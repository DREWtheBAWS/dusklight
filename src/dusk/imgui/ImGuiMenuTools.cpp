#include "fmt/format.h"
#include "imgui.h"
#include "aurora/gfx.h"

#include "ImGuiConfig.hpp"
#include "dusk/hotkeys.h"
#include "dusk/settings.h"
#include "ImGuiConsole.hpp"
#include "ImGuiMenuTools.hpp"
#include <aurora/post_render.h>

#include "ImGuiEngine.hpp"
#include "d/actor/d_a_alink.h"
#include "d/actor/d_a_horse.h"
#include "d/d_com_inf_game.h"
#include "d/d_kankyo.h"
#include "dusk/data.hpp"
#include "dusk/dusk.h"
#include "dusk/main.h"
#include "m_Do/m_Do_main.h"

#include <algorithm>
#include <aurora/lib/internal.hpp>
#include <SDL3/SDL_misc.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace aurora::gx {
extern bool enableLodBias;
}

namespace dusk {
    ImGuiMenuTools::ImGuiMenuTools() {
        m_collector.install();
        m_collector.set_draw_callback([](const AuroraGxCaptureDraw& draw, void* ud) {
            static_cast<rtao::BlasCache*>(ud)->record_draw(draw);
        }, &m_blasCache);
        aurora_set_pre_ui_callback([](WGPUDevice device, WGPUCommandEncoder encoder, void* userdata) {
            auto* self = static_cast<ImGuiMenuTools*>(userdata);

            // When the debug capture window is not open, drive everything from persistent settings.
            // The capture window overrides these each frame when it is visible.
            if (!self->m_showRtaoCapture) {
                if (!getSettings().game.rtaoEnabled.getValue()) {
                    return; // Skip all RT work — zero GPU overhead
                }
                const float dist = static_cast<float>(getSettings().game.rtaoRayLength.getValue());
                static constexpr uint32_t kQualityRays[] = {1u, 4u, 8u};
                const int q = std::clamp(getSettings().game.rtaoQuality.getValue(), 0, 2);
                self->m_aoPass.set_params({kQualityRays[q], dist, 0.01f, 0u, 0u, 0.02f});
                self->m_aoStrength     = getSettings().game.rtaoIntensity.getValue();
                self->m_shadowEnabled  = getSettings().game.rtShadowEnabled.getValue();
                self->m_shadowStrength = getSettings().game.rtShadowIntensity.getValue();
                const int iters = getSettings().game.rtaoDenoiserIterations.getValue();
                self->m_denoiseIterations = iters;
                self->m_denoiseEnabled = (iters > 0);
                self->m_useTlasBvh   = true;
                self->m_aoEnabled    = true;
                self->m_buildBvhOnly = false;
                self->m_bvhFrozen    = false;
                self->m_tlasBuilder.set_force_rebuild(false);
                self->m_collector.set_max_distance(dist * 4.f);
                self->m_collector.set_frustum_margin(dist);
                self->m_collector.set_max_edge_length(dist * 3.f);
                self->m_bvhBuilder.set_morton_range(dist * 4.f);
                self->m_blasCache.set_max_distance(dist * 4.f);
            }

            // Build new BLASes (SAH, local space) and upload to GPU.
            // Runs unconditionally so the cache stays warm even when the LBVH is frozen.
            self->m_blasCache.flush();

            // Use the game's actual sun light position (GX lighting source) for shadow direction.
            // sun_light_pos is the world-space position GX uses for sun diffuse/specular — it's
            // typically very far above the scene (directional-light approximation), which is what
            // we want for casting sun shadows.  plight_near_pos is a nearby point light (torch,
            // lamp) and causes rays to aim at the scene floor/origin, blocking everything.
            const cXyz& lightPos = g_env_light.sun_light_pos;
            self->m_collector.set_light_world_pos(lightPos.x, lightPos.y, lightPos.z);

            // Camera data is needed for the world-space TLAS build and the AO/shadow passes.
            const auto& camData  = self->m_collector.pending_camera_data();

            // Build the world-space TLAS over this frame's instances.
            // Must run after m_blasCache.flush() so all BLAS entries are available.
            self->m_tlasBuilder.build(self->m_blasCache, camData.view);
            self->m_tlasBuilder.flush(device);

            // Sync the exclude-skinned debug flag so TlasBuilder omits the dynamic instance.
            self->m_tlasBuilder.set_exclude_skinned(self->m_excludeSkinned);

            if (!self->m_useTlasBvh) {
                // Original single-level LBVH path (all captured geometry).
                const bool doRebuild = !self->m_bvhFrozen || self->m_bvhCaptureOnce;
                if (doRebuild) {
                    const auto& tris = self->m_collector.raw_triangles();
                    if (!tris.empty()) {
                        self->m_bvhBuilder.upload_triangles(device, tris);
                        self->m_bvhBuilder.build(device);
                        if (self->m_bvhCaptureOnce) {
                            self->m_bvhFrozen      = true;
                            self->m_bvhCaptureOnce = false;
                        }
                    }
                }
            } else if (!self->m_excludeSkinned) {
                // TLAS mode: GPU LBVH for skinned (multi-matrix) geometry only.
                // BlasCache separates multi-matrix draws into dynamic_triangles() and applies
                // the same distance filter as the collector, so only in-range skinned tris reach here.
                const auto& dynTris = self->m_blasCache.dynamic_triangles();
                if (!dynTris.empty()) {
                    self->m_bvhBuilder.upload_triangles(device, dynTris);
                    self->m_bvhBuilder.build(device);
                }
            }

            // AO pass: choose LBVH or BLAS/TLAS based on user toggle.
            WGPUTexture depthTex = aurora_get_depth_texture();
            const auto& texViews = self->m_collector.texture_views();
            if (!self->m_buildBvhOnly) {
                if (self->m_useTlasBvh && self->m_tlasBuilder.is_ready()) {
                    const bool dynReady = self->m_bvhBuilder.is_ready() && !self->m_excludeSkinned;
                    self->m_aoPass.execute_tlas(device, encoder, depthTex, camData,
                                                self->m_tlasBuilder.tlas_node_buf(),
                                                self->m_tlasBuilder.instance_buf(),
                                                self->m_tlasBuilder.blas_node_buf(),
                                                self->m_tlasBuilder.blas_tri_buf(),
                                                dynReady ? self->m_bvhBuilder.node_buf() : nullptr,
                                                dynReady ? self->m_bvhBuilder.tri_buf()  : nullptr,
                                                dynReady ? self->m_bvhBuilder.last_stats().nodeCount : 0u,
                                                texViews);
                } else if (self->m_bvhBuilder.is_ready()) {
                    self->m_aoPass.execute(device, encoder, depthTex, camData,
                                           self->m_bvhBuilder.node_buf(),
                                           self->m_bvhBuilder.tri_buf(),
                                           texViews);
                }

                // Shadow pass: one ray per pixel toward the sun/light source.
                if (self->m_shadowEnabled && self->m_useTlasBvh && self->m_tlasBuilder.is_ready()) {
                    const bool dynReady = self->m_bvhBuilder.is_ready() && !self->m_excludeSkinned;
                    self->m_aoPass.execute_shadow_tlas(device, encoder, depthTex, camData,
                                                       self->m_tlasBuilder.tlas_node_buf(),
                                                       self->m_tlasBuilder.instance_buf(),
                                                       self->m_tlasBuilder.blas_node_buf(),
                                                       self->m_tlasBuilder.blas_tri_buf(),
                                                       dynReady ? self->m_bvhBuilder.node_buf() : nullptr,
                                                       dynReady ? self->m_bvhBuilder.tri_buf()  : nullptr,
                                                       dynReady ? self->m_bvhBuilder.last_stats().nodeCount : 0u);
                }

                // Denoise + composite: apply filtered AO and shadow to the EFB.
                if (self->m_aoEnabled && self->m_aoPass.is_ready()) {
                    WGPUTextureView aoView = self->m_aoPass.ao_texture_view();
                    if (self->m_denoiseEnabled && self->m_denoiseIterations > 0) {
                        WGPUTextureView filtered = self->m_denoisePass.execute(
                            device, encoder, aoView, depthTex,
                            static_cast<uint32_t>(self->m_denoiseIterations),
                            self->m_denoiseSigmaZ, self->m_denoiseSigmaL);
                        if (filtered) aoView = filtered;
                    }
                    WGPUTextureView shadowView = self->m_shadowEnabled
                                              ? self->m_aoPass.shadow_texture_view() : nullptr;
                    if (shadowView && self->m_denoiseEnabled && self->m_denoiseIterations > 0) {
                        WGPUTextureView filtered = self->m_shadowDenoisePass.execute(
                            device, encoder, shadowView, depthTex,
                            static_cast<uint32_t>(self->m_denoiseIterations),
                            self->m_denoiseSigmaZ, self->m_denoiseSigmaL);
                        if (filtered) shadowView = filtered;
                    }
                    WGPUTexture colorTex = aurora_get_color_texture();
                    self->m_compositePass.execute(device, encoder, colorTex,
                                                  aoView, self->m_aoStrength,
                                                  shadowView, self->m_shadowStrength);
                }
            }
        }, this);
    }

    void ImGuiMenuTools::draw() {
        if (ImGui::BeginMenu("Tools")) {
            if (!dusk::IsGameLaunched) {
                ImGui::BeginDisabled();
            }

            ImGui::BeginDisabled(getSettings().game.speedrunMode);

            ImGui::MenuItem("Save Editor", hotkeys::SHOW_SAVE_EDITOR, &m_showSaveEditor);
            ImGui::MenuItem("State Share", hotkeys::SHOW_STATE_SHARE, &m_showStateShare);

            ImGui::EndDisabled();

            if (!dusk::IsGameLaunched) {
                ImGui::EndDisabled();
            }

#if DUSK_CAN_OPEN_DATA_FOLDER
            ImGui::Separator();
            if (ImGui::MenuItem("Open Data Folder")) {
                data::open_data_path();
            }
#endif

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Debug")) {
            ImGui::BeginDisabled(getSettings().game.speedrunMode);

            bool developmentMode = mDoMain::developmentMode == 1;
            if (ImGui::Checkbox("Development Mode", &developmentMode)) {
                mDoMain::developmentMode = developmentMode ? 1 : -1;
            }

            ImGui::Separator();

            auto& collisionView = getTransientSettings().collisionView;
            if (ImGui::BeginMenu("Graphics Settings")) {
                bool disableWaterRefraction = getSettings().game.disableWaterRefraction;
                if (ImGui::Checkbox("Disable Water Refraction", &disableWaterRefraction)) {
                    getSettings().game.disableWaterRefraction.setValue(disableWaterRefraction);
                    config::Save();
                }
                ImGui::Checkbox("Enable LOD Bias", &aurora::gx::enableLodBias);
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Collision View")) {
                ImGui::Checkbox("Enable Terrain view", &collisionView.enableTerrainView);
                ImGui::Checkbox("Enable wireframe view", &collisionView.enableWireframe);
                ImGui::SliderFloat("Opacity##terrain", &collisionView.terrainViewOpacity, 0.0f, 100.0f);
                ImGui::SliderFloat("Draw Range", &collisionView.drawRange, 0.0f, 1000.0f);
                ImGui::Separator();
                ImGui::Checkbox("Enable Attack Collider view", &collisionView.enableAtView);
                ImGui::Checkbox("Enable Target Collider view", &collisionView.enableTgView);
                ImGui::Checkbox("Enable Push Collider view", &collisionView.enableCoView);
                ImGui::SliderFloat("Opacity##colliders", &collisionView.colliderViewOpacity, 0.0f, 100.0f);
                ImGui::EndMenu();
            }

            if (!dusk::IsGameLaunched) {
                ImGui::BeginDisabled();
            }

            ImGui::MenuItem("Process Management", hotkeys::SHOW_PROCESS_MANAGEMENT, &m_showProcessManagement);
            ImGui::MenuItem("Debug Overlay", hotkeys::SHOW_DEBUG_OVERLAY, &m_showDebugOverlay);
            ImGui::MenuItem("Heap Viewer", hotkeys::SHOW_HEAP_VIEWER, &m_showHeapOverlay);
            ImGui::MenuItem("Player Info", hotkeys::SHOW_PLAYER_INFO, &m_showPlayerInfo);
            ImGui::MenuItem("Debug Camera", hotkeys::SHOW_DEBUG_CAMERA, &m_showCameraOverlay);
            ImGui::MenuItem("Audio Debug", hotkeys::SHOW_AUDIO_DEBUG, &m_showAudioDebug);
            ImGui::MenuItem("Bloom", nullptr, &m_showBloomWindow);
            ImGui::MenuItem("Stub Log", nullptr, &m_showStubLog);
            ImGui::MenuItem("Actor Spawner", nullptr, &m_showActorSpawner);
            ImGui::MenuItem("RTAO Capture", nullptr, &m_showRtaoCapture);

            if (!dusk::IsGameLaunched) {
                ImGui::EndDisabled();
            }

            ImGui::MenuItem("OSReport Force", nullptr, &OSReportReallyForceEnable);

            ImGui::EndDisabled();

            ImGui::EndMenu();
        }
    }

    void ImGuiMenuTools::ShowDebugOverlay() {
        if (!getSettings().backend.enableAdvancedSettings ||
            !ImGuiConsole::CheckMenuViewToggle(ImGuiKey_F3, m_showDebugOverlay))
        {
            return;
        }

        ImGui::PushFont(ImGuiEngine::fontMono);

        ImGuiIO& io = ImGui::GetIO();
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav;
        if (m_debugOverlayCorner != -1) {
            SetOverlayWindowLocation(m_debugOverlayCorner);
            windowFlags |= ImGuiWindowFlags_NoMove;
        }

        ImGui::SetNextWindowBgAlpha(0.65f);
        if (ImGui::Begin("Debug Overlay", nullptr, windowFlags)) {
            ImGuiStringViewText(fmt::format(FMT_STRING("FPS: {:.2f}\n"), io.Framerate));
            if (frameUsagePct > 0.f) {
                ImGuiStringViewText(fmt::format(FMT_STRING("Frame usage: {:.1f}%\n"), frameUsagePct));
            }

            ImGui::Separator();

            ImGuiStringViewText(fmt::format(FMT_STRING("Backend: {}\n"), backend_name(aurora_get_backend())));

            ImGui::Separator();

            const auto& stats = lastFrameAuroraStats;

            ImGuiStringViewText(
                fmt::format(FMT_STRING("Queued pipelines:  {}\n"), stats.queuedPipelines));
            ImGuiStringViewText(
                fmt::format(FMT_STRING("Done pipelines:    {}\n"), stats.createdPipelines));
            ImGuiStringViewText(
                fmt::format(FMT_STRING("Draw call count:   {}\n"), stats.drawCallCount));
            ImGuiStringViewText(fmt::format(FMT_STRING("Merged draw calls: {}\n"),
                stats.mergedDrawCallCount));
            ImGuiStringViewText(fmt::format(FMT_STRING("Vertex size:       {}\n"),
                BytesToString(stats.lastVertSize)));
            ImGuiStringViewText(fmt::format(FMT_STRING("Uniform size:      {}\n"),
                BytesToString(stats.lastUniformSize)));
            ImGuiStringViewText(fmt::format(FMT_STRING("Index size:        {}\n"),
                BytesToString(stats.lastIndexSize)));
            ImGuiStringViewText(fmt::format(FMT_STRING("Storage size:      {}\n"),
                BytesToString(stats.lastStorageSize)));
            ImGuiStringViewText(fmt::format(FMT_STRING("Tex upload size:   {}\n"),
                BytesToString(stats.lastTextureUploadSize)));
            ImGuiStringViewText(fmt::format(
                FMT_STRING("Total:             {}\n"),
                BytesToString(stats.lastVertSize + stats.lastUniformSize +
                    stats.lastIndexSize + stats.lastStorageSize +
                    stats.lastTextureUploadSize)));

            // TODO: persist to config
            ShowCornerContextMenu(m_debugOverlayCorner, m_cameraOverlayCorner);
        }
        ImGui::End();

        ImGui::PopFont();
    }

    void ImGuiMenuTools::ShowPlayerInfo() {
        if (!getSettings().backend.enableAdvancedSettings ||
            !ImGuiConsole::CheckMenuViewToggle(ImGuiKey_F5, m_showPlayerInfo))
        {
            return;
        }

        ImGui::PushFont(ImGuiEngine::fontMono);

        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav;
        if (m_playerInfoOverlayCorner != -1) {
            SetOverlayWindowLocation(m_playerInfoOverlayCorner);
            windowFlags |= ImGuiWindowFlags_NoMove;
        }

        ImGui::SetNextWindowBgAlpha(0.65f);

        if (ImGui::Begin("Player Info", nullptr, windowFlags)) {
            daAlink_c* player = (daAlink_c*)dComIfGp_getPlayer(0);
            daHorse_c* horse = dComIfGp_getHorseActor();

            double speedXzy = 0.0;
            if (player != nullptr) {
                speedXzy = sqrtf(player->speed.x * player->speed.x
                    + player->speed.z * player->speed.z
                    + player->speed.y * player->speed.y);
            }

            ImGui::Text("Global");
            ImGuiStringViewText(
                player != nullptr
                ? fmt::format("Stage: {}\n", dComIfGp_getStartStageName()) 
                : "Stage: ?\n"
            );

            ImGuiStringViewText(
                player != nullptr
                ? fmt::format("Layer: {0}\n", dComIfG_play_c::getLayerNo(0))
                : "Layer: ?\n"
            );

            ImGui::Separator();
            ImGui::Text("Link");
            ImGuiStringViewText(
                player != nullptr
                ? fmt::format("Position: {: .4f}, {: .4f}, {: .4f}\n", player->current.pos.x, player->current.pos.y, player->current.pos.z)
                : "Position: ?, ?, ?\n"
            );

            ImGuiStringViewText(
                player != nullptr
                ? fmt::format("Velocity (XYZ): {: .4f}, {: .4f}, {: .4f}\n", player->speed.x, player->speed.y, player->speed.z)
                : "Velocity (XYZ): ?, ?, ?\n"
            );

            ImGuiStringViewText(
                player != nullptr
                ? fmt::format("Speed (SpeedF): {: .4f}\n", player->speedF)
                : "Speed (SpeedF): ?\n"
            );

            ImGuiStringViewText(
                player != nullptr
                ? fmt::format("Speed (3D): {: .4f}\n", speedXzy)
                : "Speed (3D): ?\n"
            );

            ImGuiStringViewText(
                 player != nullptr
                 ? fmt::format("Angle: {0}\n", player->shape_angle.y)
                 : "Angle: ?\n"
            );

            ImGuiStringViewText(
                player != nullptr
                ? fmt::format("Room: {0}\n", fopAcM_GetRoomNo(player))
                : "Room: ?\n"
            );

            ImGuiStringViewText(
                player != nullptr
                ? fmt::format("Entry: {0}\n", dComIfGp_getStartStagePoint())
                : "Entry: ?\n"
            );

            ImGui::Separator();
            ImGui::Text("Epona");
            ImGuiStringViewText(
                horse != nullptr
                ? fmt::format("Position: {: .4f}, {: .4f}, {: .4f}\n", horse->current.pos.x, horse->current.pos.y, horse->current.pos.z)
                : "Position: ?, ?, ?\n"
            );

            ImGuiStringViewText(
                 horse != nullptr
                 ? fmt::format("Velocity (XYZ): {: .4f}, {: .4f}, {: .4f}\n", horse->speed.x, horse->speed.y, horse->speed.z)
                 : "Velocity (XYZ): ?, ?, ?\n"
            );

            ImGuiStringViewText(
                horse != nullptr
                ? fmt::format("Speed (SpeedF): {: .4f}\n", horse->speedF)
                : "Speed (SpeedF): ?\n"
            );

            ImGuiStringViewText(
                horse != nullptr
                ? fmt::format("Angle: {0}\n", horse->shape_angle.y)
                : "Angle: ?\n"
            );

            ImGuiStringViewText(
                horse != nullptr
                ? fmt::format("Room: {0}\n", fopAcM_GetRoomNo(horse))
                : "Room: ?\n"
            );

            ImGuiStringViewText(
                player != nullptr
                ? fmt::format("Saved Stage: {}\n", dComIfGs_getHorseRestartStageName())
                : "Saved Stage: ?\n"
            );

            ImGuiStringViewText(
                player != nullptr
                ? fmt::format("Saved Room: {0}\n", dComIfGs_getHorseRestartRoomNo())
                : "Saved Room: ?\n"
            );

            ShowCornerContextMenu(m_playerInfoOverlayCorner, m_debugOverlayCorner);
        }

        ImGui::End();
        ImGui::PopFont();
    }
}
