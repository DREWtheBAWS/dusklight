#include <algorithm>
#include <cmath>
#include <vector>
#include <mutex>

#include "dusk/logging.h"
#include "dusk/settings.h"
#include "dusk/main.h"
#include "d/d_kankyo.h"
#include "d/d_com_inf_game.h"
#include "imgui.h"
#include "ImGuiMenuTools.hpp"

namespace dusk {
    static ImGuiTextBuffer StubLogBuffer;
    static std::vector<int> LineOffsets;
    static bool StubLogPaused;
    static std::mutex StubLogMutex;

    const char* LogLevelName(const AuroraLogLevel level) {
        switch (level) {
        case LOG_DEBUG:
            return "DEBUG";
        case LOG_INFO:
            return "INFO";
        case LOG_WARNING:
            return "WARNING";
        case LOG_ERROR:
            return "ERROR";
        case LOG_FATAL:
            return "FATAL";
        default:
            return "UNKNOWN";
        }
    }

    void SendToStubLog(AuroraLogLevel level, const char* module, const char* message) {
        if (StubLogPaused) {
            return;
        }

        std::lock_guard lock(StubLogMutex);

        if (StubLogBuffer.size() > 1024 * 1024) {
            DuskLog.warn("Stub log FULL. Dropping logs!");
            return;
        }

        LineOffsets.push_back(StubLogBuffer.size());
        const auto levelName = LogLevelName(level);
        StubLogBuffer.appendf("[%s | %s] %s\n", levelName, module, message);
    }

    void ImGuiMenuTools::ShowStubLog() {
        std::lock_guard lock(StubLogMutex);

        if (!m_showStubLog) {
            return;
        }

        if (ImGui::Begin("Stub log", &m_showStubLog)) {
            ImGui::Checkbox("Redirect stub log", &StubLogEnabled);
            ImGui::SameLine();
            ImGui::Checkbox("Pause", &StubLogPaused);

            ImGui::Text("Line count (this frame): %zu", LineOffsets.size());

            ImGui::Separator();

            if (ImGui::BeginChild("scrolling")) {
                ImGuiListClipper clipper;
                clipper.Begin(static_cast<int>(LineOffsets.size()));
                while (clipper.Step()) {
                    for (int idx = clipper.DisplayStart; idx < clipper.DisplayEnd; idx++) {
                        const char* lineStart = StubLogBuffer.begin() + LineOffsets[idx];
                        const char* lineEnd = idx == LineOffsets.size() - 1 ? StubLogBuffer.end() : StubLogBuffer.begin() + LineOffsets[idx + 1];
                        ImGui::TextUnformatted(lineStart, lineEnd);
                    }
                }

                clipper.End();
            }

            ImGui::EndChild();
        }

        ImGui::End();
    }

    void ClearPastFrame() {
        if (StubLogPaused) {
            return;
        }
        StubLogBuffer.clear();
        LineOffsets.clear();
    }

    void ImGuiMenuTools::afterDraw() {
        std::lock_guard lock(StubLogMutex);
        ClearPastFrame();
        m_collector.end_frame();

        // ---- Main-thread RT prep -------------------------------------------
        // This is the one point in the frame where the main thread is between
        // frames: all of this frame's draws are recorded and the next frame's
        // draws haven't started.  Everything that reads live collector/cache
        // state runs here; the pre-UI callback (render worker, concurrent with
        // the NEXT frame's draws) only encodes GPU work from the snapshots.

        // Lazy install/uninstall of the geometry capture callback.  Done here
        // (not in the pre-UI callback) because the capture callback fires on
        // this thread during the FIFO drain — installing from the same thread
        // avoids a cross-thread race on the callback pointer.
        const bool rtaoNeeded = m_showRtaoCapture || getSettings().game.rtaoEnabled.getValue();
        if (rtaoNeeded && !m_captureInstalled) {
            m_collector.install();
            m_captureInstalled = true;
        } else if (!rtaoNeeded && m_captureInstalled) {
            m_collector.uninstall();
            m_captureInstalled = false;
        }

        // Build pending SAH BVHs on the main thread (running them on the render
        // worker caused multi-second stalls when new areas were loaded).
        m_blasCache.flush();

        if (m_captureInstalled) {
            // When the debug capture window is closed, drive everything from
            // persistent settings (the window overrides these when visible).
            if (!m_showRtaoCapture) {
                const float dist = static_cast<float>(getSettings().game.rtaoRayLength.getValue());
                static constexpr uint32_t kQualityRays[] = {1u, 4u, 8u};
                const int q = std::clamp(getSettings().game.rtaoQuality.getValue(), 0, 2);
                m_aoPass.set_params({kQualityRays[q], dist, 0.01f, 0u, 0u, 0.02f, 3000.f});
                m_aoStrength     = getSettings().game.rtaoIntensity.getValue();
                m_shadowEnabled  = getSettings().game.rtShadowEnabled.getValue();
                m_shadowStrength = getSettings().game.rtShadowIntensity.getValue();
                const int iters = getSettings().game.rtaoDenoiserIterations.getValue();
                m_denoiseIterations = iters;
                m_denoiseEnabled = (iters > 0);
                m_aoEnabled    = true;
                m_tlasBuilder.set_force_rebuild(false);
                m_collector.set_max_distance(dist * 4.f);
                m_collector.set_frustum_margin(dist);
                m_collector.set_max_edge_length(dist * 3.f);
                // Skinned (dynamic) range must cover shadow casters, not just the
                // AO radius: at dist*4 (~400u) Link sat right at the cutoff and
                // his triangles flickered in/out of the LBVH with camera drift.
                const float dynRange = std::max(dist * 4.f, 3000.f);
                m_bvhBuilder.set_morton_range(dynRange);
                m_blasCache.set_max_distance(dynRange);
            }
            // Triangle collection stays off: instances come from the BlasCache
            // callback; OBJ dumps force one collected frame via request_dump().
            m_collector.set_collect_triangles(false);

            // The camera snapshot and the TLAS both embed this frame's view
            // matrix; the render worker must never see one without the other,
            // so update them atomically under the RT prep mutex.
            std::lock_guard rtLock(m_rtPrepMutex);

            // Snapshot the camera committed by end_frame() above, and the game's
            // sun position (game state — must be read on this thread).
            // sun_light_pos is the world-space position GX uses for sun
            // diffuse/specular; plight_near_pos would aim rays at the floor.
            m_camSnapshot = m_collector.last_camera_data();

            // In-game, override with the game's authoritative camera.  Deriving
            // the camera from per-draw pnMtx heuristics breaks whenever a
            // reflection/effect pre-pass renders first at certain camera
            // positions (all-red root-AABB view, dead AO/shadows there).  We're
            // a native reimplementation — the real camera is directly readable.
            const view_class* gv = dusk::IsGameLaunched ? dComIfGd_getView() : nullptr;
            if (gv) { // null between IsGameLaunched and first scene init
                const f32 (*V)[4] = gv->viewMtx; // row-major world→view, 3×4
                const float r0len2 = V[0][0]*V[0][0] + V[0][1]*V[0][1] + V[0][2]*V[0][2];
                if (std::abs(r0len2 - 1.f) < 0.05f) { // sane rotation → camera initialized
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 4; ++c)
                            m_camSnapshot.view[r][c] = V[r][c];
                    m_camSnapshot.view[3][0] = 0.f;
                    m_camSnapshot.view[3][1] = 0.f;
                    m_camSnapshot.view[3][2] = 0.f;
                    m_camSnapshot.view[3][3] = 1.f;
                    // CameraData.proj uses the capture convention (column-major,
                    // proj[c][r]); the game's Mtx44 is row-major → transpose.
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c)
                            m_camSnapshot.proj[c][r] = gv->projMtx[r][c];
                    for (int i = 0; i < 3; ++i)
                        m_camSnapshot.worldPos[i] = -(V[0][i]*V[0][3] + V[1][i]*V[1][3] + V[2][i]*V[2][3]);
                    m_camSnapshot.valid = true;
                }
            }

            const cXyz& lightPos = g_env_light.sun_light_pos;
            m_camSnapshot.lightWorldPos[0] = lightPos.x;
            m_camSnapshot.lightWorldPos[1] = lightPos.y;
            m_camSnapshot.lightWorldPos[2] = lightPos.z;

            // World-space TLAS over this frame's instances (must follow flush()
            // so newly built BLAS entries are visible).
            m_tlasBuilder.set_exclude_skinned(m_excludeSkinned);
            m_tlasBuilder.build(m_blasCache, m_camSnapshot.view);

            // Copy the skinned triangles for the render worker's GPU LBVH build.
            if (!m_excludeSkinned) {
                m_dynTrisSnapshot = m_blasCache.dynamic_triangles();
            } else {
                m_dynTrisSnapshot.clear();
            }
        }

        m_blasCache.advance_frame();
        // m_tlasBuilder.advance_frame() intentionally NOT called: build() manages
        // m_instances internally, and TlasBuilder::flush() may still run on the
        // render worker for the frame just enqueued.
    }
}
