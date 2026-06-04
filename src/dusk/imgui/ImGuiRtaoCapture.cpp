#include "ImGuiMenuTools.hpp"
#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"
#include <SDL3/SDL_filesystem.h>
#include <string>

namespace dusk {

void ImGuiMenuTools::ShowRtaoCaptureWindow() {
    if (!m_showRtaoCapture) return;

    if (!ImGui::Begin("RTAO Capture", &m_showRtaoCapture)) {
        ImGui::End();
        return;
    }

    const auto stats = m_collector.last_stats();
    ImGui::Text("Triangles (last frame): %u", stats.triangleCount);
    ImGui::Text("Draw calls (last frame): %u", stats.drawCallCount);
    ImGui::Text("Alpha textures: %u / %u total", static_cast<uint32_t>(m_collector.texture_views().size()),
                m_collector.total_alpha_tex_count());

    const auto cam = m_collector.last_camera_data();
    if (cam.valid) {
        ImGui::Text("Camera: (%.1f, %.1f, %.1f)  FoV: %.1f deg",
                    cam.worldPos[0], cam.worldPos[1], cam.worldPos[2], cam.fovYDeg);
    } else {
        ImGui::TextDisabled("Camera: (no data yet)");
    }

    ImGui::Separator();

    // Filter controls
    static bool s_perspOnly = true;
    static float s_minW = 320.f, s_minH = 240.f;
    bool filterChanged = false;
    filterChanged |= ImGui::Checkbox("Perspective only", &s_perspOnly);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.f);
    filterChanged |= ImGui::InputFloat("Min W", &s_minW, 0.f, 0.f, "%.0f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.f);
    filterChanged |= ImGui::InputFloat("Min H", &s_minH, 0.f, 0.f, "%.0f");
    if (filterChanged) {
        m_collector.set_filter(s_perspOnly, s_minW, s_minH);
    }

    ImGui::Separator();

    // OBJ dump
    static std::string s_dumpPath;
    if (s_dumpPath.empty()) {
        const char* base = SDL_GetBasePath();
        s_dumpPath = std::string(base ? base : "") + "rtao_capture.obj";
    }
    ImGui::InputText("Path", &s_dumpPath);
    if (ImGui::Button("Dump OBJ")) {
        m_collector.request_dump(s_dumpPath);
    }
    const auto& msg = m_collector.last_dump_message();
    if (!msg.empty()) {
        ImGui::TextWrapped("%s", msg.c_str());
    }

    ImGui::Separator();

    // GPU BVH controls
    ImGui::Checkbox("Build only (skip AO)", &m_buildBvhOnly);
    ImGui::SameLine();
    ImGui::Checkbox("Freeze BVH", &m_bvhFrozen);
    ImGui::SameLine();
    if (ImGui::Button("Capture & Freeze")) {
        // do exactly one BVH rebuild on the next frame, then lock it
        m_bvhFrozen      = false;
        m_bvhCaptureOnce = true;
    }
    if (m_bvhFrozen) { ImGui::SameLine(); ImGui::TextDisabled("(frozen)"); }
    const auto bvhStats = m_bvhBuilder.last_stats();
    if (bvhStats.nodeCount > 0) {
        ImGui::Text("%u nodes, %u tris, %.2f ms", bvhStats.nodeCount, bvhStats.triCount, bvhStats.buildMs);
        {
            const float* mn = bvhStats.sceneMin;
            const float* mx = bvhStats.sceneMax;
            const float eX = mx[0]-mn[0], eY = mx[1]-mn[1], eZ = mx[2]-mn[2];
            ImGui::Text("Full AABB:   %.0f x %.0f x %.0f", eX, eY, eZ);
        }
        {
            const float* mn = bvhStats.mortonMin;
            const float* mx = bvhStats.mortonMax;
            const float eX = mx[0]-mn[0], eY = mx[1]-mn[1], eZ = mx[2]-mn[2];
            ImGui::Text("Morton AABB: %.0f x %.0f x %.0f  (mean+/-3s)", eX, eY, eZ);
            if (eX > 7000.f || eY > 7000.f || eZ > 7000.f)
                ImGui::TextColored({1.f,0.4f,0.f,1.f}, "  ^ Morton range still large");
        }
    } else {
        ImGui::TextDisabled("(not built yet)");
    }
    ImGui::Separator();

    // AO params
    static int   s_raysPerPixel = 1;
    static float s_maxDist      = 500.f;
    static float s_normalBias   = 0.01f;
    static int   s_debugMode    = 0;
    static int   s_debugMode2   = 0;
    ImGui::SetNextItemWidth(120.f);
    ImGui::SliderInt("Rays/pixel", &s_raysPerPixel, 1, 64);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.f);
    ImGui::InputFloat("Max dist", &s_maxDist, 0.f, 0.f, "%.0f");
    ImGui::SameLine();
    ImGui::TextDisabled("Morton: +-%.0f", s_maxDist * 4.f);
    ImGui::SetNextItemWidth(120.f);
    ImGui::InputFloat("Normal bias", &s_normalBias, 0.001f, 0.01f, "%.4f");
    ImGui::SetNextItemWidth(180.f);
    ImGui::Combo("Left panel",  &s_debugMode,  "AO\0Normals\0Depth dist\0Root AABB\0");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(180.f);
    ImGui::Combo("Right panel", &s_debugMode2, "Limit Hits\0AO\0Normals\0Depth dist\0Root AABB\0Visit Heat\0Limit %\0");
    m_aoPass.set_params({static_cast<uint32_t>(s_raysPerPixel), s_maxDist, s_normalBias,
                         static_cast<uint32_t>(s_debugMode),
                         static_cast<uint32_t>(s_debugMode2)});
    // Geometry sphere radius: 4× the AO ray length ensures surfaces at the far end of
    // the frustum still have nearby geometry collected for occlusion.
    // Frustum margin: exactly the AO ray length — geometry more than one ray-length outside
    // the view frustum can never occlude a visible surface, so discard it.
    m_bvhBuilder.set_morton_range(s_maxDist * 4.f);
    m_collector.set_max_distance(s_maxDist * 4.f);
    m_collector.set_frustum_margin(s_maxDist);
    // Skip triangles with any edge longer than 3× the AO ray length: these are
    // coarse terrain patches that inflate every ancestor AABB in the BVH tree
    // without contributing meaningful occlusion detail.
    m_collector.set_max_edge_length(s_maxDist * 3.f);

    static const char* kPanelNames1[] = {"AO", "Normals", "Depth dist", "Root AABB"};
    static const char* kPanelNames2[] = {"Limit Hits", "AO", "Normals", "Depth dist", "Root AABB", "Visit Heat", "Limit %"};

    if (!m_aoPass.is_ready()) {
        ImGui::Separator();
        ImGui::TextDisabled("(start the game to see AO)");
    } else {
        const float avail  = ImGui::GetContentRegionAvail().x;
        const float half   = (avail - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        const float aspect = static_cast<float>(m_aoPass.width()) /
                             static_cast<float>(m_aoPass.height());
        const float h = half / aspect;

        ImGui::Separator();
        ImGui::Text("%s", kPanelNames1[s_debugMode]);
        ImGui::SameLine(half + ImGui::GetStyle().ItemSpacing.x);
        ImGui::Text("%s", kPanelNames2[s_debugMode2]);

        ImGui::Image(m_aoPass.imgui_texture_id(),   ImVec2(half, h));
        ImGui::SameLine();
        ImGui::Image(m_aoPass.limits_texture_id(),  ImVec2(half, h));
    }

    ImGui::End();
}

} // namespace dusk
