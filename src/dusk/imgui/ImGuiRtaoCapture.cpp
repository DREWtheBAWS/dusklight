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

    // Derive a default dump path next to the executable
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

    static bool s_autoBvh = false;
    ImGui::Checkbox("Auto-build BVH", &s_autoBvh);
    ImGui::SameLine();
    if (ImGui::Button("Build BVH now")) {
        m_collector.request_bvh_build();
    }
    if (s_autoBvh) m_collector.request_bvh_build();

    const auto bvhStats = m_collector.last_bvh_stats();
    if (bvhStats.nodeCount > 0) {
        ImGui::Text("BVH: %u nodes, %.2f ms", bvhStats.nodeCount, bvhStats.buildMs);
    }

    // Queue BVH upload to GPU whenever a new build finishes.
    // Compare generation counter (not node count) so every rebuild triggers an upload,
    // even when the new BVH happens to have the same number of nodes as the old one.
    static uint32_t s_lastBvhGen = UINT32_MAX;
    const uint32_t curGen = m_collector.bvh_generation();
    if (curGen != s_lastBvhGen && !m_collector.bvh().empty()) {
        s_lastBvhGen = curGen;
        m_aoPass.queue_bvh_upload(m_collector.bvh());
    }

    ImGui::Separator();

    // AO params
    static int   s_raysPerPixel = 8;
    static float s_maxDist      = 500.f;
    static float s_normalBias   = 0.01f;
    ImGui::SetNextItemWidth(120.f);
    ImGui::SliderInt("Rays/pixel", &s_raysPerPixel, 1, 64);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.f);
    ImGui::InputFloat("Max dist", &s_maxDist, 0.f, 0.f, "%.0f");
    ImGui::SetNextItemWidth(120.f);
    ImGui::InputFloat("Normal bias", &s_normalBias, 0.001f, 0.01f, "%.4f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Scale factor for surface offset bias.\n"
                          "bias = max(dist * factor, factor * 10).\n"
                          "Increase if ground/close surfaces are black.");
    m_aoPass.set_params({static_cast<uint32_t>(s_raysPerPixel), s_maxDist, s_normalBias});


    ImGui::Separator();
    ImGui::Text("Depth Buffer");
    if (m_depthViewer.is_ready()) {
        const float avail  = ImGui::GetContentRegionAvail().x;
        const float aspect = static_cast<float>(m_depthViewer.width()) /
                             static_cast<float>(m_depthViewer.height());
        ImGui::Image(m_depthViewer.imgui_texture_id(), ImVec2(avail, avail / aspect));
    } else {
        ImGui::TextDisabled("(not yet available — start the game first)");
    }

    ImGui::Separator();
    ImGui::Text("AO Buffer");
    if (m_aoPass.is_ready()) {
        const float avail  = ImGui::GetContentRegionAvail().x;
        const float aspect = static_cast<float>(m_aoPass.width()) /
                             static_cast<float>(m_aoPass.height());
        ImGui::Image(m_aoPass.imgui_texture_id(), ImVec2(avail, avail / aspect));
    } else {
        ImGui::TextDisabled("(build BVH first, then start the game)");
    }

    ImGui::End();
}

} // namespace dusk
