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

    if (ImGui::Button("Build BVH")) {
        m_collector.request_bvh_build();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(captures current frame geometry)");

    const auto bvhStats = m_collector.last_bvh_stats();
    if (bvhStats.nodeCount > 0) {
        ImGui::Text("BVH: %u nodes, %.2f ms", bvhStats.nodeCount, bvhStats.buildMs);
    }

    ImGui::End();
}

} // namespace dusk
