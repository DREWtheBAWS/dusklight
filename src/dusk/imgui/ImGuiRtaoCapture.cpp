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

    ImGui::End();
}

} // namespace dusk
