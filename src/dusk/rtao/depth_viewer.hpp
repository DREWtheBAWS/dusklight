#pragma once
#include <webgpu/webgpu.h>
#include "imgui.h"
#include <cstdint>

namespace dusk::rtao {

class DepthTextureViewer {
public:
    ~DepthTextureViewer();

    // Called from the Aurora post-render callback each frame.
    void execute(WGPUDevice device, WGPUCommandEncoder encoder);

    // Returns the ImTextureID to pass to ImGui::Image(). Null until first execute().
    ImTextureID imgui_texture_id() const;
    bool is_ready() const { return m_outputView != nullptr; }
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }

private:
    void ensure_pipeline(WGPUDevice device);

    WGPURenderPipeline  m_pipeline    = nullptr;
    WGPUBindGroupLayout m_bgl         = nullptr;

    WGPUTexture         m_outputTex   = nullptr;
    WGPUTextureView     m_outputView  = nullptr;
    WGPUTextureView     m_depthView   = nullptr;
    WGPUBindGroup       m_bindGroup   = nullptr;
    WGPUTexture         m_lastDepthTex = nullptr;
    uint32_t            m_width       = 0;
    uint32_t            m_height      = 0;
};

} // namespace dusk::rtao
