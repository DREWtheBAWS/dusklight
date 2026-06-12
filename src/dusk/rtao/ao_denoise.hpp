#pragma once
#include <webgpu/webgpu.h>
#include <cstdint>

namespace dusk::rtao {

// A-trous wavelet denoiser for the RTAO output.
// Runs kMaxIterations of a 5-tap cross bilateral filter with step sizes 1,2,4,8,16.
// Edge stopping is keyed on depth difference and AO value difference.
class AoDenoisePass {
public:
    static constexpr uint32_t kMaxIterations = 5;

    ~AoDenoisePass();

    // Denoise aoView (from AoPass::ao_texture_view()) guided by depthTex.
    // Returns the denoised view, or aoView if numIterations == 0 / not ready.
    // sigmaZ: depth edge sensitivity   (larger = more blur across depth edges)
    // sigmaL: AO value edge sensitivity (larger = more blur across AO edges)
    WGPUTextureView execute(WGPUDevice device, WGPUCommandEncoder encoder,
                            WGPUTextureView aoView, WGPUTexture depthTex,
                            uint32_t numIterations, float sigmaZ, float sigmaL);

private:
    void ensure_pipeline(WGPUDevice device);
    void rebuild_textures(WGPUDevice device, uint32_t w, uint32_t h);
    void rebuild_depth_view(WGPUTexture depthTex);
    void rebuild_bind_groups(WGPUDevice device, WGPUTextureView aoView);
    void update_ubos(WGPUDevice device, float sigmaZ, float sigmaL);

    WGPUComputePipeline m_pipeline = nullptr;
    WGPUBindGroupLayout m_bgl      = nullptr;

    // Ping-pong textures: one is the active read, the other the write target.
    WGPUTexture     m_ping     = nullptr;
    WGPUTextureView m_pingView = nullptr;
    WGPUTexture     m_pong     = nullptr;
    WGPUTextureView m_pongView = nullptr;

    WGPUTexture     m_lastDepthTex = nullptr;
    WGPUTextureView m_depthView    = nullptr;

    WGPUTextureView m_lastAoView = nullptr;

    // One UBO and bind group per iteration slot.
    WGPUBuffer    m_ubo[kMaxIterations]        = {};
    WGPUBindGroup m_bindGroups[kMaxIterations] = {};

    float m_lastSigmaZ = -1.f;
    float m_lastSigmaL = -1.f;

    uint32_t m_width  = 0;
    uint32_t m_height = 0;
};

} // namespace dusk::rtao
