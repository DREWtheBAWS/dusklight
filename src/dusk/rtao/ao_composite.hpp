#pragma once
#include <webgpu/webgpu.h>
#include <cstdint>

namespace dusk::rtao {

// Applies the AO texture to the EFB color buffer in-place.
// Runs as a full-screen render pass inside Aurora's post-render callback,
// between GX rendering and the EFB→swapchain copy.
class AoCompositePass {
public:
    ~AoCompositePass();

    // colorTex — aurora_get_color_texture() (the EFB; TextureBinding + CopyDst)
    // aoView   — AoPass::ao_texture_view()  (RGBA8Unorm, same resolution as EFB)
    // strength — [0, 1]: 0 = no AO, 1 = full AO
    void execute(WGPUDevice device, WGPUCommandEncoder encoder,
                 WGPUTexture colorTex, WGPUTextureView aoView,
                 float strength);

private:
    void ensure_pipeline(WGPUDevice device, WGPUTextureFormat fmt);
    void rebuild_scratch(WGPUDevice device, uint32_t w, uint32_t h, WGPUTextureFormat fmt);
    void rebuild_bind_group(WGPUDevice device);

    WGPURenderPipeline  m_pipeline  = nullptr;
    WGPUBindGroupLayout m_bgl       = nullptr;
    WGPUBindGroup       m_bindGroup = nullptr;
    WGPUSampler         m_sampler   = nullptr;
    WGPUBuffer          m_paramsUbo = nullptr;

    // Scratch texture: render target for the composite; copied into the EFB after.
    WGPUTexture     m_scratchTex  = nullptr;
    WGPUTextureView m_scratchView = nullptr;

    // Cached view created from the last colorTex (stable across frames unless resized).
    WGPUTexture     m_lastColorTex       = nullptr;
    WGPUTextureView m_colorSampledView   = nullptr;

    WGPUTextureView m_lastAoView = nullptr;

    uint32_t          m_width     = 0;
    uint32_t          m_height    = 0;
    WGPUTextureFormat m_fmt       = WGPUTextureFormat_Undefined;
    WGPUTextureFormat m_scratchFmt = WGPUTextureFormat_Undefined;
};

} // namespace dusk::rtao
