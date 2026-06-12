#include "ao_denoise.hpp"
#include <cstring>

namespace dusk::rtao {

// A-trous 5-tap cross bilateral filter.
// Each iteration reads from t_src (the noisy/previous result) and writes to t_dst.
// Step size doubles each iteration so the filter covers increasing spatial ranges:
//   iter 0: step=1  (1-pixel offsets)
//   iter 1: step=2
//   iter 2: step=4
//   iter 3: step=8
//   iter 4: step=16
// Edge-stopping prevents blurring across depth discontinuities and AO value jumps.
static const char kShader[] = R"(
@group(0) @binding(0) var t_src   : texture_2d<f32>;
@group(0) @binding(1) var t_dst   : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var t_depth : texture_depth_2d;

struct Params {
    step    : u32,
    sigma_z : f32,
    sigma_l : f32,
    _pad    : f32,
};
@group(0) @binding(3) var<uniform> u : Params;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let coord = vec2<i32>(gid.xy);
    let dims  = vec2<i32>(textureDimensions(t_src));
    if (coord.x >= dims.x || coord.y >= dims.y) { return; }

    let step = i32(u.step);
    let ao_c = textureLoad(t_src,   coord, 0).r;
    let d_c  = textureLoad(t_depth, coord, 0);

    let offsets = array<vec2<i32>, 5>(
        vec2<i32>(    0,     0),
        vec2<i32>(-step,     0),
        vec2<i32>( step,     0),
        vec2<i32>(    0, -step),
        vec2<i32>(    0,  step),
    );

    var sum_w  = 0.0;
    var sum_ao = 0.0;

    for (var i = 0u; i < 5u; i++) {
        let q = coord + offsets[i];
        if (any(q < vec2<i32>(0)) || any(q >= dims)) { continue; }

        let ao_q = textureLoad(t_src,   q, 0).r;
        let d_q  = textureLoad(t_depth, q, 0);

        let w_z = exp(-abs(d_q - d_c) / (u.sigma_z + 1e-6));
        let w_l = exp(-abs(ao_q - ao_c) / (u.sigma_l + 1e-6));
        let w   = w_z * w_l;

        sum_w  += w;
        sum_ao += w * ao_q;
    }

    let result = select(ao_c, sum_ao / sum_w, sum_w > 1e-6);
    textureStore(t_dst, coord, vec4<f32>(result, result, result, 1.0));
}
)";

AoDenoisePass::~AoDenoisePass() {
    for (uint32_t i = 0; i < kMaxIterations; ++i) {
        if (m_bindGroups[i]) wgpuBindGroupRelease(m_bindGroups[i]);
        if (m_ubo[i])        wgpuBufferRelease(m_ubo[i]);
    }
    if (m_depthView) wgpuTextureViewRelease(m_depthView);
    if (m_pongView)  wgpuTextureViewRelease(m_pongView);
    if (m_pong)      wgpuTextureRelease(m_pong);
    if (m_pingView)  wgpuTextureViewRelease(m_pingView);
    if (m_ping)      wgpuTextureRelease(m_ping);
    if (m_bgl)       wgpuBindGroupLayoutRelease(m_bgl);
    if (m_pipeline)  wgpuComputePipelineRelease(m_pipeline);
}

void AoDenoisePass::ensure_pipeline(WGPUDevice device) {
    if (m_pipeline) return;

    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code        = {kShader, WGPU_STRLEN};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain  = &wgslSrc.chain;
    WGPUShaderModule sm = wgpuDeviceCreateShaderModule(device, &smDesc);

    // Layout: src texture, dst storage texture, depth texture, UBO
    WGPUBindGroupLayoutEntry entries[4] = {};

    entries[0].binding               = 0;
    entries[0].visibility            = WGPUShaderStage_Compute;
    entries[0].texture.sampleType    = WGPUTextureSampleType_Float;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;

    entries[1].binding                       = 1;
    entries[1].visibility                    = WGPUShaderStage_Compute;
    entries[1].storageTexture.access         = WGPUStorageTextureAccess_WriteOnly;
    entries[1].storageTexture.format         = WGPUTextureFormat_RGBA8Unorm;
    entries[1].storageTexture.viewDimension  = WGPUTextureViewDimension_2D;

    entries[2].binding               = 2;
    entries[2].visibility            = WGPUShaderStage_Compute;
    entries[2].texture.sampleType    = WGPUTextureSampleType_Depth;
    entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

    entries[3].binding                  = 3;
    entries[3].visibility               = WGPUShaderStage_Compute;
    entries[3].buffer.type              = WGPUBufferBindingType_Uniform;
    entries[3].buffer.minBindingSize    = 16;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 4;
    bglDesc.entries    = entries;
    m_bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts     = &m_bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    WGPUComputePipelineDescriptor pDesc{};
    pDesc.layout             = pl;
    pDesc.compute.module     = sm;
    pDesc.compute.entryPoint = {"cs_main", WGPU_STRLEN};
    m_pipeline = wgpuDeviceCreateComputePipeline(device, &pDesc);

    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(sm);

    // Allocate UBOs (one per iteration slot, written later by update_ubos).
    WGPUBufferDescriptor bd{};
    bd.size  = 16;
    bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    for (uint32_t i = 0; i < kMaxIterations; ++i) {
        if (!m_ubo[i])
            m_ubo[i] = wgpuDeviceCreateBuffer(device, &bd);
    }
}

void AoDenoisePass::rebuild_textures(WGPUDevice device, uint32_t w, uint32_t h) {
    if (m_pingView) { wgpuTextureViewRelease(m_pingView); m_pingView = nullptr; }
    if (m_ping)     { wgpuTextureRelease(m_ping);         m_ping     = nullptr; }
    if (m_pongView) { wgpuTextureViewRelease(m_pongView); m_pongView = nullptr; }
    if (m_pong)     { wgpuTextureRelease(m_pong);         m_pong     = nullptr; }

    WGPUTextureDescriptor td{};
    td.usage         = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
    td.dimension     = WGPUTextureDimension_2D;
    td.size          = {w, h, 1};
    td.format        = WGPUTextureFormat_RGBA8Unorm;
    td.mipLevelCount = 1;
    td.sampleCount   = 1;

    m_ping     = wgpuDeviceCreateTexture(device, &td);
    m_pingView = wgpuTextureCreateView(m_ping, nullptr);
    m_pong     = wgpuDeviceCreateTexture(device, &td);
    m_pongView = wgpuTextureCreateView(m_pong, nullptr);

    m_width  = w;
    m_height = h;
}

void AoDenoisePass::rebuild_depth_view(WGPUTexture depthTex) {
    if (m_depthView) { wgpuTextureViewRelease(m_depthView); m_depthView = nullptr; }

    WGPUTextureViewDescriptor dvd{};
    dvd.format          = wgpuTextureGetFormat(depthTex);
    dvd.dimension       = WGPUTextureViewDimension_2D;
    dvd.baseMipLevel    = 0;
    dvd.mipLevelCount   = 1;
    dvd.baseArrayLayer  = 0;
    dvd.arrayLayerCount = 1;
    dvd.aspect          = WGPUTextureAspect_DepthOnly;
    m_depthView    = wgpuTextureCreateView(depthTex, &dvd);
    m_lastDepthTex = depthTex;
}

void AoDenoisePass::rebuild_bind_groups(WGPUDevice device, WGPUTextureView aoView) {
    for (uint32_t i = 0; i < kMaxIterations; ++i) {
        if (m_bindGroups[i]) { wgpuBindGroupRelease(m_bindGroups[i]); m_bindGroups[i] = nullptr; }
    }

    // Iteration 0:  source=aoView → dest=ping
    // Iteration 1:  source=ping   → dest=pong
    // Iteration 2:  source=pong   → dest=ping
    // ...alternating, so result for N iters is ping (odd N) or pong (even N).
    WGPUTextureView srcViews[kMaxIterations];
    WGPUTextureView dstViews[kMaxIterations];
    srcViews[0] = aoView;
    dstViews[0] = m_pingView;
    for (uint32_t i = 1; i < kMaxIterations; ++i) {
        srcViews[i] = (i % 2 == 1) ? m_pingView : m_pongView;
        dstViews[i] = (i % 2 == 1) ? m_pongView : m_pingView;
    }

    for (uint32_t i = 0; i < kMaxIterations; ++i) {
        WGPUBindGroupEntry e[4] = {};
        e[0].binding     = 0; e[0].textureView = srcViews[i];
        e[1].binding     = 1; e[1].textureView = dstViews[i];
        e[2].binding     = 2; e[2].textureView = m_depthView;
        e[3].binding     = 3; e[3].buffer      = m_ubo[i]; e[3].size = 16;

        WGPUBindGroupDescriptor bgDesc{};
        bgDesc.layout     = m_bgl;
        bgDesc.entryCount = 4;
        bgDesc.entries    = e;
        m_bindGroups[i] = wgpuDeviceCreateBindGroup(device, &bgDesc);
    }

    m_lastAoView = aoView;
}

void AoDenoisePass::update_ubos(WGPUDevice device, float sigmaZ, float sigmaL) {
    struct alignas(16) Params { uint32_t step; float sigmaZ; float sigmaL; float pad; };

    WGPUQueue q = wgpuDeviceGetQueue(device);
    for (uint32_t i = 0; i < kMaxIterations; ++i) {
        const Params p{1u << i, sigmaZ, sigmaL, 0.f};
        wgpuQueueWriteBuffer(q, m_ubo[i], 0, &p, sizeof(p));
    }
    wgpuQueueRelease(q);

    m_lastSigmaZ = sigmaZ;
    m_lastSigmaL = sigmaL;
}

WGPUTextureView AoDenoisePass::execute(WGPUDevice device, WGPUCommandEncoder encoder,
                                        WGPUTextureView aoView, WGPUTexture depthTex,
                                        uint32_t numIterations, float sigmaZ, float sigmaL) {
    if (numIterations == 0 || !aoView || !depthTex) return aoView;
    numIterations = numIterations < kMaxIterations ? numIterations : kMaxIterations;

    const uint32_t w = wgpuTextureGetWidth(depthTex);
    const uint32_t h = wgpuTextureGetHeight(depthTex);
    if (w == 0 || h == 0) return aoView;

    ensure_pipeline(device);
    if (!m_pipeline) return aoView;

    if (w != m_width || h != m_height)
        rebuild_textures(device, w, h);

    if (depthTex != m_lastDepthTex)
        rebuild_depth_view(depthTex);

    if (aoView != m_lastAoView || !m_bindGroups[0])
        rebuild_bind_groups(device, aoView);

    if (sigmaZ != m_lastSigmaZ || sigmaL != m_lastSigmaL)
        update_ubos(device, sigmaZ, sigmaL);

    const uint32_t gx = (w + 7) / 8;
    const uint32_t gy = (h + 7) / 8;

    // One compute pass per iteration — separate passes guarantee memory visibility
    // between the write (dst) and subsequent read (src) of the same ping/pong texture.
    for (uint32_t i = 0; i < numIterations; ++i) {
        WGPUComputePassDescriptor passDesc{};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, m_pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, m_bindGroups[i], 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // ping is the result for odd numIterations, pong for even.
    return (numIterations % 2 == 1) ? m_pingView : m_pongView;
}

} // namespace dusk::rtao
