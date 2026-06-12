#include "ao_composite.hpp"

namespace dusk::rtao {

// Full-screen triangle + AO multiply shader.
// Samples the EFB color and the AO texture; outputs color * mix(1, ao, strength).
// The EFB is read as a sampled texture; the composite result lands in a scratch
// texture that is then copied back into the EFB (avoiding read-write aliasing).
static const char kShader[] = R"(
struct VertOut {
    @builtin(position) pos : vec4<f32>,
    @location(0)       uv  : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VertOut {
    var tri = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = tri[vi];
    var o : VertOut;
    o.pos = vec4<f32>(p, 0.0, 1.0);
    // NDC → UV: x [-1,1]→[0,1], y [1,-1]→[0,1] (GPU textures are top-left origin)
    o.uv  = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return o;
}

@group(0) @binding(0) var u_sampler : sampler;
@group(0) @binding(1) var t_color   : texture_2d<f32>;
@group(0) @binding(2) var t_ao      : texture_2d<f32>;

struct Params { strength : f32 };
@group(0) @binding(3) var<uniform> u_params : Params;

@fragment
fn fs_main(in : VertOut) -> @location(0) vec4<f32> {
    let color = textureSample(t_color, u_sampler, in.uv);
    let ao    = textureSample(t_ao,    u_sampler, in.uv).r;
    let scale = mix(1.0, ao, u_params.strength);
    return vec4<f32>(color.rgb * scale, color.a);
}
)";

AoCompositePass::~AoCompositePass() {
    if (m_bindGroup)       wgpuBindGroupRelease(m_bindGroup);
    if (m_bgl)             wgpuBindGroupLayoutRelease(m_bgl);
    if (m_pipeline)        wgpuRenderPipelineRelease(m_pipeline);
    if (m_sampler)         wgpuSamplerRelease(m_sampler);
    if (m_paramsUbo)       wgpuBufferRelease(m_paramsUbo);
    if (m_scratchView)     wgpuTextureViewRelease(m_scratchView);
    if (m_scratchTex)      wgpuTextureRelease(m_scratchTex);
    if (m_colorSampledView) wgpuTextureViewRelease(m_colorSampledView);
}

void AoCompositePass::ensure_pipeline(WGPUDevice device, WGPUTextureFormat fmt) {
    if (m_pipeline && m_fmt == fmt) return;

    if (m_bindGroup) { wgpuBindGroupRelease(m_bindGroup); m_bindGroup = nullptr; }
    if (m_bgl)       { wgpuBindGroupLayoutRelease(m_bgl); m_bgl       = nullptr; }
    if (m_pipeline)  { wgpuRenderPipelineRelease(m_pipeline); m_pipeline = nullptr; }

    m_fmt = fmt;

    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code        = {kShader, WGPU_STRLEN};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain  = &wgslSrc.chain;
    WGPUShaderModule sm = wgpuDeviceCreateShaderModule(device, &smDesc);

    WGPUBindGroupLayoutEntry bglEntries[4] = {};

    bglEntries[0].binding      = 0;
    bglEntries[0].visibility   = WGPUShaderStage_Fragment;
    bglEntries[0].sampler.type = WGPUSamplerBindingType_Filtering;

    bglEntries[1].binding               = 1;
    bglEntries[1].visibility            = WGPUShaderStage_Fragment;
    bglEntries[1].texture.sampleType    = WGPUTextureSampleType_Float;
    bglEntries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

    bglEntries[2].binding               = 2;
    bglEntries[2].visibility            = WGPUShaderStage_Fragment;
    bglEntries[2].texture.sampleType    = WGPUTextureSampleType_Float;
    bglEntries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

    bglEntries[3].binding             = 3;
    bglEntries[3].visibility          = WGPUShaderStage_Fragment;
    bglEntries[3].buffer.type         = WGPUBufferBindingType_Uniform;
    bglEntries[3].buffer.minBindingSize = sizeof(float);

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 4;
    bglDesc.entries    = bglEntries;
    m_bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts     = &m_bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    WGPUColorTargetState colorTarget{};
    colorTarget.format    = fmt;
    colorTarget.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState frag{};
    frag.module      = sm;
    frag.entryPoint  = {"fs_main", WGPU_STRLEN};
    frag.targetCount = 1;
    frag.targets     = &colorTarget;

    WGPURenderPipelineDescriptor rpDesc{};
    rpDesc.layout                  = pl;
    rpDesc.vertex.module           = sm;
    rpDesc.vertex.entryPoint       = {"vs_main", WGPU_STRLEN};
    rpDesc.primitive.topology      = WGPUPrimitiveTopology_TriangleList;
    rpDesc.multisample.count       = 1;
    rpDesc.multisample.mask        = 0xFFFFFFFF;
    rpDesc.fragment                = &frag;
    m_pipeline = wgpuDeviceCreateRenderPipeline(device, &rpDesc);

    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(sm);

    if (!m_sampler) {
        WGPUSamplerDescriptor sd{};
        sd.minFilter    = WGPUFilterMode_Linear;
        sd.magFilter    = WGPUFilterMode_Linear;
        sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
        sd.addressModeU = WGPUAddressMode_ClampToEdge;
        sd.addressModeV = WGPUAddressMode_ClampToEdge;
        m_sampler = wgpuDeviceCreateSampler(device, &sd);
    }

    if (!m_paramsUbo) {
        WGPUBufferDescriptor bd{};
        bd.size  = 16; // float strength + padding to meet 16-byte uniform alignment
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        m_paramsUbo = wgpuDeviceCreateBuffer(device, &bd);
    }
}

void AoCompositePass::rebuild_scratch(WGPUDevice device, uint32_t w, uint32_t h,
                                       WGPUTextureFormat fmt) {
    if (m_scratchView) { wgpuTextureViewRelease(m_scratchView); m_scratchView = nullptr; }
    if (m_scratchTex)  { wgpuTextureRelease(m_scratchTex);      m_scratchTex  = nullptr; }

    WGPUTextureDescriptor td{};
    td.usage         = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    td.dimension     = WGPUTextureDimension_2D;
    td.size          = {w, h, 1};
    td.format        = fmt;
    td.mipLevelCount = 1;
    td.sampleCount   = 1;
    m_scratchTex  = wgpuDeviceCreateTexture(device, &td);
    m_scratchView = wgpuTextureCreateView(m_scratchTex, nullptr);

    m_width     = w;
    m_height    = h;
    m_scratchFmt = fmt;
}

void AoCompositePass::rebuild_bind_group(WGPUDevice device) {
    if (m_bindGroup) { wgpuBindGroupRelease(m_bindGroup); m_bindGroup = nullptr; }

    WGPUBindGroupEntry entries[4] = {};
    entries[0].binding     = 0;
    entries[0].sampler     = m_sampler;
    entries[1].binding     = 1;
    entries[1].textureView = m_colorSampledView;
    entries[2].binding     = 2;
    entries[2].textureView = m_lastAoView;
    entries[3].binding     = 3;
    entries[3].buffer      = m_paramsUbo;
    entries[3].size        = 16;

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout     = m_bgl;
    bgDesc.entryCount = 4;
    bgDesc.entries    = entries;
    m_bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);
}

void AoCompositePass::execute(WGPUDevice device, WGPUCommandEncoder encoder,
                               WGPUTexture colorTex, WGPUTextureView aoView,
                               float strength) {
    if (!colorTex || !aoView) return;

    const uint32_t          w   = wgpuTextureGetWidth(colorTex);
    const uint32_t          h   = wgpuTextureGetHeight(colorTex);
    const WGPUTextureFormat fmt = wgpuTextureGetFormat(colorTex);
    if (w == 0 || h == 0) return;

    ensure_pipeline(device, fmt);
    if (!m_pipeline) return;

    if (w != m_width || h != m_height || fmt != m_scratchFmt)
        rebuild_scratch(device, w, h, fmt);

    bool bgDirty = false;
    if (colorTex != m_lastColorTex) {
        if (m_colorSampledView) { wgpuTextureViewRelease(m_colorSampledView); m_colorSampledView = nullptr; }
        m_colorSampledView = wgpuTextureCreateView(colorTex, nullptr);
        m_lastColorTex = colorTex;
        bgDirty = true;
    }
    if (aoView != m_lastAoView) {
        m_lastAoView = aoView;
        bgDirty = true;
    }
    if (bgDirty || !m_bindGroup)
        rebuild_bind_group(device);

    // Update strength uniform (16 bytes: float + 12 bytes padding).
    struct alignas(16) ParamsData { float strength; float pad[3]; };
    const ParamsData params{strength, {}};
    WGPUQueue q = wgpuDeviceGetQueue(device);
    wgpuQueueWriteBuffer(q, m_paramsUbo, 0, &params, sizeof(params));
    wgpuQueueRelease(q);

    // Render pass: sample colorTex → write to scratch.
    // (colorTex cannot be simultaneously sampled and rendered into; scratch avoids aliasing.)
    WGPURenderPassColorAttachment att{};
    att.view       = m_scratchView;
    att.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    att.loadOp     = WGPULoadOp_Clear;
    att.storeOp    = WGPUStoreOp_Store;
    att.clearValue = {0.0, 0.0, 0.0, 1.0};

    WGPURenderPassDescriptor passDesc{};
    passDesc.colorAttachmentCount   = 1;
    passDesc.colorAttachments       = &att;

    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
    wgpuRenderPassEncoderSetPipeline(rp, m_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, m_bindGroup, 0, nullptr);
    wgpuRenderPassEncoderDraw(rp, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);

    // Copy scratch → EFB so Aurora presents the AO-applied frame.
    WGPUTexelCopyTextureInfo src{};
    src.texture  = m_scratchTex;
    src.mipLevel = 0;
    src.origin   = {0, 0, 0};
    src.aspect   = WGPUTextureAspect_All;

    WGPUTexelCopyTextureInfo dst{};
    dst.texture  = colorTex;
    dst.mipLevel = 0;
    dst.origin   = {0, 0, 0};
    dst.aspect   = WGPUTextureAspect_All;

    const WGPUExtent3D extent{w, h, 1};
    wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &extent);
}

} // namespace dusk::rtao
