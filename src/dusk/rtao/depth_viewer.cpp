#include "depth_viewer.hpp"
#include <aurora/post_render.h>

namespace dusk::rtao {

static const char kShader[] = R"(
struct VSOut { @builtin(position) pos : vec4<f32> };

@vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
    var tri = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var o : VSOut;
    o.pos = vec4<f32>(tri[vi], 0.0, 1.0);
    return o;
}

@group(0) @binding(0) var t_depth : texture_depth_2d;

@fragment fn fs_main(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    let d = textureLoad(t_depth, vec2<i32>(i32(pos.x), i32(pos.y)), 0);
    // Gamma remap makes near-camera geometry (d≈0) visible; raw depth is very
    // non-linear so without this most of the image appears white.
    let v = pow(d, 0.1);
    return vec4<f32>(v, v, v, 1.0);
}
)";

DepthTextureViewer::~DepthTextureViewer() {
    if (m_bindGroup)  wgpuBindGroupRelease(m_bindGroup);
    if (m_depthView)  wgpuTextureViewRelease(m_depthView);
    if (m_outputView) wgpuTextureViewRelease(m_outputView);
    if (m_outputTex)  wgpuTextureRelease(m_outputTex);
    if (m_bgl)        wgpuBindGroupLayoutRelease(m_bgl);
    if (m_pipeline)   wgpuRenderPipelineRelease(m_pipeline);
}

ImTextureID DepthTextureViewer::imgui_texture_id() const {
    return reinterpret_cast<ImTextureID>(m_outputView);
}

void DepthTextureViewer::ensure_pipeline(WGPUDevice device) {
    if (m_pipeline) return;

    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code = {kShader, WGPU_STRLEN};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslSrc.chain;
    WGPUShaderModule sm = wgpuDeviceCreateShaderModule(device, &smDesc);

    WGPUBindGroupLayoutEntry bglEntry{};
    bglEntry.binding = 0;
    bglEntry.visibility = WGPUShaderStage_Fragment;
    bglEntry.texture.sampleType = WGPUTextureSampleType_Depth;
    bglEntry.texture.viewDimension = WGPUTextureViewDimension_2D;
    bglEntry.texture.multisampled = 0;
    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 1;
    bglDesc.entries = &bglEntry;
    m_bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &m_bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    WGPUColorTargetState colorTarget{};
    colorTarget.format = WGPUTextureFormat_RGBA8Unorm;
    colorTarget.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState frag{};
    frag.module = sm;
    frag.entryPoint = {"fs_main", WGPU_STRLEN};
    frag.targetCount = 1;
    frag.targets = &colorTarget;

    WGPURenderPipelineDescriptor rpDesc{};
    rpDesc.layout = pl;
    rpDesc.vertex.module = sm;
    rpDesc.vertex.entryPoint = {"vs_main", WGPU_STRLEN};
    rpDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    rpDesc.multisample.count = 1;
    rpDesc.multisample.mask = 0xFFFFFFFF;
    rpDesc.fragment = &frag;
    m_pipeline = wgpuDeviceCreateRenderPipeline(device, &rpDesc);

    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(sm);
}

void DepthTextureViewer::execute(WGPUDevice device, WGPUCommandEncoder encoder) {
    WGPUTexture depthTex = aurora_get_depth_texture();
    if (!depthTex) return;

    const uint32_t w = wgpuTextureGetWidth(depthTex);
    const uint32_t h = wgpuTextureGetHeight(depthTex);
    if (w == 0 || h == 0) return;

    ensure_pipeline(device);

    if (w != m_width || h != m_height || depthTex != m_lastDepthTex) {
        if (m_bindGroup)  { wgpuBindGroupRelease(m_bindGroup);   m_bindGroup  = nullptr; }
        if (m_depthView)  { wgpuTextureViewRelease(m_depthView); m_depthView  = nullptr; }
        if (m_outputView) { wgpuTextureViewRelease(m_outputView); m_outputView = nullptr; }
        if (m_outputTex)  { wgpuTextureRelease(m_outputTex);     m_outputTex  = nullptr; }

        WGPUTextureDescriptor outDesc{};
        outDesc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
        outDesc.dimension = WGPUTextureDimension_2D;
        outDesc.size = {w, h, 1};
        outDesc.format = WGPUTextureFormat_RGBA8Unorm;
        outDesc.mipLevelCount = 1;
        outDesc.sampleCount = 1;
        m_outputTex = wgpuDeviceCreateTexture(device, &outDesc);
        m_outputView = wgpuTextureCreateView(m_outputTex, nullptr);

        // Need a DepthOnly aspect view for texture_depth_2d binding in the shader.
        WGPUTextureViewDescriptor dvDesc{};
        dvDesc.format = wgpuTextureGetFormat(depthTex);
        dvDesc.dimension = WGPUTextureViewDimension_2D;
        dvDesc.baseMipLevel = 0;
        dvDesc.mipLevelCount = 1;
        dvDesc.baseArrayLayer = 0;
        dvDesc.arrayLayerCount = 1;
        dvDesc.aspect = WGPUTextureAspect_DepthOnly;
        m_depthView = wgpuTextureCreateView(depthTex, &dvDesc);

        WGPUBindGroupEntry bgEntry{};
        bgEntry.binding = 0;
        bgEntry.textureView = m_depthView;
        WGPUBindGroupDescriptor bgDesc{};
        bgDesc.layout = m_bgl;
        bgDesc.entryCount = 1;
        bgDesc.entries = &bgEntry;
        m_bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

        m_width = w;
        m_height = h;
        m_lastDepthTex = depthTex;
    }

    WGPURenderPassColorAttachment att{};
    att.view = m_outputView;
    att.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    att.loadOp = WGPULoadOp_Clear;
    att.storeOp = WGPUStoreOp_Store;
    att.clearValue = {0.0, 0.0, 0.0, 1.0};

    WGPURenderPassDescriptor passDesc{};
    passDesc.colorAttachmentCount = 1;
    passDesc.colorAttachments = &att;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
    wgpuRenderPassEncoderSetPipeline(pass, m_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, m_bindGroup, 0, nullptr);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

} // namespace dusk::rtao
