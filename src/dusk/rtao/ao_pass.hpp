#pragma once
#include "geometry_collector.hpp"
#include <webgpu/webgpu.h>
#include "imgui.h"
#include <cstdint>

namespace dusk::rtao {

class AoPass {
public:
    struct Params {
        uint32_t raysPerPixel    = 8;
        float    maxDistance     = 500.f;
        float    normalBias      = 0.01f;
        uint32_t debugMode       = 0;   // 0=AO, 1=normals, 2=depth, 3=root-AABB
        uint32_t debugMode2      = 0;   // 0=limit-hits,1=AO,2=normals,3=depth,4=root-AABB,5=visit-heat,6=limit%
        float    shadowConeRadius = 0.02f; // half-angle (radians) of the sun disk for soft shadows
    };

    ~AoPass();

    void set_params(const Params& p) { m_params = p; }

    // Run the AO compute pass using the GPU LBVH (single-level BVH, view space).
    // nodeBuf and triBuf come from GpuBvhBuilder each frame.
    void execute(WGPUDevice device, WGPUCommandEncoder encoder,
                 WGPUTexture depthTex,
                 const GeometryCollector::CameraData& cam,
                 WGPUBuffer nodeBuf, WGPUBuffer triBuf,
                 const std::vector<void*>& texViews);

    // Run the AO compute pass using the BLAS/TLAS two-level BVH (static geometry) plus
    // an optional GPU LBVH for dynamic (skinned) geometry in view space.
    // dynNodeBuf/dynTriBuf may be null when there is no skinned geometry this frame;
    // dynNodeCount must be 0 in that case.
    void execute_tlas(WGPUDevice device, WGPUCommandEncoder encoder,
                      WGPUTexture depthTex,
                      const GeometryCollector::CameraData& cam,
                      WGPUBuffer tlasNodeBuf, WGPUBuffer instanceBuf,
                      WGPUBuffer blasNodeBuf, WGPUBuffer blasTriBuf,
                      WGPUBuffer dynNodeBuf, WGPUBuffer dynTriBuf, uint32_t dynNodeCount,
                      const std::vector<void*>& texViews);

    // Run the shadow compute pass using the same BLAS/TLAS BVH as execute_tlas.
    // Traces one ray per pixel from the surface toward cam.lightWorldPos and writes
    // 0=shadowed / 1=lit to shadow_texture_view().
    // cam.lightWorldPos must be set (via GeometryCollector::set_light_world_pos) before calling.
    void execute_shadow_tlas(WGPUDevice device, WGPUCommandEncoder encoder,
                             WGPUTexture depthTex,
                             const GeometryCollector::CameraData& cam,
                             WGPUBuffer tlasNodeBuf, WGPUBuffer instanceBuf,
                             WGPUBuffer blasNodeBuf, WGPUBuffer blasTriBuf,
                             WGPUBuffer dynNodeBuf, WGPUBuffer dynTriBuf, uint32_t dynNodeCount);

    ImTextureID     imgui_texture_id()       const;
    ImTextureID     limits_texture_id()      const;
    ImTextureID     shadow_imgui_texture_id() const;
    WGPUTextureView ao_texture_view()        const { return m_aoView; }
    // Returns null until the first shadow dispatch has completed (uninitialized texture
    // would composite as garbage; the composite falls back to white = no effect when null).
    WGPUTextureView shadow_texture_view()    const { return m_shadowReady ? m_shadowView : nullptr; }
    bool     is_ready()      const { return m_aoView != nullptr; }
    bool     shadow_ready()  const { return m_shadowReady; }
    uint32_t width()         const { return m_width; }
    uint32_t height()        const { return m_height; }
    // Last view-space light position uploaded to the shadow UBO (for debug display).
    void last_light_view_pos(float out[3]) const {
        out[0] = m_lastLightViewPos[0];
        out[1] = m_lastLightViewPos[1];
        out[2] = m_lastLightViewPos[2];
    }

private:
    void ensure_pipeline(WGPUDevice device);
    void ensure_tlas_pipeline(WGPUDevice device);
    void ensure_shadow_pipeline(WGPUDevice device);
    void rebuild_output(WGPUDevice device, uint32_t w, uint32_t h);
    void rebuild_depth_binding(WGPUDevice device, WGPUTexture depthTex);
    void rebuild_bind_group(WGPUDevice device);
    void rebuild_tlas_bind_group(WGPUDevice device);
    void rebuild_shadow_bind_group(WGPUDevice device);

    Params m_params{};

    WGPUComputePipeline m_pipeline = nullptr;
    WGPUBindGroupLayout m_bgl      = nullptr;

    // Externally owned BVH buffers (from GpuBvhBuilder, not released here)
    WGPUBuffer m_lastNodeBuf = nullptr;
    WGPUBuffer m_lastTriBuf  = nullptr;

    WGPUBuffer m_cameraUbo = nullptr;

    WGPUTexture     m_aoTex  = nullptr;
    WGPUTextureView m_aoView = nullptr;
    WGPUTexture     m_limTex  = nullptr;
    WGPUTextureView m_limView = nullptr;

    WGPUTextureView m_depthView    = nullptr;
    WGPUTexture     m_lastDepthTex = nullptr;

    WGPUBindGroup m_bindGroup      = nullptr;
    bool          m_bindGroupDirty = true;

    // TLAS/BLAS AO path — separate pipeline + bind group
    WGPUComputePipeline m_tlasPipeline           = nullptr;
    WGPUBindGroupLayout m_tlasBgl                = nullptr;
    WGPUBuffer          m_tlasLastNodeBuf        = nullptr;
    WGPUBuffer          m_tlasLastInstBuf        = nullptr;
    WGPUBuffer          m_tlasLastBlasNodeBuf    = nullptr;
    WGPUBuffer          m_tlasLastBlasTriBuf     = nullptr;
    WGPUBuffer          m_tlasLastDynNodeBuf     = nullptr;  // nullable; dynamic LBVH nodes
    WGPUBuffer          m_tlasLastDynTriBuf      = nullptr;  // nullable; dynamic LBVH tris
    WGPUBuffer          m_dynDummyNodeBuf        = nullptr;  // 4-byte fallback when no dynamic geo
    WGPUBuffer          m_dynDummyTriBuf         = nullptr;
    WGPUBindGroup       m_tlasBindGroup          = nullptr;
    bool                m_tlasBindGroupDirty     = true;

    // Shadow pass — same BVH buffers, own pipeline + camera UBO + output texture
    WGPUComputePipeline m_shadowPipeline         = nullptr;
    WGPUBindGroupLayout m_shadowBgl              = nullptr;
    WGPUBuffer          m_shadowCamUbo           = nullptr;
    WGPUBuffer          m_shadowLastNodeBuf      = nullptr;
    WGPUBuffer          m_shadowLastInstBuf      = nullptr;
    WGPUBuffer          m_shadowLastBlasNodeBuf  = nullptr;
    WGPUBuffer          m_shadowLastBlasTriBuf   = nullptr;
    WGPUBuffer          m_shadowLastDynNodeBuf   = nullptr;
    WGPUBuffer          m_shadowLastDynTriBuf    = nullptr;
    WGPUBuffer          m_shadowDummyNodeBuf     = nullptr;
    WGPUBuffer          m_shadowDummyTriBuf      = nullptr;
    WGPUBindGroup       m_shadowBindGroup        = nullptr;
    bool                m_shadowBindGroupDirty   = true;
    WGPUTexture         m_shadowTex              = nullptr;
    WGPUTextureView     m_shadowView             = nullptr;
    bool                m_shadowReady            = false;  // true after first dispatch
    float               m_lastLightViewPos[3]    = {};     // for debug display

    // Alpha-texture sampler + 1×1 opaque-white fallback (pad empty slots).
    WGPUSampler     m_alphaSampler = nullptr;
    WGPUTexture     m_fallbackTex  = nullptr;
    WGPUTextureView m_fallbackView = nullptr;
    // Per-frame snapshot of texture_views() — invalidates bind group on change.
    std::vector<void*> m_lastTexViews;

    uint32_t m_width  = 0;
    uint32_t m_height = 0;
    uint32_t m_frame  = 0;
};

} // namespace dusk::rtao
