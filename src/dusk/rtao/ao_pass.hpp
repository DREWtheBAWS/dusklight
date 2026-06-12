#pragma once
#include "geometry_collector.hpp"
#include <webgpu/webgpu.h>
#include "imgui.h"
#include <cstdint>

namespace dusk::rtao {

class AoPass {
public:
    struct Params {
        uint32_t raysPerPixel = 8;
        float    maxDistance  = 500.f;
        float    normalBias   = 0.01f;
        uint32_t debugMode    = 0;   // 0=AO, 1=normals, 2=depth, 3=root-AABB
        uint32_t debugMode2   = 0;   // 0=limit-hits,1=AO,2=normals,3=depth,4=root-AABB,5=visit-heat,6=limit%
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

    // Run the AO compute pass using the BLAS/TLAS two-level BVH.
    // Buffers come from TlasBuilder.  Opaque-only (no alpha test) in this phase.
    // Run the AO compute pass using the BLAS/TLAS two-level BVH (view-space).
    void execute_tlas(WGPUDevice device, WGPUCommandEncoder encoder,
                      WGPUTexture depthTex,
                      const GeometryCollector::CameraData& cam,
                      WGPUBuffer tlasNodeBuf, WGPUBuffer instanceBuf,
                      WGPUBuffer blasNodeBuf, WGPUBuffer blasTriBuf,
                      const std::vector<void*>& texViews);

    ImTextureID imgui_texture_id()   const;
    ImTextureID limits_texture_id()  const;
    bool     is_ready() const { return m_aoView != nullptr; }
    uint32_t width()    const { return m_width; }
    uint32_t height()   const { return m_height; }

private:
    void ensure_pipeline(WGPUDevice device);
    void ensure_tlas_pipeline(WGPUDevice device);
    void rebuild_output(WGPUDevice device, uint32_t w, uint32_t h);
    void rebuild_depth_binding(WGPUDevice device, WGPUTexture depthTex);
    void rebuild_bind_group(WGPUDevice device);
    void rebuild_tlas_bind_group(WGPUDevice device);

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

    // TLAS/BLAS path (Phase 3) — separate pipeline + bind group
    WGPUComputePipeline m_tlasPipeline           = nullptr;
    WGPUBindGroupLayout m_tlasBgl                = nullptr;
    WGPUBuffer          m_tlasLastNodeBuf        = nullptr;
    WGPUBuffer          m_tlasLastInstBuf        = nullptr;
    WGPUBuffer          m_tlasLastBlasNodeBuf    = nullptr;
    WGPUBuffer          m_tlasLastBlasTriBuf     = nullptr;
    WGPUBindGroup       m_tlasBindGroup          = nullptr;
    bool                m_tlasBindGroupDirty     = true;

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
