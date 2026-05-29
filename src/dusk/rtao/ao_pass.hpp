#pragma once
#include "bvh.hpp"
#include "geometry_collector.hpp"
#include <webgpu/webgpu.h>
#include "imgui.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace dusk::rtao {

class AoPass {
public:
    struct Params {
        uint32_t raysPerPixel = 8;
        float    maxDistance  = 500.f;
        // View-space units to push the ray origin off the surface before casting.
        // Scales with camera distance: bias = max(dist * normalBias, normalBias * 10).
        // Increase if close surfaces are fully occluded (self-intersection); decrease
        // if thin geometry leaks.  Tune to match the game's world-unit scale.
        float    normalBias   = 0.01f;
    };

    ~AoPass();

    // Queue a BVH upload. The GPU buffers are created on the next execute() call.
    // Safe to call from the ImGui thread (no device needed here).
    void queue_bvh_upload(const Bvh& bvh);

    // Store params to use on every execute().
    void set_params(const Params& p) { m_params = p; }

    // Run the AO compute pass. Returns without doing anything if BVH hasn't been
    // uploaded yet or the camera data is invalid.
    void execute(WGPUDevice device, WGPUCommandEncoder encoder,
                 WGPUTexture depthTex,
                 const GeometryCollector::CameraData& cam);

    ImTextureID imgui_texture_id() const;
    bool     is_ready()  const { return m_aoView != nullptr; }
    uint32_t width()     const { return m_width; }
    uint32_t height()    const { return m_height; }

private:
    void ensure_pipeline(WGPUDevice device);
    void flush_pending_upload(WGPUDevice device);
    void rebuild_output(WGPUDevice device, uint32_t w, uint32_t h);
    void rebuild_depth_binding(WGPUDevice device, WGPUTexture depthTex);
    void rebuild_bind_group(WGPUDevice device);

    Params m_params{};

    WGPUComputePipeline m_pipeline = nullptr;
    WGPUBindGroupLayout m_bgl      = nullptr;

    // CPU-side pending BVH data (set by queue_bvh_upload, consumed by execute)
    struct PendingBvh {
        std::vector<uint8_t> nodeData; // packed GpuBvhNode array
        std::vector<uint8_t> triData;  // packed GpuTriangle array
        uint32_t nodeCount = 0;
        uint32_t triCount  = 0;
    };
    std::unique_ptr<PendingBvh> m_pendingBvh;

    WGPUBuffer  m_nodeBuf   = nullptr;
    WGPUBuffer  m_triBuf    = nullptr;
    uint32_t    m_nodeCount = 0;
    uint32_t    m_triCount  = 0;

    WGPUBuffer  m_cameraUbo = nullptr;

    WGPUTexture     m_aoTex  = nullptr;
    WGPUTextureView m_aoView = nullptr;

    WGPUTextureView m_depthView    = nullptr;
    WGPUTexture     m_lastDepthTex = nullptr;

    WGPUBindGroup m_bindGroup      = nullptr;
    bool          m_bindGroupDirty = true;

    uint32_t m_width  = 0;
    uint32_t m_height = 0;
    uint32_t m_frame  = 0;
};

} // namespace dusk::rtao
