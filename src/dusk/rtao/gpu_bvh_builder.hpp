#pragma once
#include "vertex_decoder.hpp"
#include <webgpu/webgpu.h>
#include <cstdint>
#include <vector>

namespace dusk::rtao {

// Builds a Linear BVH (LBVH) on the GPU every frame from the raw triangle list
// captured by GeometryCollector.  The output node buffer and sorted triangle
// buffer are consumed directly by AoPass — no CPU readback required.
//
// Call order each frame (inside the post-render callback):
//   1. upload_triangles(device, tris)  — packs CPU data into pending upload buffer
//   2. build(device)                   — creates own encoder, submits, flushes BVH
//   3. ao_pass.execute(...)            — recorded into Aurora's encoder; BVH data
//                                        guaranteed visible because build() submitted first
class GpuBvhBuilder {
public:
    struct Stats {
        uint32_t triCount  = 0;
        uint32_t nodeCount = 0;
        float    buildMs      = 0.f;
        float    sceneMin[3]  = {};   // full AABB (includes outliers)
        float    sceneMax[3]  = {};
        float    mortonMin[3] = {};   // robust mean±3σ AABB used for Morton coding
        float    mortonMax[3] = {};
    };

    ~GpuBvhBuilder();

    // Pack raw triangles into the pending upload buffer.  Resizes GPU buffers
    // when the triangle count grows.  Safe to call with an empty span (skips).
    // Does NOT touch the GPU — actual upload happens inside build() via the
    // encoder so Dawn handles COPY_DST→storage barriers within one command buffer.
    void upload_triangles(WGPUDevice device, const std::vector<Triangle>& tris);

    // Build the BVH: creates its own WGPUCommandEncoder, records all passes and
    // uploads, finishes and submits to the device queue, then releases the encoder.
    // By submitting in a dedicated command buffer BEFORE the caller's encoder is
    // submitted, the GPU queue ordering guarantees all BVH writes are visible to
    // the AO pass (recorded in the caller's encoder) without relying on intra-
    // command-buffer UAV barriers — which proved unreliable on D3D12 drivers for
    // consecutive storage-buffer write → read pairs within the same submission.
    void build(WGPUDevice device);

    // Set the half-extent used for Morton AABB clamping.
    // Geometry beyond this distance from the camera gets Morton codes at the
    // extremes of the sorted order instead of degrading quantisation resolution.
    // Good default: 4–6 × AO max_distance.
    void set_morton_range(float r) { m_mortonRange = r; }

    // Camera world position — used to centre the Morton AABB in world space.
    // Must be set each frame before build() when using a world-space BVH.
    void set_world_pos(float x, float y, float z) {
        m_worldPos[0] = x; m_worldPos[1] = y; m_worldPos[2] = z;
    }

    // Output buffers consumed by AoPass.  Always valid after the first build()
    // call.  Because build() submits its own command buffer before returning,
    // D3D12 implicit state promotion guarantees the buffers are in COMMON state
    // (readable as SRV/UAV) by the time the caller's encoder runs.
    WGPUBuffer node_buf()  const { return m_nodeBuf; }
    WGPUBuffer tri_buf()   const { return m_triBuf;  }
    uint32_t   tri_count() const { return m_triCount; }

    bool   is_ready() const { return m_nodeBuf != nullptr && m_triCount > 0; }
    Stats  last_stats() const { return m_lastStats; }

private:
    void ensure_pipelines(WGPUDevice device);
    void resize_buffers(WGPUDevice device, uint32_t n);

    // ---- GPU buffers ----
    WGPUBuffer m_triInputBuf  = nullptr;  // compact input: 9 floats/tri
    WGPUBuffer m_aabbBuf      = nullptr;  // per-tri AABB: 6 floats
    WGPUBuffer m_sceneAabbBuf = nullptr;  // 6 floats (global bounds)
    WGPUBuffer m_mortonBuf    = nullptr;  // u32 Morton codes
    WGPUBuffer m_indicesBuf   = nullptr;  // u32 sort permutation
    WGPUBuffer m_nodeBuf      = nullptr;  // GpuBvhNode output (48B each)

    WGPUBuffer m_triBuf = nullptr;

    // One buffer serves as all params: slot 0 = base {tri_count, n_padded, 0, 0},
    // slots 1..N = bitonic (k,j) steps.  Each dispatch uses a dynamic offset to
    // select its slot — no CopyBufferToBuffer between passes required.
    WGPUBuffer m_sortStepsBuf = nullptr;

    std::vector<float> m_pendingTriData;  // packed CPU-side triangles (9 floats each)

    uint32_t m_bufCapacity  = 0;     // allocated capacity (triangles)
    uint32_t m_triCount     = 0;     // current frame triangle count
    float    m_mortonRange  = 4000.f; // ±half-extent for Morton AABB clamp (see set_morton_range)
    float    m_worldPos[3]  = {};    // camera world position for Morton AABB centering

    // ---- Compute pipelines ----
    WGPUComputePipeline m_pipeBounds    = nullptr;
    WGPUComputePipeline m_pipeReduce    = nullptr;
    WGPUComputePipeline m_pipeMorton    = nullptr;
    WGPUComputePipeline m_pipeBitonic   = nullptr;
    WGPUComputePipeline m_pipeLeaves    = nullptr;  // initialise leaf nodes
    WGPUComputePipeline m_pipeLbvhAabb  = nullptr;  // Karras hierarchy + AABB fit (merged)

    WGPUBindGroupLayout m_bgl = nullptr;

    Stats m_lastStats{};
};

} // namespace dusk::rtao
