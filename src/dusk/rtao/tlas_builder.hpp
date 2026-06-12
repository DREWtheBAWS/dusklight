#pragma once
#include "blas_cache.hpp"
#include "bvh.hpp"
#include <webgpu/webgpu.h>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dusk::rtao {

// Builds a view-space Top-Level Acceleration Structure (TLAS) each frame from
// the BlasCache instance list.  Each qualifying draw-call instance becomes one
// TLAS leaf.  Produces four GPU buffers consumed by the Phase 3 AO shader:
//
//   tlas_node_buf  — BVH nodes over view-space AABBs (BvhNode / GpuNode format)
//   instance_buf   — per-instance pnMtxInv + BLAS offsets (GpuTlasInstance)
//   blas_node_buf  — monolithic BLAS node buffer (all cached BLASes concatenated)
//   blas_tri_buf   — monolithic BLAS tri buffer
//
// Call order each frame (from the post-render callback, after BlasCache::flush):
//   1. build(cache)  — CPU SAH BVH over view-space AABBs, matrix inversion
//   2. flush(device) — upload all four GPU buffers
//   3. advance_frame() — reset per-frame CPU state (from afterDraw)
class TlasBuilder {
public:
    // 64 bytes — pnMtxInv for ray transform + offsets into monolithic buffers.
    struct GpuTlasInstance {
        float    pnMtxInv[3][4]; // view→local (for ray direction transform)
        uint32_t blasNodeOffset; // first node index in monolithic blas_node_buf
        uint32_t blasTriOffset;  // first tri  index in monolithic blas_tri_buf
        uint32_t blasNodeCount;
        uint32_t blasTriCount;
    };
    static_assert(sizeof(GpuTlasInstance) == 64);

    struct Stats {
        uint32_t tlasNodeCount;
        uint32_t instanceCount;    // TLAS leaves after dedup
        uint32_t dedupRejected;    // exact (blasKey, pnMtx) duplicates removed
        uint32_t blasEntryCount;   // unique BLASes referenced by instances
        uint32_t blasNodeTotal;    // nodes in monolithic buffer
        uint32_t blasTriTotal;     // tris  in monolithic buffer
        float    rootAabbMin[3];
        float    rootAabbMax[3];
        uint64_t gpuBytesTotal;
        float    buildMs;          // CPU time for build() this frame (ms)
        float    flushMs;          // CPU time for flush() this frame (ms)
    };

    // On-demand structural validation — run once per frame when m_validateNext is set.
    struct Validation {
        bool     ran          = false;
        bool     tlasLeafOk   = true;
        bool     blasRangeOk  = true;
        uint32_t badLeaves    = 0;
        uint32_t badInstances = 0;
        uint32_t instanceCount = 0;
        uint32_t blasNodeBufSize = 0;
        uint32_t blasTriBufSize  = 0;
        uint32_t tlasNodeCount   = 0;
    };

    void request_validation() { m_validateNext = true; }
    const Validation& last_validation() const { return m_lastValidation; }

    ~TlasBuilder();

    // CPU phase: collect instances, compute view-space AABBs, build SAH BVH.
    // Call after BlasCache::flush() so all BLAS entries are populated.
    void build(const BlasCache& cache);

    // GPU phase: upload TLAS nodes, instance table, and (if cache changed) BLAS buffers.
    void flush(WGPUDevice device);

    // Reset per-frame CPU state.  Call from afterDraw().
    void advance_frame();

    Stats      last_stats()    const { return m_lastStats; }
    bool       is_ready()      const { return m_tlasNodeBuf != nullptr; }
    WGPUBuffer tlas_node_buf() const { return m_tlasNodeBuf; }
    WGPUBuffer instance_buf()  const { return m_instanceBuf; }
    WGPUBuffer blas_node_buf() const { return m_blasNodeBuf; }
    WGPUBuffer blas_tri_buf()  const { return m_blasTriBuf; }

    // Returns the view-space AABB of each TLAS leaf for debug overlays.
    void get_instance_view_aabbs(std::vector<AABB>& out) const {
        out.clear();
        out.reserve(m_instances.size());
        for (const auto& ci : m_instances) out.push_back(ci.viewAabb);
    }

private:
    struct CpuInstance {
        AABB     viewAabb;
        float    pnMtxInv[3][4];
        uint32_t blasNodeOffset;
        uint32_t blasTriOffset;
        uint32_t blasNodeCount;
        uint32_t blasTriCount;
    };

    std::vector<BvhNode>     m_tlasNodes;
    std::vector<CpuInstance> m_instances;

    // BLAS offset lookup: blasKey → (nodeOffset, triOffset)
    std::unordered_map<BlasKey, std::pair<uint32_t,uint32_t>, BlasKeyHash> m_blasOffsets;
    uint32_t m_lastBlasGeneration     = UINT32_MAX;
    uint32_t m_lastEvictionGeneration = UINT32_MAX;

    // Staging data for static monolithic BLAS buffers (rebuilt incrementally).
    std::vector<BlasCache::GpuNode> m_pendingBlasNodes;
    std::vector<BlasCache::GpuTri>  m_pendingBlasTris;
    bool     m_blasNodesDirty      = false;
    uint32_t m_staticBlasNodeCount = 0;
    uint32_t m_staticBlasTriCount  = 0;

    // Staging data for dynamic (skinned) BLAS.
    std::vector<BlasCache::GpuNode> m_dynBlasNodes;
    std::vector<BlasCache::GpuTri>  m_dynBlasTris;

    // Combined static+dynamic GPU buffer capacity (in node / tri count, not bytes).
    uint32_t m_blasNodeBufCapacity = 0;
    uint32_t m_blasTriBufCapacity  = 0;

    // Per-frame GPU buffers (uploaded each frame)
    WGPUBuffer m_tlasNodeBuf = nullptr;
    WGPUBuffer m_instanceBuf = nullptr;

    // Combined BLAS buffer (static portion stable; dynamic tail re-written each frame).
    WGPUBuffer m_blasNodeBuf = nullptr;
    WGPUBuffer m_blasTriBuf  = nullptr;

    // Per-frame instance dedup.
    std::unordered_set<size_t> m_dedupSeen;

    Stats      m_lastStats{};
    Validation m_lastValidation{};
    bool       m_validateNext = false;
};

} // namespace dusk::rtao
