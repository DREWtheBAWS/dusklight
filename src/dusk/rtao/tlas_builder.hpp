#pragma once
#include "blas_cache.hpp"
#include "bvh.hpp"
#include <webgpu/webgpu.h>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dusk::rtao {

// Builds a world-space Top-Level Acceleration Structure (TLAS) each frame from
// the BlasCache instance list.  Each qualifying draw-call instance becomes one
// TLAS leaf.  Produces four GPU buffers consumed by the Phase 3 AO shader:
//
//   tlas_node_buf  — BVH nodes over world-space AABBs (BvhNode / GpuNode format)
//   instance_buf   — per-instance pnMtxInv + BLAS offsets (GpuTlasInstance)
//   blas_node_buf  — monolithic BLAS node buffer (all cached BLASes concatenated)
//   blas_tri_buf   — monolithic BLAS tri buffer
//
// Call order each frame (from the post-render callback, after BlasCache::flush):
//   1. build(cache, viewMtx) — CPU SAH BVH over world-space AABBs, matrix inversion
//   2. flush(device)         — upload all four GPU buffers
//   3. advance_frame()       — reset per-frame CPU state (from afterDraw)
class TlasBuilder {
public:
    // 64 bytes — pnMtxInv for ray transform + offsets into monolithic buffers.
    struct GpuTlasInstance {
        float    pnMtxInv[3][4]; // view→local (BLAS rays are always in view space)
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
        bool     cached;      // true when TLAS BVH nodes were reused (world-space geometry unchanged)
        bool     instCached;  // true when instance buffer was also reused (pnMtxInv unchanged)
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

    // Debug: when true, the dynamic (skinned) BLAS is excluded from the TLAS.
    // Use to verify static-geometry caching without animated characters dirtying the hash.
    void set_exclude_skinned(bool v) { m_excludeSkinned = v; }

    ~TlasBuilder();

    // CPU phase: collect instances, compute world-space AABBs, build SAH BVH.
    // viewMtx is the 4×4 GX world→view matrix (row-major).  Call after BlasCache::flush().
    void build(const BlasCache& cache, const float viewMtx[4][4]);

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

    // Returns the world-space AABB of each TLAS leaf for debug overlays.
    void get_instance_world_aabbs(std::vector<AABB>& out) const {
        out.clear();
        out.reserve(m_instances.size());
        for (const auto& ci : m_instances) out.push_back(ci.worldAabb);
    }

private:
    struct CpuInstance {
        AABB     worldAabb;
        AABB     localAabb;     // cached for fast-path worldAabb recompute without map lookup
        BlasKey  blasKey;       // cached for fast-path draw-source verification
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
    bool     m_blasNodesDirty        = false;
    bool     m_blasFullRebuildPending = false; // true after eviction: re-upload everything
    uint32_t m_staticBlasNodeCount   = 0;
    uint32_t m_staticBlasTriCount    = 0;
    uint32_t m_uploadedNodeCount     = 0;  // how many nodes are already on the GPU
    uint32_t m_uploadedTriCount      = 0;  // how many tris  are already on the GPU

    // Static BLAS GPU buffer capacity (in node / tri count, not bytes).
    uint32_t m_blasNodeBufCapacity = 0;
    uint32_t m_blasTriBufCapacity  = 0;

    // Per-frame GPU buffers (uploaded each frame)
    WGPUBuffer m_tlasNodeBuf    = nullptr;
    WGPUBuffer m_instanceBuf    = nullptr;
    uint32_t   m_instanceBufCap = 0;  // instance count the current buffer was sized for

    // Static-only BLAS buffer (stable across frames when no new entries added/evicted).
    WGPUBuffer m_blasNodeBuf = nullptr;
    WGPUBuffer m_blasTriBuf  = nullptr;

    // Per-frame instance dedup (reused scratch, cleared each build()).
    std::unordered_set<size_t> m_dedupSeen;

    bool     m_excludeSkinned   = false; // debug: skip dynamic BLAS instance

    // Split-hash caching: structural hash covers world-space AABBs + BLAS refs (camera-independent).
    // Instance hash covers pnMtxInv (view-dependent — changes when camera moves).
    uint64_t m_lastStructHash   = 0;
    uint64_t m_lastInstHash     = 0;
    bool     m_tlasNodesDirty   = true;  // rebuild BVH nodes
    bool     m_tlasInstDirty    = true;  // re-upload instance buffer
    uint64_t m_lastTlasGpuBytes = 0;
    std::vector<uint32_t> m_lastPerm;    // sort permutation from last BVH build

    // Fast-path state: skip full instance rebuild when draw call list is identical.
    // m_instanceDrawIdx[i] = index in cache.instances() that produced m_instances[i] (sorted order).
    std::vector<uint32_t> m_instanceDrawIdx;
    uint32_t m_lastCacheInstCount    = UINT32_MAX; // cache.instances().size() last slow-path frame
    uint64_t m_lastCacheInstBlasHash = 0;          // blasKey-only hash of cache.instances()

    Stats      m_lastStats{};
    Validation m_lastValidation{};
    bool       m_validateNext = false;
};

} // namespace dusk::rtao
