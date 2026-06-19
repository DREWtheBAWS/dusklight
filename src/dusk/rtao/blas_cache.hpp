#pragma once
#include "vertex_decoder.hpp"
#include "bvh.hpp"
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct AuroraGxCaptureDraw;

namespace dusk::rtao {

// Compound key that uniquely identifies a mesh within a shared vertex buffer.
// posArray alone collapses distinct meshes that share the same buffer.
// meshHash encodes up to 8 vertex position sub-indices + indexCount — unique
// enough to distinguish different mesh shapes without false positives in practice.
struct BlasKey {
    uintptr_t posArray;  // GX position array pointer (identifies the pool)
    uint32_t  meshHash;  // hash of first ≤8 vertex pos sub-indices + indexCount
    bool operator==(const BlasKey& o) const noexcept {
        return posArray == o.posArray && meshHash == o.meshHash;
    }
};

struct BlasKeyHash {
    size_t operator()(const BlasKey& k) const noexcept {
        size_t h = std::hash<uintptr_t>{}(k.posArray);
        h ^= size_t(k.meshHash) * 0x9e3779b9u + 0x9e3779b9u + (h << 6) + (h >> 2);
        return h;
    }
};

// Maintains a persistent cache of per-mesh SAH BVHs (BLASes) built in local
// (object) space.  Keyed by a compound BlasKey (posArray + firstPosIdx + indexCount)
// that distinguishes distinct meshes even when they share a position array pool.
// Multi-matrix (skinned) and direct-attribute draws are skipped; they always fall
// back to the existing LBVH path.
//
// Call order each frame:
//   1. record_draw()   — per draw call, from the geometry-capture callback (no device)
//   2. flush()         — once in the post-render callback (has device), builds + uploads
//   3. advance_frame() — once in afterDraw(), evicts stale entries and resets per-frame state
class BlasCache {
public:
    struct Entry {
        Bvh      bvh;
        AABB     localAabb;
        uint32_t triCount      = 0;
        uint32_t nodeCount     = 0;
        uint32_t lastSeenFrame = 0;
    };

    // GPU buffer layouts — shared with TlasBuilder for monolithic BLAS packing.
    struct GpuNode {
        float    aabb_min[3]; uint32_t hit_next;
        float    aabb_max[3]; uint32_t miss_next;
        uint32_t tri_offset;  uint32_t tri_count;
        uint32_t _pad[2];
    };
    static_assert(sizeof(GpuNode) == 48);

    struct GpuTri { float v[3][4]; };
    static_assert(sizeof(GpuTri) == 48);

    // Per-frame instance: one draw call that references a cached BLAS.
    struct Instance {
        BlasKey blasKey;       // compound key → lookup key in entries()
        float   pnMtx[3][4];  // local-to-view matrix for this particular draw instance
    };

    struct Stats {
        uint32_t totalCached;       // live entries in cache
        uint32_t pendingCount;      // queued for build but not yet processed
        uint32_t seenThisFrame;     // entries that had ≥1 draw this frame
        uint32_t newThisFrame;      // BLASes newly built this frame
        uint32_t evictedLastFrame;  // entries evicted at advance_frame()
        uint64_t gpuBytesTotal;     // combined node+tri GPU buffer footprint
        // Rejection counters for diagnostics — how many draws were filtered and why.
        uint32_t rejectedDirect;      // posAttrType == GX_DIRECT or posArray == null
        uint32_t rejectedSkinned;     // per-vertex matrix check found mixed slots
        uint32_t totalCallsThisFrame; // total record_draw() calls
        uint32_t instanceCount;       // instances accumulated in last aurora_end_frame
        float    flushMs;             // CPU time for flush() this frame (ms)
        uint32_t dynTriCount;         // skinned tris accumulated for the GPU dynamic LBVH this frame
    };

    ~BlasCache();

    // Record a draw call during the geometry-capture callback.
    // Returns the triangle count of a newly queued BLAS (0 if cached or skipped).
    uint32_t record_draw(const AuroraGxCaptureDraw& draw);

    // Build CPU SAH BVH for all pending new meshes (CPU-only; no GPU upload).
    // TlasBuilder::flush() packs everything into the monolithic GPU buffer.
    // Call once per frame before TlasBuilder::build().
    // Returns number of BLASes built this call.
    uint32_t flush();

    // Advance frame counter, evict entries idle for kEvictAfterFrames, reset
    // the per-frame instance list.  Call from afterDraw().
    void advance_frame();

    Stats                                              last_stats()     const { return m_lastStats; }
    uint32_t                                           current_frame()  const { return m_frame; }
    uint32_t                                           total_record_draw_calls_ever() const { return m_totalCallsEver; }
    // Incremented whenever entries are added OR evicted.
    uint32_t                                           generation()          const { return m_generation; }
    // Incremented only on eviction; TlasBuilder uses this to decide full vs. incremental BLAS repack.
    uint32_t                                           eviction_generation() const { return m_evictionGeneration; }
    const std::vector<Instance>&                       instances()      const { return m_instances; }
    const std::unordered_map<BlasKey, Entry, BlasKeyHash>& entries()     const { return m_entries; }
    // Raw view-space skinned triangles for this frame (fed to GPU LBVH builder).
    const std::vector<Triangle>& dynamic_triangles() const { return m_dynamicTrisBuf; }

    // Match the distance filter applied by GeometryCollector so far-field skinned triangles
    // are excluded before the GPU LBVH build.  Pass 0 to disable.  Set each frame to the
    // same value as GeometryCollector::set_max_distance().
    void set_max_distance(float r) { m_maxDistSq = (r > 0.f) ? r * r : 0.f; }

    // Per-frame build budget: limits per-frame CPU cost while new areas are explored.
    // After kMaxEntries BLASes are cached the queue drains and subsequent frames are free.
    static constexpr uint32_t kEvictAfterFrames  = 300;   // ~5 s at 60 fps — reduces re-eviction churn when panning
    static constexpr uint32_t kMaxEntries        = 1024;  // bounds monolithic buffer size (~8 MB); covers large outdoor scenes
    static constexpr uint32_t kMaxBuildsPerFrame = 128;   // CPU SAH only (no GPU upload), so high budget is safe
    static constexpr uint32_t kMaxDynTris        = 12000; // cap on accumulated skinned-mesh tris

private:
    struct Pending {
        BlasKey               key;
        std::vector<Triangle> localTris;
    };

    std::vector<Pending>                               m_pending;
    std::unordered_set<BlasKey, BlasKeyHash>           m_pendingKeys;  // O(1) duplicate check
    std::vector<Instance>                              m_instances;
    std::unordered_map<BlasKey, Entry, BlasKeyHash>    m_entries;
    uint32_t                                 m_frame               = 0;
    uint32_t                                 m_seenThisFrame       = 0;
    uint32_t                                 m_rejectedDirect      = 0;
    uint32_t                                 m_rejectedSkinned     = 0;
    uint32_t                                 m_totalCallsThisFrame = 0;
    uint32_t                                 m_totalCallsEver      = 0; // never reset
    uint32_t                                 m_generation          = 0; // bumped on add/evict
    uint32_t                                 m_evictionGeneration  = 0; // bumped on evict only
    bool                                     m_instancesClearPending = false;
    Stats                                    m_lastStats{};

    // Dynamic (skinned) triangles — view-space, accumulated per frame, fed to GPU LBVH.
    std::vector<Triangle> m_dynamicTrisBuf;
    float                 m_maxDistSq = 0.f; // centroid distance² filter (0 = disabled)
};

} // namespace dusk::rtao
