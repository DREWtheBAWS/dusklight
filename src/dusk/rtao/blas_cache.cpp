#include "blas_cache.hpp"
#include "vertex_decoder.hpp"
#include <aurora/geometry_capture.h>
#include <algorithm>
#include <chrono>

namespace dusk::rtao {

// GX attribute type constants (subset — only what BlasCache needs)
static constexpr uint8_t kGxDirect  = 1;
static constexpr uint8_t kGxIndex8  = 2;
// kGxIndex16 = 3 (else branch)

// Build a compound BLAS key from a draw call.
// Hashes the first ≤8 vertex position sub-indices plus indexCount — unique
// enough to distinguish different mesh shapes sharing the same posArray.
static BlasKey compute_blas_key(const AuroraGxCaptureDraw& draw) {
    uint32_t h = draw.indexCount * 2654435761u;
    if (draw.vertData) {
        const uint32_t N = std::min(draw.vertCount, 8u);
        for (uint32_t vi = 0; vi < N; ++vi) {
            const uint8_t* p = draw.vertData + vi * draw.vertStride + draw.posOffset;
            const uint32_t idx = (draw.posAttrType == kGxIndex8) ? p[0] : (uint32_t(p[0]) << 8) | p[1];
            h = h * 1664525u + idx + 0x9e3779b9u;
        }
    }
    return { reinterpret_cast<uintptr_t>(draw.posArray), h };
}

// ---------------------------------------------------------------------------
// BlasCache
// ---------------------------------------------------------------------------

BlasCache::~BlasCache() = default;

uint32_t BlasCache::record_draw(const AuroraGxCaptureDraw& draw) {
    // Deferred clear: instances from the previous aurora_end_frame stay alive through
    // all PreDraw() calls (even multiple per aurora frame) and are cleared here on the
    // first record_draw() of the new aurora_end_frame, mirroring GeometryCollector.
    if (m_instancesClearPending) {
        m_instances.clear();
        m_dynamicTrisBuf.clear();
        m_hasDynamic = false;
        m_instancesClearPending = false;
    }

    ++m_totalCallsThisFrame;
    ++m_totalCallsEver;

    // Skip direct-attribute draws: no stable posArray pointer for caching.
    if (draw.posAttrType == kGxDirect || !draw.posArray) {
        ++m_rejectedDirect;
        return 0;
    }

    // Determine effective matrix slot.  In GX, hasPnMtxIdx is true for nearly
    // all draws (even static terrain), because PNMTXIDX is almost universally
    // present in the vertex format.  The real check for "skinned" geometry is
    // whether vertices actually switch between slots within one draw call.
    uint32_t matrixSlot = draw.currentPnMtx;
    if (draw.hasPnMtxIdx && draw.vertCount > 0) {
        matrixSlot = draw.vertData[0] / 3u;
        bool isMultiMatrix = false;
        for (uint32_t vi = 1; vi < draw.vertCount; ++vi) {
            if (draw.vertData[vi * draw.vertStride] / 3u != matrixSlot) {
                isMultiMatrix = true;
                break;
            }
        }
        if (isMultiMatrix) {
            ++m_rejectedSkinned;
            // Decode view-space triangles for the per-frame dynamic BLAS.
            // decode_triangles() applies per-vertex pnMtx to produce view-space positions.
            if (m_dynamicTrisBuf.size() < kMaxDynTris) {
                auto viewTris = decode_triangles(draw);
                const uint32_t room = kMaxDynTris - static_cast<uint32_t>(m_dynamicTrisBuf.size());
                const uint32_t take = std::min(static_cast<uint32_t>(viewTris.size()), room);
                m_dynamicTrisBuf.insert(m_dynamicTrisBuf.end(),
                                        viewTris.begin(), viewTris.begin() + take);
            }
            return 0;
        }
    }

    const BlasKey key = compute_blas_key(draw);

    // Record instance for TLAS (Phase 2) and Layer 3 overlay.
    Instance inst{};
    inst.blasKey = key;
    memcpy(inst.pnMtx, draw.pnMtx[matrixSlot], sizeof(inst.pnMtx));
    m_instances.push_back(inst);

    // Mark cached entry as seen.
    auto it = m_entries.find(key);
    if (it != m_entries.end()) {
        if (it->second.lastSeenFrame != m_frame) {
            it->second.lastSeenFrame = m_frame;
            ++m_seenThisFrame;
        }
        return 0;
    }

    // Skip if already queued for build (O(1) hash check).
    if (m_pendingKeys.count(key)) return 0;

    // Don't grow past the entry cap — bounds monolithic buffer size and GPU memory.
    if (m_entries.size() + m_pending.size() >= kMaxEntries) return 0;

    // New mesh: decode local-space triangles and queue for BVH build.
    auto localTris = decode_triangles_local(draw);
    if (localTris.empty()) return 0;

    uint32_t count = static_cast<uint32_t>(localTris.size());
    m_pendingKeys.insert(key);
    m_pending.push_back({ key, std::move(localTris) });
    return count;
}

uint32_t BlasCache::flush() {
    const auto t0 = std::chrono::high_resolution_clock::now();

    // Process at most kMaxBuildsPerFrame entries per call to spread CPU/GPU cost
    // over multiple frames.  Remaining entries stay in m_pending for the next flush.
    const uint32_t built = static_cast<uint32_t>(
        std::min(m_pending.size(), size_t(kMaxBuildsPerFrame)));

    for (uint32_t pi = 0; pi < built; ++pi) {
        auto& p = m_pending[pi];
        Entry entry{};
        entry.bvh.build(p.localTris);
        if (entry.bvh.empty()) continue;

        entry.localAabb  = entry.bvh.nodes()[0].bounds;
        entry.triCount   = static_cast<uint32_t>(entry.bvh.tris().size());
        entry.nodeCount  = entry.bvh.node_count();
        entry.lastSeenFrame = m_frame;

        m_entries[p.key] = std::move(entry);
        ++m_seenThisFrame;
        ++m_generation;
    }
    // Remove only the processed entries; keep the rest for next frame.
    for (uint32_t pi = 0; pi < built; ++pi)
        m_pendingKeys.erase(m_pending[pi].key);
    m_pending.erase(m_pending.begin(), m_pending.begin() + built);

    // Update stats (evictedLastFrame is set by advance_frame()).
    uint64_t totalBytes = 0;
    for (auto& [k, e] : m_entries)
        totalBytes += uint64_t(e.nodeCount) * sizeof(GpuNode)
                    + uint64_t(e.triCount)  * sizeof(GpuTri);

    // Build the per-frame dynamic BLAS from all accumulated skinned-mesh triangles.
    if (!m_dynamicTrisBuf.empty()) {
        m_dynamicEntry = DynamicEntry{};
        m_dynamicEntry.bvh.build(m_dynamicTrisBuf);
        if (!m_dynamicEntry.bvh.empty()) {
            m_dynamicEntry.viewAabb  = m_dynamicEntry.bvh.nodes()[0].bounds;
            m_dynamicEntry.nodeCount = m_dynamicEntry.bvh.node_count();
            m_dynamicEntry.triCount  = static_cast<uint32_t>(m_dynamicEntry.bvh.tris().size());
            m_hasDynamic = true;
        }
    }

    m_lastStats.totalCached   = static_cast<uint32_t>(m_entries.size());
    m_lastStats.pendingCount  = static_cast<uint32_t>(m_pending.size());
    m_lastStats.seenThisFrame = m_seenThisFrame;
    m_lastStats.newThisFrame  = built;
    m_lastStats.gpuBytesTotal = totalBytes;
    m_lastStats.dynTriCount   = static_cast<uint32_t>(m_dynamicTrisBuf.size());
    m_lastStats.dynNodeCount  = m_hasDynamic ? m_dynamicEntry.nodeCount : 0u;
    m_lastStats.flushMs = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    // Note: rejectedDirect/rejectedSkinned/totalCallsThisFrame are committed in
    // advance_frame() instead, so they reflect the completed frame's draws.

    return built;
}

void BlasCache::advance_frame() {
    uint32_t evicted = 0;
    for (auto it = m_entries.begin(); it != m_entries.end(); ) {
        if (m_frame - it->second.lastSeenFrame > kEvictAfterFrames) {
            it = m_entries.erase(it);
            ++evicted;
            ++m_generation;
            ++m_evictionGeneration;
        } else {
            ++it;
        }
    }
    m_lastStats.evictedLastFrame = evicted;

    // Commit per-draw counters before resetting them.
    m_lastStats.rejectedDirect      = m_rejectedDirect;
    m_lastStats.rejectedSkinned     = m_rejectedSkinned;
    m_lastStats.totalCallsThisFrame = m_totalCallsThisFrame;
    // Save instance count from the just-completed aurora_end_frame before deferring the clear.
    m_lastStats.instanceCount       = static_cast<uint32_t>(m_instances.size());

    // Deferred clear: don't destroy instances yet — PreDraw() reads them after advance_frame().
    // The actual clear happens at the start of the first record_draw() in the next aurora frame.
    m_instancesClearPending = true;
    m_seenThisFrame       = 0;
    m_rejectedDirect      = 0;
    m_rejectedSkinned     = 0;
    m_totalCallsThisFrame = 0;
    ++m_frame;
}

} // namespace dusk::rtao
