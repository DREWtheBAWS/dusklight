#include "tlas_builder.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace dusk::rtao {

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

static float get_axis(const Vec3& v, int a) {
    switch (a) { case 0: return v.x; case 1: return v.y; default: return v.z; }
}

static Vec3 xf_point(const float m[3][4], const Vec3& p) {
    return {
        m[0][0]*p.x + m[0][1]*p.y + m[0][2]*p.z + m[0][3],
        m[1][0]*p.x + m[1][1]*p.y + m[1][2]*p.z + m[1][3],
        m[2][0]*p.x + m[2][1]*p.y + m[2][2]*p.z + m[2][3]
    };
}

// Multiply two affine 3×4 matrices (treated as 4×4 with last row [0,0,0,1]).
static void mat3x4_mul(const float a[3][4], const float b[3][4], float c[3][4]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            c[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j];
        c[i][3] = a[i][0]*b[0][3] + a[i][1]*b[1][3] + a[i][2]*b[2][3] + a[i][3];
    }
}

static bool invert_3x4(const float m[3][4], float inv[3][4]) {
    const float a=m[0][0], b=m[0][1], c=m[0][2];
    const float d=m[1][0], e=m[1][1], f=m[1][2];
    const float g=m[2][0], h=m[2][1], i=m[2][2];
    const float c00= (e*i-f*h), c01=-(d*i-f*g), c02= (d*h-e*g);
    const float c10=-(b*i-c*h), c11= (a*i-c*g), c12=-(a*h-b*g);
    const float c20= (b*f-c*e), c21=-(a*f-c*d), c22= (a*e-b*d);
    const float det = a*c00 + b*c01 + c*c02;
    if (std::abs(det) < 1e-8f) return false;
    const float s = 1.f / det;
    inv[0][0]=c00*s; inv[0][1]=c10*s; inv[0][2]=c20*s;
    inv[1][0]=c01*s; inv[1][1]=c11*s; inv[1][2]=c21*s;
    inv[2][0]=c02*s; inv[2][1]=c12*s; inv[2][2]=c22*s;
    const float tx=m[0][3], ty=m[1][3], tz=m[2][3];
    inv[0][3] = -(inv[0][0]*tx + inv[0][1]*ty + inv[0][2]*tz);
    inv[1][3] = -(inv[1][0]*tx + inv[1][1]*ty + inv[1][2]*tz);
    inv[2][3] = -(inv[2][0]*tx + inv[2][1]*ty + inv[2][2]*tz);
    return true;
}

// ---------------------------------------------------------------------------
// TLAS SAH BVH
// ---------------------------------------------------------------------------

struct TlasItem { AABB aabb; uint32_t instIdx; };

static constexpr uint32_t kTlasEnd = UINT32_MAX;

static uint32_t tlas_sah_split(std::vector<TlasItem>& items,
                               uint32_t start, uint32_t end, const AABB& bounds) {
    static constexpr int kB = 8;
    const float psa = bounds.surface_area();
    if (psa < 1e-10f) return (start + end) / 2;

    float best = FLT_MAX; int baxis = -1, bsplit = -1;

    for (int axis = 0; axis < 3; ++axis) {
        const float amin = get_axis(bounds.min, axis);
        const float amax = get_axis(bounds.max, axis);
        if (amax - amin < 1e-6f) continue;
        struct Bkt { AABB b; int n=0; } bkts[kB] = {};
        for (uint32_t i = start; i < end; ++i) {
            const float cx = (get_axis(items[i].aabb.min,axis) + get_axis(items[i].aabb.max,axis)) * 0.5f;
            const int k = std::min(kB-1, (int)(kB*(cx-amin)/(amax-amin)));
            ++bkts[k].n; bkts[k].b.expand(items[i].aabb);
        }
        AABB lb[kB]; int ln[kB]={};
        AABB rb[kB]; int rn[kB]={};
        { AABB a; int n=0; for(int k=0;k<kB;++k){ a.expand(bkts[k].b); n+=bkts[k].n; lb[k]=a; ln[k]=n; } }
        { AABB a; int n=0; for(int k=kB-1;k>=0;--k){ a.expand(bkts[k].b); n+=bkts[k].n; rb[k]=a; rn[k]=n; } }
        for (int s=1; s<kB; ++s) {
            if (!ln[s-1] || !rn[s]) continue;
            const float cost = 1.f+(ln[s-1]*lb[s-1].surface_area()+rn[s]*rb[s].surface_area())/psa;
            if (cost < best) { best=cost; baxis=axis; bsplit=s; }
        }
    }
    if (baxis < 0) return (start + end) / 2;

    const float amin = get_axis(bounds.min, baxis);
    const float amax = get_axis(bounds.max, baxis);
    auto mid = std::partition(items.begin()+start, items.begin()+end, [&](const TlasItem& t){
        const float cx = (get_axis(t.aabb.min,baxis)+get_axis(t.aabb.max,baxis))*0.5f;
        return std::min(kB-1,(int)(kB*(cx-amin)/(amax-amin))) < bsplit;
    });
    const uint32_t sp = static_cast<uint32_t>(mid - items.begin());
    return (sp==start||sp==end) ? (start+end)/2 : sp;
}

static uint32_t tlas_build_node(std::vector<TlasItem>& items,
                                std::vector<BvhNode>& nodes,
                                uint32_t start, uint32_t end) {
    const uint32_t idx = static_cast<uint32_t>(nodes.size());
    nodes.push_back({});
    AABB bounds;
    for (uint32_t i = start; i < end; ++i) bounds.expand(items[i].aabb);
    const uint32_t count = end - start;
    if (count == 1) {
        nodes[idx] = { bounds, 0, 0, start, 1 };
        return idx;
    }
    const uint32_t split = tlas_sah_split(items, start, end, bounds);
    const uint32_t left  = tlas_build_node(items, nodes, start, split);
    const uint32_t right = tlas_build_node(items, nodes, split, end);
    nodes[idx] = { bounds, left, right, 0, 0 };
    return idx;
}

static void tlas_link_node(std::vector<BvhNode>& nodes, uint32_t idx, uint32_t miss) {
    BvhNode& n = nodes[idx];
    if (n.tri_count > 0) { n.miss_next = miss; return; }
    const uint32_t right = n.miss_next;
    n.miss_next = miss;
    tlas_link_node(nodes, n.hit_next, right);
    tlas_link_node(nodes, right, miss);
}

// ---------------------------------------------------------------------------
// GPU upload
// ---------------------------------------------------------------------------

static WGPUBuffer upload_gpu(WGPUDevice dev, const void* data, size_t size) {
    if (size == 0) size = 4;
    WGPUBufferDescriptor d{};
    d.size  = size;
    d.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(dev, &d);
    WGPUQueue q = wgpuDeviceGetQueue(dev);
    wgpuQueueWriteBuffer(q, buf, 0, data, size);
    wgpuQueueRelease(q);
    return buf;
}

static void release_buf(WGPUBuffer& b) { if (b) { wgpuBufferRelease(b); b = nullptr; } }

// ---------------------------------------------------------------------------
// TlasBuilder
// ---------------------------------------------------------------------------

TlasBuilder::~TlasBuilder() {
    release_buf(m_tlasNodeBuf);
    release_buf(m_instanceBuf);
    release_buf(m_blasNodeBuf);
    release_buf(m_blasTriBuf);
}

void TlasBuilder::build(const BlasCache& cache, const float viewMtx[4][4]) {
    const auto t0 = std::chrono::high_resolution_clock::now();

    // Compute view→world (invView) from the 3×4 affine portion of the view matrix.
    // Falls back to identity if the matrix is degenerate (e.g. not yet populated).
    float invView[3][4] = { {1,0,0,0}, {0,1,0,0}, {0,0,1,0} };
    {
        const float v3x4[3][4] = {
            {viewMtx[0][0], viewMtx[0][1], viewMtx[0][2], viewMtx[0][3]},
            {viewMtx[1][0], viewMtx[1][1], viewMtx[1][2], viewMtx[1][3]},
            {viewMtx[2][0], viewMtx[2][1], viewMtx[2][2], viewMtx[2][3]}
        };
        invert_3x4(v3x4, invView);  // ok to leave identity on failure
    }

    const auto& entries   = cache.entries();
    const auto& instances = cache.instances();

    // Rebuild/update the monolithic BLAS staging data when cache entries change.
    if (cache.generation() != m_lastBlasGeneration) {
        const bool fullRebuild = (cache.eviction_generation() != m_lastEvictionGeneration);
        if (fullRebuild) {
            m_blasOffsets.clear();
            m_pendingBlasNodes.clear();
            m_pendingBlasTris.clear();
            m_blasFullRebuildPending = true;
            m_uploadedNodeCount = 0;
            m_uploadedTriCount  = 0;
        }

        for (const auto& [key, e] : entries) {
            if (e.bvh.empty()) continue;
            if (!fullRebuild && m_blasOffsets.count(key)) continue;

            const uint32_t nodeOff = static_cast<uint32_t>(m_pendingBlasNodes.size());
            const uint32_t triOff  = static_cast<uint32_t>(m_pendingBlasTris.size());
            m_blasOffsets[key] = { nodeOff, triOff };

            for (uint32_t i = 0; i < e.nodeCount; ++i) {
                const BvhNode& n = e.bvh.nodes()[i];
                BlasCache::GpuNode gn{};
                gn.aabb_min[0]=n.bounds.min.x; gn.aabb_min[1]=n.bounds.min.y;
                gn.aabb_min[2]=n.bounds.min.z;
                gn.hit_next  = (n.hit_next  == UINT32_MAX) ? UINT32_MAX : n.hit_next  + nodeOff;
                gn.aabb_max[0]=n.bounds.max.x; gn.aabb_max[1]=n.bounds.max.y;
                gn.aabb_max[2]=n.bounds.max.z;
                gn.miss_next = (n.miss_next == UINT32_MAX) ? UINT32_MAX : n.miss_next + nodeOff;
                gn.tri_offset = n.tri_offset + triOff;
                gn.tri_count  = n.tri_count;
                m_pendingBlasNodes.push_back(gn);
            }

            for (uint32_t i = 0; i < e.triCount; ++i) {
                const Triangle& t = e.bvh.tris()[i];
                BlasCache::GpuTri gt{};
                gt.v[0][0]=t.a.x; gt.v[0][1]=t.a.y; gt.v[0][2]=t.a.z; gt.v[0][3]=0.f;
                gt.v[1][0]=t.b.x; gt.v[1][1]=t.b.y; gt.v[1][2]=t.b.z; gt.v[1][3]=0.f;
                gt.v[2][0]=t.c.x; gt.v[2][1]=t.c.y; gt.v[2][2]=t.c.z; gt.v[2][3]=0.f;
                m_pendingBlasTris.push_back(gt);
            }
        }
        m_blasNodesDirty = true;
        m_lastBlasGeneration     = cache.generation();
        m_lastEvictionGeneration = cache.eviction_generation();
    }

    m_staticBlasNodeCount = static_cast<uint32_t>(m_pendingBlasNodes.size());
    m_staticBlasTriCount  = static_cast<uint32_t>(m_pendingBlasTris.size());

    // ---------------------------------------------------------------------------
    // Fast path: when the draw-call list is identical to last frame (same blasKey
    // sequence, same count, no new/evicted BLASes), skip the full rebuild loop.
    // Only update pnMtxInv per instance (needed for camera movement) and recompute
    // worldAabb per instance to detect any moved objects.  Falls back to the slow
    // path if blasKey sequence changes or any instance is unmatchable.
    // ---------------------------------------------------------------------------
    const bool genStable = (cache.generation() == m_lastBlasGeneration);
    bool fastPathOk = genStable
                   && instances.size() == m_lastCacheInstCount
                   && !m_instances.empty()
                   && m_instanceDrawIdx.size() == m_instances.size();
    if (fastPathOk) {
        // Cheap blasKey-only hash (no pnMtx involved — camera-independent).
        uint64_t bkh = uint64_t(instances.size()) * 0x9e3779b97f4a7c15ULL;
        for (const auto& inst : instances)
            bkh ^= BlasKeyHash{}(inst.blasKey) * 0x517cc1b727220a95ULL + (bkh << 6) + (bkh >> 2);
        fastPathOk = (bkh == m_lastCacheInstBlasHash);
    }

    bool fastSahNeeded = false;  // true if fast path found structural change (object moved)
    if (fastPathOk) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_instances.size()); ++i) {
            const auto& draw = instances[m_instanceDrawIdx[i]];
            if (!(draw.blasKey == m_instances[i].blasKey)) { fastPathOk = false; break; }

            float modelToWorld[3][4];
            mat3x4_mul(invView, draw.pnMtx, modelToWorld);

            AABB newAabb;
            const AABB& lb = m_instances[i].localAabb;
            for (int c = 0; c < 8; ++c)
                newAabb.expand(xf_point(modelToWorld, {
                    (c&1)?lb.max.x:lb.min.x, (c&2)?lb.max.y:lb.min.y, (c&4)?lb.max.z:lb.min.z}));

            // Detect object movement via quantized AABB comparison.
            if (static_cast<int32_t>(std::round(newAabb.min.x)) != static_cast<int32_t>(std::round(m_instances[i].worldAabb.min.x)) ||
                static_cast<int32_t>(std::round(newAabb.min.y)) != static_cast<int32_t>(std::round(m_instances[i].worldAabb.min.y)) ||
                static_cast<int32_t>(std::round(newAabb.min.z)) != static_cast<int32_t>(std::round(m_instances[i].worldAabb.min.z)) ||
                static_cast<int32_t>(std::round(newAabb.max.x)) != static_cast<int32_t>(std::round(m_instances[i].worldAabb.max.x)) ||
                static_cast<int32_t>(std::round(newAabb.max.y)) != static_cast<int32_t>(std::round(m_instances[i].worldAabb.max.y)) ||
                static_cast<int32_t>(std::round(newAabb.max.z)) != static_cast<int32_t>(std::round(m_instances[i].worldAabb.max.z))) {
                m_instances[i].worldAabb = newAabb;
                fastSahNeeded = true;
            }

            invert_3x4(draw.pnMtx, m_instances[i].pnMtxInv);
        }
    }

    if (fastPathOk && !fastSahNeeded) {
        // Fully fast: same world, only camera moved.  Compute instance hash only.
        m_tlasNodesDirty = false;
        {
            uint64_t h = uint64_t(m_instances.size()) * 0x9e3779b97f4a7c15ULL;
            for (const auto& ci : m_instances) {
                const auto* b = reinterpret_cast<const uint8_t*>(ci.pnMtxInv);
                for (size_t k = 0; k < sizeof(ci.pnMtxInv); k += 4) {
                    uint32_t w = 0; std::memcpy(&w, b + k, 4);
                    h ^= uint64_t(w) * 2654435761ULL + (h << 6) + (h >> 2);
                }
            }
            m_tlasInstDirty = (h != m_lastInstHash);
            m_lastInstHash  = h;
        }
        m_lastStats.cached     = true;
        m_lastStats.instCached = !m_tlasInstDirty;
        m_lastStats.buildMs    = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return;
    }

    if (!fastPathOk) {
        // ---------------------------------------------------------------------------
        // Slow path: rebuild m_instances from scratch.
        // ---------------------------------------------------------------------------
        m_instances.clear();
        m_instanceDrawIdx.clear();
        m_instances.reserve(instances.size() + 1);
        m_instanceDrawIdx.reserve(instances.size());
        m_dedupSeen.clear();
        uint32_t dedupRej = 0;

        for (uint32_t di = 0; di < static_cast<uint32_t>(instances.size()); ++di) {
            const auto& inst = instances[di];

            // Check BLAS entry first: skip draws with no cached BLAS (avoids pnMtx hash cost).
            const auto eit = entries.find(inst.blasKey);
            if (eit == entries.end() || eit->second.bvh.empty()) continue;
            const auto oit = m_blasOffsets.find(inst.blasKey);
            if (oit == m_blasOffsets.end()) continue;

            // Dedup by (blasKey, pnMtx bytes) — only for draws that have a BLAS entry.
            size_t dh = BlasKeyHash{}(inst.blasKey);
            const auto* db = reinterpret_cast<const uint8_t*>(inst.pnMtx);
            for (int k = 0; k < 48; k += 4) {
                uint32_t w; std::memcpy(&w, db + k, 4);
                dh ^= size_t(w) * 2654435761u + (dh << 6) + (dh >> 2);
            }
            if (!m_dedupSeen.insert(dh).second) { ++dedupRej; continue; }

            CpuInstance ci;
            if (!invert_3x4(inst.pnMtx, ci.pnMtxInv)) continue;

            float modelToWorld[3][4];
            mat3x4_mul(invView, inst.pnMtx, modelToWorld);

            const AABB& lb = eit->second.localAabb;
            for (int c = 0; c < 8; ++c) {
                ci.worldAabb.expand(xf_point(modelToWorld, {
                    (c&1)?lb.max.x:lb.min.x, (c&2)?lb.max.y:lb.min.y, (c&4)?lb.max.z:lb.min.z
                }));
            }
            if (!ci.worldAabb.valid()) continue;

            ci.localAabb      = lb;
            ci.blasKey        = inst.blasKey;
            ci.blasNodeOffset = oit->second.first;
            ci.blasTriOffset  = oit->second.second;
            ci.blasNodeCount  = eit->second.nodeCount;
            ci.blasTriCount   = eit->second.triCount;
            m_instances.push_back(ci);
            m_instanceDrawIdx.push_back(di);
        }
        m_lastStats.dedupRejected = dedupRej;

        // Cache blasKey sequence for fast-path detection next frame.
        m_lastCacheInstCount = static_cast<uint32_t>(instances.size());
        {
            uint64_t h = uint64_t(instances.size()) * 0x9e3779b97f4a7c15ULL;
            for (const auto& inst : instances)
                h ^= BlasKeyHash{}(inst.blasKey) * 0x517cc1b727220a95ULL + (h << 6) + (h >> 2);
            m_lastCacheInstBlasHash = h;
        }

        // Structural hash: world-space AABBs + BLAS references (camera-independent).
        {
            uint64_t h = uint64_t(m_instances.size()) * 0x9e3779b97f4a7c15ULL;
            for (const auto& ci : m_instances) {
                const int32_t q[6] = {
                    static_cast<int32_t>(std::round(ci.worldAabb.min.x)),
                    static_cast<int32_t>(std::round(ci.worldAabb.min.y)),
                    static_cast<int32_t>(std::round(ci.worldAabb.min.z)),
                    static_cast<int32_t>(std::round(ci.worldAabb.max.x)),
                    static_cast<int32_t>(std::round(ci.worldAabb.max.y)),
                    static_cast<int32_t>(std::round(ci.worldAabb.max.z))
                };
                for (int n = 0; n < 6; ++n)
                    h ^= uint64_t(uint32_t(q[n])) * 2654435761ULL + (h << 6) + (h >> 2);
                h ^= uint64_t(ci.blasNodeOffset) * 0x517cc1b727220a95ULL;
                h ^= uint64_t(ci.blasTriOffset)  * 0x6c62272e07bb0142ULL;
                h ^= (uint64_t(ci.blasNodeCount) << 32) | ci.blasTriCount;
            }
            m_tlasNodesDirty = (h != m_lastStructHash);
            m_lastStructHash = h;
        }

        // Instance hash: pnMtxInv only (view-dependent).
        {
            uint64_t h = uint64_t(m_instances.size()) * 0x9e3779b97f4a7c15ULL;
            for (const auto& ci : m_instances) {
                const auto* b = reinterpret_cast<const uint8_t*>(ci.pnMtxInv);
                for (size_t k = 0; k < sizeof(ci.pnMtxInv); k += 4) {
                    uint32_t w = 0; std::memcpy(&w, b + k, 4);
                    h ^= uint64_t(w) * 2654435761ULL + (h << 6) + (h >> 2);
                }
            }
            m_tlasInstDirty = (h != m_lastInstHash);
            m_lastInstHash  = h;
        }

        m_lastStats.cached     = !m_tlasNodesDirty;
        m_lastStats.instCached = !m_tlasInstDirty;

        if (!m_tlasNodesDirty && !m_tlasInstDirty) {
            m_lastStats.buildMs = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            return;
        }

        if (!m_tlasNodesDirty) {
            // Nodes cached — re-sort instances + draw sources by stored permutation.
            if (!m_lastPerm.empty() && m_lastPerm.size() == m_instances.size()) {
                std::vector<CpuInstance> sorted(m_instances.size());
                std::vector<uint32_t>   sortedDraw(m_instanceDrawIdx.size());
                for (uint32_t i = 0; i < static_cast<uint32_t>(m_lastPerm.size()); ++i) {
                    sorted[i]     = m_instances[m_lastPerm[i]];
                    sortedDraw[i] = m_instanceDrawIdx[m_lastPerm[i]];
                }
                m_instances       = std::move(sorted);
                m_instanceDrawIdx = std::move(sortedDraw);
            }
            m_lastStats.buildMs = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            return;
        }
        // Nodes dirty implies instance buffer must also be re-uploaded.
        m_tlasInstDirty = true;
    } else {
        // Fast path but objects moved: m_instances already has correct worldAabb + pnMtxInv.
        // Force a full SAH rebuild to update TLAS leaf positions.
        m_tlasNodesDirty = true;
        m_tlasInstDirty  = true;
    }

    // ---------------------------------------------------------------------------
    // SAH BVH rebuild (reached from slow path nodes-dirty OR fast path structural-dirty).
    // ---------------------------------------------------------------------------
    m_tlasNodes.clear();
    if (m_instances.empty()) {
        m_lastStats.buildMs = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return;
    }

    std::vector<TlasItem> items(m_instances.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_instances.size()); ++i)
        items[i] = { m_instances[i].worldAabb, i };

    m_tlasNodes.reserve(2 * m_instances.size());
    tlas_build_node(items, m_tlasNodes, 0, static_cast<uint32_t>(items.size()));
    tlas_link_node(m_tlasNodes, 0, kTlasEnd);

    // Reorder instances (and their draw sources) to match BVH leaf traversal order.
    m_lastPerm.resize(items.size());
    std::vector<CpuInstance> sorted(m_instances.size());
    std::vector<uint32_t>    sortedDraw(m_instanceDrawIdx.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(items.size()); ++i) {
        m_lastPerm[i]  = items[i].instIdx;
        sorted[i]      = m_instances[items[i].instIdx];
        if (i < static_cast<uint32_t>(m_instanceDrawIdx.size()))
            sortedDraw[i] = m_instanceDrawIdx[items[i].instIdx];
    }
    m_instances       = std::move(sorted);
    m_instanceDrawIdx = std::move(sortedDraw);

    // Optional structural validation.
    if (m_validateNext) {
        m_validateNext = false;
        Validation v{};
        v.ran            = true;
        v.instanceCount  = static_cast<uint32_t>(m_instances.size());
        v.blasNodeBufSize = static_cast<uint32_t>(m_pendingBlasNodes.size());
        v.blasTriBufSize  = static_cast<uint32_t>(m_pendingBlasTris.size());
        v.tlasNodeCount   = static_cast<uint32_t>(m_tlasNodes.size());
        for (const auto& n : m_tlasNodes) {
            if (n.tri_count == 1u && n.tri_offset >= v.instanceCount) {
                v.tlasLeafOk = false; ++v.badLeaves;
            }
        }
        for (const auto& ci : m_instances) {
            if (ci.blasNodeOffset + ci.blasNodeCount > v.blasNodeBufSize ||
                ci.blasTriOffset  + ci.blasTriCount  > v.blasTriBufSize) {
                v.blasRangeOk = false; ++v.badInstances;
            }
        }
        m_lastValidation = v;
    }

    m_lastStats.buildMs = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

void TlasBuilder::flush(WGPUDevice device) {
    const auto t0 = std::chrono::high_resolution_clock::now();

    // Upload TLAS nodes + instance table based on which dirty flags are set.
    if (m_tlasNodesDirty) {
        // Full rebuild: recreate both buffers.
        release_buf(m_tlasNodeBuf);
        release_buf(m_instanceBuf);

        if (!m_tlasNodes.empty()) {
            // Pack and upload TLAS nodes.
            {
                std::vector<BlasCache::GpuNode> gpu(m_tlasNodes.size());
                for (uint32_t i = 0; i < static_cast<uint32_t>(m_tlasNodes.size()); ++i) {
                    const BvhNode& n = m_tlasNodes[i];
                    BlasCache::GpuNode& g = gpu[i];
                    g.aabb_min[0]=n.bounds.min.x; g.aabb_min[1]=n.bounds.min.y;
                    g.aabb_min[2]=n.bounds.min.z; g.hit_next   =n.hit_next;
                    g.aabb_max[0]=n.bounds.max.x; g.aabb_max[1]=n.bounds.max.y;
                    g.aabb_max[2]=n.bounds.max.z; g.miss_next  =n.miss_next;
                    g.tri_offset =n.tri_offset;   g.tri_count  =n.tri_count;
                    g._pad[0]=0; g._pad[1]=0;
                }
                m_tlasNodeBuf = upload_gpu(device, gpu.data(), gpu.size()*sizeof(BlasCache::GpuNode));
            }

            // Pack and upload instance table.
            {
                std::vector<GpuTlasInstance> gpu(m_instances.size());
                for (uint32_t i = 0; i < static_cast<uint32_t>(m_instances.size()); ++i) {
                    const CpuInstance& ci = m_instances[i];
                    GpuTlasInstance& gi = gpu[i];
                    memcpy(gi.pnMtxInv, ci.pnMtxInv, sizeof(gi.pnMtxInv));
                    gi.blasNodeOffset = ci.blasNodeOffset;
                    gi.blasTriOffset  = ci.blasTriOffset;
                    gi.blasNodeCount  = ci.blasNodeCount;
                    gi.blasTriCount   = ci.blasTriCount;
                }
                m_instanceBuf    = upload_gpu(device, gpu.data(), gpu.size()*sizeof(GpuTlasInstance));
                m_instanceBufCap = static_cast<uint32_t>(m_instances.size());
            }
        }
    } else if (m_tlasInstDirty) {
        // Camera moved but world-space structure unchanged: update pnMtxInv in-place.
        // m_instances size is identical to the previous full-rebuild frame (structural hash
        // matched), so we can write directly into the existing buffer.
        if (!m_instances.empty()) {
            std::vector<GpuTlasInstance> gpu(m_instances.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(m_instances.size()); ++i) {
                const CpuInstance& ci = m_instances[i];
                GpuTlasInstance& gi = gpu[i];
                memcpy(gi.pnMtxInv, ci.pnMtxInv, sizeof(gi.pnMtxInv));
                gi.blasNodeOffset = ci.blasNodeOffset;
                gi.blasTriOffset  = ci.blasTriOffset;
                gi.blasNodeCount  = ci.blasNodeCount;
                gi.blasTriCount   = ci.blasTriCount;
            }
            const uint32_t count  = static_cast<uint32_t>(m_instances.size());
            const size_t   needed = count * sizeof(GpuTlasInstance);
            if (!m_instanceBuf || count > m_instanceBufCap) {
                // Buffer too small (shouldn't happen when structural hash stable, but guard).
                release_buf(m_instanceBuf);
                WGPUBufferDescriptor d{};
                d.size  = needed;
                d.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
                m_instanceBuf    = wgpuDeviceCreateBuffer(device, &d);
                m_instanceBufCap = count;
            }
            WGPUQueue q = wgpuDeviceGetQueue(device);
            wgpuQueueWriteBuffer(q, m_instanceBuf, 0, gpu.data(), needed);
            wgpuQueueRelease(q);
        }
    }

    if (!m_tlasNodeBuf) {
        m_lastStats.flushMs = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return;
    }

    // Static-only BLAS buffer (no dynamic tail in Phase 3 — skinned geo uses GPU LBVH).
    {
        const uint32_t minNodes = std::max(m_staticBlasNodeCount, 1u);
        const uint32_t minTris  = std::max(m_staticBlasTriCount,  1u);

        if (!m_blasNodeBuf || minNodes > m_blasNodeBufCapacity) {
            release_buf(m_blasNodeBuf);
            m_blasNodeBufCapacity = minNodes + minNodes / 4 + 64;
            WGPUBufferDescriptor d{};
            d.size  = uint64_t(m_blasNodeBufCapacity) * sizeof(BlasCache::GpuNode);
            d.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            m_blasNodeBuf = wgpuDeviceCreateBuffer(device, &d);
            m_blasNodesDirty = true;
        }
        if (!m_blasTriBuf || minTris > m_blasTriBufCapacity) {
            release_buf(m_blasTriBuf);
            m_blasTriBufCapacity = minTris + minTris / 4 + 64;
            WGPUBufferDescriptor d{};
            d.size  = uint64_t(m_blasTriBufCapacity) * sizeof(BlasCache::GpuTri);
            d.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            m_blasTriBuf = wgpuDeviceCreateBuffer(device, &d);
            m_blasNodesDirty = true;
        }

        if (m_blasNodesDirty) {
            // Upload only the newly-appended delta (from m_uploadedNodeCount onward) unless a full
            // rebuild was triggered by eviction, in which case we upload everything from offset 0.
            const uint32_t nodeStart = m_blasFullRebuildPending ? 0 : m_uploadedNodeCount;
            const uint32_t triStart  = m_blasFullRebuildPending ? 0 : m_uploadedTriCount;
            WGPUQueue q = wgpuDeviceGetQueue(device);
            if (m_staticBlasNodeCount > nodeStart)
                wgpuQueueWriteBuffer(q, m_blasNodeBuf,
                    uint64_t(nodeStart) * sizeof(BlasCache::GpuNode),
                    m_pendingBlasNodes.data() + nodeStart,
                    (m_staticBlasNodeCount - nodeStart) * sizeof(BlasCache::GpuNode));
            if (m_staticBlasTriCount > triStart)
                wgpuQueueWriteBuffer(q, m_blasTriBuf,
                    uint64_t(triStart) * sizeof(BlasCache::GpuTri),
                    m_pendingBlasTris.data() + triStart,
                    (m_staticBlasTriCount - triStart) * sizeof(BlasCache::GpuTri));
            wgpuQueueRelease(q);
            m_blasNodesDirty         = false;
            m_blasFullRebuildPending = false;
            m_uploadedNodeCount      = m_staticBlasNodeCount;
            m_uploadedTriCount       = m_staticBlasTriCount;
        }
    }

    // Update stats; keep node count / root AABB from the last full rebuild when nodes are cached.
    if (m_tlasNodesDirty) {
        m_lastStats.tlasNodeCount = static_cast<uint32_t>(m_tlasNodes.size());
        m_lastStats.instanceCount = static_cast<uint32_t>(m_instances.size());
        if (!m_tlasNodes.empty()) {
            const AABB& r = m_tlasNodes[0].bounds;
            m_lastStats.rootAabbMin[0]=r.min.x; m_lastStats.rootAabbMin[1]=r.min.y; m_lastStats.rootAabbMin[2]=r.min.z;
            m_lastStats.rootAabbMax[0]=r.max.x; m_lastStats.rootAabbMax[1]=r.max.y; m_lastStats.rootAabbMax[2]=r.max.z;
        }
        m_lastTlasGpuBytes =
            m_tlasNodes.size() * sizeof(BlasCache::GpuNode) +
            m_instances.size() * sizeof(GpuTlasInstance);
    }
    m_lastStats.blasEntryCount = static_cast<uint32_t>(m_blasOffsets.size());
    m_lastStats.blasNodeTotal  = m_staticBlasNodeCount;
    m_lastStats.blasTriTotal   = m_staticBlasTriCount;
    m_lastStats.gpuBytesTotal = m_lastTlasGpuBytes +
        uint64_t(m_blasNodeBufCapacity) * sizeof(BlasCache::GpuNode) +
        uint64_t(m_blasTriBufCapacity)  * sizeof(BlasCache::GpuTri);
    m_lastStats.flushMs = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

void TlasBuilder::advance_frame() {
    m_tlasNodes.clear();
    m_instances.clear();
}

} // namespace dusk::rtao
