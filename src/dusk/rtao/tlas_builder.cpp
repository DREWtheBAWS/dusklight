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

void TlasBuilder::build(const BlasCache& cache) {
    const auto t0 = std::chrono::high_resolution_clock::now();

    const auto& entries   = cache.entries();
    const auto& instances = cache.instances();

    // Rebuild/update the monolithic BLAS staging data when cache entries change.
    if (cache.generation() != m_lastBlasGeneration) {
        const bool fullRebuild = (cache.eviction_generation() != m_lastEvictionGeneration);
        if (fullRebuild) {
            m_blasOffsets.clear();
            m_pendingBlasNodes.clear();
            m_pendingBlasTris.clear();
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

    // Pack dynamic (skinned) BLAS.
    m_dynBlasNodes.clear();
    m_dynBlasTris.clear();
    if (cache.has_dynamic_entry()) {
        const auto& dyn = cache.dynamic_entry();
        const uint32_t nodeBase = m_staticBlasNodeCount;
        const uint32_t triBase  = m_staticBlasTriCount;

        for (uint32_t i = 0; i < dyn.nodeCount; ++i) {
            const BvhNode& n = dyn.bvh.nodes()[i];
            BlasCache::GpuNode gn{};
            gn.aabb_min[0]=n.bounds.min.x; gn.aabb_min[1]=n.bounds.min.y; gn.aabb_min[2]=n.bounds.min.z;
            gn.hit_next  = (n.hit_next  == UINT32_MAX) ? UINT32_MAX : n.hit_next  + nodeBase;
            gn.aabb_max[0]=n.bounds.max.x; gn.aabb_max[1]=n.bounds.max.y; gn.aabb_max[2]=n.bounds.max.z;
            gn.miss_next = (n.miss_next == UINT32_MAX) ? UINT32_MAX : n.miss_next + nodeBase;
            gn.tri_offset = n.tri_offset + triBase;
            gn.tri_count  = n.tri_count;
            m_dynBlasNodes.push_back(gn);
        }
        for (uint32_t i = 0; i < dyn.triCount; ++i) {
            const Triangle& t = dyn.bvh.tris()[i];
            BlasCache::GpuTri gt{};
            gt.v[0][0]=t.a.x; gt.v[0][1]=t.a.y; gt.v[0][2]=t.a.z; gt.v[0][3]=0.f;
            gt.v[1][0]=t.b.x; gt.v[1][1]=t.b.y; gt.v[1][2]=t.b.z; gt.v[1][3]=0.f;
            gt.v[2][0]=t.c.x; gt.v[2][1]=t.c.y; gt.v[2][2]=t.c.z; gt.v[2][3]=0.f;
            m_dynBlasTris.push_back(gt);
        }
    }

    // Build TLAS instance list from current-frame instances (full rebuild every frame).
    m_instances.clear();
    m_instances.reserve(instances.size() + 1);
    m_dedupSeen.clear();
    uint32_t dedupRej = 0;

    for (const auto& inst : instances) {
        // Dedup by (blasKey, pnMtx bytes).
        size_t h = BlasKeyHash{}(inst.blasKey);
        const auto* b = reinterpret_cast<const uint8_t*>(inst.pnMtx);
        for (int k = 0; k < 48; k += 4) {
            uint32_t w; std::memcpy(&w, b+k, 4);
            h ^= size_t(w)*2654435761u + (h<<6) + (h>>2);
        }
        if (!m_dedupSeen.insert(h).second) { ++dedupRej; continue; }

        const auto eit = entries.find(inst.blasKey);
        if (eit == entries.end() || eit->second.bvh.empty()) continue;
        const auto oit = m_blasOffsets.find(inst.blasKey);
        if (oit == m_blasOffsets.end()) continue;

        CpuInstance ci;
        if (!invert_3x4(inst.pnMtx, ci.pnMtxInv)) continue;

        ci.viewAabb = {};
        const AABB& lb = eit->second.localAabb;
        for (int c = 0; c < 8; ++c) {
            ci.viewAabb.expand(xf_point(inst.pnMtx, {
                (c&1)?lb.max.x:lb.min.x, (c&2)?lb.max.y:lb.min.y, (c&4)?lb.max.z:lb.min.z
            }));
        }
        if (!ci.viewAabb.valid()) continue;

        ci.blasNodeOffset = oit->second.first;
        ci.blasTriOffset  = oit->second.second;
        ci.blasNodeCount  = eit->second.nodeCount;
        ci.blasTriCount   = eit->second.triCount;
        m_instances.push_back(ci);
    }
    m_lastStats.dedupRejected = dedupRej;

    // Add dynamic (skinned) BLAS as identity-transform instance (already in view space).
    if (!m_dynBlasNodes.empty()) {
        const auto& dyn = cache.dynamic_entry();
        CpuInstance di{};
        di.pnMtxInv[0][0]=1.f; di.pnMtxInv[0][1]=0.f; di.pnMtxInv[0][2]=0.f; di.pnMtxInv[0][3]=0.f;
        di.pnMtxInv[1][0]=0.f; di.pnMtxInv[1][1]=1.f; di.pnMtxInv[1][2]=0.f; di.pnMtxInv[1][3]=0.f;
        di.pnMtxInv[2][0]=0.f; di.pnMtxInv[2][1]=0.f; di.pnMtxInv[2][2]=1.f; di.pnMtxInv[2][3]=0.f;
        di.viewAabb       = dyn.viewAabb;
        di.blasNodeOffset = m_staticBlasNodeCount;
        di.blasTriOffset  = m_staticBlasTriCount;
        di.blasNodeCount  = dyn.nodeCount;
        di.blasTriCount   = dyn.triCount;
        if (di.viewAabb.valid()) m_instances.push_back(di);
    }

    // Build SAH BVH over view-space AABBs.
    m_tlasNodes.clear();
    if (m_instances.empty()) {
        m_lastStats.buildMs = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return;
    }

    std::vector<TlasItem> items(m_instances.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_instances.size()); ++i)
        items[i] = { m_instances[i].viewAabb, i };

    m_tlasNodes.reserve(2 * m_instances.size());
    tlas_build_node(items, m_tlasNodes, 0, static_cast<uint32_t>(items.size()));
    tlas_link_node(m_tlasNodes, 0, kTlasEnd);

    // Reorder instances to match BVH leaf traversal order.
    std::vector<CpuInstance> sorted(m_instances.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(items.size()); ++i)
        sorted[i] = m_instances[items[i].instIdx];
    m_instances = std::move(sorted);

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
    release_buf(m_tlasNodeBuf);
    release_buf(m_instanceBuf);

    if (m_tlasNodes.empty()) return;

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
        m_instanceBuf = upload_gpu(device, gpu.data(), gpu.size()*sizeof(GpuTlasInstance));
    }

    // Combined BLAS buffer.
    {
        const uint32_t totalNodes = m_staticBlasNodeCount + static_cast<uint32_t>(m_dynBlasNodes.size());
        const uint32_t totalTris  = m_staticBlasTriCount  + static_cast<uint32_t>(m_dynBlasTris.size());
        const uint32_t minNodes   = std::max(totalNodes, 1u);
        const uint32_t minTris    = std::max(totalTris,  1u);

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

        WGPUQueue q = wgpuDeviceGetQueue(device);

        if (m_blasNodesDirty) {
            if (m_staticBlasNodeCount > 0)
                wgpuQueueWriteBuffer(q, m_blasNodeBuf, 0,
                    m_pendingBlasNodes.data(), m_staticBlasNodeCount * sizeof(BlasCache::GpuNode));
            if (m_staticBlasTriCount > 0)
                wgpuQueueWriteBuffer(q, m_blasTriBuf, 0,
                    m_pendingBlasTris.data(),  m_staticBlasTriCount  * sizeof(BlasCache::GpuTri));
            m_blasNodesDirty = false;
        }

        if (!m_dynBlasNodes.empty()) {
            wgpuQueueWriteBuffer(q, m_blasNodeBuf,
                uint64_t(m_staticBlasNodeCount) * sizeof(BlasCache::GpuNode),
                m_dynBlasNodes.data(), m_dynBlasNodes.size() * sizeof(BlasCache::GpuNode));
        }
        if (!m_dynBlasTris.empty()) {
            wgpuQueueWriteBuffer(q, m_blasTriBuf,
                uint64_t(m_staticBlasTriCount) * sizeof(BlasCache::GpuTri),
                m_dynBlasTris.data(), m_dynBlasTris.size() * sizeof(BlasCache::GpuTri));
        }

        wgpuQueueRelease(q);
    }

    m_lastStats.tlasNodeCount  = static_cast<uint32_t>(m_tlasNodes.size());
    m_lastStats.instanceCount  = static_cast<uint32_t>(m_instances.size());
    m_lastStats.blasEntryCount = static_cast<uint32_t>(m_blasOffsets.size());
    m_lastStats.blasNodeTotal  = m_staticBlasNodeCount + static_cast<uint32_t>(m_dynBlasNodes.size());
    m_lastStats.blasTriTotal   = m_staticBlasTriCount  + static_cast<uint32_t>(m_dynBlasTris.size());
    if (!m_tlasNodes.empty()) {
        const AABB& r = m_tlasNodes[0].bounds;
        m_lastStats.rootAabbMin[0]=r.min.x; m_lastStats.rootAabbMin[1]=r.min.y; m_lastStats.rootAabbMin[2]=r.min.z;
        m_lastStats.rootAabbMax[0]=r.max.x; m_lastStats.rootAabbMax[1]=r.max.y; m_lastStats.rootAabbMax[2]=r.max.z;
    }
    const uint64_t blasBytes =
        uint64_t(m_blasNodeBufCapacity) * sizeof(BlasCache::GpuNode) +
        uint64_t(m_blasTriBufCapacity)  * sizeof(BlasCache::GpuTri);
    m_lastStats.gpuBytesTotal =
        m_tlasNodes.size()  * sizeof(BlasCache::GpuNode) +
        m_instances.size()  * sizeof(GpuTlasInstance) +
        blasBytes;
    m_lastStats.flushMs = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

void TlasBuilder::advance_frame() {
    m_tlasNodes.clear();
    m_instances.clear();
}

} // namespace dusk::rtao
