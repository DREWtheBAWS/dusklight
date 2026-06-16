#include "geometry_collector.hpp"
#include <aurora/geometry_capture.h>
#include <cstring>
#include <fstream>

namespace dusk::rtao {

void GeometryCollector::simulate_draw(const AuroraGxCaptureDraw& draw) {
    process_draw(&draw);
}

void GeometryCollector::on_capture(const AuroraGxCaptureDraw* draw, void* userdata) {
    static_cast<GeometryCollector*>(userdata)->process_draw(draw);
}

static Vec3 midpoint(const Vec3& a, const Vec3& b) {
    return {(a.x+b.x)*0.5f, (a.y+b.y)*0.5f, (a.z+b.z)*0.5f};
}

static float edge2(const Vec3& a, const Vec3& b) {
    float dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
    return dx*dx+dy*dy+dz*dz;
}

// Recursive longest-edge bisection. Stops when all edges are within th2 or depth is 0.
// Depth 3 → up to 8 virtual sub-triangles per oversized input triangle.
static void subdivide(const Triangle& t, float th2, int depth,
                      std::vector<Triangle>& out, uint32_t limit) {
    if (out.size() >= limit) return;
    const float eAB = edge2(t.a, t.b);
    const float eBC = edge2(t.b, t.c);
    const float eCA = edge2(t.c, t.a);
    if (depth == 0 || (eAB <= th2 && eBC <= th2 && eCA <= th2)) {
        out.push_back(t);
        return;
    }
    if (eAB >= eBC && eAB >= eCA) {
        Vec3 m = midpoint(t.a, t.b);
        subdivide({t.a, m,   t.c}, th2, depth-1, out, limit);
        subdivide({m,   t.b, t.c}, th2, depth-1, out, limit);
    } else if (eBC >= eCA) {
        Vec3 m = midpoint(t.b, t.c);
        subdivide({t.a, t.b, m  }, th2, depth-1, out, limit);
        subdivide({t.a, m,   t.c}, th2, depth-1, out, limit);
    } else {
        Vec3 m = midpoint(t.c, t.a);
        subdivide({t.a, t.b, m  }, th2, depth-1, out, limit);
        subdivide({m,   t.b, t.c}, th2, depth-1, out, limit);
    }
}

void GeometryCollector::process_draw(const AuroraGxCaptureDraw* draw) {
    // Deferred clear: discard previous frame's triangles on the first draw of a new frame
    // so raw_triangles() stays valid through the post-render callback.
    if (m_pendingTriClear) {
        m_triangles.clear();
        m_drawCallCount      = 0;
        m_pendingCameraData  = {};
        m_texViewToSlot.clear();
        m_textureViews.clear();
        m_totalAlphaTexCount = 0;
        m_pendingTriClear    = false;
    }

    if (draw->indexCount == 0 || m_triangles.size() >= kMaxTriangles) return;

    // Skip orthographic draws (UI, HUD) and small render targets (shadow maps, thumbnails).
    if (m_perspectiveOnly && draw->projType != 0) return;
    if (m_minViewportW > 0.f && draw->viewportWidth  < m_minViewportW) return;
    if (m_minViewportH > 0.f && draw->viewportHeight < m_minViewportH) return;

    // Skip skybox: its pnMtx is a pure rotation (translation column = 0) so the scene
    // rotates with the camera but never translates. Any real-geometry draw has a non-zero
    // translation because the camera is not at the world origin.
    {
        const auto& vm = draw->pnMtx[draw->currentPnMtx];
        const float tx = vm[0][3], ty = vm[1][3], tz = vm[2][3];
        if (tx*tx + ty*ty + tz*tz < 1e-6f) return;
    }

    // Fire the optional per-draw callback (used by BlasCache::record_draw).
    if (m_drawCb) {
        m_drawCb(*draw, m_drawCbUserdata);
        ++m_drawCbFiredTotal;
    }

    // Capture projection and view matrices from the first qualifying draw.
    // GX slot 0 is loaded with the pure view matrix (world→view) before any object draws.
    if (!m_pendingCameraData.valid) {
        memcpy(m_pendingCameraData.proj, draw->projMtx, sizeof(m_pendingCameraData.proj));
        const float (&vm)[3][4] = draw->pnMtx[0];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                m_pendingCameraData.view[r][c] = vm[r][c];
        m_pendingCameraData.view[3][0] = 0.f;
        m_pendingCameraData.view[3][1] = 0.f;
        m_pendingCameraData.view[3][2] = 0.f;
        m_pendingCameraData.view[3][3] = 1.f;
        m_pendingCameraData.valid = true;
    }

    ++m_drawCallCount;

    // Register alpha-tested texture before decoding triangles so we can pass the slot.
    // kNoSlot means "overflow / no valid slot" — decode_uvs will leave flags=0.
    static constexpr uint32_t kNoSlot = 0xFFFFFFFFu;
    uint32_t texSlot = kNoSlot;
    // GX combined alpha test: result = op(comp0(alpha, ref0), comp1(alpha, ref1)).
    // Trivially passes (no clipping) when both operands are ALWAYS, or when
    // op is OR and at least one operand is ALWAYS.
    static constexpr uint8_t kAlways = 7; // GXCompare::GX_ALWAYS
    static constexpr uint8_t kAopOr  = 1; // GXAlphaOp::GX_AOP_OR
    const bool comp0Always = (draw->alphaComp0 == kAlways);
    const bool comp1Always = (draw->alphaComp1 == kAlways);
    const bool trivialPass = (comp0Always && comp1Always)
                          || (draw->alphaOp == kAopOr && (comp0Always || comp1Always));
    if (draw->tex0View && draw->tex0HasAlpha && !trivialPass) {
        auto [it, inserted] = m_texViewToSlot.emplace(draw->tex0View,
                                                       static_cast<uint32_t>(m_textureViews.size()));
        if (inserted) {
            ++m_totalAlphaTexCount; // count all unique textures, even those that overflow
            if (m_textureViews.size() < kMaxTexSlots) {
                m_textureViews.push_back(draw->tex0View);
            } else {
                m_texViewToSlot.erase(it); // slot table full — treat as opaque
            }
        }
        auto found = m_texViewToSlot.find(draw->tex0View);
        if (found != m_texViewToSlot.end())
            texSlot = found->second;
    }

    auto tris = decode_triangles(*draw);
    decode_uvs(*draw, tris, texSlot);
    const float r2  = m_maxAoDistance * m_maxAoDistance;
    // Projection params for frustum culling: cot(fovX/2) and cot(fovY/2).
    // The first qualifying draw already set m_pendingCameraData.proj above, so these are valid.
    const float p00 = m_pendingCameraData.proj[0][0];
    const float p11 = m_pendingCameraData.proj[1][1];
    for (auto& t : tris) {
        if (m_triangles.size() >= kMaxTriangles) break;
        if (r2 > 0.f) {
            const float cx = (t.a.x + t.b.x + t.c.x) * (1.f / 3.f);
            const float cy = (t.a.y + t.b.y + t.c.y) * (1.f / 3.f);
            const float cz = (t.a.z + t.b.z + t.c.z) * (1.f / 3.f);
            // Sphere: too far from camera in any direction.
            if (cx*cx + cy*cy + cz*cz > r2) continue;
            // Frustum + margin: only keep geometry that can contribute AO to visible surfaces.
            // GX view space: -Z is forward, camera at origin.
            if (m_frustumMargin > 0.f && p00 > 0.f && p11 > 0.f) {
                // Behind the camera: keep only if within margin (AO rays from near surfaces can
                // still reach just behind the camera, but not far behind it).
                if (cz > m_frustumMargin) continue;
                // In front: discard if outside the frustum cone expanded by margin.
                // frustum half-width at depth d = d / p00; half-height = d / p11.
                if (cz < 0.f) {
                    const float d = -cz;
                    if (std::abs(cx) > d / p00 + m_frustumMargin) continue;
                    if (std::abs(cy) > d / p11 + m_frustumMargin) continue;
                }
            }
            // Large-triangle subdivision: bisect on the longest edge (depth 3 → up to 8
            // virtual sub-tris).  A single large terrain patch inflates every ancestor AABB
            // in the BVH; splitting gives each piece its own tight Morton code instead.
            if (m_maxEdgeLen > 0.f) {
                const float th2 = m_maxEdgeLen * m_maxEdgeLen;
                if (edge2(t.a,t.b) > th2 || edge2(t.b,t.c) > th2 || edge2(t.a,t.c) > th2) {
                    subdivide(t, th2, 3, m_triangles, kMaxTriangles);
                    continue;
                }
            }
        }
        m_triangles.push_back(t);
    }
}

void GeometryCollector::end_frame() {
    m_lastStats      = { static_cast<uint32_t>(m_triangles.size()), m_drawCallCount };
    m_lastCameraData = m_pendingCameraData;

    if (!m_pendingDumpPath.empty()) {
        if (write_obj(m_pendingDumpPath)) {
            m_lastDumpMsg = "Saved " + std::to_string(m_lastStats.triangleCount) +
                            " tris to " + m_pendingDumpPath;
        } else {
            m_lastDumpMsg = "Write failed: " + m_pendingDumpPath;
        }
        m_pendingDumpPath.clear();
    }

    m_pendingTriClear = true;
}

void GeometryCollector::request_dump(std::string path) {
    m_pendingDumpPath = std::move(path);
}

bool GeometryCollector::write_obj(const std::string& path) const {
    std::ofstream f(path);
    if (!f) return false;

    f << "# Dusklight RTAO capture - " << m_triangles.size() << " triangles\n";
    for (const auto& t : m_triangles) {
        f << "v " << t.a.x << ' ' << t.a.y << ' ' << t.a.z << '\n';
        f << "v " << t.b.x << ' ' << t.b.y << ' ' << t.b.z << '\n';
        f << "v " << t.c.x << ' ' << t.c.y << ' ' << t.c.z << '\n';
    }
    const uint32_t n = static_cast<uint32_t>(m_triangles.size());
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t base = i * 3 + 1;
        f << "f " << base << ' ' << base + 1 << ' ' << base + 2 << '\n';
    }
    return f.good();
}

} // namespace dusk::rtao
