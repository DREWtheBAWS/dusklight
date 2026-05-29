#include "geometry_collector.hpp"
#include <aurora/geometry_capture.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>

namespace dusk::rtao {

void GeometryCollector::simulate_draw(const AuroraGxCaptureDraw& draw) {
    process_draw(&draw);
}

void GeometryCollector::on_capture(const AuroraGxCaptureDraw* draw, void* userdata) {
    static_cast<GeometryCollector*>(userdata)->process_draw(draw);
}

GeometryCollector::CameraData GeometryCollector::extract_camera_data(const AuroraGxCaptureDraw* draw) {
    CameraData data;

    // View: pnMtx[currentPnMtx] (3×4) extended to 4×4
    const auto& vm = draw->pnMtx[draw->currentPnMtx];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            data.view[r][c] = vm[r][c];
    data.view[3][0] = 0.f; data.view[3][1] = 0.f;
    data.view[3][2] = 0.f; data.view[3][3] = 1.f;

    // Projection matrix
    memcpy(data.proj, draw->projMtx, sizeof(data.proj));

    // Camera world position: cam = -R^T * t  (R = view[0:3][0:3], t = view[*][3])
    data.worldPos[0] = -(vm[0][0]*vm[0][3] + vm[1][0]*vm[1][3] + vm[2][0]*vm[2][3]);
    data.worldPos[1] = -(vm[0][1]*vm[0][3] + vm[1][1]*vm[1][3] + vm[2][1]*vm[2][3]);
    data.worldPos[2] = -(vm[0][2]*vm[0][3] + vm[1][2]*vm[1][3] + vm[2][2]*vm[2][3]);

    // FOV Y: proj[1][1] == 1/tan(fovY/2) for symmetric perspective
    const float p11 = draw->projMtx[1][1];
    if (p11 > 0.f)
        data.fovYDeg = 2.f * std::atan(1.f / p11) * (180.f / 3.14159265f);

    data.valid = true;
    return data;
}

void GeometryCollector::process_draw(const AuroraGxCaptureDraw* draw) {
    if (draw->indexCount == 0 || m_triangles.size() >= kMaxTriangles) return;

    // Filter: reject orthographic draws (UI, shadow maps) and draws on small
    // render targets (thumbnail/shadow-map perspective passes).
    if (m_perspectiveOnly && draw->projType != 0) return;  // 0 = GX_PERSPECTIVE
    if (m_minViewportW > 0.f && draw->viewportWidth  < m_minViewportW) return;
    if (m_minViewportH > 0.f && draw->viewportHeight < m_minViewportH) return;

    if (!m_pendingCameraData.valid)
        m_pendingCameraData = extract_camera_data(draw);

    ++m_drawCallCount;
    auto tris = decode_triangles(*draw);
    for (auto& t : tris) {
        if (m_triangles.size() >= kMaxTriangles) break;
        m_triangles.push_back(t);
    }
}

void GeometryCollector::end_frame() {
    m_lastStats = { static_cast<uint32_t>(m_triangles.size()), m_drawCallCount };
    m_lastCameraData    = m_pendingCameraData;
    m_pendingCameraData = {};

    if (!m_pendingDumpPath.empty()) {
        if (write_obj(m_pendingDumpPath)) {
            m_lastDumpMsg = "Saved " + std::to_string(m_lastStats.triangleCount) +
                            " tris to " + m_pendingDumpPath;
        } else {
            m_lastDumpMsg = "Write failed: " + m_pendingDumpPath;
        }
        m_pendingDumpPath.clear();
    }

    if (m_pendingBvhBuild) {
        const auto t0 = std::chrono::steady_clock::now();
        m_bvh.build(m_triangles);
        const auto t1 = std::chrono::steady_clock::now();
        m_lastBvhStats = {
            m_bvh.node_count(),
            std::chrono::duration<float, std::milli>(t1 - t0).count(),
        };
        ++m_bvhGeneration;
        m_pendingBvhBuild = false;
    }

    m_triangles.clear();
    m_drawCallCount = 0;
}

void GeometryCollector::request_dump(std::string path) {
    m_pendingDumpPath = std::move(path);
}

void GeometryCollector::request_bvh_build() {
    m_pendingBvhBuild = true;
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
