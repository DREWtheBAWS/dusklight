#include "geometry_collector.hpp"
#include <aurora/geometry_capture.h>
#include <fstream>

namespace dusk::rtao {

void GeometryCollector::simulate_draw(const AuroraGxCaptureDraw& draw) {
    process_draw(&draw);
}

void GeometryCollector::on_capture(const AuroraGxCaptureDraw* draw, void* userdata) {
    static_cast<GeometryCollector*>(userdata)->process_draw(draw);
}

void GeometryCollector::process_draw(const AuroraGxCaptureDraw* draw) {
    if (draw->indexCount == 0 || m_triangles.size() >= kMaxTriangles) return;
    ++m_drawCallCount;
    auto tris = decode_triangles(*draw);
    for (auto& t : tris) {
        if (m_triangles.size() >= kMaxTriangles) break;
        m_triangles.push_back(t);
    }
}

void GeometryCollector::end_frame() {
    m_lastStats = { static_cast<uint32_t>(m_triangles.size()), m_drawCallCount };

    if (!m_pendingDumpPath.empty()) {
        if (write_obj(m_pendingDumpPath)) {
            m_lastDumpMsg = "Saved " + std::to_string(m_lastStats.triangleCount) +
                            " tris to " + m_pendingDumpPath;
        } else {
            m_lastDumpMsg = "Write failed: " + m_pendingDumpPath;
        }
        m_pendingDumpPath.clear();
    }

    m_triangles.clear();
    m_drawCallCount = 0;
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
