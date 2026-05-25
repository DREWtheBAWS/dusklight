#pragma once
#include "vertex_decoder.hpp"
#include <string>
#include <vector>

struct AuroraGxCaptureDraw;

namespace dusk::rtao {

class GeometryCollector {
public:
    struct Stats {
        uint32_t triangleCount = 0;
        uint32_t drawCallCount = 0;
    };

    // Call once at startup to register the Aurora capture callback.
    // Defined in geometry_collector_aurora.cpp (not linked into tests).
    void install();

    // Called at the end of each ImGui frame (from afterDraw).
    // Saves stats for display, handles pending OBJ dump, then resets.
    void end_frame();

    // Request that the next end_frame() writes an OBJ to path.
    void request_dump(std::string path);

    Stats          last_stats()       const { return m_lastStats; }
    const std::string& last_dump_message() const { return m_lastDumpMsg; }

    // Drives the callback directly — used by unit tests and simulate_draw.
    void simulate_draw(const AuroraGxCaptureDraw& draw);

private:
    static void on_capture(const AuroraGxCaptureDraw* draw, void* userdata);
    void process_draw(const AuroraGxCaptureDraw* draw);
    bool write_obj(const std::string& path) const;

    std::vector<Triangle> m_triangles;
    uint32_t m_drawCallCount = 0;

    Stats       m_lastStats;
    std::string m_pendingDumpPath;
    std::string m_lastDumpMsg;

    static constexpr uint32_t kMaxTriangles = 500'000;
};

} // namespace dusk::rtao
