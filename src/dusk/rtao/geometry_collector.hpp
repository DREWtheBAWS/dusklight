#pragma once
#include "bvh.hpp"
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

    struct BvhStats {
        uint32_t nodeCount = 0;
        float    buildMs   = 0.f;
    };

    struct CameraData {
        float proj[4][4] = {};   // GX projection matrix (row-major)
        float view[4][4] = {};   // model-view matrix for the first qualifying draw, extended to 4x4
        float worldPos[3] = {};  // camera world-space position (derived from view inverse)
        float fovYDeg = 0.f;     // vertical FOV in degrees (from proj[1][1])
        bool  valid = false;
    };

    // Call once at startup to register the Aurora capture callback.
    // Defined in geometry_collector_aurora.cpp (not linked into tests).
    void install();

    // Called at the end of each ImGui frame (from afterDraw).
    // Saves stats for display, handles pending OBJ dump, then resets.
    void end_frame();

    // Request that the next end_frame() writes an OBJ to path.
    void request_dump(std::string path);

    Stats              last_stats()        const { return m_lastStats; }
    const std::string& last_dump_message() const { return m_lastDumpMsg; }
    CameraData         last_camera_data()  const { return m_lastCameraData; }

    // Triggers a BVH build from the current frame's geometry on the next end_frame().
    // The result persists until the next build is requested.
    void          request_bvh_build();
    BvhStats      last_bvh_stats()  const { return m_lastBvhStats; }
    const Bvh&    bvh()             const { return m_bvh; }
    // Monotonically incremented each time a BVH is built. Use to detect new builds.
    uint32_t      bvh_generation()  const { return m_bvhGeneration; }

    // Filter: only collect perspective draws whose viewport meets minimum dimensions.
    // Defaults keep UI, HUD, and small shadow-map passes out of the capture.
    // Pass 0,0 to disable viewport filtering; set perspectiveOnly=false to also
    // capture orthographic draws.
    void set_filter(bool perspectiveOnly, float minViewportW = 320.f, float minViewportH = 240.f) {
        m_perspectiveOnly  = perspectiveOnly;
        m_minViewportW     = minViewportW;
        m_minViewportH     = minViewportH;
    }

    // Drives the callback directly — used by unit tests and simulate_draw.
    void simulate_draw(const AuroraGxCaptureDraw& draw);

private:
    static void on_capture(const AuroraGxCaptureDraw* draw, void* userdata);
    void process_draw(const AuroraGxCaptureDraw* draw);
    bool write_obj(const std::string& path) const;

    static CameraData extract_camera_data(const AuroraGxCaptureDraw* draw);

    std::vector<Triangle> m_triangles;
    uint32_t m_drawCallCount = 0;

    Stats       m_lastStats;
    std::string m_pendingDumpPath;
    std::string m_lastDumpMsg;

    CameraData  m_pendingCameraData;
    CameraData  m_lastCameraData;

    Bvh      m_bvh;
    BvhStats m_lastBvhStats;
    uint32_t m_bvhGeneration  = 0;
    bool     m_pendingBvhBuild = false;

    bool  m_perspectiveOnly = true;
    float m_minViewportW    = 320.f;
    float m_minViewportH    = 240.f;

    static constexpr uint32_t kMaxTriangles = 500'000;
};

} // namespace dusk::rtao
