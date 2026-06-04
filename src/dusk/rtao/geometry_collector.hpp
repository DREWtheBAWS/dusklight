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

    struct CameraData {
        float proj[4][4] = {};   // GX projection matrix (row-major), from the first qualifying draw
        float view[4][4] = {};   // reserved for future world-space use
        float worldPos[3] = {};  // reserved for future world-space use
        float fovYDeg = 0.f;
        bool  valid = false;
    };

    // Call once at startup to register the Aurora capture callback.
    void install();

    // Called at the end of each ImGui frame (from afterDraw).
    void end_frame();

    // Request that the next end_frame() writes an OBJ to path.
    void request_dump(std::string path);

    Stats              last_stats()          const { return m_lastStats; }
    const std::string& last_dump_message()   const { return m_lastDumpMsg; }
    CameraData         last_camera_data()    const { return m_lastCameraData; }
    CameraData         pending_camera_data() const { return m_pendingCameraData; }

    const std::vector<Triangle>& raw_triangles() const { return m_triangles; }

    void set_filter(bool perspectiveOnly, float minViewportW = 320.f, float minViewportH = 240.f) {
        m_perspectiveOnly = perspectiveOnly;
        m_minViewportW    = minViewportW;
        m_minViewportH    = minViewportH;
    }

    // Discard triangles whose centroid is farther than radius from the camera.
    // In view space the camera is at the origin, so length(centroid) is the
    // camera distance — same units as the AO pass maxDistance.  Pass 0 to disable.
    void set_max_distance(float radius) { m_maxAoDistance = radius; }

    // Discard triangles outside the view frustum + this margin.
    // Set to the AO ray length so only geometry that can actually cast shadows
    // on visible surfaces is collected, keeping the Morton AABB tight.
    // Pass 0 to disable frustum culling.
    void set_frustum_margin(float margin) { m_frustumMargin = margin; }

    // Subdivide triangles whose longest edge exceeds this length (longest-edge bisection,
    // up to 8 virtual sub-triangles per input).  Large terrain triangles inflate every
    // ancestor AABB in the BVH; splitting them gives each piece its own tight Morton code.
    // A good default is 3× the AO ray length.  Pass 0 to disable.
    void set_max_edge_length(float len) { m_maxEdgeLen = len; }

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

    CameraData  m_pendingCameraData;
    CameraData  m_lastCameraData;

    bool  m_pendingTriClear = false;
    bool  m_perspectiveOnly = true;
    float m_minViewportW    = 320.f;
    float m_minViewportH    = 240.f;
    float m_maxAoDistance   = 0.f;
    float m_frustumMargin   = 0.f;
    float m_maxEdgeLen      = 0.f;

    static constexpr uint32_t kMaxTriangles = 500'000;
};

} // namespace dusk::rtao
