#pragma once
#include "vertex_decoder.hpp"
#include <string>
#include <unordered_map>
#include <vector>

struct AuroraGxCaptureDraw;


namespace dusk::rtao {

class GeometryCollector {
public:
    struct Stats {
        uint32_t triangleCount = 0;
        uint32_t drawCallCount = 0;
        float    decodeMs      = 0.f; // CPU time spent in decode/cull/subdivide this frame
    };

    struct CameraData {
        float proj[4][4] = {};       // GX projection matrix (row-major), from the first qualifying draw
        float view[4][4] = {};       // GX view matrix (world→view), from GX pnMtx slot 0
        float worldPos[3] = {};      // reserved for future world-space use
        float fovYDeg = 0.f;
        float lightWorldPos[3] = {}; // world-space light position (from dKy_plight_near_pos)
        bool  valid = false;
    };

    // Register the Aurora geometry capture callback. Safe to call from render worker
    // thread when main thread is blocked in acquire_frame_slot.
    void install();
    // Clear the capture callback. Calling this stops the per-draw CPU work for
    // triangle decode/subdivision until install() is called again.
    void uninstall();

    // Called at the end of each ImGui frame (from afterDraw).
    void end_frame();

    // Request that the next end_frame() writes an OBJ to path.
    void request_dump(std::string path);

    Stats              last_stats()          const { return m_lastStats; }
    const std::string& last_dump_message()   const { return m_lastDumpMsg; }
    CameraData         last_camera_data()    const { return m_lastCameraData; }
    CameraData         pending_camera_data() const { return m_pendingCameraData; }

    const std::vector<Triangle>& raw_triangles()   const { return m_triangles; }
    // Per-frame ordered list of WGPUTextureView pointers (as void*) for alpha-tested
    // textures, indexed by Triangle::texIdx.  Valid until the next end_frame().
    const std::vector<void*>&   texture_views()    const { return m_textureViews; }
    // Total unique alpha textures seen this frame (may exceed kMaxTexSlots).
    uint32_t                    total_alpha_tex_count() const { return m_totalAlphaTexCount; }

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

    // Enable/disable full triangle decode + collection in process_draw().
    // In BLAS/TLAS mode the decoded triangles feed nothing — instances come from
    // the draw callback (BlasCache) and camera/texture capture happens before the
    // decode — so collection is disabled to skip the per-draw decode/cull/subdivide
    // cost on the main thread.  A pending OBJ dump forces collection back on until
    // a frame's triangles have been written (see request_dump()).
    void set_collect_triangles(bool v) { m_collectTriangles = v; }

    // Optional per-draw callback, fired for each qualifying draw call after the
    // projection/viewport/skybox filters.  Used to feed the BLAS cache without
    // introducing a webgpu header dependency into the test-linked geometry collector.
    using DrawCallback = void (*)(const AuroraGxCaptureDraw&, void* userdata);
    void set_draw_callback(DrawCallback cb, void* userdata) {
        m_drawCb = cb; m_drawCbUserdata = userdata;
    }

    void set_light_world_pos(float x, float y, float z) {
        m_pendingCameraData.lightWorldPos[0] = x;
        m_pendingCameraData.lightWorldPos[1] = y;
        m_pendingCameraData.lightWorldPos[2] = z;
    }

    // Diagnostic: total number of times the draw callback was actually invoked.
    // Accumulates forever (never reset).  If this stays 0, the callback is null.
    uint32_t draw_callback_fired_total() const { return m_drawCbFiredTotal; }

    void simulate_draw(const AuroraGxCaptureDraw& draw);

private:
    static void on_capture(const AuroraGxCaptureDraw* draw, void* userdata);
    void process_draw(const AuroraGxCaptureDraw* draw);
    bool write_obj(const std::string& path) const;

    std::vector<Triangle> m_triangles;
    uint32_t m_drawCallCount = 0;
    float    m_decodeMsAccum = 0.f;

    Stats       m_lastStats;
    std::string m_pendingDumpPath;
    std::string m_lastDumpMsg;

    CameraData  m_pendingCameraData;
    CameraData  m_lastCameraData;

    bool  m_pendingTriClear   = false;
    bool  m_collectTriangles  = true;  // false in TLAS mode (decode skipped per draw)
    bool  m_forceCollectFrame = false; // armed by request_dump() while collection is off
    bool  m_perspectiveOnly = true;
    float m_minViewportW    = 320.f;
    float m_minViewportH    = 240.f;
    float m_maxAoDistance   = 0.f;
    float m_frustumMargin   = 0.f;
    float m_maxEdgeLen      = 0.f;

    // Per-frame alpha-texture registry (max kMaxTexSlots entries).
    std::unordered_map<void*, uint32_t> m_texViewToSlot;
    std::vector<void*>                  m_textureViews;
    uint32_t                            m_totalAlphaTexCount = 0;

    DrawCallback m_drawCb         = nullptr;
    void*        m_drawCbUserdata = nullptr;
    uint32_t     m_drawCbFiredTotal = 0;

    // Mid-frame camera switch detection (reflection pre-pass → main camera).
    float    m_switchCandidatePos[3] = {};
    uint32_t m_switchCandidateCount  = 0;
    static constexpr uint32_t kCameraSwitchThreshold = 5;

    static constexpr uint32_t kMaxTriangles = 500'000;
    static constexpr uint32_t kMaxTexSlots  = 16;
};

} // namespace dusk::rtao
