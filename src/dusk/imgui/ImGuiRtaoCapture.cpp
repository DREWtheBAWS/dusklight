#include "ImGuiMenuTools.hpp"
#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"
#include <SDL3/SDL_filesystem.h>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>

namespace dusk {

void ImGuiMenuTools::ShowRtaoCaptureWindow() {
    if (!m_showRtaoCapture) return;

    if (!ImGui::Begin("RTAO Capture", &m_showRtaoCapture)) {
        ImGui::End();
        return;
    }

    const auto stats = m_collector.last_stats();
    ImGui::Text("Triangles (last frame): %u", stats.triangleCount);
    ImGui::Text("Draw calls (last frame): %u", stats.drawCallCount);
    ImGui::Text("BLAS callback fired (ever): %u", m_collector.draw_callback_fired_total());
    ImGui::Text("Alpha textures: %u / %u total", static_cast<uint32_t>(m_collector.texture_views().size()),
                m_collector.total_alpha_tex_count());

    const auto cam = m_collector.last_camera_data();
    if (cam.valid) {
        ImGui::Text("Camera: (%.1f, %.1f, %.1f)  FoV: %.1f deg",
                    cam.worldPos[0], cam.worldPos[1], cam.worldPos[2], cam.fovYDeg);
    } else {
        ImGui::TextDisabled("Camera: (no data yet)");
    }

    ImGui::Separator();

    // Filter controls
    static bool s_perspOnly = true;
    static float s_minW = 320.f, s_minH = 240.f;
    bool filterChanged = false;
    filterChanged |= ImGui::Checkbox("Perspective only", &s_perspOnly);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.f);
    filterChanged |= ImGui::InputFloat("Min W", &s_minW, 0.f, 0.f, "%.0f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.f);
    filterChanged |= ImGui::InputFloat("Min H", &s_minH, 0.f, 0.f, "%.0f");
    if (filterChanged) {
        m_collector.set_filter(s_perspOnly, s_minW, s_minH);
    }

    ImGui::Separator();

    // OBJ dump
    static std::string s_dumpPath;
    if (s_dumpPath.empty()) {
        const char* base = SDL_GetBasePath();
        s_dumpPath = std::string(base ? base : "") + "rtao_capture.obj";
    }
    ImGui::InputText("Path", &s_dumpPath);
    if (ImGui::Button("Dump OBJ")) {
        m_collector.request_dump(s_dumpPath);
    }
    const auto& msg = m_collector.last_dump_message();
    if (!msg.empty()) {
        ImGui::TextWrapped("%s", msg.c_str());
    }

    ImGui::Separator();

    // BVH path selection
    ImGui::Checkbox("BLAS/TLAS (Phase 3)", &m_useTlasBvh);
    ImGui::SameLine();
    ImGui::TextDisabled(m_useTlasBvh ? "| two-level, opaque-only" : "| single-level LBVH + alpha");

    ImGui::Separator();

    // GPU BVH controls
    ImGui::Checkbox("Build only (skip AO)", &m_buildBvhOnly);
    ImGui::SameLine();
    ImGui::Checkbox("Freeze BVH", &m_bvhFrozen);
    ImGui::SameLine();
    if (ImGui::Button("Capture & Freeze")) {
        // do exactly one BVH rebuild on the next frame, then lock it
        m_bvhFrozen      = false;
        m_bvhCaptureOnce = true;
    }
    if (m_bvhFrozen) { ImGui::SameLine(); ImGui::TextDisabled("(frozen)"); }
    const auto bvhStats = m_bvhBuilder.last_stats();
    if (bvhStats.nodeCount > 0) {
        ImGui::Text("%u nodes, %u tris, %.2f ms", bvhStats.nodeCount, bvhStats.triCount, bvhStats.buildMs);
        {
            const float* mn = bvhStats.sceneMin;
            const float* mx = bvhStats.sceneMax;
            const float eX = mx[0]-mn[0], eY = mx[1]-mn[1], eZ = mx[2]-mn[2];
            ImGui::Text("Full AABB:   %.0f x %.0f x %.0f", eX, eY, eZ);
        }
        {
            const float* mn = bvhStats.mortonMin;
            const float* mx = bvhStats.mortonMax;
            const float eX = mx[0]-mn[0], eY = mx[1]-mn[1], eZ = mx[2]-mn[2];
            ImGui::Text("Morton AABB: %.0f x %.0f x %.0f  (mean+/-3s)", eX, eY, eZ);
            if (eX > 7000.f || eY > 7000.f || eZ > 7000.f)
                ImGui::TextColored({1.f,0.4f,0.f,1.f}, "  ^ Morton range still large");
        }
    } else {
        ImGui::TextDisabled("(not built yet)");
    }
    // ---- Layer 2: BLAS cache stats ----------------------------------------
    ImGui::Separator();
    ImGui::TextDisabled("BLAS Cache");
    const auto blasStats = m_blasCache.last_stats();
    ImGui::Text("Cached: %u/%u  |  Pending: %u  |  Seen: %u  |  New: %u  |  Evicted: %u",
                blasStats.totalCached, dusk::rtao::BlasCache::kMaxEntries,
                blasStats.pendingCount, blasStats.seenThisFrame,
                blasStats.newThisFrame, blasStats.evictedLastFrame);
    if (blasStats.gpuBytesTotal > 0) {
        const float mb = float(blasStats.gpuBytesTotal) / (1024.f * 1024.f);
        ImGui::Text("GPU: %.2f MB", mb);
    }
    // Rejection diagnostics — vital for debugging the 0-instance problem.
    ImGui::Text("Calls/frame: %u  |  Rejected(direct): %u  |  Rejected(skinned): %u",
                blasStats.totalCallsThisFrame,
                blasStats.rejectedDirect,
                blasStats.rejectedSkinned);
    if (blasStats.totalCallsThisFrame > 0) {
        const uint32_t accepted = blasStats.totalCallsThisFrame
                                - blasStats.rejectedDirect
                                - blasStats.rejectedSkinned;
        const float pct = 100.f * float(accepted) / float(blasStats.totalCallsThisFrame);
        ImGui::Text("  => accepted: %u (%.1f%%) | instances: %u",
                    accepted, pct, blasStats.instanceCount);
    }
    ImGui::Text("record_draw() total ever: %u", m_blasCache.total_record_draw_calls_ever());
    ImGui::Text("flush: %.2f ms", blasStats.flushMs);
    ImGui::Text("Dynamic (skinned): %u tris -> %u nodes",
                blasStats.dynTriCount, blasStats.dynNodeCount);

    // ---- Layer 2b: TLAS stats -------------------------------------------
    ImGui::Separator();
    ImGui::TextDisabled("TLAS");
    const auto tlasStats = m_tlasBuilder.last_stats();
    if (m_tlasBuilder.is_ready()) {
        ImGui::Text("Nodes: %u  |  Instances: %u  |  Dedup removed: %u  |  BLASes: %u",
                    tlasStats.tlasNodeCount, tlasStats.instanceCount,
                    tlasStats.dedupRejected, tlasStats.blasEntryCount);
        ImGui::Text("Mono BLAS: %u nodes, %u tris",
                    tlasStats.blasNodeTotal, tlasStats.blasTriTotal);
        ImGui::Text("build: %.2f ms  |  flush: %.2f ms",
                    tlasStats.buildMs, tlasStats.flushMs);
        const float rootEx = tlasStats.rootAabbMax[0] - tlasStats.rootAabbMin[0];
        const float rootEy = tlasStats.rootAabbMax[1] - tlasStats.rootAabbMin[1];
        const float rootEz = tlasStats.rootAabbMax[2] - tlasStats.rootAabbMin[2];
        ImGui::Text("Root AABB: %.0f x %.0f x %.0f (view space)", rootEx, rootEy, rootEz);
        const float mb = float(tlasStats.gpuBytesTotal) / (1024.f * 1024.f);
        ImGui::Text("GPU total: %.2f MB", mb);
    } else {
        ImGui::TextDisabled("(not ready)");
    }

    // ---- Validation / stage-by-stage diagnostics ---------------------------
    ImGui::Separator();
    ImGui::TextDisabled("Pipeline Validation");

    // Stage 1 — GeometryCollector
    {
        const bool ok1 = stats.triangleCount > 0;
        ImGui::TextColored(ok1 ? ImVec4(0.3f,1.f,0.3f,1.f) : ImVec4(1.f,0.3f,0.3f,1.f),
            "Stage 1 GeoCollect: %s", ok1 ? "OK" : "FAIL");
        ImGui::SameLine(); ImGui::TextDisabled("(%u tris, %u draws)", stats.triangleCount, stats.drawCallCount);
        if (!ok1) ImGui::TextWrapped("  Expected: >0 tris. Check install() called and game is running.");
    }
    // Stage 2 — BlasCache record_draw
    {
        const bool ok2 = blasStats.instanceCount > 0;
        ImGui::TextColored(ok2 ? ImVec4(0.3f,1.f,0.3f,1.f) : ImVec4(1.f,0.3f,0.3f,1.f),
            "Stage 2 BLAS record: %s", ok2 ? "OK" : "FAIL");
        ImGui::SameLine(); ImGui::TextDisabled("(%u instances, %u cached)", blasStats.instanceCount, blasStats.totalCached);
        if (!ok2) ImGui::TextWrapped("  Expected: >0 instances. Check set_draw_callback() wired up.");
    }
    // Stage 3 — BlasCache flush (BVH builds)
    {
        const bool ok3 = blasStats.totalCached > 0 && blasStats.gpuBytesTotal > 0;
        ImGui::TextColored(ok3 ? ImVec4(0.3f,1.f,0.3f,1.f) : ImVec4(1.f,0.3f,0.3f,1.f),
            "Stage 3 BLAS flush: %s", ok3 ? "OK" : "FAIL");
        const float bkb = float(blasStats.gpuBytesTotal) / 1024.f;
        ImGui::SameLine(); ImGui::TextDisabled("(%u entries, %.0f KB GPU)", blasStats.totalCached, bkb);
        if (!ok3) ImGui::TextWrapped("  Expected: >0 cached entries after warmup. Check flush() called.");
    }
    // Stage 4 — TlasBuilder (TLAS structure)
    {
        const bool ok4 = tlasStats.instanceCount > 0 && tlasStats.tlasNodeCount > 0;
        ImGui::TextColored(ok4 ? ImVec4(0.3f,1.f,0.3f,1.f) : ImVec4(1.f,0.3f,0.3f,1.f),
            "Stage 4 TLAS build: %s", ok4 ? "OK" : "FAIL");
        ImGui::SameLine(); ImGui::TextDisabled("(%u nodes, %u inst)", tlasStats.tlasNodeCount, tlasStats.instanceCount);
        if (!ok4) ImGui::TextWrapped("  Expected: >0 TLAS nodes. Check TlasBuilder::build() called after flush.");
        // Root AABB sanity: in view space, z of near objects should be negative (camera looks -z).
        if (ok4) {
            const float rz_min = tlasStats.rootAabbMin[2];
            const float rz_max = tlasStats.rootAabbMax[2];
            if (rz_min > 0.f)
                ImGui::TextColored({1.f,0.7f,0.2f,1.f}, "  WARNING: root AABB z range [%.0f,%.0f] — expected z<0 for objects in front of camera", rz_min, rz_max);
        }
    }
    // Stage 5 — AO shader hit rate (inferred from stats)
    {
        const bool ok5 = m_tlasBuilder.is_ready();
        ImGui::TextColored(ok5 ? ImVec4(0.3f,1.f,0.3f,1.f) : ImVec4(1.f,0.3f,0.3f,1.f),
            "Stage 5 AO shader: %s", ok5 ? "ready" : "not ready");
        if (!ok5) ImGui::TextWrapped("  Expected: TLAS buffers uploaded. Check execute_tlas() called.");
        ImGui::TextDisabled("  Tip: Use debug_mode2=Visit Heat or Limit%% to check traversal cost.");
    }

    // Structural validation (on-demand)
    if (ImGui::Button("Run Structural Validation")) {
        m_tlasBuilder.request_validation();
    }
    const auto& val = m_tlasBuilder.last_validation();
    if (val.ran) {
        ImGui::Text("TLAS leaves: %s (bad=%u/%u)", val.tlasLeafOk ? "OK" : "FAIL",
                    val.badLeaves, val.tlasNodeCount);
        ImGui::Text("BLAS ranges: %s (bad=%u/%u inst)", val.blasRangeOk ? "OK" : "FAIL",
                    val.badInstances, val.instanceCount);
        ImGui::Text("  [%u TLAS nodes, %u inst, %u blas-nodes, %u blas-tris]",
                    val.tlasNodeCount, val.instanceCount, val.blasNodeBufSize, val.blasTriBufSize);
    } else {
        ImGui::TextDisabled("  (not run yet — press button above)");
    }

    // Per-entry table (up to 20 rows shown).
    static bool s_showBlasTable = false;
    ImGui::Checkbox("Show BLAS entries", &s_showBlasTable);
    if (s_showBlasTable) {
        if (ImGui::BeginTable("blas_entries", 5,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                ImVec2(0, 180.f))) {
            ImGui::TableSetupColumn("Key");
            ImGui::TableSetupColumn("Tris");
            ImGui::TableSetupColumn("Nodes");
            ImGui::TableSetupColumn("AABB extent");
            ImGui::TableSetupColumn("Stable");
            ImGui::TableHeadersRow();

            uint32_t row = 0;
            for (const auto& [key, e] : m_blasCache.entries()) {
                if (row++ >= 64) { ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0);
                    ImGui::TextDisabled("..."); break; }
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%04X:%08X",
                    static_cast<uint32_t>(key.posArray & 0xFFFF), key.meshHash);
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%u", e.triCount);
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%u", e.nodeCount);
                ImGui::TableSetColumnIndex(3);
                if (e.localAabb.valid()) {
                    float ex = e.localAabb.max.x - e.localAabb.min.x;
                    float ey = e.localAabb.max.y - e.localAabb.min.y;
                    float ez = e.localAabb.max.z - e.localAabb.min.z;
                    ImGui::Text("%.1f x %.1f x %.1f", ex, ey, ez);
                } else { ImGui::TextDisabled("--"); }
                ImGui::TableSetColumnIndex(4);
                // "Stable" = seen in a frame before the current one.
                const bool isNew = (m_blasCache.current_frame() <= e.lastSeenFrame + 1);
                ImGui::TextColored(isNew ? ImVec4(1.f,0.7f,0.2f,1.f) : ImVec4(0.3f,1.f,0.3f,1.f),
                                   isNew ? "new" : "yes");
            }
            ImGui::EndTable();
        }
    }

    ImGui::Separator();

    // AO params
    static int   s_raysPerPixel = 1;
    static float s_maxDist      = 500.f;
    static float s_normalBias   = 0.01f;
    static int   s_debugMode    = 0;
    static int   s_debugMode2   = 0;
    ImGui::SetNextItemWidth(120.f);
    ImGui::SliderInt("Rays/pixel", &s_raysPerPixel, 1, 64);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.f);
    ImGui::InputFloat("Max dist", &s_maxDist, 0.f, 0.f, "%.0f");
    ImGui::SameLine();
    ImGui::TextDisabled("Morton: +-%.0f", s_maxDist * 4.f);
    ImGui::SetNextItemWidth(120.f);
    ImGui::InputFloat("Normal bias", &s_normalBias, 0.001f, 0.01f, "%.4f");
    ImGui::SetNextItemWidth(180.f);
    ImGui::Combo("Left panel",  &s_debugMode,  "AO\0Normals\0Depth dist\0Root AABB\0");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(180.f);
    ImGui::Combo("Right panel", &s_debugMode2, "Limit Hits\0AO\0Normals\0Depth dist\0Root AABB\0Visit Heat\0Limit %\0");
    m_aoPass.set_params({static_cast<uint32_t>(s_raysPerPixel), s_maxDist, s_normalBias,
                         static_cast<uint32_t>(s_debugMode),
                         static_cast<uint32_t>(s_debugMode2)});
    // Geometry sphere radius: 4× the AO ray length ensures surfaces at the far end of
    // the frustum still have nearby geometry collected for occlusion.
    // Frustum margin: exactly the AO ray length — geometry more than one ray-length outside
    // the view frustum can never occlude a visible surface, so discard it.
    m_bvhBuilder.set_morton_range(s_maxDist * 4.f);
    m_collector.set_max_distance(s_maxDist * 4.f);
    m_collector.set_frustum_margin(s_maxDist);
    // Skip triangles with any edge longer than 3× the AO ray length: these are
    // coarse terrain patches that inflate every ancestor AABB in the BVH tree
    // without contributing meaningful occlusion detail.
    m_collector.set_max_edge_length(s_maxDist * 3.f);

    static const char* kPanelNames1[] = {"AO", "Normals", "Depth dist", "Root AABB"};
    static const char* kPanelNames2[] = {"Limit Hits", "AO", "Normals", "Depth dist", "Root AABB", "Visit Heat", "Limit %"};

    if (!m_aoPass.is_ready()) {
        ImGui::Separator();
        ImGui::TextDisabled("(start the game to see AO)");
    } else {
        const float avail  = ImGui::GetContentRegionAvail().x;
        const float half   = (avail - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        const float aspect = static_cast<float>(m_aoPass.width()) /
                             static_cast<float>(m_aoPass.height());
        const float h = half / aspect;

        ImGui::Separator();
        ImGui::Text("%s", kPanelNames1[s_debugMode]);
        ImGui::SameLine(half + ImGui::GetStyle().ItemSpacing.x);
        ImGui::Text("%s", kPanelNames2[s_debugMode2]);

        const ImVec2 imgOrigin = ImGui::GetCursorScreenPos();
        ImGui::Image(m_aoPass.imgui_texture_id(),   ImVec2(half, h));
        ImGui::SameLine();
        ImGui::Image(m_aoPass.limits_texture_id(),  ImVec2(half, h));

        // ---- Layer 3: AABB wireframe overlays --------------------------------
        // Two separate overlays so you can compare what the GAME renders vs what AO uses.
        static bool s_showAabbOverlay     = false;
        static bool s_showTlasAabbOverlay = true;
        ImGui::Checkbox("Game geometry (all, unfiltered)", &s_showAabbOverlay);
        ImGui::SameLine();
        ImGui::Checkbox("AO geometry (frozen TLAS)", &s_showTlasAabbOverlay);

        const auto& overlayCamera = m_collector.last_camera_data();
        if (overlayCamera.valid && (s_showAabbOverlay || s_showTlasAabbOverlay)) {
            ImDrawList* dl = ImGui::GetWindowDrawList();

            // Project a view-space point onto the first AO image.
            auto projectVS = [&](const rtao::Vec3& vs, ImVec2& out) -> bool {
                const float* p = reinterpret_cast<const float*>(overlayCamera.proj);
                float cx = p[0]*vs.x + p[1]*vs.y + p[2]*vs.z  + p[3];
                float cy = p[4]*vs.x + p[5]*vs.y + p[6]*vs.z  + p[7];
                float cw = p[12]*vs.x + p[13]*vs.y + p[14]*vs.z + p[15];
                if (cw <= 1e-4f) return false;
                float sx = imgOrigin.x + (cx / cw + 1.f) * 0.5f * half;
                float sy = imgOrigin.y + (1.f - cy / cw) * 0.5f * h;
                out = {sx, sy};
                return true;
            };

            // Transform local→view using a 3×4 pnMtx.
            auto xfVS = [](const float m[3][4], const rtao::Vec3& v) -> rtao::Vec3 {
                return { m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3],
                         m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3],
                         m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3] };
            };

            static constexpr std::array<std::pair<int,int>, 12> kEdges = {{
                {0,1},{2,3},{4,5},{6,7},
                {0,2},{1,3},{4,6},{5,7},
                {0,4},{1,5},{2,6},{3,7}
            }};

            // --- Green: game geometry (current-frame BlasCache instances, no radius filter) ---
            if (s_showAabbOverlay) {
                const auto& entries = m_blasCache.entries();
                std::unordered_set<size_t> overlaySeenKeys;
                overlaySeenKeys.reserve(m_blasCache.instances().size());
                auto overlayHash = [](const rtao::BlasCache::Instance& i) -> size_t {
                    size_t h = rtao::BlasKeyHash{}(i.blasKey);
                    const uint8_t* b = reinterpret_cast<const uint8_t*>(i.pnMtx);
                    for (int k = 0; k < 48; k += 4) {
                        uint32_t w; memcpy(&w, b + k, 4);
                        h ^= size_t(w) * 2654435761u + (h << 6) + (h >> 2);
                    }
                    return h;
                };
                uint32_t drawn = 0;
                for (const auto& inst : m_blasCache.instances()) {
                    if (drawn >= 200) break;
                    if (!overlaySeenKeys.insert(overlayHash(inst)).second) continue;
                    auto it = entries.find(inst.blasKey);
                    if (it == entries.end() || !it->second.localAabb.valid()) continue;
                    const rtao::AABB& lb = it->second.localAabb;
                    rtao::Vec3 corners[8];
                    for (int c = 0; c < 8; ++c)
                        corners[c] = { (c&1)?lb.max.x:lb.min.x, (c&2)?lb.max.y:lb.min.y, (c&4)?lb.max.z:lb.min.z };
                    rtao::Vec3 vsCorners[8];
                    for (int c = 0; c < 8; ++c)
                        vsCorners[c] = xfVS(inst.pnMtx, corners[c]);
                    for (auto [a, b] : kEdges) {
                        ImVec2 pa, pb;
                        if (!projectVS(vsCorners[a], pa)) continue;
                        if (!projectVS(vsCorners[b], pb)) continue;
                        dl->AddLine(pa, pb, IM_COL32(80, 220, 80, 180), 1.f);
                    }
                    ++drawn;
                }
                ImGui::Text("Game AABBs (green): %u drawn / %u raw", drawn,
                            static_cast<uint32_t>(m_blasCache.instances().size()));
            }

            // --- Yellow: AO geometry (TLAS instances, view-space AABBs) ---
            // These are EXACTLY what the AO shader traverses this frame.
            if (s_showTlasAabbOverlay) {
                std::vector<rtao::AABB> tlasAabbs;
                m_tlasBuilder.get_instance_view_aabbs(tlasAabbs);
                uint32_t drawn = 0;
                for (const auto& aabb : tlasAabbs) {
                    if (drawn >= 500) break;
                    if (!aabb.valid()) continue;
                    rtao::Vec3 corners[8];
                    for (int c = 0; c < 8; ++c)
                        corners[c] = { (c&1)?aabb.max.x:aabb.min.x, (c&2)?aabb.max.y:aabb.min.y, (c&4)?aabb.max.z:aabb.min.z };
                    for (auto [a, b] : kEdges) {
                        ImVec2 pa, pb;
                        if (!projectVS(corners[a], pa)) continue;
                        if (!projectVS(corners[b], pb)) continue;
                        dl->AddLine(pa, pb, IM_COL32(255, 200, 0, 180), 1.5f);
                    }
                    ++drawn;
                }
                ImGui::Text("AO AABBs (yellow): %u drawn / %u total", drawn,
                            static_cast<uint32_t>(tlasAabbs.size()));
            }
        }
    }

    ImGui::End();
}

} // namespace dusk
