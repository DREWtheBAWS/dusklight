#pragma once
#include <aurora/geometry_capture.h>
#include <vector>

namespace dusk::rtao {

struct Vec3 {
  float x, y, z;
  bool operator==(const Vec3&) const = default;
};

struct Vec2 {
  float u, v;
};

struct Triangle {
  Vec3     a, b, c;
  Vec2     uva, uvb, uvc;
  uint32_t texIdx = 0;
  uint32_t flags  = 0;  // bit 0: triangle has an alpha-tested texture
};

// Decode GX vertex data into view-space triangles (positions only).
// Each position is decoded from its compact GX format and transformed by the
// corresponding position matrix. Returns one Triangle per index triple.
std::vector<Triangle> decode_triangles(const AuroraGxCaptureDraw& draw);

// Decode GX vertex data into LOCAL-SPACE (object-space) triangle positions.
// Unlike decode_triangles(), does NOT apply the pnMtx transform.
// Returns empty for direct-attribute draws (no stable posArray pointer).
// hasPnMtxIdx draws are accepted — posOffset accounts for the PNMTXIDX byte;
// the caller is responsible for verifying all vertices share a single matrix slot.
// In debug builds, asserts the round-trip for single-matrix (!hasPnMtxIdx) draws.
std::vector<Triangle> decode_triangles_local(const AuroraGxCaptureDraw& draw);

// Decode GX tex coord 0 data and write UVs into the triangles produced by
// decode_triangles (must be called with the same draw and matching tris vector).
// Sets texIdx and flags on each triangle according to the draw's texture/alpha state.
// texSlot is the caller-assigned texture registry slot for this draw (0xFFFFFFFF = none).
void decode_uvs(const AuroraGxCaptureDraw& draw,
                std::vector<Triangle>& tris,
                uint32_t texSlot);

} // namespace dusk::rtao
