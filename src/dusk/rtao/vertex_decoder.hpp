#pragma once
#include <aurora/geometry_capture.h>
#include <vector>

namespace dusk::rtao {

struct Vec3 {
  float x, y, z;
  bool operator==(const Vec3&) const = default;
};

struct Triangle {
  Vec3 a, b, c;
};

// Decode GX vertex data into world-space triangles.
// Each position is decoded from its compact GX format and transformed by the
// corresponding position matrix. Returns one Triangle per index triple.
std::vector<Triangle> decode_triangles(const AuroraGxCaptureDraw& draw);

} // namespace dusk::rtao
