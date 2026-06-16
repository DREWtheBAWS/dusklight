#include "vertex_decoder.hpp"
#include <cassert>
#include <cstring>

// GXCompType constants (matches GXEnum.h)
static constexpr uint8_t kGxU8  = 0;
static constexpr uint8_t kGxS8  = 1;
static constexpr uint8_t kGxU16 = 2;
static constexpr uint8_t kGxS16 = 3;
static constexpr uint8_t kGxF32 = 4;

// GXAttrType constants
static constexpr uint8_t kGxDirect  = 1;
static constexpr uint8_t kGxIndex8  = 2;
// static constexpr uint8_t kGxIndex16 = 3;  (used implicitly as the else branch)

// GXCompCnt (position)
static constexpr uint8_t kGxPosXYZ = 1;

// GXCompCnt (tex coord)
static constexpr uint8_t kGxTexST = 1;  // 0 = S only, 1 = ST

// GXCompare (alpha compare)
static constexpr uint8_t kGxAlways = 7;

namespace dusk::rtao {

static float read_f32_be(const uint8_t* p) {
  uint32_t bits = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
                  (uint32_t(p[2]) << 8)  |  uint32_t(p[3]);
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

static float read_f32_le(const uint8_t* p) {
  float f;
  memcpy(&f, p, 4);
  return f;
}

static int16_t read_s16_be(const uint8_t* p) {
  return static_cast<int16_t>((uint16_t(p[0]) << 8) | p[1]);
}
static uint16_t read_u16_be(const uint8_t* p) {
  return (uint16_t(p[0]) << 8) | p[1];
}
static int16_t read_s16_le(const uint8_t* p) {
  return static_cast<int16_t>((uint16_t(p[1]) << 8) | p[0]);
}
static uint16_t read_u16_le(const uint8_t* p) {
  return (uint16_t(p[1]) << 8) | p[0];
}

static float decode_comp(const uint8_t* p, uint8_t type, uint8_t frac, bool le) {
  const float scale = 1.0f / float(1 << frac);
  switch (type) {
  case kGxF32: return le ? read_f32_le(p) : read_f32_be(p);
  case kGxS16: return float(le ? read_s16_le(p) : read_s16_be(p)) * scale;
  case kGxU16: return float(le ? read_u16_le(p) : read_u16_be(p)) * scale;
  case kGxS8:  return float(static_cast<int8_t>(*p)) * scale;
  case kGxU8:  return float(*p) * scale;
  default:     return 0.0f;
  }
}

static uint8_t comp_byte_size(uint8_t type) {
  switch (type) {
  case kGxF32:          return 4;
  case kGxS16: case kGxU16: return 2;
  default:              return 1;
  }
}

static Vec3 decode_pos(const AuroraGxCaptureDraw& d, uint32_t vi) {
  const uint8_t* vtx = d.vertData + vi * d.vertStride + d.posOffset;

  const uint8_t* src;
  bool le;
  if (d.posAttrType == kGxDirect) {
    src = vtx;
    le = false; // FIFO data is big-endian
  } else {
    uint32_t idx = (d.posAttrType == kGxIndex8) ? *vtx : uint32_t(read_u16_be(vtx));
    src = d.posArray + idx * d.posArrayStride;
    le  = d.posArrayLittleEndian;
  }

  const uint8_t stride = comp_byte_size(d.posCompType);
  float x = decode_comp(src,              d.posCompType, d.posFrac, le);
  float y = decode_comp(src + stride,     d.posCompType, d.posFrac, le);
  float z = 0.0f;
  if (d.posCompCnt == kGxPosXYZ) {
    z = decode_comp(src + 2 * stride, d.posCompType, d.posFrac, le);
  }
  return {x, y, z};
}

static Vec3 transform(const float m[3][4], Vec3 v) {
  return {
    m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3],
    m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3],
    m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3],
  };
}

static uint32_t matrix_idx(const AuroraGxCaptureDraw& d, uint32_t vi) {
  if (!d.hasPnMtxIdx) return d.currentPnMtx;
  // PNMTXIDX is always at byte 0; raw value is 3 * matrixIndex
  return d.vertData[vi * d.vertStride] / 3u;
}

std::vector<Triangle> decode_triangles(const AuroraGxCaptureDraw& draw) {
  std::vector<Triangle> out;
  out.reserve(draw.indexCount / 3);
  for (uint32_t i = 0; i + 2 < draw.indexCount; i += 3) {
    const uint32_t ia = draw.indices[i];
    const uint32_t ib = draw.indices[i + 1];
    const uint32_t ic = draw.indices[i + 2];
    out.push_back({
      transform(draw.pnMtx[matrix_idx(draw, ia)], decode_pos(draw, ia)),
      transform(draw.pnMtx[matrix_idx(draw, ib)], decode_pos(draw, ib)),
      transform(draw.pnMtx[matrix_idx(draw, ic)], decode_pos(draw, ic)),
    });
  }
  return out;
}

static Vec2 decode_uv(const AuroraGxCaptureDraw& d, uint32_t vi) {
  if (d.tex0AttrType == 0) return {0.f, 0.f}; // GX_NONE

  const uint8_t* vtx = d.vertData + vi * d.vertStride + d.tex0Offset;

  const uint8_t* src;
  if (d.tex0AttrType == kGxDirect) {
    src = vtx;
    // DIRECT tex coords in the FIFO are big-endian
    const uint8_t stride = comp_byte_size(d.tex0CompType);
    float u = decode_comp(src,          d.tex0CompType, d.tex0Frac, false);
    float v = (d.tex0CompCnt == kGxTexST)
              ? decode_comp(src + stride, d.tex0CompType, d.tex0Frac, false)
              : 0.f;
    return {u, v};
  }
  // Indexed: read the index, then fetch from the indirect array (little-endian)
  uint32_t idx = (d.tex0AttrType == kGxIndex8)
                 ? *vtx
                 : uint32_t(read_u16_be(vtx));
  src = d.tex0Array + idx * d.tex0ArrayStride;
  const uint8_t stride = comp_byte_size(d.tex0CompType);
  float u = decode_comp(src,          d.tex0CompType, d.tex0Frac, false);
  float v = (d.tex0CompCnt == kGxTexST)
            ? decode_comp(src + stride, d.tex0CompType, d.tex0Frac, false)
            : 0.f;
  return {u, v};
}

std::vector<Triangle> decode_triangles_local(const AuroraGxCaptureDraw& draw) {
  // Skip direct-attribute draws: no stable posArray pointer for BLAS identity.
  if (draw.posAttrType == kGxDirect || !draw.posArray) return {};
  // Note: hasPnMtxIdx draws are accepted — posOffset already accounts for the
  // PNMTXIDX byte, so decode_pos() returns correct raw local positions.
  // The BLAS cache verifies all vertices share a single matrix before calling.

  std::vector<Triangle> out;
  out.reserve(draw.indexCount / 3);
  for (uint32_t i = 0; i + 2 < draw.indexCount; i += 3) {
    const uint32_t ia = draw.indices[i];
    const uint32_t ib = draw.indices[i + 1];
    const uint32_t ic = draw.indices[i + 2];
    // Raw positions only — no pnMtx transform.
    out.push_back({ decode_pos(draw, ia), decode_pos(draw, ib), decode_pos(draw, ic) });
  }

#ifndef NDEBUG
  // Layer 1: applying pnMtx to local positions must reproduce the view-space decode.
  // Only valid (and only checked) for single-matrix draws with a known uniform slot.
  if (!draw.hasPnMtxIdx && !out.empty()) {
    auto viewTris = decode_triangles(draw);
    if (!viewTris.empty()) {
      const float (*m)[4] = draw.pnMtx[draw.currentPnMtx];
      Vec3 got = transform(m, out[0].a);
      constexpr float kEps = 0.5f;  // generous: large world coords accumulate float error
      assert(std::abs(got.x - viewTris[0].a.x) <= kEps);
      assert(std::abs(got.y - viewTris[0].a.y) <= kEps);
      assert(std::abs(got.z - viewTris[0].a.z) <= kEps);
    }
  }
#endif

  return out;
}

void decode_uvs(const AuroraGxCaptureDraw& draw,
                std::vector<Triangle>& tris,
                uint32_t texSlot) {
  // Determine if this draw uses alpha-tested transparency.
  static constexpr uint8_t kAopOr = 1; // GXAlphaOp::GX_AOP_OR
  const bool comp0Always = (draw.alphaComp0 == kGxAlways);
  const bool comp1Always = (draw.alphaComp1 == kGxAlways);
  const bool trivialPass = (comp0Always && comp1Always)
                        || (draw.alphaOp == kAopOr && (comp0Always || comp1Always));
  const bool hasAlphaTest = (draw.tex0AttrType != 0) && draw.tex0HasAlpha && !trivialPass;

  const bool activeAlpha = hasAlphaTest && (texSlot != 0xFFFFFFFFu);
  for (uint32_t i = 0; i < static_cast<uint32_t>(tris.size()); ++i) {
    const uint32_t ia = draw.indices[i * 3];
    const uint32_t ib = draw.indices[i * 3 + 1];
    const uint32_t ic = draw.indices[i * 3 + 2];
    auto& t = tris[i];
    t.uva    = decode_uv(draw, ia);
    t.uvb    = decode_uv(draw, ib);
    t.uvc    = decode_uv(draw, ic);
    t.texIdx = activeAlpha ? texSlot : 0u;
    t.flags  = activeAlpha ? 1u : 0u;
  }
}

} // namespace dusk::rtao
