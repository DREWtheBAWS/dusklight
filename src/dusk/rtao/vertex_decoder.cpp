#include "vertex_decoder.hpp"
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

} // namespace dusk::rtao
