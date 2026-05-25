#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "vertex_decoder.hpp"
#include <cstring>
#include <vector>

using dusk::rtao::Vec3;
using dusk::rtao::Triangle;
using dusk::rtao::decode_triangles;

// ---- byte-writing helpers ------------------------------------------------

static void write_f32_be(std::vector<uint8_t>& buf, float v) {
  uint32_t bits;
  memcpy(&bits, &v, 4);
  buf.push_back((bits >> 24) & 0xFF);
  buf.push_back((bits >> 16) & 0xFF);
  buf.push_back((bits >>  8) & 0xFF);
  buf.push_back( bits        & 0xFF);
}

static void write_s16_be(std::vector<uint8_t>& buf, int16_t v) {
  buf.push_back((uint16_t(v) >> 8) & 0xFF);
  buf.push_back( uint16_t(v)       & 0xFF);
}

static void write_u8(std::vector<uint8_t>& buf, uint8_t v) {
  buf.push_back(v);
}

// ---- identity matrix helper ----------------------------------------------

static void set_identity(float m[3][4]) {
  memset(m, 0, 3 * 4 * sizeof(float));
  m[0][0] = 1.f;
  m[1][1] = 1.f;
  m[2][2] = 1.f;
}

// ---- approximate Vec3 comparison -----------------------------------------

static bool approx_eq(Vec3 a, Vec3 b, float eps = 1e-4f) {
  return std::abs(a.x - b.x) < eps &&
         std::abs(a.y - b.y) < eps &&
         std::abs(a.z - b.z) < eps;
}

// ---- base capture struct -------------------------------------------------

static AuroraGxCaptureDraw make_base_draw() {
  AuroraGxCaptureDraw d{};
  d.posCompCnt  = 1; // GX_POS_XYZ
  d.posAttrType = 1; // GX_DIRECT
  d.posFrac     = 0;
  d.hasPnMtxIdx = false;
  d.currentPnMtx = 0;
  set_identity(d.pnMtx[0]);
  for (int i = 1; i < 10; ++i) set_identity(d.pnMtx[i]);
  return d;
}

// ==========================================================================
// TEST: F32 XYZ DIRECT, identity matrix
// ==========================================================================
TEST_CASE("F32 XYZ DIRECT passthrough with identity matrix") {
  // Build two vertices: (1, 2, 3) and (4, 5, 6) and (7, 8, 9)
  std::vector<uint8_t> verts;
  write_f32_be(verts, 1.f); write_f32_be(verts, 2.f); write_f32_be(verts, 3.f); // v0
  write_f32_be(verts, 4.f); write_f32_be(verts, 5.f); write_f32_be(verts, 6.f); // v1
  write_f32_be(verts, 7.f); write_f32_be(verts, 8.f); write_f32_be(verts, 9.f); // v2

  uint16_t idx[] = {0, 1, 2};

  auto d = make_base_draw();
  d.vertData    = verts.data();
  d.vertCount   = 3;
  d.vertStride  = 12; // 3 floats
  d.posOffset   = 0;
  d.posCompType = 4; // GX_F32
  d.indices     = idx;
  d.indexCount  = 3;

  auto tris = decode_triangles(d);
  REQUIRE(tris.size() == 1);
  CHECK(approx_eq(tris[0].a, {1.f, 2.f, 3.f}));
  CHECK(approx_eq(tris[0].b, {4.f, 5.f, 6.f}));
  CHECK(approx_eq(tris[0].c, {7.f, 8.f, 9.f}));
}

// ==========================================================================
// TEST: U8 XYZ DIRECT, frac=0
// ==========================================================================
TEST_CASE("U8 XYZ DIRECT frac=0 decodes correctly") {
  std::vector<uint8_t> verts;
  write_u8(verts, 10); write_u8(verts, 20); write_u8(verts, 30); // v0
  write_u8(verts, 50); write_u8(verts, 60); write_u8(verts, 70); // v1
  write_u8(verts, 0);  write_u8(verts, 128); write_u8(verts, 255); // v2

  uint16_t idx[] = {0, 1, 2};

  auto d = make_base_draw();
  d.vertData    = verts.data();
  d.vertCount   = 3;
  d.vertStride  = 3;
  d.posOffset   = 0;
  d.posCompType = 0; // GX_U8
  d.posFrac     = 0;
  d.indices     = idx;
  d.indexCount  = 3;

  auto tris = decode_triangles(d);
  REQUIRE(tris.size() == 1);
  CHECK(approx_eq(tris[0].a, {10.f, 20.f, 30.f}));
  CHECK(approx_eq(tris[0].b, {50.f, 60.f, 70.f}));
  CHECK(approx_eq(tris[0].c, {0.f,  128.f, 255.f}));
}

// ==========================================================================
// TEST: S16 XYZ DIRECT, frac=8 (fixed-point scale 1/256)
// ==========================================================================
TEST_CASE("S16 XYZ DIRECT frac=8 applies fixed-point scale") {
  // raw value 256 with frac=8 → 256 / 256 = 1.0
  // raw value -512             → -512 / 256 = -2.0
  std::vector<uint8_t> verts;
  write_s16_be(verts,  256); write_s16_be(verts,  512); write_s16_be(verts, -256); // v0
  write_s16_be(verts, -512); write_s16_be(verts,    0); write_s16_be(verts,  768); // v1
  write_s16_be(verts, 1024); write_s16_be(verts,  256); write_s16_be(verts,  256); // v2

  uint16_t idx[] = {0, 1, 2};

  auto d = make_base_draw();
  d.vertData    = verts.data();
  d.vertCount   = 3;
  d.vertStride  = 6; // 3 × 2 bytes
  d.posOffset   = 0;
  d.posCompType = 3; // GX_S16
  d.posFrac     = 8;
  d.indices     = idx;
  d.indexCount  = 3;

  auto tris = decode_triangles(d);
  REQUIRE(tris.size() == 1);
  CHECK(approx_eq(tris[0].a, { 1.f,  2.f, -1.f}));
  CHECK(approx_eq(tris[0].b, {-2.f,  0.f,  3.f}));
  CHECK(approx_eq(tris[0].c, { 4.f,  1.f,  1.f}));
}

// ==========================================================================
// TEST: F32 XYZ DIRECT, translation matrix
// ==========================================================================
TEST_CASE("F32 XYZ with translation matrix applies correctly") {
  std::vector<uint8_t> verts;
  write_f32_be(verts, 1.f); write_f32_be(verts, 0.f); write_f32_be(verts, 0.f);
  write_f32_be(verts, 0.f); write_f32_be(verts, 1.f); write_f32_be(verts, 0.f);
  write_f32_be(verts, 0.f); write_f32_be(verts, 0.f); write_f32_be(verts, 1.f);

  uint16_t idx[] = {0, 1, 2};

  // Matrix: identity with translation (10, 20, 30)
  auto d = make_base_draw();
  d.pnMtx[0][0][3] = 10.f;
  d.pnMtx[0][1][3] = 20.f;
  d.pnMtx[0][2][3] = 30.f;

  d.vertData    = verts.data();
  d.vertCount   = 3;
  d.vertStride  = 12;
  d.posOffset   = 0;
  d.posCompType = 4; // GX_F32
  d.indices     = idx;
  d.indexCount  = 3;

  auto tris = decode_triangles(d);
  REQUIRE(tris.size() == 1);
  CHECK(approx_eq(tris[0].a, {11.f, 20.f, 30.f}));
  CHECK(approx_eq(tris[0].b, {10.f, 21.f, 30.f}));
  CHECK(approx_eq(tris[0].c, {10.f, 20.f, 31.f}));
}

// ==========================================================================
// TEST: INDEX8 positions via separate array
// ==========================================================================
TEST_CASE("INDEX8 position looks up from posArray") {
  // posArray has 3 entries of big-endian F32 XYZ
  std::vector<uint8_t> arr;
  write_f32_be(arr, 100.f); write_f32_be(arr, 200.f); write_f32_be(arr, 300.f); // idx 0
  write_f32_be(arr, 400.f); write_f32_be(arr, 500.f); write_f32_be(arr, 600.f); // idx 1
  write_f32_be(arr, 700.f); write_f32_be(arr, 800.f); write_f32_be(arr, 900.f); // idx 2

  // Vertex stream: each vertex is a 1-byte index
  std::vector<uint8_t> verts = {2, 0, 1}; // references array entries 2, 0, 1

  uint16_t idx[] = {0, 1, 2};

  auto d = make_base_draw();
  d.vertData             = verts.data();
  d.vertCount            = 3;
  d.vertStride           = 1;
  d.posOffset            = 0;
  d.posCompType          = 4; // GX_F32
  d.posAttrType          = 2; // GX_INDEX8
  d.posArray             = arr.data();
  d.posArrayStride       = 12; // 3 × F32
  d.posArrayLittleEndian = false;
  d.indices              = idx;
  d.indexCount           = 3;

  auto tris = decode_triangles(d);
  REQUIRE(tris.size() == 1);
  CHECK(approx_eq(tris[0].a, {700.f, 800.f, 900.f})); // array[2]
  CHECK(approx_eq(tris[0].b, {100.f, 200.f, 300.f})); // array[0]
  CHECK(approx_eq(tris[0].c, {400.f, 500.f, 600.f})); // array[1]
}

// ==========================================================================
// TEST: per-vertex PNMTXIDX selects different matrices
// ==========================================================================
TEST_CASE("Per-vertex PNMTXIDX selects different transform matrices") {
  // Vertex layout: [pnmtxidx(1 byte)] [pos F32 XYZ (12 bytes)] = 13 bytes/vtx
  // Vertex 0: matIdx = 0*3=0 → matrix 0 (translate +1,0,0)
  // Vertex 1: matIdx = 1*3=3 → matrix 1 (translate 0,+2,0)
  // Vertex 2: matIdx = 2*3=6 → matrix 2 (translate 0,0,+3)
  std::vector<uint8_t> verts;
  verts.push_back(0);  // pnmtxidx raw = 0 → matrix 0
  write_f32_be(verts, 0.f); write_f32_be(verts, 0.f); write_f32_be(verts, 0.f);

  verts.push_back(3);  // pnmtxidx raw = 3 → matrix 1
  write_f32_be(verts, 0.f); write_f32_be(verts, 0.f); write_f32_be(verts, 0.f);

  verts.push_back(6);  // pnmtxidx raw = 6 → matrix 2
  write_f32_be(verts, 0.f); write_f32_be(verts, 0.f); write_f32_be(verts, 0.f);

  uint16_t idx[] = {0, 1, 2};

  auto d = make_base_draw();
  d.hasPnMtxIdx = true;
  d.vertData    = verts.data();
  d.vertCount   = 3;
  d.vertStride  = 13;
  d.posOffset   = 1; // after the 1-byte pnmtxidx
  d.posCompType = 4; // GX_F32
  d.indices     = idx;
  d.indexCount  = 3;

  // Matrix 0: translate +1 on X
  d.pnMtx[0][0][3] = 1.f;
  // Matrix 1: translate +2 on Y
  set_identity(d.pnMtx[1]);
  d.pnMtx[1][1][3] = 2.f;
  // Matrix 2: translate +3 on Z
  set_identity(d.pnMtx[2]);
  d.pnMtx[2][2][3] = 3.f;

  auto tris = decode_triangles(d);
  REQUIRE(tris.size() == 1);
  CHECK(approx_eq(tris[0].a, {1.f, 0.f, 0.f}));
  CHECK(approx_eq(tris[0].b, {0.f, 2.f, 0.f}));
  CHECK(approx_eq(tris[0].c, {0.f, 0.f, 3.f}));
}

// ==========================================================================
// TEST: empty draw produces no triangles
// ==========================================================================
TEST_CASE("Empty draw produces no triangles") {
  auto d = make_base_draw();
  d.vertData   = nullptr;
  d.vertCount  = 0;
  d.indexCount = 0;
  d.indices    = nullptr;

  auto tris = decode_triangles(d);
  CHECK(tris.empty());
}

// ==========================================================================
// TEST: multiple triangles from a quad (4 verts → 2 triangles, 6 indices)
// ==========================================================================
TEST_CASE("Quad (6 indices) produces 2 triangles") {
  std::vector<uint8_t> verts;
  write_f32_be(verts, 0.f); write_f32_be(verts, 0.f); write_f32_be(verts, 0.f); // v0
  write_f32_be(verts, 1.f); write_f32_be(verts, 0.f); write_f32_be(verts, 0.f); // v1
  write_f32_be(verts, 1.f); write_f32_be(verts, 1.f); write_f32_be(verts, 0.f); // v2
  write_f32_be(verts, 0.f); write_f32_be(verts, 1.f); write_f32_be(verts, 0.f); // v3

  // Two triangles: (0,1,2) and (2,3,0)
  uint16_t idx[] = {0, 1, 2, 2, 3, 0};

  auto d = make_base_draw();
  d.vertData    = verts.data();
  d.vertCount   = 4;
  d.vertStride  = 12;
  d.posOffset   = 0;
  d.posCompType = 4; // GX_F32
  d.indices     = idx;
  d.indexCount  = 6;

  auto tris = decode_triangles(d);
  REQUIRE(tris.size() == 2);
  CHECK(approx_eq(tris[0].a, {0.f, 0.f, 0.f}));
  CHECK(approx_eq(tris[0].b, {1.f, 0.f, 0.f}));
  CHECK(approx_eq(tris[0].c, {1.f, 1.f, 0.f}));
  CHECK(approx_eq(tris[1].a, {1.f, 1.f, 0.f}));
  CHECK(approx_eq(tris[1].b, {0.f, 1.f, 0.f}));
  CHECK(approx_eq(tris[1].c, {0.f, 0.f, 0.f}));
}
