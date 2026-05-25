#include <catch2/catch_test_macros.hpp>
#include "geometry_collector.hpp"
#include <cstring>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

using dusk::rtao::GeometryCollector;

// ---- byte helpers (same as vertex_decoder_test) --------------------------

static void write_f32_be(std::vector<uint8_t>& buf, float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    buf.push_back((bits >> 24) & 0xFF);
    buf.push_back((bits >> 16) & 0xFF);
    buf.push_back((bits >>  8) & 0xFF);
    buf.push_back( bits        & 0xFF);
}

static AuroraGxCaptureDraw make_triangle_draw(std::vector<uint8_t>& verts,
                                              std::vector<uint16_t>& indices,
                                              float ax, float ay, float az,
                                              float bx, float by, float bz,
                                              float cx, float cy, float cz) {
    verts.clear();
    write_f32_be(verts, ax); write_f32_be(verts, ay); write_f32_be(verts, az);
    write_f32_be(verts, bx); write_f32_be(verts, by); write_f32_be(verts, bz);
    write_f32_be(verts, cx); write_f32_be(verts, cy); write_f32_be(verts, cz);

    indices = {0, 1, 2};

    AuroraGxCaptureDraw d{};
    d.vertData    = verts.data();
    d.vertCount   = 3;
    d.vertStride  = 12;
    d.posOffset   = 0;
    d.posCompCnt  = 1;  // GX_POS_XYZ
    d.posCompType = 4;  // GX_F32
    d.posAttrType = 1;  // GX_DIRECT
    d.hasPnMtxIdx = false;
    d.currentPnMtx = 0;
    // identity matrix
    d.pnMtx[0][0][0] = 1.f; d.pnMtx[0][1][1] = 1.f; d.pnMtx[0][2][2] = 1.f;
    d.indices     = indices.data();
    d.indexCount  = 3;
    return d;
}

// ==========================================================================
// TEST: single draw populates stats
// ==========================================================================
TEST_CASE("GeometryCollector: single draw produces correct stats") {
    GeometryCollector gc;

    std::vector<uint8_t> verts;
    std::vector<uint16_t> idx;
    auto draw = make_triangle_draw(verts, idx, 1,0,0, 0,1,0, 0,0,1);
    gc.simulate_draw(draw);
    gc.end_frame();

    auto stats = gc.last_stats();
    CHECK(stats.triangleCount == 1);
    CHECK(stats.drawCallCount == 1);
}

// ==========================================================================
// TEST: multiple draws accumulate
// ==========================================================================
TEST_CASE("GeometryCollector: multiple draws accumulate across one frame") {
    GeometryCollector gc;

    std::vector<uint8_t> v1, v2;
    std::vector<uint16_t> i1, i2;
    gc.simulate_draw(make_triangle_draw(v1, i1, 0,0,0, 1,0,0, 0,1,0));
    gc.simulate_draw(make_triangle_draw(v2, i2, 2,0,0, 3,0,0, 2,1,0));
    gc.end_frame();

    auto stats = gc.last_stats();
    CHECK(stats.triangleCount == 2);
    CHECK(stats.drawCallCount == 2);
}

// ==========================================================================
// TEST: end_frame resets for the next frame
// ==========================================================================
TEST_CASE("GeometryCollector: end_frame resets counters for the next frame") {
    GeometryCollector gc;

    std::vector<uint8_t> v; std::vector<uint16_t> idx;
    gc.simulate_draw(make_triangle_draw(v, idx, 0,0,0, 1,0,0, 0,1,0));
    gc.simulate_draw(make_triangle_draw(v, idx, 0,0,0, 1,0,0, 0,1,0));
    gc.end_frame();
    REQUIRE(gc.last_stats().triangleCount == 2);

    // Frame 2: only 1 triangle
    gc.simulate_draw(make_triangle_draw(v, idx, 5,0,0, 6,0,0, 5,1,0));
    gc.end_frame();

    CHECK(gc.last_stats().triangleCount == 1);
    CHECK(gc.last_stats().drawCallCount == 1);
}

// ==========================================================================
// TEST: empty frame produces zero stats
// ==========================================================================
TEST_CASE("GeometryCollector: empty frame produces zero stats") {
    GeometryCollector gc;
    gc.end_frame();
    CHECK(gc.last_stats().triangleCount == 0);
    CHECK(gc.last_stats().drawCallCount == 0);
}

// ==========================================================================
// TEST: OBJ dump writes valid content
// ==========================================================================
TEST_CASE("GeometryCollector: OBJ dump produces valid vertex and face lines") {
    GeometryCollector gc;

    std::vector<uint8_t> v; std::vector<uint16_t> idx;
    // Two triangles
    auto draw2 = [&]() {
        v.clear();
        write_f32_be(v, 0.f); write_f32_be(v, 0.f); write_f32_be(v, 0.f);
        write_f32_be(v, 1.f); write_f32_be(v, 0.f); write_f32_be(v, 0.f);
        write_f32_be(v, 1.f); write_f32_be(v, 1.f); write_f32_be(v, 0.f);
        write_f32_be(v, 0.f); write_f32_be(v, 1.f); write_f32_be(v, 0.f);
        idx = {0,1,2, 2,3,0};
        AuroraGxCaptureDraw d{};
        d.vertData = v.data(); d.vertCount = 4; d.vertStride = 12;
        d.posOffset = 0; d.posCompCnt = 1; d.posCompType = 4; d.posAttrType = 1;
        d.currentPnMtx = 0;
        d.pnMtx[0][0][0] = 1.f; d.pnMtx[0][1][1] = 1.f; d.pnMtx[0][2][2] = 1.f;
        d.indices = idx.data(); d.indexCount = 6;
        return d;
    };
    gc.simulate_draw(draw2());

    const auto tmpPath = (std::filesystem::temp_directory_path() / "rtao_test.obj").string();
    gc.request_dump(tmpPath);
    gc.end_frame();

    REQUIRE(!gc.last_dump_message().empty());
    CHECK(gc.last_dump_message().find("failed") == std::string::npos);

    // Parse the file
    int vLines = 0, fLines = 0;
    {
        std::ifstream f(tmpPath);
        REQUIRE(f.is_open());
        std::string line;
        while (std::getline(f, line)) {
            if (line.size() >= 2 && line[0] == 'v' && line[1] == ' ') ++vLines;
            if (line.size() >= 2 && line[0] == 'f' && line[1] == ' ') ++fLines;
        }
    }

    // 2 triangles = 6 vertices (v) and 2 faces (f)
    CHECK(vLines == 6);
    CHECK(fLines == 2);

    std::filesystem::remove(tmpPath);
}

// ==========================================================================
// TEST: OBJ dump message reports triangle count
// ==========================================================================
TEST_CASE("GeometryCollector: dump message includes triangle count") {
    GeometryCollector gc;

    std::vector<uint8_t> v; std::vector<uint16_t> idx;
    gc.simulate_draw(make_triangle_draw(v, idx, 0,0,0, 1,0,0, 0,1,0));

    const auto tmpPath = (std::filesystem::temp_directory_path() / "rtao_count.obj").string();
    gc.request_dump(tmpPath);
    gc.end_frame();

    CHECK(gc.last_dump_message().find("1") != std::string::npos);

    std::filesystem::remove(tmpPath);
}
