#include <catch2/catch_test_macros.hpp>
#include "bvh.hpp"

using dusk::rtao::Bvh;
using dusk::rtao::Ray;
using dusk::rtao::Triangle;
using dusk::rtao::Vec3;
using dusk::rtao::make_ray;

// ==========================================================================
// TEST: empty input
// ==========================================================================
TEST_CASE("Bvh: empty input produces no nodes") {
    Bvh bvh;
    bvh.build({});
    CHECK(bvh.empty());
    CHECK(bvh.node_count() == 0);
    CHECK(!bvh.intersects_any(make_ray({0,0,0}, {0,0,1}), 100.f));
}

// ==========================================================================
// TEST: single triangle - hit
// ==========================================================================
TEST_CASE("Bvh: ray hits single triangle") {
    // Triangle at z=5, spanning x in [-1,1] and y in [0,1]
    Bvh bvh;
    bvh.build({{ {-1,0,5}, {1,0,5}, {0,1,5} }});
    // Centroid is at (0, 0.33, 5) — aim straight at it from z=0
    Ray r = make_ray({0, 0.33f, 0}, {0, 0, 1});
    CHECK(bvh.intersects_any(r, 10.f));
}

// ==========================================================================
// TEST: single triangle - miss (wrong direction)
// ==========================================================================
TEST_CASE("Bvh: ray misses single triangle - wrong direction") {
    Bvh bvh;
    bvh.build({{ {-1,0,5}, {1,0,5}, {0,1,5} }});
    Ray r = make_ray({0, 0.33f, 0}, {0, 0, -1});  // aimed away from z=5
    CHECK(!bvh.intersects_any(r, 10.f));
}

// ==========================================================================
// TEST: single triangle - miss (aimed beside it)
// ==========================================================================
TEST_CASE("Bvh: ray misses single triangle - aimed beside it") {
    Bvh bvh;
    bvh.build({{ {-1,0,5}, {1,0,5}, {0,1,5} }});
    Ray r = make_ray({5, 5, 0}, {0, 0, 1});  // clearly off to the side
    CHECK(!bvh.intersects_any(r, 20.f));
}

// ==========================================================================
// TEST: tmax clips the intersection
// ==========================================================================
TEST_CASE("Bvh: ray misses when tmax is less than hit distance") {
    Bvh bvh;
    bvh.build({{ {-1,0,10}, {1,0,10}, {0,1,10} }});
    Ray r = make_ray({0, 0.33f, 0}, {0, 0, 1});
    CHECK(!bvh.intersects_any(r,  5.f));   // triangle at z=10, tmax=5 → miss
    CHECK( bvh.intersects_any(r, 15.f));   // tmax=15 → hit
}

// ==========================================================================
// TEST: double-sided intersection (ray from behind)
// ==========================================================================
TEST_CASE("Bvh: ray hits triangle from behind (double-sided)") {
    Bvh bvh;
    bvh.build({{ {-1,0,5}, {1,0,5}, {0,1,5} }});
    Ray r = make_ray({0, 0.33f, 10}, {0, 0, -1});  // approaching from z=10 toward z=5
    CHECK(bvh.intersects_any(r, 10.f));
}

// ==========================================================================
// TEST: two triangles - both independently hittable
// ==========================================================================
TEST_CASE("Bvh: two separate triangles are both hittable") {
    Bvh bvh;
    bvh.build({
        { {-4,0,5}, {-2,0,5}, {-3,1,5} },   // left triangle
        { { 2,0,5}, { 4,0,5}, { 3,1,5} },   // right triangle
    });
    Ray r_left  = make_ray({-3, 0.33f, 0}, {0, 0, 1});
    Ray r_right = make_ray({ 3, 0.33f, 0}, {0, 0, 1});
    CHECK(bvh.intersects_any(r_left,  10.f));
    CHECK(bvh.intersects_any(r_right, 10.f));
}

// ==========================================================================
// TEST: ray passes between two triangles
// ==========================================================================
TEST_CASE("Bvh: ray passing between two triangles misses both") {
    Bvh bvh;
    bvh.build({
        { {-4,0,5}, {-2,0,5}, {-3,1,5} },   // left  (x in [-4,-2])
        { { 2,0,5}, { 4,0,5}, { 3,1,5} },   // right (x in [2,4])
    });
    Ray r_gap = make_ray({0, 0.33f, 0}, {0, 0, 1});  // x=0, in the gap
    CHECK(!bvh.intersects_any(r_gap, 10.f));
}

// ==========================================================================
// TEST: node count is within the 2N bound for a binary BVH
// ==========================================================================
TEST_CASE("Bvh: node count bounded by 2N for N triangles") {
    std::vector<Triangle> tris;
    for (int i = 0; i < 64; ++i) {
        float x = float(i) * 2.f;
        tris.push_back({ {x,0,0}, {x+1,0,0}, {x,1,0} });
    }
    Bvh bvh;
    bvh.build(tris);
    CHECK(bvh.node_count() >= 1);
    CHECK(bvh.node_count() <= 2 * 64);
}

// ==========================================================================
// TEST: 8x8 quad grid - every cell is hittable, off-grid misses
// ==========================================================================
TEST_CASE("Bvh: 8x8 grid - all cells hit from above, off-grid misses") {
    // 8x8 grid of quads (128 triangles) in the z=0 plane
    std::vector<Triangle> tris;
    tris.reserve(128);
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            float fx = float(x), fy = float(y);
            tris.push_back({ {fx,fy,0}, {fx+1,fy,0}, {fx,fy+1,0} });
            tris.push_back({ {fx+1,fy+1,0}, {fx,fy+1,0}, {fx+1,fy,0} });
        }
    }
    Bvh bvh;
    bvh.build(tris);

    // Each cell center should be hit by a ray from above
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            Ray r = make_ray({float(x) + 0.5f, float(y) + 0.5f, 10.f}, {0, 0, -1});
            CHECK(bvh.intersects_any(r, 20.f));
        }
    }

    // Ray far outside the grid misses
    Ray r_miss = make_ray({100.f, 100.f, 10.f}, {0, 0, -1});
    CHECK(!bvh.intersects_any(r_miss, 20.f));
}
