#include "bvh.hpp"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>

namespace dusk::rtao {

// ---------------------------------------------------------------------------
// Vec3 helpers (kept local to avoid polluting vertex_decoder.hpp)
// ---------------------------------------------------------------------------

static float get_axis(const Vec3& v, int a) {
    switch (a) { case 0: return v.x; case 1: return v.y; default: return v.z; }
}

static Vec3 vec3_sub(const Vec3& a, const Vec3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

static float vec3_dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static Vec3 vec3_cross(const Vec3& a, const Vec3& b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

void AABB::expand(const Vec3& p) {
    min.x = std::min(min.x, p.x); min.y = std::min(min.y, p.y); min.z = std::min(min.z, p.z);
    max.x = std::max(max.x, p.x); max.y = std::max(max.y, p.y); max.z = std::max(max.z, p.z);
}

void AABB::expand(const AABB& o) {
    if (!o.valid()) return;
    expand(o.min);
    expand(o.max);
}

bool AABB::valid() const {
    return min.x <= max.x && min.y <= max.y && min.z <= max.z;
}

float AABB::surface_area() const {
    if (!valid()) return 0.f;
    float dx = max.x - min.x;
    float dy = max.y - min.y;
    float dz = max.z - min.z;
    return 2.f * (dx*dy + dy*dz + dz*dx);
}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

Ray make_ray(Vec3 origin, Vec3 dir) {
    float len = std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    if (len > 1e-7f) { dir.x /= len; dir.y /= len; dir.z /= len; }
    // Safe reciprocal: zero component → large value (correct for slab AABB tests)
    auto safe_inv = [](float x) {
        return x == 0.f ? std::copysign(1e30f, x) : 1.f / x;
    };
    return { origin, dir, { safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z) } };
}

// ---------------------------------------------------------------------------
// Traversal helpers
// ---------------------------------------------------------------------------

static bool aabb_hit(const AABB& box, const Ray& ray, float tmin, float tmax) {
    for (int a = 0; a < 3; ++a) {
        float t1 = (get_axis(box.min, a) - get_axis(ray.origin, a)) * get_axis(ray.inv_dir, a);
        float t2 = (get_axis(box.max, a) - get_axis(ray.origin, a)) * get_axis(ray.inv_dir, a);
        tmin = std::max(tmin, std::min(t1, t2));
        tmax = std::min(tmax, std::max(t1, t2));
    }
    return tmin <= tmax;
}

// Möller–Trumbore, double-sided (no backface culling — correct for AO).
static bool ray_triangle_hit(const Ray& ray, const Triangle& tri, float tmax) {
    Vec3  e1  = vec3_sub(tri.b, tri.a);
    Vec3  e2  = vec3_sub(tri.c, tri.a);
    Vec3  h   = vec3_cross(ray.dir, e2);
    float det = vec3_dot(e1, h);
    if (std::abs(det) < 1e-7f) return false;
    float inv_det = 1.f / det;
    Vec3  s = vec3_sub(ray.origin, tri.a);
    float u = vec3_dot(s, h) * inv_det;
    if (u < 0.f || u > 1.f) return false;
    Vec3  q = vec3_cross(s, e1);
    float v = vec3_dot(ray.dir, q) * inv_det;
    if (v < 0.f || u + v > 1.f) return false;
    float t = vec3_dot(e2, q) * inv_det;
    // t > 1e-4 avoids self-intersection when the ray origin is on a surface.
    return t > 1e-4f && t < tmax;
}

// ---------------------------------------------------------------------------
// BVH build
// ---------------------------------------------------------------------------

static constexpr uint32_t kLeafMax = 4;  // max triangles per leaf
static constexpr uint32_t kEnd     = UINT32_MAX;

static Vec3 triangle_centroid(const Triangle& t) {
    return { (t.a.x + t.b.x + t.c.x) / 3.f,
             (t.a.y + t.b.y + t.c.y) / 3.f,
             (t.a.z + t.b.z + t.c.z) / 3.f };
}

static AABB triangle_bounds(const Triangle& t) {
    AABB b;
    b.expand(t.a); b.expand(t.b); b.expand(t.c);
    return b;
}

void Bvh::build(const std::vector<Triangle>& tris) {
    m_nodes.clear();
    m_tris.clear();
    if (tris.empty()) return;
    m_tris = tris;
    m_nodes.reserve(2 * m_tris.size());  // binary BVH upper bound: 2N-1 nodes
    build_node(0, static_cast<uint32_t>(m_tris.size()));
    link_node(0, kEnd);
}

uint32_t Bvh::build_node(uint32_t start, uint32_t end) {
    uint32_t idx = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back({});  // reserve slot; filled in after children are built

    AABB bounds;
    for (uint32_t i = start; i < end; ++i) {
        bounds.expand(m_tris[i].a);
        bounds.expand(m_tris[i].b);
        bounds.expand(m_tris[i].c);
    }

    const uint32_t count = end - start;
    if (count <= kLeafMax) {
        m_nodes[idx] = { bounds, 0, 0, start, count };
        return idx;
    }

    const uint32_t split = sah_split(start, end, bounds);
    const uint32_t left  = build_node(start, split);
    const uint32_t right = build_node(split, end);

    // hit_next = left child, miss_next stores right child temporarily
    // (link_node will convert miss_next to the correct skip-link value)
    m_nodes[idx] = { bounds, left, right, 0, 0 };
    return idx;
}

uint32_t Bvh::sah_split(uint32_t start, uint32_t end, const AABB& bounds) {
    static constexpr int kBuckets = 8;

    const float parent_sa = bounds.surface_area();
    const uint32_t count  = end - start;

    if (parent_sa < 1e-10f) return (start + end) / 2;  // degenerate

    float best_cost  = FLT_MAX;
    int   best_axis  = -1;
    int   best_bucket = -1;

    for (int axis = 0; axis < 3; ++axis) {
        const float axis_min = get_axis(bounds.min, axis);
        const float axis_max = get_axis(bounds.max, axis);
        if (axis_max - axis_min < 1e-6f) continue;

        struct Bucket { AABB bounds; int count = 0; } buckets[kBuckets] = {};

        for (uint32_t i = start; i < end; ++i) {
            const float c = get_axis(triangle_centroid(m_tris[i]), axis);
            int b = static_cast<int>(kBuckets * (c - axis_min) / (axis_max - axis_min));
            b = std::max(0, std::min(b, kBuckets - 1));  // clamp both sides: FP rounding can push c slightly outside bounds
            ++buckets[b].count;
            buckets[b].bounds.expand(triangle_bounds(m_tris[i]));
        }

        // Precompute prefix (left) and suffix (right) aggregates.
        AABB left_b[kBuckets]; int left_n[kBuckets] = {};
        AABB right_b[kBuckets]; int right_n[kBuckets] = {};
        {
            AABB lb; int ln = 0;
            for (int b = 0; b < kBuckets; ++b) {
                lb.expand(buckets[b].bounds); ln += buckets[b].count;
                left_b[b] = lb; left_n[b] = ln;
            }
            AABB rb; int rn = 0;
            for (int b = kBuckets - 1; b >= 0; --b) {
                rb.expand(buckets[b].bounds); rn += buckets[b].count;
                right_b[b] = rb; right_n[b] = rn;
            }
        }

        for (int s = 1; s < kBuckets; ++s) {
            if (left_n[s-1] == 0 || right_n[s] == 0) continue;
            const float cost = 1.f + (left_n[s-1] * left_b[s-1].surface_area() +
                                      right_n[s]   * right_b[s].surface_area()) / parent_sa;
            if (cost < best_cost) {
                best_cost   = cost;
                best_axis   = axis;
                best_bucket = s;
            }
        }
    }

    // If no split improves over leaf cost, or all tris are in one bucket,
    // fall back to median split so the tree doesn't become infinitely deep.
    if (best_axis == -1) return (start + end) / 2;

    const float axis_min = get_axis(bounds.min, best_axis);
    const float axis_max = get_axis(bounds.max, best_axis);

    auto mid = std::partition(
        m_tris.begin() + start, m_tris.begin() + end,
        [&](const Triangle& t) {
            const float c = get_axis(triangle_centroid(t), best_axis);
            const int b = std::max(0, std::min(kBuckets - 1,
                static_cast<int>(kBuckets * (c - axis_min) / (axis_max - axis_min))));
            return b < best_bucket;
        });

    const uint32_t split = static_cast<uint32_t>(mid - m_tris.begin());
    if (split == start || split == end) return (start + end) / 2;
    return split;
}

// Converts the tree's temporary right-child pointers into proper miss-links.
// Called once after build_node on the root with miss = kEnd.
//
// Interior nodes: miss_next temporarily holds the right child index.
//   After linking: miss_next = parent's miss (skip entire subtree on AABB miss).
// Leaf nodes: miss_next = parent's miss (continue DFS after exhausting this leaf).
void Bvh::link_node(uint32_t idx, uint32_t miss) {
    BvhNode& n = m_nodes[idx];
    if (n.tri_count > 0) {          // leaf
        n.miss_next = miss;
        return;
    }
    const uint32_t right = n.miss_next;  // right child stored here temporarily
    n.miss_next = miss;                  // interior miss → skip to parent's continuation
    link_node(n.hit_next, right);        // left subtree: miss → jump to right sibling
    link_node(right, miss);              // right subtree: miss → parent's continuation
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

bool Bvh::intersects_any(const Ray& ray, float tmax) const {
    if (m_nodes.empty()) return false;

    uint32_t idx = 0;
    while (idx != kEnd) {
        const BvhNode& n = m_nodes[idx];
        if (!aabb_hit(n.bounds, ray, 0.f, tmax)) {
            idx = n.miss_next;
            continue;
        }
        if (n.tri_count > 0) {
            for (uint32_t i = n.tri_offset; i < n.tri_offset + n.tri_count; ++i) {
                if (ray_triangle_hit(ray, m_tris[i], tmax))
                    return true;
            }
            idx = n.miss_next;
        } else {
            idx = n.hit_next;  // descend into left child
        }
    }
    return false;
}

} // namespace dusk::rtao
