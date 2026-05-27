#pragma once
#include "vertex_decoder.hpp"
#include <cstdint>
#include <vector>

namespace dusk::rtao {

struct AABB {
    Vec3 min{ +1e30f, +1e30f, +1e30f };
    Vec3 max{ -1e30f, -1e30f, -1e30f };

    void  expand(const Vec3& p);
    void  expand(const AABB& o);
    float surface_area() const;
    bool  valid() const;
};

struct Ray {
    Vec3 origin;
    Vec3 dir;
    Vec3 inv_dir;  // 1/dir, precomputed for slab AABB tests
};

// Normalizes dir and precomputes inv_dir.
Ray make_ray(Vec3 origin, Vec3 dir);

struct BvhNode {
    AABB     bounds;
    uint32_t hit_next;    // interior: left child index; leaf: unused
    uint32_t miss_next;   // node to jump to on AABB miss; UINT32_MAX = done
    uint32_t tri_offset;  // leaf: first triangle index in Bvh::m_tris
    uint32_t tri_count;   // 0 = interior, >0 = leaf
};

class Bvh {
public:
    // Builds the BVH from the given triangle list (makes an internal copy).
    void build(const std::vector<Triangle>& tris);

    // Returns true if any triangle is hit within [1e-4, tmax).
    // Double-sided: backfaces occlude just like frontfaces (correct for AO).
    bool intersects_any(const Ray& ray, float tmax) const;

    uint32_t node_count() const { return static_cast<uint32_t>(m_nodes.size()); }
    bool     empty()      const { return m_nodes.empty(); }

private:
    std::vector<BvhNode>  m_nodes;
    std::vector<Triangle> m_tris;  // reordered for spatial locality during build

    uint32_t build_node(uint32_t start, uint32_t end);
    uint32_t sah_split(uint32_t start, uint32_t end, const AABB& bounds);
    void     link_node(uint32_t idx, uint32_t miss);
};

} // namespace dusk::rtao
