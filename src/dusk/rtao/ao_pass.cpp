#include "ao_pass.hpp"
#include <aurora/post_render.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace dusk::rtao {

// ---------------------------------------------------------------------------
// GPU-side struct layouts (must match WGSL structs byte-for-byte)
// ---------------------------------------------------------------------------

// 48 bytes — matches WGSL BvhNode
struct GpuBvhNode {
    float    boundsMin[3]; // offset 0
    uint32_t leftChild;    // offset 12
    float    boundsMax[3]; // offset 16
    uint32_t rightChild;   // offset 28
    uint32_t triOffset;    // offset 32
    uint32_t triCount;     // offset 36
    uint32_t _pad[2];      // offset 40 → total 48
};
static_assert(sizeof(GpuBvhNode) == 48);

// 48 bytes — matches WGSL GpuTriangle
struct GpuTriangle {
    float a[3]; float _p0;
    float b[3]; float _p1;
    float c[3]; float _p2;
};
static_assert(sizeof(GpuTriangle) == 48);

// 112 bytes — matches WGSL Camera uniform
struct GpuCamera {
    float    invViewProj[16]; // col-major mat4x4 = inv(VP), offset 0
    uint32_t screenWidth;     // offset 64
    uint32_t screenHeight;    // offset 68
    uint32_t raysPerPixel;    // offset 72
    uint32_t frameSeed;       // offset 76
    float    maxDistance;     // offset 80
    float    normalBias;      // offset 84
    uint32_t debugMode;       // offset 88  0=AO,1=normals,2=depth,3=root-AABB
    uint32_t debugMode2;      // offset 92  0=limit-hits,1=AO,2=normals,3=depth,4=root-AABB
    float    camWorldX;       // offset 96  camera world position (for world-space AO)
    float    camWorldY;       // offset 100
    float    camWorldZ;       // offset 104
    uint32_t _pad2;           // offset 108
};
static_assert(sizeof(GpuCamera) == 112);

// ---------------------------------------------------------------------------
// CPU-side 4×4 matrix math
//
// Convention: mathematical matrices stored row-major (M[row][col]).
// The GPU receives column-major (transposed) so WGSL `v * M` = row-vector math.
// ---------------------------------------------------------------------------

// Standard cofactor 4×4 inverse (row-major flat input/output).
// Returns false if the matrix is singular.
static bool mat4_invert_flat(const float m[16], float inv[16]) {
    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];

    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (std::abs(det) < 1e-10f) return false;
    float id = 1.f / det;
    for (int i = 0; i < 16; ++i) inv[i] *= id;
    return true;
}

// Build inv(P) for the WGSL mat4x4 uniform.
//
// BVH triangles are in view space (camera at origin), so the unproject only
// needs to invert the GX projection matrix.  Forward transform:
//   clip = P_GX × view_pos  →  inv(P_GX) × clip = view_pos
//
// P_GX comes from cam.proj (row-major: projMtx[row][col]).
//
// WGSL receives inv(P) stored row-major in the column-major slot so that
//   WGSL  v * M_wgsl  =  (inv(P) × v)^T   ✓
//
// GX NDC Z: near=-1, far=0.  Aurora reversed-Z: load_depth returns -raw ∈ [-1,0].
static bool compute_inv_proj(const GeometryCollector::CameraData& cam,
                              float out_col_major[16]) {
    float P_flat[16] = {};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            P_flat[r*4 + c] = cam.proj[r][c];

    float invP_rm[16];
    if (!mat4_invert_flat(P_flat, invP_rm)) return false;

    // Storing invP_rm row-major into out_col_major puts inv(P)^T in WGSL
    // column-major: `v * M_wgsl = (inv(P) × v)^T`. ✓
    std::memcpy(out_col_major, invP_rm, 16 * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// WGSL compute shader
// ---------------------------------------------------------------------------

static const char kShader[] = R"(
// ---- GPU-side BVH structs (must match GpuBvhNode / GpuTriangle on CPU) ----

struct BvhNode {
    bounds_min  : vec3<f32>,   // offset 0
    left_child  : u32,         // offset 12  (0xFFFFFFFF for leaf)
    bounds_max  : vec3<f32>,   // offset 16
    right_child : u32,         // offset 28
    tri_offset  : u32,         // offset 32
    tri_count   : u32,         // offset 36  (0=interior, 1=leaf)
    range_first : u32,         // offset 40  (build-time, ignored here)
    range_last  : u32,         // offset 44  (build-time, ignored here)
}

struct GpuTriangle {
    a  : vec3<f32>,
    _p0: f32,
    b  : vec3<f32>,
    _p1: f32,
    c  : vec3<f32>,
    _p2: f32,
}

struct Camera {
    inv_view_proj : mat4x4<f32>,  // offset 0 (column-major) = inv(P), unprojects NDC → view space
    screen_width  : u32,          // offset 64
    screen_height : u32,          // offset 68
    rays_per_pixel: u32,          // offset 72
    frame_seed    : u32,          // offset 76
    max_distance  : f32,          // offset 80
    normal_bias   : f32,          // offset 84
    debug_mode    : u32,          // offset 88  0=AO, 1=normals, 2=depth, 3=root-AABB
    debug_mode2   : u32,          // offset 92  0=limit-hits,1=AO,2=normals,3=depth,4=root-AABB,5=visit-heat,6=limit%
    cam_world_x   : f32,          // offset 96  camera world position
    cam_world_y   : f32,          // offset 100
    cam_world_z   : f32,          // offset 104
    _pad2         : u32,          // offset 108
}

@group(0) @binding(0) var<storage, read> bvh_nodes  : array<BvhNode>;
@group(0) @binding(1) var<storage, read> triangles   : array<GpuTriangle>;
@group(0) @binding(2) var<uniform>       cam         : Camera;
@group(0) @binding(3) var               t_depth     : texture_depth_2d;
@group(0) @binding(4) var               ao_out      : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var               limits_out  : texture_storage_2d<rgba8unorm, write>;

// ---- PCG random number generator ----

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rand_f(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed) / 4294967295.0;
}

// ---- Cosine-weighted hemisphere sample around `normal` ----

fn cosine_hemisphere(normal: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand_f(seed);
    let r2 = rand_f(seed);
    let phi = 6.2831853 * r1;
    let sr2 = sqrt(r2);
    let lx  = cos(phi) * sr2;
    let ly  = sin(phi) * sr2;
    let lz  = sqrt(max(0.0, 1.0 - r2));

    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.99) { up = vec3<f32>(1.0, 0.0, 0.0); }
    let tangent   = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return lx * tangent + ly * bitangent + lz * normal;
}

// ---- AABB slab test ----

fn aabb_hit(node: BvhNode, origin: vec3<f32>, inv_dir: vec3<f32>, tmax: f32) -> bool {
    let t1 = (node.bounds_min - origin) * inv_dir;
    let t2 = (node.bounds_max - origin) * inv_dir;
    let tmin_v = min(t1, t2);
    let tmax_v = max(t1, t2);
    let tmin_s = max(max(tmin_v.x, tmin_v.y), max(tmin_v.z, 0.0));
    let tmax_s = min(min(tmax_v.x, tmax_v.y), min(tmax_v.z, tmax));
    return tmin_s <= tmax_s;
}

// ---- Möller–Trumbore triangle intersection (double-sided) ----

fn ray_tri_hit(origin: vec3<f32>, dir: vec3<f32>,
               tri: GpuTriangle, tmax: f32) -> bool {
    let a = tri.a;
    let b = tri.b;
    let c = tri.c;
    let e1 = b - a;
    let e2 = c - a;
    let h  = cross(dir, e2);
    let det = dot(e1, h);
    if (abs(det) < 1e-7) { return false; }
    let inv_det = 1.0 / det;
    let s = origin - a;
    let u = dot(s, h) * inv_det;
    if (u < 0.0 || u > 1.0) { return false; }
    let q = cross(s, e1);
    let v = dot(dir, q) * inv_det;
    if (v < 0.0 || u + v > 1.0) { return false; }
    let t = dot(e2, q) * inv_det;
    return t > 1e-4 && t < tmax;
}

// ---- Stack-based BVH traversal (left_child / right_child format) ----
// kMaxNodeVisits guards against runaway traversal in degenerate BVH subtrees.
// 4096 is generous for a depth-≤62 LBVH (30 Morton bits + 32 index tie-break
// bits = max depth 62; stack-based DFS visits each node at most once per ray).
// Raising this causes GPU stalls: kMaxNodeVisits × rays_per_pixel × num_pixels
// of traversal work quickly reaches billions of iterations per frame.
const kMaxNodeVisits: u32 = 4096u;

// Returns 0=miss, 1=hit, 2=visit-limit reached.
// Accumulates per-traversal node visits into *acc_visits for the heatmap.
fn intersects_any(origin: vec3<f32>, dir: vec3<f32>, tmax: f32,
                  acc_visits: ptr<function, u32>) -> u32 {
    if (arrayLength(&bvh_nodes) == 0u) { return 0u; }
    let inv_dir = vec3<f32>(1.0/dir.x, 1.0/dir.y, 1.0/dir.z);
    // 128-entry stack: Karras LBVH with 30-bit Morton codes has max depth ~30;
    // duplicate codes allow tie-breaking up to ~62 levels deep.  128 entries
    // gives comfortable headroom for two children pushed per level.
    var stk: array<u32, 128>;
    var sp: i32 = 1;
    stk[0] = 0u;
    var visits: u32 = 0u;
    loop {
        if (sp <= 0 || visits >= kMaxNodeVisits) { break; }
        sp -= 1;
        visits += 1u;
        let idx  = stk[u32(sp)];
        let node = bvh_nodes[idx];
        if (!aabb_hit(node, origin, inv_dir, tmax)) { continue; }
        if (node.tri_count > 0u) {
            for (var i = node.tri_offset; i < node.tri_offset + node.tri_count; i += 1u) {
                if (ray_tri_hit(origin, dir, triangles[i], tmax)) {
                    *acc_visits += visits;
                    return 1u;
                }
            }
        } else {
            // Guard: need sp+2 <= 127 before pushing two children.
            if (sp < 126) {
                // Push right first so left is processed first (LIFO)
                stk[u32(sp)] = node.right_child; sp += 1;
                stk[u32(sp)] = node.left_child;  sp += 1;
            }
        }
    }
    *acc_visits += visits;
    return select(0u, 2u, visits >= kMaxNodeVisits);
}

// ---- Depth → view-space position ----
// Aurora uses reversed-Z (near=1, far=0 in the depth buffer), so we invert
// before unprojecting through inv(P).  The result is in GX view space.

fn load_depth(px: u32, py: u32) -> f32 {
    let raw = textureLoad(t_depth, vec2<i32>(i32(px), i32(py)), 0);
    // Aurora reversed-Z: GPU shader does out.pos.z = -out.pos.z on GX clip Z,
    // where GX produces clip Z in [-w, 0] (near=-1, far=0 in NDC).
    // Negating gives depth in [0, 1] (near=1, far=0 in the buffer).
    // To recover the original GX NDC Z expected by inv(P_GX): z_ndc = -raw.
    return -raw;
}

fn unproject(px: u32, py: u32) -> vec4<f32> {
    let d = load_depth(px, py);
    let ndc_x = (f32(px) + 0.5) / f32(cam.screen_width)  *  2.0 - 1.0;
    let ndc_y = (f32(py) + 0.5) / f32(cam.screen_height) * -2.0 + 1.0;
    let ndc_h = vec4<f32>(ndc_x, ndc_y, d, 1.0);
    let world_h = ndc_h * cam.inv_view_proj;
    if (abs(world_h.w) < 1e-6) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    return vec4<f32>(world_h.xyz / world_h.w, 1.0);
}

// ---- Normal estimation from depth-buffer gradients ----

fn estimate_normal(px: u32, py: u32, pos: vec3<f32>) -> vec3<f32> {
    let W = cam.screen_width;
    let H = cam.screen_height;
    // Use the two neighbors that avoid going out of bounds
    let px1 = select(px + 1u, px - 1u, px + 1u >= W);
    let py1 = select(py + 1u, py - 1u, py + 1u >= H);
    let sx   = select(1.0, -1.0, px + 1u >= W);
    let sy   = select(1.0, -1.0, py + 1u >= H);

    let p1 = unproject(px1, py);
    let p2 = unproject(px,  py1);
    if (p1.w < 0.5 || p2.w < 0.5) { return vec3<f32>(0.0, 1.0, 0.0); }

    let dx = (p1.xyz - pos) * sx;
    let dy = (p2.xyz - pos) * sy;
    let n  = cross(dx, dy);
    let nl = length(n);
    if (nl < 1e-6) { return vec3<f32>(0.0, 1.0, 0.0); }
    var n_norm = n / nl;
    // Orient toward camera.  BVH and positions are in view space; camera is at origin.
    if (dot(n_norm, normalize(-pos)) < 0.0) { n_norm = -n_norm; }
    return n_norm;
}

// ---- Main compute entry point ----

// Dispatched at half resolution; each thread writes a 2×2 pixel block.
// Always runs AO rays so both panels have fresh hit/limit data regardless
// of which debug view is selected.
@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x * 2u;
    let py = gid.y * 2u;
    if (px >= cam.screen_width || py >= cam.screen_height) { return; }

    let W = i32(cam.screen_width);
    let H = i32(cam.screen_height);
    let c00 = vec2<i32>(i32(px),   i32(py));
    let c10 = vec2<i32>(i32(px)+1, i32(py));
    let c01 = vec2<i32>(i32(px),   i32(py)+1);
    let c11 = vec2<i32>(i32(px)+1, i32(py)+1);

    let d = textureLoad(t_depth, vec2<i32>(i32(px), i32(py)), 0);
    if (d <= 0.0001) {
        // Sky pixel — red in left panel, black in right panel
        let ao_sky  = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        let lim_sky = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        textureStore(ao_out,    vec2<i32>(i32(px),   i32(py)),   ao_sky);
        textureStore(ao_out,    vec2<i32>(i32(px)+1, i32(py)),   ao_sky);
        textureStore(ao_out,    vec2<i32>(i32(px),   i32(py)+1), ao_sky);
        textureStore(ao_out,    vec2<i32>(i32(px)+1, i32(py)+1), ao_sky);
        textureStore(limits_out, vec2<i32>(i32(px),   i32(py)),   lim_sky);
        textureStore(limits_out, vec2<i32>(i32(px)+1, i32(py)),   lim_sky);
        textureStore(limits_out, vec2<i32>(i32(px),   i32(py)+1), lim_sky);
        textureStore(limits_out, vec2<i32>(i32(px)+1, i32(py)+1), lim_sky);
        return;
    }

    let pos_h = unproject(px, py);
    if (pos_h.w < 0.5) {
        let bad = vec4<f32>(0.0, 0.0, 1.0, 1.0);
        textureStore(ao_out,    vec2<i32>(i32(px),   i32(py)),   bad);
        textureStore(ao_out,    vec2<i32>(i32(px)+1, i32(py)),   bad);
        textureStore(ao_out,    vec2<i32>(i32(px),   i32(py)+1), bad);
        textureStore(ao_out,    vec2<i32>(i32(px)+1, i32(py)+1), bad);
        textureStore(limits_out, vec2<i32>(i32(px),   i32(py)),   bad);
        textureStore(limits_out, vec2<i32>(i32(px)+1, i32(py)),   bad);
        textureStore(limits_out, vec2<i32>(i32(px),   i32(py)+1), bad);
        textureStore(limits_out, vec2<i32>(i32(px)+1, i32(py)+1), bad);
        return;
    }
    let pos    = pos_h.xyz;
    let normal = estimate_normal(px, py, pos);

    // Always run AO rays — both panels need hit/limit_hit/visit data.
    // BVH and positions are in view space: camera is at the origin.
    let dist_from_cam  = length(pos);
    let ray_origin = pos + normal * max(dist_from_cam * cam.normal_bias, cam.normal_bias * 10.0);
    var seed = pcg(((py * cam.screen_width + px) * 1664525u) ^ cam.frame_seed);
    var hits: u32 = 0u;
    var limit_hits: u32 = 0u;
    var total_visits: u32 = 0u;
    let N = cam.rays_per_pixel;
    for (var i = 0u; i < N; i += 1u) {
        let dir = cosine_hemisphere(normal, &seed);
        let r = intersects_any(ray_origin, dir, cam.max_distance, &total_visits);
        if (r == 1u) { hits       += 1u; }
        if (r == 2u) { limit_hits += 1u; }
    }

    // Normalize only over rays that completed traversal — limit-reached rays are
    // neither hits nor misses, so including them in the denominator biases AO bright.
    let valid = N - limit_hits;
    let ao = select(0.5, 1.0 - f32(hits) / f32(valid), valid > 0u);

    // Root AABB hit — reused by both panels when debug_mode == 3 / debug_mode2 == 4.
    let root = bvh_nodes[0];
    let root_hit = aabb_hit(root, pos, vec3<f32>(1.0/0.1, 1.0/0.1, 1.0/(-1.0)), cam.max_distance);

    // Left panel (ao_out): select based on debug_mode
    var ao_val: vec4<f32>;
    if (cam.debug_mode == 1u) {
        ao_val = vec4<f32>(normal * 0.5 + 0.5, 1.0);
    } else if (cam.debug_mode == 2u) {
        ao_val = vec4<f32>(dist_from_cam / 2000.0, 0.0, 0.0, 1.0);
    } else if (cam.debug_mode == 3u) {
        ao_val = select(vec4<f32>(1.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 1.0, 0.0, 1.0), root_hit);
    } else {
        ao_val = vec4<f32>(ao, ao, ao, 1.0);
    }

    // Right panel (limits_out): select based on debug_mode2
    var lim_val: vec4<f32>;
    if (cam.debug_mode2 == 1u) {
        lim_val = vec4<f32>(ao, ao, ao, 1.0);
    } else if (cam.debug_mode2 == 2u) {
        lim_val = vec4<f32>(normal * 0.5 + 0.5, 1.0);
    } else if (cam.debug_mode2 == 3u) {
        lim_val = vec4<f32>(dist_from_cam / 2000.0, 0.0, 0.0, 1.0);
    } else if (cam.debug_mode2 == 4u) {
        lim_val = select(vec4<f32>(1.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 1.0, 0.0, 1.0), root_hit);
    } else if (cam.debug_mode2 == 5u) {
        // Visit heat: avg node visits per ray, clamped to [0, 4096].
        // Bright red = expensive traversal (bad BVH quality / inflated scene AABB).
        let avg = f32(total_visits) / f32(N);
        lim_val = vec4<f32>(min(avg / 4096.0, 1.0), 0.0, 0.0, 1.0);
    } else if (cam.debug_mode2 == 6u) {
        // Limit %: fraction of rays that hit kMaxNodeVisits, shown as magenta gradient.
        let frac = f32(limit_hits) / f32(N);
        lim_val = vec4<f32>(frac, 0.0, frac, 1.0);
    } else {
        // 0: Limit Hits — magenta where kMaxNodeVisits was reached (BVH corruption)
        lim_val = select(vec4<f32>(0.0, 0.0, 0.0, 1.0),
                         vec4<f32>(1.0, 0.0, 1.0, 1.0),
                         limit_hits > N / 2u);
    }

    textureStore(ao_out, c00, ao_val);
    if (c10.x < W) { textureStore(ao_out, c10, ao_val); }
    if (c01.y < H) { textureStore(ao_out, c01, ao_val); }
    if (c10.x < W && c01.y < H) { textureStore(ao_out, c11, ao_val); }

    textureStore(limits_out, c00, lim_val);
    if (c10.x < W) { textureStore(limits_out, c10, lim_val); }
    if (c01.y < H) { textureStore(limits_out, c01, lim_val); }
    if (c10.x < W && c01.y < H) { textureStore(limits_out, c11, lim_val); }
}
)";

// ---------------------------------------------------------------------------
// WebGPU helpers
// ---------------------------------------------------------------------------

static WGPUBuffer create_buffer(WGPUDevice device, uint64_t size, WGPUBufferUsage usage,
                                 const void* data = nullptr) {
    WGPUBufferDescriptor desc{};
    desc.size             = size;
    desc.usage            = usage | WGPUBufferUsage_CopyDst;
    desc.mappedAtCreation = false;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(device, &desc);
    if (data) {
        WGPUQueue q = wgpuDeviceGetQueue(device);
        wgpuQueueWriteBuffer(q, buf, 0, data, size);
        wgpuQueueRelease(q);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// AoPass implementation
// ---------------------------------------------------------------------------

AoPass::~AoPass() {
    if (m_bindGroup)   wgpuBindGroupRelease(m_bindGroup);
    if (m_depthView)   wgpuTextureViewRelease(m_depthView);
    if (m_limView)     wgpuTextureViewRelease(m_limView);
    if (m_limTex)      wgpuTextureRelease(m_limTex);
    if (m_aoView)      wgpuTextureViewRelease(m_aoView);
    if (m_aoTex)       wgpuTextureRelease(m_aoTex);
    if (m_cameraUbo)   wgpuBufferRelease(m_cameraUbo);
    if (m_bgl)         wgpuBindGroupLayoutRelease(m_bgl);
    if (m_pipeline)    wgpuComputePipelineRelease(m_pipeline);
}

void AoPass::ensure_pipeline(WGPUDevice device) {
    if (m_pipeline) return;

    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code        = {kShader, WGPU_STRLEN};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslSrc.chain;
    WGPUShaderModule sm = wgpuDeviceCreateShaderModule(device, &smDesc);

    // Bind group layout: 6 entries
    WGPUBindGroupLayoutEntry entries[6] = {};

    entries[0].binding          = 0;
    entries[0].visibility       = WGPUShaderStage_Compute;
    entries[0].buffer.type      = WGPUBufferBindingType_ReadOnlyStorage;

    entries[1].binding          = 1;
    entries[1].visibility       = WGPUShaderStage_Compute;
    entries[1].buffer.type      = WGPUBufferBindingType_ReadOnlyStorage;

    entries[2].binding          = 2;
    entries[2].visibility       = WGPUShaderStage_Compute;
    entries[2].buffer.type      = WGPUBufferBindingType_Uniform;

    entries[3].binding                     = 3;
    entries[3].visibility                  = WGPUShaderStage_Compute;
    entries[3].texture.sampleType          = WGPUTextureSampleType_Depth;
    entries[3].texture.viewDimension       = WGPUTextureViewDimension_2D;
    entries[3].texture.multisampled        = 0;

    entries[4].binding                       = 4;
    entries[4].visibility                    = WGPUShaderStage_Compute;
    entries[4].storageTexture.access         = WGPUStorageTextureAccess_WriteOnly;
    entries[4].storageTexture.format         = WGPUTextureFormat_RGBA8Unorm;
    entries[4].storageTexture.viewDimension  = WGPUTextureViewDimension_2D;

    entries[5].binding                       = 5;
    entries[5].visibility                    = WGPUShaderStage_Compute;
    entries[5].storageTexture.access         = WGPUStorageTextureAccess_WriteOnly;
    entries[5].storageTexture.format         = WGPUTextureFormat_RGBA8Unorm;
    entries[5].storageTexture.viewDimension  = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 6;
    bglDesc.entries    = entries;
    m_bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts     = &m_bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout               = pl;
    cpDesc.compute.module       = sm;
    cpDesc.compute.entryPoint   = {"cs_main", WGPU_STRLEN};
    m_pipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);

    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(sm);
}

void AoPass::rebuild_output(WGPUDevice device, uint32_t w, uint32_t h) {
    if (m_aoView)  { wgpuTextureViewRelease(m_aoView);  m_aoView  = nullptr; }
    if (m_aoTex)   { wgpuTextureRelease(m_aoTex);       m_aoTex   = nullptr; }
    if (m_limView) { wgpuTextureViewRelease(m_limView); m_limView = nullptr; }
    if (m_limTex)  { wgpuTextureRelease(m_limTex);      m_limTex  = nullptr; }

    WGPUTextureDescriptor td{};
    td.usage         = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
    td.dimension     = WGPUTextureDimension_2D;
    td.size          = {w, h, 1};
    td.format        = WGPUTextureFormat_RGBA8Unorm;
    td.mipLevelCount = 1;
    td.sampleCount   = 1;
    m_aoTex  = wgpuDeviceCreateTexture(device, &td);
    m_aoView = wgpuTextureCreateView(m_aoTex, nullptr);
    m_limTex  = wgpuDeviceCreateTexture(device, &td);
    m_limView = wgpuTextureCreateView(m_limTex, nullptr);

    m_width  = w;
    m_height = h;
    m_bindGroupDirty = true;
}

void AoPass::rebuild_depth_binding(WGPUDevice device, WGPUTexture depthTex) {
    if (m_depthView) { wgpuTextureViewRelease(m_depthView); m_depthView = nullptr; }

    WGPUTextureViewDescriptor dvd{};
    dvd.format          = wgpuTextureGetFormat(depthTex);
    dvd.dimension       = WGPUTextureViewDimension_2D;
    dvd.baseMipLevel    = 0;
    dvd.mipLevelCount   = 1;
    dvd.baseArrayLayer  = 0;
    dvd.arrayLayerCount = 1;
    dvd.aspect          = WGPUTextureAspect_DepthOnly;
    m_depthView    = wgpuTextureCreateView(depthTex, &dvd);
    m_lastDepthTex = depthTex;
    m_bindGroupDirty = true;
}

void AoPass::rebuild_bind_group(WGPUDevice device) {
    if (m_bindGroup) { wgpuBindGroupRelease(m_bindGroup); m_bindGroup = nullptr; }

    WGPUBindGroupEntry entries[6] = {};
    entries[0].binding = 0; entries[0].buffer = m_lastNodeBuf; entries[0].size = WGPU_WHOLE_SIZE;
    entries[1].binding = 1; entries[1].buffer = m_lastTriBuf;  entries[1].size = WGPU_WHOLE_SIZE;
    entries[2].binding = 2; entries[2].buffer = m_cameraUbo;   entries[2].size = sizeof(GpuCamera);
    entries[3].binding = 3; entries[3].textureView = m_depthView;
    entries[4].binding = 4; entries[4].textureView = m_aoView;
    entries[5].binding = 5; entries[5].textureView = m_limView;

    WGPUBindGroupDescriptor bgd{};
    bgd.layout     = m_bgl;
    bgd.entryCount = 6;
    bgd.entries    = entries;
    m_bindGroup      = wgpuDeviceCreateBindGroup(device, &bgd);
    m_bindGroupDirty = false;
}

void AoPass::execute(WGPUDevice device, WGPUCommandEncoder encoder,
                     WGPUTexture depthTex,
                     const GeometryCollector::CameraData& cam,
                     WGPUBuffer nodeBuf, WGPUBuffer triBuf) {
    if (!cam.valid || !depthTex || !nodeBuf || !triBuf) return;

    // Invalidate bind group when the BVH buffers change frame-to-frame
    if (nodeBuf != m_lastNodeBuf || triBuf != m_lastTriBuf) {
        m_lastNodeBuf    = nodeBuf;
        m_lastTriBuf     = triBuf;
        m_bindGroupDirty = true;
    }

    const uint32_t w = wgpuTextureGetWidth(depthTex);
    const uint32_t h = wgpuTextureGetHeight(depthTex);
    if (w == 0 || h == 0) return;

    ensure_pipeline(device);
    if (!m_pipeline) return;

    // Resize output texture if needed
    if (w != m_width || h != m_height)
        rebuild_output(device, w, h);

    // Re-bind depth texture if it changed
    if (depthTex != m_lastDepthTex)
        rebuild_depth_binding(device, depthTex);

    // Upload camera UBO
    GpuCamera gpuCam{};
    gpuCam.screenWidth   = w;
    gpuCam.screenHeight  = h;
    gpuCam.raysPerPixel  = m_params.raysPerPixel;
    gpuCam.frameSeed     = m_frame++;
    gpuCam.maxDistance   = m_params.maxDistance;
    gpuCam.normalBias    = m_params.normalBias;
    gpuCam.debugMode     = m_params.debugMode;
    gpuCam.debugMode2    = m_params.debugMode2;
    if (!compute_inv_proj(cam, gpuCam.invViewProj)) return;
    // BVH triangles and depth are in view space; camera is at the origin.
    gpuCam.camWorldX = 0.f;
    gpuCam.camWorldY = 0.f;
    gpuCam.camWorldZ = 0.f;

    if (!m_cameraUbo) {
        m_cameraUbo = create_buffer(device, sizeof(GpuCamera), WGPUBufferUsage_Uniform);
        m_bindGroupDirty = true;
    }
    WGPUQueue q = wgpuDeviceGetQueue(device);
    wgpuQueueWriteBuffer(q, m_cameraUbo, 0, &gpuCam, sizeof(GpuCamera));
    wgpuQueueRelease(q);

    if (m_bindGroupDirty)
        rebuild_bind_group(device);

    // Dispatch compute
    WGPUComputePassDescriptor passDesc{};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
    wgpuComputePassEncoderSetPipeline(pass, m_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, m_bindGroup, 0, nullptr);
    // Half-resolution dispatch — each thread writes a 2×2 block (4× work reduction).
    const uint32_t gx = (w / 2 + 7) / 8;
    const uint32_t gy = (h / 2 + 7) / 8;
    wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

ImTextureID AoPass::imgui_texture_id()    const { return reinterpret_cast<ImTextureID>(m_aoView);  }
ImTextureID AoPass::limits_texture_id()  const { return reinterpret_cast<ImTextureID>(m_limView); }

} // namespace dusk::rtao
