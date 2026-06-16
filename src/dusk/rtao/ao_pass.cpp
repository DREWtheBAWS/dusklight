#include "ao_pass.hpp"
#include <aurora/post_render.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
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

// 80 bytes — matches WGSL GpuTriangle (positions + UVs + alpha-texture index/flags)
struct GpuTriangle {
    float    a[3]; float    _p0;      // @ 0-15
    float    b[3]; float    _p1;      // @ 16-31
    float    c[3]; float    _p2;      // @ 32-47
    float    uva[2]; float  uvb[2];   // @ 48-63
    float    uvc[2]; uint32_t texIdx; uint32_t flags; // @ 64-79
};
static_assert(sizeof(GpuTriangle) == 80);

// 176 bytes — matches WGSL Camera uniform (both LBVH and TLAS shaders)
struct GpuCamera {
    float    invViewProj[16]; // col-major mat4x4 = inv(P), offset 0; unprojects NDC → view space
    uint32_t screenWidth;     // offset 64
    uint32_t screenHeight;    // offset 68
    uint32_t raysPerPixel;    // offset 72
    uint32_t frameSeed;       // offset 76
    float    maxDistance;     // offset 80
    float    normalBias;      // offset 84
    uint32_t debugMode;       // offset 88  0=AO,1=normals,2=depth,3=root-AABB
    uint32_t debugMode2;      // offset 92  0=limit-hits,1=AO,2=normals,3=depth,4=root-AABB
    float    camWorldX;       // offset 96  camera world position (used by LBVH path)
    float    camWorldY;       // offset 100
    float    camWorldZ;       // offset 104
    uint32_t _pad2;           // offset 108
    float    viewToWorld[12]; // offset 112 view→world 3×4 affine matrix, row-major
    uint32_t dynNodeCount;    // offset 160 GPU LBVH node count for dynamic (skinned) geometry; 0=none
    uint32_t _pad3[3];        // offset 164 padding to 176
};
static_assert(sizeof(GpuCamera) == 176);

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

// Compute the view→world 3×4 affine matrix (inverse of the view matrix) and
// pack it row-major into out[12].  Returns false if the view matrix is singular.
static bool compute_view_to_world(const GeometryCollector::CameraData& cam,
                                   float out[12]) {
    float V[16] = {};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            V[r*4+c] = cam.view[r][c];
    if (V[15] == 0.f) V[15] = 1.f;  // ensure homogeneous row is [0,0,0,1]

    float invV[16];
    if (!mat4_invert_flat(V, invV)) return false;

    // Pack the 3×4 portion row-major: out[r*4+c] = invV[r][c], r=0..2, c=0..3.
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            out[r*4+c] = invV[r*4+c];
    return true;
}

// ---------------------------------------------------------------------------
// WGSL compute shader (built dynamically to generate N alpha-texture slots)
// ---------------------------------------------------------------------------

static std::string build_ao_shader() {
    std::string s;
    s += R"(
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
    a       : vec3<f32>,   // offset 0
    _p0     : f32,         // offset 12
    b       : vec3<f32>,   // offset 16
    _p1     : f32,         // offset 28
    c       : vec3<f32>,   // offset 32
    _p2     : f32,         // offset 44
    uva     : vec2<f32>,   // offset 48
    uvb     : vec2<f32>,   // offset 56
    uvc     : vec2<f32>,   // offset 64
    tex_idx : u32,         // offset 72
    flags   : u32,         // offset 76
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
@group(0) @binding(1) var<storage, read> triangles  : array<GpuTriangle>;
@group(0) @binding(2) var<uniform>       cam        : Camera;
@group(0) @binding(3) var               t_depth    : texture_depth_2d;
@group(0) @binding(4) var               ao_out     : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var               limits_out : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var               s_alpha    : sampler;
)";
    // Generate 16 alpha-texture slots (bindings 7-22)
    for (int i = 0; i < 16; ++i) {
        s += "@group(0) @binding(";
        s += std::to_string(7 + i);
        s += ") var t_alpha_";
        s += std::to_string(i);
        s += " : texture_2d<f32>;\n";
    }
    // sample_alpha: barycentric UV interpolation + switch-based texture sample.
    // bar_u/bar_v are the Möller–Trumbore u/v coordinates: weight for b and c
    // respectively; weight for a = 1 - bar_u - bar_v.
    s += R"(
fn sample_alpha(tri: GpuTriangle, bar_u: f32, bar_v: f32) -> f32 {
    if ((tri.flags & 1u) == 0u) { return 1.0; }
    let w  = 1.0 - bar_u - bar_v;
    let uv = w * tri.uva + bar_u * tri.uvb + bar_v * tri.uvc;
    switch tri.tex_idx {
)";
    for (int i = 0; i < 16; ++i) {
        s += "        case ";
        s += std::to_string(i);
        s += "u: { return textureSampleLevel(t_alpha_";
        s += std::to_string(i);
        s += ", s_alpha, uv, 0.0).a; }\n";
    }
    s += "        default: { return 1.0; }\n    }\n}\n";
    s += R"(

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

// ---- Möller–Trumbore triangle intersection (double-sided) with alpha test ----

fn ray_tri_hit(origin: vec3<f32>, dir: vec3<f32>,
               tri: GpuTriangle, tmax: f32) -> bool {
    let e1 = tri.b - tri.a;
    let e2 = tri.c - tri.a;
    let h  = cross(dir, e2);
    let det = dot(e1, h);
    if (abs(det) < 1e-7) { return false; }
    let inv_det = 1.0 / det;
    let sv = origin - tri.a;
    let u = dot(sv, h) * inv_det;
    if (u < 0.0 || u > 1.0) { return false; }
    let q = cross(sv, e1);
    let v = dot(dir, q) * inv_det;
    if (v < 0.0 || u + v > 1.0) { return false; }
    let t = dot(e2, q) * inv_det;
    if (t <= 1e-4 || t >= tmax) { return false; }
    // Alpha test: sample the texture at the interpolated UV and discard if transparent.
    if (sample_alpha(tri, u, v) < 0.5) { return false; }
    return true;
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
    let W = cam.screen_width; let H = cam.screen_height;
    // Sample all 4 axis-aligned neighbors; for each axis pick the one on the same
    // surface (no depth discontinuity) to avoid wrong normals at mesh silhouettes.
    let d0     = length(pos);
    let thresh = max(d0 * 0.15, 0.1);

    let xp = unproject(select(px + 1u, px - 1u, px + 1u >= W), py);
    let xn = unproject(select(px - 1u, px + 1u, px == 0u),     py);
    let yp = unproject(px, select(py + 1u, py - 1u, py + 1u >= H));
    let yn = unproject(px, select(py - 1u, py + 1u, py == 0u));

    let xp_ok = xp.w > 0.5 && abs(length(xp.xyz) - d0) < thresh && (px + 1u < W);
    let xn_ok = xn.w > 0.5 && abs(length(xn.xyz) - d0) < thresh && (px > 0u);
    let yp_ok = yp.w > 0.5 && abs(length(yp.xyz) - d0) < thresh && (py + 1u < H);
    let yn_ok = yn.w > 0.5 && abs(length(yn.xyz) - d0) < thresh && (py > 0u);

    // Pick best tangent for each axis; return camera-facing fallback at silhouettes.
    var tx: vec3<f32>;
    if      (xp_ok) { tx = xp.xyz - pos; }
    else if (xn_ok) { tx = pos - xn.xyz; }
    else            { return normalize(-pos); }

    var ty: vec3<f32>;
    if      (yp_ok) { ty = yp.xyz - pos; }
    else if (yn_ok) { ty = pos - yn.xyz; }
    else            { return normalize(-pos); }

    let n  = cross(tx, ty);
    let nl = length(n);
    if (nl < 1e-6) { return normalize(-pos); }
    var nn = n / nl;
    if (dot(nn, normalize(-pos)) < 0.0) { nn = -nn; }
    return nn;
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
    return s;
}

// ---------------------------------------------------------------------------
// TLAS/BLAS two-level AO shader
// ---------------------------------------------------------------------------

static std::string build_tlas_ao_shader() {
    std::string s;
    s += R"(
// ---- Structs ---------------------------------------------------------------
// AcNode: shared 48-byte format for both TLAS and monolithic BLAS nodes.
// Offsets match GpuNode on the CPU (BlasCache::GpuNode).
struct AcNode {
    bounds_min : vec3<f32>,  // offset 0
    hit_next   : u32,        // offset 12  (0xFFFFFFFF = done)
    bounds_max : vec3<f32>,  // offset 16
    miss_next  : u32,        // offset 28
    tri_offset : u32,        // offset 32  TLAS leaf: instance index; BLAS leaf: tri index (monolithic)
    tri_count  : u32,        // offset 36  0=interior; TLAS leaf=1; BLAS leaf=tri count
    _pad0      : u32,        // offset 40
    _pad1      : u32,        // offset 44
}

// TlasInstance: 64 bytes, matches GpuTlasInstance on the CPU.
struct TlasInstance {
    pnMtxInv_r0     : vec4<f32>,   // offset 0   row 0 of view→local 3×4 matrix
    pnMtxInv_r1     : vec4<f32>,   // offset 16  row 1
    pnMtxInv_r2     : vec4<f32>,   // offset 32  row 2
    blas_node_offset: u32,          // offset 48  (informational, already baked into node indices)
    blas_tri_offset : u32,          // offset 52  (informational, already baked into tri_offset)
    blas_node_count : u32,          // offset 56
    blas_tri_count  : u32,          // offset 60
}

// BlasTri: 48 bytes (positions only; UVs deferred to Phase 3b).
struct BlasTri {
    v0 : vec4<f32>,  // xyz = local-space position, w = pad
    v1 : vec4<f32>,
    v2 : vec4<f32>,
}

// Camera UBO: 176 bytes, matches GpuCamera on the CPU.
struct Camera {
    inv_view_proj  : mat4x4<f32>,  // offset 0   inv(P), unprojects NDC → view space
    screen_width   : u32,          // offset 64
    screen_height  : u32,          // offset 68
    rays_per_pixel : u32,          // offset 72
    frame_seed     : u32,          // offset 76
    max_distance   : f32,          // offset 80
    normal_bias    : f32,          // offset 84
    debug_mode     : u32,          // offset 88
    debug_mode2    : u32,          // offset 92
    cam_world_x    : f32,          // offset 96
    cam_world_y    : f32,          // offset 100
    cam_world_z    : f32,          // offset 104
    _pad2          : u32,          // offset 108
    vtw_r0         : vec4<f32>,    // offset 112  row 0 of view→world 3×4 matrix
    vtw_r1         : vec4<f32>,    // offset 128  row 1
    vtw_r2         : vec4<f32>,    // offset 144  row 2
    dyn_node_count : u32,          // offset 160  GPU LBVH nodes for skinned geo; 0 = none
    _pad3a         : u32,          // offset 164
    _pad3b         : u32,          // offset 168
    _pad3c         : u32,          // offset 172
}

// ---- Bindings ---------------------------------------------------------------
@group(0) @binding(0) var<storage, read> tlas_nodes     : array<AcNode>;
@group(0) @binding(1) var<storage, read> tlas_instances : array<TlasInstance>;
@group(0) @binding(2) var<uniform>       cam            : Camera;
@group(0) @binding(3) var               t_depth        : texture_depth_2d;
@group(0) @binding(4) var               ao_out         : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var               limits_out     : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var               s_alpha        : sampler;
@group(0) @binding(7) var<storage, read> blas_nodes     : array<AcNode>;
@group(0) @binding(8) var<storage, read> blas_tris      : array<BlasTri>;
)";
    // Alpha textures: same 16 slots as LBVH for future UV support.
    for (int i = 0; i < 16; ++i) {
        s += "@group(0) @binding(";
        s += std::to_string(9 + i);
        s += ") var t_alpha_";
        s += std::to_string(i);
        s += " : texture_2d<f32>;\n";
    }
    // Dynamic LBVH bindings (bindings 25 and 26).
    // LbvhNode matches GpuBvhNode layout (48 bytes) with left/right children for stack traversal.
    // DynBlasTri matches GpuTriangle layout (80 bytes) from GpuBvhBuilder; only positions read.
    s += R"(
@group(0) @binding(25) var<storage, read> dyn_blas_nodes : array<LbvhNode>;
@group(0) @binding(26) var<storage, read> dyn_blas_tris  : array<DynBlasTri>;

struct LbvhNode {
    bounds_min  : vec3<f32>,  // offset 0
    left_child  : u32,        // offset 12  (0xFFFFFFFF for leaf)
    bounds_max  : vec3<f32>,  // offset 16
    right_child : u32,        // offset 28
    tri_offset  : u32,        // offset 32
    tri_count   : u32,        // offset 36
    _pad0       : u32,        // offset 40
    _pad1       : u32,        // offset 44
}

struct DynBlasTri {
    a    : vec3<f32>,  // offset 0
    _p0  : f32,        // offset 12
    b    : vec3<f32>,  // offset 16
    _p1  : f32,        // offset 28
    c    : vec3<f32>,  // offset 32
    _p2  : f32,        // offset 44
    _u0  : vec4<f32>,  // offset 48 (UVs — unused for skinned meshes)
    _u1  : vec4<f32>,  // offset 64 (UVs + texIdx + flags — unused)
}
)";
    s += R"(

// ---- PCG RNG ----------------------------------------------------------------
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn rand_f(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed) / 4294967295.0;
}

// ---- Cosine-weighted hemisphere sample ----
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

// ---- View→world transform (for TLAS-level ray) ----
// cam.vtw_r0/r1/r2 are the rows of the 3×4 view→world affine matrix.
fn xf_vtw_point(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(cam.vtw_r0, vec4<f32>(p, 1.0)),
        dot(cam.vtw_r1, vec4<f32>(p, 1.0)),
        dot(cam.vtw_r2, vec4<f32>(p, 1.0))
    );
}
fn xf_vtw_dir(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(cam.vtw_r0.xyz, d),
        dot(cam.vtw_r1.xyz, d),
        dot(cam.vtw_r2.xyz, d)
    );
}

// ---- AABB slab test (world-space for TLAS, local-space for BLAS) ----
fn aabb_hit(node: AcNode, origin: vec3<f32>, inv_dir: vec3<f32>, tmax: f32) -> bool {
    let t1 = (node.bounds_min - origin) * inv_dir;
    let t2 = (node.bounds_max - origin) * inv_dir;
    let tmin_v = min(t1, t2);
    let tmax_v = max(t1, t2);
    let tmin_s = max(max(tmin_v.x, tmin_v.y), max(tmin_v.z, 0.0));
    let tmax_s = min(min(tmax_v.x, tmax_v.y), min(tmax_v.z, tmax));
    return tmin_s <= tmax_s;
}

// ---- Ray transform into instance local space ----
// pnMtxInv is stored as three vec4 rows of a 3×4 affine matrix.
fn xf_point(inst: TlasInstance, p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(inst.pnMtxInv_r0, vec4<f32>(p, 1.0)),
        dot(inst.pnMtxInv_r1, vec4<f32>(p, 1.0)),
        dot(inst.pnMtxInv_r2, vec4<f32>(p, 1.0))
    );
}
fn xf_dir(inst: TlasInstance, d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(inst.pnMtxInv_r0.xyz, d),
        dot(inst.pnMtxInv_r1.xyz, d),
        dot(inst.pnMtxInv_r2.xyz, d)
    );
}

// ---- BLAS triangle intersection (position only, no alpha) ----
// The t-parameter is the same in both view space and local space (it's the
// ray parameter, not a distance), so tmax can be passed through unchanged.
fn blas_ray_tri_hit(origin: vec3<f32>, dir: vec3<f32>, tri: BlasTri, tmax: f32) -> bool {
    let e1 = tri.v1.xyz - tri.v0.xyz;
    let e2 = tri.v2.xyz - tri.v0.xyz;
    let h  = cross(dir, e2);
    let det = dot(e1, h);
    if (abs(det) < 1e-7) { return false; }
    let inv_det = 1.0 / det;
    let sv = origin - tri.v0.xyz;
    let u = dot(sv, h) * inv_det;
    if (u < 0.0 || u > 1.0) { return false; }
    let q = cross(sv, e1);
    let v = dot(dir, q) * inv_det;
    if (v < 0.0 || u + v > 1.0) { return false; }
    let t = dot(e2, q) * inv_det;
    return (t > 1e-4 && t < tmax);
}

// ---- Stackless BLAS traversal in local space ----
// Node indices in blas_nodes are already absolute (monolithic buffer offsets
// were baked in at build time).
//
// Per-BLAS visit cap is independent of the TLAS cap so a single complex BLAS
// cannot exhaust the budget for remaining TLAS instances in the same ray.
// Returns 0=miss, 1=hit, 2=blas-visit-limit reached.
const kMaxTlasVisits: u32 = 1024u;  // TLAS-level node budget per ray
const kMaxBlasVisits: u32 = 2048u;  // per-BLAS node budget (each leaf capped separately)

fn traverse_blas(inst: TlasInstance, l_origin: vec3<f32>, l_dir: vec3<f32>,
                 tmax: f32, heat_acc: ptr<function, u32>) -> u32 {
    let inv_dir = vec3<f32>(1.0/l_dir.x, 1.0/l_dir.y, 1.0/l_dir.z);
    var idx: u32 = inst.blas_node_offset;
    var local_v: u32 = 0u;
    loop {
        if (idx == 0xFFFFFFFFu) { break; }
        if (local_v >= kMaxBlasVisits) { *heat_acc += local_v; return 2u; }
        local_v += 1u;
        let node = blas_nodes[idx];
        if (!aabb_hit(node, l_origin, inv_dir, tmax)) {
            idx = node.miss_next; continue;
        }
        if (node.tri_count > 0u) {
            for (var i = node.tri_offset; i < node.tri_offset + node.tri_count; i += 1u) {
                if (blas_ray_tri_hit(l_origin, l_dir, blas_tris[i], tmax)) {
                    *heat_acc += local_v; return 1u;
                }
            }
            idx = node.miss_next;
        } else {
            idx = node.hit_next;
        }
    }
    *heat_acc += local_v;
    return 0u;
}

// ---- Dynamic LBVH traversal (view space, stack-based, GPU LBVH format) ----
// Checks the skinned-mesh GPU LBVH.  The BVH and triangles are in view space so
// no coordinate transform is needed — origin_view/dir_view are used directly.
// Skipped immediately when cam.dyn_node_count == 0 (no dynamic geometry this frame).
fn aabb_hit_lbvh(node: LbvhNode, origin: vec3<f32>, inv_dir: vec3<f32>, tmax: f32) -> bool {
    let t1 = (node.bounds_min - origin) * inv_dir;
    let t2 = (node.bounds_max - origin) * inv_dir;
    let tmin_v = min(t1, t2);
    let tmax_v = max(t1, t2);
    let tmin_s = max(max(tmin_v.x, tmin_v.y), max(tmin_v.z, 0.0));
    let tmax_s = min(min(tmax_v.x, tmax_v.y), min(tmax_v.z, tmax));
    return tmin_s <= tmax_s;
}

fn dyn_ray_tri_hit(origin: vec3<f32>, dir: vec3<f32>, tri: DynBlasTri, tmax: f32) -> bool {
    let e1 = tri.b - tri.a;
    let e2 = tri.c - tri.a;
    let h   = cross(dir, e2);
    let det = dot(e1, h);
    if (abs(det) < 1e-7) { return false; }
    let inv_det = 1.0 / det;
    let sv = origin - tri.a;
    let u  = dot(sv, h) * inv_det;
    if (u < 0.0 || u > 1.0) { return false; }
    let q = cross(sv, e1);
    let v = dot(dir, q) * inv_det;
    if (v < 0.0 || u + v > 1.0) { return false; }
    let t = dot(e2, q) * inv_det;
    return (t > 1e-4 && t < tmax);
}

fn traverse_dyn_lbvh(origin: vec3<f32>, dir: vec3<f32>, tmax: f32,
                      heat_acc: ptr<function, u32>) -> u32 {
    if (cam.dyn_node_count == 0u) { return 0u; }
    let inv_dir = vec3<f32>(1.0/dir.x, 1.0/dir.y, 1.0/dir.z);
    var stk: array<u32, 64>;
    var sp: i32 = 1; stk[0] = 0u;
    var visits: u32 = 0u;
    loop {
        if (sp <= 0 || visits >= kMaxBlasVisits) { break; }
        sp -= 1; visits += 1u;
        let idx  = stk[u32(sp)];
        let node = dyn_blas_nodes[idx];
        if (!aabb_hit_lbvh(node, origin, inv_dir, tmax)) { continue; }
        if (node.tri_count > 0u) {
            for (var i = node.tri_offset; i < node.tri_offset + node.tri_count; i += 1u) {
                if (dyn_ray_tri_hit(origin, dir, dyn_blas_tris[i], tmax)) {
                    *heat_acc += visits; return 1u;
                }
            }
        } else {
            if (sp < 62) {
                stk[u32(sp)] = node.right_child; sp += 1;
                stk[u32(sp)] = node.left_child;  sp += 1;
            }
        }
    }
    *heat_acc += visits;
    return select(0u, 2u, visits >= kMaxBlasVisits);
}

// ---- Two-level TLAS traversal (world space at TLAS, view space at BLAS) ----
// origin_view/dir_view are in view space (from the depth unproject + hemisphere sample).
// The TLAS is in world space; ray is converted once at the top.
// BLAS leaves still receive the original view-space ray via pnMtxInv (view→local).
// After the TLAS, the dynamic GPU LBVH (skinned geo, view space) is checked as a second pass.
// Returns 0=miss, 1=hit, 2=visit-limit reached.
fn intersects_any(origin_view: vec3<f32>, dir_view: vec3<f32>, tmax: f32,
                  acc_visits: ptr<function, u32>) -> u32 {
    if (arrayLength(&tlas_nodes) > 0u && arrayLength(&tlas_instances) > 0u) {
        let origin  = xf_vtw_point(origin_view);
        let dir     = xf_vtw_dir(dir_view);
        let inv_dir = vec3<f32>(1.0/dir.x, 1.0/dir.y, 1.0/dir.z);
        var idx: u32 = 0u;
        var tlas_v: u32 = 0u;
        loop {
            if (idx == 0xFFFFFFFFu) { break; }
            if (tlas_v >= kMaxTlasVisits) { *acc_visits += tlas_v; return 2u; }
            tlas_v += 1u;
            let node = tlas_nodes[idx];
            if (!aabb_hit(node, origin, inv_dir, tmax)) {
                idx = node.miss_next; continue;
            }
            if (node.tri_count == 1u) {  // TLAS leaf
                let inst = tlas_instances[node.tri_offset];
                let l_origin = xf_point(inst, origin_view);
                let l_dir    = xf_dir(inst, dir_view);
                let r = traverse_blas(inst, l_origin, l_dir, tmax, acc_visits);
                if (r == 1u) { *acc_visits += tlas_v; return 1u; }
                if (r == 2u) { *acc_visits += tlas_v; return 2u; }
                idx = node.miss_next;
            } else {
                idx = node.hit_next;
            }
        }
        *acc_visits += tlas_v;
    }
    // Second pass: dynamic (skinned) geometry via GPU LBVH in view space.
    return traverse_dyn_lbvh(origin_view, dir_view, tmax, acc_visits);
}

// ---- Depth / unproject / normal (identical to LBVH shader) ----
fn load_depth(px: u32, py: u32) -> f32 { return -textureLoad(t_depth, vec2<i32>(i32(px), i32(py)), 0); }

fn unproject(px: u32, py: u32) -> vec4<f32> {
    let d = load_depth(px, py);
    let ndc_x = (f32(px) + 0.5) / f32(cam.screen_width)  *  2.0 - 1.0;
    let ndc_y = (f32(py) + 0.5) / f32(cam.screen_height) * -2.0 + 1.0;
    let ndc_h = vec4<f32>(ndc_x, ndc_y, d, 1.0);
    let world_h = ndc_h * cam.inv_view_proj;
    if (abs(world_h.w) < 1e-6) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    return vec4<f32>(world_h.xyz / world_h.w, 1.0);
}

fn estimate_normal(px: u32, py: u32, pos: vec3<f32>) -> vec3<f32> {
    let W = cam.screen_width; let H = cam.screen_height;
    let d0     = length(pos);
    let thresh = max(d0 * 0.15, 0.1);

    let xp = unproject(select(px + 1u, px - 1u, px + 1u >= W), py);
    let xn = unproject(select(px - 1u, px + 1u, px == 0u),     py);
    let yp = unproject(px, select(py + 1u, py - 1u, py + 1u >= H));
    let yn = unproject(px, select(py - 1u, py + 1u, py == 0u));

    let xp_ok = xp.w > 0.5 && abs(length(xp.xyz) - d0) < thresh && (px + 1u < W);
    let xn_ok = xn.w > 0.5 && abs(length(xn.xyz) - d0) < thresh && (px > 0u);
    let yp_ok = yp.w > 0.5 && abs(length(yp.xyz) - d0) < thresh && (py + 1u < H);
    let yn_ok = yn.w > 0.5 && abs(length(yn.xyz) - d0) < thresh && (py > 0u);

    var tx: vec3<f32>;
    if      (xp_ok) { tx = xp.xyz - pos; }
    else if (xn_ok) { tx = pos - xn.xyz; }
    else            { return normalize(-pos); }

    var ty: vec3<f32>;
    if      (yp_ok) { ty = yp.xyz - pos; }
    else if (yn_ok) { ty = pos - yn.xyz; }
    else            { return normalize(-pos); }

    let n  = cross(tx, ty);
    let nl = length(n);
    if (nl < 1e-6) { return normalize(-pos); }
    var nn = n / nl;
    if (dot(nn, normalize(-pos)) < 0.0) { nn = -nn; }
    return nn;
}
)";
    s += R"(
// ---- Main compute entry point ----
// Dispatched at half resolution; each thread handles a 2×2 pixel block.
// One depth/normal sample is shared across the block (same as before), but
// the N-ray budget is split evenly among the four sub-pixels with independent
// seeds so neighbouring pixels draw distinct sample sequences.
// Distance LOD reduces the ray count for far geometry.
@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x * 2u;
    let py = gid.y * 2u;
    if (px >= cam.screen_width || py >= cam.screen_height) { return; }

    let W = i32(cam.screen_width); let H = i32(cam.screen_height);
    let c00 = vec2<i32>(i32(px),   i32(py));
    let c10 = vec2<i32>(i32(px)+1, i32(py));
    let c01 = vec2<i32>(i32(px),   i32(py)+1);
    let c11 = vec2<i32>(i32(px)+1, i32(py)+1);

    let d = textureLoad(t_depth, vec2<i32>(i32(px), i32(py)), 0);
    if (d <= 0.0001) {
        let sky = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        let blk = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        textureStore(ao_out,    c00, sky); textureStore(ao_out,    c10, sky);
        textureStore(ao_out,    c01, sky); textureStore(ao_out,    c11, sky);
        textureStore(limits_out, c00, blk); textureStore(limits_out, c10, blk);
        textureStore(limits_out, c01, blk); textureStore(limits_out, c11, blk);
        return;
    }

    let pos_h = unproject(px, py);
    if (pos_h.w < 0.5) {
        let bad = vec4<f32>(0.0, 0.0, 1.0, 1.0);
        textureStore(ao_out,    c00, bad); textureStore(ao_out,    c10, bad);
        textureStore(ao_out,    c01, bad); textureStore(ao_out,    c11, bad);
        textureStore(limits_out, c00, bad); textureStore(limits_out, c10, bad);
        textureStore(limits_out, c01, bad); textureStore(limits_out, c11, bad);
        return;
    }
    let pos    = pos_h.xyz;
    let normal = estimate_normal(px, py, pos);

    let dist_from_cam = length(pos);
    let ray_origin = pos + normal * max(dist_from_cam * cam.normal_bias, cam.normal_bias * 10.0);

    // Distance LOD: far surfaces get fewer rays; the A-trous denoiser covers residual noise.
    let N = cam.rays_per_pixel;
    let lod_n = select(N,
                    select(max(1u, N / 2u), 1u, dist_from_cam > cam.max_distance * 8.0),
                    dist_from_cam > cam.max_distance * 4.0);

    // Split the budget across the four sub-pixels (same total GPU work as before).
    // Each sub-pixel uses a seed seeded from its own screen coordinate so adjacent
    // pixels draw independent cosine-hemisphere sequences every frame.
    let n_each = max(1u, lod_n / 4u);

    var seed00 = pcg(((py        * cam.screen_width + px)        * 1664525u) ^ cam.frame_seed);
    var hits00: u32 = 0u; var lim00: u32 = 0u; var vis00: u32 = 0u;
    for (var i = 0u; i < n_each; i += 1u) {
        let r = intersects_any(ray_origin, cosine_hemisphere(normal, &seed00), cam.max_distance, &vis00);
        if (r == 1u) { hits00 += 1u; } if (r == 2u) { lim00 += 1u; }
    }

    var seed10 = pcg(((py        * cam.screen_width + px + 1u)   * 1664525u) ^ cam.frame_seed);
    var hits10: u32 = 0u; var lim10: u32 = 0u; var vis10: u32 = 0u;
    for (var i = 0u; i < n_each; i += 1u) {
        let r = intersects_any(ray_origin, cosine_hemisphere(normal, &seed10), cam.max_distance, &vis10);
        if (r == 1u) { hits10 += 1u; } if (r == 2u) { lim10 += 1u; }
    }

    var seed01 = pcg((((py + 1u) * cam.screen_width + px)        * 1664525u) ^ cam.frame_seed);
    var hits01: u32 = 0u; var lim01: u32 = 0u; var vis01: u32 = 0u;
    for (var i = 0u; i < n_each; i += 1u) {
        let r = intersects_any(ray_origin, cosine_hemisphere(normal, &seed01), cam.max_distance, &vis01);
        if (r == 1u) { hits01 += 1u; } if (r == 2u) { lim01 += 1u; }
    }

    var seed11 = pcg((((py + 1u) * cam.screen_width + px + 1u)   * 1664525u) ^ cam.frame_seed);
    var hits11: u32 = 0u; var lim11: u32 = 0u; var vis11: u32 = 0u;
    for (var i = 0u; i < n_each; i += 1u) {
        let r = intersects_any(ray_origin, cosine_hemisphere(normal, &seed11), cam.max_distance, &vis11);
        if (r == 1u) { hits11 += 1u; } if (r == 2u) { lim11 += 1u; }
    }

    let valid00 = n_each - lim00; let ao00 = select(0.5, 1.0 - f32(hits00) / f32(valid00), valid00 > 0u);
    let valid10 = n_each - lim10; let ao10 = select(0.5, 1.0 - f32(hits10) / f32(valid10), valid10 > 0u);
    let valid01 = n_each - lim01; let ao01 = select(0.5, 1.0 - f32(hits01) / f32(valid01), valid01 > 0u);
    let valid11 = n_each - lim11; let ao11 = select(0.5, 1.0 - f32(hits11) / f32(valid11), valid11 > 0u);

    // Aggregate stats for debug views (based on c00 sub-pixel as representative).
    let total_visits = vis00 + vis10 + vis01 + vis11;
    let total_lim    = lim00 + lim10 + lim01 + lim11;
    let total_n      = n_each * 4u;

    let root = tlas_nodes[0];
    let root_hit = aabb_hit(root, xf_vtw_point(pos), vec3<f32>(1.0/0.1, 1.0/0.1, 1.0/(-1.0)), cam.max_distance);

    // Debug/non-AO modes produce one value shared across the block; AO mode writes
    // the unique per-sub-pixel estimate so the denoiser sees four distinct samples.
    var ao_val00: vec4<f32>; var ao_val10: vec4<f32>;
    var ao_val01: vec4<f32>; var ao_val11: vec4<f32>;
    if (cam.debug_mode == 1u) {
        let v = vec4<f32>(normal * 0.5 + 0.5, 1.0);
        ao_val00 = v; ao_val10 = v; ao_val01 = v; ao_val11 = v;
    } else if (cam.debug_mode == 2u) {
        let v = vec4<f32>(dist_from_cam / 2000.0, 0.0, 0.0, 1.0);
        ao_val00 = v; ao_val10 = v; ao_val01 = v; ao_val11 = v;
    } else if (cam.debug_mode == 3u) {
        let v = select(vec4<f32>(1.0,0.0,0.0,1.0), vec4<f32>(0.0,1.0,0.0,1.0), root_hit);
        ao_val00 = v; ao_val10 = v; ao_val01 = v; ao_val11 = v;
    } else {
        ao_val00 = vec4<f32>(ao00, ao00, ao00, 1.0);
        ao_val10 = vec4<f32>(ao10, ao10, ao10, 1.0);
        ao_val01 = vec4<f32>(ao01, ao01, ao01, 1.0);
        ao_val11 = vec4<f32>(ao11, ao11, ao11, 1.0);
    }

    var lim_val: vec4<f32>;
    if (cam.debug_mode2 == 1u)      { lim_val = vec4<f32>(ao00, ao00, ao00, 1.0); }
    else if (cam.debug_mode2 == 2u) { lim_val = vec4<f32>(normal * 0.5 + 0.5, 1.0); }
    else if (cam.debug_mode2 == 3u) { lim_val = vec4<f32>(dist_from_cam / 2000.0, 0.0, 0.0, 1.0); }
    else if (cam.debug_mode2 == 4u) { lim_val = select(vec4<f32>(1.0,0.0,0.0,1.0), vec4<f32>(0.0,1.0,0.0,1.0), root_hit); }
    else if (cam.debug_mode2 == 5u) { lim_val = vec4<f32>(min(f32(total_visits) / f32(total_n * 256u), 1.0), 0.0, 0.0, 1.0); }
    else if (cam.debug_mode2 == 6u) { lim_val = vec4<f32>(f32(total_lim)/f32(total_n), 0.0, f32(total_lim)/f32(total_n), 1.0); }
    else                            { lim_val = select(vec4<f32>(0.0,0.0,0.0,1.0), vec4<f32>(1.0,0.0,1.0,1.0), total_lim > total_n / 2u); }

    textureStore(ao_out, c00, ao_val00);
    if (c10.x < W) { textureStore(ao_out, c10, ao_val10); }
    if (c01.y < H) { textureStore(ao_out, c01, ao_val01); }
    if (c10.x < W && c01.y < H) { textureStore(ao_out, c11, ao_val11); }
    textureStore(limits_out, c00, lim_val);
    if (c10.x < W) { textureStore(limits_out, c10, lim_val); }
    if (c01.y < H) { textureStore(limits_out, c01, lim_val); }
    if (c10.x < W && c01.y < H) { textureStore(limits_out, c11, lim_val); }
}
)";
    return s;
}

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
    if (m_dynDummyNodeBuf) wgpuBufferRelease(m_dynDummyNodeBuf);
    if (m_dynDummyTriBuf)  wgpuBufferRelease(m_dynDummyTriBuf);
    if (m_tlasBindGroup)  wgpuBindGroupRelease(m_tlasBindGroup);
    if (m_tlasBgl)        wgpuBindGroupLayoutRelease(m_tlasBgl);
    if (m_tlasPipeline)   wgpuComputePipelineRelease(m_tlasPipeline);
    if (m_bindGroup)      wgpuBindGroupRelease(m_bindGroup);
    if (m_depthView)      wgpuTextureViewRelease(m_depthView);
    if (m_limView)        wgpuTextureViewRelease(m_limView);
    if (m_limTex)         wgpuTextureRelease(m_limTex);
    if (m_aoView)         wgpuTextureViewRelease(m_aoView);
    if (m_aoTex)          wgpuTextureRelease(m_aoTex);
    if (m_fallbackView)   wgpuTextureViewRelease(m_fallbackView);
    if (m_fallbackTex)    wgpuTextureRelease(m_fallbackTex);
    if (m_alphaSampler)   wgpuSamplerRelease(m_alphaSampler);
    if (m_cameraUbo)      wgpuBufferRelease(m_cameraUbo);
    if (m_bgl)            wgpuBindGroupLayoutRelease(m_bgl);
    if (m_pipeline)       wgpuComputePipelineRelease(m_pipeline);
}

void AoPass::ensure_pipeline(WGPUDevice device) {
    if (m_pipeline) return;

    const std::string shaderSrc = build_ao_shader();
    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code        = {shaderSrc.c_str(), WGPU_STRLEN};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslSrc.chain;
    WGPUShaderModule sm = wgpuDeviceCreateShaderModule(device, &smDesc);

    // Bind group layout: 6 fixed + 1 sampler + 16 alpha textures = 23 entries
    WGPUBindGroupLayoutEntry entries[23] = {};

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

    // Binding 6: filtering sampler for alpha texture lookup
    entries[6].binding       = 6;
    entries[6].visibility    = WGPUShaderStage_Compute;
    entries[6].sampler.type  = WGPUSamplerBindingType_Filtering;

    // Bindings 7-22: 16 alpha-tested texture slots (float sampling)
    for (int i = 0; i < 16; ++i) {
        entries[7 + i].binding               = 7 + i;
        entries[7 + i].visibility            = WGPUShaderStage_Compute;
        entries[7 + i].texture.sampleType    = WGPUTextureSampleType_Float;
        entries[7 + i].texture.viewDimension = WGPUTextureViewDimension_2D;
        entries[7 + i].texture.multisampled  = 0;
    }

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 23;
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

void AoPass::ensure_tlas_pipeline(WGPUDevice device) {
    if (m_tlasPipeline) return;

    const std::string src = build_tlas_ao_shader();
    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code        = {src.c_str(), WGPU_STRLEN};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslSrc.chain;
    WGPUShaderModule sm = wgpuDeviceCreateShaderModule(device, &smDesc);

    // 9 fixed bindings (0-8) + 1 sampler (6) + 16 alpha textures (9-24)
    // + 2 dynamic LBVH buffers (25-26) = 27 total
    WGPUBindGroupLayoutEntry entries[27] = {};

    // 0: TLAS nodes
    entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    // 1: TLAS instances
    entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    // 2: Camera UBO
    entries[2].binding = 2; entries[2].visibility = WGPUShaderStage_Compute;
    entries[2].buffer.type = WGPUBufferBindingType_Uniform;
    // 3: Depth texture
    entries[3].binding = 3; entries[3].visibility = WGPUShaderStage_Compute;
    entries[3].texture.sampleType    = WGPUTextureSampleType_Depth;
    entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;
    // 4: AO output
    entries[4].binding = 4; entries[4].visibility = WGPUShaderStage_Compute;
    entries[4].storageTexture.access        = WGPUStorageTextureAccess_WriteOnly;
    entries[4].storageTexture.format        = WGPUTextureFormat_RGBA8Unorm;
    entries[4].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
    // 5: Limits output
    entries[5].binding = 5; entries[5].visibility = WGPUShaderStage_Compute;
    entries[5].storageTexture.access        = WGPUStorageTextureAccess_WriteOnly;
    entries[5].storageTexture.format        = WGPUTextureFormat_RGBA8Unorm;
    entries[5].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
    // 6: Sampler
    entries[6].binding = 6; entries[6].visibility = WGPUShaderStage_Compute;
    entries[6].sampler.type = WGPUSamplerBindingType_Filtering;
    // 7: Monolithic BLAS nodes (static geometry only)
    entries[7].binding = 7; entries[7].visibility = WGPUShaderStage_Compute;
    entries[7].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    // 8: Monolithic BLAS tris (static geometry only)
    entries[8].binding = 8; entries[8].visibility = WGPUShaderStage_Compute;
    entries[8].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    // 9-24: Alpha textures
    for (int i = 0; i < 16; ++i) {
        entries[9 + i].binding               = 9 + i;
        entries[9 + i].visibility            = WGPUShaderStage_Compute;
        entries[9 + i].texture.sampleType    = WGPUTextureSampleType_Float;
        entries[9 + i].texture.viewDimension = WGPUTextureViewDimension_2D;
    }
    // 25: Dynamic LBVH nodes (skinned geo, GPU LBVH format — empty dummy when no dyn geo)
    entries[25].binding = 25; entries[25].visibility = WGPUShaderStage_Compute;
    entries[25].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    // 26: Dynamic LBVH tris (GpuTriangle 80-byte format, positions only used)
    entries[26].binding = 26; entries[26].visibility = WGPUShaderStage_Compute;
    entries[26].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 27;
    bglDesc.entries    = entries;
    m_tlasBgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts     = &m_tlasBgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout             = pl;
    cpDesc.compute.module     = sm;
    cpDesc.compute.entryPoint = {"cs_main", WGPU_STRLEN};
    m_tlasPipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);

    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(sm);
}

void AoPass::rebuild_tlas_bind_group(WGPUDevice device) {
    if (m_tlasBindGroup) { wgpuBindGroupRelease(m_tlasBindGroup); m_tlasBindGroup = nullptr; }

    // Create 4-byte dummy storage buffers for dynamic LBVH when no skinned geometry exists.
    // The shader guards against access via cam.dyn_node_count == 0.
    if (!m_dynDummyNodeBuf) {
        WGPUBufferDescriptor d{}; d.size = 4;
        d.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        m_dynDummyNodeBuf = wgpuDeviceCreateBuffer(device, &d);
    }
    if (!m_dynDummyTriBuf) {
        WGPUBufferDescriptor d{}; d.size = 4;
        d.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        m_dynDummyTriBuf = wgpuDeviceCreateBuffer(device, &d);
    }

    WGPUBuffer dynNodeBuf = m_tlasLastDynNodeBuf ? m_tlasLastDynNodeBuf : m_dynDummyNodeBuf;
    WGPUBuffer dynTriBuf  = m_tlasLastDynTriBuf  ? m_tlasLastDynTriBuf  : m_dynDummyTriBuf;

    WGPUBindGroupEntry entries[27] = {};
    entries[0].binding = 0; entries[0].buffer = m_tlasLastNodeBuf;     entries[0].size = WGPU_WHOLE_SIZE;
    entries[1].binding = 1; entries[1].buffer = m_tlasLastInstBuf;     entries[1].size = WGPU_WHOLE_SIZE;
    entries[2].binding = 2; entries[2].buffer = m_cameraUbo;           entries[2].size = sizeof(GpuCamera);
    entries[3].binding = 3; entries[3].textureView = m_depthView;
    entries[4].binding = 4; entries[4].textureView = m_aoView;
    entries[5].binding = 5; entries[5].textureView = m_limView;
    entries[6].binding = 6; entries[6].sampler = m_alphaSampler;
    entries[7].binding = 7; entries[7].buffer = m_tlasLastBlasNodeBuf; entries[7].size = WGPU_WHOLE_SIZE;
    entries[8].binding = 8; entries[8].buffer = m_tlasLastBlasTriBuf;  entries[8].size = WGPU_WHOLE_SIZE;
    for (int i = 0; i < 16; ++i) {
        entries[9 + i].binding = 9 + i;
        entries[9 + i].textureView =
            (i < static_cast<int>(m_lastTexViews.size()))
            ? static_cast<WGPUTextureView>(m_lastTexViews[i])
            : m_fallbackView;
    }
    entries[25].binding = 25; entries[25].buffer = dynNodeBuf; entries[25].size = WGPU_WHOLE_SIZE;
    entries[26].binding = 26; entries[26].buffer = dynTriBuf;  entries[26].size = WGPU_WHOLE_SIZE;

    WGPUBindGroupDescriptor bgd{};
    bgd.layout     = m_tlasBgl;
    bgd.entryCount = 27;
    bgd.entries    = entries;
    m_tlasBindGroup      = wgpuDeviceCreateBindGroup(device, &bgd);
    m_tlasBindGroupDirty = false;
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
    m_bindGroupDirty     = true;
    m_tlasBindGroupDirty = true;
}

void AoPass::rebuild_bind_group(WGPUDevice device) {
    if (m_bindGroup) { wgpuBindGroupRelease(m_bindGroup); m_bindGroup = nullptr; }

    WGPUBindGroupEntry entries[23] = {};
    entries[0].binding = 0; entries[0].buffer = m_lastNodeBuf; entries[0].size = WGPU_WHOLE_SIZE;
    entries[1].binding = 1; entries[1].buffer = m_lastTriBuf;  entries[1].size = WGPU_WHOLE_SIZE;
    entries[2].binding = 2; entries[2].buffer = m_cameraUbo;   entries[2].size = sizeof(GpuCamera);
    entries[3].binding = 3; entries[3].textureView = m_depthView;
    entries[4].binding = 4; entries[4].textureView = m_aoView;
    entries[5].binding = 5; entries[5].textureView = m_limView;
    entries[6].binding = 6; entries[6].sampler = m_alphaSampler;
    // Bindings 7-22: real alpha textures, empty slots filled with the 1×1 white fallback.
    for (int i = 0; i < 16; ++i) {
        entries[7 + i].binding = 7 + i;
        entries[7 + i].textureView =
            (i < static_cast<int>(m_lastTexViews.size()))
            ? static_cast<WGPUTextureView>(m_lastTexViews[i])
            : m_fallbackView;
    }

    WGPUBindGroupDescriptor bgd{};
    bgd.layout     = m_bgl;
    bgd.entryCount = 23;
    bgd.entries    = entries;
    m_bindGroup      = wgpuDeviceCreateBindGroup(device, &bgd);
    m_bindGroupDirty = false;
}

void AoPass::execute(WGPUDevice device, WGPUCommandEncoder encoder,
                     WGPUTexture depthTex,
                     const GeometryCollector::CameraData& cam,
                     WGPUBuffer nodeBuf, WGPUBuffer triBuf,
                     const std::vector<void*>& texViews) {
    if (!cam.valid || !depthTex || !nodeBuf || !triBuf) return;

    // Invalidate bind group when the BVH buffers change frame-to-frame
    if (nodeBuf != m_lastNodeBuf || triBuf != m_lastTriBuf) {
        m_lastNodeBuf    = nodeBuf;
        m_lastTriBuf     = triBuf;
        m_bindGroupDirty = true;
    }

    // Invalidate bind group when the alpha-texture set changes
    if (texViews != m_lastTexViews) {
        m_lastTexViews   = texViews;
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
    gpuCam.camWorldX = 0.f;
    gpuCam.camWorldY = 0.f;
    gpuCam.camWorldZ = 0.f;

    if (!m_cameraUbo) {
        m_cameraUbo = create_buffer(device, sizeof(GpuCamera), WGPUBufferUsage_Uniform);
        m_bindGroupDirty = true;
    }

    // Create alpha-texture sampler and 1×1 opaque-white fallback texture (once).
    if (!m_alphaSampler) {
        WGPUSamplerDescriptor sd{};
        sd.addressModeU  = WGPUAddressMode_Repeat;
        sd.addressModeV  = WGPUAddressMode_Repeat;
        sd.minFilter     = WGPUFilterMode_Linear;
        sd.magFilter     = WGPUFilterMode_Linear;
        sd.mipmapFilter  = WGPUMipmapFilterMode_Nearest;
        sd.maxAnisotropy = 1;
        m_alphaSampler = wgpuDeviceCreateSampler(device, &sd);
        m_bindGroupDirty = true;
    }
    if (!m_fallbackTex) {
        WGPUTextureDescriptor ftd{};
        ftd.usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        ftd.dimension     = WGPUTextureDimension_2D;
        ftd.size          = {1, 1, 1};
        ftd.format        = WGPUTextureFormat_RGBA8Unorm;
        ftd.mipLevelCount = 1;
        ftd.sampleCount   = 1;
        m_fallbackTex  = wgpuDeviceCreateTexture(device, &ftd);
        m_fallbackView = wgpuTextureCreateView(m_fallbackTex, nullptr);
        // 1×1 fully opaque white — sample_alpha returns 1.0 (opaque) for empty slots
        const uint8_t white[4] = {255, 255, 255, 255};
        WGPUQueue fq = wgpuDeviceGetQueue(device);
        WGPUTexelCopyTextureInfo ict{};
        ict.texture  = m_fallbackTex;
        WGPUTexelCopyBufferLayout tdl{};
        tdl.bytesPerRow  = 4;
        tdl.rowsPerImage = 1;
        WGPUExtent3D ext{1, 1, 1};
        wgpuQueueWriteTexture(fq, &ict, white, 4, &tdl, &ext);
        wgpuQueueRelease(fq);
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

void AoPass::execute_tlas(WGPUDevice device, WGPUCommandEncoder encoder,
                          WGPUTexture depthTex,
                          const GeometryCollector::CameraData& cam,
                          WGPUBuffer tlasNodeBuf, WGPUBuffer instanceBuf,
                          WGPUBuffer blasNodeBuf, WGPUBuffer blasTriBuf,
                          WGPUBuffer dynNodeBuf, WGPUBuffer dynTriBuf, uint32_t dynNodeCount,
                          const std::vector<void*>& texViews) {
    if (!cam.valid || !depthTex) return;
    if (!tlasNodeBuf || !instanceBuf || !blasNodeBuf || !blasTriBuf) return;

    if (tlasNodeBuf != m_tlasLastNodeBuf || instanceBuf != m_tlasLastInstBuf ||
        blasNodeBuf != m_tlasLastBlasNodeBuf || blasTriBuf != m_tlasLastBlasTriBuf ||
        dynNodeBuf  != m_tlasLastDynNodeBuf  || dynTriBuf  != m_tlasLastDynTriBuf) {
        m_tlasLastNodeBuf     = tlasNodeBuf;
        m_tlasLastInstBuf     = instanceBuf;
        m_tlasLastBlasNodeBuf = blasNodeBuf;
        m_tlasLastBlasTriBuf  = blasTriBuf;
        m_tlasLastDynNodeBuf  = dynNodeBuf;
        m_tlasLastDynTriBuf   = dynTriBuf;
        m_tlasBindGroupDirty  = true;
    }
    if (texViews != m_lastTexViews) {
        m_lastTexViews       = texViews;
        m_tlasBindGroupDirty = true;
        m_bindGroupDirty     = true; // also invalidate LBVH bind group
    }

    const uint32_t w = wgpuTextureGetWidth(depthTex);
    const uint32_t h = wgpuTextureGetHeight(depthTex);
    if (w == 0 || h == 0) return;

    ensure_tlas_pipeline(device);
    if (!m_tlasPipeline) return;

    if (w != m_width || h != m_height) rebuild_output(device, w, h);
    if (depthTex != m_lastDepthTex)    rebuild_depth_binding(device, depthTex);

    // Upload camera UBO (shared with LBVH path).
    GpuCamera gpuCam{};
    gpuCam.screenWidth  = w;
    gpuCam.screenHeight = h;
    gpuCam.raysPerPixel = m_params.raysPerPixel;
    gpuCam.frameSeed    = m_frame++;
    gpuCam.maxDistance  = m_params.maxDistance;
    gpuCam.normalBias   = m_params.normalBias;
    gpuCam.debugMode    = m_params.debugMode;
    gpuCam.debugMode2   = m_params.debugMode2;
    if (!compute_inv_proj(cam, gpuCam.invViewProj)) return;
    // Default viewToWorld to identity; overwrite if view matrix is populated.
    gpuCam.viewToWorld[0]=1; gpuCam.viewToWorld[1]=0; gpuCam.viewToWorld[2]=0;  gpuCam.viewToWorld[3]=0;
    gpuCam.viewToWorld[4]=0; gpuCam.viewToWorld[5]=1; gpuCam.viewToWorld[6]=0;  gpuCam.viewToWorld[7]=0;
    gpuCam.viewToWorld[8]=0; gpuCam.viewToWorld[9]=0; gpuCam.viewToWorld[10]=1; gpuCam.viewToWorld[11]=0;
    compute_view_to_world(cam, gpuCam.viewToWorld);
    gpuCam.dynNodeCount = dynNodeCount;

    if (!m_cameraUbo) {
        m_cameraUbo = create_buffer(device, sizeof(GpuCamera), WGPUBufferUsage_Uniform);
        m_tlasBindGroupDirty = true;
        m_bindGroupDirty     = true;
    }
    if (!m_alphaSampler) {
        WGPUSamplerDescriptor sd{};
        sd.addressModeU = WGPUAddressMode_Repeat; sd.addressModeV = WGPUAddressMode_Repeat;
        sd.minFilter = WGPUFilterMode_Linear;     sd.magFilter = WGPUFilterMode_Linear;
        sd.mipmapFilter = WGPUMipmapFilterMode_Nearest; sd.maxAnisotropy = 1;
        m_alphaSampler = wgpuDeviceCreateSampler(device, &sd);
        m_tlasBindGroupDirty = true;
        m_bindGroupDirty     = true;
    }
    if (!m_fallbackTex) {
        WGPUTextureDescriptor ftd{};
        ftd.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        ftd.dimension = WGPUTextureDimension_2D; ftd.size = {1,1,1};
        ftd.format = WGPUTextureFormat_RGBA8Unorm; ftd.mipLevelCount = 1; ftd.sampleCount = 1;
        m_fallbackTex  = wgpuDeviceCreateTexture(device, &ftd);
        m_fallbackView = wgpuTextureCreateView(m_fallbackTex, nullptr);
        const uint8_t white[4] = {255,255,255,255};
        WGPUQueue fq = wgpuDeviceGetQueue(device);
        WGPUTexelCopyTextureInfo ict{}; ict.texture = m_fallbackTex;
        WGPUTexelCopyBufferLayout tdl{}; tdl.bytesPerRow = 4; tdl.rowsPerImage = 1;
        WGPUExtent3D ext{1,1,1};
        wgpuQueueWriteTexture(fq, &ict, white, 4, &tdl, &ext);
        wgpuQueueRelease(fq);
        m_tlasBindGroupDirty = true;
        m_bindGroupDirty     = true;
    }

    WGPUQueue q = wgpuDeviceGetQueue(device);
    wgpuQueueWriteBuffer(q, m_cameraUbo, 0, &gpuCam, sizeof(GpuCamera));
    wgpuQueueRelease(q);

    if (m_tlasBindGroupDirty) rebuild_tlas_bind_group(device);

    WGPUComputePassDescriptor passDesc{};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
    wgpuComputePassEncoderSetPipeline(pass, m_tlasPipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, m_tlasBindGroup, 0, nullptr);
    const uint32_t gx = (w / 2 + 7) / 8;
    const uint32_t gy = (h / 2 + 7) / 8;
    wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

ImTextureID AoPass::imgui_texture_id()    const { return reinterpret_cast<ImTextureID>(m_aoView);  }
ImTextureID AoPass::limits_texture_id()  const { return reinterpret_cast<ImTextureID>(m_limView); }

} // namespace dusk::rtao
