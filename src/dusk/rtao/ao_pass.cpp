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
    uint32_t hitNext;      // offset 12
    float    boundsMax[3]; // offset 16
    uint32_t missNext;     // offset 28
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

// 96 bytes — matches WGSL Camera uniform
struct GpuCamera {
    float    invViewProj[16]; // col-major mat4x4, offset 0
    uint32_t screenWidth;     // offset 64
    uint32_t screenHeight;    // offset 68
    uint32_t raysPerPixel;    // offset 72
    uint32_t frameSeed;       // offset 76
    float    maxDistance;     // offset 80
    float    normalBias;      // offset 84 — view-space units to push ray origin off surface
    uint32_t _pad[2];         // offset 88 → total 96
};
static_assert(sizeof(GpuCamera) == 96);

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

// Build the inv(proj) uniform for the WGSL mat4x4.
//
// The BVH triangles are in GX view/camera space (after pnMtx), so the unproject
// must yield view-space positions → only inv(P) is needed, not inv(view*proj).
//
// geometry_capture.h: projMtx[4][4] is the GX projection matrix stored row-major
// (projMtx[row][col]).  Reading it as cam.proj[c][r] with c treated as the first
// index recovers P_GX transposed into P_flat.  mat4_invert_flat then gives
// inv(P_GX^T) = inv(P_GX)^T.  The final store into out_col_major (column-major
// for WGSL) transposes again, so WGSL computes v * M_wgsl = inv(P_GX)*v. ✓
//
// GX NDC Z range: near=-1, far=0 (OpenGL-style).  Aurora reversed-Z stores
// depth = -z_ndc_gx, so load_depth returns z_ndc_gx = -raw ∈ [-1, 0].
static bool compute_inv_proj(const GeometryCollector::CameraData& cam,
                              float out_col_major[16]) {
    // Recover P in row-major [r][c] from Aurora's column-major [c][r].
    float P_flat[16];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            P_flat[r*4 + c] = cam.proj[c][r];

    float invP_rm[16];
    if (!mat4_invert_flat(P_flat, invP_rm)) return false;

    // Transpose to column-major for WGSL mat4x4
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out_col_major[c*4 + r] = invP_rm[r*4 + c];

    return true;
}

// ---------------------------------------------------------------------------
// WGSL compute shader
// ---------------------------------------------------------------------------

static const char kShader[] = R"(
// ---- GPU-side BVH structs (must match GpuBvhNode / GpuTriangle on CPU) ----

struct BvhNode {
    bounds_min : vec3<f32>,  // offset 0
    hit_next   : u32,        // offset 12
    bounds_max : vec3<f32>,  // offset 16
    miss_next  : u32,        // offset 28
    tri_offset : u32,        // offset 32
    tri_count  : u32,        // offset 36
    _pad       : vec2<u32>,  // offset 40
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
    inv_view_proj : mat4x4<f32>,  // offset 0 (column-major), size 64
    screen_width  : u32,          // offset 64
    screen_height : u32,          // offset 68
    rays_per_pixel: u32,          // offset 72
    frame_seed    : u32,          // offset 76
    max_distance  : f32,          // offset 80
    normal_bias   : f32,          // offset 84
}

@group(0) @binding(0) var<storage, read> bvh_nodes : array<BvhNode>;
@group(0) @binding(1) var<storage, read> triangles  : array<GpuTriangle>;
@group(0) @binding(2) var<uniform>       cam        : Camera;
@group(0) @binding(3) var               t_depth    : texture_depth_2d;
@group(0) @binding(4) var               ao_out     : texture_storage_2d<rgba8unorm, write>;

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

// ---- Stackless BVH traversal ----

fn intersects_any(origin: vec3<f32>, dir: vec3<f32>, tmax: f32) -> bool {
    let n = arrayLength(&bvh_nodes);
    if (n == 0u) { return false; }
    let inv_dir = vec3<f32>(1.0/dir.x, 1.0/dir.y, 1.0/dir.z);
    var idx: u32 = 0u;
    loop {
        if (idx >= n || idx == 0xFFFFFFFFu) { break; }
        let node = bvh_nodes[idx];
        if (!aabb_hit(node, origin, inv_dir, tmax)) {
            idx = node.miss_next;
            continue;
        }
        if (node.tri_count > 0u) {
            for (var i = node.tri_offset; i < node.tri_offset + node.tri_count; i += 1u) {
                if (ray_tri_hit(origin, dir, triangles[i], tmax)) { return true; }
            }
            idx = node.miss_next;
        } else {
            idx = node.hit_next;
        }
    }
    return false;
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
    // The cross product sign depends on surface orientation; orient toward camera.
    // In view space the camera is at the origin, so the toward-camera direction is -normalize(pos).
    // All visible surfaces must face the camera, so flip if needed.
    if (dot(n_norm, -normalize(pos)) < 0.0) { n_norm = -n_norm; }
    return n_norm;
}

// ---- Main compute entry point ----

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= cam.screen_width || py >= cam.screen_height) { return; }

    // Sample depth; treat far-plane pixels as fully lit (no geometry).
    // Aurora uses reversed-Z (near=1, far=0), so sky pixels have d near 0.
    let d = textureLoad(t_depth, vec2<i32>(i32(px), i32(py)), 0);
    if (d <= 0.0001) {
        textureStore(ao_out, vec2<i32>(i32(px), i32(py)), vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }

    let pos_h = unproject(px, py);
    if (pos_h.w < 0.5) {
        textureStore(ao_out, vec2<i32>(i32(px), i32(py)), vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }
    let pos    = pos_h.xyz;
    let normal = estimate_normal(px, py, pos);

    // Offset the ray origin along the surface normal to avoid self-intersection.
    // The bias scales with distance from camera; the constant factor (normal_bias)
    // is tunable so it can be matched to the game's world-unit scale.
    let dist_from_cam = length(pos);
    let ray_origin = pos + normal * max(dist_from_cam * cam.normal_bias, cam.normal_bias * 10.0);

    // Per-pixel seed = (y * W + x) * large_prime XOR frame_seed
    var seed = pcg(((py * cam.screen_width + px) * 1664525u) ^ cam.frame_seed);

    var hits: u32 = 0u;
    let N = cam.rays_per_pixel;
    for (var i = 0u; i < N; i += 1u) {
        let dir = cosine_hemisphere(normal, &seed);
        if (intersects_any(ray_origin, dir, cam.max_distance)) {
            hits += 1u;
        }
    }

    let ao = 1.0 - f32(hits) / f32(N);
    textureStore(ao_out, vec2<i32>(i32(px), i32(py)), vec4<f32>(ao, ao, ao, 1.0));
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
    if (m_bindGroup)  wgpuBindGroupRelease(m_bindGroup);
    if (m_depthView)  wgpuTextureViewRelease(m_depthView);
    if (m_aoView)     wgpuTextureViewRelease(m_aoView);
    if (m_aoTex)      wgpuTextureRelease(m_aoTex);
    if (m_cameraUbo)  wgpuBufferRelease(m_cameraUbo);
    if (m_triBuf)     wgpuBufferRelease(m_triBuf);
    if (m_nodeBuf)    wgpuBufferRelease(m_nodeBuf);
    if (m_bgl)        wgpuBindGroupLayoutRelease(m_bgl);
    if (m_pipeline)   wgpuComputePipelineRelease(m_pipeline);
}

void AoPass::queue_bvh_upload(const Bvh& bvh) {
    if (bvh.empty()) return;

    auto pending = std::make_unique<PendingBvh>();

    const auto& src_nodes = bvh.nodes();
    pending->nodeCount = static_cast<uint32_t>(src_nodes.size());
    pending->nodeData.resize(src_nodes.size() * sizeof(GpuBvhNode));
    auto* gpu_nodes = reinterpret_cast<GpuBvhNode*>(pending->nodeData.data());
    for (size_t i = 0; i < src_nodes.size(); ++i) {
        const auto& n = src_nodes[i];
        gpu_nodes[i].boundsMin[0] = n.bounds.min.x;
        gpu_nodes[i].boundsMin[1] = n.bounds.min.y;
        gpu_nodes[i].boundsMin[2] = n.bounds.min.z;
        gpu_nodes[i].hitNext      = n.hit_next;
        gpu_nodes[i].boundsMax[0] = n.bounds.max.x;
        gpu_nodes[i].boundsMax[1] = n.bounds.max.y;
        gpu_nodes[i].boundsMax[2] = n.bounds.max.z;
        gpu_nodes[i].missNext     = n.miss_next;
        gpu_nodes[i].triOffset    = n.tri_offset;
        gpu_nodes[i].triCount     = n.tri_count;
        gpu_nodes[i]._pad[0]      = 0;
        gpu_nodes[i]._pad[1]      = 0;
    }

    const auto& src_tris = bvh.tris();
    pending->triCount = static_cast<uint32_t>(src_tris.size());
    pending->triData.resize(src_tris.size() * sizeof(GpuTriangle));
    auto* gpu_tris = reinterpret_cast<GpuTriangle*>(pending->triData.data());
    for (size_t i = 0; i < src_tris.size(); ++i) {
        gpu_tris[i].a[0] = src_tris[i].a.x; gpu_tris[i].a[1] = src_tris[i].a.y; gpu_tris[i].a[2] = src_tris[i].a.z; gpu_tris[i]._p0 = 0;
        gpu_tris[i].b[0] = src_tris[i].b.x; gpu_tris[i].b[1] = src_tris[i].b.y; gpu_tris[i].b[2] = src_tris[i].b.z; gpu_tris[i]._p1 = 0;
        gpu_tris[i].c[0] = src_tris[i].c.x; gpu_tris[i].c[1] = src_tris[i].c.y; gpu_tris[i].c[2] = src_tris[i].c.z; gpu_tris[i]._p2 = 0;
    }

    m_pendingBvh = std::move(pending);
}

void AoPass::flush_pending_upload(WGPUDevice device) {
    if (!m_pendingBvh) return;

    if (m_nodeBuf) { wgpuBufferRelease(m_nodeBuf); m_nodeBuf = nullptr; }
    if (m_triBuf)  { wgpuBufferRelease(m_triBuf);  m_triBuf  = nullptr; }

    m_nodeBuf = create_buffer(device, m_pendingBvh->nodeData.size(),
                              WGPUBufferUsage_Storage, m_pendingBvh->nodeData.data());
    m_triBuf  = create_buffer(device, m_pendingBvh->triData.size(),
                              WGPUBufferUsage_Storage, m_pendingBvh->triData.data());
    m_nodeCount = m_pendingBvh->nodeCount;
    m_triCount  = m_pendingBvh->triCount;
    m_pendingBvh.reset();
    m_bindGroupDirty = true;
}

void AoPass::ensure_pipeline(WGPUDevice device) {
    if (m_pipeline) return;

    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code        = {kShader, WGPU_STRLEN};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslSrc.chain;
    WGPUShaderModule sm = wgpuDeviceCreateShaderModule(device, &smDesc);

    // Bind group layout: 5 entries
    WGPUBindGroupLayoutEntry entries[5] = {};

    // 0: BVH nodes (read-only storage buffer)
    entries[0].binding          = 0;
    entries[0].visibility       = WGPUShaderStage_Compute;
    entries[0].buffer.type      = WGPUBufferBindingType_ReadOnlyStorage;

    // 1: triangles (read-only storage buffer)
    entries[1].binding          = 1;
    entries[1].visibility       = WGPUShaderStage_Compute;
    entries[1].buffer.type      = WGPUBufferBindingType_ReadOnlyStorage;

    // 2: camera uniform
    entries[2].binding          = 2;
    entries[2].visibility       = WGPUShaderStage_Compute;
    entries[2].buffer.type      = WGPUBufferBindingType_Uniform;

    // 3: depth texture
    entries[3].binding                     = 3;
    entries[3].visibility                  = WGPUShaderStage_Compute;
    entries[3].texture.sampleType          = WGPUTextureSampleType_Depth;
    entries[3].texture.viewDimension       = WGPUTextureViewDimension_2D;
    entries[3].texture.multisampled        = 0;

    // 4: AO output (write-only storage texture)
    entries[4].binding                       = 4;
    entries[4].visibility                    = WGPUShaderStage_Compute;
    entries[4].storageTexture.access         = WGPUStorageTextureAccess_WriteOnly;
    entries[4].storageTexture.format         = WGPUTextureFormat_RGBA8Unorm;
    entries[4].storageTexture.viewDimension  = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 5;
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
    if (m_aoView) { wgpuTextureViewRelease(m_aoView); m_aoView = nullptr; }
    if (m_aoTex)  { wgpuTextureRelease(m_aoTex);      m_aoTex  = nullptr; }

    WGPUTextureDescriptor td{};
    td.usage         = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
    td.dimension     = WGPUTextureDimension_2D;
    td.size          = {w, h, 1};
    td.format        = WGPUTextureFormat_RGBA8Unorm;
    td.mipLevelCount = 1;
    td.sampleCount   = 1;
    m_aoTex  = wgpuDeviceCreateTexture(device, &td);
    m_aoView = wgpuTextureCreateView(m_aoTex, nullptr);

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

    WGPUBindGroupEntry entries[5] = {};
    // size = WGPU_WHOLE_SIZE (0) binds the entire buffer
    entries[0].binding = 0; entries[0].buffer = m_nodeBuf;   entries[0].size = WGPU_WHOLE_SIZE;
    entries[1].binding = 1; entries[1].buffer = m_triBuf;    entries[1].size = WGPU_WHOLE_SIZE;
    entries[2].binding = 2; entries[2].buffer = m_cameraUbo; entries[2].size = sizeof(GpuCamera);
    entries[3].binding = 3; entries[3].textureView = m_depthView;
    entries[4].binding = 4; entries[4].textureView = m_aoView;

    WGPUBindGroupDescriptor bgd{};
    bgd.layout     = m_bgl;
    bgd.entryCount = 5;
    bgd.entries    = entries;
    m_bindGroup      = wgpuDeviceCreateBindGroup(device, &bgd);
    m_bindGroupDirty = false;
}

void AoPass::execute(WGPUDevice device, WGPUCommandEncoder encoder,
                     WGPUTexture depthTex,
                     const GeometryCollector::CameraData& cam) {
    if (!cam.valid || !depthTex) return;

    flush_pending_upload(device);
    if (!m_nodeBuf) return;

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
    if (!compute_inv_proj(cam, gpuCam.invViewProj)) return;

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
    const uint32_t gx = (w + 7) / 8;
    const uint32_t gy = (h + 7) / 8;
    wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

ImTextureID AoPass::imgui_texture_id() const {
    return reinterpret_cast<ImTextureID>(m_aoView);
}

} // namespace dusk::rtao
