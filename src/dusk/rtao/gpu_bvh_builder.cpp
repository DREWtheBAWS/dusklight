#include "gpu_bvh_builder.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>

namespace dusk::rtao {

// ---------------------------------------------------------------------------
// WGSL — one module, 7 entry points, one shared bind group layout
//
// Bind group layout (group 0):
//   0: b_tri_in   read-only storage  — compact input triangles (9 f32 each)
//   1: b_aabb     read-write storage — per-triangle AABB      (6 f32 each)
//   2: b_scene    read-write storage — global scene AABB      (6 f32)
//   3: b_morton   read-write storage — Morton codes           (u32 each)
//   4: b_idx      read-write storage — sort permutation       (u32 each)
//   5: b_nodes    read-write storage — BVH nodes              (BvhNode each)
//   6: b_tri_out  read-write storage — sorted GpuTriangle out (20 f32 each)
//   7: b_params   uniform            — Params (4 x u32)
//
// BvhNode layout (48 bytes, identical to what AoPass expects):
//   bounds_min  : vec3<f32>  @ 0    (12 B)
//   left_child  : u32        @ 12   ( 4 B)  interior→left child, leaf→0xFFFF
//   bounds_max  : vec3<f32>  @ 16   (12 B)
//   right_child : u32        @ 28   ( 4 B)  interior→right child
//   tri_offset  : u32        @ 32   ( 4 B)  leaf only
//   tri_count   : u32        @ 36   ( 4 B)  0=interior, 1=leaf
//   range_first : u32        @ 40   ( 4 B)  build-time: leaf range start
//   range_last  : u32        @ 44   ( 4 B)  build-time: leaf range end
// ---------------------------------------------------------------------------

static const char kShader[] = R"(

struct Params {
    tri_count  : u32,   // number of real triangles
    n_padded   : u32,   // next power of 2 >= tri_count (bitonic sort)
    bitonic_k  : u32,   // outer stage for bitonic sort
    bitonic_j  : u32,   // inner step  for bitonic sort
}

struct BvhNode {
    bounds_min  : vec3<f32>,
    left_child  : u32,
    bounds_max  : vec3<f32>,
    right_child : u32,
    tri_offset  : u32,
    tri_count   : u32,
    range_first : u32,
    range_last  : u32,
}

@group(0) @binding(0) var<storage, read>       b_tri_in : array<f32>;
@group(0) @binding(1) var<storage, read_write> b_aabb   : array<f32>;
@group(0) @binding(2) var<storage, read_write> b_scene  : array<f32>;
@group(0) @binding(3) var<storage, read_write> b_morton : array<u32>;
@group(0) @binding(4) var<storage, read_write> b_idx    : array<u32>;
@group(0) @binding(5) var<storage, read_write> b_nodes  : array<BvhNode>;
@group(0) @binding(6) var<storage, read_write> b_tri_out: array<f32>;
@group(0) @binding(7) var<uniform>             b_params : Params;

// ==========================================================================
// cs_bounds: compute per-triangle AABB — one thread per triangle
// ==========================================================================
@compute @workgroup_size(256)
fn cs_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= b_params.tri_count) { return; }
    // GpuTriangle layout (20 f32): a[0-2], _p0[3], b[4-6], _p1[7], c[8-10], ...
    let s = i * 20u;
    let ax = b_tri_in[s];     let ay = b_tri_in[s+1u]; let az = b_tri_in[s+2u];
    let bx = b_tri_in[s+4u];  let by = b_tri_in[s+5u]; let bz = b_tri_in[s+6u];
    let cx = b_tri_in[s+8u];  let cy = b_tri_in[s+9u]; let cz = b_tri_in[s+10u];
    let d = i * 6u;
    b_aabb[d]   = min(ax, min(bx, cx));
    b_aabb[d+1u] = min(ay, min(by, cy));
    b_aabb[d+2u] = min(az, min(bz, cz));
    b_aabb[d+3u] = max(ax, max(bx, cx));
    b_aabb[d+4u] = max(ay, max(by, cy));
    b_aabb[d+5u] = max(az, max(bz, cz));
}

// ==========================================================================
// cs_reduce: reduce all per-tri AABBs to one global scene AABB.
// Single workgroup of 256 — each thread loops over ceil(n/256) entries.
// ==========================================================================
var<workgroup> wg_mn: array<vec3<f32>, 256>;
var<workgroup> wg_mx: array<vec3<f32>, 256>;

@compute @workgroup_size(256)
fn cs_reduce(@builtin(local_invocation_index) li: u32) {
    let n = b_params.tri_count;
    var mn = vec3<f32>( 1e30,  1e30,  1e30);
    var mx = vec3<f32>(-1e30, -1e30, -1e30);
    var i = li;
    loop {
        if (i >= n) { break; }
        let d = i * 6u;
        mn = min(mn, vec3<f32>(b_aabb[d], b_aabb[d+1u], b_aabb[d+2u]));
        mx = max(mx, vec3<f32>(b_aabb[d+3u], b_aabb[d+4u], b_aabb[d+5u]));
        i += 256u;
    }
    wg_mn[li] = mn;
    wg_mx[li] = mx;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (li < s) {
            wg_mn[li] = min(wg_mn[li], wg_mn[li + s]);
            wg_mx[li] = max(wg_mx[li], wg_mx[li + s]);
        }
        workgroupBarrier();
        s >>= 1u;
    }
    if (li == 0u) {
        b_scene[0u] = wg_mn[0u].x; b_scene[1u] = wg_mn[0u].y; b_scene[2u] = wg_mn[0u].z;
        b_scene[3u] = wg_mx[0u].x; b_scene[4u] = wg_mx[0u].y; b_scene[5u] = wg_mx[0u].z;
    }
}

// ==========================================================================
// cs_morton: compute 30-bit Morton codes from normalised triangle centroids.
// Also initialises the sort permutation b_idx[i] = i and pads extras with
// sentinel 0xFFFFFFFF so they sort to the end.
// ==========================================================================
fn expand_bits(v_in: u32) -> u32 {
    var v = v_in & 0x3FFu;
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v <<  8u)) & 0x0300F00Fu;
    v = (v | (v <<  4u)) & 0x030C30C3u;
    v = (v | (v <<  2u)) & 0x09249249u;
    return v;
}

@compute @workgroup_size(256)
fn cs_morton(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n_pad = b_params.n_padded;
    if (i >= n_pad) { return; }
    b_idx[i] = i;
    if (i >= b_params.tri_count) { b_morton[i] = 0xFFFFFFFFu; return; }

    let scene_mn = vec3<f32>(b_scene[0u], b_scene[1u], b_scene[2u]);
    let scene_mx = vec3<f32>(b_scene[3u], b_scene[4u], b_scene[5u]);
    let extent   = max(scene_mx - scene_mn, vec3<f32>(1e-6));
    let d = i * 6u;
    let tmn = vec3<f32>(b_aabb[d], b_aabb[d+1u], b_aabb[d+2u]);
    let tmx = vec3<f32>(b_aabb[d+3u], b_aabb[d+4u], b_aabb[d+5u]);
    let c = clamp(((tmn + tmx) * 0.5 - scene_mn) / extent, vec3<f32>(0.0), vec3<f32>(1.0));
    let xi = u32(c.x * 1023.0);
    let yi = u32(c.y * 1023.0);
    let zi = u32(c.z * 1023.0);
    b_morton[i] = (expand_bits(zi) << 2u) | (expand_bits(yi) << 1u) | expand_bits(xi);
}

// ==========================================================================
// cs_bitonic: one step of a global bitonic sort on (b_morton, b_idx).
// Parameterised by b_params.bitonic_k (outer stage) and bitonic_j (step).
// Dispatch with n_padded/2 threads total.
// ==========================================================================
@compute @workgroup_size(256)
fn cs_bitonic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i   = gid.x;
    let n   = b_params.n_padded;
    if (i >= n >> 1u) { return; }
    let k   = b_params.bitonic_k;
    let j   = b_params.bitonic_j;
    let ixj = i ^ j;
    if (ixj <= i) { return; }
    let ascending = (i & k) == 0u;
    let a = b_morton[i];
    let b_ = b_morton[ixj];
    if ((ascending && a > b_) || (!ascending && a < b_)) {
        b_morton[i] = b_; b_morton[ixj] = a;
        let t = b_idx[i]; b_idx[i] = b_idx[ixj]; b_idx[ixj] = t;
    }
}

// ==========================================================================
// cs_init_leaves: initialise leaf nodes in b_nodes and write sorted
// triangles to b_tri_out in GpuTriangle layout (12 f32 each, with padding).
// Leaves occupy slots [n-1, 2n-2] in b_nodes.
// ==========================================================================
@compute @workgroup_size(256)
fn cs_init_leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = b_params.tri_count;
    if (i >= n) { return; }
    let leaf = n - 1u + i;
    let tri  = b_idx[i];            // original triangle index (sorted permutation)
    let ao   = tri * 6u;
    // Clamp leaf AABB to the mortonAabb (b_scene).  Large triangles that straddle
    // the boundary of the AO-relevant region would otherwise inflate every ancestor
    // node's AABB, causing AO rays to descend into subtrees they can never hit.
    // The geometry collector already excludes triangles whose centroid is beyond the
    // mortonAabb, so the clamped AABB is always non-degenerate (min ≤ max).
    let scene_mn = vec3<f32>(b_scene[0u], b_scene[1u], b_scene[2u]);
    let scene_mx = vec3<f32>(b_scene[3u], b_scene[4u], b_scene[5u]);
    let raw_mn   = vec3<f32>(b_aabb[ao],    b_aabb[ao+1u], b_aabb[ao+2u]);
    let raw_mx   = vec3<f32>(b_aabb[ao+3u], b_aabb[ao+4u], b_aabb[ao+5u]);
    b_nodes[leaf].bounds_min  = max(raw_mn, scene_mn);
    b_nodes[leaf].bounds_max  = min(raw_mx, scene_mx);
    b_nodes[leaf].left_child  = 0xFFFFFFFFu;
    b_nodes[leaf].right_child = 0xFFFFFFFFu;
    b_nodes[leaf].tri_offset  = i;   // index into b_tri_out (sorted order)
    b_nodes[leaf].tri_count   = 1u;
    b_nodes[leaf].range_first = i;
    b_nodes[leaf].range_last  = i;
    // Write sorted GpuTriangle (80 B = 20 f32): verbatim copy preserves positions,
    // padding, UVs, texIdx, and flags (u32 bits stored as f32 bits in b_tri_in).
    let src = tri * 20u;
    let dst = i   * 20u;
    for (var k = 0u; k < 20u; k += 1u) {
        b_tri_out[dst + k] = b_tri_in[src + k];
    }
}

// ==========================================================================
// cs_lbvh: Karras 2012 parallel LBVH hierarchy construction.
// One thread per internal node (index 0..n-2).
// Internal nodes occupy [0, n-2]; leaves occupy [n-1, 2n-2].
// ==========================================================================

// Length of the longest common prefix of Morton codes at positions i and j.
// Tie-breaks duplicate codes using the index.
fn delta(i_: i32, j_: i32, n_: i32) -> i32 {
    if (j_ < 0 || j_ >= n_) { return -1i; }
    let mi = b_morton[u32(i_)];
    let mj = b_morton[u32(j_)];
    if (mi != mj) { return i32(countLeadingZeros(mi ^ mj)); }
    // Duplicate Morton codes — tie-break with index XOR
    return 32i + i32(countLeadingZeros(u32(i_) ^ u32(j_)));
}

@compute @workgroup_size(256)
fn cs_lbvh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    let n   = i32(b_params.tri_count);
    if (idx >= n - 1i) { return; }

    // Direction of the range from this node
    let d_fwd = delta(idx, idx + 1i, n);
    let d_bwd = delta(idx, idx - 1i, n);
    let d_sign = select(-1i, 1i, d_fwd > d_bwd);

    // Upper bound for range length (exponential search)
    let delta_min = delta(idx, idx - d_sign, n);
    var l_max = 2i;
    for (var iter = 0i; iter < 64i; iter++) {
        if (delta(idx, idx + l_max * d_sign, n) <= delta_min) { break; }
        l_max = l_max << 1u;   // WGSL: shift amount must be u32
        if (l_max > n) { l_max = n; break; }
    }

    // Binary search for the exact range end.
    // The initial step must be the highest power of 2 <= l_max.  We compute
    // this with an explicit doubling loop (avoids countLeadingZeros, which is
    // not reliably available in all Dawn/WGSL versions).  The first probe lands
    // exactly at l_max, which always fails by definition of the exponential
    // search, so the subsequent halvings correctly narrow to the true range end.
    var l = 0i;
    var t = 1i;
    loop { if ((t << 1u) > l_max) { break; } t = t << 1u; }
    loop {
        if (t == 0i) { break; }
        if (delta(idx, idx + (l + t) * d_sign, n) > delta_min) { l += t; }
        t = t >> 1u;
    }
    let j     = idx + l * d_sign;
    let first = min(idx, j);
    let last  = max(idx, j);

    // Binary search for the split position within [first, last].
    // Same approach: start at highest power of 2 <= (last - first).  The first
    // probe tests the boundary (delta == delta_node, never strictly greater),
    // so it always misses and the halvings find the correct split.
    let delta_node = delta(first, last, n);
    var s  = 0i;
    var t2 = 1i;
    loop { if ((t2 << 1u) > last - first) { break; } t2 = t2 << 1u; }
    loop {
        if (t2 == 0i) { break; }
        if (delta(first, first + s + t2, n) > delta_node) { s += t2; }
        t2 = t2 >> 1u;
    }
    let gamma = first + s;

    // Child indices: if child covers a single leaf it IS a leaf node
    let left_child  = select(u32(gamma),       u32(n - 1i + gamma),       gamma == first);
    let right_child = select(u32(gamma + 1i),  u32(n - 1i + gamma + 1i),  gamma + 1i == last);

    b_nodes[u32(idx)].left_child  = left_child;
    b_nodes[u32(idx)].right_child = right_child;
    b_nodes[u32(idx)].tri_offset  = 0u;
    b_nodes[u32(idx)].tri_count   = 0u;
    b_nodes[u32(idx)].range_first = u32(first);
    b_nodes[u32(idx)].range_last  = u32(last);
}

// ==========================================================================
// cs_aabb: compute AABB for each internal node by reducing its leaf range.
// One thread per internal node (0..n-2).
//
// Reads ONLY from leaf nodes (n-1..2n-2), which are written exclusively by
// cs_init_leaves in the previous pass.  Internal nodes are written but never
// read within this dispatch, so there are no intra-dispatch races and no
// cross-workgroup barriers are needed — a single dispatch is correct.
//
// (A multi-pass child-union approach was tried but caused frame-to-frame
// instability on D3D12: UAV cache flushes between consecutive compute passes
// are not guaranteed to be visible in the next pass's L1/L2 reads on all
// drivers, leading to stale propagated values and a wrong root AABB.)
// ==========================================================================
@compute @workgroup_size(256)
fn cs_aabb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n   = b_params.tri_count;
    if (idx >= n - 1u) { return; }

    let first = b_nodes[idx].range_first;
    let last  = b_nodes[idx].range_last;

    var mn = vec3<f32>( 1e30,  1e30,  1e30);
    var mx = vec3<f32>(-1e30, -1e30, -1e30);
    for (var li = first; li <= last; li++) {
        let leaf = n - 1u + li;
        mn = min(mn, b_nodes[leaf].bounds_min);
        mx = max(mx, b_nodes[leaf].bounds_max);
    }
    b_nodes[idx].bounds_min = mn;
    b_nodes[idx].bounds_max = mx;
}
)";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static uint32_t next_pow2(uint32_t n) {
    if (n <= 1) return 1;
    --n;
    n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    return ++n;
}

// Create a one-shot staging (upload) buffer that is already mapped and written.
// Record a CopyBufferToBuffer into the encoder, then call wgpuBufferRelease —
// Dawn keeps the underlying resource alive until the GPU finishes the copy.
static WGPUBuffer make_upload_buf(WGPUDevice dev, const void* data, uint64_t size) {
    const uint64_t aligned = (size + 3ull) & ~3ull;
    WGPUBufferDescriptor d{};
    d.size             = aligned;
    d.usage            = WGPUBufferUsage_CopySrc;
    d.mappedAtCreation = true;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(dev, &d);
    void* ptr = wgpuBufferGetMappedRange(buf, 0, aligned);
    memcpy(ptr, data, size);
    wgpuBufferUnmap(buf);
    return buf;
}

static WGPUBuffer make_buf(WGPUDevice dev, uint64_t size, WGPUBufferUsage usage,
                            const void* data = nullptr) {
    if (size == 0) size = 4; // WebGPU rejects zero-size buffers
    WGPUBufferDescriptor d{};
    d.size  = size;
    d.usage = usage | WGPUBufferUsage_CopyDst;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(dev, &d);
    if (data) {
        WGPUQueue q = wgpuDeviceGetQueue(dev);
        wgpuQueueWriteBuffer(q, buf, 0, data, size);
        wgpuQueueRelease(q);
    }
    return buf;
}

static WGPUComputePipeline make_pipeline(WGPUDevice dev, WGPUShaderModule sm,
                                          WGPUPipelineLayout pl, const char* entry) {
    WGPUComputePipelineDescriptor d{};
    d.layout             = pl;
    d.compute.module     = sm;
    d.compute.entryPoint = {entry, WGPU_STRLEN};
    return wgpuDeviceCreateComputePipeline(dev, &d);
}

// ---------------------------------------------------------------------------
// GpuBvhBuilder
// ---------------------------------------------------------------------------

GpuBvhBuilder::~GpuBvhBuilder() {
    if (m_pipeBounds)    wgpuComputePipelineRelease(m_pipeBounds);
    if (m_pipeReduce)    wgpuComputePipelineRelease(m_pipeReduce);
    if (m_pipeMorton)    wgpuComputePipelineRelease(m_pipeMorton);
    if (m_pipeBitonic)   wgpuComputePipelineRelease(m_pipeBitonic);
    if (m_pipeLeaves)    wgpuComputePipelineRelease(m_pipeLeaves);
    if (m_pipeLbvh)      wgpuComputePipelineRelease(m_pipeLbvh);
    if (m_pipeAabb)      wgpuComputePipelineRelease(m_pipeAabb);
    if (m_bgl)           wgpuBindGroupLayoutRelease(m_bgl);
    if (m_triInputBuf)   wgpuBufferRelease(m_triInputBuf);
    if (m_aabbBuf)       wgpuBufferRelease(m_aabbBuf);
    if (m_sceneAabbBuf)  wgpuBufferRelease(m_sceneAabbBuf);
    if (m_mortonBuf)     wgpuBufferRelease(m_mortonBuf);
    if (m_indicesBuf)    wgpuBufferRelease(m_indicesBuf);
    if (m_nodeBuf)       wgpuBufferRelease(m_nodeBuf);
    if (m_triBuf)        wgpuBufferRelease(m_triBuf);
    if (m_sortStepsBuf)  wgpuBufferRelease(m_sortStepsBuf);
}

void GpuBvhBuilder::ensure_pipelines(WGPUDevice device) {
    if (m_pipeBounds) return;

    // Shader module
    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code        = {kShader, WGPU_STRLEN};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslSrc.chain;
    WGPUShaderModule sm = wgpuDeviceCreateShaderModule(device, &smDesc);

    // Bind group layout (8 entries, shared by all pipelines)
    WGPUBindGroupLayoutEntry bgle[8] = {};

    auto storage_ro = [](uint32_t binding) {
        WGPUBindGroupLayoutEntry e{};
        e.binding       = binding;
        e.visibility    = WGPUShaderStage_Compute;
        e.buffer.type   = WGPUBufferBindingType_ReadOnlyStorage;
        return e;
    };
    auto storage_rw = [](uint32_t binding) {
        WGPUBindGroupLayoutEntry e{};
        e.binding       = binding;
        e.visibility    = WGPUShaderStage_Compute;
        e.buffer.type   = WGPUBufferBindingType_Storage;
        return e;
    };
    // Binding 7: dynamic uniform — each dispatch selects its own 256-byte-aligned slot
    // in m_sortStepsBuf via SetBindGroup's dynamicOffsets argument.  No buffer copies
    // between passes are needed, which avoids D3D12 resource-state conflicts.
    WGPUBindGroupLayoutEntry dynUniform{};
    dynUniform.binding                = 7;
    dynUniform.visibility             = WGPUShaderStage_Compute;
    dynUniform.buffer.type            = WGPUBufferBindingType_Uniform;
    dynUniform.buffer.hasDynamicOffset = true;
    dynUniform.buffer.minBindingSize  = 16;

    bgle[0] = storage_ro(0); // b_tri_in
    bgle[1] = storage_rw(1); // b_aabb
    bgle[2] = storage_rw(2); // b_scene
    bgle[3] = storage_rw(3); // b_morton
    bgle[4] = storage_rw(4); // b_idx
    bgle[5] = storage_rw(5); // b_nodes
    bgle[6] = storage_rw(6); // b_tri_out
    bgle[7] = dynUniform;    // b_params  (dynamic offset selects step)

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 8;
    bglDesc.entries    = bgle;
    m_bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts     = &m_bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    m_pipeBounds  = make_pipeline(device, sm, pl, "cs_bounds");
    m_pipeReduce  = make_pipeline(device, sm, pl, "cs_reduce");
    m_pipeMorton  = make_pipeline(device, sm, pl, "cs_morton");
    m_pipeBitonic = make_pipeline(device, sm, pl, "cs_bitonic");
    m_pipeLeaves  = make_pipeline(device, sm, pl, "cs_init_leaves");
    m_pipeLbvh    = make_pipeline(device, sm, pl, "cs_lbvh");
    m_pipeAabb    = make_pipeline(device, sm, pl, "cs_aabb");

    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(sm);
}

void GpuBvhBuilder::resize_buffers(WGPUDevice device, uint32_t n) {
    if (n <= m_bufCapacity) return;
    const uint32_t np = next_pow2(n);

    auto rel = [](WGPUBuffer& b) { if (b) { wgpuBufferRelease(b); b = nullptr; } };
    rel(m_triInputBuf); rel(m_aabbBuf); rel(m_mortonBuf);
    rel(m_indicesBuf);  rel(m_nodeBuf); rel(m_triBuf);
    if (!m_sceneAabbBuf)
        m_sceneAabbBuf = make_buf(device, 24, WGPUBufferUsage_Storage);
    // Params stored with 256-byte stride (D3D12 minUniformBufferOffsetAlignment).
    // Slot 0 = base {tri_count, n_padded, 0, 0}; slots 1..N = bitonic (k,j) steps.
    // 512 slots × 256 bytes covers up to 2^31-padded (496 sort steps max).
    if (!m_sortStepsBuf)
        m_sortStepsBuf = make_buf(device, 512 * 256, WGPUBufferUsage_Uniform);

    m_triInputBuf = make_buf(device, uint64_t(n)    * 80, WGPUBufferUsage_Storage);
    m_aabbBuf     = make_buf(device, uint64_t(n)    * 24, WGPUBufferUsage_Storage);
    m_mortonBuf   = make_buf(device, uint64_t(np)   *  4, WGPUBufferUsage_Storage);
    m_indicesBuf  = make_buf(device, uint64_t(np)   *  4, WGPUBufferUsage_Storage);
    // AoPass reads these directly — no copy needed; UAV state decays to D3D12
    // COMMON between command lists, allowing implicit SRV promotion on the next read.
    m_nodeBuf = make_buf(device, uint64_t(2*n-1) * 48, WGPUBufferUsage_Storage);
    m_triBuf  = make_buf(device, uint64_t(n)     * 80, WGPUBufferUsage_Storage);
    m_bufCapacity = n;
}

void GpuBvhBuilder::upload_triangles(WGPUDevice device,
                                      const std::vector<Triangle>& tris) {
    if (tris.empty()) { m_triCount = 0; m_pendingTriData.clear(); return; }
    const uint32_t n = static_cast<uint32_t>(tris.size());
    resize_buffers(device, n);
    m_triCount = n;

    // Pack triangles into CPU vector matching GpuTriangle layout (20 f32 = 80 B).
    // Fields: a[3], _p0, b[3], _p1, c[3], _p2, uva[2], uvb[2], uvc[2], texIdx, flags.
    // texIdx and flags are u32s stored as bit-identical f32 (read back as u32 via struct).
    // The actual GPU upload happens inside build() via a staging buffer.
    m_pendingTriData.resize(n * 20u);
    for (uint32_t i = 0; i < n; ++i) {
        const auto& t = tris[i];
        float* p = m_pendingTriData.data() + i * 20u;
        p[ 0] = t.a.x;   p[ 1] = t.a.y;   p[ 2] = t.a.z;   p[ 3] = 0.f;
        p[ 4] = t.b.x;   p[ 5] = t.b.y;   p[ 6] = t.b.z;   p[ 7] = 0.f;
        p[ 8] = t.c.x;   p[ 9] = t.c.y;   p[10] = t.c.z;   p[11] = 0.f;
        p[12] = t.uva.u; p[13] = t.uva.v;
        p[14] = t.uvb.u; p[15] = t.uvb.v;
        p[16] = t.uvc.u; p[17] = t.uvc.v;
        uint32_t ti = t.texIdx, fl = t.flags;
        memcpy(&p[18], &ti, 4);
        memcpy(&p[19], &fl, 4);
    }
}

// ---------------------------------------------------------------------------
// build() — record uploads + all passes + copy into the caller's encoder
// ---------------------------------------------------------------------------

void GpuBvhBuilder::build(WGPUDevice device, WGPUCommandEncoder encoder) {
    if (m_triCount == 0 || !m_triInputBuf) return;

    ensure_pipelines(device);

    const uint32_t n      = m_triCount;
    const uint32_t np     = next_pow2(n);
    const uint32_t g256_n = (n  + 255u) / 256u;
    const uint32_t g256_p = (np / 2u + 255u) / 256u;
    const uint32_t g256_i = (n  > 1u) ? ((n - 1u + 255u) / 256u) : 0u;

    // Pre-upload all bitonic sort step params (one 256-byte slot per step).
    struct Params { uint32_t tri_count, n_padded, bitonic_k, bitonic_j; };
    static constexpr uint32_t kStride = 256u;
    std::vector<uint8_t> stepsBlob;
    stepsBlob.reserve(512 * kStride);
    auto pushStep = [&](uint32_t k, uint32_t j) {
        const size_t base = stepsBlob.size();
        stepsBlob.resize(base + kStride, 0);
        *reinterpret_cast<Params*>(stepsBlob.data() + base) = {n, np, k, j};
    };
    pushStep(0, 0); // slot 0 = base params for all non-bitonic passes
    uint32_t sortStepCount = 0;
    for (uint32_t k = 2; k <= np; k <<= 1u)
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) { pushStep(k, j); ++sortStepCount; }

    // Single CPU pass: full scene AABB for display + per-vertex min/max.
    // Must run before the GPU uploads below since mortonAabb is uploaded as a staging buffer.
    float sceneMin[3] = { 1e30f,  1e30f,  1e30f};
    float sceneMax[3] = {-1e30f, -1e30f, -1e30f};
    for (uint32_t i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k) {
            // GpuTriangle stride = 20 f32; a at [0], b at [4], c at [8]
            const float va = m_pendingTriData[i*20 + k];
            const float vb = m_pendingTriData[i*20 + 4 + k];
            const float vc = m_pendingTriData[i*20 + 8 + k];
            sceneMin[k] = std::min({sceneMin[k], va, vb, vc});
            sceneMax[k] = std::max({sceneMax[k], va, vb, vc});
        }
    }

    // Morton AABB: clamp the scene AABB to a cube of half-side m_mortonRange centred
    // on the camera world position.  Triangles are now in world space so we centre on
    // m_worldPos rather than the origin.  Geometry beyond m_mortonRange can't affect AO
    // results and ends up in fringe subtrees that are rarely visited.
    float mortonAabb[6]; // [minX,minY,minZ, maxX,maxY,maxZ]
    for (int k = 0; k < 3; ++k) {
        mortonAabb[k]     = std::max(sceneMin[k], m_worldPos[k] - m_mortonRange);
        mortonAabb[k + 3] = std::min(sceneMax[k], m_worldPos[k] + m_mortonRange);
        if (mortonAabb[k] >= mortonAabb[k + 3]) { // whole scene is outside the cap
            mortonAabb[k]     = sceneMin[k];
            mortonAabb[k + 3] = sceneMax[k];
        }
    }

    // Upload triangle data, sort-step params, and Morton AABB as the first encoder
    // commands.  The Morton AABB replaces cs_reduce: it's the ±m_mortonRange clamp
    // computed above, isolating the relevant scene geometry from distant outliers.
    {
        const uint64_t triBytes = uint64_t(m_triCount) * 80u;
        WGPUBuffer stag = make_upload_buf(device, m_pendingTriData.data(), triBytes);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, stag, 0, m_triInputBuf, 0, triBytes);
        wgpuBufferRelease(stag);
    }
    {
        WGPUBuffer stag = make_upload_buf(device, stepsBlob.data(), stepsBlob.size());
        wgpuCommandEncoderCopyBufferToBuffer(encoder, stag, 0, m_sortStepsBuf, 0, stepsBlob.size());
        wgpuBufferRelease(stag);
    }
    {
        // Upload robust Morton AABB to b_scene — cs_reduce is skipped.
        WGPUBuffer stag = make_upload_buf(device, mortonAabb, sizeof(mortonAabb));
        wgpuCommandEncoderCopyBufferToBuffer(encoder, stag, 0, m_sceneAabbBuf, 0, sizeof(mortonAabb));
        wgpuBufferRelease(stag);
    }

    WGPUBindGroupEntry bge[8] = {};
    bge[0] = { .binding = 0, .buffer = m_triInputBuf,  .size = WGPU_WHOLE_SIZE };
    bge[1] = { .binding = 1, .buffer = m_aabbBuf,      .size = WGPU_WHOLE_SIZE };
    bge[2] = { .binding = 2, .buffer = m_sceneAabbBuf, .size = WGPU_WHOLE_SIZE };
    bge[3] = { .binding = 3, .buffer = m_mortonBuf,    .size = WGPU_WHOLE_SIZE };
    bge[4] = { .binding = 4, .buffer = m_indicesBuf,   .size = WGPU_WHOLE_SIZE };
    bge[5] = { .binding = 5, .buffer = m_nodeBuf,      .size = WGPU_WHOLE_SIZE };
    bge[6] = { .binding = 6, .buffer = m_triBuf,       .size = WGPU_WHOLE_SIZE };
    bge[7] = { .binding = 7, .buffer = m_sortStepsBuf, .size = 16              };

    WGPUBindGroupDescriptor bgd{};
    bgd.layout     = m_bgl;
    bgd.entryCount = 8;
    bgd.entries    = bge;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgd);

    const auto t0 = std::chrono::steady_clock::now();

    // Record all 7 BVH compute passes directly into the provided encoder.
    // The caller (post-render callback) also records the AO pass into the same
    // encoder, so the entire BVH-build + copy + AO sequence lives in one
    // command buffer.  Dawn inserts the correct D3D12 resource-state barriers
    // at every pass boundary within a single command buffer — UAV→UAV between
    // compute passes, UAV→COPY_SRC before the copy, and COPY_DST→storage-read
    // before the AO pass — without any cross-submission tracking uncertainty.
    auto dispatch = [&](WGPUComputePipeline pipe, uint32_t groups_x, uint32_t off) {
        if (groups_x == 0) return;
        WGPUComputePassDescriptor pd{};
        WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        wgpuComputePassEncoderSetPipeline(cp, pipe);
        wgpuComputePassEncoderSetBindGroup(cp, 0, bg, 1, &off);
        wgpuComputePassEncoderDispatchWorkgroups(cp, groups_x, 1, 1);
        wgpuComputePassEncoderEnd(cp);
        wgpuComputePassEncoderRelease(cp);
    };

    dispatch(m_pipeBounds,  g256_n, 0u);
    // cs_reduce skipped — robust Morton AABB already uploaded from CPU above.
    dispatch(m_pipeMorton,  (np + 255u) / 256u, 0u);
    for (uint32_t step = 0; step < sortStepCount; ++step)
        dispatch(m_pipeBitonic, g256_p, (step + 1u) * kStride);
    dispatch(m_pipeLeaves, g256_n, 0u);
    dispatch(m_pipeLbvh,   g256_i, 0u);
    dispatch(m_pipeAabb,   g256_i, 0u);

    const auto t1 = std::chrono::steady_clock::now();

    wgpuBindGroupRelease(bg);

    m_lastStats = {n, 2u * n - 1u,
                   std::chrono::duration<float, std::milli>(t1 - t0).count(),
                   {sceneMin[0], sceneMin[1], sceneMin[2]},
                   {sceneMax[0], sceneMax[1], sceneMax[2]},
                   {mortonAabb[0], mortonAabb[1], mortonAabb[2]},
                   {mortonAabb[3], mortonAabb[4], mortonAabb[5]}};
}

} // namespace dusk::rtao
