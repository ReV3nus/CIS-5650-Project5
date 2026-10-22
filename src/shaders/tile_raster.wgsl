// tile rasterizer, same structure as the original 3DGS CUDA code

override sortKeysPerDispatch: u32 = 3840u;

const TILE_SIZE = 16u;
const BATCH = 256u;     // one gaussian per thread

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct TileParams {
    tiles_x: u32,
    tiles_y: u32,
    depth_bits: u32,
    capacity: u32,
}

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct DispatchIndirect {
    x: u32,
    y: u32,
    z: u32,
}

// same layout as preprocess.wgsl
struct Splat {
    center: vec2<f32>,
    conic_xy: vec2<f32>,
    conic_z: f32,
    size: u32,
    color: array<u32,2>,
};

// each entry point only uses some of these (auto layout)
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> tile_params: TileParams;
@group(0) @binding(2) var<storage, read_write> sort_infos: SortInfos;
@group(0) @binding(3) var<storage, read_write> sort_dispatch: DispatchIndirect;
@group(0) @binding(4) var<storage, read_write> ranges_dispatch: DispatchIndirect;
@group(0) @binding(5) var<storage, read> sorted_keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> ranges: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(8) var<storage, read> splats: array<Splat>;
@group(0) @binding(9) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(10) var blit_src: texture_2d<f32>;
@group(0) @binding(11) var<storage, read_write> requested_keys: u32;   // unclamped count, read back on the CPU

// clamp the key count and set up the indirect dispatches
@compute @workgroup_size(1)
fn finalize() {
    requested_keys = sort_infos.keys_size;
    let n = min(sort_infos.keys_size, tile_params.capacity);
    sort_infos.keys_size = n;

    sort_dispatch.x = (n + sortKeysPerDispatch - 1u) / sortKeysPerDispatch;
    sort_dispatch.y = 1u;
    sort_dispatch.z = 1u;
    // only pad up to the blocks we actually dispatch
    sort_infos.padded_size = sort_dispatch.x * sortKeysPerDispatch;

    let groups = (n + 255u) / 256u;
    let gx = min(groups, 65535u);
    ranges_dispatch.x = gx;
    ranges_dispatch.y = select(1u, (groups + gx - 1u) / max(gx, 1u), gx > 0u);
    ranges_dispatch.z = 1u;
}

// [start, end) of each tile in the sorted keys
@compute @workgroup_size(256)
fn identify_ranges(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * nwg.x * 256u + gid.x;
    let n = sort_infos.keys_size;
    if i >= n {
        return;
    }
    let depth_bits = tile_params.depth_bits;
    let tile = sorted_keys[i] >> depth_bits;
    if i == 0u {
        ranges[tile].x = 0u;
    } else {
        let prev = sorted_keys[i - 1u] >> depth_bits;
        if prev != tile {
            ranges[prev].y = i;
            ranges[tile].x = i;
        }
    }
    if i == n - 1u {
        ranges[tile].y = n;
    }
}

var<workgroup> ws_range: vec2<u32>;
var<workgroup> ws_done_count: atomic<u32>;
var<workgroup> ws_all_done: u32;
var<workgroup> ws_center: array<vec2<f32>, BATCH>;
var<workgroup> ws_conic_opacity: array<vec4<f32>, BATCH>;
var<workgroup> ws_color: array<vec3<f32>, BATCH>;

// one workgroup per tile, front to back
@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn rasterize(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lindex: u32,
) {
    let tile_id = wid.y * tile_params.tiles_x + wid.x;
    if lindex == 0u {
        ws_range = ranges[tile_id];
        atomicStore(&ws_done_count, 0u);
    }
    let range = workgroupUniformLoad(&ws_range);

    let viewport = camera.viewport;
    let pix = wid.xy * TILE_SIZE + lid.xy;
    let inside = f32(pix.x) < viewport.x && f32(pix.y) < viewport.y;
    let pixf = vec2<f32>(pix) + 0.5;

    var T = 1.0;
    var C = vec3<f32>(0.0);
    var done = !inside;
    if done {
        atomicAdd(&ws_done_count, 1u);
    }

    let total = select(0u, range.y - range.x, range.y > range.x);
    let rounds = (total + BATCH - 1u) / BATCH;
    for (var r = 0u; r < rounds; r++) {
        // whole tile saturated
        workgroupBarrier();
        if lindex == 0u {
            ws_all_done = select(0u, 1u, atomicLoad(&ws_done_count) == BATCH);
        }
        if workgroupUniformLoad(&ws_all_done) == 1u {
            break;
        }

        // load the next batch into shared memory
        let k = range.x + r * BATCH + lindex;
        if k < range.y {
            let s = splats[sorted_indices[k]];
            ws_center[lindex] = vec2<f32>((s.center.x * 0.5 + 0.5) * viewport.x, (0.5 - s.center.y * 0.5) * viewport.y);
            let ba = unpack2x16float(s.color[1]);
            ws_conic_opacity[lindex] = vec4<f32>(s.conic_xy, s.conic_z, ba.y);
            ws_color[lindex] = vec3<f32>(unpack2x16float(s.color[0]), ba.x);
        }
        workgroupBarrier();

        let count = min(BATCH, total - r * BATCH);
        for (var j = 0u; j < count; j++) {
            if done {
                break;
            }
            let d = pixf - ws_center[j];
            let co = ws_conic_opacity[j];
            let power = -0.5 * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            if power > 0.0 {
                continue;
            }
            let alpha = min(0.99, co.w * exp(power));
            if alpha < 1.0 / 255.0 {
                continue;
            }
            let next_T = T * (1.0 - alpha);
            if next_T < 0.0001 {
                done = true;
                atomicAdd(&ws_done_count, 1u);
                break;
            }
            C += ws_color[j] * alpha * T;
            T = next_T;
        }
    }

    if inside {
        textureStore(out_tex, pix, vec4<f32>(C, 1.0));
    }
}

@vertex
fn blit_vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // fullscreen triangle
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn blit_fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(blit_src, vec2<i32>(p.xy), 0);
}
