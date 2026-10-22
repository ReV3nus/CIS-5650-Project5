// `real` (f32 or f16) and unpack_real() are prepended in gaussian-preprocess.ts

const SH_C0 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<real,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<real,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

const ZNEAR_CULL = 0.2;
const ZFAR = 100.0;     // same as camera.ts
const TILE_SIZE = 16.0;

override workgroupSize: u32;
override sortKeyPerThread: u32;
override preprocessWorkgroupSize: u32 = 256u;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
    frustum_culling: u32,
    _pad: u32,
}

struct TileParams {
    tiles_x: u32,
    tiles_y: u32,
    depth_bits: u32,
    capacity: u32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    center: vec2<f32>,      // NDC
    conic_xy: vec2<f32>,    // inverse 2D cov in pixels (a, b, c)
    conic_z: f32,
    size: u32,              // half extent in NDC, packed f16
    color: array<u32,2>,    // (r, g), (b, opacity), packed f16
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> settings: RenderSettings;
@group(0) @binding(2)
var<uniform> tile_params: TileParams;

@group(1) @binding(0)
var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh_buffer : array<u32>;
@group(1) @binding(2)
var<storage, read_write> splats : array<Splat>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<real> {
    // 16 coefs x 3 channels of f16 per gaussian
    let base = splat_idx * 48u + c_idx * 3u;
    let w = base / 2u;
    let a = unpack_real(sh_buffer[w]);
    let b = unpack_real(sh_buffer[w + 1u]);
    if (base & 1u) == 0u {
        return vec3<real>(a.x, a.y, b.x);
    }
    return vec3<real>(a.y, b.x, b.y);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<real>, v_idx: u32, sh_deg: u32) -> vec3<real> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<real>(0.), result);
}

fn sigmoid(x: real) -> real {
    return 1.0 / (1.0 + exp(-x));
}

struct Projected {
    visible: bool,
    view_z: f32,
    center_px: vec2<f32>,
    radius: f32,
}

fn process_gaussian(idx: u32) -> Projected {
    var res: Projected;
    res.visible = false;

    let g = gaussians[idx];
    let a = unpack2x16float(g.pos_opacity[0]);
    let b = unpack2x16float(g.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.0);

    let view_pos = camera.view * pos;
    let tz = view_pos.z;
    if tz < ZNEAR_CULL || tz > ZFAR {
        return res;
    }
    let clip = camera.proj * view_pos;
    let ndc = clip.xy / clip.w;
    // 1.2x margin, splats centered just off screen can still cover it
    if settings.frustum_culling != 0u && (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2) {
        return res;
    }

    // Sigma = R S S^T R^T
    let r01 = unpack_real(g.rot[0]);
    let r23 = unpack_real(g.rot[1]);
    let q = normalize(vec4<real>(r01.x, r01.y, r23.x, r23.y));   // (w, x, y, z)
    let qw = q.x;
    let qx = q.y;
    let qy = q.z;
    let qz = q.w;
    let R = mat3x3<real>(
        vec3<real>(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy)),
        vec3<real>(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx)),
        vec3<real>(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)),
    );
    let s01 = unpack_real(g.scale[0]);
    let s23 = unpack_real(g.scale[1]);
    let s = exp(vec3<real>(s01.x, s01.y, s23.x)) * real(settings.gaussian_scaling);
    let M_r = mat3x3<real>(R[0] * s.x, R[1] * s.y, R[2] * s.z);
    // Sigma can overflow f16, so the rest stays in f32
    let M = mat3x3<f32>(M_r);
    let Sigma = M * transpose(M);

    // cov2d = J W Sigma W^T J^T
    let focal = camera.focal;
    let lim = 1.3 * 0.5 * camera.viewport / focal;
    let tx = clamp(view_pos.x / tz, -lim.x, lim.x) * tz;
    let ty = clamp(view_pos.y / tz, -lim.y, lim.y) * tz;
    let J = mat3x3<f32>(
        vec3<f32>(focal.x / tz, 0.0, 0.0),
        vec3<f32>(0.0, focal.y / tz, 0.0),
        vec3<f32>(-focal.x * tx / (tz * tz), -focal.y * ty / (tz * tz), 0.0),
    );
    let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let T = J * W;
    let cov = T * Sigma * transpose(T);

    // low-pass, at least ~1px wide
    let c00 = cov[0][0] + 0.3;
    let c01 = cov[0][1];
    let c11 = cov[1][1] + 0.3;
    let det = c00 * c11 - c01 * c01;
    if det <= 0.0 {
        return res;
    }
    let conic = vec3<f32>(c11, -c01, c00) / det;

    // 3 sigma of the larger eigenvalue
    let mid = 0.5 * (c00 + c11);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(lambda1));
    if radius <= 0.0 {
        return res;
    }

    let cam_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos.xyz - cam_pos);
    let color = vec3<f32>(computeColorFromSH(vec3<real>(dir), idx, u32(settings.sh_deg)));
    let opacity = f32(sigmoid(real(b.y)));

    let size = vec2<f32>(radius) * 2.0 / camera.viewport;

    splats[idx] = Splat(
        ndc,
        conic.xy,
        conic.z,
        pack2x16float(size),
        array<u32,2>(pack2x16float(color.rg), pack2x16float(vec2<f32>(color.b, opacity))),
    );

    res.visible = true;
    res.view_z = tz;
    res.center_px = vec2<f32>((ndc.x * 0.5 + 0.5) * camera.viewport.x, (0.5 - ndc.y * 0.5) * camera.viewport.y);
    res.radius = radius;
    return res;
}

// 2D dispatch, large scenes need more than 65535 workgroups
fn thread_index(gid: vec3<u32>, wgs: vec3<u32>) -> u32 {
    return gid.y * wgs.x * preprocessWorkgroupSize + gid.x;
}

@compute @workgroup_size(preprocessWorkgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = thread_index(gid, wgs);
    if idx >= arrayLength(&gaussians) {
        return;
    }

    let p = process_gaussian(idx);
    if !p.visible {
        return;
    }

    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    let slot = atomicAdd(&sort_infos.keys_size, 1u);
    sort_indices[slot] = idx;
    // sort is ascending, so flip depth to get back to front. positive floats sort fine as u32
    sort_depths[slot] = bitcast<u32>(ZFAR - p.view_z);
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if slot % keys_per_dispatch == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}

// one key per overlapped tile: tile_id << depth_bits | depth
@compute @workgroup_size(preprocessWorkgroupSize,1,1)
fn preprocess_tiles(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = thread_index(gid, wgs);
    if idx >= arrayLength(&gaussians) {
        return;
    }

    let p = process_gaussian(idx);
    if !p.visible {
        return;
    }

    let tiles = vec2<f32>(f32(tile_params.tiles_x), f32(tile_params.tiles_y));
    let rect_min = vec2<u32>(clamp(floor((p.center_px - p.radius) / TILE_SIZE), vec2<f32>(0.0), tiles));
    let rect_max = vec2<u32>(clamp(ceil((p.center_px + p.radius) / TILE_SIZE), vec2<f32>(0.0), tiles));
    let count = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
    if count == 0u {
        return;
    }
    // full, keep the counter from wrapping
    if atomicLoad(&sort_infos.keys_size) >= tile_params.capacity {
        return;
    }

    let depth_bits = tile_params.depth_bits;
    let depth_max = (1u << depth_bits) - 1u;
    let depth_key = u32(clamp(p.view_z / ZFAR, 0.0, 1.0) * f32(depth_max));

    var slot = atomicAdd(&sort_infos.keys_size, count);
    for (var y = rect_min.y; y < rect_max.y; y++) {
        for (var x = rect_min.x; x < rect_max.x; x++) {
            if slot >= tile_params.capacity {
                return;
            }
            let tile_id = y * tile_params.tiles_x + x;
            sort_depths[slot] = (tile_id << depth_bits) | depth_key;
            sort_indices[slot] = idx;
            slot++;
        }
    }
}
