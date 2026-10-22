struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) center_px: vec2<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
};

// same layout as preprocess.wgsl
struct Splat {
    center: vec2<f32>,
    conic_xy: vec2<f32>,
    conic_z: f32,
    size: u32,
    color: array<u32,2>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> splats: array<Splat>;
@group(0) @binding(2)
var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let splat = splats[sort_indices[instance_index]];
    let size = unpack2x16float(splat.size);

    // strip order: (-1,-1) (1,-1) (-1,1) (1,1)
    let corner = vec2<f32>(f32(vertex_index & 1u), f32((vertex_index >> 1u) & 1u)) * 2.0 - 1.0;

    var out: VertexOutput;
    out.position = vec4<f32>(splat.center + corner * size, 0.0, 1.0);
    out.center_px = vec2<f32>(
        (splat.center.x * 0.5 + 0.5) * camera.viewport.x,
        (0.5 - splat.center.y * 0.5) * camera.viewport.y,
    );
    out.conic = vec3<f32>(splat.conic_xy, splat.conic_z);
    let rg = unpack2x16float(splat.color[0]);
    let ba = unpack2x16float(splat.color[1]);
    out.color = vec4<f32>(rg, ba);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // pixel y and camera y both point down, so no flip needed
    let d = in.position.xy - in.center_px;
    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;
    if power > 0.0 {
        discard;
    }
    let alpha = min(0.99, in.color.a * exp(power));
    if alpha < 1.0 / 255.0 {
        discard;
    }
    // premultiplied
    return vec4<f32>(in.color.rgb * alpha, alpha);
}
