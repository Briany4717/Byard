// TextureSampler pipeline (M21, RFC-0001 §3.1): a UV-mapped, optionally
// rounded quad that samples a decoded image. `fit` (fill/contain/cover/none) is
// resolved on the CPU into a UV transform so the shader stays a plain sampler.

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
};

struct InstanceInput {
    @location(1) rect: vec4<f32>,
    @location(2) radii: vec4<f32>,
    // (uv_scale_x, uv_scale_y, uv_offset_x, uv_offset_y) — the `fit` transform.
    @location(3) uv_xform: vec4<f32>,
    // (opacity, _, _, _)
    @location(4) misc: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) local_pos: vec2<f32>,
    @location(2) half_size: vec2<f32>,
    @location(3) radii: vec4<f32>,
    @location(4) opacity: f32,
};

@group(0) @binding(0) var<uniform> viewport_size: vec2<f32>;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    let w = instance.rect.z;
    let h = instance.rect.w;
    out.half_size = vec2<f32>(w, h) * 0.5;
    out.local_pos = (vertex.quad_pos - 0.5) * vec2<f32>(w, h);

    let world_pos = instance.rect.xy + vertex.quad_pos * vec2<f32>(w, h);
    // misc.y carries the draw-order depth (NDC-z), written per instance by the
    // encoder so images honour global paint order against solids/decorated/text.
    out.position = vec4<f32>(
        (world_pos.x / viewport_size.x) * 2.0 - 1.0,
        1.0 - (world_pos.y / viewport_size.y) * 2.0,
        instance.misc.y,
        1.0
    );

    // Apply the CPU-computed fit transform to the [0,1] quad UVs.
    out.uv = vertex.quad_pos * instance.uv_xform.xy + instance.uv_xform.zw;
    out.radii = instance.radii;
    out.opacity = instance.misc.x;
    return out;
}

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: vec4<f32>) -> f32 {
    var r_corner = r.x;
    if (p.x > 0.0 && p.y < 0.0) { r_corner = r.y; }
    if (p.x > 0.0 && p.y > 0.0) { r_corner = r.z; }
    if (p.x < 0.0 && p.y > 0.0) { r_corner = r.w; }

    let q = abs(p) - b + vec2<f32>(r_corner);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r_corner;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Outside the source image (possible under `contain`/`none`) → transparent.
    if (in.uv.x < 0.0 || in.uv.x > 1.0 || in.uv.y < 0.0 || in.uv.y > 1.0) {
        discard;
    }

    var sample = textureSample(tex, samp, in.uv);

    // Clip to the rounded-rect boundary.
    let dist = sd_rounded_box(in.local_pos, in.half_size, in.radii);
    let edge_softness = max(length(vec2<f32>(dpdx(dist), dpdy(dist))), 1e-5);
    let alpha = smoothstep(edge_softness, 0.0, dist);
    if (alpha <= 0.0) {
        discard;
    }

    return vec4<f32>(sample.rgb, sample.a * alpha * in.opacity);
}
