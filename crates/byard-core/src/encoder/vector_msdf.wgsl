// VectorMSDF pipeline (RFC-0009 §1, the fifth pipeline). Samples a multi-channel
// signed-distance-field (MSDF) array-texture atlas to draw crisp, resolution-
// independent monochrome glyphs at any scale.
//
// Two byard corrections to the RFC draft are baked in here:
//   §2-D — no `discard` by default: the fragment outputs premultiplied alpha and
//          relies on blending, so a transparent fragment costs nothing on the
//          TBDR mobile GPUs RFC-0001 §3.1 targets (a `discard` would defeat
//          early-Z there).
//   §2-E — anti-aliasing from the baked `px_range` and the screen-space
//          derivative of the sampled UV, not an unspecified helper.

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
};

struct InstanceInput {
    // (u0, v0, u1, v1) UV rect within the atlas layer.
    @location(1) atlas_uv: vec4<f32>,
    // (x, y, width, height) screen rect in logical pixels.
    @location(2) screen_rect: vec4<f32>,
    @location(3) color: vec4<f32>,
    @location(4) px_range: f32,
    @location(5) atlas_layer: u32,
    // Draw-order NDC-z (RFC-0011 cross-pass paint order).
    @location(6) depth: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) px_range: f32,
    @location(3) @interpolate(flat) layer: u32,
};

@group(0) @binding(0) var<uniform> viewport_size: vec2<f32>;
@group(1) @binding(0) var t_msdf: texture_2d_array<f32>;
@group(1) @binding(1) var s_msdf: sampler;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    let wh = instance.screen_rect.zw;
    let world_pos = instance.screen_rect.xy + vertex.quad_pos * wh;
    out.position = vec4<f32>(
        (world_pos.x / viewport_size.x) * 2.0 - 1.0,
        1.0 - (world_pos.y / viewport_size.y) * 2.0,
        instance.depth,
        1.0
    );

    // Map the [0,1] quad to the glyph's UV rect within the atlas.
    out.uv = mix(instance.atlas_uv.xy, instance.atlas_uv.zw, vertex.quad_pos);
    out.color = instance.color;
    out.px_range = instance.px_range;
    out.layer = instance.atlas_layer;
    return out;
}

// Median of three channels — the MSDF reconstruction of the true signed distance
// (Chlumský 2015). Sharp corners survive because the median of the three field
// channels keeps the discontinuity a single triangle pair cannot represent.
fn median3(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // §2-E screen-space anti-aliasing: convert the baked distance range (atlas
    // texels per em) into a screen-pixel coverage width via the UV derivative.
    let tex_dims = vec2<f32>(textureDimensions(t_msdf, 0));
    let unit_range = vec2<f32>(in.px_range, in.px_range) / tex_dims;
    let screen_tex_size = vec2<f32>(1.0, 1.0) / fwidth(in.uv);
    let screen_px_range = max(0.5 * dot(unit_range, screen_tex_size), 1.0);

    let msdf = textureSample(t_msdf, s_msdf, in.uv, i32(in.layer)).rgb;
    let sd = median3(msdf.r, msdf.g, msdf.b);
    let screen_dist = screen_px_range * (sd - 0.5);
    let coverage = clamp(screen_dist + 0.5, 0.0, 1.0);

    // §2-D premultiplied-alpha output; a zero-coverage fragment blends to nothing
    // without a `discard`.
    let a = coverage * in.color.a;
    return vec4<f32>(in.color.rgb * a, a);
}
