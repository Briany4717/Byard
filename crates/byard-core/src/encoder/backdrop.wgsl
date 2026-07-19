// Backdrop composite pipeline (RFC-0023 §2): draws one frosted-glass pane
// inside the main UI pass, sampling the blurred copy of the scene behind it.
//
// The UV is derived from the *framebuffer position* of each fragment mapped
// into the copied region — "what is behind this pixel" is by definition the
// same physical pixel of the colour target, so the sample stays perfectly
// aligned regardless of the pane's transform. The fragment then boosts
// saturation (the iOS vibrancy look), blends `backdrop_tint` on top, and
// clips everything to the element's rounded rect with the same SDF as
// `DecoratedBox`.

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
};

struct InstanceInput {
    // Element rect (x, y, w, h) in logical px.
    @location(1) rect: vec4<f32>,
    // Per-corner clip radii (tl, tr, br, bl).
    @location(2) radii: vec4<f32>,
    // backdrop_tint colour; `a = 0` disables.
    @location(3) tint: vec4<f32>,
    // (saturation, opacity, depth, unused)
    @location(4) params: vec4<f32>,
    // Copied region mapping: (origin_x, origin_y, 1/width, 1/height), all in
    // physical pixels of the colour target.
    @location(5) region: vec4<f32>,
    // Paint-time transform (RFC-0011).
    @location(6) t_translate: vec2<f32>,
    @location(7) t_scale: vec2<f32>,
    @location(8) t_rotate: f32,
    @location(9) t_origin: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) half_size: vec2<f32>,
    @location(2) radii: vec4<f32>,
    @location(3) tint: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) region: vec4<f32>,
};

@group(0) @binding(0) var<uniform> viewport_size: vec2<f32>;
@group(1) @binding(0) var blurred_tex: texture_2d<f32>;
@group(1) @binding(1) var blurred_smp: sampler;

const QUAD_PADDING: f32 = 2.0;

// Identical helper to `decorated_box.wgsl` — a transformed pane carries its
// glass with it.
fn apply_transform(
    world: vec2<f32>,
    translate: vec2<f32>,
    scale: vec2<f32>,
    rotate: f32,
    origin: vec2<f32>,
) -> vec2<f32> {
    let p = world - origin;
    let scaled = vec2<f32>(p.x * scale.x, p.y * scale.y);
    let c = cos(rotate);
    let s = sin(rotate);
    let rotated = vec2<f32>(scaled.x * c - scaled.y * s, scaled.x * s + scaled.y * c);
    return rotated + origin + translate;
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    let w = instance.rect.z;
    let h = instance.rect.w;
    out.half_size = vec2<f32>(w, h) * 0.5;

    let padded = vec2<f32>(w, h) + vec2<f32>(QUAD_PADDING) * 2.0;
    out.local_pos = (vertex.quad_pos - 0.5) * padded;
    let world_pos = instance.rect.xy - vec2<f32>(QUAD_PADDING) + vertex.quad_pos * padded;

    let transformed = apply_transform(
        world_pos,
        instance.t_translate,
        instance.t_scale,
        instance.t_rotate,
        instance.t_origin,
    );

    out.position = vec4<f32>(
        (transformed.x / viewport_size.x) * 2.0 - 1.0,
        1.0 - (transformed.y / viewport_size.y) * 2.0,
        instance.params.z,
        1.0
    );

    out.radii = instance.radii;
    out.tint = instance.tint;
    out.params = instance.params;
    out.region = instance.region;
    return out;
}

// Rounded-box SDF, shared shape with `decorated_box.wgsl` / `ripple.wgsl`.
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
    // Clip to the element's rounded rect.
    let dist = sd_rounded_box(in.local_pos, in.half_size, in.radii);
    let soft = max(length(vec2<f32>(dpdx(dist), dpdy(dist))), 1e-5);
    let coverage = smoothstep(soft, 0.0, dist);
    if (coverage <= 0.0) {
        discard;
    }

    // "What is behind this pixel": the fragment's own framebuffer position
    // mapped into the copied region (bilinear upscale from the downsampled
    // blur — the smoothing the RFC's resolution answer relies on).
    let uv = (in.position.xy - in.region.xy) * in.region.zw;
    var color = textureSampleLevel(blurred_tex, blurred_smp, uv, 0.0).rgb;

    // Saturation boost (RFC-0023 `blur_saturation`, the vibrancy cast).
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    color = mix(vec3<f32>(luma), color, in.params.x);

    // `backdrop_tint` blended on top of the blurred sample.
    color = mix(color, in.tint.rgb, in.tint.a);

    return vec4<f32>(color, coverage * in.params.y);
}
