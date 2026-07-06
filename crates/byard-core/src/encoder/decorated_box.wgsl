// DecoratedBox pipeline (M21, RFC-0001 §3.1): a rounded rectangle with an
// optional inner border, a blurred drop shadow, and an overall opacity. Plain
// solid fills (no border/shadow/opacity) stay on the SolidBox pipeline; this one
// is used only when the compiler promotes a box via `RenderFrame::push_decorated`.

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
};

struct InstanceInput {
    @location(1) rect: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) radii: vec4<f32>,
    @location(4) border_color: vec4<f32>,
    @location(5) shadow_color: vec4<f32>,
    // (border_width, shadow_dx, shadow_dy, shadow_blur)
    @location(6) params: vec4<f32>,
    // (opacity, _, _, _)
    @location(7) misc: vec4<f32>,
    // Paint-time transform (RFC-0011); identity is a free no-op below.
    // `opacity` isn't part of this block — `misc.x` above stays authoritative.
    @location(8) t_translate: vec2<f32>,
    @location(9) t_scale: vec2<f32>,
    @location(10) t_rotate: f32,
    @location(11) t_origin: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) half_size: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) radii: vec4<f32>,
    @location(4) border_color: vec4<f32>,
    @location(5) shadow_color: vec4<f32>,
    @location(6) params: vec4<f32>,
    @location(7) misc: vec4<f32>,
};

@group(0) @binding(0) var<uniform> viewport_size: vec2<f32>;

const QUAD_PADDING: f32 = 2.0;

/// Applies a paint-time transform (RFC-0011) to a world-space (logical-pixel)
/// position: rotate + scale about `origin`, then translate. Identity inputs
/// collapse to `world` unchanged.
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

    // Inflate the quad to cover the shape, its anti-alias fringe, and the full
    // shadow extent (offset + blur + positive spread), so no shadow fragment is
    // clipped away. `misc.z` is the shadow spread.
    let shadow_margin = abs(vec2<f32>(instance.params.y, instance.params.z))
        + vec2<f32>(instance.params.w)
        + vec2<f32>(max(instance.misc.z, 0.0));
    let margin = vec2<f32>(QUAD_PADDING) + shadow_margin;
    let padded = vec2<f32>(w, h) + margin * 2.0;

    out.local_pos = (vertex.quad_pos - 0.5) * padded;
    let world_pos = instance.rect.xy - margin + vertex.quad_pos * padded;

    let transformed = apply_transform(
        world_pos,
        instance.t_translate,
        instance.t_scale,
        instance.t_rotate,
        instance.t_origin,
    );

    // misc.y carries the draw-order depth (NDC-z); the encoder writes it per
    // instance so decorated boxes honour global paint order against solids/text.
    out.position = vec4<f32>(
        (transformed.x / viewport_size.x) * 2.0 - 1.0,
        1.0 - (transformed.y / viewport_size.y) * 2.0,
        instance.misc.y,
        1.0
    );

    out.color = instance.color;
    out.radii = instance.radii;
    out.border_color = instance.border_color;
    out.shadow_color = instance.shadow_color;
    out.params = instance.params;
    out.misc = instance.misc;
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
    let border_width = in.params.x;
    let shadow_offset = vec2<f32>(in.params.y, in.params.z);
    let shadow_blur = in.params.w;
    let shadow_spread = in.misc.z;
    let opacity = in.misc.x;

    // ── Drop shadow (drawn beneath the surface) ───────────────────────────
    // `spread` grows/shrinks the shadow shape (and its corner radii) before the
    // blur, matching CSS `box-shadow` spread; clamp to non-negative extents.
    var shadow_a = 0.0;
    if (in.shadow_color.a > 0.0 && (shadow_blur > 0.0 || shadow_spread != 0.0
        || abs(shadow_offset.x) > 0.0 || abs(shadow_offset.y) > 0.0)) {
        let s_half = max(in.half_size + vec2<f32>(shadow_spread), vec2<f32>(0.0));
        let s_radii = max(in.radii + vec4<f32>(shadow_spread), vec4<f32>(0.0));
        let sdist = sd_rounded_box(in.local_pos - shadow_offset, s_half, s_radii);
        let soft = max(shadow_blur, 0.5);
        shadow_a = (1.0 - smoothstep(0.0, soft, sdist)) * in.shadow_color.a;
    }

    // ── Surface fill + inner border ───────────────────────────────────────
    let fdist = sd_rounded_box(in.local_pos, in.half_size, in.radii);
    let edge_softness = max(length(vec2<f32>(dpdx(fdist), dpdy(fdist))), 1e-5);
    let fill_cov = smoothstep(edge_softness, 0.0, fdist);

    var surface = in.color;
    if (border_width > 0.0 && fdist > -border_width) {
        surface = in.border_color;
    }

    let a_top = fill_cov * surface.a;
    let a_bot = shadow_a * (1.0 - a_top);
    let out_a = a_top + a_bot;
    if (out_a <= 0.0) {
        discard;
    }
    let out_rgb = (surface.rgb * a_top + in.shadow_color.rgb * a_bot) / out_a;
    return vec4<f32>(out_rgb, out_a * opacity);
}
