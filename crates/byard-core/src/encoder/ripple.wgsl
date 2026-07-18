// Ripple pipeline (RFC-0023): the Material ink reveal. One instance is one
// expanding, fading circle — centred on the tap point, clipped to its
// element's rounded rect, composited *above* the element background and
// *below* its children (the instance's draw-order depth, stamped between the
// two by the evaluator's emission order, resolves that against the shared
// depth buffer).
//
// The logic thread samples the expansion/fade each tick (RFC-0010 model as
// landed) and re-emits `params.z` (radius) / `params.w` (fade alpha); this
// shader only rasterises the current circle analytically. Blending is
// premultiplied-alpha "over", so a dark ink darkens a light surface (pure
// addition could only ever brighten) while simultaneous ripples from rapid
// taps still accumulate where their circles overlap (RFC-0023 §1).

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
};

struct InstanceInput {
    // Element rect (x, y, w, h) in logical px — quad geometry and clip bounds.
    @location(1) rect: vec4<f32>,
    // (center_x, center_y, radius, fade_alpha); centre in absolute logical px.
    @location(2) params: vec4<f32>,
    // Ink colour; `a` is the ink's own peak alpha.
    @location(3) color: vec4<f32>,
    // Per-corner clip radii (tl, tr, br, bl) — the element's border radii.
    @location(4) radii: vec4<f32>,
    // Paint-time transform (RFC-0011); identity is a free no-op.
    @location(5) t_translate: vec2<f32>,
    @location(6) t_scale: vec2<f32>,
    @location(7) t_rotate: f32,
    @location(8) t_origin: vec2<f32>,
    // Draw-order depth (NDC-z), stamped by `RenderFrame::push_ripple`.
    @location(9) depth: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) half_size: vec2<f32>,
    @location(2) center_local: vec2<f32>,
    @location(3) radius_alpha: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) radii: vec4<f32>,
};

@group(0) @binding(0) var<uniform> viewport_size: vec2<f32>;

// Anti-alias fringe head-room around the quad, matching `decorated_box.wgsl`.
const QUAD_PADDING: f32 = 2.0;

// Applies a paint-time transform (RFC-0011) to a world-space (logical-pixel)
// position: rotate + scale about `origin`, then translate — identical to
// `decorated_box.wgsl`'s helper, so a transformed element carries its ink
// with it.
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
        instance.depth,
        1.0
    );

    // The tap point, in the same rect-centred local space as `local_pos` —
    // the fragment circle SDF is then transform-agnostic (the transform moves
    // the whole quad, ink included).
    let rect_center = instance.rect.xy + out.half_size;
    out.center_local = instance.params.xy - rect_center;
    out.radius_alpha = instance.params.zw;
    out.color = instance.color;
    out.radii = instance.radii;
    return out;
}

// Rounded-box SDF, shared shape with `decorated_box.wgsl` — the clip must
// match the element's own outline exactly or the ink visibly bleeds past (or
// falls short of) a rounded corner.
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
    // Clip to the element's rounded rect (RFC-0023: always, no opt-out).
    let box_dist = sd_rounded_box(in.local_pos, in.half_size, in.radii);
    let box_soft = max(length(vec2<f32>(dpdx(box_dist), dpdy(box_dist))), 1e-5);
    let box_cov = smoothstep(box_soft, 0.0, box_dist);

    // The expanding ink circle, anti-aliased over the same screen-space fringe.
    let circle_dist = length(in.local_pos - in.center_local) - in.radius_alpha.x;
    let circle_cov = smoothstep(box_soft, 0.0, circle_dist);

    let a = box_cov * circle_cov * in.color.a * in.radius_alpha.y;
    if (a <= 0.0) {
        discard;
    }
    // Premultiplied output for the PREMULTIPLIED_ALPHA_BLENDING "over" state:
    // ink composites onto light and dark surfaces alike, and rapid taps pool
    // where circles overlap.
    return vec4<f32>(in.color.rgb * a, a);
}
