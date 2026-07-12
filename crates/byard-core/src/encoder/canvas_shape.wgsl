// CanvasShape pipeline (RFC-0020 §2, Tier 1): programmatic 2-D shapes — arcs,
// circles, lines, and (rounded) rectangles — rendered by evaluating each
// shape's closed-form signed-distance function analytically per fragment.
// Resolution-independent, atlas-free, and trivially animatable: an animated
// `sweep`/`dash_offset` is just new per-instance data, never a re-tessellation.
//
// Complex `path(d: …)` commands do not reach this shader — they rasterize
// through the VectorMSDF pipeline (RFC-0020 §2, Tier 2).

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
};

struct InstanceInput {
    // Quad bounds in logical px, already inflated to cover stroke + AA fringe
    // (`CanvasShape::bounds`).
    @location(1) rect: vec4<f32>,
    // Shape params, absolute logical px / radians (layout per kind — see
    // `frame::CANVAS_SHAPE_*`).
    @location(2) params0: vec4<f32>,
    @location(3) params1: vec4<f32>,
    @location(4) stroke_color: vec4<f32>,
    @location(5) fill_color: vec4<f32>,
    // (stroke_width, dash_len, dash_gap, dash_offset)
    @location(6) stroke_dash: vec4<f32>,
    // (opacity, draw-order depth, kind, cap) — kind/cap are small integers
    // carried exactly in f32.
    @location(7) misc: vec4<f32>,
    // Paint-time transform (RFC-0011); identity is a free no-op.
    @location(8) t_translate: vec2<f32>,
    @location(9) t_scale: vec2<f32>,
    @location(10) t_rotate: f32,
    @location(11) t_origin: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // Un-transformed logical-px position: interpolating it across the
    // *transformed* triangle is the standard inverse mapping, so the SDF is
    // evaluated in shape-local space and shapes rotate/scale correctly.
    @location(0) world_pos: vec2<f32>,
    @location(1) params0: vec4<f32>,
    @location(2) params1: vec4<f32>,
    @location(3) stroke_color: vec4<f32>,
    @location(4) fill_color: vec4<f32>,
    @location(5) stroke_dash: vec4<f32>,
    @location(6) misc: vec4<f32>,
};

@group(0) @binding(0) var<uniform> viewport_size: vec2<f32>;

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
// A distance far past any quad extent — "no coverage from this term".
const FAR: f32 = 1e6;

const KIND_ARC: u32 = 0u;
const KIND_CIRCLE: u32 = 1u;
const KIND_LINE: u32 = 2u;
const KIND_RECT: u32 = 3u;

const CAP_BUTT: u32 = 0u;
const CAP_ROUND: u32 = 1u;
const CAP_SQUARE: u32 = 2u;

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

    let world_pos = instance.rect.xy + vertex.quad_pos * instance.rect.zw;
    out.world_pos = world_pos;

    let transformed = apply_transform(
        world_pos,
        instance.t_translate,
        instance.t_scale,
        instance.t_rotate,
        instance.t_origin,
    );

    // misc.y carries the draw-order depth (NDC-z, RFC-0011) so shapes honour
    // global paint order against every other pipeline.
    out.position = vec4<f32>(
        (transformed.x / viewport_size.x) * 2.0 - 1.0,
        1.0 - (transformed.y / viewport_size.y) * 2.0,
        instance.misc.y,
        1.0
    );

    out.params0 = instance.params0;
    out.params1 = instance.params1;
    out.stroke_color = instance.stroke_color;
    out.fill_color = instance.fill_color;
    out.stroke_dash = instance.stroke_dash;
    out.misc = instance.misc;
    return out;
}

// Wraps an angle to [-PI, PI].
fn wrap_angle(a: f32) -> f32 {
    return a - TAU * round(a / TAU);
}

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

// Per-fragment shape evaluation: signed stroke distance (< 0 inside the
// stroked band), signed fill distance (< 0 inside the filled region), and the
// arc-length parameter `t` (logical px along the path) driving dashes.
struct ShapeDist {
    stroke: f32,
    fill: f32,
    t: f32,
};

fn eval_shape(p: vec2<f32>, kind: u32, cap: u32, half_w: f32,
              params0: vec4<f32>, params1: vec4<f32>) -> ShapeDist {
    var out: ShapeDist;
    out.stroke = FAR;
    out.fill = FAR;
    out.t = 0.0;

    if (kind == KIND_CIRCLE) {
        let c = params0.xy;
        let r = max(params0.z, 0.0);
        let rel = p - c;
        let len = length(rel);
        out.stroke = abs(len - r) - half_w;
        out.fill = len - r;
        // Dash parameter: arc length from the +X axis; `dash_offset` sets the
        // phase, so the (arbitrary) zero point is not observable.
        out.t = (wrap_angle(atan2(rel.y, rel.x)) + PI) * r;
        return out;
    }

    if (kind == KIND_ARC) {
        let c = params0.xy;
        let r = max(params0.z, 0.0);
        let start = params0.w;
        let sweep = params1.x;
        let rel = p - c;
        let len = length(rel);
        let ang = atan2(rel.y, rel.x);

        var half_sweep = min(abs(sweep), TAU) * 0.5;
        // A square cap extends the arc by half the stroke width of arc
        // length past each endpoint, then ends flat like a butt cap.
        if (cap == CAP_SQUARE) {
            half_sweep = half_sweep + half_w / max(r, 1e-3);
        }
        let mid = start + sweep * 0.5;
        let delta = wrap_angle(ang - mid);
        let ring = abs(len - r) - half_w;

        if (abs(delta) <= half_sweep) {
            out.stroke = ring;
        } else if (cap == CAP_ROUND) {
            // Round caps: semicircular ends centred on the endpoints.
            let p0 = c + r * vec2<f32>(cos(start), sin(start));
            let p1 = c + r * vec2<f32>(cos(start + sweep), sin(start + sweep));
            out.stroke = min(length(p - p0), length(p - p1)) - half_w;
        } else {
            // Butt/square: flat angular cutoff. `(|delta| - half) * len`
            // approximates the px distance past the end plane, keeping the
            // cut anti-aliased instead of a hard step.
            out.stroke = max(ring, (abs(delta) - half_sweep) * max(len, 1e-3));
        }

        // Fill = the swept circular sector (pie wedge).
        out.fill = max(len - r, (abs(delta) - half_sweep) * max(len, 1e-3));

        // Arc length from the sweep's starting endpoint, following the sweep
        // direction, so dashes march from `start` regardless of sign.
        out.t = (half_sweep + delta * sign(sweep)) * r;
        return out;
    }

    if (kind == KIND_LINE) {
        let a = params0.xy;
        let b = params0.zw;
        let ba = b - a;
        let pa = p - a;
        let len2 = max(dot(ba, ba), 1e-6);
        let seg_len = sqrt(len2);
        let h = clamp(dot(pa, ba) / len2, 0.0, 1.0);
        // Perpendicular distance to the infinite line, and the overshoot past
        // either endpoint along it.
        let u = dot(pa, ba) / seg_len;
        let perp = abs(pa.x * ba.y - pa.y * ba.x) / seg_len;
        let over = max(max(-u, u - seg_len), 0.0);

        if (cap == CAP_ROUND) {
            out.stroke = length(pa - ba * h) - half_w;
        } else if (cap == CAP_SQUARE) {
            // Chebyshev-style: the stroke reaches half_w past the endpoints.
            out.stroke = max(perp, over) - half_w;
        } else {
            // Butt: the band ends exactly at the endpoints.
            out.stroke = max(perp - half_w, over);
        }
        out.t = clamp(u, 0.0, seg_len);
        return out;
    }

    // KIND_RECT: params0 = (x, y, w, h), params1.x = corner radius.
    let half_size = max(params0.zw * 0.5, vec2<f32>(0.0));
    let center = params0.xy + half_size;
    let radius = clamp(params1.x, 0.0, min(half_size.x, half_size.y));
    let sd = sd_rounded_box(p - center, half_size, radius);
    out.stroke = abs(sd) - half_w;
    out.fill = sd;
    // Dashes are not defined along a rect perimeter in v1 (RFC-0020): `t`
    // stays 0, which the dash mask treats as "always on" → a solid stroke.
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let opacity = in.misc.x;
    let kind = u32(in.misc.z + 0.5);
    let cap = u32(in.misc.w + 0.5);
    let stroke_width = in.stroke_dash.x;
    let dash_len = in.stroke_dash.y;
    let dash_gap = in.stroke_dash.z;
    let dash_offset = in.stroke_dash.w;

    // Screen-space AA width in logical px (`fwidth`-based — RFC-0020 §"AA
    // quality"), from the interpolated world position so it scales with DPI
    // and any paint-time transform.
    let aa = max(fwidth(in.world_pos.x) + fwidth(in.world_pos.y), 1e-4) * 0.5;

    // Sub-pixel strokes: clamp the rendered half-width up to one AA unit and
    // scale alpha down proportionally, so a 0.5px stroke renders as a fainter
    // 1px line instead of shimmering in and out of coverage.
    let half_w = stroke_width * 0.5;
    let half_w_eff = max(half_w, aa);
    let thin_alpha = clamp(half_w / half_w_eff, 0.0, 1.0);

    let d = eval_shape(in.world_pos, kind, cap, half_w_eff, in.params0, in.params1);

    // ── Coverages ─────────────────────────────────────────────────────────
    var stroke_cov = 1.0 - smoothstep(0.0, aa, d.stroke);
    let fill_cov = 1.0 - smoothstep(0.0, aa, d.fill);

    // Dash mask along the path parameter (logical px, so `aa` applies).
    if (dash_len > 0.0 && stroke_cov > 0.0) {
        let period = dash_len + max(dash_gap, 0.0);
        let s = fract((d.t + dash_offset) / period) * period;
        // > 0 inside the "on" interval [0, dash_len], < 0 in the gap.
        let edge = min(s, dash_len - s);
        stroke_cov = stroke_cov * smoothstep(-aa, aa, edge);
    }

    // ── Composite: stroke over fill ───────────────────────────────────────
    let a_stroke = stroke_cov * in.stroke_color.a * thin_alpha;
    let a_fill = fill_cov * in.fill_color.a * (1.0 - a_stroke);
    let out_a = a_stroke + a_fill;
    if (out_a <= 0.001) {
        discard;
    }
    let rgb = (in.stroke_color.rgb * a_stroke + in.fill_color.rgb * a_fill) / out_a;
    return vec4<f32>(rgb, out_a * opacity);
}
