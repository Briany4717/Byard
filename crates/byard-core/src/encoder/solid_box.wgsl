struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
};

struct InstanceInput {
    @location(1) rect: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) radii: vec4<f32>,
    // Paint-time transform (RFC-0011); identity is a free no-op below.
    @location(4) t_translate: vec2<f32>,
    @location(5) t_scale: vec2<f32>,
    @location(6) t_rotate: f32,
    @location(7) t_origin: vec2<f32>,
    @location(8) t_opacity: f32,
};

// Draw-order depth (NDC-z), fed from a separate per-instance vertex buffer so
// `BoxInstance`'s Pod layout stays unchanged. Earlier-emitted primitives get a
// value nearer 1.0 (farther); later ones nearer 0.0 (closer). See `draw_depth`.
struct DepthInput {
    @location(9) depth: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    /// Fragment position relative to the rectangle centre, in logical pixels.
    ///
    /// Covers the padded quad (see vertex shader), so values outside
    /// ±half_size are valid and represent the anti-alias margin. This is
    /// deliberately computed *before* the transform (RFC-0011: transforms
    /// reposition the painted quad as a rigid unit; the SDF still measures
    /// distance to the shape's own, untransformed boundary).
    @location(0) local_pos: vec2<f32>,
    /// Half-size of the *original* (un-padded) rectangle, in logical pixels.
    ///
    /// Passed unchanged to `sd_rounded_box` so the SDF always measures distance
    /// to the intended shape boundary, regardless of quad inflation.
    @location(1) half_size: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) radii: vec4<f32>,
    @location(4) opacity: f32,
};

@group(0) @binding(0) var<uniform> viewport_size: vec2<f32>;

/// Extra pixels added on every side of the quad beyond the shape boundary.
///
/// Without padding the quad ends exactly where the SDF crosses zero, so
/// the GPU never rasterises the fragments needed to fade alpha to 0 —
/// the anti-alias fringe is clipped at the cardinal points, making circles
/// look flat-edged. Two pixels is enough for one-pixel smoothstep at 2× DPI.
const QUAD_PADDING: f32 = 2.0;

/// Applies a paint-time transform (RFC-0011) to a world-space (logical-pixel)
/// position: rotate + scale about `origin`, then translate. Identity inputs
/// (`scale = (1,1)`, `rotate = 0`, `translate = (0,0)`) collapse to `world`
/// unchanged — a few cheap ALU ops, never a relayout (INV-8).
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
fn vs_main(vertex: VertexInput, instance: InstanceInput, depth: DepthInput) -> VertexOutput {
    var out: VertexOutput;

    let w = instance.rect.z;
    let h = instance.rect.w;

    // half_size stays at the original, un-padded dimensions.
    // The SDF always measures distance to this boundary.
    out.half_size = vec2<f32>(w, h) * 0.5;

    // The quad is inflated by QUAD_PADDING on every side so fragments can
    // exist beyond the shape boundary and carry out the fade-to-zero.
    let padded = vec2<f32>(w + QUAD_PADDING * 2.0, h + QUAD_PADDING * 2.0);

    // local_pos: fragment position relative to the rect centre.
    //   quad_pos = [0,0] → local_pos = -padded/2  (QUAD_PADDING px outside TL corner)
    //   quad_pos = [0.5, 0.5] → local_pos = [0, 0]  (centre, always)
    //   quad_pos = [1,1] → local_pos = +padded/2  (QUAD_PADDING px outside BR corner)
    out.local_pos = (vertex.quad_pos - 0.5) * padded;

    // world_pos: the quad top-left shifts by QUAD_PADDING so the inflation
    // is symmetric — QUAD_PADDING px of extra space on every side of the rect.
    let world_pos = instance.rect.xy - vec2<f32>(QUAD_PADDING) + vertex.quad_pos * padded;

    // Paint-time transform (RFC-0011), applied after layout placement and
    // before projection — layout itself never re-runs for this.
    let transformed = apply_transform(
        world_pos,
        instance.t_translate,
        instance.t_scale,
        instance.t_rotate,
        instance.t_origin,
    );

    // Convert logical-pixel world position to NDC.
    // viewport_size is in logical pixels (set from window.scale_factor() in the engine),
    // so this division is unit-consistent regardless of display density.
    out.position = vec4<f32>(
        (transformed.x / viewport_size.x) * 2.0 - 1.0,
        1.0 - (transformed.y / viewport_size.y) * 2.0,
        depth.depth,
        1.0
    );

    out.color = instance.color;
    out.radii = instance.radii;
    out.opacity = instance.t_opacity;
    return out;
}

/// SDF for a rounded rectangle with per-corner radii.
///
/// `p`  — fragment position relative to the rectangle centre (logical pixels).
/// `b`  — half-size of the rectangle (width/2, height/2) (logical pixels).
/// `r`  — corner radii [top_left, top_right, bottom_right, bottom_left].
///
/// Returns a negative value inside the shape, zero on the boundary, and a
/// positive value outside. Screen Y increases downward, so the top half has
/// y < 0 in local space.
///
/// The default case (p.x <= 0 && p.y <= 0) covers the top-left quadrant and
/// the coordinate axes, where the top-left radius applies.
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: vec4<f32>) -> f32 {
    var r_corner = r.x; // Top-Left (default — also covers axes p.x == 0 or p.y == 0)
    if (p.x > 0.0 && p.y < 0.0) { r_corner = r.y; } // Top-Right
    if (p.x > 0.0 && p.y > 0.0) { r_corner = r.z; } // Bottom-Right
    if (p.x < 0.0 && p.y > 0.0) { r_corner = r.w; } // Bottom-Left

    let q = abs(p) - b + vec2<f32>(r_corner);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r_corner;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let distance = sd_rounded_box(in.local_pos, in.half_size, in.radii);

    // Anti-alias the edge over one pixel using the Euclidean gradient magnitude.
    // We use length([dpdx, dpdy]) instead of fwidth (Manhattan norm) to get a
    // uniform anti-alias band in every direction — fwidth is up to √2 wider at
    // diagonals, which makes circles look slightly diamond-shaped.
    let edge_softness = length(vec2<f32>(dpdx(distance), dpdy(distance)));
    let alpha = smoothstep(edge_softness, 0.0, distance);

    if (alpha <= 0.0) {
        discard;
    }

    return vec4<f32>(in.color.rgb, in.color.a * alpha * in.opacity);
}
