struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
};

struct InstanceInput {
    @location(1) rect: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) radii: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) half_size: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) radii: vec4<f32>,
};

@group(0) @binding(0) var<uniform> viewport_size: vec2<f32>;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    let w = instance.rect.z;
    let h = instance.rect.w;
    out.half_size = vec2<f32>(w, h) * 0.5;
    out.local_pos = (vertex.quad_pos - 0.5) * vec2<f32>(w, h);

    let world_pos = instance.rect.xy + vertex.quad_pos * vec2<f32>(w, h);

    out.position = vec4<f32>(
        (world_pos.x / viewport_size.x) * 2.0 - 1.0,
        1.0 - (world_pos.y / viewport_size.y) * 2.0,
        0.0,
        1.0
    );

    out.color = instance.color;
    out.radii = instance.radii;
    return out;
}

/// SDF for a rounded rectangle with per-corner radii.
///
/// `p`  — fragment position relative to the rectangle centre.
/// `b`  — half-size of the rectangle (width/2, height/2).
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

    // Anti-alias the edge over one pixel using screen-space derivatives.
    // fwidth(distance) approximates how fast the SDF changes across a pixel,
    // giving sub-pixel accuracy without multisampling.
    let edge_softness = fwidth(distance);
    let alpha = smoothstep(edge_softness, 0.0, distance);

    if (alpha <= 0.0) {
        discard;
    }

    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
