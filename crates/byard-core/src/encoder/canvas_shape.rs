//! `CanvasShape` render pipeline (RFC-0020 §2, Tier 1) — the sixth pipeline.
//!
//! Draws programmatic 2-D shapes — arcs, circles, lines, and (rounded)
//! rectangles — by evaluating each shape's closed-form signed-distance
//! function analytically in the fragment shader (`canvas_shape.wgsl`).
//! Resolution-independent, atlas-free, and animation-friendly: a reactive
//! `sweep`/`dash_offset` change is just fresh instance data, never a
//! re-tessellation or a re-rasterization.
//!
//! Like `DecoratedBox`, everything this pipeline draws is **transparent
//! geometry** (anti-aliased strokes and fills with fractional-coverage
//! edges), so its pipeline *tests* the shared draw-order depth buffer but
//! never writes it — the RFC-0017 opaque/transparent split. Complex
//! `path(d: …)` commands never reach this pipeline; they rasterize through
//! `VectorMSDF` (RFC-0020 §2, Tier 2).

use wgpu::util::DeviceExt;

use crate::ByardError;
use crate::frame::CanvasShape;

/// GPU-side per-instance data for the `CanvasShape` pipeline. The field
/// layout must match `canvas_shape.wgsl`'s `InstanceInput` exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CanvasShapeInstance {
    /// Quad bounds `[x, y, w, h]` in logical px ([`CanvasShape::bounds`] —
    /// already inflated to cover the stroke and the AA fringe).
    pub rect: [f32; 4],
    /// Shape params, first half (layout per kind — `frame::CANVAS_SHAPE_*`).
    pub params0: [f32; 4],
    /// Shape params, second half.
    pub params1: [f32; 4],
    /// Stroke colour `[r, g, b, a]`.
    pub stroke_color: [f32; 4],
    /// Fill colour `[r, g, b, a]`.
    pub fill_color: [f32; 4],
    /// `[stroke_width, dash_len, dash_gap, dash_offset]`.
    pub stroke_dash: [f32; 4],
    /// `[opacity, draw-order depth, kind, cap]` — kind/cap are small integers
    /// carried exactly in `f32`.
    pub misc: [f32; 4],
    /// Paint-time transform translate (RFC-0011).
    pub t_translate: [f32; 2],
    /// Paint-time transform per-axis scale (RFC-0011).
    pub t_scale: [f32; 2],
    /// Paint-time transform rotation in radians (RFC-0011).
    pub t_rotate: f32,
    /// Paint-time transform pivot (RFC-0011).
    pub t_origin: [f32; 2],
}

impl CanvasShapeInstance {
    /// Builds the GPU instance for one [`CanvasShape`] at draw-order depth
    /// `depth`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn new(s: &CanvasShape, depth: f32) -> Self {
        let b = s.bounds();
        Self {
            rect: [b.x, b.y, b.width, b.height],
            params0: [s.params[0], s.params[1], s.params[2], s.params[3]],
            params1: [s.params[4], s.params[5], s.params[6], s.params[7]],
            stroke_color: s.stroke_color,
            fill_color: s.fill_color,
            stroke_dash: [s.stroke_width, s.dash[0], s.dash[1], s.dash_offset],
            misc: [s.opacity, depth, s.kind as f32, s.cap as f32],
            t_translate: s.transform.translate,
            t_scale: s.transform.scale,
            t_rotate: s.transform.rotate,
            t_origin: s.transform.origin,
        }
    }

    /// Vertex buffer layout for the per-instance step (locations 1..=11; the
    /// static quad occupies location 0).
    #[must_use]
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
            1 => Float32x4, // rect
            2 => Float32x4, // params0
            3 => Float32x4, // params1
            4 => Float32x4, // stroke_color
            5 => Float32x4, // fill_color
            6 => Float32x4, // stroke_dash
            7 => Float32x4, // misc
            8 => Float32x2, // transform.translate
            9 => Float32x2, // transform.scale
            10 => Float32, // transform.rotate
            11 => Float32x2, // transform.origin
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CanvasShapeInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRS,
        }
    }
}

/// Compiles the WGSL shader and assembles the `CanvasShape` pipeline, wrapping
/// the whole create sequence in a single validation error scope (RFC-0001 §8).
///
/// # Errors
///
/// [`ByardError::PipelineCompilation`] if the shader or pipeline fails
/// GPU-side validation — never a panic, never a software fallback.
pub async fn build_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    quad_layout: wgpu::VertexBufferLayout<'static>,
    surface_format: wgpu::TextureFormat,
    depth_stencil: wgpu::DepthStencilState,
) -> Result<wgpu::RenderPipeline, ByardError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ByardCore - CanvasShape Pipeline Layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ByardCore - CanvasShape WGSL Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "canvas_shape.wgsl"
        ))),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ByardCore - CanvasShape Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[quad_layout, CanvasShapeInstance::layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(depth_stencil),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    if let Some(error) = scope.pop().await {
        return Err(ByardError::PipelineCompilation {
            pipeline: "CanvasShape".to_string(),
            reason: error.to_string(),
        });
    }

    Ok(pipeline)
}

/// Draws every [`CanvasShape`], scissored to its content clip (RFC-0005).
/// Everything here is transparent geometry — the pipeline tests the shared
/// draw-order depth buffer but never writes it (see the module docs).
#[allow(clippy::too_many_arguments)]
pub fn draw(
    render_pass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    quad_buffer: &wgpu::Buffer,
    shapes: &[CanvasShape],
    depths: &[f32],
    clip_slice: &[Option<u16>],
    ctx: super::ClipCtx<'_>,
) {
    if shapes.is_empty() {
        return;
    }
    // `depths` is parallel to `shapes`; a short/empty slice falls back to the
    // far plane so a missing depth can't push a shape in front of others.
    let instances: Vec<CanvasShapeInstance> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let depth = depths
                .get(i)
                .copied()
                .unwrap_or(crate::frame::DRAW_DEPTH_CLEAR);
            CanvasShapeInstance::new(s, depth)
        })
        .collect();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ByardCore - CanvasShape Instance Buffer"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::VERTEX,
    });

    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
    render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
    // Content-clip runs (RFC-0005): scissor each run to its ScrollView viewport.
    super::for_each_clip_run(render_pass, instances.len(), clip_slice, ctx, |p, s, e| {
        p.draw(0..4, s..e);
    });
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// CPU mirrors of the WGSL distance functions, following the project's
// established pattern (`encoder::mod`'s `cpu_sd_rounded_box`): the decision
// geometry is pinned down deterministically here, without a GPU; the
// GPU-readback test in `byard-platform` proves the wired-up pipeline paints.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{
        CANVAS_CAP_BUTT, CANVAS_CAP_ROUND, CANVAS_SHAPE_ARC, CANVAS_SHAPE_LINE, Transform,
    };

    const TAU: f32 = std::f32::consts::TAU;

    /// CPU twin of the WGSL `wrap_angle`: wraps to `[-PI, PI]`.
    fn wrap_angle(a: f32) -> f32 {
        a - TAU * (a / TAU).round()
    }

    /// CPU twin of the WGSL arc stroke distance (round/butt caps).
    fn cpu_arc_stroke(
        p: [f32; 2],
        c: [f32; 2],
        r: f32,
        start: f32,
        sweep: f32,
        half_w: f32,
        round_cap: bool,
    ) -> f32 {
        let rel = [p[0] - c[0], p[1] - c[1]];
        let len = (rel[0] * rel[0] + rel[1] * rel[1]).sqrt();
        let ang = rel[1].atan2(rel[0]);
        let half_sweep = sweep.abs().min(TAU) * 0.5;
        let mid = start + sweep * 0.5;
        let delta = wrap_angle(ang - mid);
        let ring = (len - r).abs() - half_w;
        if delta.abs() <= half_sweep {
            ring
        } else if round_cap {
            let p0 = [c[0] + r * start.cos(), c[1] + r * start.sin()];
            let e = start + sweep;
            let p1 = [c[0] + r * e.cos(), c[1] + r * e.sin()];
            let d0 = ((p[0] - p0[0]).powi(2) + (p[1] - p0[1]).powi(2)).sqrt();
            let d1 = ((p[0] - p1[0]).powi(2) + (p[1] - p1[1]).powi(2)).sqrt();
            d0.min(d1) - half_w
        } else {
            ring.max((delta.abs() - half_sweep) * len.max(1e-3))
        }
    }

    /// CPU twin of the WGSL line stroke distance (round cap).
    fn cpu_line_stroke_round(p: [f32; 2], a: [f32; 2], b: [f32; 2], half_w: f32) -> f32 {
        let ba = [b[0] - a[0], b[1] - a[1]];
        let pa = [p[0] - a[0], p[1] - a[1]];
        let len2 = (ba[0] * ba[0] + ba[1] * ba[1]).max(1e-6);
        let h = ((pa[0] * ba[0] + pa[1] * ba[1]) / len2).clamp(0.0, 1.0);
        let dx = pa[0] - ba[0] * h;
        let dy = pa[1] - ba[1] * h;
        (dx * dx + dy * dy).sqrt() - half_w
    }

    /// CPU twin of the WGSL dash mask edge value (`> 0` = on-segment).
    fn cpu_dash_edge(t: f32, dash_len: f32, dash_gap: f32, dash_offset: f32) -> f32 {
        let period = dash_len + dash_gap.max(0.0);
        let s = ((t + dash_offset) / period).fract() * period;
        let s = if s < 0.0 { s + period } else { s };
        s.min(dash_len - s)
    }

    // ── Arc geometry ──────────────────────────────────────────────────────────

    #[test]
    fn arc_point_on_ring_within_sweep_is_inside_the_stroke() {
        // Quarter arc from 0 to 90° at r=20, stroke 4 (half 2): the point at
        // 45° on the ring is inside.
        let ang = 45f32.to_radians();
        let p = [20.0 * ang.cos(), 20.0 * ang.sin()];
        let d = cpu_arc_stroke(p, [0.0, 0.0], 20.0, 0.0, 90f32.to_radians(), 2.0, false);
        assert!(d < 0.0, "on-ring in-sweep point must be covered, got {d}");
    }

    #[test]
    fn arc_point_outside_sweep_is_not_covered_with_butt_caps() {
        // The same ring point at 180° is far outside the 0..90° sweep.
        let p = [-20.0, 0.0];
        let d = cpu_arc_stroke(p, [0.0, 0.0], 20.0, 0.0, 90f32.to_radians(), 2.0, false);
        assert!(d > 0.0, "out-of-sweep point must not be covered, got {d}");
    }

    #[test]
    fn arc_round_cap_covers_just_past_the_endpoint() {
        // A point 1px past the sweep end, on the ring: a round cap (radius
        // half_w = 2) still covers it; a butt cap does not.
        let end = 90f32.to_radians();
        let past = end + (1.0 / 20.0); // ≈1px of arc length past the end
        let p = [20.0 * past.cos(), 20.0 * past.sin()];
        let round = cpu_arc_stroke(p, [0.0, 0.0], 20.0, 0.0, end, 2.0, true);
        let butt = cpu_arc_stroke(p, [0.0, 0.0], 20.0, 0.0, end, 2.0, false);
        assert!(
            round < 0.0,
            "round cap must cover 1px past the end: {round}"
        );
        assert!(butt > 0.0, "butt cap must stop at the end: {butt}");
    }

    #[test]
    fn full_circle_sweep_covers_every_ring_angle() {
        for deg in [0.0f32, 45.0, 133.0, 200.0, 359.0] {
            let a = deg.to_radians();
            let p = [30.0 * a.cos(), 30.0 * a.sin()];
            let d = cpu_arc_stroke(p, [0.0, 0.0], 30.0, 0.0, TAU, 1.5, false);
            assert!(d < 0.0, "sweep=360° must cover the whole ring at {deg}°");
        }
    }

    // ── Line geometry ─────────────────────────────────────────────────────────

    #[test]
    fn line_stroke_covers_the_segment_band_and_round_cap() {
        // Horizontal segment (0,0)→(100,0), stroke 6 (half 3).
        let inside = cpu_line_stroke_round([50.0, 2.0], [0.0, 0.0], [100.0, 0.0], 3.0);
        let outside = cpu_line_stroke_round([50.0, 5.0], [0.0, 0.0], [100.0, 0.0], 3.0);
        let cap = cpu_line_stroke_round([102.0, 0.0], [0.0, 0.0], [100.0, 0.0], 3.0);
        assert!(inside < 0.0 && outside > 0.0);
        assert!(cap < 0.0, "round cap overhangs the endpoint: {cap}");
    }

    // ── Dash mask ─────────────────────────────────────────────────────────────

    #[test]
    fn dash_mask_alternates_on_and_off_and_offset_shifts_the_phase() {
        // Pattern (10 on, 10 off): t=5 on, t=15 off.
        assert!(cpu_dash_edge(5.0, 10.0, 10.0, 0.0) > 0.0);
        assert!(cpu_dash_edge(15.0, 10.0, 10.0, 0.0) < 0.0);
        // A 10px offset flips both.
        assert!(cpu_dash_edge(5.0, 10.0, 10.0, 10.0) < 0.0);
        assert!(cpu_dash_edge(15.0, 10.0, 10.0, 10.0) > 0.0);
    }

    // ── Instance packing ──────────────────────────────────────────────────────

    #[test]
    fn instance_packs_kind_cap_depth_and_quad_bounds() {
        let s = CanvasShape {
            kind: CANVAS_SHAPE_ARC,
            params: [24.0, 24.0, 20.0, -1.0, 2.0, 0.0, 0.0, 0.0],
            stroke_color: [1.0, 0.0, 0.0, 1.0],
            stroke_width: 4.0,
            cap: CANVAS_CAP_ROUND,
            dash: [6.0, 4.0],
            dash_offset: 2.0,
            opacity: 0.8,
            transform: Transform::IDENTITY,
            fill_color: [0.0; 4],
            dirty: true,
        };
        let inst = CanvasShapeInstance::new(&s, 0.5);
        // misc = [opacity, depth, kind (ARC = 0), cap (ROUND = 1)]. The
        // packing copies these fields verbatim — no arithmetic — so *bitwise*
        // equality is precisely the claim (and satisfies `float_cmp`).
        assert_eq!(
            inst.misc.map(f32::to_bits),
            [0.8f32, 0.5, 0.0, 1.0].map(f32::to_bits)
        );
        assert_eq!(
            inst.stroke_dash.map(f32::to_bits),
            [4.0f32, 6.0, 4.0, 2.0].map(f32::to_bits)
        );
        // The quad covers the circle plus the stroke margin.
        assert!(inst.rect[0] <= 0.0 && inst.rect[1] <= 0.0);
        assert!(inst.rect[2] >= 48.0 && inst.rect[3] >= 48.0);
        // Line instances use segment bounds, not circle bounds.
        let l = CanvasShape {
            kind: CANVAS_SHAPE_LINE,
            params: [10.0, 20.0, 90.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            stroke_width: 2.0,
            cap: CANVAS_CAP_BUTT,
            ..CanvasShape::default()
        };
        let li = CanvasShapeInstance::new(&l, 0.25);
        assert!(li.rect[0] <= 8.0 && li.rect[0] + li.rect[2] >= 92.0);
    }
}
