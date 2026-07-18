//! `Ripple` render pipeline (RFC-0023): the Material ink reveal.
//!
//! Draws one expanding, fading circle per [`RippleInstance`] — centred on the
//! tap point, clipped in-shader to the element's rounded rect, and composited
//! with **premultiplied-alpha "over" blending**: a light ink brightens a dark
//! surface, a dark ink darkens a light one (a purely additive blend could
//! only ever add light, making dark ink invisible on light surfaces), and
//! simultaneous ripples from rapid taps still accumulate where their circles
//! overlap (RFC-0023 §1). The pipeline is transparent geometry: it *tests*
//! the shared draw-order depth buffer but never *writes* it, exactly like
//! `DecoratedBox` — its depth (stamped by
//! [`RenderFrame::push_ripple`](crate::frame::RenderFrame::push_ripple)
//! between the element's background and its children) is what places the ink
//! in the RFC-0023 compositing slot: background → ripple → children.
//!
//! The expansion/fade animation is sampled on the logic thread each tick
//! through the shared [`Motion`](crate::frame::Motion) closed forms (the
//! RFC-0010 model as landed); this pipeline only rasterises the current
//! circle. Zero cost when no ripple is live — the draw call is skipped
//! entirely on an empty pool.

use wgpu::util::DeviceExt;

use crate::ByardError;
use crate::frame::RippleInstance;

impl RippleInstance {
    /// Vertex buffer layout for the per-instance step (locations 1..=9; the
    /// static quad occupies location 0). Field order must match both this
    /// struct's declaration and `ripple.wgsl`'s `InstanceInput`.
    #[must_use]
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
            1 => Float32x4, // rect
            2 => Float32x4, // params (center.xy, radius, fade alpha)
            3 => Float32x4, // color
            4 => Float32x4, // radii
            5 => Float32x2, // transform.translate
            6 => Float32x2, // transform.scale
            7 => Float32,   // transform.rotate
            8 => Float32x2, // transform.origin
            9 => Float32,   // draw-order depth
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<RippleInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRS,
        }
    }
}

/// Compiles the WGSL shader and assembles the `Ripple` pipeline, wrapping the
/// whole create sequence in a single validation error scope (RFC-0001 §8).
///
/// # Errors
///
/// [`ByardError::PipelineCompilation`] if the shader or pipeline fails GPU-side
/// validation — never a panic, never a software fallback.
pub async fn build_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    quad_layout: wgpu::VertexBufferLayout<'static>,
    surface_format: wgpu::TextureFormat,
    depth_stencil: wgpu::DepthStencilState,
) -> Result<wgpu::RenderPipeline, ByardError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ByardCore - Ripple Pipeline Layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ByardCore - Ripple WGSL Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("ripple.wgsl"))),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ByardCore - Ripple Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[quad_layout, RippleInstance::layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                // Premultiplied "over" (the shader outputs `rgb·a`): ink works
                // on both dark and light surfaces, and overlapping simultaneous
                // ripples accumulate as `1 − (1−a)ⁿ` where their circles cross.
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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
            pipeline: "Ripple".to_string(),
            reason: error.to_string(),
        });
    }

    Ok(pipeline)
}

/// Draws every [`RippleInstance`], scissored to its content clip (RFC-0005).
/// The instances upload as-is — depth is already stamped on the struct by
/// `push_ripple`, so unlike `decorated_box::draw` no per-instance rewrite
/// happens here.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    render_pass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    quad_buffer: &wgpu::Buffer,
    ripples: &[RippleInstance],
    clip_slice: &[Option<u16>],
    ctx: super::ClipCtx<'_>,
) {
    if ripples.is_empty() {
        return;
    }
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ByardCore - Ripple Instance Buffer"),
        contents: bytemuck::cast_slice(ripples),
        usage: wgpu::BufferUsages::VERTEX,
    });

    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
    render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
    // Content-clip runs (RFC-0005): scissor each run to its ScrollView viewport.
    super::for_each_clip_run(render_pass, ripples.len(), clip_slice, ctx, |p, s, e| {
        p.draw(0..4, s..e);
    });
}
