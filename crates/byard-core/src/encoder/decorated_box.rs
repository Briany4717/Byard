//! `DecoratedBox` render pipeline (M21, RFC-0001 §3.1).
//!
//! Draws a rounded rectangle with an optional inner border, a blurred drop
//! shadow, and an overall opacity. The compiler promotes a box to this pipeline
//! (via [`RenderFrame::push_decorated`](crate::frame::RenderFrame::push_decorated))
//! only when one of those decoration fields is non-trivial; plain solid fills
//! (radius alone included) stay on the cheaper `SolidBox` pipeline.

use wgpu::util::DeviceExt;

use crate::ByardError;
use crate::frame::DecoratedBox;

/// GPU-side per-instance data for the `DecoratedBox` pipeline. The field layout
/// must match `decorated_box.wgsl`'s `InstanceInput` exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DecoratedInstance {
    /// `[x, y, width, height]` in logical pixels.
    pub rect: [f32; 4],
    /// Fill colour `[r, g, b, a]`.
    pub color: [f32; 4],
    /// Per-corner radii `[tl, tr, br, bl]`.
    pub radii: [f32; 4],
    /// Border colour `[r, g, b, a]`.
    pub border_color: [f32; 4],
    /// Shadow colour `[r, g, b, a]`.
    pub shadow_color: [f32; 4],
    /// `[border_width, shadow_dx, shadow_dy, shadow_blur]`.
    pub params: [f32; 4],
    /// `[opacity, 0, 0, 0]`.
    pub misc: [f32; 4],
    /// Paint-time transform translate (RFC-0011), from `d.base.transform`.
    /// Only the geometric fields (`translate`/`scale`/`rotate`/`origin`) are
    /// read here — `d.base.transform.opacity` is **not** consulted; `misc.x`
    /// (above) is the authoritative opacity for decorated boxes, unchanged
    /// since M21.
    pub t_translate: [f32; 2],
    /// Paint-time transform per-axis scale (RFC-0011).
    pub t_scale: [f32; 2],
    /// Paint-time transform rotation in radians (RFC-0011).
    pub t_rotate: f32,
    /// Paint-time transform pivot (RFC-0011).
    pub t_origin: [f32; 2],
}

impl From<&DecoratedBox> for DecoratedInstance {
    fn from(d: &DecoratedBox) -> Self {
        Self {
            rect: d.base.rect,
            color: d.base.color,
            radii: d.base.radii,
            border_color: d.border_color,
            shadow_color: d.shadow_color,
            params: [d.border_width, d.shadow_dx, d.shadow_dy, d.shadow_blur],
            misc: [d.opacity, 0.0, 0.0, 0.0],
            t_translate: d.base.transform.translate,
            t_scale: d.base.transform.scale,
            t_rotate: d.base.transform.rotate,
            t_origin: d.base.transform.origin,
        }
    }
}

impl DecoratedInstance {
    /// Vertex buffer layout for the per-instance step (locations 1..=11; the
    /// static quad occupies location 0).
    #[must_use]
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
            1 => Float32x4, // rect
            2 => Float32x4, // color
            3 => Float32x4, // radii
            4 => Float32x4, // border_color
            5 => Float32x4, // shadow_color
            6 => Float32x4, // params
            7 => Float32x4, // misc
            8 => Float32x2, // transform.translate
            9 => Float32x2, // transform.scale
            10 => Float32, // transform.rotate
            11 => Float32x2, // transform.origin
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DecoratedInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRS,
        }
    }
}

/// Compiles the WGSL shader and assembles the `DecoratedBox` pipeline, wrapping
/// the whole create sequence in a single validation error scope (RFC-0001 §8).
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
) -> Result<wgpu::RenderPipeline, ByardError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ByardCore - DecoratedBox Pipeline Layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ByardCore - DecoratedBox WGSL Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "decorated_box.wgsl"
        ))),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ByardCore - DecoratedBox Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[quad_layout, DecoratedInstance::layout()],
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
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    if let Some(error) = scope.pop().await {
        return Err(ByardError::PipelineCompilation {
            pipeline: "DecoratedBox".to_string(),
            reason: error.to_string(),
        });
    }

    Ok(pipeline)
}

/// Draws every [`DecoratedBox`] using the pipeline. The active GPU scissor (if
/// any) bounds which pixels are actually touched.
pub fn draw(
    render_pass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    quad_buffer: &wgpu::Buffer,
    boxes: &[DecoratedBox],
) {
    if boxes.is_empty() {
        return;
    }
    let instances: Vec<DecoratedInstance> = boxes.iter().map(DecoratedInstance::from).collect();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ByardCore - DecoratedBox Instance Buffer"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::VERTEX,
    });

    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
    render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
    #[allow(clippy::cast_possible_truncation)]
    render_pass.draw(0..4, 0..instances.len() as u32);
}
