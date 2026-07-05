//! `VectorMSDF` render pipeline (RFC-0009 §1, the fifth pipeline).
//!
//! Samples a multi-channel signed-distance-field (MSDF) **array-texture** atlas
//! to draw crisp monochrome glyphs at any scale. The dev (JIT) and release
//! (AOT-baked) paths produce the same [`VectorInstance`] shape (INV-7), so this
//! one pipeline serves both modes.
//!
//! Two byard corrections to the RFC draft are realised here and in the WGSL:
//!   - §2-D: no `discard`; premultiplied-alpha output (TBDR-friendly).
//!   - §2-E: anti-aliasing from the baked `px_range` and the UV derivative.
//!
//! Atlas uploads cross `frame.rs` as data ([`AtlasUpload`]) and are applied
//! **only** on the render thread via [`VectorAtlas::apply_uploads`] (INV-8); a
//! background worker never touches the `wgpu::Queue`.

use wgpu::util::DeviceExt;

use crate::ByardError;
use crate::frame::{AtlasUpload, VectorInstance};

/// Default MSDF atlas edge length in texels (one layer). 2048² holds many glyph
/// cells; the dev allocator (M48) grows layers on top of this.
pub const ATLAS_SIZE: u32 = 2048;

/// Vertex buffer layout for [`VectorInstance`] (locations 1..=5; the static quad
/// occupies location 0). Matches `vector_msdf.wgsl`'s `InstanceInput`.
#[must_use]
pub fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
        1 => Float32x4, // atlas_uv
        2 => Float32x4, // screen_rect
        3 => Float32x4, // color
        4 => Float32,   // px_range
        5 => Uint32,    // atlas_layer
        6 => Float32,   // depth (RFC-0011 cross-pass paint order)
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<VectorInstance>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: ATTRS,
    }
}

/// The MSDF atlas bind group layout (group 1): an array texture + linear sampler.
#[must_use]
pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ByardCore - VectorMSDF Atlas Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// The linear-filtering sampler — the whole point of an MSDF: hardware
/// interpolates the field, keeping edges crisp under magnification.
#[must_use]
pub fn sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ByardCore - VectorMSDF Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}

/// The MSDF atlas: an RGBA8 array texture plus its sampling bind group. Glyph
/// fields land in cells via [`apply_uploads`](Self::apply_uploads).
pub struct VectorAtlas {
    texture: wgpu::Texture,
    /// The group-1 bind group (atlas view + sampler), rebuilt when the texture
    /// grows.
    bind_group: wgpu::BindGroup,
    layers: u32,
}

impl VectorAtlas {
    /// Creates an empty `size × size` atlas with `layers` array layers.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        size: u32,
        layers: u32,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ByardCore - VectorMSDF Atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers.max(1),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bind_group = Self::make_bind_group(device, layout, sampler, &texture);
        Self {
            texture,
            bind_group,
            layers: layers.max(1),
        }
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        texture: &wgpu::Texture,
    ) -> wgpu::BindGroup {
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ByardCore - VectorMSDF Atlas View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ByardCore - VectorMSDF Atlas Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// The group-1 bind group for the draw call.
    #[must_use]
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// The number of array layers in the atlas.
    #[must_use]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Applies pending MSDF-field uploads to the atlas (RFC-0009 §2-C / INV-8).
    /// **Render thread only** — this is the single place a `Queue::write_texture`
    /// for the atlas is issued. Out-of-bounds uploads are skipped defensively.
    pub fn apply_uploads(&self, queue: &wgpu::Queue, uploads: &[AtlasUpload]) {
        for up in uploads {
            if up.layer >= self.layers || up.width == 0 || up.height == 0 {
                continue;
            }
            let expected = (up.width as usize) * (up.height as usize) * 4;
            if up.bytes.len() < expected {
                continue;
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: up.x,
                        y: up.y,
                        z: up.layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &up.bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(up.width * 4),
                    rows_per_image: Some(up.height),
                },
                wgpu::Extent3d {
                    width: up.width,
                    height: up.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

/// Compiles the WGSL shader and assembles the `VectorMSDF` pipeline inside a
/// single validation error scope (RFC-0001 §8).
///
/// # Errors
///
/// [`ByardError::PipelineCompilation`] (`pipeline: "VectorMSDF"`) on GPU-side
/// validation failure — never a panic.
pub async fn build_pipeline(
    device: &wgpu::Device,
    viewport_layout: &wgpu::BindGroupLayout,
    atlas_layout: &wgpu::BindGroupLayout,
    quad_layout: wgpu::VertexBufferLayout<'static>,
    surface_format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, ByardError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ByardCore - VectorMSDF Pipeline Layout"),
        bind_group_layouts: &[Some(viewport_layout), Some(atlas_layout)],
        immediate_size: 0,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ByardCore - VectorMSDF WGSL Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "vector_msdf.wgsl"
        ))),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ByardCore - VectorMSDF Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[quad_layout, instance_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                // §2-D: premultiplied-alpha output blends correctly with
                // standard "over"; `PREMULTIPLIED_ALPHA_BLENDING` expects
                // already-multiplied rgb, which `fs_main` produces.
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
        // Shares the pass's draw-order depth buffer with the other pipelines
        // (RFC-0011 cross-pass paint order) — required for pipeline/pass
        // compatibility even where a caller doesn't care about ordering.
        depth_stencil: Some(crate::encoder::draw_depth_stencil()),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    if let Some(error) = scope.pop().await {
        return Err(ByardError::PipelineCompilation {
            pipeline: "VectorMSDF".to_string(),
            reason: error.to_string(),
        });
    }

    Ok(pipeline)
}

/// Draws every [`VectorInstance`], batched against the atlas bind group. Group 0
/// (viewport) must already be bound by the caller. A no-op on an empty list, so
/// the render path is unchanged when no vector glyphs are present.
pub fn draw(
    render_pass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    viewport_bind_group: &wgpu::BindGroup,
    quad_buffer: &wgpu::Buffer,
    atlas: &VectorAtlas,
    instances: &[VectorInstance],
) {
    if instances.is_empty() {
        return;
    }
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ByardCore - VectorMSDF Instance Buffer"),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    });
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, viewport_bind_group, &[]);
    render_pass.set_bind_group(1, atlas.bind_group(), &[]);
    render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
    render_pass.set_vertex_buffer(1, buffer.slice(..));
    let count = u32::try_from(instances.len()).unwrap_or(u32::MAX);
    render_pass.draw(0..4, 0..count);
}

/// Median of three channels — the MSDF reconstruction of the true signed
/// distance (Chlumský 2015). The CPU twin of the WGSL `median3`, used by
/// generator/packer tests to verify field correctness without a GPU.
#[must_use]
pub fn median3(r: f32, g: f32, b: f32) -> f32 {
    (r.min(g)).max((r.max(g)).min(b))
}

/// Screen-pixel coverage width from the baked distance range and the screen-space
/// UV derivative (RFC-0009 §2-E). The CPU twin of the WGSL computation, exposed
/// for unit testing the anti-aliasing math. `uv_fwidth` is the per-axis
/// `fwidth(uv)`; `atlas_dims` is the atlas size in texels.
#[must_use]
pub fn screen_px_range(px_range: f32, uv_fwidth: [f32; 2], atlas_dims: [f32; 2]) -> f32 {
    let unit_range = [px_range / atlas_dims[0], px_range / atlas_dims[1]];
    let screen_tex_size = [1.0 / uv_fwidth[0], 1.0 / uv_fwidth[1]];
    let dot = unit_range[0] * screen_tex_size[0] + unit_range[1] * screen_tex_size[1];
    (0.5 * dot).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median3_matches_the_middle_value() {
        assert!((median3(0.1, 0.9, 0.5) - 0.5).abs() < 1e-6);
        assert!((median3(0.9, 0.5, 0.1) - 0.5).abs() < 1e-6);
        assert!((median3(0.5, 0.5, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn screen_px_range_is_floored_at_one() {
        // Extremely large texels per screen pixel (tiny derivative) → big range,
        // but a degenerate case must never fall below 1 (avoids div-by-zero AA).
        let big = screen_px_range(4.0, [0.0001, 0.0001], [2048.0, 2048.0]);
        assert!(big >= 1.0);
        // A glyph drawn near 1:1 still clamps to at least one pixel of softness.
        let small = screen_px_range(4.0, [1.0, 1.0], [2048.0, 2048.0]);
        assert!((small - 1.0).abs() < 1e-6);
    }
}
