//! `TextureSampler` render pipeline (M21, RFC-0001 §3.1).
//!
//! Draws a decoded image into a (optionally rounded) quad with a `fit` policy.
//! Host-side decode uses the `image` crate (IMPL-32): the runner owns decode, the
//! interpreter only carries the `Str` path. Decoded textures are cached by path
//! so a static image is uploaded once.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use crate::ByardError;
use crate::frame::{ImageFit, TextureSampler};

/// GPU-side per-instance data for the `TextureSampler` pipeline; matches
/// `texture_sampler.wgsl`'s `InstanceInput`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextureInstance {
    /// `[x, y, width, height]` in logical pixels.
    pub rect: [f32; 4],
    /// Per-corner radii `[tl, tr, br, bl]`.
    pub radii: [f32; 4],
    /// `[uv_scale_x, uv_scale_y, uv_offset_x, uv_offset_y]` — the `fit` transform.
    pub uv_xform: [f32; 4],
    /// `[opacity, 0, 0, 0]`.
    pub misc: [f32; 4],
}

impl TextureInstance {
    /// Vertex buffer layout (locations 1..=4; the static quad occupies 0).
    #[must_use]
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
            1 => Float32x4, // rect
            2 => Float32x4, // radii
            3 => Float32x4, // uv_xform
            4 => Float32x4, // misc
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TextureInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRS,
        }
    }
}

/// One decoded, uploaded texture plus its sampling bind group.
pub struct TextureEntry {
    /// Bind group (group 1): the texture view + sampler.
    pub bind_group: wgpu::BindGroup,
    /// Source pixel width.
    pub width: u32,
    /// Source pixel height.
    pub height: u32,
}

/// Path-keyed cache of decoded textures so a static image uploads once.
#[derive(Default)]
pub struct TextureCache {
    entries: HashMap<String, Option<TextureEntry>>,
}

impl TextureCache {
    /// Decodes and uploads `src` if not already cached. A decode failure is
    /// cached as `None` (the image simply does not draw — no panic, IMPL-32).
    pub fn ensure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        src: &str,
    ) {
        if self.entries.contains_key(src) {
            return;
        }
        let entry = decode_and_upload(device, queue, layout, sampler, src);
        self.entries.insert(src.to_string(), entry);
    }

    /// Looks up a previously-ensured entry.
    #[must_use]
    pub fn get(&self, src: &str) -> Option<&TextureEntry> {
        self.entries.get(src).and_then(Option::as_ref)
    }
}

/// Decodes an image file to RGBA8 and uploads it to a fresh GPU texture.
fn decode_and_upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    src: &str,
) -> Option<TextureEntry> {
    let img = match image::open(src) {
        Ok(img) => img.to_rgba8(),
        Err(err) => {
            // Alert the dev that their asset isn't where the engine looked. The
            // cache stores this `None` so the warning fires once per path, not
            // every frame (IMPL-32: a missing image simply does not draw).
            let cwd = std::env::current_dir()
                .map_or_else(|_| "?".to_string(), |p| p.display().to_string());
            eprintln!(
                "byard: warning: image not found or could not be decoded: \
                 '{src}' (searched relative to {cwd}): {err}"
            );
            return None;
        }
    };
    let (width, height) = img.dimensions();
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ByardCore - Sampled Image Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ByardCore - Texture Bind Group"),
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
    });
    Some(TextureEntry {
        bind_group,
        width,
        height,
    })
}

/// Builds the texture+sampler bind group layout (group 1).
#[must_use]
pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ByardCore - Texture Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
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

/// Computes the `[scale_x, scale_y, offset_x, offset_y]` UV transform that
/// realizes `fit` for a `(img_w × img_h)` image inside a `(rect_w × rect_h)`
/// rectangle. UVs outside `[0,1]` are discarded by the shader (letterbox bars).
#[must_use]
#[allow(clippy::cast_precision_loss)] // image dimensions are small; f32 is exact in practice.
pub fn uv_transform(fit: ImageFit, img_w: u32, img_h: u32, rect_w: f32, rect_h: f32) -> [f32; 4] {
    if img_w == 0 || img_h == 0 || rect_w <= 0.0 || rect_h <= 0.0 {
        return [1.0, 1.0, 0.0, 0.0];
    }
    let img_aspect = img_w as f32 / img_h as f32;
    let rect_aspect = rect_w / rect_h;
    match fit {
        ImageFit::Fill => [1.0, 1.0, 0.0, 0.0],
        ImageFit::Cover => {
            // Sample a centered sub-rectangle so the image covers the rect.
            if img_aspect > rect_aspect {
                let s = rect_aspect / img_aspect;
                [s, 1.0, (1.0 - s) / 2.0, 0.0]
            } else {
                let s = img_aspect / rect_aspect;
                [1.0, s, 0.0, (1.0 - s) / 2.0]
            }
        }
        ImageFit::Contain => {
            // Fit the whole image inside; the quad over-extends on the
            // letterboxed axis so the bars fall outside [0,1] and discard.
            if img_aspect > rect_aspect {
                let s = img_aspect / rect_aspect;
                [1.0, s, 0.0, (1.0 - s) / 2.0]
            } else {
                let s = rect_aspect / img_aspect;
                [s, 1.0, (1.0 - s) / 2.0, 0.0]
            }
        }
        ImageFit::None => {
            // Natural pixel size, top-left aligned.
            [rect_w / img_w as f32, rect_h / img_h as f32, 0.0, 0.0]
        }
    }
}

/// Builds the linear-filtering sampler shared by all sampled images.
#[must_use]
pub fn sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ByardCore - Image Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}

/// Compiles the WGSL shader and assembles the `TextureSampler` pipeline inside a
/// single validation error scope (RFC-0001 §8).
///
/// # Errors
///
/// [`ByardError::PipelineCompilation`] on GPU-side validation failure.
pub async fn build_pipeline(
    device: &wgpu::Device,
    viewport_layout: &wgpu::BindGroupLayout,
    texture_layout: &wgpu::BindGroupLayout,
    quad_layout: wgpu::VertexBufferLayout<'static>,
    surface_format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, ByardError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ByardCore - TextureSampler Pipeline Layout"),
        bind_group_layouts: &[Some(viewport_layout), Some(texture_layout)],
        immediate_size: 0,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ByardCore - TextureSampler WGSL Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "texture_sampler.wgsl"
        ))),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ByardCore - TextureSampler Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[quad_layout, TextureInstance::layout()],
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
            pipeline: "TextureSampler".to_string(),
            reason: error.to_string(),
        });
    }

    Ok(pipeline)
}

/// Draws every cached [`TextureSampler`]; unresolved (failed-decode) sources are
/// skipped. Group 0 (viewport) must already be bound by the caller.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    render_pass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    viewport_bind_group: &wgpu::BindGroup,
    quad_buffer: &wgpu::Buffer,
    cache: &TextureCache,
    textures: &[TextureSampler],
) {
    if textures.is_empty() {
        return;
    }
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, viewport_bind_group, &[]);
    render_pass.set_vertex_buffer(0, quad_buffer.slice(..));

    for t in textures {
        let Some(entry) = cache.get(&t.src) else {
            continue;
        };
        let uv_xform = uv_transform(t.fit, entry.width, entry.height, t.rect[2], t.rect[3]);
        let instance = TextureInstance {
            rect: t.rect,
            radii: t.radii,
            uv_xform,
            misc: [t.opacity, 0.0, 0.0, 0.0],
        };
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ByardCore - TextureSampler Instance Buffer"),
            contents: bytemuck::bytes_of(&instance),
            usage: wgpu::BufferUsages::VERTEX,
        });
        render_pass.set_bind_group(1, &entry.bind_group, &[]);
        render_pass.set_vertex_buffer(1, buffer.slice(..));
        render_pass.draw(0..4, 0..1);
    }
}
