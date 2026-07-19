//! Backdrop blur pipelines (RFC-0023 §2): the iOS frosted-glass / vibrancy
//! effect.
//!
//! A [`BackdropInstance`](crate::frame::BackdropInstance) is a **barrier**:
//! everything emitted before it must be rasterised into the colour target
//! before the pane can sample it. The encoder therefore splits its UI pass
//! at each backdrop and, between the split passes, this module records the
//! off-screen work (RFC-0023 §2 steps 2–4):
//!
//! 1. copy the region behind the pane (its on-screen AABB + the ±2.5σ
//!    kernel halo) out of the persistent colour target,
//! 2. blur it into a downsampled scratch target with a two-pass separable
//!    21-tap Gaussian (`σ` = the `blur` prop, the CSS `backdrop-filter`
//!    convention the RFC cites). The quality tiers differ only in *base
//!    resolution* — high 0.75×, auto-on-capable-GPUs 0.5×, low (and auto on
//!    software adapters) 0.25× — and the downsample deepens **adaptively**
//!    with σ so tap spacing never exceeds the kernel's clean coverage: the
//!    anti-ghosting guarantee (see `blur.wgsl`),
//! 3. composite the result back inside the resumed main pass, clipped to
//!    the element's rounded rect, saturation-boosted and tinted
//!    (`backdrop.wgsl`).
//!
//! Scratch textures are cached per backdrop slot and only recreated when
//! the region size changes, so a steady frosted nav-bar allocates nothing
//! per frame after the first.

use wgpu::util::DeviceExt;

use crate::ByardError;
use crate::frame::{BLUR_QUALITY_HIGH, BLUR_QUALITY_LOW, BackdropInstance};

/// GPU-side per-instance data for the composite quad. Field order must match
/// `backdrop.wgsl`'s `InstanceInput`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeQuad {
    /// Element rect `[x, y, w, h]` in logical px.
    rect: [f32; 4],
    /// Per-corner clip radii.
    radii: [f32; 4],
    /// `backdrop_tint` colour.
    tint: [f32; 4],
    /// `(saturation, opacity, depth, 0)`.
    params: [f32; 4],
    /// Copied-region mapping `(origin_x, origin_y, 1/w, 1/h)` in physical px.
    region: [f32; 4],
    /// Paint-time transform (RFC-0011).
    t_translate: [f32; 2],
    t_scale: [f32; 2],
    t_rotate: f32,
    t_origin: [f32; 2],
}

impl CompositeQuad {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
            1 => Float32x4, // rect
            2 => Float32x4, // radii
            3 => Float32x4, // tint
            4 => Float32x4, // params
            5 => Float32x4, // region
            6 => Float32x2, // transform.translate
            7 => Float32x2, // transform.scale
            8 => Float32,   // transform.rotate
            9 => Float32x2, // transform.origin
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CompositeQuad>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRS,
        }
    }
}

/// Uniforms of one off-screen blur pass; must match `blur.wgsl`'s
/// `BlurParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    texel: [f32; 2],
    dir: [f32; 2],
    sigma: f32,
    _pad: [f32; 3],
}

/// The compiled backdrop pipelines plus their shared bind-group layouts and
/// sampler, built once at encoder init.
pub struct BackdropPipelines {
    /// Off-screen blur pass pipeline (fullscreen triangle, `blur.wgsl`).
    blur: wgpu::RenderPipeline,
    /// Bind-group layout of a blur pass: source texture + sampler + params.
    blur_bgl: wgpu::BindGroupLayout,
    /// In-pass composite pipeline (`backdrop.wgsl`).
    composite: wgpu::RenderPipeline,
    /// Bind-group layout of the composite's group 1: blurred texture + sampler.
    composite_bgl: wgpu::BindGroupLayout,
    /// Shared bilinear sampler (clamping — the halo padding keeps edge taps
    /// in-region, and clamping avoids wrap artefacts on the border).
    sampler: wgpu::Sampler,
}

/// Adaptive-downsample cap: the largest Gaussian σ, in destination-scaled
/// texels, that the 21-tap kernel covers cleanly (σ/4 tap spacing ≤ 2
/// texels — see `blur.wgsl`'s anti-ghosting contract).
const SIGMA_CAP: f32 = 8.0;

/// The per-slot scratch cache: backdrop pool index → its blur textures.
pub type ScratchCache = std::collections::HashMap<usize, BlurScratch>;

/// Per-backdrop-slot scratch textures, cached across frames and recreated
/// only when the region size changes.
pub struct BlurScratch {
    /// Full-resolution copy of the region behind the pane.
    src_size: (u32, u32),
    src: wgpu::Texture,
    src_view: wgpu::TextureView,
    /// Downsampled ping/pong targets for the blur passes.
    scaled_size: (u32, u32),
    a_view: wgpu::TextureView,
    b_view: wgpu::TextureView,
    /// Keep the ping/pong textures alive alongside their views.
    _a: wgpu::Texture,
    _b: wgpu::Texture,
}

/// A blur job prepared between two main-pass segments: everything
/// `draw_composite` needs once the next segment's pass is open.
pub struct PreparedBackdrop {
    /// Group-1 bind group sampling the blurred scratch target.
    bind_group: wgpu::BindGroup,
    /// The single-instance vertex buffer for the composite quad.
    instance_buffer: wgpu::Buffer,
}

/// Builds a texture+sampler+uniform bind-group layout entry set shared by
/// the blur passes.
fn texture_bgl(device: &wgpu::Device, with_uniform: bool) -> wgpu::BindGroupLayout {
    let mut entries = vec![
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
    ];
    if with_uniform {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ByardCore - Backdrop Texture BGL"),
        entries: &entries,
    })
}

/// Assembles the off-screen blur pipeline (`blur.wgsl`, fullscreen
/// triangle, no depth). Called inside [`build_pipelines`]'s validation
/// error scope, so any GPU-side failure is captured there (RFC-0001 §8).
fn blur_pipeline(
    device: &wgpu::Device,
    blur_bgl: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ByardCore - Blur WGSL Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("blur.wgsl"))),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ByardCore - Blur Pipeline Layout"),
        bind_group_layouts: &[Some(blur_bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ByardCore - Blur Render Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Assembles the in-pass frosted-glass composite pipeline (`backdrop.wgsl`).
/// Called inside [`build_pipelines`]'s validation error scope (RFC-0001 §8).
fn composite_pipeline(
    device: &wgpu::Device,
    viewport_layout: &wgpu::BindGroupLayout,
    composite_bgl: &wgpu::BindGroupLayout,
    quad_layout: wgpu::VertexBufferLayout<'static>,
    surface_format: wgpu::TextureFormat,
    depth_stencil: wgpu::DepthStencilState,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ByardCore - Backdrop WGSL Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("backdrop.wgsl"))),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ByardCore - Backdrop Pipeline Layout"),
        bind_group_layouts: &[Some(viewport_layout), Some(composite_bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ByardCore - Backdrop Render Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[quad_layout, CompositeQuad::layout()],
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
    })
}

/// Compiles both backdrop pipelines, wrapping the whole create sequence in a
/// single validation error scope (RFC-0001 §8).
///
/// # Errors
///
/// [`ByardError::PipelineCompilation`] if a shader or pipeline fails GPU-side
/// validation — never a panic, never a software fallback.
pub async fn build_pipelines(
    device: &wgpu::Device,
    viewport_layout: &wgpu::BindGroupLayout,
    quad_layout: wgpu::VertexBufferLayout<'static>,
    surface_format: wgpu::TextureFormat,
    depth_stencil: wgpu::DepthStencilState,
) -> Result<BackdropPipelines, ByardError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let blur_bgl = texture_bgl(device, true);
    let composite_bgl = texture_bgl(device, false);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ByardCore - Backdrop Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let blur = blur_pipeline(device, &blur_bgl, surface_format);
    let composite = composite_pipeline(
        device,
        viewport_layout,
        &composite_bgl,
        quad_layout,
        surface_format,
        depth_stencil,
    );

    if let Some(error) = scope.pop().await {
        return Err(ByardError::PipelineCompilation {
            pipeline: "Backdrop".to_string(),
            reason: error.to_string(),
        });
    }

    Ok(BackdropPipelines {
        blur,
        blur_bgl,
        composite,
        composite_bgl,
        sampler,
    })
}

/// The *base* downsample factor of a backdrop's scratch targets (RFC-0023
/// resolved questions "blur quality tiers" + "blur texture resolution"):
/// the tiers differ only in resolution — the kernel is always the same
/// separable Gaussian. `high` forces 0.75×, `low` forces the cheap 0.25×,
/// and `auto` picks 0.5× on capable GPUs (the startup probe) or 0.25× on
/// software/virtual adapters. The adaptive σ cap in [`prepare`] may deepen
/// this further for large radii.
fn base_downsample(b: &BackdropInstance, auto_capable: bool) -> f32 {
    match b.quality {
        BLUR_QUALITY_HIGH => 0.75,
        BLUR_QUALITY_LOW => 0.25,
        _ if auto_capable => 0.5,
        _ => 0.25,
    }
}

/// Creates one scratch texture of `size` in the target's format.
fn scratch_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    size: (u32, u32),
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Ensures the scratch cache holds textures of exactly the required sizes
/// for `slot`, reusing the previous frame's when unchanged. Keyed by slot
/// (the backdrop's pool index) so a skipped slot — a pane whose region
/// degenerated off-screen — never misaligns the others.
fn ensure_scratch(
    scratch: &mut ScratchCache,
    slot: usize,
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    src_size: (u32, u32),
    scaled_size: (u32, u32),
) {
    let stale = scratch
        .get(&slot)
        .is_none_or(|s| s.src_size != src_size || s.scaled_size != scaled_size);
    if !stale {
        return;
    }
    let (src, src_view) = scratch_texture(device, format, src_size, "ByardCore - Backdrop Src");
    let (a, a_view) = scratch_texture(device, format, scaled_size, "ByardCore - Backdrop A");
    let (b, b_view) = scratch_texture(device, format, scaled_size, "ByardCore - Backdrop B");
    scratch.insert(
        slot,
        BlurScratch {
            src_size,
            src,
            src_view,
            scaled_size,
            a_view,
            b_view,
            _a: a,
            _b: b,
        },
    );
}

/// One off-screen blur pass: fullscreen triangle from `src` into `dst`.
#[allow(clippy::too_many_arguments)]
fn blur_pass(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    pipes: &BackdropPipelines,
    src: &wgpu::TextureView,
    dst: &wgpu::TextureView,
    params: BlurParams,
    label: &str,
) {
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ByardCore - Blur Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ByardCore - Blur Pass BG"),
        layout: &pipes.blur_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&pipes.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: dst,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&pipes.blur);
    pass.set_bind_group(0, &bind, &[]);
    pass.draw(0..3, 0..1);
}

/// The physical-pixel copy region behind a pane: the **on-screen** AABB of
/// its rect — mapped through its paint transform, which is also where any
/// enclosing scroll displacement lives (RFC-0005) — plus the ±2.5σ kernel
/// halo, clamped to the target. The composite shader maps each fragment's
/// framebuffer position into this region, so the two must agree on where
/// the pane actually is, not where layout placed it. `None` when the region
/// degenerates to zero pixels (a fully off-screen pane).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn sample_region(
    b: &BackdropInstance,
    scale: f32,
    phys: (u32, u32),
    halo_phys: f32,
) -> Option<(u32, u32, u32, u32)> {
    let corners = [
        [b.rect[0], b.rect[1]],
        [b.rect[0] + b.rect[2], b.rect[1]],
        [b.rect[0], b.rect[1] + b.rect[3]],
        [b.rect[0] + b.rect[2], b.rect[1] + b.rect[3]],
    ]
    .map(|p| b.transform.apply_point(p));
    let min_x = corners.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let min_y = corners.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let max_x = corners
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let max_y = corners
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max);
    let x0 = ((min_x * scale) - halo_phys).floor().max(0.0) as u32;
    let y0 = ((min_y * scale) - halo_phys).floor().max(0.0) as u32;
    let x1 = (((max_x * scale) + halo_phys).ceil() as u32).min(phys.0);
    let y1 = (((max_y * scale) + halo_phys).ceil() as u32).min(phys.1);
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

/// Records the whole between-segments blur job for one backdrop: region
/// copy out of `persistent`, the tier's blur pass(es) into the slot's
/// scratch targets, and returns the [`PreparedBackdrop`] the resumed main
/// pass composites with. `None` when the region degenerates to zero pixels
/// (a fully off-screen pane).
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn prepare(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    pipes: &BackdropPipelines,
    scratch: &mut ScratchCache,
    slot: usize,
    persistent: &wgpu::Texture,
    b: &BackdropInstance,
    scale: f32,
    phys: (u32, u32),
    format: wgpu::TextureFormat,
    auto_capable: bool,
) -> Option<PreparedBackdrop> {
    // The `blur` prop is the Gaussian σ (the CSS `backdrop-filter: blur(N)`
    // convention the RFC cites); the kernel samples ±2.5σ around each pixel.
    let sigma_phys = (b.blur * scale).max(0.0);
    let (x0, y0, x1, y1) = sample_region(b, scale, phys, 2.5 * sigma_phys)?;
    let src_size = (x1 - x0, y1 - y0);

    // Adaptive downsample (the anti-ghosting guarantee): deepen the tier's
    // base factor until σ, expressed in destination texels, fits the 21-tap
    // kernel's clean coverage ([`SIGMA_CAP`] keeps the σ/4 tap spacing ≤ 2
    // texels). Sparse taps at high resolution read as ghosted copies, not
    // blur; the composite's bilinear upscale hides the reduced resolution
    // (the RFC's own resolution rationale).
    let mut ds = base_downsample(b, auto_capable);
    if sigma_phys * ds > SIGMA_CAP {
        ds *= SIGMA_CAP / (sigma_phys * ds);
    }
    let ds = ds.max(1.0 / 16.0);
    let sigma_scaled = (sigma_phys * ds).min(SIGMA_CAP);
    let scaled_size = (
        ((src_size.0 as f32) * ds).ceil().max(1.0) as u32,
        ((src_size.1 as f32) * ds).ceil().max(1.0) as u32,
    );
    ensure_scratch(scratch, slot, device, format, src_size, scaled_size);
    let s = &scratch[&slot];

    // 1. Copy the region behind the pane.
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: persistent,
            mip_level: 0,
            origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &s.src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: src_size.0,
            height: src_size.1,
            depth_or_array_layers: 1,
        },
    );

    // 2. Blur into the scaled scratch (`b_view` ends holding the final
    //    result). Offsets are uniform in destination-scaled texels for both
    //    passes (`blur.wgsl` contract): the first pass samples the
    //    full-resolution copy at scaled-grid positions — the bilinear
    //    minification is the downsample.
    let scaled_texel = [1.0 / scaled_size.0 as f32, 1.0 / scaled_size.1 as f32];
    blur_pass(
        encoder,
        device,
        pipes,
        &s.src_view,
        &s.a_view,
        BlurParams {
            texel: scaled_texel,
            dir: [1.0, 0.0],
            sigma: sigma_scaled,
            _pad: [0.0; 3],
        },
        "ByardCore - Backdrop Blur H",
    );
    blur_pass(
        encoder,
        device,
        pipes,
        &s.a_view,
        &s.b_view,
        BlurParams {
            texel: scaled_texel,
            dir: [0.0, 1.0],
            sigma: sigma_scaled,
            _pad: [0.0; 3],
        },
        "ByardCore - Backdrop Blur V",
    );

    // 3. Everything the resumed main pass needs to composite.
    Some(composite_resources(
        device,
        pipes,
        &s.b_view,
        b,
        (x0, y0, x1, y1),
    ))
}

/// Builds the bind group and single-instance quad the resumed main pass
/// composites with — the hand-off half of [`prepare`].
#[allow(clippy::cast_precision_loss)]
fn composite_resources(
    device: &wgpu::Device,
    pipes: &BackdropPipelines,
    blurred: &wgpu::TextureView,
    b: &BackdropInstance,
    (x0, y0, x1, y1): (u32, u32, u32, u32),
) -> PreparedBackdrop {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ByardCore - Backdrop Composite BG"),
        layout: &pipes.composite_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(blurred),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&pipes.sampler),
            },
        ],
    });
    let quad = CompositeQuad {
        rect: b.rect,
        radii: b.radii,
        tint: b.tint,
        params: [b.saturation, b.opacity, b.depth, 0.0],
        region: [
            x0 as f32,
            y0 as f32,
            1.0 / (x1 - x0) as f32,
            1.0 / (y1 - y0) as f32,
        ],
        t_translate: b.transform.translate,
        t_scale: b.transform.scale,
        t_rotate: b.transform.rotate,
        t_origin: b.transform.origin,
    };
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ByardCore - Backdrop Composite Instance"),
        contents: bytemuck::bytes_of(&quad),
        usage: wgpu::BufferUsages::VERTEX,
    });
    PreparedBackdrop {
        bind_group,
        instance_buffer,
    }
}

/// Draws one prepared backdrop composite inside the (resumed) main UI pass,
/// scissored to its content clip (RFC-0005).
pub fn draw_composite(
    render_pass: &mut wgpu::RenderPass<'_>,
    pipes: &BackdropPipelines,
    viewport_bind_group: &wgpu::BindGroup,
    quad_buffer: &wgpu::Buffer,
    prepared: &PreparedBackdrop,
    clip: Option<u16>,
    ctx: super::ClipCtx<'_>,
) {
    render_pass.set_pipeline(&pipes.composite);
    render_pass.set_bind_group(0, viewport_bind_group, &[]);
    render_pass.set_bind_group(1, &prepared.bind_group, &[]);
    render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
    render_pass.set_vertex_buffer(1, prepared.instance_buffer.slice(..));
    super::for_each_clip_run(render_pass, 1, &[clip], ctx, |p, s, e| {
        p.draw(0..4, s..e);
    });
}
