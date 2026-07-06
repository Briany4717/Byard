//! `TextureSampler` render pipeline (M21, RFC-0001 §3.1).
//!
//! Draws a decoded image into a (optionally rounded) quad with a `fit` policy.
//! Host-side decode uses the `image` crate: the runner owns decode, the
//! interpreter only carries the `Str` path. Decoded textures are cached by path
//! so a static image is uploaded once.

use std::any::Any;
use std::collections::HashMap;

use tokio::sync::mpsc::UnboundedSender;
use wgpu::util::DeviceExt;

use crate::ByardError;
use crate::frame::{ImageFit, TextureSampler};

/// Type-erased I/O result channel sender, structurally identical to
/// `relay::IoResult`'s sender. Spelled out here (rather than imported from
/// `relay`) so the encoder never gains a dependency on the relay subsystem —
/// the async-decode result flows through the existing type-erased channel,
/// not a new cross-module call (RFC-0001 §9 / INV-11).
pub type IoResultSender = UnboundedSender<Box<dyn Any + Send>>;

/// Raw decoded RGBA8 pixels produced off the render thread by the I/O pool
/// (M29). Carries no `wgpu` handles — `Device`/`Queue` are used only on their
/// owning (render) thread, where [`TextureCache::apply_decoded`] performs the
/// upload.
#[derive(Debug)]
pub struct DecodedRgba {
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Tightly packed RGBA8 rows (`4 * width * height` bytes).
    pub bytes: Vec<u8>,
}

/// The result of one async decode, sent back through the I/O channel and
/// drained on the render thread. `result` is `Err(message)` on a missing or
/// corrupt file — the message is logged once when the entry transitions to
/// [`TextureState::Failed`].
#[derive(Debug)]
pub struct DecodedImage {
    /// The source path/key this decode was started for.
    pub src: String,
    /// Decoded pixels, or the human-readable decode error.
    pub result: Result<DecodedRgba, String>,
}

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

/// Lifecycle of a single cached texture (M29).
///
/// Replaces the old `Option<TextureEntry>` (which conflated "decode failed"
/// with "not yet decoded"). The render thread observes `Pending` for a freshly
/// requested image while its decode runs on the I/O pool, then `Ready` once
/// [`TextureCache::apply_decoded`] uploads it, or `Failed` on a bad/corrupt
/// source.
enum TextureState {
    /// Decode spawned on the I/O pool; not yet uploaded. Draws nothing.
    Pending,
    /// Decoded and uploaded; ready to sample.
    Ready(TextureEntry),
    /// Decode failed (logged once). Draws nothing.
    Failed,
}

/// Path-keyed cache of decoded textures so a static image uploads once.
///
/// Decode is **asynchronous** (RFC-0001 §5.1): [`ensure`](Self::ensure)
/// never touches the filesystem on the calling (render) thread — it inserts a
/// [`TextureState::Pending`] marker and spawns the blocking `image::open` decode
/// on the relay's I/O runtime. The decoded pixels return through the type-erased
/// I/O channel and are uploaded by [`apply_decoded`](Self::apply_decoded), which
/// runs on the render thread (where the `wgpu` `Device`/`Queue` live).
#[derive(Default)]
pub struct TextureCache {
    entries: HashMap<String, TextureState>,
}

impl TextureCache {
    /// Requests `src` without blocking: if unseen, inserts a `Pending` marker
    /// and spawns the decode on `io_handle`, returning immediately. A second
    /// call for a still-`Pending` (or already-resolved) `src` is a no-op — the
    /// `contains_key` guard ensures exactly one decode task per source.
    ///
    /// Decode happens entirely off the calling thread (INV-12); only the cheap
    /// GPU upload, later, runs on the render thread.
    pub fn ensure(
        &mut self,
        io_handle: &tokio::runtime::Handle,
        io_tx: &IoResultSender,
        src: &str,
    ) {
        if self.entries.contains_key(src) {
            return;
        }
        self.entries.insert(src.to_string(), TextureState::Pending);

        let src_owned = src.to_string();
        let tx = io_tx.clone();
        io_handle.spawn(async move {
            let result = decode_rgba(&src_owned);
            // The receiver (the render thread) may already be gone on shutdown;
            // a dropped send is fine — nothing left to paint into.
            let _ = tx.send(Box::new(DecodedImage {
                src: src_owned,
                result,
            }) as Box<dyn Any + Send>);
        });
    }

    /// Uploads an async decode result on the render thread, transitioning the
    /// entry to `Ready` (on success) or `Failed` (on a decode error, logged
    /// once). Idempotent if the entry was already resolved.
    pub fn apply_decoded(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        decoded: DecodedImage,
    ) {
        let state = match decoded.result {
            Ok(rgba) => TextureState::Ready(upload_rgba(device, queue, layout, sampler, &rgba)),
            Err(err) => {
                // A missing/corrupt image simply does not draw. The
                // warning fires once per path (the entry is now `Failed`, so
                // `ensure` never re-spawns it).
                eprintln!(
                    "byard: warning: image not found or could not be decoded: '{}': {err}",
                    decoded.src
                );
                TextureState::Failed
            }
        };
        self.entries.insert(decoded.src, state);
    }

    /// Looks up a `Ready` entry. Returns `None` for both `Pending` and
    /// `Failed` — a not-yet-loaded image draws nothing, exactly like a missing
    /// one, so callers need no new pending-state handling.
    #[must_use]
    pub fn get(&self, src: &str) -> Option<&TextureEntry> {
        match self.entries.get(src) {
            Some(TextureState::Ready(entry)) => Some(entry),
            _ => None,
        }
    }
}

/// Decodes an image file to RGBA8 **off the render thread** (no `wgpu` calls).
///
/// Blocking file read + CPU decode; this is what the I/O pool runs. The result
/// is uploaded later by [`upload_rgba`]. `pub(crate)` so the encoder's
/// no-relay fallback path can decode synchronously.
pub(crate) fn decode_rgba(src: &str) -> Result<DecodedRgba, String> {
    let img = image::open(src).map_err(|err| {
        let cwd =
            std::env::current_dir().map_or_else(|_| "?".to_string(), |p| p.display().to_string());
        format!("searched relative to {cwd}: {err}")
    })?;
    let img = img.to_rgba8();
    let (width, height) = img.dimensions();
    Ok(DecodedRgba {
        width,
        height,
        bytes: img.into_raw(),
    })
}

/// Uploads decoded RGBA8 pixels to a fresh GPU texture + bind group. Runs on
/// the render thread (the only place `Device`/`Queue` may be used).
fn upload_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    rgba: &DecodedRgba,
) -> TextureEntry {
    let DecodedRgba {
        width,
        height,
        bytes,
    } = rgba;
    let (width, height) = (*width, *height);
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
        bytes,
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
    TextureEntry {
        bind_group,
        width,
        height,
    }
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
        depth_stencil: Some(super::draw_depth_stencil()),
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
    depths: &[f32],
    clip_slice: &[Option<u16>],
    ctx: super::ClipCtx<'_>,
) {
    if textures.is_empty() {
        return;
    }
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, viewport_bind_group, &[]);
    render_pass.set_vertex_buffer(0, quad_buffer.slice(..));

    for (i, t) in textures.iter().enumerate() {
        let Some(entry) = cache.get(&t.src) else {
            continue;
        };
        // Content clip (RFC-0005): scissor this image to its ScrollView
        // viewport; skip it entirely if scrolled fully out of view.
        let Some((sx, sy, sw, sh)) = super::clip_scissor(ctx, clip_slice.get(i).copied().flatten())
        else {
            continue;
        };
        render_pass.set_scissor_rect(sx, sy, sw, sh);
        let uv_xform = uv_transform(t.fit, entry.width, entry.height, t.rect[2], t.rect[3]);
        // misc.y carries this image's draw-order depth (NDC-z); the shader reads
        // it as position.z. Missing depth falls back to the far plane.
        let depth = depths
            .get(i)
            .copied()
            .unwrap_or(crate::frame::DRAW_DEPTH_CLEAR);
        let instance = TextureInstance {
            rect: t.rect,
            radii: t.radii,
            uv_xform,
            misc: [t.opacity, depth, 0.0, 0.0],
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Builds a multi-threaded Tokio runtime mirroring `Relay`'s (no time/io
    /// drivers — decode is pure compute, results travel a plain channel).
    fn io_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .build()
            .expect("io runtime")
    }

    /// Writes a `w×h` solid-colour PNG fixture to a unique temp path and
    /// returns it. The caller removes it when done.
    fn write_png_fixture(tag: &str, w: u32, h: u32) -> std::path::PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("byard_m29_{tag}_{pid}_{nanos}.png"));
        let img = image::RgbaImage::from_pixel(w, h, image::Rgba([10, 20, 30, 255]));
        img.save(&path).expect("save fixture png");
        path
    }

    /// INV-12: `ensure` must return without doing the decode itself — the
    /// blocking `image::open` runs on the I/O pool, not the calling thread.
    #[test]
    fn ensure_does_not_block_when_decoding_a_slow_fixture() {
        let rt = io_runtime();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Box<dyn Any + Send>>();
        // A 1024×1024 PNG: its decode (inflate + unfilter) takes many
        // milliseconds — far longer than the microseconds `ensure` needs just
        // to spawn the task and return.
        let path = write_png_fixture("inv12", 1024, 1024);
        let src = path.to_str().unwrap();

        // Warm the pool so its worker threads are already running — we are
        // timing `ensure`'s enqueue, not Tokio's one-time lazy thread spin-up.
        rt.block_on(async {
            tokio::runtime::Handle::current()
                .spawn(async {})
                .await
                .unwrap();
        });

        let mut cache = TextureCache::default();
        let start = Instant::now();
        cache.ensure(rt.handle(), &tx, src);
        let elapsed = start.elapsed();

        // The load-bearing INV-12 check is *structural*, not a tight timing
        // bound (CI runners are too jittery for sub-millisecond wall-clock
        // assertions): immediately after `ensure` returns, the decode result
        // has **not** arrived on the channel. A blocking `ensure` (one that ran
        // `image::open` inline before returning) would have already sent its
        // result, so this is empty only because the decode is still in flight
        // on the I/O pool.
        assert!(
            rx.try_recv().is_err(),
            "ensure must not have completed the decode synchronously"
        );
        // The decode is deferred, so nothing is `Ready` yet either.
        assert!(
            cache.get(src).is_none(),
            "image must be Pending (not Ready) immediately after ensure"
        );
        // Secondary sanity bound, deliberately generous: `ensure` only spawns,
        // so it returns far below the multi-ms decode even on a loaded runner.
        assert!(
            elapsed < Duration::from_millis(25),
            "ensure should return promptly (only spawns); took {elapsed:?}"
        );

        // It does complete off-thread, proving the decode actually ran there.
        let received = rt.block_on(rx.recv()).expect("decode result");
        let decoded = received
            .downcast::<DecodedImage>()
            .expect("result is a DecodedImage");
        assert!(decoded.result.is_ok(), "fixture should decode cleanly");

        let _ = std::fs::remove_file(&path);
    }

    /// `ensure` called twice for the same still-`Pending` path must spawn only
    /// one decode task (the `contains_key` guard) — so exactly one result lands
    /// on the channel.
    #[test]
    fn ensure_called_twice_for_the_same_pending_path_spawns_one_decode_task() {
        let rt = io_runtime();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Box<dyn Any + Send>>();
        let path = write_png_fixture("dedup", 8, 8);
        let src = path.to_str().unwrap();

        let mut cache = TextureCache::default();
        cache.ensure(rt.handle(), &tx, src);
        cache.ensure(rt.handle(), &tx, src);

        // Block until the (single) task completes, then assert the channel is
        // empty — the second `ensure` provably never spawned, since the first
        // inserted `Pending` synchronously before either task could run.
        let _first = rt.block_on(rx.recv()).expect("one decode result");
        assert!(
            rx.try_recv().is_err(),
            "a second decode task must not have been spawned"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Returns `(device, queue)` for a real adapter, or `None` headless.
    fn try_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        Some((Arc::new(device), Arc::new(queue)))
    }

    /// A texture is `Pending` (drawing nothing) until its decode result is
    /// drained and applied, after which it is `Ready`. Since every primitive is
    /// re-emitted dirty each tick, that `Ready` texture then paints on the next
    /// frame with no extra dirty signal needed.
    #[test]
    fn texture_becomes_ready_after_io_result_drain() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter — skipping texture-ready drain test");
            return;
        };
        let layout = bind_group_layout(&device);
        let smp = sampler(&device);

        let rt = io_runtime();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Box<dyn Any + Send>>();
        let path = write_png_fixture("ready", 16, 16);
        let src = path.to_str().unwrap();

        let mut cache = TextureCache::default();
        cache.ensure(rt.handle(), &tx, src);
        assert!(cache.get(src).is_none(), "Pending before drain → no draw");

        let received = rt.block_on(rx.recv()).expect("decode result");
        let decoded = *received.downcast::<DecodedImage>().unwrap();
        cache.apply_decoded(&device, &queue, &layout, &smp, decoded);

        assert!(
            cache.get(src).is_some(),
            "Ready after drain → entry available to draw"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// A bad path decodes to `Failed` (drawing nothing), never a panic, and
    /// `get` keeps returning `None`.
    #[test]
    fn missing_image_resolves_to_failed_not_panic() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter — skipping failed-decode test");
            return;
        };
        let layout = bind_group_layout(&device);
        let smp = sampler(&device);
        let mut cache = TextureCache::default();

        let decoded = DecodedImage {
            src: "/no/such/byard/image.png".to_string(),
            result: decode_rgba("/no/such/byard/image.png"),
        };
        assert!(decoded.result.is_err(), "missing file must fail to decode");
        cache.apply_decoded(&device, &queue, &layout, &smp, decoded);
        assert!(cache.get("/no/such/byard/image.png").is_none());
    }
}
