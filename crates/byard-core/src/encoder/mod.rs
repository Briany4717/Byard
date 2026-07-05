//! # Encoder
//!
//! Multi-pipeline `wgpu` command dispatch.
//!
//! This subsystem owns the specialised render pipelines compiled at startup:
//!
//! - **`SolidBox`** — Axis-aligned rectangles with solid fill **and per-corner
//!   `border-radius` via an analytical SDF**. This absorbs the basic rounded-rect
//!   case from the RFC §3.1 `DecoratedBox` column by design: a single instanced
//!   pipeline handles both plain and rounded rectangles with zero extra GPU state
//!   switches, because the radius parameters are part of the per-instance vertex
//!   data rather than a pipeline variant.
//! - **`DecoratedBox`** — Rectangles with gradients, box-shadows, and parametric
//!   decorations (future sub-issue).
//! - **`TextGlyph`** — Text rendering via a `glyphon` glyph atlas (future).
//! - **`TextureSampler`** — UV-mapped quads for decoded images and icons (future).
//!
//! Primitives are batched into Z-bins (stacking contexts) and ordered first by
//! pipeline, then by local Z-index, to minimise GPU context switches.
//!
//! Pipeline creation is wrapped in `Device::push_error_scope` (returns an
//! `ErrorScopeGuard` in wgpu 28+) and `scope.pop().await` with
//! `ErrorFilter::Validation`. Failures are surfaced as
//! [`ByardError::PipelineCompilation`](crate::ByardError::PipelineCompilation)
//! — the engine never panics on a GPU error.

pub mod decorated_box;
pub mod gpu_timer;
pub mod text_glyph;
pub mod texture_sampler;
pub mod vector_msdf;

pub use gpu_timer::GpuTimer;

/// Name of the single GPU pass this codebase currently times (RFC-0013 §"GPU
/// timing"): `SolidBox`, `DecoratedBox`, `TextureSampler`, and `TextGlyph` all
/// draw within one `wgpu::RenderPass` (see [`draw_ui_pass`]), so — unlike the
/// RFC's four-pipeline illustration — there is exactly one pass boundary to
/// time today. Per-pipeline GPU timing needs the encoder to split that pass
/// first; tracked as a follow-up, not attempted here.
pub const GPU_UI_PASS_SCOPE: &str = "gpu.ui_pass";

use std::sync::Arc;

use bytemuck;
use wgpu::util::DeviceExt;

use crate::ByardError;
use crate::frame::{AtlasUpload, Rect, RenderFrame, Transform, VectorInstance, Viewport};
use text_glyph::{TextGlyphPipeline, TextLine};
use vector_msdf::VectorAtlas;

/// Re-exported from [`crate::frame`] — the canonical definition now lives
/// there so the Logic thread can populate [`RenderFrame::instances`] without
/// importing from the Encoder subsystem (RFC-0001 §9).
pub use crate::frame::BoxInstance;

/// Re-exported so the engine's render-thread drain (M29) can downcast the
/// type-erased I/O result back to a decoded image and hand it to
/// [`EncoderSubsystem::apply_decoded`].
pub use texture_sampler::DecodedImage;

/// The async-decode plumbing the engine hands the encoder (M29), cloned out of
/// `Relay`: a Tokio handle to spawn the blocking `image::open` decode on, and
/// the type-erased result sender those tasks report back through. Held as a
/// plain struct (not a `relay` import) so the encoder never depends on the
/// relay subsystem (RFC-0001 §9 / INV-11).
struct IoContext {
    handle: tokio::runtime::Handle,
    tx: texture_sampler::IoResultSender,
}

impl BoxInstance {
    /// Returns the `wgpu` vertex buffer layout for the instance buffer.
    ///
    /// Step mode is [`wgpu::VertexStepMode::Instance`] — the GPU advances one
    /// entry per drawn rectangle, not per vertex of the shared unit quad.
    #[must_use]
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // rect: [x, y, w, h]
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // color: [r, g, b, a]
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // radii: [tl, tr, br, bl]
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // transform.translate
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // transform.scale
                wgpu::VertexAttribute {
                    offset: 56,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // transform.rotate
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                },
                // transform.origin
                wgpu::VertexAttribute {
                    offset: 68,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // transform.opacity
                wgpu::VertexAttribute {
                    offset: 76,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Vertices of the shared unit quad (two triangles via `TriangleStrip`).
///
/// All rectangle instances share this single buffer. The vertex shader scales
/// and translates the quad to match each instance's `rect` field.
const QUAD_VERTICES: &[f32] = &[
    0.0, 0.0, // Top-Left
    1.0, 0.0, // Top-Right
    0.0, 1.0, // Bottom-Left
    1.0, 1.0, // Bottom-Right
];

/// Owns all wgpu resources for the `SolidBox` render pipeline.
///
/// Initialised once via [`EncoderSubsystem::init`]. Holds `Arc` handles to
/// the device and queue so the render thread can submit commands without
/// locking the logic thread.
pub struct EncoderSubsystem {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    render_pipeline: wgpu::RenderPipeline,
    /// No-blend variant of `SolidBox`'s pipeline, used only to paint a fully
    /// transparent "clear quad" over a dirty rect before it is repainted.
    ///
    /// `render_pipeline`'s `ALPHA_BLENDING` state means
    /// `dst_new = src.rgb * src.a + dst.rgb * (1 - src.a)` — wherever a
    /// fragment's alpha is 0 (most of a glyph's bounding box, since
    /// letterforms are sparse), the destination is left **unchanged**.
    /// Combined with `LoadOp::Load` on an incremental frame, that means old
    /// ink can never be erased by simply redrawing new content with
    /// standard "over" blending.
    /// `clear_pipeline` uses `blend: None`, so the fragment shader's output
    /// unconditionally **replaces** the destination regardless of its
    /// alpha, making it possible to genuinely wipe a rect.
    clear_pipeline: wgpu::RenderPipeline,
    quad_buffer: wgpu::Buffer,
    viewport_buffer: wgpu::Buffer,
    viewport_bind_group: wgpu::BindGroup,
    /// Text rendering pipeline — shares the UI render pass with `SolidBox`.
    text_pipeline: TextGlyphPipeline,
    /// `DecoratedBox` pipeline (M21) — border/shadow/opacity boxes. Shares the
    /// viewport bind group (group 0) with `SolidBox`.
    decorated_pipeline: wgpu::RenderPipeline,
    /// `TextureSampler` pipeline (M21) — `Image` quads.
    texture_pipeline: wgpu::RenderPipeline,
    /// Texture+sampler bind group layout (group 1) for `texture_pipeline`.
    texture_bind_group_layout: wgpu::BindGroupLayout,
    /// Shared linear sampler for all sampled images.
    image_sampler: wgpu::Sampler,
    /// Path-keyed cache of decoded image textures (M21).
    texture_cache: texture_sampler::TextureCache,
    /// `VectorMSDF` pipeline (RFC-0009 §1, the fifth pipeline) — samples
    /// [`vector_atlas`](Self::vector_atlas) to draw crisp monochrome icons.
    vector_pipeline: wgpu::RenderPipeline,
    /// The MSDF atlas: an array texture uploaded to by the JIT/AOT paths via
    /// [`RenderFrame::atlas_uploads`] (RFC-0009 §2-C, INV-8 — this is the only
    /// place `Queue::write_texture` is called for it).
    vector_atlas: VectorAtlas,
    /// Async-decode plumbing (M29): the relay's I/O runtime handle and the
    /// type-erased result sender, installed by the engine via
    /// [`set_io_context`](Self::set_io_context). `None` for a bare encoder
    /// constructed without a relay (e.g. GPU-readback tests that never load
    /// images) — in that case [`encode_frame_with_decorations`] decodes
    /// synchronously, since there is no I/O pool to offload to.
    io: Option<IoContext>,
    /// DPI scale factor, derived once per resize from the OS-reported value.
    ///
    /// Stored here so `encode_frame` can pass it to `TextGlyphPipeline::prepare`
    /// without requiring the caller to supply it per-frame.
    scale_factor: f32,
    /// Set by [`update_viewport`](EncoderSubsystem::update_viewport) and
    /// consumed (cleared) by [`encode_frame`](EncoderSubsystem::encode_frame).
    ///
    /// Forces `TextGlyphPipeline::prepare` to re-prepare even when no text
    /// content has changed — necessary after a viewport resize because glyphon's
    /// `Viewport` resolution changed.
    viewport_dirty: bool,
    /// Persistent off-screen colour target that incremental (scissored) draws
    /// actually land on.
    ///
    /// RFC §3.3's scissor clipping only makes sense against a render target
    /// with *retained* content across frames. The swapchain image returned by
    /// `wgpu::Surface::get_current_texture` does not offer that guarantee
    /// under multi-buffering — wgpu is free to rotate in a stale or
    /// uninitialised image on any given frame. `persistent_color` is the
    /// real, always-retained surface that `LoadOp::Load` + `set_scissor_rect`
    /// draw into; the swapchain image only ever receives a full, unscissored
    /// copy of this texture's current contents once per frame (see
    /// `encode_frame`'s final `copy_texture_to_texture` call).
    persistent_color: wgpu::Texture,
    /// View of [`persistent_color`](Self::persistent_color), cached to avoid
    /// recreating it every frame.
    persistent_view: wgpu::TextureView,
    /// Frame-local draw-order depth buffer (RFC-0011 cross-pass paint order):
    /// cleared to the far plane every pass and rebuilt from the current
    /// primitive list, so it needs no cross-frame bookkeeping. Recreated
    /// alongside `persistent_color` on resize so the two always match in size.
    persistent_depth_view: wgpu::TextureView,
    /// Pixel format shared by `persistent_color` and the swapchain surface.
    ///
    /// Stored so [`update_viewport`](Self::update_viewport) can recreate
    /// `persistent_color` at the new size on resize without requiring the
    /// caller to pass the format again every time.
    surface_format: wgpu::TextureFormat,
    /// Physical-pixel width of [`persistent_color`](Self::persistent_color).
    phys_w: u32,
    /// Physical-pixel height of [`persistent_color`](Self::persistent_color).
    phys_h: u32,
    /// `true` when the next [`encode_frame`](EncoderSubsystem::encode_frame)
    /// must draw everything, ignoring per-`TextLine` dirty bits.
    ///
    /// Set on construction (nothing has been drawn into `persistent_color`
    /// yet) and whenever the surface is resized (the recreated texture's
    /// contents are undefined). Cleared at the end of every `encode_frame`
    /// call. This is intentionally independent of `EvaluatorTick`'s dirty
    /// collection: a freshly registered `Signal` reports an empty dirty set
    /// on its very first collection (nothing has mutated yet), so relying on
    /// `TextLine::dirty` alone would mean the first frame draws nothing.
    needs_full_redraw: bool,
    /// Number of `BoxInstance`s passed to the previous `encode_frame` call.
    ///
    /// `BoxInstance`s carry no per-instance dirty bit (nothing in the current
    /// codebase mutates a `BoxInstance` after construction), so a *count*
    /// change is the only structural signal
    /// available that the instance list changed shape. A mismatch forces a
    /// full redraw so a future caller that does start mutating the instance
    /// list cannot silently lose a newly added box to the scissor rect.
    last_instance_count: usize,
    /// Number of `TextLine`s passed to the previous `encode_frame` call, for
    /// the same structural-change reasoning as
    /// [`last_instance_count`](Self::last_instance_count).
    last_text_count: usize,
    /// Per-line bounding boxes (logical pixels) from the previous
    /// `encode_frame` call, positionally aligned with that call's `texts`
    /// slice.
    ///
    /// A dirty line's scissor contribution must cover **both** its current
    /// bounds and its bounds from the previous frame: if a line shrinks or
    /// moves, its old footprint can fall entirely outside the new bounds,
    /// leaving stale ink permanently outside the scissor rect (and
    /// therefore never cleared). See [`dirty_text_bounds`].
    last_text_bounds: Vec<Rect>,
    /// Per-`BoxInstance` bounding boxes from the previous `encode_frame` call,
    /// positionally aligned with that call's `instances` slice. Mirrors
    /// [`last_text_bounds`](Self::last_text_bounds) for the solid-box pipeline
    /// (M26) so a moved/shrunk box still clears its old footprint.
    last_box_bounds: Vec<Rect>,
    /// Per-`DecoratedBox` bounding boxes from the previous call (M27), aligned
    /// with that call's `decorated` slice. Same shrink/move-safety contract.
    last_decorated_bounds: Vec<Rect>,
    /// Per-`TextureSampler` bounding boxes from the previous call (M27),
    /// aligned with that call's `textures` slice.
    last_texture_bounds: Vec<Rect>,
    /// The [`RenderFrame::version`] value from the last frame rendered via
    /// [`encode_frame_from_relay`](Self::encode_frame_from_relay).
    ///
    /// When the relay advances beyond this version (meaning at least one
    /// frame was published that the render thread did not process), the
    /// encoder forces a full redraw and text reshape to ensure no content
    /// changes are silently skipped — even if the current frame's
    /// `TextLine::dirty` bits are all `false` because they were cleared
    /// before the render thread read the frame.
    last_relay_version: u64,
    /// Async GPU pass timing (RFC-0013 §"GPU timing"), or `None` if the
    /// device lacks `wgpu::Features::TIMESTAMP_QUERY` (P5) — checked once at
    /// construction, never re-probed per frame.
    gpu_timer: Option<GpuTimer>,
    /// This frame's `Gpu`-tagged samples, drained from `gpu_timer` at the
    /// start of the next [`encode_frame_with_decorations`](Self::encode_frame_with_decorations)
    /// call and pushed onto the calling (render) thread's telemetry ring —
    /// reused across calls so draining never allocates once warmed up.
    gpu_samples_scratch: Vec<crate::telemetry::Sample>,
    /// Set when [`GpuTimer::resolve_and_copy`] ran during the last
    /// [`encode_frame_with_decorations`](Self::encode_frame_with_decorations)
    /// call (i.e. a pass was actually timed this frame); consumed by
    /// [`submit`](Self::submit), which must only call
    /// [`GpuTimer::request_map`] when there is a fresh copy to map.
    gpu_timing_pending: bool,
}

impl EncoderSubsystem {
    /// Compiles all GPU pipelines using an already-created device and queue.
    ///
    /// Adapter selection and device creation are the responsibility of the
    /// caller (typically [`Engine::init`](crate::engine::Engine::init)), which
    /// also configures the `wgpu::Surface` before calling this method.
    ///
    /// Shader compilation and pipeline creation are wrapped in a
    /// `push_error_scope` / `pop_error_scope` pair (RFC §8). Any GPU-side
    /// validation failure is returned as
    /// [`ByardError::PipelineCompilation`](crate::ByardError::PipelineCompilation)
    /// — this method never panics on a GPU error.
    ///
    /// `width`/`height` are the surface's initial dimensions in **physical
    /// pixels**, used to allocate the persistent intermediate colour target
    /// (see [`persistent_color`](Self::persistent_color)) at construction
    /// time so the very first [`encode_frame`](Self::encode_frame) call has
    /// somewhere to draw into.
    ///
    /// # Errors
    ///
    /// - [`ByardError::PipelineCompilation`] — the WGSL shader or the pipeline
    ///   descriptor failed GPU-side validation.
    // A resource-wiring constructor: it allocates the quad/viewport buffers,
    // five pipelines, the persistent target and the texture cache. Splitting it
    // further would scatter one cohesive setup across helpers with no clarity
    // gain, so the line-count lint is allowed here specifically.
    #[allow(clippy::too_many_lines)]
    pub async fn init(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        scale_factor: f32,
        width: u32,
        height: u32,
    ) -> Result<Self, ByardError> {
        // Static geometry shared by every SolidBox instance.
        let quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ByardCore - Static Quad Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let viewport_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ByardCore - Viewport Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ByardCore - Viewport Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(16),
                },
                count: None,
            }],
        });

        let viewport_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ByardCore - Viewport Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: viewport_buffer.as_entire_binding(),
            }],
        });

        let quad_layout = wgpu::VertexBufferLayout {
            array_stride: 8, // 2 × f32
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };

        // `bind_group_layout` is passed into the helper so that `pipeline_layout`
        // can be created inside the error scope alongside the shader and pipeline,
        // matching the full sequence required by RFC §8.
        let render_pipeline = build_solid_box_pipeline(
            &device,
            &bind_group_layout,
            quad_layout,
            surface_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            draw_depth_stencil(),
            "ByardCore - SolidBox Render Pipeline",
        )
        .await?;

        // See `EncoderSubsystem::clear_pipeline`'s doc comment — same shader
        // and layout, only the blend state differs (no blending → the
        // fragment output unconditionally replaces the destination).
        let clear_pipeline = build_solid_box_pipeline(
            &device,
            &bind_group_layout,
            wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
            surface_format,
            None,
            clear_depth_stencil(),
            "ByardCore - SolidBox Clear Pipeline",
        )
        .await?;

        let text_pipeline = TextGlyphPipeline::new(&device, &queue, surface_format).await?;

        // M21 pipelines (RFC-0001 §3.1).
        let (decorated_pipeline, texture_pipeline, texture_bind_group_layout, image_sampler) =
            build_m21_pipelines(&device, &bind_group_layout, surface_format).await?;

        // `VectorMSDF` pipeline (RFC-0009 §1, the fifth pipeline).
        let vector_atlas_layout = vector_msdf::bind_group_layout(&device);
        let vector_sampler = vector_msdf::sampler(&device);
        let vector_pipeline = vector_msdf::build_pipeline(
            &device,
            &bind_group_layout,
            &vector_atlas_layout,
            wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
            surface_format,
        )
        .await?;
        // One layer to start; the dev allocator (M48) grows this on demand.
        let vector_atlas = VectorAtlas::new(
            &device,
            &vector_atlas_layout,
            &vector_sampler,
            vector_msdf::ATLAS_SIZE,
            1,
        );

        let (persistent_color, persistent_view) =
            create_persistent_target(&device, surface_format, width, height);
        let persistent_depth_view = create_depth_target(&device, width, height);

        let gpu_timer = GpuTimer::new(&device, &queue, &[GPU_UI_PASS_SCOPE]);

        Ok(Self {
            device,
            queue,
            render_pipeline,
            clear_pipeline,
            quad_buffer,
            viewport_buffer,
            viewport_bind_group,
            text_pipeline,
            decorated_pipeline,
            texture_pipeline,
            texture_bind_group_layout,
            image_sampler,
            texture_cache: texture_sampler::TextureCache::default(),
            vector_pipeline,
            vector_atlas,
            io: None,
            scale_factor,
            viewport_dirty: false,
            persistent_depth_view,
            persistent_color,
            persistent_view,
            surface_format,
            phys_w: width,
            phys_h: height,
            // Nothing has been drawn into `persistent_color` yet — the first
            // `encode_frame` call must draw everything unconditionally.
            needs_full_redraw: true,
            last_instance_count: 0,
            last_text_count: 0,
            last_text_bounds: Vec::new(),
            last_box_bounds: Vec::new(),
            last_decorated_bounds: Vec::new(),
            last_texture_bounds: Vec::new(),
            last_relay_version: 0,
            gpu_timer,
            gpu_samples_scratch: Vec::new(),
            gpu_timing_pending: false,
        })
    }

    /// Returns a reference to the underlying `wgpu` device.
    ///
    /// Used by [`Engine`](crate::engine::Engine) to configure and reconfigure
    /// the `wgpu::Surface` without duplicating the device handle.
    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Whether this encoder's GPU pass timing is active (RFC-0013 **P5**) —
    /// `false` when the device lacks `wgpu::Features::TIMESTAMP_QUERY`. Used
    /// by the overlay/CLI to show a clear "GPU timing unavailable" notice
    /// instead of a silently empty GPU section.
    #[must_use]
    pub fn gpu_timing_available(&self) -> bool {
        self.gpu_timer.is_some()
    }

    /// Submits a command buffer to the GPU queue.
    ///
    /// Thin wrapper around `queue.submit` so that callers outside this module
    /// do not need to hold a separate reference to the queue. Also requests
    /// this frame's GPU-timing readback map, if a pass was timed — `wgpu`
    /// requires the `map_async` request to happen only after the command
    /// buffer that writes the mapped buffer has actually been submitted
    /// (see [`GpuTimer::resolve_and_copy`]'s doc comment).
    pub(crate) fn submit(&mut self, buffer: wgpu::CommandBuffer) {
        self.queue.submit(std::iter::once(buffer));
        if self.gpu_timing_pending {
            self.gpu_timing_pending = false;
            if let Some(timer) = &mut self.gpu_timer {
                timer.request_map();
            }
        }
    }

    /// Drains any GPU pass timings that finished resolving since the last
    /// call (RFC-0013 "GPU timing": never blocks, so a slot may still be
    /// pending — it is simply checked again next time) and pushes them onto
    /// this (render) thread's own telemetry ring, alongside whatever this
    /// thread profiles directly — the overlay drains both. Extracted out of
    /// [`encode_frame_with_decorations`](Self::encode_frame_with_decorations)
    /// purely to keep that function under clippy's line-count threshold.
    fn drain_gpu_samples_into_telemetry(&mut self) {
        let Some(timer) = &mut self.gpu_timer else {
            return;
        };
        self.gpu_samples_scratch.clear();
        timer.drain_ready(&self.device, &mut self.gpu_samples_scratch);
        for sample in self.gpu_samples_scratch.drain(..) {
            crate::telemetry::push_sample(sample);
        }
    }

    /// Uploads updated viewport dimensions to the GPU uniform buffer and
    /// notifies the text pipeline of the new resolution.
    ///
    /// `phys_w`/`phys_h` are the new surface dimensions in **physical pixels**.
    /// `scale` is the OS DPI scale factor; it is stored so that `encode_frame`
    /// can pass the correct value to [`TextGlyphPipeline::prepare`] without
    /// requiring the caller to supply it per-frame.
    ///
    /// Must be called whenever the surface is resized before the next frame.
    ///
    /// If `phys_w`/`phys_h` differ from the currently allocated
    /// [`persistent_color`](Self::persistent_color) size, that texture is
    /// recreated at the new size and `needs_full_redraw` is set — the
    /// recreated texture's contents are undefined, so the next
    /// `encode_frame` must repopulate it in full rather than trying to
    /// incrementally patch stale (or garbage) pixels.
    pub fn update_viewport(&mut self, viewport: Viewport, phys_w: u32, phys_h: u32, scale: f32) {
        // SolidBox viewport uniform (logical pixels, padded to 16 bytes).
        let size_data = [viewport.width, viewport.height, 0.0_f32, 0.0];
        self.queue
            .write_buffer(&self.viewport_buffer, 0, bytemuck::cast_slice(&size_data));

        // glyphon Viewport (physical pixels — glyphon always operates in physical px).
        self.text_pipeline
            .update_resolution(&self.queue, phys_w, phys_h);

        self.scale_factor = scale;
        self.viewport_dirty = true;

        if phys_w != self.phys_w || phys_h != self.phys_h {
            let (persistent_color, persistent_view) =
                create_persistent_target(&self.device, self.surface_format, phys_w, phys_h);
            self.persistent_color = persistent_color;
            self.persistent_view = persistent_view;
            self.persistent_depth_view = create_depth_target(&self.device, phys_w, phys_h);
            self.phys_w = phys_w;
            self.phys_h = phys_h;
            self.needs_full_redraw = true;
        }
    }

    /// Encodes a single UI frame into a `CommandBuffer` ready for queue submission.
    ///
    /// Implements RFC-0001 §3.3 (dirty rectangles and scissor clipping). The
    /// actual incremental drawing target is **not** `target` — it is
    /// [`persistent_color`](Self::persistent_color), an off-screen texture
    /// that, unlike the swapchain image, is guaranteed to retain its
    /// contents across frames. See that field's doc comment for why this
    /// indirection exists.
    ///
    /// Three cases, decided by [`needs_full_redraw_this_frame`]:
    ///
    /// - **Full redraw** (first call, or after a resize, or the
    ///   instance/text count changed shape since the previous call): the
    ///   inner pass clears `persistent_color` and draws every `BoxInstance`
    ///   and every `TextLine` unscissored — identical to this function's
    ///   pre-#31 behaviour.
    /// - **Incremental** (not a full redraw, and at least one `TextLine` is
    ///   dirty): the inner pass loads (does not clear) `persistent_color`,
    ///   restricts fragment writes to `wgpu::RenderPass::set_scissor_rect`
    ///   for the union of the dirty lines' *current and previous* bounding
    ///   boxes (see [`dirty_text_bounds`]), then draws a fully transparent
    ///   "clear quad" over exactly that rect via [`clear_pipeline`](
    ///   Self::clear_pipeline) before redrawing — standard alpha blending
    ///   alone cannot erase stale content (see that field's doc comment),
    ///   so this step is required, not optional. Every `BoxInstance` is
    ///   then redrawn too (the clear quad may have wiped one), bounded by
    ///   the active scissor rect. `TextGlyphPipeline::prepare`/`render`
    ///   still receive the **full, unfiltered** `texts` slice; see the note
    ///   above the `render` call below for why.
    /// - **Nothing dirty**: the inner pass is skipped entirely — zero GPU
    ///   work beyond the mandatory composite step below.
    ///
    /// In every case, the frame ends with an unscissored
    /// `copy_texture_to_texture` of `persistent_color`'s current contents
    /// onto `target` (the swapchain image), since the swapchain's own
    /// previous contents are never assumed valid.
    ///
    /// # Instance buffer lifetime
    ///
    /// The `SolidBox` instance buffer is allocated per call and dropped after
    /// `encoder.finish()`. A persistent ring-buffer strategy is a future sub-issue.
    ///
    /// # Errors
    ///
    /// - [`ByardError::TextPrepare`] — glyphon atlas upload failed.
    /// - [`ByardError::TextRender`] — glyphon render recording failed.
    pub fn encode_frame(
        &mut self,
        target: &wgpu::Texture,
        instances: &[BoxInstance],
        texts: &[TextLine],
    ) -> Result<wgpu::CommandBuffer, ByardError> {
        // No `RenderFrame` here (raw solid+text convenience path): empty depths
        // → far-plane fallback, i.e. the pre-depth type-grouped pass order.
        self.encode_frame_with_decorations(
            target,
            instances,
            texts,
            &[],
            &[],
            &[],
            &[],
            DrawDepths::default(),
        )
    }

    /// Full encode path including the M21 `DecoratedBox`/`TextureSampler`
    /// primitives. [`encode_frame`](Self::encode_frame) forwards here with empty
    /// decoration slices, keeping the common (solid + text) path byte-identical.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn encode_frame_with_decorations(
        &mut self,
        target: &wgpu::Texture,
        instances: &[BoxInstance],
        texts: &[TextLine],
        decorated: &[crate::frame::DecoratedBox],
        textures: &[crate::frame::TextureSampler],
        vectors: &[VectorInstance],
        atlas_uploads: &[AtlasUpload],
        depths: DrawDepths<'_>,
    ) -> Result<wgpu::CommandBuffer, ByardError> {
        // RFC-0009 §2-C / INV-8: the single place this atlas is ever written
        // to. Applied unconditionally (not gated on `should_draw` below) so a
        // pending upload is never silently dropped on a skip-frame.
        self.vector_atlas.apply_uploads(&self.queue, atlas_uploads);

        self.request_textures(textures);

        self.drain_gpu_samples_into_telemetry();

        let full_redraw = needs_full_redraw_this_frame(
            self.needs_full_redraw,
            self.last_instance_count,
            instances.len(),
            self.last_text_count,
            texts.len(),
        );
        // M27: `DecoratedBox`/`TextureSampler` no longer force a full,
        // unscissored redraw just by being present — they now carry their own
        // `dirty` bit and contribute to the same incremental scissor union as
        // text and solid boxes (RFC-0001 §3.3).

        // Every `BoxInstance` in the frame is treated as dirty: the lowering
        // re-emits the whole instance list each tick and `BoxInstance`
        // is a pure GPU `Pod` type with no room for a per-instance dirty bit,
        // so the honest, layout-safe representation of "a box may
        // have changed" is "all of them might have." This is what fixes the M26
        // bug: a box-only mutation (no dirty text) now always produces a
        // non-empty scissor and reaches the screen.
        let instances_dirty = vec![true; instances.len()];

        // Only meaningful on a non-full-redraw frame — every primitive is
        // drawn regardless of its dirty bit when `full_redraw` is true.
        let scissor = if full_redraw {
            None
        } else {
            compute_scissor(
                &ScissorInputs {
                    texts,
                    prev_texts: &self.last_text_bounds,
                    instances,
                    instances_dirty: &instances_dirty,
                    prev_boxes: &self.last_box_bounds,
                    decorated,
                    prev_decorated: &self.last_decorated_bounds,
                    textures,
                    prev_textures: &self.last_texture_bounds,
                },
                self.scale_factor,
                self.phys_w,
                self.phys_h,
            )
        };

        // Nothing to (re)draw into `persistent_color` this frame: not a full
        // redraw, no `TextLine` is dirty, and no vector glyph just landed. The
        // swapchain still gets a fresh copy of `persistent_color` below — just
        // unchanged from last frame. A fresh `atlas_uploads` entry forces a
        // draw even with an empty scissor, since a placeholder→resident
        // transition changes a `VectorInstance`'s content but not its rect —
        // the scissor union (rect-based) would otherwise miss it entirely.
        let should_draw = full_redraw || scissor.is_some() || !atlas_uploads.is_empty();

        // ── Text prepare (before the render pass) ─────────────────────────────
        if should_draw {
            let viewport_dirty = self.viewport_dirty;
            self.text_pipeline.prepare(
                &self.device,
                &self.queue,
                texts,
                depths.text,
                self.scale_factor,
                viewport_dirty,
            )?;
        }
        self.viewport_dirty = false;

        // ── Command encoding ──────────────────────────────────────────────────
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ByardCore - Frame Command Encoder"),
            });

        if should_draw {
            draw_ui_pass(
                &mut encoder,
                &self.persistent_view,
                &self.persistent_depth_view,
                &self.device,
                &DrawPipelines {
                    solid: &self.render_pipeline,
                    clear: &self.clear_pipeline,
                    decorated: &self.decorated_pipeline,
                    texture: &self.texture_pipeline,
                    vector: &self.vector_pipeline,
                },
                &self.viewport_bind_group,
                &self.quad_buffer,
                &mut self.text_pipeline,
                full_redraw,
                scissor,
                &DrawPrimitives {
                    instances,
                    texts,
                    decorated,
                    textures,
                    texture_cache: &self.texture_cache,
                    vectors,
                    vector_atlas: &self.vector_atlas,
                    solid_depths: depths.solid,
                    decorated_depths: depths.decorated,
                    texture_depths: depths.texture,
                },
                self.gpu_timer.as_ref(),
            )?;
            // Only when a pass actually ran this frame — resolving an
            // untouched query set would read stale or never-written slots.
            // The matching `request_map` happens in `submit`, once this
            // encoder's command buffer has actually reached the queue.
            if let Some(timer) = &mut self.gpu_timer {
                timer.resolve_and_copy(&mut encoder);
                self.gpu_timing_pending = true;
            }
        }

        // ── Composite onto the swapchain image ────────────────────────────────
        //
        // Always a full, unscissored copy, every frame, regardless of
        // `should_draw` — see `persistent_color`'s doc comment for why the
        // swapchain's own previous contents can never be assumed valid.
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.persistent_color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.phys_w,
                height: self.phys_h,
                depth_or_array_layers: 1,
            },
        );

        update_frame_bookkeeping(self, instances, texts, decorated, textures);

        Ok(encoder.finish())
    }

    /// Encodes a frame from a [`RenderFrame`] published by the Relay, with
    /// version-based forced-redraw semantics (RFC-0001 §5.2).
    ///
    /// When `frame.version()` exceeds
    /// [`last_relay_version`](Self::last_relay_version), the render thread
    /// skipped at least one frame that had `TextLine::dirty = true`. To
    /// prevent silently displaying stale glyphs this forces a full redraw and
    /// marks all texts dirty before passing them to
    /// [`encode_frame`](Self::encode_frame). Setting `dirty = true` on every
    /// line guarantees glyph reshape for all lines and also satisfies the
    /// debug-mode assertion that content-changed lines must carry a set dirty
    /// bit — the assertion's invariant holds in the normal evaluator pipeline
    /// but is intentionally bypassed here because the render thread may have
    /// missed the original dirty=true frame.
    ///
    /// # Errors
    ///
    /// Same error variants as [`encode_frame`](Self::encode_frame).
    pub fn encode_frame_from_relay(
        &mut self,
        target: &wgpu::Texture,
        frame: &RenderFrame,
    ) -> Result<wgpu::CommandBuffer, ByardError> {
        if frame.version() > self.last_relay_version {
            self.needs_full_redraw = true;
            // Rebuild the text list with every dirty bit set. This ensures
            // (a) every glyph buffer is reshaped (its cached copy may reflect a
            // stale version) and (b) the debug-only hash-change assertion in
            // TextGlyphPipeline::prepare cannot fire on "hash changed but
            // dirty=false" — a state that is legitimate here but would indicate
            // a transpiler bug in the normal evaluator path.
            let texts_dirty: Vec<TextLine> = frame
                .texts()
                .iter()
                .map(|t| TextLine {
                    dirty: true,
                    ..t.clone()
                })
                .collect();
            let cmd = self.encode_frame_with_decorations(
                target,
                frame.instances(),
                &texts_dirty,
                frame.decorated(),
                frame.textures(),
                frame.vector_instances(),
                frame.atlas_uploads(),
                frame_draw_depths(frame),
            )?;
            self.last_relay_version = frame.version();
            return Ok(cmd);
        }
        let cmd = self.encode_frame_with_decorations(
            target,
            frame.instances(),
            frame.texts(),
            frame.decorated(),
            frame.textures(),
            frame.vector_instances(),
            frame.atlas_uploads(),
            frame_draw_depths(frame),
        )?;
        self.last_relay_version = frame.version();
        Ok(cmd)
    }

    /// Requests decode of every texture source in `textures` before the render
    /// pass (M29). With an I/O context the decode runs on the relay's pool and
    /// the upload happens later via [`apply_decoded`](Self::apply_decoded), so
    /// the render thread never blocks here (INV-12). A bare encoder with no
    /// relay falls back to a synchronous decode+upload — used only by
    /// GPU-readback tests, which never carry images, so this branch is a safety
    /// net rather than a hot path.
    fn request_textures(&mut self, textures: &[crate::frame::TextureSampler]) {
        if let Some(io) = &self.io {
            let handle = io.handle.clone();
            let tx = io.tx.clone();
            for t in textures {
                self.texture_cache.ensure(&handle, &tx, &t.src);
            }
        } else {
            for t in textures {
                let decoded = texture_sampler::DecodedImage {
                    src: t.src.clone(),
                    result: texture_sampler::decode_rgba(&t.src),
                };
                self.texture_cache.apply_decoded(
                    &self.device,
                    &self.queue,
                    &self.texture_bind_group_layout,
                    &self.image_sampler,
                    decoded,
                );
            }
        }
    }

    /// Installs the async-decode plumbing (M29): the relay's I/O runtime handle
    /// (decode tasks are spawned here) and the type-erased result sender (they
    /// report decoded pixels back through it). Called once by the engine after
    /// it has both a `Relay` and this encoder. Until set, image decode falls
    /// back to a synchronous path (see [`encode_frame_with_decorations`]).
    pub fn set_io_context(
        &mut self,
        handle: tokio::runtime::Handle,
        tx: texture_sampler::IoResultSender,
    ) {
        self.io = Some(IoContext { handle, tx });
    }

    /// Uploads one async decode result on the render thread (M29). Called by
    /// the engine for each [`DecodedImage`] drained from the relay's I/O
    /// channel, before encoding the next frame. The GPU upload is fast; the
    /// expensive decode already happened off-thread. Because every primitive is
    /// re-emitted dirty each tick, the newly-`Ready` texture repaints
    /// on the next frame without any extra dirty signal.
    pub fn apply_decoded(&mut self, decoded: DecodedImage) {
        self.texture_cache.apply_decoded(
            &self.device,
            &self.queue,
            &self.texture_bind_group_layout,
            &self.image_sampler,
            decoded,
        );
    }
}

/// Updates the structural-change bookkeeping consulted by
/// [`needs_full_redraw_this_frame`] and [`compute_scissor`] on the *next*
/// `encode_frame` call.
///
/// Extracted out of `encode_frame` purely to keep that function under
/// clippy's line-count threshold.
fn update_frame_bookkeeping(
    state: &mut EncoderSubsystem,
    instances: &[BoxInstance],
    texts: &[TextLine],
    decorated: &[crate::frame::DecoratedBox],
    textures: &[crate::frame::TextureSampler],
) {
    state.needs_full_redraw = false;
    state.last_instance_count = instances.len();
    state.last_text_count = texts.len();
    // Recomputed for every primitive (not just dirty ones) — a clean
    // primitive's bounds are unchanged from last frame anyway, so this is a
    // no-op for it, and it keeps each `last_*_bounds` positionally aligned with
    // its slice without needing a separate "did this move" check.
    state.last_text_bounds = texts.iter().map(text_line_bounds).collect();
    state.last_box_bounds = instances.iter().map(|b| rect_of(b.rect)).collect();
    state.last_decorated_bounds = decorated.iter().map(|d| rect_of(d.base.rect)).collect();
    state.last_texture_bounds = textures.iter().map(|t| rect_of(t.rect)).collect();
}

/// Draws the UI render pass: a scissored clear quad (incremental frames
/// only), every `SolidBox` instance, then every `TextLine` — see
/// `EncoderSubsystem::encode_frame`'s doc comment for the full three-case
/// behaviour this implements.
///
/// Extracted out of `encode_frame` purely to keep that function under
/// clippy's line-count threshold. Every parameter here is a field
/// `encode_frame` already owns or borrows — the long parameter list is
/// mechanical (one argument per resource the pass needs), not a sign of
/// fresh coupling between subsystems, so `too_many_arguments` is allowed
/// rather than worked around with an ad-hoc bundling struct.
/// The four UI pipelines `draw_ui_pass` needs, bundled to keep its argument
/// count within the lint threshold (mechanical grouping, not fresh coupling).
#[derive(Clone, Copy)]
struct DrawPipelines<'a> {
    solid: &'a wgpu::RenderPipeline,
    clear: &'a wgpu::RenderPipeline,
    decorated: &'a wgpu::RenderPipeline,
    texture: &'a wgpu::RenderPipeline,
    vector: &'a wgpu::RenderPipeline,
}

/// Draw-order depth slices for one frame (RFC-0011 cross-pass paint order),
/// parallel to the four primitive pools. Bundled so the encode entry points
/// stay within the argument-count lint. An empty slice means "no depth info"
/// and falls back to the far plane, reproducing the pre-depth pass order.
#[derive(Clone, Copy, Default)]
pub struct DrawDepths<'a> {
    /// Parallel to solid `BoxInstance`s.
    pub solid: &'a [f32],
    /// Parallel to `DecoratedBox`es.
    pub decorated: &'a [f32],
    /// Parallel to `TextureSampler`s.
    pub texture: &'a [f32],
    /// Parallel to `TextLine`s (applied via glyphon per-glyph metadata).
    pub text: &'a [f32],
}

/// Bundles a frame's parallel draw-order depth slices into a [`DrawDepths`].
fn frame_draw_depths(frame: &RenderFrame) -> DrawDepths<'_> {
    DrawDepths {
        solid: frame.solid_depths(),
        decorated: frame.decorated_depths(),
        texture: frame.texture_depths(),
        text: frame.text_depths(),
    }
}

/// The per-frame primitive lists `draw_ui_pass` draws, similarly bundled.
#[derive(Clone, Copy)]
struct DrawPrimitives<'a> {
    instances: &'a [BoxInstance],
    texts: &'a [TextLine],
    decorated: &'a [crate::frame::DecoratedBox],
    textures: &'a [crate::frame::TextureSampler],
    texture_cache: &'a texture_sampler::TextureCache,
    /// MSDF vector-glyph instances (RFC-0009 §1).
    vectors: &'a [VectorInstance],
    /// The MSDF atlas these `vectors` sample; not drawn without one.
    vector_atlas: &'a VectorAtlas,
    /// Draw-order depths, parallel to `instances`/`decorated`/`textures`
    /// respectively (RFC-0011 cross-pass paint order). `texts` depth is applied
    /// inside `TextGlyphPipeline::prepare` via glyphon's per-glyph metadata;
    /// `vectors`' depth is a field on `VectorInstance` itself (stamped by
    /// `RenderFrame::push_vector`), not a parallel slice like these three.
    solid_depths: &'a [f32],
    decorated_depths: &'a [f32],
    texture_depths: &'a [f32],
}

#[allow(clippy::too_many_arguments)]
fn draw_ui_pass(
    encoder: &mut wgpu::CommandEncoder,
    persistent_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    device: &wgpu::Device,
    pipelines: &DrawPipelines<'_>,
    viewport_bind_group: &wgpu::BindGroup,
    quad_buffer: &wgpu::Buffer,
    text_pipeline: &mut TextGlyphPipeline,
    full_redraw: bool,
    scissor: Option<(Rect, u32, u32, u32, u32)>,
    primitives: &DrawPrimitives<'_>,
    gpu_timer: Option<&GpuTimer>,
) -> Result<(), ByardError> {
    let DrawPipelines {
        solid: render_pipeline,
        clear: clear_pipeline,
        decorated: decorated_pipeline,
        texture: texture_pipeline,
        vector: vector_pipeline,
    } = *pipelines;
    let DrawPrimitives {
        instances,
        texts,
        decorated,
        textures,
        texture_cache,
        vectors,
        vector_atlas,
        solid_depths,
        decorated_depths,
        texture_depths,
    } = *primitives;
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("ByardCore - UI Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: persistent_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: if full_redraw {
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            },
            // wgpu 29: new field; None = standard 2-D rendering (no depth slice).
            depth_slice: None,
        })],
        // The draw-order depth buffer is frame-local scratch: every primitive is
        // re-emitted (and re-depth-stamped) each frame, so it's cleared to the
        // far plane every pass, even incremental ones. That keeps depth ordering
        // correct within the frame without any cross-frame depth bookkeeping,
        // and (colour is `Load`ed on incremental frames) doesn't disturb the
        // preserved pixels outside the scissor.
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(crate::frame::DRAW_DEPTH_CLEAR),
                store: wgpu::StoreOp::Discard,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: gpu_timer.and_then(|t| t.timestamp_writes(GPU_UI_PASS_SCOPE)),
        occlusion_query_set: None,
        // wgpu 28: new required field; None disables multiview rendering.
        multiview_mask: None,
    });

    // Restrict fragment writes to the dirty region on an incremental frame,
    // and wipe exactly that region first — see
    // `EncoderSubsystem::clear_pipeline`'s doc comment for why a clear pass
    // is required before standard alpha-blended redraw can erase stale
    // content. A full redraw draws unscissored and already cleared the
    // whole target via `LoadOp::Clear` above, so no clear quad is needed in
    // that case.
    if let Some((bounds, x, y, w, h)) = scissor {
        render_pass.set_scissor_rect(x, y, w, h);
        draw_clear_quad(
            &mut render_pass,
            device,
            clear_pipeline,
            viewport_bind_group,
            quad_buffer,
            bounds,
        );
    }

    // Drawn on every call to this function, not just a full redraw — the
    // clear quad above can wipe a box's area on an incremental frame, so
    // boxes must be repainted afterwards or they would stay erased. The
    // active GPU scissor rect (set above, incremental frames only) bounds
    // which pixels this actually touches, so the cost is still proportional
    // to the dirty region, not the full instance list.
    if !instances.is_empty() {
        draw_solid_box_instances(
            &mut render_pass,
            device,
            render_pipeline,
            viewport_bind_group,
            quad_buffer,
            instances,
            solid_depths,
        );
    }

    // M21: decorated boxes (border/shadow/opacity), then textured images. The
    // pass order is unchanged, but the shared depth buffer (each primitive
    // carrying its emission-order z) is now what resolves visibility — so a
    // container's border no longer paints over a child that was emitted after
    // it, and text (below) no longer sits unconditionally on top.
    decorated_box::draw(
        &mut render_pass,
        device,
        decorated_pipeline,
        viewport_bind_group,
        quad_buffer,
        decorated,
        decorated_depths,
    );
    texture_sampler::draw(
        &mut render_pass,
        device,
        texture_pipeline,
        viewport_bind_group,
        quad_buffer,
        texture_cache,
        textures,
        texture_depths,
    );
    // RFC-0009 §1: crisp monochrome icons, sampled from the same MSDF atlas
    // the JIT/AOT paths upload to. Each instance carries its own draw-order
    // depth (RFC-0011), so paint order across pipelines is honoured here too.
    vector_msdf::draw(
        &mut render_pass,
        device,
        vector_pipeline,
        viewport_bind_group,
        quad_buffer,
        vector_atlas,
        vectors,
    );

    // Always the full, unfiltered `texts` slice (the `prepare` call before
    // this pass began did too) — `TextGlyphPipeline`'s internal cache is
    // positionally index-aligned with whatever slice it last saw, so
    // slicing down to only the dirty lines here would silently associate a
    // non-dirty line's cached glyph buffer with the wrong line. The scissor
    // rect set above (on incremental frames) is what actually limits which
    // pixels this call may write — not the slice contents.
    if !texts.is_empty() {
        text_pipeline.render(&mut render_pass)?;
    }

    Ok(())
}

/// Creates the persistent off-screen colour target and its view.
///
/// Shared by [`EncoderSubsystem::init`] and
/// [`EncoderSubsystem::update_viewport`] so both call sites build the
/// texture identically. `RENDER_ATTACHMENT` lets the UI render pass draw
/// into it; `COPY_SRC` lets [`EncoderSubsystem::encode_frame`] copy its
/// contents onto the swapchain image every frame.
fn create_persistent_target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    // wgpu textures must be at least 1×1; a window can in principle report
    // a momentary zero-sized client area mid-resize.
    let width = width.max(1);
    let height = height.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ByardCore - Persistent UI Colour Target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Creates the frame-local draw-order depth target's view (RFC-0011). Sized to
/// match [`create_persistent_target`]; `RENDER_ATTACHMENT` only (never copied to
/// the swapchain — depth is scratch, discarded at the end of each pass).
fn create_depth_target(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let width = width.max(1);
    let height = height.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ByardCore - Draw-Order Depth Target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Deliberately generous (over-estimating) bounding box for a [`TextLine`],
/// in logical pixels.
///
/// `TextLine` exposes no measured glyph extents — the shaped
/// `glyphon::Buffer` is private to `TextGlyphPipeline` — so this estimates
/// width from `font_size` and character count using a generous per-character
/// advance and line-height multiplier. Over-estimating only wastes a little
/// scissor-rect area (a little extra fragment-write bandwidth);
/// under-estimating would visibly clip glyphs, which is a real correctness
/// bug. A measured-extent version is a natural follow-up once `TextLine` (or
/// the Atlas) carries real shaped-glyph bounds.
// A `TextLine` will never hold remotely enough characters (2^24 = 16M+) to
// make this cast lossy in practice; the line's logical-pixel width would
// exceed any real display by many orders of magnitude well before that.
#[allow(clippy::cast_precision_loss)]
fn text_line_bounds(line: &TextLine) -> Rect {
    // Generous enough to cover the widest glyphs in typical Latin text.
    const CHAR_WIDTH_FACTOR: f32 = 0.75;
    // Generous enough to cover ascender + descender.
    const LINE_HEIGHT_FACTOR: f32 = 1.5;

    let char_count = line.text.chars().count() as f32;
    let width = line.font_size * CHAR_WIDTH_FACTOR * char_count;
    let height = line.font_size * LINE_HEIGHT_FACTOR;
    Rect::new(line.x, line.y, width, height)
}

/// Computes the merged bounding box — RFC §3.3's "bounding box of the
/// affected region" — over every **dirty** entry in `texts`, unioned with
/// that same line's bounds from the *previous* frame (`previous`,
/// positionally aligned with `texts`).
///
/// The previous-frame union matters because a line's bounds can shrink or
/// move between frames (e.g. a reactive label whose text gets shorter):
/// without it, the new (smaller) bounds would leave the line's old
/// footprint entirely outside the computed scissor rect, and that old
/// content would never be cleared.
///
/// Returns `None` when no entry is dirty, the caller's signal to skip the
/// incremental render pass entirely for this frame. Multiple simultaneously
/// dirty lines are merged into a single bounding box via repeated
/// [`Rect::union`] rather than issued as separate scissored sub-passes —
/// one scissor + draw call instead of N, at the cost of a marginally larger
/// over-draw region when the dirty lines are far apart on screen.
fn dirty_text_bounds(texts: &[TextLine], previous: &[Rect]) -> Option<Rect> {
    texts
        .iter()
        .enumerate()
        .filter(|(_, line)| line.dirty)
        .map(|(i, line)| {
            let current = text_line_bounds(line);
            match previous.get(i) {
                Some(prev) => current.union(prev),
                None => current,
            }
        })
        .reduce(|acc, r| acc.union(&r))
}

/// Builds a [`Rect`] from a primitive's `[x, y, width, height]` paint rect.
const fn rect_of(rect: [f32; 4]) -> Rect {
    Rect::new(rect[0], rect[1], rect[2], rect[3])
}

/// The generalisation of [`dirty_text_bounds`] to any primitive type
/// (RFC-0001 §3.3): unions the current bounds of every **dirty** item with
/// that item's bounds from the previous frame (`previous`, positionally
/// aligned), so a shrunk or moved primitive still clears its old footprint.
///
/// `items` yields `(current_bounds, is_dirty)` in slice order. Returns `None`
/// when nothing is dirty.
fn union_dirty_rects(items: impl Iterator<Item = (Rect, bool)>, previous: &[Rect]) -> Option<Rect> {
    items
        .enumerate()
        .filter(|(_, (_, dirty))| *dirty)
        .map(|(i, (current, _))| match previous.get(i) {
            Some(prev) => current.union(prev),
            None => current,
        })
        .reduce(|acc, r| acc.union(&r))
}

/// `dirty_text_bounds` for the `SolidBox` pipeline (M26).
///
/// `dirty` is positionally aligned with `instances`. Each box's bounds are its
/// paint rect directly (a solid box's geometry *is* its bounds, unlike text
/// whose extents are heuristically estimated). See [`union_dirty_rects`] for
/// the previous-frame-union contract this shares with text.
fn dirty_box_bounds(instances: &[BoxInstance], dirty: &[bool], previous: &[Rect]) -> Option<Rect> {
    union_dirty_rects(
        instances
            .iter()
            .zip(dirty.iter())
            .map(|(b, d)| (rect_of(b.rect), *d)),
        previous,
    )
}

/// `dirty_text_bounds` for the `DecoratedBox` pipeline (M27); dirtiness comes
/// from each decoration's own [`DecoratedBox::dirty`](crate::frame::DecoratedBox::dirty)
/// bit and its bounds from its `base` rect.
fn dirty_decorated_bounds(
    decorated: &[crate::frame::DecoratedBox],
    previous: &[Rect],
) -> Option<Rect> {
    union_dirty_rects(
        decorated.iter().map(|d| (rect_of(d.base.rect), d.dirty)),
        previous,
    )
}

/// `dirty_text_bounds` for the `TextureSampler` pipeline (M27); dirtiness comes
/// from each sampler's own [`TextureSampler::dirty`](crate::frame::TextureSampler::dirty)
/// bit.
fn dirty_texture_bounds(
    textures: &[crate::frame::TextureSampler],
    previous: &[Rect],
) -> Option<Rect> {
    union_dirty_rects(
        textures.iter().map(|t| (rect_of(t.rect), t.dirty)),
        previous,
    )
}

/// Unions two optional rects: `Some ∪ Some` merges, `Some ∪ None` (either
/// order) passes the `Some` through, `None ∪ None` is `None`.
fn union_opt(a: Option<Rect>, b: Option<Rect>) -> Option<Rect> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.union(&y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

/// Every per-primitive dirty input [`compute_scissor`] unions into one
/// scissor rect, bundled so the call site stays under the argument-count lint
/// and reads as one coherent "what changed this frame" snapshot. Each
/// `prev_*` slice is positionally aligned with its primitive slice (the
/// previous frame's bounds) for the shrink/move-safety contract in
/// [`union_dirty_rects`].
struct ScissorInputs<'a> {
    texts: &'a [TextLine],
    prev_texts: &'a [Rect],
    instances: &'a [BoxInstance],
    instances_dirty: &'a [bool],
    prev_boxes: &'a [Rect],
    decorated: &'a [crate::frame::DecoratedBox],
    prev_decorated: &'a [Rect],
    textures: &'a [crate::frame::TextureSampler],
    prev_textures: &'a [Rect],
}

#[cfg(test)]
impl<'a> ScissorInputs<'a> {
    /// Builds inputs that carry only text (no boxes/decorations/textures) —
    /// the original text-only `compute_scissor` shape, kept so the text-path
    /// unit tests read unchanged.
    fn text_only(texts: &'a [TextLine], prev_texts: &'a [Rect]) -> Self {
        Self {
            texts,
            prev_texts,
            instances: &[],
            instances_dirty: &[],
            prev_boxes: &[],
            decorated: &[],
            prev_decorated: &[],
            textures: &[],
            prev_textures: &[],
        }
    }
}

/// Unions every dirty primitive's bounds (text + solid boxes + decorations +
/// textures, RFC-0001 §3.3) and converts the result into the single
/// [`EncoderSubsystem::encode_frame`] needs to scissor an incremental frame:
/// the logical bounds (needed to size the clear quad) alongside the physical
/// `(x, y, width, height)` tuple `wgpu::RenderPass::set_scissor_rect` expects.
///
/// Pure and unit-testable independent of any `wgpu` state, following the
/// project's established pattern of extracting CPU-mirror decision logic
/// into free functions (see `text_glyph::needs_reshape`). Returns `None`
/// both when nothing is dirty and when the dirty bounds degenerate to a
/// zero-size physical rect (wgpu rejects a zero-size scissor rect).
fn compute_scissor(
    inputs: &ScissorInputs<'_>,
    scale: f32,
    max_w: u32,
    max_h: u32,
) -> Option<(Rect, u32, u32, u32, u32)> {
    let bounds = union_opt(
        union_opt(
            dirty_text_bounds(inputs.texts, inputs.prev_texts),
            dirty_box_bounds(inputs.instances, inputs.instances_dirty, inputs.prev_boxes),
        ),
        union_opt(
            dirty_decorated_bounds(inputs.decorated, inputs.prev_decorated),
            dirty_texture_bounds(inputs.textures, inputs.prev_textures),
        ),
    )?;
    let (x, y, w, h) = logical_rect_to_physical_scissor(bounds, scale, max_w, max_h);
    if w > 0 && h > 0 {
        Some((bounds, x, y, w, h))
    } else {
        None
    }
}

/// Converts a logical-pixel `Rect` into the `(x, y, width, height)` tuple
/// expected by `wgpu::RenderPass::set_scissor_rect`, in physical pixels,
/// clamped to `[0, max_w] × [0, max_h]`.
///
/// wgpu validates that a scissor rect lies entirely within the render
/// target's bounds — a rect computed from logical coordinates can overshoot
/// the physical target by a few pixels from rounding (`x * scale` truncation
/// at the high end), so clamping here is required, not defensive cruft.
// `max_w_f`/`max_h_f` are intentionally parallel names for parallel
// quantities (the f32 form of `max_w`/`max_h`, used only for the `.min`
// clamp below) — not a real ambiguity risk. The u32 → f32 cast is lossless
// in practice: a physical surface dimension exceeding 2^24px (16M+) does
// not exist on any real display.
#[allow(clippy::similar_names, clippy::cast_precision_loss)]
fn logical_rect_to_physical_scissor(
    rect: Rect,
    scale: f32,
    max_w: u32,
    max_h: u32,
) -> (u32, u32, u32, u32) {
    let x0 = (rect.x * scale).floor().max(0.0);
    let y0 = (rect.y * scale).floor().max(0.0);
    let x1 = ((rect.x + rect.width) * scale).ceil().max(x0);
    let y1 = ((rect.y + rect.height) * scale).ceil().max(y0);

    let max_w_f = max_w as f32;
    let max_h_f = max_h as f32;

    let x0 = x0.min(max_w_f);
    let y0 = y0.min(max_h_f);
    let x1 = x1.min(max_w_f);
    let y1 = y1.min(max_h_f);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    (x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)
}

/// Decides whether [`EncoderSubsystem::encode_frame`] must perform a full
/// redraw this frame.
///
/// Pure and unit-testable independent of any `wgpu` state, following the
/// project's established pattern of extracting CPU-mirror decision logic
/// into free functions (see `text_glyph::needs_reshape`).
///
/// A full redraw is forced by `sticky` (set on construction and after a
/// resize — see [`EncoderSubsystem::needs_full_redraw`]) OR by a structural
/// change in the instance/text counts since the previous frame, since
/// neither `BoxInstance` nor `TextLine` carries an "added this frame" bit.
fn needs_full_redraw_this_frame(
    sticky: bool,
    prev_instance_count: usize,
    instance_count: usize,
    prev_text_count: usize,
    text_count: usize,
) -> bool {
    sticky || prev_instance_count != instance_count || prev_text_count != text_count
}

/// Compiles the WGSL shader and assembles a `SolidBox`-shaped render
/// pipeline, parameterised by `blend` so the same shader and vertex layout
/// can back both [`EncoderSubsystem::render_pipeline`] (alpha-blended) and
/// [`EncoderSubsystem::clear_pipeline`] (`blend: None` — unconditional
/// replace).
///
/// Separated from [`EncoderSubsystem::init`] to keep that function under the
/// 100-line lint threshold.
///
/// Per RFC §8, the full creation sequence — `create_pipeline_layout`,
/// `create_shader_module`, and `create_render_pipeline` — is wrapped inside a
/// single `push_error_scope` / `pop_error_scope` pair so that any GPU-side
/// validation failure is captured and returned as
/// [`ByardError::PipelineCompilation`].
/// Builds the M21 `DecoratedBox` and `TextureSampler` pipelines plus the texture
/// bind-group layout and shared sampler. Extracted from
/// [`EncoderSubsystem::init`] to keep that function under the line-count lint.
async fn build_m21_pipelines(
    device: &wgpu::Device,
    viewport_layout: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> Result<
    (
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::BindGroupLayout,
        wgpu::Sampler,
    ),
    ByardError,
> {
    let quad = || wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x2,
        }],
    };

    let decorated_pipeline =
        decorated_box::build_pipeline(device, viewport_layout, quad(), surface_format).await?;

    let texture_bind_group_layout = texture_sampler::bind_group_layout(device);
    let image_sampler = texture_sampler::sampler(device);
    let texture_pipeline = texture_sampler::build_pipeline(
        device,
        viewport_layout,
        &texture_bind_group_layout,
        quad(),
        surface_format,
    )
    .await?;

    Ok((
        decorated_pipeline,
        texture_pipeline,
        texture_bind_group_layout,
        image_sampler,
    ))
}

/// Depth-buffer format used to resolve draw order across the four UI pipelines.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Depth-stencil state for the *drawing* pipelines (solid/decorated/texture/
/// vector/text): write each primitive's draw-order z and keep the nearest —
/// i.e. the later-emitted — fragment (`LessEqual`, since later = smaller z,
/// buffer cleared to the far plane). `pub(crate)` so `vector_msdf` (a sibling
/// submodule) shares it instead of duplicating the state.
pub(crate) fn draw_depth_stencil() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Depth-stencil state for the clear pipeline: never tests or writes depth
/// (`Always` + no write), so wiping colour in the incremental scissor region
/// leaves the draw-order depth buffer (already cleared this frame) untouched.
fn clear_depth_stencil() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(false),
        depth_compare: Some(wgpu::CompareFunction::Always),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

async fn build_solid_box_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    quad_layout: wgpu::VertexBufferLayout<'static>,
    surface_format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    depth_stencil: wgpu::DepthStencilState,
    debug_name: &str,
) -> Result<wgpu::RenderPipeline, ByardError> {
    // --- GPU VALIDATION ERROR SCOPE (RFC §8) ---
    // Covers create_pipeline_layout + create_shader_module + create_render_pipeline,
    // the three operations listed in RFC §8 as requiring capture.
    // wgpu 28+: push_error_scope returns an owned scope handle; pop is on the handle.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ByardCore - SolidBox Pipeline Layout"),
        // wgpu 29: bind_group_layouts is now &[Option<&BindGroupLayout>].
        bind_group_layouts: &[Some(bind_group_layout)],
        // wgpu 28: push_constant_ranges removed; replaced by immediate_size: u32.
        immediate_size: 0,
    });

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ByardCore - SolidBox WGSL Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "solid_box.wgsl"
        ))),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(debug_name),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: Some("vs_main"),
            buffers: &[quad_layout, BoxInstance::layout(), solid_depth_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend,
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
            pipeline: debug_name.to_string(),
            reason: error.to_string(),
        });
    }

    Ok(pipeline)
}

/// Draws a single fully transparent quad covering `bounds` (logical pixels)
/// using `pipeline`'s no-blend state, so the fragment shader's output
/// unconditionally **replaces** the destination instead of blending with
/// it — see [`EncoderSubsystem::clear_pipeline`]'s doc comment for why this
/// is required before an incremental redraw can erase stale content.
///
/// Must be called while `render_pass`'s active scissor rect already
/// restricts writes to (at most) `bounds` — otherwise this would wipe
/// unrelated content outside the dirty region.
fn draw_clear_quad(
    render_pass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    quad_buffer: &wgpu::Buffer,
    bounds: Rect,
) {
    let clear_instance = BoxInstance {
        rect: [bounds.x, bounds.y, bounds.width, bounds.height],
        color: [0.0, 0.0, 0.0, 0.0],
        radii: [0.0; 4],
        transform: Transform::IDENTITY,
    };
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ByardCore - Clear Quad Instance Buffer"),
        contents: bytemuck::bytes_of(&clear_instance),
        usage: wgpu::BufferUsages::VERTEX,
    });
    // The clear pipeline shares `solid_box.wgsl`, which now reads a depth at
    // location 9, so the clear draw must still supply the buffer. The value is
    // irrelevant: the clear pipeline runs with depth-write disabled and an
    // `Always` compare (see `build_solid_box_pipeline`), so it never touches the
    // depth buffer — it only wipes colour in the scissor region.
    let depth_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ByardCore - Clear Quad Depth Buffer"),
        contents: bytemuck::bytes_of(&crate::frame::DRAW_DEPTH_CLEAR),
        usage: wgpu::BufferUsages::VERTEX,
    });

    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
    render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
    render_pass.set_vertex_buffer(2, depth_buffer.slice(..));
    render_pass.draw(0..4, 0..1);
}

/// Draws every `BoxInstance` in `instances` using `pipeline`'s alpha-blended
/// state.
///
/// Extracted from [`EncoderSubsystem::encode_frame`] to keep that function
/// under the 100-line lint threshold. On an incremental frame, the caller's
/// active GPU scissor rect (not this function) is what actually bounds the
/// pixels touched here, so calling this unconditionally on every
/// `should_draw` frame is still proportional to the dirty region's
/// bandwidth, not the full instance list's.
fn draw_solid_box_instances(
    render_pass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    quad_buffer: &wgpu::Buffer,
    instances: &[BoxInstance],
    depths: &[f32],
) {
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ByardCore - SolidBox Instance Buffer"),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    });
    // Draw-order depths, parallel to `instances`, fed as a second per-instance
    // vertex buffer (shader location 9) — keeps `BoxInstance`'s Pod layout
    // untouched. Padded to the instance count with the far plane so a
    // length mismatch can never index out of range on the GPU.
    let mut depth_data = depths.to_vec();
    depth_data.resize(instances.len(), crate::frame::DRAW_DEPTH_CLEAR);
    let depth_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ByardCore - SolidBox Depth Buffer"),
        contents: bytemuck::cast_slice(&depth_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
    render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
    render_pass.set_vertex_buffer(2, depth_buffer.slice(..));
    // Safety: no UI frame will ever hold 2^32 instances. The cast is
    // bounded by system memory (each BoxInstance is 80 bytes, so 2^32
    // would require 320 GiB of RAM before reaching this code).
    #[allow(clippy::cast_possible_truncation)]
    render_pass.draw(0..4, 0..instances.len() as u32);
}

/// Vertex buffer layout for the `SolidBox` pipeline's parallel draw-order depth
/// buffer (a lone `f32` per instance at shader location 9).
fn solid_depth_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![9 => Float32];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: ATTRS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── INV-8: paint-time transforms never trigger a relayout ─────────────────

    #[test]
    fn encoder_module_never_calls_layout_atlas_compute() {
        // RFC-0011 (INV-8): a paint-time `Transform` must never cause a Taffy
        // relayout. Structurally enforced by module boundaries (`encoder`
        // never imports `crate::atlas`) — this test scans the encoder's own
        // sources for the literal call so a future edit can't reintroduce it
        // without at least this test noticing.
        //
        // Built at runtime (not a literal in this file) so this very
        // assertion doesn't trip on itself via `include_str!`. The file list
        // is every `.rs` file in this directory (`ls src/encoder/*.rs`) —
        // keep it in sync when adding a new one, since `include_str!` can't
        // glob a directory.
        let forbidden_call = ["LayoutAtlas", "::", "compute"].concat();
        for (name, src) in [
            ("mod.rs", include_str!("mod.rs")),
            ("decorated_box.rs", include_str!("decorated_box.rs")),
            ("gpu_timer.rs", include_str!("gpu_timer.rs")),
            ("text_glyph.rs", include_str!("text_glyph.rs")),
            ("texture_sampler.rs", include_str!("texture_sampler.rs")),
        ] {
            assert!(
                !src.contains(&forbidden_call),
                "{name} must never call into layout recomputation (INV-8)"
            );
        }
    }

    // ── BoxInstance layout ────────────────────────────────────────────────────
    //
    // The GPU relies on the byte layout of BoxInstance being exactly what
    // `BoxInstance::layout()` declares. Any mismatch silently corrupts every
    // rendered rectangle. These tests catch such regressions at compile time.

    #[test]
    fn box_instance_size_and_alignment() {
        // 3 fields × [f32; 4] × 4 bytes = 48, + Transform's 8 × f32 × 4 bytes
        // = 32, for 80 bytes total (RFC-0011). The GPU stride declaration in
        // `layout()` hardcodes this value.
        assert_eq!(
            std::mem::size_of::<BoxInstance>(),
            80,
            "BoxInstance must be exactly 80 bytes"
        );
        // f32 requires 4-byte alignment; wgpu vertex attributes assume this.
        assert_eq!(std::mem::align_of::<BoxInstance>(), 4);
    }

    #[test]
    fn box_instance_field_offsets_match_shader_locations() {
        // `BoxInstance::layout()` declares offsets 0, 16, 32, 48 for
        // rect/color/radii/transform. If any field is reordered or padded,
        // the shader sees garbage.
        assert_eq!(std::mem::offset_of!(BoxInstance, rect), 0);
        assert_eq!(std::mem::offset_of!(BoxInstance, color), 16);
        assert_eq!(std::mem::offset_of!(BoxInstance, radii), 32);
        assert_eq!(std::mem::offset_of!(BoxInstance, transform), 48);
        assert_eq!(std::mem::offset_of!(Transform, translate), 0);
        assert_eq!(std::mem::offset_of!(Transform, scale), 8);
        assert_eq!(std::mem::offset_of!(Transform, rotate), 16);
        assert_eq!(std::mem::offset_of!(Transform, origin), 20);
        assert_eq!(std::mem::offset_of!(Transform, opacity), 28);
    }

    #[test]
    fn box_instance_layout_stride_step_mode_and_attributes() {
        let layout = BoxInstance::layout();

        assert_eq!(
            layout.array_stride, 80,
            "stride must equal size_of::<BoxInstance>()"
        );
        assert_eq!(
            layout.step_mode,
            wgpu::VertexStepMode::Instance,
            "must advance per instance, not per vertex"
        );

        // Verify each attribute's (shader_location, offset) pair.
        let attrs = layout.attributes;
        assert_eq!(attrs.len(), 8);

        assert_eq!(attrs[0].shader_location, 1); // rect
        assert_eq!(attrs[0].offset, 0);
        assert_eq!(attrs[0].format, wgpu::VertexFormat::Float32x4);

        assert_eq!(attrs[1].shader_location, 2); // color
        assert_eq!(attrs[1].offset, 16);
        assert_eq!(attrs[1].format, wgpu::VertexFormat::Float32x4);

        assert_eq!(attrs[2].shader_location, 3); // radii
        assert_eq!(attrs[2].offset, 32);
        assert_eq!(attrs[2].format, wgpu::VertexFormat::Float32x4);

        assert_eq!(attrs[3].shader_location, 4); // transform.translate
        assert_eq!(attrs[3].offset, 48);
        assert_eq!(attrs[3].format, wgpu::VertexFormat::Float32x2);

        assert_eq!(attrs[4].shader_location, 5); // transform.scale
        assert_eq!(attrs[4].offset, 56);
        assert_eq!(attrs[4].format, wgpu::VertexFormat::Float32x2);

        assert_eq!(attrs[5].shader_location, 6); // transform.rotate
        assert_eq!(attrs[5].offset, 64);
        assert_eq!(attrs[5].format, wgpu::VertexFormat::Float32);

        assert_eq!(attrs[6].shader_location, 7); // transform.origin
        assert_eq!(attrs[6].offset, 68);
        assert_eq!(attrs[6].format, wgpu::VertexFormat::Float32x2);

        assert_eq!(attrs[7].shader_location, 8); // transform.opacity
        assert_eq!(attrs[7].offset, 76);
        assert_eq!(attrs[7].format, wgpu::VertexFormat::Float32);
    }

    #[test]
    fn box_instance_bytemuck_cast_produces_correct_byte_count() {
        // bytemuck::cast_slice is used in encode_frame to upload instances.
        // A wrong Pod impl (e.g. accidental padding) would give the wrong length.
        let instances = [
            BoxInstance {
                rect: [0.0, 0.0, 100.0, 50.0],
                color: [1.0, 0.0, 0.5, 1.0],
                radii: [8.0, 8.0, 8.0, 8.0],
                transform: Transform::IDENTITY,
            },
            BoxInstance {
                rect: [10.0, 20.0, 200.0, 80.0],
                color: [0.0, 1.0, 0.0, 0.8],
                radii: [0.0; 4],
                transform: Transform::IDENTITY,
            },
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        assert_eq!(bytes.len(), 2 * 80, "2 instances × 80 bytes each");
    }

    #[test]
    // Exact bit-level zero is what Zeroable guarantees, so strict equality is
    // correct here: we are not comparing computed floats but literal bit patterns.
    #[allow(clippy::float_cmp)]
    fn box_instance_zeroed_is_valid_and_all_zero() {
        // bytemuck::Zeroable guarantees that all-zero bytes form a valid
        // BoxInstance. Used implicitly when zero-filling instance buffers.
        let z: BoxInstance = bytemuck::Zeroable::zeroed();
        assert_eq!(z.rect, [0.0; 4]);
        assert_eq!(z.color, [0.0; 4]);
        assert_eq!(z.radii, [0.0; 4]);
    }

    // ── QUAD_VERTICES ─────────────────────────────────────────────────────────

    #[test]
    // QUAD_VERTICES is a compile-time constant array of exact integer-valued
    // floats (0.0 and 1.0). Strict equality is intentional: we are verifying
    // that no rounding crept in, not comparing computed results.
    #[allow(clippy::float_cmp)]
    fn quad_vertices_form_unit_square() {
        // 4 vertices × 2 coords (x, y) = 8 floats.
        assert_eq!(QUAD_VERTICES.len(), 8);

        // Every coordinate must be exactly 0.0 or 1.0.
        for &v in QUAD_VERTICES {
            assert!(
                v == 0.0 || v == 1.0,
                "unexpected quad vertex coordinate: {v}"
            );
        }

        // All four corners of the unit square [0,1]² must be present.
        let pairs: Vec<(f32, f32)> = QUAD_VERTICES.chunks(2).map(|c| (c[0], c[1])).collect();

        assert!(pairs.contains(&(0.0, 0.0)), "missing top-left  (0,0)");
        assert!(pairs.contains(&(1.0, 0.0)), "missing top-right  (1,0)");
        assert!(pairs.contains(&(0.0, 1.0)), "missing bottom-left  (0,1)");
        assert!(pairs.contains(&(1.0, 1.0)), "missing bottom-right (1,1)");
    }

    // ── SDF ───────────────────────────────────────────────────────────────────

    /// CPU reimplementation of the `sd_rounded_box` WGSL function.
    ///
    /// Mirrors the shader logic exactly so algebraic properties can be
    /// asserted in unit tests without spinning up a GPU backend.
    fn cpu_sd_rounded_box(p: [f32; 2], b: [f32; 2], r: [f32; 4]) -> f32 {
        // Screen Y increases downward, so top half has p[1] < 0.
        // Default case (p.x <= 0 && p.y <= 0) → top-left radius.
        let mut r_corner = r[0]; // Top-Left
        if p[0] > 0.0 && p[1] < 0.0 {
            r_corner = r[1]; // Top-Right
        }
        if p[0] > 0.0 && p[1] > 0.0 {
            r_corner = r[2]; // Bottom-Right
        }
        if p[0] < 0.0 && p[1] > 0.0 {
            r_corner = r[3]; // Bottom-Left
        }

        let q_x = p[0].abs() - b[0] + r_corner;
        let q_y = p[1].abs() - b[1] + r_corner;

        let length_max_q = (q_x.max(0.0) * q_x.max(0.0) + q_y.max(0.0) * q_y.max(0.0)).sqrt();

        q_x.max(q_y).min(0.0) + length_max_q - r_corner
    }

    #[test]
    fn sdf_zero_radii_degenerates_to_axis_aligned_rect() {
        // With r=[0,0,0,0] the SDF must equal the plain AABB SDF.
        // This is the most common production case (solid rectangle, no rounding).
        let half = [50.0_f32, 50.0];
        let r = [0.0_f32; 4];

        // Centre: strictly inside.
        assert!(cpu_sd_rounded_box([0.0, 0.0], half, r) < 0.0);

        // Right edge midpoint: on the boundary → SDF = 0.
        // q_x = 50 − 50 + 0 = 0, q_y = 0 − 50 + 0 = −50
        // → min(max(0, −50), 0) + length((0, 0)) − 0 = 0
        let d = cpu_sd_rounded_box([50.0, 0.0], half, r);
        assert!(d.abs() < 0.001, "right edge: {d}");

        // Outside right edge (x=55, y=0): SDF = 5.
        // q_x = 55 − 50 = 5, q_y = 0 − 50 = −50
        // → min(max(5, −50), 0) + length((5, 0)) − 0 = 0 + 5 = 5
        let d = cpu_sd_rounded_box([55.0, 0.0], half, r);
        assert!((d - 5.0).abs() < 0.001, "right exterior: {d}");

        // Outside corner (x=55, y=55): SDF = √(5²+5²) ≈ 7.071.
        // q_x = 5, q_y = 5 → 0 + √50 ≈ 7.071
        let d = cpu_sd_rounded_box([55.0, 55.0], half, r);
        assert!((d - (50.0_f32).sqrt()).abs() < 0.001, "sharp corner: {d}");
    }

    #[test]
    fn sdf_all_four_quadrants_select_correct_radius() {
        // Fully asymmetric radii: TL=10, TR=20, BR=30, BL=40.
        // For each quadrant, place a point at a distance that depends only
        // on the corner radius of that quadrant and verify the expected SDF.
        //
        // Strategy: at (±45, ∓45) (all inside the box), the SDF depends on
        // r_corner because q_* includes `+ r_corner`. By varying only the
        // active radius we can isolate each quadrant.
        //
        // Each expected value computed analytically:
        // q = 45 − 50 + r = r − 5  (same for both axes at this symmetric point)
        // When r ≥ 5: both q components ≥ 0 → result = √2·(r−5) − r
        // When r < 5: both q components < 0 → result = (r−5) − r = −5
        fn expected(r: f32) -> f32 {
            let q = 45.0 - 50.0 + r; // = r - 5
            if q <= 0.0 {
                q - r
            } else {
                (2.0_f32).sqrt() * q - r
            }
        }

        let half = [50.0_f32, 50.0];
        let r = [10.0_f32, 20.0, 30.0, 40.0]; // TL, TR, BR, BL

        // Top-Left (p.x < 0, p.y < 0) → r[0] = 10
        let d = cpu_sd_rounded_box([-45.0, -45.0], half, r);
        assert!((d - expected(10.0)).abs() < 0.001, "TL: {d}");

        // Top-Right (p.x > 0, p.y < 0) → r[1] = 20
        let d = cpu_sd_rounded_box([45.0, -45.0], half, r);
        assert!((d - expected(20.0)).abs() < 0.001, "TR: {d}");

        // Bottom-Right (p.x > 0, p.y > 0) → r[2] = 30
        let d = cpu_sd_rounded_box([45.0, 45.0], half, r);
        assert!((d - expected(30.0)).abs() < 0.001, "BR: {d}");

        // Bottom-Left (p.x < 0, p.y > 0) → r[3] = 40
        let d = cpu_sd_rounded_box([-45.0, 45.0], half, r);
        assert!((d - expected(40.0)).abs() < 0.001, "BL: {d}");
    }

    #[test]
    // The recovered values are the same bytes written as the original —
    // no arithmetic involved, so strict bit-equality is the correct assertion.
    #[allow(clippy::float_cmp)]
    fn box_instance_bytemuck_round_trip_preserves_values() {
        // Verifies that casting BoxInstance → &[u8] → BoxInstance returns
        // identical field values. Catches any Pod impl that shuffles bytes.
        let original = BoxInstance {
            rect: [1.0, 2.0, 300.0, 400.0],
            color: [0.25, 0.5, 0.75, 1.0],
            radii: [8.0, 16.0, 24.0, 32.0],
            transform: Transform::IDENTITY,
        };

        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let recovered: &BoxInstance = bytemuck::from_bytes(bytes);

        assert_eq!(recovered.rect, original.rect);
        assert_eq!(recovered.color, original.color);
        assert_eq!(recovered.radii, original.radii);
    }

    #[test]
    fn sdf_mathematical_quadrants_and_boundaries() {
        // 100×100 box centred at origin → half_size = (50, 50).
        let half_size = [50.0_f32, 50.0];
        // Asymmetric radii to verify per-corner selection.
        let radii = [10.0_f32, 15.0, 20.0, 25.0];

        // Centre is deep inside — SDF must be strongly negative.
        let dist_center = cpu_sd_rounded_box([0.0, 0.0], half_size, radii);
        assert!(dist_center < -20.0, "centre: {dist_center}");

        // Top edge midpoint (x=0, y=−50): SDF ≈ 0.
        // q_x = 0 − 50 + 10 = −40, q_y = 50 − 50 + 10 = 10
        // → min(max(−40, 10), 0) + length((0, 10)) − 10 = 0 + 10 − 10 = 0
        let dist_edge = cpu_sd_rounded_box([0.0, -50.0], half_size, radii);
        assert!(dist_edge.abs() < 0.001, "top edge: {dist_edge}");

        // Far outside corner — SDF must be substantially positive.
        let dist_outer = cpu_sd_rounded_box([100.0, 100.0], half_size, radii);
        assert!(dist_outer > 40.0, "outer: {dist_outer}");

        // Asymmetry: TL radius=10 gives a different SDF than BR radius=20
        // at the same distance from their respective corners.
        let dist_tl = cpu_sd_rounded_box([-45.0, -45.0], half_size, radii);
        let dist_br = cpu_sd_rounded_box([45.0, 45.0], half_size, radii);
        assert!(
            (dist_tl - dist_br).abs() > 1.0,
            "TL={dist_tl}, BR={dist_br} should differ"
        );

        // Axis boundary (x=0, y=−45): default case (TL radius=10) must apply.
        // q_x = 0 − 50 + 10 = −40, q_y = 45 − 50 + 10 = 5
        // → min(max(−40, 5), 0) + length((0, 5)) − 10 = 0 + 5 − 10 = −5
        let dist_boundary = cpu_sd_rounded_box([0.0, -45.0], half_size, radii);
        assert!(
            (dist_boundary - (-5.0)).abs() < 0.001,
            "axis boundary: {dist_boundary}"
        );
    }

    // ── SDF: corner arc ───────────────────────────────────────────────────────

    #[test]
    fn sdf_point_exactly_on_corner_arc_has_zero_distance() {
        // For the BR corner with r=20 and half=[50,50]:
        //   arc centre = (50−20, 50−20) = (30, 30)
        //   point at 45° on the arc: p = (30 + 20/√2, 30 + 20/√2) ≈ (44.14, 44.14)
        //
        // Manual calculation (BR quadrant → r_corner=20):
        //   q_x = 44.14 − 50 + 20 = 14.14
        //   q_y = same
        //   result = 0 + √(14.14² + 14.14²) − 20 = √400 − 20 = 0
        let half = [50.0_f32, 50.0];
        let r = [0.0_f32, 0.0, 20.0, 0.0]; // only BR has radius
        let offset = 20.0_f32 / (2.0_f32).sqrt();
        let p = [30.0 + offset, 30.0 + offset];
        let d = cpu_sd_rounded_box(p, half, r);
        assert!(d.abs() < 0.001, "on arc: {d}");
    }

    // ── SDF: degenerate radii ─────────────────────────────────────────────────

    #[test]
    fn sdf_full_radius_equals_half_size_acts_like_circle() {
        // When r == half for all corners the rounded rect degenerates to a circle.
        // For half=[25,25] and r=[25,25,25,25] the boundary lies at distance 25
        // from the origin in every direction.
        //
        // Right midpoint p=(25,0) — falls into the TL default case since p.y==0:
        //   q_x = 25−25+25 = 25, q_y = 0−25+25 = 0
        //   result = 0 + length((25,0)) − 25 = 25 − 25 = 0 (on boundary) ✓
        //
        // TR diagonal p=(25/√2, −25/√2) — TR case (p.x>0, p.y<0):
        //   q_x = q_y = 25/√2 − 25 + 25 = 25/√2 ≈ 17.68
        //   result = 0 + √(17.68²+17.68²) − 25 = 25 − 25 = 0 (on boundary) ✓
        let half = [25.0_f32, 25.0];
        let r = [25.0_f32; 4];

        let d = cpu_sd_rounded_box([25.0, 0.0], half, r);
        assert!(d.abs() < 0.001, "right midpoint: {d}");

        let diag = 25.0_f32 / (2.0_f32).sqrt();
        let d = cpu_sd_rounded_box([diag, -diag], half, r);
        assert!(d.abs() < 0.001, "TR diagonal: {d}");

        let d = cpu_sd_rounded_box([0.0, 0.0], half, r);
        assert!((d - (-25.0)).abs() < 0.001, "centre: {d}");
    }

    #[test]
    fn sdf_radius_exceeding_half_size_is_finite() {
        // The SDF function does not clamp radii to half-size. A radius larger
        // than half-size produces a mathematically valid (though visually odd)
        // value. This test documents that the function is total — it never
        // panics or returns NaN/±inf for any finite inputs.
        let half = [50.0_f32, 50.0];
        let r_big = [60.0_f32; 4];
        let r_huge = [1000.0_f32; 4];

        assert!(cpu_sd_rounded_box([0.0, 0.0], half, r_big).is_finite());
        assert!(cpu_sd_rounded_box([0.0, 0.0], half, r_huge).is_finite());
        assert!(cpu_sd_rounded_box([100.0, 100.0], half, r_big).is_finite());
    }

    // ── bytemuck: empty slice and non-finite values ───────────────────────────

    #[test]
    fn box_instance_cast_slice_empty_gives_zero_bytes() {
        // encode_frame guards with `if !instances.is_empty()` before creating
        // a buffer. Verify that casting an empty slice is safe and produces
        // zero bytes — not UB, not a panic.
        let empty: &[BoxInstance] = &[];
        let bytes: &[u8] = bytemuck::cast_slice(empty);
        assert_eq!(bytes.len(), 0);
    }

    #[test]
    fn box_instance_pod_accepts_non_finite_floats() {
        // bytemuck::Pod requires every bit pattern to be a valid value.
        // NaN and ±inf are valid f32 bit patterns, so Pod must accept them.
        // encode_frame calls bytemuck::cast_slice on instances it receives;
        // if the caller passes NaN coordinates, the cast must not panic.
        let inst = BoxInstance {
            rect: [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0],
            color: [f32::NAN; 4],
            radii: [f32::INFINITY; 4],
            transform: Transform::IDENTITY,
        };
        let bytes = bytemuck::bytes_of(&inst);
        assert_eq!(bytes.len(), 80, "NaN/inf must not change struct size");
    }

    // ── QUAD_VERTICES: TriangleStrip geometry ────────────────────────────────

    #[test]
    fn quad_vertices_triangle_strip_tiles_unit_square_without_gaps() {
        // TriangleStrip with 4 vertices produces exactly 2 triangles:
        //   T1: indices 0,1,2 → (TL, TR, BL)
        //   T2: indices 1,2,3 → (TR, BL, BR)
        //
        // Verify their combined area equals 1.0 (the unit square), which
        // proves they tile the surface without gaps or overlaps.
        fn tri_area(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
            ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)).abs() * 0.5
        }

        let p: Vec<(f32, f32)> = QUAD_VERTICES.chunks(2).map(|c| (c[0], c[1])).collect();

        let a1 = tri_area(p[0].0, p[0].1, p[1].0, p[1].1, p[2].0, p[2].1);
        let a2 = tri_area(p[1].0, p[1].1, p[2].0, p[2].1, p[3].0, p[3].1);

        assert!((a1 - 0.5).abs() < 0.001, "T1 area = {a1} (expected 0.5)");
        assert!((a2 - 0.5).abs() < 0.001, "T2 area = {a2} (expected 0.5)");
        assert!(
            (a1 + a2 - 1.0).abs() < 0.001,
            "total area = {} (expected 1.0)",
            a1 + a2
        );
    }

    // ── #31 scissor clipping: pure decision/heuristic functions ──────────────
    //
    // None of these touch wgpu — they're the CPU-mirror logic that decides
    // *what* `encode_frame` will do, extracted so it's testable without a
    // GPU device (project convention — see `text_glyph::needs_reshape`).

    /// Tolerance-based f32 comparison, mirroring `engine.rs`'s test helper
    /// of the same name — used in place of `assert_eq!` on raw floats to
    /// satisfy `clippy::float_cmp` without losing the precision these
    /// tests actually need (well under one logical pixel).
    #[track_caller]
    fn assert_f32_eq(actual: f32, expected: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.001,
            "expected {expected}, got {actual} (diff = {diff})"
        );
    }

    fn line(x: f32, y: f32, text: &str, font_size: f32, dirty: bool) -> TextLine {
        TextLine {
            x,
            y,
            text: text.to_string(),
            font_size,
            color: [0.0, 0.0, 0.0, 1.0],
            dirty,
        }
    }

    #[test]
    fn text_line_bounds_grows_with_character_count() {
        let short = text_line_bounds(&line(0.0, 0.0, "a", 16.0, false));
        let long = text_line_bounds(&line(0.0, 0.0, "a much longer string", 16.0, false));
        assert!(
            long.width > short.width,
            "more characters must widen the bound"
        );
        // height depends only on font_size, not character count.
        assert_f32_eq(short.height, long.height);
    }

    #[test]
    fn text_line_bounds_grows_with_font_size() {
        let small = text_line_bounds(&line(0.0, 0.0, "label", 12.0, false));
        let large = text_line_bounds(&line(0.0, 0.0, "label", 48.0, false));
        assert!(large.width > small.width);
        assert!(large.height > small.height);
    }

    #[test]
    fn text_line_bounds_is_positioned_at_the_line_origin() {
        let r = text_line_bounds(&line(123.0, 45.0, "x", 16.0, false));
        assert_f32_eq(r.x, 123.0);
        assert_f32_eq(r.y, 45.0);
    }

    #[test]
    fn text_line_bounds_never_yields_negative_dimensions() {
        // Defensive: an empty string is a degenerate but legitimate TextLine.
        let r = text_line_bounds(&line(0.0, 0.0, "", 16.0, false));
        assert!(r.width >= 0.0);
        assert!(r.height >= 0.0);
    }

    #[test]
    fn dirty_text_bounds_is_none_when_nothing_is_dirty() {
        let texts = [
            line(0.0, 0.0, "a", 16.0, false),
            line(50.0, 50.0, "b", 16.0, false),
        ];
        assert!(dirty_text_bounds(&texts, &[]).is_none());
    }

    #[test]
    fn dirty_text_bounds_is_none_for_empty_slice() {
        assert!(dirty_text_bounds(&[], &[]).is_none());
    }

    #[test]
    fn dirty_text_bounds_ignores_non_dirty_lines() {
        let dirty_line = line(10.0, 10.0, "dirty", 16.0, true);
        let clean_line = line(1000.0, 1000.0, "clean", 16.0, false);
        let bounds = dirty_text_bounds(&[dirty_line.clone(), clean_line], &[]).unwrap();
        let expected = text_line_bounds(&dirty_line);
        assert_eq!(bounds, expected, "clean line must not widen the union");
    }

    #[test]
    fn dirty_text_bounds_merges_multiple_dirty_lines() {
        let a = line(0.0, 0.0, "a", 16.0, true);
        let b = line(200.0, 300.0, "b", 16.0, true);
        let merged = dirty_text_bounds(&[a.clone(), b.clone()], &[]).unwrap();
        let expected = text_line_bounds(&a).union(&text_line_bounds(&b));
        assert_eq!(merged, expected);
    }

    #[test]
    fn dirty_text_bounds_unions_with_previous_frame_bounds() {
        // A line that shrinks between frames: its NEW bounds alone would
        // leave the old (wider) footprint outside the scissor rect,
        // exactly the bug behind the shrinking-line visual-verification finding.
        let shrunk = line(0.0, 0.0, "a", 16.0, true);
        let previous_bounds =
            text_line_bounds(&line(0.0, 0.0, "a much longer string", 16.0, false));
        let bounds = dirty_text_bounds(std::slice::from_ref(&shrunk), &[previous_bounds]).unwrap();
        let current = text_line_bounds(&shrunk);
        let expected = current.union(&previous_bounds);
        assert_eq!(
            bounds, expected,
            "must cover both current and previous bounds for a dirty line"
        );
        assert!(
            bounds.width >= previous_bounds.width,
            "must not be narrower than the previous frame's footprint"
        );
    }

    #[test]
    fn dirty_text_bounds_unions_when_line_grows_between_frames() {
        // The inverse of the shrink case above: when the new bounds fully
        // contain the old ones, the union must still equal the new bounds
        // (not be artificially clamped back down to the smaller, previous
        // footprint).
        let grown = line(0.0, 0.0, "a much longer string", 16.0, true);
        let previous_bounds = text_line_bounds(&line(0.0, 0.0, "a", 16.0, false));
        let bounds = dirty_text_bounds(std::slice::from_ref(&grown), &[previous_bounds]).unwrap();
        let current = text_line_bounds(&grown);
        assert_eq!(
            bounds, current,
            "previous bounds are fully contained, so union must equal current bounds"
        );
    }

    #[test]
    fn dirty_text_bounds_unions_when_line_moves_without_resizing() {
        // A line that translates (same size, new position) between frames —
        // current and previous bounds do not overlap at all, so the union
        // must be the bounding box that spans both, not just one of them.
        let moved = line(500.0, 500.0, "a", 16.0, true);
        let previous_bounds = text_line_bounds(&line(0.0, 0.0, "a", 16.0, false));
        let bounds = dirty_text_bounds(std::slice::from_ref(&moved), &[previous_bounds]).unwrap();
        let current = text_line_bounds(&moved);
        let expected = current.union(&previous_bounds);
        assert_eq!(bounds, expected);
        // Sanity: the union must still reach back to the old (top-left)
        // position, not just the new one.
        assert_f32_eq(bounds.x, 0.0);
        assert_f32_eq(bounds.y, 0.0);
    }

    #[test]
    fn dirty_text_bounds_handles_previous_shorter_than_texts() {
        // `previous` is positionally aligned with `texts`, but a brand-new
        // line added this frame has no corresponding entry from the last
        // call (the slice is shorter than `texts`). `previous.get(i)` must
        // return `None` for it rather than panicking on an out-of-bounds
        // index, and the line's current bounds alone must be used.
        let existing = line(0.0, 0.0, "a", 16.0, false);
        let new_line = line(50.0, 50.0, "b", 16.0, true);
        let previous = [text_line_bounds(&existing)];
        let bounds = dirty_text_bounds(&[existing, new_line.clone()], &previous).unwrap();
        let expected = text_line_bounds(&new_line);
        assert_eq!(
            bounds, expected,
            "a newly added dirty line with no previous entry must use only its current bounds"
        );
    }

    #[test]
    fn logical_rect_to_physical_scissor_scales_by_dpi_factor() {
        let rect = Rect::new(10.0, 20.0, 30.0, 40.0);
        let scissor = logical_rect_to_physical_scissor(rect, 2.0, 10_000, 10_000);
        assert_eq!(scissor, (20, 40, 60, 80));
    }

    #[test]
    fn logical_rect_to_physical_scissor_clamps_to_target_bounds() {
        // A rect that overshoots the physical target (e.g. from rounding,
        // or a heuristic text bound near the edge of the window) must be
        // clamped — wgpu rejects a scissor rect that exceeds the target.
        let rect = Rect::new(90.0, 90.0, 50.0, 50.0);
        let (scissor_x, scissor_y, scissor_w, scissor_h) =
            logical_rect_to_physical_scissor(rect, 1.0, 100, 100);
        assert_eq!(scissor_x, 90);
        assert_eq!(scissor_y, 90);
        assert_eq!(scissor_w, 10, "clamped to max_w - x");
        assert_eq!(scissor_h, 10, "clamped to max_h - y");
    }

    #[test]
    fn logical_rect_to_physical_scissor_handles_origin_outside_bounds() {
        // A rect entirely past the target's edge collapses to a zero-size
        // scissor rather than going negative-width.
        let r = Rect::new(200.0, 200.0, 50.0, 50.0);
        let (_, _, w, h) = logical_rect_to_physical_scissor(r, 1.0, 100, 100);
        assert_eq!(w, 0);
        assert_eq!(h, 0);
    }

    #[test]
    fn needs_full_redraw_true_when_sticky_flag_set() {
        assert!(needs_full_redraw_this_frame(true, 3, 3, 3, 3));
    }

    #[test]
    fn needs_full_redraw_false_when_nothing_changed_and_not_sticky() {
        assert!(!needs_full_redraw_this_frame(false, 3, 3, 3, 3));
    }

    #[test]
    fn needs_full_redraw_true_when_instance_count_changes() {
        assert!(needs_full_redraw_this_frame(false, 3, 4, 3, 3));
        assert!(needs_full_redraw_this_frame(false, 4, 3, 3, 3));
    }

    #[test]
    fn needs_full_redraw_true_when_text_count_changes() {
        assert!(needs_full_redraw_this_frame(false, 3, 3, 2, 3));
        assert!(needs_full_redraw_this_frame(false, 3, 3, 3, 2));
    }

    // ── compute_scissor ───────────────────────────────────────────────────────
    //
    // `encode_frame` never calls `dirty_text_bounds` or
    // `logical_rect_to_physical_scissor` directly on an incremental frame —
    // it goes through `compute_scissor`, so the composition of the two
    // (including the zero-size-rect rejection) needs its own coverage, not
    // just each half in isolation.

    #[test]
    fn compute_scissor_is_none_when_nothing_is_dirty() {
        let texts = [line(0.0, 0.0, "a", 16.0, false)];
        assert!(compute_scissor(&ScissorInputs::text_only(&texts, &[]), 1.0, 1000, 1000).is_none());
    }

    #[test]
    fn compute_scissor_is_none_for_empty_texts() {
        assert!(compute_scissor(&ScissorInputs::text_only(&[], &[]), 1.0, 1000, 1000).is_none());
    }

    #[test]
    fn compute_scissor_returns_physical_rect_for_a_dirty_line() {
        let texts = [line(10.0, 20.0, "hello", 16.0, true)];
        let (bounds, x, y, w, h) =
            compute_scissor(&ScissorInputs::text_only(&texts, &[]), 2.0, 1000, 1000).unwrap();
        let expected_bounds = text_line_bounds(&texts[0]);
        assert_eq!(bounds, expected_bounds, "logical bounds must be unscaled");
        let (expected_x, expected_y, expected_w, expected_h) =
            logical_rect_to_physical_scissor(expected_bounds, 2.0, 1000, 1000);
        assert_eq!(
            (x, y, w, h),
            (expected_x, expected_y, expected_w, expected_h)
        );
    }

    #[test]
    // `previous_bounds.width.ceil() as u32` mirrors the same lossless-in-practice
    // cast already allowed in `logical_rect_to_physical_scissor` (no real text
    // bound is anywhere near 2^24 logical pixels wide).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn compute_scissor_unions_with_previous_bounds() {
        // Mirrors `dirty_text_bounds_unions_with_previous_frame_bounds`, but
        // through the full `compute_scissor` path that `encode_frame`
        // actually calls — a shrinking line's stale footprint must still be
        // covered by the resulting *physical* scissor rect, not just the
        // logical bounds in isolation.
        let shrunk = line(0.0, 0.0, "a", 16.0, true);
        let previous_bounds =
            text_line_bounds(&line(0.0, 0.0, "a much longer string", 16.0, false));
        let texts = [shrunk];
        let (bounds, _, _, w, _) = compute_scissor(
            &ScissorInputs::text_only(&texts, &[previous_bounds]),
            1.0,
            1000,
            1000,
        )
        .unwrap();
        assert!(
            bounds.width >= previous_bounds.width,
            "logical union must retain the previous (wider) footprint"
        );
        assert!(
            w >= previous_bounds.width.ceil() as u32,
            "physical scissor width must be wide enough to cover the stale footprint"
        );
    }

    #[test]
    fn compute_scissor_is_none_when_dirty_rect_lies_entirely_outside_target() {
        // The dirty bounds are non-empty but fall entirely past the
        // physical target's edge, so `logical_rect_to_physical_scissor`
        // collapses them to a zero-size rect — wgpu rejects a zero-size
        // scissor, so `compute_scissor` must surface `None` rather than a
        // degenerate `Some((..., 0, 0))`.
        let texts = [line(2000.0, 2000.0, "offscreen", 16.0, true)];
        assert!(compute_scissor(&ScissorInputs::text_only(&texts, &[]), 1.0, 100, 100).is_none());
    }

    // ── M26/M27: box / decorated / texture dirty bounds + combined scissor ────

    /// Builds a `BoxInstance` at `(x, y, w, h)` (colour/radii irrelevant to
    /// the bounds helpers under test).
    fn box_at(x: f32, y: f32, w: f32, h: f32) -> BoxInstance {
        BoxInstance {
            rect: [x, y, w, h],
            color: [0.0; 4],
            radii: [0.0; 4],
            transform: Transform::IDENTITY,
        }
    }

    fn decorated_at(x: f32, y: f32, w: f32, h: f32, dirty: bool) -> crate::frame::DecoratedBox {
        crate::frame::DecoratedBox {
            base: box_at(x, y, w, h),
            dirty,
            ..Default::default()
        }
    }

    fn texture_at(x: f32, y: f32, w: f32, h: f32, dirty: bool) -> crate::frame::TextureSampler {
        crate::frame::TextureSampler {
            rect: [x, y, w, h],
            src: String::new(),
            fit: crate::frame::ImageFit::Fill,
            radii: [0.0; 4],
            opacity: 1.0,
            dirty,
        }
    }

    #[test]
    fn dirty_box_bounds_is_none_when_nothing_is_dirty() {
        let boxes = [box_at(0.0, 0.0, 10.0, 10.0), box_at(50.0, 50.0, 10.0, 10.0)];
        assert!(dirty_box_bounds(&boxes, &[false, false], &[]).is_none());
    }

    #[test]
    fn dirty_box_bounds_ignores_non_dirty_boxes() {
        let boxes = [
            box_at(10.0, 10.0, 20.0, 20.0),
            box_at(900.0, 900.0, 5.0, 5.0),
        ];
        let bounds = dirty_box_bounds(&boxes, &[true, false], &[]).unwrap();
        assert_eq!(
            bounds,
            rect_of(boxes[0].rect),
            "a clean box must not widen the union"
        );
    }

    #[test]
    fn dirty_box_bounds_unions_with_previous_frame_bounds() {
        // A box that shrinks between frames: its old (wider) footprint must
        // still be covered, mirroring `dirty_text_bounds_unions_with_previous_*`.
        let shrunk = box_at(0.0, 0.0, 10.0, 10.0);
        let previous = Rect::new(0.0, 0.0, 200.0, 200.0);
        let bounds = dirty_box_bounds(std::slice::from_ref(&shrunk), &[true], &[previous]).unwrap();
        assert_eq!(bounds, rect_of(shrunk.rect).union(&previous));
        assert!(bounds.width >= previous.width);
    }

    #[test]
    fn compute_scissor_is_some_when_only_a_box_is_dirty_and_no_text_exists() {
        // The M26 regression test: no text at all, one dirty box. The old
        // text-only `compute_scissor` returned `None` here, so `should_draw`
        // was false and the box mutation never reached the screen.
        let boxes = [box_at(10.0, 20.0, 30.0, 40.0)];
        let inputs = ScissorInputs {
            instances: &boxes,
            instances_dirty: &[true],
            ..ScissorInputs::text_only(&[], &[])
        };
        let scissor = compute_scissor(&inputs, 1.0, 1000, 1000);
        assert!(
            scissor.is_some(),
            "a box-only mutation must produce a non-empty scissor"
        );
        let (bounds, ..) = scissor.unwrap();
        assert_eq!(bounds, rect_of(boxes[0].rect));
    }

    #[test]
    fn compute_scissor_unions_box_and_text_dirty_regions() {
        // One dirty box far from one dirty text line; the scissor must cover
        // both regions, not just one.
        let texts = [line(0.0, 0.0, "a", 16.0, true)];
        let boxes = [box_at(500.0, 500.0, 40.0, 40.0)];
        let inputs = ScissorInputs {
            instances: &boxes,
            instances_dirty: &[true],
            ..ScissorInputs::text_only(&texts, &[])
        };
        let (bounds, ..) = compute_scissor(&inputs, 1.0, 1000, 1000).unwrap();
        let expected = text_line_bounds(&texts[0]).union(&rect_of(boxes[0].rect));
        assert_eq!(bounds, expected);
    }

    #[test]
    fn dirty_texture_bounds_is_none_when_nothing_is_dirty() {
        let textures = [texture_at(0.0, 0.0, 10.0, 10.0, false)];
        assert!(dirty_texture_bounds(&textures, &[]).is_none());
    }

    #[test]
    fn dirty_texture_bounds_ignores_non_dirty_textures() {
        let textures = [
            texture_at(10.0, 10.0, 20.0, 20.0, true),
            texture_at(900.0, 900.0, 5.0, 5.0, false),
        ];
        let bounds = dirty_texture_bounds(&textures, &[]).unwrap();
        assert_eq!(bounds, rect_of(textures[0].rect));
    }

    #[test]
    fn dirty_texture_bounds_unions_with_previous_frame_bounds() {
        let shrunk = texture_at(0.0, 0.0, 10.0, 10.0, true);
        let previous = Rect::new(0.0, 0.0, 200.0, 200.0);
        let bounds = dirty_texture_bounds(std::slice::from_ref(&shrunk), &[previous]).unwrap();
        assert_eq!(bounds, rect_of(shrunk.rect).union(&previous));
    }

    #[test]
    fn compute_scissor_does_not_force_full_redraw_when_a_clean_decorated_box_is_present() {
        // The actual point of M27: a scene with one *non-dirty* DecoratedBox
        // and one dirty text line must scissor to the text's bounds only — not
        // the whole viewport (which is what the old forced-`full_redraw` block,
        // now deleted, effectively did).
        let texts = [line(10.0, 20.0, "hi", 16.0, true)];
        let decorated = [decorated_at(0.0, 0.0, 999.0, 999.0, false)];
        let inputs = ScissorInputs {
            decorated: &decorated,
            ..ScissorInputs::text_only(&texts, &[])
        };
        let (bounds, ..) = compute_scissor(&inputs, 1.0, 1000, 1000).unwrap();
        assert_eq!(
            bounds,
            text_line_bounds(&texts[0]),
            "a clean decorated box must not expand the scissor"
        );
    }

    #[test]
    fn compute_scissor_includes_a_dirty_decorated_box() {
        let decorated = [decorated_at(100.0, 100.0, 50.0, 50.0, true)];
        let inputs = ScissorInputs {
            decorated: &decorated,
            ..ScissorInputs::text_only(&[], &[])
        };
        let (bounds, ..) = compute_scissor(&inputs, 1.0, 1000, 1000).unwrap();
        assert_eq!(bounds, rect_of(decorated[0].base.rect));
    }
}
