//! # Engine
//!
//! Top-level orchestrator that binds the encoder subsystem to a `wgpu` surface.
//!
//! [`Engine`] is the public entry point for platform code. The platform
//! creates one `Engine` per window, notifies it of resize events via
//! [`Engine::on_resize`], and drives it frame-by-frame via
//! [`Engine::render_frame`]. The engine never imports windowing primitives
//! (`winit`, `raw-window-handle`, etc.) — surface creation and window
//! lifecycle are entirely the platform's responsibility (RFC-0001 §6).
//!
//! ## Coordinate system
//!
//! All instance coordinates (rect, radii, etc.) and the viewport uniform are
//! in **logical pixels** — the same density-independent unit used by CSS,
//! `SwiftUI` points, and Android dp. On `HiDPI` displays (Retina 2×, etc.) the
//! platform must supply the OS-reported `scale_factor` so that the engine can
//! internally convert physical pixels → logical pixels for the viewport
//! uniform, while keeping the `wgpu` surface in physical pixels as required
//! by the API.
//!
//! ```text
//!   Platform (winit / etc.)                    byard-core
//!   ──────────────────────                     ──────────────────────────────
//!   window resize event (physical + scale) ──► Engine::on_resize(w, h, scale)
//!   RedrawRequested event                  ──► Engine::render_frame(&instances)
//!                                                  └─ EncoderSubsystem::encode_frame
//!                                                  └─ queue.submit + surface.present
//! ```

use std::sync::Arc;

use crate::ByardError;
use crate::atlas::{LayoutAtlas, LeafSize};
use crate::encoder::text_glyph::TextLine;
use crate::encoder::{BoxInstance, EncoderSubsystem};
use crate::evaluator::{EvaluatorTick, Signal, ViewArena};
use crate::frame::{TargetId, TargetKind, Viewport};

/// The Byard rendering engine for a single window surface.
///
/// Owns the GPU device, queue, compiled pipelines, and the `wgpu` surface.
/// Constructed once at startup via [`Engine::init`]; thereafter the platform
/// calls [`on_resize`](Engine::on_resize) and [`render_frame`](Engine::render_frame)
/// in response to OS events.
///
/// `Engine` also owns one Signal-driven reactive text label ([`ReactiveLabel`]):
/// every [`render_frame`](Engine::render_frame) call runs a real
/// `Evaluator → Atlas` tick against it (not just the unit tests in
/// `atlas/layout.rs`), and [`Engine::set_label_text`] is the production
/// mutation path. This is what closes Phase 1's stated closure criterion —
/// "the engine renders ... a reactive text label driven by a Signal" — by
/// exercising Evaluator, Atlas, and Encoder together outside of test code.
pub struct Engine {
    encoder: EncoderSubsystem,
    surface: wgpu::Surface<'static>,
    /// Cached surface configuration (physical pixels), updated on every resize
    /// so that surface-loss recovery can reconfigure without external input.
    surface_config: wgpu::SurfaceConfiguration,
    /// Cached scale factor so surface-loss recovery can recompute the logical
    /// viewport without additional platform input.
    scale_factor: f64,
    /// The engine's one Signal-driven text label. See the struct doc above.
    label: ReactiveLabel,
    /// Reused per-frame scratch buffer for the combined text list (the
    /// reactive label plus whatever static lines the caller supplies) —
    /// avoids a per-frame allocation once warmed up, per RFC-0001's
    /// deterministic-memory goals.
    text_scratch: Vec<TextLine>,
}

/// Backing state for [`Engine`]'s one Signal-driven text label.
///
/// Bundles a [`ViewArena`], the [`Signal<String>`](Signal) allocated from
/// it, an [`EvaluatorTick`] tracking that signal, and a trivial single-node
/// [`LayoutAtlas`] that receives the resulting dirty-target broadcast — the
/// same Evaluator → Atlas flow RFC-0001 §2.2/§4.1 describes, now exercised
/// by production code instead of only by `atlas/layout.rs`'s unit tests.
///
/// # Self-referential lifetime
///
/// `Signal<'a, T>` ties its lifetime to the [`ViewArena`] it was allocated
/// from. Storing both as sibling fields is therefore self-referential,
/// which safe Rust cannot express directly. This is resolved the same way
/// any self-referential owner must: heap-allocate the arena (`Box`, so its
/// address is stable even if `ReactiveLabel` itself moves) and erase the
/// signal's lifetime to `'static` via [`Signal::erase_lifetime`]. This is
/// sound because `arena` and `signal` are dropped together — nothing
/// outside this struct ever holds a copy of `signal`, so it is never used
/// after `arena` is gone.
struct ReactiveLabel {
    // Boxed so the heap address backing `signal`'s slot never moves, even
    // if `ReactiveLabel` (or the `Engine` that owns it) is moved. Never
    // read directly — its only job is to keep that heap allocation (and
    // therefore `signal`'s backing slot) alive for as long as `ReactiveLabel`
    // exists; the value is reached exclusively through `signal`'s erased
    // `'static` handle, not through this field.
    #[allow(dead_code, reason = "kept alive only to back `signal`'s slot")]
    arena: Box<ViewArena>,
    signal: Signal<'static, String>,
    tick: EvaluatorTick<'static>,
    atlas: LayoutAtlas,
    target: TargetId,
    x: f32,
    y: f32,
    font_size: f32,
    color: [f32; 4],
}

impl ReactiveLabel {
    fn new(text: impl Into<String>, x: f32, y: f32, font_size: f32, color: [f32; 4]) -> Self {
        let arena = Box::new(ViewArena::new());
        // `arena` is boxed (stable heap address) and dropped together with
        // `signal` when `ReactiveLabel` (and the `Engine` that owns it) is
        // dropped — see the struct doc above. `Signal::new_in_boxed` is the
        // safe wrapper around the `unsafe` lifetime erasure that pattern
        // requires; the `unsafe` block itself stays inside `signal.rs`,
        // the evaluator subsystem file that owns this invariant.
        let signal: Signal<'static, String> = Signal::new_in_boxed(&arena, text.into());

        // A single trivial (zero-sized) leaf is enough to give the label a
        // real AtlasNode TargetId to subscribe to and mark dirty — Phase 1
        // does not yet thread Atlas-resolved geometry into `TextLine` (x/y
        // are authored directly), so this leaf's only job is to participate
        // honestly in the dirty-broadcast flow, not to compute position.
        let mut atlas = LayoutAtlas::new();
        let node = atlas
            .add_leaf(LeafSize {
                width: 0.0,
                height: 0.0,
            })
            .expect("a single freshly created leaf can never fail to add");
        atlas
            .set_root(node)
            .expect("the node just created always belongs to this atlas");
        atlas
            .compute(Viewport::new(0.0, 0.0))
            .expect("computing layout for one zero-sized leaf can never fail");

        let target = TargetId::new(0, atlas.current_generation(), TargetKind::AtlasNode as u16);
        signal.subscribe(target);

        let mut tick = EvaluatorTick::new();
        tick.register(signal);

        Self {
            arena,
            signal,
            tick,
            atlas,
            target,
            x,
            y,
            font_size,
            color,
        }
    }

    /// Overwrites the label's text content. The signal's version counter
    /// advances; the next [`ReactiveLabel::text_line`] call observes it.
    fn set_text(&self, text: impl Into<String>) {
        self.signal.write(|v| *v = text.into());
    }

    /// Runs one Evaluator → Atlas tick and returns this frame's `TextLine`,
    /// with `dirty` reflecting the real dirty-flag pipeline output — never
    /// a hardcoded value.
    fn text_line(&mut self) -> Result<TextLine, ByardError> {
        let dirty_targets = self.tick.collect_dirty();
        let dirty = if dirty_targets.is_empty() {
            false
        } else {
            self.atlas.mark_dirty_all(&dirty_targets);
            self.atlas.recompute_dirty(Viewport::new(0.0, 0.0))?;
            dirty_targets.contains(&self.target)
        };

        Ok(TextLine {
            x: self.x,
            y: self.y,
            text: self.signal.read(String::clone),
            font_size: self.font_size,
            color: self.color,
            dirty,
        })
    }
}

impl Engine {
    /// Initialises the engine, selects a GPU adapter, and compiles all pipelines.
    ///
    /// Performs adapter selection, device creation, surface format negotiation,
    /// surface configuration, and pipeline compilation. All GPU errors are
    /// captured and returned as [`ByardError`] — this method never panics.
    ///
    /// ## Parameters
    ///
    /// - `width`, `height` — initial surface dimensions in **physical pixels**
    ///   (`window.inner_size()` in winit). Used for the `wgpu` surface only.
    /// - `scale_factor` — OS DPI scale factor (`window.scale_factor()` in
    ///   winit; typically `1.0` on non-HiDPI, `2.0` on Retina). Used to
    ///   convert physical pixels → logical pixels for the viewport uniform so
    ///   that all instance coordinates (rect, radii, etc.) can be authored in
    ///   logical pixels regardless of display density.
    ///
    /// # Errors
    ///
    /// - [`ByardError::UnsupportedBackend`] — no compatible GPU adapter found,
    ///   or logical device creation failed.
    /// - [`ByardError::PipelineCompilation`] — WGSL shader or pipeline
    ///   descriptor failed GPU-side validation.
    pub async fn init(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
        scale_factor: f64,
    ) -> Result<Self, ByardError> {
        // --- Adapter selection ---
        // wgpu 29: request_adapter returns Result<Adapter, RequestAdapterError>
        // instead of Option<Adapter>; use map_err instead of ok_or.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| ByardError::UnsupportedBackend)?;

        // --- Logical device ---
        // wgpu 29: request_device takes only &DeviceDescriptor (trace path moved
        // into DeviceDescriptor::trace); use ..Default::default() to zero-fill
        // the new `experimental_features` and `trace` fields.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ByardCore - Engine Device"),
                required_features: wgpu::Features::empty(),
                // Use the adapter's own limits — no artificial WebGL2 cap.
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|_| ByardError::UnsupportedBackend)?;

        // --- Surface format negotiation ---
        // Prefer sRGB so that linear-space colours in shaders display correctly.
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        // --- Surface configuration (physical pixels — wgpu requirement) ---
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // --- Encoder subsystem ---
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let scale_f32 = scale_factor as f32;
        let mut encoder = EncoderSubsystem::init(
            Arc::clone(&device),
            Arc::clone(&queue),
            surface_format,
            scale_f32,
        )
        .await?;

        // Viewport uniform uses LOGICAL pixels so all instance coordinates can
        // be authored in density-independent units. cast precision loss is
        // acceptable: logical pixel counts fit well within f32's 24-bit mantissa.
        #[allow(clippy::cast_precision_loss)]
        encoder.update_viewport(
            logical_viewport(width, height, scale_factor),
            width,
            height,
            scale_f32,
        );

        Ok(Self {
            encoder,
            surface,
            surface_config,
            scale_factor,
            label: ReactiveLabel::new("Byard — Phase 1", 110.0, 110.0, 20.0, [1.0, 1.0, 1.0, 1.0]),
            text_scratch: Vec::new(),
        })
    }

    /// Replaces the engine's reactive label text.
    ///
    /// The next [`Engine::render_frame`] call picks up the change, with the
    /// resulting `TextLine`'s `dirty` bit set from the real Evaluator →
    /// Atlas dirty-flag pipeline (RFC-0001 §2.2, §4.1) — this is the
    /// production mutation path for Phase 1's "reactive text label driven
    /// by a Signal" closure criterion.
    pub fn set_label_text(&self, text: impl Into<String>) {
        self.label.set_text(text);
    }

    /// Notifies the engine that the window surface has been resized.
    ///
    /// `width` and `height` are the new dimensions in **physical pixels**
    /// (`window.inner_size()` in winit). `scale_factor` is the OS DPI scale
    /// factor — pass `window.scale_factor()` and also call this method from
    /// `WindowEvent::ScaleFactorChanged` so the viewport uniform stays correct
    /// when the window moves between displays of different densities.
    ///
    /// Calls with `width == 0` or `height == 0` are silently ignored — zero-size
    /// surfaces are invalid in wgpu (occurs on window minimise on some platforms).
    pub fn on_resize(&mut self, width: u32, height: u32, scale_factor: f64) {
        if width == 0 || height == 0 {
            return;
        }
        self.scale_factor = scale_factor;
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface
            .configure(self.encoder.device(), &self.surface_config);
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        self.encoder.update_viewport(
            logical_viewport(width, height, scale_factor),
            width,
            height,
            scale_factor as f32,
        );
    }

    /// Renders one frame to the window surface.
    ///
    /// Acquires the next surface texture, encodes a render pass that draws
    /// all `instances`, submits the command buffer to the GPU queue, and
    /// presents the frame to the display.
    ///
    /// # Surface loss
    ///
    /// If the surface is lost or outdated (window minimise/restore, driver
    /// reset), the engine silently reconfigures it and returns `Ok(())`.
    /// The next call to `render_frame` will produce output normally. The
    /// platform does not need to handle surface loss explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`ByardError::RenderSurface`] only on unrecoverable surface
    /// errors such as out-of-memory or GPU timeout.
    pub fn render_frame(
        &mut self,
        instances: &[BoxInstance],
        texts: &[TextLine],
    ) -> Result<(), ByardError> {
        // wgpu 29: get_current_texture() returns CurrentSurfaceTexture (an enum),
        // replacing the old Result<SurfaceTexture, SurfaceError> API.
        let frame = match self.surface.get_current_texture() {
            // Suboptimal: surface is valid but should be reconfigured (e.g. wrong
            // DPI or format). Draw this frame and let the next on_resize fix it.
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            // Lost / Outdated: reconfigure with the last known dimensions and
            // skip this frame. The surface will be healthy on the next call.
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface
                    .configure(self.encoder.device(), &self.surface_config);
                return Ok(());
            }
            // Timeout / Occluded: transient; skip frame, next will succeed.
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(ByardError::RenderSurface(
                    "GPU validation error during surface texture acquire".to_string(),
                ));
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Run one Evaluator → Atlas tick for the engine's reactive label
        // and fold it into this frame's text list, ahead of whatever
        // static lines the caller supplies. This is the production code
        // path that exercises Signal/EvaluatorTick/LayoutAtlas — not just
        // the unit tests in `atlas/layout.rs`.
        let label_line = self.label.text_line()?;
        self.text_scratch.clear();
        self.text_scratch.push(label_line);
        self.text_scratch.extend_from_slice(texts);

        let cmd = self
            .encoder
            .encode_frame(&view, instances, &self.text_scratch)?;
        self.encoder.submit(cmd);
        frame.present();

        Ok(())
    }
}

/// Converts physical pixel dimensions + DPI scale factor to a [`Viewport`] in
/// logical pixels.
///
/// The viewport uniform is always in logical pixels so that instance
/// coordinates (rect, radii, etc.) can be authored once and render correctly
/// on every display density.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn logical_viewport(phys_w: u32, phys_h: u32, scale: f64) -> Viewport {
    Viewport::new(phys_w as f32 / scale as f32, phys_h as f32 / scale as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts two `f32` values are equal within a small tolerance.
    ///
    /// `assert_eq!` on `f32` triggers `clippy::float_cmp`; every value
    /// checked here is either passed straight through `ReactiveLabel`
    /// untouched or produced by a simple division, so an exact bit-pattern
    /// match would normally hold, but this tolerance keeps the intent
    /// (approximate equality) honest rather than relying on that
    /// incidental exactness. Mirrors `atlas::layout::tests::assert_f32_eq`.
    #[track_caller]
    fn assert_f32_eq(actual: f32, expected: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.001,
            "expected {expected}, got {actual} (diff = {diff})",
        );
    }

    /// Asserts two `[f32; 4]` colors are equal within [`assert_f32_eq`]'s
    /// tolerance, component by component.
    ///
    /// `assert_eq!` on a float array triggers `clippy::float_cmp` just like
    /// it does on a bare `f32` (the lint looks through array equality), so
    /// every color comparison in this module goes through here instead.
    #[track_caller]
    fn assert_color_eq(actual: [f32; 4], expected: [f32; 4]) {
        for i in 0..4 {
            assert_f32_eq(actual[i], expected[i]);
        }
    }

    /// Builds a `ReactiveLabel` with a fixed, distinctive position/size/color
    /// so tests that only care about text/dirty behaviour don't repeat the
    /// same four literals everywhere.
    fn label_with_text(text: &str) -> ReactiveLabel {
        ReactiveLabel::new(text, 10.0, 20.0, 16.0, [1.0, 0.5, 0.25, 0.75])
    }

    // ── construction: text content ─────────────────────────────────────

    #[test]
    fn new_stores_initial_text() {
        let mut label = label_with_text("hello");
        let line = label.text_line().expect("first tick never fails");
        assert_eq!(line.text, "hello");
    }

    #[test]
    fn new_with_empty_string_text() {
        let mut label = label_with_text("");
        let line = label.text_line().expect("first tick never fails");
        assert_eq!(line.text, "");
    }

    #[test]
    fn new_with_unicode_text() {
        let mut label = label_with_text("Byard — 🦀 ñ");
        let line = label.text_line().expect("first tick never fails");
        assert_eq!(line.text, "Byard — 🦀 ñ");
    }

    #[test]
    fn new_with_long_text() {
        let long = "x".repeat(10_000);
        let mut label = label_with_text(&long);
        let line = label.text_line().expect("first tick never fails");
        assert_eq!(line.text.len(), 10_000);
    }

    #[test]
    fn new_accepts_owned_string() {
        let owned: String = String::from("owned");
        let mut label = ReactiveLabel::new(owned, 0.0, 0.0, 12.0, [0.0, 0.0, 0.0, 1.0]);
        let line = label.text_line().expect("first tick never fails");
        assert_eq!(line.text, "owned");
    }

    #[test]
    fn new_accepts_str_slice() {
        let mut label = ReactiveLabel::new("slice", 0.0, 0.0, 12.0, [0.0, 0.0, 0.0, 1.0]);
        let line = label.text_line().expect("first tick never fails");
        assert_eq!(line.text, "slice");
    }

    // ── construction: position, size, color pass-through ────────────────

    #[test]
    fn new_stores_x_position() {
        let mut label = label_with_text("t");
        let line = label.text_line().expect("first tick never fails");
        assert_f32_eq(line.x, 10.0);
    }

    #[test]
    fn new_stores_y_position() {
        let mut label = label_with_text("t");
        let line = label.text_line().expect("first tick never fails");
        assert_f32_eq(line.y, 20.0);
    }

    #[test]
    fn new_stores_font_size() {
        let mut label = label_with_text("t");
        let line = label.text_line().expect("first tick never fails");
        assert_f32_eq(line.font_size, 16.0);
    }

    #[test]
    fn new_stores_color() {
        let mut label = label_with_text("t");
        let line = label.text_line().expect("first tick never fails");
        assert_color_eq(line.color, [1.0, 0.5, 0.25, 0.75]);
    }

    #[test]
    fn new_with_zero_font_size() {
        let mut label = ReactiveLabel::new("t", 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 1.0]);
        let line = label.text_line().expect("first tick never fails");
        assert_f32_eq(line.font_size, 0.0);
    }

    #[test]
    fn new_with_negative_position() {
        let mut label = ReactiveLabel::new("t", -50.0, -75.0, 12.0, [0.0, 0.0, 0.0, 1.0]);
        let line = label.text_line().expect("first tick never fails");
        assert_f32_eq(line.x, -50.0);
        assert_f32_eq(line.y, -75.0);
    }

    #[test]
    fn new_with_alpha_zero_color() {
        let mut label = ReactiveLabel::new("t", 0.0, 0.0, 12.0, [1.0, 1.0, 1.0, 0.0]);
        let line = label.text_line().expect("first tick never fails");
        assert_color_eq(line.color, [1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn new_with_all_color_components_at_max() {
        let mut label = ReactiveLabel::new("t", 0.0, 0.0, 12.0, [1.0, 1.0, 1.0, 1.0]);
        let line = label.text_line().expect("first tick never fails");
        assert_color_eq(line.color, [1.0, 1.0, 1.0, 1.0]);
    }

    // ── dirty-flag lifecycle ──────────────────────────────────────────

    #[test]
    fn first_text_line_is_not_dirty() {
        let mut label = label_with_text("t");
        let line = label.text_line().expect("first tick never fails");
        assert!(
            !line.dirty,
            "a label that was never written to has nothing to mark dirty"
        );
    }

    #[test]
    fn first_text_line_returns_ok() {
        let mut label = label_with_text("t");
        assert!(label.text_line().is_ok());
    }

    #[test]
    fn set_text_marks_next_tick_dirty() {
        let mut label = label_with_text("before");
        label.set_text("after");
        let line = label.text_line().expect("tick after a write never fails");
        assert!(line.dirty);
    }

    #[test]
    fn set_text_updates_text_content() {
        let mut label = label_with_text("before");
        label.set_text("after");
        let line = label.text_line().expect("tick after a write never fails");
        assert_eq!(line.text, "after");
    }

    #[test]
    fn set_text_does_not_change_x() {
        let mut label = label_with_text("before");
        label.set_text("after");
        let line = label.text_line().expect("tick after a write never fails");
        assert_f32_eq(line.x, 10.0);
    }

    #[test]
    fn set_text_does_not_change_y() {
        let mut label = label_with_text("before");
        label.set_text("after");
        let line = label.text_line().expect("tick after a write never fails");
        assert_f32_eq(line.y, 20.0);
    }

    #[test]
    fn set_text_does_not_change_font_size() {
        let mut label = label_with_text("before");
        label.set_text("after");
        let line = label.text_line().expect("tick after a write never fails");
        assert_f32_eq(line.font_size, 16.0);
    }

    #[test]
    fn set_text_does_not_change_color() {
        let mut label = label_with_text("before");
        label.set_text("after");
        let line = label.text_line().expect("tick after a write never fails");
        assert_color_eq(line.color, [1.0, 0.5, 0.25, 0.75]);
    }

    #[test]
    fn second_tick_after_dirty_tick_is_clean() {
        let mut label = label_with_text("before");
        label.set_text("after");
        let first = label.text_line().expect("tick after a write never fails");
        assert!(first.dirty);

        let second = label.text_line().expect("tick with no writes never fails");
        assert!(
            !second.dirty,
            "no write happened between the two ticks, so the second must be clean"
        );
    }

    #[test]
    fn second_tick_after_dirty_tick_keeps_the_same_text() {
        let mut label = label_with_text("before");
        label.set_text("after");
        let _ = label.text_line().expect("tick after a write never fails");
        let second = label.text_line().expect("tick with no writes never fails");
        assert_eq!(second.text, "after");
    }

    #[test]
    fn multiple_writes_between_ticks_yield_a_single_dirty_tick() {
        let mut label = label_with_text("v0");
        label.set_text("v1");
        label.set_text("v2");
        label.set_text("v3");

        let line = label
            .text_line()
            .expect("tick after several writes never fails");
        assert!(line.dirty);

        let next = label.text_line().expect("tick with no writes never fails");
        assert!(
            !next.dirty,
            "the three writes must collapse into exactly one dirty tick"
        );
    }

    #[test]
    fn multiple_writes_between_ticks_yield_the_latest_text() {
        let mut label = label_with_text("v0");
        label.set_text("v1");
        label.set_text("v2");
        label.set_text("v3");

        let line = label
            .text_line()
            .expect("tick after several writes never fails");
        assert_eq!(line.text, "v3");
    }

    #[test]
    fn set_text_with_same_value_still_marks_dirty() {
        // A Signal::write always advances the version counter, even when the
        // new value is identical to the old one — there is no value-equality
        // short-circuit anywhere in the Evaluator. This is intentional (see
        // RFC-0001 §2.2): cheaply comparing arbitrary `T` would not be
        // possible in general, and skipping it keeps the pipeline simple.
        let mut label = label_with_text("same");
        label.set_text("same");
        let line = label.text_line().expect("tick after a write never fails");
        assert!(line.dirty);
        assert_eq!(line.text, "same");
    }

    #[test]
    fn set_text_accepts_owned_string() {
        let mut label = label_with_text("before");
        let owned: String = String::from("after");
        label.set_text(owned);
        let line = label.text_line().expect("tick after a write never fails");
        assert_eq!(line.text, "after");
    }

    #[test]
    fn set_text_accepts_str_slice() {
        let mut label = label_with_text("before");
        label.set_text("after");
        let line = label.text_line().expect("tick after a write never fails");
        assert_eq!(line.text, "after");
    }

    #[test]
    fn set_text_accepts_formatted_string() {
        let mut label = label_with_text("before");
        label.set_text(format!("clicked {} time(s)", 3));
        let line = label.text_line().expect("tick after a write never fails");
        assert_eq!(line.text, "clicked 3 time(s)");
    }

    #[test]
    fn sequential_set_text_calls_apply_in_order() {
        let mut label = label_with_text("start");
        for n in 1..=5 {
            label.set_text(format!("step {n}"));
            let line = label.text_line().expect("each tick never fails");
            assert_eq!(line.text, format!("step {n}"));
            assert!(line.dirty, "each step wrote, so each tick must be dirty");
        }
    }

    #[test]
    fn alternating_write_and_tick_toggles_dirty_each_time() {
        let mut label = label_with_text("v0");
        for n in 1..=4 {
            label.set_text(format!("v{n}"));
            let dirty_line = label.text_line().expect("tick after a write never fails");
            assert!(
                dirty_line.dirty,
                "iteration {n}: expected dirty after write"
            );

            let clean_line = label.text_line().expect("tick with no writes never fails");
            assert!(
                !clean_line.dirty,
                "iteration {n}: expected clean with no intervening write"
            );
        }
    }

    #[test]
    fn repeated_ticks_with_no_writes_stay_clean() {
        let mut label = label_with_text("static");
        // Consume the initial (clean) state, then tick several more times
        // with no writes in between — every one must stay clean.
        for _ in 0..10 {
            let line = label.text_line().expect("tick with no writes never fails");
            assert!(!line.dirty);
            assert_eq!(line.text, "static");
        }
    }

    #[test]
    fn text_line_can_be_called_many_times_consecutively() {
        let mut label = label_with_text("t");
        for _ in 0..100 {
            assert!(label.text_line().is_ok());
        }
    }

    // ── independence between separate labels ─────────────────────────

    #[test]
    fn two_labels_have_independent_text() {
        let mut a = label_with_text("a-text");
        let mut b = label_with_text("b-text");
        assert_eq!(a.text_line().unwrap().text, "a-text");
        assert_eq!(b.text_line().unwrap().text, "b-text");
    }

    #[test]
    fn two_labels_have_independent_dirty_state() {
        let mut a = label_with_text("a");
        let mut b = label_with_text("b");

        a.set_text("a2");
        let a_line = a.text_line().expect("tick after a write never fails");
        let b_line = b.text_line().expect("tick with no writes never fails");

        assert!(a_line.dirty, "a was written to");
        assert!(!b_line.dirty, "b was never touched");
    }

    #[test]
    fn set_text_on_one_label_does_not_affect_another() {
        let mut a = label_with_text("a");
        let mut b = label_with_text("b");

        a.set_text("changed");
        let _ = a.text_line().expect("tick after a write never fails");
        let b_line = b.text_line().expect("tick with no writes never fails");

        assert_eq!(b_line.text, "b", "writing to `a` must not leak into `b`");
    }

    // ── logical_viewport: pure conversion helper ──────────────────────

    #[test]
    fn logical_viewport_identity_at_scale_one() {
        let viewport = logical_viewport(800, 600, 1.0);
        assert_f32_eq(viewport.width, 800.0);
        assert_f32_eq(viewport.height, 600.0);
    }

    #[test]
    fn logical_viewport_halves_dimensions_at_scale_two() {
        let viewport = logical_viewport(800, 600, 2.0);
        assert_f32_eq(viewport.width, 400.0);
        assert_f32_eq(viewport.height, 300.0);
    }

    #[test]
    fn logical_viewport_scales_fractionally() {
        // A 1.5x DPI scale (common on some Windows/Linux HiDPI setups).
        let viewport = logical_viewport(1200, 900, 1.5);
        assert_f32_eq(viewport.width, 800.0);
        assert_f32_eq(viewport.height, 600.0);
    }

    #[test]
    fn logical_viewport_zero_dimensions_stay_zero() {
        let viewport = logical_viewport(0, 0, 2.0);
        assert_f32_eq(viewport.width, 0.0);
        assert_f32_eq(viewport.height, 0.0);
    }

    #[test]
    fn logical_viewport_large_dimensions_do_not_overflow() {
        // 8K physical resolution at 1x scale — comfortably within f32's
        // 24-bit mantissa (RFC-0001 already accepts this precision
        // trade-off for viewport math; see `Engine::init`'s `cast_precision_loss`
        // allow).
        let viewport = logical_viewport(7680, 4320, 1.0);
        assert_f32_eq(viewport.width, 7680.0);
        assert_f32_eq(viewport.height, 4320.0);
    }
}
