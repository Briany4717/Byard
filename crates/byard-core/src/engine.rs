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
use crate::encoder::text_glyph::TextLine;
use crate::encoder::{BoxInstance, EncoderSubsystem};
use crate::frame::Viewport;

/// The Byard rendering engine for a single window surface.
///
/// Owns the GPU device, queue, compiled pipelines, and the `wgpu` surface.
/// Constructed once at startup via [`Engine::init`]; thereafter the platform
/// calls [`on_resize`](Engine::on_resize) and [`render_frame`](Engine::render_frame)
/// in response to OS events.
pub struct Engine {
    encoder: EncoderSubsystem,
    surface: wgpu::Surface<'static>,
    /// Cached surface configuration (physical pixels), updated on every resize
    /// so that surface-loss recovery can reconfigure without external input.
    surface_config: wgpu::SurfaceConfiguration,
    /// Cached scale factor so surface-loss recovery can recompute the logical
    /// viewport without additional platform input.
    scale_factor: f64,
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
        })
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

        let cmd = self.encoder.encode_frame(&view, instances, texts)?;
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
