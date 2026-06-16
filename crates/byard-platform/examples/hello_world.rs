//! Phase 1 visual verification — renders two `SolidBox` instances to a window.
//!
//! This example is intentionally minimal: it implements [`PlatformHost`] and
//! hands it to [`WinitHost`], drawing a static scene to confirm that the
//! `SolidBox` pipeline, SDF border-radius shader, and NDC transform all
//! produce correct output. It used to drive `winit`'s `ApplicationHandler`
//! directly inside `byard-core`; it now lives here because `byard-core` has
//! zero direct references to `winit` (RFC-0001 §6) and this is the crate
//! that owns the window.
//!
//! Run with:
//! ```sh
//! cargo run --example hello_world
//! ```

use byard_core::{BoxInstance, ByardError, Engine, PlatformHost, TextLine, WindowSize};
use byard_platform::WinitHost;

fn main() {
    let host = WinitHost::new("Byard — Phase 1 Verification", 800, 600);
    host.run(App::default()).expect("event loop error");
}

/// Application state implementing [`PlatformHost`].
///
/// `engine` is `None` until [`PlatformHost::on_resume`] fires (the first
/// point at which a window/surface exists on every platform `winit`
/// supports, including Android and iOS).
#[derive(Default)]
struct App {
    engine: Option<Engine>,
    /// Static scene: three shapes exercising radii, colour, and transparency.
    ///
    /// All coordinates are in **logical pixels** (density-independent units).
    /// The engine converts to physical pixels internally using the window's
    /// DPI scale factor, so this list renders identically on Retina and
    /// non-HiDPI displays.
    instances: Vec<BoxInstance>,
    /// Static text overlay rendered in the same pass as `instances`.
    texts: Vec<TextLine>,
}

impl PlatformHost for App {
    fn on_resume(
        &mut self,
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        size: WindowSize,
    ) -> Result<(), ByardError> {
        let engine = pollster::block_on(Engine::init(
            instance,
            surface,
            size.width,
            size.height,
            size.scale_factor,
        ))?;

        self.instances = vec![
            // Solid blue rectangle with uniform 16 px border-radius (logical).
            BoxInstance {
                rect: [100.0, 100.0, 300.0, 200.0],
                color: [0.18, 0.55, 1.0, 1.0],
                radii: [16.0; 4],
            },
            // Semi-transparent orange rectangle with asymmetric radii (logical).
            BoxInstance {
                rect: [500.0, 180.0, 200.0, 150.0],
                color: [1.0, 0.42, 0.18, 0.9],
                radii: [0.0, 32.0, 0.0, 32.0],
            },
            // White circle: radius == half of the square side (40 logical px).
            // With all four corner radii equal to half_width == half_height the
            // rounded-rect SDF degenerates to a perfect circle.
            BoxInstance {
                rect: [340.0, 420.0, 80.0, 80.0],
                color: [1.0, 1.0, 1.0, 0.85],
                radii: [40.0; 4],
            },
        ];

        self.texts = vec![
            // Label over the blue rounded rectangle.
            TextLine {
                x: 110.0,
                y: 110.0,
                text: "Byard — Phase 1".to_string(),
                font_size: 20.0,
                color: [1.0, 1.0, 1.0, 1.0],
                // Static label: never mutated after this first frame, so it
                // never needs `dirty: true` on a later tick. `prepare`
                // shapes it once regardless, since it's new to the cache.
                dirty: false,
            },
            // Smaller label over the orange rectangle.
            TextLine {
                x: 510.0,
                y: 190.0,
                text: "TextGlyph".to_string(),
                font_size: 14.0,
                color: [0.1, 0.05, 0.0, 1.0],
                dirty: false,
            },
        ];

        self.engine = Some(engine);
        Ok(())
    }

    fn on_resize(&mut self, size: WindowSize) {
        // Engine reconfigures the surface (physical px) and updates the
        // viewport uniform (logical px) in one call.
        if let Some(engine) = self.engine.as_mut() {
            engine.on_resize(size.width, size.height, size.scale_factor);
        }
    }

    fn on_redraw(&mut self) -> Result<(), ByardError> {
        if let Some(engine) = self.engine.as_mut() {
            engine.render_frame(&self.instances, &self.texts)?;
        }
        Ok(())
    }
}
