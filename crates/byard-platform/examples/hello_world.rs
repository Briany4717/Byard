//! Phase 1 visual verification — renders three `SolidBox` instances and two
//! text labels to a window, one of which is a Signal-driven reactive label.
//!
//! This example is intentionally minimal: it implements [`PlatformHost`] and
//! hands it to [`WinitHost`], drawing a mostly-static scene to confirm that
//! the `SolidBox` pipeline, SDF border-radius shader, and NDC transform all
//! produce correct output, plus a click-to-mutate label that exercises
//! [`Engine`]'s Signal/`EvaluatorTick`/`LayoutAtlas` plumbing in production
//! (RFC-0001's Phase 1 closure criterion). It used to drive `winit`'s
//! `ApplicationHandler` directly inside `byard-core`; it now lives here
//! because `byard-core` has zero direct references to `winit` (RFC-0001 §6)
//! and this is the crate that owns the window.
//!
//! Click anywhere in the window to mutate the reactive label's text via
//! [`Engine::set_label_text`] — the only authored content for that label is
//! the click count; everything about *how* the new text reaches the screen
//! (Signal write → `EvaluatorTick::collect_dirty` → `LayoutAtlas::mark_dirty_all`
//! → `TextLine::dirty`) happens inside [`Engine`], not here.
//!
//! Run with:
//! ```sh
//! cargo run --example hello_world
//! ```

use byard_core::{
    BoxInstance, ByardError, Engine, PlatformHost, PointerButton, PointerState, TextLine,
    WindowSize,
};
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
    ///
    /// This does **not** include the engine's Signal-driven label — that one
    /// lives inside [`Engine`] itself and is folded into the frame by
    /// [`Engine::render_frame`]; this `Vec` is only the lines that genuinely
    /// never change.
    texts: Vec<TextLine>,
    /// Number of left-button clicks seen so far, used to author the
    /// reactive label's text in [`App::on_pointer_input`].
    click_count: u32,
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
            // Smaller label over the orange rectangle. The label over the
            // blue rectangle is intentionally absent here — it's owned by
            // `Engine`'s reactive `ReactiveLabel` and folded into the frame
            // automatically by `Engine::render_frame`.
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

    fn on_pointer_input(&mut self, button: PointerButton, state: PointerState) {
        if button != PointerButton::Left || state != PointerState::Pressed {
            return;
        }

        self.click_count += 1;

        if let Some(engine) = self.engine.as_ref() {
            // Engine handles every step from here: writing the Signal,
            // running `EvaluatorTick::collect_dirty`, marking the Atlas
            // node dirty, and recomputing it — this call only supplies the
            // new text.
            engine.set_label_text(format!("Byard — clicked {} time(s)", self.click_count));
        }
    }
}
