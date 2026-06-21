//! Renders `hello_world.byd` in a real winit/wgpu window.
//!
//! The byld `Interpreter` runs on the logic thread (via `start_logic_from_view`)
//! and publishes a `RenderFrame` each tick. It forwards cursor coordinates and
//! mouse input events directly to the interpreter, which dispatches them to
//! layout-aware elements in the layout tree.
//!
//! Run with:
//! ```sh
//! cargo run --example hello_world_byd -p byard-platform
//! ```

use byard_compiler::interp::eval::{Interpreter, RenderNode};
use byard_compiler::parser::parse;
use byard_core::frame::{RenderFrame, TargetId};
use byard_core::{
    ByardError, Engine, LogicRuntime, PlatformHost, PointerButton, PointerState, WindowSize,
};
use byard_platform::WinitHost;
use std::time::{SystemTime, UNIX_EPOCH};

const SRC: &str = include_str!("../../byard-compiler/examples/hello_world.byd");

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() * 1000 + u64::from(d.subsec_millis()))
}

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

fn main() {
    let host = WinitHost::new("Byard — hello_world.byd", 800, 600);
    host.run(App::default()).expect("event loop error");
}

struct ByldRuntime {
    interp: Interpreter,
    tree: Vec<RenderNode>,
    width_bits: Arc<AtomicU32>,
    height_bits: Arc<AtomicU32>,
}

impl LogicRuntime for ByldRuntime {
    fn evaluate_tick(
        &mut self,
        frame: &mut RenderFrame,
        input_events: &[byard_core::platform::InputEvent],
        _dirty: &[TargetId],
    ) {
        self.interp.dispatch_events(input_events);
        self.interp.tick();
        let w = f32::from_bits(self.width_bits.load(Ordering::Relaxed));
        let h = f32::from_bits(self.height_bits.load(Ordering::Relaxed));
        self.interp.render(&self.tree, frame, w, h);
    }
}

#[derive(Default)]
struct App {
    engine: Option<Engine>,
    width_bits: Option<Arc<AtomicU32>>,
    height_bits: Option<Arc<AtomicU32>>,
}

impl PlatformHost for App {
    fn on_resume(
        &mut self,
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        size: WindowSize,
    ) -> Result<(), ByardError> {
        let parsed = parse(SRC);
        assert!(
            parsed.errors.is_empty(),
            "parse errors: {:?}",
            parsed.errors
        );
        let view = parsed.views[0].clone();

        let mut engine = pollster::block_on(Engine::init(
            instance,
            surface,
            size.width,
            size.height,
            size.scale_factor,
        ))?;

        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let w = size.width as f32 / size.scale_factor as f32;
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let h = size.height as f32 / size.scale_factor as f32;
        let width_bits = Arc::new(AtomicU32::new(w.to_bits()));
        let height_bits = Arc::new(AtomicU32::new(h.to_bits()));

        let w_clone = Arc::clone(&width_bits);
        let h_clone = Arc::clone(&height_bits);

        engine.start_logic_from_view(move |_arena| {
            let mut interp = Interpreter::new();
            let tree = interp.lower_view(&view, &[]);
            interp.tick();
            Box::new(ByldRuntime {
                interp,
                tree,
                width_bits: w_clone,
                height_bits: h_clone,
            })
        })?;

        self.engine = Some(engine);
        self.width_bits = Some(width_bits);
        self.height_bits = Some(height_bits);
        Ok(())
    }

    fn on_resize(&mut self, size: WindowSize) {
        if let Some(e) = self.engine.as_mut() {
            e.on_resize(size.width, size.height, size.scale_factor);
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let w = size.width as f32 / size.scale_factor as f32;
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let h = size.height as f32 / size.scale_factor as f32;
            if let Some(w_bits) = &self.width_bits {
                w_bits.store(w.to_bits(), Ordering::Relaxed);
            }
            if let Some(h_bits) = &self.height_bits {
                h_bits.store(h.to_bits(), Ordering::Relaxed);
            }
        }
    }

    fn on_redraw(&mut self) -> Result<(), ByardError> {
        if let Some(e) = self.engine.as_mut() {
            e.render_latest()?;
        }
        Ok(())
    }

    fn on_pointer_input(&mut self, _button: PointerButton, state: PointerState, x: f32, y: f32) {
        if let Some(engine) = &self.engine {
            let kind = match state {
                PointerState::Pressed => byard_core::platform::EventKind::PointerDown,
                PointerState::Released => byard_core::platform::EventKind::PointerUp,
            };
            engine.push_input(byard_core::platform::InputEvent {
                kind,
                pos: (x, y),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: now_ms(),
            });
        }
    }

    fn on_cursor_moved(&mut self, x: f32, y: f32) {
        if let Some(engine) = &self.engine {
            engine.push_input(byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerMove,
                pos: (x, y),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: now_ms(),
            });
        }
    }
}
