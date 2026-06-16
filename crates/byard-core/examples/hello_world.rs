//! Phase 1 visual verification — renders two `SolidBox` instances to a window.
//!
//! This example is intentionally minimal: it wires `winit` event loop to the
//! `byard-core` [`Engine`] API and draws a static scene to confirm that the
//! `SolidBox` pipeline, SDF border-radius shader, and NDC transform all produce
//! correct output before any higher-level subsystems exist.
//!
//! Run with:
//! ```sh
//! cargo run --example hello_world
//! ```

use std::sync::Arc;

use byard_core::{BoxInstance, Engine};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    // Wait: sleep until the OS sends an event; no busy-loop for a static scene.
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop
        .run_app(&mut App::default())
        .expect("event loop error");
}

/// Top-level application state.
///
/// `state` is `None` until `resumed()` fires (the first chance to create a
/// window on all platforms, including Android and iOS).
#[derive(Default)]
struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    engine: Engine,
    /// Static scene: three shapes exercising radii, colour, and transparency.
    ///
    /// All coordinates are in **logical pixels** (density-independent units).
    /// The engine converts to physical pixels internally using the window's
    /// DPI scale factor, so this list renders identically on Retina and
    /// non-HiDPI displays.
    instances: Vec<BoxInstance>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Byard — Phase 1 Verification")
                        .with_inner_size(winit::dpi::LogicalSize::new(800_u32, 600_u32)),
                )
                .expect("failed to create window"),
        );

        // wgpu 29: InstanceDescriptor::default() removed.
        // `new_without_display_handle_from_env` is the direct equivalent:
        // reads the WGPU_BACKEND env var and works on Metal/Dx12/Vulkan.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        // create_surface requires the window to be Arc'd ('static bound).
        // No `unsafe` needed with wgpu 24+ + winit 0.30.
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create wgpu surface");

        let phys_size = window.inner_size();
        let scale = window.scale_factor();
        let engine = pollster::block_on(Engine::init(
            &instance,
            surface,
            phys_size.width,
            phys_size.height,
            scale,
        ))
        .expect("Byard engine initialisation failed");

        let instances = vec![
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

        // Request the first frame now that the engine is ready.
        window.request_redraw();

        self.state = Some(AppState {
            window,
            engine,
            instances,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(new_size) => {
                // Engine reconfigures the surface (physical px) and updates the
                // viewport uniform (logical px) in one call.
                let scale = state.window.scale_factor();
                state
                    .engine
                    .on_resize(new_size.width, new_size.height, scale);
                state.window.request_redraw();
            }

            // Fired when the window moves between displays with different DPI
            // (e.g. from a Retina to a non-HiDPI external monitor). The new
            // physical size after the factor change is supplied by winit.
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let new_phys = state.window.inner_size();
                state
                    .engine
                    .on_resize(new_phys.width, new_phys.height, scale_factor);
                state.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                state
                    .engine
                    .render_frame(&state.instances)
                    .expect("render_frame failed");
            }

            _ => {}
        }
    }
}
