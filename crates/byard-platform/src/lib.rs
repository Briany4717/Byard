//! # byard-platform
//!
//! `winit`-backed implementation of [`byard_core::PlatformHost`] (RFC-0001 §6).
//!
//! This is the only crate in the workspace allowed to depend on `winit`.
//! [`WinitHost`] owns the actual window and event loop; it translates
//! `winit`'s `WindowEvent`s into [`byard_core::PlatformHost`] calls and never
//! leaks a `winit` type through its own public API (`WinitHost::run` takes a
//! generic `PlatformHost` and returns a plain `Result<(), ByardError>`), so
//! downstream crates can depend on `byard-platform` without ever importing
//! `winit` themselves.

use std::sync::Arc;

use byard_core::{ByardError, PlatformHost, PointerButton, PointerState, WindowSize};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// A `winit`-backed [`PlatformHost`] driver.
///
/// Construct with [`WinitHost::new`] and hand it a [`PlatformHost`]
/// implementation via [`WinitHost::run`], which blocks until the window is
/// closed (or a [`PlatformHost`] callback returns an error).
pub struct WinitHost {
    title: String,
    width: u32,
    height: u32,
}

impl WinitHost {
    /// Creates a host that will open a window with the given title and
    /// initial size (in logical pixels).
    pub fn new(title: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            title: title.into(),
            width,
            height,
        }
    }

    /// Runs the `winit` event loop until the window closes, dispatching
    /// every relevant OS event to `host`.
    ///
    /// This call blocks the calling thread for the lifetime of the window —
    /// per RFC-0001 §6, this is the render thread; it never shares mutable
    /// state with a logic thread, so blocking here is exactly what the
    /// concurrency model expects.
    ///
    /// # Errors
    ///
    /// Returns [`ByardError::Platform`] if the event loop or window itself
    /// fails to initialise, or whatever [`ByardError`] a [`PlatformHost`]
    /// callback returns (e.g. [`Engine::init`](byard_core::engine::Engine::init)
    /// failing inside [`PlatformHost::on_resume`]) — `winit`'s
    /// `ApplicationHandler` callbacks are themselves infallible, so such
    /// errors are captured internally and surfaced here once the loop exits.
    pub fn run<H: PlatformHost>(self, host: H) -> Result<(), ByardError> {
        let event_loop = EventLoop::new().map_err(|e| ByardError::Platform(e.to_string()))?;
        // Wait: sleep until the OS sends an event; no busy-loop for a static scene.
        event_loop.set_control_flow(ControlFlow::Wait);

        let mut app = WinitApp {
            host,
            window: None,
            title: self.title,
            width: self.width,
            height: self.height,
            fatal: None,
            cursor_pos: (0.0, 0.0),
        };

        event_loop
            .run_app(&mut app)
            .map_err(|e| ByardError::Platform(e.to_string()))?;

        match app.fatal {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

/// Adapts a [`PlatformHost`] to `winit`'s `ApplicationHandler`.
///
/// `window` is `None` until `resumed()` fires — the first point at which
/// creating a window is valid on every platform `winit` supports, including
/// Android and iOS.
struct WinitApp<H: PlatformHost> {
    host: H,
    window: Option<Arc<Window>>,
    title: String,
    width: u32,
    height: u32,
    /// Set when a [`PlatformHost`] callback returns `Err`, since
    /// `ApplicationHandler`'s methods can't themselves return a `Result`.
    /// `WinitHost::run` checks this after the event loop exits and surfaces
    /// it to its own caller.
    fatal: Option<ByardError>,
    cursor_pos: (f32, f32),
}

impl<H: PlatformHost> WinitApp<H> {
    /// Records a fatal error and asks the event loop to exit on its next
    /// iteration.
    fn fail(&mut self, event_loop: &ActiveEventLoop, err: ByardError) {
        self.fatal = Some(err);
        event_loop.exit();
    }
}

impl<H: PlatformHost> ApplicationHandler for WinitApp<H> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Already initialised — e.g. resumed again after a mobile suspend.
        if self.window.is_some() {
            return;
        }

        let window = match event_loop.create_window(
            Window::default_attributes()
                .with_title(&self.title)
                .with_inner_size(LogicalSize::new(self.width, self.height)),
        ) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.fail(event_loop, ByardError::Platform(e.to_string()));
                return;
            }
        };

        // wgpu 29: InstanceDescriptor::default() removed.
        // `new_without_display_handle_from_env` is the direct equivalent:
        // reads the WGPU_BACKEND env var and works on Metal/Dx12/Vulkan.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        // create_surface requires the window to be Arc'd ('static bound).
        // No `unsafe` needed with wgpu 24+ + winit 0.30.
        let surface = match instance.create_surface(window.clone()) {
            Ok(s) => s,
            Err(e) => {
                self.fail(event_loop, ByardError::Platform(e.to_string()));
                return;
            }
        };

        let size = to_window_size(window.inner_size(), window.scale_factor());

        if let Err(e) = self.host.on_resume(&instance, surface, size) {
            self.fail(event_loop, e);
            return;
        }

        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };

        match event {
            // on_close_requested() == false means "keep the window open";
            // there's nothing else to do, so it falls through to the `_`
            // arm below rather than getting its own empty branch here.
            WindowEvent::CloseRequested if self.host.on_close_requested() => {
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                let size = to_window_size(new_size, window.scale_factor());
                self.host.on_resize(size);
                window.request_redraw();
            }

            // Fired when the window moves between displays with different
            // DPI (e.g. Retina to a non-HiDPI external monitor). The new
            // physical size after the factor change is supplied by winit.
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let size = to_window_size(window.inner_size(), scale_factor);
                self.host.on_resize(size);
                window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                if let Err(e) = self.host.on_redraw() {
                    self.fail(event_loop, e);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let scale_factor = window.scale_factor();
                let logical = position.to_logical::<f64>(scale_factor);
                #[allow(clippy::cast_possible_truncation)]
                let x = logical.x as f32;
                #[allow(clippy::cast_possible_truncation)]
                let y = logical.y as f32;
                self.cursor_pos = (x, y);
                self.host.on_cursor_moved(x, y);
                window.request_redraw();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let (x, y) = self.cursor_pos;
                self.host.on_pointer_input(
                    to_pointer_button(button),
                    to_pointer_state(state),
                    x,
                    y,
                );
                window.request_redraw();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                let key_str = key_to_str(&event.logical_key);
                if !key_str.is_empty() {
                    self.host.on_key(&key_str, pressed);
                }
                // Fire text input only for printable characters on press
                if pressed {
                    if let Key::Character(s) = &event.logical_key {
                        self.host.on_text(s.as_str());
                    }
                }
                window.request_redraw();
            }

            _ => {}
        }
    }
}

/// Converts a `winit` physical size + DPI scale factor into a
/// [`WindowSize`].
///
/// Extracted as a pure function so the conversion is testable without a real
/// `winit` event loop or display — neither of which exist in CI.
fn to_window_size(size: PhysicalSize<u32>, scale_factor: f64) -> WindowSize {
    WindowSize {
        width: size.width,
        height: size.height,
        scale_factor,
    }
}

/// Converts a `winit` mouse button into the plain, windowing-crate-agnostic
/// [`PointerButton`] that crosses into `byard-core`.
///
/// Extracted as a pure function for the same reason as [`to_window_size`] —
/// testable without a real event loop.
fn to_pointer_button(button: MouseButton) -> PointerButton {
    match button {
        MouseButton::Left => PointerButton::Left,
        MouseButton::Right => PointerButton::Right,
        MouseButton::Middle => PointerButton::Middle,
        MouseButton::Back => PointerButton::Back,
        MouseButton::Forward => PointerButton::Forward,
        MouseButton::Other(code) => PointerButton::Other(code),
    }
}

/// Converts a `winit` element state into the plain [`PointerState`] that
/// crosses into `byard-core`.
fn to_pointer_state(state: ElementState) -> PointerState {
    match state {
        ElementState::Pressed => PointerState::Pressed,
        ElementState::Released => PointerState::Released,
    }
}

/// Converts a `winit` logical key to a string key name.
///
/// Returns an empty string for keys we don't model (so the caller can skip).
fn key_to_str(key: &Key) -> String {
    match key {
        Key::Character(s) => s.to_string(),
        Key::Named(named) => match named {
            NamedKey::Backspace => "Backspace".to_string(),
            NamedKey::Delete => "Delete".to_string(),
            NamedKey::Enter => "Enter".to_string(),
            NamedKey::Tab => "Tab".to_string(),
            NamedKey::Escape => "Escape".to_string(),
            NamedKey::ArrowLeft => "ArrowLeft".to_string(),
            NamedKey::ArrowRight => "ArrowRight".to_string(),
            NamedKey::ArrowUp => "ArrowUp".to_string(),
            NamedKey::ArrowDown => "ArrowDown".to_string(),
            NamedKey::Home => "Home".to_string(),
            NamedKey::End => "End".to_string(),
            _ => String::new(),
        },
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_window_size_carries_fields_through_unchanged() {
        let phys = PhysicalSize::new(1024_u32, 768_u32);
        let size = to_window_size(phys, 2.0);

        assert_eq!(size.width, 1024);
        assert_eq!(size.height, 768);
        assert!((size.scale_factor - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn to_window_size_handles_non_hidpi_scale() {
        let phys = PhysicalSize::new(800_u32, 600_u32);
        let size = to_window_size(phys, 1.0);

        assert_eq!(size.width, 800);
        assert_eq!(size.height, 600);
        assert!((size.scale_factor - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn to_pointer_button_maps_every_winit_variant() {
        assert_eq!(to_pointer_button(MouseButton::Left), PointerButton::Left);
        assert_eq!(to_pointer_button(MouseButton::Right), PointerButton::Right);
        assert_eq!(
            to_pointer_button(MouseButton::Middle),
            PointerButton::Middle
        );
        assert_eq!(to_pointer_button(MouseButton::Back), PointerButton::Back);
        assert_eq!(
            to_pointer_button(MouseButton::Forward),
            PointerButton::Forward
        );
        assert_eq!(
            to_pointer_button(MouseButton::Other(9)),
            PointerButton::Other(9)
        );
    }

    #[test]
    fn to_pointer_state_maps_both_winit_variants() {
        assert_eq!(
            to_pointer_state(ElementState::Pressed),
            PointerState::Pressed
        );
        assert_eq!(
            to_pointer_state(ElementState::Released),
            PointerState::Released
        );
    }
}
