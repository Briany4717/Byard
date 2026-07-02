//! # Platform abstraction
//!
//! Defines [`PlatformHost`] — the only point of contact between the engine
//! core and a concrete windowing backend (`winit`, a future mobile host,
//! the headless *Coreolis* embedding, etc.), per RFC-0001 §6.
//!
//! `byard-core` has zero direct references to `winit`, Wayland, Win32, or
//! any other OS primitive. This module does not either: [`WindowSize`] is
//! expressed in plain `u32`/`f64`, and the only non-`std` types in the trait
//! signature ([`wgpu::Instance`], [`wgpu::Surface`]) are already a dependency
//! of `byard-core` for rendering, not for windowing. A concrete host (e.g.
//! `byard-platform`'s `WinitHost`) owns the actual window and event loop,
//! and calls into a [`PlatformHost`] implementation in response to OS events:
//!
//! ```text
//!   Host (WinitHost / etc.)                     Application code
//!   ──────────────────────                      ──────────────────────────────
//!   window + surface created  ──────────────►   PlatformHost::on_resume
//!   resize / DPI change       ──────────────►   PlatformHost::on_resize
//!   RedrawRequested            ──────────────►   PlatformHost::on_redraw
//!   mouse button press/release ──────────────►   PlatformHost::on_pointer_input
//!   close button / Cmd+Q       ──────────────►   PlatformHost::on_close_requested
//! ```
//!
//! The application implements [`PlatformHost`] once and it works unchanged
//! against any host. The host crate is the only place `winit` (or whatever
//! the next backend is) appears in the dependency tree.

use crate::ByardError;

/// Physical window dimensions plus the OS DPI scale factor.
///
/// This is the minimal data every [`PlatformHost`] callback needs about the
/// window, expressed without depending on any windowing crate's types.
/// `width`/`height` are in **physical pixels** (e.g. winit's
/// `window.inner_size()`); `scale_factor` is the OS-reported DPI scale (e.g.
/// winit's `window.scale_factor()`) — see [`crate::engine`]'s module docs for
/// why the engine needs both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSize {
    /// Surface width in physical pixels.
    pub width: u32,
    /// Surface height in physical pixels.
    pub height: u32,
    /// OS DPI scale factor (`1.0` on non-`HiDPI`, `2.0` on Retina, etc.).
    pub scale_factor: f64,
}

/// A mouse/pointer button, expressed without depending on any windowing
/// crate's types — mirrors the variants `winit::event::MouseButton` exposes
/// today, since that is the only host that currently exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerButton {
    /// The primary (usually left) button.
    Left,
    /// The secondary (usually right) button.
    Right,
    /// The middle/wheel button.
    Middle,
    /// A "navigate back" side button, where the device has one.
    Back,
    /// A "navigate forward" side button, where the device has one.
    Forward,
    /// Any other, vendor-specific button, identified by its raw code.
    Other(u16),
}

/// A pointer button's transition state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerState {
    /// The button was just pressed down.
    Pressed,
    /// The button was just released.
    Released,
}

/// The event kinds the Phase-2 router models.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EventKind {
    /// A pointer press.
    PointerDown,
    /// A pointer release.
    PointerUp,
    /// A qualifying tap (down+up within thresholds).
    Tap,
    /// Continuous pointer movement.
    PointerMove,
    /// Continuous scroll.
    Scroll,
    /// Continuous wheel.
    Wheel,
    /// A value change from a value-carrying intrinsic.
    Change,
    /// A keyboard key press; key name in `InputEvent.payload` as `InputPayload::Key`.
    KeyDown,
    /// A keyboard key release; key name in `InputEvent.payload` as `InputPayload::Key`.
    KeyUp,
    /// Printable text input (character key or IME commit); text in `InputEvent.payload` as `InputPayload::Key`.
    TextInput,
    // ── M24: remaining event catalog ─────────────────────────────────────
    /// Cursor entered an element's hit rect (synthesized by the router).
    PointerEnter,
    /// Cursor left an element's hit rect (synthesized by the router).
    PointerExit,
    /// Pointer hovering inside an element (fires on `PointerMove` while inside, no button).
    Hover,
    /// Press held > 500 ms without significant movement.
    LongPress,
    /// Two qualifying taps within `DOUBLE_TAP_MS` ms.
    DoubleTap,
    /// Secondary (right) button tap.
    Secondary,
}

impl EventKind {
    /// Whether this is a continuous (coalescible) event.
    #[must_use]
    pub fn is_continuous(self) -> bool {
        matches!(self, Self::PointerMove | Self::Scroll | Self::Wheel)
    }
}

/// A simple payload for input events.
#[derive(Clone, Debug, PartialEq)]
pub enum InputPayload {
    /// A string payload (e.g. text input value).
    Str(String),
    /// A boolean payload (e.g. toggle state).
    Bool(bool),
    /// A float payload (e.g. slider position).
    Float(f32),
    /// A key name or printable text (keyboard events, M17).
    Key(String),
}

/// A normalized, `Send`-able input event produced by the platform thread.
#[derive(Clone, Debug, PartialEq)]
pub struct InputEvent {
    /// The event kind.
    pub kind: EventKind,
    /// Absolute cursor position (logical px).
    pub pos: (f32, f32),
    /// Incremental delta for continuous events.
    pub delta: (f32, f32),
    /// The new value for a `Change` event (write-back payload).
    pub payload: Option<InputPayload>,
    /// Event time in milliseconds (for the tap interval).
    pub time_ms: u64,
}

/// Application hooks driven by a concrete platform host.
///
/// Implement this trait once per application; a host (e.g. `WinitHost` from
/// `byard-platform`) owns the window and event loop and calls these methods
/// in response to OS events. Neither side needs to know the other's concrete
/// type, which is what keeps `winit` (or any future backend) out of
/// `byard-core`'s dependency tree (RFC-0001 §6).
///
/// ## Scope: this trait drives [`Engine`](crate::engine::Engine)
///
/// `PlatformHost` owns an `Engine` and calls it in response to OS events.
/// [`on_resume`](PlatformHost::on_resume) initialises the engine and calls
/// [`Engine::start_logic`](crate::engine::Engine::start_logic) to spawn the
/// logic thread; [`on_redraw`](PlatformHost::on_redraw) calls
/// [`Engine::render_latest`](crate::engine::Engine::render_latest), which
/// reads the latest [`RenderFrame`](crate::frame::RenderFrame) from the
/// engine's [`Relay`](crate::relay::Relay) — the logic thread never blocks
/// the render thread, per RFC-0001 §5.
pub trait PlatformHost {
    /// Called once, after the host has created its window and `wgpu`
    /// surface, before the first redraw. Implementations typically call
    /// [`Engine::init`](crate::engine::Engine::init) here (via
    /// `pollster::block_on` or similar, since this method is synchronous)
    /// and store the resulting `Engine`.
    ///
    /// `instance` is borrowed only for the duration of this call — adapter
    /// and device creation must happen before it returns; `surface` is
    /// moved in because the resulting `Engine` owns it for its lifetime.
    ///
    /// `waker` is a frame-waker tied to the host's event loop (see
    /// [`Engine::set_frame_waker`](crate::engine::Engine::set_frame_waker)).
    /// An event-driven (`Wait`-mode) host should install it on the `Engine` it
    /// creates here — `engine.set_frame_waker(waker)` — so input results are
    /// presented as soon as the logic thread publishes them. A
    /// continuously-redrawing (`Poll`) host may ignore it.
    ///
    /// # Errors
    ///
    /// Returns whatever [`ByardError`] engine initialisation produces (see
    /// [`Engine::init`](crate::engine::Engine::init)'s own error
    /// documentation). The host should treat this as fatal startup failure.
    fn on_resume(
        &mut self,
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        size: WindowSize,
        waker: crate::relay::FrameWaker,
    ) -> Result<(), ByardError>;

    /// Called whenever the window is resized or the OS DPI scale changes.
    ///
    /// Implementations typically forward this directly to
    /// [`Engine::on_resize`](crate::engine::Engine::on_resize).
    fn on_resize(&mut self, size: WindowSize);

    /// Called when the host requests a redraw (e.g. winit's
    /// `WindowEvent::RedrawRequested`).
    ///
    /// Implementations typically call
    /// [`Engine::render_latest`](crate::engine::Engine::render_latest) here.
    ///
    /// # Errors
    ///
    /// Returns whatever [`ByardError`] frame rendering produces. Transient
    /// surface loss is already handled internally and never reaches this
    /// method as an error — only unrecoverable surface errors do.
    fn on_redraw(&mut self) -> Result<(), ByardError>;

    /// Called when the user requests the window close (close button,
    /// Cmd+Q, etc.). Returning `true` tells the host it is safe to exit its
    /// event loop; returning `false` keeps the window open (e.g. to show an
    /// unsaved-changes prompt in a future application).
    ///
    /// Defaults to always permitting the close, since most applications —
    /// including every example in this crate today — have nothing to guard.
    fn on_close_requested(&mut self) -> bool {
        true
    }

    /// Called when a pointer (mouse) button changes state over the window at coordinates (x, y).
    ///
    /// Defaults to a no-op.
    fn on_pointer_input(&mut self, _button: PointerButton, _state: PointerState, _x: f32, _y: f32) {
    }

    /// Called when the cursor moves to coordinates (x, y).
    ///
    /// Defaults to a no-op.
    fn on_cursor_moved(&mut self, _x: f32, _y: f32) {}

    /// Called when a keyboard key is pressed or released.
    ///
    /// `key` is the logical key name (e.g. `"a"`, `"Enter"`, `"Backspace"`,
    /// `"Tab"`). `pressed` is `true` for key-down, `false` for key-up.
    ///
    /// Defaults to a no-op.
    fn on_key(&mut self, _key: &str, _pressed: bool) {}

    /// Called when printable text is committed (character keys, IME commit).
    ///
    /// `text` is the committed string (usually one character but may be more
    /// for IME). Defaults to a no-op.
    fn on_text(&mut self, _text: &str) {}

    /// Called on a trackpad-style scroll gesture at cursor position `(x, y)`,
    /// with `(dx, dy)` the scroll delta in logical pixels (RFC-0012 §A loose
    /// end: `scroll`'s hardware origin).
    ///
    /// Defaults to a no-op.
    fn on_scroll(&mut self, _dx: f32, _dy: f32, _x: f32, _y: f32) {}

    /// Called on a physical mouse-wheel tick at cursor position `(x, y)`,
    /// with `(dx, dy)` the wheel delta (RFC-0012 §A loose end: `wheel`'s
    /// hardware origin).
    ///
    /// Defaults to a no-op.
    fn on_wheel(&mut self, _dx: f32, _dy: f32, _x: f32, _y: f32) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal `PlatformHost` that exercises only the trait's default
    /// methods, so the default `on_close_requested` body is covered without
    /// needing a real `wgpu`/window environment.
    struct DefaultsOnlyHost;

    impl PlatformHost for DefaultsOnlyHost {
        fn on_resume(
            &mut self,
            _instance: &wgpu::Instance,
            _surface: wgpu::Surface<'static>,
            _size: WindowSize,
            _waker: crate::relay::FrameWaker,
        ) -> Result<(), ByardError> {
            Ok(())
        }

        fn on_resize(&mut self, _size: WindowSize) {}

        fn on_redraw(&mut self) -> Result<(), ByardError> {
            Ok(())
        }
    }

    #[test]
    fn on_close_requested_defaults_to_true() {
        let mut host = DefaultsOnlyHost;
        assert!(host.on_close_requested());
    }

    #[test]
    fn on_pointer_input_default_is_a_no_op() {
        let mut host = DefaultsOnlyHost;
        // Must not panic for any button/state combination.
        host.on_pointer_input(PointerButton::Left, PointerState::Pressed, 0.0, 0.0);
        host.on_pointer_input(PointerButton::Other(7), PointerState::Released, 10.0, 20.0);
        host.on_cursor_moved(15.0, 25.0);
    }

    #[test]
    fn on_scroll_and_on_wheel_defaults_are_no_ops() {
        let mut host = DefaultsOnlyHost;
        // Must not panic for any delta/position combination.
        host.on_scroll(1.0, -2.0, 10.0, 20.0);
        host.on_wheel(0.0, 3.0, 30.0, 40.0);
    }

    #[test]
    fn pointer_button_and_state_are_copy_clone_eq_and_debug() {
        let a = PointerButton::Left;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_ne!(PointerButton::Left, PointerButton::Right);
        assert_eq!(PointerButton::Other(3), PointerButton::Other(3));
        assert_ne!(PointerButton::Other(3), PointerButton::Other(4));
        let _ = format!("{a:?}");

        let pressed = PointerState::Pressed;
        assert_ne!(pressed, PointerState::Released);
        let _ = format!("{pressed:?}");
    }

    #[test]
    fn window_size_is_copy_clone_eq_and_debug() {
        let a = WindowSize {
            width: 800,
            height: 600,
            scale_factor: 2.0,
        };
        let b = a; // Copy
        assert_eq!(a, b);

        let c = WindowSize { width: 801, ..a };
        assert_ne!(a, c);

        // Debug must not panic.
        let _ = format!("{a:?}");
    }
}
