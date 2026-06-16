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

/// Application hooks driven by a concrete platform host.
///
/// Implement this trait once per application; a host (e.g. `WinitHost` from
/// `byard-platform`) owns the window and event loop and calls these methods
/// in response to OS events. Neither side needs to know the other's concrete
/// type, which is what keeps `winit` (or any future backend) out of
/// `byard-core`'s dependency tree (RFC-0001 §6).
///
/// ## Scope: this trait drives an [`Engine`](crate::engine::Engine) directly
///
/// `PlatformHost` calls an `Engine` the same way the original Phase 1
/// example did by hand — it does not route frames through
/// [`crate::relay::Relay`]. `Relay`'s own module documentation already
/// states that wiring it into `Engine` is deliberately deferred until the
/// Atlas actually populates frames on a logic thread; until that lands,
/// having `PlatformHost` go through `Relay` would just be speculative glue
/// with no producer behind it. A future issue can swap what's behind
/// `on_redraw` from a direct `Engine::render_frame` call to
/// `Relay::current()` without changing this trait's shape at all.
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
    /// [`Engine::render_frame`](crate::engine::Engine::render_frame) here.
    ///
    /// # Errors
    ///
    /// Returns whatever [`ByardError`] frame rendering produces. Per
    /// [`Engine::render_frame`]'s own documentation, transient surface loss
    /// is already handled internally and never reaches this method as an
    /// error — only unrecoverable surface errors do.
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
