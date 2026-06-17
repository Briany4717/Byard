//! # byard-core
//!
//! The engine core of the Byard UI framework.
//!
//! This crate contains the four subsystems that compose the rendering engine:
//!
//! - [`evaluator`] — Reactive state (`Signal<T>`), per-view memory arenas (`ViewArena`),
//!   and dirty-flag collection.
//! - [`atlas`] — Layout computation via Taffy and spatial hit-testing via a hash grid.
//! - [`encoder`] — Multi-pipeline `wgpu` command dispatch. `SolidBox` and
//!   `TextGlyph` are implemented today; `DecoratedBox` and `TextureSampler`
//!   are planned (RFC-0001 §3.1) but not yet built.
//! - [`relay`] — Thread management, double-buffered frame swap, and async I/O pool.
//!
//! Cross-subsystem communication goes exclusively through the types defined in
//! [`frame`]. No subsystem module imports from another subsystem directly.
//!
//! ```text
//! encoder  ──┐
//! atlas    ──┤─→  frame  ←─  relay
//! evaluator ─┘
//! ```

pub mod atlas;
pub mod encoder;
pub mod engine;
pub mod evaluator;
pub mod frame;
pub mod platform;
pub mod relay;

pub use encoder::BoxInstance;
pub use encoder::text_glyph::TextLine;
pub use engine::Engine;
pub use platform::{PlatformHost, PointerButton, PointerState, WindowSize};

use std::fmt;

/// Errors produced by the Byard engine.
///
/// This enum is `#[non_exhaustive]` — new variants may be added in future
/// releases without breaking downstream code.
#[non_exhaustive]
#[derive(Debug)]
pub enum ByardError {
    /// A render pipeline failed to compile during initialisation.
    PipelineCompilation {
        /// Name of the pipeline that failed (e.g. `"SolidBox"`).
        pipeline: String,
        /// The underlying error message from `wgpu`.
        reason: String,
    },

    /// The GPU backend does not meet Byard's minimum requirements.
    UnsupportedBackend,

    /// An error from the Atlas subsystem (layout, hit-testing).
    Layout(crate::atlas::AtlasError),

    /// An unrecoverable error occurred while acquiring or presenting a surface texture.
    ///
    /// Transient surface losses (e.g. window minimise/restore) are handled
    /// transparently by [`Engine::render_latest`] and never produce this variant.
    RenderSurface(String),

    /// [`glyphon`] failed to upload shaped glyphs to the GPU atlas.
    ///
    /// The inner string is the original [`glyphon::PrepareError`] message.
    TextPrepare(String),

    /// [`glyphon`] failed to record text draw calls into the render pass.
    ///
    /// The inner string is the original [`glyphon::RenderError`] message.
    TextRender(String),

    /// The OS refused to spawn a thread (e.g. [`relay::Relay`]'s logic thread).
    ///
    /// The inner string is the original [`std::io::Error`] message. Per
    /// RFC-0001 §8, thread-spawn failure is a recoverable condition, not a
    /// panic — the caller decides whether to retry or abort startup.
    ThreadSpawn(String),

    /// The async I/O Tokio runtime failed to initialise.
    ///
    /// The inner string is the original [`std::io::Error`] message produced
    /// by [`tokio::runtime::Builder::build`].
    RuntimeCreation(String),

    /// A [`platform::PlatformHost`] implementation failed to create its
    /// window, `wgpu` surface, or event loop.
    ///
    /// The inner string is the original platform-specific error message
    /// (e.g. from `winit::event_loop::EventLoop::new` or
    /// `wgpu::Instance::create_surface`). `byard-core` never constructs this
    /// variant itself — it exists so host crates (e.g. `byard-platform`)
    /// have a `ByardError` to return from [`platform::PlatformHost::on_resume`]
    /// without inventing their own error type.
    Platform(String),
}

impl fmt::Display for ByardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PipelineCompilation { pipeline, reason } => {
                write!(f, "pipeline '{pipeline}' failed to compile: {reason}")
            }
            Self::UnsupportedBackend => write!(f, "no compatible wgpu backend found"),
            Self::Layout(e) => write!(f, "{e}"),
            Self::RenderSurface(msg) => write!(f, "surface error: {msg}"),
            Self::TextPrepare(msg) => write!(f, "text prepare error: {msg}"),
            Self::TextRender(msg) => write!(f, "text render error: {msg}"),
            Self::ThreadSpawn(msg) => write!(f, "failed to spawn thread: {msg}"),
            Self::RuntimeCreation(msg) => write!(f, "failed to create async runtime: {msg}"),
            Self::Platform(msg) => write!(f, "platform host error: {msg}"),
        }
    }
}

impl From<crate::atlas::AtlasError> for ByardError {
    fn from(e: crate::atlas::AtlasError) -> Self {
        Self::Layout(e)
    }
}

impl std::error::Error for ByardError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Layout(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byard_error_layout_chains_source() {
        use crate::atlas::AtlasError;
        use std::error::Error;

        let inner_msg = "backend exploded".to_string();
        let atlas_err = AtlasError::Backend(inner_msg);
        let byard_err: ByardError = atlas_err.into();

        assert!(
            byard_err.source().is_some(),
            "Layout variant must expose source"
        );
        assert_eq!(
            byard_err.to_string(),
            "layout backend error: backend exploded"
        );
    }

    #[test]
    fn byard_error_pipeline_has_no_source() {
        use std::error::Error;

        let err = ByardError::PipelineCompilation {
            pipeline: "SolidBox".to_string(),
            reason: "bad shader".to_string(),
        };
        assert!(err.source().is_none());
    }

    #[test]
    fn byard_error_unsupported_backend_display_and_no_source() {
        use std::error::Error;

        let err = ByardError::UnsupportedBackend;
        assert_eq!(err.to_string(), "no compatible wgpu backend found");
        assert!(err.source().is_none());
    }

    #[test]
    fn byard_error_render_surface_display_and_no_source() {
        use std::error::Error;

        let msg = "out of memory on present".to_string();
        let err = ByardError::RenderSurface(msg.clone());
        assert_eq!(err.to_string(), format!("surface error: {msg}"));
        assert!(err.source().is_none());
    }

    #[test]
    fn byard_error_pipeline_display() {
        let err = ByardError::PipelineCompilation {
            pipeline: "SolidBox".to_string(),
            reason: "bad shader".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "pipeline 'SolidBox' failed to compile: bad shader"
        );
    }

    #[test]
    fn byard_error_is_debug() {
        // Debug must not panic for any variant.
        let _ = format!("{:?}", ByardError::UnsupportedBackend);
        let _ = format!(
            "{:?}",
            ByardError::PipelineCompilation {
                pipeline: "x".into(),
                reason: "y".into(),
            }
        );
        let _ = format!("{:?}", ByardError::RenderSurface("z".into()));
    }
}
