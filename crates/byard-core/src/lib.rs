//! # byard-core
//!
//! The engine core of the Byard UI framework.
//!
//! This crate contains the four subsystems that compose the rendering engine:
//!
//! - [`evaluator`] — Reactive state (`Signal<T>`), per-view memory arenas (`ViewArena`),
//!   and dirty-flag collection.
//! - [`atlas`] — Layout computation via Taffy and spatial hit-testing via a hash grid.
//! - [`encoder`] — Multi-pipeline `wgpu` command dispatch (`SolidBox`, `DecoratedBox`,
//!   `TextGlyph`, `TextureSampler`).
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
pub mod evaluator;
pub mod frame;
pub mod relay;

use std::fmt;

/// Errors produced by the Byard engine.
///
/// This enum is `#[non_exhaustive]` — new variants may be added in future
/// releases without breaking downstream code.
#[non_exhaustive]
#[derive(Debug)]
pub enum ByardError {
    PipelineCompilation { pipeline: String, reason: String },
    UnsupportedBackend,
    /// An error from the Atlas subsystem (layout, hit-testing).
    Layout(crate::atlas::AtlasError),
}

impl fmt::Display for ByardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PipelineCompilation { pipeline, reason } => {
                write!(f, "pipeline '{pipeline}' failed to compile: {reason}")
            }
            Self::UnsupportedBackend => write!(f, "no compatible wgpu backend found"),
            Self::Layout(e) => write!(f, "{e}"),
        }
    }
}

impl From<crate::atlas::AtlasError> for ByardError {
    fn from(e: crate::atlas::AtlasError) -> Self {
        Self::Layout(e)
    }
}

impl std::error::Error for ByardError {}
