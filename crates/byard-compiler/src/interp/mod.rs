//! The Dev-mode tree-walking interpreter (RFC-0002 §"Dev-mode interpreter",
//! RFC-0004).
//!
//! Submodules are added milestone by milestone: [`env`] (the per-View binding
//! environment + `inject`) lands first; the reactive core, eval driver,
//! intrinsics, style, events, and hot-reload follow.

pub mod env;
pub mod eval;
pub mod events;
pub mod intrinsics;
pub mod reactive;
pub mod reload;
pub mod style;
