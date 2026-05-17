//! # Frame
//!
//! Shared data types for cross-subsystem communication.
//!
//! This module defines [`RenderFrame`] and the primitive types that flow between
//! the evaluator, atlas, encoder, and relay subsystems. It is the **only** module
//! that all subsystems may depend on.
//!
//! ```text
//! encoder  ──┐
//! atlas    ──┤─→  frame  ←─  relay
//! evaluator ─┘
//! ```
//!
//! Adding a dependency from one subsystem to another (e.g. `encoder` importing
//! from `evaluator`) is a design defect. If data needs to cross that boundary,
//! it must be modelled as a type in this module.

/// An immutable snapshot of all render primitives for a single frame.
///
/// Produced by the logic thread (evaluator + atlas) and consumed by the render
/// thread (encoder) via an atomic pointer swap managed by the relay.
///
/// This type is intentionally cheap to clone — it will be wrapped in an `Arc`
/// for the double-buffer exchange.
#[derive(Debug, Default)]
pub struct RenderFrame {
    // Primitives will be added here as the subsystems are implemented.
}
