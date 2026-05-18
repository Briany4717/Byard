//! # Atlas
//!
//! Layout computation and spatial hit-testing.
//!
//! This subsystem owns:
//!
//! - **Taffy integration** — All layout is delegated to
//!   [`taffy`](https://github.com/DioxusLabs/taffy). The engine never
//!   computes box geometry itself. [`LayoutAtlas`] initialises the Taffy
//!   tree, feeds it node constraints, and reads back resolved rectangles.
//!
//! - **Spatial hash grid** *(future sub-issue)* — A partitioned data
//!   structure that indexes a mapping between 2D screen coordinates and
//!   event descriptors.
//!
//! Atlas exposes resolved geometry to the encoder exclusively through
//! [`crate::frame::Rect`].
//!
//! # State machine
//!
//! [`LayoutAtlas`] enforces a strict two-phase lifecycle:
//!
//! 1. **Building** — nodes can be added and modified. `compute` is the
//!    only valid transition out. Querying resolved geometry panics.
//! 2. **Computed** — resolved geometry is accessible. Adding or modifying
//!    nodes panics. `reset` returns to Building, preserving capacity.
//!
//! Per RFC-0001 §4.1, `compute` is called exactly once per frame, after
//! all mutations have been applied by the Logic thread.

pub mod layout;

pub use layout::{AtlasError, AtlasNodeId, ContainerStyle, LayoutAtlas, LeafSize};
