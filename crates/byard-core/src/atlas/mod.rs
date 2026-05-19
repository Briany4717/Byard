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
//! [`crate::frame::Rect`] and reacts to dirty-flag notifications produced
//! by the Evaluator subsystem via [`crate::frame::TargetId`] +
//! [`crate::frame::TargetKind::AtlasNode`].
//!
//! # State machine
//!
//! [`LayoutAtlas`] enforces a strict lifecycle with two states:
//!
//! 1. **Building** — nodes can be added (`add_leaf`, `add_container`) and
//!    the root can be set. Querying resolved geometry or marking nodes
//!    dirty panics.
//! 2. **Computed** — `compute(viewport)` transitions here. Resolved
//!    geometry is accessible via `resolved_rect` and `populate_frame`.
//!    Dirty subtrees can be re-laid out incrementally via
//!    `mark_dirty_all` + `recompute_dirty`. Adding or modifying nodes
//!    panics until `clear` is called.
//!
//! ## Transitions
//!
//! - `compute(viewport)` — Building → Computed.
//! - `clear()` — Computed → Building. Preserves internal capacity and
//!   increments the view generation so any
//!   [`TargetId`](crate::frame::TargetId)s from the previous view are
//!   silently rejected by future `mark_dirty_all` calls.
//!
//! Per RFC-0001 §4.1, `compute` is called exactly once per frame at the
//! end of the mutation phase, then `recompute_dirty` is called on
//! subsequent frames whenever the Evaluator reports dirty targets.
//!
//! # Cross-subsystem flow
//!
//! The Atlas is one consumer of the broadcast `TargetId` stream produced
//! by the Logic thread:
//!
//! ```text
//! signals mutate  →  EvaluatorTick::collect_dirty()  →  Vec<TargetId>
//!                                                       │
//!                            ┌──────────────────────────┼──────────────┐
//!                            ▼                          ▼              ▼
//!                  atlas.mark_dirty_all(...)   encoder.mark_dirty_all(...)   ...
//! ```
//!
//! Each subsystem filters the broadcast by [`TargetKind`](crate::frame::TargetKind)
//! and ignores foreign or stale entries. See [`LayoutAtlas::mark_dirty_all`]
//! for the filtering rules.

pub mod layout;

pub use layout::{AtlasError, AtlasNodeId, ContainerStyle, LayoutAtlas, LeafSize};
