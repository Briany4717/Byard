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
//! # Builder API
//!
//! [`LayoutAtlasBuilder`] sits on top of `add_leaf` /
//! `add_container` / `set_root` to let a multi-level tree be expressed as
//! a single chained expression, instead of one imperative call per node:
//!
//! ```
//! use byard_core::atlas::{ContainerStyle, LayoutAtlas, LayoutAtlasBuilder as B, LeafSize};
//!
//! let mut atlas = LayoutAtlas::new();
//! let root = atlas.build_root(
//!     B::container(ContainerStyle::new(Some(300.0), Some(200.0)), [
//!         B::leaf(LeafSize::new(50.0, 50.0)),
//!         B::container(ContainerStyle::default(), [
//!             B::leaf(LeafSize::new(20.0, 20.0)),
//!         ]),
//!     ]),
//! ).unwrap();
//! # let _ = root;
//! ```
//!
//! `LayoutAtlasBuilder::leaf` / `container` only build an [`AtlasNodeSpec`]
//! description — a plain value, no Taffy or atlas access — which
//! `LayoutAtlas::build` / `build_root` then commits in one depth-first
//! pass via the same low-level methods, in the same order an equivalent
//! imperative call sequence would use. The low-level API from PR #14 is
//! unchanged; the builder is purely additive sugar over it.
//!
//! # Cross-subsystem flow
//!
//! The Atlas is one consumer of the broadcast `TargetId` stream produced
//! by the Logic thread:
//!
//! ```text
//! signals mutate  →  EvaluatorTick::collect_dirty()  →  Vec<TargetId>
//!                                                       │
//!                                                       ▼
//!                                          atlas.mark_dirty_all(...)
//!                                                       │
//!                                                       ▼
//!                                          atlas.recompute_dirty(...)
//!                                                       │
//!                                                       ▼
//!                              per-target `dirty` bit, read off the
//!                              resolved [`TargetId`] and copied onto the
//!                              matching `TextLine`/`BoxInstance` in
//!                              `RenderFrame` — the Atlas is the only
//!                              subsystem that calls `mark_dirty_all`; the
//!                              encoder never broadcasts, it only reads the
//!                              dirty bit already attached to each primitive.
//! ```
//!
//! The Atlas filters the broadcast by [`TargetKind`](crate::frame::TargetKind)
//! and ignores foreign or stale entries. See [`LayoutAtlas::mark_dirty_all`]
//! for the filtering rules.

pub mod layout;
pub mod spatial;

pub use layout::{
    Align, AtlasError, AtlasNodeId, AtlasNodeSpec, ContainerStyle, FlexDir, GridItemPlacement,
    GridTrack, Justify, LayoutAtlas, LayoutAtlasBuilder, LeafSize, Spacing, StackAlign,
};

pub use spatial::{CELL_SIZE, SpatialGrid};
