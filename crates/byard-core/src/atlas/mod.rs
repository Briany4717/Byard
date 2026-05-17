//! # Atlas
//!
//! Layout computation and spatial hit-testing.
//!
//! This subsystem owns:
//!
//! - **Taffy integration** — All layout is delegated to
//!   [`taffy`](https://github.com/DioxusLabs/taffy). The engine never computes box
//!   geometry itself. Atlas initialises the Taffy tree, feeds it node constraints,
//!   and reads back resolved rectangles.
//!
//! - **Spatial hash grid** — A partitioned data structure that indexes a mapping
//!   between 2D screen coordinates and event descriptors. Event handlers declared
//!   in the DSL register their Taffy-resolved rect into the grid. Pointer queries
//!   resolve in amortised `O(1)` by computing a hash index — the UI tree is never
//!   walked during event dispatch.
//!
//! Atlas exposes resolved geometry to the encoder exclusively through
//! [`crate::frame`] primitives.
