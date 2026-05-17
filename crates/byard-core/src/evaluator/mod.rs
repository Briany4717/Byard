//! # Evaluator
//!
//! Reactive state and per-view memory management.
//!
//! This subsystem owns:
//!
//! - **`Signal<T>`** — A reactive value that holds data of type `T` and a set of
//!   atomic dirty flags. Mutating a signal marks the corresponding flags without
//!   triggering a full tree rebuild.
//!
//! - **`ViewArena`** — A contiguous memory block allocated when a `View` is mounted.
//!   Stores all `Signal` values, Taffy node references, and spatial grid entries for
//!   that view. Released in `O(1)` via `Drop` when the view is unmounted.
//!
//! - **Dirty-flag tick** — On each logic tick, the evaluator collects all dirty
//!   flags set since the last tick and produces the minimal set of dirty rectangles
//!   to hand off to the encoder via [`crate::frame`].
