//! # Relay
//!
//! Thread management, double-buffered visual state, and async I/O.
//!
//! This subsystem owns:
//!
//! - **Render thread** — Holds an `Arc<RenderFrame>` (immutable) and runs the
//!   `request_redraw` loop. Never blocks.
//!
//! - **Logic thread** — Mutates signals, runs Taffy, and updates the spatial grid.
//!   Produces a new [`RenderFrame`](crate::frame::RenderFrame) and performs an
//!   atomic pointer swap with the render thread — no mutex on the hot path.
//!
//! - **Tokio pool** — Executes async I/O from Rust controllers. Results are sent
//!   to the logic thread via `tokio::sync::mpsc`.
//!
//! The render thread and the logic thread never share mutable state. The frame
//! swap is a single atomic pointer exchange.
