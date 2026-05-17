//! # Encoder
//!
//! Multi-pipeline `wgpu` command dispatch.
//!
//! This subsystem owns the specialised render pipelines compiled at startup:
//!
//! - **`SolidBox`** — Axis-aligned rectangles with solid fill and `border-radius`.
//! - **`DecoratedBox`** — Rectangles with gradients, box-shadows, and paramétric
//!   decorations.
//! - **`TextGlyph`** — Text rendering via a `glyphon` glyph atlas.
//! - **`TextureSampler`** — UV-mapped quads for decoded images and icons.
//!
//! Primitives are batched into Z-bins (stacking contexts) and ordered first by
//! pipeline, then by local Z-index, to minimise GPU context switches.
//!
//! Partial screen updates use `wgpu::RenderPass::set_scissor_rect` driven by
//! dirty rectangles from the evaluator, limiting VRAM bandwidth to only the
//! affected region.
//!
//! Pipeline creation is wrapped in `Device::push_error_scope` /
//! `Device::pop_error_scope` with `ErrorFilter::Validation`. Failures are
//! surfaced as [`ByardError::PipelineCompilation`](crate::ByardError::PipelineCompilation)
//! — the engine never panics on a GPU error.
