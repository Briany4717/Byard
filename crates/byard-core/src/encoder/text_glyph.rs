//! # `TextGlyph` pipeline
//!
//! GPU text rendering via a [`glyphon`] glyph atlas, integrated into the
//! single UI render pass shared with [`SolidBox`](super::EncoderSubsystem).
//!
//! ## Design constraints
//!
//! - **Single render pass** — `TextGlyphPipeline::render_layer` is called
//!   *inside* the same `wgpu::RenderPass` already started by the `SolidBox`
//!   draw. On Apple Silicon (TBDR architecture) every render pass break
//!   flushes the tile buffer to VRAM; sharing the pass with `SolidBox`
//!   eliminates that cost.
//! - **Layered text draws, one shared atlas** — RFC-0017 z-layers need a
//!   later layer's transparent geometry (a modal scrim, a shadow) to
//!   alpha-blend *over* an earlier layer's text, which a single frame-final
//!   text draw can never provide. Instead of one `TextRenderer`, the pipeline
//!   holds one **per z-layer** — all sharing the *same* `FontSystem`,
//!   `SwashCache`, `TextAtlas`, `Viewport`, and shaped-buffer cache, so every
//!   line is still shaped exactly once and every glyph is rasterised into one
//!   atlas regardless of layer count. The only per-layer cost is one small
//!   glyph vertex buffer and one draw call *inside* the existing pass — never
//!   an extra render pass, never a re-shape, never a duplicate atlas. A
//!   single-layer frame uses exactly one renderer: the pre-layering fast
//!   path, unchanged.
//! - **Upstream dirty flag, trusted in release** — each [`TextLine`] carries
//!   a `dirty` bit set by the Evaluator → Atlas → `RenderFrame` pipeline
//!   (see `frame.rs` and `atlas::layout::LayoutAtlas::populate_frame`).
//!   `--release` builds re-shape a line's glyph buffer if and only if that
//!   bit (or `viewport_dirty`) is set — zero hashing, zero extra CPU cost
//!   for static text. Debug builds additionally compute an `FxHasher`
//!   content hash as a secondary safety net and panic if it disagrees with
//!   the upstream flag, catching dependency-tracking bugs in the byld
//!   transpiler before they reach production. See [`needs_reshape`] and
//!   [`assert_dirty_flag_consistency`].
//! - **Three-pass borrow pattern** — `prepare` splits work across three
//!   sequential passes to satisfy Rust's field-split borrowing rules (see the
//!   method documentation for a precise explanation).
//! - **No panics** — every fallible operation returns [`ByardError`].

// Both imports below feed only `content_hash`, the debug-only secondary
// safety net described in the module documentation — absent in `--release`,
// where the upstream dirty flag is trusted exclusively and no hash is ever
// computed.
#[cfg(debug_assertions)]
use std::hash::Hasher as _;

use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, SwashCache, TextArea,
    TextAtlas, TextBounds, TextRenderer, Viewport,
};

/// Converts a logical-pixel content clip ([`crate::frame::ClipRect`]) into
/// glyphon's physical-pixel [`TextBounds`] (RFC-0005 `ScrollView`), so a text
/// line inside a scroll viewport is clipped to it per-area rather than via a
/// render-pass scissor.
#[allow(clippy::cast_possible_truncation)]
fn clip_to_text_bounds(rect: crate::frame::Rect, scale: f32) -> TextBounds {
    TextBounds {
        left: (rect.x * scale).floor() as i32,
        top: (rect.y * scale).floor() as i32,
        right: ((rect.x + rect.width) * scale).ceil() as i32,
        bottom: ((rect.y + rect.height) * scale).ceil() as i32,
    }
}
#[cfg(debug_assertions)]
use rustc_hash::FxHasher;
use wgpu::MultisampleState;

use crate::ByardError;

// ── Public surface ────────────────────────────────────────────────────────────

/// Re-exported from [`crate::frame`] — the canonical definition now lives
/// there so the Logic thread can populate [`RenderFrame::texts`] without
/// importing from a subsystem that it must not depend on (RFC-0001 §9).
pub use crate::frame::TextLine;

// ── Internal cache entry ──────────────────────────────────────────────────────

/// Cached GPU-side state for a single [`TextLine`].
///
/// Lives entirely inside [`TextGlyphPipeline`]; never exposed outside this
/// module. `buffer` is the shaped glyph run, kept across frames so unchanged
/// lines cost nothing beyond the `needs_reshape` check. `content_hash`
/// (debug builds only) is a secondary safety net — see its own doc comment.
struct CachedLine {
    buffer: Buffer,
    /// `FxHasher` digest of `(text, font_size as bits, color as bits)`.
    ///
    /// **Debug-only secondary safety net.** `prepare` never uses this to
    /// *decide* whether to re-shape — [`needs_reshape`] is the sole
    /// decision point in both build profiles, and it only consults the
    /// upstream `dirty` bit. This hash exists purely so
    /// [`assert_dirty_flag_consistency`] can catch a transpiler bug where
    /// content changed but the dirty bit was not set. Absent in
    /// `--release` — there, the upstream flag is fully trusted and no hash
    /// is ever computed, for zero extra CPU cost on static text.
    #[cfg(debug_assertions)]
    content_hash: u64,
}

/// Computes a cheap content hash for a [`TextLine`].
///
/// Uses [`FxHasher`] (≈3× faster than `SipHash` for small keys) over the
/// fields that affect the GPU glyph run. Position (`x`, `y`) is excluded —
/// a translation never invalidates shaped glyphs.
///
/// **Debug-only.** This function does not exist in `--release` builds —
/// see the module documentation for the trust-the-upstream-flag rationale.
#[cfg(debug_assertions)]
fn content_hash(text: &str, font_size: f32, color: [f32; 4]) -> u64 {
    let mut h = FxHasher::default();
    h.write(text.as_bytes());
    h.write_u32(font_size.to_bits());
    for c in color {
        h.write_u32(c.to_bits());
    }
    h.finish()
}

/// Decides whether a text line's glyph buffer needs to be re-shaped this
/// frame.
///
/// This is the **only** decision point `prepare` consults to skip
/// re-shaping, in both build profiles. It never looks at a content hash —
/// only the caller-supplied dirty bits — so encoder pipelines never
/// re-derive "did this change" the way the old `content_hash`-only check
/// did. Pulled out as a free, pure function so it is unit-testable without
/// any glyphon or wgpu state.
fn needs_reshape(viewport_dirty: bool, line_dirty: bool) -> bool {
    viewport_dirty || line_dirty
}

/// Debug-only safety net: panics if a line's content actually changed
/// (`hash_changed`) but the upstream dirty flag (`line_dirty`) was not set.
///
/// `prepare`'s reshape decision ([`needs_reshape`]) never consults the
/// hash — only `line_dirty`. So if this fires, the line would have been
/// silently left stale on screen, and critically, the same staleness would
/// occur in `--release` too, where this check does not exist at all. This
/// is the deliberate trade: paying a hash comparison in debug builds to
/// catch a transpiler dependency-tracking bug before it ships, in exchange
/// for zero hashing cost in release.
///
/// Absent in `--release` builds.
#[cfg(debug_assertions)]
fn assert_dirty_flag_consistency(hash_changed: bool, line_dirty: bool) {
    assert!(
        !hash_changed || line_dirty,
        "State mutation undetected! A text primitive content changed but its upstream dirty flag was not set. This is a bug in the byld transpiler dependency tracking."
    );
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// GPU text pipeline owned by [`EncoderSubsystem`](super::EncoderSubsystem).
///
/// Wraps all glyphon state into a single struct so that it can be initialised
/// once in [`EncoderSubsystem::init`] and driven frame-by-frame by
/// [`prepare`](TextGlyphPipeline::prepare) + [`render`](TextGlyphPipeline::render).
pub struct TextGlyphPipeline {
    /// glyphon font system — owns the loaded font data and shaped buffers.
    font_system: FontSystem,
    /// glyphon swash cache — rasterises shaped glyphs on demand.
    swash_cache: SwashCache,
    /// glyphon glyph atlas — GPU texture containing rasterised glyphs.
    atlas: TextAtlas,
    /// glyphon viewport — maps logical → physical pixels for the render pass.
    viewport: Viewport,
    /// glyphon renderers, **one per z-layer** (RFC-0017 layered draw batches),
    /// all sharing `atlas`/`viewport`/`font_system`/`swash_cache` above. Grown
    /// lazily by `prepare` to the frame's layer count and truncated when it
    /// shrinks (dropping a stale layer's glyph vertex buffer). Index = layer.
    /// The common single-layer frame keeps exactly one entry.
    renderers: Vec<TextRenderer>,
    /// Per-line cache: shaped buffers and content hashes.
    ///
    /// Index-aligned with the **full** `text_lines` slice passed to `prepare`
    /// — *global* across layers, so layering never re-shapes a line.
    /// Entries are added as new lines appear and never removed (Phase 1).
    cache: Vec<CachedLine>,
}

impl TextGlyphPipeline {
    /// Creates the pipeline.
    ///
    /// Initialises all glyphon resources in the correct order:
    /// `Cache` → `TextAtlas` → `Viewport` → `TextRenderer`. This sequence is
    /// wrapped in a single `Device::push_error_scope` / `pop_error_scope`
    /// pair (RFC §8) — glyphon's constructors are opaque to byard-core, but
    /// an error scope captures any validation error raised on `device`
    /// during the scope regardless of which crate triggered it, so the
    /// guarantee holds even though byard-core never calls
    /// `create_render_pipeline` itself here.
    ///
    /// # Errors
    ///
    /// Returns [`ByardError::PipelineCompilation`] if glyphon's internal
    /// pipeline/shader construction fails GPU-side validation.
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Result<Self, ByardError> {
        // --- GPU VALIDATION ERROR SCOPE (RFC §8) ---
        // Covers Cache::new + TextAtlas::new + Viewport::new + TextRenderer::new.
        // glyphon's pipeline/shader creation is opaque to byard-core, but the
        // scope still captures any validation error wgpu raises on `device`
        // while it runs.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let glyph_cache = Cache::new(device);
        let mut atlas = TextAtlas::new(device, queue, &glyph_cache, format);
        let viewport = Viewport::new(device, &glyph_cache);
        // Enable the same draw-order depth state as the box/texture pipelines so
        // glyphon's text participates in cross-pass paint ordering (RFC-0011)
        // instead of always drawing on top. This first renderer is layer 0 —
        // the only one a frame without overlays ever touches; `prepare` grows
        // the vec on demand when a frame carries more z-layers.
        let renderer = TextRenderer::new(
            &mut atlas,
            device,
            MultisampleState::default(),
            Some(super::draw_depth_stencil()),
        );

        if let Some(error) = scope.pop().await {
            return Err(ByardError::PipelineCompilation {
                pipeline: "TextGlyph".to_string(),
                reason: error.to_string(),
            });
        }

        Ok(Self {
            font_system: FontSystem::new(),
            swash_cache: SwashCache::new(),
            atlas,
            viewport,
            renderers: vec![renderer],
            cache: Vec::new(),
        })
    }

    /// Uploads updated viewport dimensions to the glyphon `Viewport`.
    ///
    /// Must be called whenever the surface is resized and before the next
    /// `prepare`. The `phys_w`/`phys_h` pair is in **physical pixels** —
    /// glyphon's `Resolution` always works in physical pixels.
    pub fn update_resolution(&mut self, queue: &wgpu::Queue, phys_w: u32, phys_h: u32) {
        self.viewport.update(
            queue,
            Resolution {
                width: phys_w,
                height: phys_h,
            },
        );
    }

    /// Shapes and uploads text for the next frame.
    ///
    /// `scale_factor` converts logical → physical pixels so that glyph metrics
    /// stay DPI-correct. `viewport_dirty` forces a re-prepare even when no
    /// text content has changed (e.g. after a window resize).
    ///
    /// `layer_ranges` partitions `text_lines` into the frame's z-layers
    /// (RFC-0017 layered draw batches): contiguous, non-overlapping index
    /// ranges covering the whole slice, one per layer, in draw order. Each
    /// layer gets its own `TextRenderer` (grown here on demand) so the
    /// Encoder can interleave text draws with the other pools per layer; the
    /// shaping pass below stays **global**, so a line is shaped exactly once
    /// no matter how many layers the frame has. Pass a single `0..len` range
    /// for the ordinary single-layer frame.
    ///
    /// ## Three-pass borrow pattern
    ///
    /// Rust's field-split borrowing cannot reason across a Vec of structs when
    /// the same loop body needs both `&mut entry.buffer` (for layout) and
    /// `&entry.buffer` (for the `TextArea` slice). Three sequential passes solve
    /// this cleanly:
    ///
    /// 1. **Mutation pass** — mutably borrows `self.cache` and
    ///    `self.font_system` to grow the cache and re-shape dirty buffers.
    /// 2. **Collection pass** — immutably borrows `self.cache` to build a
    ///    `Vec<TextArea<'_>>` holding `&entry.buffer` references.
    /// 3. **Prepare pass** — borrows `self.renderers`, `self.font_system`,
    ///    `self.atlas`, `self.viewport`, `self.swash_cache` — all distinct
    ///    from `self.cache`, which is already borrowed by `text_areas`.
    ///
    /// Passes 2–3 run once per layer over that layer's subrange; every
    /// renderer is re-prepared each call (an empty range clears its layer's
    /// previous glyph buffer — required, or a text line that moved layers
    /// would ghost in both).
    ///
    /// # Errors
    ///
    /// Returns [`ByardError::TextPrepare`] if glyphon's `prepare` fails.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        text_lines: &[TextLine],
        depths: &[f32],
        scale_factor: f32,
        viewport_dirty: bool,
        clips: &[crate::frame::ClipRect],
        text_clips: &[Option<u16>],
        layer_ranges: &[std::ops::Range<usize>],
    ) -> Result<(), ByardError> {
        // ── Pass 1: grow cache and re-shape dirty lines ───────────────────────
        //
        // Each CachedLine is grown lazily (push when missing) rather than
        // resize_with so the closure does not need to capture &mut font_system
        // at the same time as &mut cache — which would be a double borrow.
        let preexisting_len = self.cache.len();
        while self.cache.len() < text_lines.len() {
            let metrics = Metrics::new(12.0, 14.0); // placeholder; overwritten below
            let buffer = Buffer::new(&mut self.font_system, metrics);
            self.cache.push(CachedLine {
                buffer,
                #[cfg(debug_assertions)]
                content_hash: 0,
            });
        }

        for (i, line) in text_lines.iter().enumerate() {
            // A line beyond the cache's previous length has no shaped
            // buffer yet — it must always be shaped on its first
            // appearance, regardless of `line.dirty` or the debug-only
            // hash. (`line.dirty` reflects whether the *value* changed
            // since last tick, not whether this line existed before;
            // requiring callers to set it for brand-new lines would be an
            // easy-to-miss footgun, so we detect "new" structurally here.)
            let is_new = i >= preexisting_len;
            let entry = &mut self.cache[i];

            #[cfg(debug_assertions)]
            {
                let hash = content_hash(&line.text, line.font_size, line.color);
                if !is_new {
                    assert_dirty_flag_consistency(hash != entry.content_hash, line.dirty);
                }
                entry.content_hash = hash;
            }

            if !is_new && !needs_reshape(viewport_dirty, line.dirty) {
                continue; // unchanged — skip re-shaping
            }

            let metrics = Metrics::new(line.font_size, line.font_size * 1.2);
            entry.buffer.set_metrics(&mut self.font_system, metrics);
            entry.buffer.set_size(
                &mut self.font_system,
                None, // unbounded width
                None, // unbounded height
            );

            // Color is applied per-TextArea in pass 2 (default_color field).
            // Here we only need to shape the text; color does not affect layout.
            // Tag every glyph of this line with its line index as glyphon
            // `metadata`, so pass 3's `metadata_to_depth` can look up this
            // line's draw-order depth. Lines are re-shaped every tick (all
            // dirty), so the metadata stays current with the line's index.
            entry.buffer.set_text(
                &mut self.font_system,
                &line.text,
                &Attrs::new().family(Family::SansSerif).metadata(i),
                glyphon::Shaping::Advanced,
                None, // align: no paragraph-level override
            );
            entry
                .buffer
                .shape_until_scroll(&mut self.font_system, false);
        }

        // ── Grow/shrink the per-layer renderer pool ───────────────────────────
        //
        // One `TextRenderer` per z-layer, all sharing the atlas/viewport/font
        // stack. Growth is rare (a new overlay depth appears); truncation
        // drops a vanished layer's glyph vertex buffer instead of leaking it.
        let layer_count = layer_ranges.len().max(1);
        while self.renderers.len() < layer_count {
            self.renderers.push(TextRenderer::new(
                &mut self.atlas,
                device,
                MultisampleState::default(),
                Some(super::draw_depth_stencil()),
            ));
        }
        self.renderers.truncate(layer_count);

        // A missing/empty partition means "everything is layer 0" — the
        // ordinary frame — expressed as one full-slice range.
        let full_range = 0..text_lines.len();
        let ranges: &[std::ops::Range<usize>] = if layer_ranges.is_empty() {
            std::slice::from_ref(&full_range)
        } else {
            layer_ranges
        };

        // ── Passes 2 + 3, once per layer ──────────────────────────────────────
        for (renderer, range) in self.renderers.iter_mut().zip(ranges) {
            // Clamp defensively: a malformed range must never panic the
            // render thread — worst case a line draws in the wrong layer.
            let start = range.start.min(text_lines.len());
            let end = range.end.clamp(start, text_lines.len());

            // ── Pass 2: collect immutable TextArea refs for this layer ────────
            //
            // A free function over `&self.cache` (not a `&self` method) so the
            // returned borrow is of the `cache` field alone — a method would
            // freeze all of `self` and collide with the `&mut` field borrows
            // pass 3 needs.
            let text_areas = collect_layer_text_areas(
                &self.cache,
                text_lines,
                start..end,
                scale_factor,
                clips,
                text_clips,
            );

            // ── Pass 3: glyphon prepare (with draw-order depth) ───────────────
            //
            // Borrows: this layer's renderer (from `self.renderers`),
            // font_system, atlas, viewport, swash_cache. All distinct fields
            // from `cache` (borrowed by text_areas).
            //
            // `metadata_to_depth` maps each glyph's metadata (its *global* line
            // index, set in pass 1) to that line's draw-order NDC-z, so text is
            // depth-sorted against solids/decorated/textures instead of always
            // painting on top (RFC-0011 cross-pass paint order). A missing/
            // out-of-range depth falls back to the far plane. An empty layer
            // still prepares — that is what clears its renderer's previous
            // glyph buffer.
            renderer
                .prepare_with_depth(
                    device,
                    queue,
                    &mut self.font_system,
                    &mut self.atlas,
                    &self.viewport,
                    text_areas,
                    &mut self.swash_cache,
                    |meta| {
                        depths
                            .get(meta)
                            .copied()
                            .unwrap_or(crate::frame::DRAW_DEPTH_CLEAR)
                    },
                )
                .map_err(|e| ByardError::TextPrepare(e.to_string()))?;
        }
        Ok(())
    }

    /// Records **one z-layer's** text draw commands into the active render
    /// pass (RFC-0017 layered draw batches).
    ///
    /// Must be called after that layer's box/texture/vector draws, inside the
    /// same `wgpu::RenderPass`. On TBDR architectures (Apple Silicon), keeping
    /// every layer in one pass eliminates a tile-buffer flush. A `layer` with
    /// no renderer (out of range — e.g. the frame carries more layer marks
    /// than the last `prepare` saw) is a no-op rather than an error, so a
    /// racing layer-count change can never crash the render thread.
    ///
    /// # Errors
    ///
    /// Returns [`ByardError::TextRender`] if glyphon's `render` fails (e.g.
    /// atlas overflow — rare after a successful `prepare`).
    pub fn render_layer<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        layer: usize,
    ) -> Result<(), ByardError> {
        let Some(renderer) = self.renderers.get(layer) else {
            return Ok(());
        };
        renderer
            .render(&self.atlas, &self.viewport, render_pass)
            .map_err(|e| ByardError::TextRender(e.to_string()))
    }
}

/// Pass 2 of [`TextGlyphPipeline::prepare`]: builds one z-layer's
/// `TextArea`s — the `range` subslice of `text_lines`, each referencing its
/// *globally* cached shaped buffer (index-aligned with the full slice, like
/// `clips`/`text_clips`).
///
/// A free function over the `cache` slice rather than a `&self` method: the
/// returned `TextArea`s must borrow **only** the `cache` field, so pass 3 can
/// still take `&mut` borrows of the pipeline's other fields (the module's
/// field-split borrow pattern). `range` is pre-clamped by the caller.
fn collect_layer_text_areas<'cache>(
    cache: &'cache [CachedLine],
    text_lines: &[TextLine],
    range: std::ops::Range<usize>,
    scale_factor: f32,
    clips: &[crate::frame::ClipRect],
    text_clips: &[Option<u16>],
) -> Vec<TextArea<'cache>> {
    let start = range.start;
    text_lines[range]
        .iter()
        .enumerate()
        .map(|(offset, line)| {
            let global = start + offset; // global line index (cache/clips/depths)
            let [red, green, blue, alpha] = line.color;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let default_color = Color::rgba(
                (red.clamp(0.0, 1.0) * 255.0) as u8,
                (green.clamp(0.0, 1.0) * 255.0) as u8,
                (blue.clamp(0.0, 1.0) * 255.0) as u8,
                (alpha.clamp(0.0, 1.0) * 255.0) as u8,
            );
            TextArea {
                buffer: &cache[global].buffer,
                // glyphon's Viewport/Resolution is configured in PHYSICAL
                // pixels (see EncoderSubsystem::update_viewport), but
                // TextLine.x/y are authored in logical pixels like every
                // other public coordinate in this crate. cosmic-text's
                // glyph positioning does not rescale this offset — only
                // the buffer's own shaped glyph extents are scaled by
                // `scale` — so `left`/`top` must already be physical
                // pixels or text lands at `logical / scale_factor`,
                // visibly drifting toward the origin on HiDPI displays.
                left: line.x * scale_factor,
                top: line.y * scale_factor,
                scale: scale_factor,
                // Content clip (RFC-0005 `ScrollView`): a line inside a
                // scroll viewport is clipped to it via glyphon's own
                // `TextBounds` (physical px) — the clean, per-area way to
                // clip text without a render-pass scissor. Unclipped
                // lines stay unbounded.
                bounds: text_clips
                    .get(global)
                    .copied()
                    .flatten()
                    .and_then(|idx| clips.get(idx as usize))
                    .map_or(
                        TextBounds {
                            left: 0,
                            top: 0,
                            right: i32::MAX,
                            bottom: i32::MAX,
                        },
                        |c| clip_to_text_bounds(c.rect, scale_factor),
                    ),
                default_color,
                custom_glyphs: &[],
            }
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// `needs_reshape` and `assert_dirty_flag_consistency` are extracted as pure,
// glyphon/wgpu-free functions specifically so the dirty-flag decision logic
// from the acceptance criteria ("encoder pipelines never recompute
// did-this-change") can be exercised deterministically here, without a real
// `wgpu::Device` — the same CPU-mirror-of-decision-logic style already used
// by `encoder::mod`'s `cpu_sd_rounded_box` tests.
#[cfg(test)]
mod tests {
    use super::*;

    // ── needs_reshape: all four (viewport_dirty, line_dirty) combinations ──

    #[test]
    fn needs_reshape_false_when_nothing_is_dirty() {
        assert!(!needs_reshape(false, false));
    }

    #[test]
    fn needs_reshape_true_when_only_viewport_is_dirty() {
        assert!(needs_reshape(true, false));
    }

    #[test]
    fn needs_reshape_true_when_only_line_is_dirty() {
        assert!(needs_reshape(false, true));
    }

    #[test]
    fn needs_reshape_true_when_both_are_dirty() {
        assert!(needs_reshape(true, true));
    }

    // ── assert_dirty_flag_consistency: the debug-only safety net ───────────

    #[test]
    #[cfg(debug_assertions)]
    fn consistency_check_passes_when_hash_unchanged_and_not_dirty() {
        assert_dirty_flag_consistency(false, false);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn consistency_check_passes_when_hash_unchanged_but_dirty_anyway() {
        // Over-marking dirty is wasteful, never unsound — must not panic.
        assert_dirty_flag_consistency(false, true);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn consistency_check_passes_when_hash_changed_and_dirty_was_set() {
        assert_dirty_flag_consistency(true, true);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "State mutation undetected! A text primitive content changed but its upstream dirty flag was not set. This is a bug in the byld transpiler dependency tracking."
    )]
    fn consistency_check_panics_when_hash_changed_but_dirty_was_not_set() {
        assert_dirty_flag_consistency(true, false);
    }

    // ── content_hash: debug-only helper feeding the safety net ─────────────

    #[test]
    #[cfg(debug_assertions)]
    fn content_hash_is_stable_for_identical_input() {
        let a = content_hash("hello", 14.0, [1.0, 1.0, 1.0, 1.0]);
        let b = content_hash("hello", 14.0, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(a, b);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn content_hash_changes_with_text() {
        let a = content_hash("hello", 14.0, [1.0, 1.0, 1.0, 1.0]);
        let b = content_hash("world", 14.0, [1.0, 1.0, 1.0, 1.0]);
        assert_ne!(a, b);
    }

    // ── TextLine: dirty field is a plain, independent bit ──────────────────

    #[test]
    fn text_line_dirty_field_round_trips() {
        let dirty_line = TextLine {
            x: 0.0,
            y: 0.0,
            text: "hi".to_string(),
            font_size: 12.0,
            color: [0.0, 0.0, 0.0, 1.0],
            dirty: true,
        };
        assert!(dirty_line.dirty);

        let clean_line = TextLine {
            dirty: false,
            ..dirty_line
        };
        assert!(!clean_line.dirty);
    }
}
