//! # `TextGlyph` pipeline
//!
//! GPU text rendering via a [`glyphon`] glyph atlas, integrated into the
//! single UI render pass shared with [`SolidBox`](super::EncoderSubsystem).
//!
//! ## Design constraints
//!
//! - **Single render pass** — `TextGlyphPipeline::render` is called *inside*
//!   the same `wgpu::RenderPass` already started by the `SolidBox` draw. On
//!   Apple Silicon (TBDR architecture) every render pass break flushes the
//!   tile buffer to VRAM; sharing the pass with `SolidBox` eliminates that cost.
//! - **Dirty detection via `FxHasher`** — each [`TextLine`] carries a
//!   content hash computed from `(text, font_size, color)`. [`CachedLine`]
//!   stores the hash of the last-uploaded version; glyphon layout is skipped
//!   when the hash matches, so unchanged text costs only a hash comparison.
//! - **Three-pass borrow pattern** — `prepare` splits work across three
//!   sequential passes to satisfy Rust's field-split borrowing rules (see the
//!   method documentation for a precise explanation).
//! - **No panics** — every fallible operation returns [`ByardError`].

use std::hash::Hasher as _;

use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, SwashCache, TextArea,
    TextAtlas, TextBounds, TextRenderer, Viewport,
};
use rustc_hash::FxHasher;
use wgpu::MultisampleState;

use crate::ByardError;

// ── Public surface ────────────────────────────────────────────────────────────

/// A single line of text to be rendered in a frame.
///
/// All fields are primitives — no glyphon types are exposed at the public API
/// boundary. The pipeline owns all GPU-side state internally.
///
/// Coordinates are in **logical pixels**, consistent with [`BoxInstance`](super::BoxInstance).
#[derive(Debug, Clone)]
pub struct TextLine {
    /// X position of the text baseline in logical pixels.
    pub x: f32,
    /// Y position of the text baseline in logical pixels.
    pub y: f32,
    /// Text content.
    pub text: String,
    /// Font size in logical pixels.
    pub font_size: f32,
    /// Text colour: `[r, g, b, a]` in linear space, each component 0–1.
    pub color: [f32; 4],
}

// ── Internal cache entry ──────────────────────────────────────────────────────

/// Cached GPU-side state for a single [`TextLine`].
///
/// Lives entirely inside [`TextGlyphPipeline`]; never exposed outside this
/// module. `buffer` is the shaped glyph run; `content_hash` lets `prepare`
/// skip re-shaping when the visible content has not changed.
struct CachedLine {
    buffer: Buffer,
    /// `FxHasher` digest of `(text, font_size as bits, color as bits)`.
    ///
    /// Re-computed each frame from the incoming [`TextLine`]. When it matches
    /// the stored value we skip `buffer.set_text` + `buffer.shape_until_scroll`.
    content_hash: u64,
}

/// Computes a cheap content hash for a [`TextLine`].
///
/// Uses [`FxHasher`] (≈3× faster than `SipHash` for small keys) over the
/// fields that affect the GPU glyph run. Position (`x`, `y`) is excluded —
/// a translation never invalidates shaped glyphs.
fn content_hash(text: &str, font_size: f32, color: [f32; 4]) -> u64 {
    let mut h = FxHasher::default();
    h.write(text.as_bytes());
    h.write_u32(font_size.to_bits());
    for c in color {
        h.write_u32(c.to_bits());
    }
    h.finish()
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
    /// glyphon renderer — records text draw calls into a `RenderPass`.
    renderer: TextRenderer,
    /// Per-line cache: shaped buffers and content hashes.
    ///
    /// Index-aligned with the `text_lines` slice passed to `prepare`.
    /// Entries are added as new lines appear and never removed (Phase 1).
    cache: Vec<CachedLine>,
}

impl TextGlyphPipeline {
    /// Creates the pipeline.
    ///
    /// Initialises all glyphon resources in the correct order:
    /// `Cache` → `TextAtlas` → `Viewport` → `TextRenderer`.
    ///
    /// # Errors
    ///
    /// Currently infallible. Returns `Result` so callers already write `?` and
    /// the signature stays stable when fallible GPU operations are added.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Result<Self, ByardError> {
        let glyph_cache = Cache::new(device);
        let mut atlas = TextAtlas::new(device, queue, &glyph_cache, format);
        let viewport = Viewport::new(device, &glyph_cache);
        let renderer = TextRenderer::new(&mut atlas, device, MultisampleState::default(), None);

        Ok(Self {
            font_system: FontSystem::new(),
            swash_cache: SwashCache::new(),
            atlas,
            viewport,
            renderer,
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
    /// 3. **Prepare pass** — borrows `self.renderer`, `self.font_system`,
    ///    `self.atlas`, `self.viewport`, `self.swash_cache` — all distinct
    ///    from `self.cache`, which is already borrowed by `text_areas`.
    ///
    /// # Errors
    ///
    /// Returns [`ByardError::TextPrepare`] if glyphon's `prepare` fails.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        text_lines: &[TextLine],
        scale_factor: f32,
        viewport_dirty: bool,
    ) -> Result<(), ByardError> {
        // ── Pass 1: grow cache and re-shape dirty lines ───────────────────────
        //
        // Each CachedLine is grown lazily (push when missing) rather than
        // resize_with so the closure does not need to capture &mut font_system
        // at the same time as &mut cache — which would be a double borrow.
        while self.cache.len() < text_lines.len() {
            let metrics = Metrics::new(12.0, 14.0); // placeholder; overwritten below
            let buffer = Buffer::new(&mut self.font_system, metrics);
            self.cache.push(CachedLine {
                buffer,
                content_hash: 0,
            });
        }

        for (i, line) in text_lines.iter().enumerate() {
            let hash = content_hash(&line.text, line.font_size, line.color);
            let entry = &mut self.cache[i];

            if !viewport_dirty && entry.content_hash == hash {
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
            entry.buffer.set_text(
                &mut self.font_system,
                &line.text,
                &Attrs::new().family(Family::SansSerif),
                glyphon::Shaping::Advanced,
                None, // align: no paragraph-level override
            );
            entry
                .buffer
                .shape_until_scroll(&mut self.font_system, false);
            entry.content_hash = hash;
        }

        // ── Pass 2: collect immutable TextArea refs ───────────────────────────
        let text_areas: Vec<TextArea<'_>> = text_lines
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let [r, g, b, a] = line.color;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let default_color = Color::rgba(
                    (r.clamp(0.0, 1.0) * 255.0) as u8,
                    (g.clamp(0.0, 1.0) * 255.0) as u8,
                    (b.clamp(0.0, 1.0) * 255.0) as u8,
                    (a.clamp(0.0, 1.0) * 255.0) as u8,
                );
                TextArea {
                    buffer: &self.cache[i].buffer,
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
                    // Unbounded clip region: text is visible anywhere on screen.
                    // A real layout system would tighten this to the widget's rect.
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: i32::MAX,
                        bottom: i32::MAX,
                    },
                    default_color,
                    custom_glyphs: &[],
                }
            })
            .collect();

        // ── Pass 3: glyphon prepare ───────────────────────────────────────────
        //
        // Borrows: renderer, font_system, atlas, viewport, swash_cache.
        // These are all distinct fields from `cache` (borrowed by text_areas).
        self.renderer
            .prepare(
                device,
                queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .map_err(|e| ByardError::TextPrepare(e.to_string()))
    }

    /// Records text draw commands into the active render pass.
    ///
    /// Must be called **after** `SolidBox` draw calls inside the same
    /// `wgpu::RenderPass`. On TBDR architectures (Apple Silicon), keeping both
    /// in one pass eliminates a tile-buffer flush.
    ///
    /// # Errors
    ///
    /// Returns [`ByardError::TextRender`] if glyphon's `render` fails (e.g.
    /// atlas overflow — rare after a successful `prepare`).
    pub fn render<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) -> Result<(), ByardError> {
        self.renderer
            .render(&self.atlas, &self.viewport, render_pass)
            .map_err(|e| ByardError::TextRender(e.to_string()))
    }
}
