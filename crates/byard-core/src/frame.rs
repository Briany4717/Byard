//! Shared data types for cross-subsystem communication.
//!
//! This module defines [`RenderFrame`] and [`TargetId`], the primitive types
//! that flow between the evaluator, atlas, encoder, and relay subsystems.
//! It is the **only** module that all subsystems may depend on.
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

/// An opaque, copyable identifier for a dirty-flag target.
///
/// Internally packs three fields into a single 64-bit word:
///
/// - bits 0–31  — `index`, the position inside the owning subsystem's table
/// - bits 32–47 — `generation`, a monotonic counter that lets stale IDs be
///   detected when the underlying slot is reused
/// - bits 48–63 — `kind`, a discriminant identifying which subsystem owns
///   the target (atlas node, encoder primitive, …)
///
/// The internal representation is private; consumers must use [`TargetId::new`]
/// to construct an ID and the accessor methods to read its parts.
///
/// Lives in `frame` rather than any subsystem module so all subsystems may
/// reference it without violating the dependency graph in RFC-0001 §9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TargetId(u64);

/// Discriminant identifying which subsystem owns a [`TargetId`].
///
/// Stored in the high 16 bits of every `TargetId` so subsystems can filter
/// the broadcast `mark_dirty_all` calls down to their own targets without
/// coordination.
///
/// `#[repr(u16)]` guarantees the in-memory representation matches the
/// `TargetId` bit layout, so `TargetKind::Foo as u16` is a zero-cost cast.
#[repr(u16)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    /// A layout node owned by `LayoutAtlas`.
    AtlasNode = 1,
    /// A render primitive owned by an Encoder pipeline (`SolidBox`,
    /// `TextGlyph`, …), addressed by its position in the `RenderFrame`.
    EncoderPrimitive = 2,
}

impl TargetId {
    /// Constructs a `TargetId` from its three components.
    ///
    /// The `index`, `generation`, and `kind` are packed into a single
    /// 64-bit word — see the [`TargetId`] type documentation for the
    /// bit layout.
    #[must_use]
    pub const fn new(index: u32, generation: u16, kind: u16) -> Self {
        let raw = (index as u64) | ((generation as u64) << 32) | ((kind as u64) << 48);
        Self(raw)
    }

    /// Returns the index part of the ID.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn index(self) -> u32 {
        // Truncation is intentional: we mask to the low 32 bits.
        (self.0 & 0xFFFF_FFFF) as u32
    }

    /// Returns the generation part of the ID.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn generation(self) -> u16 {
        // Truncation is intentional: we mask to bits 32-47.
        ((self.0 >> 32) & 0xFFFF) as u16
    }

    /// Returns the kind part of the ID.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn kind(self) -> u16 {
        // Truncation is intentional: we mask to the high 16 bits.
        ((self.0 >> 48) & 0xFFFF) as u16
    }

    /// Returns the raw 64-bit representation of the ID.
    #[must_use]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// An axis-aligned rectangle in logical pixel coordinates.
///
/// Produced by the Atlas as the resolved position and size of a node,
/// consumed by the Encoder to issue draw commands. Lives in `frame`
/// because it crosses the subsystem boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rect {
    /// Top-left X coordinate in logical pixels.
    pub x: f32,
    /// Top-left Y coordinate in logical pixels.
    pub y: f32,
    /// Width in logical pixels.
    pub width: f32,
    /// Height in logical pixels.
    pub height: f32,
}

impl Rect {
    /// Constructs a new rectangle.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Returns `true` if the rectangle contains the given point.
    ///
    /// Uses half-open bounds: the left (`x`) and top (`y`) edges are
    /// **inclusive**, while the right (`x + width`) and bottom
    /// (`y + height`) edges are **exclusive**. This matches the convention
    /// used by the spatial hash grid (sub-issue pending) and avoids
    /// off-by-one disagreements during hit-testing.
    #[must_use]
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    /// Returns the smallest rectangle that fully covers both `self` and
    /// `other`.
    ///
    /// Used by the Encoder (RFC-0001 §3.3) to merge several dirty-region
    /// bounding boxes into the single bounding box passed to
    /// `wgpu::RenderPass::set_scissor_rect`. Degenerate (zero-area) rects are
    /// handled the same as any other rect: the union still expands to cover
    /// their `(x, y)` corner.
    #[must_use]
    pub fn union(&self, other: &Rect) -> Rect {
        let min_x = self.x.min(other.x);
        let min_y = self.y.min(other.y);
        let max_x = (self.x + self.width).max(other.x + other.width);
        let max_y = (self.y + self.height).max(other.y + other.height);
        Rect {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }
}

/// GPU-ready instance data for a single solid rectangle.
///
/// Shared between the logic thread (which populates [`RenderFrame::instances`])
/// and the Encoder (which uploads the slice to the GPU instance buffer). Lives
/// in `frame` rather than `encoder` because it crosses the subsystem boundary
/// between the Logic thread's layout pass and the Encoder's GPU dispatch —
/// see the RFC-0001 §9 dependency graph.
///
/// `#[repr(C)]` and `bytemuck` derives match the layout declared in
/// [`BoxInstance::layout`](crate::encoder::BoxInstance::layout).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoxInstance {
    /// Rectangle in logical pixels: `[x, y, width, height]`.
    pub rect: [f32; 4],
    /// Linear-space fill colour: `[r, g, b, a]`.
    pub color: [f32; 4],
    /// Per-corner border radii: `[top_left, top_right, bottom_right, bottom_left]`.
    pub radii: [f32; 4],
}

/// How an image is scaled/positioned inside its bounding rect.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ImageFit {
    /// Stretch to fill, ignoring aspect ratio.
    #[default]
    Fill,
    /// Scale uniformly to contain inside the rect (letterbox).
    Contain,
    /// Scale uniformly to cover the rect (crop).
    Cover,
    /// No scaling — image at natural size, top-left aligned.
    None,
}

/// A `DecoratedBox` extends a [`BoxInstance`] with an optional border and
/// drop shadow (M21 pipeline). Fields that don't apply are zeroed.
///
/// The Encoder promotes a plain `BoxInstance` to `DecoratedBox` when any of
/// border or shadow fields are non-trivial.
#[derive(Copy, Clone, Debug)]
pub struct DecoratedBox {
    /// The underlying fill/radii data.
    pub base: BoxInstance,
    /// Border width in logical pixels (0.0 = no border).
    pub border_width: f32,
    /// Border colour `[r, g, b, a]`.
    pub border_color: [f32; 4],
    /// Drop-shadow offset X in logical pixels.
    pub shadow_dx: f32,
    /// Drop-shadow offset Y in logical pixels.
    pub shadow_dy: f32,
    /// Drop-shadow blur radius in logical pixels.
    pub shadow_blur: f32,
    /// Drop-shadow colour `[r, g, b, a]`.
    pub shadow_color: [f32; 4],
    /// Element opacity `0.0–1.0`.
    pub opacity: f32,
    /// Whether this decoration changed since the last tick.
    ///
    /// The encoder's analogue of [`TextLine::dirty`] for the `DecoratedBox`
    /// pipeline (RFC-0001 §3.3): set upstream by the Evaluator → `RenderFrame`
    /// lowering, trusted by the Encoder when computing the incremental scissor
    /// union. A decoration's `base` is a [`BoxInstance`], which is a pure GPU
    /// `Pod` vertex type and therefore cannot itself carry a dirty bit — so the
    /// flag lives here on the (non-`Pod`) wrapper instead.
    pub dirty: bool,
}

impl Default for DecoratedBox {
    fn default() -> Self {
        Self {
            base: BoxInstance {
                rect: [0.0; 4],
                color: [0.0; 4],
                radii: [0.0; 4],
            },
            border_width: 0.0,
            border_color: [0.0; 4],
            shadow_dx: 0.0,
            shadow_dy: 0.0,
            shadow_blur: 0.0,
            shadow_color: [0.0; 4],
            opacity: 1.0,
            dirty: false,
        }
    }
}

/// A texture-sampled rectangle: `Image` intrinsic lowered to a GPU primitive
/// (M21 pipeline). Texture data is identified by a host-opaque `texture_id`
/// (registered outside the engine boundary via the controller boundary, M23).
#[derive(Clone, Debug)]
pub struct TextureSampler {
    /// Rectangle in logical pixels `[x, y, width, height]`.
    pub rect: [f32; 4],
    /// Texture source path or ID (resolved by the controller boundary at M23).
    pub src: String,
    /// How the image is scaled within the rect.
    pub fit: ImageFit,
    /// Per-corner border radii.
    pub radii: [f32; 4],
    /// Opacity `0.0–1.0`.
    pub opacity: f32,
    /// Whether this image primitive changed since the last tick.
    ///
    /// The `TextureSampler` analogue of [`TextLine::dirty`] (RFC-0001 §3.3) —
    /// set upstream by the lowering, trusted by the Encoder's incremental
    /// scissor union. Also set by the Encoder itself the frame after an async
    /// decode completes (M29), so a freshly-loaded image paints without a full
    /// redraw.
    pub dirty: bool,
}

/// A single line of text to be rendered in a frame.
///
/// Shared between the logic thread (which populates [`RenderFrame::texts`]) and
/// the Encoder's `TextGlyphPipeline`. Lives in `frame` rather than
/// `encoder::text_glyph` because it crosses the subsystem boundary between the
/// Evaluator/Atlas and the Encoder — see RFC-0001 §9.
///
/// All coordinates are in **logical pixels**, consistent with [`BoxInstance`].
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
    /// Whether this line's content changed since the last tick.
    ///
    /// Set upstream by the Evaluator → Atlas → `RenderFrame` pipeline — never
    /// derived locally by the Encoder. The Encoder trusts this bit completely
    /// in `--release` builds; see `encoder::text_glyph`'s module documentation.
    pub dirty: bool,
}

/// Logical-pixel dimensions of the surface that hosts a layout.
///
/// Passed to [`LayoutAtlas::compute`](crate::atlas::LayoutAtlas::compute) as
/// the available space for the root node.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Viewport {
    /// Width of the host surface in logical pixels.
    pub width: f32,
    /// Height of the host surface in logical pixels.
    pub height: f32,
}

impl Viewport {
    /// Constructs a new viewport.
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

/// A snapshot of all render primitives for a single frame.
///
/// Built by the Logic thread (Evaluator + Atlas) and read by the Render
/// thread (Encoder). The Logic thread mutates the frame during construction
/// via crate-private APIs; once handed off to the Render thread (via the
/// Relay's atomic pointer swap) it is treated as immutable for the duration
/// of that frame.
///
/// The structure is intentionally SoA-friendly for batched GPU dispatch: each
/// primitive type lives in its own `Vec` so the Encoder can cast a slice
/// directly to bytes and upload it with zero copy.
///
/// [`version`](RenderFrame::version) is a monotonic counter incremented by the
/// Logic thread whenever any content changes. The Encoder compares it against
/// the version it saw on the previous frame to detect missed-dirty-frame
/// scenarios (see `EncoderSubsystem::encode_frame_from_relay`).
#[derive(Debug, Default)]
pub struct RenderFrame {
    /// Resolved geometry produced by the Atlas.
    ///
    /// Each entry is a rectangle in logical pixels, ready for the Encoder
    /// to translate into a draw command.
    rects: Vec<Rect>,

    /// Per-entry dirty state, parallel to `rects`.
    ///
    /// `dirty[i]` is `true` when `rects[i]` changed since the previous tick.
    dirty: Vec<bool>,

    /// Solid-rectangle instances populated by the Logic thread each tick.
    instances: Vec<BoxInstance>,

    /// Decorated-box instances (M21) — boxes with border/shadow/opacity.
    decorated: Vec<DecoratedBox>,

    /// Texture-sampled image instances (M21).
    textures: Vec<TextureSampler>,

    /// Text lines populated by the Logic thread each tick.
    texts: Vec<TextLine>,

    /// Monotonic version counter, incremented by the Logic thread whenever any
    /// content in this frame changes relative to the previous tick.
    ///
    /// The Encoder compares this against the last version it rendered. A version
    /// advance means the render thread skipped at least one dirty frame and must
    /// force a full redraw + text reshape to avoid displaying stale glyphs.
    version: u64,

    /// This tick's CPU scope samples (RFC-0013 "Hand-off"), piggybacked on
    /// the existing atomic frame swap instead of a dedicated channel. Empty
    /// when the `telemetry` feature is off or nothing was profiled this tick.
    telemetry: crate::telemetry::SampleBlock,
}

impl RenderFrame {
    /// Creates an empty frame.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears the frame, retaining internal buffer capacity.
    ///
    /// After the first frame, subsequent populations pay zero allocation cost
    /// as long as primitive counts stay within the high-water mark. Version is
    /// reset to zero; the Logic thread always calls [`set_version`](Self::set_version)
    /// immediately after acquiring a recycled frame.
    pub fn clear(&mut self) {
        self.rects.clear();
        self.dirty.clear();
        self.instances.clear();
        self.decorated.clear();
        self.textures.clear();
        self.texts.clear();
        self.version = 0;
        self.telemetry = crate::telemetry::SampleBlock::default();
    }

    /// Appends a resolved rectangle and its dirty state to the frame.
    pub fn push_rect(&mut self, rect: Rect, dirty: bool) {
        self.rects.push(rect);
        self.dirty.push(dirty);
    }

    /// Appends a [`BoxInstance`] to the frame.
    pub fn push_instance(&mut self, instance: BoxInstance) {
        self.instances.push(instance);
    }

    /// Appends a [`DecoratedBox`] (border/shadow/opacity) to the frame (M21).
    pub fn push_decorated(&mut self, d: DecoratedBox) {
        self.decorated.push(d);
    }

    /// Appends a [`TextureSampler`] (image) to the frame (M21).
    pub fn push_texture(&mut self, t: TextureSampler) {
        self.textures.push(t);
    }

    /// Appends a [`TextLine`] to the frame.
    pub fn push_text(&mut self, text: TextLine) {
        self.texts.push(text);
    }

    /// Sets the frame's version counter.
    pub fn set_version(&mut self, version: u64) {
        self.version = version;
    }

    /// Returns the resolved rectangles in this frame.
    #[must_use]
    pub fn rects(&self) -> &[Rect] {
        &self.rects
    }

    /// Returns the per-entry dirty state, parallel to [`rects`](Self::rects).
    #[must_use]
    pub fn dirty(&self) -> &[bool] {
        &self.dirty
    }

    /// Returns the solid-rectangle instances in this frame.
    #[must_use]
    pub fn instances(&self) -> &[BoxInstance] {
        &self.instances
    }

    /// Returns the decorated-box instances in this frame (M21).
    #[must_use]
    pub fn decorated(&self) -> &[DecoratedBox] {
        &self.decorated
    }

    /// Returns the texture-sampled image instances in this frame (M21).
    #[must_use]
    pub fn textures(&self) -> &[TextureSampler] {
        &self.textures
    }

    /// Returns the text lines in this frame.
    #[must_use]
    pub fn texts(&self) -> &[TextLine] {
        &self.texts
    }

    /// Returns the monotonic version counter for this frame.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Attaches this tick's CPU scope samples (RFC-0013 "Hand-off"),
    /// piggybacked on this frame instead of a dedicated channel.
    pub fn set_telemetry(&mut self, block: crate::telemetry::SampleBlock) {
        self.telemetry = block;
    }

    /// Returns this tick's CPU scope samples, if any were captured.
    #[must_use]
    pub fn telemetry(&self) -> &crate::telemetry::SampleBlock {
        &self.telemetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_all_fields() {
        let id = TargetId::new(0x1234_5678, 0xABCD, 0x9F00);
        assert_eq!(id.index(), 0x1234_5678);
        assert_eq!(id.generation(), 0xABCD);
        assert_eq!(id.kind(), 0x9F00);
    }

    #[test]
    fn maximum_values_do_not_overflow_neighbouring_fields() {
        let id = TargetId::new(u32::MAX, u16::MAX, u16::MAX);
        assert_eq!(id.index(), u32::MAX);
        assert_eq!(id.generation(), u16::MAX);
        assert_eq!(id.kind(), u16::MAX);
    }

    #[test]
    fn zero_id_has_all_zero_fields() {
        let id = TargetId::new(0, 0, 0);
        assert_eq!(id.as_raw(), 0);
        assert_eq!(id.index(), 0);
        assert_eq!(id.generation(), 0);
        assert_eq!(id.kind(), 0);
    }

    #[test]
    fn is_copy_and_cheap_to_clone() {
        const fn assert_copy<T: Copy>() {}
        assert_copy::<TargetId>();
        assert_eq!(std::mem::size_of::<TargetId>(), 8);
    }

    #[test]
    fn rect_contains_point_inside() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(r.contains(50.0, 30.0));
    }

    #[test]
    fn rect_does_not_contain_point_on_right_edge() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(!r.contains(110.0, 30.0), "right edge is exclusive");
    }

    #[test]
    fn rect_does_not_contain_point_outside() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(!r.contains(0.0, 0.0));
    }

    #[test]
    fn rect_union_of_disjoint_rects_covers_both() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(100.0, 200.0, 10.0, 10.0);
        let u = a.union(&b);
        assert_eq!(u, Rect::new(0.0, 0.0, 110.0, 210.0));
    }

    #[test]
    fn rect_union_with_self_is_identity() {
        let a = Rect::new(5.0, 5.0, 20.0, 30.0);
        assert_eq!(a.union(&a), a);
    }

    #[test]
    fn rect_union_is_commutative() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(-5.0, 3.0, 4.0, 50.0);
        assert_eq!(a.union(&b), b.union(&a));
    }

    #[test]
    fn rect_union_where_one_fully_contains_the_other_returns_the_larger() {
        let outer = Rect::new(0.0, 0.0, 100.0, 100.0);
        let inner = Rect::new(10.0, 10.0, 5.0, 5.0);
        assert_eq!(outer.union(&inner), outer);
        assert_eq!(inner.union(&outer), outer);
    }

    #[test]
    fn rect_union_of_overlapping_rects_merges_correctly() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 10.0, 10.0);
        assert_eq!(a.union(&b), Rect::new(0.0, 0.0, 15.0, 15.0));
    }

    #[test]
    fn rect_union_of_zero_area_rects_covers_both_corners() {
        // A degenerate (zero-size) rect can still arise from a TextLine whose
        // heuristic bounds collapse to a point; union must not panic or
        // silently drop it.
        let a = Rect::new(0.0, 0.0, 0.0, 0.0);
        let b = Rect::new(50.0, 50.0, 0.0, 0.0);
        assert_eq!(a.union(&b), Rect::new(0.0, 0.0, 50.0, 50.0));
    }

    #[test]
    fn render_frame_starts_empty() {
        let frame = RenderFrame::new();
        assert!(frame.rects().is_empty());
    }

    #[test]
    fn render_frame_clear_empties_rects() {
        let mut frame = RenderFrame::new();
        frame.push_rect(Rect::new(0.0, 0.0, 10.0, 10.0), false);
        frame.push_rect(Rect::new(10.0, 0.0, 10.0, 10.0), true);
        assert_eq!(frame.rects().len(), 2);

        frame.clear();
        assert!(frame.rects().is_empty());
        assert!(frame.dirty().is_empty());
    }

    #[test]
    fn target_kind_round_trips_through_target_id() {
        let id = TargetId::new(7, 3, TargetKind::AtlasNode as u16);
        assert_eq!(id.kind(), TargetKind::AtlasNode as u16);
        assert_eq!(id.index(), 7);
        assert_eq!(id.generation(), 3);
    }

    // ── Rect::contains edge cases ─────────────────────────────────────────────

    #[test]
    fn rect_contains_point_on_left_edge_is_inclusive() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(
            r.contains(10.0, 30.0),
            "left edge (x == rect.x) is inclusive"
        );
    }

    #[test]
    fn rect_contains_point_on_top_edge_is_inclusive() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(
            r.contains(50.0, 20.0),
            "top edge (y == rect.y) is inclusive"
        );
    }

    #[test]
    fn rect_does_not_contain_point_on_bottom_edge() {
        // Half-open: y == rect.y + rect.height is exclusive.
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(
            !r.contains(50.0, 70.0),
            "bottom edge (y == y + height) is exclusive"
        );
    }

    #[test]
    fn zero_size_rect_contains_nothing() {
        // A Rect with width=0 or height=0 has no interior; every point is outside.
        let zero_w = Rect::new(10.0, 10.0, 0.0, 50.0);
        assert!(
            !zero_w.contains(10.0, 20.0),
            "zero-width rect contains nothing"
        );

        let zero_h = Rect::new(10.0, 10.0, 50.0, 0.0);
        assert!(
            !zero_h.contains(20.0, 10.0),
            "zero-height rect contains nothing"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // comparing literal → stored literal, no arithmetic, always bit-exact
    fn rect_default_is_all_zeros() {
        let r = Rect::default();
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.width, 0.0);
        assert_eq!(r.height, 0.0);
    }

    // ── Viewport ─────────────────────────────────────────────────────────────

    #[test]
    #[allow(clippy::float_cmp)] // round-trip through Viewport::new: no arithmetic, bit-exact
    fn viewport_new_round_trips() {
        let vp = Viewport::new(1920.0, 1080.0);
        assert_eq!(vp.width, 1920.0);
        assert_eq!(vp.height, 1080.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // Default-derived zero: no arithmetic, bit-exact
    fn viewport_default_is_zero() {
        let vp = Viewport::default();
        assert_eq!(vp.width, 0.0);
        assert_eq!(vp.height, 0.0);
    }

    #[test]
    fn viewport_is_copy() {
        const fn assert_copy<T: Copy>() {}
        assert_copy::<Viewport>();
        assert_eq!(std::mem::size_of::<Viewport>(), 8);
    }

    // ── RenderFrame ───────────────────────────────────────────────────────────

    #[test]
    fn render_frame_push_rect_preserves_order() {
        let mut frame = RenderFrame::new();
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(20.0, 20.0, 30.0, 30.0);
        frame.push_rect(a, false);
        frame.push_rect(b, true);
        assert_eq!(frame.rects()[0], a);
        assert_eq!(frame.rects()[1], b);
    }

    #[test]
    fn render_frame_dirty_is_parallel_to_rects() {
        let mut frame = RenderFrame::new();
        frame.push_rect(Rect::new(0.0, 0.0, 10.0, 10.0), false);
        frame.push_rect(Rect::new(10.0, 0.0, 10.0, 10.0), true);
        frame.push_rect(Rect::new(20.0, 0.0, 10.0, 10.0), false);

        assert_eq!(frame.dirty(), &[false, true, false]);
        assert_eq!(frame.dirty().len(), frame.rects().len());
    }

    #[test]
    fn render_frame_starts_with_no_dirty_entries() {
        let frame = RenderFrame::new();
        assert!(frame.dirty().is_empty());
    }

    #[test]
    #[allow(clippy::float_cmp)] // x=99.0 stored from a literal, no arithmetic, bit-exact
    fn render_frame_clear_retains_capacity_for_reuse() {
        // Clearing a frame with N rects and immediately re-populating with N
        // rects should not reallocate. We verify correctness (no stale data),
        // not performance — allocation is observable only via Miri/asan.
        let mut frame = RenderFrame::new();
        for i in 0..10 {
            #[allow(clippy::cast_precision_loss)]
            frame.push_rect(Rect::new(i as f32, 0.0, 10.0, 10.0), false);
        }
        frame.clear();
        assert!(frame.rects().is_empty(), "clear must empty the frame");

        frame.push_rect(Rect::new(99.0, 0.0, 1.0, 1.0), true);
        assert_eq!(frame.rects().len(), 1, "can push after clear");
        assert_eq!(frame.rects()[0].x, 99.0);
        assert_eq!(frame.dirty(), &[true]);
    }
}
