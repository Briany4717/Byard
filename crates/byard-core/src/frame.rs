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
    // Future: EncoderPrimitive = 2, etc.
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
/// Built by the Logic thread (evaluator + atlas) and read by the Render
/// thread (encoder). The Logic thread mutates the frame during
/// construction via crate-private APIs; once handed off to the Render
/// thread (via the Relay's atomic pointer swap) it is treated as
/// immutable for the duration of that frame.
///
/// Produced by the logic thread (evaluator + atlas) and consumed by the render
/// thread (encoder) via an atomic pointer swap managed by the relay.
///
/// Phase 1 only carries the resolved rectangles produced by the Atlas. As
/// the Encoder grows, additional primitives (text glyph runs, decorated
/// boxes, texture samplers) will be added as parallel `Vec`s — the
/// structure is intentionally SoA-friendly for batched GPU dispatch.
#[derive(Debug, Default)]
pub struct RenderFrame {
    /// Resolved geometry produced by the Atlas.
    ///
    /// Each entry is a rectangle in logical pixels, ready for the Encoder
    /// to translate into a draw command. Order is determined by Atlas tree
    /// traversal (currently pre-order over the layout tree; will become
    /// Z-bin order in a future sub-issue).
    rects: Vec<Rect>,
}

impl RenderFrame {
    /// Creates an empty frame.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears the frame, retaining internal capacity.
    ///
    /// After the first frame, subsequent populations pay zero allocation
    /// cost as long as primitive counts stay within the high-water mark.
    pub fn clear(&mut self) {
        self.rects.clear();
    }

    /// Appends a resolved rectangle to the frame.
    ///
    /// Called by the Atlas during `populate_frame`. Not part of the public
    /// engine API — external code reads frames, it does not build them.
    pub(crate) fn push_rect(&mut self, rect: Rect) {
        self.rects.push(rect);
    }

    /// Returns the resolved rectangles in this frame.
    #[must_use]
    pub fn rects(&self) -> &[Rect] {
        &self.rects
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
    fn render_frame_starts_empty() {
        let frame = RenderFrame::new();
        assert!(frame.rects().is_empty());
    }

    #[test]
    fn render_frame_clear_empties_rects() {
        let mut frame = RenderFrame::new();
        frame.push_rect(Rect::new(0.0, 0.0, 10.0, 10.0));
        frame.push_rect(Rect::new(10.0, 0.0, 10.0, 10.0));
        assert_eq!(frame.rects().len(), 2);

        frame.clear();
        assert!(frame.rects().is_empty());
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
        frame.push_rect(a);
        frame.push_rect(b);
        assert_eq!(frame.rects()[0], a);
        assert_eq!(frame.rects()[1], b);
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
            frame.push_rect(Rect::new(i as f32, 0.0, 10.0, 10.0));
        }
        frame.clear();
        assert!(frame.rects().is_empty(), "clear must empty the frame");

        frame.push_rect(Rect::new(99.0, 0.0, 1.0, 1.0));
        assert_eq!(frame.rects().len(), 1, "can push after clear");
        assert_eq!(frame.rects()[0].x, 99.0);
    }
}
