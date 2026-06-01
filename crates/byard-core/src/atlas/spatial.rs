//! Spatial hash grid for `O(1)` amortised hit-testing.
//!
//! See RFC-0001 §4.1 for the architectural intent. The grid maps screen
//! coordinates to [`TargetId`]s so the engine can resolve pointer events
//! (clicks, hovers, drags) without walking the UI tree.
//!
//! # How it works
//!
//! The 2D plane is divided into uniform cells of [`CELL_SIZE`] logical
//! pixels. Each inserted [`Rect`] registers in every cell its
//! axis-aligned bounding box touches. A point query computes which cell
//! the point falls in, iterates the entries in that single cell, and
//! returns the matching [`TargetId`].
//!
//! # Z-order contract
//!
//! When two rectangles overlap and a point falls inside both, the entry
//! inserted **later** wins — this matches the convention that later
//! draw calls render on top.
//!
//! **The caller is responsible** for inserting entries in the same order
//! they will be dispatched by the Encoder. The Atlas's [`populate_frame`]
//! visits the tree in pre-order, which produces parent → child ordering;
//! children render on top of parents, which is the natural Z-order for
//! a UI tree.
//!
//! If the engine ever supports stacking contexts (floating modals,
//! tooltips, popovers), the orchestrator must insert those entries
//! after the rest of the tree, so they keep priority on hit-testing —
//! mirroring exactly what the renderer will do.
//!
//! # Large rectangles
//!
//! A rectangle that spans multiple cells is inserted into every cell its
//! AABB touches. This is the standard approach for spatial hash grids
//! and keeps queries fast (one cell lookup per point). For very large
//! rectangles spanning many cells, memory cost grows linearly with the
//! number of touched cells; in typical UI workloads this is bounded by
//! ~9 cells per rectangle.
//!
//! [`populate_frame`]: super::layout::LayoutAtlas::populate_frame

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::frame::{Rect, TargetId};

/// Side length of each grid cell in logical pixels.
///
/// Chosen as a power-of-two divisor of the typical UI element size.
/// Smaller cells reduce per-cell occupancy at the cost of more cells
/// per rectangle; larger cells do the opposite. 128 is a balanced
/// choice for typical desktop UIs (button-sized targets fit in one
/// cell; container backgrounds span 4–9 cells).
pub const CELL_SIZE: f32 = 128.0;

/// Stack-allocated capacity for entries per cell.
///
/// UI workloads typically have 1–3 entries per 128×128 cell. Allocating
/// inline for up to 4 entries avoids heap allocation for the common
/// case, which dominates frame-time variance.
const INLINE_ENTRIES: usize = 4;

/// A single entry in the spatial grid.
#[derive(Debug, Clone, Copy, PartialEq)]
struct GridEntry {
    rect: Rect,
    target: TargetId,
}

/// Spatial hash grid for `O(1)` hit-testing.
///
/// See the module documentation for design rationale.
#[derive(Debug, Default)]
pub struct SpatialGrid {
    cells: FxHashMap<u64, SmallVec<[GridEntry; INLINE_ENTRIES]>>,
}

impl SpatialGrid {
    /// Creates an empty grid.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cells: FxHashMap::default(),
        }
    }

    /// Clears the grid, retaining internal capacity.
    ///
    /// Use this between frames so allocations from previous layouts are
    /// reused.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Returns the number of cells currently holding at least one entry.
    ///
    /// Primarily useful for diagnostics and tests.
    #[must_use]
    pub fn occupied_cells(&self) -> usize {
        self.cells.len()
    }

    /// Inserts a rectangle into every cell its AABB touches.
    ///
    /// Insertion order matters for overlapping rectangles — the later
    /// entry wins point queries. See the module-level Z-order contract.
    pub fn insert(&mut self, rect: Rect, target: TargetId) {
        let entry = GridEntry { rect, target };

        let (min_col, min_row) = Self::quantize(rect.x, rect.y);
        let (max_col, max_row) = Self::quantize(
            (rect.x + rect.width).next_down(),
            (rect.y + rect.height).next_down(),
        );

        for row in min_row..=max_row {
            for col in min_col..=max_col {
                let key = Self::pack_key(row, col);
                self.cells.entry(key).or_default().push(entry);
            }
        }
    }

    /// Returns the [`TargetId`] of the topmost rectangle containing the
    /// point, or `None` if no rectangle contains it.
    ///
    /// "Topmost" means the entry inserted most recently into the
    /// containing cell. See the module-level Z-order contract.
    #[must_use]
    pub fn query(&self, px: f32, py: f32) -> Option<TargetId> {
        let (col, row) = Self::quantize(px, py);
        let key = Self::pack_key(row, col);

        let cell = self.cells.get(&key)?;

        // Iterate in reverse so the most recently inserted entry wins.
        cell.iter()
            .rev()
            .find(|entry| entry.rect.contains(px, py))
            .map(|entry| entry.target)
    }

    /// Maps a world-space coordinate to a cell index (col, row).
    ///
    /// Uses `.floor()` rather than `as i32` truncation so negative
    /// coordinates produce the geometrically correct cell. For example,
    /// `quantize(-10.0, -10.0)` with `CELL_SIZE = 128.0` returns
    /// `(-1, -1)`, not `(0, 0)`.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn quantize(x: f32, y: f32) -> (i32, i32) {
        // Truncation from f32 → i32 is bounded by the realistic
        // coordinate range of a UI surface (≤ a few thousand pixels in
        // logical units). f32 can represent every i32 up to 2^24
        // exactly; we are nowhere near that limit.
        let col = (x / CELL_SIZE).floor() as i32;
        let row = (y / CELL_SIZE).floor() as i32;
        (col, row)
    }

    /// Packs two signed cell coordinates into a single `u64` key.
    ///
    /// Uses bit-level reinterpretation (`i32 as u32`) so negative
    /// coordinates produce distinct keys from their positive
    /// counterparts. `(-1, -1)` ≠ `(0, 0)` ≠ `(1, 1)` etc.
    #[inline]
    #[allow(clippy::cast_sign_loss)]
    const fn pack_key(row: i32, col: i32) -> u64 {
        // Bit-level reinterpretation: i32 → u32 preserves the bit pattern,
        // so negative coordinates produce distinct keys from their positive
        // counterparts. This is intentional, not a sign-loss bug.
        ((row as u32 as u64) << 32) | (col as u32 as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::TargetKind;

    fn make_target(index: u32) -> TargetId {
        TargetId::new(index, 0, TargetKind::AtlasNode as u16)
    }

    #[test]
    fn empty_grid_returns_none() {
        let grid = SpatialGrid::new();
        assert_eq!(grid.query(0.0, 0.0), None);
        assert_eq!(grid.query(100.0, 100.0), None);
    }

    #[test]
    fn query_inside_inserted_rect_returns_target() {
        let mut grid = SpatialGrid::new();
        let target = make_target(1);
        grid.insert(Rect::new(0.0, 0.0, 100.0, 100.0), target);

        assert_eq!(grid.query(50.0, 50.0), Some(target));
    }

    #[test]
    fn query_outside_inserted_rect_returns_none() {
        let mut grid = SpatialGrid::new();
        grid.insert(Rect::new(0.0, 0.0, 100.0, 100.0), make_target(1));

        // Far away, in a different cell.
        assert_eq!(grid.query(1000.0, 1000.0), None);
    }

    #[test]
    fn query_at_top_left_corner_hits() {
        let mut grid = SpatialGrid::new();
        let target = make_target(1);
        grid.insert(Rect::new(10.0, 20.0, 50.0, 50.0), target);

        // Top-left edges are inclusive (matches Rect::contains).
        assert_eq!(grid.query(10.0, 20.0), Some(target));
    }

    #[test]
    fn query_on_bottom_right_edge_misses() {
        let mut grid = SpatialGrid::new();
        grid.insert(Rect::new(10.0, 20.0, 50.0, 50.0), make_target(1));

        // Bottom-right edges are exclusive (matches Rect::contains).
        assert_eq!(grid.query(60.0, 70.0), None);
    }

    #[test]
    fn overlapping_rects_return_last_inserted() {
        let mut grid = SpatialGrid::new();
        let bottom = make_target(1);
        let top = make_target(2);

        // Both contain the point (50, 50).
        grid.insert(Rect::new(0.0, 0.0, 100.0, 100.0), bottom);
        grid.insert(Rect::new(0.0, 0.0, 100.0, 100.0), top);

        assert_eq!(grid.query(50.0, 50.0), Some(top), "topmost wins");
    }

    #[test]
    fn negative_coordinates_quantize_correctly() {
        // (-10, -10) sits in cell (-1, -1) when CELL_SIZE = 128.
        // Truncation via `as i32` would put it in cell (0, 0), so this
        // test exists to catch a regression to truncation-based cells.
        let mut grid = SpatialGrid::new();
        let target = make_target(1);
        grid.insert(Rect::new(-20.0, -20.0, 30.0, 30.0), target);

        assert_eq!(grid.query(-10.0, -10.0), Some(target));
    }

    #[test]
    fn large_rect_spans_multiple_cells() {
        // 400×300 with CELL_SIZE=128 → spans 4 cols × 3 rows = 12 cells.
        let mut grid = SpatialGrid::new();
        let target = make_target(1);
        grid.insert(Rect::new(0.0, 0.0, 400.0, 300.0), target);

        // Hit-test from a cell far from the origin.
        assert_eq!(grid.query(350.0, 250.0), Some(target));
        // Each cell that the AABB touches holds an entry.
        assert_eq!(grid.occupied_cells(), 12);
    }

    #[test]
    fn small_rect_fits_in_one_cell() {
        let mut grid = SpatialGrid::new();
        grid.insert(Rect::new(10.0, 10.0, 50.0, 50.0), make_target(1));
        assert_eq!(grid.occupied_cells(), 1);
    }

    #[test]
    fn rect_exactly_one_cell_wide_spans_one_cell() {
        // Rect [0, 128) × [0, 128) — exactly one cell. The exclusive
        // upper bound (128.0) must NOT spill into cell (1, _).
        let mut grid = SpatialGrid::new();
        grid.insert(Rect::new(0.0, 0.0, CELL_SIZE, CELL_SIZE), make_target(1));
        assert_eq!(
            grid.occupied_cells(),
            1,
            "half-open bound must not spill into next cell"
        );
    }

    #[test]
    fn rect_crossing_cell_boundary_spans_two_cells() {
        // Rect from (120, 0) to (140, 50) crosses x=128, the boundary
        // between cell (0,0) and cell (1,0).
        let mut grid = SpatialGrid::new();
        grid.insert(Rect::new(120.0, 0.0, 20.0, 50.0), make_target(1));
        assert_eq!(grid.occupied_cells(), 2);
    }

    #[test]
    fn clear_empties_grid_but_query_still_works() {
        let mut grid = SpatialGrid::new();
        grid.insert(Rect::new(0.0, 0.0, 100.0, 100.0), make_target(1));
        assert_eq!(grid.occupied_cells(), 1);

        grid.clear();
        assert_eq!(grid.occupied_cells(), 0);
        assert_eq!(grid.query(50.0, 50.0), None);

        grid.insert(Rect::new(0.0, 0.0, 100.0, 100.0), make_target(2));
        assert_eq!(grid.query(50.0, 50.0), Some(make_target(2)));
    }

    #[test]
    fn pack_key_distinguishes_signed_coordinates() {
        // Sanity check: -1 and 0 must produce distinct keys.
        let key_neg = SpatialGrid::pack_key(-1, -1);
        let key_zero = SpatialGrid::pack_key(0, 0);
        let key_pos = SpatialGrid::pack_key(1, 1);

        assert_ne!(key_neg, key_zero);
        assert_ne!(key_zero, key_pos);
        assert_ne!(key_neg, key_pos);
    }

    #[test]
    fn quantize_handles_negative_coordinates_correctly() {
        // (-10, -10) → cell (-1, -1) when CELL_SIZE = 128.
        assert_eq!(SpatialGrid::quantize(-10.0, -10.0), (-1, -1));
        // (0, 0) → cell (0, 0).
        assert_eq!(SpatialGrid::quantize(0.0, 0.0), (0, 0));
        // (127.9, 127.9) → cell (0, 0) (last point inside the first cell).
        assert_eq!(SpatialGrid::quantize(127.9, 127.9), (0, 0));
        // (128.0, 128.0) → cell (1, 1) (boundary).
        assert_eq!(SpatialGrid::quantize(128.0, 128.0), (1, 1));
    }

    #[test]
    fn many_non_overlapping_rects_all_findable() {
        // Stress test: 100 rects in a 10x10 grid, each in its own cell.
        let mut grid = SpatialGrid::new();
        #[allow(clippy::cast_precision_loss)]
        {
            for row in 0..10_u32 {
                for col in 0..10_u32 {
                    let x = col as f32 * 200.0;
                    let y = row as f32 * 200.0;
                    let target = make_target(row * 10 + col);
                    grid.insert(Rect::new(x, y, 100.0, 100.0), target);
                }
            }

            // Every rect must be findable at its center.
            for row in 0..10_u32 {
                for col in 0..10_u32 {
                    let center_x = col as f32 * 200.0 + 50.0;
                    let center_y = row as f32 * 200.0 + 50.0;
                    let expected = make_target(row * 10 + col);
                    assert_eq!(
                        grid.query(center_x, center_y),
                        Some(expected),
                        "rect at row={row} col={col} not found"
                    );
                }
            }
        }
    }

    #[test]
    fn clear_and_reinsert_does_not_leak_stale_entries() {
        // Simulates a frame cycle: insert rects, clear, insert new rects.
        // The old rects must be unreachable after clear + re-insert.
        let mut grid = SpatialGrid::new();

        // Frame 1: rect at (0, 0).
        let old = make_target(1);
        grid.insert(Rect::new(0.0, 0.0, 100.0, 100.0), old);
        assert_eq!(grid.query(50.0, 50.0), Some(old));

        // Simulate layout recompute: clear and insert new geometry.
        grid.clear();

        // Frame 2: no rect at (0, 0); rect moved to (500, 500).
        let new = make_target(2);
        grid.insert(Rect::new(500.0, 500.0, 100.0, 100.0), new);

        // Old position must return None — no stale ghost.
        assert_eq!(
            grid.query(50.0, 50.0),
            None,
            "stale entry from frame 1 must not survive clear",
        );

        // New position must return the new target.
        assert_eq!(grid.query(550.0, 550.0), Some(new));
    }

    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn repeated_clear_reinsert_cycles_stay_correct() {
        // Run multiple frame cycles to verify no accumulation of stale data.
        let mut grid = SpatialGrid::new();

        for frame in 0..10_u32 {
            grid.clear();

            let x = frame as f32 * 50.0;
            let target = make_target(frame);
            grid.insert(Rect::new(x, 0.0, 40.0, 40.0), target);

            // Current frame's rect is findable.
            assert_eq!(
                grid.query(x + 20.0, 20.0),
                Some(target),
                "frame {frame}: current rect not found",
            );

            // Previous frame's rect (if any) must be gone.
            if frame > 0 {
                let prev_x = (frame - 1) as f32 * 50.0;
                assert_eq!(
                    grid.query(prev_x + 20.0, 20.0),
                    None,
                    "frame {frame}: stale rect from frame {} still present",
                    frame - 1,
                );
            }
        }
    }
}
