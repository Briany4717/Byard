//! MaxRects bounding-box packing for the AOT atlas (RFC-0009 §4, M49).
//!
//! `byard build` bakes every statically referenced icon into one immutable
//! array-texture. Fields are packed with the **MaxRects** algorithm (Jylänki
//! 2010, Best-Short-Side-Fit heuristic): each glyph is placed in the free
//! rectangle whose shorter leftover side is smallest, then the free list is
//! split around the placement and pruned of contained rectangles. When a glyph
//! no longer fits the current layer a fresh layer is opened, so the packer
//! spans an array texture rather than failing on a full sheet.
//!
//! The dev JIT uses a simpler fixed-grid bump allocator ([`super::jit`]); this
//! tighter packer is the AOT counterpart, where the full set is known up front.

/// A box to place, identified by its index in the caller's input slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size {
    /// Width in texels.
    pub w: u32,
    /// Height in texels.
    pub h: u32,
}

/// Where a box landed: its input index and its top-left cell in the atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Index into the `sizes` slice passed to [`pack_layers`].
    pub index: usize,
    /// Top-left x in texels within `layer`.
    pub x: u32,
    /// Top-left y in texels within `layer`.
    pub y: u32,
    /// Array-texture layer.
    pub layer: u32,
}

#[derive(Clone, Copy, Debug)]
struct FreeRect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// Packs `sizes` into one or more `bin_w × bin_h` layers with MaxRects-BSSF,
/// opening a new layer whenever a box does not fit the current one. Returns one
/// [`Placement`] per input (order not significant) and the number of layers
/// used, or `None` if any single box is larger than a whole layer (unplaceable).
///
/// Placements never overlap and always lie fully within `[0, bin_w) × [0,
/// bin_h)` on their layer — both are property-checked in the tests.
#[must_use]
pub fn pack_layers(sizes: &[Size], bin_w: u32, bin_h: u32) -> Option<(Vec<Placement>, u32)> {
    if sizes
        .iter()
        .any(|s| s.w > bin_w || s.h > bin_h || s.w == 0 || s.h == 0)
    {
        return None;
    }

    // Largest-area-first placement is the standard MaxRects pre-sort; it packs
    // markedly tighter than input order while staying fully deterministic.
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| {
        let area = |i: usize| u64::from(sizes[i].w) * u64::from(sizes[i].h);
        area(b).cmp(&area(a)).then(a.cmp(&b))
    });

    let mut placements = Vec::with_capacity(sizes.len());
    let mut layer = 0u32;
    let mut free = vec![FreeRect {
        x: 0,
        y: 0,
        w: bin_w,
        h: bin_h,
    }];

    for &i in &order {
        let size = sizes[i];
        let mut spot = find_best(&free, size);
        if spot.is_none() {
            // Current layer is full for this box — open the next one.
            layer += 1;
            free = vec![FreeRect {
                x: 0,
                y: 0,
                w: bin_w,
                h: bin_h,
            }];
            spot = find_best(&free, size);
        }
        // A box that fits an empty layer (guaranteed by the bounds check above)
        // always has a spot after the reset, so this never panics.
        let node = spot.expect("a box within bin bounds always fits a fresh layer");
        placements.push(Placement {
            index: i,
            x: node.x,
            y: node.y,
            layer,
        });
        place(&mut free, node.x, node.y, size);
    }

    Some((placements, layer + 1))
}

/// Best-Short-Side-Fit: the free rect that leaves the smallest shorter leftover
/// side, tie-broken by the longer leftover side, then by position for
/// determinism. Returns the chosen top-left as a zero-size `FreeRect`.
fn find_best(free: &[FreeRect], size: Size) -> Option<FreeRect> {
    let mut best: Option<(u32, u32, FreeRect)> = None;
    for f in free {
        if f.w < size.w || f.h < size.h {
            continue;
        }
        let leftover_h = f.w - size.w;
        let leftover_v = f.h - size.h;
        let short = leftover_h.min(leftover_v);
        let long = leftover_h.max(leftover_v);
        let cand = (
            short,
            long,
            FreeRect {
                x: f.x,
                y: f.y,
                w: 0,
                h: 0,
            },
        );
        let better = match &best {
            None => true,
            Some((bs, bl, br)) => (short, long, (f.x, f.y)) < (*bs, *bl, (br.x, br.y)),
        };
        if better {
            best = Some(cand);
        }
    }
    best.map(|(_, _, r)| r)
}

/// Splits every free rect that overlaps the just-placed box into up to four
/// residual rects, then prunes any free rect fully contained in another
/// (the MaxRects maintenance step that keeps the free list maximal).
fn place(free: &mut Vec<FreeRect>, px: u32, py: u32, size: Size) {
    let placed = FreeRect {
        x: px,
        y: py,
        w: size.w,
        h: size.h,
    };
    let mut next = Vec::with_capacity(free.len() + 4);
    for f in free.drain(..) {
        if !overlaps(&f, &placed) {
            next.push(f);
            continue;
        }
        // Left slab.
        if placed.x > f.x {
            next.push(FreeRect {
                x: f.x,
                y: f.y,
                w: placed.x - f.x,
                h: f.h,
            });
        }
        // Right slab.
        if placed.x + placed.w < f.x + f.w {
            next.push(FreeRect {
                x: placed.x + placed.w,
                y: f.y,
                w: (f.x + f.w) - (placed.x + placed.w),
                h: f.h,
            });
        }
        // Top slab.
        if placed.y > f.y {
            next.push(FreeRect {
                x: f.x,
                y: f.y,
                w: f.w,
                h: placed.y - f.y,
            });
        }
        // Bottom slab.
        if placed.y + placed.h < f.y + f.h {
            next.push(FreeRect {
                x: f.x,
                y: placed.y + placed.h,
                w: f.w,
                h: (f.y + f.h) - (placed.y + placed.h),
            });
        }
    }
    prune(&mut next);
    *free = next;
}

fn overlaps(a: &FreeRect, b: &FreeRect) -> bool {
    a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y
}

fn contains(outer: &FreeRect, inner: &FreeRect) -> bool {
    inner.x >= outer.x
        && inner.y >= outer.y
        && inner.x + inner.w <= outer.x + outer.w
        && inner.y + inner.h <= outer.y + outer.h
}

fn prune(rects: &mut Vec<FreeRect>) {
    let mut i = 0;
    while i < rects.len() {
        let mut removed = false;
        let mut j = i + 1;
        while j < rects.len() {
            if contains(&rects[j], &rects[i]) {
                rects.swap_remove(i);
                removed = true;
                break;
            }
            if contains(&rects[i], &rects[j]) {
                rects.swap_remove(j);
            } else {
                j += 1;
            }
        }
        if !removed {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overlaps_any(ps: &[Placement], sizes: &[Size]) -> bool {
        for a in ps {
            for b in ps {
                if a.index >= b.index || a.layer != b.layer {
                    continue;
                }
                let (sa, sb) = (sizes[a.index], sizes[b.index]);
                let ra = FreeRect {
                    x: a.x,
                    y: a.y,
                    w: sa.w,
                    h: sa.h,
                };
                let rb = FreeRect {
                    x: b.x,
                    y: b.y,
                    w: sb.w,
                    h: sb.h,
                };
                if overlaps(&ra, &rb) {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn packs_uniform_cells_into_one_layer_without_overlap() {
        let sizes = vec![Size { w: 32, h: 32 }; 16];
        let (ps, layers) = pack_layers(&sizes, 128, 128).unwrap();
        assert_eq!(ps.len(), 16);
        assert_eq!(layers, 1, "16 32px cells fit a 128² (4×4) layer exactly");
        assert!(!overlaps_any(&ps, &sizes), "placements must not overlap");
        for p in &ps {
            assert!(p.x + 32 <= 128 && p.y + 32 <= 128, "must stay in bounds");
        }
    }

    #[test]
    fn opens_a_new_layer_when_the_sheet_is_full() {
        // 5 cells of 32px into a 64² sheet: only 4 fit per layer → 2 layers.
        let sizes = vec![Size { w: 32, h: 32 }; 5];
        let (ps, layers) = pack_layers(&sizes, 64, 64).unwrap();
        assert_eq!(layers, 2);
        assert!(!overlaps_any(&ps, &sizes));
        let on_layer_0 = ps.iter().filter(|p| p.layer == 0).count();
        assert_eq!(on_layer_0, 4, "a 64² sheet holds four 32px cells");
    }

    #[test]
    fn packs_mixed_sizes_without_overlap_or_out_of_bounds() {
        let sizes = vec![
            Size { w: 64, h: 32 },
            Size { w: 32, h: 64 },
            Size { w: 32, h: 32 },
            Size { w: 48, h: 48 },
            Size { w: 16, h: 96 },
        ];
        let (ps, _layers) = pack_layers(&sizes, 128, 128).unwrap();
        assert_eq!(ps.len(), sizes.len());
        assert!(!overlaps_any(&ps, &sizes));
        for p in &ps {
            let s = sizes[p.index];
            assert!(p.x + s.w <= 128 && p.y + s.h <= 128);
        }
    }

    #[test]
    fn is_deterministic() {
        let sizes = vec![
            Size { w: 40, h: 20 },
            Size { w: 20, h: 40 },
            Size { w: 32, h: 32 },
            Size { w: 50, h: 50 },
        ];
        let a = pack_layers(&sizes, 100, 100).unwrap();
        let b = pack_layers(&sizes, 100, 100).unwrap();
        assert_eq!(a, b, "same input must pack identically");
    }

    #[test]
    fn rejects_a_box_larger_than_a_whole_layer() {
        let sizes = vec![Size { w: 200, h: 32 }];
        assert!(pack_layers(&sizes, 128, 128).is_none());
    }

    #[test]
    fn empty_input_packs_to_zero_boxes_one_layer() {
        let (ps, layers) = pack_layers(&[], 128, 128).unwrap();
        assert!(ps.is_empty());
        assert_eq!(layers, 1);
    }
}
