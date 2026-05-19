//! Performance benchmarks for the Atlas subsystem.
//!
//! Run with `cargo bench --bench atlas`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use std::hint::black_box;
use std::time::Instant;

use byard_core::atlas::{AtlasNodeId, ContainerStyle, LayoutAtlas, LeafSize};
use byard_core::frame::{TargetId, TargetKind, Viewport};

/// Builds a deep, balanced tree representative of real UI hierarchies.
///
/// Each non-leaf node has `branch_factor` children, nested to `depth`
/// levels. Total leaf count is `branch_factor^depth`.
///
/// Examples:
/// - depth=3, branch=4 → 64 leaves, depth 3 (small panel)
/// - depth=4, branch=5 → 625 leaves, depth 4 (medium app view)
/// - depth=5, branch=5 → 3125 leaves, depth 5 (full IDE)
///
/// These match the shape of real applications measured publicly:
/// VS Code averages depth 12-18, Figma depth 8-15. We stop at depth 5
/// because the recompute-vs-full speedup saturates beyond that.
fn build_deep_tree_building(depth: u32, branch_factor: u32) -> LayoutAtlas {
    fn build_subtree(atlas: &mut LayoutAtlas, depth: u32, branch_factor: u32) -> AtlasNodeId {
        if depth == 0 {
            atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap()
        } else {
            let children: Vec<AtlasNodeId> = (0..branch_factor)
                .map(|_| build_subtree(atlas, depth - 1, branch_factor))
                .collect();
            atlas
                .add_container(ContainerStyle::new(None, None), &children)
                .unwrap()
        }
    }

    let mut atlas = LayoutAtlas::new();
    let root = build_subtree(&mut atlas, depth, branch_factor);
    atlas.set_root(root);
    atlas
}

fn build_deep_tree_computed(depth: u32, branch_factor: u32) -> (LayoutAtlas, TargetId) {
    let mut atlas = build_deep_tree_building(depth, branch_factor);
    atlas.compute(Viewport::new(1024.0, 768.0)).unwrap();

    // Target a deep leaf (last one created during recursion → highest index).
    let leaf_index = atlas.next_target_index().wrapping_sub(1);
    let dirty_target = TargetId::new(
        leaf_index,
        atlas.current_generation(),
        TargetKind::AtlasNode as u16,
    );

    (atlas, dirty_target)
}

fn bench_deep_full(name: &str, depth: u32, branch: u32, iters: u64) {
    bench_with_setup(
        name,
        iters,
        || build_deep_tree_building(depth, branch),
        |mut atlas| {
            atlas.compute(Viewport::new(1024.0, 768.0)).unwrap();
            black_box(&atlas);
        },
    );
}

fn bench_deep_incremental(name: &str, depth: u32, branch: u32, iters: u64) {
    bench_with_setup(
        name,
        iters,
        || build_deep_tree_computed(depth, branch),
        |(mut atlas, dirty)| {
            atlas.mark_dirty_all(&[dirty]);
            atlas.recompute_dirty(Viewport::new(1024.0, 768.0)).unwrap();
            black_box(&atlas);
        },
    );
}

/// Builds a balanced tree with the given number of leaf nodes.
///
/// Returns the atlas (already computed once) plus the `TargetId` pointing
/// at one specific leaf — used as the "dirty" target in the incremental
/// benchmark.
/// Builds a balanced tree with `leaf_count` leaves, in `Building` state.
fn build_tree_building(leaf_count: usize) -> LayoutAtlas {
    let mut atlas = LayoutAtlas::new();

    let mut leaves = Vec::with_capacity(leaf_count);
    for _ in 0..leaf_count {
        leaves.push(atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap());
    }

    let root = atlas
        .add_container(ContainerStyle::new(Some(1000.0), Some(1000.0)), &leaves)
        .unwrap();
    atlas.set_root(root);
    atlas
}

/// Builds a balanced tree and runs the initial compute, leaving the
/// atlas in `Computed` state. Returns the `TargetId` of the middle leaf
/// for the dirty-recompute benchmark.
fn build_tree_computed(leaf_count: usize) -> (LayoutAtlas, TargetId) {
    let mut atlas = build_tree_building(leaf_count);
    atlas.compute(Viewport::new(1024.0, 768.0)).unwrap();

    let middle = leaf_count / 2;
    let dirty_target = TargetId::new(
        middle as u32,
        atlas.current_generation(),
        TargetKind::AtlasNode as u16,
    );

    (atlas, dirty_target)
}

fn bench_full_recompute(name: &str, leaf_count: usize, iters: u64) {
    bench_with_setup(
        name,
        iters,
        || build_tree_building(leaf_count),
        |mut atlas| {
            atlas.compute(Viewport::new(1024.0, 768.0)).unwrap();
            black_box(&atlas);
        },
    );
}

fn bench_incremental_recompute(name: &str, leaf_count: usize, iters: u64) {
    bench_with_setup(
        name,
        iters,
        || build_tree_computed(leaf_count),
        |(mut atlas, dirty)| {
            atlas.mark_dirty_all(&[dirty]);
            atlas.recompute_dirty(Viewport::new(1024.0, 768.0)).unwrap();
            black_box(&atlas);
        },
    );
}

fn bench_with_setup<S, F, T>(name: &str, iters: u64, mut setup: S, mut measure: F)
where
    S: FnMut() -> T,
    F: FnMut(T),
{
    // Warm-up
    for _ in 0..100 {
        let state = setup();
        measure(state);
    }

    let mut total = std::time::Duration::ZERO;
    for _ in 0..iters {
        let state = setup();
        let start = Instant::now();
        measure(state);
        total += start.elapsed();
    }

    let nanos_per_op = total.as_nanos() as f64 / iters as f64;
    let micros_per_op = nanos_per_op / 1000.0;
    println!("{name:60} {micros_per_op:>10.3} µs/op   ({iters} iters)");
}

fn main() {
    println!("\n=== Atlas recompute benchmarks ===\n");

    println!("── Flat trees (worst case for incremental layout) ──");
    bench_full_recompute("atlas: flat full compute (10 leaves)", 10, 10_000);
    bench_incremental_recompute("atlas: flat incremental 1/10", 10, 10_000);

    bench_full_recompute("atlas: flat full compute (100 leaves)", 100, 1_000);
    bench_incremental_recompute("atlas: flat incremental 1/100", 100, 1_000);

    bench_full_recompute("atlas: flat full compute (1000 leaves)", 1000, 100);
    bench_incremental_recompute("atlas: flat incremental 1/1000", 1000, 100);

    println!("\n── Deep balanced trees (realistic UI hierarchies) ──");
    println!("\n  Small panel: depth=3, branch=4 → 64 leaves");
    bench_deep_full("atlas: deep full compute (3x4 = 64 leaves)", 3, 4, 10_000);
    bench_deep_incremental("atlas: deep incremental 1/64", 3, 4, 10_000);

    println!("\n  Medium app view: depth=4, branch=5 → 625 leaves");
    bench_deep_full("atlas: deep full compute (4x5 = 625 leaves)", 4, 5, 1_000);
    bench_deep_incremental("atlas: deep incremental 1/625", 4, 5, 1_000);

    println!("\n  Full IDE: depth=5, branch=5 → 3125 leaves");
    bench_deep_full("atlas: deep full compute (5x5 = 3125 leaves)", 5, 5, 100);
    bench_deep_incremental("atlas: deep incremental 1/3125", 5, 5, 100);

    println!();
}
