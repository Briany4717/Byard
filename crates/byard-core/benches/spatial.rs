//! Performance benchmarks for the spatial hash grid.
//!
//! Run with `cargo bench --bench spatial`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use std::hint::black_box;
use std::time::Instant;

use byard_core::atlas::SpatialGrid;
use byard_core::frame::{Rect, TargetId, TargetKind};

/// Populates a grid with `count` non-overlapping rectangles arranged in
/// a grid-of-grids pattern.
///
/// Returns the populated grid plus a point known to hit one of the
/// rectangles (used by query benchmarks for hit measurements).
fn populate(count: usize) -> (SpatialGrid, (f32, f32)) {
    let mut grid = SpatialGrid::new();
    let cols = (count as f32).sqrt().ceil() as usize;

    for i in 0..count {
        let col = i % cols;
        let row = i / cols;
        let x = col as f32 * 100.0;
        let y = row as f32 * 100.0;
        let target = TargetId::new(i as u32, 0, TargetKind::AtlasNode as u16);
        grid.insert(Rect::new(x, y, 80.0, 80.0), target);
    }

    // Hit point: centre of the rectangle at index count / 2.
    let middle = count / 2;
    let col = middle % cols;
    let row = middle / cols;
    let hit_x = col as f32 * 100.0 + 40.0;
    let hit_y = row as f32 * 100.0 + 40.0;

    (grid, (hit_x, hit_y))
}

fn bench<F: FnMut()>(name: &str, iters: u64, mut f: F) {
    // Warm-up
    for _ in 0..1000 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per_op = elapsed.as_nanos() as f64 / iters as f64;

    println!("{name:55} {per_op:>10.2} ns/op   ({iters} iters)");
}

fn bench_with_setup<S, F, T>(name: &str, iters: u64, mut setup: S, mut measure: F)
where
    S: FnMut() -> T,
    F: FnMut(T),
{
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

    let per_op = total.as_nanos() as f64 / iters as f64;
    println!("{name:55} {per_op:>10.2} ns/op   ({iters} iters)");
}

fn bench_insert(count: usize) {
    bench_with_setup(
        &format!("spatial: insert {count} rects"),
        100,
        SpatialGrid::new,
        |mut grid| {
            for i in 0..count {
                let col = (i as f32) * 0.137;
                let row = (i as f32) * 0.241;
                let x = (col * 800.0) % 4000.0;
                let y = (row * 600.0) % 3000.0;
                let target = TargetId::new(i as u32, 0, TargetKind::AtlasNode as u16);
                grid.insert(Rect::new(x, y, 50.0, 50.0), target);
            }
            black_box(&grid);
        },
    );
}

fn bench_query_hit(count: usize) {
    let (grid, (x, y)) = populate(count);

    bench(
        &format!("spatial: query hit ({count} rects)"),
        1_000_000,
        || {
            black_box(grid.query(x, y));
        },
    );
}

fn bench_query_miss(count: usize) {
    let (grid, _) = populate(count);
    // Far outside any inserted rect.
    let (x, y) = (-9999.0, -9999.0);

    bench(
        &format!("spatial: query miss ({count} rects)"),
        1_000_000,
        || {
            black_box(grid.query(x, y));
        },
    );
}

fn main() {
    println!("\n=== Spatial grid benchmarks ===\n");

    println!("── Insertion ──");
    bench_insert(100);
    bench_insert(1_000);
    bench_insert(10_000);

    println!("\n── Query (hit) ──");
    bench_query_hit(100);
    bench_query_hit(1_000);
    bench_query_hit(10_000);

    println!("\n── Query (miss, far outside) ──");
    bench_query_miss(100);
    bench_query_miss(1_000);
    bench_query_miss(10_000);

    println!();
}
