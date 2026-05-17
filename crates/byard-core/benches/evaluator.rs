#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
//! Performance benchmarks for the Evaluator subsystem.
//!
//! Run with `cargo bench`.

use std::hint::black_box;
use std::time::Instant;

use byard_core::evaluator::{Signal, TargetId, ViewArena};

fn bench<F: FnMut()>(name: &str, iters: u64, ops_per_iter: u64, mut f: F) {
    // Warm-up
    for _ in 0..1000 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();

    let total_ops = iters * ops_per_iter;
    let per_op = elapsed.as_nanos() as f64 / total_ops as f64;
    let per_batch = elapsed.as_nanos() as f64 / iters as f64;

    if ops_per_iter == 1 {
        println!("{name:50} {per_op:>10.2} ns/op   ({iters} iters)");
    } else {
        println!(
            "{name:50} {per_op:>10.2} ns/op   {per_batch:>10.2} ns/batch   ({iters} iters x {ops_per_iter} ops)"
        );
    }
}

fn main() {
    println!("\n=== Evaluator benchmarks ===\n");

    // ── ViewArena allocation ─────────────────────────────────────────────
    bench(
        "arena: alloc u64 (trivially-droppable)",
        1_000_000,
        100,
        || {
            let arena = ViewArena::new();
            for i in 0..100_u64 {
                black_box(arena.alloc(black_box(i)));
            }
        },
    );

    bench(
        "arena: alloc String (drop-registered)",
        1_000_000,
        100,
        || {
            let arena = ViewArena::new();
            for _ in 0..100 {
                black_box(arena.alloc(String::from("hello")));
            }
        },
    );

    bench(
        "arena: apoptosis with 1000 String drops",
        10_000,
        1000,
        || {
            let arena = ViewArena::new();
            for _ in 0..1000 {
                arena.alloc(String::from("x"));
            }
            drop(black_box(arena));
        },
    );

    let arena = ViewArena::new();
    let signal = Signal::new_in(&arena, 0_u64);

    bench("signal: read u64", 10_000_000, 1, || {
        black_box(signal.read(|v| *v));
    });

    bench("signal: write u64", 10_000_000, 1, || {
        signal.write(|v| *v = black_box(42));
    });

    bench("signal: write with increment", 10_000_000, 1, || {
        signal.write(|v| *v = v.wrapping_add(1));
    });

    bench("signal: new_in (full allocation)", 1_000_000, 100, || {
        let arena = ViewArena::new();
        for i in 0..100_u64 {
            black_box(Signal::new_in(&arena, i));
        }
    });

    bench("signal: subscribe 1000 targets", 10_000, 1000, || {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u64);
        for i in 0..1000 {
            signal.subscribe(TargetId::new(i, 0, 0));
        }
    });

    bench(
        "signal: 100 signals × 10 subs × 100 writes",
        1_000,
        100 * 10 + 100 * 100, // subscribes + writes
        || {
            let arena = ViewArena::new();
            let mut signals = Vec::with_capacity(100);
            for i in 0..100_u64 {
                signals.push(Signal::new_in(&arena, i));
            }

            for (i, sig) in signals.iter().enumerate() {
                for j in 0..10 {
                    sig.subscribe(TargetId::new(i as u32, j, 0));
                }
            }

            for _ in 0..100 {
                for (i, sig) in signals.iter().enumerate() {
                    sig.write(|v| *v = i as u64);
                }
            }
        },
    );

    println!();
}
