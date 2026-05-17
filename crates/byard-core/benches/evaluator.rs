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

    println!("{name:50} {per_op:>10.2} ns/op   ({iters} iters)");
}

fn main() {
    println!("\n=== Evaluator benchmarks ===\n");

    // ── ViewArena allocation ─────────────────────────────────────────────
    bench("arena: alloc u64 (trivially-droppable)", 1_000_000, || {
        let mut arena = ViewArena::new();
        for i in 0..100 {
            black_box(arena.alloc(black_box(i as u64)));
        }
    });

    bench("arena: alloc String (drop-registered)", 1_000_000, || {
        let mut arena = ViewArena::new();
        for _ in 0..100 {
            black_box(arena.alloc(String::from("hello")));
        }
    });

    // ── Apoptosis (drop entire arena) ────────────────────────────────────
    bench("arena: apoptosis with 1000 String drops", 10_000, || {
        let mut arena = ViewArena::new();
        for _ in 0..1000 {
            arena.alloc(String::from("x"));
        }
        drop(black_box(arena));
    });

    // ── Signal operations ────────────────────────────────────────────────
    let mut arena = ViewArena::new();
    let signal = Signal::new_in(&mut arena, 0_u64);

    bench("signal: read u64", 10_000_000, || {
        black_box(signal.read(|v| *v));
    });

    bench("signal: write u64", 10_000_000, || {
        signal.write(|v| *v = black_box(42));
    });

    bench("signal: write with increment", 10_000_000, || {
        signal.write(|v| *v = v.wrapping_add(1));
    });

    // ── Signal creation ──────────────────────────────────────────────────
    bench("signal: new_in (full allocation)", 1_000_000, || {
        let mut arena = ViewArena::new();
        for i in 0..100 {
            black_box(Signal::new_in(&mut arena, i as u64));
        }
    });

    // ── Subscribe scaling ────────────────────────────────────────────────
    bench("signal: subscribe 1000 targets", 10_000, || {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 0_u64);
        for i in 0..1000 {
            signal.subscribe(TargetId::new(i, 0, 0));
        }
    });

    // ── Worst case: many signals + many subscribers + many writes ────────
    bench(
        "signal: 100 signals × 10 subs × 100 writes",
        1_000,
        || {
            let mut arena = ViewArena::new();
            let signals: Vec<_> = (0..100)
                .map(|i| Signal::new_in(&mut arena, i as u64))
                .collect();

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
