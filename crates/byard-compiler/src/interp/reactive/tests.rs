//! The RFC-0004 §"Test fixtures" matrix (all 10) plus a `proptest` backstop
//! against a naive recompute-everything reference evaluator.
//!
//! These are written to the public [`ReactiveCtx`] API and are the acceptance
//! criteria for the reactive core.
//!
//! The fixtures use the RFC-0004 graph notation (`a`, `b`, `c`, `d`) verbatim,
//! so single-character bindings are intentional here.
#![allow(clippy::many_single_char_names)]

use super::{FrameTarget, ReactiveCtx, untrack};
use crate::interp::env::Value;

fn int(n: i64) -> Value {
    Value::Int(n)
}

/// Fixture 1 — diamond, single compute. `a → b,c → d=b+c`.
#[test]
fn diamond_computes_d_exactly_once() {
    let mut ctx = ReactiveCtx::new();
    let a = ctx.create_signal(int(1));
    let b = ctx.open_memo(move |c| int(c.read_signal(a).as_int().unwrap() * 10));
    let cc = ctx.open_memo(move |c| int(c.read_signal(a).as_int().unwrap() * 100));
    let d = ctx.open_value_binding(FrameTarget(0), move |c| {
        int(c.read_memo(b).as_int().unwrap() + c.read_memo(cc).as_int().unwrap())
    });

    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.binding_value(d), Some(int(110)));
    assert_eq!(ctx.eval_count(b), 1);
    assert_eq!(ctx.eval_count(cc), 1);
    assert_eq!(ctx.eval_count(d), 1);

    ctx.write_signal(a, int(2));
    let e = ctx.begin_tick();
    ctx.pull(e);
    // d, b, c each recompute exactly once against the settled post-mutation a.
    assert_eq!(ctx.eval_count(b), 2);
    assert_eq!(ctx.eval_count(cc), 2);
    assert_eq!(ctx.eval_count(d), 2);
    assert_eq!(ctx.binding_value(d), Some(int(220)));
}

/// Fixture 2 — idempotent marking, wide diamond. `a → m1..m50 → d`.
#[test]
fn wide_diamond_marks_each_node_once() {
    let mut ctx = ReactiveCtx::new();
    let a = ctx.create_signal(int(1));
    let memos: Vec<_> = (0..50)
        .map(|_| ctx.open_memo(move |c| c.read_signal(a)))
        .collect();
    let memos2 = memos.clone();
    let d = ctx.open_value_binding(FrameTarget(0), move |c| {
        int(memos2
            .iter()
            .map(|&m| c.read_memo(m).as_int().unwrap())
            .sum())
    });

    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.binding_value(d), Some(int(50)));

    let before = ctx.mark_effective_visits();
    ctx.write_signal(a, int(2));
    // 50 memos + d, each dirtied exactly once — never exponential.
    assert_eq!(ctx.mark_effective_visits() - before, 51);

    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.eval_count(d), 2, "d evaluates once despite 50 paths");
    assert_eq!(ctx.binding_value(d), Some(int(100)));
}

/// Fixture 3 — dynamic dependencies. Binding reads `a` when `flag` else `b`.
#[test]
fn dynamic_dependencies_swap_on_branch_change() {
    let mut ctx = ReactiveCtx::new();
    let flag = ctx.create_signal(Value::Bool(true));
    let a = ctx.create_signal(int(1));
    let b = ctx.create_signal(int(2));
    let bind = ctx.open_value_binding(FrameTarget(0), move |c| {
        if c.read_signal(flag).as_bool().unwrap() {
            c.read_signal(a)
        } else {
            c.read_signal(b)
        }
    });

    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.signal_subscriber_count(a), 1);
    assert_eq!(
        ctx.signal_subscriber_count(b),
        0,
        "b not read while flag=true"
    );

    // Mutating b does nothing; mutating a updates.
    let n = ctx.eval_count(bind);
    ctx.write_signal(b, int(99));
    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.eval_count(bind), n, "b had no subscriber");

    ctx.write_signal(a, int(5));
    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.eval_count(bind), n + 1);
    assert_eq!(ctx.binding_value(bind), Some(int(5)));

    // Flip the flag: subscription set swaps to b.
    ctx.write_signal(flag, Value::Bool(false));
    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.signal_subscriber_count(a), 0, "a no longer read");
    assert_eq!(ctx.signal_subscriber_count(b), 1);
    assert_eq!(ctx.binding_value(bind), Some(int(99)));
}

/// Fixture 4 — glitch-freedom. `b=a+1; c=a+1; d=(b==c)` is always `true`.
#[test]
fn glitch_freedom_d_is_always_true() {
    let mut ctx = ReactiveCtx::new();
    let a = ctx.create_signal(int(0));
    let b = ctx.open_memo(move |c| int(c.read_signal(a).as_int().unwrap() + 1));
    let cc = ctx.open_memo(move |c| int(c.read_signal(a).as_int().unwrap() + 1));
    let d = ctx.open_value_binding(FrameTarget(0), move |c| {
        Value::Bool(c.read_memo(b) == c.read_memo(cc))
    });

    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.binding_value(d), Some(Value::Bool(true)));

    for i in 1..20 {
        ctx.write_signal(a, int(i));
        let e = ctx.begin_tick();
        ctx.pull(e);
        // The tick is the consistency boundary: b and c are always equal.
        assert_eq!(
            ctx.binding_value(d),
            Some(Value::Bool(true)),
            "observed a transient glitch at a={i}"
        );
    }
}

/// Fixture 5 — `untrack`. `x = a + untrack(|| b)` never subscribes to `b`.
#[test]
fn untrack_does_not_subscribe() {
    let mut ctx = ReactiveCtx::new();
    let a = ctx.create_signal(int(1));
    let b = ctx.create_signal(int(10));
    let x = ctx.open_memo(move |c| {
        let av = c.read_signal(a).as_int().unwrap();
        let bv = untrack(|| c.read_signal(b).as_int().unwrap());
        int(av + bv)
    });
    let bind = ctx.open_value_binding(FrameTarget(0), move |c| c.read_memo(x));

    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.binding_value(bind), Some(int(11)));
    assert_eq!(
        ctx.signal_subscriber_count(b),
        0,
        "untracked read installs no edge"
    );

    // Mutating b does not recompute x.
    let n = ctx.eval_count(x);
    ctx.write_signal(b, int(99));
    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.eval_count(x), n, "x is not subscribed to b");

    // Mutating a does recompute x.
    ctx.write_signal(a, int(2));
    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.eval_count(x), n + 1);
    assert_eq!(ctx.signal_subscriber_count(b), 0);
}

/// Fixture 6 — over-mark bounded. Memo input changes, output equal: the binding
/// re-evaluates but writes no frame field.
#[test]
fn over_mark_re_evaluates_but_writes_nothing() {
    let mut ctx = ReactiveCtx::new();
    let a = ctx.create_signal(int(1));
    let m = ctx.open_memo(move |c| int(c.read_signal(a).as_int().unwrap().abs()));
    let bind = ctx.open_value_binding(FrameTarget(0), move |c| c.read_memo(m));

    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.frame_writes().len(), 1, "first projection writes once");

    let n = ctx.eval_count(bind);
    // 1 → -1: |a| is unchanged at 1.
    ctx.write_signal(a, int(-1));
    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(
        ctx.eval_count(bind),
        n + 1,
        "over-mark forces a re-evaluation"
    );
    assert_eq!(
        ctx.frame_writes().len(),
        0,
        "value-equality cut: no frame write"
    );
}

/// Fixture 7 — structural `when`. N toggles ⇒ N mount/unmount pairs, no leaked
/// subscriptions.
#[test]
fn structural_when_mounts_and_unmounts_without_leaks() {
    let mut ctx = ReactiveCtx::new();
    let cond = ctx.create_signal(Value::Bool(false));
    let inner = ctx.create_signal(int(0));
    ctx.open_when(
        move |c| c.read_signal(cond),
        move |c| {
            // then-branch: one binding that reads `inner`.
            let b = c.open_value_binding(FrameTarget(1), move |c| c.read_signal(inner));
            vec![b]
        },
        None,
    );

    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(
        ctx.signal_subscriber_count(inner),
        0,
        "branch not mounted yet"
    );

    for i in 0..5 {
        // mount
        ctx.write_signal(cond, Value::Bool(true));
        let e = ctx.begin_tick();
        ctx.pull(e);
        assert_eq!(
            ctx.signal_subscriber_count(inner),
            1,
            "iteration {i}: mounted branch subscribes to inner"
        );

        // unmount
        ctx.write_signal(cond, Value::Bool(false));
        let e = ctx.begin_tick();
        ctx.pull(e);
        assert_eq!(
            ctx.signal_subscriber_count(inner),
            0,
            "iteration {i}: unmounted branch leaked no subscription"
        );
    }
}

/// Fixture 8 — structural `for` (coarse). Mutating the list drops and rebuilds.
#[test]
fn structural_for_drops_and_rebuilds() {
    let mut ctx = ReactiveCtx::new();
    let list = ctx.create_signal(Value::List(vec![int(1), int(2)]));
    ctx.open_for(
        move |c| c.read_signal(list),
        move |c, item| {
            let v = item.clone();
            let b = c.open_value_binding(FrameTarget(2), move |_| v.clone());
            vec![b]
        },
    );

    let e = ctx.begin_tick();
    ctx.pull(e);
    let values: Vec<_> = ctx.frame_writes().iter().map(|(_, v)| v.clone()).collect();
    assert_eq!(values, vec![int(1), int(2)]);

    // Mutate the list: coarse reconciliation drops all and rebuilds.
    ctx.write_signal(list, Value::List(vec![int(3), int(4), int(5)]));
    let e = ctx.begin_tick();
    ctx.pull(e);
    let values: Vec<_> = ctx.frame_writes().iter().map(|(_, v)| v.clone()).collect();
    assert_eq!(values, vec![int(3), int(4), int(5)]);
}

/// Fixture 9 — cycle trip-wire (debug). Two mutually-reading memos panic the
/// `evaluating` `debug_assert!` instead of hanging.
#[test]
#[should_panic(expected = "reactive cycle")]
fn cycle_trip_wire_fires() {
    let mut ctx = ReactiveCtx::new();
    let b = ctx.open_memo(|_| int(0)); // placeholder body, replaced below
    let c = ctx.open_memo(move |cx| cx.read_memo(b));
    ctx.set_compute(b, move |cx| cx.read_memo(c));
    // Forcing a read drives b → c → b, tripping the cycle guard.
    let _ = ctx.read_memo(b);
}

/// Fixture 10 — tick boundary. Batched mutations settle before any evaluation,
/// and each dirty scope evaluates once.
#[test]
fn tick_boundary_batches_marks_before_evaluation() {
    let mut ctx = ReactiveCtx::new();
    let a = ctx.create_signal(int(1));
    let b = ctx.create_signal(int(2));
    let d = ctx.open_value_binding(FrameTarget(0), move |c| {
        int(c.read_signal(a).as_int().unwrap() + c.read_signal(b).as_int().unwrap())
    });

    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.eval_count(d), 1);

    // Batch two mutations before the single pull.
    ctx.write_signal(a, int(10));
    ctx.write_signal(b, int(20));
    assert_eq!(ctx.eval_count(d), 1, "mutations only mark; no eval yet");

    let e = ctx.begin_tick();
    ctx.pull(e);
    assert_eq!(ctx.eval_count(d), 2, "evaluates once despite two marks");
    assert_eq!(ctx.binding_value(d), Some(int(30)));
}

// ─────────────────────────────────────────────────────────────────────────
// proptest backstop: random (graph, mutations) vs a naive reference evaluator.
// ─────────────────────────────────────────────────────────────────────────

use proptest::prelude::*;

proptest! {
    #[test]
    fn matches_naive_recompute(
        n_sigs in 1usize..5,
        // Each memo is a non-empty subset of signal indices (as a bitmask).
        memo_masks in proptest::collection::vec(1u32..32, 1..6),
        writes in proptest::collection::vec((0usize..5, -50i64..50), 0..40),
    ) {
        let mut ctx = ReactiveCtx::new();
        let mut values = vec![0i64; n_sigs];
        let sigs: Vec<_> = (0..n_sigs).map(|_| ctx.create_signal(int(0))).collect();

        // Build the memos from their bitmasks, restricted to live signal indices.
        let subsets: Vec<Vec<usize>> = memo_masks
            .iter()
            .map(|mask| {
                let s: Vec<usize> = (0..n_sigs).filter(|i| mask & (1 << i) != 0).collect();
                if s.is_empty() { vec![0] } else { s }
            })
            .collect();

        let mut memos = Vec::new();
        for subset in &subsets {
            let subset = subset.clone();
            let sigs = sigs.clone();
            memos.push(ctx.open_memo(move |c| {
                int(subset.iter().map(|&i| c.read_signal(sigs[i]).as_int().unwrap()).sum())
            }));
        }

        let memos2 = memos.clone();
        let bind = ctx.open_value_binding(FrameTarget(0), move |c| {
            int(memos2.iter().map(|&m| c.read_memo(m).as_int().unwrap()).sum())
        });

        let e = ctx.begin_tick();
        ctx.pull(e);

        let naive = |values: &[i64]| -> i64 {
            subsets.iter().map(|sub| sub.iter().map(|&i| values[i]).sum::<i64>()).sum()
        };
        prop_assert_eq!(ctx.binding_value(bind), Some(int(naive(&values))));

        for (idx, val) in writes {
            let idx = idx % n_sigs;
            values[idx] = val;
            ctx.write_signal(sigs[idx], int(val));
            let e = ctx.begin_tick();
            ctx.pull(e);
            prop_assert_eq!(ctx.binding_value(bind), Some(int(naive(&values))));
        }
    }
}
