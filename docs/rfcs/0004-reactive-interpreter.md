# RFC-0004: Reactive Interpreter (`interp/reactive.rs`) — read-tracking, pull-based memos, structural effects

- **Status:** Active — implemented (M8 reactive core, all 10 fixtures + proptest green). Mark-and-Pull, memos, structural effects, `untrack` landed in `interp/reactive.rs`.
- **Author(s):** Briany4717
- **Created:** 2026-06-20
- **Last updated:** 2026-06-20
- **Depends on:** RFC-0001 (§2.2 `Signal<T>` dirty-flag vector, §2.1 `ViewArena`, §5.1 logic thread, §3.3 dirty rectangles), RFC-0002 (**D1** Mark-and-Pull, **D2** `untrack`, **D3** reactivity metadata, structural effects), RFC-0003 (events as a mutation source, per-tick ordering).
- **Implements:** RFC-0002 complication 2 ("the interpreter grows a reactivity subsystem") and the D1 fixture-test mandate.

---

## Summary

This RFC specifies `interp/reactive.rs`, the subsystem that makes `byld`'s
automatic reactivity work. It is the single largest piece of net-new Phase 2 code
(RFC-0002 complication 2) and the one with the subtle correctness conditions, so
it is specified in full here, with the fixture tests RFC-0002 D1 required *before*
implementation.

The design rests on one load-bearing observation from RFC-0002: **RFC-0001's
`Signal<T>` already is a subscription mechanism.** §2.2 defines a `Signal` as a
value plus "a vector of atomic dirty flags pointing to specific render or spatial
subsystem entries." Automatic reactivity does not add a new graph type — it adds
a **read-tracking layer** that *fills in* those subscriber links by observing
which `Signal`s each expression reads, and a **Mark-and-Pull** update discipline
(D1) over them. Three scope kinds sit on top: **value bindings** (an intrinsic
field that projects an expression), **computed memos** (`let` / `fn`), and
**structural effects** (`for` / `when`). All of it runs on the logic thread
(RFC-0001 §5.1), so the "current scope" tracking pointer is a logic-thread-local
with no synchronization.

---

## Motivation

RFC-0002 adopted automatic reactivity (`var` + mutation, no hand-wired signals)
and resolved its *discipline* questions (D1 Mark-and-Pull, D2 `untrack`, D3
metadata). What it did not do is specify the *machine*: the scope types, the
tracking pointer, the mark cascade, the pull phase, dynamic-dependency clearing,
memo memoization, and how `for`/`when` reconcile arenas. Without that, two
contributors implement reactivity two incompatible ways, and the correctness
conditions D1 flagged (glitch-freedom, dynamic deps, idempotent marking) stay as
prose hopes instead of tested invariants. This RFC turns them into a concrete
module with a test matrix.

---

## Guide-level explanation

The developer writes this (RFC-0002 surface):

```byld
View Search() {
    var query = ""
    var items: List<Str> = ["apple", "pear", "plum"]

    let filtered = items.filter(|x| x.starts_with(query))   // computed memo
    fn count() -> Int => filtered.len()                      // computed memo

    Column {
        Text("Found {count()}")                              // value binding
        for item in filtered { Text(item) }                  // structural effect
        when filtered.is_empty() { Text("none") }            // structural effect
    }
}
```

The compiler/interpreter sees a **reactive graph**, built by *observation*:

```
   query ─────┐
              ├──►  filtered (memo) ──►  count (memo) ──►  Text("Found …") binding
   items ─────┘            │
                           ├──►  for-effect  (mounts one Text per item)
                           └──►  when-effect (mounts/unmounts "none")
```

Nobody declared an edge. Each edge exists because, while evaluating `filtered`,
the interpreter saw it read `query` and `items`; while evaluating the
`Text("Found {count()}")` binding, it saw it read `count`, which read `filtered`.
Mutating `query` marks `filtered` dirty, which marks `count` and the two
structural effects, which mark the bindings — and the next tick pulls exactly the
affected ones.

---

## Reference-level explanation

### 1. Core types

All scope state lives in the owning `View`'s `ViewArena` (RFC-0001 §2.1), so it is
reclaimed in one linear pass on unmount — reactivity adds no separate lifetime
management.

```rust
/// Identifies any reactive node. A ValueBinding's id doubles as the
/// render/spatial dirty-flag target of RFC-0001 §2.2.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct ScopeId(u32);

pub enum Scope {
    /// Projects an expression into one RenderFrame field (a Text string,
    /// a bg color, a gap, …). Its ScopeId IS the §2.2 dirty target.
    ValueBinding {
        target: FrameTarget,     // which primitive field this writes
        expr:   AstId,           // the sub-expression to re-walk
        deps:   SmallVec<[SignalId; 4]>,
        dirty:  bool,
        last_epoch: u32,         // per-tick epoch guard (D1)
    },
    /// A `let` / `fn` computed value. Pull-based, memoized.
    Memo {
        expr:   AstId,
        cache:  Value,
        deps:   SmallVec<[SignalId; 4]>,
        subs:   SmallVec<[ScopeId; 4]>,  // who reads this memo
        dirty:  bool,
        evaluating: bool,                // cycle guard (debug)
    },
    /// `for` / `when`. Re-runs a mount/unmount effect, not a field write.
    Structural {
        kind:   StructuralKind,          // When | For
        expr:   AstId,                   // condition / iterator
        deps:   SmallVec<[SignalId; 4]>,
        dirty:  bool,
        mounted: StructuralState,        // current branch / child arenas
    },
}
```

A `Signal` (RFC-0001 §2.2) is extended only in interpretation: its dirty-flag
vector holds `ScopeId`s (value bindings, memos, and structural effects are all
valid subscribers). No change to the `Signal` memory layout is required — a
`ScopeId` is exactly the "specific render or spatial subsystem entry" §2.2 already
points at, generalized to also name memos and effects.

### 2. The tracking pointer

```rust
thread_local! {
    /// The scope currently being evaluated, or None under `untrack`.
    /// Logic-thread-local: RFC-0001 §5.1 confines Signal access here, so no
    /// atomics/locks are needed.
    static CURRENT_SCOPE: Cell<Option<ScopeId>> = const { Cell::new(None) };
}
```

`Signal::read` becomes tracking-aware:

```rust
impl<T> Signal<T> {
    pub fn read(&self) -> T {
        if let Some(s) = CURRENT_SCOPE.get() {
            self.subscribers.insert_unique(s);   // §2.2 dirty-flag link
            scope_deps_record(s, self.id);       // reverse edge, for clearing
        }
        self.value.clone()
    }
}
```

Reading outside any scope (e.g. inside an event handler body, which mutates but is
not itself a tracked projection) simply does not subscribe — correct, because a
handler is an action, not a binding.

### 3. Scope evaluation with dynamic-dependency clearing

Every (re)evaluation of a scope first **clears its old subscriptions**, then
re-tracks. This is what makes dependencies *dynamic*: a binding that reads `a`
this tick and `b` next tick ends subscribed to exactly what it read this time
(D1 "dynamic dependencies").

```rust
fn evaluate_scope(s: ScopeId) -> Value {
    clear_deps(s);                       // remove s from each old dep's sub-list
    let prev = CURRENT_SCOPE.replace(Some(s));
    let v = walk_expr(scope[s].expr);    // reads call Signal::read → re-track
    CURRENT_SCOPE.set(prev);             // restored even on the error path
    v
}
```

`clear_deps` walks `scope[s].deps`, removes `s` from each `Signal`'s subscriber
vector, then empties `deps`. `walk_expr` repopulates `deps` via `read`.

### 4. Mark phase (synchronous, idempotent) — D1

A `var` mutation runs a synchronous mark cascade. It **never computes a value**;
it only sets dirty bits and enqueues work.

```rust
fn mark(sig: SignalId) {
    for s in signal[sig].subscribers.iter() {
        mark_scope(s);
    }
}

fn mark_scope(s: ScopeId) {
    if scope[s].dirty { return; }        // IDEMPOTENT — stop re-traversal (D1)
    scope[s].dirty = true;
    match scope[s] {
        ValueBinding { .. } => tick.dirty_bindings.push(s),
        Structural  { .. } => tick.dirty_structural.push(s),
        Memo { subs, .. }  => for sub in subs { mark_scope(sub); }  // cascade
    }
}
```

The idempotent guard is not an optimization — without it, a wide diamond (one
source feeding *k* memos that all feed one binding) degrades the cascade from
`O(nodes)` to `O(paths)`, which is exponential. Memos propagate the mark to their
subscribers but are **not** recomputed here; their recompute is deferred to a
lazy pull (§6).

Mutations from all three sources — event handlers (RFC-0003 step 2), async
controller results (RFC-0001 §5.1, step 3), and hot-reload value patches
(RFC-0002, step 1) — funnel through `mark`. Because every source only *marks*
during its step and the single pull is step 4, **the tick is the consistency
boundary** (D1): every pull observes a fully-settled mark set, so no scope can
read a half-updated graph. This is the glitch-freedom guarantee, and it is a
property of the *ordering*, not of any per-node cleverness.

### 5. Pull phase (tick step 4) — D1

```rust
fn pull(frame: &mut RenderFrame, epoch: u32) {
    for s in tick.dirty_structural.drain(..) {   // structure first: mounts may
        reconcile_structural(s, frame);          // create new value bindings
    }
    for s in tick.dirty_bindings.drain(..) {
        if scope[s].last_epoch == epoch { continue; }  // EPOCH GUARD (D1)
        scope[s].last_epoch = epoch;
        let v = evaluate_scope(s);               // pulls dirty memos on demand
        write_frame_field(frame, scope[s].target, v);   // value-equality cut → §7
        scope[s].dirty = false;
    }
}
```

Structural effects are reconciled before value bindings because mounting a new
arena creates new value bindings that must also be evaluated this tick. The epoch
guard ensures each binding evaluates **at most once per tick** even when reached
through several dirty paths.

### 6. Pull-based memos (the diamond solution) — D1

A memo is recomputed **only when read while dirty**, never eagerly:

```rust
fn read_memo(m: ScopeId) -> Value {
    if let Some(s) = CURRENT_SCOPE.get() {       // the reader subscribes to m
        scope[m].subs.insert_unique(s);
        scope_deps_record(s, m.as_signal_like());
    }
    if scope[m].dirty {
        debug_assert!(!scope[m].evaluating, "reactive cycle through memo {m:?}");
        scope[m].evaluating = true;
        scope[m].cache = evaluate_scope(m);      // recursive: pulls its own deps
        scope[m].dirty = false;
        scope[m].evaluating = false;
    }
    scope[m].cache.clone()
}
```

**Why this is glitch-free on a diamond.** Take `a` → memos `b`, `c` → binding `d
= b + c`. Mutating `a` marks `b` and `c` dirty, each cascades a mark to `d`
(second arrival is the idempotent no-op). Pull evaluates `d`: walking `b + c`
reads `b` (dirty → recompute against the *settled* `a`, cache, clear) then `c`
(same). `d` computes once, from two fresh operands. No double-compute, no stale
read — and no runtime topological scheduler, exactly as D1 promised. The
`evaluating` flag is a debug-only cycle trip-wire; the Phase 2 grammar cannot
express a reactive cycle, but a future `fn` amendment could, and this catches it
loudly instead of hanging.

### 7. Over-marking vs. over-rendering (an honest bound)

Mark-and-Pull is a *push-marks / pull-values* scheme, so it can **over-mark**: a
binding that reads a memo whose inputs changed but whose *output* did not will
still be re-evaluated, because the mark cascade fires before any value is known.
This is bounded and acceptable:

- It causes at most one extra **evaluation** (an AST re-walk), never an extra
  **GPU command** — the frame write in §5 is value-equality–gated (the same cut
  RFC-0003 E1 relies on), so an unchanged projected value writes nothing and
  emits no draw call (RFC-0001 §3.3 produces no dirty rect).
- Dev mode is explicitly not the throughput path (RFC-0002), so a few redundant
  AST walks per tick are irrelevant.
- The Phase 4 transpiler eliminates the gap entirely: it can compute the static
  dependency graph at compile time and emit value-versioned memos, so Prod has
  neither over-marking nor over-rendering.

A future optional refinement (memo value-versioning: a memo that recomputes to an
equal value bumps no version, and readers comparing versions skip) is noted in
*Future possibilities* — deliberately **not** in Phase 2, to keep the scheme
small and obviously correct.

### 8. Structural effects: `when` and `for`

```rust
fn reconcile_structural(s: ScopeId, frame: &mut RenderFrame) {
    if scope[s].last_epoch == current_epoch { return; }
    scope[s].last_epoch = current_epoch;
    match scope[s].kind {
        When => {
            let take = evaluate_scope(s).as_bool();   // re-tracks the condition
            if take != scope[s].mounted.active_branch() {
                scope[s].mounted.drop_current();      // RFC-0001 §2 O(1) arena drop
                scope[s].mounted = mount_branch(take); // alloc child arena + Env,
            }                                         // evaluate body (self-registers)
        }
        For => {
            // Phase 2 coarse reconciliation (RFC-0002 D7):
            let list = evaluate_scope(s).as_list();
            scope[s].mounted.drop_all();              // one linear arena pass
            for item in list {
                let child = mount_child_arena();      // binds `item` in child Env
                scope[s].mounted.push(child);
            }
        }
    }
}
```

Mounting a branch/child runs that body's declarations, which open their own value
bindings and memos (self-registering against their child `ViewArena`). Unmounting
drops the child arena (RFC-0001 §2), which removes its scopes, its `Signal`
subscriptions, and its §4.2 grid entries in the same linear pass — so a structural
change cannot leak a subscription or a stale hit-test rect. The `for` upgrade to
keyed reconciliation is gated exactly by RFC-0002 D7's churn/FPS triggers.

### 9. `untrack` — D2

```rust
pub fn untrack<R>(thunk: impl FnOnce() -> R) -> R {
    struct Restore(Option<ScopeId>);
    impl Drop for Restore {                       // RAII: restores even on unwind
        fn drop(&mut self) { CURRENT_SCOPE.set(self.0); }
    }
    let _g = Restore(CURRENT_SCOPE.replace(None));
    thunk()                                        // Signal::read inside skips tracking
}
```

`untrack(expr)` is a reserved intrinsic recognized in `interp/eval.rs` (D2): it
evaluates `expr` with `CURRENT_SCOPE = None`, so reads inside install no
subscription, then restores the previous scope (correctly nested, via the RAII
guard). The canonical use is seeding a `var` from a one-shot read of another
without creating a permanent edge.

### 10. D3 reactivity metadata (no AST mutation)

Whether a `let`/`fn` is reactive (reads ≥1 `Signal`) is observed during its first
`evaluate_scope` — if its `deps` is non-empty, it is reactive. Per D3 this is
**not** written back onto the AST node (the AST stays immutable owned data so
hot-reload diffing is clean); it is recorded in `ResolvedView.reactive:
HashSet<Symbol>` and emitted to the LSP JSON. `interp/reactive.rs` exposes
`deps_nonempty(scope) -> bool` for the resolver to populate that set.

### 11. Hot-reload hook (RFC-0002 boundary)

On a **reactive-compatible** patch (RFC-0002 case 1: `var`/`param`/`inject` shape
unchanged, only expressions/elements differ), `interp/reload.rs` keeps the live
`Signal`s and asks `reactive.rs` to **rebuild the scope set from the new AST** and
re-point it at the existing `Signal`s (matched by `(position, name)`). Because
every scope re-evaluates and re-tracks from scratch (§3), the dependency graph is
re-derived automatically — there is no separate "patch the graph" path. On a
**structure-incompatible** patch (case 2), the arena drops and remounts, taking
all scopes with it. RFC-0003 E5 gates structural patches behind an in-flight
gesture.

### 12. Public API surface (`interp/reactive.rs`)

```rust
pub struct ReactiveCtx { /* scope arena, tick queues, epoch */ }

impl ReactiveCtx {
    pub fn open_value_binding(&mut self, target: FrameTarget, expr: AstId) -> ScopeId;
    pub fn open_memo(&mut self, expr: AstId) -> ScopeId;
    pub fn open_structural(&mut self, kind: StructuralKind, expr: AstId) -> ScopeId;

    pub fn read_signal<T>(&self, sig: &Signal<T>) -> T;   // tracking-aware
    pub fn read_memo(&mut self, m: ScopeId) -> Value;     // pull-on-read
    pub fn mark(&mut self, sig: SignalId);                // mutation entry point

    pub fn begin_tick(&mut self) -> u32;                  // bumps epoch
    pub fn pull(&mut self, frame: &mut RenderFrame, epoch: u32);

    pub fn untrack<R>(&self, f: impl FnOnce() -> R) -> R;
    pub fn deps_nonempty(&self, s: ScopeId) -> bool;      // for D3 metadata
}
```

This is the entire surface the rest of the interpreter (`eval.rs`,
`intrinsics.rs`, `reload.rs`) touches; everything in §1–§11 is private.

---

## Test fixtures (required before implementation — D1 mandate)

Each is a deterministic unit/property test exercising one invariant. They are the
acceptance criteria for the module.

1. **Diamond — single compute.** `a → b,c → d=b+c`. Mutate `a`; assert `d`
   evaluates exactly once, equals the value computed from the post-mutation `a`,
   and `b`/`c` each recompute once.
2. **Idempotent marking — wide diamond.** `a → m1..m50 → d`. Mutate `a`; assert
   the mark cascade visits each node once (instrument a counter) and `d`
   evaluates once (epoch guard).
3. **Dynamic dependencies.** Binding reads `a` when `flag` else `b`. With
   `flag=true`, mutate `b` → no update; mutate `a` → update. Flip `flag`; assert
   subscription set swaps (mutating `a` now no-ops, `b` updates).
4. **Glitch-freedom.** `a; b=a+1; c=a+1; d=(b==c)`. Mutate `a` repeatedly; assert
   `d` is observed `true` at every pull (never a transient `false`).
5. **Untrack.** `let x = a + untrack(|| b)`. Mutate `b` → `x` not recomputed;
   mutate `a` → recomputed. Assert `x` never subscribes to `b`.
6. **Over-mark bounded.** Memo whose inputs change but output is equal feeds a
   binding. Mutate input; assert the binding *re-evaluates* (counter +1) but
   writes **no** frame field (value-equality cut → zero dirty rects).
7. **Structural `when`.** Toggle a `when` condition N times; assert exactly N
   mount/unmount pairs, and that the unmounted branch's `Signal` subscriptions and
   §4.2 grid entries are gone (no leak).
8. **Structural `for` (coarse).** Mutate the list; assert all child arenas drop
   and rebuild, child count matches, and each child's `item` binding is correct.
9. **Cycle trip-wire (debug).** Construct two mutually-reading memos via a test
   hook; assert the `evaluating` `debug_assert!` fires rather than hanging.
10. **Tick boundary.** Batch several `var` mutations before a single
    `pull(epoch)`; assert all marks settle before any evaluation and each dirty
    scope evaluates once.

A `proptest` generator over random `(build graph, mutate, pull)` sequences,
checked against a simple non-incremental reference evaluator (recompute
everything every tick), backstops the hand-written fixtures.

---

## Drawbacks

- **Subtle correctness surface.** Tracking, clearing, idempotent marking, epoch
  guarding, and pull-on-read must all be right together; the fixtures exist
  precisely because inspection is not enough. This is the module most likely to
  harbor a reactivity bug.
- **Over-marking** can re-evaluate bindings whose value did not change (§7). Bounded
  to wasted AST walks (never wasted GPU work), accepted for Dev, eliminated in
  Prod — but real.
- **Coarse `for`** (drop-and-rebuild) loses child state across list changes until
  the D7-gated keyed upgrade lands.
- **`SmallVec` sizing** for `deps`/`subs` is a guess (inline 4); pathological
  fan-out spills to the heap. Measure and tune; not a correctness issue.

---

## Rationale and alternatives

**Why reuse `Signal`'s dirty-flag vector instead of a separate reactive graph?**
RFC-0002's whole feasibility argument: §2.2 already built the subscriber store;
a parallel graph would duplicate it and risk divergence. Tracking is an
*observation* layer over the existing primitive.

**Why pull-based memos rather than eager push-recompute?** Glitch-freedom with the
least machinery (D1 Rationale). Eager push needs topological scheduling to avoid
double-compute and stale reads on diamonds; pull-on-read with a dirty bit is
correct for free at the cost of laziness, which Dev can afford.

**Why clear-then-retrack every evaluation instead of diffing dependency sets?**
Clearing is `O(deps)` and trivially correct for dynamic dependencies; a diff is
more code for no Dev-mode benefit. The Phase 4 transpiler computes deps statically
and needs neither.

**Why thread-local `CURRENT_SCOPE` and not pass a context explicitly?** The
interpreter is a recursive AST walk; threading a `&mut ctx` through every `read`
fights the borrow checker (a `Signal::read` needs `&self` and the scope needs
`&mut`). A logic-thread-local pointer is the standard Solid/reactively approach and
is sound here because §5.1 already confines all of this to one thread.

---

## Prior art

- **Solid.js / reactively / Leptos.** Read-tracking via a current-computation
  pointer, pull-based memos, clear-and-retrack dynamic deps — the direct model.
- **RFC-0001 §2.2 / §3.3.** The dirty-flag vector and dirty-rectangle machinery
  this layer drives.
- **MobX (derivations).** The "derived values recompute lazily on read" lineage
  for §6.
- **Adapton / incremental computation.** The theory behind pull-based incremental
  recomputation and why a consistency boundary (the tick) gives glitch-freedom.

---

## Unresolved questions

- **During implementation:**
  - [ ] **`SmallVec` inline sizes** for `deps`/`subs` — pick after measuring real
    `byld` views.
  - [ ] **Memo value-versioning** (§7 over-mark cut) — confirm it stays out of
    Phase 2 and lands (if ever) as a measured optimization, not speculatively.
  - [ ] **Effect ordering within structural reconciliation** — when a single tick
    both toggles a `when` and dirties a binding inside the branch being mounted,
    confirm the "structural before bindings" order in §5 fully covers it, or
    specify a fixed-point loop bound.
  - [ ] **`for` child identity for the coarse path** — define what "child state"
    (e.g. an inner `var`) is expected to survive a coarse rebuild (answer for
    Phase 2: none; documented so it is not a surprise).

---

## Future possibilities

- **Memo value-versioning** to eliminate over-marking (§7), once measured to
  matter.
- **Keyed `for` reconciliation** (RFC-0002 D7 trigger) reusing the structural
  scope with child-arena preservation.
- **Phase 4 transpiler** lowering this exact graph to static Rust: discovered
  subscriptions become const wiring, pull-memos become const-folded computations,
  structural effects become generated mount/unmount code — zero runtime tracking.
- **Async/suspense scopes** (a binding awaiting a controller result showing a
  fallback) as a fourth scope kind, once Phase 4 and the async story are firm.
