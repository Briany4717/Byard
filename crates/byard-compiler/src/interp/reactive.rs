//! The reactive core: read-tracking, Mark-and-Pull, pull-based memos, and
//! structural effects (RFC-0004 in full; RFC-0002 D1/D2).
//!
//! This is the highest-risk Phase 2 module. It implements automatic reactivity
//! as a *read-tracking* layer over a subscription store, exactly as RFC-0004
//! specifies:
//!
//! - A logic-thread-local [`struct@CURRENT_SCOPE`] names the computation being
//!   evaluated (§2). Every `read_signal`/`read_memo` while a scope is active
//!   records a subscription, so dependency edges are *discovered*, never
//!   declared.
//! - **Mark** (§4) is synchronous and idempotent: a mutation sets dirty bits
//!   and enqueues work, never computing a value. Memos cascade the mark to
//!   their subscribers; bindings/structurals enqueue.
//! - **Pull** (§5) runs once per tick (the consistency boundary): structural
//!   effects first (mounts create new bindings), then value bindings, each at
//!   most once per tick via an epoch guard. Memos are **pull-on-read** (§6),
//!   recomputing lazily against a fully-settled mark set — the glitch-free
//!   diamond solution.
//! - Dynamic dependencies (§3): every (re)evaluation clears the scope's old
//!   subscriptions before re-tracking.
//!
//! ## How computations are plugged in
//!
//! RFC-0004's `walk_expr(scope.expr)` is the eval driver's job (M9). To keep
//! this module testable in isolation against the §"Test fixtures" matrix, a
//! scope's computation is a boxed `FnMut(&mut ReactiveCtx) -> Value` closure
//! (the AST walker, in M9; a plain Rust closure, in these tests). The closure
//! is *taken out* of the context for the duration of its own evaluation, so the
//! borrow checker is satisfied without `RefCell` and a self-read cycle is caught
//! by the `evaluating` trip-wire (§6) rather than aliasing.

use std::cell::Cell;

use smallvec::SmallVec;

use super::env::{SignalId, Value};

thread_local! {
    /// The scope currently being evaluated, or `None` outside any scope (and
    /// under [`untrack`]). Logic-thread-local: RFC-0001 §5.1 confines all of
    /// this to one thread, so no atomics or locks are needed (INV-2).
    static CURRENT_SCOPE: Cell<Option<ScopeId>> = const { Cell::new(None) };
}

/// Identifies a reactive scope (value binding, memo, or structural effect).
/// A value binding's id doubles as its RFC-0001 §2.2 dirty-flag target.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ScopeId(u32);

/// Identifies a `RenderFrame` field a value binding projects into. Opaque to
/// the reactive core; the eval driver (M9) maps it to a concrete primitive
/// field.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FrameTarget(pub u32);

/// Which structural construct a [`ScopeKind::Structural`] scope drives.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum StructuralKind {
    /// `when cond { … } else { … }`.
    When,
    /// `for item in list { … }`.
    For,
}

/// A computation: re-walked to produce a [`Value`], recording its reads as
/// subscriptions while it runs.
type Compute = Box<dyn FnMut(&mut ReactiveCtx) -> Value>;

/// Mounts a `when` branch, returning the scope ids it created (so they can be
/// dropped on the next toggle).
type BranchMount = Box<dyn FnMut(&mut ReactiveCtx) -> Vec<ScopeId>>;

/// Mounts one `for` item, returning the scope ids it created.
type ItemMount = Box<dyn FnMut(&mut ReactiveCtx, &Value) -> Vec<ScopeId>>;

/// A recorded dependency edge from a scope to a source it read.
#[derive(Copy, Clone, Eq, PartialEq)]
enum Dep {
    Signal(SignalId),
    Memo(ScopeId),
}

/// Per-`when`/`for` mount state, kept outside the scope so it can be taken out
/// during reconciliation without aliasing the scope arena.
enum Mount {
    When {
        then: BranchMount,
        els: Option<BranchMount>,
        active: Option<bool>,
        children: Vec<ScopeId>,
    },
    For {
        item: ItemMount,
        children: Vec<ScopeId>,
    },
}

/// What kind of scope this is, plus its kind-specific state.
enum ScopeKind {
    /// Projects an expression into one `RenderFrame` field.
    ValueBinding {
        target: FrameTarget,
        last_written: Option<Value>,
    },
    /// A `let`/`fn` computed value: pull-based, memoized.
    Memo {
        cache: Value,
        subs: SmallVec<[ScopeId; 4]>,
        evaluating: bool,
    },
    /// A `for`/`when` structural effect.
    Structural { kind: StructuralKind },
}

/// One reactive scope. Liveness is tracked so structural unmount can retire a
/// scope without shifting other scopes' ids.
struct Scope {
    kind: ScopeKind,
    deps: SmallVec<[Dep; 4]>,
    dirty: bool,
    last_epoch: u32,
    eval_count: u32,
    live: bool,
}

/// A reactive source: a value plus the scopes subscribed to it.
struct SignalCell {
    value: Value,
    subscribers: SmallVec<[ScopeId; 4]>,
}

/// The reactive context: owns the scope arena, the signal store, the per-tick
/// dirty queues, and the epoch counter (RFC-0004 §12).
#[derive(Default)]
pub struct ReactiveCtx {
    scopes: Vec<Scope>,
    /// Parallel to `scopes`: each scope's computation, taken out while it runs.
    computes: Vec<Option<Compute>>,
    /// Parallel to `scopes`: structural mount state, present only for
    /// `Structural` scopes.
    mounts: Vec<Option<Mount>>,
    signals: Vec<SignalCell>,
    dirty_bindings: Vec<ScopeId>,
    dirty_structural: Vec<ScopeId>,
    epoch: u32,
    /// Observability for the fixtures: how many frame-field writes the last
    /// pull(s) performed (the value-equality cut means an unchanged projection
    /// records nothing).
    frame_writes: Vec<(FrameTarget, Value)>,
    /// Cumulative count of *effective* mark visits (a scope dirtied for the
    /// first time), to prove the cascade is idempotent and never exponential.
    mark_effective: u32,
}

impl ReactiveCtx {
    /// Creates an empty reactive context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── construction ────────────────────────────────────────────────────

    /// Allocates a reactive source with `initial` value.
    pub fn create_signal(&mut self, initial: Value) -> SignalId {
        let id = SignalId(self.signals.len() as u32);
        self.signals.push(SignalCell {
            value: initial,
            subscribers: SmallVec::new(),
        });
        id
    }

    fn push_scope(
        &mut self,
        kind: ScopeKind,
        compute: Option<Compute>,
        mount: Option<Mount>,
    ) -> ScopeId {
        let id = ScopeId(self.scopes.len() as u32);
        self.scopes.push(Scope {
            kind,
            deps: SmallVec::new(),
            dirty: true,
            last_epoch: 0,
            eval_count: 0,
            live: true,
        });
        self.computes.push(compute);
        self.mounts.push(mount);
        id
    }

    /// Opens a computed memo (`let`/`fn`). Starts dirty: its first read
    /// computes it.
    pub fn open_memo(
        &mut self,
        compute: impl FnMut(&mut ReactiveCtx) -> Value + 'static,
    ) -> ScopeId {
        self.push_scope(
            ScopeKind::Memo {
                cache: Value::Unit,
                subs: SmallVec::new(),
                evaluating: false,
            },
            Some(Box::new(compute)),
            None,
        )
    }

    /// Opens a value binding projecting `compute` into `target`. Enqueued dirty
    /// so the next [`ReactiveCtx::pull`] evaluates it.
    pub fn open_value_binding(
        &mut self,
        target: FrameTarget,
        compute: impl FnMut(&mut ReactiveCtx) -> Value + 'static,
    ) -> ScopeId {
        let id = self.push_scope(
            ScopeKind::ValueBinding {
                target,
                last_written: None,
            },
            Some(Box::new(compute)),
            None,
        );
        self.dirty_bindings.push(id);
        id
    }

    /// Opens a `when` structural effect. `cond` evaluates the condition; `then`
    /// / `els` mount the corresponding branch.
    pub fn open_when(
        &mut self,
        cond: impl FnMut(&mut ReactiveCtx) -> Value + 'static,
        then: impl FnMut(&mut ReactiveCtx) -> Vec<ScopeId> + 'static,
        els: Option<BranchMount>,
    ) -> ScopeId {
        let id = self.push_scope(
            ScopeKind::Structural {
                kind: StructuralKind::When,
            },
            Some(Box::new(cond)),
            Some(Mount::When {
                then: Box::new(then),
                els,
                active: None,
                children: Vec::new(),
            }),
        );
        self.dirty_structural.push(id);
        id
    }

    /// Opens a `for` structural effect. `iter` evaluates the list; `item` mounts
    /// one child per element.
    pub fn open_for(
        &mut self,
        iter: impl FnMut(&mut ReactiveCtx) -> Value + 'static,
        item: impl FnMut(&mut ReactiveCtx, &Value) -> Vec<ScopeId> + 'static,
    ) -> ScopeId {
        let id = self.push_scope(
            ScopeKind::Structural {
                kind: StructuralKind::For,
            },
            Some(Box::new(iter)),
            Some(Mount::For {
                item: Box::new(item),
                children: Vec::new(),
            }),
        );
        self.dirty_structural.push(id);
        id
    }

    /// Replaces a scope's computation (used by the cycle fixture and by
    /// hot-reload, M13). Marks the scope dirty.
    pub fn set_compute(
        &mut self,
        s: ScopeId,
        compute: impl FnMut(&mut ReactiveCtx) -> Value + 'static,
    ) {
        self.computes[s.0 as usize] = Some(Box::new(compute));
        self.scopes[s.0 as usize].dirty = true;
    }

    // ── reads (tracking-aware) ──────────────────────────────────────────

    /// Reads a signal, subscribing the current scope (if any) to it (§2).
    pub fn read_signal(&mut self, sig: SignalId) -> Value {
        if let Some(s) = CURRENT_SCOPE.with(Cell::get) {
            self.subscribe_signal(sig, s);
            self.record_dep(s, Dep::Signal(sig));
        }
        self.signals[sig.0 as usize].value.clone()
    }

    /// Reads a memo, subscribing the current scope to it and pulling it on
    /// demand if dirty (§6).
    pub fn read_memo(&mut self, m: ScopeId) -> Value {
        if let Some(s) = CURRENT_SCOPE.with(Cell::get) {
            self.subscribe_memo(m, s);
            self.record_dep(s, Dep::Memo(m));
        }
        let dirty = self.scopes[m.0 as usize].dirty;
        if dirty {
            if let ScopeKind::Memo { evaluating, .. } = &self.scopes[m.0 as usize].kind {
                debug_assert!(!evaluating, "reactive cycle through memo {m:?}");
            }
            self.set_memo_evaluating(m, true);
            let v = self.evaluate_scope(m);
            self.set_memo_cache(m, v);
            self.scopes[m.0 as usize].dirty = false;
            self.set_memo_evaluating(m, false);
        }
        match &self.scopes[m.0 as usize].kind {
            ScopeKind::Memo { cache, .. } => cache.clone(),
            _ => Value::Unit,
        }
    }

    // ── mutation: the mark cascade (§4) ─────────────────────────────────

    /// Reads a signal's current value **without** tracking — for mutation
    /// l-values (`count++`) and other actions that are not reactive
    /// projections.
    #[must_use]
    pub fn peek_signal(&self, sig: SignalId) -> Value {
        self.signals[sig.0 as usize].value.clone()
    }

    /// Writes a new value to a signal and runs the synchronous mark cascade
    /// (the mutation entry point).
    pub fn write_signal(&mut self, sig: SignalId, value: Value) {
        self.signals[sig.0 as usize].value = value;
        self.mark(sig);
    }

    /// Marks every subscriber of `sig` dirty (§4).
    pub fn mark(&mut self, sig: SignalId) {
        let subs = self.signals[sig.0 as usize].subscribers.clone();
        for s in subs {
            self.mark_scope(s);
        }
    }

    fn mark_scope(&mut self, s: ScopeId) {
        {
            let scope = &mut self.scopes[s.0 as usize];
            if !scope.live || scope.dirty {
                return; // IDEMPOTENT — stop re-traversal (D1)
            }
            scope.dirty = true;
        }
        self.mark_effective += 1;
        // Extract the cascade target into owned data before recursing, so no
        // borrow of `self.scopes` is held across the recursive `mark_scope`.
        let memo_subs: Option<SmallVec<[ScopeId; 4]>> = match &self.scopes[s.0 as usize].kind {
            ScopeKind::ValueBinding { .. } => {
                self.dirty_bindings.push(s);
                None
            }
            ScopeKind::Structural { .. } => {
                self.dirty_structural.push(s);
                None
            }
            ScopeKind::Memo { subs, .. } => Some(subs.clone()),
        };
        if let Some(subs) = memo_subs {
            for sub in subs {
                self.mark_scope(sub); // cascade; memo not recomputed here
            }
        }
    }

    // ── the tick: begin + pull (§5) ─────────────────────────────────────

    /// Begins a tick, bumping the epoch. Each dirty scope evaluates at most
    /// once per epoch.
    pub fn begin_tick(&mut self) -> u32 {
        self.epoch += 1;
        self.frame_writes.clear();
        self.epoch
    }

    /// Pulls all dirty scopes for `epoch`: structural effects first (their
    /// mounts create new bindings), then value bindings, each guarded to one
    /// evaluation per epoch. The frame write is value-equality–gated (§5/§7).
    pub fn pull(&mut self, epoch: u32) {
        let structural: Vec<ScopeId> = self.dirty_structural.drain(..).collect();
        for s in structural {
            self.reconcile_structural(s, epoch);
        }
        // Reconciliation may have enqueued freshly-mounted bindings; process the
        // queue in FIFO order (so `for` children project in list order) and keep
        // going as it grows, until the whole tick settles.
        let mut i = 0;
        while i < self.dirty_bindings.len() {
            let s = self.dirty_bindings[i];
            i += 1;
            if !self.scopes[s.0 as usize].live {
                continue;
            }
            if self.scopes[s.0 as usize].last_epoch == epoch {
                continue; // EPOCH GUARD (D1)
            }
            self.scopes[s.0 as usize].last_epoch = epoch;
            let v = self.evaluate_scope(s);
            self.write_frame_field(s, v);
            self.scopes[s.0 as usize].dirty = false;
        }
        self.dirty_bindings.clear();
    }

    fn write_frame_field(&mut self, s: ScopeId, v: Value) {
        let target = match &self.scopes[s.0 as usize].kind {
            ScopeKind::ValueBinding { target, .. } => *target,
            _ => return,
        };
        let changed = match &self.scopes[s.0 as usize].kind {
            ScopeKind::ValueBinding { last_written, .. } => last_written.as_ref() != Some(&v),
            _ => false,
        };
        if changed {
            // value-equality cut: unchanged projection writes nothing (§7).
            if let ScopeKind::ValueBinding { last_written, .. } =
                &mut self.scopes[s.0 as usize].kind
            {
                *last_written = Some(v.clone());
            }
            self.frame_writes.push((target, v));
        }
    }

    // ── scope evaluation with dynamic-dep clearing (§3) ─────────────────

    fn evaluate_scope(&mut self, s: ScopeId) -> Value {
        self.clear_deps(s);
        let mut compute = self.computes[s.0 as usize]
            .take()
            .expect("scope has a computation (or a reactive cycle re-entered it)");
        let prev = CURRENT_SCOPE.replace(Some(s));
        let value = compute(self);
        CURRENT_SCOPE.set(prev);
        self.computes[s.0 as usize] = Some(compute);
        self.scopes[s.0 as usize].eval_count += 1;
        value
    }

    /// Removes `s` from each of its dependencies' subscriber lists, then empties
    /// its dep set (§3).
    fn clear_deps(&mut self, s: ScopeId) {
        let deps = std::mem::take(&mut self.scopes[s.0 as usize].deps);
        for dep in &deps {
            match dep {
                Dep::Signal(sig) => {
                    self.signals[sig.0 as usize].subscribers.retain(|x| *x != s);
                }
                Dep::Memo(m) => {
                    if let ScopeKind::Memo { subs, .. } = &mut self.scopes[m.0 as usize].kind {
                        subs.retain(|x| *x != s);
                    }
                }
            }
        }
    }

    // ── structural reconciliation (§8) ──────────────────────────────────

    fn reconcile_structural(&mut self, s: ScopeId, epoch: u32) {
        if !self.scopes[s.0 as usize].live || self.scopes[s.0 as usize].last_epoch == epoch {
            return;
        }
        self.scopes[s.0 as usize].last_epoch = epoch;
        let kind = match &self.scopes[s.0 as usize].kind {
            ScopeKind::Structural { kind } => *kind,
            _ => return,
        };
        let value = self.evaluate_scope(s);
        self.scopes[s.0 as usize].dirty = false;

        let mut mount = self.mounts[s.0 as usize]
            .take()
            .expect("structural scope has mount state");
        match (&mut mount, kind) {
            (
                Mount::When {
                    then,
                    els,
                    active,
                    children,
                },
                StructuralKind::When,
            ) => {
                let take = value.as_bool().unwrap_or(false);
                if *active != Some(take) {
                    for child in children.drain(..) {
                        self.drop_scope(child);
                    }
                    *active = Some(take);
                    *children = if take {
                        then(self)
                    } else if let Some(els) = els {
                        els(self)
                    } else {
                        Vec::new()
                    };
                }
            }
            (Mount::For { item, children }, StructuralKind::For) => {
                // Phase 2 coarse reconciliation (D7): drop all, rebuild.
                for child in children.drain(..) {
                    self.drop_scope(child);
                }
                let items = value.as_list().map(<[Value]>::to_vec).unwrap_or_default();
                for it in &items {
                    let mut mounted = item(self, it);
                    children.append(&mut mounted);
                }
            }
            _ => {}
        }
        self.mounts[s.0 as usize] = Some(mount);
    }

    /// Retires a scope on unmount: clears its subscriptions (so no `Signal` or
    /// memo keeps a stale link), drops any structural children recursively, and
    /// marks the slot dead. The §4.2 grid entry removal is the eval driver's
    /// job (M9/M10); here we guarantee no leaked reactive subscription (§8).
    fn drop_scope(&mut self, s: ScopeId) {
        if !self.scopes[s.0 as usize].live {
            return;
        }
        // Recursively drop structural children first.
        if let Some(Mount::When { children, .. } | Mount::For { children, .. }) =
            self.mounts[s.0 as usize].as_mut()
        {
            let children = std::mem::take(children);
            for child in children {
                self.drop_scope(child);
            }
        }
        self.clear_deps(s);
        // Remove this scope from any memo subscriber lists it appears in.
        let scope_is_memo = matches!(self.scopes[s.0 as usize].kind, ScopeKind::Memo { .. });
        if scope_is_memo {
            if let ScopeKind::Memo { subs, .. } = &mut self.scopes[s.0 as usize].kind {
                subs.clear();
            }
        }
        self.scopes[s.0 as usize].live = false;
        self.computes[s.0 as usize] = None;
        self.mounts[s.0 as usize] = None;
    }

    // ── small helpers ───────────────────────────────────────────────────

    fn subscribe_signal(&mut self, sig: SignalId, s: ScopeId) {
        let subs = &mut self.signals[sig.0 as usize].subscribers;
        if !subs.contains(&s) {
            subs.push(s);
        }
    }

    fn subscribe_memo(&mut self, m: ScopeId, s: ScopeId) {
        if let ScopeKind::Memo { subs, .. } = &mut self.scopes[m.0 as usize].kind {
            if !subs.contains(&s) {
                subs.push(s);
            }
        }
    }

    fn record_dep(&mut self, s: ScopeId, dep: Dep) {
        let deps = &mut self.scopes[s.0 as usize].deps;
        if !deps.contains(&dep) {
            deps.push(dep);
        }
    }

    fn set_memo_evaluating(&mut self, m: ScopeId, v: bool) {
        if let ScopeKind::Memo { evaluating, .. } = &mut self.scopes[m.0 as usize].kind {
            *evaluating = v;
        }
    }

    fn set_memo_cache(&mut self, m: ScopeId, v: Value) {
        if let ScopeKind::Memo { cache, .. } = &mut self.scopes[m.0 as usize].kind {
            *cache = v;
        }
    }

    // ── escape hatch + metadata (§9, §10) ───────────────────────────────

    /// Whether a scope read at least one source on its last evaluation — the
    /// D3 "is this `let`/`fn` reactive?" signal (§10), recorded in a side-table,
    /// never on the AST.
    #[must_use]
    pub fn deps_nonempty(&self, s: ScopeId) -> bool {
        !self.scopes[s.0 as usize].deps.is_empty()
    }

    // ── observability (tests) ───────────────────────────────────────────

    /// How many times `s` has been evaluated since creation.
    #[must_use]
    pub fn eval_count(&self, s: ScopeId) -> u32 {
        self.scopes[s.0 as usize].eval_count
    }

    /// Number of frame-field writes performed since the last
    /// [`ReactiveCtx::begin_tick`] (an unchanged projection records none).
    #[must_use]
    pub fn frame_write_count(&self) -> usize {
        self.frame_writes.len()
    }

    /// Number of scopes currently subscribed to `sig` (leak check for §7/§8).
    #[must_use]
    pub fn signal_subscriber_count(&self, sig: SignalId) -> usize {
        self.signals[sig.0 as usize].subscribers.len()
    }

    /// The most recently projected value of a value binding (its `last_written`).
    #[must_use]
    pub fn binding_value(&self, s: ScopeId) -> Option<Value> {
        match &self.scopes[s.0 as usize].kind {
            ScopeKind::ValueBinding { last_written, .. } => last_written.clone(),
            _ => None,
        }
    }

    /// The frame-field writes performed since the last
    /// [`ReactiveCtx::begin_tick`].
    #[must_use]
    pub fn frame_writes(&self) -> &[(FrameTarget, Value)] {
        &self.frame_writes
    }

    /// Cumulative count of *effective* mark visits (each scope dirtied once),
    /// proving the cascade is idempotent (fixture 2).
    #[must_use]
    pub fn mark_effective_visits(&self) -> u32 {
        self.mark_effective
    }
}

/// Evaluates `thunk` with read-tracking suspended (`CURRENT_SCOPE = None`), so
/// `read_signal`/`read_memo` inside install no subscription, then restores the
/// previous scope — correctly nested, even on unwind (§9 / D2).
pub fn untrack<R>(thunk: impl FnOnce() -> R) -> R {
    struct Restore(Option<ScopeId>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT_SCOPE.set(self.0);
        }
    }
    let _guard = Restore(CURRENT_SCOPE.replace(None));
    thunk()
}

#[cfg(test)]
mod tests;
