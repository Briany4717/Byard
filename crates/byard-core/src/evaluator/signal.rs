//! Reactive value handles backed by arena-allocated slots.
//!
//! A [`Signal<'a, T>`] is a `Copy` handle that points to an internal slot
//! living inside a [`ViewArena`]. Mutating a signal
//! does **not** rebuild a virtual tree: it updates the value in place and
//! increments an atomic version counter so downstream subsystems can detect
//! changes without taking locks.
//!
//! This implements the reactivity model described in RFC-0001 §2.2.
//!
//! # Lifetime binding
//!
//! `Signal<'a, T>` carries the lifetime of the arena that allocated it. Safe
//! code cannot construct a `Signal` that outlives its arena, eliminating any
//! possibility of use-after-free.
//!
//! # Thread safety
//!
//! `Signal<'a, T>` is `!Send` and `!Sync` by construction. Per RFC-0001 §5.1,
//! signals are only ever read or mutated from the Logic thread; the
//! compiler enforces this statically.
//!
//! # Reentrancy
//!
//! Because `Signal` is `Copy`, two copies of the same handle could in
//! principle alias each other on the same thread. To prevent this, each
//! slot carries a runtime borrow counter. Nested `read` calls are allowed
//! (multiple shared borrows); nested `write` calls, or a `write` nested
//! inside a `read`, will panic with a clear message rather than producing
//! aliased references.

#![allow(unsafe_code)]

use std::cell::{Cell, UnsafeCell};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use super::arena::ViewArena;
use crate::frame::TargetId;

/// The arena-allocated backing storage for a [`Signal<T>`].
///
/// Crate-private: external code interacts with `Signal` handles only.
pub(crate) struct SignalSlot<T> {
    value: UnsafeCell<T>,
    dirty_targets: UnsafeCell<Vec<TargetId>>,
    dirty_version: AtomicU64,
    /// Runtime borrow counter. Positive = shared borrows in progress;
    /// `BORROW_MUT_SENTINEL` = exclusive borrow in progress; 0 = idle.
    borrow_state: Cell<isize>,
}

impl<T> SignalSlot<T> {
    /// Returns a reference to the atomic version counter.
    ///
    /// Used by [`EvaluatorTick`](crate::evaluator::EvaluatorTick) to poll
    /// versions without going through `Signal`'s borrow-guarded API.
    /// Safe because `AtomicU64` is itself thread-safe.
    pub(crate) fn dirty_version_ref(&self) -> &AtomicU64 {
        &self.dirty_version
    }

    /// Returns a shared reference to the subscriber list.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that no exclusive borrow of `dirty_targets`
    /// is active. This is upheld by the Logic-thread-only invariant when
    /// called from `EvaluatorTick::collect_dirty`.
    pub(crate) unsafe fn subscribers_ref(&self) -> &[TargetId] {
        // SAFETY: caller upholds the contract above.
        unsafe { (*self.dirty_targets.get()).as_slice() }
    }
}

/// Marker value for `borrow_state` indicating an exclusive borrow.
const BORROW_MUT_SENTINEL: isize = -1;

/// A `Copy` handle to a reactive value living in a [`ViewArena`].
///
/// The lifetime parameter `'a` ties the handle to its owning arena,
/// preventing use-after-free in safe code.
pub struct Signal<'a, T: 'static> {
    slot: NonNull<SignalSlot<T>>,
    _arena: PhantomData<&'a SignalSlot<T>>,
    _not_send: PhantomData<*mut ()>,
}

/// RAII guard that restores `borrow_state` when dropped.
///
/// Decrementing back to a known state on drop ensures the slot's
/// reentrancy counter is correct even if the user's closure panics.
struct BorrowGuard<'g> {
    state: &'g Cell<isize>,
    on_drop: isize,
}

impl<'g> BorrowGuard<'g> {
    fn shared(state: &'g Cell<isize>) -> Self {
        let current = state.get();
        state.set(current + 1);
        Self {
            state,
            on_drop: current,
        }
    }
}

impl Drop for BorrowGuard<'_> {
    fn drop(&mut self) {
        self.state.set(self.on_drop);
    }
}

/// RAII guard for exclusive borrows that also bumps `dirty_version`
/// atomically on drop — even if the user closure panics.
///
/// This guarantees that any observable mutation of the value is reflected
/// in the version counter before control returns to the caller (whether
/// via normal return or unwinding).
struct WriteGuard<'g> {
    state: &'g Cell<isize>,
    version: Option<&'g AtomicU64>,
}

impl<'g> WriteGuard<'g> {
    fn new(state: &'g Cell<isize>, version: &'g AtomicU64) -> Self {
        state.set(BORROW_MUT_SENTINEL);
        Self {
            state,
            version: Some(version),
        }
    }

    fn for_subscribe(state: &'g Cell<isize>) -> Self {
        state.set(BORROW_MUT_SENTINEL);
        Self {
            state,
            version: None,
        }
    }
}

impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        self.state.set(0);
        if let Some(version) = self.version {
            version.fetch_add(1, Ordering::Release);
        }
    }
}

// `Signal` is intentionally `Copy`: it is a thin handle, not a value.
impl<T: 'static> Copy for Signal<'_, T> {}
impl<T: 'static> Clone for Signal<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: 'static> Signal<'a, T> {
    /// Allocates a new signal inside `arena` with the given initial value.
    #[must_use]
    pub fn new_in(arena: &'a ViewArena, initial: T) -> Self {
        let slot = arena.alloc(SignalSlot {
            value: UnsafeCell::new(initial),
            dirty_targets: UnsafeCell::new(Vec::new()),
            dirty_version: AtomicU64::new(0),
            borrow_state: Cell::new(0),
        });
        let slot = NonNull::from(slot);
        Self {
            slot,
            _arena: PhantomData,
            _not_send: PhantomData,
        }
    }

    /// Reads the current value of the signal.
    ///
    /// # Panics
    ///
    /// Panics if a `write` is currently in progress on the same slot
    /// (i.e. this is a `read` nested inside a `write` via another copy
    /// of the handle).
    pub fn read<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        // SAFETY: the slot is alive (lifetime `'a` ensures the arena
        // outlives this handle). `Signal` is `!Send`, so we are the only
        // thread that can access this slot.
        let slot = unsafe { self.slot.as_ref() };

        let state = slot.borrow_state.get();
        assert!(
            state >= 0,
            "Signal::read called while a Signal::write is in progress on the same slot",
        );
        let _guard = BorrowGuard::shared(&slot.borrow_state);

        // SAFETY: borrow counter is positive, so no `write` can produce a
        // mutable reference concurrently. Multiple shared borrows are fine.
        let value: &T = unsafe { &*slot.value.get() };
        f(value)
    }

    /// Mutates the value of the signal and increments the version counter
    /// atomically.
    ///
    /// The version increment happens inside `WriteGuard`'s `Drop`, which
    /// runs after the user closure returns (or while unwinding from a
    /// panic). This guarantees that any observable mutation is reflected
    /// in `Signal::version` before control returns to the caller, even if
    /// the closure panics.
    ///
    /// # Panics
    ///
    /// Panics if any other borrow (read or write) is currently in
    /// progress on the same slot.
    pub fn write<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        // SAFETY: the slot is alive (lifetime `'a` ensures the arena
        // outlives this handle). `Signal` is `!Send`, so we are the only
        // thread that can access this slot.
        let slot = unsafe { self.slot.as_ref() };

        let state = slot.borrow_state.get();
        assert_eq!(
            state, 0,
            "Signal::write called while another borrow is in progress on the same slot \
         (current borrow_state = {state})"
        );
        let _guard = WriteGuard::new(&slot.borrow_state, &slot.dirty_version);

        // SAFETY: borrow counter is at the exclusive sentinel, so no other
        // reference (shared or exclusive) to this value exists.
        let value: &mut T = unsafe { &mut *slot.value.get() };
        f(value)
    }

    /// Registers a dependency on this signal.
    ///
    /// # Panics
    ///
    /// Panics if any other borrow on the same slot is in progress — including
    /// `read`, `write`, or `with_subscribers` via a copied handle.
    pub fn subscribe(&self, target: TargetId) {
        // SAFETY: see `read`.
        let slot = unsafe { self.slot.as_ref() };

        let state = slot.borrow_state.get();
        assert_eq!(
            state, 0,
            "Signal::subscribe called while another borrow is in progress on the same slot \
         (current borrow_state = {state})"
        );
        let _guard = WriteGuard::for_subscribe(&slot.borrow_state);

        // SAFETY: borrow counter is at the exclusive sentinel; no other
        // reference to `dirty_targets` exists.
        let targets: &mut Vec<TargetId> = unsafe { &mut *slot.dirty_targets.get() };
        targets.push(target);
    }

    /// Calls `f` with the targets currently registered on this signal.
    ///
    /// The subscriber list is borrowed only for the duration of the closure.
    /// This is the preferred API for hot-path iteration.
    ///
    /// # Panics
    ///
    /// Panics if a `write` or `subscribe` is currently in progress on the
    /// same slot through a copied handle.
    pub fn with_subscribers<R>(&self, f: impl FnOnce(&[TargetId]) -> R) -> R {
        // SAFETY: see `read`.
        let slot = unsafe { self.slot.as_ref() };

        let state = slot.borrow_state.get();
        assert!(
            state >= 0,
            "Signal::with_subscribers called while an exclusive borrow is in progress",
        );
        let _guard = BorrowGuard::shared(&slot.borrow_state);

        // SAFETY: borrow counter is positive; no mutator can run concurrently.
        let targets: &Vec<TargetId> = unsafe { &*slot.dirty_targets.get() };
        f(targets.as_slice())
    }

    /// Returns an owned snapshot of the targets currently registered.
    ///
    /// Allocates. Prefer [`Signal::with_subscribers`] in hot paths.
    #[must_use]
    pub fn subscribers(&self) -> Vec<TargetId> {
        self.with_subscribers(<[TargetId]>::to_vec)
    }

    /// Returns the current version counter of this signal.
    #[must_use]
    pub fn version(&self) -> u64 {
        // SAFETY: `dirty_version` is atomic; safe to load through a
        // shared reference.
        unsafe { self.slot.as_ref() }
            .dirty_version
            .load(Ordering::Acquire)
    }

    /// Returns the raw address of this signal's backing slot.
    ///
    /// Used internally by [`EvaluatorTick`](crate::evaluator::EvaluatorTick) to
    /// detect duplicate registrations in debug builds. Two `Signal` handles
    /// that compare equal by this pointer refer to the same underlying slot.
    ///
    /// The returned pointer is opaque — it must not be dereferenced.
    #[must_use]
    pub(crate) fn slot_ptr(self) -> *const () {
        self.slot.as_ptr().cast()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_returns_initial_value() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 42_u32);
        assert_eq!(signal.read(|v| *v), 42);
    }

    #[test]
    fn write_updates_value() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.write(|v| *v = 7);
        assert_eq!(signal.read(|v| *v), 7);
    }

    #[test]
    fn write_can_use_previous_value() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 10_i32);
        signal.write(|v| *v += 5);
        signal.write(|v| *v *= 2);
        assert_eq!(signal.read(|v| *v), 30);
    }

    #[test]
    fn read_works_with_non_copy_types() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, String::from("hello"));
        let len = signal.read(String::len);
        assert_eq!(len, 5);
        signal.write(|s| s.push_str(", world"));
        assert_eq!(signal.read(String::clone), "hello, world");
    }

    #[test]
    fn signal_is_copy() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 100_u32);
        let copy = signal;
        copy.write(|v| *v = 200);
        assert_eq!(signal.read(|v| *v), 200);
    }

    #[test]
    fn subscribers_start_empty() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        assert!(signal.subscribers().is_empty());
    }

    #[test]
    fn subscribe_adds_targets_in_order() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        let a = TargetId::new(1, 0, 0);
        let b = TargetId::new(2, 0, 0);
        let c = TargetId::new(3, 0, 0);

        signal.subscribe(a);
        signal.subscribe(b);
        signal.subscribe(c);

        assert_eq!(signal.subscribers(), vec![a, b, c]);
    }

    #[test]
    fn version_starts_at_zero() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        assert_eq!(signal.version(), 0);
    }

    #[test]
    fn write_increments_version() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);

        signal.write(|v| *v = 1);
        assert_eq!(signal.version(), 1);

        signal.write(|v| *v = 2);
        assert_eq!(signal.version(), 2);
    }

    #[test]
    fn read_does_not_change_version() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 42_u32);
        let _ = signal.read(|v| *v);
        assert_eq!(signal.version(), 0);
    }

    #[test]
    fn version_observable_across_copies_of_handle() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        let copy = signal;
        signal.write(|v| *v = 100);
        assert_eq!(signal.version(), 1);
        assert_eq!(copy.version(), 1);
    }

    #[test]
    fn drop_runs_for_non_trivial_signal_values() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct Guard(Rc<Cell<u32>>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let counter = Rc::new(Cell::new(0));
        {
            let arena = ViewArena::new();
            let _s = Signal::new_in(&arena, Guard(Rc::clone(&counter)));
            assert_eq!(counter.get(), 0);
        }
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn nested_reads_allowed() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 10_u32);
        let result = signal.read(|outer| signal.read(|inner| *outer + *inner));
        assert_eq!(result, 20);
    }

    #[test]
    #[should_panic(expected = "Signal::write called while another borrow is in progress")]
    fn write_inside_read_panics() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.read(|_| {
            signal.write(|v| *v = 1);
        });
    }

    #[test]
    #[should_panic(expected = "Signal::read called while a Signal::write is in progress")]
    fn read_inside_write_panics() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.write(|_| {
            signal.read(|v| *v);
        });
    }

    #[test]
    #[should_panic(expected = "Signal::write called while another borrow is in progress")]
    fn nested_write_panics() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.write(|_| {
            signal.write(|v| *v = 1);
        });
    }

    #[test]
    fn write_panic_still_marks_signal_dirty() {
        use std::panic;

        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            signal.write(|v| {
                *v = 99;
                panic!("intentional");
            });
        }));

        assert!(
            result.is_err(),
            "the panic should propagate to catch_unwind"
        );
        assert_eq!(
            signal.version(),
            1,
            "version must reflect the observable mutation"
        );
        assert_eq!(signal.read(|v| *v), 99, "the value was actually mutated");
    }

    #[test]
    fn with_subscribers_borrows_without_allocating() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.subscribe(TargetId::new(1, 0, 0));
        signal.subscribe(TargetId::new(2, 0, 0));

        let sum: u32 = signal.with_subscribers(|targets| targets.iter().map(|t| t.index()).sum());
        assert_eq!(sum, 3);
    }

    #[test]
    #[should_panic(expected = "Signal::subscribe called while another borrow is in progress")]
    fn subscribe_inside_with_subscribers_panics() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        let copy = signal;
        signal.with_subscribers(|_| {
            copy.subscribe(TargetId::new(1, 0, 0));
        });
    }

    #[test]
    #[should_panic(expected = "Signal::with_subscribers called while an exclusive borrow")]
    fn with_subscribers_inside_write_panics() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        let copy = signal;
        signal.write(|_| {
            copy.with_subscribers(|_| {});
        });
    }

    #[test]
    fn subscribe_does_not_change_version() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.subscribe(TargetId::new(1, 0, 0));
        signal.subscribe(TargetId::new(2, 0, 0));
        assert_eq!(signal.version(), 0, "subscribe is not a value mutation");
    }
}
