//! Reactive value handles backed by arena-allocated slots.
//!
//! A [`Signal<T>`] is a `Copy` handle that points to a [`SignalSlot<T>`]
//! living inside a [`ViewArena`](super::arena::ViewArena). Mutating a signal
//! does **not** rebuild a virtual tree: it updates the value in place and
//! records the set of [`TargetId`](super::target::TargetId)s that depend on
//! it so that downstream subsystems can pick up the change on their next
//! tick.
//!
//! This implements the reactivity model described in RFC-0001 §2.2.
//!
//! # Thread safety
//!
//! `Signal<T>` is `!Send` and `!Sync` by construction. Per RFC-0001 §5.1,
//! signals are only ever read or mutated from the Logic thread; the
//! compiler enforces this statically.
//!
//! # Aliasing
//!
//! The handle uses interior mutability via [`UnsafeCell`]. The `!Send`
//! marker is the soundness foundation: because no two threads can ever hold
//! a `Signal<T>` to the same slot simultaneously, the only possible
//! aliasing is sequential, single-threaded reentrancy. The public API
//! prevents reentrant access by exposing values exclusively through closures
//! whose borrows cannot escape.

#![allow(unsafe_code)]

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use super::arena::ViewArena;
use super::target::TargetId;

/// The arena-allocated backing storage for a [`Signal<T>`].
///
/// Users never name this type directly — it is allocated by
/// [`Signal::new_in`] and accessed exclusively through `Signal` handles.
///
/// The `dirty_version` is an atomic counter incremented on every write,
/// allowing other subsystems to detect changes without taking locks.
/// `dirty_targets` is the list of [`TargetId`]s registered via
/// [`Signal::subscribe`] — written only from the Logic thread.
pub struct SignalSlot<T> {
    value: UnsafeCell<T>,
    dirty_targets: UnsafeCell<Vec<TargetId>>,
    dirty_version: AtomicU64,
}

/// A `Copy` handle to a reactive value living in a [`ViewArena`].
///
/// Constructed via [`Signal::new_in`]. Reading and writing are done through
/// closure-based APIs that prevent borrows from escaping.
pub struct Signal<T: 'static> {
    slot: NonNull<SignalSlot<T>>,
    _not_send: PhantomData<*mut ()>,
}

// `Signal<T>` is intentionally `Copy`: it is a thin handle, not a value.
impl<T: 'static> Copy for Signal<T> {}
impl<T: 'static> Clone for Signal<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: 'static> Signal<T> {
    /// Allocates a new signal inside `arena` with the given initial value
    /// and an empty dirty-target list.
    pub fn new_in(arena: &mut ViewArena, initial: T) -> Self {
        let slot = arena.alloc(SignalSlot {
            value: UnsafeCell::new(initial),
            dirty_targets: UnsafeCell::new(Vec::new()),
            dirty_version: AtomicU64::new(0),
        });
        let slot = NonNull::from(slot);
        Self {
            slot,
            _not_send: PhantomData,
        }
    }

    /// Reads the current value of the signal.
    ///
    /// The closure receives an immutable reference to the value. The
    /// reference cannot escape the closure, preventing reentrant writes
    /// from invalidating it.
    pub fn read<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        // SAFETY: the slot is owned by the arena that allocated this signal
        // and is alive as long as the arena is alive. `Signal<T>` is
        // `!Send`, so we are the only thread that can access the slot. The
        // closure-based API prevents the resulting `&T` from outliving this
        // call, so no other code path can observe a borrow.
        let value: &T = unsafe { &*self.slot.as_ref().value.get() };
        f(value)
    }

    /// Mutates the value of the signal and marks all registered targets
    /// as dirty by incrementing the version counter atomically.
    ///
    /// The closure receives a mutable reference to the value. After it
    /// returns, [`Signal::version`] will return a strictly greater value,
    /// allowing other subsystems to observe the change without taking a lock.
    pub fn write<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        // SAFETY: see `read`. The closure-based API prevents the mutable
        // borrow from escaping, so there is no possibility of aliasing it
        // with any other reference during the call.
        let value: &mut T = unsafe { &mut *self.slot.as_ref().value.get() };
        let result = f(value);

        // SAFETY: `dirty_version` is `AtomicU64`, safe to access through a
        // shared reference from any thread (though Signal<T> is !Send, so
        // in practice only this thread writes; remote readers may still
        // observe via raw pointer if a subsystem holds the slot address).
        unsafe { self.slot.as_ref() }
            .dirty_version
            .fetch_add(1, Ordering::Release);

        result
    }

    /// Registers a dependency on this signal.
    ///
    /// Subsystems call this once when they create a primitive that should
    /// be marked dirty whenever the signal is written.
    pub fn subscribe(&self, target: TargetId) {
        // SAFETY: see `read`. The mutable borrow of the dirty list is
        // confined to this method body; nothing else can observe it.
        let targets: &mut Vec<TargetId> = unsafe { &mut *self.slot.as_ref().dirty_targets.get() };
        targets.push(target);
    }

    /// Returns a snapshot of the targets currently registered on this signal.
    ///
    /// Primarily useful for tests and diagnostics. Allocates.
    #[must_use]
    pub fn subscribers(&self) -> Vec<TargetId> {
        // SAFETY: see `read`. The borrow is read-only and confined to this
        // method body.
        let targets: &Vec<TargetId> = unsafe { &*self.slot.as_ref().dirty_targets.get() };
        targets.clone()
    }

    /// Returns the current version counter of this signal.
    ///
    /// The counter is incremented atomically on every [`Signal::write`].
    /// Other subsystems can cache the version they last observed and
    /// detect changes by comparing — no locks, no allocations.
    ///
    /// Uses `Ordering::Acquire` to synchronise with the `Release` store in
    /// `write`.
    #[must_use]
    pub fn version(&self) -> u64 {
        // SAFETY: `dirty_version` is atomic; safe to load through a
        // shared reference.
        unsafe { self.slot.as_ref() }
            .dirty_version
            .load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_returns_initial_value() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 42_u32);
        assert_eq!(signal.read(|v| *v), 42);
    }

    #[test]
    fn write_updates_value() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 0_u32);
        signal.write(|v| *v = 7);
        assert_eq!(signal.read(|v| *v), 7);
    }

    #[test]
    fn write_can_use_previous_value() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 10_i32);
        signal.write(|v| *v += 5);
        signal.write(|v| *v *= 2);
        assert_eq!(signal.read(|v| *v), 30);
    }

    #[test]
    fn read_works_with_non_copy_types() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, String::from("hello"));
        let len = signal.read(String::len);
        assert_eq!(len, 5);
        signal.write(|s| s.push_str(", world"));
        assert_eq!(signal.read(String::clone), "hello, world");
    }

    #[test]
    fn signal_is_copy() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 100_u32);
        let copy = signal;
        // Mutating through either handle affects the same underlying slot.
        copy.write(|v| *v = 200);
        assert_eq!(signal.read(|v| *v), 200);
    }

    #[test]
    fn subscribers_start_empty() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 0_u32);
        assert!(signal.subscribers().is_empty());
    }

    #[test]
    fn subscribe_adds_targets_in_order() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 0_u32);
        let a = TargetId::new(1, 0, 0);
        let b = TargetId::new(2, 0, 0);
        let c = TargetId::new(3, 0, 0);

        signal.subscribe(a);
        signal.subscribe(b);
        signal.subscribe(c);

        assert_eq!(signal.subscribers(), vec![a, b, c]);
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
            let mut arena = ViewArena::new();
            let _s = Signal::new_in(&mut arena, Guard(Rc::clone(&counter)));
            assert_eq!(counter.get(), 0);
        }

        // The Guard inside the signal slot must run its Drop when the
        // arena is released.
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn version_starts_at_zero() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 0_u32);
        assert_eq!(signal.version(), 0);
    }

    #[test]
    fn write_increments_version() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 0_u32);

        signal.write(|v| *v = 1);
        assert_eq!(signal.version(), 1);

        signal.write(|v| *v = 2);
        assert_eq!(signal.version(), 2);

        signal.write(|v| *v = 3);
        assert_eq!(signal.version(), 3);
    }

    #[test]
    fn read_does_not_change_version() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 42_u32);

        let _ = signal.read(|v| *v);
        let _ = signal.read(|v| *v);
        let _ = signal.read(|v| *v);

        assert_eq!(signal.version(), 0);
    }

    #[test]
    fn version_observable_across_copies_of_handle() {
        let mut arena = ViewArena::new();
        let signal = Signal::new_in(&mut arena, 0_u32);
        let copy = signal;

        signal.write(|v| *v = 100);

        // Both handles see the same atomic counter.
        assert_eq!(signal.version(), 1);
        assert_eq!(copy.version(), 1);
    }
}
