//! Per-view memory arena with deterministic destruction.
//!
//! `ViewArena` wraps a [`bumpalo::Bump`] allocator and a registry of `Drop`
//! glue functions. Values allocated into the arena have their destructors
//! recorded; when the arena is dropped, every registered destructor runs in
//! reverse registration order before the underlying memory is released.
//!
//! This is the implementation of the apoptosis model described in
//! RFC-0001 2.1: deterministic destruction with no global heap
//! fragmentation and no garbage collector. The drop pass itself is
//! `O(n)` in the number of registered destructors; the underlying memory
//! release is then a single `O(1)` operation.
//!
//! # Allocation through shared references
//!
//! `alloc` takes `&self`, not `&mut self`. This is the standard `bumpalo`
//! pattern: the underlying `Bump` already supports it, and the drop
//! registry uses interior mutability (a `RefCell`) so that multiple
//! handles to arena-allocated values can coexist without aliasing the
//! arena itself.
//!
//! `ViewArena` is `!Send` and `!Sync` — it must only be touched from the
//! Logic thread (see RFC-0001 5.1).

#![allow(unsafe_code)]

use std::cell::RefCell;
use std::marker::PhantomData;
use std::ptr;

use bumpalo::Bump;

/// Type-erased destructor entry.
///
/// Each entry stores a raw pointer to a value living in the bump arena and
/// a monomorphised function pointer that calls `ptr::drop_in_place` for the
/// concrete type. No `Box`, no vtable.
struct DropEntry {
    ptr: *mut u8,
    drop_fn: unsafe fn(*mut u8),
}

/// Monomorphised drop glue for a concrete type `T`.
///
/// # Safety
///
/// The caller must guarantee that `ptr` points to a valid, initialised `T`
/// that has not already been dropped.
unsafe fn drop_glue<T>(ptr: *mut u8) {
    // SAFETY: caller upholds the contract documented above.
    unsafe { ptr::drop_in_place(ptr.cast::<T>()) };
}

/// A contiguous memory arena scoped to the lifetime of a single `View`.
///
/// Allocate values with [`ViewArena::alloc`]; they live until the arena is
/// dropped, at which point every registered `Drop` runs in **reverse
/// registration order** (LIFO), then the underlying memory is released in
/// one operation.
///
/// LIFO ordering matches Rust's standard RAII semantics for stack values
/// and is required for panic safety: each destructor pops from the
/// registry before running, so a panic mid-pass leaves the remaining
/// entries in the `Vec` to be cleaned up by its own `Drop`.
///
/// `ViewArena` is `!Send` and `!Sync` by construction.
pub struct ViewArena {
    bump: Bump,
    drops: RefCell<Vec<DropEntry>>,
    _not_send: PhantomData<*mut ()>,
}

impl ViewArena {
    /// Creates a new, empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bump: Bump::new(),
            drops: RefCell::new(Vec::new()),
            _not_send: PhantomData,
        }
    }

    /// Allocates `value` inside the arena and returns a reference valid for
    /// the arena's lifetime.
    ///
    /// Takes `&self` so multiple allocations can coexist with their
    /// resulting references. If `T` needs to be dropped (per
    /// [`std::mem::needs_drop`]), a destructor entry is registered so
    /// `Drop` runs when the arena is released.
    pub fn alloc<T: 'static>(&self, value: T) -> &mut T {
        let slot: &mut T = self.bump.alloc(value);

        if std::mem::needs_drop::<T>() {
            self.drops.borrow_mut().push(DropEntry {
                ptr: ptr::from_mut(slot).cast::<u8>(),
                drop_fn: drop_glue::<T>,
            });
        }

        slot
    }

    /// Returns the number of pending destructors registered in this arena.
    ///
    /// Primarily useful for tests and diagnostics.
    #[must_use]
    pub fn pending_drops(&self) -> usize {
        self.drops.borrow().len()
    }
}

impl Default for ViewArena {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ViewArena {
    fn drop(&mut self) {
        // Pop-based loop: each iteration is independent, so a panicking
        // destructor leaves the remaining entries in `self.drops`. The
        // `Vec` then runs its own `Drop` during unwinding, ensuring the
        // remaining `DropEntry` records are released (though the inner
        // values they pointed at will leak — Rust's normal behaviour on
        // panic during drop).
        let mut drops = self.drops.borrow_mut();
        while let Some(entry) = drops.pop() {
            // SAFETY: each `entry.ptr` was produced by `Bump::alloc` for a
            // value of the type that `entry.drop_fn` was monomorphised for,
            // and the value has not been dropped before.
            unsafe { (entry.drop_fn)(entry.ptr) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn allocates_trivially_droppable_types_without_registering_drops() {
        let arena = ViewArena::new();
        let n = arena.alloc(42_u64);
        assert_eq!(*n, 42);
        assert_eq!(arena.pending_drops(), 0);
    }

    #[test]
    fn registers_drop_for_non_trivial_types() {
        let arena = ViewArena::new();
        let _s = arena.alloc(String::from("hello"));
        assert_eq!(arena.pending_drops(), 1);
    }

    #[test]
    fn runs_registered_drops_on_arena_drop() {
        struct Guard(Rc<Cell<u32>>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let counter = Rc::new(Cell::new(0));

        {
            let arena = ViewArena::new();
            arena.alloc(Guard(Rc::clone(&counter)));
            arena.alloc(Guard(Rc::clone(&counter)));
            arena.alloc(Guard(Rc::clone(&counter)));
            assert_eq!(counter.get(), 0);
        }

        assert_eq!(counter.get(), 3);
    }

    #[test]
    fn drops_run_in_reverse_registration_order() {
        // With a pop-based loop, the most recently registered drop runs
        // first. This matches Rust's RAII semantics for stack values
        // (LIFO) and is panic-safe.
        struct OrderedGuard {
            id: u32,
            log: Rc<Cell<Vec<u32>>>,
        }
        impl Drop for OrderedGuard {
            fn drop(&mut self) {
                let mut v = self.log.take();
                v.push(self.id);
                self.log.set(v);
            }
        }

        let order = Rc::new(Cell::new(Vec::<u32>::new()));

        {
            let arena = ViewArena::new();
            arena.alloc(OrderedGuard {
                id: 1,
                log: Rc::clone(&order),
            });
            arena.alloc(OrderedGuard {
                id: 2,
                log: Rc::clone(&order),
            });
            arena.alloc(OrderedGuard {
                id: 3,
                log: Rc::clone(&order),
            });
        }

        assert_eq!(order.take(), vec![3, 2, 1]);
    }

    #[test]
    fn allocates_heterogeneous_types() {
        let arena = ViewArena::new();

        let a = arena.alloc(1_u32);
        let b = arena.alloc(String::from("two"));
        let c = arena.alloc(vec![3_u8, 3, 3]);

        assert_eq!(*a, 1);
        assert_eq!(b, "two");
        assert_eq!(c, &[3, 3, 3]);
        assert_eq!(arena.pending_drops(), 2, "u32 is trivially droppable");
    }
}
