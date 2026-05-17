#![allow(unsafe_code)]
//! Per-view memory arena with deterministic destruction.
//!
//! `ViewArena` wraps a [`bumpalo::Bump`] allocator and a registry of `Drop`
//! glue functions. Values allocated into the arena have their destructors
//! recorded; when the arena is dropped, every registered destructor runs in
//! a single linear pass before the underlying memory is released.
//!
//! This is the implementation of the "apoptosis" model described in
//! RFC-0001 §2.1: `O(1)` amortised destruction with no global heap
//! fragmentation and no garbage collector.

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
/// dropped, at which point every value's `Drop` runs in registration order,
/// then the underlying memory is released in one operation.
///
/// `ViewArena` is `!Send` and `!Sync` by construction — it must only be
/// touched from the Logic thread (see RFC-0001 §5.1).
pub struct ViewArena {
    bump: Bump,
    drops: Vec<DropEntry>,
    _not_send: PhantomData<*mut ()>,
}

impl ViewArena {
    /// Creates a new, empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bump: Bump::new(),
            drops: Vec::new(),
            _not_send: PhantomData,
        }
    }

    /// Allocates `value` inside the arena and returns a mutable reference
    /// valid for the arena's lifetime.
    ///
    /// If `T` needs to be dropped (per [`std::mem::needs_drop`]), a
    /// destructor entry is registered so that `Drop` runs when the arena
    /// is released.
    pub fn alloc<T: 'static>(&mut self, value: T) -> &mut T {
        let slot: &mut T = self.bump.alloc(value);

        if std::mem::needs_drop::<T>() {
            self.drops.push(DropEntry {
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
        self.drops.len()
    }
}

impl Default for ViewArena {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ViewArena {
    fn drop(&mut self) {
        // Run all registered destructors in registration order, then let
        // `Bump` release the underlying memory when it drops.
        for entry in self.drops.drain(..) {
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
        let mut arena = ViewArena::new();
        let n = arena.alloc(42_u64);
        assert_eq!(*n, 42);
        assert_eq!(arena.pending_drops(), 0);
    }

    #[test]
    fn registers_drop_for_non_trivial_types() {
        let mut arena = ViewArena::new();
        let _s = arena.alloc(String::from("hello"));
        assert_eq!(arena.pending_drops(), 1);
    }

    #[test]
    fn runs_registered_drops_on_arena_drop() {
        // A counter shared between the test and a guard whose Drop
        // increments it. If the arena drops the guard correctly, the
        // counter reaches the expected value.
        struct Guard(Rc<Cell<u32>>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let counter = Rc::new(Cell::new(0));

        {
            let mut arena = ViewArena::new();
            arena.alloc(Guard(Rc::clone(&counter)));
            arena.alloc(Guard(Rc::clone(&counter)));
            arena.alloc(Guard(Rc::clone(&counter)));
            assert_eq!(counter.get(), 0, "no drops should run before arena drop");
        }

        assert_eq!(counter.get(), 3, "all three drops must run when arena dies");
    }

    #[test]
    fn drops_run_in_registration_order() {
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
            let mut arena = ViewArena::new();
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

        assert_eq!(order.take(), vec![1, 2, 3]);
    }

    #[test]
    fn allocates_heterogeneous_types() {
        let mut arena = ViewArena::new();

        {
            let a: &mut u32 = arena.alloc(1);
            assert_eq!(*a, 1);
        }
        {
            let b: &mut String = arena.alloc(String::from("two"));
            assert_eq!(b, "two");
        }
        {
            let c: &mut Vec<u8> = arena.alloc(vec![3, 3, 3]);
            assert_eq!(c, &[3, 3, 3]);
        }

        assert_eq!(arena.pending_drops(), 2, "u32 is trivially droppable");
    }
}
