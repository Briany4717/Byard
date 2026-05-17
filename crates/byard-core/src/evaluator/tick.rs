//! Dirty-flag collection loop for the Evaluator subsystem.
//!
//! `EvaluatorTick` is the coordination layer that consumes the atomic
//! version counters of registered [`Signal`]s and produces the set of
//! [`TargetId`]s that downstream subsystems (Atlas, Encoder) must
//! recompute for the current tick.
//!
//! This implements the tick cycle described in RFC-0001 §2.2.
//!
//! # Model
//!
//! 1. The Logic thread registers each `Signal` it wants tracked, once,
//!    via [`EvaluatorTick::register`].
//! 2. On every tick, the Logic thread calls
//!    [`EvaluatorTick::collect_dirty`], which:
//!    - Compares each registered signal's current `version()` against
//!      the version observed at the previous tick.
//!    - For every signal whose version advanced, appends its subscriber
//!      list to the output.
//!    - Updates the observed versions to the current ones.
//!    - Deduplicates the output before returning it.
//! 3. Downstream subsystems consume the returned `TargetId`s and mark
//!    their corresponding entities as dirty.
//!
//! # Lifetime
//!
//! `EvaluatorTick<'a>` carries the same arena lifetime as the [`Signal`]s
//! it tracks. It cannot outlive its arena.
//!
//! # Duplicate registration
//!
//! A `debug_assert!` checks for duplicate registrations in debug builds.
//! In release, duplicates are allowed and the per-tick deduplication step
//! ensures correctness. This favours availability (no production panics
//! from a transpiler edge case) over strict validation.
//!
#![allow(unsafe_code)]

use std::marker::PhantomData;
use std::sync::atomic::Ordering;

use crate::frame::TargetId;

use super::signal::{Signal, SignalSlot};

/// Type-erased entry in the tick's source list.
///
/// Stores a pointer to the signal's slot, the last observed version, and
/// two monomorphised function pointers: one to read the current version,
/// and one to enumerate the slot's subscribers without allocating.
struct TickSource {
    slot_ptr: *const (),
    last_version: u64,
    read_version: unsafe fn(*const ()) -> u64,
    enumerate_subscribers: unsafe fn(*const (), &mut Vec<TargetId>),
}

/// Monomorphised glue: reads the version counter of a `SignalSlot<T>`.
///
/// # Safety
///
/// `slot` must point to a valid, live `SignalSlot<T>`.
unsafe fn read_version_glue<T>(slot: *const ()) -> u64 {
    // SAFETY: caller upholds the contract above.
    let slot = unsafe { &*slot.cast::<SignalSlot<T>>() };
    slot.dirty_version_ref().load(Ordering::Acquire)
}

/// Monomorphised glue: pushes a `SignalSlot<T>`'s subscribers into `out`.
///
/// # Safety
///
/// `slot` must point to a valid, live `SignalSlot<T>`. The slot must not
/// currently have an exclusive borrow active (this is upheld by the
/// single-threaded invariant: `collect_dirty` runs on the Logic thread,
/// no `write` can be in progress simultaneously).
unsafe fn enumerate_subscribers_glue<T>(slot: *const (), out: &mut Vec<TargetId>) {
    // SAFETY: caller upholds the contract above.
    let slot = unsafe { &*slot.cast::<SignalSlot<T>>() };
    // SAFETY: no exclusive borrow is active (see above), so this shared
    // borrow of the subscriber list is sound.
    let subscribers = unsafe { slot.subscribers_ref() };
    out.extend_from_slice(subscribers);
}

/// Collects dirty targets each tick by polling registered [`Signal`]s.
pub struct EvaluatorTick<'a> {
    sources: Vec<TickSource>,
    scratch: Vec<TargetId>,
    _arena: PhantomData<&'a ()>,
    _not_send: PhantomData<*mut ()>,
}

impl<'a> EvaluatorTick<'a> {
    /// Creates a new, empty tick.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            scratch: Vec::new(),
            _arena: PhantomData,
            _not_send: PhantomData,
        }
    }

    /// Registers `signal` for dirty-flag tracking.
    ///
    /// Subsequent calls to [`EvaluatorTick::collect_dirty`] will include
    /// the signal's subscribed [`TargetId`]s whenever its value is
    /// mutated.
    ///
    /// # Duplicate registration
    ///
    /// In debug builds, a `debug_assert!` fires if `signal` is registered
    /// twice. In release builds, duplicates are accepted silently —
    /// `collect_dirty` deduplicates the output, so correctness is
    /// preserved at the cost of one wasted slot per duplicate.
    pub fn register<T: 'static>(&mut self, signal: Signal<'a, T>) {
        let slot_ptr = signal.slot_ptr();

        debug_assert!(
            !self
                .sources
                .iter()
                .any(|s| std::ptr::addr_eq(s.slot_ptr, slot_ptr)),
            "Signal registered twice on the same EvaluatorTick — likely a \
             bylang transpiler bug. The engine will deduplicate dirty \
             targets at collect time, so this is not a correctness issue, \
             but each duplicate wastes a tracking slot.",
        );

        self.sources.push(TickSource {
            slot_ptr,
            last_version: 0,
            read_version: read_version_glue::<T>,
            enumerate_subscribers: enumerate_subscribers_glue::<T>,
        });
    }

    /// Returns the number of signals currently registered.
    #[must_use]
    pub fn registered(&self) -> usize {
        self.sources.len()
    }

    /// Collects the set of dirty [`TargetId`]s for this tick and resets
    /// observed versions.
    ///
    /// Each registered signal is polled exactly once. Signals whose
    /// version has not advanced contribute nothing. The returned `Vec` is
    /// sorted and deduplicated, so each `TargetId` appears at most once.
    ///
    /// The internal scratch buffer is reused across calls — subsequent
    /// ticks of similar size allocate nothing.
    pub fn collect_dirty(&mut self) -> Vec<TargetId> {
        self.scratch.clear();

        for source in &mut self.sources {
            // SAFETY: `slot_ptr` was produced by a valid `Signal<'a, T>`
            // whose lifetime is still alive (enforced by 'a). The function
            // pointer was monomorphised for the same T.
            let current = unsafe { (source.read_version)(source.slot_ptr) };

            if current != source.last_version {
                // SAFETY: same as above. The slot is alive and no exclusive
                // borrow can be active here because we are on the Logic
                // thread, which is the only thread that mutates signals.
                unsafe {
                    (source.enumerate_subscribers)(source.slot_ptr, &mut self.scratch);
                }
                source.last_version = current;
            }
        }

        // Sort + dedup. For the sizes expected (tens to low hundreds of
        // targets per tick), this is more cache-friendly than a HashSet.
        self.scratch.sort_unstable_by_key(|id| id.as_raw());
        self.scratch.dedup();

        // Return the scratch contents and leave an empty buffer for next
        // tick. `mem::take` reuses the allocated capacity in subsequent
        // calls via the empty Vec we leave behind.
        std::mem::take(&mut self.scratch)
    }
}

impl Default for EvaluatorTick<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::ViewArena;

    #[test]
    fn empty_tick_produces_no_dirty_targets() {
        let mut tick = EvaluatorTick::new();
        assert!(tick.collect_dirty().is_empty());
    }

    #[test]
    fn never_written_signal_produces_no_dirty_targets() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.subscribe(TargetId::new(1, 0, 0));

        let mut tick = EvaluatorTick::new();
        tick.register(signal);

        assert!(tick.collect_dirty().is_empty());
    }

    #[test]
    fn written_signal_produces_its_subscribers() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.subscribe(TargetId::new(1, 0, 0));
        signal.subscribe(TargetId::new(2, 0, 0));

        let mut tick = EvaluatorTick::new();
        tick.register(signal);

        signal.write(|v| *v = 1);
        let dirty = tick.collect_dirty();

        assert_eq!(dirty, vec![TargetId::new(1, 0, 0), TargetId::new(2, 0, 0)]);
    }

    #[test]
    fn multiple_writes_between_ticks_produce_each_target_once() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.subscribe(TargetId::new(42, 0, 0));

        let mut tick = EvaluatorTick::new();
        tick.register(signal);

        signal.write(|v| *v = 1);
        signal.write(|v| *v = 2);
        signal.write(|v| *v = 3);

        let dirty = tick.collect_dirty();
        assert_eq!(dirty, vec![TargetId::new(42, 0, 0)]);
    }

    #[test]
    fn second_tick_with_no_writes_is_empty() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.subscribe(TargetId::new(1, 0, 0));

        let mut tick = EvaluatorTick::new();
        tick.register(signal);

        signal.write(|v| *v = 1);
        let first = tick.collect_dirty();
        assert_eq!(first.len(), 1);

        let second = tick.collect_dirty();
        assert!(
            second.is_empty(),
            "no writes between ticks → no dirty targets"
        );
    }

    #[test]
    fn shared_target_between_signals_is_deduplicated() {
        let arena = ViewArena::new();
        let a = Signal::new_in(&arena, 0_u32);
        let b = Signal::new_in(&arena, 0_u32);

        let shared = TargetId::new(100, 0, 0);
        a.subscribe(shared);
        b.subscribe(shared);

        let mut tick = EvaluatorTick::new();
        tick.register(a);
        tick.register(b);

        a.write(|v| *v = 1);
        b.write(|v| *v = 2);

        let dirty = tick.collect_dirty();
        assert_eq!(dirty, vec![shared], "shared target appears exactly once");
    }

    #[test]
    fn heterogeneous_signal_types_share_a_tick() {
        let arena = ViewArena::new();
        let int_sig = Signal::new_in(&arena, 0_u32);
        let str_sig = Signal::new_in(&arena, String::new());

        int_sig.subscribe(TargetId::new(1, 0, 0));
        str_sig.subscribe(TargetId::new(2, 0, 0));

        let mut tick = EvaluatorTick::new();
        tick.register(int_sig);
        tick.register(str_sig);

        int_sig.write(|v| *v = 99);
        str_sig.write(|s| s.push_str("hi"));

        let dirty = tick.collect_dirty();
        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&TargetId::new(1, 0, 0)));
        assert!(dirty.contains(&TargetId::new(2, 0, 0)));
    }

    #[test]
    fn registered_returns_correct_count() {
        let arena = ViewArena::new();
        let a = Signal::new_in(&arena, 0_u32);
        let b = Signal::new_in(&arena, 0_u32);

        let mut tick = EvaluatorTick::new();
        assert_eq!(tick.registered(), 0);
        tick.register(a);
        assert_eq!(tick.registered(), 1);
        tick.register(b);
        assert_eq!(tick.registered(), 2);
    }

    #[test]
    #[should_panic(expected = "Signal registered twice")]
    #[cfg(debug_assertions)]
    fn duplicate_registration_panics_in_debug() {
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);

        let mut tick = EvaluatorTick::new();
        tick.register(signal);
        tick.register(signal);
    }
}
