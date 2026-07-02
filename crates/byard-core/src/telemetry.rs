//! Zero-allocation telemetry & profiling (RFC-0013).
//!
//! Each engine thread owns a fixed-capacity, thread-local ring of
//! [`Sample`]s. [`profile_scope!`] wraps a block in an RAII [`Guard`] that
//! writes one sample on drop; the ring never grows and never allocates on
//! the hot path (P1). At end-of-tick, [`crate::relay::Relay::publish`] calls
//! [`crate::frame::RenderFrame::drain_telemetry`], which pulls the calling
//! thread's ring into the frame's own [`SampleBlock`], piggybacking on the
//! existing atomic frame swap instead of opening a new channel.
//!
//! With the `telemetry` Cargo feature off, [`profile_scope!`] expands to a
//! no-op statement — zero cost in a build that disables it (e.g. release).

use std::cell::RefCell;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Fixed capacity of each thread's sample ring (RFC-0013 **P1**).
pub const RING_CAPACITY: usize = 4096;

/// A compile-time-interned scope identifier.
///
/// Looked up once per call site (cached in a call-site-local `OnceLock` by
/// [`profile_scope!`]), never per sample.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScopeId(pub u16);

/// Looks up (or registers, on first touch) the [`ScopeId`] for a scope name.
///
/// Backed by a small `Mutex<Vec<&'static str>>` registry — touched once per
/// unique call site, never on the per-sample hot path.
///
/// # Panics
///
/// Panics if more than `u16::MAX` distinct scope names are ever registered
/// (a build-time authoring error, not something user input can trigger) —
/// silently wrapping the index would alias two unrelated scopes under the
/// same `ScopeId` and corrupt profiling data.
pub fn scope_id(name: &'static str) -> ScopeId {
    static REGISTRY: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let registry = REGISTRY.get_or_init(|| Mutex::new(Vec::new()));
    let mut names = registry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(pos) = names.iter().position(|&n| n == name) {
        return ScopeId(index_to_u16(pos));
    }
    names.push(name);
    ScopeId(index_to_u16(names.len() - 1))
}

fn index_to_u16(index: usize) -> u16 {
    u16::try_from(index).expect("telemetry scope registry exceeded u16::MAX distinct scope names")
}

/// Returns the engine's telemetry epoch, established on first use.
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Nanoseconds elapsed since the engine's telemetry epoch.
///
/// Backed by `std::time::Instant` (portable, correct everywhere) rather than
/// an always-on `rdtsc` fast path — see RFC-0013 "Rationale and alternatives".
#[must_use]
#[allow(clippy::cast_possible_truncation)] // u64 ns covers ~584 years; never truncates in practice
pub fn now_ns() -> u64 {
    epoch().elapsed().as_nanos() as u64
}

/// One CPU scope timing: a scope identifier and its start/end timestamps in
/// nanoseconds since the telemetry [`epoch`].
///
/// `#[repr(C)]` with explicit padding fields (no implicit tail/interior
/// padding) so the type is a clean `bytemuck::Pod` — required to pack a flat
/// byte block that can cross the frame swap as `Send` data (RFC-0013
/// "Hand-off").
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sample {
    /// Which scope this sample belongs to.
    pub scope: ScopeId,
    /// Explicit padding to keep the layout free of compiler-inserted gaps.
    _reserved_a: u16,
    /// Explicit padding to align `start`/`end` on an 8-byte boundary.
    _reserved_b: u32,
    /// Scope entry time, nanoseconds since the telemetry epoch.
    pub start: u64,
    /// Scope exit time, nanoseconds since the telemetry epoch.
    pub end: u64,
}

/// A flat, `Send` snapshot of one thread's ring for a single tick.
///
/// Attached to the [`crate::frame::RenderFrame`] on the existing atomic
/// frame swap (RFC-0013 "Hand-off") — no new channel, no new lock. `samples`
/// holds only [`Sample`], a `Pod` type, so the block is plain data end to
/// end even though the `Vec` wrapper itself isn't `Pod`.
#[derive(Debug, Clone, Default)]
pub struct SampleBlock {
    /// The samples captured this tick, in push order.
    pub samples: Vec<Sample>,
    /// How many samples were dropped this tick because the ring was full.
    pub dropped: u64,
}

/// A fixed-capacity, non-circular sample buffer: once full, new samples are
/// dropped (not the oldest) so an in-flight capture is never overwritten
/// mid-frame (RFC-0013 **P1**).
struct Ring {
    buf: Box<[Sample]>,
    len: usize,
    dropped: u64,
}

impl Ring {
    fn new() -> Self {
        Self {
            // Built directly on the heap (never as a stack array first) —
            // `RING_CAPACITY * size_of::<Sample>()` is too large to build on
            // the stack before boxing.
            buf: vec![Sample::default(); RING_CAPACITY].into_boxed_slice(),
            len: 0,
            dropped: 0,
        }
    }

    /// Pushes a sample. Never allocates: writes into the preallocated
    /// buffer, or increments the dropped counter once full.
    fn push(&mut self, sample: Sample) {
        if self.len < RING_CAPACITY {
            self.buf[self.len] = sample;
            self.len += 1;
        } else {
            self.dropped += 1;
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Copies out the current contents into `out` and resets the ring for
    /// the next tick.
    ///
    /// Reuses `out.samples`' existing heap allocation (`Vec::clear` +
    /// `extend_from_slice`) instead of allocating a fresh `Vec` every tick,
    /// so a caller that keeps its `SampleBlock` around (e.g. a recycled
    /// [`crate::frame::RenderFrame`]) drains at steady-state zero
    /// allocation once its capacity has grown to fit a typical tick.
    fn drain_into(&mut self, out: &mut SampleBlock) {
        out.samples.clear();
        out.samples.extend_from_slice(&self.buf[..self.len]);
        out.dropped = self.dropped;
        self.len = 0;
        self.dropped = 0;
    }
}

thread_local! {
    static RING: RefCell<Ring> = RefCell::new(Ring::new());
}

/// Writes one [`Sample`] into the calling thread's ring. Never allocates.
pub fn push_sample(sample: Sample) {
    RING.with(|r| r.borrow_mut().push(sample));
}

/// Returns the number of samples currently held in the calling thread's ring.
#[must_use]
pub fn ring_len() -> usize {
    RING.with(|r| r.borrow().len())
}

/// Returns the number of samples dropped so far this tick on the calling
/// thread because its ring was full.
#[must_use]
pub fn ring_dropped() -> u64 {
    RING.with(|r| r.borrow().dropped())
}

/// Drains the calling thread's ring into `out`, resetting the ring for the
/// next tick and reusing `out`'s existing `Vec` allocation.
///
/// This is the hot path — [`crate::frame::RenderFrame::drain_telemetry`]
/// calls this on the logic thread right before [`crate::relay::Relay::publish`]
/// swaps the frame in, so a recycled frame's `SampleBlock` never reallocates
/// once it has grown to fit a typical tick.
pub fn drain_samples_into(out: &mut SampleBlock) {
    RING.with(|r| r.borrow_mut().drain_into(out));
}

/// Drains the calling thread's ring into a freshly allocated [`SampleBlock`].
///
/// Convenience for tests and one-off call sites; steady-state hot paths
/// should prefer [`drain_samples_into`] with a reused buffer.
#[must_use]
pub fn drain_samples() -> SampleBlock {
    let mut block = SampleBlock::default();
    drain_samples_into(&mut block);
    block
}

/// RAII guard produced by [`profile_scope!`]; writes one [`Sample`] to the
/// calling thread's ring when dropped.
pub struct Guard {
    scope: ScopeId,
    start: u64,
}

impl Guard {
    /// Starts timing `scope` now.
    #[must_use]
    pub fn new(scope: ScopeId) -> Self {
        Self {
            scope,
            start: now_ns(),
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        push_sample(Sample {
            scope: self.scope,
            _reserved_a: 0,
            _reserved_b: 0,
            start: self.start,
            end: now_ns(),
        });
    }
}

/// Times the rest of the enclosing block as a named scope.
///
/// ```
/// # use byard_core::profile_scope;
/// fn build_frame() {
///     profile_scope!("frame.total");
///     // ... work ...
/// }
/// ```
///
/// Expands to a no-op when the `telemetry` feature is off — zero cost in a
/// build that disables it.
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        #[cfg(feature = "telemetry")]
        let _guard = {
            static SCOPE: ::std::sync::OnceLock<$crate::telemetry::ScopeId> =
                ::std::sync::OnceLock::new();
            let id = *SCOPE.get_or_init(|| $crate::telemetry::scope_id($name));
            $crate::telemetry::Guard::new(id)
        };
        #[cfg(not(feature = "telemetry"))]
        let _ = ();
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_writes_a_sample_and_increments_len() {
        // Isolate from other tests sharing the same thread-local ring by
        // draining first.
        let _ = drain_samples();
        push_sample(Sample {
            scope: ScopeId(1),
            _reserved_a: 0,
            _reserved_b: 0,
            start: 1,
            end: 2,
        });
        assert_eq!(ring_len(), 1);
        let block = drain_samples();
        assert_eq!(block.samples.len(), 1);
        assert_eq!(block.samples[0].scope, ScopeId(1));
    }

    #[test]
    fn full_ring_drops_newest_and_honors_capacity() {
        let _ = drain_samples();
        for i in 0..RING_CAPACITY {
            #[allow(clippy::cast_possible_truncation)]
            push_sample(Sample {
                scope: ScopeId(0),
                _reserved_a: 0,
                _reserved_b: 0,
                start: i as u64,
                end: i as u64,
            });
        }
        assert_eq!(ring_len(), RING_CAPACITY);
        assert_eq!(ring_dropped(), 0);

        // One more push once full: dropped, not overwriting slot 0.
        push_sample(Sample {
            scope: ScopeId(99),
            _reserved_a: 0,
            _reserved_b: 0,
            start: 999,
            end: 999,
        });
        assert_eq!(
            ring_len(),
            RING_CAPACITY,
            "capacity is honored, not exceeded"
        );
        assert_eq!(ring_dropped(), 1, "the overflowing sample is dropped");

        let block = drain_samples();
        assert_eq!(block.samples.len(), RING_CAPACITY);
        assert_eq!(block.samples[0].start, 0, "slot 0 was never overwritten");
        assert_eq!(block.dropped, 1);

        // Draining resets the ring for the next tick.
        assert_eq!(ring_len(), 0);
        assert_eq!(ring_dropped(), 0);
    }

    #[test]
    fn drain_samples_into_reuses_the_callers_vec_allocation() {
        let _ = drain_samples();
        let mut block = SampleBlock::default();
        block.samples.reserve(RING_CAPACITY);
        let reused_capacity = block.samples.capacity();
        assert!(reused_capacity >= RING_CAPACITY);

        for _ in 0..RING_CAPACITY {
            push_sample(Sample::default());
        }
        drain_samples_into(&mut block);
        assert_eq!(block.samples.len(), RING_CAPACITY);
        assert_eq!(
            block.samples.capacity(),
            reused_capacity,
            "draining into an already-sized buffer must not reallocate"
        );

        // A second tick with fewer samples reuses the same allocation again.
        push_sample(Sample::default());
        drain_samples_into(&mut block);
        assert_eq!(block.samples.len(), 1);
        assert_eq!(block.samples.capacity(), reused_capacity);
    }

    #[test]
    fn scope_id_is_stable_and_interned_per_name() {
        let a1 = scope_id("telemetry.test.scope_a");
        let a2 = scope_id("telemetry.test.scope_a");
        let b = scope_id("telemetry.test.scope_b");
        assert_eq!(a1, a2, "the same name always resolves to the same id");
        assert_ne!(a1, b, "distinct names get distinct ids");
    }

    #[test]
    fn profile_scope_writes_one_sample_via_guard() {
        let _ = drain_samples();
        {
            crate::profile_scope!("telemetry.test.profile_scope_writes_one_sample_via_guard");
        }
        #[cfg(feature = "telemetry")]
        assert_eq!(ring_len(), 1, "the guard's Drop wrote exactly one sample");
        #[cfg(not(feature = "telemetry"))]
        assert_eq!(ring_len(), 0, "with telemetry off, the macro is a no-op");
    }

    #[test]
    #[cfg(not(feature = "telemetry"))]
    fn profile_scope_is_noop_without_telemetry_feature() {
        // Exercised via `cargo test -p byard-core --no-default-features`.
        let _ = drain_samples();
        crate::profile_scope!("telemetry.test.noop");
        assert_eq!(ring_len(), 0, "no guard is constructed without the feature");
    }

    #[test]
    fn sample_block_is_send_pod_data() {
        const fn assert_send<T: Send>() {}
        assert_send::<SampleBlock>();
        assert_send::<Sample>();

        let sample = Sample {
            scope: ScopeId(3),
            _reserved_a: 0,
            _reserved_b: 0,
            start: 10,
            end: 20,
        };
        // Round-trips as raw bytes, as it will across the frame swap.
        let bytes: &[u8] = bytemuck::bytes_of(&sample);
        let back: Sample = bytemuck::pod_read_unaligned(bytes);
        assert_eq!(back, sample);
        assert_eq!(std::mem::size_of::<Sample>(), 24, "no implicit padding");
    }

    // ── Allocation-free push (RFC-0013 P1: "no allocation in push") ────────

    #[allow(unsafe_code)] // SAFETY: thin passthrough wrapper around `System`, test-only.
    mod counting_alloc {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        // Thread-local, not a shared atomic: `cargo test` runs many unrelated
        // tests concurrently on other threads, all sharing one global
        // allocator, so a process-wide counter would be polluted by them.
        // Isolating the count per-thread lets this test see only the
        // allocations its own calling thread performed.
        thread_local! {
            static COUNT: Cell<usize> = const { Cell::new(0) };
        }

        /// Number of allocations observed on the calling thread since
        /// process start. Test-only: this crate ships no global allocator
        /// outside `cfg(test)`.
        pub fn count() -> usize {
            COUNT.with(Cell::get)
        }

        pub struct CountingAllocator;

        // SAFETY: forwards every call unchanged to `System`, which is
        // itself a valid `GlobalAlloc`; the only addition is a thread-local
        // counter increment with no effect on the allocation contract.
        unsafe impl GlobalAlloc for CountingAllocator {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                COUNT.with(|c| c.set(c.get() + 1));
                // SAFETY: `layout` is passed through unchanged from the caller.
                unsafe { System.alloc(layout) }
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                // SAFETY: `ptr`/`layout` are passed through unchanged from the caller.
                unsafe { System.dealloc(ptr, layout) }
            }
        }
    }

    #[global_allocator]
    static GLOBAL: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

    #[test]
    fn push_does_not_allocate() {
        let _ = drain_samples();
        // Warm the ring so any one-time thread-local init has already run.
        push_sample(Sample::default());
        let _ = drain_samples();

        let before = counting_alloc::count();
        for _ in 0..64 {
            push_sample(Sample::default());
        }
        let after = counting_alloc::count();
        assert_eq!(after, before, "push must not allocate");

        let _ = drain_samples();
    }
}
