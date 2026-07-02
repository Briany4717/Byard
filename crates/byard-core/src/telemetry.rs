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

/// Which cost bucket a scope belongs to (RFC-0013 §"The interpreter tax
/// segmentation"): `Interpreter` scopes evaporate in an AOT release build,
/// `Native` scopes don't, and `Gpu` scopes are async pass timings rather than
/// CPU wall-clock at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScopeKind {
    /// Tree-walking eval, dynamic dispatch, env lookups — the cost an AOT
    /// (transpiled) build does not pay.
    Interpreter,
    /// Ordinary CPU work that costs the same in dev and in an AOT release.
    #[default]
    Native,
    /// A `wgpu` render-pass timing, resolved asynchronously (RFC-0013
    /// "GPU timing").
    Gpu,
}

struct ScopeEntry {
    name: &'static str,
    kind: ScopeKind,
}

fn registry() -> &'static Mutex<Vec<ScopeEntry>> {
    static REGISTRY: OnceLock<Mutex<Vec<ScopeEntry>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Looks up (or registers, on first touch) the [`ScopeId`] for a scope name,
/// tagged `Native` (see [`scope_id_tagged`] for `Interpreter`/`Gpu` scopes).
///
/// Backed by a small `Mutex<Vec<_>>` registry — touched once per unique call
/// site, never on the per-sample hot path.
#[must_use]
pub fn scope_id(name: &'static str) -> ScopeId {
    scope_id_tagged(name, ScopeKind::Native)
}

/// Looks up (or registers, on first touch) the [`ScopeId`] for a scope name
/// tagged with `kind`. Re-registering an existing name with a different
/// `kind` is a programming error and panics — a scope's cost bucket is
/// determined once, at its call site, and must not drift.
///
/// # Panics
///
/// Panics if more than `u16::MAX` distinct scope names are ever registered
/// (a build-time authoring error, not something user input can trigger) —
/// silently wrapping the index would alias two unrelated scopes under the
/// same `ScopeId` and corrupt profiling data. Also panics if `name` was
/// already registered under a different `kind`.
pub fn scope_id_tagged(name: &'static str, kind: ScopeKind) -> ScopeId {
    let mut names = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(pos) = names.iter().position(|e| e.name == name) {
        assert!(
            names[pos].kind == kind,
            "telemetry scope {name:?} re-registered with a different ScopeKind"
        );
        return ScopeId(index_to_u16(pos));
    }
    names.push(ScopeEntry { name, kind });
    ScopeId(index_to_u16(names.len() - 1))
}

/// Returns the [`ScopeKind`] a [`ScopeId`] was registered with.
#[must_use]
pub fn scope_kind(id: ScopeId) -> Option<ScopeKind> {
    let names = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    names.get(usize::from(id.0)).map(|e| e.kind)
}

/// Returns the name a [`ScopeId`] was registered with — the overlay/CLI's
/// only way to turn a `Sample` back into a human-readable scope label.
#[must_use]
pub fn scope_name(id: ScopeId) -> Option<&'static str> {
    let names = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    names.get(usize::from(id.0)).map(|e| e.name)
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

impl Sample {
    /// Builds a `Gpu`-tagged sample from an already-resolved pass duration
    /// (RFC-0013 "GPU timing") rather than two wall-clock timestamps: GPU
    /// passes are timed by the device's own timestamp queries, resolved
    /// asynchronously, so there is no meaningful CPU-epoch `start` for them.
    /// By convention `start` is `0` and `end` is the duration itself, so
    /// `end - start` (the quantity every consumer actually wants) is still
    /// the pass duration in nanoseconds.
    #[must_use]
    pub fn gpu_duration(scope: ScopeId, duration_ns: u64) -> Self {
        Self {
            scope,
            _reserved_a: 0,
            _reserved_b: 0,
            start: 0,
            end: duration_ns,
        }
    }

    /// This sample's duration (`end - start`) in nanoseconds.
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
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

impl SampleBlock {
    /// Sums the duration of every sample whose scope is tagged `kind`
    /// (RFC-0013 "the interpreter tax segmentation").
    #[must_use]
    pub fn sum_by_kind(&self, kind: ScopeKind) -> u64 {
        self.samples
            .iter()
            .filter(|s| scope_kind(s.scope) == Some(kind))
            .map(Sample::duration_ns)
            .sum()
    }

    /// The total `Interpreter`-tagged time this tick — the tax an AOT release
    /// build does not pay. Overlay/CLI consumers sum this bucket separately
    /// from the rest of `frame.total` (RFC-0013 "the honest number").
    #[must_use]
    pub fn interpreter_tax_ns(&self) -> u64 {
        self.sum_by_kind(ScopeKind::Interpreter)
    }
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

/// A calibrated interpreter-vs-native cost ratio (RFC-0013 **P4**): a fixed
/// set of microbenchmarks (e.g. `byard-core/benches/telemetry_calibration.rs`,
/// signal read / element construct / memo eval), refreshed per release —
/// never measured live, which would re-add observer overhead.
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    /// Where this ratio came from — always shown alongside a projection so
    /// the number is legible as an estimate, never a hard promise (P3).
    pub basis: &'static str,
    /// `native_ns ≈ interpreter_ns * ratio` for representative interpreter
    /// operations, as measured by the calibration benchmarks.
    pub interpreter_to_native_ratio: f64,
}

/// A projected "what would this cost in an AOT release build" estimate
/// (RFC-0013 **P3**): opt-in, and always carries its [`Calibration::basis`]
/// so the overlay/CLI can show the number is an estimate, not a measurement.
#[derive(Debug, Clone, Copy)]
pub struct Projection {
    /// The projected total frame cost in nanoseconds.
    pub projected_ns: u64,
    /// The calibration basis this projection was computed from.
    pub basis: &'static str,
}

/// Projects an AOT estimate from a tick's measured total and its
/// [`SampleBlock::interpreter_tax_ns`] (RFC-0013 "The interpreter tax
/// segmentation"): `native ≈ total − interp_measured + interp_native_equiv`,
/// where `interp_native_equiv` comes from `calibration`.
///
/// Never called implicitly — a caller opts in by calling this and choosing to
/// display the result (P3: "opt-in, always shown with its basis").
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)] // frame times never approach f64's precision or u64's range
pub fn project_aot(total_ns: u64, interpreter_ns: u64, calibration: &Calibration) -> Projection {
    let interp_native_equiv =
        (interpreter_ns as f64 * calibration.interpreter_to_native_ratio) as u64;
    Projection {
        projected_ns: total_ns.saturating_sub(interpreter_ns) + interp_native_equiv,
        basis: calibration.basis,
    }
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
        $crate::profile_scope!($name, $crate::telemetry::ScopeKind::Native);
    };
    ($name:expr, $kind:expr) => {
        #[cfg(feature = "telemetry")]
        let _guard = {
            static SCOPE: ::std::sync::OnceLock<$crate::telemetry::ScopeId> =
                ::std::sync::OnceLock::new();
            let id = *SCOPE.get_or_init(|| $crate::telemetry::scope_id_tagged($name, $kind));
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
    fn scope_id_tagged_records_its_kind() {
        let interp = scope_id_tagged("telemetry.test.kind.interp", ScopeKind::Interpreter);
        let native = scope_id_tagged("telemetry.test.kind.native", ScopeKind::Native);
        let gpu = scope_id_tagged("telemetry.test.kind.gpu", ScopeKind::Gpu);
        assert_eq!(scope_kind(interp), Some(ScopeKind::Interpreter));
        assert_eq!(scope_kind(native), Some(ScopeKind::Native));
        assert_eq!(scope_kind(gpu), Some(ScopeKind::Gpu));
        assert_eq!(
            scope_kind(scope_id("telemetry.test.kind.default_native")),
            Some(ScopeKind::Native),
            "plain scope_id defaults to Native"
        );
    }

    #[test]
    fn scope_name_round_trips_through_scope_id() {
        let id = scope_id("telemetry.test.name.round_trip");
        assert_eq!(scope_name(id), Some("telemetry.test.name.round_trip"));
    }

    #[test]
    #[should_panic(expected = "re-registered with a different ScopeKind")]
    fn scope_id_tagged_rejects_a_kind_change_for_an_existing_name() {
        let _ = scope_id_tagged("telemetry.test.kind.stable", ScopeKind::Native);
        let _ = scope_id_tagged("telemetry.test.kind.stable", ScopeKind::Interpreter);
    }

    #[test]
    fn interpreter_tax_sums_only_interpreter_tagged_samples() {
        let interp = scope_id_tagged("telemetry.test.tax.interp", ScopeKind::Interpreter);
        let native = scope_id_tagged("telemetry.test.tax.native", ScopeKind::Native);
        let gpu = scope_id_tagged("telemetry.test.tax.gpu", ScopeKind::Gpu);
        let block = SampleBlock {
            samples: vec![
                Sample {
                    scope: interp,
                    _reserved_a: 0,
                    _reserved_b: 0,
                    start: 0,
                    end: 100,
                },
                Sample {
                    scope: native,
                    _reserved_a: 0,
                    _reserved_b: 0,
                    start: 0,
                    end: 50,
                },
                Sample::gpu_duration(gpu, 30),
                Sample {
                    scope: interp,
                    _reserved_a: 0,
                    _reserved_b: 0,
                    start: 100,
                    end: 175,
                },
            ],
            dropped: 0,
        };
        assert_eq!(block.interpreter_tax_ns(), 100 + 75);
        assert_eq!(block.sum_by_kind(ScopeKind::Native), 50);
        assert_eq!(block.sum_by_kind(ScopeKind::Gpu), 30);
    }

    #[test]
    fn project_aot_replaces_measured_interpreter_cost_with_its_native_equivalent() {
        let calibration = Calibration {
            basis: "test calibration",
            interpreter_to_native_ratio: 0.5,
        };
        // total = 10ms, of which 6ms was interpreter; native equivalent is
        // half that (3ms), so the projection is 10 - 6 + 3 = 7ms.
        let projection = project_aot(10_000_000, 6_000_000, &calibration);
        assert_eq!(projection.projected_ns, 7_000_000);
        assert_eq!(projection.basis, "test calibration");
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
