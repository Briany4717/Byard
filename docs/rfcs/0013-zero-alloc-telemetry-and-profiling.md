# RFC-0013: Zero-Allocation Telemetry & Profiling

- **Status:** Draft — design proposal
- **Author(s):** Brian (byard_v2)
- **Created:** 2026-07-01
- **Last updated:** 2026-07-01
- **Depends on:** RFC-0001 (§5 concurrency, atomic frame hand-off, `frame.rs` boundary), RFC-0004 (tick), RFC-0006 (dev runner & overlay).
- **Recommended sequencing:** land **first** — it is the measurement tool that justifies (or rejects) RFC-0014 (JIT) and the layout-tween future of RFC-0010.

---

## Summary

A built-in profiler that measures Byard's own frame cost **without perturbing what
it measures**: strictly zero heap allocation on the hot path, thread-local
lock-free capture, RAII scope timers, GPU timing via `wgpu` timestamp queries
resolved asynchronously two frames later, and a transparent segmentation of the
**"interpreter tax"** so a dev in `byard dev` can read off what the AOT-compiled
build will actually cost. Telemetry piggybacks on the existing atomic frame
channel — it introduces no new locks and no `Mutex` contention.

## Motivation

Byard's entire pitch is performance-as-a-floor. That claim is only credible if it
is **measured**, continuously, by a profiler that doesn't itself distort the
numbers (the observer effect). Three specific needs:

1. **Honest dev numbers.** `byard dev` runs a tree-walking AST interpreter
   (RFC-0004/0006). Its frame time is *not* the shipped app's frame time. A dev
   staring at "12ms/frame" in dev needs to know how much of that is the
   interpreter tax that evaporates in AOT release.
2. **No perturbation.** A profiler that allocates, locks, or syscalls per sample
   changes the very timings it reports. It must be effectively free.
3. **CPU *and* GPU.** Frame cost is split across both; measuring only CPU hides
   shader/overdraw regressions. GPU timing must never make the CPU block on the
   GPU.

Without this, every performance claim in the project is an assertion, not a
measurement — and RFC-0014's JIT would be a solution to an *unquantified* problem.

## Guide-level explanation

Instrument a scope with an RAII macro; read results in the CLI or the dev overlay:

```rust
fn build_frame(&mut self) {
    profile_scope!("frame.total");
    { profile_scope!("interp.tick");   self.tick(); }
    { profile_scope!("layout.taffy");  self.layout(); }
    { profile_scope!("encode.frame");  self.encode(); }
}
```

The overlay shows a flat, per-scope breakdown with the interpreter tax called out
separately:

```
frame.total     8.9ms   (interp tax 5.1ms → AOT proj… ~3.8ms)
  interp.tick   5.1ms   [INTERPRETER — 0 in release]
  layout.taffy  1.2ms
  encode.frame  0.9ms
  gpu.solidbox  1.4ms   (async, −2 frames)
  gpu.msdf      0.3ms
```

Turn it off entirely with a feature flag; when compiled out, `profile_scope!`
expands to nothing — zero cost in release.

## Reference-level explanation

### CPU capture — thread-local ring, zero alloc

Each engine thread owns a **fixed-capacity, thread-local ring buffer** of samples
(`thread_local!`, no shared state, no locks):

```rust
#[repr(C)]
#[derive(Clone, Copy)]
struct Sample { scope: ScopeId, start: u64, end: u64 }   // u64 = ns since engine epoch

thread_local! {
    static RING: RefCell<Ring<Sample, 4096>> = /* preallocated, fixed */;
}
```

`ScopeId` is a compile-time-interned `&'static str` → `u16` (a static registry
built at first touch; no per-sample string work). `profile_scope!(name)` expands
to a guard whose `Drop` writes one `Sample`:

```rust
struct Guard { id: ScopeId, start: u64 }
impl Drop for Guard { fn drop(&mut self) { RING.with(|r| r.borrow_mut().push(
    Sample { scope: self.id, start: self.start, end: now_ns() })); } }
```

`now_ns()` uses `std::time::Instant` by default (portable, ~tens of ns). An
optional `rdtsc`+calibration fast path is available behind a feature for
sub-scopes where `Instant` overhead is itself significant — but `Instant` is the
default because correctness/portability beat a few ns (Byard principle).

**Zero allocation:** the ring is preallocated and fixed; `push` overwrites the
oldest slot when full (bounded memory, never grows). No `Vec` growth, no boxing,
no formatting on the hot path.

### Hand-off — piggyback on the atomic frame channel

At end-of-tick, the logic thread packs the tick's samples into a flat POD block
and ships it on **the same atomic frame swap** already used to hand `RenderFrame`
to the renderer (RFC-0001 §5.1). No new channel, no `Mutex`, no contention — the
telemetry rides the frame it describes. Only `Send` PODs cross the boundary
(INV-2). The renderer/overlay consumer reads the block after presenting.

### GPU timing — async timestamp queries, never blocking

Per RFC-0001 §3.1 pipelines, the `Encoder` writes timestamps into a
`wgpu::QuerySet` before/after each render pass:

1. Allocate a `QuerySet` (timestamp) and a resolve buffer once (reused).
2. `encoder.write_timestamp(set, i)` around each pass (`SolidBox`,
   `VectorMSDF`, `DecoratedBox`, `TextureSampler`).
3. `resolve_query_set` into a buffer; `map_async` it.
4. **Read it two frames later** when the map has completed — the CPU never waits
   synchronously on the GPU. Results are matched to their frame by index.

GPU samples land in the same overlay stream, tagged `(async, −2 frames)` so the
dev knows they lag by design.

### The interpreter tax segmentation (the honest number)

Scopes are tagged `Interpreter | Native | Gpu`. `interp.*` scopes (tree-walking
eval, dynamic dispatch, env lookups) are `Interpreter`. The overlay/CLI sums the
`Interpreter` bucket separately and **projects** an AOT estimate:

- The projection is a calibrated ratio, not a guess: a set of micro-benchmarks
  (already the shape of `byard-core/benches/*`) measures the interpreter-vs-native
  cost of representative ops (signal read, element construct, memo eval). The tax
  is projected as `native ≈ total − interp_measured + interp_native_equiv`, where
  `interp_native_equiv` comes from the calibration table.
- The number is presented as an **estimate with its basis**, never as a hard
  promise — consistent with "measure, don't assert."

This directly answers RFC-0014's gating question: *is the tree-walker actually the
bottleneck worth a JIT?* You cannot answer that honestly without this bucket.

### Threading & invariants

Ring buffers are `!Send` and thread-local; only packed POD sample blocks cross
threads, on the existing frame channel. No back-dependency (INV-1), no shared
mutable state, no lock on the hot path. Compiled out in release → the profiler
cannot affect shipped performance.

## Drawbacks

- Fixed-capacity rings can drop samples under pathological over-instrumentation
  (bounded-memory trade-off; surfaced as a "dropped N samples" counter).
- GPU timestamp support/precision varies by backend; some targets report coarse
  or no timestamps (degrade gracefully to CPU-only).
- The AOT projection is an estimate; over-trusting it is a risk (mitigated by
  always showing its basis).

## Rationale and alternatives

- **Thread-local ring vs a shared queue.** Shared queues need synchronization =
  contention = perturbation. Thread-local is contention-free by construction.
- **Piggyback vs dedicated telemetry channel.** Reusing the atomic frame swap adds
  zero new synchronization and auto-correlates telemetry with its frame.
- **Async GPU timestamps vs `device.poll(Wait)`.** Blocking the CPU on the GPU to
  read timings would destroy the very frame pacing being measured.
- **`Instant` default vs always-`rdtsc`.** `rdtsc` needs calibration and has
  cross-core caveats; `Instant` is correct everywhere, fast enough, and honest.
- **Not doing it:** performance claims stay unverifiable and RFC-0014 is
  unjustifiable.

## Prior art

Tracy profiler (scoped zone capture, GPU zones), Optick, Chrome trace event
format, `puffin` (Rust, egui overlays), `wgpu` timestamp-query examples, and the
Rust `tracing` span model (but allocation-free and fixed-capacity here).

## Resolved decisions (2026-07-01)

- **P1 — ring capacity / overflow:** **4096 samples/thread, drop-newest + visible
  "N dropped" counter** (overwriting the oldest would corrupt an in-flight frame's
  capture).
- **P2 — overlay format:** **flat list by default + toggleable flamegraph** (flat is
  the at-a-glance read; flamegraph for deep nesting).
- **P3 — AOT projection:** **opt-in, always shown with its basis** (it's a calibrated
  estimate; default-prominent invites over-trust — "measure, don't assert").
- **P4 — calibration:** **fixed microbenchmarks in `benches/`**, refreshed per release
  (reproducible; live per-op measurement would re-add observer overhead).
- **P5 — backends without timestamp queries:** **degrade to CPU-only with a clear
  overlay notice** (honesty over invented numbers).

## Unresolved questions (deferred to implementation)

- [ ] Whether the flamegraph view ships in the first cut or a follow-up.
- [ ] Calibration refresh automation (CI job vs manual per release).

## Future possibilities

- Export to Chrome/Tracy trace formats for external tooling.
- Per-`View` and per-signal attribution ("which view costs the most").
- Regression gates in CI: fail a PR if `frame.total` p99 regresses on a fixture.
- Feed the numbers back into RFC-0010's layout-tween go/no-go and RFC-0014's JIT
  decision.
