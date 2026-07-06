# RFC-0015: Polyglot Controller Bridge (isolated guest runtimes)

- **Status:** Draft — design proposal
- **Author(s):** Brian (byard_v2)
- **Created:** 2026-07-01
- **Last updated:** 2026-07-01
- **Depends on:** RFC-0001 (§5 concurrency, `!Send`/`!Sync`, `frame.rs` boundary, controller model), RFC-0003 (§6 callback props), RFC-0002 (`inject` / ambient values), RFC-0006/M23 (Rust controller boundary — this generalizes it).
- **Positioning:** an *optional, feature-gated* extension of the Rust controller boundary — never a replacement, never on the hot path.

---

## Summary

Let controllers — the `.rs`-side logic layer — be written in **guest interpreted
languages (JavaScript via QuickJS/`rquickjs`, Python via `pyo3`)** for adoption,
**without ever letting a GC pause, GIL, or interpreter stall touch a frame.**
Guests run behind the same compile-time boundary as Rust controllers: they are
**thread-isolated**, exchange only `Send` POD messages and **validated**
shared-memory ring buffers with the logic thread, and can never reach the arena or
the `RenderFrame`. The renderer keeps drawing the last atomic frame at full rate
while a guest blocks. Zero-copy is offered, but through a *validated, versioned*
binary layout — not blind `&[u8]` reinterpretation, which would be a soundness
hole this project refuses to ship.

## Motivation

Byard's two-file model (`.byd` UI + `.rs` controllers) is principled but narrows
the audience to Rust developers. A huge amount of business logic, ML glue, and
prototyping lives in Python and JS. Letting those languages drive *variables*
(never UI, never memory) would massively widen adoption — *if* it can be done
without betraying the performance floor.

The danger is obvious: CPython's GIL and GC, or a long JS task, can freeze a
naive integration mid-frame. Byard's answer must guarantee the **UI thread never
waits on a guest**. The architecture below makes that a structural property, not a
best-effort hope.

## Guide-level explanation

A controller can be declared in a guest language and `inject`ed exactly like a
Rust controller; the `byld` side is unchanged and unaware of the language:

```python
# weather.py — a guest controller
@byard_controller
class Weather:
    async def fetch(self, city: str) -> float:
        r = await http.get(f"/temp/{city}")   # a 2s network stall here…
        return r.json()["celsius"]
```

```byld
inject Weather as weather

Text("{weather.temp}°")                 // reflects a var the controller updates
Button("Refresh") => weather.fetch(city) // fire-and-forget; result lands next tick
```

While `fetch` stalls for 2 seconds, **the UI keeps rendering at 60/144 Hz** — the
render thread is drawing the last atomic `RenderFrame` and never calls into
Python. When the result arrives, it crosses back as a `Send` message and updates
the bound `var` on the logic thread (RFC-0004 tick step 3), marking it dirty.

For bulk data (sensor buses, numeric matrices), the guest writes raw bytes into a
**shared-memory ring** that Byard reads by reference — no per-value marshalling.

## Reference-level explanation

### Isolation model — the guest never touches Byard memory

Each guest runtime runs on **its own OS thread** (or process; see below), separate
from both the logic thread and the render thread:

```
 render thread ── draws last RenderFrame ── never enters a guest
 logic thread  ── arena, signals, tick ── owns the Byard↔guest mailbox
 guest thread  ── QuickJS / CPython VM ── only sends/receives Send POD + rings
```

- The guest can only call a **narrow FFI surface** (`emit(var, value)`, `call`,
  `subscribe`) that enqueues `Send` messages onto a `bounded(N)` channel drained
  at tick step 3. It has **no handle** to `ViewArena`, `Signal`, or `RenderFrame`.
- Values crossing the boundary are a closed `enum GuestValue` of POD scalars +
  the shared-ring reference (below). No Rust reference ever escapes into guest
  memory and vice-versa. This preserves INV-1/INV-2 and the `!Send`/`!Sync`
  discipline: nothing non-`Send` crosses.

### GIL / GC isolation — the frame guarantee

Because the guest is on its own thread and the render thread never calls it, a GIL
acquisition or GC pause inside the guest **cannot** block rendering: the render
thread's only input is the atomic frame swap, which is always the last-good frame.
A guest that stalls simply doesn't post new messages that tick; the UI shows the
last state until it does. This is the same "async result arrives later" model as
Rust controllers — the guest is just a slower, sandboxed producer.

Back-pressure: the Byard→guest and guest→Byard channels are `bounded` with a
latest-wins or drop policy per channel, so a runaway guest cannot grow memory
unboundedly (bounded-memory principle).

### Zero-copy for bulk flows — *validated*, not blind

The original sketch (guest writes raw bytes, Byard consumes by reference) is right
in spirit but must not be `unsafe` transmute of arbitrary bytes — that would be a
soundness bug the project explicitly forbids. The design:

- A **shared-memory ring** (`memmap`'d region, single-producer/single-consumer)
  carries records of a **declared, versioned POD schema** (think `bytemuck::Pod`
  or `rkyv` archived types with a schema id + length prefix).
- The consumer (logic thread) **validates** each record cheaply before viewing it:
  schema id matches, length in bounds, alignment correct. Only then is it read as
  `&T` (a `bytemuck`-style checked cast, no UB path). Malformed records are
  rejected, not dereferenced.
- For truly hot numeric matrices, the validated header is `O(1)` and the payload
  bytes are viewed in place — so the *marshalling* cost is gone while soundness is
  kept. This is "zero-copy with a seatbelt."

### Guest footprint policy

- **JavaScript (QuickJS via `rquickjs`)** is the recommended default guest:
  tiny RAM footprint, fast startup, embeds cleanly, `no_std`-friendly. Good fit
  for Byard's minimalism.
- **Python (`pyo3`/CPython)** is heavier (large binary, GC, GIL). It is
  **feature-gated (`guest-python`) and opt-in**, and documented as a
  footprint/latency trade-off. For the strictest targets, an **out-of-process**
  Python option (guest in a child process, same message/ring protocol over a pipe)
  keeps CPython out of the app binary entirely. The message contract is identical
  whether the guest is in-thread or out-of-process.

### The `@byard_controller` metadata (shared with M23)

Guest controllers expose the same metadata the interpreter's `inject`/member-type
inference needs (RFC-0006 M23). For guests, the macro/decorator emits a small
manifest (method names, arg/return POD types) so the `byld` side can type-check
`weather.temp`/`weather.fetch(city)` at compile time even though the body is
Python/JS. Type mismatches at the boundary are compile errors, not runtime
surprises.

### What guests may **not** do

- No synchronous call from the render thread into a guest (structurally
  impossible — render thread has no guest handle).
- No access to the arena, signals, layout, or frame.
- No non-POD, non-validated data across the boundary.
- No unbounded channel or ring (bounded-memory invariant).

## Drawbacks

- Embedding CPython contradicts minimal-footprint goals; hence opt-in + gating +
  out-of-process option.
- Two+ guest runtimes multiply the FFI/marshalling surface and the test matrix.
- Shared-memory rings + schema validation are non-trivial to get right (though far
  safer than blind transmute).
- Async-only result model means guests can't do synchronous UI queries — a
  deliberate constraint some devs will need to learn.

## Rationale and alternatives

- **Thread/process isolation vs in-line embedding.** Isolation is what makes the
  frame guarantee structural rather than aspirational. In-lining a GIL into the
  logic thread would risk stalls.
- **Validated zero-copy vs blind reinterpretation.** The project's "no soundness
  bugs" rule forbids transmuting arbitrary guest bytes; a cheap checked cast keeps
  the performance while closing the hole.
- **QuickJS-first, Python-gated.** Matches the footprint priority; still serves the
  large Python audience via an explicit, honest trade-off.
- **Keep the Rust boundary primary.** Guests are an adoption on-ramp, not the
  recommended path for perf-critical controllers.
- **Not doing it:** Byard stays Rust-only on the logic side, capping adoption.

## Prior art

Tauri (JS frontends + Rust core, IPC boundary), Neovim (embedded Lua/`msgpack`
RPC), Blender (embedded CPython on a separate concern from the draw loop), game
engines embedding Lua/JS for scripting, `rquickjs`/`deno_core` embedding patterns,
Apache Arrow / shared-memory columnar transfer (validated zero-copy), `bytemuck`/
`rkyv` for checked POD casts.

## Resolved decisions (2026-07-01)

- **X1 — runtimes:** **QuickJS (`rquickjs`) in-process by default**; **Python
  out-of-process by default**, with an opt-in `guest-python-inproc` feature. Protects
  footprint/isolation; identical message+ring protocol either way.
- **X2 — `GuestValue` / ring schema:** **`bytemuck`-POD with a schema id + length/
  alignment validation** (simplest checked, zero-runtime cast; enough for scalars +
  numeric matrices). No blind transmute. (`rkyv` rejected as over-heavy for PODs.)
- **X3 — channel policy:** **per direction** — Byard→guest **latest-wins** (only the
  last state matters, like the D10 watcher); guest→Byard **bounded + visible error**
  when full (silent result loss is dangerous).
- **X4 — error propagation:** guest exceptions become a **`ByardError`** with guest
  context (file/line when available) — no silent swallowing (INV-4).
- **X5 — guest hot-reload:** **reload only the affected guest, preserving bound
  `var`s** (same promise as `.byd` hot-reload, RFC-0006 E5).
- **X6 — sandbox:** **declared per-controller capabilities** (which resources a
  controller may touch) **+ process isolation** reinforcing it when out-of-process.
  Declaration enters at design; fine-grained enforcement deferred to implementation.

## Unresolved questions (deferred to implementation)

- [ ] The concrete capability vocabulary (net/disk/env/…) and its enforcement points.
- [ ] `bytemuck` schema versioning across guest/host build skew.

## Future possibilities

- WASM guests (`wasmtime`) as a third, strongly-sandboxed runtime — arguably the
  cleanest isolation of all.
- A language-agnostic controller ABI so any runtime implementing the message/ring
  protocol plugs in.
- Capability-scoped guests (declare which OS resources a controller may touch).
