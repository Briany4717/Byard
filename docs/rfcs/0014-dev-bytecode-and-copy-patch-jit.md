# RFC-0014: Dev Bytecode IR + Optional Copy-and-Patch JIT

- **Status:** Draft — design proposal
- **Author(s):** Brian (byard_v2)
- **Created:** 2026-07-01
- **Last updated:** 2026-07-01
- **Depends on:** RFC-0002 (compiler pipeline, lexer/parser/lower), RFC-0004 (reactive interpreter — the current tree-walker), RFC-0006 (dev runner, hot-reload), RFC-0013 (telemetry — the gate).
- **Gated by:** RFC-0013's interpreter-tax measurement. **Do not build the JIT until the profiler shows the tree-walker is the measured bottleneck.**

---

## Summary

Replace the `byard dev` tree-walking AST interpreter with a **two-tier execution
model** for a linear UI bytecode:

- **Tier 0 — a flat bytecode interpreter** over a linear UI opcode stream. Big,
  portable, safe win over AST tree-walking; ships everywhere (including targets
  that forbid executable memory); no `unsafe`.
- **Tier 1 (optional) — a Copy-and-Patch JIT** that `memcpy`s pre-compiled
  machine-code *stencils* (built by `rustc` at Byard's own AOT build time) into an
  executable page and patches in dynamic operands. Compilation is memory-copy
  speed (microseconds), eliminating Dev-Runner jank.

The deliberate framing: **Tier 0 first, measured; Tier 1 only if RFC-0013 proves
it's worth the `unsafe`/platform cost.** This respects "correctness before
optimization" and avoids reinventing a heavy backend (Cranelift) whose RAM and
latency overhead the original analysis correctly rejects.

## Motivation

`byard dev` today walks the AST every tick (RFC-0004). Tree-walking pays for
pointer-chasing, `enum` dispatch, and env lookups on every node, every frame —
the "interpreter tax" RFC-0013 measures. When a `.byd` file is saved, hot-reload
re-lowers and re-walks. For large views this can stutter the Dev Runner, which is
exactly where DX is felt most.

The wrong fix is a general JIT backend: Cranelift (or LLVM) brings hundreds of ms
of compile latency and tens of MB of RAM — antithetical to Byard. The right fixes,
in order:

1. **Lower the AST to a linear bytecode** and interpret *that* — removes tree
   pointer-chasing and shrinks dispatch to a tight loop. Portable, safe, big win.
2. **Optionally** turn the same bytecode into native code by *copying* stencils —
   near-zero compile latency, no backend, no IR optimizer.

Copy-and-Patch (Xu & Kjolstad, PLDI 2021) is the technique that makes (2) cheap:
the optimizing compiler (`rustc`) runs *once, ahead of time*, on the framework
primitives; runtime "compilation" is just `memcpy` + operand patching.

## Guide-level explanation

Nothing changes in the `byld` a developer writes. The change is entirely under
`byard dev`: saves feel instant and frames get cheaper. The dev overlay (RFC-0013)
shows the interpreter-tax bucket shrink as tiers engage, and shows which tier is
live (`Tier0-bytecode` / `Tier1-jit` / `Tier0-fallback`).

Platforms that forbid W^X executable pages (iOS, some embedded/console targets)
**automatically** run Tier 0 — identical behaviour, same bytecode, just
interpreted. There is no separate code path to maintain: Tier 1 is an accelerator
over Tier 0's IR, not a fork of the logic.

## Reference-level explanation

### The UI bytecode IR

`byard-compiler` lowers a validated view to a **flat, linear opcode stream** —
one pass, no tree. Opcodes are the framework's reactive/build primitives, e.g.:

```
LOAD_SIG  s3            ; read signal 3 (tracked)
CONST_I   10
SELECT                  ; ternary pick from bool on stack
BUILD_BOX box_tmpl, ...
SET_ATTR  radius
BIND_MEMO m2
REGISTER_HANDLER tap, action_ref
```

The stream is `Send` POD (crosses to no other thread, but stays alloc-friendly and
cache-linear). Hot-reload diffs bytecode ranges instead of AST subtrees. This IR
is the **shared contract** both tiers execute.

### Tier 0 — flat bytecode interpreter

A `while pc < len { match op { … } }` loop over the stream with a small operand
stack. Removes AST pointer-chasing and recursion; dispatch is a dense `match`
(the compiler builds a jump table). No `unsafe`, no executable memory. This alone
is expected to cut a large fraction of the interpreter tax and is the **mandatory,
portable baseline**. It also becomes the semantic reference against which Tier 1
is differentially tested (§Correctness).

### Tier 1 — Copy-and-Patch

**Stencils (AOT, in the CLI binary).** For each opcode, a tiny Rust function is
written so `rustc` compiles it to a relocatable machine-code fragment with
well-known **holes** (operand slots, continuation addresses). At Byard's build
time these fragments are extracted (a build script emits their bytes + hole
offsets) and embedded as `&'static [u8]` tables in the CLI. `rustc`'s optimizer
has already run — there is no runtime optimizer.

**Runtime patching.** On save/first-run, the JIT walks the bytecode and, per
opcode:

1. `memcpy` the stencil bytes into a `mmap`'d page (allocated W, later flipped to
   X — never simultaneously writable+executable).
2. Patch the holes: write dynamic operands (signal addresses, style constants,
   handler thunk pointers) into the reserved offsets.
3. Chain to the next stencil (fall-through or patched jump).

Then `mprotect` the page to executable and call it. "Compilation" is
`O(bytes)` `memcpy` + a handful of stores — microseconds, no jank.

**Isolation of `unsafe`.** All raw-memory/W^X work is confined to one audited
module (`interp/jit/`), behind a safe API (`compile(&Bytecode) -> ExecutablePage`),
with `ExecutablePage` an RAII owner that `munmap`s on drop. The rest of the engine
never sees a raw pointer.

### Fallback & platform gating

Tier 1 is compiled in only for targets with dynamic executable memory and is
selected at runtime after a capability probe (can we `mmap` + `mprotect` to X?).
Any failure — capability probe, patch mismatch, unknown opcode — **degrades to
Tier 0 for that stream**, logged in the overlay. Correctness never depends on
Tier 1 succeeding.

### Interaction with the Phase-4 AOT transpiler

Byard's endgame (RFC-0002 D-future) is transpiling `byld` → native Rust for
release. This RFC is explicitly the **dev-loop** accelerator, not that. But the
bytecode IR is a useful common lowering target that the Phase-4 transpiler can
also consume, so the work is not throwaway. This RFC does **not** change release
builds.

### Correctness (non-negotiable)

- Tier 0 is the reference semantics; the existing RFC-0004 fixtures + proptest are
  re-run against it.
- Tier 1 is validated **differentially**: for a corpus of views and input
  sequences, Tier 0 and Tier 1 must produce byte-identical `RenderFrame`s and
  signal states. A divergence fails CI.
- Stencil extraction is checked by a build-time test that each stencil's holes
  match its opcode's operand arity.
- `unsafe` blocks carry `// SAFETY:` invariants and are covered by Miri where the
  memory model allows (JIT'd code excepted; the patcher's bookkeeping is not).

## Drawbacks

- Tier 1 introduces `unsafe`, raw executable memory, and a build-time stencil
  extraction step — real complexity and a security surface (W^X discipline is
  mandatory).
- Two execution tiers to keep semantically identical (mitigated: shared IR +
  differential testing).
- Stencil extraction is toolchain-sensitive (relocations, calling convention) and
  may need per-arch care.
- If RFC-0013 shows the tree-walker *isn't* the bottleneck, Tier 1's cost isn't
  justified — hence the gate.

## Rationale and alternatives

- **Copy-and-Patch vs Cranelift/LLVM.** The original analysis is right: heavy
  backends bring unacceptable RAM/latency. Copy-and-Patch moves optimization to
  AOT (`rustc`) and makes runtime codegen a `memcpy`.
- **Bytecode-first vs JIT-first.** Bytecode is most of the win, portable, and
  `unsafe`-free; it also gives Tier 1 a reference to be tested against. Shipping
  it first de-risks everything.
- **Shared IR vs two independent engines.** One IR, two executors, keeps semantics
  single-sourced.
- **Not doing it:** Dev Runner stutter on large views persists; the framework's
  "impeccable DX" claim weakens where it's most visible.

## Prior art

Copy-and-Patch Compilation (Xu & Kjolstad, PLDI 2021); LuaJIT interpreter design;
CPython 3.13 experimental copy-and-patch JIT; WebKit's template/baseline JITs;
`wasmtime`'s Winch baseline. The W^X + fallback model mirrors how JS engines run
on iOS (interpreter when JIT is disallowed).

## Resolved decisions (2026-07-01)

- **J1 — the gate:** build **Tier-0 (bytecode) first and measure** with RFC-0013;
  **Tier-1 only if the measured interpreter tax justifies it**. Core "measure before
  optimize" discipline.
- **J2 — bytecode encoding:** **fixed-width opcodes + a side constant pool** (dense,
  predictable, cache-linear loop; serves both tiers; literals don't bloat the stream).
- **J3 — stencil extraction:** **build script over `rustc` output** with a build-time
  test asserting each stencil's holes match its opcode arity (`rustc` stays the
  optimizer). Hand-written asm only where an arch demands it.
- **J4 — target matrix:** **x86-64 and aarch64 first** (desktop + Apple Silicon/ARM =
  ~99% of real `byard dev`); other arches run Tier-0.
- **J5 — W→X lifecycle:** **`mmap` W → patch → `mprotect` X**, with `__clear_cache`
  on aarch64; strict W^X, confined to `interp/jit/` behind `// SAFETY:`.
- **J6 — hot-reload:** **recompile the whole stream in Phase 1** (copy-and-patch is
  microseconds; incremental range-patching is a later optimization gated on 0013).

## Unresolved questions (deferred to implementation)

- [ ] Exact opcode set once the bytecode IR is drafted.
- [ ] Per-arch relocation specifics surfaced by the stencil build script.

## Future possibilities

- Share the bytecode IR with the Phase-4 AOT Rust transpiler.
- Tiered warmup: interpret cold streams, JIT hot ones (guided by RFC-0013).
- Inline caches for member-access/dispatch opcodes.
