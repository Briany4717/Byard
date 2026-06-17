# RFC-0001: Core Architecture

- **Status:** Active
- **Author:** Briany4717
- **Created:** 2026-05-03
- **Last updated:** 2026-05-14

---

## Summary

This RFC defines the foundational architecture of Byard: a high-performance,
cross-platform, direct-GPU-rendering UI framework built on `wgpu` and `winit`,
with a zero-garbage-collector memory model and a companion declarative DSL called
`byld`. It covers the four engine subsystems, the memory arena model, the
multi-pipeline renderer, the spatial hit-testing grid, the multi-threaded
concurrency design, and the `byld` compiler pipeline.

---

## Motivation

Every major UI stack carries a structural cost that Byard is designed to avoid.

**Web / DOM.** The DOM is a document model retrofitted for interactivity. The
result is high memory use, CPU pressure, and a rendering architecture that
cannot reason about GPU budget. JavaScript's garbage collector introduces
non-deterministic frame pauses.

**Flutter / Dart.** Excellent cross-platform ergonomics, but the widget model
encourages deep "wrapper hell" trees (a standalone `Padding` widget, an
`Align` widget, …) that generate boilerplate with no semantic value. The Dart
VM adds a runtime layer with its own GC.

**Pure-Rust UI toolkits.** Full memory-safety and concurrency guarantees, but
the borrow checker makes stateful, reactive UI trees hostile to write. The
developer experience destroys creative flow.

Byard's thesis: the ergonomics and readability of React or SwiftUI are
achievable with the memory safety, concurrency, and low-level control of Rust,
provided the declarative and the systems layers are **strictly separated** and
the declarative layer never owns complex Rust lifetimes.

---

## Guide-level explanation

### The two layers

A Byard application consists of exactly two kinds of files:

- **`.byd` files** — written in `byld`. These describe UI structure, styling,
  and visual reactivity. They contain no network calls, no file I/O, and no
  business logic.
- **`.rs` files** — written in Rust. These implement controllers: networking,
  disk, cryptography, OS integration, and application state that outlives a
  single view. Controllers are declared with `#[byard_controller]` and exposed
  to `byld` through a zero-cost, compile-time-generated boundary.

The two are never mixed in a single file. This is a hard architectural rule,
not a style guideline.

### `byld` at a glance

```
View UserCard() {
    signal clicks = 0
    inject AppEnvironment as env

    Column(gap: 12, bg: env.theme.surface, radius: 16, p: 20) {
        Text("Clicks: {clicks}", typo: m3.titleLarge)
        Button("Action", onClick: () => clicks++)
    }
}
```

`View` is the fundamental unit. Wrapper components (`Padding`, `Align`, …) do
not exist; spatial and decorative properties are passed as named arguments.
`signal` declares reactive local state. `inject` pulls a typed value from the
ambient environment, equivalent to React Context.

### Rust controller

```rust
#[byard_controller]
pub struct NetworkController {
    base_url: String,
}

impl NetworkController {
    pub async fn fetch_user(&self, id: u64) -> Result<User, ApiError> { … }
}
```

`#[byard_controller]` generates shared-memory bindings and a typed metadata
file consumed by the `byld` LSP, giving the developer full autocompletion
across the language boundary without a serialization round-trip.

---

## Reference-level explanation

### 1. The four subsystems

The engine is composed of four concurrent subsystems:

| Subsystem | Internal name | Responsibility |
|-----------|---------------|----------------|
| Logic | `Evaluator` | Signal state, memory arenas, dirty-flag collection |
| Spatial | `Atlas` | Taffy layout, spatial hash grid for hit-testing |
| Render | `Encoder` | Multi-pipeline wgpu command dispatch |
| Concurrency | `Relay` | Thread management, double-buffered visual state, Tokio I/O pool |

### 2. Memory model (Zero-GC)

**No `Box::new` for per-view allocations.** Individual heap allocations fragment
memory and add GC pressure even in Rust (via reference counting or arena growth).

#### 2.1 `ViewArena`

When a `View` is mounted, the engine allocates one contiguous memory block
called a `ViewArena`. It stores:

- All `Signal<T>` values for that view instance.
- References to the view's Taffy nodes.
- Spatial hash grid entries for the view's interactive regions.

When the view is unmounted (e.g. a navigation pop), `ViewArena::drop` is called
once. The entire block is reclaimed in `O(1)`. There is no deferred collection,
no reference-count decrement cascade, and no latency spike.

#### 2.2 `Signal<T>`

A `Signal<T>` contains:

1. The current value of type `T`.
2. A vector of atomic dirty flags pointing to specific render or spatial
   subsystem entries.

Mutating a `Signal` sets the relevant dirty flags. The Logic subsystem collects
dirty flags on each tick and derives the minimal set of dirty rectangles to
re-dispatch — without diffing a virtual DOM tree.

### 3. Render subsystem

#### 3.1 Multi-pipeline architecture

To support tile-based deferred rendering (TBDR) GPUs common on mobile — where
an über-shader causes bandwidth exhaustion — the renderer uses small, specialized
pipelines compiled at startup from parallel WGSL modules:

| Pipeline | Draws |
|----------|-------|
| `SolidBox` | Axis-aligned rectangles with solid fill |
| `DecoratedBox` | Rectangles with `border-radius`, gradients, box-shadows |
| `TextGlyph` | Text via a `glyphon` glyph atlas |
| `TextureSampler` | UV-mapped quads (images, icons) |

#### 3.2 Batching and stacking contexts

Primitives are not sorted by global Z-order (that would cause GPU context
switches per primitive). Instead they are organized into **Z-bins** (stacking
contexts). Within each bin, draw calls are ordered first by pipeline, then by
local Z, allowing hundreds of primitives to be flushed in a single `draw_call`.

#### 3.3 Dirty rectangles and scissor clipping

When a `Signal` mutates:

1. The Logic subsystem computes the bounding box of the affected region.
2. The Render subsystem issues `wgpu::RenderPass::set_scissor_rect` for that
   bounding box.
3. Only primitives that intersect the scissor rect are re-submitted to the
   command buffer, minimising VRAM bandwidth.

### 4. Spatial subsystem

#### 4.1 Layout

All layout is delegated to [`taffy`](https://github.com/DioxusLabs/taffy).
The engine never computes box geometry itself.

#### 4.2 Spatial hash grid (hit-testing)

Hit-testing is treated as a **collision problem**, not a tree-traversal problem.
A spatial hash grid runs parallel to the UI tree.

- **Registration.** When an event handler (e.g. `onClick`) is declared, the
  engine registers the Taffy-resolved rect into the corresponding grid cell.
- **Query.** On `PointerDown(x, y)`, the engine computes the hash index
  `H(x, y)` and retrieves the handler in amortised `O(1)`. The UI tree is never
  walked during event dispatch.

### 5. Concurrency model

#### 5.1 Thread layout

| Thread | Role |
|--------|------|
| **Render thread** | Holds `Arc<RenderFrame>` (immutable). Runs the `request_redraw` loop. Never blocks. |
| **Logic thread** | Mutates signals, runs Taffy, updates the spatial grid. Produces a new `RenderFrame` and performs an atomic pointer swap with the render thread. |
| **Tokio pool** | Executes async I/O from Rust controllers. Sends results back to the logic thread via `tokio::sync::mpsc`. |

#### 5.2 Double buffering

The render thread and the logic thread never share mutable state. The logic
thread writes a new `RenderFrame`; the render thread reads the previous one.
The swap is a single atomic pointer exchange — no mutex, no stall.

### 6. Platform abstraction

The engine core has zero direct references to `winit`, Wayland, Win32, or any
other OS primitive. All platform interaction goes through the `PlatformHost`
trait. The concrete implementation (e.g. `WinitHost`) is injected at binary
compile time, giving zero-cost abstraction at runtime.

This design also allows the engine to be embedded as a pure rendering backend
with no windowing layer at all (the planned *Coreolis* use-case).

### 7. `byld` compiler pipeline

#### 7.1 Lexer

The lexer is generated by [`logos`](https://github.com/maciejhirsz/logos), which
produces a DFA-based tokeniser running at memory-bandwidth speed.

#### 7.2 Parser

A hand-written recursive descent parser consumes the token stream. This choice
prioritises:

- **Error quality.** A hand-written parser can emit precise, context-aware error
  messages with source spans — something PEG generators make difficult.
- **AST fidelity.** Full control over the AST shape.

#### 7.3 Execution modes

| Mode | Mechanism | Use-case |
|------|-----------|----------|
| **Dev** | AST interpreter | Hot-reload: changes are picked up without recompilation |
| **Prod** | Transpiler → Rust | AOT: `byld` compiles to native Rust that generates GPU primitives directly |

### 8. GPU error handling

Byard does not panic on GPU errors. All fallible engine operations return
`Result<T, ByardError>`. `ByardError` is a non-exhaustive enum defined in
`byard-core` that wraps `wgpu` error types with additional context (backend
name, shader stage, feature flag).

The error boundary is at initialisation time. During the pipeline compilation
phase at startup, the engine wraps the full creation sequence of each pipeline
— `Device::create_shader_module`, `Device::create_pipeline_layout`, and
`Device::create_render_pipeline` — inside a single
`Device::push_error_scope` / `Device::pop_error_scope` pair with
`ErrorFilter::Validation`. `pop_error_scope` returns a future that is driven
to completion before the render loop begins, guaranteeing that any validation
or compilation error from any stage of pipeline creation is captured. If any
pipeline fails, the engine returns
`Err(ByardError::PipelineCompilation { pipeline, reason })` to the caller.
The application decides how to handle it — log and exit, show a fallback UI,
or surface the error to the developer.

**There is no software rendering fallback.** Byard requires a `wgpu`-compatible
GPU with support for the features used by its pipelines. This constraint is
explicit and documented. Attempting to run on an unsupported backend returns
`Err(ByardError::UnsupportedBackend)` at device creation time.

### 9. `byard-core` crate layout

The `byard-core` crate maps directly to the four subsystems. Each subsystem is
a top-level module. No subsystem module imports from another subsystem module
directly — cross-subsystem communication goes through the types defined in the
`frame.rs` module, which is the shared data boundary.

```
crates/byard-core/src/
├── lib.rs          — public API surface, re-exports, ByardError
├── evaluator/      — Signal<T>, ViewArena, dirty-flag collection
├── atlas/          — Taffy integration, spatial hash grid
├── encoder/        — wgpu pipelines (SolidBox, DecoratedBox, TextGlyph, TextureSampler)
├── relay/          — thread management, RenderFrame, atomic frame swap, Tokio pool
└── frame.rs        — RenderFrame and the primitive types shared across subsystems
```

The dependency graph within the crate is strictly layered:

```
encoder  ──┐
atlas    ──┤─→  frame.rs  ←─  relay
evaluator ─┘
```

`frame.rs` is the only module that all subsystems may depend on. `encoder` and
`atlas` never depend on `relay`. `evaluator` never depends on `encoder`. Any
cross-subsystem dependency that violates this graph is a design defect and must
be resolved before merging.

---

## Drawbacks

**Parser maintenance.** A hand-written recursive descent parser requires
sustained discipline to extend without introducing regressions. Grammar changes
are more expensive than with a generated parser.

**Multi-language overhead.** Developers must context-switch between `.byd` and
`.rs` files. This is mitigated by the LSP providing seamless cross-language
navigation, but the initial learning curve is real for developers used to
single-file component models (e.g. Svelte, Vue SFCs).

**Accessibility.** Not using OS-native primitives means the engine must maintain
its own accessibility tree via
[`accesskit`](https://github.com/AccessKit/accesskit), mapped to Taffy's layout
output. Keeping this mapping correct and complete is a sustained engineering
commitment.

---

## Rationale and alternatives

**Why `wgpu` and not Vulkan/Metal directly?** Cross-platform without per-backend
maintenance. `wgpu` is the same backend already used by every major Rust UI
project in this space.

**Why hand-written parser and not `pest` / `chumsky`?** Error quality. Parser
combinator and PEG frameworks produce adequate errors for simple grammars but
struggle with the contextual hints (`byld` intends to produce SwiftUI-quality
diagnostics). The maintenance cost is real but the developer experience benefit
justifies it.

**Why separate threads and not async everywhere?** The render thread must hold a
hard vsync deadline. Async runtimes introduce polling latency that would cause
frame jitter. The render thread is a tight, synchronous loop; everything else
uses Tokio.

**Why Taffy and not a custom layout engine?** Taffy is battle-tested, covers the
full flexbox + CSS Grid spec, and is maintained by the Dioxus team. Writing a
layout engine is a multi-year distraction from the differentiated parts of Byard.

---

## Prior art

- **Flutter / Skia.** GPU-accelerated retained UI. Validated the direct rendering
  approach, but Dart's GC and the widget tree verbosity are the exact problems
  Byard solves.
- **Iced.** Rust-native UI using `wgpu`. The elm-like message model and lack of
  fine-grained reactivity are the main differences from Byard's signal model.
- **Bevy UI.** ECS-based UI in a game engine. Excellent concurrency, but the ECS
  model is not ergonomic for application UI trees.
- **SwiftUI.** The DX reference for `byld`. Declarative, property-chain styling,
  environment injection. Byard's DSL is directly inspired by it.
- **Solid.js signals.** The fine-grained reactivity reference: mutations touch
  only the affected DOM nodes, never re-run the whole component. Byard's `Signal`
  model targets the same property at the GPU primitive level.

---

## Unresolved questions

**Before merging / Phase 1 scope:**

- [x] **Phase 1 scope.** Resolved: Phase 1 closes when the engine renders a
  solid rectangle with `border-radius` and a reactive text label driven by a
  `Signal`. This exercises all four subsystems. Tracked in the Phase 1 milestone.
- [x] **Runtime GPU error handling.** Resolved: see section 8. No panic, no
  software fallback. `Result<T, ByardError>` at the initialisation boundary.
- [ ] **Accessibility tree.** Deferred to Phase 2. A dedicated `AccessBridge`
  subsystem will own the `accesskit` tree, subscribing to mount/unmount events
  from `Atlas`. `Atlas` exposes a notification interface and has no knowledge of
  accessibility internals. A sub-RFC is required before Phase 2 begins.

**During implementation:**

- [ ] **Testing strategy.** The double-buffered multi-threaded model is hard to
  test with standard `#[test]`. What is the approach — headless wgpu, snapshot
  tests of render output, deterministic frame replay?
- [ ] **`byld` syntax stabilisation.** The syntax shown in this RFC is
  conceptual. A grammar RFC must be written before the parser is implemented.
- [ ] **MSRV policy.** How aggressively do we track stable Rust? This affects
  `wgpu` version selection and feature availability.
- [ ] **Hot-reload boundary.** In dev mode, which changes trigger a full restart
  vs. an in-place signal update? Struct layout changes in a controller require a
  restart; `Signal` value changes do not. The rules need to be explicit.

---

## Future possibilities

- **System compositor.** Embedding the engine as a system-level GPU compositor, with no
  windowing layer. The `PlatformHost` abstraction is specifically designed to
  make this possible.
- **Native mobile targets.** The `PlatformHost` abstraction covers Android and
  iOS; a mobile platform implementation is a natural Phase N deliverable.
- **`byld` → native widgets.** A backend that maps `byld` views to OS-native
  controls (AppKit, Win32, GTK) for accessibility and system integration, as an
  alternative to the GPU path.