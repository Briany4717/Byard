# Byard

**A high-performance, cross-platform UI framework with direct-to-GPU rendering, written in Rust 🦀**

[![CI](https://github.com/Briany4717/byard/actions/workflows/ci.yml/badge.svg)](https://github.com/Briany4717/byard/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#project-status)

---

> **Project status: pre-alpha.**
> The engine, compiler, and dev toolchain are functional and tested — you can
> write a `.byd` file today and run it live-reloading in a native window.
> Working end to end: interactive widgets (`Toggle`/`Slider`/`TextField`) with
> focus and keyboard input; decorated rendering (border, shadow, opacity,
> per-corner `radius`); paint-time transforms (`translate`/`scale`/`rotate`)
> with coherent cross-pass paint ordering; a theme system; the
> `#[byard_controller]` Rust boundary; dirty-rectangle incremental redraw; async
> image loading; and a zero-allocation frame profiler. Per-property animations,
> interactive style states, and richer view composition are actively in
> progress. Public APIs and the `byld` syntax will change before the first
> stable release.

## What is Byard?

Byard is a UI framework built around a single idea: **the declarative layer and the
systems layer should never live in the same file.**

- **`byld`** — a statically-typed DSL used *exclusively* to declare UI structure,
  styling, and visual reactivity.
- **Rust** — used *exclusively* for business logic: networking, disk, cryptography,
  OS integration, and anything that touches the real world.

The two communicate through compile-time-generated, zero-cost bindings. There is no
IPC, no serialization boundary, and no runtime glue.

Byard renders directly to the GPU through [`wgpu`](https://github.com/gfx-rs/wgpu),
lays out with [`taffy`](https://github.com/DioxusLabs/taffy), and rasterizes text
with [`glyphon`](https://github.com/grovesNL/glyphon). It has **no garbage
collector**: memory is owned by component-scoped arenas released in a single `O(1)`
operation when a view unmounts.

## Why does Byard exist?

Existing UI stacks each carry a structural cost Byard is designed to avoid:

| Ecosystem | Strength | Cost Byard rejects |
|-----------|----------|--------------------|
| Web / DOM | Universal reach | A document model forced to be interactive — heavy RAM and CPU use |
| Flutter / Dart | Excellent cross-platform story | Verbosity and deep "wrapper hell" widget trees |
| Pure Rust UI | Memory safety, real concurrency | A hostile developer experience fighting the borrow checker for layout |

Byard's goal is the ergonomics and readability of React/SwiftUI with the memory
safety, concurrency, and low-level control of Rust — and **deterministic
performance**: stable frame times, no GC pauses, no VRAM spikes.

## Design principles

1. **Strict domain separation.** `byld` is for design; Rust is for logic. They are
   never mixed in one file.
2. **Zero garbage collector.** Memory is managed through Rust ownership and
   component-scoped memory arenas.
3. **Deterministic, raw performance.** Stable frame rate and bounded VRAM are
   first-class correctness criteria, not aspirations.
4. **No raw math in the view.** The declarative layer exposes organic concepts —
   views, signals, environments — never graphs, pointers, or Z-indices.
5. **Live reload by default.** `byard dev` reflects every save in the running window,
   with state preserved on reactive-compatible changes and gesture safety on
   structural ones.

## A taste of `byld`

```
View Counter() {
    var count = 0

    Column #[gap: 20, p: 32, align: center, justify: center] {
        Text("{count} taps") #[size: 24, color: 0xFFFFFF]

        Button("+") #[bg: 0x3B82F6, radius: 8, p: 10,
                      color: 0xFFFFFF, weight: bold] => count++
    }
}
```

Wrapper components (`Padding`, `Align`, …) are intentionally absent — spatial and
decorative properties are inline arguments on the element they affect.

`radius` also accepts a positional 4-tuple for independent per-corner control —
`radius: (4, 8, 12, 16)` in CSS-style `top_left, top_right, bottom_right,
bottom_left` order — with a plain scalar still broadcasting to all four corners.

Paint-time transforms move, resize, and rotate an element *visually* without
touching layout: a hover-to-lift card is just `scale: hovered ? 1.03 : 1.0`, and
its siblings never reflow.

## Getting started

```sh
# Scaffold a new project
byard new my_app
cd my_app

# Start the live-reload dev window
byard dev

# Validate without opening a window (CI-friendly)
byard check
```

Edit `main.byd` and save — the window updates within one frame. No recompile,
no `cargo run`, no hot key.

> **Running from this repo (pre-release).** `byard` isn't published yet, so
> there's no `byard` on your `PATH`. Invoke the CLI through Cargo and pass a
> `.byd` path directly (no `byard.toml` needed):
>
> ```sh
> # Live-reload dev window for the bundled demo
> cargo run -p byard-cli -- dev crates/byard-compiler/examples/hello_world.byd
>
> # Validate only (no window)
> cargo run -p byard-cli -- check crates/byard-compiler/examples/hello_world.byd
> ```

## Architecture at a glance

The engine is four concurrent subsystems:

- **Logic subsystem** — interprets state (`var`/`let` signals) and owns the
  per-view memory arenas. Runs on a dedicated thread.
- **Spatial subsystem** — Taffy-based layout plus a spatial hash grid for `O(1)`
  hit-testing, decoupled from the UI tree.
- **Render subsystem** — a multi-pipeline `wgpu` command dispatcher (`SolidBox`,
  `TextGlyph`, `DecoratedBox`, `TextureSampler`), with dirty-rectangle tracking
  and GPU scissor clipping so an incremental frame only repaints what changed
  (RFC-0001 §3.3).
- **Concurrency subsystem** — double-buffered visual state, the Relay signal bus,
  and a Tokio pool for async I/O.

The `byard-cli` dev runner wires these together with a `notify` OS file watcher.
On every save, the view's shape is diffed: reactive-compatible patches apply
instantly (signal state preserved); structure-incompatible patches are held past
any in-flight pointer gesture, then applied cleanly.

The foundational design is specified across six RFCs in [`docs/rfcs/`](docs/rfcs/):

| RFC | Topic |
|-----|-------|
| [0001](docs/rfcs/0001-core-architecture.md) | Core architecture, crate layering, memory model, `PlatformHost` |
| [0002](docs/rfcs/0002-byld-language-and-compiler-pipeline.md) | `byld` language, compiler pipeline, hot-reload boundary |
| [0003](docs/rfcs/0003-interactive-events-and-view-mutation.md) | Event system, gesture recognition, write-back |
| [0004](docs/rfcs/0004-reactive-interpreter.md) | Reactive core: Mark-and-Pull, memos, structural scopes |
| [0005](docs/rfcs/0005-intrinsic-view-catalog.md) | Built-in view catalog (`Column`, `Button`, `TextField`, …) |
| [0006](docs/rfcs/0006-cli-and-dev-runner.md) | `byard` CLI, dev runner, live-reload wiring |

Later subsystems — transforms, animations, interactive styling, telemetry, and
view composition — are specified in their own RFCs, each landing alongside the
feature it describes.

## Crate layout

```
crates/
  byard-core/       — engine subsystems (renderer, atlas/layout, relay, frame)
  byard-compiler/   — byld lexer, parser, reactive interpreter, hot-reload logic
  byard-platform/   — PlatformHost implementations (winit + wgpu)
  byard-cli/        — the `byard` binary (new / dev / check / build)
  byard-macro/      — #[byard_controller] proc-macro (the byld ↔ Rust boundary)
  byld-lsp/         — language server (in progress)
```

## Roadmap

| Phase | Status | Scope |
|-------|--------|-------|
| **0 — Design** | ✅ complete | Core architecture, crate layering, memory model |
| **1 — Engine core** | ✅ complete | `wgpu` renderer, Taffy layout, Relay threading, `PlatformHost` |
| **2 — `byld` compiler & dev toolchain** | ✅ complete | Lexer, parser, reactive interpreter (Mark-and-Pull), event router, hot-reload, `byard-cli` |
| **3 — Interactive widgets & rendering polish** | ✅ complete | `Toggle`/`Slider`/`TextField`, focus and keyboard input, `for`/`when` in render, decorated rendering, theming, the controller boundary, dirty-rect redraw, async assets |
| **4 — Motion & interactive styling** | 🟡 in progress | Paint-time transforms and coherent cross-pass paint order (done); a zero-allocation frame profiler (done); per-property `with` animations and interactive style states (underway) |
| **5 — View composition & packages** | 🔲 planned | Composing user `View`s within and across files; modules, dependencies, and distribution |
| **6 — Production transpiler** | 🔲 planned | `byld` → native Rust AOT compilation and an accessibility bridge |

## Contributing

Byard is open to contributions. The best entry points are:

- Read the relevant RFC before touching a subsystem — the design decisions are the
  contract; the code is the implementation.
- `cargo test --workspace` and `cargo clippy --workspace` must stay green.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be
dual-licensed as above, without any additional terms or conditions.

Made with love for the Rust community.
For those who believe UI deserves the same rigor as systems code.
