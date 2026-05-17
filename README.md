# Byard

**A high-performance, cross-platform UI framework with direct-to-GPU rendering, written in Rust 🦀**

[![CI](https://github.com/Briany4717/byard/actions/workflows/ci.yml/badge.svg)](https://github.com/Briany4717/byard/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#project-status)

---

> **Project status: pre-alpha — design phase.**
> Byard is currently a set of design documents and an architectural plan. There is
> no usable build yet. The public interface, the `bylang` DSL syntax, and the
> crate layout are all expected to change. This README describes the **intended**
> system; see [the RFCs](docs/rfcs/) for the authoritative design.

## What is Byard?

Byard is a UI framework built around a single idea: **the declarative layer and the
systems layer should never live in the same file.**

- **`bylang`** — a statically-typed DSL used *exclusively* to declare UI structure,
  styling, and visual reactivity.
- **Rust** — used *exclusively* for business logic: networking, disk, cryptography,
  OS integration, and anything that touches the real world.

The two communicate through compile-time-generated, zero-cost bindings. There is no
IPC, no serialization boundary, and no runtime glue.

Byard renders directly to the GPU through [`wgpu`](https://github.com/gfx-rs/wgpu),
lays out with [`taffy`](https://github.com/DioxusLabs/taffy), and rasterizes text
with [`glyphon`](https://github.com/grovesNL/glyphon). It has **no garbage
collector**: memory is owned by component-scoped arenas that are released in a
single `O(1)` operation when a view is unmounted.

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

1. **Strict domain separation.** `bylang` is for design; Rust is for logic. They are
   never mixed in one file.
2. **Zero garbage collector.** Memory is managed through Rust ownership and
   component-scoped memory arenas.
3. **Deterministic, raw performance.** If it does not hold a stable frame rate, or it
   spikes VRAM, the architecture has failed. Asynchronous hardware acceleration is
   preferred over CPU-side logic.
4. **No raw math in the view.** The declarative layer exposes organic concepts —
   views, signals, environments — never graphs, pointers, or Z-indices.

## Architecture at a glance

The engine is four concurrent subsystems:

- **Logic subsystem** — interprets state (`Signal`s) and owns the per-view memory
  arenas.
- **Spatial subsystem** — topological layout via Taffy plus a parallel spatial hash
  grid for `O(1)` hit-testing, fully decoupled from the UI tree.
- **Render subsystem** — a multi-pipeline `wgpu` command dispatcher (no über-shader).
- **Concurrency subsystem** — thread management, double-buffered visual state, and a
  Tokio pool for async I/O.

The full design — memory model, the multi-pipeline renderer, the spatial hit-testing
grid, the threading model, and the `bylang` compiler pipeline — is specified in
[**RFC-0001: Core Architecture**](docs/rfcs/0001-core-architecture.md).

## A taste of `bylang`

```
// Conceptual bylang — syntax is not final.
View UserCard() {
    signal clicks = 0
    inject AppEnvironment as env

    Column(gap: 12, bg: env.theme.surface, radius: 16, p: 20) {
        Text("Clicks: {clicks}", typo: m3.titleLarge)
        Button("Action", onClick: () => clicks++)
    }
}
```

Wrapper components (`Padding`, `Align`, …) are intentionally absent — spatial and
decorative properties are passed as arguments to the base component.

## Roadmap

Byard is being built in phases.

- **Phase 0 — Design** *(complete)* — RFCs, architecture, crate layout.
- **Phase 1 — Engine core** *(current)* — `wgpu` multi-pipeline renderer, Taffy
  integration, the spatial hash grid, and the double-buffered threading model.
  Scope defined in RFC-0001 9. Progress tracked in the
  [Phase 1 milestone](https://github.com/Briany4717/byard/milestones).
- **Phase 2 — `bylang` compiler** — `logos` lexer, hand-written recursive descent
  parser, and the dev-mode AST interpreter.
- **Phase 3 — Rust ↔ `bylang` bridge** — the `#[byard_controller]` macro and LSP
  metadata generation.
- **Phase 4 — Production transpiler** — `bylang` → native Rust for AOT builds.

Roadmap items will be tracked as
[GitHub milestones](https://github.com/Briany4717/byard/milestones) as the project moves
out of the design phase.

## Contributing

Byard is open to contributions from day one. The
[Phase 1 milestone](https://github.com/Briany4717/byard/milestones) tracks the
current implementation work — look for issues labelled `phase-1` and
`good first issue` to get started.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md) before opening an issue or pull request.

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
