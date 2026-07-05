# Changelog

All notable changes to Byard will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Byard uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- When writing entries use these categories:
     Added / Changed / Deprecated / Removed / Fixed / Security -->

## [Unreleased]

### Added

- **RFC-0009 `VectorIcon` renders live in `byard dev`.** The `VectorMSDF`
  pipeline is now actually wired into the render loop (atlas + pipeline built
  at startup, drawn every frame) and participates in cross-pipeline paint
  order like every other primitive. A background dev-mode dispatcher generates
  each icon's field on its own worker thread the first time it's referenced;
  the call site paints a zero-opacity placeholder until the field lands, then
  the icon appears — no stall, no re-render trigger needed from the caller.
- **RFC-0009 vector/icon MSDF generator.** `byard_compiler::vector` turns an
  SVG icon into a multi-channel signed distance field: a structural complexity
  guardrail (rejects gradients, patterns, filters, and oversized path sets),
  and a generator that parses/normalizes with `usvg` and produces the field
  with a pure-Rust generator, deterministically and with sharp corners
  preserved at any scale.
- **RFC-0008 package ecosystem.** The `use` import surface with explicit
  namespacing (`use material as m` → `m.Card`, `use material.{Card}`); a
  module resolver in `byard_compiler::resolve` with package-cycle detection
  and a program-wide span `SourceMap`; strict `[dependencies]` parsing;
  `byard add`/`byard install`/`byard get` with a content-hashed `byard.lock`
  and a global `~/.byard/cache`; multi-file + `path`-dependency hot-reload; and
  package-aware LSP completions (`use <TAB>`, `m.<TAB>`, package-view params).
- Repository scaffolding: README, licenses, contributing guide, CI workflow.
- `docs/rfcs/0001-core-architecture.md` — consolidated design document covering
  the memory model, multi-pipeline renderer, spatial hit-testing grid, threading
  model, and the `byld` compiler pipeline.
- RFC template at `docs/rfcs/0000-template.md`.

---

<!-- New versions go above this line, oldest at the bottom. -->
<!-- Example entry:
## [0.1.0] - 2026-MM-DD
### Added
- First working renderer prototype.
-->
