# Changelog

All notable changes to Byard will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Byard uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- When writing entries use these categories:
     Added / Changed / Deprecated / Removed / Fixed / Security -->

## [Unreleased]

### Added

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
