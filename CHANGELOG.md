# Changelog

All notable changes to Byard will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Byard uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- When writing entries use these categories:
     Added / Changed / Deprecated / Removed / Fixed / Security -->

## [Unreleased]

### Fixed

- **`DecoratedBox` inner border edge is now anti-aliased.** The rounded-rect
  SDF shader smoothed only the *outer* edge; the transition from the border to
  the (possibly transparent) interior used a hard threshold, leaving the inner
  edge jagged. Most visible on thin rings such as `RadioButton`. The inner edge
  now uses the same screen-space-derivative smoothstep as the outer edge, so
  both edges of any border are crisp at every size and DPI.

### Added

- **RFC-0018 `RadioButton` intrinsic.** Single-selection within a group: each
  button carries a `value: Str` identity and a `bind: Str` to the shared group
  `var`, and is selected when `bind == value`. Tapping a button writes its
  `value` to the group var, so the previously selected sibling deselects
  reactively — automatic mutual exclusion, no explicit coordination (the
  standard group-var model). Focusable by default; arrow keys move selection
  within the group (Down/Right next, Up/Left previous, wrapping at both ends).
  Visual is an engine-owned outer ring plus an inner accent dot when selected;
  `bg` is the selected accent. Fires `change` with `bind:` write-back
  (RFC-0003 E1). See `crates/byard-cli/examples/radio_button`.
- **RFC-0018 `Checkbox` intrinsic.** A first-class boolean control with a
  distinct square identity from `Toggle`: reflected two-way `value`/`bind: Bool`
  (`true` = checked), an `indeterminate` mixed state, focusable by default
  (Space toggles), and a `change` event with `bind:` write-back (RFC-0003 E1).
  It owns its visuals — a rounded square that fills with the `bg` accent and
  shows an engine-drawn checkmark when checked, a muted filled slot when
  unchecked, and a horizontal dash when indeterminate, all borderless so the
  mark stays crisp at control sizes — so `bg` is the checked accent, not a
  background slab (parity with `Toggle`/`Slider`). Replaces the Box+Text
  approximation design systems used for selection controls; see
  `crates/byard-cli/examples/checkbox`.
- **Binary arithmetic in `byld` (`+ - * /`).** Expressions can now compute:
  `width: base * 2 + 10`, `sweep: percent * 3.6 with anim.spring()`. Standard
  precedence, left-associative, Int/Float promotion; required by RFC-0020's
  reactive shape parameters and useful everywhere a prop is derived.
- **RFC-0020 `Canvas` intrinsic & path/shape primitives.** A fixed-size
  drawing surface whose children are declarative shape commands — `arc`,
  `circle`, `line`, `rect`, `bezier`, `path(d: …)`, and `text` — rendered by
  a new analytic-SDF GPU pipeline: resolution-independent anti-aliasing,
  stroke caps (`butt`/`round`/`square`), dash patterns with an animatable
  `dash_offset`, fills (including arc sectors), and per-parameter reactivity
  (`sweep: percent * 3.6` animates with no re-tessellation, no atlas churn).
  Complex SVG `path` data rasterizes through the existing MSDF pipeline at
  icon quality. Unblocks circular progress indicators, spinners, gauges, and
  custom decorations; see `crates/byard-cli/examples/canvas_shapes`.
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
