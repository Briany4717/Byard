# Changelog

All notable changes to Byard will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Byard uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- When writing entries use these categories:
     Added / Changed / Deprecated / Removed / Fixed / Security -->

## [Unreleased]

### Changed

- **Text now wraps to its parent's width by default (RFC-0005).** A `Text` with
  no explicit `width` reflows to the width its container offers — like a block of
  text in a browser — instead of overflowing on a single line. This is done
  properly through Taffy's measure protocol: `Text` becomes a measured leaf that
  the layout atlas sizes via the shared, cached `TextMeasurer` during layout
  (`LayoutAtlas::add_text_leaf` + `compute_with_text`), so it re-wraps when its
  container resizes with no per-`Text` bookkeeping. `wrap: false` opts out to a
  single line; an explicit `width` still pins the wrap width. Previously wrapping
  required both `wrap: true` and an explicit `width`, so unbounded text overflowed
  — the catalog documented `wrap` as defaulting to `true`, but the leaf-measured
  model couldn't honour it. New engine surface: `atlas::TextLeaf`,
  `atlas::LayoutAtlas::{add_text_leaf, compute_with_text}`, and the
  `text::TextSizer` trait.

### Fixed

- **Enum keyword props can no longer be shadowed by a same-named `var`.** A
  keyword-valued prop (`snap: page`, `axis: horizontal`, `align: center`,
  `justify: …`, `direction: …`, `fit: …`, `alignment: …`, `anchor: …`) is a
  closed token set the type-checker reads as a bare identifier — but the runtime
  was resolving that identifier through the reactive environment, so a view
  declaring a `var` with the same name as the keyword silently evaluated the
  *variable* instead of the token. Most visibly, RFC-0021's `snap: page` carousel
  reflects its page through a `var page`, and `snap: page` next to `var page`
  read as the page index (`0`), disabling snapping entirely. Enum props are now
  read directly from the AST at the single resolution point, matching the
  checker: they can never be shadowed, and the read skips lowering an expression
  for a value that is always a compile-time keyword. Fixes the `scroll_snap`
  example (`cargo run -p byard-cli -- dev` in
  `crates/byard-cli/examples/scroll_snap`).
- **`DecoratedBox` inner border edge is now anti-aliased.** The rounded-rect
  SDF shader smoothed only the *outer* edge; the transition from the border to
  the (possibly transparent) interior used a hard threshold, leaving the inner
  edge jagged. Most visible on thin rings such as `RadioButton`. The inner edge
  now uses the same screen-space-derivative smoothstep as the outer edge, so
  both edges of any border are crisp at every size and DPI.

### Added

- **RFC-0021 advanced scroll behaviours — page snap, pagination, infinite
  scroll (first slice).** `ScrollView` gains the full RFC-0021 prop/event surface
  (`snap`, `snap_align`, `pull_refresh`, `refreshing`, `collapse_header`, `page`,
  `page_count`, `end_threshold`; events `end_reached`/`page_change`/`scroll_end`/
  `refresh`). Implemented in this slice: **`snap: page`** glides the offset to the
  nearest viewport-sized page with a **spring** (RFC-0010) when scrolling stops —
  on drag release *and* after wheel/trackpad scrolling goes quiet (there is no
  release event for a wheel). The settle is momentum-aware: it waits for the
  fling's shrinking deltas to actually stop before snapping, so the snap animation
  never fights an in-progress scroll, and any fresh scroll/drag cancels the glide
  so the user always takes over cleanly. It reflects the `page:` var **both ways**:
  `page` tracks the current page *continuously* as you scroll (wheel or drag,
  firing `page_change`), and setting `page` scrolls the offset to that page
  (edge-triggered so it never fights a drag). **`on_end_reached`** fires once when
  the visible bottom crosses `end_threshold` (debounced until the offset falls
  back, so appending items re-arms it) — the infinite-scroll trigger. All other
  props parse and validate. `snap: item` boundary snapping, pull-to-refresh
  (overscroll + indicator), and collapsing headers (layout-during-scroll +
  implicit `scroll_fraction`) are follow-up passes — each needs a new
  physics/layout subsystem. See `crates/byard-cli/examples/scroll_snap`.
- **RFC-0021 `snap: item` + `snap_align`.** `snap: item` settles the scroll to
  the nearest **direct-child boundary** instead of a fixed page, so a carousel of
  unequal-width cards snaps each card to the viewport edge. `snap_align` places
  the snapped item at the viewport's `start` (default), `center`, or `end`. The
  item boundaries are read from the laid-out child rects each render (offset is a
  paint-time translate, so layout positions are the natural content coordinates),
  aligned, and clamped to the scroll extent — the settle then picks the boundary
  nearest the current offset and reuses the same spring glide and momentum-aware
  quiet detection as `snap: page`. When the content is wrapped in a single
  `Row`/`Column` (the usual scroll layout), the items are that container's
  children. New engine surface: `LayoutAtlas::children`. `snap_spring` overrides
  and fling-velocity projection remain follow-ups. See
  `crates/byard-cli/examples/scroll_snap`.
- **RFC-0024 extended style states + combined selectors.** The style-state
  system (RFC-0012/0016) gains five engine-managed pseudo-states — `checked`
  (a value-widget's value is true), `selected` (the `selected:` prop, or a
  `RadioButton` whose `bind == value`), `invalid` (the `invalid:` prop),
  `indeterminate` (a `Checkbox`'s mixed prop), and `dragging` (the element being
  dragged past an 8px threshold) — plus **combined selectors**: `on focused+hover
  { … }` applies only when *all* its states are active. `selected`/`invalid` are
  universal opt-in props on any element; `checked`/`indeterminate` are mutually
  exclusive. Resolution is by specificity (a combined selector beats a
  single-state one) then declaration order. This completes RFC-0012's remaining
  states and lets `Checkbox`/`RadioButton`/`TextField` theme their states through
  `on <state>` blocks instead of duplicating the element tree with `when/else`.
  See `crates/byard-cli/examples/style_states`.
- **RFC-0018 `ZStack` intrinsic.** Overlapping children within the layout tree:
  all children occupy the same rect (painted in declaration order, last on top),
  the stack sizes to its largest child (the SwiftUI model), and
  `alignment: Align2D` (`center` default, plus the eight edge/corner tokens)
  positions children smaller than the stack. Implemented as a single-cell CSS
  grid, so it composes with the rest of the layout system; unblocks badges on
  avatars, a play button over a thumbnail, and floating action buttons over
  content. See `crates/byard-cli/examples/zstack`.
- **RFC-0018 `Grid` intrinsic.** A CSS-grid container backed by Taffy's grid
  mode. `columns`/`rows` take a template string (`"1fr 2fr 100"`,
  `"repeat(3, 1fr)"`, `auto`) parsed into engine tracks — a malformed template
  is a `CompileError::InvalidGridTemplate`. `gap`, plus per-axis `col_gap`/
  `row_gap`, space the cells. Children auto-place left-to-right, top-to-bottom by
  default, or place explicitly with the child props `col`/`row` (1-based grid
  lines) and `col_span`/`row_span`. Replaces the nested-`Row`/`Column` "wrapper
  hell" for two-dimensional layouts (dashboards, galleries, label+field forms);
  see `crates/byard-cli/examples/grid`.
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
  shows an engine-drawn checkmark when checked, an outlined box (or a muted
  filled slot) when unchecked, and a horizontal dash when indeterminate. The
  container is a `DecoratedBox`, so a style can give it a `border`/`on checked
  { border }` (RFC-0024); `bg` is the checked accent, not a background slab
  (parity with `Toggle`/`Slider`). Replaces the Box+Text approximation design
  systems used for selection controls; see `crates/byard-cli/examples/checkbox`.
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
