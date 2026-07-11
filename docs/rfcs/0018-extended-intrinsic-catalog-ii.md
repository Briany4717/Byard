# RFC-0018: Extended Intrinsic Catalog II — Checkbox, RadioButton, Grid, ZStack

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-10
- **Last updated:** 2026-07-10
- **Depends on:** RFC-0001 (§3.1 pipelines, §4.1 Taffy), RFC-0002 (D4 attribute contract, D9 types), RFC-0003 (events, reflected props, `bind:`), RFC-0005 (intrinsic catalog — this extends it), RFC-0012 (style states — `checked`/`selected` states from RFC-0024).
- **Amends:** RFC-0005 (adds four new intrinsics to the closed table).
- **Enables:** Native checkbox/radio in `byard-material` and `byard-cupertino`, CSS-grid layouts, explicit z-layering within the layout tree.

---

## Summary

Add four intrinsics to the catalog defined in RFC-0005: **`Checkbox`** (boolean
toggle with a distinct visual identity from `Toggle`), **`RadioButton`** (single-
selection within a group), **`Grid`** (CSS-grid layout via Taffy's grid support),
and **`ZStack`** (explicit z-layering of overlapping children within the layout
tree). These close the "Future possibilities" items RFC-0005 listed and unblock
native selection controls that `byard-material` currently approximates with
Box+Text workarounds.

---

## Motivation

### Checkbox and RadioButton

The byard-material package currently fakes these with `Box` + `Text("check")` —
visually passable but fundamentally broken: there is no `checked` state for the
engine to track, no accessibility semantics, no platform-native interaction
pattern (keyboard Space to toggle, arrow keys to move within a radio group), and
no `bind:` support. Every design system (Material, Cupertino, Fluent) has
checkboxes and radios as first-class controls. They are as fundamental as
`Toggle` and `Slider`.

### Grid

`Column` and `Row` (flex-direction presets of `Box`) handle one-dimensional
layouts. Two-dimensional layouts — dashboard grids, image galleries, form layouts
with label+input columns — require either deep nesting (Column of Rows, the
Flutter "wrapper hell" RFC-0001 rejects) or a proper grid. Taffy already supports
CSS-grid; the intrinsic surface is the missing piece.

### ZStack

`Column`/`Row` place children sequentially. Overlapping children (a badge on top
of an avatar, a play button centered on a thumbnail, a floating action button
above content) require z-layering *within* the layout tree — not a full overlay
escape (RFC-0017) but just "these children occupy the same space, last-child on
top." SwiftUI's `ZStack` and Flutter's `Stack` serve this role.

---

## Reference-level explanation

### 1. `Checkbox` — boolean selection with distinct visual

- **Content:** none. **Children:** none.
- **Props:** Layout + Decoration + `indeterminate: Bool` (default `false`).
- **Reflected (two-way):** `value: Bool` (or `bind:`). `true` = checked.
- **States:** RFC-0024's `checked` pseudo-state is driven by `value`.
- **Events:** `change(e: ChangeEvent<Bool>)`, all pointer + focus; **focusable by
  default**. Space key toggles.
- **Pipeline:** `DecoratedBox` (container square) + `VectorMSDF` or `TextGlyph`
  (the checkmark path). The checkmark is an engine-owned vector asset — **not** a
  design-system icon (the engine draws the check; the design system themes the
  container colors via style states).
- **Accessibility role:** `checkbox`.

Interaction model: tap or Space toggles `value` → `change` event → `bind:`
write-back (RFC-0003 E1). Identical pattern to `Toggle` but with checkbox
semantics and a square visual.

### 2. `RadioButton` — single-selection within a group

- **Content:** none. **Children:** none.
- **Props:** Layout + Decoration + `value: Str` (required — the value this button
  represents within the group).
- **Reflected (two-way):** `bind: Str` (required — binds to the group's shared
  `var`). The RadioButton is selected when `bind == value`.
- **States:** RFC-0024's `selected` pseudo-state is driven by `bind == value`.
- **Events:** `change(e: ChangeEvent<Str>)`, all pointer + focus; **focusable**.
  Arrow keys move selection within the group (accessibility pattern).
- **Pipeline:** `DecoratedBox` (outer circle) + `DecoratedBox` (inner filled
  circle when selected). Both use corner radius = 50%.
- **Accessibility role:** `radio`.

**Group-var model:** RadioButtons sharing the same `bind:` var form a mutual-
exclusion group automatically:

```byld
var choice = "home"
RadioButton(value: "home", bind: choice)   // selected when choice == "home"
RadioButton(value: "work", bind: choice)   // selected when choice == "work"
RadioButton(value: "other", bind: choice)  // selected when choice == "other"
```

When one is tapped (or selected via keyboard), the engine writes its `value`
to the bound var (`choice = "work"`). Because all RadioButtons in the group
read the same var, the previously selected one deselects reactively via
Mark-and-Pull — no explicit mutual-exclusion logic needed. This is the
standard group-var model (SwiftUI Picker, Flutter RadioListTile `groupValue`,
HTML `<input type="radio" name="group">`).

### 3. `Grid` — CSS-grid layout

- **Content:** none. **Children:** any.
- **Props:** Layout + Decoration +

| Prop | Type | Default | Meaning |
|---|---|---|---|
| `columns` | `GridTemplate` | `auto` | column track definitions |
| `rows` | `GridTemplate` | `auto` | row track definitions |
| `gap` | `Len` | `0` | gap between cells (or `row_gap`/`col_gap`) |
| `col_gap` | `Int` | from `gap` | column gap override |
| `row_gap` | `Int` | from `gap` | row gap override |

`GridTemplate` syntax: `"1fr 2fr 100"` or `"repeat(3, 1fr)"` — a string parsed
at compile time into Taffy's `GridTrackVec`. Invalid syntax is a
`CompileError::InvalidGridTemplate`.

- **Child placement props** (on children of `Grid`):

| Prop | Type | Meaning |
|---|---|---|
| `col` | `Int` or `(Int, Int)` | column start (or start..end span) |
| `row` | `Int` or `(Int, Int)` | row start (or start..end span) |
| `col_span` | `Int` | columns to span (default 1) |
| `row_span` | `Int` | rows to span (default 1) |

- **Events:** all pointer; focusable if events registered.
- **Pipeline:** `DecoratedBox` (background, same as `Box`).

Grid lowers directly to Taffy's CSS-grid mode — the layout engine already
supports it; this intrinsic exposes the surface.

### 4. `ZStack` — overlapping children

- **Content:** none. **Children:** any.
- **Props:** Layout + Decoration + `alignment: Align2D` (default `center`).

`Align2D` is a 2D alignment: `center`, `top_start`, `top_end`, `bottom_start`,
`bottom_end`, `top`, `bottom`, `start`, `end`. Controls how children that are
smaller than the stack are positioned within it.

- **Events:** all pointer; keyboard/focus if focusable.
- **Pipeline:** `DecoratedBox` (background). Children paint in **declaration
  order** (last child on top), all occupying the same rect.

`ZStack` is `Box` with `position: absolute` semantics for its children: all
children are laid out against the `ZStack`'s rect (not sequentially), and
they overlap. Hit-testing walks children in reverse declaration order (topmost
first), consistent with the visual order.

Taffy implementation: each child has `position: absolute` in the Taffy tree,
with alignment derived from the `ZStack`'s `alignment` prop (overridable per
child via `align_self`).

---

## Drawbacks

- **Four more intrinsics in the closed table.** Each one is a pipeline-aware
  lowering in `interp/intrinsics.rs`, adding to the validation surface. The
  catalog grows from 12 to 16.
- **`RadioButton` group coordination** adds intra-tick state management that
  crosses element boundaries — the first time the interpreter does this (Toggle
  and Checkbox are self-contained).
- **`Grid` template string parsing** adds a mini-parser inside the attribute
  validator. Invalid templates must produce clear diagnostics.
- **`ZStack` hit-testing** in reverse order requires the spatial grid to record
  sibling order, not just rect overlap.

---

## Rationale and alternatives

**Why not compose Checkbox/RadioButton from `Toggle`?** They have distinct
semantics (checkbox = independent boolean, radio = mutual exclusion within a
group), distinct accessibility roles, and distinct keyboard interaction patterns.
Pretending they're the same control themed differently leads to the accessibility
failures that `byard-material`'s Box+Text workaround demonstrates.

**Why `Grid` instead of nested `Row`/`Column`?** Nested flex is the "wrapper
hell" RFC-0001 §Motivation explicitly rejects. A grid with `columns: "1fr 2fr"`
replaces a `Row` containing a `Column(grow: 1)` and a `Column(grow: 2)` and
their inner content — fewer nodes, clearer intent, and proper 2D alignment.

**Why `ZStack` instead of `Box` with `position: absolute` children?** `Box`
children are flex items. Making some absolute would require a per-child
`position` prop that changes Taffy's layout mode — error-prone and ambiguous.
A dedicated `ZStack` makes the overlap intent explicit and keeps `Box`/`Column`/
`Row` purely flex-based.

---

## Prior art

- **SwiftUI:** `Toggle`, `Picker` (radio-like), `LazyVGrid`/`LazyHGrid`, `ZStack`.
- **Flutter:** `Checkbox`, `Radio`, `GridView`, `Stack` + `Positioned`.
- **Jetpack Compose:** `Checkbox`, `RadioButton`, `LazyVerticalGrid`, `Box` with
  `Modifier.align`.
- **CSS/HTML:** `<input type="checkbox">`, `<input type="radio">`, CSS Grid,
  `position: relative` + `position: absolute`.

---

## Resolved questions

- **Before merge:**
  - [x] **RadioButton value type.** **Group-var model.** A single `var` per
    group, each RadioButton carries a `value` prop, and the group is linked
    via `bind:` to the shared var:
    ```byld
    var choice = "home"
    RadioButton(value: "home", bind: choice)
    RadioButton(value: "work", bind: choice)
    ```
    The engine enforces mutual exclusion: setting `choice = "work"` deselects
    "home" automatically. This is the standard model (SwiftUI Picker, Flutter
    RadioListTile groupValue, HTML radio name). The per-button Bool model is
    rejected — it forces the developer to manually coordinate exclusion,
    which is error-prone for 3+ options.
  - [x] **Grid auto-placement.** Implicit default. Children without explicit
    `col`/`row` props fill cells left-to-right, top-to-bottom (Taffy's
    auto-placement). This matches CSS Grid's default behavior and requires
    zero configuration for the common case. Explicit `col`/`row` overrides
    are available for manual placement.
  - [x] **ZStack sizing.** Sizes to its **largest child** (SwiftUI model).
    The ZStack's intrinsic size is `(max(child.width), max(child.height))`.
    Explicit `width`/`height` props override this. Rationale: the SwiftUI
    model is intuitive — the stack is "as big as it needs to be" — and avoids
    forcing dimensions on every ZStack.

- **During implementation:**
  - [x] **Checkbox checkmark asset.** Engine-provided **SVG path** baked into
    the VectorMSDF atlas. The path is a simple polyline (two strokes forming
    a checkmark), rendered via the existing MSDF pipeline (RFC-0009). This
    gives crisp rendering at all sizes and DPI scales. `TextGlyph("✓")` is
    rejected because glyph appearance varies across fonts and is not
    pixel-consistent.
  - [x] **RadioButton arrow-key wrapping.** Yes, arrow-key focus **wraps**
    from the last radio to the first within a group. This matches WAI-ARIA
    Radio Group pattern and platform behavior (HTML radio groups, iOS/Android
    accessibility). Up/Left goes to previous, Down/Right goes to next, both
    wrap.

---

## Future possibilities

- **`Divider` intrinsic** — currently composed from `Box #[height: 1]` in
  `byard-material`, could become a semantic intrinsic with theme-aware color.
- **`Menu` / `Dialog` intrinsics** — paired with RFC-0017's overlay system,
  these could become semantic intrinsics with built-in dismiss behavior.
- **`LazyGrid`** — virtualized grid that only instantiates visible cells.
- **`RadioButton` with enum values** — `bind:` to an enum `var` instead of
  per-button `Bool`s, once the type system supports enums.
