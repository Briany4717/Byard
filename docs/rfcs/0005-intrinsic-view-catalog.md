# RFC-0005: Intrinsic View Catalog — content, properties, and events per built-in

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-06-20
- **Last updated:** 2026-06-20
- **Depends on:** RFC-0001 (§3.1 render pipelines, §4.1 Taffy layout, §4.2 grid), RFC-0002 (**D4** intrinsic attribute contract, **D5** style precedence, **D9** types/`Str`), RFC-0003 (event catalog, `:` vs `=>` attribute syntax, reflected props).
- **Implements:** RFC-0002 D4's open item "exact arg names/types per intrinsic" and RFC-0003 §4's pairing of events with reactive props.

---

## Summary

This RFC enumerates the Phase 2 **intrinsic views** — the fixed table in
`interp/intrinsics.rs` (RFC-0002) that lowers reserved element names to
`BoxInstance` / `TextLine` / texture writes on the `RenderFrame`. For each
intrinsic it fixes: the positional **content** arity (`(...)`), the typed
**properties** (`#[name: value]`), the **events** it can raise (`#[name =>
action]`), and the **reflected reactive props** that command it without a
reference (RFC-0003 §3). It also defines the shared property vocabulary, the
scalar types those props use, and the exact validation/diagnostic rules from D4.

Intrinsics are **not** user `View`s — a user `View` is pure composition that
bottoms out in these. An element name that is neither an intrinsic nor a
`ViewDecl` in scope is `CompileError::UnknownView`; an unrecognized attribute is
`CompileError::UnknownAttribute` with a Levenshtein hint (D4).

Phase 2 ships eleven intrinsics: **Box, Column, Row, Spacer, Text, Button,
TextField, Toggle, Slider, Image, ScrollView.**

---

## Motivation

RFC-0002 D4 resolved the *rules* of the attribute contract (zone separation, hard
errors, Levenshtein hints) but explicitly deferred "exact arg names/types per
intrinsic" to a follow-up. RFC-0003 §4 fixed the event catalog and noted events
"pair with" reactive props (`focused:`, `offset:`) without saying which intrinsic
carries which. Until those are enumerated, `interp/intrinsics.rs` cannot be
written, the type-inference rules (D9) have nothing concrete to infer against, and
the LSP has no completion data. This RFC closes that gap.

---

## Guide-level explanation

```byld
View ProfileCard(name: Str) {
    var liked = false

    Column #[gap: 12, p: 16, bg: 0x1E1E1E, radius: 16, width: 280] {
        Row #[gap: 8, align: center] {
            Image("avatar.png") #[width: 40, height: 40, radius: 20, fit: cover]
            Text(name) #[typo: titleMedium]
            Spacer #[grow: 1]
            Toggle #[bind: liked]
        }
        Text("{name} {liked ? \"♥ liked\" : \"\"}") #[color: 0xAAAAAA, lines: 1]
        Button("Follow") #[bg: 0x3B82F6, radius: 8, p: (8, 16)] => follow()
    }
}
```

Every value uses `:`; the one event uses `=>` (RFC-0003). `bind:` makes the
`Toggle` two-way against `liked`. No wrapper views, no refs.

---

## Reference-level explanation

### 1. Shared scalar types (D9)

| Type | Literal / form | Notes |
|---|---|---|
| `Int` | `12`, `-4` | logical pixels for lengths; `i64` in Phase 4 |
| `Float` | `0.5`, `1.0` | `f64`; opacities, slider values |
| `Bool` | `true` / `false` | |
| `Str` | `"…"` (interpolated) | scalar string (**not** `Text`, which is the view) |
| `Color` | `0xRRGGBB` or `0xAARRGGBB` | hex int; 6-digit ⇒ opaque, 8-digit ⇒ alpha-first |
| `Len` | `Int`, or `(Int, Int)`, or `(Int×4)` | scalar = all sides; pair = (vertical, horizontal); quad = (top, right, bottom, left) |
| `Typo` | a typography token (`titleLarge`, `bodyMedium`, …) | resolved from the active theme |
| `Align` | `start` `center` `end` `stretch` | cross-axis alignment |
| `Justify` | `start` `center` `end` `between` `around` `evenly` | main-axis distribution |
| `Weight` | `thin` `regular` `medium` `bold` | text weight |
| `TextAlign` | `start` `center` `end` `justify` | |
| `Fit` | `fill` `contain` `cover` `none` | image scaling |
| `Vec2` | `(Float, Float)` | scroll offsets, positions |
| `Fn(Args) -> Ret` | `Fn(ChangeEvent<Str>)` | callback-prop type (RFC-0003 E2) |

Enum-token types (`Align`, `Justify`, `Weight`, `TextAlign`, `Fit`) are written as
bare identifiers in attribute position (`align: center`) and validated against the
fixed token set; an unknown token is `CompileError::UnknownAttribute`-style with a
Levenshtein hint over the valid tokens.

### 2. Shared property vocabulary

Most intrinsics draw their props from these groups (each intrinsic's spec lists
which groups it accepts). All map to RFC-0001's pipelines/Taffy.

**Layout (Taffy, §4.1)** — accepted by containers and most elements:

| Prop | Type | Default | Meaning |
|---|---|---|---|
| `width` / `height` | `Int` | auto | fixed logical size |
| `gap` | `Int` | `0` | space between children (containers only) |
| `p` | `Len` | `0` | padding |
| `m` | `Len` | `0` | margin |
| `align` | `Align` | `stretch` | cross-axis alignment of children |
| `justify` | `Justify` | `start` | main-axis distribution of children |
| `grow` | `Int` | `0` | flex-grow factor |
| `basis` | `Int` | auto | flex basis |

**Decoration (`DecoratedBox`/`SolidBox`, §3.1)** — accepted by box-like intrinsics:

| Prop | Type | Default | Meaning |
|---|---|---|---|
| `bg` | `Color` | transparent | solid fill (`SolidBox` if no radius/shadow, else `DecoratedBox`) |
| `radius` | `Len` | `0` | corner radius (scalar = all, quad = per-corner) |
| `border` | `Border` `{ width: Int, color: Color }` | none | |
| `shadow` | `Shadow` token | none | box shadow from theme |
| `opacity` | `Float` (0–1) | `1.0` | |

**Text (`TextGlyph`, §3.1)** — `Text` and text-bearing intrinsics:

| Prop | Type | Default | Meaning |
|---|---|---|---|
| `typo` | `Typo` | theme body | typography token (sets size+weight+line-height) |
| `color` | `Color` | theme on-surface | glyph color |
| `size` | `Int` | from `typo` | overrides token size |
| `weight` | `Weight` | from `typo` | overrides token weight |
| `align` | `TextAlign` | `start` | |
| `lines` | `Int` | `0` | max lines (`0` = unbounded); overflow ellipsizes |
| `wrap` | `Bool` | `true` | |

**Universal** — accepted by every intrinsic:

| Attr | Kind | Notes |
|---|---|---|
| `style` | prop `: .class` | merges a scoped style class (D5 layer 2) |
| `focused` | reflected prop `: Bool` | only on focusable intrinsics (§4) |
| any event | `name => action` | per the element's allowed event set (§3) |

### 3. Event applicability and focusability

Pointer events (`tap`, `pointer_down/up/move`, `pointer_enter/exit`, `hover`,
`long_press`, `double_tap`, `secondary`, `wheel`) may attach to **any** intrinsic
— attaching one registers the element's rect in the §4.2 grid (with the D4-bis
`=>` syntax) and inflates it to the 44×44 hit minimum (RFC-0003 E8).

Keyboard/focus events (`key_down`, `key_up`, `focus`, `blur`) and the `focused:`
reflected prop require the element to be **focusable**. An element is focusable
iff it is an inherently-focusable intrinsic (`Button`, `TextField`, `Toggle`,
`Slider`) **or** it registers any keyboard/focus listener (which makes a plain
`Box` focusable and inserts it into the Tab ring — RFC-0003 E3). Edit events
(`change`, `input`, `submit`) and value props (`value:`, `bind:`) are only valid
on value-carrying intrinsics (`TextField`, `Toggle`, `Slider`); using them
elsewhere is `CompileError::UnknownAttribute`.

### 4. The intrinsics

Each spec lists: positional **content** (arity + type), accepted **prop groups**
plus intrinsic-specific props, **reflected** two-way props, **events**, and the
**pipeline** it lowers to.

#### `Box` — generic styleable container
- **Content:** none (arity 0).
- **Children:** any.
- **Props:** Layout + Decoration + `direction: (row|column)` (default `column`).
- **Reflected:** `focused: Bool` (if it registers focus/key events).
- **Events:** all pointer; keyboard/focus if focusable.
- **Pipeline:** `SolidBox` / `DecoratedBox`. The base of all containers.

#### `Column` — vertical stack
- **Content:** none. **Children:** any.
- **Props:** Layout + Decoration. (`Box` with `direction: column` preset; `gap`
  applies along the vertical main axis.)
- **Events:** all pointer; keyboard/focus if focusable.

#### `Row` — horizontal stack
- Identical to `Column` with `direction: row` preset; `gap` along the horizontal
  main axis.

#### `Spacer` — flexible gap
- **Content:** none. **Children:** none.
- **Props:** `grow: Int` (default `1`), `basis: Int`. No decoration, no events.
- **Pipeline:** none (layout-only Taffy node).

#### `Text` — text run
- **Content:** **arity 1**, `Str` (the text; interpolation allowed). `Text()` with
  no content or >1 positional arg is `CompileError::ArityMismatch`.
- **Children:** none.
- **Props:** Text group + `m`, `width`, `style`.
- **Events:** all pointer (a tappable label is just `Text(...) #[tap => …]`);
  focusable only if it registers focus/key events.
- **Pipeline:** `TextGlyph`.

#### `Button` — pressable
- **Content:** arity 1, `Str` (label). (A child-block form for icon+label buttons
  is deferred — see *Unresolved*.)
- **Props:** Layout + Decoration + Text (for the label).
- **Reflected:** `focused: Bool`.
- **Events:** all pointer + keyboard/focus; **focusable by default**. The
  element-tail `=>` shorthand maps to `tap` (RFC-0003).
- **Pipeline:** `DecoratedBox` (background) + `TextGlyph` (label).

#### `TextField` — single-line text input
- **Content:** none. **Children:** none.
- **Props:** Layout + Decoration + Text + `placeholder: Str`.
- **Reflected (two-way):** `value: Str` (or `bind: query` sugar → `value:` +
  `change`), `focused: Bool`, `selection: Range`.
- **Events:** `change(e: ChangeEvent<Str>)`, `input`, `submit`, `focus`, `blur`,
  `key_down`, `key_up`, all pointer; **focusable by default**.
- **Pipeline:** `DecoratedBox` + `TextGlyph` (text + caret).

#### `Toggle` — boolean switch
- **Content:** none.
- **Props:** Layout + Decoration (track/thumb colors via theme).
- **Reflected (two-way):** `value: Bool` (or `bind:`).
- **Events:** `change(e: ChangeEvent<Bool>)`, all pointer + focus; focusable by
  default.
- **Pipeline:** `DecoratedBox` ×2 (track + thumb).

#### `Slider` — numeric range
- **Content:** none.
- **Props:** Layout + Decoration + `min: Float` (default `0.0`), `max: Float`
  (default `1.0`), `step: Float` (default continuous).
- **Reflected (two-way):** `value: Float` (or `bind:`).
- **Events:** `change(e: ChangeEvent<Float>)`, all pointer + focus; focusable by
  default.
- **Pipeline:** `DecoratedBox` (track + fill + thumb).

#### `Image` — raster/texture
- **Content:** arity 1, `Str` (source path/handle). `CompileError::ArityMismatch`
  otherwise.
- **Props:** Layout + `radius`, `opacity`, `fit: Fit` (default `contain`).
- **Events:** all pointer.
- **Pipeline:** `TextureSampler`.

#### `ScrollView` — scroll container
- **Content:** none. **Children:** any (a single content subtree).
- **Props:** Layout + Decoration + `axis: (vertical|horizontal|both)` (default
  `vertical`).
- **Reflected (two-way):** `offset: Vec2` (read current scroll; set to scroll
  programmatically — RFC-0003 §3, no ref).
- **Events:** `scroll(e: ScrollEvent)`, `wheel`, all pointer.
- **Pipeline:** clips children via scissor (§3.3); content drawn at `-offset`.

### 5. Argument validation (D4, made concrete)

For every intrinsic call `Name(content...) #[attrs...] { children }`,
`interp/intrinsics.rs` validates, in order, each producing a precise span-anchored
diagnostic:

1. **Name resolution.** Not an intrinsic and not a `ViewDecl` in scope →
   `CompileError::UnknownView { name, span }` with a Levenshtein hint over
   intrinsic names + in-scope views.
2. **Content arity.** Positional count ≠ the intrinsic's declared arity →
   `CompileError::ArityMismatch { name, expected, found, span }` (e.g.
   `Text("a", "b")` or `Column("x")`).
3. **Content type.** A positional arg whose inferred type (D9) ≠ the declared
   content type → `CompileError::TypeMismatch { expected, found, span }`.
4. **Attribute name.** An attr not in the intrinsic's accepted set →
   `CompileError::UnknownAttribute { name, span, suggestion }` where `suggestion`
   is the nearest accepted attr by Levenshtein (≤ 2 and < half the typo length —
   D4). Includes using a value prop (`value:`) on a non-value intrinsic, or a
   keyboard event on a non-focusable one.
5. **Attribute kind.** A property written with `=>` or an event written with `:`
   → `CompileError::WrongAttributeSeparator { name, expected_kind, span }` ("`gap`
   is a property — use `gap: …`, not `gap => …`").
6. **Attribute type.** Value type ≠ the prop's declared type →
   `CompileError::TypeMismatch`. Enum-token props validate the token against the
   fixed set (Levenshtein hint over valid tokens).
7. **`bind:` target.** `bind:` to a non-`var` l-value →
   `CompileError::NotAssignable` (RFC-0003).
8. **Children.** Children given to a childless intrinsic (`Text`, `Spacer`,
   `Image`, `TextField`, `Toggle`, `Slider`) → `CompileError::UnexpectedChildren
   { name, span }`.

No failure is ever silent (the MAUI lesson, D4). Every diagnostic carries a
`Span` for the LSP.

### 6. Theme-resolved defaults

Props left unset fall to the intrinsic's **default typed base style** (D5 layer 1),
resolved from the active theme injected via the environment (RFC-0001 `inject`).
This is why `Text("hi")` renders with sensible color/typography and `Button("x")`
gets a default surface — the catalog defaults above are the *fallbacks* when the
theme does not override. Theme tokens (`titleLarge`, shadow tokens, on-surface
color) are resolved at mount; making them reactive is deferred with dynamic
styles (RFC-0002 D8 → Phase 3).

---

## Drawbacks

- **Fixed surface.** A closed intrinsic table means new built-ins require a code
  change, not a library. Intentional for Phase 2 (the set is small and the
  renderer pipelines are fixed at four), but it is a real extensibility limit
  until a user-defined-primitive story exists.
- **`Button` label-only.** No icon+label child form in Phase 2; composite buttons
  must wrap a `Row` in a tappable `Box`. Deferred deliberately.
- **Enum tokens as bare identifiers** (`align: center`) share the
  `UnknownAttribute` diagnostic surface with misspelled prop names; the error must
  distinguish "unknown prop" from "unknown token value" to stay clear.
- **`Len` overloading** (scalar / pair / quad) is ergonomic but adds an arity
  dimension to type-checking that the inference (D9) must handle precisely.

---

## Rationale and alternatives

**Why a closed intrinsic table instead of trait-based extensible widgets?**
RFC-0001 fixes exactly four render pipelines; every intrinsic lowers to those, so
the set is naturally bounded and a table gives the fastest dispatch and the
clearest diagnostics. Extensibility (user primitives) is a post-Phase-2 question.

**Why `(...)` = content and `#[...]` = config, repeated here?** It is the D4
contract; this RFC just assigns each intrinsic's content arity. The split is what
lets `Text("hi") #[color: …]` read unambiguously and keeps spatial/decoration
props out of the content slot.

**Why reflected props (`value:`, `offset:`, `focused:`) instead of methods?**
RFC-0003 §3: no widget references. The catalog therefore exposes every imperative
capability as a two-way prop bound to a `var`.

**Why `Column`/`Row` as presets of `Box` rather than distinct types?** They share
all layout/decoration props and differ only in `direction`; modeling them as
presets keeps the intrinsic table small and the lowering uniform.

---

## Unresolved questions

- **Before merge:**
  - [ ] **`Button` composite content.** Confirm label-only for Phase 2 vs. a
    child-block form (`Button { Icon(...); Text(...) }`); if deferred, document the
    `Box`+`Row`+`tap` workaround as canonical.
  - [ ] **`Len` quad order.** Ratify `(top, right, bottom, left)` (CSS order) vs.
    `(top, bottom, left, right)`; pick the one matching Taffy's API to avoid a
    translation layer.
  - [ ] **Theme token namespace.** Are typography tokens bare (`titleLarge`) or
    namespaced (`m3.titleLarge` as in RFC-0001's example)? Pick one and apply
    across the catalog.
  - [ ] **`color` vs `bg` on `Text`.** Confirm `Text` uses `color` for glyphs and
    has no `bg` (a background behind text = wrap in a `Box`), to keep `Text`
    glyph-only.
- **During implementation:**
  - [ ] **Default `gap`/`p` from theme** vs. hard `0` — decide whether spacing
    defaults come from the theme or are always explicit.
  - [ ] **`ScrollView` nested-scroll** and how `offset` reflects during an active
    drag (ties into RFC-0003 gesture deferral).
  - [ ] **`Image` source type** — is `Str` a path, an asset handle, or a
    controller-provided texture id? Affects the `#[byard_controller]` boundary.

---

## Future possibilities

- **More intrinsics:** `Grid` (CSS-grid via Taffy), `Stack`/`ZStack` (explicit
  Z-layering), `Divider`, `Icon`, `Checkbox`, `RadioGroup`, `Menu`, `Dialog`
  (paired with `when`).
- **Composite `Button`** and other child-bearing interactive intrinsics.
- **User-defined primitives** that register new pipeline lowerings, once the
  closed-table constraint becomes limiting.
- **Reactive theme tokens** (RFC-0002 D8 / Phase 3) so `typo`/`color`/`shadow`
  can depend on a `var` (e.g. live theme switching).
- **Responsive props** (breakpoint-aware values) layered over the same attribute
  syntax.
