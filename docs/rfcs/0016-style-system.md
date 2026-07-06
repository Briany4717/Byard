# RFC-0016: Style System — styles as first-class values (Hybrid D+B+C)

- **Status:** Draft — design proposal (approach chosen 2026-07-01: **Hybrid D+B+C**)
- **Author(s):** Brian (byard_v2)
- **Created:** 2026-07-01
- **Last updated:** 2026-07-01
- **Depends on:** RFC-0002 (D5 resolution, D8 static styles, Pratt parser, `inject`), RFC-0005 (§6 theme tokens / M22), RFC-0010 (`with` animation), RFC-0011 (transform), RFC-0012 (engine state source + event exposure).
- **Supersedes:** RFC-0012's **§B declarative pseudo-state layer** (this RFC owns the declarative style/state surface). RFC-0012's **§A event exposure** and the engine `StyleState` mask **remain** and are consumed here.
- **Relation to the existing `style { .class }`:** kept as desugaring sugar (§7 Migration) — no existing `.byd` breaks.

---

## Summary

Byard styles become **first-class, typed, composable values** — not a cascade, not
a global stylesheet, not string class names. A `Style` is created with `style { }`,
named with `let`, composed with `merge` / the `..` spread, parameterized with
**variants/recipes** (named axes like `size`/`tone`), applied inline as **colocated
semantic modifiers**, carries **interaction states** via `on <state> { }`, and
animates any property with `with` (RFC-0010). Everything is **static and type-safe**:
an invalid token, variant, property, or state is a compile error, and the whole
thing resolves at lower time to a flat attribute set with **zero runtime cost**.

This is the "reimagine CSS, don't copy it" answer: no cascade to reason about, no
specificity wars, no global leakage — styles are ordinary language values you can
name, pass, compose, and analyze, layered with the professional variant ergonomics
of Panda/`cva` and the beloved colocation of Tailwind/SwiftUI, all type-checked.

## Motivation

The audit and RFC-0016-exploration established: Byard already has scoped, non-
cascading classes (`style { .class #[...] }`, RFC-0002 D5) but **no tokens, no
variants, no composition, no states**. The 2024–2025 DX evidence points three ways
at once — devs love *colocation* (Tailwind), still want *named/scoped* styles (CSS
Modules), and the modern professional baseline is *type-safe tokens + variants*
(Panda `cva`, StyleX, vanilla-extract). SwiftUI notably lacks centralized theming —
a gap Byard can win. No single existing model gives all of this cleanly.

The unifying insight: if a **style is a value**, then colocation, naming,
composition, and variants all fall out of ordinary language mechanisms rather than a
bespoke cascade. That is both simpler (one model) and strictly more powerful than
CSS (fully analyzable, no global state).

## Guide-level explanation

### 1. A style is a value

```byld
let btn = style {
    bg: surface, color: onSurface, radius: md,
    scale: 1.0 with anim.spring(),        // animatable (RFC-0010)
    on hover   { bg: surfaceHover }
    on pressed { scale: 0.97 }
    on focused { border: (2, accent) }
    on disabled{ opacity: 0.4 }
}

Button("Guardar") #[ ..btn ]              // spread the style value onto the element
```

No cascade, no global. `btn` is a value with a type (`Style`). `on <state> { }`
blocks describe how the style changes in the four engine-owned states (RFC-0012
provides the state truth; this RFC provides the declaration).

### 2. Composition by value

```byld
let danger  = btn merge style { bg: error, color: onError }   // explicit merge
let bigCard = card merge elevated                              // compose two styles

Card #[ ..base, ..elevated, radius: lg ]   // spread several; inline `radius` wins
```

`merge` and `..` are the *only* composition mechanism — later wins, deterministically.
There is no specificity to memorize; the order you write is the order that applies.

### 3. Variants / recipes (the professional, type-safe layer)

For design-system components, declare named **axes** with a default combination:

```byld
style Button {                    // a named recipe → produces a parameterized Style
    base #[ radius: md, weight: 600, scale: 1.0 with anim.spring() ]
    size {
        sm #[ p: (6, 10),  typo: labelSmall ]
        md #[ p: (8, 14),  typo: labelMedium ]
        lg #[ p: (12, 20), typo: labelLarge ]
    }
    tone {
        neutral #[ bg: surface, color: onSurface ]
        primary #[ bg: accent,  color: onAccent ]
        danger  #[ bg: error,   color: onError ]
    }
    on hover    { elevation: 2 }
    on pressed  { scale: 0.97 }
    on disabled { opacity: 0.4 }
    default #[ size: md, tone: neutral ]
}

Button("Borrar") #[ variant: (size: lg, tone: danger) ]
Button("OK")                       // uses defaults (md, neutral)
```

Asking for a variant that doesn't exist (`tone: teal`) is a **compile error** with a
Levenshtein suggestion (D4-style). Variants are just a parameterized way to build a
`Style` — under the hood it is still value composition.

### 4. Colocated semantic modifiers (the terse, prototyping layer)

For one-off elements, compose inline from **semantic tokens** — no `style {}` block,
maximum colocation, but type-safe (not Tailwind's untyped "class soup"):

```byld
Button("Guardar") #[
    .surface, .round(md), .pad(md), .tone(primary),
    on hover   { bg: surfaceHover },
    on pressed { scale: 0.97 },
]
```

`.surface`/`.round(md)`/`.tone(primary)` are **token modifiers** resolved against the
theme (RFC-0005 §6). They produce an anonymous `Style` merged inline. This is the
same `Style` value model, just written at the terse end — the exact "verbose vs
inferred" spectrum you asked for, at the style-system level.

### 5. Three ceremonies, one model

Named value (`let btn = style{}`), recipe (`style Button {}`), and inline modifiers
(`#[.surface, ...]`) are **three surfaces over the same `Style` value**. Pick the
ceremony that fits the case; they compose freely (an inline element can spread a
named style *and* add modifiers *and* override inline).

## Reference-level explanation

### The `Style` type & data model

```rust
// interp/style.rs (evolved)
pub struct Style {
    base:   AttrSet,                       // resolved base properties
    states: [Option<AttrSet>; 4],          // hover, pressed, focused, disabled (RFC-0012 order)
    // recipes carry their axes until a variant selection collapses them:
    axes:   Option<Recipe>,                // Some(..) only for `style Name { }` recipes
}
```

- `AttrSet` is a small, interned, order-preserving map (last-write-wins), arena-
  allocated with the view, `!Send` (RFC-0001 discipline). A resolved `Style` holds
  no `var` reads (still static, D8) except through inline `with`/ternary at the use
  site, which is the element's concern, not the style's.
- `Style` values are **immutable**; `merge` produces a new `AttrSet` by
  last-write-wins overlay. All of this happens at lower time — the emitted element
  carries a flat resolved attr set, so there is **zero runtime style cost**.

### Grammar (RFC-0002 additions)

```
style_value  := "style" "{" style_body "}"
style_body   := ( attr | state_block )*
state_block  := "on" state_name "{" attr* "}"
state_name   := "hover" | "pressed" | "focused" | "disabled"

recipe_decl  := "style" IDENT "{" recipe_body "}"
recipe_body  := ("base" attr_block)? axis_group* state_block* ("default" attr_block)?
axis_group   := IDENT "{" (IDENT attr_block)+ "}"

spread       := ".." expr            // in an attribute list: spread a Style value
merge_expr   := expr "merge" expr    // infix, left-assoc, low precedence
token_mod    := "." IDENT ( "(" arg_list ")" )?   // .surface / .round(md) / .tone(primary)
```

- `on` and `merge` become contextual keywords (like `with` in RFC-0010). `merge`
  registers in the Pratt parser as a low-precedence infix (`left: 5, right: 6`,
  above ternary `4` so `a merge b ? …` needs parens — intentional).
- `..expr` spread is parsed in the attribute list (`register_event_attrs` sibling)
  and requires an `expr` of type `Style` (checked).
- `token_mod` is the **single-dot** `.name` form (a new `Expr::TokenMod`); the
  **double-dot** `..name` is the `Style`-value spread (decision **M2**). The two
  sigils are distinct, so there is no symbol-table disambiguation: `.` = token,
  `..` = whole style. Legacy `.class` references desugar to `..class` (§7).

### Resolution order (evolves D5 into value composition)

At lower time, an element's effective attrs resolve last-wins in this order:

1. **theme defaults** (RFC-0005 §6 / M22, D5 layer 1),
2. **spread/merged styles** in written order (`..a, ..b` → b overrides a),
3. **active interaction-state blocks**, in fixed priority
   `disabled > pressed > focused > hover` (RFC-0012 **S3**),
4. **inline properties** (`#[ radius: lg ]`) — always win.

This replaces D5's "classes" middle layer with "composed Style values," preserving
the last-wins mental model and the "inline wins" invariant. There is no cascade and
no specificity: **written order + this fixed 4-tier stack is the whole rule.**

### Interaction states (consumes RFC-0012 engine truth)

The `on <state> { }` blocks bind to RFC-0012's engine `StyleState` mask
(`router.style_state(elem)`), which is the *only* dynamic input (engine-owned
booleans, read on the logic thread) — so this does **not** reopen D8's general
dynamic-style deferral, exactly as RFC-0012 argued. When a state bit flips,
`render()` re-resolves only elements that *have* state blocks (RFC-0012 **S4**
bitset). If a base property carries `with anim.*`, the state change updates that
property's `Motion.to` and the transition animates for free (RFC-0010).

`disabled` also gates event dispatch (RFC-0012 **S5**).

### Type-safety & diagnostics

- Unknown property → `UnknownAttribute` + Levenshtein (D4).
- Unknown token/variant/axis/state → `UnknownStyleToken` / `UnknownVariant` /
  `UnknownStyleState` with suggestions.
- `..x` where `x` is not a `Style` → `NotAStyle`.
- Reading a `var` inside a `style { }` value (not inline) → still
  `DynamicStyleForbidden` (D8). States are the only sanctioned dynamism.
- Requesting an axis value the recipe doesn't declare → `UnknownVariant`.

### Ownership, threading, cost

Styles are interned, arena-scoped, immutable, `!Send`; composition is a lower-time
overlay producing a flat `AttrSet`; the emitted element carries only resolved values
+ any `Motion` PODs. Nothing style-related crosses the frame boundary except the
already-defined primitive/`Transform`/`Motion` PODs (RFC-0011/0010). Zero runtime
cascade, zero allocation on the hot path.

## Migration & backward compatibility (§7)

The existing `style { .class #[...] }` block and `#[style: .class]` reference **keep
working** by desugaring: a `style { }` block of `.class` rules lowers to a set of
named `Style` values in the view scope, and `#[style: .btn]` desugars to `#[..btn]`.
So no existing `.byd` breaks; the new surface is additive. A lint can suggest
migrating `.class` references to `..style` spreads over time.

## Drawbacks

- Biggest design surface of the cluster: a real `Style` type, `merge`/spread,
  recipes, token modifiers, and state blocks — more to build, test, document.
- Two ways to name styles (`let` value vs `style Name` recipe) — must document when
  to use which (value for ad-hoc/composition, recipe for design-system axes).
- Token modifiers vs old `.class` references need careful disambiguation (§Grammar).
- `merge` precedence with ternary requires parens in edge cases (intentional, but a
  papercut).

## Rationale and alternatives

- **Styles as values vs cascade.** A value model is simpler to reason about (order
  + 4 tiers, no specificity) and strictly more analyzable/composable than CSS. It is
  the differentiator that makes Byard's styling "better, not a copy."
- **Hybrid vs a single surface.** Named values (D) give power; recipes (B) give
  design-system ergonomics; inline modifiers (C) give prototyping speed. One model,
  three ceremonies — mirrors the verbose/inferred duality chosen for transforms.
- **Why not just A (evolved classes).** A is "CSS slightly better" and misses the DX
  leap that is a Byard pillar.
- **Type-safe tokens vs Tailwind's atomic strings.** Semantic, checked tokens keep
  colocation's speed without the untyped class-soup downside.
- **Not doing it:** styling stays class-only with no tokens/variants/states — below
  the modern professional baseline and far below Byard's DX ambition.

## Prior art

Panda CSS recipes / `cva` (variants, tokens, type-safe), StyleX & vanilla-extract
(zero-runtime, type-safe styles-as-objects), Tailwind (colocation DX, AI-friendly),
CSS Modules (scoped named styles), SwiftUI modifiers + `ViewStyle` (and its missing
central theming, which Byard's token layer fixes), Jetpack Compose Material3 theming.

## Resolved decisions (2026-07-01)

- **Approach:** Hybrid **D (styles as values) + B (recipes) + C (inline modifiers)**,
  states via `on <state> { }`.
- **State syntax:** `on hover { }` (prose form), consistent with `with` and the
  declarative philosophy; chosen over `:hover` suffix, nested `:hover {}`, and
  `.on(hover){}`.
- **State set / priority / dispatch gating:** inherit RFC-0012 **S1/S3/S5**.
- **Composition:** `merge` + `..` spread only; last-wins; inline overrides.
- **Backward compat:** `style { .class }` and `#[style: .class]` desugar (§7).
- **M1 — recipe naming:** a recipe is a **free name** applied via
  `#[variant: (…)]`, **not** bound to an intrinsic name. Rationale: keeps recipes
  reusable/composable and decouples the style system from the intrinsic catalog; an
  intrinsic-bound default (all `Button`s adopt a `Button` recipe) is deferred sugar,
  not load-bearing.
- **M2 — token vs style disambiguation → distinct sigils.** Single-dot **`.token`**
  = semantic token modifier (resolved against the theme); double-dot **`..style`** =
  spread a `Style` *value*. Legacy `.class` references desugar to `..class`.
  Rationale: eliminates any symbol-table collision, is visually unambiguous
  (`.` = token, `..` = whole style), and matches the §7 migration.
- **M3 — composition operator → keyword `merge`.** Infix `merge` for building named
  composed styles; `..` spread stays for inline. Rejected `+`/`&`: reads worse and
  overloads arithmetic with style semantics. `merge` is prose, consistent with
  `with`/`on`. A recipe may be `merge`d (it collapses to a plain `Style` first).

## Unresolved questions (deferred to implementation)

- [ ] `AttrSet` inline capacity before spilling (mirror IMPL-01's `SmallVec<[_;4]>`).
- [ ] Lint/codemod to auto-migrate `.class` → `..style` spreads.
- [ ] Exact theme-token vocabulary shared with RFC-0005 §6 / M22.

## Future possibilities

- Responsive/adaptive variants (axis keyed on viewport/platform).
- Theme switching at runtime via `inject`ed theme (dark/light) with animated token
  transitions (RFC-0010).
- `checked`/`selected`/`invalid` value-widget states (RFC-0012 Phase 2).
- A visual style inspector in the dev overlay (RFC-0013) showing the 4-tier resolve.
