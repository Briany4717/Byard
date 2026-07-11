# RFC-0024: Extended Style States — `checked`, `selected`, `invalid`, `indeterminate`, and combined selectors

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-10
- **Last updated:** 2026-07-10
- **Depends on:** RFC-0003 (E3 focus, event catalog), RFC-0005 (intrinsic catalog — value-widget states), RFC-0012 (style states system — `hover`/`pressed`/`focused`/`disabled` already landed; this extends it), RFC-0016 (style system — `on <state> {}` blocks), RFC-0018 (Checkbox, RadioButton — new intrinsics that source these states).
- **Completes:** RFC-0012's "Remaining" items: `checked`/`selected`/`invalid` value-widget states and combined selectors.

---

## Summary

Extend the style-state system (RFC-0012, RFC-0016) with five new engine-managed
pseudo-states and **combined selectors**:

| State | Sourced by | Meaning |
|---|---|---|
| `checked` | `Checkbox`, `Toggle` | value is true |
| `selected` | `RadioButton`, tabs, nav items | this item is the active selection |
| `invalid` | `TextField` (validation), form fields | value fails validation |
| `indeterminate` | `Checkbox` | neither checked nor unchecked (tri-state) |
| `dragging` | any element during a drag gesture | element is being dragged |

Combined selectors (`on hover+pressed { ... }`) allow styles that activate only
when multiple states are simultaneously true.

---

## Motivation

RFC-0012 landed four states (`hover`, `pressed`, `focused`, `disabled`) and
noted the rest as "remaining." The byard-material gap analysis confirmed the
impact: without `checked`, the `Checkbox` workaround can't theme the checkmark
container differently when checked vs. unchecked using engine-managed states —
it resorts to `when checked { ... } else { ... }` which duplicates the entire
element tree. Without `invalid`, a `TextField` can't show an error border
reactively. Without combined selectors, expressing "hovered AND focused" requires
nesting `when` blocks.

These states complete the interactive-style vocabulary needed for production
component libraries. Material and Cupertino both rely on them:

- **Checkbox checked/indeterminate** — container fill, checkmark visibility.
- **RadioButton selected** — inner dot visibility, ring color.
- **TextField invalid** — border and label color change, error text visibility.
- **Tab/NavItem selected** — active indicator, color weight.
- **Combined states** — a focused+hovered input has a different outline than
  focused-only or hovered-only.

---

## Guide-level explanation

### Value-widget states

```byld
let checkbox_style = style {
    bg: 0xFEF7FF, border: 0x79747E, radius: 2
    on checked  { bg: 0x6750A4, border: 0x6750A4 }
    on hover    { border: 0x6750A4 }
    on disabled { opacity: 0.38 }
    on indeterminate { bg: 0x6750A4, border: 0x6750A4 }
}

Checkbox #[..checkbox_style, bind: isAccepted]
```

When `isAccepted` is `true`, the `checked` state activates and the checkbox
gets a purple fill. When the user hovers, `hover` also activates. The states
compose via the existing precedence rules (RFC-0016): later `on` blocks override
earlier ones for conflicting properties.

### Validation and `invalid`

```byld
View EmailField() {
    var email = ""
    let is_invalid = !email.contains("@") && email.len() > 0

    TextField #[bind: email, placeholder: "email@example.com",
                invalid: is_invalid] {
        style {
            border: 0x79747E
            on invalid { border: 0xB3261E, color: 0xB3261E }
            on focused { border: 0x6750A4 }
            on focused+invalid { border: 0xB3261E }
        }
    }
}
```

`invalid: Bool` is a prop on `TextField` (and potentially other form fields)
that activates the `invalid` pseudo-state. It's set by the developer, not by
the engine — validation logic is app-specific.

### Combined selectors

```byld
let input_style = style {
    border: 0x79747E
    on hover           { border: 0x1D1B20 }
    on focused         { border: 0x6750A4, border_width: 2 }
    on focused+hover   { border: 0x6750A4, shadow: "sm" }
    on disabled        { opacity: 0.38, border: 0xCAC4D0 }
    on invalid+focused { border: 0xB3261E, border_width: 2 }
}
```

`on focused+hover` activates only when **both** states are true. Combined
selectors use `+` as the conjunction operator, parsed in `style {}` blocks.

Precedence: combined selectors with more states have higher specificity than
single-state selectors. Among selectors with equal state count, later declaration
wins (consistent with RFC-0016's override model).

### The `selected` state

For elements that participate in a selection group (tabs, navigation items,
segmented buttons):

```byld
View NavTab(label: Str, active: Bool = false) {
    let s = style {
        color: 0x49454F
        on selected { color: 0x6750A4 }
        on hover    { bg: 0xF3EDF7 }
    }
    Button(label) #[..s, selected: active]
}
```

`selected: Bool` is a universal prop (available on any intrinsic). Setting it to
true activates the `selected` pseudo-state on that element. Unlike `checked`
(which is a value-widget state driven by `bind:`), `selected` is caller-driven —
the developer passes the selection state explicitly.

---

## Reference-level explanation

### 1. State mask extension

RFC-0012 defined a `StyleState` bitmask:

```rust
bitflags! {
    struct StyleState: u16 {
        const HOVER    = 0b0000_0001;
        const PRESSED  = 0b0000_0010;
        const FOCUSED  = 0b0000_0100;
        const DISABLED = 0b0000_1000;
        // --- new ---
        const CHECKED       = 0b0001_0000;
        const SELECTED      = 0b0010_0000;
        const INVALID       = 0b0100_0000;
        const INDETERMINATE = 0b1000_0000;
        const DRAGGING      = 0b1_0000_0000;
    }
}
```

The mask fits in a `u16` with room for future states. Each rendered element
carries a `StyleState` that the engine updates per tick based on:

| State | Source |
|---|---|
| `hover` | pointer position inside element rect |
| `pressed` | pointer_down active on this element |
| `focused` | focus system (RFC-0003 E3) |
| `disabled` | `disabled: true` prop |
| `checked` | value-widget `value: Bool` is `true` |
| `selected` | `selected: true` prop |
| `invalid` | `invalid: true` prop |
| `indeterminate` | `indeterminate: true` prop (Checkbox only) |
| `dragging` | active drag gesture on this element |

### 2. Combined selector resolution

A style block like `on focused+hover { ... }` is parsed as a combined selector
with mask `FOCUSED | HOVER`. At style resolution time:

1. Compute the element's current `StyleState` mask.
2. For each `on` block, check if its required mask is a **subset** of the
   current mask (all required states are active).
3. Apply matching blocks in order of **specificity** (number of states in the
   selector), then **declaration order** for equal specificity.

```rust
struct ConditionalStyle {
    required: StyleState,  // mask that must be fully active
    properties: Vec<(StyleProperty, Value)>,
}

fn specificity(s: &ConditionalStyle) -> usize {
    s.required.bits().count_ones() as usize
}
```

### 3. New props on intrinsics

| Intrinsic | New prop | Type | Activates |
|---|---|---|---|
| `Checkbox` | (none — `value: Bool` drives `checked`) | — | `checked` |
| `Checkbox` | `indeterminate: Bool` | `Bool` | `indeterminate` |
| `Toggle` | (none — `value: Bool` drives `checked`) | — | `checked` |
| `RadioButton` | (none — `value: Bool` drives `selected`) | — | `selected` |
| `TextField` | `invalid: Bool` | `Bool` | `invalid` |
| *any* | `selected: Bool` | `Bool` | `selected` |
| *any* | `invalid: Bool` | `Bool` | `invalid` |

`selected` and `invalid` are **universal props** — any element can opt into
them. This supports custom selection patterns (nav items, tabs, chips).

### 4. Grammar extension

The `on` keyword in `style {}` blocks accepts combined selectors:

```
on_block := "on" state_list "{" property* "}"
state_list := state ("+" state)*
state := "hover" | "pressed" | "focused" | "disabled"
       | "checked" | "selected" | "invalid" | "indeterminate" | "dragging"
```

An unknown state name after `on` is `CompileError::UnknownStyleState` with a
Levenshtein hint.

---

## Drawbacks

- **Nine states.** The combinatorial space of combined selectors is large. Two-
  state combinations alone give 36 pairs. Developers need guidance on which
  combinations are meaningful. The compiler could warn on nonsensical combinations
  (e.g., `checked+indeterminate` — mutually exclusive on Checkbox).
- **`selected` on any element** is permissive. It's a manual state, not engine-
  driven — the developer is responsible for setting it correctly. Misuse (setting
  `selected: true` on unrelated elements) won't cause errors but will confuse
  styling.
- **`indeterminate` is Checkbox-only.** Adding it as a universal state would be
  misleading; keeping it Checkbox-only means the state enum has a heterogeneous
  scope.

---

## Rationale and alternatives

**Why engine-managed states, not `var`-driven `when` blocks?** RFC-0012 already
answered this: engine states are bounded, synthesized, and resolve at style-
layer-4 precedence. `var`-driven states reopen the general D8 dynamic-style
problem. `checked`/`selected`/`invalid` are bounded like `hover`/`pressed` —
they come from intrinsic state or explicit props, not arbitrary computation.

**Why `+` for combined selectors, not nesting?** Nesting (`on hover { on pressed
{ ... } }`) would work but is verbose and ambiguous about specificity. `+` is
concise, mirrors CSS compound selectors (`:hover:focus`), and the specificity
rule (more states = higher precedence) is intuitive.

**Why `invalid` as a prop, not computed from a `validate` function?** Validation
logic varies wildly (regex, async server check, cross-field dependencies).
Making it a simple `Bool` prop keeps the state system bounded and delegates
complexity to the developer. A future `validate:` prop could provide built-in
patterns (`validate: email`, `validate: /regex/`).

---

## Prior art

- **CSS `:checked`, `:invalid`, `:indeterminate`:** direct naming precedent.
- **Material Design state layers:** checked, selected, error states with
  distinct color overlays.
- **SwiftUI:** no pseudo-states; selection is managed via `@State` bindings.
  Byard's approach is more ergonomic for theming.
- **Jetpack Compose `InteractionSource`:** tracks pressed, hovered, focused,
  dragged. Similar bounded-state model.

---

## Unresolved questions

- **Before merge:**
  - [ ] **Mutual exclusion.** Should `checked` and `indeterminate` be mutually
    exclusive (setting one clears the other)? Recommendation: yes.
  - [ ] **`dragging` scope.** Should `dragging` activate during any pointer-move
    after pointer-down, or only after a drag threshold? Recommendation: after
    threshold (consistent with platform conventions).
  - [ ] **Negation.** Should `on !hover { ... }` (not-hovered) be supported?
    Recommendation: deferred — it's rarely needed and adds parser complexity.

- **During implementation:**
  - [ ] **Animation on state change.** When `checked` transitions, do properties
    in `on checked { ... }` animate? Recommendation: yes, through RFC-0010's
    existing `with` on each property.

---

## Future possibilities

- **Custom states** — developer-defined state names beyond the built-in set
  (e.g., `on loading { ... }`). Would require a state registry per View.
- **State-based transitions** — `on checked(enter) { ... }` / `on checked(exit)
  { ... }` for mount/unmount animations.
- **Form validation integration** — `validate: Fn(Str) -> Bool` prop on
  `TextField` that auto-drives `invalid`.
- **Negation and disjunction** — `on !disabled { ... }`, `on hover|focused
  { ... }`.
