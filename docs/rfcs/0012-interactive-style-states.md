# RFC-0012: Interactive Style States (`:hover`, `:pressed`, `:focused`, `:disabled`) & Full Event Exposure

- **Status:** Active — partially implemented (M32 full event exposure + wheel origin, M37 `StyleState` mask + disabled gate landed). All design decisions (S1–S5) and formerly-unresolved questions resolved. IMPL-76–80 logged. Remaining: `checked`/`selected`/`invalid` value-widget states, combined selectors.
- **Author(s):** Brian (byard_v2)
- **Created:** 2026-07-01
- **Last updated:** 2026-07-01
- **Depends on:** RFC-0002 (D5 three-layer style resolution, D8 dynamic-style deferral — *this RFC lifts a bounded slice of D8*), RFC-0003 (E3 focus, §4 event catalog, §7 no bubbling), RFC-0010 (animation), RFC-0011 (transform/opacity).
- **Supersedes (partially):** RFC-0002 **D8** for the specific case of *engine-managed interaction states* (not arbitrary `var`-driven styles, which stay deferred).

---

## Summary

Give developers the CSS-grade ergonomics they expect — *"this button turns
`accent` on hover and lifts 2px on press"* — as a first-class declarative feature,
without holding a widget reference and without hand-writing state `var`s and
`pointer_enter`/`pointer_exit` handlers per element. Two coordinated changes:

1. **Interaction pseudo-states in `style {}`**: `.btn:hover`, `.btn:pressed`,
   `.btn:focused`, `.btn:disabled` blocks, resolved as a *fourth* precedence layer
   above classes, and animatable through `with` (RFC-0010).
2. **Full event exposure**: wire the six events the engine already models and
   synthesizes but never exposed to the `byld` surface (`hover`,
   `pointer_enter`, `pointer_exit`, `long_press`, `double_tap`, `secondary`),
   plus optional `focus`/`blur` sugar and a `click`→`tap` alias.

An audit of the current event/style plumbing established that the router and
`byard_core` already recognize all these gestures with passing tests;
the only gap is the six-line `match` in `register_event_attrs` and the missing
declarative state layer. This RFC closes both.

## Motivation

**The DX gap.** To make a button change color on hover *today*, a dev must write:

```byld
var hovered = false
Button("Save") #[
    bg: hovered ? accent : surface,
    pointer_enter => hovered = true,   // ← and this event isn't even exposed yet
    pointer_exit  => hovered = false,
]
```

Three lines of boilerplate state per interactive element, multiplied across every
button, card and row — and it doesn't even compile because `pointer_enter`/
`pointer_exit` fall through `register_event_attrs`' `_ => continue`. A component
with hover + press + focus + disabled needs four `var`s and up to eight handlers.
That is the opposite of "irresistible DX."

**The engine is ready.** RFC-0003's router already tracks press (`down`), hover
(`hovered`), and focus (`focused`) state and fires the corresponding events. We
are not building new machinery — we are exposing and *sugaring* what exists.

**Why lift a slice of D8.** D8 deferred *arbitrary* dynamic styles (reading any
`var` inside `style {}`) to Phase 3, for good reasons (unbounded reactivity in the
style layer). Interaction states are a **bounded, engine-owned** subset: the state
booleans are synthesized by the router, not user `var`s, so they don't reopen the
general dynamic-style problem. This RFC lifts exactly that slice and no more.

## Guide-level explanation

### Pseudo-states in `style {}`

```byld
style {
    .btn #[ bg: surface, color: onSurface, radius: 8, scale: 1.0 ]
    .btn:hover   #[ bg: surfaceHover ]
    .btn:pressed #[ scale: 0.97 ]
    .btn:focused #[ border: (2, accent) ]
    .btn:disabled#[ opacity: 0.4 ]
}

Button("Save") #[ style: .btn ]                 // gets all four states for free
Button("Delete") #[ style: .btn, disabled: true ]
```

No state `var`s, no handlers. The engine flips the states from real input and
resolves the winning attributes each frame. Combine with `with` for motion:

```byld
style {
    .btn         #[ scale: 1.0   with anim.spring() ]
    .btn:hover   #[ scale: 1.03 ]
    .btn:pressed #[ scale: 0.97 ]
}
```

The `with` on the base property means *every* transition into/out of a state
animates — the spring is declared once, the states just change the target.

### Precedence (extends D5)

Resolution becomes **four** layers, last-wins:

1. framework/theme default base style (D5 layer 1),
2. referenced classes in element order (D5 layer 2),
3. **active interaction states** (new layer 3), in a fixed priority
   `disabled > pressed > focused > hover` when several are active,
4. inline `#[...]` attributes (D5 layer 3 → now layer 4, still wins).

Inline still overrides everything, preserving the D5 mental model.

### Newly exposed events

All six become writable inline handlers, same syntax as the existing ten:

```byld
Row #[
    hover         => tooltip = true,
    pointer_enter => glow = true,
    pointer_exit  => glow = false,
    long_press    => showContextMenu(),
    double_tap    => zoomIn(),
    secondary     => showContextMenu(),   // right-click
]
```

Plus sugar: `focus =>` / `blur =>` (fire when the element's focus state flips) and
`click` as an alias of `tap`.

## Reference-level explanation

### Part A — Event exposure (small, mechanical)

Extend `interp/eval.rs::register_event_attrs` (currently the 10-arm `match` at
~line 1599) with the six modeled kinds and the aliases:

```rust
"hover"         => EventKind::Hover,
"pointer_enter" => EventKind::PointerEnter,
"pointer_exit"  => EventKind::PointerExit,
"long_press"    => EventKind::LongPress,
"double_tap"    => EventKind::DoubleTap,
"secondary"     => EventKind::Secondary,
"click"         => EventKind::Tap,          // alias
// focus/blur handled below (not raw EventKinds)
```

`focus`/`blur` are **not** raw events (RFC-0003 has no `Focus` kind — focus is
state E3). Sugar them: when a handler names `focus`/`blur`, register a focusable
(as `focused:` already does) and attach the action to the *rising*/*falling* edge
of `focused_sig`. This keeps the E3 model intact.

**Also (loose end from the audit):** connect `WindowEvent::MouseWheel` in
`byard-platform/src/lib.rs` so `scroll`/`wheel` reach the router from hardware,
not just from tests.

The §5 attribute checker must add these names to each intrinsic's accepted event
set so they don't report `UnknownAttribute`.

### Part B — Interaction states (the new layer)

**State source.** The router already owns the truth: `is_pointer_pressed()`,
`hovered`, `is_focused(elem)`. Add a per-element **state mask** the router can
report:

```rust
bitflags StyleState { HOVER, PRESSED, FOCUSED, DISABLED }
impl EventRouter { pub fn style_state(&self, elem: u32) -> StyleState { ... } }
```

`HOVER` from `self.hovered == Some(elem)`; `PRESSED` from `self.down.elem == Some(elem)`;
`FOCUSED` from `self.focused == Some(elem)`; `DISABLED` from the element's
`disabled:` prop (a static/`var` bool resolved at lower time).

**Parsing.** The `style {}` grammar's selector gains an optional pseudo-state
suffix: `class_sel := "." IDENT (":" state)?` where `state ∈ {hover, pressed,
focused, disabled}`. `StyleRule` carries `Option<StyleState>`:

```rust
pub struct StyleRule { pub class: Symbol, pub state: Option<StyleState>, pub attrs: Vec<Attr> }
```

An unknown pseudo-state is `CompileError::UnknownStyleState` with a suggestion.

**Resolution.** `interp/style.rs::StyleMap` keys classes by `(Symbol, Option<StyleState>)`.
`resolve` gains the active-state mask and inserts, after the base classes and
before inline, the attrs of every matching `(class, state)` whose state bit is set
— applied in the fixed priority `disabled > pressed > focused > hover` so a
higher-priority state's attr wins ties. Because the state mask changes per frame,
`render()` re-resolves the affected elements; this is bounded to elements that
*have* state rules (tracked at lower time), not the whole tree.

**Why this doesn't reopen D8.** The only "dynamic" inputs are the four
engine-owned booleans, read from the router on the logic thread — never arbitrary
user `var`s inside `style {}`. `check_static` keeps rejecting user-`var` reads in
`style {}`; it only whitelists the pseudo-state selectors. The general dynamic-
style problem D8 deferred stays deferred.

**Animation hook.** If a base property carries `with anim.*` (RFC-0010), a state
change updates that property's `Motion.to`; the transition animates automatically.
No extra syntax at the state site.

### Ordering & correctness (RFC-0003 §8)

State reads happen at `render()` time, which already runs after the tick's event
dispatch settles (RFC-0003 §8, single pull). So the state mask a frame sees is
consistent with the events that produced it — no half-updated resolution. Hot-
reload during a gesture (E5) already holds structure-incompatible patches until
`PointerUp`; state resolution piggybacks on the same rebuild.

## Drawbacks

- Lifts part of D8 — must be scoped tightly to engine states or it becomes the
  general dynamic-style can of worms it was deferred to avoid.
- Per-frame re-resolution for stateful elements (bounded, but non-zero).
- `focus`/`blur` sugar is a second focus API alongside `focused:` — must document
  when to use which (event for one-shot side effects, `focused:` for bound state).
- Four fixed state priorities may not fit every design; escape hatch is inline.

## Rationale and alternatives

- **Pseudo-states vs `var`+handlers.** Pseudo-states collapse the most common
  interaction pattern from ~3 lines/element to zero and reuse the existing engine
  state — this is the single biggest DX win in the visual cluster.
- **Fixed priority vs specificity cascade.** A tiny fixed priority
  (`disabled>pressed>focused>hover`) is predictable and needs no CSS-style
  specificity engine; inline remains the ultimate override.
- **Exposing events vs leaving them internal.** They're already built, tested and
  synthesized; not exposing them is pure waste.
- **Not doing it:** the framework technically "supports hover" but no dev can
  ergonomically use it — failing the core promise.

## Prior art

CSS pseudo-classes (`:hover`/`:active`/`:focus`/`:disabled`), SwiftUI
`.hoverEffect`/`ButtonStyle` `configuration.isPressed`, Flutter
`WidgetStateProperty`/`MaterialState`. Byard's twist: states are engine-owned
booleans resolved reference-free, and transitions ride RFC-0010's GPU springs.

## Resolved decisions (2026-07-01)

- **S1 — Phase-1 state set:** the **four** (`hover/pressed/focused/disabled`);
  `checked/selected/invalid` deferred to Phase 2 (they need form value write-back).
- **S2 — focus API:** **both** — keep `focused:` (bound state) *and* add `focus =>`/
  `blur =>` event sugar (one-shot side effects); both ride the same `focused_sig`.
- **S3 — priority:** **`disabled > pressed > focused > hover`** (disabled always
  wins; active press over persistent focus over transient hover). Not configurable;
  inline is the escape.
- **S4 — bounded re-resolution:** **lower-time bitset** marking elements that have
  state blocks; only those re-resolve per frame.
- **S5 — disabled gates dispatch:** **yes** — a `disabled` element's events are not
  fired (hover-for-tooltip optionally excepted).

> Note: RFC-0016 (styles-as-values) supersedes this RFC's **§B declarative
> pseudo-state layer** (the `.class:hover` selector). The **§A event exposure** and
> the engine `StyleState` mask below remain and are consumed by RFC-0016's
> `on <state> { }` blocks.

## Resolved questions (formerly unresolved)

- [x] **`hover` on `disabled` elements for tooltips:** resolved as **yes, hover fires on disabled** (S5 edge, decided during M37 implementation). A `disabled` element gates *dispatch* of action events (`tap`, `pointer_down`, `key_down`, etc.) but **not** `hover`/`pointer_enter`/`pointer_exit`. Rationale: tooltips are the primary use case for hovering a disabled control ("this button is disabled because…"), and blocking hover would force developers to wrap every disabled element in an invisible hover-target — anti-DX. The gate is applied at the `EventRouter::dispatch` level: `disabled` checks `is_action_event(kind)`, which returns `false` for `Hover`/`PointerEnter`/`PointerExit`. This matches web platform behavior (`pointer-events` on disabled elements still fires `mouseenter`).
- [x] **`StyleState` layout shared with RFC-0016:** resolved as a **4-variant enum** (`StyleStateKind { Hover, Pressed, Focused, Disabled }`) in `parser/ast.rs`, not a bitflag. The `StyleState` type in `interp/events.rs` is a simple wrapper around a `u8` bitmask built from these variants — the enum names the bits, the mask stores them. RFC-0016's `on <state> { }` resolver queries `EventRouter::style_state(element_idx)` which returns the live mask; each `StyleBlock.state` is matched against it. The layout is: `Hover = 0x01`, `Pressed = 0x02`, `Focused = 0x04`, `Disabled = 0x08`. Future value-widget states (`checked`/`selected`/`invalid`) extend the mask at bits 4–6 without breaking existing styles.

## Future possibilities

- `:checked`/`:selected`/`:invalid` value-widget states.
- Combined selectors (`.btn:hover:focused`).
- Group/parent states (`:hover` on a container styling children) — needs the
  transform/opacity group model from RFC-0011's future work.
- Full D8 dynamic styles (arbitrary `var` in `style {}`) as the eventual Phase-3
  generalization this RFC's bounded slice paves the way for.
