# RFC-0017: Overlay & Z-Layer System — portals, modals, and floating surfaces

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-10
- **Last updated:** 2026-07-10
- **Depends on:** RFC-0001 (§3.1 render pipelines, §4.2 spatial grid, §5 concurrency), RFC-0002 (D1 Mark-and-Pull, `when` structural effect), RFC-0003 (event dispatch, hit-testing), RFC-0005 (intrinsic catalog), RFC-0011 (opacity/transform), RFC-0012 (style states), RFC-0016 (style system).
- **Enables:** Complete implementations of dialogs, menus, tooltips, snackbars, bottom sheets, side sheets, date/time pickers, popovers, context menus, and dropdown surfaces in `byard-material` and `byard-cupertino`.

---

## Summary

Add a **z-layer rendering model** and an `Overlay` intrinsic that lets a subtree
escape the normal layout flow and render above everything else in a separate
compositing layer. Overlays are stacked in declaration order within a global
**overlay stack** managed by the engine, hit-tested top-down (topmost overlay
first), and dismissed declaratively through `when` — no imperative show/hide API,
no widget references, consistent with RFC-0003's reference-free design.

The overlay system adds exactly **one new render pass** (a sorted, back-to-front
composite of overlay layers after the main scene) and **one new intrinsic**
(`Overlay`). Everything else — positioning, animation, scrim, dismissal — is
composed from existing primitives (`Box`, `when`, transforms, style states).

---

## Motivation

The byard-material gap analysis identified **overlay rendering** as the single
largest blocker: dialogs, menus, tooltips, snackbars, bottom sheets, side sheets,
date/time pickers, search result surfaces, and dropdown selects all require
content to render *above* the normal layout tree. Without this, these components
are limited to inline approximations that the caller must manually position — a
DX failure that makes a component library unusable for real apps.

The same need arises in Cupertino: action sheets, popovers, context menus with
previews, and the iOS notification center all float above the app's content.

Every major UI framework solves this: Flutter's `Overlay`/`OverlayEntry`, the
DOM's stacking contexts and `position: fixed`, SwiftUI's `.sheet`/`.popover`
modifiers. Byard needs an answer that:

1. **Respects the reference-free model** — no handle to show/hide an overlay.
2. **Is declarative** — mount/unmount via `when`, not imperative push/pop.
3. **Integrates with existing hit-testing** — overlays intercept input above the
   main tree, with proper dismissal on outside tap.
4. **Costs zero when unused** — no extra render pass if no overlay is mounted.

---

## Guide-level explanation

### The `Overlay` intrinsic

```byld
View ConfirmDialog(visible: Bool, title: Str, body: Str, content) {
    when visible {
        Overlay #[modal: true] {
            // Scrim — tapping outside dismisses (caller handles via on_dismiss)
            Box #[bg: 0x1D1B20, opacity: 0.32, grow: 1]

            // Dialog surface, centered
            Column #[bg: 0xECE6F0, radius: 28, shadow: "lg",
                     p: 24, gap: 16, width: 312,
                     anchor: center] {
                Text(title) #[size: 24, weight: medium]
                Text(body) #[color: 0x49454F, size: 14]
                Row #[gap: 8, justify: end] {
                    content
                }
            }
        }
    }
}
```

`Overlay` removes its children from the normal layout flow and places them in a
separate compositing layer rendered **above** all non-overlay content. Multiple
overlays stack in **mount order** (most recently mounted = topmost). The
`Overlay` itself occupies zero space in its parent's layout.

### Positioning

Children of an `Overlay` are laid out in a **full-viewport coordinate space** by
default. An `anchor` prop on any child controls placement:

| `anchor` value | Meaning |
|---|---|
| `center` | centered in viewport (default for modal content) |
| `top`, `bottom`, `start`, `end` | edge-aligned |
| `(x, y)` | absolute logical-pixel offset from viewport top-left |
| `relative(ref_element)` | **deferred** — anchored to a specific element |

For the initial implementation, `center` and edge anchors cover dialogs,
snackbars, bottom sheets, and side sheets. Relative anchoring (for dropdown
menus, tooltips near a trigger) is a follow-up — the workaround is to compute
position from the trigger's known layout coordinates and pass as `(x, y)`.

### Modality

`modal: true` (the default) makes the overlay **capture all input**: taps outside
the overlay's content children hit the scrim (the first child, typically a
semi-transparent `Box`), not the main tree below. This is how dismissal works —
the scrim's tap handler sets the `visible` var to `false`, which unmounts the
`when` and therefore the `Overlay`.

`modal: false` allows input to pass through to the main tree for non-modal
surfaces (tooltips, snackbars that don't block interaction).

### Stacking order

Overlays stack in mount order. If multiple overlays are mounted simultaneously,
the one whose `when` became true later renders on top. Within a single overlay,
children follow normal z-painting order (last child on top). This is
deterministic and requires no explicit z-index values.

---

## Reference-level explanation

### 1. The overlay stack

A new field on `Engine` (or the interpreter's frame state):

```rust
struct OverlayEntry {
    id: OverlayId,          // unique per mount
    modal: bool,
    children: Vec<RenderNode>, // the overlay's content subtree
    order: u64,             // monotonic mount counter
}

struct OverlayStack {
    entries: Vec<OverlayEntry>, // sorted by order ascending
    next_id: u64,
}
```

`OverlayStack` lives on the logic thread (`!Send`). It is cleared and rebuilt
each reactive tick alongside the main render tree (consistent with the current
full-rebuild model). A future incremental update can diff overlay entries by
`OverlayId` for state preservation.

### 2. Rendering

After the main scene's render pass completes, the encoder runs a **second pass**
over `OverlayStack.entries` in order (back to front). Each entry's children are
laid out against the viewport rect (not the parent's layout rect) and painted
using the same pipelines (`SolidBox`, `DecoratedBox`, `TextGlyph`, `VectorMSDF`,
`TextureSampler`). The overlay pass shares the same GPU texture atlas and glyph
cache — no resource duplication.

If `OverlayStack` is empty, the second pass is skipped entirely — zero cost when
no overlay is mounted.

### 3. Hit-testing

The spatial hash grid (RFC-0001 §4.2) is extended with a **layer tag** per entry:

```rust
struct GridEntry {
    rect: Rect,
    handler_id: HandlerId,
    layer: LayerTag, // MainScene | Overlay(OverlayId)
}
```

Hit-testing walks layers **top-down**: the topmost overlay's entries are tested
first. If a modal overlay is mounted and the hit point falls outside all of that
overlay's rects, the hit is consumed by the overlay's scrim (or discarded if no
scrim handler exists) — it never reaches lower layers. Non-modal overlays that
miss fall through to the next layer.

### 4. The `Overlay` intrinsic

Added to RFC-0005's catalog:

- **Content:** none. **Children:** any (the overlay's content subtree).
- **Props:** `modal: Bool` (default `true`), `dismiss_on_outside: Bool` (default
  `true` when modal — fires a `dismiss` event on scrim tap).
- **Events:** `dismiss` (fired when modal + outside tap), all pointer.
- **Pipeline:** none (layout-only; children use their own pipelines).
- **Constraint:** an `Overlay` whose `when` guard is not in scope is a
  `CompileWarning` ("overlay is always mounted; wrap in `when` for
  conditional display").

### 5. Dismissal

No imperative `dismiss()` method. Dismissal is `when visible { Overlay { ... } }`
where the overlay's scrim handler sets `visible = false`. The `dismiss` event is
sugar: `#[dismiss => showDialog = false]` is equivalent to placing a full-viewport
tap handler as the first child.

### 6. Interaction with transforms and animations

Overlay children support all RFC-0011 transforms and RFC-0010 animations. A
bottom sheet sliding up is:

```byld
Overlay #[modal: true] {
    Box #[bg: 0x1D1B20, opacity: 0.32, grow: 1]
    Column #[bg: 0xFEF7FF, radius: (28, 28, 0, 0),
             anchor: bottom, width: viewport,
             translate: (0, 0) with anim.spring()] {
        // sheet content
    }
}
```

The translate animation runs on the GPU (RFC-0010); the overlay compositor simply
reads the final transformed quads.

---

## Drawbacks

- **New render pass.** Even though it's skipped when empty, the code path exists
  and must be maintained. The compositor ordering adds complexity to the encoder.
- **Full-viewport layout context.** Overlay children cannot participate in the
  parent's layout (they're removed from the flow). This is intentional but means
  a tooltip can't automatically position itself "next to" its trigger without
  coordinate passing — the `relative` anchor mode (deferred) would solve this.
- **Stacking order by mount time.** Two overlays mounted in the same tick have
  undefined relative order (deterministic within a single tick by declaration
  order, but across ticks it's mount order). This is fine for typical use
  (one modal at a time) but could surprise if multiple overlays compete.

---

## Rationale and alternatives

**Why a new intrinsic, not a style prop?** `z-index` as a style prop (CSS model)
creates a combinatorial stacking-context nightmare and fights the reference-free
model. A dedicated `Overlay` intrinsic makes the escape-hatch explicit: "this
subtree leaves the flow." It's the Flutter `Overlay` model, not the CSS model.

**Why not `Stack`/`ZStack`?** A `ZStack` (RFC-0018) layers children *within* the
layout tree. An `Overlay` escapes the tree entirely — it's viewport-relative,
not parent-relative. Both are needed; they serve different purposes.

**Why modal by default?** The vast majority of overlay use cases (dialogs,
bottom sheets, action sheets) are modal. Making non-modal the default would
lead to accidental input pass-through bugs.

**Why declaration-order stacking, not explicit z-index?** Explicit z-index is
the single biggest source of CSS layout bugs. Declaration order is deterministic,
requires no coordination between components, and matches how overlays actually
work (one at a time, the latest one wins).

---

## Prior art

- **Flutter `Overlay` / `OverlayEntry`:** imperative push/pop model. Byard
  rejects the imperative API but adopts the concept of a global overlay stack.
- **DOM stacking contexts + `position: fixed`:** the general model, but CSS
  z-index coordination is a known DX failure.
- **SwiftUI `.sheet` / `.fullScreenCover` / `.popover`:** declarative modifiers
  bound to a `Bool`. Closest to Byard's `when` + `Overlay` pattern.
- **Jetpack Compose `Dialog` / `Popup`:** composable wrappers that render in a
  separate window. Similar to Byard's overlay layer.

---

## Unresolved questions

- **Before merge:**
  - [ ] **Relative anchoring.** Should `anchor: relative(element_id)` ship in
    v1, or is coordinate-passing sufficient for menus/tooltips initially?
  - [ ] **Nested overlays.** Can an overlay's content mount another overlay?
    (Needed for a menu inside a dialog.) Recommendation: yes, they stack.
  - [ ] **Transition on dismiss.** When `when` unmounts the overlay, animated
    children are immediately destroyed. Should there be a `leave` transition
    mechanism (deferred to RFC-0025)?

- **During implementation:**
  - [ ] **Scissor interaction.** An overlay child that extends beyond the
    viewport — clamp or allow offscreen painting?
  - [ ] **Accessibility.** Modal overlays must trap focus (Tab cycles within the
    overlay). Wire this with RFC-0003 E3's focus system.

---

## Future possibilities

- **Relative anchoring** for dropdown menus, tooltips, and autocomplete surfaces
  that position themselves near a trigger element.
- **Shared-element transitions** between an overlay and the main tree (hero
  animations).
- **Multi-window** — each window has its own overlay stack; cross-window overlays
  are explicitly out of scope.
- **Priority overlays** (e.g., system-level alerts above all app overlays) with
  a tier system.
