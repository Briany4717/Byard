# RFC-0003: Interactive Events and View Mutation (reference-free)

- **Status:** Active — implemented (M12 events, M16–M18 write-back/keyboard/focus, M24 full catalog, M32 extended exposure). All design decisions (E1–E8) resolved 2026-06-20; code landed and verified green.
- **Author(s):** Briany4717
- **Created:** 2026-06-20
- **Last updated:** 2026-06-20
- **Depends on:** RFC-0001 (§4.2 spatial hash grid, §5 concurrency, §6 `PlatformHost`), RFC-0002 (Lume surface, automatic reactivity, **D1** Mark-and-Pull, **D4** intrinsic attribute contract, **D9** inference, **D10** tick channels).
- **Resolutions:** the 8 questions formerly in *Unresolved questions* are answered in *Resolved design decisions (E1–E8)*. Three proposed answers were amended for correctness — see **E1** (the value-equality compare reads a `Signal`, so it must run on the logic thread, not the platform thread), **E4** (the proposed `tap`-before-`pointer_up` order was inverted to `pointer_up` → `tap` to match every platform convention — ratified 2026-06-20), and **E6** (the dispatch-vs-layout safety comes from intra-tick step ordering, not from double-buffering as proposed). Event-attribute syntax also changed to the `=>` separator (decided 2026-06-20): engine events use bare names with `=>` (`tap => …`), properties keep `:` — see **§ Attribute syntax**.

---

## Summary

This RFC specifies how a `byld` view **mutates in response to interactive
input**, and the catalog of **natively-supported interactive events** — both
designed around one hard rule: **the developer never holds a reference, handle,
ref, controller, or key to a widget.** Neither to attach an event listener, nor
to command a widget imperatively.

That rule has two halves:

- **Input (event → app).** An interactive handler is declared inline as an
  attribute (`#[tap => count++]`). At mount, the engine registers the
  handler's resolved rectangle into RFC-0001 §4.2's spatial hash grid. The grid
  *is* the listener registry — there is no `addEventListener`, no `GlobalKey`, no
  `useRef`. On input, hit-testing is an `O(1)` grid lookup, never a tree walk.
- **Output (app → widget).** Anything you would normally reach for a reference
  to do imperatively — focus a field, scroll to an offset, select text — is
  instead a **reactive property bound to a `var`**. Mutating the `var` drives the
  widget; the widget reflects state back into the `var`. This is SwiftUI's
  `@FocusState` model and QML's property-binding model, and it means the
  imperative API surface for "command a widget" is *zero*.

Mutation itself is just `var` assignment. There is no `setState`, no
`notifyListeners`, no `ref.current.x`. A handler writes `count++`; RFC-0002's
**D1 Mark-and-Pull** turns that single write into the minimal GPU update on the
next tick. This RFC defines the event pipeline that feeds D1, the event catalog,
the two-way `bind:` sugar, typed event payloads, the mutation-scope rules
(including cross-`View` communication without shared mutable state), and the
per-tick ordering that keeps the whole thing glitch-free.

---

## Motivation

RFC-0002's examples already write `Button("Action") => clicks++` and
`#[tap => …]`, and RFC-0002's own motivation notes that today's
`hello_world.rs` drives a label "from a pointer-input handler" by hand in Rust.
But the *semantics* of that arrow — when the handler runs, on which thread, what
it is allowed to mutate, how the mutation reaches the screen, and how a widget is
commanded back — were never specified. Without that specification, two
contributors will implement events two incompatible ways, and the "no widget
references" promise that distinguishes `byld` from the DOM / Flutter /
imperative-ref world is just an unstated assumption.

The reference-free stance is not stylistic. References are the root of an entire
class of UI bugs and lifetime headaches: dangling refs to unmounted nodes, stale
closures capturing old refs, `GlobalKey` collisions, controllers that outlive
their widget, and — in Rust specifically — the borrow-checker hostility RFC-0001
§Motivation cites as the reason pure-Rust UI "destroys creative flow." A view
whose only state primitive is a `var`, whose only event mechanism is an inline
handler, and whose only imperative channel is a reactive prop, has none of those
failure modes by construction. RFC-0001 §4.2 already built the piece that makes
this possible (hit-testing as an `O(1)` collision query, "the UI tree is never
walked during event dispatch"); this RFC builds the language model on top of it.

---

## Guide-level explanation

### Mutating the view: just assign a `var`

```byld
View Counter() {
    var count = 0

    Column #[gap: 8, p: 16] {
        Text("Count: {count}")
        Button("+") => count++          //  =>  is the primary-action shorthand
        Button("−") #[tap => count--]   // explicit form, same effect
        Button("Reset") #[tap => count = 0]
    }
}
```

`count++`, `count--`, `count = 0` are ordinary mutations of a reactive `var`. The
developer attaches nothing, references nothing, and calls no update function.
RFC-0002 D1 marks the `Text` binding (the only thing that read `count`) dirty and
re-evaluates exactly it on the next tick. The two `Button`s that did *not* read
`count` are untouched.

### Commanding a widget without a reference: reactive props

The imperative things other frameworks need a ref for are reactive props here:

```byld
View SearchBar() {
    var query   = ""
    var editing = false        // drives focus — no ref to the field

    Column #[gap: 8] {
        TextField #[bind: query, focused: editing, placeholder: "Search…"]
        Button("Focus") => editing = true     // sets focus by mutating a var
        Button("Clear") => { query = ""; editing = false }  // blur + clear
    }
}
```

`#[focused: editing]` is a **two-way** binding: setting `editing = true` focuses
the field; when the user taps elsewhere and the field loses focus, the engine
writes `false` back into `editing`. No `field.focus()`, no `FocusNode`, no ref.
Scrolling (`#[offset: scrollY]`), selection, and toggle state work the same way.

### `bind:` — two-way value binding

```byld
TextField #[bind: query]
// desugars to:
TextField #[value: query, change(e) => query = e.value]
```

`bind:` is sugar over a `value:` prop plus the matching `change` handler, for any
value-carrying intrinsic (`TextField`, `Slider`, `Toggle`, `ScrollView`, …). Its
target must be an assignable `var`; `bind:`-ing a `let` or a literal is a
`CompileError::NotAssignable`.

### Attribute syntax: `:` for properties, `=>` for events

Inside `#[...]` the separator encodes the kind of attribute (formalized in
RFC-0002 grammar and decision **D4-bis**). The rule is read-at-a-glance: `:` is
appearance/state, `=>` is behavior.

```byld
Column #[
    gap: 12,                    // property  — `name: value`
    bg: 0x222222,
    focused: editing,           // reactive prop (still a value) uses `:`
    tap => count++,             // engine event — `name => action`
    pointer_move(e) => cur = e.pos,   // `name(e) => action` binds the payload
] { }
```

- **Properties** use `name: value` and bind a value — including reactive props
  (`focused:`, `offset:`, `value:`, `bind:`) and function-valued **callback
  props** passed to a child `View` (`onPick: (e) => …`), because passing a
  function *value* is still a value binding.
- **Engine events** use `name => action` (or `name(payload) => action`). Event
  names are **bare** — no `on` prefix, since `=>` already marks them. This is the
  same `=>` as the element-tail shorthand (`Button("+") => count++` is the
  primary `tap` event hoisted), and `name => expr` desugars to
  `name: (payload) => expr` internally. The catalog (§4) lists every event name.

---

## Reference-level explanation

### 1. The event pipeline (input → mutation → screen)

An interactive event travels a fixed path, ending in RFC-0002's D1 tick:

```
 [platform/event-loop thread]                 [logic thread]                 [render thread]
  winit event  ─► PlatformHost                                                
   normalize ─► InputEvent{kind,pos,payload}                                  
        └──► crossbeam channel ──────────────►  drained at tick start         
                                                 │                            
                                                 ├─1 apply hot-reload (D10)    
                                                 ├─2 dispatch InputEvents:     
                                                 │     H(x,y) grid lookup (§4.2)
                                                 │     walk handler AST lambda 
                                                 │     → var writes → D1 marks 
                                                 ├─3 drain controller results  
                                                 │     → var writes → D1 marks 
                                                 ├─4 D1 PULL: re-eval dirty     
                                                 │     bindings + structural    
                                                 │     effects → RenderFrame    
                                                 └─5 atomic frame swap ────────►  reads frame
```

- **Capture & normalize (platform thread).** RFC-0001 §6's `PlatformHost`
  (e.g. `WinitHost`) receives the OS event and normalizes it into a thread-safe,
  `Send + 'static` `InputEvent { kind: EventKind, pos: Vec2, payload: Payload }`.
  No `Signal` is touched here — §5.1 restricts `Signal` access to the logic
  thread.
- **Transport.** The `InputEvent` is pushed onto a `crossbeam-channel` to the
  logic thread, the same kind of channel D10 uses for hot-reload delivery and
  RFC-0001 §5.1 uses for Tokio results. Input is an unbounded (or large-bounded)
  queue — input must never be silently dropped the way stale hot-reloads are
  coalesced.
- **Dispatch (logic thread, tick step 2).** For a positional event the engine
  computes `H(pos)` and retrieves the topmost registered handler in amortised
  `O(1)` (RFC-0001 §4.2) — *the UI tree is never walked.* The handler is an AST
  lambda held by reference in the owning `View`'s `Env` (RFC-0002's interpreter
  model: lambdas are AST subtrees, not Rust closures), so dispatch is "walk this
  sub-tree now," with the event payload bound as its parameter.
- **Mutation → marks.** Every `var` write inside the handler runs D1's
  synchronous dirty-mark cascade. **No pull happens during dispatch.**
- **Settle & pull (tick step 4).** After *all* queued events (and async results)
  for this tick have been dispatched and their marks have accumulated, D1's pull
  phase re-evaluates the dirty value-bindings and structural effects exactly
  once each (D1's per-tick epoch guard), producing the new `RenderFrame`.

This ordering is the whole game: because dispatch only *marks* and never *pulls*,
no handler can observe a half-updated view, multiple events in one tick compose
cleanly, and D1's "the tick is the consistency boundary" invariant holds for
interactive input exactly as it does for any other mutation source.

### 2. Registration — why no reference is needed (input side)

When an element carrying an event attribute mounts, the interpreter's intrinsic
lowering (RFC-0002 `interp/intrinsics.rs`) does two things with the handler:

1. Stores the handler lambda's AST node in the `View`'s `Env`.
2. After `Atlas` resolves the element's Taffy rectangle, registers
   `(rect, z, event_kind) → handler_id` into the spatial hash grid cell(s) the
   rect covers (RFC-0001 §4.2 "Registration").

That registration *is* the event subscription. There is no widget object exposed
to the developer to attach to, and nothing to detach: on unmount, the element's
`ViewArena` drop (RFC-0001 §2) removes its grid entries in the same linear pass.
A handler therefore cannot dangle past its element's lifetime — the lifetime of
the subscription is exactly the lifetime of the arena, enforced by construction,
not by developer discipline.

### 3. Commanding a widget — why no reference is needed (output side)

The reference-free rule cuts both ways. The actions other frameworks expose as
imperative methods on a ref — `focus()`, `blur()`, `scrollTo()`, `select()`,
`open()/close()` — are modeled as **reflected reactive props**: a prop whose
value the engine both *reads from* and *writes back to* a `var`.

| Imperative API elsewhere | `byld` reactive prop | Direction |
|---|---|---|
| `field.focus()` / `FocusNode` | `#[focused: editing]` | two-way (`var` ⇄ widget) |
| `scrollController.jumpTo(y)` | `#[offset: scrollY]` | two-way |
| `field.select(a, b)` | `#[selection: range]` | two-way |
| `dialog.open()` | `when isOpen { Dialog { … } }` | structural (D1) |
| `animation.play()` | `#[playing: isPlaying]` | two-way (Phase 3) |

Setting the `var` commands the widget; the widget reflects user-driven changes
back into the same `var`. The mental model is uniform: **state lives in `var`s,
the widget is a pure projection of that state, and "doing something to a widget"
is always "mutating the state it projects."** This is why no ref is ever needed
to *drive* a widget, only to *read* its state — and reading its state is just
reading the `var`.

### 4. Native event catalog (Phase 2 core)

Each event is a bare-named `=>` attribute on an intrinsic (events live in
`#[...]`, never in `(...)`; the `=>` separator marks them — § Attribute syntax).
Unknown event names are a hard `CompileError::UnknownAttribute` with a Levenshtein
hint (D4). Written `name => action`, or `name(e) => action` to bind the payload.
The Phase 2 core set:

| Category | Event (use with `=>`) | Payload | Notes |
|---|---|---|---|
| Tap | `tap` (alias `click`) | `PointerEvent` | primary action; element-tail `=>` maps here |
| Tap | `long_press` | `PointerEvent` | duration threshold from theme (= `tap` upper bound) |
| Tap | `double_tap` | `PointerEvent` | |
| Tap | `secondary` | `PointerEvent` | right-click / two-finger |
| Pointer | `pointer_down` / `pointer_up` | `PointerEvent` | raw button transitions |
| Pointer | `pointer_move` | `PointerEvent` | only while pointer is over the rect |
| Hover | `pointer_enter` / `pointer_exit` | `PointerEvent` | desktop hover |
| Hover | `hover` | `ChangeEvent<Bool>` | convenience: enter/exit as a bool |
| Keyboard | `key_down` / `key_up` | `KeyEvent` | requires focus |
| Focus | `focus` / `blur` | `FocusEvent` | paired with the `focused:` prop |
| Edit | `change` / `input` | `ChangeEvent<T>` | `T` from the field's value type |
| Edit | `submit` | `ChangeEvent<T>` | Enter / commit |
| Scroll | `scroll` | `ScrollEvent` | paired with the `offset:` prop |
| Wheel | `wheel` | `ScrollEvent` | discrete wheel deltas |

Note the reactive **props** these events pair with — `focused:`, `offset:`,
`value:` — use the `:` separator, not `=>`, because they bind state rather than
map an event (§ Attribute syntax).

**Deferred (post-Phase-2, follow-up RFC):** full gesture recognizers
(`drag_start`/`drag_update`/`drag_end`, pan, pinch, rotate), multi-touch /
multiple simultaneous pointers, IME composition events, and pointer-capture
semantics (a drag that continues outside the originating rect). These need either
gesture state machines or pointer capture that the §4.2 grid does not model yet;
called out explicitly rather than implied.

### 5. Event payloads and the D9 inference exception

A handler is either **zero-argument** or takes **one typed payload**:

```byld
Button("+")            => count++                       // zero-arg (primary `tap`)
Box #[pointer_move(e)  => cursor = e.pos]               // typed payload `e`
TextField #[change(e)  => query = e.value]              // ChangeEvent<Str>
```

Payload record types (defined in `byard-core`, exposed to `byld`):

```rust
struct PointerEvent  { pos: Vec2, button: Button, mods: Mods }
struct KeyEvent      { key: Key, mods: Mods, repeat: bool }
struct ChangeEvent<T>{ value: T }
struct FocusEvent    { gained: bool }
struct ScrollEvent   { offset: Vec2, delta: Vec2 }
```

**Inference exception to D9.** D9 requires explicit annotations on `fn`
parameters. **Event-handler lambda parameters are exempt:** the intrinsic's attr
signature already declares the payload type (`pointer_move` → `PointerEvent`),
so the interpreter infers `e`'s type from the attribute, and the Phase 4
transpiler emits the concrete type from the same table. This exemption is the
specific case of a general rule formalized in **E2**: an inline lambda's
parameter types are inferred from the *expected function type* at its use site —
whether that type comes from an intrinsic attr signature (`pointer_move` →
`Fn(PointerEvent)`) or a developer-declared `Fn(...)` callback prop. Named `fn`
declarations and `View`/`fn` signatures always require annotations (D9). `bind:`'s
generated `change` lambda is covered by the same rule.

### 6. Mutation scope — what a handler may write

A handler is an AST lambda living in a `View`'s `Env`. Its mutation rights follow
lexical scope, with one bright line at the `View` boundary:

- **Within its own `View`:** a handler may read and write any `var` of that
  `View`. Nested intrinsic elements (`Column { Button … }`) share the enclosing
  `View`'s single `Env` (RFC-0002: "one `Env` per `View` instance; nothing below
  the `View` level introduces a scope"), so a deeply nested `Button`'s handler
  mutating a top-of-`View` `var` is ordinary lexical access — no prop-drilling
  required *inside* a `View`.
- **Across a `View` boundary:** a handler **cannot** reach into another `View`'s
  `Env`. There is no shared mutable state across `View`s, which is what keeps the
  whole model single-owner and free of the cross-component mutation hazards refs
  enable. Two sanctioned channels exist:

  **(a) Callback props (child → parent, ephemeral).** The parent passes a
  function-typed attribute; the child declares a function-typed parameter and
  invokes it.

  ```byld
  View Parent() {
      var picked = ""
      Picker #[options: ["a","b"], onPick: (e) => picked = e.value]
  }
  View Picker(options: List<Str>, onPick: Fn(ChangeEvent<Str>)) {
      for opt in options {
          Button(opt) #[tap => onPick(ChangeEvent { value: opt })]
      }
  }
  ```

  The `onPick` lambda is an AST node owned by `Parent`'s `Env`; `Picker` only
  invokes it. The mutation of `picked` therefore *executes in `Parent`'s scope*,
  on the logic thread, with no shared reference and no cross-`Env` write — the
  child never names `picked`.

  **(b) Injected controllers (shared / persistent).** For state that outlives a
  single `View` or is shared across the tree, a handler calls a method on an
  `inject`-ed controller (RFC-0001's `#[byard_controller]` boundary). Async work
  (network, disk) returns to the logic thread via the Tokio-result channel
  (RFC-0001 §5.1) and lands as tick-step-3 mutations.

- **Structural mutation is automatic.** If a handler mutates a `var` that a
  `when` or `for` reads, D1 marks that structural effect dirty and the tick's
  pull phase mounts/unmounts the affected arenas (RFC-0002 structural effects).
  Opening a dialog, revealing a row, or clearing a list is therefore *also* just
  a `var` write — no separate navigation or visibility API.

### 7. Hit-testing details: topmost-wins, no implicit bubbling

The §4.2 grid returns the **topmost** handler for a given event kind at a given
point, ordered by the Z-bin / stacking-context rules of RFC-0001 §3.2. Phase 2
deliberately does **not** implement DOM-style capture/bubble propagation:

- **Decision:** an event is delivered to exactly one handler — the topmost
  element registered for that event kind under the pointer. There is no implicit
  walk up an ancestor chain (the grid has no ancestor chain to walk; that is the
  point of §4.2).
- **Rationale:** bubbling presupposes a tree traversal during dispatch, which
  §4.2 exists specifically to avoid. Topmost-wins keeps dispatch `O(1)` and makes
  "which handler runs?" unambiguous.
- **Consequences handled without bubbling:** common bubbling use-cases get
  explicit, declarative answers instead — "tap outside to dismiss" is a
  full-bleed backdrop element with its own `tap` behind the dialog (and `when
  isOpen { … }` controls both), not a document-level capturing listener. A row
  that is tappable *and* contains a tappable button simply registers both; the
  button, being on top, wins inside its rect and the row wins elsewhere.
- **Flagged:** if real applications show topmost-only is too restrictive (e.g.
  they genuinely need an outer handler to also see an inner event), a future RFC
  can add opt-in propagation — but it must define how propagation coexists with
  the grid without reintroducing a tree walk. Left as an unresolved question
  rather than silently assuming bubbling exists.

### 8. Threading and ordering (formal)

All handler execution is on the **logic thread** (RFC-0001 §5.1), because
handlers touch `Signal`s. The per-tick order, extending RFC-0002 D10's
"drain channels at tick start":

1. **Hot-reload** channel drained (D10) — apply code changes first, so input
   applies to the latest code.
2. **Input** queue drained in **FIFO arrival order** — each event hit-tested and
   dispatched; mutations accumulate D1 marks. No pull.
3. **Controller results** channel drained (RFC-0001 §5.1) — async I/O results
   mutate `var`s; marks accumulate. No pull.
4. **D1 pull** — dirty value-bindings and structural effects re-evaluated once
   each (epoch guard); `RenderFrame` produced.
5. **Atomic frame swap** to the render thread (RFC-0001 §5.2).

Re-entrancy is impossible by construction: because steps 2–3 only mark and step 4
is the sole pull, a handler can never trigger a synchronous re-render, and no
handler ever runs against a partially-updated frame. Input dispatched in tick *T*
is hit-tested against the layout/grid produced by tick *T−1* (what the user
actually saw and clicked); structural changes from those handlers register new
grid entries when their arenas mount, visible to tick *T+1*. This is correct and
must be stated, because it means a handler that spawns a new button cannot
receive a click on that button within the same tick — the user has not seen it
yet.

---

## Drawbacks

- **No event bubbling in Phase 2.** Some established patterns (event delegation,
  ancestor-level interception) have no direct translation and must be rewritten
  declaratively (§7). This is a real ergonomic cost for developers coming from
  the DOM, mitigated by explicit backdrop/`when` patterns.
- **Reactive-prop imperatives can feel indirect.** "Focus this field" becoming
  "set a `var` the field is bound to" is unfamiliar to developers used to
  `ref.focus()`, and a poorly-named driver `var` can obscure intent. The win
  (no refs, no lifetime bugs) is worth it, but the learning curve is real —
  exactly the SwiftUI `@FocusState` adjustment.
- **One payload per handler.** Handlers take zero or one payload, not a
  positional argument list. Composite needs (e.g. pointer + keyboard mods) are
  handled by fields on the payload record, not multiple params — a deliberate
  simplification that the catalog's payload types must anticipate.
- **Gestures are deferred.** Drag/pan/pinch/multi-touch are out of Phase 2
  (§4). Apps needing rich gestures wait for the follow-up gesture RFC.
- **Input queue growth under stall.** Unlike coalesced hot-reloads, input is
  never dropped, so a pathologically stalled logic thread grows the input queue.
  Acceptable (a stalled logic thread is already a bug), but worth monitoring.

---

## Rationale and alternatives

**Why no widget references at all?** References are the source of dangling-handle
bugs, stale closures, key collisions, and (in Rust) borrow-checker hostility that
RFC-0001 §Motivation names as the reason native-Rust UI is unpleasant. A model
where state is `var`s, events are inline handlers registered via the §4.2 grid,
and imperatives are reflected props has none of those failure modes structurally.

**Why reflected props instead of an imperative escape hatch (e.g. a
`ViewHandle`)?** An escape hatch would reintroduce exactly the lifetime and
aliasing problems the design eliminates, and would have to be `!Send`-juggled
across the render/logic threads. Reflected props keep every imperative action on
the existing reactive substrate (D1), so "focus" is not a special path — it is a
`var` write like any other, and it composes with `when`/`for` for free.

**Why topmost-wins instead of bubbling?** §4.2 hit-testing is a collision query,
not a tree traversal; bubbling would require the tree walk the architecture
exists to avoid. Topmost-only preserves `O(1)` dispatch and unambiguous delivery;
the rare genuine need for propagation is left to an opt-in future mechanism
rather than paid for on every event.

**Why drain input at tick boundaries instead of dispatching synchronously on
arrival?** Synchronous dispatch on the platform thread would touch `Signal`s
off the logic thread (violating §5.1) and would let a handler observe a
half-updated frame (violating D1's consistency boundary). Tick-boundary draining
makes input just another mutation source that settles before the single pull.

**Why callback props for child→parent instead of letting children mutate parent
`var`s?** Allowing cross-`View` `var` writes is shared mutable state across
ownership boundaries — the precise hazard the no-reference rule removes. Callback
props keep the mutation executing in the owner's scope while letting the child
stay ignorant of the parent's state.

**Why a single payload record, not positional args?** Fixed-shape typed records
make the D9 inference exception sound (the engine knows the type) and keep the
handler grammar identical to an ordinary one-or-zero-arg lambda — no special
call form.

---

## Prior art

- **RFC-0001 §4.2 spatial hash grid.** The substrate that makes reference-free,
  tree-walk-free event dispatch possible; this RFC is the language model over it.
- **SwiftUI `@FocusState` / `.focused($x)`.** The direct model for reflected
  imperative props: focus is a piece of bindable state, not a method on a ref.
- **QML property bindings.** Two-way property binding as the universal mechanism
  for both reading and driving widget state — the lineage of `bind:` and reflected
  props.
- **Solid.js / Svelte event handlers.** Inline `onClick={…}` / `on:click` with no
  ref and no manual subscription — the input-side ergonomics target.
- **Elm / Flutter callback props.** Child→parent communication via passed
  functions rather than shared mutable references — the model for §6(a).
- **React `useRef` / Flutter `GlobalKey` / DOM `addEventListener`.** The
  reference-based approaches this RFC deliberately rejects, named so the contrast
  is explicit.

---

## Resolved design decisions (E1–E8)

All eight questions formerly in *Unresolved questions* are resolved here. Each is
authoritative for implementation; three carry a correctness amendment marked in
**bold**.

### E1 — Reflected write-back: source-gated + value-deduplicated (loop dies at length 1)

Two independent guards close the feedback loop between a two-way prop and its
`var`:

1. **Source gate.** A write-back is emitted **only** by a direct physical user
   action (a keystroke into a field, a physical scroll, a click elsewhere that
   blurs). A `var` mutation triggered by application code **never** emits a
   write-back. This alone breaks code→var→widget→var cycles.
2. **Value dedup.** Even for a physical action, the intrinsic's change path reads
   the current `Signal` value and compares: if `new == current`, the event is
   discarded at zero cost; if they differ, a synthetic `ReflectedWriteBack`
   mutation is injected into the in-flight tick's step-2 queue, updating the
   `Signal` and running D1's lazy marking.

**Amendment (thread placement).** The proposal had the intrinsic "intercept the
input and read the current `var` before the event loop ends." That comparison
reads a `Signal`, and RFC-0001 §5.1 restricts `Signal` access to the logic
thread. So the platform thread forwards only the **raw new value** as an ordinary
`InputEvent`; the value-equality compare runs in **step 2 on the logic thread**,
never on the platform thread. With that placement, the closure argument holds
exactly as proposed: at step 4 (pull) the `value:` binding re-evaluates to the
value the widget already physically shows, the renderer emits no GPU command,
and programmatic widget writes never re-emit input events — so the cycle
terminates at length 1, by construction.

### E2 — `Fn(...)` callback-prop type syntax + the lambda-inference rule

Callback-prop parameters are typed with a `Fn(...)` type, mandatory in `View`/`fn`
signatures (no D9 exception for signatures):

```byld
View Picker(options: List<Str>, onPick: Fn(ChangeEvent<Str>)) { … }
// with a return type when needed:
//   predicate: Fn(Int) -> Bool
```

Syntax rule: keyword `Fn`, argument types in parentheses, return type omitted for
the common void case, otherwise `Fn(Args) -> Ret`. The parser produces a
`Type::Function` node. The **general inference rule** (generalizing §5's event
exception): an inline lambda passed at a use site infers its parameter types from
the **expected `Fn` type** there — an intrinsic attr signature or a declared
`Fn(...)` param — and the compiler validates the lambda's arity and payload
against that `Type::Function`. Callback lambdas are AST nodes owned by the
**parent** `Env` (§6a), not Rust closures, so they carry no `Send`/lifetime cost.

### E3 — Keyboard focus: one global scalar, no `FocusNode`

Focus is a single logic-thread scalar `focused_arena_id: Option<ArenaId>` (one
focused element per window). `#[focused: editing]` associates the `var` `editing`
with that arena's focus state, two-way via E1's write-back.

- **Focus stealing (mutual exclusion).** Clicking another focusable element does
  two things in the same tick: inject a `false` write-back to the previously
  focused element's `var`, then set `focused_arena_id` to the new id and write
  `true` to its `var`.
- **Tab navigation.** On `Tab`, the engine does a pre-order traversal of the
  previous tick's Taffy tree, filtered to elements that registered a keyboard or
  text-input listener, and advances focus to the next id (`Shift+Tab` → previous).

Refinements: (a) the Tab traversal is an acceptable infrequent `O(n)` walk
**precisely because it is not the hot path** — pointer hit-testing stays `O(1)`
via the grid (§4.2); only the rare `Tab` keypress walks, and it can be
precomputed into an ordered focus-ring later if needed. (b) Binding the *same*
`var` to `#[focused:]` on two elements is undefined (last-mount wins) — bind
distinct `var`s. (c) Explicit tab-index ordering is deferred; the default order
is pre-order (declaration/visual order). (d) Multi-window focus is deferred
(single window in Phase 2).

### E4 — `tap` vs `pointer_up`: thresholds + precedence (ratified)

A pointer interaction is a valid `tap` iff **all three** hold: `PointerDown` and
`PointerUp` land within the same element's grid rect; total cursor displacement
< **8 logical px**; elapsed interval < **500 ms**. The 500 ms upper bound is the
same themeable constant as the `long_press` threshold, so tap and long-press
partition cleanly at one boundary.

**Precedence (ratified 2026-06-20): low-level before high-level —
`pointer_up` → `tap`.** When both are registered on one element, the engine
dispatches `pointer_up` first, then `tap`, in the same tick step 2. This matches
every mainstream platform (DOM `pointerup` precedes `click`; the same on
iOS/Android), so developer muscle memory carries over. The earlier
`tap`-before-`pointer_up` proposal is dropped. Both orders were mechanically safe
(both mutations accumulate in D1 flags for the single pull); the choice was
purely a DX convention, now settled in favor of platform consistency. The full
firing order for a qualifying tap is therefore `pointer_down` → `pointer_up` →
`tap` (and `double_tap`/`long_press`, when they qualify, fire after `tap` /
instead of `tap` respectively, per their threshold rules).

### E5 — Hot-reload during an in-flight gesture: Gesture State Safety

A shared `pointer_pressed: AtomicBool` is set by the platform thread. At tick
step 1, the reload gate uses RFC-0002's existing hot-reload diff classification:

- **Reactive-compatible** patches (RFC-0002 case 1 — pure expression/value
  changes that do not alter tree shape) apply immediately, even mid-gesture.
- **Structure-incompatible** patches (case 2 — `var`/`param`/`inject` shape
  changes that tear down and rebuild an arena) are **held** while
  `pointer_pressed`, coalesced latest-wins in the bounded(1) channel (D10), and
  applied synchronously on the first tick after `PointerUp`.

This prevents destroying a view arena in the middle of a physical interaction.
Edge: a pointer held indefinitely starves structural reloads (rare, acceptable —
the latest held patch still applies on release).

### E6 — Grid-vs-dispatch safety: **intra-tick step ordering, not double-buffering**

Ratified: step-2 dispatch hit-tests against the previous tick's grid, so the user
always interacts with what they see. **Amendment (the reason why):** the proposal
attributed the lock-free safety to RFC-0001 §5.2 double-buffering. That is the
wrong mechanism. Double-buffering protects the **render thread** reading a frame
while the **logic thread** writes the next one. The dispatch-vs-layout safety is
different: dispatch (step 2) and re-layout / grid-rebuild (step 4, `Atlas`) are
**sequential steps on the same logic thread**, so when step 2 runs, this tick's
new geometry has not been computed yet and the grid is unambiguously last tick's.
No lock is needed because the two operations never overlap in time, not because
they are double-buffered. Both guarantees coexist; stating the right one prevents
a future contributor from "optimizing" away the step ordering on the false belief
that double-buffering covers it.

### E7 — Input coalescing: continuous events only, by `(kind, element_id)`

At step 2, before dispatching, the drainer inspects the whole buffer and
coalesces **only continuous events** — `pointer_move`, `scroll`, `wheel`,
and hover-move — per `(kind, element_id)`: keep the **latest absolute position**,
**sum the deltas**, discard intermediate samples, and dispatch one consolidated
call so the AST lambda is walked once per frame.

**Discrete events are never coalesced** — `pointer_down`, `pointer_up`,
`tap`, `double_tap`, `long_press`, `pointer_enter`/`pointer_exit`,
`key_down`/`key_up`, `submit` — each carries distinct semantics and every
occurrence is delivered. This is the one place the catalog is partitioned into
"continuous/idempotent" vs "discrete/significant," and the partition is fixed in
the intrinsic table.

### E8 — Hit-test slop: backend inflation to a 44×44 minimum

When `interp/intrinsics.rs` registers an interactive element's rect into the grid,
if either dimension is below **44×44 logical px** it inflates the **collision area
only** (never the visual/Taffy layout) to a centered 44×44 minimum, **clamped to
the parent's scissor/clip bounds** (§3.3) so the hit area never invades hidden or
clipped regions.

Refinements: (a) 44 logical px follows Apple HIG and is themeable (Material uses
48 dp). (b) Two inflated sibling targets that would overlap are resolved by §7
topmost/registration order, and inflation is clamped so it does not cross a
sibling's center. (c) Inflation applies only to elements that registered a
listener; static elements never enter the grid. (d) The visual layout is
untouched — only the invisible touch target grows, so the developer's layout
stays exactly as authored.

---

## Future possibilities

- **Gesture recognizers** (drag, pan, pinch, rotate, long-press-drag) and
  **pointer capture** as a dedicated follow-up RFC, building gesture state
  machines over the pointer events defined here.
- **Opt-in event propagation** if topmost-only proves insufficient, defined so it
  does not reintroduce a tree walk (e.g. a per-cell handler stack the dispatcher
  may optionally fold over).
- **Multi-touch / multi-pointer** for tablet and touch-table targets, extending
  `PointerEvent` with a pointer id.
- **Animation as reflected state** (`#[playing: …]`, transition `var`s) once the
  Phase 3 animation system lands, keeping animation imperative-free like focus.
- **Declarative keyboard shortcuts** (`#[shortcut: "Cmd+S" => save()]`) as a
  focus-independent global event source registered outside the spatial grid.
