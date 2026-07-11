# RFC-0019: Callback Props & Event Forwarding — `Fn` parameters for user Views

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-10
- **Last updated:** 2026-07-10
- **Depends on:** RFC-0001 (§2.2 `Signal<T>`, §5.1 logic thread), RFC-0002 (D9 type inference, `Fn` type in the type table), RFC-0003 (§6 callback props — this RFC implements it), RFC-0005 (§1 `Fn(Args) -> Ret` type), RFC-0007 (user-view instantiation, parameter binding).
- **Enables:** Interactive wrapper Views in packages — `byard-material`'s `TappableCard`, `MenuItem`, `IconButton` can accept and forward `on_tap` callbacks. Essential for any component library.

---

## Summary

Implement **callback parameters** for user `View`s: a `View` can declare a
parameter of type `Fn()` (or `Fn(event)`) and the caller can pass an inline
action expression. The callback is bound at instantiation time (RFC-0007) and
invoked when the inner intrinsic fires the corresponding event. This closes
RFC-0003 §6's "callback props" item and RFC-0005 §1's `Fn(Args) -> Ret` type.

```byld
View TappableCard(on_tap: Fn() = {}, content) {
    Box #[radius: 12, p: 16, shadow: "sm",
          tap => on_tap()] {
        content
    }
}

// Caller:
TappableCard(on_tap: { navigate("detail") }) {
    Text("Tap me")
}
```

Callbacks are **not closures over mutable state** — they are **action
expressions** evaluated in the caller's scope. This keeps them compatible with
the reference-free model: the callee doesn't hold a reference to the caller's
state, it invokes an action that the caller defined.

---

## Motivation

Today, a user `View` that wraps an interactive intrinsic (a tappable `Box`, a
`Button`) cannot forward the event to its caller. The only way to make
`TappableCard` do something on tap is to hardcode the action inside the View
body — which destroys reusability. This is the fundamental reason `byard-material`
components like `MenuItem`, `TappableElevatedCard`, and `IconButton` are
visual-only: they look right but can't *do* anything because the caller can't
pass an `on_tap` handler.

RFC-0003 §6 already sketched this: "callback props bind an inline action
expression from the call site." RFC-0005 §1 already has `Fn(Args) -> Ret` in the
type table. Neither was implemented because the instantiation machinery (RFC-0007)
didn't exist yet. Now that RFC-0007 is fully landed, callback props are the
natural next step.

---

## Guide-level explanation

### Declaring a callback parameter

```byld
View ActionButton(label: Str, on_tap: Fn() = {}, disabled: Bool = false) {
    let s = style {
        bg: 0x6750A4, color: 0xFFFFFF, radius: 20
        on hover { bg: 0x7462AE }
        on pressed { scale: 0.98 }
        on disabled { opacity: 0.62 }
    }
    Button(label) #[..s, tap => on_tap()]
}
```

`on_tap: Fn()` declares a callback parameter with no arguments. The default
`{}` is a no-op. The `=> on_tap()` syntax in the Button's attributes invokes
the callback when the tap event fires.

### Passing a callback

```byld
View App() {
    var count = 0

    ActionButton(label: "Increment", on_tap: { count++ })
    ActionButton(label: "Reset", on_tap: { count = 0 })
    Text("Count: {count}")
}
```

The `{ count++ }` expression is evaluated **in the caller's scope** — it can
read and write the caller's `var`s. The callee (`ActionButton`) never sees
`count`; it just invokes `on_tap()` which triggers the caller's action.

### Callbacks with event data

```byld
View SearchField(on_change: Fn(Str) = {|_|}, on_submit: Fn(Str) = {|_|}) {
    var query = ""
    TextField #[bind: query,
                change => on_change(query),
                submit => on_submit(query)]
}
```

`Fn(Str)` accepts a single string argument. The callee passes `query` when
invoking; the caller receives it:

```byld
SearchField(on_change: {|text| results = search(text)})
```

### Typing rules

| Declaration | Caller syntax | Meaning |
|---|---|---|
| `on_tap: Fn()` | `on_tap: { action }` | no-arg callback |
| `on_change: Fn(Str)` | `on_change: {\|val\| action }` | one-arg callback |
| `on_select: Fn(Int, Str)` | `on_select: {\|i, s\| action }` | multi-arg callback |
| `on_tap: Fn() = {}` | (omitted) | optional with no-op default |

Type mismatch (wrong arity or wrong arg types) is `CompileError::TypeMismatch`.

---

## Reference-level explanation

### 1. AST representation

`Param` gains a `Fn` type variant:

```rust
enum ParamType {
    Scalar(ScalarType),  // Int, Float, Bool, Str, Color, ...
    Callback {
        args: Vec<ScalarType>,  // argument types
        // return type always Void for event callbacks
    },
}
```

A callback default value `{}` is parsed as an `ActionExpr::Block(vec![])`.

### 2. Scope binding (RFC-0007 extension)

At view instantiation, callback arguments are **not evaluated eagerly** like
scalar args. Instead, the caller's action expression AST is captured along with
a **scope snapshot** (the caller's `Env` at the call site). When the callee
invokes `on_tap()`, the action is evaluated against that captured scope.

This is a **lexical closure over the caller's scope** — but critically, the
closure only captures `var` `SignalId`s, not mutable references. Writing
`count++` inside a callback writes to the caller's `Signal` via `SignalId`,
which is an `usize` — `Copy`, `Send`-safe, no lifetime entanglement.

```rust
struct CallbackBinding {
    body: Vec<ActionExpr>,   // the caller's action AST
    params: Vec<String>,     // parameter names (|val| → ["val"])
    scope: EnvSnapshot,      // caller's signal bindings at call time
}
```

### 3. Invocation

When the callee's event handler runs `on_tap()`:

1. Look up `on_tap` in the current scope — it resolves to a `CallbackBinding`.
2. Create a child `Env` from the callback's `scope` snapshot.
3. Bind invocation arguments (`on_change(query)` → bind `val = query_value`).
4. Evaluate the `body` action expressions in that child `Env`.
5. Any `var` writes go through normal `Signal` mutation → Mark-and-Pull.

This all happens within the same tick, on the logic thread. There is no
async boundary, no thread crossing, no `Send` requirement on the callback
itself — it's an intra-tick scope switch.

### 4. Validation

- A `Fn` param used outside `=> invoke()` is `CompileError::CallbackNotInvocable`
  ("callback props can only be invoked in event handlers").
- A `Fn` param invoked with the wrong arity is `CompileError::ArityMismatch`.
- A non-`Fn` expression passed to a `Fn` param is `CompileError::TypeMismatch`.
- Recursive callback invocation (a callback that triggers an event that invokes
  itself) hits RFC-0007's recursion depth limit and is a runtime error.

### 5. Interaction with hot-reload

Callback bindings are rebuilt on hot-reload: when a callee `View`'s body changes,
its callback invocation sites are re-bound. When a caller's action expression
changes, the `CallbackBinding.body` is replaced. The `scope` snapshot references
`SignalId`s, which are stable across reloads (RFC-0002 D11).

---

## Drawbacks

- **Scope capture adds complexity.** The `EnvSnapshot` must be a lightweight
  copy of signal bindings, not a deep clone of the entire environment.
  Implementing this correctly requires the `Env` to support cheap snapshots.
- **No first-class function values.** Callbacks are not general closures — they
  can't be stored in `var`s, passed through data structures, or returned. This
  is intentional (it prevents the callback-spaghetti anti-pattern) but limits
  expressiveness.
- **Debugging.** A callback invoked deep in a View hierarchy produces a stack
  trace that crosses scope boundaries. The diagnostic must show both the
  invocation site (callee) and the definition site (caller).

---

## Rationale and alternatives

**Why action expressions, not general closures?** General closures in a reactive
system create the exact reference/lifetime problems RFC-0003 was designed to
eliminate. Action expressions are *fire-and-forget*: they run, write to signals,
and disappear. No stored references, no dangling closures, no stale captures.

**Why capture by SignalId, not by value?** The callback must write to the
caller's `var`s (`count++`). If it captured by value, writes would be lost.
Capturing by `SignalId` (an `usize`) is zero-cost and correctly routes writes
through the reactive system.

**Why not an `@event` decorator or separate event-forwarding syntax?** A new
syntax adds grammar complexity without benefit. `Fn()` is already in the type
table (RFC-0005 §1); treating callbacks as typed parameters is the natural
extension of the existing parameter model.

---

## Prior art

- **React:** `onClick: () => void` props — function references passed as props.
  Byard's model is similar but with reactive action expressions instead of JS
  closures.
- **SwiftUI:** `action: () -> Void` initializer parameters on `Button`, etc.
  Direct precedent.
- **Flutter:** `VoidCallback` / `ValueChanged<T>` — callback typedefs. Byard's
  `Fn()` / `Fn(T)` is the same concept, type-checked at compile time.
- **Solid.js:** event handler props on components — functions passed as props
  and called from `on:click` handlers.

---

## Unresolved questions

- **Before merge:**
  - [ ] **Async callbacks.** Can a callback invoke a controller method that
    returns asynchronously? Recommendation: yes, the async result arrives on
    the next tick via the relay channel (RFC-0001 §5.1), same as any controller
    call. The callback itself returns immediately.
  - [ ] **Multi-statement callbacks.** Is `{ a++; b = 0 }` valid? Recommendation:
    yes, the body is a `Vec<ActionExpr>` — same as an event handler body.
  - [ ] **Naming convention.** Should callback params be prefixed `on_` by
    convention? The compiler could warn on non-`on_` Fn params as a style lint.

- **During implementation:**
  - [ ] **EnvSnapshot cost.** Profile the snapshot size for typical Views. If
    too large, switch to a reference-counted `Env` with COW semantics.
  - [ ] **LSP support.** Callback invocations need go-to-definition support
    that jumps from `on_tap()` in the callee to the `{ count++ }` in the caller.

---

## Future possibilities

- **Typed event callbacks** — `Fn(TapEvent)` that receive the full event payload
  (coordinates, modifiers) from the intrinsic.
- **Callback chains** — multiple callbacks on the same event
  (`on_tap: { a() }, on_tap: { b() }` — or a dedicated `also` combinator).
- **Generic callbacks** for controller method references — `on_tap: ctrl.save`
  where `ctrl` is an injected controller.
