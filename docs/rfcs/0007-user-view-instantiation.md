# RFC-0007: User-View Instantiation — Composing `View`s Within a File

- **Status:** Active — implemented (M30–M35 in `IMPLEMENTATION_4.md`). ViewTable, arg binding, body expansion, recursion/cycles, hot-reload, slots/defaults all landed. Decisions D-A through D-E resolved (IMPL-46–50).
- **Author(s):** Briany4717
- **Created:** 2026-06-24
- **Last updated:** 2026-06-24
- **Depends on:**
  - RFC-0001 §2 (memory model — `ViewArena`, `Signal<T>`), §5.1 (logic-thread-only signals), §7.3 (Dev interpreter vs. Prod transpiler)
  - RFC-0002 **D11** (multiple `View`s per file; per-`ViewDecl` reload), the AST shape (`ViewDecl`, `Param`, `ElementNode`, `Arg`)
  - RFC-0004 (reactive interpreter — memos, structural `when`/`for`, mount/unmount)
  - RFC-0005 (intrinsic catalog — the closed set of built-ins a user view is *not*)
- **Enables:** RFC-0008 (Package Ecosystem). Instantiation is the language-level
  prerequisite; importing a `View` from another file is worthless until a `View`
  call actually expands. This RFC contains **no** multi-file, network, or CLI work.

---

## Summary

Today a user-defined `View` can be *declared* and *referenced*, but a reference
never expands: `interp/eval.rs::lower_element` validates the name against
`known_views` and then falls through its `_ =>` arm, lowering the call as a
generic container `Box` — its inline children are lowered, but the referenced
view's **body is never expanded and its parameters are never bound**. This RFC
specifies **user-view instantiation**: turning a `View` call site into a real
subtree, with positional/named argument binding, a per-instance reactive scope,
recursion/cycle protection, and hot-reload that re-derives instances when a
referenced view changes. Scope is deliberately limited to **a single `.byd`
file**; cross-file resolution is RFC-0008.

---

## Motivation

### The composition primitive is missing

RFC-0001 §"`byld` at a glance" presents `View` as "the fundamental unit," and
RFC-0002 D11 already allows many `View`s per file. The natural expectation —
that one `View` can call another and get its rendered subtree inlined — does not
hold. Concretely, in `crates/byard-compiler/src/interp/eval.rs::lower_element`:

```rust
// "Toggle" | "Slider" | "TextField" => { ... }   // value widgets
// "Image" => { ... }                             // TextureSampler
_ => {
    // Box / Column / Row / ScrollView and any other container.
    let children = self.lower_members(&el.children, known_views);
    RenderNode::Box { name: el.name.clone(), attrs: el.attrs.clone(),
                      children, action: el.action.clone(), bound_sig: None }
}
```

A call such as `MyCard(title: "Hi")` reaches this `_` arm. `validate_element`
(`interp/intrinsics.rs`) has already confirmed `MyCard` is in `known_views`, so
no diagnostic fires — but the result is an empty decorated `Box`: the body of
`MyCard` is discarded and `title` is dropped on the floor. There is no way to
factor a UI into reusable pieces, which is the single most load-bearing
ergonomic in every framework `byld` takes inspiration from (SwiftUI views,
Flutter `StatelessWidget`, Solid components).

### Why this is its own RFC, ahead of packages

A package is just a `View` (and assets) authored in another file. If a *local*
`View` call does not expand, an *imported* one cannot either. Splitting this out
means the highest-risk, highest-value language work lands and is testable with
zero network, zero CLI, and zero registry surface — and RFC-0008 inherits a
proven instantiation path instead of co-developing two hard problems at once.

---

## Guide-level explanation

A `View` with parameters is declared exactly as today (RFC-0002 grammar,
`View NAME(params) { ... }`). The new behavior is at the **call site**:

```byld
View Avatar(url: Str, size: Int) {
    Image(url) #[width: size, height: size, radius: size]
}

View UserRow(name: Str, avatar: Str) {
    Row(gap: 12, align: .center) {
        Avatar(url: avatar, size: 40)      // ← expands to Avatar's body
        Text(name) #[typo: m3.titleMedium]
    }
}

View App() {
    Column(gap: 8) {
        UserRow(name: "Ada",  avatar: "ada.png")
        UserRow(name: "Alan", avatar: "alan.png")
    }
}
```

Each call site is replaced by the callee's rendered body. Arguments bind to the
callee's parameters by name (or by position); inside the callee, a parameter is
an ordinary reactive value, so passing a parent `var` makes the child update
when the parent mutates. Each instance has its **own** local `var`s — two
`UserRow`s do not share state.

Two ergonomics are explicitly deferred to decisions below, not assumed: whether
a user view may accept a **trailing child block** (`Card { ... }`, i.e. slots),
and whether parameters may declare **default values**.

---

## Reference-level explanation

### 1. View registry on the interpreter

The interpreter already threads `known_views: &[&str]` for validation. Lowering
needs more than names — it needs the `ViewDecl`s. Introduce a resolved view
table on the interpreter, built once per program load from `ParsedFile::views`:

```text
ViewTable {
    by_name: HashMap<Symbol, ViewId>,   // Symbol → dense id
    decls:   Vec<ViewDecl>,             // owned, Send
}
```

`lower_element` consults `ViewTable` *before* its intrinsic match: a name that
resolves to a `ViewId` is a **user-view call**; otherwise it is an intrinsic or
container as today. The intrinsic catalog (RFC-0005) remains closed and takes
precedence — a user may not shadow `Column`/`Text`/etc. (diagnostic, see §6).

### 2. The instantiation algorithm

When `lower_element` sees a user-view call `Callee(args)` it performs:

1. **Resolve** `Callee` → `ViewDecl` via `ViewTable`. Unknown ⇒ diagnostic
   (already covered by `validate_element`'s `UnknownElement`).
2. **Bind arguments → parameters** (§3). Produces a set of per-instance
   bindings, one reactive value per declared `Param`.
3. **Open an instance scope.** Push a fresh lexical frame on `Env` containing the
   parameter bindings; allocate the instance's reactive state in its own arena
   region (RFC-0001 §2.1 `ViewArena` — one contiguous block per mounted view
   instance, reclaimed in `O(1)` on unmount).
4. **Lower the body.** Recursively `lower_members(&callee.body, known_views)`
   within the instance scope, exactly as a top-level view is lowered by
   `lower_view`. Local `var`/`let`/`fn` declarations open in the instance scope,
   so they are isolated per instance.
5. **Splice.** Insert the resulting `RenderNode`s at the call site. A call that
   yields multiple roots is spliced as siblings (consistent with how `when`/`for`
   already splice multiple nodes via `lower_member_into`).
6. **Close** the instance scope (truncate `Env`), leaving the parent untouched.

Inline children at the call site (`Callee { ... }`) are **not** silently lowered
into a `Box` anymore; their handling is governed by the slot decision (D-A).

### 3. Argument → parameter binding

`ElementNode` carries `content: Vec<Arg>` (positional) and `attrs: Vec<Attr>`
(named `#[...]`). Binding rules:

- **Named** arguments match a `Param` by `Symbol`; **positional** arguments match
  by declaration order. Mixing is allowed only positional-before-named.
- Each bound argument is projected as a **memo** over the parent scope (reuse
  `bind_value`/`open_value_binding`), not a snapshot — so a parameter fed a
  parent `var` stays live and propagates on mutation (RFC-0004 reactivity).
  A literal argument lowers to a constant memo (no dirty edges).
- Arity, unknown-parameter, duplicate-parameter, and missing-required-parameter
  errors are diagnostics (§6), not panics.

> **Signals stay on the logic thread (RFC-0001 §5.1).** Binding creates
> memos and arena entries only on the logic thread; nothing here crosses a thread
> boundary.

### 4. Recursion and cycles

Instantiation is recursive, so it can diverge. Two protections:

- **Static cycle detection (single file).** Build a call graph over `ViewTable`
  (edge `A → B` when `A`'s body references user view `B`). A cycle that is not
  guarded by a structural `when`/`for` boundary is reported at load time. (A
  recursive tree view guarded by `when has_children` is legal; an unconditional
  `A` calls `A` is not.)
- **Runtime depth bound.** Lowering carries a depth counter; exceeding a
  configured maximum (decision D-C) yields a diagnostic and a truncated subtree
  rather than a stack overflow — correctness before cleverness.

### 5. Hot-reload across instances

`interp/reload.rs` diffs **per `ViewDecl` name** (D11). Instantiation changes the
blast radius: editing `Avatar` must now re-derive **every** `Avatar` instance,
including those nested inside other views. Required changes:

- The reload pass resolves the set of views *transitively affected* by a changed
  `ViewDecl` (the changed view plus its callers, via the §4 call graph) and
  re-lowers the affected subtrees.
- Per-instance state preservation follows the existing reactive-compatible vs.
  structure-incompatible split (RFC-0002 D11, RFC-0003 gesture gate): a body edit
  that preserves shape rebinds instance `var`s in place; a shape change remounts
  the affected instances (their `ViewArena`s drop, new ones allocate).
- This RFC keeps the file-watcher single-file. RFC-0008 generalizes the watcher
  to a file graph; the per-instance invalidation logic specified here is the
  piece it reuses.

### 6. Diagnostics

New compile errors (extend `diagnostics`/`intrinsics::validate_element` and the
checker, not the parser):

| Code (proposed) | Trigger |
|---|---|
| `ViewArityMismatch` | call supplies more positional args than params |
| `UnknownParam` | named arg not declared on the callee |
| `MissingParam` | required param has no argument (interacts with D-B defaults) |
| `DuplicateParam` | same param bound twice (positional + named, or named twice) |
| `IntrinsicShadowed` | a `View` is named like an RFC-0005 intrinsic |
| `RecursiveView` | unguarded static call cycle |

Diagnostics must carry source spans and, where useful, the callee's parameter
list as a hint (SwiftUI-quality errors, RFC-0001 §"Rationale").

### 7. Prod-transpiler forward compatibility (design constraint, not work)

RFC-0001 §7.3 commits to a Prod path that transpiles `byld` → Rust. The
instantiation model above must be expressible by that path: argument binding as
memo projection, per-instance scope, and body splicing all have a static AOT
analogue (a generated function per `View`, parameters as typed inputs). This RFC
does **not** implement the transpiler, but every decision here is checked against
"can the AOT backend emit this without a runtime view-lookup table." Anything
that would force runtime-only semantics must be flagged before merge.

---

## Open decisions

Each decision below is recorded with Context / Decision / Why / Consequences.
**Never decide silently.**

- **D-A — Slot / child-block model.** Does `Callee { ... }` pass a child block to
  the callee, and how does the callee receive it? *Recommendation:* introduce an
  explicit content parameter (a `View`-typed or `slot` param) rather than magic
  trailing-block capture; keep first implementation to named params only and log
  slots as a follow-up sub-decision. Resolve before the package ecosystem (RFC-0008).
- **D-B — Parameter defaults.** Allow `View Card(elevated: Bool = false)`?
  *Recommendation:* yes, since it sharply reduces call-site noise and is trivial
  to transpile; defaults evaluate in the callee scope. Affects `MissingParam`.
- **D-C — Recursion policy.** Static-cycle-as-error plus a runtime depth bound
  (recommended), and the bound's value. Must not be a silent truncation without a
  diagnostic.
- **D-D — Per-instance arena granularity.** One `ViewArena` per instance
  (RFC-0001 §2.1, recommended) vs. a pooled region. Measure before optimizing
  (project ethos): start with per-instance, record allocation cost as was
  done for the render path.
- **D-E — Intrinsic precedence.** Confirm intrinsics (RFC-0005) always win over a
  same-named user `View` and that the collision is a hard error, not shadowing.

---

## Drawbacks

- **Reload blast radius grows.** Per-`ViewDecl` diffing must now chase callers;
  an incorrect affected-set computation shows stale UI. Mitigated by the §4 call
  graph being the single source of truth for both cycle detection and reload.
- **Divergence risk.** Recursion makes a malformed program able to diverge at
  lower time; the depth bound is a guard, not a proof. A future totality check is
  out of scope.
- **AOT coupling.** Committing to a model the transpiler can emit constrains the
  interpreter's freedom (no runtime-only view tricks). This is intentional but
  real.

---

## Rationale and alternatives

- **Why expand at lower time, not introduce a runtime `Component` node?** A
  dedicated runtime node would keep the render tree smaller but pushes a
  view-lookup indirection into every frame and complicates §7's AOT path.
  Lowering-time expansion keeps the per-frame render tree flat (consistent with
  the existing `when`/`for` splice model) and transpiles cleanly.
- **Why memos for arguments, not value snapshots?** Snapshots break reactivity
  across the call boundary — a child fed a parent `var` would freeze. Memo
  projection is the same mechanism `let`/`bind_value` already use (RFC-0004).
- **Why static cycle detection at all?** Without it the first recursive view a
  developer writes is an unbounded lower-time loop. A diagnostic is strictly
  better DX than a crash, matching the RFC-0001 error-quality thesis.

---

## Prior art

- **SwiftUI `View` composition.** Direct DX reference: value-type views, body
  re-evaluation, environment injection. `byld`'s parameter-as-reactive-value
  mirrors SwiftUI's `@Binding`/plain-value distinction.
- **Flutter `StatelessWidget`/`StatefulWidget`.** Per-instance state isolation
  and the build-method-returns-subtree model map onto §2's instance scope.
- **Solid.js components.** Components run once and wire fine-grained signals;
  `byld`'s memo-projected arguments target the same "props are reactive" behavior
  without a re-render.

---

## Resolved questions (formerly unresolved)

- [x] **Generic / typed `View` params (slots, D-A).** Resolved by IMPL-46: slots use an explicit `content` parameter — the callee declares `content: View` and the caller passes a child block. This is a **special form** in the interpreter (the content block is lowered as a closure-like scope splice, not a first-class `View` value), avoiding the complexity of `View`-typed values in the type system. The type checker sees `content` as an opaque child slot, not a generic type parameter. This covers the 95% case (wrappers like `Card`, `Dialog`); multi-slot composition is a future extension.
- [x] **Keying for `for`-instantiated views.** Deferred — RFC-0002 D7's coarse reconciliation (drop-and-rebuild) applies uniformly to user views and intrinsics. Keyed reconciliation lands only when `N>50` **and** churn is measured (D7's own trigger). State preservation on reorder is a sub-RFC concern, not a user-view concern — the identity model is the same regardless of whether the `for` body is a user view or an intrinsic.
- [x] **Inspector/devtools.** Resolved: instance boundaries are **fully erased** by lowering. The interpreter expands a user view into its body's intrinsics inline — no marker node survives in the render tree. A future inspector can reconstruct boundaries from the `ViewTable` + call-site spans (the data exists in the compiler, not in the render tree). Keeping the render tree lean is a performance invariant; inspector metadata belongs in a debug side-channel, not in the hot path.
