# RFC-0002: `byld` Language Definition (Lume surface) — Compiler Pipeline, Automatic Reactivity, and Dev-Mode Interpreter

- **Status:** Draft — all open questions resolved 2026-06-20 (ready for merge review)
- **Author(s):** Briany4717
- **Created:** 2026-06-20
- **Last updated:** 2026-06-20
- **Supersedes:** the surface-syntax examples in RFC-0001 §"`byld` at a glance" and §"Rust controller" (see *Implications* → erratum).
- **Resolutions:** the 12 questions formerly in *Unresolved questions* are now answered in *Resolved design decisions (Phase 2, final)*. Two of the proposed answers were amended for correctness — see decisions **D6** (`CompileError` would have broken the `byard-core` ⇄ `byard-compiler` layering) and **D9** (the `Text` type name collided with the `Text` intrinsic).

---

## Summary

This RFC promotes `byld` from the conceptual sketch in RFC-0001 to a **defined
language**, adopting the *Lume* design as its surface syntax, and specifies the
Phase 2 deliverable: the `byard-compiler` crate that turns `byld` source into a
running view tree.

The single most important change from the prior draft is the **reactivity
model**. RFC-0001's `signal` keyword (explicit reactive declarations) is replaced
by **automatic, fine-grained reactivity**: the developer writes ordinary
mutable bindings (`var`) and the compiler/interpreter discovers the dependency
graph by *tracking reads*, exactly the Solid.js model RFC-0001 already cites as
its reactivity reference. Crucially, this is **not a new memory primitive** —
it is an automatic way of populating the subscriber links that RFC-0001's
`Signal<T>` dirty-flag vector (§2.2) already provides. The developer stops
wiring reactivity by hand; the engine wires it for them.

Three stages, one crate, unchanged in spirit from the prior draft: a
`logos`-generated lexer, a hand-written recursive-descent parser whose
expression layer uses Pratt parsing, and a tree-walking AST interpreter that
runs in Dev mode by binding directly against the `Signal` / `ViewArena` /
`EvaluatorTick` API Phase 1 shipped. On top of the prior draft it adds: a
**reactive read-tracking scope**, **computed bindings** (`let` / `fn`),
**first-class control flow** (`for` / `when`) as *structural* reactive effects,
a **three-zone element syntax** (`(...)` content · `#[...]` props · `{ }`
children), and a **scoped `style` block**. Production transpilation (`byld` →
native Rust, Phase 4) remains **out of scope**, but every Phase 2 decision is
checked against it so the transpiler inherits the same reactive semantics
without a redesign.

A versioned grammar is fixed (see *Reference-level explanation* → Grammar),
satisfying RFC-0001's "a grammar RFC must be written before the parser is
implemented" unresolved item — now for the full Lume surface, not just the
minimal `UserCard` subset.

---

## Motivation

Phase 1 closed with a working engine core (`evaluator`, `atlas`, `encoder`,
`relay` wired into `Engine`, proven by `hello_world`). But `hello_world.rs`
authors its view by hand in Rust. There is no `byld` surface in the codebase,
and therefore RFC-0001's central thesis — *the declarative layer and the
systems layer never share a file; `byld` describes UI, Rust controls the world*
— is still unverified.

The prior draft of RFC-0002 set out to parse RFC-0001's `UserCard` example
"verbatim" with a minimal grammar. Since then, a dedicated design study of
frontend-syntax preferences (Flutter, JSX/TSX, QML, SwiftUI, Jetpack Compose,
.NET MAUI, plus the Svelte/Solid/Angular reactivity convergence) produced the
**Lume** design, which the project has decided to adopt as `byld`'s definitive
surface. Lume's thesis matches Byard's thesis exactly: declarativity *without*
the pyramid-of-doom (Flutter), reactivity *without* hand-wired signals (the
"too programmational" failure mode of explicit-signal APIs), styling
*co-located but isolated* (the CSS-vs-Tailwind false dilemma), and diagnostics
*that never lie* (the SwiftUI "type-checker gives up" and MAUI "binding
silently degrades" failure modes).

Adopting Lume therefore is not a cosmetic rename — it is the decision that the
DSL should **do the reactive bookkeeping for the developer**. That decision
cascades into the compiler design below, and it is the reason this draft
supersedes the previous one rather than amending it.

Dev mode (interpreted) is still specified before Prod mode (transpiled, Phase
4), for the same two reasons as the prior draft: an interpreter is the faster
path to a *working* language while the grammar is still young, and locking a
transpilation target before the grammar has been exercised would mean compiling
against a moving target twice. RFC-0001 §7.3 scopes Dev = interpreter,
Prod = transpiler; this RFC specifies the Dev half and the language it
interprets.

---

## Guide-level explanation

### `byld` at a glance (Lume surface)

```byld
View UserCard() {
    var clicks = 0                       // reactive source — no `signal` keyword
    inject AppEnvironment as env         // ambient value (React Context model)

    // Three zones:  (...) content  ·  #[...] props/config  ·  { } children
    Column #[gap: 12, bg: env.theme.surface, radius: 16, p: 20] {
        Text("Clicks: {clicks}") #[typo: m3.titleLarge]
        Button("Action") => clicks++     // `=>` is the primary-action shorthand
    }
}
```

Compared to the RFC-0001 sketch, three things changed and one stayed:

- `signal clicks = 0` became `var clicks = 0`. **There is no `signal` keyword
  in `byld` anymore.** Any `var` is a reactive source; any expression that
  reads it becomes a subscriber automatically. `clicks++` mutates it and the
  engine updates only the `Text` that read it.
- Decorative/spatial properties moved out of the call parentheses into a
  `#[...]` attribute block. `(...)` now carries only the element's *primary
  positional content* (a `Text`'s string, a `Button`'s label).
- Events use the `=>` separator with bare names (no `on` prefix): the primary
  action is the element-tail shorthand `Button("Action") => clicks++`, and the
  explicit form is `#[tap => clicks++]` (with `#[tap => …, secondary => …]` when
  an element needs several handlers). See **D4-bis** and RFC-0003 for the full
  event/attribute-syntax rule.
- `inject` is unchanged.

### State, derived values, and "no signals to write"

```byld
View Search() {
    var query = ""                                   // reactive source
    var items: List<Str> = ["apple", "pear", "plum"] // `Str` = string scalar (see D9)

    // `let` / `fn` are *computed*: reactive iff they read a `var`, memoized.
    let filtered = items.filter(|x| x.starts_with(query))
    fn greeting() -> Str => filtered.is_empty() ? "No matches" : "Results"

    Column #[gap: 8, p: 16] {
        Text(greeting()) #[style: .title]
        TextField #[bind: query, placeholder: "Filter…"]

        for item in filtered {                       // first-class iteration
            Text(item) #[style: .row]
        }
        when filtered.is_empty() {                   // first-class conditional
            Text("Nothing here") #[style: .muted]
        }
    }

    style {                                          // scoped, no global cascade
        .title #[size: 20, weight: bold]
        .row   #[p: (4, 8)]
        .muted #[color: 0x888888]
    }
}
```

The developer never writes `Signal::new`, `.get()`, `.set()`, a dependency
array, a `useMemo`, or a `@Composable`-style recomposition annotation. They
write `var`, read it, and mutate it. Everything reactive is inferred.

### Rust controller (unchanged boundary)

```rust
#[byard_controller]
pub struct NetworkController { base_url: String }

impl NetworkController {
    pub async fn fetch_user(&self, id: u64) -> Result<User, ApiError> { /* … */ }
}
```

`#[byard_controller]` still generates the shared-memory bindings and the typed
metadata file the `byld` LSP consumes. Note the deliberate visual rhyme:
`#[...]` means "attribute/configuration" in *both* languages now — props in
`.byd`, macro attributes in `.rs`. They never collide (different files,
different grammars) and the shared spelling is an asset for a Rust-first
author, not a hazard. (See *Implications* for the one caveat.)

---

## Reference-level explanation

### Crate layout

```
crates/byard-compiler/src/
├── lib.rs          — public API: compile_view(), interpret_tick(), CompileError
├── lexer/          — logos-generated Token enum + StrLit callback driver
├── parser/
│   ├── ast.rs       — typed AST node definitions
│   ├── expr.rs      — Pratt expression parser (nud/led tables)
│   └── error.rs     — parse diagnostics (span, message, hint)
├── interp/
│   ├── env.rs        — per-View binding environment, `inject` resolution
│   ├── eval.rs       — AST → engine calls (Signal read/write, ViewArena alloc)
│   ├── reactive.rs   — read-tracking scopes, computed memos, structural effects  ← NEW
│   ├── intrinsics.rs — Column/Row/Text/Button/… → BoxInstance/TextLine writes
│   ├── style.rs      — scoped style-table resolution                              ← NEW
│   └── reload.rs     — hot-reload AST diff → restart-or-patch decision
└── diagnostics.rs   — shared error/span types
```

`byard-compiler` depends on `byard-core` (for `Signal`, `ViewArena`,
`EvaluatorTick`, the new `LogicRuntime` trait); **never the reverse**, the same
direction `byard-platform → byard-core` already follows. This is the layering
discipline of RFC-0001 §9 applied one level up; `byard-compiler` is a crate
*above* `byard-core`, not a fifth subsystem inside it. The two `NEW` modules
(`reactive.rs`, `style.rs`) are where adopting Lume costs the most
implementation work relative to the prior draft.

### Data structures

- `Span { start: u32, end: u32 }` — byte offsets, `Copy`; every AST node carries
  one (diagnostics + future LSP). Unchanged from the prior draft.
- `Symbol` — `Arc<str>` from a single process-global, content-addressed interner
  (append-only `HashSet<Arc<str>>`). `Arc`, not `Rc`: `CompiledView` must be
  `Send + 'static` to cross the file-watcher → logic-thread channel, and every
  `Symbol` in its AST has to satisfy that bound. Identity is stable across
  reparses by construction (content-addressed, not encounter-order). Unchanged
  from the prior draft and still resolved.
- `Token` (logos derive) — terminals now include the Lume additions:
  `View`, `Var`, `Let`, `Fn`, `Inject`, `As`, `For`, `In`, `When`, `Else`,
  `Style`, `Ident(Symbol)`, `IntLit(i64)`, `FloatLit(f64)`, `StrLit` (raw),
  `LParen`/`RParen`, `LBrace`/`RBrace`, `HashBracket` (`#[`), `RBracket` (`]`),
  `LBrack` (`[`, array literals), `Comma`, `Colon`, `Dot`, `Arrow` (`=>`),
  `Eq`, `PlusEq`, `MinusEq`, `PlusPlus`, `MinusMinus`, `Lt`/`Gt` (generic
  arguments), `Pipe` (`|`, lambda params in `filter(|x| …)`), etc. `#[` is one
  token (`HashBracket`), not `#` + `[`, so it never lexes ambiguously against a
  future standalone `#`.
- AST (`ast.rs`), sketch — note `Stmt` is replaced by a richer `Member` because
  a View body now contains declarations, elements, control flow, and a style
  block, not just statements:

  ```rust
  struct ViewDecl { name: Symbol, params: Vec<Param>, body: Vec<Member>, span: Span }

  enum Member {
      Var    { name: Symbol, ty: Option<Type>, init: Expr, span: Span },  // reactive source
      Let    { name: Symbol, ty: Option<Type>, init: Expr, span: Span },  // computed/const
      Fn     { name: Symbol, params: Vec<Param>, ret: Option<Type>, body: Expr, span: Span },
      Inject { ty: Type, name: Symbol, span: Span },
      Element(ElementNode),
      For    { var: Symbol, iter: Expr, body: Vec<Member>, span: Span },     // structural
      When   { cond: Expr, then: Vec<Member>, els: Option<Vec<Member>>, span: Span }, // structural
      Style  { rules: Vec<StyleRule>, span: Span },
      Expr(Expr),
  }

  struct ElementNode {
      name: Symbol,                 // intrinsic (Column, Text, …) or user View
      content: Vec<Arg>,            // (...) positional content
      attrs: Vec<Attr>,             // #[...] props/config
      action: Option<Expr>,         //  => shorthand (primary action)
      children: Vec<Member>,        // { } block
      span: Span,
  }
  struct Attr { name: Symbol, value: Expr, span: Span }
  struct StyleRule { class: Symbol, attrs: Vec<Attr>, span: Span }

  enum Expr {
      IntLit(i64, Span), FloatLit(f64, Span),
      StrLit(Vec<StrPart>, Span),   // StrPart::Text(String) | StrPart::Interp(Box<Expr>)
      Ident(Symbol, Span),
      Array(Vec<Expr>, Span),
      Member { base: Box<Expr>, field: Symbol, span: Span },
      Call   { callee: Box<Expr>, args: Vec<Arg>, span: Span },
      Lambda { params: Vec<Symbol>, body: Box<Expr>, span: Span },
      Assign { target: Box<Expr>, op: AssignOp, value: Box<Expr>, span: Span }, // = += -=
      Postfix{ target: Box<Expr>, op: PostfixOp, span: Span },                  // ++ --
      Ternary{ cond: Box<Expr>, then: Box<Expr>, els: Box<Expr>, span: Span },
      Error(Span),
  }
  struct Arg { name: Option<Symbol>, value: Expr }
  ```

  The AST owns all its data (no borrows into source text), so hot-reload can
  re-parse and diff against the previous tree without lifetime entanglement.

### Lexer

A single `#[derive(Logos)]` enum compiled to one DFA. `#[logos(skip
r"[ \t\r\n]+")]` and `#[logos(skip r"//[^\n]*")]` drop whitespace and line
comments. String interpolation (`"Clicks: {clicks}"`) is handled exactly as the
prior draft specified — the lexer emits one raw `StrLit` token whose end is
located by a `#[logos(callback = …)]` function maintaining a fixed-depth
two-state stack (`InString` / `InBraceExpr`), and the *parser* recursively
re-invokes the pipeline on each `{...}` span (the PEP 701 model). This part is
unchanged by the Lume adoption; the only delta is the larger terminal set
listed above.

### Parser

Declaration / statement / element / control-flow level is ordinary recursive
descent — one function per production. Expression level uses Pratt
(precedence-climbing) parsing. The Lume surface *strengthens* the original
argument for Pratt: the expression grammar now mixes member access
(`env.theme.surface`), calls (`filter(|x| …)`), postfix `++`/`--`, assignment
`= += -=`, the ternary `?:`, and arrow lambdas at different binding strengths.
A one-function-per-precedence-level grammar would be unmanageable; Pratt's
`nud`/`led` table scales by adding entries:

```rust
fn parse_expr(&mut self, min_bp: u8) -> Expr {
    let mut lhs = self.parse_nud();              // literal, ident, array, paren, lambda
    loop {
        let Some((op, l_bp, r_bp)) = self.peek_led() else { break };
        if l_bp < min_bp { break; }
        self.advance();
        lhs = self.parse_led(op, lhs, r_bp);     // ., (, ++, --, =, +=, -=, ?:
    }
    lhs
}
```

The element production is recursive descent: parse the head `IDENT`, then
optionally `(` content `)`, then optionally a `#[` attr-block `]`, then *either*
a `{` children `}` block *or* a `=>` action shorthand. `for` / `when` / `style`
are keyword-led statements inside a View body.

Error recovery is unchanged: any `parse_*` that hits an unexpected token records
a `CompileError`, substitutes `Expr::Error(span)` (or skips to the next member
boundary at statement level), and continues, so one pass collects multiple
diagnostics. As before, this RFC does **not** propose a lossless/rowan tree;
that is flagged for Phase 3's LSP.

### Dev-mode interpreter — values

A tree-walking interpreter, no bytecode. Its environment binds straight to the
Phase 1 `Evaluator` types. `Env`, the per-View binding environment, is a flat
`Vec<(Symbol, Value)>` walked in reverse (shadowing for free), allocated once
per View instance and never truncated until the View's `ViewArena` drops — the
same design and the same soundness argument as the prior draft (nothing below
the View level introduces a scope; nested elements get their own child arena +
`Env`).

The binding rules, updated for Lume:

- `var x = <init>` lowers to `let sig = Signal::new_in(arena, eval(init))` —
  *identical* to how the prior draft lowered `signal x = …`. The keyword changed;
  the lowering did not. The handle lives in the per-View `Env`.
- `let y = <expr>` and `fn f(...) => <expr>` lower to a **computed memo** (see
  next section) — reactive and cached if the expression transitively reads a
  `var`, a plain constant otherwise. The reactive-vs-const determination is made
  by the read-tracking machinery at first evaluation, not by a separate static
  pass.
- Reading a `var`-bound identifier evaluates to `sig.read(|v| …)` **and**
  registers a subscription against the currently-active reactive scope (next
  section). RFC-0001 §5.1 restricts `Signal` access to the logic thread and the
  interpreter only ever runs inside an `EvaluatorTick`, so the read-tracking
  thread-local lives on the logic thread by construction.
- `inject T as name` resolves by walking a parent-pointer chain of `Env`
  *references* (one borrow per ancestor View), a runtime lookup (React Context),
  unchanged from the prior draft.
- Lambdas (`|x| …`, `() => clicks++`) are **not** Rust closures — they are AST
  subtrees held by reference in `Env`, re-walked when the bound event fires or
  the iterator applies them. This sidesteps `Send`/lifetime questions inside the
  interpreter entirely, unchanged from the prior draft, and a `debug_assert!`
  still guards the "lambda never outlives its View arena" invariant.

### Dev-mode interpreter — automatic reactivity (NEW: `interp/reactive.rs`)

This is the core addition. The insight that makes it cheap: **RFC-0001's
`Signal<T>` already *is* a subscription mechanism.** §2.2 defines a `Signal` as
"the current value + a vector of atomic dirty flags pointing to specific render
or spatial subsystem entries," and §3.3 already turns a mutation into minimal
dirty rectangles. The explicit-`signal` model left the developer to arrange
*which* render entry each signal pointed at. Lume's automatic model just fills
those pointers in by observing reads.

**Reactive scope.** A logic-thread-local `current_scope: Cell<Option<ScopeId>>`
names the computation currently being evaluated. Three things open a scope:

1. A **value binding** — the content/attr expressions of an intrinsic element
   that lower to a `BoxInstance`/`TextLine` field (e.g. the string of
   `Text("Clicks: {clicks}")`, the `bg:` of a `Column`). The scope's "output"
   is that primitive's field; its `ScopeId` *is* the render/spatial dirty-flag
   target RFC-0001 §2.2 already defines.
2. A **computed memo** — a `let`/`fn` body. Its output is a cached `Value` plus
   a dirty bit; downstream scopes subscribe to *it*, and it subscribes to the
   `var`s it reads. Memos are **pull-based**: a read recomputes only if the
   dirty bit is set, then re-tracks. (Pull-based memoization, rather than eager
   push, is chosen to avoid the diamond-dependency glitch problem — see
   *Unresolved questions*.)
3. A **structural effect** — a `for` or `when` body (next subsection).

**Tracking.** While a scope is active, every `Signal::read` calls
`signal.track(current_scope)`, which (a) adds `current_scope` to that signal's
dirty-flag subscriber vector and (b) records the signal in the scope's own
dependency set. When the scope finishes, its dependency set is the exact set of
sources that can dirty it — discovered, not declared.

**Update.** On `var` mutation, the existing §2.2/§3.3 dirty-flag collection runs
unchanged; it now finds the subscriber links the tracking pass installed. Dirty
value-binding scopes are re-evaluated on the next `EvaluatorTick` (re-walk that
one AST sub-expression, rewrite that one primitive field, recompute the dirty
rect). Dirty memos are marked, recomputed lazily on next read. **Dynamic
dependencies** are handled the standard Solid/reactively way: before
re-evaluating a scope, its previous subscriptions are cleared, so a binding that
reads `a` on one tick and `b` on the next ends up subscribed to exactly what it
read this time.

The developer-visible result is precisely RFC-0001's stated goal ("mutations
touch only the affected GPU primitives, never re-run the whole component") — but
now reached without the developer arranging a single subscription.

### Dev-mode interpreter — structural reactivity: `for` / `when` (NEW)

`for` and `when` are not value bindings; they change the *shape* of the view
tree, which touches `Atlas` (layout) and the per-view arena lifetime (RFC-0001
§2). They are modeled as **structural effects**:

- `when cond { … } else { … }` opens a scope over `cond`. When `cond` flips, the
  effect mounts the appropriate child arena subtree (RFC-0001 §2 alloc) and
  drops the other (RFC-0001 §2 `O(1)` linear arena drop). No new memory
  primitive — mount/unmount is exactly what navigation push/pop already does.
- `for item in list { … }` opens a scope over `list`. Each element gets a child
  `ViewArena` with `item` bound in its `Env`. When `list` changes, children are
  added/removed.

**Phase 2 reconciliation is deliberately coarse:** on any change to a `for`'s
source list, drop and rebuild the whole list's child arenas (one linear arena
pass, cheap to implement, correct, and well within Dev mode's "fast enough to
feel instant" budget). Keyed minimal-diff reconciliation (preserve child arenas
across reorders, animate moves) is a refinement deferred to a follow-up, in the
same "start coarse, measure, refine" spirit the prior draft applied to
hot-reload granularity. This is called out as a real gap, not glossed over.

### Dev-mode interpreter — intrinsics and styling

**Intrinsics.** `Column`, `Row`, `Text`, `Button`, `TextField`, … are a fixed
table in `interp/intrinsics.rs` mapping reserved names to Rust functions that
read the element's positional `content` and `#[...]` attrs and push the
corresponding `BoxInstance`/`TextLine` entries onto the `RenderFrame`, using the
same `Atlas`/`BoxInstance`/`TextLine` APIs `Engine` already drives by hand. The
table now reads from *two* sources per element: `(...)` = primary content,
`#[...]` = modifiers. An element name that is not an intrinsic must resolve to a
`ViewDecl` in lexical scope, else `CompileError::UnknownView`. An attr name an
intrinsic does not recognize is `CompileError::UnknownAttribute` — a hard error,
**never silently ignored** (this is the explicit lesson from MAUI's
silently-degrading bindings; diagnostics must not lie).

**Style (`interp/style.rs`).** A `style { .title #[…] }` block is interpreted to
a `StyleMap: HashMap<Symbol, Vec<Attr>>` scoped to the View's `Env` — there is
**no global cascade**. An element's `#[style: .title]` resolves `.title` in the
enclosing View's `StyleMap` and merges its attrs into the element's attrs.
**Precedence is explicit:** inline `#[...]` attrs override class attrs of the
same name; later classes override earlier ones if a list is given. For Phase 2,
style values are **static** (no `var` reads inside a `style` block); dynamic
style values are deferred (see *Unresolved questions*), so the style table needs
no reactive scope.

### Integration with `Engine` (unchanged contract, reaffirmed)

The `LogicRuntime` / `start_logic_from_view` contract from the prior draft is
**unchanged** and, if anything, more clearly necessary now. The interpreter
holds live `Signal` handles *and* a logic-thread-local `current_scope`, so it
must stay `!Send` and never leave the logic thread:

```rust
// byard-core — implementors may freely be !Send.
pub trait LogicRuntime {
    fn evaluate_tick(&mut self, frame: &mut RenderFrame, dirty_targets: &[TargetId]);
}

impl Engine {
    pub fn start_logic_from_view<F>(build: F) -> Result<JoinHandle<()>, ByardError>
    where
        F: for<'a> FnOnce(&'a ViewArena) -> Box<dyn LogicRuntime + 'a> + Send + 'static,
    { /* spawn thread, build interpreter inside it, loop evaluate_tick */ }
}
```

The `Send + 'static` bound is on `build` (the recipe, which closes over a plain
owned `CompiledView` with no `Signal`s in it), never on the `!Send`
`LogicRuntime` it produces. `byard-core` defines `LogicRuntime` and the generic
`start_logic_from_view`; `byard-compiler` is the only crate that ever constructs
a `build` closure. The dependency edge stays `byard-compiler → byard-core`,
never the reverse. (See the prior draft for the full argument; nothing about
adopting Lume changes it.)

### Hot-reload boundary (updated for `var`)

On a `.byd` change, re-parse the affected `View` and structurally diff the new
`ViewDecl` against the running one:

1. **Reactive-compatible.** The `var` / `let` / `inject` / parameter lists are
   unchanged in shape (names + positions + declared types); only expressions,
   elements, control flow, or the `style` block differ. Action: keep the live
   `ViewArena`/`Signal` instances — match each `var` in the new `ViewDecl` to the
   running `Env` entry by `(position, name)` and **rebind the existing `Signal`
   handle** rather than calling `Signal::new_in` again (which would reset state
   to the new initializer and defeat the patch). Re-evaluate the non-`var`
   members on the next tick; the read-tracking pass re-derives subscriptions
   automatically as it goes, so reload "just works" for the reactive graph with
   no special-casing. No restart, no flicker.
2. **Structure-incompatible.** The `var` / `let` / parameter / `inject` list
   changed shape (added, removed, reordered, or retyped an entry). Action: tear
   down the View's arena subtree via the same linear arena-drop pass RFC-0001 §2
   defines for normal unmount, and rebuild from the new AST — a fresh mount, one
   tick early.

This is the prior draft's proposal with `signal` → `var` substituted; the
mechanics are identical because `var` lowers to the same `Signal`. The
file-watcher still runs off the logic thread and ships only plain `Send +
'static` data (`CompiledView` or `CompileError`) over a channel the logic thread
drains once per tick before the tick body; a failed parse keeps the
last-known-good `CompiledView` running and surfaces the error as a diagnostic.
The model is JetBrains Compose Hot Reload's: preserved state = `Signal` values,
fallback = arena teardown/rebuild.

### Grammar (versioned — full Lume surface)

```ebnf
view_decl   := "View" IDENT "(" param_list? ")" "{" member* "}"
param_list  := param ("," param)*
param       := IDENT (":" type)?
member      := var_stmt | let_stmt | fn_decl | inject_stmt
             | element | for_stmt | when_stmt | style_block | expr_stmt

var_stmt    := "var" IDENT (":" type)? "=" expr
let_stmt    := "let" IDENT (":" type)? "=" expr
fn_decl     := "fn" IDENT "(" param_list? ")" ("->" type)? "=>" expr
inject_stmt := "inject" type "as" IDENT

element     := IDENT ("(" arg_list? ")")? attr_block? element_tail?
element_tail:= "{" member* "}" | "=>" expr
attr_block  := "#[" attr_list? "]"
attr_list   := attr ("," attr)*
attr        := prop_attr | event_attr
prop_attr   := IDENT ":" expr                 // property — binds a value
event_attr  := IDENT ("(" IDENT ")")? "=>" expr   // engine event — maps to an action

for_stmt    := "for" IDENT "in" expr "{" member* "}"
when_stmt   := "when" expr "{" member* "}" ("else" "{" member* "}")?

style_block := "style" "{" style_rule* "}"
style_rule  := "." IDENT attr_block

arg_list    := arg ("," arg)*
arg         := (IDENT ":")? expr

type        := IDENT ("<" type ("," type)* ">")?
expr        := primary (postfix | binary | ternary | assign)*
primary     := INT | FLOAT | STRING | IDENT | array | "(" expr ")" | lambda
array       := "[" (expr ("," expr)*)? "]"
lambda      := "(" param_list? ")" "=>" expr | "|" param_list? "|" expr
postfix     := "++" | "--" | "." IDENT | "(" arg_list? ")"
assign      := ("=" | "+=" | "-=") expr
ternary     := "?" expr ":" expr
STRING      := '"' (CHAR | "{" expr "}")* '"'
```

This is roughly twice the prior draft's grammar. The growth is the explicit cost
of adopting full Lume now (the alternative — "core reactive now, defer
`for`/`when`/`style`" — was considered and rejected per the project decision;
see *Rationale*). Operators (`%`, `&&`, `||`, comparisons) and richer types are
out of scope for Phase 2 and require a grammar-amendment RFC.

---

## Implications and complications

This section exists because adopting Lume genuinely collides with the existing
RFCs in places. None are blockers, but each is a real cost or a decision that
must be ratified.

### 1. RFC-0001's surface examples are superseded — erratum required

RFC-0001 §"`byld` at a glance" uses `signal clicks = 0` and packs props into
`Column(gap: 12, bg: …, radius: 16, p: 20)`. That is no longer valid `byld`.
RFC-0001 is **Active**, so it needs a one-paragraph erratum pointing at this RFC
for the canonical surface, with the corrected snippet:

```byld
View UserCard() {
    var clicks = 0
    inject AppEnvironment as env
    Column #[gap: 12, bg: env.theme.surface, radius: 16, p: 20] {
        Text("Clicks: {clicks}") #[typo: m3.titleLarge]
        Button("Action") => clicks++
    }
}
```

Recommendation: add an "Amended by RFC-0002" note to RFC-0001 rather than
editing its body, preserving the historical record.

### 2. The interpreter grows a reactivity subsystem (the bulk of new work)

The prior draft's interpreter was a pure tree-walker with *no read tracking*.
Automatic reactivity adds `interp/reactive.rs`: the read-tracking scope stack,
computed-memo pull-evaluation, dynamic-dependency clearing, and structural
effects for `for`/`when`. This is the single largest delta and the main schedule
risk. It is **feasible** precisely because it reuses RFC-0001 §2.2's existing
`Signal` dirty-flag vector as the subscription store — but it is net-new code
with subtle correctness conditions (see complication 5).

### 3. Grammar / parser surface roughly doubles

New tokens (`#[`, `]`, `[`, `++`/`--`, `=`/`+=`/`-=`, `<`/`>`, `|`, `var`/`let`/
`fn`/`for`/`in`/`when`/`else`/`style`), new productions (element three-zone form,
control flow, style block, arrays, ternary, assignment). Pratt absorbs the
expression-operator growth gracefully (this was the original justification for
choosing Pratt), but the statement-level recursive-descent surface and the
intrinsic argument-handling both grow. Still hand-written, still the RFC-0001
diagnostic-quality bet — just bigger.

### 4. `for` is structural reactivity, not value reactivity

`for`/`when` touch `Atlas` and arena lifetimes, not just primitive fields. Phase
2 ships the *coarse* version (drop-and-rebuild the list on any change). Anyone
expecting keyed reconciliation, preserved scroll state, or list-move animations
in Phase 2 will not get them; that is a deliberate, documented deferral.

### 5. Reactive correctness conditions that did not exist before

Three correctness questions are introduced by automatic reactivity and must be
answered during implementation, not assumed:

- **Glitch-freedom / diamonds.** If `c = a + b` and both `a` and `b` derive from
  one source, a naive push can recompute `c` twice or with a stale operand.
  **Resolved in D1** (Mark-and-Pull with idempotent marking, the tick as the
  consistency boundary, and a per-tick epoch guard).
- **Dynamic dependencies.** A binding whose read set changes between ticks (e.g.
  behind a `when`) must clear stale subscriptions before re-tracking, or it will
  leak subscribers and over-update. The clearing step is specified above but is
  easy to get wrong; covered by D1's fixture tests.
- **Untracked reads.** Some reads must *not* subscribe (e.g. reading a `var`
  only to seed another `var`'s initializer once). **Resolved in D2**:
  `untrack(expr)` ships in Phase 2.

### 6. `let` means two things, and that must be observable

`let` is a *const* if it reads no `var`, a *reactive computed* if it does. This
is convenient but the dual nature must be **diagnosable** — the LSP/Phase 3 must
be able to tell the developer "this `let` is reactive because it reads `query`,"
or debugging "why did this recompute?" becomes guesswork. The read-tracking pass
already has the information; it just needs to be surfaced. **Resolved in D3**
(reactivity recorded in per-View resolved metadata, emitted to the LSP JSON).

### 7. Optional typing now, real types needed for Phase 4

The Dev interpreter is dynamically typed (values are engine values); annotations
are checked locally where present and inferred for LSP metadata. But Phase 4's
`byld → Rust` transpiler will need *sound enough* types to emit Rust. So the
local inference designed now must be a deliberate subset of what the transpiler
will require, or Phase 4 reopens the type system. This is a forward-compatibility
constraint on a Phase 2 component, flagged so it is not discovered late.
**Resolved in D9** (annotations mandatory on `View`/`fn` signatures; `var`/`let`
infer locally from their initializer; scalar string type is `Str`, not `Text`).

### 8. `#[...]` overload with Rust attributes — a feature with one caveat

`#[...]` reading as "configuration" in both `.byd` and `.rs` is intentional and
good for a Rust-first author. The one caveat: error messages and docs must be
careful to say "byld attribute block" vs "Rust attribute macro" so a newcomer
grepping for `#[` does not conflate the two. Purely a documentation discipline,
not a technical conflict.

### 9. Performance framing must be even louder than before

Automatic reactivity routes *every* `var` read through `track()` in Dev mode.
That overhead is irrelevant (Dev is not the throughput path) and **zero in
Prod** (Phase 4 transpiles the discovered subscriptions into static wiring). But
because the mechanism is now implicit, the "do not benchmark Dev mode and
conclude Byard is slow" warning from the prior draft applies doubly — an
implicit cost is easier to misattribute than an explicit one.

---

## Drawbacks

- **Largest single-RFC scope in the project so far.** Adopting full Lume in
  Phase 2 (vs. the deferred-features alternative) front-loads the grammar, the
  reactivity subsystem, and the structural-effect machinery into one deliverable.
  Mitigated by the coarse-first tactics (drop-and-rebuild `for`, whole-View-body
  reload patching) but the surface area is real.
- **Tree-walking + read-tracking is slower than the prior pure tree-walker**,
  which was already slower than bytecode/native. Acceptable only because Dev mode
  is explicitly not the throughput path — must be documented loudly (see
  complication 9).
- **Reactive correctness is subtle** (complication 5). The prior draft's
  interpreter could be proven correct by inspection; the reactive one needs a
  dedicated test strategy (property tests over read/mutate sequences, diamond
  fixtures, dynamic-dependency fixtures).
- **No lossless/rowan tree** still — multi-error recovery exists, but trivia is
  not preserved, so Phase 3's LSP may need a second lossless pass over this same
  grammar.
- **`for` reconciliation is coarse** in Phase 2 (complication 4).

---

## Rationale and alternatives

**Why adopt Lume wholesale rather than keep RFC-0001's `signal` surface?**
Because the project's decision is that `byld` should *do the reactive bookkeeping
for the developer*. Explicit `signal`/`.get()`/`.set()` is the "too
programmational" failure mode the Lume study identified across React hooks and
hand-wired signal APIs. RFC-0001 already names Solid.js as the reactivity
reference; automatic read-tracking is simply Solid's model taken to its
conclusion at the language surface instead of the library API.

**Why full Lume in Phase 2 instead of a deferred core?** Two narrower options
were on the table: (a) `var` auto-reactivity + `#[...]` now, defer `for`/`when`/
`style` to grammar-amendment RFCs; (b) only swap the reactivity model, keep
`(...)` props. Both were rejected: the project wants the *whole* Lume surface
validated by a real interpreter before the Phase 4 transpiler locks onto it, and
the anti-nesting promise (`for`/`when`) and the styling story (`style {}`) are
load-bearing parts of "agradable" that a half-language cannot demonstrate.

**Why reuse `Signal`'s dirty-flag vector as the subscription store rather than a
new reactive graph type?** Because RFC-0001 §2.2 already built a subscription
mechanism; a parallel graph would duplicate it and risk divergence. Automatic
reactivity is an *observation* layer over the existing primitive, not a
replacement.

**Why pull-based memos for `let`/`fn`?** Glitch-freedom with the least
machinery. Eager push needs topological scheduling to avoid double-computation
and stale reads on diamonds; pull-on-read with a dirty bit gives correctness for
free at the cost of laziness, which Dev mode can afford.

**Why coarse `for` reconciliation first?** Same reason the prior draft chose
coarse hot-reload: keyed diffing is a real subsystem with no payoff until real
usage shows the coarse version stutters. Start coarse, measure, refine.

**Why keep `View` rather than Lume's `component` keyword?** Engine coherence.
The core already speaks `ViewArena`, `ViewDecl`, `RenderFrame` of views; renaming
the surface keyword would desync the language from its own runtime vocabulary for
no benefit.

**Why not a parser generator / lossless tree / bytecode VM?** All three settled
in the prior draft and RFC-0001; adopting Lume does not reopen them.

---

## Prior art

- **Solid.js / reactively.** Fine-grained automatic reactivity via read-tracking
  and pull-based memos — the direct model for `interp/reactive.rs`. RFC-0001
  already cites Solid as the reactivity reference; this RFC operationalizes it.
- **Svelte 5 runes / Vue Composition API / Angular signals.** The 2026 industry
  convergence on tracked fine-grained reactivity that motivated dropping the
  explicit `signal` keyword.
- **JetBrains Compose Hot Reload.** State-preservation-on-reload with a
  full-rebuild fallback — the model for the hot-reload boundary.
- **PEP 701 (Python f-strings).** Re-tokenizing interpolated `{...}` segments —
  the precedent for the string-interpolation handling (unchanged).
- **matklad, "Simple but Powerful Pratt Parsing."** The recursive-descent +
  Pratt split this parser uses, now carrying more operators.
- **SwiftUI / Flutter / QML / .NET MAUI.** The DX study behind Lume: SwiftUI's
  type-checker-gives-up and MAUI's silently-degrading-bindings are the explicit
  reasons for "diagnostics must never lie" (hard `UnknownAttribute` errors);
  Flutter's nesting hell is the reason for first-class `for`/`when`; the
  CSS/Tailwind dilemma is the reason for the scoped `style` block.

---

## Resolved design decisions (Phase 2, final)

All twelve questions formerly in *Unresolved questions* are resolved here. Each
decision below is authoritative for Phase 2 implementation. Where a proposed
answer was amended for correctness, the amendment is called out in **bold**.

### D1 — Reactive update discipline: Mark-and-Pull (synchronous mark, lazy pull)

Adopted formally. On a `var` mutation, a **synchronous dirty-mark cascade**
propagates through the subscriber graph (memos and value bindings); **no value
is recomputed during the mark phase**. On the next `EvaluatorTick`, value
bindings re-evaluate their expressions; reading a dirty memo triggers that memo
to **lazily** re-walk its AST subtree, refresh its cached `Value`, and clear its
flag. This resolves diamonds without a runtime topological scheduler.

Three refinements added for correctness (each gets a fixture test before
`interp/reactive.rs` is written):

- **Idempotent marking.** The cascade stops at any node already marked dirty —
  a node that is already dirty cannot become "more dirty," so its subtree is not
  re-traversed. Without this, a wide diamond degrades the synchronous mark from
  O(affected nodes) to O(paths), which is exponential in the worst case.
- **The tick is the consistency boundary.** All mutations that land between two
  ticks accumulate marks *before* any pull happens at the next tick. Because
  every pull therefore observes a fully-settled mark set, no binding can read a
  half-updated graph — this is what makes the scheme glitch-free, and it must be
  stated as an invariant, not left implicit.
- **Per-tick epoch guard.** Each value binding carries a `last_evaluated_epoch`;
  a binding re-evaluates at most once per tick even if reached through several
  dirty paths. A `debug_assert!` flags any reactive cycle (a memo transitively
  reading itself), which this grammar cannot express today but a future `fn`
  amendment could.

### D2 — `untrack` escape hatch: included in Phase 2

Confirmed for Phase 2. Without it, seeding one binding from a one-shot read of
another would install a perpetual, unwanted subscription. `untrack(expr)` is a
**reserved intrinsic** (recognized by name in `interp/eval.rs`, not a user
`View`): the interpreter saves `current_scope`, sets it to `None`, evaluates
`expr` so every `Signal::read` inside skips subscriber registration, then
restores the previous scope. The save/restore uses an **RAII guard** so the
scope is restored even on an early return or a debug panic, and `untrack` nests
correctly (restores to the *previous* scope, not unconditionally to `Some`).
Syntactically it is an ordinary call (`primary "(" arg_list ")"`), so the
grammar needs no new production — only the reserved-name check.

### D3 — `let` reactivity visibility: derived metadata, not a mutated AST

Adopted with one amendment. The proposal stored an `is_reactive: bool` *on the
`Member::Let` AST node*, toggled during cold evaluation. **Amended:** the AST
stays immutable owned data (the property this RFC relies on for clean
hot-reload diffing — a mutated-during-eval AST would diff against itself
incorrectly after a reload). Instead, reactivity is recorded in a side-table in
the per-View *resolved* metadata (`ResolvedView.reactive: HashSet<Symbol>`),
populated when the read-tracker observes that a `let`/`fn` initializer read at
least one `Signal`. The compiler emits this set into the JSON metadata file the
LSP consumes, so the editor can visually distinguish dynamic computed bindings
from constant ones (subtle color/icon) and answer "why is this reactive?". The
determination extends to `fn`, not just `let`.

### D4 — Intrinsic attribute contract: strict zone separation + Levenshtein hints

Adopted. **`(...)` carries only the primary semantic content** (the plain text
of a `Text`, the base label of a `Button`); spatial, style, and event
properties live **exclusively** in `#[...]`. Each intrinsic declares a fixed
arity for its positional content; a wrong count is `CompileError::ArityMismatch`
(e.g. `Column("x")` — `Column` takes zero positional args). An unrecognized
attribute is a hard `CompileError::UnknownAttribute` — **never silently
ignored** (the MAUI lesson) — and the message suggests the nearest valid
attribute by **Levenshtein distance**, emitting a "did you mean `gap`?" hint
when the best candidate is within distance `≤ 2` (and strictly less than half
the typo's length, so short names don't match everything).

**Property vs event separator (D4-bis).** Inside `#[...]`, the separator encodes
the kind of attribute — a deliberate, zero-token distinction so that, at a
glance, `:` is appearance and `=>` is behavior:

- **Property** (`name: value`) binds a value: `gap: 12`, `bg: 0x222222`,
  `focused: editing`, `style: .title`. This includes function-valued callback
  props passed to a child `View` (`onPick: (e) => picked = e.value`) — passing a
  function *value* is still a value binding.
- **Engine event** (`name => action`, or `name(e) => action` to bind the
  payload) maps an event the engine dispatches to an action: `tap => clicks++`,
  `pointer_move(e) => cursor = e.pos`. Event names are **bare** — no `on`
  prefix, because `=>` already marks them as events. This is the same `=>` as
  the element-tail shorthand (`Button("+") => clicks++` is just the primary
  `tap` event hoisted). `name => expr` desugars to `name: (payload) => expr`
  internally; the full event catalog and payload types are defined in RFC-0003
  §4–§5.

An unrecognized property *or* event name is still `CompileError::UnknownAttribute`
with the Levenshtein hint above.

### D5 — `style` precedence and scope: static, three-layer, last-write-wins

Confirmed static for Phase 2 (reading a `var` inside `style {}` is a hard
error). Resolution order over one element is linear and cumulative, lowest to
highest priority:

1. The framework's **default typed base style** for that intrinsic.
2. **Local `style` classes** referenced via `#[style: …]`, merged in the order
   the classes are *listed on the element* (left-to-right; later wins). *(This
   is a deliberate choice over "style-block declaration order": element-list
   order is what CSS-in-JS conventions lead developers to expect. The two
   coincide for the common single-class case; the rule only matters when an
   element lists several classes.)*
3. **Inline `#[...]` attributes** on the element — the last word; they override
   any colliding property from layers 1–2.

### D6 — `CompileError` location — **amended to preserve the layering rule**

The proposal placed `CompileError` in `byard-compiler/diagnostics.rs` (correct)
but wrapped it as `ByardError::Compiler(CompileError)`. **That wrapper is
unsound:** `ByardError` is defined in `byard-core` (RFC-0001 §8), so a variant
naming `CompileError` would force `byard-core` to depend on `byard-compiler` —
the exact reverse of the one-way edge this whole RFC commits to, and the same
trap RFC-0002's prior draft already flagged. `byard-core` must stay agnostic to
compiler failures.

**Resolution:** `CompileError` lives entirely in `byard-compiler/diagnostics.rs`
and `byard-core` gains **no** compiler-specific variant. Unification happens one
layer up, where it belongs. Two acceptable forms, pick per ergonomics:

- *(Recommended)* The **application/runtime crate** — which already depends on
  both `byard-core` and `byard-compiler` — owns a top-level error enum:
  `enum AppError { Engine(ByardError), Compile(CompileError) }`. `byard-core`
  never learns the compiler exists.
- If a single `ByardError` surface is genuinely wanted at the boundary, add a
  **type-erased** variant in `byard-core` —
  `ByardError::External(Box<dyn std::error::Error + Send + Sync + 'static>)` —
  which carries a `CompileError` without naming it. `byard-core` stays agnostic;
  the consumer downcasts if it needs the concrete type.

Either way the dependency edge stays `byard-compiler → byard-core`, never the
reverse.

### D7 — `for` reconciliation upgrade trigger: churn-gated, not size-gated

Ship the blunt strategy first: any mutation of the iterator destroys and rebuilds
the whole `for` arena subtree. Migrate a given `for` to keyed diffing only when
**both** churn and cost show up — not on size alone, because a 500-row list built
once and never mutated is perfectly happy with the coarse path. Concretely, the
trigger is: list length `N > 50` **and** the list mutates frequently (a measured
rebuild rate above a threshold the Engine telemetry tracks, e.g. rebuilds per
second over a rolling window), **or** the Engine's frame telemetry records
sustained drops below 60 FPS attributable to per-tick allocation pressure from
that `for`. Size with no churn never triggers it.

### D8 — Dynamic `style` values: deferred to Phase 3

Formally deferred to Phase 3 (alongside LSP maturation). Allowing `var` reads
inside style classes would require wrapping the attribute dispatcher in an extra
reactive scope, spending CPU in Dev mode for a feature with no Phase 2 consumer.
Static-only keeps style resolution entirely outside the reactive graph (zero
tracking overhead), consistent with D5.

### D9 — Phase 4 inference subset — **amended for a type/intrinsic name clash**

Adopted with one fix. The subset is deliberately **not** Hindley-Milner:

- **`View` parameters and `fn` signatures (params + return) require explicit
  type annotations.** This gives the read-tracker and the Phase 4 transpiler a
  fixed point at every boundary.
- **`var` / `let` infer strictly from their initializer expression**, locally:
  integer literal → `i64`, float → `f64`, string → `Str`, bool → `bool`, array
  → `List<T>` with `T` from the (homogeneous) elements, call → the callee's
  declared return type, member access → the field type from the controller
  metadata. A non-inferable initializer (the empty array `[]`, a heterogeneous
  array) **requires an annotation** (`var items: List<Str> = []`) rather than
  guessing. This lets the Phase 4 transpiler emit flat Rust (`let mut clicks =
  0i64;`) with no ambiguity.
- **Amendment — `Text` is a `View`, not a type.** The earlier examples wrote
  `fn greeting() -> Text`, but `Text` is already the name of the text intrinsic,
  so using it as the string scalar type is an ambiguous collision. The scalar
  string type is **`Str`**; `Text` refers only to the intrinsic view. (All
  examples in this RFC and the design study should be read with `Text`-the-type
  replaced by `Str`.)

### D10 — File-watcher: dedicated OS thread + `notify`, coalescing `bounded(1)`

Adopted with two robustness refinements. The watcher uses **`notify` on a
dedicated, isolated OS thread**, *not* `relay.rs`'s Tokio pool — keeping it off
the async pool avoids latency and contention with user I/O. On a disk-change
event it parses the AST on that thread (in parallel with the logic thread) and
hands the resulting `CompiledView` (or `CompileError`) to the logic thread over
a **`crossbeam-channel::bounded(1)`**, drained at the start of the next tick.
Refinements:

- **Debounce raw FS events.** Editors save via write-temp-then-rename and often
  emit several events per save; a short debounce (≈50–100 ms, e.g.
  `notify-debouncer-mini`) collapses them into one parse.
- **Latest-wins coalescing.** With `bounded(1)`, if the developer saves several
  times faster than the logic thread drains, the watcher must **replace** the
  pending value rather than block (the watcher thread never stalls, and only the
  newest `CompiledView` is ever applied). The interner is process-global and
  thread-safe, so parsing off-thread is sound.

### D11 — Multiple `View`s per `.byd` file: allowed, diffed per `ViewDecl`

Allowed. A file is `file := view_decl*`. Hot-reload does **not** invalidate the
whole file; it maps each change to the unique `Symbol` of its `ViewDecl` and
applies D-style (the §"Hot-reload boundary" two-case diff) **per View**. Editing
the internal logic of one secondary View touches only that View's arena and
definition, leaving the rest of the file's live state intact. Two cross-View
cases are made explicit: renaming a `View` reads as delete-old + add-new (the
old instances unmount, the new name mounts fresh); changing a `View`'s
*signature* (params/`inject`/`var` shape) forces a structure-incompatible rebuild
at every call site that mounts it, since those call sites' child arenas no longer
match.

### D12 — `StrLit` interpolation nesting limit: hard cap at 3 levels

A strict limit of **3 levels** of interpolation nesting (`"L1 { "L2 { "L3" }" }"`)
guards the parser against stack overflow from accidental recursion. Exceeding it
is a hard `CompileError::StringNestingTooDeep`, not a panic or a silent
truncation. The lexer's fixed-size two-state stack is sized to **detect** the
violation (capacity for depth 4 so the 4th level is observed and rejected
cleanly, rather than overflowing the array). Beyond three levels the visual
cleanliness and DX are gone anyway, so the cap costs nothing real.

---

## Future possibilities

- **Phase 4 transpiler (`byld` → native Rust).** Inherits the reactive semantics
  defined here: the discovered subscription graph becomes static wiring, the
  pull-based memos become const-folded computations, and the `#[...]`/`style`
  surface lowers to direct `BoxInstance`/`TextLine` construction — zero runtime
  reactivity cost, the whole point of the model.
- **Keyed `for` reconciliation** with preserved child arenas and move animation.
- **Lossless/rowan CST** as a second pass over this grammar for Phase 3's LSP
  (incremental reparse, format-on-save).
- **Richer expression operators and types** (`&&`, `||`, comparisons, `%`,
  generics beyond one level) as incremental grammar-amendment RFCs.
- **Dynamic, reactive `style` values** once the static version is proven.
- **`untrack` / explicit effect blocks** if real usage shows automatic tracking
  is too eager in some patterns.
