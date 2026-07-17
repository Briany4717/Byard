# RFC-0027: Data & Collection Operations — comparison, logic, string and list expressions

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-17
- **Last updated:** 2026-07-17
- **Depends on:** RFC-0002 (grammar, `Value`, reactivity D1, `Str` type D9), RFC-0003 (events / view mutation, E1 write-back, `Action`), RFC-0004 (reactive interpreter — read-tracking, memos), RFC-0018 (structural `for`/`when` in the render tree), RFC-0019 (callback props / `Fn`, lambda param inference E2).
- **Extends:** RFC-0002 (the expression grammar and `eval_binary`), RFC-0005 (`Text` interpolation gains scalar formatting).
- **Enables:** Todo apps, filterable/sortable lists, computed derived collections, boolean-driven `when`/ternary conditions. Unblocks every app that must *transform* state rather than replace it wholesale. Prerequisite for the data returned by RFC-0028 controllers to be useful in the view.

---

## Summary

The `byld` expression language today evaluates only four operators — `+ - * /`
on `Int`/`Float` — and has **no boolean, comparison, string, or collection
operations at all** (`interp/eval.rs::eval_binary`, `parser/ast.rs::BinOp`). A
`var` can hold a `Value::List`, and `for`/`when` can render it (RFC-0018), but
the language cannot *derive* one list from another, test `a == b`, negate a
`Bool`, or concatenate two `Str`s. This RFC adds the missing **pure, value-level
data layer**: comparison operators, short-circuiting logic, string
concatenation, a closed set of immutable list operations (`len`, index,
`push`, `removeAt`, `contains`, `map`, `filter`, spread/concat), and
**lambda expressions in value position** so `map`/`filter` can take a predicate.
Every operation is a *pure expression that returns a new value* — nothing
mutates in place — which keeps the Mark-and-Pull discipline (D1) and the
reference-free model (RFC-0003) intact.

```byld
View TodoList() {
    var todos: List = []
    var draft: Str = ""

    Column #[gap: 8, p: 16] {
        TextField #[value: draft, placeholder: "New task…"]
        Button("Add") => {
            todos = todos.push({ text: draft, done: false })   // list op
            draft = ""
        }

        let remaining = todos.filter(t => !t.done).len          // logic + list + lambda

        Text("{remaining} of {todos.len} left")                 // scalar interpolation

        for t in todos {
            Row #[gap: 8] {
                Toggle #[value: t.done]
                Text(t.text) #[color: t.done ? 0x888888 : 0x111111]  // comparison-free ternary today; `==` now available
            }
        }
    }
}
```

---

## Motivation

The [gap analysis](../../support/GAP_ANALYSIS_real_apps.md) identified three
structural blockers between Byard-the-visual-framework and Byard-the-app-
framework. This RFC closes the first and most self-contained one: **the language
cannot manipulate data.**

Concretely, in the current tree:

- **`eval_binary` handles only `Int`/`Float` arithmetic.** Any other operand
  combination falls through to `Value::Unit` (`interp/eval.rs:7490-7508`). So
  `"a" + "b"` is `Unit`, `todos + [x]` is `Unit`, `count == 3` is unparseable.
- **`BinOp` has exactly four variants** — `Add`, `Sub`, `Mul`, `Div`
  (`parser/ast.rs:369-378`). There is no `==`, `!=`, `<`, `<=`, `>`, `>=`,
  `&&`, `||`, or `!`.
- **There are no collection builtins** — no `len`, no indexing, no
  `push`/`filter`/`map` (grep of `interp/` returns none).

The smoking gun is `crates/byard-compiler/examples/reactive_demo.byd`: its
"Add Grace" button does not append — it **replaces** the list with a hard-coded
literal `["Ada", "Alan", "Grace", "Katherine"]`, and its toggle writes
`showList = showList ? false : true` instead of `!showList`, because neither
dynamic append nor boolean negation is expressible. A todo app — the simplest
"real" program — is impossible without this RFC.

This is deliberately the **cheapest** blocker to close: it touches only the
interpreter's expression evaluator and the parser's operator/postfix surface. It
does not touch concurrency, the relay, or the render pipelines.

---

## Guide-level explanation

### 1. Comparison operators → `Bool`

`==  !=  <  <=  >  >=` compare two values and produce a `Bool`.

- `Int`/`Float` compare numerically (with the same Int→Float promotion as
  arithmetic).
- `Str` compares by value (`==`/`!=` and lexicographic ordering).
- `Bool` compares with `==`/`!=` only.
- Two operands of **incompatible types** are a compile error
  (`CompileError::TypeMismatch`), not a silent `Unit` (INV-4: no silent
  failure). Comparing `List`/record values with `==` is a **structural**
  equality (element-wise), ordering (`<`) on them is a `TypeMismatch`.

### 2. Logical operators with short-circuit

`&&  ||  !` operate on `Bool`.

- `a && b` evaluates `b` **only if** `a` is `true`; `a || b` evaluates `b` only
  if `a` is `false`. Short-circuit is observable through read-tracking (RFC-0004):
  a `var` read only in the non-taken branch is **not** subscribed this tick, so
  the memo does not re-run when it changes — matching the reactive semantics of
  `when`.
- `!a` negates. This finally makes `!showList` legal.

### 3. String concatenation and scalar interpolation

- `Str + Str` concatenates. `Str + Int`/`Str + Float`/`Str + Bool` coerces the
  scalar to its display form and concatenates (so `"n=" + count` works), keeping
  parity with the `Text("{count}")` interpolation path (RFC-0005), which this
  RFC formalizes as the single scalar-formatting function `format_scalar`.
- Non-`Str` `+` where the **left** side is a `Str` triggers coercion; a `List`
  operand to `+` follows §4 (list concat), never string coercion.

### 4. List operations — pure, immutable, value-returning

Lists are values (`Value::List`). Every operation **returns a new list** and
never mutates an alias; the caller writes the result back to a `var`
(`xs = xs.push(y)`), exactly the E1 write-back model. This preserves
reference-freedom: there are no aliased mutable containers, so no hidden
observers, so Mark-and-Pull stays correct.

| Form | Result | Notes |
|---|---|---|
| `xs.len` | `Int` | property access, not a call |
| `xs[i]` | element | `IndexOutOfBounds` → `Unit` + logic-thread diagnostic (INV-4), never panic |
| `xs.push(v)` | new `List` with `v` appended | |
| `xs.removeAt(i)` | new `List` without index `i` | out-of-range → unchanged list + diagnostic |
| `xs.contains(v)` | `Bool` | structural equality (§1) |
| `xs.map(f)` | new `List` | `f` is a lambda (§5) |
| `xs.filter(f)` | new `List` | `f: T -> Bool` |
| `xs + ys` | concatenated `List` | both operands `List` |
| `[..xs, v]` | spread literal | sugar for `xs.push(v)` at literal sites |

The v1 set is deliberately closed (no `sort`/`reduce`/`find` yet — see Future
possibilities). It is exactly what a todo/list app needs.

### 5. Lambda expressions in value position

`map`/`filter` need a predicate. RFC-0019 already defines the `Fn(Args) -> Ret`
type and lambda **parameter** inference (E2) for callback *props*; this RFC lifts
lambdas to **general value position** so they can be arguments to collection
builtins:

```byld
todos.filter(t => !t.done)
todos.map(t => t.text)
```

The lambda body is a pure expression evaluated once per element on the logic
thread. It may read `var`s (tracked normally) and its parameter (`t`), but it may
**not** perform side effects (no assignment, no event action) — enforced at
compile time (`CompileError::EffectInPureLambda`), which keeps `map`/`filter`
referentially transparent and safe to re-run during a pull.

### 6. Records (object literals)

`{ text: draft, done: false }` in the todo example is a **record** — an ordered
set of named fields, added as `Value::Record(Vec<(Symbol, Value)>)`. Records are
values (immutable, structurally compared) with field access `r.field`. They are
the natural element type for lists of structured data and mirror the shape a
controller (RFC-0028) returns. Record field *update* is `{ ..r, done: true }`
(spread), returning a new record.

---

## Reference-level explanation

### 1. Grammar & AST changes (RFC-0002 extension)

`BinOp` gains: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`, `And`, `Or`. A new unary
`UnOp { Not, Neg }` node is added (`Neg` unifies with existing `-` where used as
prefix). Precedence, lowest→highest: `||` < `&&` < comparison < `+ -` < `* /` <
unary < postfix (`.`/`[]`/call). `&&`/`||` are right-recursive with
short-circuit lowering (not eager `eval_binary`).

New expression nodes:

```rust
Expr::Unary   { op: UnOp, rhs: Box<Expr>, span: Span }
Expr::Index   { base: Box<Expr>, index: Box<Expr>, span: Span }
Expr::Method  { recv: Box<Expr>, name: Symbol, args: Vec<Expr>, span: Span }
Expr::Field   { recv: Box<Expr>, name: Symbol, span: Span }   // r.field, xs.len
Expr::Lambda  { params: Vec<Symbol>, body: Box<Expr>, span: Span }
Expr::Record  { fields: Vec<(Symbol, Expr)>, spread: Option<Box<Expr>>, span: Span }
```

`Value` (in `interp/env.rs`) today has `Int`, `Float`, `Bool`, `Str`, `List`,
`Tuple(Vec<(Option<Symbol>, Value)>)`, `Fn(AstId)`, `Signal`, `Memo`,
`Theme`, and `Unit`. This RFC adds `Value::Record(SmallVec<[(Symbol, Value); 4]>)`.

`Record` is **distinct from the existing `Tuple`**: `Tuple` is the positional,
optionally-named aggregate produced by attribute-value syntax like `p: (vertical:
8, horizontal: 16)` (RFC-0005 `Len` pairs) — its fields are *positional first*
and it is not a general keyed data type. `Record` is a **name-keyed data
aggregate** for app state (`{ text, done }`), always accessed by field name, and
is the element type this RFC's list ops and RFC-0028's `HostValue` speak. Keeping
them separate avoids overloading `Tuple`'s layout/attribute role with data
semantics. `Value::Fn` already exists for callback props; lambdas reuse it
(`Fn(AstId)` pointing at the lambda body plus its captured param names).

INV-3 (AST immutable after parse) is preserved: these are new node kinds, not
mutations of existing nodes.

### 2. Evaluation

`eval_binary` is refactored into three total functions, each pure and
unit-testable, none panicking on user data (INV-4):

```rust
fn eval_arith(op: BinOp, a: Value, b: Value) -> Value;    // + - * /  (existing, unchanged)
fn eval_compare(op: BinOp, a: Value, b: Value) -> Value;  // == != < <= > >=  → Bool
fn eval_concat(a: Value, b: Value) -> Value;              // Str+*, List+List
```

`&&`/`||`/`!` are **not** in these tables — they are lowered as control flow in
the expression evaluator so the RHS is only evaluated (and only read-tracked)
when reached. The dispatch order for `+`: both `Int`/`Float` → `eval_arith`;
either `Str` or both `List` → `eval_concat`; else `TypeMismatch`.

Collection builtins are dispatched from `Expr::Method`/`Expr::Field` against a
small static table keyed by `(receiver kind, name)`. `map`/`filter` evaluate the
`Value::Fn` body per element in a child scope binding the lambda param; the
per-element evaluation is `untrack`-neutral (reads still subscribe, so a memo
over `xs.filter(...)` re-runs when any element changes — coarse but correct,
consistent with RFC-0018 D7's "coarse first").

### 3. Reactivity interaction (RFC-0004)

All of the above are ordinary reads inside a memo or a value binding. `let
remaining = todos.filter(t => !t.done).len` opens a memo (RFC-0004) that
subscribes to `todos`; writing `todos` marks it dirty; it is pulled next tick.
Short-circuit `&&`/`||` narrows the subscription set exactly like `when`, so
no over-subscription. No new reactive machinery is required — this RFC is
**purely additive to the evaluator** and reuses the existing subscription store.

### 4. Type inference (RFC-0002 D9 subset)

The Phase-4 inference subset annotates `View`/`fn` signatures and infers locals.
Comparison/logic yield `Bool`; `.len` yields `Int`; `.filter` preserves the
element type; a lambda's param type is inferred from the receiver's element type
(E2 mechanism, generalized). A `filter` predicate that does not yield `Bool` is
`CompileError::PredicateNotBool`.

### 5. Diagnostics (new `CompileError` variants — INV-5, defined in `byard-compiler`)

`TypeMismatch { op, lhs_ty, rhs_ty, span }`, `PredicateNotBool { span }`,
`EffectInPureLambda { span }`, `UnknownMethod { recv_ty, name, span }` (with
Levenshtein suggestion, matching the D4 `UnknownAttribute` treatment). Runtime
index/removeAt out-of-range is **not** a `CompileError` — it degrades to `Unit`/
unchanged and emits a logic-thread diagnostic, because the index may be a runtime
`var` (INV-4: no panic on user-derived data).

---

## Drawbacks

- **Grows the language surface.** Records, lambdas-in-values, and a method-call
  syntax are meaningfully more language than the pre-RFC arithmetic-only
  expression grammar. This is unavoidable for real apps but must be documented
  and LSP-supported.
- **Coarse reactive re-derivation.** `map`/`filter` re-run over the whole list
  when any dependency changes (RFC-0018 D7 coarse model). For very large lists
  this is wasteful until keyed reconciliation lands. Acceptable for v1.
- **No in-place mutation.** `xs.push(y)` allocates a new list. For large lists in
  hot loops this is a real cost; mitigated by the arena model and by keeping the
  op set small. In-place/persistent-data-structure optimizations are a future
  possibility, invisible to the surface syntax.

---

## Rationale and alternatives

**Why immutable, value-returning ops instead of `xs.push(y)` mutating in place?**
Reference-freedom (RFC-0003) is the load-bearing invariant of the whole reactive
model: there are no handles, so the only way state changes is a `var` write,
which the write-tracker sees. An in-place mutation of a `Value::List` behind a
`Signal` would change state **without** going through `write_signal`, defeating
Mark-and-Pull. Returning a new list and assigning it keeps every mutation on the
single audited path (E1). This is the SolidJS/Elm store model, chosen for exactly
this reason.

**Why lambdas only as pure expressions?** Allowing side effects inside
`map`/`filter` would make a pull non-idempotent (D1 requires idempotent marks and
safe re-runs). Restricting collection lambdas to pure expressions preserves that;
event actions remain the place for effects (RFC-0003 `Action`).

**Why not defer comparison/logic to a separate RFC?** They are the same
evaluator change, share the same diagnostics, and a todo app needs all of them
together (`filter(t => !t.done)`). Splitting would fragment one cohesive
interpreter change across two documents.

**Alternative rejected: a general expression VM (RFC-0014 bytecode).** That is a
performance vehicle for dev-mode, orthogonal to *which* operations exist. This
RFC defines the operations; RFC-0014 can later compile them faster.

---

## Prior art

- **SolidJS stores / Elm / Redux:** immutable updates, derive-don't-mutate. Direct
  model for §4.
- **Swift/Kotlin collection methods** (`map`/`filter`/`contains`): the closed,
  method-call collection surface.
- **Rust iterators:** the pure-transformation mental model (though eager here, not
  lazy — a UI list is small and fully materialized each tick).
- **Jetpack Compose `derivedStateOf`:** the memoized-derived-collection pattern
  `let remaining = todos.filter(...)` mirrors.

---

## Resolved questions

- **Before merge:**
  - [x] **Mutation model — in-place vs value-returning.** **Value-returning.**
    See Rationale; required by reference-freedom.
  - [x] **Equality on `List`/`Record`.** **Structural (element/field-wise).**
    Ordering (`<`) on them is a `TypeMismatch`.
  - [x] **`Str + scalar` coercion direction.** **Left-`Str` coerces the right
    operand** via the single `format_scalar` shared with `Text` interpolation;
    `scalar + Str` is also allowed and coerces the left. Two non-`Str` scalars
    keep arithmetic semantics.
  - [x] **v1 collection op set.** **`len`, index, `push`, `removeAt`, `contains`,
    `map`, `filter`, `+`/spread.** `sort`/`reduce`/`find`/`slice` deferred.
  - [x] **Lambda purity.** **Pure expressions only** in collection lambdas;
    effects are `EffectInPureLambda`.

- **During implementation:**
  - [x] **Precedence table.** As in §1; add a parser precedence-climbing test per
    operator pair. Ternary `? :` (already present) binds tighter than `||`.
  - [x] **`removeAt`/index bounds.** Runtime-degrade to `Unit`/unchanged +
    diagnostic (INV-4), never a `CompileError` and never a panic, because the
    index can be a runtime value.
  - [x] **Record key ordering.** Declaration order preserved (`SmallVec`), so
    structural equality is order-sensitive on construction but field access is
    by name.

---

## Future possibilities

- **`sort`/`reduce`/`find`/`slice`/`indexOf`** as a second closed batch once the
  v1 set proves the dispatch design.
- **Keyed `for` reconciliation** (RFC-0002 D7) consuming a stable-id lambda
  (`for t in todos key t.id`) so `map`/`filter`-derived lists diff instead of
  rebuild.
- **Persistent (structural-sharing) list backing** so `push` on a large list is
  O(log n) without changing the surface.
- **`Map`/`Set` value kinds** for keyed collections (e.g. entity caches from
  RFC-0028 controllers).
- **Numeric formatting spec** in interpolation (`{price:.2}`) built on
  `format_scalar`.
