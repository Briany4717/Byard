# RFC-0002: `byld` Compiler Pipeline — Lexer, Parser, and Dev-Mode Interpreter

- **Status:** Draft
- **Author(s):** <!-- fill in GitHub username -->
- **Created:** 2026-06-17
- **Last updated:** 2026-06-17

---

## Summary

This RFC proposes the Phase 2 deliverable from the project Roadmap: a new
`byard-compiler` crate that turns `byld` source into a running view tree.
Three stages, in one crate: a `logos`-generated lexer, a hand-written
recursive-descent parser whose expression layer uses Pratt parsing, and a
tree-walking AST interpreter that runs in Dev mode by binding directly against
the existing `Signal` / `ViewArena` / `EvaluatorTick` API that Phase 1 already
built and shipped. It also proposes a concrete answer to RFC-0001's open
"Hot-reload boundary" question, and fixes a versioned grammar for the `byld`
subset RFC-0001 already shows in its example — satisfying the "a grammar RFC
must be written before the parser is implemented" item from RFC-0001's
Unresolved Questions. Production transpilation (`byld` → native Rust, Phase
4) is explicitly **out of scope**.

## Motivation

Phase 1 closed with a working engine core — `evaluator`, `atlas`, `encoder`,
`relay` are all wired into `Engine`, proven end-to-end by the `hello_world`
example (issue #31's audit confirmed this). But `hello_world.rs` authors its
view by hand in Rust: a `Vec<BoxInstance>`, a `Vec<TextLine>`, and a
`set_label_text` call from a pointer-input handler. There is no `byld`
surface anywhere in the codebase today.

RFC-0001's central claim is that the declarative layer and the systems layer
never live in the same file — `byld` describes UI, Rust controls the world,
and the two communicate through compile-time bindings, not a widget tree or
serialization boundary. None of that exists yet. Until a compiler exists,
that claim is unverified, and every contributor who wants to write a real
`byld` view is blocked.

Dev mode (interpreted) is proposed before Prod mode (transpiled, Phase 4) for
two reasons. First, an interpreter is the faster path to a *working* language
— hot-reload and fast iteration matter more than throughput while the grammar
itself is still unstable. Second, locking a transpilation target before the
grammar has been exercised by a real interpreter would mean compiling against
a moving target twice. RFC-0001 §7.3 already scopes Dev = interpreter,
Prod = transpiler; this RFC only specifies the Dev half.

## Guide-level explanation

A `.byd` file is compiled in three stages, all inside the new
`byard-compiler` crate:

1. **Lexer** (`lexer/`) — turns source text into a `Token` stream. Generated
   from a `#[derive(Logos)]` enum; whitespace and line comments are skipped
   at this stage via `#[logos(skip ...)]`.
2. **Parser** (`parser/`) — turns the token stream into a typed AST
   (`ViewDecl`, `Stmt`, `Expr`, ...). Declarations and statements (`View`,
   `signal`, `inject`) are parsed by ordinary recursive descent, one function
   per production. Expressions (member access, named-argument calls, postfix
   `++`, arrow lambdas) are parsed with a Pratt (precedence-climbing) table,
   because a flat one-function-per-precedence-level grammar gets unwieldy
   once that many operator forms coexist.
3. **Dev interpreter** (`interp/`) — walks the AST directly, with no
   intermediate bytecode. `signal` statements allocate real `Signal` handles
   in a real `ViewArena`; view-call statements allocate nested arenas exactly
   the way `Engine::start_logic`'s hand-rolled tick loop already does it
   today. There is no separate "interpreter runtime" object model — the
   interpreter's values *are* engine values.

A contributor's day-to-day loop: edit a `.byd` file, a file watcher notices,
the changed `View`'s AST is re-parsed and diffed against the version
currently running. If only signal-affecting expressions changed, the existing
`Signal`/`ViewArena` instances are kept and just re-evaluated on the next
tick — no restart, no flicker. If the View's shape changed (a `signal`, a
parameter, or an `inject` was added, removed, or retyped), the View's arena
subtree is torn down and rebuilt from scratch, the same single linear
arena-drop pass RFC-0001 §2 already defines for ordinary unmount. This is the
proposed resolution to the "Hot-reload boundary" unresolved question (see
Reference-level explanation and Unresolved Questions below).

## Reference-level explanation

### Crate layout

This matches the layout already sketched as a comment in the workspace
`Cargo.toml`:

```
crates/byard-compiler/src/
├── lib.rs          — public API: compile_view(), interpret_tick(), CompileError
├── lexer/          — logos-generated Token enum + driver
├── parser/
│   ├── ast.rs       — typed AST node definitions
│   ├── expr.rs      — Pratt expression parser (nud/led tables)
│   └── error.rs     — parse diagnostics (span, message, hint)
├── interp/
│   ├── env.rs        — variable/binding environment, `inject` resolution
│   ├── eval.rs       — AST → engine calls (Signal read/write, ViewArena alloc)
│   └── reload.rs      — hot-reload AST diff → restart-or-patch decision
└── diagnostics.rs   — shared error/span types
```

`byard-compiler` depends on `byard-core` (for `Signal`, `ViewArena`,
`EvaluatorTick`) but not the reverse — the same relationship `byard-platform`
already has with `byard-core`. Note this is an *inter-crate* dependency
direction, not a literal application of RFC-0001 §9: §9 governs the module
graph *inside* `byard-core` (no subsystem module imports another directly,
everything crosses `frame.rs`). `byard-compiler` is a separate crate sitting
above `byard-core`, not a fifth subsystem inside it. The principle this RFC
is actually following is the same layering discipline §9 embodies, applied
one level up — see "Integration with `Engine`" below for why `byard-core`
specifically must **not** gain a dependency on `byard-compiler` in the
other direction.

### Data structures

- `Span { start: u32, end: u32 }` — byte offsets into the source file, `Copy`.
  Every AST node carries one, so diagnostics and (later) LSP features can
  point at exact source ranges.
- `Token` (logos derive) — one variant per terminal: `View`, `Signal`,
  `Inject`, `As`, `Ident(Symbol)`, `IntLit(i64)`, `StrLit` (raw, unparsed —
  interpolation is resolved by the parser, not the lexer; see below),
  `LParen`, `RParen`, `LBrace`, `RBrace`, `Comma`, `Colon`, `Arrow` (`=>`),
  `PlusPlus`, `Dot`, etc. Logos 0.16 lexers return `Result<Token,
  Token::Error>` per token (no manual `#[error]` variant is required); the
  driver wraps a lex failure as `CompileError::UnexpectedChar { span }`.
- `Symbol` — `Arc<str>`, returned by a single process-global, content-
  addressed interner (`HashSet<Arc<str>>`, append-only, populated behind a
  lock taken only on a *new* identifier text — repeat identifiers, which
  dominate after the first few parses, hit a read path that returns a
  clone of the existing `Arc`). Two `Symbol`s for the same text are the
  same allocation, so equality is `Arc::ptr_eq` in the common case with a
  content-equality fallback, and identity is stable across reparses by
  construction — no per-session integer table whose IDs shift depending on
  encounter order. `Arc` (not `Rc`) is not a style choice: `CompiledView`
  must be `Send + 'static` to cross from the file-watcher thread to the
  logic thread (see "Integration with `Engine`" below), and every `Symbol`
  inside its AST has to satisfy that bound. `Rc<str>` would silently make
  the whole AST `!Send` and the channel send would fail to compile. Memory
  growth is bounded by the number of *distinct* identifier strings seen in
  the process lifetime — hundreds for a UI source tree, not a leak in
  practice — so the interner is never freed and does not need to be.
- AST (`ast.rs`), sketch:
  ```rust
  struct ViewDecl { name: Symbol, params: Vec<Param>, body: Block, span: Span }
  enum Stmt {
      Signal { name: Symbol, init: Expr, span: Span },
      Inject { ty: Symbol, name: Symbol, span: Span },
      ViewCall { name: Symbol, args: Vec<Arg>, block: Option<Block>, span: Span },
      Expr(Expr),
  }
  enum Expr {
      IntLit(i64, Span),
      StrLit(Vec<StrPart>, Span),       // StrPart::Text(String) | StrPart::Interp(Box<Expr>)
      Ident(Symbol, Span),
      Member { base: Box<Expr>, field: Symbol, span: Span },
      Call { callee: Box<Expr>, args: Vec<Arg>, span: Span },
      Lambda { params: Vec<Symbol>, body: Box<Expr>, span: Span },
      PostfixIncr { target: Box<Expr>, span: Span },
      Error(Span), // parse-error sentinel — see "Parser" below
  }
  struct Arg { name: Option<Symbol>, value: Expr }
  ```
  The AST owns all of its data (no borrows into the source text) — this is
  what makes re-parsing on hot-reload simple to diff against the previous
  tree without lifetime entanglement with the file's contents.

### Lexer

A single `#[derive(Logos)]` enum, compiled by logos into one DFA at build
time — dispatch is a table lookup per character, no backtracking, no runtime
regex engine. `#[logos(skip r"[ \t\r\n]+")]` and `#[logos(skip
r"//[^\n]*")]` remove whitespace and line comments before the parser ever
sees them.

String interpolation (`"Clicks: {clicks}"`) is *not* handled by a dedicated,
top-level lexer mode (no `logos` mode-switching on the token stream). The
lexer emits a single `StrLit` token holding the literal's raw inner text and
span. The parser, while building the `Expr::StrLit` node, splits that raw
text into literal segments and `{...}` segments, and for each `{...}`
segment re-invokes the same lexer+parser pipeline recursively to produce a
nested `Expr`. This mirrors CPython's PEP 701 approach to f-strings
(re-tokenizing the interpolated spans rather than inventing a parallel
interpolation-aware lexer mode) and keeps the `Token` enum itself simple. A
stateful logos mode that pushes/pops on `{`/`}` for the *entire* lexer was
considered and rejected — see Rationale.

**Locating the literal's end is not a plain regex.** The grammar already
allows a string literal inside an interpolation block, recursively
(`STRING := '"' (CHAR | "{" expr "}")* '"'`, and `expr`'s `primary` includes
`STRING`) — e.g. `"Hola {signals.get_string("usuario")}"`. A naive
string-literal regex (`r#""[^"\\]*(?:\\.[^"\\]*)*""#`) closes the token at
the first unescaped `"`, which is the one opening `"usuario"`, not the
literal's real end. `StrLit` is therefore produced by a `#[logos(callback =
...)]` function, not a `#[regex(...)]` pattern: the callback scans forward
from the opening `"` maintaining a small fixed-depth stack of two states,
`InString` and `InBraceExpr`. It pushes `InBraceExpr` on an unescaped `{`
seen while the top is `InString`, pushes `InString` on an unescaped `"` seen
while the top is `InBraceExpr` (entering a nested literal), and pops on the
matching `}` or closing `"`. The literal ends at the `"` that empties the
stack. A plain brace-depth counter without this two-state distinction is
*not* sufficient — a literal `}` inside a nested string (e.g.
`"{f("}")}"`) would decrement the counter and close the interpolation early,
because the counter has no way to know that `}` was string content rather
than a real brace. Nesting depth is small for UI source (a handful of
levels at most), so the stack is a fixed-size array, not a `Vec` — no
allocation inside the lexer's hot path.

### Parser

Declaration and statement level (`View`, `signal`, `inject`, view-call) is
ordinary recursive descent: one function per production, deciding which
branch to take by peeking the next token.

Expression level uses Pratt (precedence-climbing) parsing. This grammar
already mixes member access (`env.theme.surface`), named-argument calls
(`Button("Action", onClick: () => clicks++)`), postfix increment
(`clicks++`), and arrow lambdas at different binding strengths — a
one-function-per-precedence-level recursive-descent grammar would need
several mutually recursive functions today and a new one for every future
operator. Pratt's `nud`/`led` split scales by adding table entries:

```rust
fn parse_expr(&mut self, min_bp: u8) -> Expr {
    let mut lhs = self.parse_nud();              // literal, ident, paren, lambda
    loop {
        let Some((op, l_bp, r_bp)) = self.peek_led() else { break };
        if l_bp < min_bp { break; }
        self.advance();
        lhs = self.parse_led(op, lhs, r_bp);      // `.`, `(`, `++`, binary ops
    }
    lhs
}
```

`nud` handles atoms: literals, identifiers, parenthesized expressions, and
lambdas (`() => expr` / `(params) => expr`). `led` handles everything that
follows an existing expression: `.` (member access — very high binding
power, consumes one trailing identifier), `(` (call — parses a named/
positional argument list), `++` (postfix — emits `PostfixIncr` without
consuming an rhs), and ordinary binary operators if and when the grammar
grows them.

Error recovery: any `parse_*` function that hits an unexpected token records
a `CompileError` and substitutes `Expr::Error(span)` rather than aborting the
whole parse. This lets one parse pass collect multiple diagnostics instead of
stopping at the first error — in line with RFC-0001's stated diagnostic-
quality goal. This RFC deliberately does **not** propose a lossless/rowan-
style tree (every byte preserved for IDE round-tripping) — see Rationale and
Drawbacks.

### Dev-mode interpreter

A tree-walking interpreter, no bytecode compilation step. Dev mode's job is
correctness and hot-reload latency, not throughput — Phase 4's transpiler is
already the throughput path, so Dev mode never has to compete with it. The
interpreter's environment binds straight to the Evaluator subsystem types
Phase 1 already shipped:

`Env`, the interpreter's per-View binding environment, is a flat
`Vec<(Symbol, Value)>`, not a hash map. A View's `signal`s and parameters
are pushed once when the View mounts and the vector is never truncated
until the View itself unmounts (its `ViewArena` drops) — there is no
sub-block scoping below "one `Env` per View instance" in this grammar,
because the only construct that introduces a new scope is a nested
`view_call_stmt`, and that already allocates its own child `ViewArena` and
`Env`. Lookup walks the vector in reverse (supporting shadowing for free)
and is O(n) in the View's own binding count, which is single digits in
practice — faster and more cache-friendly than hashing for that size, and
allocation-free after the vector's capacity stabilizes across the first
few ticks. This flat-stack design is only sound *because* nothing below
the View level needs its own scope; if a future grammar amendment adds
block-scoped statements inside a View body, this invariant has to be
revisited before that lands.

- `signal x = <init>` compiles to `let sig = Signal::new_in(arena,
  eval(init))`; the resulting handle lives in the interpreter's per-View
  `Env`.
- Reading a signal-bound identifier evaluates to `sig.read(|v| ...)`.
  RFC-0001 §5.1 restricts `Signal` access to the logic thread, and the
  interpreter only ever executes as part of an `EvaluatorTick`, so this
  invariant holds without extra enforcement — the interpreter is born
  running on the logic thread, the same place `Engine::start_logic`'s
  hand-rolled tick closure already captures its `ReactiveLabel`/`Signal`
  state today (see RFC-0001 §5.1 and the `relay.rs` "Engine integration"
  doc, which documents why that closure cannot be `Send`).
- `inject T as name` resolves by walking up a separate parent-pointer chain
  of `Env` *references* (one borrow per ancestor View), not the flat Vec
  itself — each View's own arena and `Env` can die independently per
  RFC-0001 §2's per-view lifetime, so the chain borrows ancestor `Env`s
  rather than copying or merging them into the child's vector. This is a
  runtime lookup (React Context's model), not a Rust-level type lookup.
- A nested view-call statement (`Column(...) { Text(...) }`) allocates a
  child `ViewArena` scoped to that nested view, mirroring RFC-0001 §2's
  per-view arena lifetime — the child arena dies in the same linear pass as
  its parent, no GC, no separate lifetime management code in the
  interpreter.
- Lambdas (`() => clicks++`) are not Rust closures. They are AST subtrees
  held by reference inside `Env`, re-walked by the interpreter when the
  bound event (`onClick`) fires. This sidesteps `Send`/lifetime questions
  entirely inside the interpreter — there is no Rust closure capturing a
  `!Send` `Signal`, just an AST node evaluated in place. This only stays
  correct because `Env` is never truncated before its View's arena drops
  (see above) — a lambda captured against a View-level binding outlives
  any narrower scope by construction, since none exists. A
  `debug_assert!` in `interp/env.rs` should check this invariant directly
  rather than leaving it implicit.

### Integration with `Engine`, and intrinsic views

Two integration questions are easy to skip past while looking at the lexer/
parser/interpreter in isolation, but Phase 2 is not "done" until both are
answered, because without them the interpreter has no way to actually
produce a frame:

**How does `Engine` run a compiled View instead of hand-authored Rust?**
Today, `Engine::start_logic(instances, texts)` takes a `Vec<BoxInstance>` and
a `Vec<TextLine>` and, per the `relay.rs` "Engine integration" doc, builds
its tick closure *inside* the spawned thread because that closure captures a
`!Send` `Signal`-bearing `ReactiveLabel`. This RFC proposes a parallel entry
point built around a trait that deliberately carries **no** `Send` bound,
because the running interpreter holds `Signal` handles and must stay exactly
as `!Send` as today's hand-rolled closure:

```rust
// byard-core — implementors may freely be !Send; nothing here ever
// crosses a thread boundary after construction.
pub trait LogicRuntime {
    fn evaluate_tick(&mut self, frame: &mut RenderFrame, dirty_targets: &[TargetId]);
}

impl Engine {
    pub fn start_logic_from_view<F>(build: F) -> Result<JoinHandle<()>, ByardError>
    where
        F: for<'a> FnOnce(&'a ViewArena) -> Box<dyn LogicRuntime + 'a> + Send + 'static,
    {
        // spawns a thread via std::thread::Builder, constructs the
        // ViewArena inside that thread, calls build(&arena) there — the
        // resulting LogicRuntime (and the Signals it holds) never leaves
        // the thread — then loops `runtime.evaluate_tick(...)` the same
        // way today's loop runs its hand-written body.
        ...
    }
}
```

The `Send + 'static` bound belongs on `build`, not on `LogicRuntime` itself.
`build` only needs to be the *recipe* for constructing the interpreter's
`Env` (which closes over a `CompiledView` — plain, owned, `Send + 'static`
data with no `Signal`s in it) — that recipe is what crosses from the
caller's thread (or the file-watcher thread, see below) into the spawned
logic thread over the channel. The object the recipe *produces*, once
called inside that thread, is free to be `!Send`, exactly as `Env` needs to
be. A `LogicRuntime: Send` bound, which an earlier version of this proposal
carried, would be a contradiction in terms — no implementor holding live
`Signal` handles could ever satisfy it, since `Signal` access is restricted
to the logic thread by RFC-0001 §5.1. The bound has to live on the factory
that crosses the boundary, never on the thing built after it.

Critically, **this does not require `byard-core` to depend on
`byard-compiler`.** `byard-core` defines `LogicRuntime` and the generic
`start_logic_from_view`; `byard-compiler` is the only crate that ever
constructs a closure satisfying `build`'s signature (capturing a
`CompiledView` and producing an `Env`-backed `LogicRuntime` impl inside the
thread). `byard-core` never names `byard-compiler` or `ViewDecl`. The
dependency edge stays `byard-compiler → byard-core`, never the reverse,
exactly as `byard-platform → byard-core` already does for the same reason
(`PlatformHost` keeps `winit` out of `byard-core`; the same trick keeps
`byard-compiler` out of it too).

**How do `Column`, `Text`, and `Button` actually draw anything?** A
user-defined `View` (like RFC-0001's `UserCard`) has no Rust-side renderer —
it is pure composition. But its *body* bottoms out in calls to a small set of
built-in views (`Column`, `Text`, `Button`, and friends) that must lower to
real `BoxInstance`/`TextLine` writes on the `RenderFrame`, the same
primitives `Engine::start_logic`'s hand-written tick body constructs today.
This RFC proposes that these are **intrinsics**, not user `ViewDecl`s: a
fixed table in `interp/eval.rs` mapping a small set of reserved names
(`Column`, `Row`, `Text`, `Button`, ...) directly to Rust functions that read
the call's named arguments (`gap`, `bg`, `radius`, `p`, the text content,
`onClick`, ...) and push the corresponding entries into the frame, using the
same `Atlas`/`BoxInstance`/`TextLine` APIs `Engine` already uses by hand. A
`view_call_stmt` whose name matches an intrinsic dispatches there; any other
name must resolve to a `ViewDecl` in lexical scope, or the parser/interpreter
reports `CompileError::UnknownView`. Without this table, the grammar in this
RFC parses RFC-0001's `UserCard` example but the interpreter has no way to
turn `Column(...)`/`Text(...)`/`Button(...)` into pixels — this was a real
gap in the first draft of this RFC and is called out explicitly rather than
left implicit.

### Hot-reload boundary (proposed resolution to RFC-0001's open question)

On detecting a `.byd` file change, re-parse the affected `View` and diff the
new `ViewDecl` against the one currently running, by structure:

1. **Signal-update-compatible.** The `signal` list (names + types, by
   position) and the `inject`/parameter lists are unchanged; only
   expressions inside the body differ. Action: keep the live
   `ViewArena`/`Signal` instances — concretely, match each `signal` statement
   in the new `ViewDecl` to the corresponding entry in the running `Env` by
   `(position, name)` and rebind the *existing* `Signal` handle to it rather
   than calling `Signal::new_in` again (which would silently reset state to
   the new initializer and defeat the entire point of patching). Only the
   non-signal statements are re-evaluated against the existing `Env` on the
   next tick. No restart, no flicker.
2. **Struct-incompatible.** The `signal`, parameter, or `inject` list changed
   shape (added, removed, or retyped an entry). Action: tear down the
   View's arena subtree using the same linear arena-drop pass RFC-0001 §2
   already defines for normal unmount, and rebuild from the new AST — a
   fresh mount, one tick early.

This is a structural diff over two typed `ViewDecl`s, not a text diff, so it
slots into the existing arena-lifetime model without inventing a new memory
primitive: case 2 is simply "this view's arena dies and repopulates a tick
early," which Phase 1's unmount path already has to support. This is
presented as a proposal for review, not a settled answer — see Unresolved
Questions for the parts intentionally left coarse.

**Delivery into the logic thread and reload failure behaviour.** The
file-watcher runs outside the logic thread (its own thread, or a task on
`relay.rs`'s existing Tokio pool — see Unresolved Questions) and only ever
produces plain `Send + 'static` data: a freshly parsed `CompiledView`, or a
`CompileError`. It hands that value to the logic thread over the same kind
of channel `relay.rs` already uses for its event/I/O-result paths; the logic
thread drains this channel once per tick, alongside its existing event-queue
drain, *before* running the tick body. If the parse failed
(`CompileError`), the logic thread keeps running the last-known-good
`CompiledView` completely unchanged and the error is surfaced as a log/
diagnostic — a bad save must never crash the dev session, stall the render
thread, or silently roll back state.

### Grammar (versioned, scoped to RFC-0001's example)

```ebnf
view_decl   := "View" IDENT "(" param_list? ")" "{" stmt* "}"
param_list  := param ("," param)*
param       := IDENT (":" type)?
stmt        := signal_stmt | inject_stmt | view_call_stmt | expr_stmt
signal_stmt := "signal" IDENT "=" expr
inject_stmt := "inject" type "as" IDENT
view_call_stmt := IDENT "(" arg_list? ")" ("{" stmt* "}")?
arg_list    := arg ("," arg)*
arg         := (IDENT ":")? expr
expr        := primary (postfix | binary)*
primary     := INT | STRING | IDENT | "(" expr ")" | lambda
lambda      := "(" param_list? ")" "=>" expr
postfix     := "++" | "." IDENT | "(" arg_list? ")"
STRING      := '"' (CHAR | "{" expr "}")* '"'
```

This is intentionally the minimal grammar that parses RFC-0001's `UserCard`
example verbatim. Control flow (`if`/`else`, loops) and additional operators
are out of scope for this RFC and require a follow-up grammar-amendment RFC.

## Drawbacks

- Sub-lexing `{...}` interpolation segments (re-invoking the pipeline
  recursively per string literal) adds a small constant overhead and a bit
  of recursive-call complexity compared to a dedicated interpolation lexer
  mode. Locating each `StrLit`'s real end also requires a hand-written
  `logos` callback with a small nesting-depth stack instead of a plain
  regex (see "Lexer") — more code than a single `#[regex(...)]` line, and
  more surface for a subtle off-by-one in the push/pop logic than a regex
  would have.
- Tree-walking interpretation is slower than bytecode or transpiled native
  code. This is acceptable only because Dev mode is explicitly not the
  throughput path — but it must be documented loudly, so nobody benchmarks
  Dev mode and concludes Byard itself is slow.
- The hot-reload diff proposed here (signal/param/inject list unchanged ⇒
  patch, else ⇒ restart) is a coarse, whole-View-body granularity. It does
  not yet handle reordering signals, renaming a signal while preserving its
  meaning, or patching only the changed sub-expression inside an otherwise-
  compatible View. These gaps are real and called out explicitly in
  Unresolved Questions rather than glossed over.
- No lossless/rowan-style syntax tree is proposed here. Multi-error recovery
  exists (`Expr::Error` sentinels, collected diagnostics), but the AST does
  not preserve whitespace or trivia, so IDE round-tripping (format-on-save
  preserving exact formatting, incremental reparse per keystroke) is not
  free. Phase 3's LSP work may need a second, lossless pass over this same
  grammar — flagged honestly here instead of silently deferred.

## Rationale and alternatives

**Why not a parser generator (`pest`/`chumsky`)?** RFC-0001 already settled
this in its own Rationale section, citing diagnostic quality. This RFC
extends that decision down to the expression grammar specifically, choosing
Pratt parsing over either a parser-combinator expression grammar or a flat
precedence-climbing if-chain, because Pratt's table-driven `nud`/`led` split
is the standard way to keep a hand-written parser's expression layer
maintainable as operators are added — and this grammar already has four
different postfix/infix forms (`.`, `(`, `++`, `=>`) competing for precedence.

**Why not a lossless (rowan/cstree) tree from day one?** Considered and
rejected for Phase 2 specifically, not forever. Phase 2's only consumer is
the Dev interpreter, which needs a typed AST, not a lossless CST. Building a
red-green tree (green-tree builder, red-tree wrapper, typed AST view on top)
has real implementation cost with no payoff until Phase 3's LSP work actually
needs incremental reparsing and whitespace-preserving edits. Revisit
explicitly when Phase 3 begins — flagged in Future Possibilities.

**Why tree-walking and not a bytecode VM for Dev mode?** Dev mode's
performance budget is "fast enough to feel instant on save," not "fast
enough to ship." A bytecode VM is a second execution strategy to keep
correct, on top of Phase 4's eventual native transpiler — three strategies
(AST interpreter, bytecode VM, native transpiler) is one more than a young
codebase should maintain. Two (interpret now, transpile later) is the
minimum that covers both Dev and Prod.

**Why structural AST diffing for hot-reload, not a text/line diff?** A line
diff cannot distinguish "the user renamed a local variable" from "the user
changed what the signal represents" — exactly the ambiguity RFC-0001 left
open. Diffing the typed `ViewDecl` (signal list, param list, inject list)
gives a mechanical, principled answer to "restart or patch?" instead of a
heuristic over raw text.

**Why re-lex interpolation segments instead of a stateful lexer mode?**
logos supports mode-switching, but adding lexer-level state for a single
grammar feature (string interpolation) was judged not worth the complexity
when the parser can just recursively re-invoke the same pipeline on the
inner span. Keeps the lexer itself simple and dependency-free.

## Prior art

- **PEP 701 (Python f-strings).** Re-tokenizes interpolated `{...}` segments
  as a nested token stream rather than a dedicated interpolation-aware lexer
  mode — the precedent for this RFC's string-interpolation handling.
- **rust-analyzer / rowan.** Lossless red-green syntax trees, purpose-built
  for IDE-grade error recovery and incremental reparsing — the reference
  design for what Phase 3's LSP work will likely need, explicitly *not*
  adopted in this RFC (see Rationale).
- **matklad, "Simple but Powerful Pratt Parsing."** The standard modern
  reference for combining recursive descent (statements/declarations) with
  Pratt parsing (expressions) inside one hand-written parser — exactly this
  RFC's parser-layer split.
- **JetBrains Compose Hot Reload.** State-preservation-on-reload: capture
  state, reconstruct the component from new code, reattach preserved state,
  falling back to a full rebuild when a change can't be hot-swapped in
  place. The direct model for this RFC's Hot-reload boundary proposal —
  `Signal` state is the preserved state, arena teardown/rebuild is the
  fallback.
- **Solid.js / SwiftUI** (already cited in RFC-0001) — informed the
  `View`/`signal`/`inject` surface this parser accepts; not re-litigated
  here.

## Unresolved questions

- **Before merge:**
  - [ ] Does the grammar need `if`/`else` or loops inside a View body for
    Phase 2, or is that explicitly deferred to a follow-up grammar-amendment
    RFC? RFC-0001's example has neither; the grammar above currently omits
    them.
  - [x] **Symbol interning strategy.** Resolved: a single process-global,
    content-addressed `HashSet<Arc<str>>` interner (see "Data structures").
    This avoids both horns of the original dilemma — it never reassigns an
    identifier's identity between reparses (content-addressed, not
    encounter-order-addressed), and it never needs per-session freeing
    (growth is bounded by distinct identifier text, not by reload count).
    `Arc` rather than `Rc` because `Symbol` has to stay `Send` for
    `CompiledView` to cross the file-watcher → logic-thread channel.
  - [ ] Where does `CompileError` live relative to `ByardError`? A new
    `ByardError::Compile(...)` variant in `byard-core`, or a fully separate
    error type owned by `byard-compiler`? Given the dependency direction
    this RFC commits to (`byard-compiler → byard-core`, never the reverse —
    see "Integration with `Engine`"), `byard-core` cannot know about a
    compiler-specific error variant by name, which leans toward a separate
    type — but this should be settled explicitly.
  - [x] **Shape of the `byard-core` contract for "run a compiled view."**
    Resolved: a `LogicRuntime` trait with no `Send` bound, plus a generic
    `start_logic_from_view<F>` that takes a `Send + 'static` factory closure
    (`F: for<'a> FnOnce(&'a ViewArena) -> Box<dyn LogicRuntime + 'a>`). The
    `Send` bound lives on the factory, never on the runtime it produces —
    see "Integration with `Engine`" for why a `LogicRuntime: Send` bound is
    unsatisfiable by any implementor holding live `Signal` handles.
  - [ ] The intrinsic-view table (`Column`, `Row`, `Text`, `Button`, ...) is
    proposed but not specified here: exact argument names/types per
    intrinsic, and how an unknown named argument (e.g. a typo'd `gp:` for
    `gap:`) is reported — a compile error, or silently ignored? Needs a
    decision before `interp/eval.rs` is implemented.
- **During implementation:**
  - [ ] Hot-reload diff granularity: should patching ever apply inside a
    single View's body (only the changed sub-expression re-evaluated)
    rather than always re-running the whole tick function on any
    non-structural change? Proposal: start coarse (whole-body re-run),
    measure actual reload latency, refine only if it matters in practice.
  - [ ] File-watcher mechanism (`notify` crate vs. polling) and whether it
    runs on `relay.rs`'s existing Tokio I/O pool or needs its own thread.
  - [ ] One-View-per-file vs. multiple-Views-per-file as a convention —
    affects how the hot-reload diff is scoped (per-file or per-View).
  - [ ] Diagnostic rendering (terminal output vs. LSP) is out of scope for
    Phase 2 itself, but the `CompileError`/`Span` design here should be
    checked against Phase 3's LSP needs so it doesn't need a rewrite later.
  - [ ] `StrLit` callback nesting-depth limit: the lexer's two-state stack
    (`InString`/`InBraceExpr`) is a fixed-size array, not a `Vec`, to avoid
    allocation in the lexer's hot path. What depth is "enough" for real UI
    source, and what happens on overflow —
    `CompileError::StringNestingTooDeep` rather than a panic or silent
    truncation?

## Future possibilities

- A lossless/rowan-style CST as a second pass over this same grammar, once
  Phase 3's LSP actually needs incremental reparse and whitespace-preserving
  edits.
- A bytecode or SSA IR as an intermediate step before Phase 4's native
  transpiler, if direct AST-to-Rust-source generation proves too large a
  leap. This RFC's AST is designed to be a reasonable input to either path
  without a redesign.
- Richer, sub-expression-level hot-reload patching, once the coarse
  whole-View-body version proposed here has real usage data showing it's too
  coarse.
- `if`/`else`, loops, and other grammar constructs as incremental
  grammar-amendment RFCs once the `View`/`signal`/`inject` core proven here
  is stable.
