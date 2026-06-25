# Erratum to RFC-0001: `byld` surface syntax superseded by RFC-0002

- **Status:** Active erratum (amends, does not replace, RFC-0001)
- **Author(s):** Briany4717
- **Created:** 2026-06-20
- **Applies to:** RFC-0001 §"`byld` at a glance", §"Rust controller", and any inline `byld` snippet in RFC-0001.
- **Authority:** RFC-0002 (Lume surface, decisions D1–D12), RFC-0003 (events, decisions E1–E8).

---

## Why this erratum exists

RFC-0001 introduced `byld` with conceptual snippets written before the language
was defined. RFC-0002 then adopted the **Lume** design as `byld`'s definitive
surface and RFC-0003 fixed the interactive-event syntax. Three of RFC-0001's
illustrative choices are now **invalid `byld`** and would not parse under the
RFC-0002 grammar. RFC-0001 is **Active**, so rather than edit its body (which
would erase the historical record of the design's evolution), this erratum lists
the corrections and the canonical replacements. Where RFC-0001's prose and the
snippets disagree, **this erratum and RFC-0002/0003 win.**

This erratum changes **surface syntax only**. Every architectural claim in
RFC-0001 — the four subsystems, the zero-GC arena model, `Signal<T>` and its
dirty-flag vector, the multi-pipeline renderer, the §4.2 spatial hash grid, the
concurrency model, `PlatformHost`, and the Dev-interpreter / Prod-transpiler
split — stands unchanged. In fact the Lume surface is implemented *on top of*
those primitives without altering them (RFC-0002 §"automatic reactivity" reuses
the §2.2 dirty-flag vector as its subscription store).

---

## Corrections

### C1 — `signal` keyword removed; state is `var` / `let`

RFC-0001 declared reactive state with a `signal` keyword. **There is no `signal`
keyword in `byld`.** Reactive sources are declared with `var` (mutable) and
computed/constant bindings with `let`; reactivity is **automatic** (the compiler
discovers dependencies by read-tracking — RFC-0002 D1). The lowering is
unchanged: a `var` still becomes a `Signal::new_in(arena, …)` exactly as the old
`signal` did; only the keyword and the wiring (now implicit) changed.

### C2 — Properties move from `(...)` to `#[...]`; three-zone elements

RFC-0001 packed decorative/spatial properties into the call parentheses
(`Column(gap: 12, bg: …, radius: 16, p: 20)`). Under Lume an element has **three
zones** (RFC-0002 D4):

- `(...)` — **primary positional content only** (a `Text`'s string, a `Button`'s
  label).
- `#[...]` — **properties / config / events**.
- `{ }` — **children**.

### C3 — Event syntax: bare names with `=>`, no `on` prefix

RFC-0001 wrote handlers as `onClick: () => …`. Under RFC-0003 (decision
"Attribute syntax" / D4-bis), engine events are **bare-named** and use the `=>`
separator (`tap => …`, `pointer_move(e) => …`); `:` is reserved for value
bindings (properties, including reactive props like `focused:` and
function-valued callback props). Event names drop the `on` prefix because `=>`
already marks them.

### C4 — Scalar string type is `Str`, not `Text`

Any RFC-0001 prose using `Text` as a type name is wrong: `Text` is the text
**intrinsic view**. The scalar string type is **`Str`** (RFC-0002 D9).

---

## Canonical replacements

### RFC-0001 §"`byld` at a glance" — corrected

```byld
View UserCard() {
    var clicks = 0                       // was: signal clicks = 0   (C1)
    inject AppEnvironment as env

    Column #[gap: 12, bg: env.theme.surface, radius: 16, p: 20] {   // C2
        Text("Clicks: {clicks}") #[typo: m3.titleLarge]             // C2
        Button("Action") => clicks++                                // C3
    }
}
```

`View` remains the fundamental unit (kept over Lume's `component` for coherence
with `ViewArena`/`ViewDecl` — RFC-0002 Rationale). `inject` is unchanged. Wrapper
views (`Padding`, `Align`, …) still do not exist; spatial properties are still
named attributes — now in `#[...]` rather than `(...)`.

### RFC-0001 §"Rust controller" — unchanged

```rust
#[byard_controller]
pub struct NetworkController { base_url: String }

impl NetworkController {
    pub async fn fetch_user(&self, id: u64) -> Result<User, ApiError> { /* … */ }
}
```

The Rust side is **not** affected. Note the now-deliberate visual rhyme: `#[...]`
means "configuration" in both languages — properties/events in `.byd`, attribute
macros in `.rs`. They never collide (different files, different grammars).

---

## Unresolved item from RFC-0001 newly closed

RFC-0001's *Unresolved questions* listed **"`byld` syntax stabilisation — a
grammar RFC must be written before the parser is implemented."** That item is now
**closed**: RFC-0002 fixes the versioned grammar for the full Lume surface, and
RFC-0003 fixes the event/attribute syntax. Mark it resolved in RFC-0001's
tracking, pointing here.

RFC-0001's other open items are unaffected by this erratum: the accessibility
`AccessBridge` sub-RFC, the testing strategy for the double-buffered model, the
MSRV policy, and the hot-reload boundary (the last now answered by RFC-0002's
§"Hot-reload boundary" and RFC-0003 E5).

---

## Summary table

| RFC-0001 as written | Canonical `byld` | Authority |
|---|---|---|
| `signal clicks = 0` | `var clicks = 0` | RFC-0002 C1 / D1 |
| `Column(gap: 12, bg: …, radius: 16, p: 20)` | `Column #[gap: 12, bg: …, radius: 16, p: 20]` | RFC-0002 C2 / D4 |
| `Button("Action", onClick: () => clicks++)` | `Button("Action") => clicks++` | RFC-0003 C3 |
| `Text("…", typo: m3.titleLarge)` | `Text("…") #[typo: m3.titleLarge]` | RFC-0002 C2 / D4 |
| `Text` used as a string type | `Str` | RFC-0002 C4 / D9 |
| `View`, `inject`, `#[byard_controller]` | unchanged | — |
