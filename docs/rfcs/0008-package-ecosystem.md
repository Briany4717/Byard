# RFC-0008: Package Ecosystem — Multi-File Modules, Dependencies, and Distribution

- **Status:** Active — implemented 2026-07-04. Pillars A/B/C/E landed;
  Pillar D (asset distribution) remains a design proposal. Open decisions
  D-F…D-K are resolved; the first out-of-tree package (`byard-material`)
  exercises the pipeline end to end.
- **Author(s):** Briany4717
- **Created:** 2026-06-24
- **Last updated:** 2026-07-04
- **Depends on:**
  - **RFC-0007** (User-View Instantiation) — a `View` call must expand before an
    *imported* `View` can mean anything. Hard prerequisite.
  - RFC-0001 §1 (two-layer rule — `.byd` has no I/O), §3.1/§7.3 (pipelines,
    Dev/Prod), §9 (strict, layered crate dependency graph)
  - RFC-0002 **D10** (file-watcher, `LatestWins`, debounce), **D11** (per-`ViewDecl` reload)
  - RFC-0005 (intrinsic catalog — packages ship `View`s, never new intrinsics)
  - RFC-0006 (CLI — `byard new/dev/check/build`, `manifest.rs`)
- **Provides:** a Flutter/`pub`-style modular distribution model so libraries such
  as a Material or Liquid-Glass component set can be declared, fetched,
  resolved, namespaced, and consumed — reproducibly.

---

## Implementation status (2026-07-04)

| Pillar | State | Where |
| --- | --- | --- |
| A — `use` + module resolver | **Landed** | `Token::Use`, `UseDecl`, `byard_compiler::resolve` (graph, cycle detection, span-rebasing `SourceMap`, flatten into RFC-0007's `ViewTable`) |
| B — Explicit namespacing | **Landed** | Alias/selective forms, canonical `pkg.View` names, `NameCollision`/`UnknownImportSymbol`/`DuplicateViewName` diagnostics |
| C — Dependency manager | **Landed** | Strict `[dependencies]` parsing, `byard add`/`install`/`get`, `byard.lock` (commit + `sha256:` content hash), `~/.byard/cache`, transitive path/git deps |
| D — Asset distribution | Design proposal | Unchanged below — fonts/icons/themes wait on RFC-0009 + asset-loading integration |
| E — Multi-file hot-reload | **Landed** | Watcher covers project dir + `path` deps (D-J: yes); any change re-runs the resolver; cache packages not watched |

Two deliberate deviations from the text below: sibling files of one package
share a **flat namespace** (no relative imports at all), and
`byard add <bare-name>` resolves through a small built-in index that writes an
explicit pinned source into the manifest, as the honest stopgap for the deferred
registry.

---

## Summary

This RFC adds the four pillars a package ecosystem needs, on top of RFC-0007's
single-file instantiation: (A) an `import` surface and a **module resolver** that
turns a set of `.byd` files into one validated module graph; (B) **explicit
namespacing** so two packages may both define `Button`; (C) a **dependency
manager** in `byard-cli` (`byard add` / `byard get`, `[dependencies]` in
`byard.toml`, and a content-hashed **`byard.lock`** for reproducible builds);
and (D) **asset distribution** (fonts, icons, themes) handled as *engine policy*,
not as `.byd`-author file I/O — consistent with RFC-0001 §1 and its
asset-loading-as-engine-policy precedent. Git-first acquisition; a central
registry is explicitly deferred.

---

## Motivation

After RFC-0007, a `View` composes within a file but the compiler still consumes a
**single** entry file: `parser::parse` returns a flat `Vec<ViewDecl>` from one
source, and `byard-cli/src/manifest.rs` parses only `[project].entry`, warning
on and ignoring every other key (so `[dependencies]` is silently dropped today).
There is no `import`, no resolver, no namespacing, and no notion of an external
artifact. Reusable component libraries — the entire value proposition of a
"frontend ecosystem" — are impossible. This RFC closes that gap while protecting
two invariants the project will not trade away: **reproducibility** (a lockfile
from day one) and the **two-layer rule** (`.byd` never does I/O).

---

## Design tension to resolve up front: imports must not smuggle I/O into `byld`

The intuitive syntax `import "pkg:material/buttons/raised_button.byd"` encodes a
*filesystem path and a resolution strategy as a string literal inside surface
`byld`*. That sits uneasily with RFC-0001 §1 ("`.byd` files … contain no network
calls, no file I/O") and with the principle that asset loading is engine policy,
not an author choice. Two coherent directions:

- **Path-string imports** (Dart-like): familiar, but bakes path/resolution
  semantics into the language surface and invites authors to think in files.
- **Package-qualified symbol imports** (recommended): the *manifest* declares
  packages; a `.byd` file imports **symbols from a named, already-resolved
  package**, e.g. `use material as m` / `use material.{RaisedButton}` — the
  string path and the I/O live entirely in the compiler/CLI, never in `byld`.

This RFC proposes the package-qualified form and logs the choice as a decision
(D-F). Either way, **resolution is a compile-time concern owned by the resolver
and the CLI**, and `byld` source never expresses a filesystem read.

---

## Reference-level explanation

### Pillar A — Import surface and module resolver

- **Lexer/parser (RFC-0002).** Add the `use` keyword and an `ImportDecl` AST node
  parsed only at file top (before any `View`). Keep the recursive-descent parser
  hand-written (RFC-0001 §7.2) — error quality over a generated grammar.
- **Module graph.** A resolver builds a DAG of `.byd` modules: nodes are files,
  edges are `use` dependencies. It must:
  - resolve a package-qualified import to a concrete file in the resolved
    dependency set (Pillar C output), and a relative import to a sibling file;
  - **detect import cycles** between files and report them (mirrors RFC-0007 §4's
    intra-file call-graph cycle detection, one level up);
  - produce a single, deterministic, validated symbol table consumed by lowering.
- **Output is one program.** The resolver flattens the graph into the same
  `ViewTable` shape RFC-0007 introduced, so instantiation is unchanged — it
  simply now sees views sourced from many files. This keeps the per-frame render
  path and the AOT path (RFC-0001 §7.3) identical to the single-file case.

### Pillar B — Explicit namespacing

Collisions are inevitable (`material.Button` vs. `liquid_glass.Button`). Two
options were in the original proposal; this RFC recommends **explicit**:

- **Explicit alias (recommended, D-G).** Imports are namespaced
  (`use material as m;` → `m.RaisedButton`). Resolution is deterministic and
  order-independent — consistent with the project ethos that correctness and
  determinism precede convenience. Bare `use material.{X}` is allowed only when
  unambiguous; ambiguity is a `NameCollision` diagnostic demanding an alias.
- **Implicit-by-order (rejected).** Last-import-wins is order-sensitive and makes
  a name's meaning depend on edit history — a silent-correctness hazard.

### Pillar C — Dependency manager (`byard-cli`)

Extend RFC-0006's CLI and `manifest.rs`:

- **Manifest.** `byard.toml` gains a `[dependencies]` table. `manifest.rs`
  currently warns-and-ignores unknown keys — that path must instead parse
  dependencies and **error on malformed entries** (no silent drop):

  ```toml
  [project]
  name  = "my_app"
  entry = "main.byd"

  [dependencies]
  material     = { git = "https://github.com/byard-framework/material", rev = "..." }
  liquid_glass = { git = "https://github.com/.../liquid_glass", tag = "v1.2.0" }
  local_kit    = { path = "../local_kit" }
  ```

- **Acquisition — git-first (D-H).** `byard add <dep>` records a dependency;
  `byard get` resolves and fetches. Sources: `git` (pinned by `rev`/`tag`),
  `path` (local, for cooperative dev). A central hosted registry (npm/pub-style)
  is **deferred** — it is infrastructure, not language, and can be layered on the
  same lockfile later.
- **Reproducibility — `byard.lock` (D-I, the gap in the original proposal).**
  `byard get` writes a lockfile pinning every resolved dependency to an exact
  commit and a **content hash**. Builds resolve from the lock, not the manifest
  ranges; `byard get` is the only command that may change it. Without this,
  "version `^1.2.0`" is non-reproducible — unacceptable for a framework that
  sells determinism.
- **Cache + project layout.** Fetched packages live in a global user cache
  (e.g. `~/.byard/cache/`) and are referenced from the project via
  `.byard/packages/` (links or copies). `.byard/` is git-ignored; `byard.lock`
  is committed.

### Pillar D — Asset distribution as engine policy

Component libraries ship fonts, icons, and theme tokens. RFC-0001 §1 forbids
`.byd` I/O, and the engine already establishes that loading is **engine policy,
not a `byld`-author choice**. Therefore:

- A package declares its assets in *its* manifest; the resolver registers them
  into an **asset table** keyed by package-qualified virtual id (e.g.
  `material/fonts/Roboto`).
- The engine's text/texture subsystems (glyphon / cosmic-text, the
  `TextureSampler` pipeline, RFC-0001 §3.1) consume assets through that table.
  The `.byd` author references a token (`typo: m.titleLarge`), never a path.
- Theme tokens extend the existing theme provenance model, namespaced
  per package so two themes coexist.

### Pillar E — Multi-file hot-reload

RFC-0006/D10 wires a single-file watcher delivering one `ParsedFile` over a
`LatestWins` channel; RFC-0007 §5 added per-instance invalidation. This RFC
generalizes the watcher to the **module graph**:

- Watch every file in the resolved graph (project sources; optionally `path`
  dependencies for cooperative dev — D-J, recommended **yes**).
- On a change, re-parse the affected file, recompute the affected slice of the
  graph, and reuse RFC-0007's transitive instance invalidation to patch only the
  affected subtrees — preserving state per D11 / RFC-0003's gesture gate.
- Fetched (immutable, locked) cache packages are **not** watched.

---

## Open decisions

- **D-F — Import surface.** Package-qualified symbol imports (`use`) vs.
  path-string imports. *Recommendation:* `use`, keeping I/O out of `byld`.
- **D-G — Namespacing.** Explicit alias (recommended) vs. implicit-by-order
  (rejected). Ambiguity ⇒ `NameCollision` diagnostic.
- **D-H — Acquisition.** Git + path first; hosted registry deferred. Define the
  pin model (`rev`/`tag` required for `git`).
- **D-I — Lockfile.** `byard.lock` format (commit + content hash), who writes it,
  resolve-from-lock build semantics.
- **D-J — Cooperative dev watch.** Whether `path` dependencies are hot-reloaded
  in `byard dev`. *Recommendation:* yes.
- **D-K — Version resolution.** With git-first, do we need SemVer range solving
  now, or pin exact refs and defer a solver to the registry phase?
  *Recommendation:* defer the solver; exact refs + lock first.
- **D-L — Asset manifest schema.** How a package declares fonts/icons/themes and
  how the resolver validates them against the engine's asset-loading model.

---

## Drawbacks

- **Resolver + lockfile + cache is real infrastructure.** It is the largest
  non-engine surface in the project to date and adds CLI/network failure modes
  (auth, offline, corrupt cache) that need first-class error handling
  (RFC-0001's `Result`-not-panic ethos extends to the CLI).
- **Namespacing changes call sites.** `m.RaisedButton` is more verbose than a
  bare name; the cost is borne for determinism.
- **Asset table couples engine and package metadata.** The §9 crate-dependency
  graph (RFC-0001) must be respected — package metadata must not leak a back-edge
  into a subsystem module; it crosses through the existing boundary types.

---

## Rationale and alternatives

- **Why git-first, registry-later?** Matches early Go/Cargo. A registry is an
  ops commitment (hosting, availability, moderation, security) orthogonal to the
  language; the lockfile makes a later registry a drop-in source.
- **Why a lockfile from day one (not later)?** Reproducible builds are a
  correctness property, not an optimization. Adding it retroactively means a
  window where two developers get different trees from the same manifest — the
  opposite of what the project promises.
- **Why flatten the module graph into RFC-0007's `ViewTable`?** It keeps exactly
  one instantiation/lowering path for single-file and multi-file programs, so the
  AOT transpiler and the Dev interpreter never diverge.

---

## Prior art

- **Dart `pub` / `pubspec.yaml` + `pubspec.lock`.** The closest model: manifest +
  lock + hosted/git/path sources. `byld` adopts the lock and source taxonomy,
  rejects path-string imports.
- **Cargo (`Cargo.toml` + `Cargo.lock`, git/path deps before crates.io maturity).**
  Direct precedent for git-first-with-lock and the global-cache + per-project
  link layout.
- **npm/pnpm namespacing pain.** Motivation for *explicit* namespacing rather
  than flat global names with shadowing.

---

## Unresolved questions

- **Package versioning of `byld` itself / compatibility ranges.** How does a
  package declare which `byld` language/intrinsic-catalog version it targets
  (RFC-0005 is currently "closed per phase")?
- **Security & integrity.** Beyond content hashes, do git sources need signature
  or provenance verification before a registry exists?
- **Prod build packaging.** How fetched packages and their assets are embedded by
  the future `byard build` AOT path (RFC-0001 §7.3) — a Prod-mode sub-RFC.
- **Transitive dependency conflicts.** Two deps requiring incompatible versions
  of a third — resolution strategy is tied to D-K and likely a registry-phase
  concern.
