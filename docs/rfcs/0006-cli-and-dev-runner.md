# RFC-0006: `byard` CLI and Dev Runner — Project Scaffolding, Live Reload, and the Developer Loop

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-06-21
- **Last updated:** 2026-06-21
- **Depends on:**
  - RFC-0001 §6 (`PlatformHost` trait, `Engine`), §7 (byld compiler pipeline, Dev/Prod modes)
  - RFC-0002 **D10** (file-watcher, `notify`, `bounded(1)` latest-wins, debounce), **D11** (per-`ViewDecl` reload), hot-reload boundary (reactive-compatible vs. structure-incompatible)
  - RFC-0003 §per-tick ordering (events dispatched before reload)
  - RFC-0004 §structural reactivity (mount/unmount on reload)
  - RFC-0005 (intrinsic catalog, project's `.byd` surface)
- **Implements:**
  - RFC-0001 §7.3 "Dev mode" end-to-end (currently exercised only through an internal example, not a shipped tool)
  - RFC-0002 D10's open wire: `start_watcher` + `LatestWins<ParsedFile>` channel exists in `byard-compiler/src/interp/reload.rs` but is not connected to any live runner
  - The "IMPL-23" item from the implementation log: notify file watcher concrete binding

---

## Summary

This RFC introduces **`crates/byard-cli`**, a new workspace crate that ships the
`byard` binary, and specifies **the dev runner** — the unified host that `byard
dev` runs, which replaces the ad-hoc `hello_world_byd.rs` example as the
canonical real-app entry point. Together they close the gap between "the engine
can render a `.byd` file" (Phase 2 thesis, proven) and "a developer can write,
run, and iterate on a `.byd` file with zero manual restarts" (the actual usable
product).

Four commands are defined:

| Command | What it does |
|---------|--------------|
| `byard new <name>` | Scaffold a new project with `main.byd`, `byard.toml`, `.gitignore` |
| `byard dev [file]` | Open a GPU window and live-reload the `.byd` file on every save |
| `byard check [file]` | Parse + validate without opening a window; CI-friendly, exits non-zero on errors |
| `byard build` | Phase 3+ stub; prints a clear "not yet available" message |

The central specification is **`byard dev`**: it owns the complete developer loop
— parse once, open window, run logic thread, watch file, apply reactive-compatible
patches instantly, defer structure-incompatible patches past in-flight gestures
(RFC-0002 E5 / RFC-0003), and display parse errors as an overlay in the window
without crashing.

---

## Motivation

### The current state is a prototype, not a tool

After Phase 2, `byard-compiler` can parse, interpret, render, and hot-reload
`.byd` views. The machinery is complete. But it is only accessible through
`cargo run --example hello_world_byd -p byard-platform`, which:

- Hard-codes the source file via `include_str!`, so any edit requires a
  `cargo run` restart — *exactly the problem hot reload was built to solve.*
- Is an internal development fixture, not a public interface.
- Has no project conventions, no error display, and no scaffolding.

A developer who downloads Byard today cannot write their first `.byd` file and
see it running in under two minutes. This RFC changes that.

### Why a CLI, not a GUI launcher

`byld` is a language for UI, but the tool that runs it must be scriptable: CI
pipelines run `byard check`, package managers invoke `byard build`, and
developers type `byard dev` in a terminal. A CLI is also the simplest artifact —
one binary, no installer required on macOS/Linux. A GUI launcher is a future
possibility (§ Future possibilities) and can compose *on top of* the CLI.

### Why specify it in an RFC, not just ship code

The dev runner introduces the first **observable public contract** for how Byard
projects are structured (`byard.toml`, entry-point conventions, error display),
and it connects three previously-separate Phase 2 pieces:

1. The `Interpreter` + `Engine` runner (RFC-0001, RFC-0002)
2. The file-watcher channel (RFC-0002 D10, now wired)
3. The hot-reload decision logic `diff_view` / `diff_program` / `gate`
   (RFC-0002 §hot-reload boundary)

Decisions about how these connect — tick ordering, error overlay, gesture gate
window, channel drain position — are load-bearing design choices that belong in a
document, not buried in example code.

---

## Guide-level explanation

### First run in two commands

```sh
byard new my_app
cd my_app
byard dev
```

A GPU window opens showing the starter view. Edit `main.byd`, save — the window
updates within one tick cycle (typically < 16 ms on any modern machine) without
closing or restarting. No recompile. No `cargo run`. No hot key. Just save.

### Project layout

```
my_app/
├── byard.toml          # project manifest
├── main.byd            # entry-point view (configurable)
├── src/                # optional Rust controller crates
└── .gitignore
```

`byard.toml` is minimal by design:

```toml
[project]
name    = "my_app"
entry   = "main.byd"     # relative to byard.toml; default "main.byd"
```

No build system integration, no dependency resolver, no lock file — those belong
to the Rust/Cargo side. `byard.toml` only describes what `byard dev` needs to
find and run the view.

### `byard new <name>`

```
$ byard new my_app
  Created my_app/
  Created my_app/byard.toml
  Created my_app/main.byd
  Created my_app/.gitignore
```

`main.byd` is a runnable starter that demonstrates the core language features:
a reactive counter and a text input, matching the hello_world demo already in
the repo. It compiles and runs without errors from the first keystroke.

### `byard dev`

```
$ cd my_app && byard dev
  Byard 0.0.0 — dev mode
  Entry: main.byd
  Watching for changes…
  ↳ Loaded (0 errors)
```

The window title shows `"Byard dev — <name>"`. On every save:

- A **reactive-compatible** patch (body changes only): applied on the very next
  tick — the window updates while any in-flight pointer gesture continues safely.
- A **structure-incompatible** patch (`var`/`let`/`inject` shape changed): applied
  after the current gesture ends (RFC-0003 E5 gate). A small "⟳ reload pending"
  indicator is shown during the wait. Applied immediately if no gesture is
  in-flight.
- A **parse error**: the window switches to an error overlay showing the file
  name, line, column, and message. The last successfully-rendered view stays as
  a blurred background. Fixing the file and saving dismisses the overlay.

`byard dev` never crashes on a bad save. It holds the last good state and waits.

### `byard check`

```
$ byard check main.byd
main.byd:7:5: error: unknown attribute `colour` — did you mean `color`?
$ echo $?
1
```

No window. Returns exit code 0 on success, 1 on any error. Designed for CI.
Accepts an explicit path or reads `entry` from `byard.toml` in the current
directory.

### `byard build`

```
$ byard build
  byard build is not yet available (Phase 3+).
  For production builds, see https://github.com/Briany4717/byard/issues/XXX.
```

Exits 0, so scripts do not fail; the message is on stderr.

---

## Reference-level explanation

### 1. Crate layout

```
crates/byard-cli/
├── Cargo.toml
└── src/
    ├── main.rs              # CLI entry: parse args, dispatch
    ├── commands/
    │   ├── new.rs           # byard new
    │   ├── dev.rs           # byard dev  (the dev runner)
    │   ├── check.rs         # byard check
    │   └── build.rs         # byard build (stub)
    ├── manifest.rs          # byard.toml parse + project-root discovery
    └── error_overlay.rs     # in-window error display
```

**Dependencies:**

```toml
[dependencies]
clap            = { version = "4", features = ["derive"] }
byard-compiler  = { path = "../byard-compiler" }
byard-platform  = { path = "../byard-platform" }
byard-core      = { path = "../byard-core" }
toml            = "0.8"
```

`byard-cli` is above the existing crates in the dependency graph:

```
byard-cli
  ├── byard-compiler   (Interpreter, parse, reload, LatestWins, start_watcher)
  ├── byard-platform   (WinitHost)
  └── byard-core       (Engine, LogicRuntime, RenderFrame)
```

It never inverts any existing edge. `byard-core` continues to have zero knowledge
of the CLI; `byard-compiler` continues to have zero knowledge of the window
system. The CLI is the integration point.

### 2. `manifest.rs` — project-root discovery and `byard.toml`

**Discovery algorithm** (decision **C1**):

1. If a path argument was given, use it as the entry file directly (no manifest
   required).
2. Otherwise, walk from `CWD` upward until `byard.toml` is found or the
   filesystem root is reached.
3. If no `byard.toml` is found, fall back to `main.byd` in `CWD` (allows running
   in a directory without a manifest, useful for quick experiments).

```rust
pub struct Manifest {
    pub project_root: PathBuf,
    pub entry:        PathBuf,   // absolute, resolved from project_root + [project].entry
    pub name:         String,
}

impl Manifest {
    /// Discover and parse the manifest, or return a synthetic one for a bare file.
    pub fn discover(override_path: Option<&Path>) -> Result<Self, CliError> { … }
}
```

`[project].name` defaults to the directory name. `[project].entry` defaults to
`"main.byd"`. Unknown keys in `byard.toml` emit a warning, never an error
(forward-compatible with future manifest additions — decision **C2**).

### 3. The dev runner — `commands/dev.rs`

The dev runner is the most complex piece. Its design is a direct consequence of
the constraints in RFC-0001 §5, RFC-0002 D10, and RFC-0003.

#### 3.1 Thread model

```
  Main / winit thread
    │  OS events (resize, input, close)
    │  on_resume → spawn logic thread, spawn watcher thread
    ▼
  Logic thread   (ByldRuntime::evaluate_tick — must stay !Send)
    │  drain reload channel (step 0)
    │  dispatch_events (step 1)
    │  tick (step 2)
    │  render (step 3)
    ▼
  Watcher thread  (notify OS thread)
    │  file change → re-parse → publish to LatestWins<ParsedFile>
```

The watcher thread is separate from the Tokio pool (RFC-0002 D10: "not relay's
Tokio pool — to avoid latency and contention"). The `Arc<LatestWins<ParsedFile>>`
is the only shared state between the watcher and logic threads; it is already
designed for exactly this (bounded(1), try_recv before try_send — latest-wins).

#### 3.2 `ByldRuntime` — per-tick logic

```rust
struct ByldRuntime {
    interp:          Interpreter,
    tree:            Vec<RenderNode>,
    current_view:    ViewDecl,            // for structural diffing
    reload_channel:  Arc<LatestWins<ParsedFile>>,
    pending_reload:  Option<(ViewDecl, ReloadKind)>,  // E5 gesture gate
    error_state:     Option<Vec<CompileError>>,        // parse errors to overlay
    width_bits:      Arc<AtomicU32>,
    height_bits:     Arc<AtomicU32>,
}
```

**Per-tick ordering** (decision **C3**, respects RFC-0003 §per-tick ordering):

```
Step 0: drain reload channel
  ├── ParsedFile with errors? → set error_state, discard views
  └── ParsedFile ok?
        ├── diff_program(current_view, new_views)
        │    └── per-ViewDecl: Added / Removed / Patch(ReloadKind)
        └── gate(kind, pointer_pressed = router.is_pressed())
              ├── Gated::Apply  → interp.reload(&new_view, kind); rebuild tree
              └── Gated::Defer  → pending_reload = Some((new_view, kind))

Step 0b: apply pending_reload if pointer was released
  └── (checked every tick, not just on new file events)

Step 1: dispatch_events(input_events)

Step 2: interp.tick()

Step 3: render(tree, frame) or render_error_overlay(error_state, frame)
```

This ordering guarantees:
- Events from the *current* tick always run against the *current* view, never a
  partially-applied patch (D1 consistency boundary).
- A reactive-compatible patch is applied *before* dispatch, so the very next
  render already reflects the new source — zero extra tick of lag.
- A structure-incompatible patch that arrives during a gesture never tears down
  the view mid-gesture (RFC-0003 E5): the user finishes their tap, the tap fires,
  *then* the view rebuilds.

#### 3.3 Reload application

When `gate` returns `Gated::Apply`:

```rust
fn apply_reload(&mut self, new_view: &ViewDecl, kind: ReloadKind) {
    self.interp.reload(new_view, kind);       // RFC-0002 case 1 or case 2
    self.tree = self.interp.lower_view(new_view, &[]);
    self.current_view = new_view.clone();
    self.error_state = None;
}
```

`Interpreter::reload` is the existing method that either keeps `Signal` state
(ReactiveCompatible) or resets it (StructureIncompatible) — RFC-0002 §hot-reload
boundary. The tree is rebuilt from the new `ViewDecl` on the same call. The
reactive graph re-derives from scratch on the next tick; no special-casing needed.

#### 3.4 Error overlay

When `error_state` is `Some(errors)`, `render` emits a semi-transparent dark
overlay with the first error's message, file, line, and column. The last
successfully-rendered frame is not re-emitted (it has already been displayed);
the overlay is a simple `BoxInstance` + `TextLine` written directly to the frame
without going through the interpreter. This keeps the error display path entirely
independent of the interpreter state — a parser bug or evaluator panic cannot
prevent the error from being shown (decision **C4**).

Structure of the overlay (rendered directly, not via `byld`):

```
┌──────────────────────────────────────────────┐
│  ✕  Parse error — main.byd                   │
│     line 7, col 5                            │
│     unknown attribute `colour`               │
│     did you mean `color`?                    │
│                                              │
│  Fix the file and save to dismiss.           │
└──────────────────────────────────────────────┘
```

Background: `0xCC000000` (80% opaque black). Text: white. No border-radius
(keeps the overlay renderer trivially simple — no `DecoratedBox` needed).

#### 3.5 Watcher setup

```rust
// In App::on_resume (main thread, before start_logic_from_view):
let reload_channel = Arc::new(LatestWins::<ParsedFile>::new());
let channel_for_watcher = Arc::clone(&reload_channel);

// Hold the handle — dropping it stops the watcher.
let _watcher = start_watcher(&manifest.entry, channel_for_watcher)?;
```

The `_watcher` handle is stored in `App` (decision **C5**: the watcher's lifetime
is tied to the `App`, not the logic thread; it survives logic-thread restarts
caused by structure-incompatible reloads, since the channel `Arc` is shared).

The initial parse happens on the main thread in `on_resume` (before the logic
thread starts) so that parse errors at startup can be shown immediately, and the
logic thread always receives a valid `current_view` to start from.

### 4. `commands/new.rs` — scaffolding

Three files are written:

**`byard.toml`**
```toml
[project]
name  = "<name>"
entry = "main.byd"
```

**`main.byd`** — a runnable starter that demonstrates `var`, reactivity, a
`Button`, and a `TextField` (same feature set as the existing `hello_world.byd`,
but with cleaner structure and inline comments explaining the syntax):

```byld
// <name> — starter view
// Edit and save; the window updates instantly.
View Main() {
    var count = 0
    var label = "hello"

    Column #[gap: 20, p: 32, align: center, justify: center] {
        Text("{label} — tapped {count} times") #[size: 24, color: 0xFFFFFF]

        Button("Tap me") #[bg: 0x3B82F6, radius: 8, px: 20, py: 10,
                           color: 0xFFFFFF, weight: bold] => count++

        TextField #[bg: 0x374151, radius: 6, px: 12, height: 36,
                    color: 0xFFFFFF, bind: label]
    }
}
```

**`.gitignore`**
```
/target
```

All three files are written atomically: if any write fails, the directory is
removed and an error is reported. Partial scaffolding is never left on disk
(decision **C6**).

### 5. `commands/check.rs` — headless validation

Reads the entry file, runs `byard_compiler::parser::parse`, and prints diagnostics
in a format compatible with terminal linkers (file:line:col: error: message).
Exits 0 on success, 1 on any error. Does not open a window; has no dependency on
`wgpu`, `winit`, or any GPU primitive. Suitable for `git` pre-commit hooks and CI.

```
$ byard check
Checking main.byd…
main.byd:7:5: error[UnknownAttribute]: unknown attribute `colour`
  note: did you mean `color`? (Levenshtein distance 1)
1 error.
$ echo $?
1
```

The output format is `file:line:col: error[kind]: message\n  note: hint` —
matching the Rust compiler's format so existing IDE integrations that parse `rustc`
output will work without custom parsers (decision **C7**).

### 6. CLI surface — `main.rs`

```rust
#[derive(Parser)]
#[command(name = "byard", version, about = "The Byard UI framework CLI")]
enum Cli {
    /// Scaffold a new project.
    New { name: String },
    /// Start the dev window with live reload.
    Dev { file: Option<PathBuf> },
    /// Parse and validate without opening a window.
    Check { file: Option<PathBuf> },
    /// (Phase 3+) Compile to a production binary.
    Build,
}
```

`clap derive` is used throughout. `clap`'s version field reads from
`Cargo.toml`'s `package.version`. All subcommands print `--help` output that
mirrors this document's table.

---

## Resolved design decisions (CLI and dev runner, final)

### C1 — Project-root discovery: manifest-walk then CWD fallback

Adopted as described in §2. The upward walk mirrors Cargo's `Cargo.toml`
discovery: it makes `byard dev` work from any subdirectory of the project, which
is the ergonomic expectation for a modern CLI. The CWD fallback (bare `main.byd`)
lowers the barrier to experimentation: `byard dev main.byd` works in a directory
with no manifest, and `byard dev` works if there is a `main.byd` present.

The walk stops at the filesystem root, not at a git boundary. A git-boundary stop
(like Cargo) would prevent `byard dev` from running inside a mono-repo where
`byard.toml` is above the git root of a sub-project. Since Byard has no
dependency resolution at this stage, there is no safety reason for the git-root
stop.

### C2 — `byard.toml` unknown keys: warn, not error

Forward-compatible. Future fields (`[dependencies]`, `[build]`, `[features]`)
will land in Phase 3+. If earlier manifests treat unknown keys as errors,
upgrading `byard` in an existing project breaks `byard dev` until the manifest is
updated — a poor upgrade story. A warning preserves the signal that the key is
unrecognized while keeping the project runnable.

### C3 — Tick step ordering: drain-reload before dispatch-events

The critical ordering decision. Two alternatives were considered:

- **Drain reload after dispatch:** the current tick's events fire against the old
  view, and the new view only renders on the next tick. Correct but wastes a tick
  of reload lag.
- **Drain reload before dispatch (adopted):** the new view is in effect when
  events are dispatched. This means that an event handler that was *removed* in
  the reload will not fire, and an event handler that was *added* will be
  available immediately. Both behaviors are correct and expected — the developer
  saved the file; the new behavior is what they intended.

The only edge case is a tap that was in progress during a structure-incompatible
reload: the E5 gate (Gated::Defer) ensures such reloads are held until the
gesture ends, so the tap fires against the old view, completes, and *then* the
new view mounts. No event is lost; no view is torn down under a live gesture.

### C4 — Error overlay rendered directly, not via `byld`

If the overlay were a `byld` view, a parse error in the byld surface syntax would
block the overlay from rendering — the exact scenario where the overlay is needed.
Direct-to-frame rendering (a few `BoxInstance` and `TextLine` writes, no
interpreter involved) keeps the error path independent of the interpreter's
correctness. This mirrors how OSes render crash dialogs: the crash handler cannot
use the same runtime that just crashed.

Consequence: the overlay is intentionally plain (no border-radius, no animation,
fixed colors). Aesthetic polish of the overlay is a future possibility; correctness
is non-negotiable.

### C5 — Watcher lifetime tied to `App`, not to logic thread

A structure-incompatible reload rebuilds the `Interpreter` (teardown + remount).
If the watcher were owned by the logic thread, rebuilding the interpreter would
drop the watcher and stop watching — the developer would have to restart `byard
dev` after changing a `var` declaration. Owning the watcher in `App` keeps it
alive for the full session; the `Arc<LatestWins<ParsedFile>>` is shared across
any number of logic-thread rebuilds.

### C6 — Atomic scaffolding: remove partial state on failure

`byard new` creates a directory and writes several files. If any write fails
(permissions, disk full), a partial project is left in a state that `byard dev`
cannot run from. This is more confusing than "no directory was created." The
fix — track files written, delete them all on first error — costs negligible code
and eliminates a class of confusing UX failures. This applies the same principle
as database transactions to file creation.

### C7 — `byard check` output format: rustc-compatible `file:line:col: error[kind]: message`

The Rust compiler's output format is already parsed by every major IDE extension,
CI log parser, and terminal emulator. Using the same format means `byard check`
works in CI pipelines, `make`-based scripts, and problem-matcher configurations
(VS Code, JetBrains, GitHub Actions) without any Byard-specific integration.
Alternative (JSON output) is a future addition for language-server use cases, not
a replacement.

---

## Drawbacks

- **`clap` dependency.** Adds a compile-time dependency. Mitigated: `byard-cli`
  is a *binary* crate; `clap` does not appear in any library crate's public API,
  so it cannot leak into downstream dependency graphs.
- **`byard.toml` is another project file.** Developers already have `Cargo.toml`.
  Adding a second manifest adds cognitive overhead. Mitigated: it is optional
  (`byard dev main.byd` works without it) and intentionally minimal (two keys).
  Phase 3 will evaluate whether `byard.toml` should be a section inside `Cargo.toml`
  instead.
- **The error overlay adds a rendering path outside the interpreter.** Two
  rendering paths means two code paths to maintain. Mitigated: the overlay is
  trivially simple (four primitives) and explicitly not reusing the interpreter
  (C4); it is unlikely to change.
- **`byard dev` opens a GPU window.** A headless dev server (e.g. for remote
  development) is not possible with this design. This is a Phase 3 concern;
  the `PlatformHost` abstraction already provides the seam for a headless host.

---

## Rationale and alternatives

**Why `clap` and not a custom argument parser?** `clap derive` produces correct
`--help`, version strings, and error messages with zero maintenance. The
alternative (hand-written argument parsing) is several hundred lines of code for
identical observable behavior, with worse error messages.

**Why `byard.toml` and not piggyback on `Cargo.toml`?** Two reasons. First, a
`byld`-only project has no `Cargo.toml` (no Rust controllers); requiring one
would force non-Rust users into the Cargo ecosystem unnecessarily. Second,
`Cargo.toml` has a fixed schema enforced by `cargo`; adding an `[byard]` section
today might collide with a future Cargo workspace feature. A separate file avoids
the collision risk. Phase 3 will reconsider once the feature set is stable.

**Why hold the watcher in `App` and not in the logic thread?** See C5. The
alternative (logic-thread ownership) was rejected because structure-incompatible
reloads would stop file-watching, requiring a manual `byard dev` restart on any
`var` declaration change — precisely the interaction most likely during early
development.

**Why drain the reload channel at Step 0 (before dispatch) rather than Step 3
(after render)?** See C3. "After render" would mean the user sees a stale frame
for one full tick after every save — a noticeable flicker at 60 Hz. "Before
dispatch" applies the patch in the same tick it arrives, with the E5 gate
ensuring gesture safety.

**Why no `byard run` (non-reloading)? Why no `byard test`?**  `byard dev` *is*
the non-reloading runner when the file does not change — no separate `run` is
needed. `byard test` (headless test runner for `.byd` views) is a Phase 3
concern; the `byard check` command covers the parse/validate use case that CI
needs today.

---

## Prior art

- **Vite / `npm run dev`.** The "save and see" DX reference. Byard's `byard dev`
  targets the same sub-16ms reload latency Vite delivers for web projects.
- **Flutter Hot Reload.** The closest existing analogy: reactive-compatible
  patches (Dart VM's "hot reload") apply in-place; structure-breaking changes
  require a "hot restart." RFC-0002 §hot-reload boundary formalizes the same
  distinction; this RFC wires it.
- **JetBrains Compose Hot Reload.** State-preservation on compatible patches,
  full rebuild on incompatible ones — the model RFC-0002 already cited.
- **`cargo-watch`.** External file watcher that re-runs Cargo commands. `byard
  dev` provides a first-class equivalent within the framework, with gesture-safety
  and error overlay that `cargo-watch` cannot offer.
- **SwiftUI Previews.** In-editor live preview without a separate window. A future
  `byld` LSP integration could offer this; the current RFC targets the terminal
  workflow.
- **`gleam` CLI.** A language-first CLI with `gleam new`, `gleam run`, `gleam
  check`, `gleam build` — the same four-command shape this RFC adopts.
  Demonstrates that a minimal CLI surface is sufficient for a complete developer
  experience.

---

## Unresolved questions

**Before implementation:**

- [ ] **`byard.toml` in `Cargo.toml`?** Phase 3 will evaluate `[workspace.byard]`
  as an alternative to a separate file. Decision deferred.
- [ ] **Window title / taskbar icon.** `byard dev` sets the title to
  `"Byard dev — <name>"`. Should it use the `.byd` file's first `View` name
  instead? No strong opinion; resolved during implementation.
- [ ] **`byard dev --no-watch`.** A flag to run without the file watcher (for
  embedded / headless scenarios). Possibly needed for testing; not required for
  Phase 2.

**During implementation:**

- [ ] **Error overlay position and opacity.** The RFC specifies a full-screen
  overlay at 80% opacity. If the error message is very long, it may overflow.
  A scrollable overlay or truncation heuristic is an implementation decision.
- [ ] **Multiple view files.** `byard.toml` currently specifies one entry file.
  A multi-file project (`import` or a `views/` directory) is not in scope for
  Phase 2 and will require a grammar-amendment RFC.
- [ ] **`byard check` exit code for warnings.** Currently 0 on warnings. If
  "unknown `byard.toml` key" becomes a structured warning, CI authors may want
  `--strict` to treat warnings as errors. Deferred.

---

## Future possibilities

- **`byard test <file>`** — headless test runner for `.byd` views, exercising
  reactive state and event dispatch without a GPU window. Natural Phase 3
  deliverable using the existing `Interpreter` + headless `Engine` path.
- **`byard format` / `byard fmt`** — formatter for `.byd` files. Requires the
  lossless rowan-based CST RFC-0002 deferred to Phase 3.
- **`byard lsp`** — start the language server (already exists as `byld-lsp`;
  this would add a CLI alias).
- **GUI launcher** — a graphical project browser that embeds `byard dev` sessions,
  aimed at developers less comfortable with the terminal. Composable on top of
  the CLI; no architectural changes required.
- **`byard add <controller>`** — scaffold a Rust controller crate and wire it into
  the project. Requires Phase 3's `#[byard_controller]` metadata system.
- **Remote dev server** — a headless `byard dev` mode that streams the rendered
  frame to a browser via WebSockets, enabling remote development and mobile
  preview without native builds. The `PlatformHost` abstraction makes this
  possible without engine changes.
