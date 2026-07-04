# RFC-0008 multi-file / package example

A tiny two-package workspace showing user-view instantiation (RFC-0007) across
files and packages (RFC-0008):

- `app/` — the project. Its `src/main.byd` uses a sibling file (`widgets.byd`,
  same project namespace) and a `path` dependency `kit` via `use kit as k`.
- `kit/` — a local package exposing `Card`.

Check the whole module graph (from inside `app/`):

```
cd crates/byard-cli/examples/workspace/app
cargo run -p byard-cli -- check          # → 3 file(s), 1 package(s); ok (0 errors)
```

Or point the CLI at the project directory or its manifest:

```
cargo run -p byard-cli -- check crates/byard-cli/examples/workspace/app
```
