# Contributing to Byard

Thanks for your interest in Byard. Contributions are welcome from day one.

Byard is in **Phase 1 — Engine core**. Phase 0 (design) is complete, and the
engine core is now being built piece by piece. The codebase builds, tests run,
and CI is green. The `bylang` DSL and the public engine API are still expected
to change as Phase 1 evolves, but the foundations are in place.

## Ways to contribute

| Type | How |
|------|-----|
| **Phase 1 implementation** | Pick an issue from the [Phase 1 milestone](https://github.com/Briany4717/byard/milestones). Look for `good first issue` and `help wanted` labels. |
| **Design review** | Comment on an RFC, or open a discussion. New architectural decisions still go through the RFC process. |
| **New design proposal** | Copy [`docs/rfcs/0000-template.md`](docs/rfcs/0000-template.md), fill it in, open a PR. |
| **Bug report** | Use the bug report issue template. |
| **Feature request** | Use the feature request issue template. Large features should go through the RFC process. |
| **Documentation** | Fixes to the README, RFCs, or this file are always welcome. |

The [Unresolved Questions](docs/rfcs/0001-core-architecture.md#unresolved-questions)
section of RFC-0001 still lists open design problems — accessibility mapping,
runtime error handling, and the testing strategy for a multi-threaded engine —
where experienced reviewers can have outsized impact.

## Before you open an issue

- **Search existing issues and discussions first.**
- For anything that changes the architecture, open a **discussion** or an
  **RFC PR**, not a regular issue. Architectural decisions need the RFC paper trail.
- For bugs, include enough detail to reproduce: OS, GPU, driver, `wgpu` backend, and
  the steps. (This matters more for a GPU project than almost anywhere else.)

## Pull request process

1. **Open an issue or discussion first** for anything non-trivial, so the approach
   can be agreed on before you spend time on it. Typo fixes and small doc changes
   can skip this.
2. Fork the repository and create a branch from `main`.
3. Make your change. Keep the commit history clean — small, focused commits.
4. Make sure the checks pass locally (see below).
5. Open the PR, fill in the template, and link the issue it closes.
6. A maintainer will review. Expect discussion — Phase 1 iterates a lot.

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(render): add scissor-rect dispatch to the SolidBox pipeline
fix(spatial): correct hash bucket index for negative coordinates
docs(rfc): clarify the apoptosis model in RFC-0001
```

This keeps the history readable and lets the `CHANGELOG` be assembled
semi-automatically later.

## Local development

```sh
# Format — must be clean before a PR
cargo fmt --all

# Lints — CI treats warnings as errors
cargo clippy --workspace --all-targets -- -D warnings

# Tests
cargo test --workspace

# Build
cargo build --workspace

# Benchmarks (Evaluator subsystem and onwards)
cargo bench
```

CI runs all of the above on every pull request. A PR cannot be merged until it is
green.

### Toolchain

Byard tracks **stable Rust**. The current MSRV is declared in the workspace
`Cargo.toml`.

## Code style

- `rustfmt` is the source of truth for formatting. Do not hand-format.
- `clippy` warnings are errors. If a lint is wrong for a specific case, `#[allow(...)]`
  it with a comment explaining why — don't silence it globally.
- Follow the **strict domain separation** principle: engine crates must not depend
  on `winit` or any concrete platform. Platform code goes behind the `PlatformHost`
  trait. A PR that couples the core to a windowing library will be asked to change.
- Public items need doc comments. The engine is meant to be embeddable; treat the
  API surface as a product.

### Unsafe code

The workspace lint is `unsafe_code = "deny"`. Modules that legitimately need
`unsafe` (currently `evaluator/arena.rs` and `evaluator/signal.rs`) opt out
with a file-level `#![allow(unsafe_code)]` and must follow these rules:

1. Every `unsafe` block has a `// SAFETY:` comment documenting why each
   invariant of the unsafe operation is upheld.
2. New `#![allow(unsafe_code)]` files require explicit reviewer justification
   in the PR description. The bar is: "could this be done in safe code without
   significant cost or correctness loss?"
3. Unsafe abstractions should be unit-tested for the failure modes their
   `SAFETY` comments rule out (panics on misuse, miri-compatible tests where
   possible).

We use `deny` rather than `forbid` because `forbid` cannot be overridden even
with `#![allow]`, which would make the modules above uncompilable. The intent
is the same: `unsafe` is the exception, not the norm, and every use is audited.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold it. Report unacceptable behaviour to the
contact listed there.

## Licensing of contributions

Byard is dual-licensed under MIT and Apache-2.0. Unless you state otherwise, any
contribution you submit is dual-licensed under the same terms, as described in the
Apache-2.0 license. You do not need to sign a CLA.

## Questions

If something here is unclear, open a
[discussion](https://github.com/Briany4717/byard/discussions) — improving this document
is itself a welcome contribution.
