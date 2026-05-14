# Contributing to Byard

Thanks for your interest in Byard. Contributions are welcome from day one.

Byard is in its **design phase**. There is no usable build yet — the codebase is a
set of design documents and an architectural plan. This shapes what "contributing"
means right now, so please read this whole document before opening anything.

## The most valuable contribution right now is design review

Because the architecture is still being settled, **discussion on the RFCs is worth
more than code**. If you have experience with `wgpu`, retained-mode UI, layout
engines, language frontends, or memory-arena designs, the best thing you can do is:

1. Read [RFC-0001: Core Architecture](docs/rfcs/0001-core-architecture.md).
2. Open a [discussion](https://github.com/Briany4717/byard/discussions) or an issue with
   the `rfc` label challenging or refining a decision.

The [Unresolved Questions](docs/rfcs/0001-core-architecture.md#unresolved-questions)
section of RFC-0001 lists the open design problems we most need help thinking
through — accessibility mapping, runtime error handling, the testing strategy for a
multi-threaded engine, and the exact scope of Phase 1.

## Ways to contribute

| Type | How |
|------|-----|
| **Design review** | Comment on an RFC, or open a discussion. |
| **New design proposal** | Copy [`docs/rfcs/0000-template.md`](docs/rfcs/0000-template.md), fill it in, open a PR. |
| **Bug report** | Use the bug report issue template. (Applies once there is code to run.) |
| **Feature request** | Use the feature request issue template. Note that large features should go through the RFC process. |
| **Documentation** | Fixes to the README, RFCs, or this file are always welcome. |
| **Code** | Once Phase 1 is scoped and tracked in issues, code contributions open up. Look for issues labelled `good first issue` and `help wanted`. |

## Before you open an issue

- **Search existing issues and discussions first.** The design is in flux and your
  question may already be under discussion.
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
6. A maintainer will review. Expect discussion — design-stage projects iterate a lot.

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

> These commands describe the intended workflow. Until Phase 1 lands there is no
> code to build — for now, `cargo fmt` and `cargo clippy` on the workspace skeleton
> are all that apply.

```sh
# Format — must be clean before a PR
cargo fmt --all

# Lints — CI treats warnings as errors
cargo clippy --workspace --all-targets -- -D warnings

# Tests
cargo test --workspace

# Build
cargo build --workspace
```

CI runs all of the above on every pull request. A PR cannot be merged until it is
green.

### Toolchain

Byard tracks **stable Rust**. The minimum supported Rust version (MSRV) will be
declared in the workspace `Cargo.toml` once Phase 1 begins.

## Code style

- `rustfmt` is the source of truth for formatting. Do not hand-format.
- `clippy` warnings are errors. If a lint is wrong for a specific case, `#[allow(...)]`
  it with a comment explaining why — don't silence it globally.
- Follow the **strict domain separation** principle: engine crates must not depend
  on `winit` or any concrete platform. Platform code goes behind the `PlatformHost`
  trait. A PR that couples the core to a windowing library will be asked to change.
- Public items need doc comments. The engine is meant to be embeddable; treat the
  API surface as a product.

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
