# byard_v2

Byard is a cross-platform, direct-GPU UI framework in Rust (`wgpu` + `winit`)
with a companion declarative DSL, `byld`.

## Design & decisions record

`support/` (gitignored, AI-only — never leak its internal M-codes/IMPL-IDs into
commits, PRs, issues, or source) holds the living design/implementation record:

- `README_CLAUDE.md` — document map and reading order
- `DESICIONS.md` / `DECISIONS_unresolved_questions.md` — decision log
- `IMPLEMENTATION*.md` — milestone-by-milestone build guides
- `ROADMAP.md` / `PLAN_advanced_features.md` — what's built vs. planned
- `AUDIT_events_and_interactive_styles.md` — audit notes

RFCs (the numbered `docs/rfcs/000N-*.md` files) are the upstream architectural
proposals; `support/` tracks how they were actually implemented, amended, and
what's still open. Check `support/` for context before assuming a feature is
unbuilt or a decision is unresolved.
