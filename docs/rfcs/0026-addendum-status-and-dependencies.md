# Addendum to RFC-0026: implementation status & dependencies on the data/async layer

- **Status:** Active addendum (amends, does not replace, RFC-0026)
- **Author(s):** Briany4717
- **Created:** 2026-07-17
- **Applies to:** RFC-0026 (Navigation & Routing).
- **Authority:** RFC-0027 (Data & Collection Operations), RFC-0028 (Controller Boundary & Async Bridge), RFC-0029 (Async I/O Capabilities).

---

## Why this addendum exists

RFC-0026 is a complete, well-specified design, but it is **not implemented**: a
source audit (2026-07-17) found nothing in `crates/byard-compiler/src/interp/`
that provides a navigation stack, route table, or `navigate`/`back`/`replace`
actions (the only "router" in the tree is the *event* `EventRouter`, unrelated).
RFC-0026 remains the authoritative design for navigation; this addendum records
its status and its **hard dependencies on the newly proposed data/async layer**,
so the 0027–0029 set can reference navigation without duplicating it and so the
milestone plan sequences them correctly.

This addendum adds **no navigation design** — RFC-0026 owns that. It only fixes
the dependency order.

---

## N-A1 — Status: designed, not built

Mark RFC-0026 as **"Active — not implemented"** in the document map. Its three
primitives (`NavStack`, `NavHost`, navigation actions) and its "navigation state
is a reactive `var`" model are unchanged and correct; they simply have no code
yet. Navigation is **out of scope** for the 0027–0029 cluster (which targets the
todo/weather blockers); it is the next tier, unblocked *by* them.

## N-A2 — Route params need RFC-0027 data operations

RFC-0026's `route "/detail/:id" { DetailPage(id: route.params.id) }` reads a
field off a `route.params` record and compares/derives route state. That requires
**record field access, comparison, and (for query lists) collection ops** — none
of which exist pre-RFC-0027. Navigation cannot be implemented before RFC-0027
lands the expression layer. Add RFC-0027 to RFC-0026's *Depends on* list.

## N-A3 — Deep linking needs the RFC-0028 boundary and RFC-0029 I/O

RFC-0026 §"deep linking from OS intents" delivers an external URL into the app.
An OS intent is a platform I/O event that must cross to the logic thread as
`Send` data and land on a navigation `var` — exactly the RFC-0028 apply path
(§5). Deep-link handlers that load data for the target screen (the common case)
call controllers (RFC-0028) and I/O capabilities (RFC-0029). So RFC-0026's
deep-linking tier depends on RFC-0028; its data-backed screens depend on
RFC-0029. Basic in-app push/pop navigation depends only on RFC-0027.

## N-A4 — Sequencing

Implement in this order: **RFC-0027 → RFC-0028 → RFC-0029 → RFC-0026**. RFC-0027
alone unblocks in-app navigation (push/pop/replace over a `var` stack); RFC-0028/
0029 unblock deep linking and data-backed screens. The IMPLEMENTATION_7 milestone
plan schedules 0027–0029 only; navigation gets its own milestone set once this
cluster is green.

---

## Summary table

| RFC-0026 feature | Depends on (new) | Available after |
|---|---|---|
| `NavStack` push/pop/replace over a `var` | RFC-0027 (records, comparison) | RFC-0027 |
| `NavHost` route matching, `route.params` | RFC-0027 (field access, lists) | RFC-0027 |
| Data-backed screens (fetch on navigate) | RFC-0028 + RFC-0029 | RFC-0029 |
| Deep linking from OS intents | RFC-0028 (Send delivery to logic thread) | RFC-0028 |
