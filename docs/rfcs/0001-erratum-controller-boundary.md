# Erratum to RFC-0001: controller boundary status & `#[byard_controller]` scope

- **Status:** Active erratum (amends, does not replace, RFC-0001)
- **Author(s):** Briany4717
- **Created:** 2026-07-17
- **Applies to:** RFC-0001 §"Rust controller", §5.1 (Tokio pool row), and the implied claim that `#[byard_controller]` is a working boundary.
- **Authority:** RFC-0028 (Controller Boundary & Async Bridge), RFC-0029 (Async I/O Capabilities). Where RFC-0001's prose implies a working controller path, **RFC-0028/0029 define what actually exists and what is being built.**

---

## Why this erratum exists

RFC-0001 describes the controller boundary as if it were complete: §"Rust
controller" says `#[byard_controller]` "generates shared-memory bindings and a
typed metadata file," and §5.1 lists a Tokio pool that "executes async I/O from
Rust controllers \[and] sends results back to the logic thread via
`tokio::sync::mpsc`." A source audit (2026-07-17,
`support/GAP_ANALYSIS_real_apps.md`) found this path is **not wired end to end**.
This erratum records the true state and points to the RFCs that close the gap. As
with the surface-syntax erratum, RFC-0001 is **Active**, so its body is not
edited; the architectural intent stands, and RFC-0028/0029 realize it.

This erratum changes **no architecture**. The two-layer split, the `!Send`
`Signal`, the double-buffered relay, the Tokio pool, and `tokio::sync::mpsc`
result delivery all stand exactly as RFC-0001 states. What changes is the honest
status of the *implementation* and the precise **scope** of what
`#[byard_controller]` emits.

---

## Corrections

### CB1 — `#[byard_controller]` today emits **only field metadata**

The macro (`crates/byard-macro/src/lib.rs`) generates `BYARD_FIELDS` and
`byard_field_type()` — the "typed metadata" of RFC-0001 §"Rust controller" — and
**nothing else**. Its current doc-comment claims it "can provide ambient values
via `inject`, expose async methods, and deliver results back to the logic thread
through the relay's I/O channel." **None of that is implemented by the macro
today.** That doc-comment is aspirational and should be corrected in code to
describe only the metadata it emits, with a pointer to RFC-0028 for the full
boundary. RFC-0028 §2 specifies the dispatch shim (`impl Controller`) the macro
*will* generate to make the async-method claim real.

### CB2 — `inject` does not resolve a controller (yet)

RFC-0001's example `inject AppEnvironment as env` and the "shared-memory
bindings" language imply `inject` can reach a Rust controller. Today
`Env::provide`/`resolve_inject` (`interp/env.rs`) resolve an ambient **`Value`**
inside the `byld` tree only; there is no controller handle and no method call.
RFC-0028 §3 generalizes providers from values to controller handles
(`Value::Controller`) so `inject WeatherApi as api` resolves and `api.method(…)`
dispatches — the mechanism RFC-0001 assumed.

### CB3 — the Tokio result channel is not drained on the logic thread

RFC-0001 §5.1 says results flow "back to the logic thread via
`tokio::sync::mpsc`." The channel exists (`Relay::io_result_sender` /
`try_recv_io_result`) and `Interpreter::apply_io_results` exists
(`eval.rs:995`), but the logic-thread loop (`relay.rs::spawn_logic_from_view`)
**never connects them** — the only real consumer is image decode on the
**render** thread (`engine.rs:610`). So no controller result can reach a `var`
today. RFC-0028 §5 adds the missing logic-thread drain (tick step 0) and
RFC-0029 §2 wakes a `Wait`-mode render loop on such ticks.

### CB4 — the Tokio runtime cannot do socket/timer I/O (yet)

RFC-0001 §5.1's "executes async I/O" implies sockets/timers. The runtime is built
without `.enable_io()`/`.enable_time()` and Cargo enables tokio with only
`["rt-multi-thread","sync"]` (`relay.rs:180-185`), so only compute futures run. A
`TcpStream`/`sleep` would panic. RFC-0029 O1 enables the `net`+`time` drivers
behind a `runtime-io` feature — the change the `relay.rs` comment explicitly
deferred.

---

## What is unchanged

Every architectural claim in RFC-0001 stands: the `.byd`/`.rs` two-layer model,
`#[byard_controller]` as *the* boundary marker, `!Send` `Signal`s never crossing
threads, the double-buffered `RenderFrame` swap, the Tokio pool, and result
delivery via `tokio::sync::mpsc`. RFC-0028/0029 do not alter any of these — they
**implement** them. The erratum-0001-byld-surface example
`impl NetworkController { pub async fn fetch_user(&self, id: u64) -> Result<User,
ApiError> { … } }` is exactly the shape RFC-0028 makes callable.

---

## Summary table

| RFC-0001 as written / implied | True status (2026-07-17) | Realized by |
|---|---|---|
| `#[byard_controller]` "generates bindings" | Emits field metadata only | RFC-0028 §2 (dispatch shim) |
| `inject` reaches a controller | Resolves ambient `Value` only | RFC-0028 §3 (`Value::Controller`) |
| Results flow to the logic thread via mpsc | Channel exists, logic-thread drain missing | RFC-0028 §5 (tick-step-0 drain) |
| Tokio pool "executes async I/O" | Runtime has no `net`/`time` driver | RFC-0029 O1 (`runtime-io`) |
| Two-layer model, `!Send` `Signal`, double buffering | **Unchanged, correct** | — |
