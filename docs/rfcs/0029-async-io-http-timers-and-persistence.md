# RFC-0029: Async I/O Capabilities — runtime enablement, HTTP, JSON, timers, and persistence

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-17
- **Last updated:** 2026-07-17
- **Depends on:** RFC-0001 (§5.1 Tokio I/O pool, INV-12 "decode off caller" generalized to "I/O off the logic/render threads"), RFC-0028 (the controller boundary — every capability here is reached through it), RFC-0027 (`HostValue`/`Record` shapes for parsed data), RFC-0004 (tick / waker for delivering time-driven and I/O-driven updates).
- **Extends:** `relay.rs` (the Tokio runtime gains the `net`+`time` drivers; the frame waker fires on I/O-driven ticks), the Cargo feature set (new `runtime-io`, `net`, `json`, `storage` features, mirroring the existing `telemetry`/`image` gating).
- **Enables:** Weather-API consumers, feed/list apps backed by remote JSON, periodic refresh, offline persistence of todos and settings — the concrete capabilities the audit's three target apps need once RFC-0028 gives `byld` a way to call Rust.
- **Requires:** RFC-0028 merged first (this RFC has no `byld`-visible surface of its own except the timer effect; everything else is controller-delivered).

---

## Summary

RFC-0028 wires a *path* from `byld` to Rust and back, but there is nothing at the
far end to *do*: the Tokio runtime is built **without** the `net` and `time`
drivers (`relay.rs:180-185` — "nothing here uses sockets or timers yet"), there
is no HTTP client anywhere in the tree, no JSON parser outside `byld-lsp`, no
timer primitive, and no persistence. This RFC adds the **capability layer**:
(O1) flip on the Tokio `net`+`time` drivers behind a `runtime-io` feature so
controllers may use sockets and timers; (O2) a first-party `Http` capability
(reqwest + rustls, `net` feature) provided by default so a weather app works
out of the box, while controllers remain free to bring their own client; (O3)
JSON via `serde_json` (`json` feature) with a direct `JSON ⇄ HostValue`
mapping; (O4) a **timer effect** (`every`/`after`) that delivers ticks through
the same continuation/apply path as a controller reply, waking a `Wait`-mode
render loop; (O5) a minimal durable key/value `Store` capability (`storage`
feature) for offline state. Every capability runs its blocking/async work on the
Tokio pool and delivers only `Send` `HostValue` to the logic thread — INV-12
generalized: **no capability ever blocks the logic or render thread.**

```byld
View Weather() {
    inject Http as http
    inject Store as store
    var report: Record = { state: "loading" }

    // Load persisted last-city on mount, then fetch.
    on mount => store.get("last_city") ok city => http.get("/wx?q={city}")
        ok  res => { report = { state: "ok", data: res.json } }
        err e   => { report = { state: "error", message: e.message } }

    // Refresh every 5 minutes.
    every 300s => http.get("/wx?q=Tokyo") ok res => { report = { state: "ok", data: res.json } }

    when report.state == "ok" { Text("{report.data.tempC}°C") }
}
```

---

## Motivation

The [gap analysis](../../support/GAP_ANALYSIS_real_apps.md) third blocker: even
with a bridge, there is *nothing to call*. Specifically:

- **The Tokio runtime cannot do I/O.** `relay.rs` builds
  `Builder::new_multi_thread()` with no `.enable_io()`/`.enable_time()`, and
  `Cargo.toml` enables only `tokio = { features = ["rt-multi-thread", "sync"] }`.
  A `TcpStream` or a `tokio::time::sleep` would panic ("no reactor running"). The
  code comment explicitly defers this "the day a real async I/O task needs them"
  — that day is now.
- **No HTTP client.** No `reqwest`/`ureq`/`hyper` in any `Cargo.toml`. A weather
  or YouTube-like app cannot make a request.
- **No JSON.** `serde`/`serde_json` live only in `byld-lsp`; controllers have
  nothing to parse an API response with.
- **No timers.** No `every`/`interval`/`after` primitive (grep of
  `intrinsics.rs` is empty). Periodic refresh is impossible.
- **No persistence.** `std::fs` is used only for the vector glyph cache
  (`vector/cache.rs`), never for app state.

This RFC supplies exactly these, each feature-gated so an app pays for only what
it uses (the `telemetry`/`image` precedent).

---

## Guide-level explanation

### 1. Turning on real async (O1)

`byard-core` gains a default-on `runtime-io` feature that enables Tokio's `net`
and `time` drivers and builds the runtime with `.enable_all()`. With it,
**any** controller (RFC-0028) may `.await` sockets and timers. This is the single
prerequisite that makes the other four capabilities possible; a headless/embedded
build that wants none of it can disable the feature.

### 2. HTTP as a provided controller (O2)

`Http` is a built-in controller the engine `provide`s by default (opt-out). It
exposes async methods returning a `HostValue` response record:

```
http.get(url)                 -> { status: Int, body: Str, json: Record|List }
http.post(url, body)          -> { status, body, json }
http.request(record)          -> full control (method, headers, body, timeout)
```

`json` is populated when the response `Content-Type` is JSON (O3); otherwise it
is `Unit` and the caller reads `body`. Non-2xx and transport failures go to the
`err` arm as `{ kind, message, status? }`. Controllers that want a different
client (auth, retries, gRPC) simply don't use `Http` and call their own crate —
`Http` is a convenience, not a lock-in.

### 3. JSON (O3)

The `json` feature adds `serde_json` and a total `serde_json::Value → HostValue`
mapping (objects → `Record`, arrays → `List`, numbers → `Int`/`Float`, etc.), so
API responses land directly in the reactive tree as RFC-0027 records. Controllers
returning `#[derive(HostValue)]` structs get typed parsing for free
(`serde::Deserialize` + `IntoHostValue`).

### 4. Timers as effects (O4)

`every 300s => action` and `after 2s => action` are **effects** bound to the
view: they schedule on the Tokio time driver and deliver a tick to the logic
thread through the RFC-0028 apply path, so the action runs reference-free on the
logic thread and its `var` writes render normally. Timers stop when their view
unmounts (no leaked subscriptions, INV-10). A `Wait`-mode render loop is woken on
each timer tick via the existing frame waker.

### 5. Persistence (O5)

`Store` is a built-in controller (opt-in via the `storage` feature) giving durable
key/value storage in the platform data dir:

```
store.get(key)        -> value | Unit
store.set(key, value) -> Unit          // value is any HostValue (JSON-encoded on disk)
store.remove(key)     -> Unit
```

All operations run on the Tokio pool (file I/O never blocks the logic thread).
It is intentionally a KV store, not a database — enough for todo lists, drafts,
auth tokens, and settings.

---

## Reference-level explanation

### 1. Runtime enablement (O1) — `relay.rs`

`Relay::new` builds the runtime with `.enable_all()` when `runtime-io` is on
(default), `.enable_time()` minimum. `Cargo.toml`:
`tokio = { version = "1", features = ["rt-multi-thread", "sync", "net", "time",
"macros"] }` gated so `--no-default-features` drops `net`/`time`. This is the
change the `relay.rs:180-185` comment anticipated; the existing 27 relay tests
stay green (they use compute futures, unaffected).

### 2. Frame-waker on I/O-driven ticks (O1/O4) — the second half of the drain

RFC-0028 §5 added the logic-thread I/O drain. This RFC amends the loop so a tick
that **applied any I/O result or timer tick** counts as "changed" and calls
`relay.wake_renderer()` (today only input-bearing ticks wake it, `relay.rs:447`).
Without this, an async result or timer would update state but a `Wait`-mode host
would not repaint until the next unrelated OS event. The idle `park_timeout`
(`IDLE_PARK`) is replaced by parking on a condvar that the I/O-result sender and
the timer driver signal, so a pending result wakes the logic thread immediately
instead of after up to 6 ms — bounded latency for network/timer updates.

### 3. HTTP capability (O2) — `byard-core::cap::http`, `net` feature

Backed by `reqwest` with `default-features = false, features = ["rustls-tls",
"json", "gzip"]` — **rustls, not OpenSSL**, so the dependency is pure-Rust,
cross-platform, and needs no system TLS (`unsafe_code = "deny"` friendly). `Http`
implements `Controller` (RFC-0028 §2): each method builds a `reqwest::Request`,
`.await`s it on the pool, and maps the response into the response record. A
default 30 s timeout and a bounded connection pool are set; `http.request({...})`
overrides them. TLS roots come from `webpki-roots` (bundled) so there is no
platform cert-store dependency.

### 4. JSON (O3) — `byard-core::cap::json`, `json` feature

A `fn json_to_host(v: serde_json::Value) -> HostValue` and its inverse
`host_to_json`. Numbers: integral → `Int`, fractional → `Float`. Objects preserve
key order (serde_json `preserve_order`) so round-trips are stable (matches
RFC-0027 record ordering). Exposed both as the `Http` response `.json` field and
as a standalone `Json` capability (`json.parse(str)`, `json.stringify(value)`).

### 5. Timers (O4) — grammar + driver

Grammar (RFC-0002/0003 extension): `timer_effect := ("every" | "after") duration
"=>" action`, a new `Member::Timer { every: bool, dur_ms: u64, action }`, valid at
View-member position (like `on mount`). Lowered to a structural effect (RFC-0018)
so it mounts/unmounts with its scope. On mount, the interpreter registers the
timer with a `TimerDriver` running on the Tokio time driver: `every` uses
`tokio::time::interval`, `after` a one-shot `sleep`. Each fire sends a
`TimerTick { continuation_id }` `IoResult` (the same typed logic-thread channel
as controller replies, RFC-0028 §7); `apply_io_results` runs the timer's action.
Unmount cancels the Tokio task (drops the interval), so no tick fires after the
view is gone (INV-10). Durations build on the existing duration literal: today the
lexer parses only the `ms` suffix (`DurationLit(u32)`, milliseconds, RFC-0010,
`lexer/mod.rs:127`). This RFC adds `s` and `min` suffixes (a one-line regex
extension lowering to the same millisecond `DurationLit`), so `300s`/`5min` read
naturally at timer sites; `ms` stays valid. The parser must also accept a
`DurationLit` in timer-effect position (today it is only meaningful inside
`anim.*` calls).

### 6. Persistence (O5) — `byard-core::cap::store`, `storage` feature

`Store` implements `Controller`. Backing: one JSON file per app in the OS data dir
(`directories`-crate resolved path), loaded into an in-memory `HashMap<String,
HostValue>` at first access, written atomically (temp-file + rename) on `set`/
`remove`. All disk work runs via `spawn_blocking` on the pool. Concurrent `set`s
are serialized by an async `Mutex` inside `Store`. v1 is single-file KV; a
`storage-sqlite` feature is a future possibility for large datasets. Values are
`HostValue`, JSON-encoded on disk via O3 — so a persisted todo list is human-
readable and portable.

### 7. Capability registration & opt-out

The engine `provide`s `Http`, `Json`, and (if `storage`) `Store` by default;
`App::without_default_capabilities()` or per-capability `App::without::<Http>()`
opts out (e.g. a controller that owns its own HTTP stack). Built-in capabilities
occupy reserved type names (`Http`, `Json`, `Store`, `Timer`); a user controller
of the same name is a `CompileError::ReservedControllerName` (INV-5).

### 8. Feature matrix

| Feature | Default | Pulls in | Gives |
|---|---|---|---|
| `runtime-io` | on | tokio `net`+`time` | sockets/timers for any controller |
| `net` | on | `reqwest` (rustls), `webpki-roots` | `Http` capability |
| `json` | on | `serde_json` | JSON parse/stringify, `Http.json` |
| `storage` | off | `directories` | `Store` capability |

Disabling `runtime-io` disables `net`/`storage` (compile-time `cfg` requires it).
Everything degrades cleanly: no feature, no capability, no dependency cost — the
`telemetry`/`image` precedent.

---

## Drawbacks

- **Dependency weight.** `reqwest`+rustls is a non-trivial tree. Mitigated by
  feature-gating (a viz-only app disables `net`) and by rustls avoiding native
  TLS/OpenSSL entirely. Controllers can also use a lighter client and the
  framework ships no client at all with `--no-default-features`.
- **First-party HTTP is opinionated.** `Http`'s response shape and error model are
  choices some apps will outgrow; they escape to their own controller. `Http` is a
  convenience floor, not a ceiling.
- **KV persistence is minimal.** No queries, no migrations, no large-blob story.
  Deliberate for v1; `storage-sqlite` is the escape hatch.
- **Timer semantics are logic-thread coarse.** `every 16ms` is not a substitute
  for the animation runtime (RFC-0010, GPU-driven); timers are for
  seconds-scale refresh, not per-frame motion. Documented to avoid misuse.

---

## Rationale and alternatives

**Why enable `net`/`time` rather than keep the runtime compute-only?** It is the
literal precondition the code left a TODO for, and every target app needs it.
Gating it behind a default-on feature keeps embedded/headless builds lean.

**Why bundle `Http` at all instead of pure BYO-client?** A weather app should work
from `byard new` without the developer wiring reqwest by hand — DX floor. But
because `Http` is just a `Controller` (RFC-0028), it is fully opt-out and
non-privileged, so power users lose nothing.

**Why rustls over native TLS?** `unsafe_code = "deny"` and cross-platform
determinism: rustls + webpki-roots need no OpenSSL, no platform cert store, and
build identically on every OS — matching Byard's "performance/correctness is the
floor" ethos and its arena/no-surprise philosophy.

**Why deliver timer ticks through the RFC-0028 apply path instead of a bespoke
channel?** One delivery mechanism, one consistency boundary (D1), one waker
amendment. A timer tick is semantically a zero-arg controller reply; reusing the
path means no second set of ordering/leak invariants to maintain.

**Why KV, not SQLite, for v1 persistence?** The audit's apps (todo, weather,
YouTube-like) need settings/drafts/cache, not relational queries. KV is a few
hundred lines with no schema story; SQLite is a heavy dependency deferred until a
real dataset demands it.

**Alternative rejected: a synchronous `ureq` client on `spawn_blocking`.** Simpler
deps, but blocking a pool worker per request scales worse than async reqwest under
concurrency (a feed view firing many requests), and loses streaming for a future
YouTube-like use case.

---

## Prior art

- **Flutter `http`/`dio` + `shared_preferences` + `Timer`:** the exact capability
  set (HTTP, KV prefs, timers) a declarative UI toolkit ships. Direct model.
- **SwiftUI `URLSession` + `UserDefaults` + `Timer.publish`:** platform-provided
  networking/persistence/timing consumed reactively.
- **Tauri:** rustls-based HTTP + a KV/store plugin exposed to a declarative front
  end via commands — the feature-gated, controller-delivered approach.
- **reqwest + rustls + webpki-roots:** the standard pure-Rust cross-platform HTTP
  stack.
- **`directories` crate:** canonical per-OS data-dir resolution for `Store`.

---

## Resolved questions

- **Before merge:**
  - [x] **Enable `net`/`time`?** **Yes, behind default-on `runtime-io`.** Executes
    the `relay.rs` TODO.
  - [x] **Ship an HTTP client?** **Yes, `Http` via reqwest+rustls, default-on
    `net`, fully opt-out.** DX floor without lock-in.
  - [x] **TLS backend.** **rustls + webpki-roots**, never native/OpenSSL
    (`unsafe`-deny, cross-platform).
  - [x] **Persistence scope.** **KV single-file JSON, opt-in `storage`.** SQLite
    deferred.
  - [x] **Timer delivery.** **Through the RFC-0028 apply path**, not a bespoke
    channel; one consistency boundary.
  - [x] **Waker on async updates.** **Yes** — I/O-result and timer ticks wake a
    `Wait`-mode render loop (amends `relay.rs` `wake_renderer`).

- **During implementation:**
  - [x] **Idle parking vs latency.** Replace `IDLE_PARK` spin/park with a condvar
    signalled by the I/O sender and timer driver, so async results wake the logic
    thread immediately (was ≤6 ms). Keep a timeout fallback so a lost signal never
    hangs.
  - [x] **HTTP defaults.** 30 s timeout, gzip, redirect-follow (≤10), bounded pool;
    all overridable via `http.request({...})`.
  - [x] **`Store` durability.** Atomic temp-file + rename write; load-once cache;
    async `Mutex` serializes writers. Corrupt file → empty store + diagnostic
    (INV-4), never a panic.
  - [x] **Reserved names.** `Http`/`Json`/`Store`/`Timer` reserved;
    `ReservedControllerName` on collision.
  - [x] **Feature dependencies.** `net`/`storage` `cfg`-require `runtime-io`; a
    build enabling `net` without `runtime-io` fails at compile time with a clear
    message.

---

## Future possibilities

- **`storage-sqlite`** for large/queryable datasets (feeds, offline video
  metadata) behind a heavier feature.
- **Streaming HTTP / websockets / SSE** feeding the RFC-0028 `stream` arm — the
  substrate for a YouTube-like live feed.
- **Request cache & de-dup** layer on `Http` (ETag/Cache-Control) so repeated
  `get`s of the same URL coalesce.
- **File-system capability** (`Fs`) for user documents, mirroring `Store`.
- **Background tasks** that survive view unmount (download queues) via a
  registry-owned, non-view-scoped continuation.
- **Environment/secret injection** so `base_url`/API keys come from config, not
  literals.
