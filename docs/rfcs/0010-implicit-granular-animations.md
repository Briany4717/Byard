# RFC-0010: Implicit Granular Animations (`with` syntax, GPU-analytic springs)

- **Status:** Draft — design proposal
- **Author(s):** Brian (byard_v2)
- **Created:** 2026-07-01
- **Last updated:** 2026-07-01
- **Depends on:** RFC-0001 (§3.1 render pipelines, §5 concurrency & `!Send`/`!Sync`, `frame.rs` boundary), RFC-0002 (D1 Mark-and-Pull, D4 attribute contract, Pratt parser), RFC-0011 (transform/paint properties — the animatable set), RFC-0004 (reactive tick).
- **Enables:** RFC-0012 (interactive style states animate through this).

---

## Summary

Introduce a per-property, infix `with` operator inside `#[...]` that attaches an
animation to a single attribute value: `radius: pressed ? 3 : 10 with anim.spring(...)`.
Animations are **granular** (one property, one curve, no global "animate
everything" contamination), **implicit** (the dev names the target value and the
curve; the engine drives the transition), and **evaluated on the GPU** for the
paint-time animatable set. The CPU computes only the *target* in `O(1)` on
mutation; the shader evaluates the closed-form damped-spring (or eased) curve per
frame. The CPU never runs an interpolation loop.

The one correctness addition over the original sketch: because a spring never
mathematically reaches rest, the engine keeps a small **active-animation set** so
it can stop requesting frames once every animation has settled within an epsilon.
Without it, "GPU does the interpolation" would silently pin the app at 144 Hz
forever and burn battery — violating Byard's efficiency floor.

## Motivation

Today a value flips instantly: `radius: pressed ? 3 : 10` snaps. Every UI worth
using needs motion, but the mainstream models are bad fits for Byard:

- **CSS `transition: all 200ms`** contaminates globally, re-runs layout, and is
  impossible to reason about for performance.
- **Retained animation objects** (Flutter `AnimationController`, imperative
  tweens) require a *reference* to the widget — which RFC-0003 forbids outright.
- **CPU tween loops** spend cycles every frame per animated property and fight the
  zero-GC arena model.

Byard needs motion that (a) needs no widget reference, (b) is declared next to the
value it animates, (c) costs zero CPU cycles while running, and (d) never forces
relayout. `with` + GPU-analytic springs delivers all four.

## Guide-level explanation

You attach a curve to any animatable property with `with`:

```byld
Button("Save") #[
    radius:  pressed ? 3 : 10        with anim.spring(stiffness: 210, damping: 20),
    scale:   hovered ? 1.05 : 1.0    with anim.spring(),        // sensible defaults
    bg:      hovered ? accent : surf with anim.linear(200ms),
]
```

Read it as: *"the radius is 3 when pressed and 10 otherwise, and it moves there
with a spring."* When `pressed` flips, the CPU computes the new **target**
(3 or 10) once. The transition from wherever the property currently is, to that
target, is drawn by the GPU. You never hold a handle, never tick a controller,
never write an `onFrame`.

Parentheses are optional by design — `with anim.spring()` and
`with anim.spring(stiffness: 210, damping: 20)` both parse. `with` binds looser
than the ternary `? :`, so the whole conditional is the animated value and the
`anim.*` call is the curve.

**What can be animated:** the *paint-time* set that never changes layout —
`opacity`, `bg`/`color`, `radius`, and the transform props from RFC-0011
(`translate`, `scale`, `rotate`). These are per-instance shader parameters, so
the GPU can interpolate them for free. **Layout-affecting** props (`width`,
`height`, `padding`, `gap`, flex) are *not* GPU-animatable in this RFC — see
§"Layout properties" for why and what happens if you try.

**Curves:** `anim.linear(duration)`, `anim.ease(duration)` (and named easings),
and `anim.spring(stiffness, damping, [initial_velocity])`. Springs are
duration-free and physically continuous under interruption (the classic reason to
prefer them for interactive UI).

## Reference-level explanation

### Grammar & AST (RFC-0002 Pratt parser)

`with` becomes a contextual keyword. Two viable encodings; this RFC picks (A):

- **(A) New `Token::With`** emitted by the lexer for the identifier `with` when
  it appears in operator position. Cleaner diagnostics; `with` becomes reserved
  inside `#[...]` values.
- (B) Keep it an `Ident` and special-case it in `led`. Rejected: fragile,
  leaks `with` as a usable variable name into value position.

In `parser/expr.rs::peek_led`, register `With` as an infix operator with binding
power **`left: 3, right: 3`** — strictly *below* the ternary (`Question` is
`left: 4`) so `a ? b : c with k` groups as `(a ? b : c) with k`, and strictly
*above* assignment (`left: 2`) so it never captures a `=`. In `parse_led`, on
`With`, parse the RHS with `parse_expr(3)` (an `anim.*` call/member) and build:

```rust
// parser/ast.rs
pub enum Expr {
    // ...existing variants
    Animated {
        value: Box<Expr>,        // the scalar/ternary target expression
        anim:  Box<Expr>,        // the anim.* call, resolved to AnimationSpec at lower time
        span:  Span,
    },
}
```

`anim` stays an `Expr` (an ordinary call) at parse time; it is *resolved* to a
typed `AnimationSpec` during lowering so the parser needs no knowledge of the
curve catalog (keeps layers decoupled, mirrors D6).

### Typed animation spec (lowering)

```rust
// interp/anim.rs (new)
#[derive(Clone, Copy, PartialEq)]
pub enum Curve {
    Linear { ms: u32 },
    Ease   { ms: u32, kind: EaseKind },
    Spring { stiffness: f32, damping: f32, v0: f32 },
}
```

Lowering validates: `anim.spring(...)` only accepts `stiffness`/`damping`/
`initial_velocity`; `anim.linear`/`anim.ease` require a duration; any other name
is a `CompileError::UnknownAnimation` with Levenshtein suggestion (D4 style).
An `Animated { value }` whose `value` resolves to a **layout property** is a
`CompileError::LayoutPropNotAnimatable` (see §"Layout properties").

### The animatable value model — `Motion<T>` (CPU side, zero loop)

Each animated property on an element gets a tiny per-instance record, allocated in
the same view-scoped arena as the element (RFC-0001 arena discipline, no GC):

```rust
// A paint-time animatable scalar. No per-frame CPU work: only written on target change.
#[repr(C)]
pub struct Motion {
    from:      f32,     // value at the moment the target last changed
    to:        f32,     // current target (CPU O(1) on mutation)
    start_ms:  u32,     // absolute engine time when `to` was set
    curve:     Curve,   // packed to a POD for the GPU
}
```

**Tick (CPU), on mutation only.** RFC-0004's tick recomputes the target `to`. If
`to` changed, the engine samples the *current* on-screen value `v(now)` (closed
form — see below), writes it into `from`, sets `start_ms = now`, and marks the
element dirty. This is the "interruptible spring" behaviour: a mid-flight reversal
starts from the real current position and velocity, not from the old endpoint.
Cost: one closed-form evaluation, `O(1)`, only when the target actually changes.

**Frame (GPU).** The `Motion` is packed into the per-instance data of its
primitive (`BoxInstance`/`DecoratedBox`/`TextLine`/transform block — see
RFC-0011). The shader receives the **global engine time** as a uniform and
evaluates the curve per vertex/fragment:

```wgsl
// analytic underdamped spring: x(t) = target + (from-target)*e^{-ζω t}(cos ω_d t + ...)
fn spring(from: f32, to: f32, t: f32, k: f32, c: f32) -> f32 {
    let omega = sqrt(k);
    let zeta  = c / (2.0 * sqrt(k));
    let dx    = from - to;
    if (zeta < 1.0) {                       // underdamped
        let wd = omega * sqrt(1.0 - zeta*zeta);
        let e  = exp(-zeta * omega * t);
        return to + e * (dx * cos(wd*t) + ((zeta*omega*dx) / wd) * sin(wd*t));
    } else {                                // critically / overdamped: closed forms
        let e = exp(-omega * t);
        return to + dx * e * (1.0 + omega*t);
    }
}
```

Linear/eased curves are trivial `clamp(t/ms, 0, 1)` remaps. The CPU never touches
these; the GPU redraws the interpolated value every frame the animation is active.

### The active-animation set — settling & frame scheduling (the correctness fix)

A spring's analytic value approaches the target asymptotically; the GPU would
happily redraw forever. To respect the efficiency floor, the **CPU owns a
`Vec<MotionId>` of still-moving animations** (again arena-scoped, `!Send`, logic
thread). Per tick:

1. For each active `Motion`, evaluate `v(now)` **and** the analytic velocity
   `v'(now)` (both closed form, one FMA-heavy expression).
2. If `|v(now) - to| < EPS_POS` **and** `|v'(now)| < EPS_VEL`, snap `from = to`,
   mark the `Motion` **settled**, and remove it from the active set.
3. **Frame scheduling:** the runner requests a redraw next frame **iff the active
   set is non-empty** (or an input/reactive mark occurred). When the set empties,
   the app returns to its idle, zero-frame state. This is what keeps an idle
   animated UI at 0% GPU.

`EPS_POS`/`EPS_VEL` are per-unit (px, ratio, degrees, color channel) and tunable;
defaults chosen so settling is imperceptible (<0.5px, <0.5px/s).

### Threading & the `frame.rs` boundary (RFC-0001)

`Motion` is `!Send` and lives on the logic thread with the arena; only the
**packed POD** (`from,to,start_ms,curve`) crosses into `RenderFrame` — the same
atomic frame hand-off already used for every primitive. The render thread reads
the global clock uniform; it never reads a `Signal` and never writes back. This
preserves INV-1/INV-2 (no back-dependency; only `Send` PODs cross threads).

### Layout properties

Animating `width`/`height`/`padding`/`gap`/flex would require re-running Taffy per
frame — exactly the cost this design avoids, and a relayout cannot happen in the
shader. This RFC **rejects** GPU animation of layout props: `size: … with anim…`
is a compile error `LayoutPropNotAnimatable` with a note pointing at the two
supported alternatives — (1) animate a *transform* `scale` instead (visual size,
no relayout), or (2) a future CPU-driven layout-tween (deferred; see Future).
Being explicit here prevents the "why does animating width tank my FPS" trap.

## Drawbacks

- Adds a reserved contextual keyword (`with`) and a new `Expr` variant + shader
  paths. More surface to test and document.
- The paint-time/layout-time split is a real constraint the dev must learn
  (mitigated by a precise compile error, not a silent slowdown).
- Per-instance `Motion` PODs grow the vertex/instance buffers slightly for
  animated elements (bounded: only animated props pay).

## Rationale and alternatives

- **GPU-analytic springs vs CPU tween loop.** The closed form removes all
  per-frame CPU cost and is exact under interruption; a CPU loop costs cycles per
  property per frame and needs its own scheduler.
- **`with` infix vs a `#[animate(...)]` block.** Infix keeps the curve *next to
  the value it animates* (granularity is visible in the source) and reads as prose.
- **Springs default, easing available.** Springs interrupt gracefully (interactive
  UI); fixed-duration easings remain for entrance/exit choreography.
- **Not doing it:** motion stays impossible without dropping to Rust controllers,
  which breaks the two-file model and the "no references" promise.

## Prior art

SwiftUI `.animation(_:value:)` (per-value, implicit, spring-first), React-Spring
/ Framer Motion (spring physics for UI), Robert Penner easing, the Apple/Google
"interruptible spring" interaction model, and GPU-side vertex animation from game
engines. Copy-and-Patch-style closed-form evaluation on GPU mirrors how shaders
already parameterize time-based effects.

## Resolved decisions (2026-07-01)

- **A1 — `with` token:** reserved **`Token::With`** (contextual only inside `#[...]`).
  Clean diagnostics; unambiguous. (Rejected: contextual `led` special-case — fragile.)
- **A2 — default spring:** **`stiffness: 210, damping: 20`** ("snappy", iOS-feel,
  good on 60/120/144 Hz) + named presets (`.gentle/.snappy/.bouncy`) as future sugar.
- **A3 — color interpolation space:** **OKLab** — perceptually uniform transitions
  (no muddy midpoints); the GPU conversion cost is negligible. Aligns with "beautiful
  by default."
- **A4 — `EPS_POS`/`EPS_VEL`:** **internal fixed constants per unit** (not public):
  <0.5px position, <0.5px/s velocity, 1/256 per color channel. Exposing them would
  let devs break settling (battery). Revisable internally.
- **A5 — `Motion` packing:** **inline 4 with arena spill** (mirrors IMPL-01's
  `SmallVec<[_;4]>`); covers opacity+scale+bg+radius without heap.

## Unresolved questions (deferred to implementation)

- [ ] Exact OKLab↔sRGB conversion placement (vertex vs fragment) for color springs.
- [ ] Named-preset spring constants once A2's default is validated on-device.

## Future possibilities

- CPU-driven **layout-tween** track for the deliberately-excluded layout props
  (measured against the profiler from RFC-0013 before shipping).
- Keyframed/staggered choreography built on the same `Motion` primitive.
- `anim.spring` presets (`.gentle`, `.snappy`, `.bouncy`) as named constants.
