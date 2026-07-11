# RFC-0025: Looping & Indefinite Animations — repeat, reverse, keyframes, stagger

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-10
- **Last updated:** 2026-07-10
- **Depends on:** RFC-0001 (§5 concurrency, `frame.rs`), RFC-0010 (animation system — `with` syntax, `Motion` runtime, active-set settling), RFC-0011 (transform/paint-time properties — the animatable set), RFC-0020 (Canvas — animated shape properties like `start`/`sweep`/`dash_offset`).
- **Extends:** RFC-0010 with repeat/reverse/infinite modifiers and a keyframe sequence syntax.
- **Enables:** Indeterminate progress bars, loading spinners, shimmer/skeleton effects, pulsing badges, rotating icons. Completes the last animation gap for `byard-material` and `byard-cupertino`.

---

## Summary

Extend RFC-0010's `with anim.*()` system with three capabilities:

1. **Repeat modifiers** — `repeat: N`, `repeat: infinite`, `reverse: true` on
   any `anim.*()` curve.
2. **`anim.keyframes()`** — a multi-step sequence of timed values for a single
   property, replacing the single from→to model.
3. **`anim.stagger()`** — delay offsets for children in a `for` loop, producing
   wave/cascade effects.

All three are **declarative extensions to the `with` syntax**, GPU-evaluated
(RFC-0010's model), and compatible with the active-set settling system (infinite
animations keep the active set alive; finite ones remove themselves on
completion).

---

## Motivation

RFC-0010 gives Byard beautiful one-shot transitions: a property springs from
value A to value B when its reactive source changes. But several essential UI
patterns need **continuous or multi-step motion**:

- **Indeterminate progress** (Material, Cupertino): a bar or arc that cycles
  endlessly while `loading` is true.
- **Loading spinner** (Cupertino activity indicator): a rotating set of arcs
  fading in/out in sequence.
- **Shimmer/skeleton** (Material): a highlight sweep across placeholder blocks.
- **Pulsing badge**: a notification dot that gently scales up/down to draw
  attention.
- **Page transition**: a slide-in from the right with an overshoot bounce.

RFC-0010's `with anim.spring()` can't express these: a spring goes from A to B
and settles. There's no "keep going," no "go A→B→C," no "start this one after
that one."

---

## Guide-level explanation

### Repeat and reverse

```byld
// Pulsing badge — scales up and down forever
Box #[width: 12, height: 12, radius: 6, bg: 0xB3261E,
      scale: 1.3 with anim.spring(repeat: infinite, reverse: true)]
```

`repeat: infinite` restarts the animation when it settles. `reverse: true` plays
the animation back-to-front on even iterations (A→B→A→B→...). Together they
create a continuous oscillation.

`repeat: 3` plays the animation exactly 3 times, then stops (the property holds
its final value).

### Controlling start/stop

Looping animations are **conditional on their reactive source**:

```byld
var loading = true

Canvas #[width: 48, height: 48] {
    when loading {
        arc(cx: 24, cy: 24, r: 20,
            start: 0 with anim.linear(800ms, repeat: infinite),
            sweep: 270,
            stroke: 0x6750A4, stroke_width: 4, cap: round)
    }
}
```

When `loading` becomes false, the `when` unmounts the arc, which removes the
animation from the active set. No separate "stop animation" API — the animation
lives and dies with its element.

### Keyframes

```byld
// Material indeterminate linear progress
Box #[bg: 0xE8DEF8, radius: 2, grow: 1, height: 4] {
    Box #[bg: 0x6750A4, radius: 2, height: 4,
          width: anim.keyframes(
              0%: 0,     // start narrow
              50%: 200,  // expand
              100%: 0    // contract
          , loop: true, duration: 2000ms),
          translate: anim.keyframes(
              0%: (-100, 0),
              50%: (50, 0),
              100%: (300, 0)
          , loop: true, duration: 2000ms)]
    }
}
```

`anim.keyframes()` defines a sequence of (percentage, value) pairs animated over
`duration`. `loop: true` is sugar for `repeat: infinite`. Each segment between
keyframes is linearly interpolated by default; an optional easing per segment
can be specified:

```byld
width: anim.keyframes(
    0%: 0,
    50%: 200 ease_out,    // ease out from 0% to 50%
    100%: 0 ease_in       // ease in from 50% to 100%
, duration: 2000ms, loop: true)
```

### Stagger

```byld
Column #[gap: 4] {
    for i, item in items {
        Box #[opacity: 1.0 with anim.spring(delay: i * 50ms),
              translate: (0, 0) with anim.spring(delay: i * 50ms)] {
            Text(item.name)
        }
    }
}
```

`delay: Duration` on any `anim.*()` offsets the animation start by that amount.
Combined with a `for` loop's index `i`, it produces staggered entrance
animations where each item appears slightly after the previous one.

A convenience `anim.stagger(base: spring(), step: 50ms)` syntax is sugar for
the `delay: i * step` pattern:

```byld
for i, item in items {
    Text(item.name) #[opacity: 1.0 with anim.stagger(spring(), 50ms, i)]
}
```

---

## Reference-level explanation

### 1. Motion extensions

RFC-0010's `Motion` struct gains repeat/reverse fields:

```rust
struct Motion {
    curve: Curve,           // Spring { stiffness, damping } | Linear(ms) | Ease(ms, fn)
    target: f64,            // target value
    current: f64,           // current animated value
    velocity: f64,          // for springs
    start_time: f64,        // engine time at animation start
    // --- new ---
    repeat: RepeatMode,     // Once | Count(u32) | Infinite
    reverse: bool,          // alternate direction on even iterations
    delay: f64,             // delay before first start (seconds)
    iteration: u32,         // current iteration count
}

enum RepeatMode {
    Once,
    Count(u32),
    Infinite,
}
```

### 2. Active-set behavior

Infinite animations **stay in the active set permanently** (until their element
unmounts). This means the engine continues requesting frames. To avoid burning
CPU/GPU when the app is idle but a spinner is spinning, the active set
distinguishes:

- **Visible infinite animations** — in the viewport, requesting frames.
- **Offscreen infinite animations** — outside the viewport or in a collapsed
  `when` branch. These are **paused** (removed from the active set) and
  **resumed** when they become visible again.

The active-set check runs once per frame: if all remaining animations are
offscreen, the engine stops requesting frames (idle mode).

### 3. Keyframe representation

```rust
struct Keyframes {
    steps: Vec<KeyframeStep>,  // sorted by percent 0.0–1.0
    duration: f64,             // total duration in seconds
    repeat: RepeatMode,
    reverse: bool,
}

struct KeyframeStep {
    percent: f64,              // 0.0 to 1.0
    value: AnimatableValue,    // the target value at this keyframe
    easing: Easing,            // interpolation from previous step to this one
}
```

Evaluation: at time `t`, compute `progress = (t % duration) / duration` (with
reverse flipping on even iterations). Find the two surrounding keyframe steps
and interpolate between their values using the segment's easing function.

Keyframes are evaluated on the **GPU** for the paint-time animatable set
(RFC-0011): the shader receives the keyframe steps as a small uniform buffer
and evaluates the interpolation per fragment. For layout-affecting properties
(width, height, padding — which RFC-0010 already rejects for spring animation),
keyframes are also rejected with `CompileError::LayoutPropertyNotAnimatable`.

### 4. Grammar extensions

The `with anim.*()` syntax gains optional trailing modifiers:

```
anim_expr := "anim" "." curve_fn "(" params ("," modifier)* ")"
curve_fn  := "spring" | "linear" | "ease" | "ease_in" | "ease_out"
           | "ease_in_out" | "keyframes" | "stagger"
modifier  := "repeat" ":" (int_lit | "infinite")
           | "reverse" ":" bool_lit
           | "delay" ":" duration_lit
           | "loop" ":" bool_lit          // sugar for repeat: infinite
           | "duration" ":" duration_lit  // for keyframes/linear

duration_lit := int_lit "ms" | float_lit "s"
```

Keyframes use a special argument syntax:

```
keyframes_args := keyframe_step ("," keyframe_step)*
keyframe_step  := percent ":" value (easing_name)?
percent        := int_lit "%"
```

### 5. `delay` interaction with reactive changes

When a property's reactive source changes (new target), any active delay is
**cancelled** and the animation starts immediately toward the new target. This
prevents a delayed animation from overwriting a more recent user interaction.

Exception: `anim.stagger()` delays in a `for` loop mount are not cancellable
(they represent intentional entrance staggering, not responsive transitions).

---

## Drawbacks

- **Infinite animations burn power.** Even with the offscreen optimization, a
  visible spinner means the GPU runs every frame. The active-set settling
  (RFC-0010) was designed to *stop* frame requests; infinite animations
  undermine that. Mitigation: idle detection (if no user interaction for N
  seconds, reduce animation framerate or pause).
- **Keyframe GPU evaluation.** Sending keyframe data to the GPU as uniforms
  adds per-element data. For many keyframe animations, this could pressure the
  uniform buffer. Limit: max 8 keyframe steps per property (sufficient for
  indeterminate progress, which needs 2–4 steps).
- **Stagger index dependency.** `anim.stagger(...)` implicitly depends on the
  `for` loop index, which is a new implicit binding. The compiler must verify
  `i` is in scope and is an integer.

---

## Rationale and alternatives

**Why extend `with anim.*()` instead of a separate animation system?** RFC-0010's
`with` is the established animation surface. Adding a parallel `animate { ... }`
block would fragment the animation model and create two ways to animate the same
property. Extensions to `with` keep one model, one syntax, one GPU pipeline.

**Why GPU-evaluated keyframes, not CPU-driven?** Consistency with RFC-0010: the
CPU computes targets, the GPU drives curves. Keyframes are just a multi-segment
curve — the GPU evaluates `progress → value` from the step table each frame.
Zero CPU cost per frame, same as springs.

**Why no `timeline` or `animation controller`?** Those are imperative models
that require references (RFC-0003 forbids). Declarative repeat/keyframes/stagger
express the same patterns without any handles.

---

## Prior art

- **CSS `@keyframes` + `animation-iteration-count: infinite`:** direct precedent
  for keyframes + repeat. Byard's keyframes are per-property, not per-selector.
- **SwiftUI `.repeatForever()`, `.delay()`:** method chaining on animations.
  Similar to Byard's modifier syntax.
- **Flutter `AnimationController.repeat()`:** imperative repeat. Byard's is
  declarative.
- **Framer Motion `variants` + `staggerChildren`:** stagger API. Inspiration
  for `anim.stagger()`.
- **CSS `animation-delay`:** direct precedent for `delay`.
- **Lottie:** complex multi-property keyframe animations. Byard's keyframes are
  simpler (single property per `with`), but composable across properties.

---

## Unresolved questions

- **Before merge:**
  - [ ] **Max keyframe steps.** Cap at 8 (covers all real-world UI patterns) or
    allow arbitrary? Recommendation: 8, with a `CompileError` above the limit.
  - [ ] **Easing per keyframe segment.** Is per-segment easing essential for v1,
    or can all segments be linear with a global easing? Recommendation: per-
    segment — indeterminate progress needs `ease_in_out` on its segments.
  - [ ] **`delay` on springs.** Does delay compose well with spring physics (the
    spring starts from rest after the delay)? Recommendation: yes, the spring
    simply begins at `t = start_time + delay`.

- **During implementation:**
  - [ ] **Keyframe uniform buffer format.** Pack steps as `vec4` arrays
    (percent, value, easing_id, padding) for GPU consumption.
  - [ ] **Framerate reduction for idle+animated.** When no user interaction is
    happening but an infinite animation is visible, should the engine drop to
    30fps to save power?

---

## Future possibilities

- **Orchestrated animations** — a `sequence { anim1 then anim2 then anim3 }`
  syntax for chaining animations across properties or elements.
- **Physics simulations** — gravity, bounce, friction as curve types alongside
  spring/linear/ease.
- **Lottie/Rive import** — parse a Lottie JSON into a series of keyframe
  animations applied to `Canvas` shapes.
- **Enter/exit transitions** — `on mount { ... } on unmount { ... }` animation
  blocks that play when a `when` branch mounts or unmounts (requires deferred
  unmount lifecycle).
- **Gesture-driven animations** — scroll position or drag distance as the
  animation progress input (0.0–1.0), replacing time.
