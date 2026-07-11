# RFC-0023: Paint Effects — ripple, blur, vibrancy, and composited visual effects

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-10
- **Last updated:** 2026-07-10
- **Depends on:** RFC-0001 (§3.1 render pipelines, `frame.rs`), RFC-0003 (pointer events — tap coordinates for ripple origin), RFC-0010 (animation — the effects animate), RFC-0011 (opacity/transform), RFC-0012 (style states — effects triggered by hover/pressed).
- **Enables:** Material ripple/ink effect, iOS frosted-glass blur/vibrancy, custom visual effects. Completes the visual fidelity gap between Byard and native platform renderers.

---

## Summary

Add three **paint-time visual effects** as style properties and a backing
**effects pipeline** that composites them at render time:

1. **`ripple`** — a radial ink reveal originating from the tap point, expanding
   and fading. The signature Material interaction feedback.
2. **`blur`** — a Gaussian (or box) blur applied to the element's background,
   reading pixels behind it. The foundation of iOS vibrancy/frosted glass.
3. **`backdrop_tint`** — a colored overlay blended with the blurred background.
   Combined with blur, it produces the vibrancy effect.

All three are **style properties** (usable in `style {}` blocks, animatable with
`with`, driven by style states), not intrinsics. They are paint-time only — they
never affect layout.

---

## Motivation

### Material ripple

Every Material interactive surface shows a ripple on tap. `byard-material`
currently approximates this with `on pressed { bg: darkerColor }` — a flat color
swap, not a radial reveal. The difference is viscerally obvious: flat swaps feel
dead, ripples feel responsive. Material Design's identity is built on ink
physics; without it, the package looks like a wireframe.

### iOS blur/vibrancy

Cupertino's visual language is defined by layered translucency: navigation bars,
tab bars, sheets, and notification panels show blurred content behind them. A
`byard-cupertino` package without blur would be visually unrecognizable as iOS.
This is not a nice-to-have; it's the platform's core visual identity.

### Both require compositor-level support

Neither effect can be faked at the `byld` level: ripple needs tap coordinates
and a time-driven expansion animation; blur needs to sample pixels behind the
element. Both are inherently render-pipeline concerns.

---

## Guide-level explanation

### Ripple

```byld
let btn = style {
    bg: 0x6750A4, radius: 20
    ripple: 0xFFFFFF       // ripple color (semi-transparent white)
    on pressed { ripple_active: true }
}

Button("Save") #[..btn]
```

When `pressed` becomes true, a ripple circle begins expanding from the tap point
(the pointer coordinates at `pointer_down`). It fades out over ~300ms. The
ripple is rendered as a composited circle *above* the element's background but
*below* its children — the text remains crisp on top of the ripple.

Ripple properties:

| Prop | Type | Default | Meaning |
|---|---|---|---|
| `ripple` | `Color` | none (disabled) | ripple color; typically semi-transparent |
| `ripple_active` | `Bool` | `false` | trigger the ripple animation |
| `ripple_radius` | `Float` | auto (expands to cover the element) | max radius |
| `ripple_duration` | `Duration` | `300ms` | fade-out duration |

### Blur and vibrancy

```byld
let glass = style {
    blur: 20                   // 20px Gaussian blur on background
    backdrop_tint: 0x80FFFFFF  // semi-transparent white overlay
    radius: 16
}

// iOS-style navigation bar
Row #[..glass, height: 44, p: (0, 16)] {
    Text("Title") #[color: 0x000000, weight: medium]
}
```

The `blur` prop samples the scene behind the element (everything already
rendered below it in the painter's algorithm), applies a Gaussian blur, and draws
the blurred result as the element's background. `backdrop_tint` is blended on
top of the blur for the vibrancy color cast.

Blur properties:

| Prop | Type | Default | Meaning |
|---|---|---|---|
| `blur` | `Float` | `0` (disabled) | blur radius in logical px |
| `backdrop_tint` | `Color` | transparent | color blended on blurred backdrop |
| `blur_saturation` | `Float` | `1.8` | saturation boost (iOS vibrancy look) |

### Combined with animation

```byld
let frosted = style {
    blur: 0 with anim.spring()
    backdrop_tint: 0x00FFFFFF with anim.spring()
    on hover {
        blur: 16
        backdrop_tint: 0x80FFFFFF
    }
}
```

A card that gains a frosted-glass effect on hover, animated smoothly.

---

## Reference-level explanation

### 1. Ripple rendering

The ripple is rendered as a **composited radial mask** between the element's
background and its children:

1. On `pointer_down`, record the tap coordinates relative to the element rect.
2. When `ripple_active` becomes true, create a `RippleInstance`:
   ```rust
   struct RippleInstance {
       center: Vec2,       // tap point relative to element
       color: Color,
       max_radius: f32,    // auto-computed: distance to farthest corner
       start_time: f64,    // engine time at activation
       duration: f32,      // fade-out duration
       element_rect: Rect, // for clipping
       corner_radii: [f32; 4], // clip to element's border radius
   }
   ```
3. The ripple shader draws a filled circle at `center` with radius interpolated
   from 0 to `max_radius` over `duration`, with alpha fading from the color's
   alpha to 0. The circle is clipped to the element's rounded rect.
4. Compositing order: background → ripple → children. The ripple sits in the
   `DecoratedBox` pipeline as an optional effect layer.

The ripple animation is **GPU-driven** (RFC-0010 model): the CPU sets `start_time`
once, and the shader computes radius and alpha each frame from elapsed time.
Active ripples are in the RFC-0010 active-animation set and stop requesting
frames once fully faded.

Multiple simultaneous ripples (rapid tapping) are supported — each tap creates
a new `RippleInstance`, and they blend additively.

### 2. Blur rendering

Blur requires **render-to-texture** for the scene behind the element:

1. The encoder renders all primitives *below* the blurred element into the main
   render target as normal.
2. Before rendering the blurred element, copy the region behind it (the element's
   rect in screen coordinates) from the render target into a temporary texture.
3. Apply a **two-pass Gaussian blur** (horizontal then vertical) to the temporary
   texture. The kernel size is derived from `blur` radius.
4. Draw the blurred texture as the element's background (with `backdrop_tint`
   blended on top).
5. Continue rendering the element's children and subsequent elements on top.

**Performance considerations:**

- The texture copy + two-pass blur is expensive. It runs once per blurred element
  per frame — not per pixel.
- For static scenes, the blurred background can be cached until the content
  behind it changes (tracked via dirty rectangles, RFC-0001 §3.3).
- Blur radius is clamped to a maximum (e.g., 40px) to bound kernel size.
- On low-end GPUs, blur falls back to a box blur (cheaper, lower quality) or
  is disabled entirely with a `CompileWarning`.

### 3. Style property integration

Both `ripple` and `blur` are style properties (RFC-0016), usable in `style {}`
blocks with `on <state>` and `with anim`:

```rust
enum StyleProperty {
    // ... existing (bg, color, radius, scale, translate, rotate, opacity)
    Ripple(Color),
    RippleActive(bool),
    Blur(f32),
    BackdropTint(Color),
    BlurSaturation(f32),
}
```

`blur` and `backdrop_tint` are in the GPU-animatable set (RFC-0010): the shader
interpolates them per frame. `ripple_active` is a boolean trigger, not a
continuous value — it starts the ripple animation.

### 4. Effects pipeline

A new render pass stage: **effects compositing**, inserted between
`DecoratedBox` background and child rendering:

```
Background (SolidBox / DecoratedBox)
  → Blur sampling (if blur > 0)
    → Backdrop tint blend (if backdrop_tint set)
      → Ripple overlay (if ripple_active)
        → Children (TextGlyph, VectorMSDF, etc.)
```

This is per-element, not a global post-processing pass. Only elements with
effects enabled run through the effects stage — zero cost for normal elements.

---

## Drawbacks

- **Blur is expensive.** The render-to-texture + two-pass blur is the most
  computationally expensive operation in Byard's pipeline. On mobile GPUs, a
  full-screen blur can consume the entire frame budget. Mitigations: caching,
  radius clamping, quality tiers.
- **Render order dependency.** Blur reads pixels already rendered — it requires
  strict painter's-order rendering, which Byard already uses. But it prevents
  future batching optimizations that might reorder draw calls.
- **Platform inconsistency.** The blur effect looks slightly different across
  GPU vendors due to floating-point precision in the Gaussian kernel. This is
  acceptable for a visual effect (not a correctness concern).
- **Three new style properties** add to the already-growing property namespace.

---

## Rationale and alternatives

**Why style properties, not intrinsics?** `Ripple` and `Blur` are visual
properties of an element, not elements themselves. Making `Ripple` an intrinsic
(`Ripple { Button("Save") }`) creates the wrapper-hell RFC-0001 rejects. A
style property (`ripple: 0xFFFFFF` on the Button itself) is the Byard way.

**Why not a general shader/effect system?** A user-definable shader pipeline
would be powerful but adds massive complexity (shader compilation, safety
validation, GPU resource management). Ripple and blur cover 95% of real-world
effect needs for Material and Cupertino. A general system can layer on top later.

**Why GPU-driven ripple, not a CPU-animated circle overlay?** A CPU-animated
overlay would require a render-tree mutation every frame (adding/resizing a
circle node). The GPU-driven approach updates zero state after the initial
trigger — consistent with RFC-0010's "CPU computes target once, GPU drives the
curve" model.

---

## Prior art

- **Flutter `InkWell` / `InkResponse`:** Material ripple via `RenderBox` paint
  override. Byard achieves the same with a style property.
- **iOS `UIVisualEffectView` / `UIBlurEffect`:** system-level backdrop blur.
  The gold standard for vibrancy.
- **CSS `backdrop-filter: blur(20px)`:** the web equivalent. Direct inspiration
  for Byard's `blur` prop name.
- **Jetpack Compose `Modifier.blur()`:** modifier-based blur. Similar to Byard's
  style-property approach.
- **wgpu blur implementations:** common pattern is two-pass separable Gaussian
  with compute shaders or fragment shaders.

---

## Unresolved questions

- **Before merge:**
  - [ ] **Blur quality tiers.** Should the engine auto-select quality (Gaussian
    vs. box blur) based on GPU capability, or let the developer choose?
    Recommendation: auto-select with an optional `blur_quality` prop.
  - [ ] **Ripple clipping.** Should ripple clip to the element's border-radius
    automatically? Recommendation: yes, always — an unclipped ripple looks
    wrong.
  - [ ] **Blur on `Canvas`.** Can a `Canvas` (RFC-0020) element have a blur
    background? Recommendation: yes, the blur sampling doesn't care about the
    element type.

- **During implementation:**
  - [ ] **Blur texture resolution.** Full resolution or downsampled (e.g., 0.5x)?
    Downsampled is cheaper and the blur hides the resolution loss.
  - [ ] **Multiple blurred elements overlap.** If two blurred elements overlap,
    does the upper one blur the lower one's blur? Recommendation: yes (natural
    painter's order), but flag if performance is poor.

---

## Future possibilities

- **Custom shader effects** — a general `effect: shader("...")` prop for
  user-defined fragment shaders (post-MVP).
- **Drop shadow blur** — `DecoratedBox` shadow could use the same blur pipeline
  for soft shadows instead of approximations.
- **Motion blur** — during fast scroll or fling, apply directional blur to
  content. Expensive but cinematic.
- **Color matrix effects** — grayscale, sepia, hue-rotate applied as paint-time
  transformations (CSS `filter` equivalents).
