# RFC-0020: Path & Shape Primitives — arcs, circles, and custom vector drawing

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-10
- **Last updated:** 2026-07-10
- **Depends on:** RFC-0001 (§3.1 render pipelines, `frame.rs`), RFC-0005 (intrinsic catalog), RFC-0009 (VectorMSDF pipeline, MSDF atlas), RFC-0010 (animations — animatable path properties), RFC-0011 (transforms).
- **Extends:** RFC-0009 (the VectorMSDF pipeline gains programmatic shape support beyond SVG-file icons).
- **Enables:** Circular progress indicators, loading spinners, arc-based gauges, custom shape decorations, clip masks. Unblocks `byard-material` circular progress and `byard-cupertino` activity indicators.

---

## Summary

Add a **`Canvas`** intrinsic and a **path-drawing DSL** within `byld` that lets
developers define arbitrary 2D vector shapes — arcs, circles, lines, beziers —
rendered through the existing VectorMSDF pipeline (RFC-0009) or a new lightweight
**stroke pipeline** for non-filled paths. Shapes are declared inline, animated
per-property (arc sweep, stroke dash offset), and rasterized at the same MSDF
quality as `VectorIcon`.

```byld
// Circular progress indicator
Canvas #[width: 48, height: 48] {
    arc(cx: 24, cy: 24, r: 20,
        start: -90, sweep: progress * 360,
        stroke: 0x6750A4, stroke_width: 4, cap: round)
    arc(cx: 24, cy: 24, r: 20,
        start: -90, sweep: 360,
        stroke: 0xE8DEF8, stroke_width: 4, cap: round)
}
```

---

## Motivation

The byard-material gap analysis identified **arc/path drawing** as a hard blocker
for circular progress indicators, loading spinners, and arc-based gauges. The
VectorMSDF pipeline (RFC-0009) renders SVG-sourced icons but has no way to define
shapes programmatically — you can't draw a partial circle whose sweep angle is
bound to a reactive `var`.

Both Material and Cupertino rely heavily on circular shapes: Material's circular
progress, Cupertino's `UIActivityIndicatorView` spinner, gauge charts, pie
charts, ring counters. Without programmatic path drawing, these are impossible.

Beyond the design-system need, a `Canvas` primitive enables custom decorations
(wave backgrounds, curved dividers, pie charts, sparklines) that distinguish
polished apps from boxed layouts.

---

## Guide-level explanation

### The `Canvas` intrinsic

`Canvas` is a fixed-size drawing surface. Its children are **shape commands**,
not View elements:

```byld
Canvas #[width: 200, height: 200] {
    // Background track
    arc(cx: 100, cy: 100, r: 80,
        start: 0, sweep: 360,
        stroke: 0xE8DEF8, stroke_width: 6)

    // Progress arc — sweep is reactive
    arc(cx: 100, cy: 100, r: 80,
        start: -90, sweep: percent * 3.6,
        stroke: 0x6750A4, stroke_width: 6, cap: round)
        with anim.spring()

    // Center text
    text("75%", x: 100, y: 100,
         color: 0x1D1B20, size: 24, align: center)
}
```

### Shape commands

| Command | Parameters | Description |
|---|---|---|
| `arc` | `cx, cy, r, start, sweep` + stroke/fill | Circular arc |
| `circle` | `cx, cy, r` + stroke/fill | Full circle (sugar for `arc` with sweep=360) |
| `line` | `x1, y1, x2, y2` + stroke | Line segment |
| `rect` | `x, y, w, h, radius?` + stroke/fill | Rectangle |
| `path` | `d: "M10 10 L90 90 ..."` + stroke/fill | SVG path data |
| `bezier` | `x1,y1, cx1,cy1, cx2,cy2, x2,y2` + stroke | Cubic bezier |
| `text` | `content, x, y` + text props | Text at coordinates |

### Stroke and fill properties

Every shape accepts:

| Prop | Type | Default | Meaning |
|---|---|---|---|
| `stroke` | `Color` | none | stroke color |
| `stroke_width` | `Float` | `1.0` | stroke width in logical px |
| `cap` | `butt\|round\|square` | `butt` | line cap style |
| `join` | `miter\|round\|bevel` | `miter` | line join style |
| `fill` | `Color` | none | fill color |
| `dash` | `(Float, Float)` | none | dash pattern (length, gap) |
| `dash_offset` | `Float` | `0.0` | dash phase offset (animatable!) |
| `opacity` | `Float` | `1.0` | shape opacity |

### Animation

Shape properties are animatable with `with` (RFC-0010):

```byld
// Indeterminate spinner — rotating dash offset
arc(cx: 24, cy: 24, r: 20,
    start: spin_angle with anim.linear(1000ms, repeat: infinite),
    sweep: 270,
    stroke: 0x6750A4, stroke_width: 4, cap: round,
    dash: (60, 200), dash_offset: 0)
```

`start`, `sweep`, `dash_offset`, `stroke_width`, `opacity`, and `fill`/`stroke`
colors are all in the GPU-animatable set (RFC-0010).

---

## Reference-level explanation

### 1. The `Canvas` intrinsic

Added to RFC-0005's catalog:

- **Content:** none. **Children:** shape commands only
  (`CompileError::UnexpectedChildren` for View/intrinsic children).
- **Props:** Layout (`width`, `height` required; `grow` applies), `bg: Color`
  (background fill), universal `style`.
- **Events:** all pointer (the canvas rect is hit-testable).
- **Pipeline:** new `CanvasShape` pipeline (see §2), or VectorMSDF for
  pre-rasterized complex paths.

### 2. Rendering strategy

Two tiers, chosen per shape:

**Tier 1: Analytic stroke shader (new pipeline).** For arcs, circles, lines,
and rects — shapes with closed-form SDF functions. A fragment shader computes
the signed distance to the shape and applies stroke/fill/anti-aliasing
analytically. This is resolution-independent, requires no atlas allocation,
and is trivially animatable (the shader reads `start`/`sweep` from a uniform
that the CPU updates per frame only when animated).

```wgsl
// Simplified arc SDF
fn arc_sdf(p: vec2<f32>, center: vec2<f32>, radius: f32,
           start_rad: f32, sweep_rad: f32) -> f32 {
    let d = distance(p, center);
    let angle = atan2(p.y - center.y, p.x - center.x);
    let in_sweep = angle_in_range(angle, start_rad, sweep_rad);
    return select(1e10, abs(d - radius), in_sweep);
}
```

Per-instance data (`CanvasShapeInstance`):

```rust
struct CanvasShapeInstance {
    rect: Rect,              // canvas viewport rect
    shape_type: u32,         // arc, circle, line, rect
    params: [f32; 8],        // shape-specific: cx,cy,r,start,sweep,...
    stroke_color: [f32; 4],
    fill_color: [f32; 4],
    stroke_width: f32,
    cap_join: u32,           // packed cap + join enum
    dash_pattern: [f32; 2],
    dash_offset: f32,
    opacity: f32,
    transform: Transform2D,  // from RFC-0011
}
```

**Tier 2: VectorMSDF rasterization.** For `path(d: "...")` commands with complex
SVG path data — these are too expensive for analytic SDF. The path is tessellated
into the MSDF atlas (RFC-0009) at the requested size and rendered as a textured
quad. This path is **not real-time animatable** (re-rasterization on geometry
change), but color/opacity/transform still animate on the GPU.

### 3. Grammar extension

Shape commands are parsed as a new `Member::ShapeCmd` AST node, valid only inside
a `Canvas`. The parser recognizes `arc(`, `circle(`, `line(`, `rect(`, `path(`,
`bezier(`, `text(` as shape-command keywords when inside a `Canvas` body.
Outside `Canvas`, they are `CompileError::ShapeOutsideCanvas`.

### 4. Clip masks (future, designed for)

`Canvas` naturally extends to clip masks:

```byld
Box #[clip: Canvas { circle(cx: 50, cy: 50, r: 50) }] {
    Image("avatar.png")
}
```

The `clip` prop takes a `Canvas` whose shapes define the clip region. This is
designed for but deferred — the initial implementation focuses on visible shapes.

---

## Drawbacks

- **New render pipeline.** The `CanvasShape` pipeline adds a vertex/fragment
  shader pair and a new instance buffer to the encoder. It's the sixth pipeline
  (after `SolidBox`, `DecoratedBox`, `TextGlyph`, `TextureSampler`, `VectorMSDF`).
- **Shape grammar.** Shape commands are a mini-DSL within `byld`. While they
  reuse the same `name(key: value)` syntax as intrinsics, they're semantically
  different (no children, no events per shape, no style states). The parser must
  distinguish them contextually.
- **Not a general-purpose 2D renderer.** `Canvas` is for shapes and decorations,
  not for building a game or a drawing app. Complex real-time vector animation
  (Lottie-style) is out of scope.

---

## Rationale and alternatives

**Why analytic SDF, not tessellation?** Tessellation (triangulating shapes into
meshes) is the traditional approach but requires re-tessellation when geometry
changes (bad for animation) and produces aliased edges without MSAA. Analytic
SDF gives perfect anti-aliasing, resolution independence, and zero-cost
animation (just update a uniform). The tradeoff is that only simple shapes have
closed-form SDFs — complex paths fall back to Tier 2 (VectorMSDF).

**Why not extend `VectorIcon` with programmatic paths?** `VectorIcon` is
purpose-built for monochrome icon rendering from SVG files. Adding real-time
animated shapes to it would violate its single-responsibility (JIT atlas
allocation, MSDF baking) and its optimization assumptions (static geometry,
batch-upload). A separate `Canvas` keeps both simple.

**Why not a `<canvas>`-style imperative API?** Imperative drawing (`ctx.moveTo`,
`ctx.arc`, ...) requires a reference to a drawing context — exactly what
RFC-0003 forbids. Declarative shape commands fit the reactive model: when
`percent` changes, the arc's `sweep` updates automatically.

---

## Prior art

- **SwiftUI `Canvas` / `Path`:** declarative shape drawing with `Path { ... }`.
  Direct inspiration for `Canvas` + shape commands.
- **Flutter `CustomPainter`:** imperative `canvas.drawArc(...)`. Powerful but
  requires a class, a paint object, and a `shouldRepaint` callback.
- **Jetpack Compose `Canvas` + `DrawScope`:** lambda-based drawing. Closer to
  imperative than declarative.
- **CSS `conic-gradient` + `clip-path`:** hack-level arc rendering. Byard's
  `Canvas` is the proper primitive.
- **Skia / vello / lyon:** GPU-accelerated 2D renderers. Byard's analytic SDF
  approach is more constrained but cheaper for the simple shapes UI needs.

---

## Unresolved questions

- **Before merge:**
  - [ ] **Gradient support.** Should `fill` accept gradients (`linear(...)`,
    `radial(...)`) or only solid colors? Recommendation: solid only in v1;
    gradients are a separate concern.
  - [ ] **Shape hit-testing.** Are individual shapes within a Canvas
    hit-testable, or only the Canvas rect? Recommendation: Canvas rect only;
    per-shape hit-testing adds complexity for little gain.
  - [ ] **Tier selection.** Is the tier (analytic vs. MSDF) automatic based on
    shape type, or can the developer force one? Recommendation: automatic.

- **During implementation:**
  - [ ] **Anti-aliasing quality.** The analytic SDF approach needs tuning for
    thin strokes at small sizes. Test with 1px strokes on high-DPI displays.
  - [ ] **Batch limits.** How many shape instances per Canvas before performance
    degrades? Set a reasonable limit with a diagnostic.

---

## Future possibilities

- **Clip masks** (`clip: Canvas { ... }`) for circular image crops, custom
  shaped containers.
- **Gradient fills** — linear, radial, conic gradients as fill types.
- **SVG path animation** — morph between two SVG path strings (interpolate
  control points).
- **Sparkline / chart primitives** — sugar on top of `Canvas` for common data
  visualization patterns.
- **`Canvas` children as children of `ZStack`** — compositing drawn shapes
  with laid-out Views.
