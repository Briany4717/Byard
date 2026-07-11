# RFC-0011: Transform & Paint-Time Properties

- **Status:** Active — partially implemented (M33 engine primitives, M34 attribute surface, group-transform inheritance landed). All design decisions (T1–T4) and formerly-unresolved questions resolved. Remaining: hierarchical transform stack, group opacity (render-to-texture), text transforms (glyphon limitation).
- **Author(s):** Brian (byard_v2)
- **Created:** 2026-07-01
- **Last updated:** 2026-07-01
- **Depends on:** RFC-0001 (§3.1 pipelines, `frame.rs` primitives), RFC-0005 (intrinsic attribute catalog & `Len` model), RFC-0002 (D4 attribute contract, D5 style layers).
- **Pairs with:** RFC-0010 (these are exactly the GPU-animatable set), RFC-0012 (interactive states drive them).

---

## Summary

Add a family of **paint-time transform properties** — `translate`, `scale`,
`rotate`, `origin`, and `opacity` — that reposition/resize/rotate an element
*visually* without touching layout. Transforms are applied in the vertex shader
via a per-instance 2D affine matrix baked from a decomposed TRS; Taffy layout is
computed exactly once and never re-run because a transform changed. This is the
CSS-`transform` model (transforms don't reflow), chosen precisely so it composes
for free with RFC-0010's GPU animation.

## Motivation

Byard can lay out and paint boxes, text and vectors, but has **no way to nudge,
scale, or rotate an element** — the building blocks of every micro-interaction
(a button that grows 5% on hover, an icon that spins while loading, a card that
lifts on press). `frame.rs` primitives (`BoxInstance`, `DecoratedBox`,
`TextLine`, `VectorInstance`) carry a `Rect` and `radii` but **no transform
field** today.

Doing this as a *layout* change would be a disaster: scaling a button 5% would
reflow its siblings and re-run Taffy every frame of the animation. The correct
model — the one that makes motion cheap — is a **paint-time transform** that
leaves layout geometry untouched and only rewrites how the already-laid-out quad
is drawn. That also makes transforms the natural, free-to-animate partner for
RFC-0010.

## Guide-level explanation

```byld
Card #[
    scale:     hovered ? 1.03 : 1.0,
    translate: pressed ? (0, 1) : (0, 0),   // (x, y) in logical px
    rotate:    spinning ? 360deg : 0deg,
    origin:    center,                       // pivot for scale/rotate
    opacity:   loading ? 0.5 : 1.0,
]
```

- **`translate: (x, y)`** — moves the painted element by a logical-px offset.
  Layout position is unchanged; siblings don't move.
- **`scale: n`** or **`scale: (sx, sy)`** — visual scaling about `origin`.
- **`rotate: deg`** — rotation (accepts `deg`/`rad` unit suffix) about `origin`.
- **`origin:`** — the pivot: `center` (default), `top_left`, `(px, px)`, or a
  fractional `(0.5, 0.5)`. Resolved against the element's laid-out rect.
- **`opacity: 0.0–1.0`** — element alpha (already present on `DecoratedBox`;
  this RFC promotes it to a first-class, animatable style prop for every primitive).

Every one of these is in RFC-0010's animatable set, so
`scale: hovered ? 1.03 : 1.0 with anim.spring()` "just works" and costs zero CPU.

**Transforms never affect layout or hit-testing geometry by default.** The hit
rect registered in the router (RFC-0003 §4.2) stays the laid-out rect — a scaled
button is still clickable over its layout bounds. (A future `hit_follows_transform`
opt-in is noted under Future.)

## Reference-level explanation

### Attribute grammar (RFC-0005 catalog additions)

New props, validated per-intrinsic by the §5 checker (available on every visual
intrinsic — `Box`/`Row`/`Column`/`Button`/`Text`/`Image`/vectors):

| Prop | Type | Default | Notes |
|---|---|---|---|
| `translate` | `Len` pair `(x, y)` | `(0,0)` | logical px; reuses the `Len` tuple model |
| `scale` | `Float` or `(Float,Float)` | `1.0` | uniform or per-axis |
| `rotate` | `Angle` (`deg`/`rad`) | `0deg` | new `Angle` scalar w/ unit suffix |
| `origin` | `Origin` enum/tuple | `center` | pivot; `center`/`top_left`/`(fx,fy)`/`(px,px)` |
| `opacity` | `Float 0..=1` | `1.0` | promoted from `DecoratedBox` |

`rotate` requires a small lexer/`Angle` addition (`360deg`, `1.5rad`); degrees
are canonicalized to radians at lower time. `origin` fractional vs absolute is
disambiguated by value range/`px` suffix (mirrors the `Len`/fraction split).

### Dual surface: inferred (terse) and verbose (explicit) forms

Every transform prop whose value has structure accepts **both** a terse inferred
form (fast to write, great for prototyping) and a verbose explicit form
(self-documenting, better in review/large teams). Both parse to the *same*
`Transform` — this is pure surface sugar, resolved at lower time, so there is no
runtime cost to either. The dual surface is a deliberate DX multiplier: the same
mental model reads at two levels of ceremony.

| Prop | Inferred (terse) | Verbose (explicit) | Notes |
|---|---|---|---|
| `scale` | `scale: 1.05` | `scale: (x: 1.05, y: 1.0)` | scalar = uniform on both axes; named tuple = per-axis |
| `scale` (per-axis terse) | `scale: (1.05, 1.0)` | `scale: (x: 1.05, y: 1.0)` | positional tuple = `(x, y)` |
| `translate` | `translate: (0, 2)` | `translate: (x: 0, y: 2)` | positional `(x, y)` vs named |
| `translate` (single axis) | `translate.y: 2` | `translate: (x: 0, y: 2)` | sub-property sets one axis, other defaults |
| `rotate` | `rotate: 90deg` | `rotate: (angle: 90deg)` | terse is already ideal; verbose exists for consistency/future params |
| `origin` | `origin: center` | `origin: (x: 0.5, y: 0.5)` | named token vs explicit fraction/px |
| `origin` (corner) | `origin: top_left` | `origin: (x: 0, y: 0)` | token expands to a fraction pair |

Rules:

- **Scalar → uniform.** A single `Float` on a two-axis prop (`scale`) fills both
  axes. This is the common case (uniform scale) and should be the shortest to write.
- **Positional tuple → canonical order.** `(a, b)` binds to the prop's canonical
  axes `(x, y)` — same convention already used by the `Len` pair model (RFC-0005),
  so no new rule to learn.
- **Named tuple → any subset, any order.** `(y: 2)` sets `y`, leaves `x` at its
  default; `(y: 2, x: 0)` is order-independent. Names are the prop's axis names
  (`x`/`y` for translate/scale/origin, `angle` for rotate).
- **Sub-property access `prop.axis: value`.** `translate.x: 5` /`scale.y: 1.2` set
  one component inline without a tuple. Resolves by merging over the default; the
  §5 checker rejects an unknown axis (`translate.z`) with a Levenshtein hint (D4).
- **Named tokens are inferred sugar over explicit values.** `origin: center` ≡
  `origin: (x: 0.5, y: 0.5)`; `top_left` ≡ `(0, 0)`; other corners analogous. The
  token set is closed and checked.

The checker enforces one-way-only ambiguity resolution so a value is never
silently misread: a bare scalar is *always* uniform (never "x only"); a positional
tuple is *always* `(x, y)`; to set a single axis you use either the named tuple or
the sub-property form. This keeps the terse form unambiguous while the verbose
form stays maximally clear.

### `frame.rs` — the per-instance transform block (engine change)

Add one packed, POD transform to the primitives that can be transformed. To keep
instance buffers lean, store the **decomposed TRS + origin** (not a full matrix)
and let the shader build the matrix; an untransformed element uses the identity
default and pays nothing extra beyond a branch-predictable check:

```rust
// frame.rs
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Transform {
    pub translate: [f32; 2],
    pub scale:     [f32; 2],
    pub rotate:    f32,       // radians
    pub origin:    [f32; 2],  // resolved absolute pivot in element space
    pub opacity:   f32,
}
impl Transform {
    pub const IDENTITY: Transform = Transform {
        translate: [0.0, 0.0], scale: [1.0, 1.0], rotate: 0.0,
        origin: [0.0, 0.0], opacity: 1.0,
    };
    pub fn is_identity(&self) -> bool { /* cheap compare vs IDENTITY */ }
}
```

`BoxInstance`, `DecoratedBox`, `TextLine`, `VectorInstance` gain a
`transform: Transform` field (default `IDENTITY`). The encoder skips uploading a
transform for identity instances (keeps the common, static case free).

### WGSL — vertex-shader application

The vertex shader builds the affine matrix about the pivot and applies it after
layout positioning, before projection:

```wgsl
fn apply_transform(local: vec2<f32>, t: Transform) -> vec2<f32> {
    let p = local - t.origin;                     // to pivot space
    let s = vec2(p.x * t.scale.x, p.y * t.scale.y);
    let c = cos(t.rotate); let sn = sin(t.rotate);
    let r = vec2(s.x*c - s.y*sn, s.x*sn + s.y*c); // rotate
    return r + t.origin + t.translate;            // back + translate
}
```

`opacity` multiplies the fragment alpha. Because this is per-instance data, an
animated transform (RFC-0010) is just a time-varying `Transform` evaluated in the
shader — the CPU still uploads only endpoints.

### Interaction with the D5 style layers

Transform props flow through the normal three-layer resolution
(`interp/style.rs`): default (identity) → classes → inline. They are **static**
today unless read via inline ternary or animated via `with` (RFC-0010), exactly
like other props — no special casing.

### Composition & nesting

Phase-1 scope: transforms are **element-local** (each instance transforms its own
quad). Nested transforms are **not** composed into a parent matrix stack in this
RFC (a child's transform is relative to its own laid-out rect, not the parent's
transformed space). Full hierarchical transform composition is deferred (Future)
because it interacts with scissor/clip and the flat instance model; element-local
covers the entire micro-interaction use case that motivates this RFC.

## Drawbacks

- Grows four primitive structs by one `Transform` each (mitigated: identity is
  free to upload; only transformed elements pay).
- A new `Angle` scalar + `origin` disambiguation add lexer/checker surface.
- Element-local (non-composing) transforms can surprise devs expecting CSS nested
  transform semantics — must be documented clearly.
- Transform vs hit-rect divergence (clickable area = layout rect, not visual)
  needs a clear mental model.

## Rationale and alternatives

- **Paint-time vs layout-time.** The whole point is to *not* reflow; a layout-time
  transform defeats the purpose and kills animation performance.
- **Decomposed TRS vs baked `[f32;6]` matrix.** TRS is smaller to upload, trivial
  to interpolate per-component in the shader (RFC-0010), and human-readable in a
  debugger. A baked matrix would be cheaper in the shader but can't be
  spring-interpolated component-wise and hides intent.
- **Element-local vs hierarchical composition.** Local ships the 95% use case now
  with zero clip/scissor entanglement; hierarchical is a clean future extension.
- **Not doing it:** no micro-interactions at all, or forcing devs into Rust — both
  unacceptable for the DX bar.

## Prior art

CSS `transform`/`transform-origin` (no-reflow paint transforms), SwiftUI
`.scaleEffect`/`.rotationEffect`/`.offset`, Flutter `Transform` widget, and the
standard game-engine model-matrix-in-vertex-shader approach.

## Resolved decisions (2026-07-01)

- **T1 — `Angle` literal:** **suffix form** `360deg` / `1.5rad` (terse, reads as
  prose, one small lexer addition). Rejected: `deg(360)` call form.
- **T2 — `origin` disambiguation:** **tokens + fractional tuple by default, explicit
  `px` suffix for absolute** (`center` ≡ `(0.5,0.5)`; `(0.5,0.5)` fractional;
  `(10px,10px)` absolute). Removes the range ambiguity.
- **T3 — instance-buffer layout:** **std430** (less padding → smaller buffers → less
  bandwidth); validated by `wgsl_validation.rs`.
- **T4 — container `opacity`:** **per-instance in Phase 1**; correct **group opacity
  deferred** to the hierarchical transform stack (needs render-to-texture). Document
  the difference.

## Resolved questions (formerly unresolved)

- [x] **std430 field ordering/padding for `Transform`:** resolved by implementation (IMPL-82, M33). `Transform` is a plain Rust struct with `translate: [f32; 2]`, `scale: [f32; 2]`, `rotate: f32`, `origin: [f32; 2]`, `opacity: f32` — no explicit `#[repr(C)]` padding needed because the fields are passed to the GPU through `BoxInstance`'s existing per-instance vertex buffer layout (already std430-aligned and validated by `wgsl_validation.rs`). `DecoratedBox` reads transform fields individually from `base.transform` into `DecoratedInstance::from`, not as a raw copy — so field ordering in the Rust struct has no cross-backend consequence. The key invariant: the GPU never sees a `Transform` struct directly; it sees the unpacked fields in the vertex buffer.
- [x] **Group-opacity compositing path:** deferred by design (IMPL-81, T4). Per-instance opacity landed for `BoxInstance` (via `Transform.opacity`) and `DecoratedBox` (via its own `opacity` field, IMPL-82). **Group opacity** (a container's opacity applied to its composited subtree, not per-child) requires render-to-texture and is explicitly deferred to the hierarchical transform stack — a separate, larger RFC. `TextLine` also lacks transform support (IMPL-81) because `glyphon`'s API has no transform matrix per `TextArea`. Both gaps are documented; neither blocks the current feature set.

## Future possibilities

- **Hierarchical transform stack** (parent matrix composition) with clip
  interaction.
- **`hit_follows_transform`** opt-in so a scaled/rotated element hit-tests against
  its visual bounds.
- **3D transforms / perspective** (`rotateX`, `translateZ`) — a much larger,
  separate RFC.
- **Skew** and matrix escape hatch for power users.
