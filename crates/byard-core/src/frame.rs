//! Shared data types for cross-subsystem communication.
//!
//! This module defines [`RenderFrame`] and [`TargetId`], the primitive types
//! that flow between the evaluator, atlas, encoder, and relay subsystems.
//! It is the **only** module that all subsystems may depend on.
//!
//! ```text
//! encoder  ──┐
//! atlas    ──┤─→  frame  ←─  relay
//! evaluator ─┘
//! ```
//!
//! Adding a dependency from one subsystem to another (e.g. `encoder` importing
//! from `evaluator`) is a design defect. If data needs to cross that boundary,
//! it must be modelled as a type in this module.

/// An opaque, copyable identifier for a dirty-flag target.
///
/// Internally packs three fields into a single 64-bit word:
///
/// - bits 0–31  — `index`, the position inside the owning subsystem's table
/// - bits 32–47 — `generation`, a monotonic counter that lets stale IDs be
///   detected when the underlying slot is reused
/// - bits 48–63 — `kind`, a discriminant identifying which subsystem owns
///   the target (atlas node, encoder primitive, …)
///
/// The internal representation is private; consumers must use [`TargetId::new`]
/// to construct an ID and the accessor methods to read its parts.
///
/// Lives in `frame` rather than any subsystem module so all subsystems may
/// reference it without violating the dependency graph in RFC-0001 §9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TargetId(u64);

/// Discriminant identifying which subsystem owns a [`TargetId`].
///
/// Stored in the high 16 bits of every `TargetId` so subsystems can filter
/// the broadcast `mark_dirty_all` calls down to their own targets without
/// coordination.
///
/// `#[repr(u16)]` guarantees the in-memory representation matches the
/// `TargetId` bit layout, so `TargetKind::Foo as u16` is a zero-cost cast.
#[repr(u16)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    /// A layout node owned by `LayoutAtlas`.
    AtlasNode = 1,
    /// A render primitive owned by an Encoder pipeline (`SolidBox`,
    /// `TextGlyph`, …), addressed by its position in the `RenderFrame`.
    EncoderPrimitive = 2,
}

impl TargetId {
    /// Constructs a `TargetId` from its three components.
    ///
    /// The `index`, `generation`, and `kind` are packed into a single
    /// 64-bit word — see the [`TargetId`] type documentation for the
    /// bit layout.
    #[must_use]
    pub const fn new(index: u32, generation: u16, kind: u16) -> Self {
        let raw = (index as u64) | ((generation as u64) << 32) | ((kind as u64) << 48);
        Self(raw)
    }

    /// Returns the index part of the ID.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn index(self) -> u32 {
        // Truncation is intentional: we mask to the low 32 bits.
        (self.0 & 0xFFFF_FFFF) as u32
    }

    /// Returns the generation part of the ID.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn generation(self) -> u16 {
        // Truncation is intentional: we mask to bits 32-47.
        ((self.0 >> 32) & 0xFFFF) as u16
    }

    /// Returns the kind part of the ID.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn kind(self) -> u16 {
        // Truncation is intentional: we mask to the high 16 bits.
        ((self.0 >> 48) & 0xFFFF) as u16
    }

    /// Returns the raw 64-bit representation of the ID.
    #[must_use]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// An axis-aligned rectangle in logical pixel coordinates.
///
/// Produced by the Atlas as the resolved position and size of a node,
/// consumed by the Encoder to issue draw commands. Lives in `frame`
/// because it crosses the subsystem boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rect {
    /// Top-left X coordinate in logical pixels.
    pub x: f32,
    /// Top-left Y coordinate in logical pixels.
    pub y: f32,
    /// Width in logical pixels.
    pub width: f32,
    /// Height in logical pixels.
    pub height: f32,
}

impl Rect {
    /// Constructs a new rectangle.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Returns `true` if the rectangle contains the given point.
    ///
    /// Uses half-open bounds: the left (`x`) and top (`y`) edges are
    /// **inclusive**, while the right (`x + width`) and bottom
    /// (`y + height`) edges are **exclusive**. This matches the convention
    /// used by the spatial hash grid (sub-issue pending) and avoids
    /// off-by-one disagreements during hit-testing.
    #[must_use]
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    /// Returns the smallest rectangle that fully covers both `self` and
    /// `other`.
    ///
    /// Used by the Encoder (RFC-0001 §3.3) to merge several dirty-region
    /// bounding boxes into the single bounding box passed to
    /// `wgpu::RenderPass::set_scissor_rect`. Degenerate (zero-area) rects are
    /// handled the same as any other rect: the union still expands to cover
    /// their `(x, y)` corner.
    #[must_use]
    pub fn union(&self, other: &Rect) -> Rect {
        let min_x = self.x.min(other.x);
        let min_y = self.y.min(other.y);
        let max_x = (self.x + self.width).max(other.x + other.width);
        let max_y = (self.y + self.height).max(other.y + other.height);
        Rect {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }
}

/// A paint-time affine transform (RFC-0011): translate/scale/rotate about a
/// pivot, plus an opacity multiplier. Applied in the vertex/fragment shader
/// *after* Taffy has placed the element — layout geometry and hit-testing
/// rects are never affected, and Taffy is never re-run because a transform
/// changed (INV-8). The identity value is a free no-op in the shader.
///
/// Deliberately a decomposed TRS (not a baked matrix): smaller to upload,
/// trivial to interpolate per-component (RFC-0010's GPU springs animate one
/// field at a time), and legible in a debugger.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Transform {
    /// Logical-pixel offset applied after layout placement; siblings never move.
    pub translate: [f32; 2],
    /// Per-axis scale about `origin`; `[1.0, 1.0]` is unscaled.
    pub scale: [f32; 2],
    /// Rotation about `origin`, in radians.
    pub rotate: f32,
    /// The pivot for `scale`/`rotate`, resolved at lower time into the same
    /// absolute logical-pixel space as the element's laid-out rectangle
    /// (e.g. `center` resolves to the rect's own midpoint).
    pub origin: [f32; 2],
    /// Element alpha multiplier, `0.0..=1.0`.
    pub opacity: f32,
}

impl Transform {
    /// The no-op transform: no offset, unit scale, no rotation, full opacity.
    pub const IDENTITY: Transform = Transform {
        translate: [0.0, 0.0],
        scale: [1.0, 1.0],
        rotate: 0.0,
        origin: [0.0, 0.0],
        opacity: 1.0,
    };

    /// Whether this transform is a no-op (bit-exact match against `IDENTITY`
    /// — every field is set from either a literal default or an exact
    /// user-authored value, never accumulated arithmetic, so exact float
    /// comparison is safe here).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// Maps a point (in the same absolute logical-pixel space as `origin`)
    /// through this transform: scale and rotate it about `origin`, then translate.
    /// Used to propagate an ancestor transform onto a descendant's laid-out
    /// geometry (RFC-0011 group transforms).
    #[must_use]
    pub fn apply_point(&self, p: [f32; 2]) -> [f32; 2] {
        let dx = (p[0] - self.origin[0]) * self.scale[0];
        let dy = (p[1] - self.origin[1]) * self.scale[1];
        let (sin, cos) = self.rotate.sin_cos();
        [
            self.origin[0] + dx * cos - dy * sin + self.translate[0],
            self.origin[1] + dx * sin + dy * cos + self.translate[1],
        ]
    }

    /// Maps a free vector (a displacement) through this transform: scale and
    /// rotate only — no pivot, no translation.
    #[must_use]
    fn apply_vec(&self, v: [f32; 2]) -> [f32; 2] {
        let dx = v[0] * self.scale[0];
        let dy = v[1] * self.scale[1];
        let (sin, cos) = self.rotate.sin_cos();
        [dx * cos - dy * sin, dx * sin + dy * cos]
    }

    /// The mean of the two scale axes — the single factor a scalar-only sink
    /// (text `font_size`, a uniform glyph scale) uses when an ancestor's group
    /// transform is non-uniform. Exact for the common uniform-scale case.
    #[must_use]
    pub fn uniform_scale(&self) -> f32 {
        (self.scale[0] + self.scale[1]) * 0.5
    }

    /// Composes `self` (the outer / ancestor transform) with `inner` (a
    /// descendant's own transform), yielding the single [`Transform`] that maps
    /// the descendant's laid-out geometry as if `inner` were applied first and
    /// `self` after it — the basis of RFC-0011 group transforms (a scaled or
    /// translated container carries its children, text, and widgets with it).
    ///
    /// Exact when the outer transform has no rotation (the common scale/translate
    /// inheritance) or when scales are uniform. A rotating, *non-uniformly* scaled
    /// ancestor is approximated: a general affine no longer decomposes to a single
    /// pivot-based TRS, so full fidelity there would need render-to-texture
    /// (RFC-0011 T4). Opacity multiplies.
    #[must_use]
    pub fn compose(&self, inner: &Transform) -> Transform {
        let moved_origin = self.apply_point(inner.origin);
        let moved_translate = self.apply_vec(inner.translate);
        Transform {
            translate: [
                moved_translate[0] + moved_origin[0] - inner.origin[0],
                moved_translate[1] + moved_origin[1] - inner.origin[1],
            ],
            scale: [
                self.scale[0] * inner.scale[0],
                self.scale[1] * inner.scale[1],
            ],
            rotate: self.rotate + inner.rotate,
            // Keep the descendant's own pivot (in absolute coords); the ancestor's
            // effect is folded into `translate`/`scale`/`rotate` above.
            origin: inner.origin,
            opacity: self.opacity * inner.opacity,
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A packed, POD animation curve (RFC-0010) — a `u32` tag plus three `f32`
/// parameters, so it crosses the frame boundary as plain data and the engine
/// never needs to know the compiler's typed `Curve`. The compiler packs its
/// resolved curve into this at lower time; both the CPU (settling) and the GPU
/// (drawing) read the same closed forms from it.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MotionCurve {
    /// The curve family — one of the `MotionCurve::*` tag constants.
    pub kind: u32,
    /// Curve parameters, interpreted by `kind`:
    /// - linear / ease: `[duration_ms, _, _]`
    /// - spring: `[stiffness, damping, initial_velocity]`
    pub params: [f32; 3],
}

impl MotionCurve {
    /// Fixed-duration linear ramp; `params[0]` is the duration in ms.
    pub const LINEAR: u32 = 0;
    /// Ease-in (cubic); `params[0]` is the duration in ms.
    pub const EASE_IN: u32 = 1;
    /// Ease-out (cubic); `params[0]` is the duration in ms.
    pub const EASE_OUT: u32 = 2;
    /// Ease-in-out (cubic); `params[0]` is the duration in ms.
    pub const EASE_IN_OUT: u32 = 3;
    /// Damped spring; `params` is `[stiffness, damping, initial_velocity]`.
    pub const SPRING: u32 = 4;
}

/// A paint-time animatable scalar (RFC-0010 §"The animatable value model").
///
/// Carries only endpoints and a curve — **no per-frame CPU work**: the CPU
/// rewrites `to` (and reseeds `from`/`start_ms`) once, on a target change, and
/// the shader interpolates every active frame. The CPU also evaluates the same
/// closed forms ([`sample`](Self::sample)/[`velocity`](Self::velocity)) to
/// decide when a motion has [`settled`](Self::is_settled) so the app can stop
/// requesting frames. Times are absolute engine milliseconds.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Motion {
    /// The value at the moment `to` last changed (the interruption point).
    pub from: f32,
    /// The current target (rewritten `O(1)` on mutation).
    pub to: f32,
    /// Absolute engine time (ms) when `to` was set.
    pub start_ms: u32,
    /// The curve driving `from` → `to`.
    pub curve: MotionCurve,
}

impl Motion {
    /// Default settling threshold on position (RFC-0010 A4): below half a
    /// logical pixel a transition is imperceptible. This is a safe scalar
    /// default; per-property callers whose unit isn't pixels (opacity 0..1,
    /// a colour channel, degrees) should pass their own to
    /// [`is_settled_with_eps`](Self::is_settled_with_eps).
    pub const DEFAULT_EPS_POS: f32 = 0.5;
    /// Default settling threshold on velocity, in units per second (RFC-0010 A4).
    pub const DEFAULT_EPS_VEL: f32 = 0.5;

    /// A settled motion pinned at `value` (no movement) with a linear curve.
    #[must_use]
    pub fn resting(value: f32) -> Self {
        Self {
            from: value,
            to: value,
            start_ms: 0,
            curve: MotionCurve {
                kind: MotionCurve::LINEAR,
                params: [0.0, 0.0, 0.0],
            },
        }
    }

    /// The animated value at absolute engine time `now_ms`.
    #[must_use]
    pub fn sample(&self, now_ms: u32) -> f32 {
        let t = seconds_since(self.start_ms, now_ms);
        match self.curve.kind {
            MotionCurve::SPRING => spring_position(self, t),
            MotionCurve::LINEAR => self.from + (self.to - self.from) * duration_progress(self, t),
            _ => {
                let p = ease(self.curve.kind, duration_progress(self, t));
                self.from + (self.to - self.from) * p
            }
        }
    }

    /// The analytic velocity (units per second) at `now_ms`.
    #[must_use]
    pub fn velocity(&self, now_ms: u32) -> f32 {
        let t = seconds_since(self.start_ms, now_ms);
        if self.curve.kind == MotionCurve::SPRING {
            spring_velocity(self, t)
        } else {
            // Finite-difference the eased/linear ramp — cheap and only used for
            // settling, where a derivative-free estimate is plenty.
            const H: f32 = 1.0 / 240.0;
            (self.sample_at(t + H) - self.sample_at(t)) / H
        }
    }

    /// Position at an explicit elapsed time `t` seconds (used by the
    /// finite-difference velocity of the non-spring curves).
    fn sample_at(&self, t: f32) -> f32 {
        match self.curve.kind {
            MotionCurve::SPRING => spring_position(self, t),
            MotionCurve::LINEAR => self.from + (self.to - self.from) * clamp01(progress(self, t)),
            _ => {
                self.from
                    + (self.to - self.from) * ease(self.curve.kind, clamp01(progress(self, t)))
            }
        }
    }

    /// Whether the motion has effectively reached rest, using the default
    /// per-pixel epsilons ([`DEFAULT_EPS_POS`](Self::DEFAULT_EPS_POS) /
    /// [`DEFAULT_EPS_VEL`](Self::DEFAULT_EPS_VEL)).
    #[must_use]
    pub fn is_settled(&self, now_ms: u32) -> bool {
        self.is_settled_with_eps(now_ms, Self::DEFAULT_EPS_POS, Self::DEFAULT_EPS_VEL)
    }

    /// Whether the motion has reached rest under caller-supplied thresholds —
    /// within `eps_pos` of `to` and moving slower than `eps_vel`. The runtime
    /// scales these to the animated property's unit (px vs. opacity vs. colour
    /// channel) so settling is neither too eager nor too lax.
    #[must_use]
    pub fn is_settled_with_eps(&self, now_ms: u32, eps_pos: f32, eps_vel: f32) -> bool {
        (self.sample(now_ms) - self.to).abs() < eps_pos && self.velocity(now_ms).abs() < eps_vel
    }
}

/// Elapsed seconds from `start_ms` to `now_ms`, never negative.
fn seconds_since(start_ms: u32, now_ms: u32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let ms = now_ms.saturating_sub(start_ms) as f32;
    ms / 1000.0
}

/// Clamps `x` to `[0, 1]`.
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Raw (unclamped) progress `t / duration` for a fixed-duration curve; a
/// zero/absent duration is treated as an instant jump (progress ≥ 1).
fn progress(m: &Motion, t_seconds: f32) -> f32 {
    let dur_s = m.curve.params[0] / 1000.0;
    if dur_s <= 0.0 { 1.0 } else { t_seconds / dur_s }
}

/// Clamped progress for a fixed-duration curve.
fn duration_progress(m: &Motion, t_seconds: f32) -> f32 {
    clamp01(progress(m, t_seconds))
}

/// Cubic easing remap of a `0..=1` progress for the ease-* curve kinds.
fn ease(kind: u32, p: f32) -> f32 {
    match kind {
        MotionCurve::EASE_IN => p * p * p,
        MotionCurve::EASE_OUT => {
            let q = 1.0 - p;
            1.0 - q * q * q
        }
        // EASE_IN_OUT and any unknown kind fall back to symmetric cubic.
        _ => {
            if p < 0.5 {
                4.0 * p * p * p
            } else {
                let q = -2.0 * p + 2.0;
                1.0 - (q * q * q) / 2.0
            }
        }
    }
}

/// Damping ratios within this band of `1.0` are treated as critically damped.
///
/// A computed `zeta` never lands exactly on `1.0`, and the under-/over-damped
/// closed forms divide by `wd = ω√(1−ζ²)` / `r1−r2 = 2ω√(ζ²−1)`, both of which
/// vanish as `ζ → 1` and would amplify float error into extreme values. Routing
/// the whole neighbourhood through the (division-free) critical form is exact at
/// `ζ = 1` and an imperceptibly-close approximation across so narrow a band.
const SPRING_CRITICAL_BAND: f32 = 1e-2;

/// Analytic damped-spring position at elapsed time `t` seconds (RFC-0010).
/// `params` = `[stiffness, damping, initial_velocity]`; handles the under-,
/// critically-, and over-damped closed forms with an initial velocity. A
/// negative damping is clamped to zero so a mistuned curve can never turn into
/// unbounded exponential growth.
#[allow(clippy::many_single_char_names)] // standard spring-physics notation
fn spring_position(m: &Motion, t: f32) -> f32 {
    let [k, c, v0] = m.curve.params;
    let c = c.max(0.0);
    let d = m.from - m.to; // displacement from target at t=0
    let omega = k.max(0.0).sqrt();
    if omega == 0.0 {
        return m.to + d; // no restoring force: stays put
    }
    let zeta = c / (2.0 * omega);
    if (zeta - 1.0).abs() < SPRING_CRITICAL_BAND {
        // Critically damped (and the near-critical neighbourhood).
        let e = (-omega * t).exp();
        m.to + e * (d + (v0 + omega * d) * t)
    } else if zeta < 1.0 {
        // Underdamped.
        let wd = omega * (1.0 - zeta * zeta).sqrt();
        let e = (-zeta * omega * t).exp();
        let a = d;
        let b = (v0 + zeta * omega * d) / wd;
        m.to + e * (a * (wd * t).cos() + b * (wd * t).sin())
    } else {
        // Overdamped: two real roots.
        let s = omega * (zeta * zeta - 1.0).sqrt();
        let r1 = -zeta * omega + s;
        let r2 = -zeta * omega - s;
        let a = (v0 - r2 * d) / (r1 - r2);
        let b = d - a;
        m.to + a * (r1 * t).exp() + b * (r2 * t).exp()
    }
}

/// Analytic damped-spring velocity (units/second) at elapsed time `t` — the
/// exact derivative of [`spring_position`], so it is accurate even at `t = 0`
/// where the initial acceleration is large. Each branch satisfies `v(0) = v0`.
#[allow(clippy::many_single_char_names)] // standard spring-physics notation
fn spring_velocity(m: &Motion, t: f32) -> f32 {
    let [k, c, v0] = m.curve.params;
    let c = c.max(0.0);
    let d = m.from - m.to;
    let omega = k.max(0.0).sqrt();
    if omega == 0.0 {
        return 0.0;
    }
    let zeta = c / (2.0 * omega);
    if (zeta - 1.0).abs() < SPRING_CRITICAL_BAND {
        let e = (-omega * t).exp();
        e * (v0 - omega * (v0 + omega * d) * t)
    } else if zeta < 1.0 {
        let wd = omega * (1.0 - zeta * zeta).sqrt();
        let b = (v0 + zeta * omega * d) / wd;
        let e = (-zeta * omega * t).exp();
        e * ((b * wd - zeta * omega * d) * (wd * t).cos()
            - (d * wd + zeta * omega * b) * (wd * t).sin())
    } else {
        let s = omega * (zeta * zeta - 1.0).sqrt();
        let r1 = -zeta * omega + s;
        let r2 = -zeta * omega - s;
        let a = (v0 - r2 * d) / (r1 - r2);
        let b = d - a;
        a * r1 * (r1 * t).exp() + b * r2 * (r2 * t).exp()
    }
}

/// GPU-ready instance data for a single solid rectangle.
///
/// Shared between the logic thread (which populates [`RenderFrame::instances`])
/// and the Encoder (which uploads the slice to the GPU instance buffer). Lives
/// in `frame` rather than `encoder` because it crosses the subsystem boundary
/// between the Logic thread's layout pass and the Encoder's GPU dispatch —
/// see the RFC-0001 §9 dependency graph.
///
/// `#[repr(C)]` and `bytemuck` derives match the layout declared in
/// [`BoxInstance::layout`](crate::encoder::BoxInstance::layout).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoxInstance {
    /// Rectangle in logical pixels: `[x, y, width, height]`.
    pub rect: [f32; 4],
    /// Linear-space fill colour: `[r, g, b, a]`.
    pub color: [f32; 4],
    /// Per-corner border radii: `[top_left, top_right, bottom_right, bottom_left]`.
    pub radii: [f32; 4],
    /// Paint-time transform (RFC-0011); `Transform::IDENTITY` for an
    /// untransformed box.
    pub transform: Transform,
}

/// How an image is scaled/positioned inside its bounding rect.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ImageFit {
    /// Stretch to fill, ignoring aspect ratio.
    #[default]
    Fill,
    /// Scale uniformly to contain inside the rect (letterbox).
    Contain,
    /// Scale uniformly to cover the rect (crop).
    Cover,
    /// No scaling — image at natural size, top-left aligned.
    None,
}

/// A `DecoratedBox` extends a [`BoxInstance`] with an optional border and
/// drop shadow (M21 pipeline). Fields that don't apply are zeroed.
///
/// The Encoder promotes a plain `BoxInstance` to `DecoratedBox` when any of
/// border or shadow fields are non-trivial.
#[derive(Copy, Clone, Debug)]
pub struct DecoratedBox {
    /// The underlying fill/radii data. `base.transform`'s `translate`/`scale`/
    /// `rotate`/`origin` apply as usual; `base.transform.opacity` is **not**
    /// consulted for decorated boxes — [`DecoratedBox::opacity`] (below) is
    /// the one that reaches the shader, since it predates RFC-0011 and
    /// already has an established call-site contract.
    pub base: BoxInstance,
    /// Border width in logical pixels (0.0 = no border).
    pub border_width: f32,
    /// Border colour `[r, g, b, a]`.
    pub border_color: [f32; 4],
    /// Drop-shadow offset X in logical pixels.
    pub shadow_dx: f32,
    /// Drop-shadow offset Y in logical pixels.
    pub shadow_dy: f32,
    /// Drop-shadow blur radius in logical pixels.
    pub shadow_blur: f32,
    /// Drop-shadow spread in logical pixels (RFC-0011): grows (`> 0`) or shrinks
    /// (`< 0`) the shadow shape before blurring, like CSS `box-shadow` spread.
    pub shadow_spread: f32,
    /// Drop-shadow colour `[r, g, b, a]`.
    pub shadow_color: [f32; 4],
    /// Element opacity `0.0–1.0`.
    pub opacity: f32,
    /// Whether this decoration changed since the last tick.
    ///
    /// The encoder's analogue of [`TextLine::dirty`] for the `DecoratedBox`
    /// pipeline (RFC-0001 §3.3): set upstream by the Evaluator → `RenderFrame`
    /// lowering, trusted by the Encoder when computing the incremental scissor
    /// union. A decoration's `base` is a [`BoxInstance`], which is a pure GPU
    /// `Pod` vertex type and therefore cannot itself carry a dirty bit — so the
    /// flag lives here on the (non-`Pod`) wrapper instead.
    pub dirty: bool,
}

impl Default for DecoratedBox {
    fn default() -> Self {
        Self {
            base: BoxInstance {
                rect: [0.0; 4],
                color: [0.0; 4],
                radii: [0.0; 4],
                transform: Transform::IDENTITY,
            },
            border_width: 0.0,
            border_color: [0.0; 4],
            shadow_dx: 0.0,
            shadow_dy: 0.0,
            shadow_blur: 0.0,
            shadow_spread: 0.0,
            shadow_color: [0.0; 4],
            opacity: 1.0,
            dirty: false,
        }
    }
}

/// Shape-kind discriminant for a [`CanvasShape`] (RFC-0020 §2, Tier 1): a
/// circular arc. `params = [cx, cy, r, start_rad, sweep_rad, 0, 0, 0]`.
pub const CANVAS_SHAPE_ARC: u32 = 0;
/// [`CanvasShape`] kind: a full circle. `params = [cx, cy, r, 0, 0, 0, 0, 0]`.
pub const CANVAS_SHAPE_CIRCLE: u32 = 1;
/// [`CanvasShape`] kind: a line segment. `params = [x1, y1, x2, y2, 0, 0, 0, 0]`.
pub const CANVAS_SHAPE_LINE: u32 = 2;
/// [`CanvasShape`] kind: a (rounded) rectangle.
/// `params = [x, y, w, h, radius, 0, 0, 0]`.
pub const CANVAS_SHAPE_RECT: u32 = 3;

/// [`CanvasShape`] line-cap style (RFC-0020 §"Stroke and fill"): flat end
/// exactly at the path endpoint.
pub const CANVAS_CAP_BUTT: u32 = 0;
/// [`CanvasShape`] line-cap style: semicircular end centred on the endpoint.
pub const CANVAS_CAP_ROUND: u32 = 1;
/// [`CanvasShape`] line-cap style: square end extending half the stroke width
/// past the endpoint.
pub const CANVAS_CAP_SQUARE: u32 = 2;

/// A programmatic 2-D shape (RFC-0020): one `Canvas` shape command lowered to
/// a GPU primitive for the `CanvasShape` analytic-SDF pipeline (Tier 1 —
/// arcs, circles, lines, rects). Coordinates in `params` are **absolute
/// logical pixels** (the evaluator has already offset the shape by its
/// `Canvas`'s resolved origin).
///
/// Complex `path(d: …)` commands do not use this type — they rasterize
/// through the `VectorMSDF` pipeline (RFC-0020 §2 Tier 2) as a
/// [`VectorInstance`]; `text(…)` commands lower to [`TextLine`]s.
#[derive(Clone, Debug, PartialEq)]
pub struct CanvasShape {
    /// Shape kind: one of [`CANVAS_SHAPE_ARC`], [`CANVAS_SHAPE_CIRCLE`],
    /// [`CANVAS_SHAPE_LINE`], [`CANVAS_SHAPE_RECT`].
    pub kind: u32,
    /// Shape-specific geometry (see each kind constant's doc for the layout).
    /// Angles are radians; `start` is measured from the positive X axis,
    /// increasing clockwise in screen space (positive Y is down).
    pub params: [f32; 8],
    /// Stroke colour `[r, g, b, a]`; `a == 0` disables the stroke.
    pub stroke_color: [f32; 4],
    /// Fill colour `[r, g, b, a]`; `a == 0` disables the fill. A filled arc
    /// paints the circular sector (pie wedge) swept by `start..start+sweep`.
    pub fill_color: [f32; 4],
    /// Stroke width in logical pixels.
    pub stroke_width: f32,
    /// Line-cap style: [`CANVAS_CAP_BUTT`], [`CANVAS_CAP_ROUND`], or
    /// [`CANVAS_CAP_SQUARE`].
    pub cap: u32,
    /// Dash pattern `(dash_length, gap_length)` in logical pixels along the
    /// path; a non-positive dash length renders a solid stroke.
    pub dash: [f32; 2],
    /// Dash phase offset in logical pixels (animatable — RFC-0020 §"Animation").
    pub dash_offset: f32,
    /// Shape opacity `0.0–1.0` (already multiplied by the canvas/inherited
    /// opacity by the lowering).
    pub opacity: f32,
    /// Paint-time transform (RFC-0011), inherited from the `Canvas` element's
    /// ancestors. `transform.opacity` is **not** consulted — `opacity` above
    /// is authoritative, mirroring [`DecoratedBox`]'s contract.
    pub transform: Transform,
    /// Whether this shape changed since the last tick — the [`CanvasShape`]
    /// analogue of [`TextLine::dirty`] (RFC-0001 §3.3), consumed by the
    /// Encoder's incremental scissor union.
    pub dirty: bool,
}

impl Default for CanvasShape {
    fn default() -> Self {
        Self {
            kind: CANVAS_SHAPE_CIRCLE,
            params: [0.0; 8],
            stroke_color: [0.0; 4],
            fill_color: [0.0; 4],
            stroke_width: 1.0,
            cap: CANVAS_CAP_BUTT,
            dash: [0.0; 2],
            dash_offset: 0.0,
            opacity: 1.0,
            transform: Transform::IDENTITY,
            dirty: false,
        }
    }
}

impl CanvasShape {
    /// Conservative bounding box of this shape in logical pixels, including
    /// the stroke extent and cap overhang. Used by the Encoder both to size
    /// the instance quad and for the incremental dirty-scissor union
    /// (RFC-0001 §3.3). Over-estimating costs a few wasted fragments;
    /// under-estimating would visibly clip the shape, so every branch here
    /// leans generous.
    #[must_use]
    pub fn bounds(&self) -> Rect {
        // Round/square caps overhang endpoints by half the stroke width; the
        // stroke body itself extends half the width to each side. A full
        // stroke width of margin covers both plus the AA fringe.
        let m = self.stroke_width.max(1.0);
        let [p0, p1, p2, p3, ..] = self.params;
        match self.kind {
            CANVAS_SHAPE_LINE => {
                let (x0, x1) = (p0.min(p2), p0.max(p2));
                let (y0, y1) = (p1.min(p3), p1.max(p3));
                Rect::new(x0 - m, y0 - m, (x1 - x0) + m * 2.0, (y1 - y0) + m * 2.0)
            }
            CANVAS_SHAPE_RECT => Rect::new(p0 - m, p1 - m, p2 + m * 2.0, p3 + m * 2.0),
            // Arc and circle: the full circle's box. An arc's true box is a
            // subset, but sweep-dependent tightening is not worth the CPU per
            // frame — the quad is still tiny.
            _ => {
                let r = p2.max(0.0) + m;
                Rect::new(p0 - r, p1 - r, r * 2.0, r * 2.0)
            }
        }
    }
}

/// A texture-sampled rectangle: `Image` intrinsic lowered to a GPU primitive
/// (M21 pipeline). Texture data is identified by a host-opaque `texture_id`
/// (registered outside the engine boundary via the controller boundary, M23).
#[derive(Clone, Debug)]
pub struct TextureSampler {
    /// Rectangle in logical pixels `[x, y, width, height]`.
    pub rect: [f32; 4],
    /// Texture source path or ID (resolved by the controller boundary at M23).
    pub src: String,
    /// How the image is scaled within the rect.
    pub fit: ImageFit,
    /// Per-corner border radii.
    pub radii: [f32; 4],
    /// Opacity `0.0–1.0`.
    pub opacity: f32,
    /// Whether this image primitive changed since the last tick.
    ///
    /// The `TextureSampler` analogue of [`TextLine::dirty`] (RFC-0001 §3.3) —
    /// set upstream by the lowering, trusted by the Encoder's incremental
    /// scissor union. Also set by the Encoder itself the frame after an async
    /// decode completes (M29), so a freshly-loaded image paints without a full
    /// redraw.
    pub dirty: bool,
}

/// GPU-ready instance data for a single MSDF vector glyph (RFC-0009 §1, the
/// fifth pipeline). A `VectorIcon` lowers to one of these; the render thread
/// samples the multi-channel signed-distance-field atlas to draw a crisp,
/// resolution-independent monochrome glyph at any scale.
///
/// `#[repr(C)]` + `bytemuck` so the slice uploads to the instance buffer with
/// zero copy, exactly like [`BoxInstance`]. The shape is identical in dev (JIT
/// atlas) and release (AOT-baked atlas) — INV-7 — so the render path is the same
/// in both modes.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorInstance {
    /// UV rectangle within the MSDF atlas `[u0, v0, u1, v1]` (normalised 0–1).
    pub atlas_uv_rect: [f32; 4],
    /// Taffy-resolved screen rectangle in logical pixels `[x, y, width, height]`.
    pub screen_rect: [f32; 4],
    /// Linear-space tint colour `[r, g, b, a]` (monochrome glyph, RFC-0009).
    pub color: [f32; 4],
    /// Distance range baked at generation time, in atlas texels (§2-E). Drives
    /// the screen-space anti-aliasing in the fragment shader.
    pub px_range: f32,
    /// Array-texture layer holding this glyph's cell.
    pub atlas_layer: u32,
    /// Draw-order depth (NDC-z, RFC-0011 cross-pass paint order — see
    /// [`draw_depth`]). Set by [`RenderFrame::push_vector`], not by
    /// [`VectorInstance::new`] (which defaults it to the far plane); a bare
    /// `VectorInstance` built outside a `RenderFrame` therefore always loses
    /// the depth test to one that went through `push_vector`.
    pub depth: f32,
}

impl VectorInstance {
    /// Builds an instance from logical-pixel screen geometry and an atlas UV
    /// rect (both as [`Rect`]), a colour, the baked `px_range`, and the atlas
    /// layer. `depth` defaults to the far plane; [`RenderFrame::push_vector`]
    /// stamps the real draw-order value on push.
    #[must_use]
    pub fn new(screen: Rect, atlas_uv: Rect, color: [f32; 4], px_range: f32, layer: u32) -> Self {
        Self {
            atlas_uv_rect: [
                atlas_uv.x,
                atlas_uv.y,
                atlas_uv.x + atlas_uv.width,
                atlas_uv.y + atlas_uv.height,
            ],
            screen_rect: [screen.x, screen.y, screen.width, screen.height],
            color,
            px_range,
            atlas_layer: layer,
            depth: DRAW_DEPTH_CLEAR,
        }
    }
}

/// An owned MSDF-field upload destined for the vector atlas (RFC-0009 §2-C /
/// INV-8). A background worker generates the field and sends this record over a
/// channel; the **logic thread** allocates the UV slot and records the upload on
/// the next [`RenderFrame`]; only the **render thread** performs the actual
/// `Queue::write_texture` during frame application. Workers never touch the GPU
/// queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtlasUpload {
    /// Destination array-texture layer.
    pub layer: u32,
    /// Destination pixel x within the layer.
    pub x: u32,
    /// Destination pixel y within the layer.
    pub y: u32,
    /// Field cell width in pixels.
    pub width: u32,
    /// Field cell height in pixels.
    pub height: u32,
    /// The RGBA8 MSDF field bytes (`width * height * 4`), owned (INV-3).
    pub bytes: Vec<u8>,
    /// Caller-assigned identity for this upload (monotonic per-generator),
    /// echoed back through an acknowledgment channel once the render thread
    /// actually applies it — lets the caller resend an upload indefinitely
    /// until confirmed, rather than guessing a fixed retry window.
    pub id: u64,
}

/// A single line of text to be rendered in a frame.
///
/// Shared between the logic thread (which populates [`RenderFrame::texts`]) and
/// the Encoder's `TextGlyphPipeline`. Lives in `frame` rather than
/// `encoder::text_glyph` because it crosses the subsystem boundary between the
/// Evaluator/Atlas and the Encoder — see RFC-0001 §9.
///
/// All coordinates are in **logical pixels**, consistent with [`BoxInstance`].
#[derive(Debug, Clone)]
pub struct TextLine {
    /// X position of the text baseline in logical pixels.
    pub x: f32,
    /// Y position of the text baseline in logical pixels.
    pub y: f32,
    /// Text content.
    pub text: String,
    /// Font size in logical pixels.
    pub font_size: f32,
    /// Text colour: `[r, g, b, a]` in linear space, each component 0–1.
    pub color: [f32; 4],
    /// Whether this line's content changed since the last tick.
    ///
    /// Set upstream by the Evaluator → Atlas → `RenderFrame` pipeline — never
    /// derived locally by the Encoder. The Encoder trusts this bit completely
    /// in `--release` builds; see `encoder::text_glyph`'s module documentation.
    pub dirty: bool,
}

/// Logical-pixel dimensions of the surface that hosts a layout.
///
/// Passed to [`LayoutAtlas::compute`](crate::atlas::LayoutAtlas::compute) as
/// the available space for the root node.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Viewport {
    /// Width of the host surface in logical pixels.
    pub width: f32,
    /// Height of the host surface in logical pixels.
    pub height: f32,
}

impl Viewport {
    /// Constructs a new viewport.
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

/// A snapshot of all render primitives for a single frame.
///
/// Built by the Logic thread (Evaluator + Atlas) and read by the Render
/// thread (Encoder). The Logic thread mutates the frame during construction
/// via crate-private APIs; once handed off to the Render thread (via the
/// Relay's atomic pointer swap) it is treated as immutable for the duration
/// of that frame.
///
/// The structure is intentionally SoA-friendly for batched GPU dispatch: each
/// primitive type lives in its own `Vec` so the Encoder can cast a slice
/// directly to bytes and upload it with zero copy.
///
/// [`version`](RenderFrame::version) is a monotonic counter incremented by the
/// Logic thread whenever any content changes. The Encoder compares it against
/// the version it saw on the previous frame to detect missed-dirty-frame
/// scenarios (see `EncoderSubsystem::encode_frame_from_relay`).
#[derive(Debug, Default)]
pub struct RenderFrame {
    /// Resolved geometry produced by the Atlas.
    ///
    /// Each entry is a rectangle in logical pixels, ready for the Encoder
    /// to translate into a draw command.
    rects: Vec<Rect>,

    /// Per-entry dirty state, parallel to `rects`.
    ///
    /// `dirty[i]` is `true` when `rects[i]` changed since the previous tick.
    dirty: Vec<bool>,

    /// Solid-rectangle instances populated by the Logic thread each tick.
    instances: Vec<BoxInstance>,

    /// Decorated-box instances (M21) — boxes with border/shadow/opacity.
    decorated: Vec<DecoratedBox>,

    /// Texture-sampled image instances (M21).
    textures: Vec<TextureSampler>,

    /// Text lines populated by the Logic thread each tick.
    texts: Vec<TextLine>,

    /// MSDF vector-glyph instances (RFC-0009 §1, the fifth pipeline).
    vector_instances: Vec<VectorInstance>,

    /// Programmatic 2-D shapes (RFC-0020): `Canvas` shape commands lowered to
    /// the `CanvasShape` analytic-SDF pipeline (the sixth pipeline).
    canvas_shapes: Vec<CanvasShape>,

    /// Pending MSDF-atlas uploads recorded by the logic thread this tick
    /// (RFC-0009 §2-C / INV-8). Applied by the render thread via a single
    /// `Queue::write_texture` each, during frame application, before the draw.
    atlas_uploads: Vec<AtlasUpload>,

    /// Per-primitive **draw-order depth**, one parallel vec per drawable pool.
    ///
    /// The Encoder draws in four type-grouped passes (solids → decorated →
    /// textures → text), which alone can never honour paint order *across*
    /// passes — a container's border (decorated) would always sit above its
    /// children (solids), and text above everything. To fix that coherently we
    /// stamp every primitive, in global emission order, with a monotonically
    /// *nearer* NDC-z (see [`draw_depth`]) and let a shared depth buffer
    /// (cleared to the far plane every frame, `LessEqual` test) resolve
    /// visibility. Emission order is tree pre-order = the intended painter's
    /// order, so a later-emitted primitive correctly wins.
    ///
    /// Kept as parallel `f32` vecs (not fields on the primitives) so the `Pod`
    /// instance structs and their vertex layouts stay byte-for-byte unchanged.
    solid_depths: Vec<f32>,
    decorated_depths: Vec<f32>,
    texture_depths: Vec<f32>,
    text_depths: Vec<f32>,
    canvas_depths: Vec<f32>,

    /// Content-clip rectangles (RFC-0005 `ScrollView`, §3.3 scissor). A
    /// [`ScrollView`] wraps its children in [`begin_clip`](Self::begin_clip) /
    /// [`end_clip`](Self::end_clip); every drawable emitted inside carries the
    /// index of the active clip in the parallel `*_clips` slices below, and the
    /// Encoder sets the GPU scissor to that rect (intersected with the dirty
    /// region) while drawing it. Distinct from the *dirty-region* scissor, which
    /// is a per-frame redraw optimisation — this is a semantic content clip.
    clips: Vec<ClipRect>,
    /// Stack of active clip indices during emission (not serialized). The top is
    /// stamped onto each `push_*`; nested clips store their **intersection** with
    /// the parent, so the Encoder only ever sets one rect. Empty after a frame.
    clip_stack: Vec<u16>,

    /// Per-primitive clip index, parallel to each drawable pool (like the
    /// `*_depths` vecs — kept off the `Pod` instance types so vertex layouts stay
    /// byte-for-byte unchanged). `None` = unclipped (the whole viewport).
    solid_clips: Vec<Option<u16>>,
    decorated_clips: Vec<Option<u16>>,
    texture_clips: Vec<Option<u16>>,
    text_clips: Vec<Option<u16>>,
    vector_clips: Vec<Option<u16>>,
    canvas_clips: Vec<Option<u16>>,

    /// Per-text-line wrap width in logical pixels, parallel to `texts` (kept off
    /// the `TextLine` struct like the depth/clip vecs, so its 19 construction
    /// sites stay untouched). `Some(w)` shapes that line bounded to `w` so it
    /// wraps onto multiple lines; `None` is the natural single-line width.
    /// Populated by [`push_text_wrapped`](Self::push_text_wrapped); the plain
    /// [`push_text`](Self::push_text) records `None`.
    text_wrap: Vec<Option<f32>>,

    /// Running global emission counter, mapped to a depth by [`draw_depth`].
    /// Reset each [`clear`](Self::clear); advanced by every `push_*` drawable.
    draw_seq: u32,

    /// Z-layer boundaries recorded by [`begin_layer`](Self::begin_layer)
    /// (RFC-0017 layered draw batches). Empty for the overwhelmingly common
    /// single-layer frame — the Encoder then draws the exact pre-layering
    /// stream. See [`LayerMark`] for the full model.
    layer_marks: Vec<LayerMark>,

    /// Monotonic version counter, incremented by the Logic thread whenever any
    /// content in this frame changes relative to the previous tick.
    ///
    /// The Encoder compares this against the last version it rendered. A version
    /// advance means the render thread skipped at least one dirty frame and must
    /// force a full redraw + text reshape to avoid displaying stale glyphs.
    version: u64,

    /// This tick's CPU scope samples (RFC-0013 "Hand-off"), piggybacked on
    /// the existing atomic frame swap instead of a dedicated channel. Empty
    /// when the `telemetry` feature is off or nothing was profiled this tick.
    telemetry: crate::telemetry::SampleBlock,
}

/// A content-clip rectangle (RFC-0005 `ScrollView`). `rect` is in logical
/// pixels, like every other [`RenderFrame`] geometry; the Encoder scales it to
/// physical pixels and intersects it with the dirty-region scissor before the
/// draw. A [`ScrollView`]'s clip is the intersection of its own viewport with
/// any ancestor scroll viewport, so nested scrolling needs no per-draw math.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClipRect {
    /// The clip region in logical pixels.
    pub rect: Rect,
}

/// Axis-aligned intersection of two logical-pixel rects (empty if disjoint).
#[must_use]
fn intersect_rect(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.width).min(b.x + b.width);
    let y1 = (a.y + a.height).min(b.y + b.height);
    Rect::new(x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0))
}

/// A z-layer boundary: the length of every drawable pool at the instant
/// [`RenderFrame::begin_layer`] was called (RFC-0017 layered draw batches).
///
/// The Encoder turns consecutive marks into per-pool index ranges and draws
/// each layer's primitives — solids, decorated, textures, vectors, **and
/// text** — as one interleaved group inside the single UI render pass, so a
/// later layer's *transparent* geometry (a modal scrim, a dialog shadow)
/// alpha-blends **over** an earlier layer's text and images instead of being
/// painted before them. Within a layer, the shared draw-order depth buffer
/// keeps resolving paint order exactly as before; across layers, draw order
/// itself is now correct for blending. A frame with no marks is one layer —
/// the exact pre-layering draw stream, byte for byte.
///
/// Kept as pool *cursors* (not per-primitive tags) because emission is
/// strictly sequential — the main tree first, then each overlay — so a layer
/// is always a contiguous range of every pool. Five `u32`s per layer instead
/// of one tag per primitive.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct LayerMark {
    /// `instances` (solid boxes) length at the boundary.
    pub solid: u32,
    /// `decorated` length at the boundary.
    pub decorated: u32,
    /// `textures` length at the boundary.
    pub texture: u32,
    /// `vector_instances` length at the boundary.
    pub vector: u32,
    /// `texts` length at the boundary.
    pub text: u32,
    /// `canvas_shapes` length at the boundary (RFC-0020).
    pub canvas: u32,
}

/// NDC far-plane depth the shared draw-order depth buffer is cleared to at the
/// start of every frame. Every drawable's [`draw_depth`] is strictly nearer, so
/// it passes the `LessEqual` test against this cleared value.
pub const DRAW_DEPTH_CLEAR: f32 = 1.0;

/// NDC-z granted per emitted primitive. `1/65536` spaces ~65k primitives across
/// the usable near-1.0 range while staying far above f32 depth resolution
/// (~6e-8 near 1.0), so adjacent primitives never z-fight.
const DRAW_DEPTH_STEP: f32 = 1.0 / 65_536.0;

/// Maps a global emission sequence number to a draw-order NDC-z: earlier =
/// farther (toward `1.0`), later = nearer (toward `0.0`). Saturating, so a
/// pathologically deep frame clamps to the near plane rather than wrapping.
#[must_use]
pub fn draw_depth(seq: u32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let steps = seq.saturating_add(1) as f32 * DRAW_DEPTH_STEP;
    (DRAW_DEPTH_CLEAR - steps).max(0.0)
}

impl RenderFrame {
    /// Creates an empty frame.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears the frame, retaining internal buffer capacity.
    ///
    /// After the first frame, subsequent populations pay zero allocation cost
    /// as long as primitive counts stay within the high-water mark. Version is
    /// reset to zero; the Logic thread always calls [`set_version`](Self::set_version)
    /// immediately after acquiring a recycled frame.
    pub fn clear(&mut self) {
        self.rects.clear();
        self.dirty.clear();
        self.instances.clear();
        self.decorated.clear();
        self.textures.clear();
        self.texts.clear();
        self.vector_instances.clear();
        self.canvas_shapes.clear();
        self.atlas_uploads.clear();
        self.solid_depths.clear();
        self.decorated_depths.clear();
        self.texture_depths.clear();
        self.text_depths.clear();
        self.canvas_depths.clear();
        self.clips.clear();
        self.clip_stack.clear();
        self.solid_clips.clear();
        self.decorated_clips.clear();
        self.texture_clips.clear();
        self.text_clips.clear();
        self.vector_clips.clear();
        self.canvas_clips.clear();
        self.text_wrap.clear();
        self.layer_marks.clear();
        self.draw_seq = 0;
        self.version = 0;
        // `Vec::clear` only, not `SampleBlock::default()` — the latter would
        // drop the block's existing allocation and defeat the capacity
        // retention this method promises once telemetry is attached.
        self.telemetry.samples.clear();
        self.telemetry.dropped = 0;
    }

    /// Appends a resolved rectangle and its dirty state to the frame.
    pub fn push_rect(&mut self, rect: Rect, dirty: bool) {
        self.rects.push(rect);
        self.dirty.push(dirty);
    }

    /// Advances the global emission counter and returns the draw-order depth
    /// (NDC-z) for the primitive about to be pushed. See [`solid_depths`] for
    /// the ordering model.
    ///
    /// [`solid_depths`]: Self::solid_depths
    fn next_depth(&mut self) -> f32 {
        let d = draw_depth(self.draw_seq);
        self.draw_seq = self.draw_seq.saturating_add(1);
        d
    }

    /// The clip index stamped onto the next `push_*` (the top of the clip
    /// stack), or `None` when nothing is being clipped.
    fn active_clip(&self) -> Option<u16> {
        self.clip_stack.last().copied()
    }

    /// Opens a content clip (RFC-0005 `ScrollView`): every drawable pushed until
    /// the matching [`end_clip`](Self::end_clip) is clipped to `rect`. A nested
    /// clip is stored as its **intersection** with the enclosing clip, so the
    /// Encoder only ever sets one scissor rect. Returns the clip's index.
    pub fn begin_clip(&mut self, rect: Rect) -> u16 {
        let clipped = match self.clip_stack.last() {
            Some(&parent) => intersect_rect(self.clips[parent as usize].rect, rect),
            None => rect,
        };
        let id = u16::try_from(self.clips.len()).unwrap_or(u16::MAX);
        self.clips.push(ClipRect { rect: clipped });
        self.clip_stack.push(id);
        id
    }

    /// Closes the most recently opened content clip.
    pub fn end_clip(&mut self) {
        self.clip_stack.pop();
    }

    /// Appends a [`BoxInstance`] to the frame.
    pub fn push_instance(&mut self, instance: BoxInstance) {
        let d = self.next_depth();
        let c = self.active_clip();
        self.instances.push(instance);
        self.solid_depths.push(d);
        self.solid_clips.push(c);
    }

    /// Appends a [`DecoratedBox`] (border/shadow/opacity) to the frame (M21).
    pub fn push_decorated(&mut self, d: DecoratedBox) {
        let depth = self.next_depth();
        let c = self.active_clip();
        self.decorated.push(d);
        self.decorated_depths.push(depth);
        self.decorated_clips.push(c);
    }

    /// Appends a [`TextureSampler`] (image) to the frame (M21).
    pub fn push_texture(&mut self, t: TextureSampler) {
        let d = self.next_depth();
        let c = self.active_clip();
        self.textures.push(t);
        self.texture_depths.push(d);
        self.texture_clips.push(c);
    }

    /// Appends a [`TextLine`] to the frame (natural single-line width).
    pub fn push_text(&mut self, text: TextLine) {
        self.push_text_wrapped(text, None);
    }

    /// Appends a [`TextLine`] shaped to an optional wrap width (RFC-0018 text
    /// wrap). `Some(w)` wraps the line onto multiple lines bounded to `w`
    /// logical pixels; `None` behaves exactly like [`push_text`](Self::push_text).
    pub fn push_text_wrapped(&mut self, text: TextLine, wrap_width: Option<f32>) {
        let d = self.next_depth();
        let c = self.active_clip();
        self.texts.push(text);
        self.text_depths.push(d);
        self.text_clips.push(c);
        self.text_wrap.push(wrap_width);
    }

    /// Appends a [`VectorInstance`] (MSDF glyph) to the frame (RFC-0009 §1).
    pub fn push_vector(&mut self, mut v: VectorInstance) {
        v.depth = self.next_depth();
        let c = self.active_clip();
        self.vector_instances.push(v);
        self.vector_clips.push(c);
    }

    /// Appends a [`CanvasShape`] (RFC-0020 Tier-1 shape command) to the frame.
    pub fn push_canvas_shape(&mut self, s: CanvasShape) {
        let d = self.next_depth();
        let c = self.active_clip();
        self.canvas_shapes.push(s);
        self.canvas_depths.push(d);
        self.canvas_clips.push(c);
    }

    /// Records a pending [`AtlasUpload`] for the render thread to apply before
    /// drawing this frame (RFC-0009 §2-C / INV-8).
    pub fn push_atlas_upload(&mut self, upload: AtlasUpload) {
        self.atlas_uploads.push(upload);
    }

    /// Opens a new z-layer (RFC-0017): everything pushed from here on is drawn
    /// — solids, decorated, textures, vectors, *and text*, interleaved — after
    /// **everything** already in the frame, inside the same GPU render pass.
    ///
    /// Called by the overlay phase before emitting each overlay, so a modal
    /// scrim genuinely alpha-blends over the main tree's text and images
    /// instead of being drawn before the frame-final text batch. Consecutive
    /// calls with no primitives in between are deduplicated, so an overlay
    /// that emits nothing costs nothing. A frame that never calls this is a
    /// single layer and renders through the exact pre-layering draw stream.
    pub fn begin_layer(&mut self) {
        let mark = LayerMark {
            solid: u32::try_from(self.instances.len()).unwrap_or(u32::MAX),
            decorated: u32::try_from(self.decorated.len()).unwrap_or(u32::MAX),
            texture: u32::try_from(self.textures.len()).unwrap_or(u32::MAX),
            vector: u32::try_from(self.vector_instances.len()).unwrap_or(u32::MAX),
            text: u32::try_from(self.texts.len()).unwrap_or(u32::MAX),
            canvas: u32::try_from(self.canvas_shapes.len()).unwrap_or(u32::MAX),
        };
        if self.layer_marks.last() == Some(&mark) {
            return; // empty layer — dedup, an overlay that emitted nothing is free
        }
        self.layer_marks.push(mark);
    }

    /// The z-layer boundaries recorded this frame (RFC-0017 layered draw
    /// batches); empty for a single-layer frame. See [`LayerMark`].
    #[must_use]
    pub fn layer_marks(&self) -> &[LayerMark] {
        &self.layer_marks
    }

    /// Sets the frame's version counter.
    pub fn set_version(&mut self, version: u64) {
        self.version = version;
    }

    /// Returns the resolved rectangles in this frame.
    #[must_use]
    pub fn rects(&self) -> &[Rect] {
        &self.rects
    }

    /// Returns the per-entry dirty state, parallel to [`rects`](Self::rects).
    #[must_use]
    pub fn dirty(&self) -> &[bool] {
        &self.dirty
    }

    /// Returns the solid-rectangle instances in this frame.
    #[must_use]
    pub fn instances(&self) -> &[BoxInstance] {
        &self.instances
    }

    /// Returns the decorated-box instances in this frame (M21).
    #[must_use]
    pub fn decorated(&self) -> &[DecoratedBox] {
        &self.decorated
    }

    /// Returns the texture-sampled image instances in this frame (M21).
    #[must_use]
    pub fn textures(&self) -> &[TextureSampler] {
        &self.textures
    }

    /// Returns the per-line wrap widths, parallel to [`texts`](Self::texts).
    #[must_use]
    pub fn text_wraps(&self) -> &[Option<f32>] {
        &self.text_wrap
    }

    /// Returns the text lines pushed to the frame.
    #[must_use]
    pub fn texts(&self) -> &[TextLine] {
        &self.texts
    }

    /// Returns the MSDF vector-glyph instances in this frame (RFC-0009 §1).
    #[must_use]
    pub fn vector_instances(&self) -> &[VectorInstance] {
        &self.vector_instances
    }

    /// Returns the `Canvas` shape primitives in this frame (RFC-0020).
    #[must_use]
    pub fn canvas_shapes(&self) -> &[CanvasShape] {
        &self.canvas_shapes
    }

    /// Returns the pending atlas uploads recorded this frame (RFC-0009 §2-C).
    #[must_use]
    pub fn atlas_uploads(&self) -> &[AtlasUpload] {
        &self.atlas_uploads
    }

    /// Draw-order depths parallel to [`instances`](Self::instances).
    #[must_use]
    pub fn solid_depths(&self) -> &[f32] {
        &self.solid_depths
    }

    /// Draw-order depths parallel to [`decorated`](Self::decorated).
    #[must_use]
    pub fn decorated_depths(&self) -> &[f32] {
        &self.decorated_depths
    }

    /// Draw-order depths parallel to [`textures`](Self::textures).
    #[must_use]
    pub fn texture_depths(&self) -> &[f32] {
        &self.texture_depths
    }

    /// Draw-order depths parallel to [`texts`](Self::texts).
    #[must_use]
    pub fn text_depths(&self) -> &[f32] {
        &self.text_depths
    }

    /// Draw-order depths parallel to [`canvas_shapes`](Self::canvas_shapes).
    #[must_use]
    pub fn canvas_depths(&self) -> &[f32] {
        &self.canvas_depths
    }

    /// The content-clip table (RFC-0005 `ScrollView`); a primitive's
    /// `*_clips` entry indexes into this.
    #[must_use]
    pub fn clips(&self) -> &[ClipRect] {
        &self.clips
    }
    /// Per-`BoxInstance` clip index (parallel to [`instances`](Self::instances)).
    #[must_use]
    pub fn solid_clips(&self) -> &[Option<u16>] {
        &self.solid_clips
    }
    /// Per-`DecoratedBox` clip index (parallel to [`decorated`](Self::decorated)).
    #[must_use]
    pub fn decorated_clips(&self) -> &[Option<u16>] {
        &self.decorated_clips
    }
    /// Per-`TextureSampler` clip index (parallel to [`textures`](Self::textures)).
    #[must_use]
    pub fn texture_clips(&self) -> &[Option<u16>] {
        &self.texture_clips
    }
    /// Per-`TextLine` clip index (parallel to [`texts`](Self::texts)).
    #[must_use]
    pub fn text_clips(&self) -> &[Option<u16>] {
        &self.text_clips
    }
    /// Per-`VectorInstance` clip index (parallel to [`vector_instances`](Self::vector_instances)).
    #[must_use]
    pub fn vector_clips(&self) -> &[Option<u16>] {
        &self.vector_clips
    }
    /// Per-`CanvasShape` clip index (parallel to [`canvas_shapes`](Self::canvas_shapes)).
    #[must_use]
    pub fn canvas_clips(&self) -> &[Option<u16>] {
        &self.canvas_clips
    }

    /// Returns the monotonic version counter for this frame.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Pulls the calling thread's CPU scope samples into this frame
    /// (RFC-0013 "Hand-off"), piggybacked on this frame's atomic swap instead
    /// of a dedicated channel.
    ///
    /// Called from the logic thread, once per tick, by
    /// [`crate::relay::Relay::publish`] right before the frame is swapped
    /// in — so every publish path picks up telemetry automatically, with no
    /// per-call-site wiring needed. Reuses this frame's existing
    /// `SampleBlock` allocation (see [`RenderFrame::clear`]) rather than
    /// allocating a fresh one each tick.
    pub fn drain_telemetry(&mut self) {
        crate::telemetry::drain_samples_into(&mut self.telemetry);
    }

    /// Returns this tick's CPU scope samples, if any were captured.
    #[must_use]
    pub fn telemetry(&self) -> &crate::telemetry::SampleBlock {
        &self.telemetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether two 2-vectors agree within a tight float tolerance.
    fn approx(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4
    }

    #[test]
    fn compose_with_identity_is_a_no_op() {
        let t = Transform {
            translate: [3.0, -4.0],
            scale: [1.5, 2.0],
            rotate: 0.3,
            origin: [10.0, 20.0],
            opacity: 0.8,
        };
        // An identity *outer* preserves the inner transform field-for-field.
        assert_eq!(Transform::IDENTITY.compose(&t), t);
        // An identity *inner* re-anchors the pivot to (0,0) but stays the same
        // mapping — check a couple of sample points rather than the fields.
        let reanchored = t.compose(&Transform::IDENTITY);
        for p in [[0.0, 0.0], [12.0, -7.0], [30.0, 40.0]] {
            let a = t.apply_point(p);
            let b = reanchored.apply_point(p);
            assert!(
                (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4,
                "identity-inner compose maps {p:?} the same: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn parent_scale_carries_a_child_point_and_scale() {
        // A parent scaling 2× about the origin (0,0) should map a child point at
        // (10, 5) to (20, 10) and double the child's scale — the group transform.
        let parent = Transform {
            scale: [2.0, 2.0],
            origin: [0.0, 0.0],
            ..Transform::IDENTITY
        };
        // Child is an unscaled box pivoting at its own centre (10, 5).
        let child = Transform {
            origin: [10.0, 5.0],
            ..Transform::IDENTITY
        };
        let composed = parent.compose(&child);
        assert!(approx(composed.scale, [2.0, 2.0]), "scale multiplies");
        // The composed transform, applied to the child pivot, lands where the
        // parent would have put it: 2×(10,5) = (20,10).
        assert!(approx(composed.apply_point([10.0, 5.0]), [20.0, 10.0]));
    }

    #[test]
    fn parent_translate_offsets_children() {
        let parent = Transform {
            translate: [7.0, -3.0],
            ..Transform::IDENTITY
        };
        let child = Transform {
            origin: [4.0, 4.0],
            ..Transform::IDENTITY
        };
        let composed = parent.compose(&child);
        // A pure translate just shifts the child's mapped geometry.
        assert!(approx(composed.apply_point([4.0, 4.0]), [11.0, 1.0]));
        assert!(approx(composed.scale, [1.0, 1.0]));
    }

    #[test]
    fn round_trip_preserves_all_fields() {
        let id = TargetId::new(0x1234_5678, 0xABCD, 0x9F00);
        assert_eq!(id.index(), 0x1234_5678);
        assert_eq!(id.generation(), 0xABCD);
        assert_eq!(id.kind(), 0x9F00);
    }

    #[test]
    fn maximum_values_do_not_overflow_neighbouring_fields() {
        let id = TargetId::new(u32::MAX, u16::MAX, u16::MAX);
        assert_eq!(id.index(), u32::MAX);
        assert_eq!(id.generation(), u16::MAX);
        assert_eq!(id.kind(), u16::MAX);
    }

    #[test]
    fn zero_id_has_all_zero_fields() {
        let id = TargetId::new(0, 0, 0);
        assert_eq!(id.as_raw(), 0);
        assert_eq!(id.index(), 0);
        assert_eq!(id.generation(), 0);
        assert_eq!(id.kind(), 0);
    }

    #[test]
    fn vector_instance_is_pod_and_round_trips_through_the_frame() {
        let v = VectorInstance::new(
            Rect::new(10.0, 20.0, 16.0, 16.0),
            Rect::new(0.0, 0.0, 0.25, 0.25),
            [1.0, 1.0, 1.0, 1.0],
            4.0,
            0,
        );
        // UV rect is stored as [u0, v0, u1, v1].
        let close = |a: [f32; 4], b: [f32; 4]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        assert!(close(v.atlas_uv_rect, [0.0, 0.0, 0.25, 0.25]));
        assert!(close(v.screen_rect, [10.0, 20.0, 16.0, 16.0]));
        // Pod: 15 × 4-byte fields, no padding (so it uploads zero-copy).
        assert_eq!(std::mem::size_of::<VectorInstance>(), 60);
        let _bytes: &[u8] = bytemuck::bytes_of(&v);

        let mut frame = RenderFrame::new();
        frame.push_vector(v);
        frame.push_atlas_upload(AtlasUpload {
            layer: 0,
            x: 0,
            y: 0,
            width: 2,
            height: 2,
            id: 0,
            bytes: vec![0u8; 2 * 2 * 4],
        });
        assert_eq!(frame.vector_instances().len(), 1);
        assert_eq!(frame.atlas_uploads().len(), 1);
        frame.clear();
        assert!(frame.vector_instances().is_empty());
        assert!(frame.atlas_uploads().is_empty());
    }

    #[test]
    fn is_copy_and_cheap_to_clone() {
        const fn assert_copy<T: Copy>() {}
        assert_copy::<TargetId>();
        assert_eq!(std::mem::size_of::<TargetId>(), 8);
    }

    #[test]
    fn rect_contains_point_inside() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(r.contains(50.0, 30.0));
    }

    #[test]
    fn rect_does_not_contain_point_on_right_edge() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(!r.contains(110.0, 30.0), "right edge is exclusive");
    }

    #[test]
    fn rect_does_not_contain_point_outside() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(!r.contains(0.0, 0.0));
    }

    #[test]
    fn rect_union_of_disjoint_rects_covers_both() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(100.0, 200.0, 10.0, 10.0);
        let u = a.union(&b);
        assert_eq!(u, Rect::new(0.0, 0.0, 110.0, 210.0));
    }

    #[test]
    fn rect_union_with_self_is_identity() {
        let a = Rect::new(5.0, 5.0, 20.0, 30.0);
        assert_eq!(a.union(&a), a);
    }

    #[test]
    fn rect_union_is_commutative() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(-5.0, 3.0, 4.0, 50.0);
        assert_eq!(a.union(&b), b.union(&a));
    }

    #[test]
    fn rect_union_where_one_fully_contains_the_other_returns_the_larger() {
        let outer = Rect::new(0.0, 0.0, 100.0, 100.0);
        let inner = Rect::new(10.0, 10.0, 5.0, 5.0);
        assert_eq!(outer.union(&inner), outer);
        assert_eq!(inner.union(&outer), outer);
    }

    #[test]
    fn rect_union_of_overlapping_rects_merges_correctly() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 10.0, 10.0);
        assert_eq!(a.union(&b), Rect::new(0.0, 0.0, 15.0, 15.0));
    }

    #[test]
    fn rect_union_of_zero_area_rects_covers_both_corners() {
        // A degenerate (zero-size) rect can still arise from a TextLine whose
        // heuristic bounds collapse to a point; union must not panic or
        // silently drop it.
        let a = Rect::new(0.0, 0.0, 0.0, 0.0);
        let b = Rect::new(50.0, 50.0, 0.0, 0.0);
        assert_eq!(a.union(&b), Rect::new(0.0, 0.0, 50.0, 50.0));
    }

    #[test]
    fn render_frame_starts_empty() {
        let frame = RenderFrame::new();
        assert!(frame.rects().is_empty());
    }

    #[test]
    fn render_frame_clear_empties_rects() {
        let mut frame = RenderFrame::new();
        frame.push_rect(Rect::new(0.0, 0.0, 10.0, 10.0), false);
        frame.push_rect(Rect::new(10.0, 0.0, 10.0, 10.0), true);
        assert_eq!(frame.rects().len(), 2);

        frame.clear();
        assert!(frame.rects().is_empty());
        assert!(frame.dirty().is_empty());
    }

    #[test]
    fn target_kind_round_trips_through_target_id() {
        let id = TargetId::new(7, 3, TargetKind::AtlasNode as u16);
        assert_eq!(id.kind(), TargetKind::AtlasNode as u16);
        assert_eq!(id.index(), 7);
        assert_eq!(id.generation(), 3);
    }

    // ── Rect::contains edge cases ─────────────────────────────────────────────

    #[test]
    fn rect_contains_point_on_left_edge_is_inclusive() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(
            r.contains(10.0, 30.0),
            "left edge (x == rect.x) is inclusive"
        );
    }

    #[test]
    fn rect_contains_point_on_top_edge_is_inclusive() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(
            r.contains(50.0, 20.0),
            "top edge (y == rect.y) is inclusive"
        );
    }

    #[test]
    fn rect_does_not_contain_point_on_bottom_edge() {
        // Half-open: y == rect.y + rect.height is exclusive.
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(
            !r.contains(50.0, 70.0),
            "bottom edge (y == y + height) is exclusive"
        );
    }

    #[test]
    fn zero_size_rect_contains_nothing() {
        // A Rect with width=0 or height=0 has no interior; every point is outside.
        let zero_w = Rect::new(10.0, 10.0, 0.0, 50.0);
        assert!(
            !zero_w.contains(10.0, 20.0),
            "zero-width rect contains nothing"
        );

        let zero_h = Rect::new(10.0, 10.0, 50.0, 0.0);
        assert!(
            !zero_h.contains(20.0, 10.0),
            "zero-height rect contains nothing"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // comparing literal → stored literal, no arithmetic, always bit-exact
    fn rect_default_is_all_zeros() {
        let r = Rect::default();
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.width, 0.0);
        assert_eq!(r.height, 0.0);
    }

    // ── Viewport ─────────────────────────────────────────────────────────────

    #[test]
    #[allow(clippy::float_cmp)] // round-trip through Viewport::new: no arithmetic, bit-exact
    fn viewport_new_round_trips() {
        let vp = Viewport::new(1920.0, 1080.0);
        assert_eq!(vp.width, 1920.0);
        assert_eq!(vp.height, 1080.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // Default-derived zero: no arithmetic, bit-exact
    fn viewport_default_is_zero() {
        let vp = Viewport::default();
        assert_eq!(vp.width, 0.0);
        assert_eq!(vp.height, 0.0);
    }

    #[test]
    fn viewport_is_copy() {
        const fn assert_copy<T: Copy>() {}
        assert_copy::<Viewport>();
        assert_eq!(std::mem::size_of::<Viewport>(), 8);
    }

    // ── RenderFrame ───────────────────────────────────────────────────────────

    #[test]
    fn render_frame_push_rect_preserves_order() {
        let mut frame = RenderFrame::new();
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(20.0, 20.0, 30.0, 30.0);
        frame.push_rect(a, false);
        frame.push_rect(b, true);
        assert_eq!(frame.rects()[0], a);
        assert_eq!(frame.rects()[1], b);
    }

    #[test]
    fn render_frame_dirty_is_parallel_to_rects() {
        let mut frame = RenderFrame::new();
        frame.push_rect(Rect::new(0.0, 0.0, 10.0, 10.0), false);
        frame.push_rect(Rect::new(10.0, 0.0, 10.0, 10.0), true);
        frame.push_rect(Rect::new(20.0, 0.0, 10.0, 10.0), false);

        assert_eq!(frame.dirty(), &[false, true, false]);
        assert_eq!(frame.dirty().len(), frame.rects().len());
    }

    #[test]
    fn render_frame_starts_with_no_dirty_entries() {
        let frame = RenderFrame::new();
        assert!(frame.dirty().is_empty());
    }

    #[test]
    #[allow(clippy::float_cmp)] // x=99.0 stored from a literal, no arithmetic, bit-exact
    fn render_frame_clear_retains_capacity_for_reuse() {
        // Clearing a frame with N rects and immediately re-populating with N
        // rects should not reallocate. We verify correctness (no stale data),
        // not performance — allocation is observable only via Miri/asan.
        let mut frame = RenderFrame::new();
        for i in 0..10 {
            #[allow(clippy::cast_precision_loss)]
            frame.push_rect(Rect::new(i as f32, 0.0, 10.0, 10.0), false);
        }
        frame.clear();
        assert!(frame.rects().is_empty(), "clear must empty the frame");

        frame.push_rect(Rect::new(99.0, 0.0, 1.0, 1.0), true);
        assert_eq!(frame.rects().len(), 1, "can push after clear");
        assert_eq!(frame.rects()[0].x, 99.0);
        assert_eq!(frame.dirty(), &[true]);
    }
}

#[cfg(test)]
mod motion_tests {
    use super::*;

    fn spring(from: f32, to: f32, start_ms: u32) -> Motion {
        Motion {
            from,
            to,
            start_ms,
            curve: MotionCurve {
                kind: MotionCurve::SPRING,
                // The RFC-0010 A2 default: snappy 210/20, no initial velocity.
                params: [210.0, 20.0, 0.0],
            },
        }
    }

    #[test]
    fn spring_starts_at_from_and_approaches_to() {
        let m = spring(10.0, 3.0, 1_000);
        // At the start instant, the value is exactly `from`.
        assert!((m.sample(1_000) - 10.0).abs() < 1e-4);
        // Far in the future it has settled onto `to`.
        assert!((m.sample(1_000 + 10_000) - 3.0).abs() < Motion::DEFAULT_EPS_POS);
    }

    #[test]
    fn spring_velocity_starts_near_its_initial_velocity() {
        let mut m = spring(0.0, 100.0, 0);
        m.curve.params[2] = 50.0; // initial velocity
        assert!(
            (m.velocity(0) - 50.0).abs() < 2.0,
            "v(0) should be ~50, got {}",
            m.velocity(0)
        );
    }

    #[test]
    fn spring_is_unsettled_in_flight_and_settled_at_rest() {
        let m = spring(0.0, 100.0, 0);
        assert!(!m.is_settled(0), "a just-started spring is moving");
        assert!(
            m.is_settled(10_000),
            "a spring long past its start has settled"
        );
    }

    #[test]
    fn overdamped_and_critically_damped_springs_still_reach_the_target() {
        // Critically damped: c = 2*sqrt(k). k=100 -> c=20.
        let mut m = spring(0.0, 5.0, 0);
        m.curve.params = [100.0, 20.0, 0.0];
        assert!((m.sample(0)).abs() < 1e-4);
        assert!((m.sample(6_000) - 5.0).abs() < Motion::DEFAULT_EPS_POS);
        // Overdamped: c well above 2*sqrt(k).
        m.curve.params = [100.0, 60.0, 0.0];
        assert!((m.sample(0)).abs() < 1e-4);
        assert!((m.sample(6_000) - 5.0).abs() < Motion::DEFAULT_EPS_POS);
    }

    #[test]
    fn linear_curve_interpolates_over_its_duration() {
        let m = Motion {
            from: 0.0,
            to: 200.0,
            start_ms: 0,
            curve: MotionCurve {
                kind: MotionCurve::LINEAR,
                params: [200.0, 0.0, 0.0], // 200 ms
            },
        };
        assert!((m.sample(0) - 0.0).abs() < 1e-4);
        assert!((m.sample(100) - 100.0).abs() < 1e-3, "halfway at 100ms");
        assert!((m.sample(200) - 200.0).abs() < 1e-4, "arrived at 200ms");
        assert!((m.sample(999) - 200.0).abs() < 1e-4, "clamped past the end");
        assert!(m.is_settled(200));
    }

    #[test]
    fn ease_in_out_hits_its_endpoints_and_midpoint() {
        let m = Motion {
            from: 0.0,
            to: 1.0,
            start_ms: 0,
            curve: MotionCurve {
                kind: MotionCurve::EASE_IN_OUT,
                params: [100.0, 0.0, 0.0],
            },
        };
        assert!((m.sample(0) - 0.0).abs() < 1e-4);
        assert!((m.sample(100) - 1.0).abs() < 1e-4);
        // Symmetric ease passes through 0.5 at the temporal midpoint.
        assert!((m.sample(50) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn motion_is_pod_and_resting_is_settled() {
        // A resting motion never moves and reports settled immediately.
        let m = Motion::resting(42.0);
        assert!((m.sample(0) - 42.0).abs() < 1e-6);
        assert!((m.sample(9_999) - 42.0).abs() < 1e-6);
        assert!(m.is_settled(0));
        // POD round-trip (crosses the frame boundary as bytes).
        let bytes = bytemuck::bytes_of(&m);
        let back: Motion = *bytemuck::from_bytes(bytes);
        assert_eq!(back, m);
    }

    #[test]
    fn near_critical_and_negative_damping_stay_finite() {
        // A damping ratio a hair off critical must not divide by a vanishing
        // `wd`/`r1−r2` and blow up — the near-critical band routes it through
        // the division-free form.
        let mut m = spring(0.0, 10.0, 0);
        m.curve.params = [100.0, 20.05, 0.0]; // critical is c = 2√k = 20
        for t_ms in [0_u32, 1, 8, 100, 1_000, 6_000] {
            assert!(m.sample(t_ms).is_finite(), "sample must stay finite");
            assert!(m.velocity(t_ms).is_finite(), "velocity must stay finite");
        }
        assert!((m.sample(6_000) - 10.0).abs() < Motion::DEFAULT_EPS_POS);

        // Negative damping is clamped to zero, so the worst case is an undamped
        // (bounded) oscillation — never unbounded exponential growth.
        m.curve.params = [100.0, -50.0, 0.0];
        for t_ms in [0_u32, 100, 1_000, 5_000] {
            let v = m.sample(t_ms);
            assert!(
                v.is_finite() && v.abs() < 1.0e4,
                "clamped damping stays bounded"
            );
        }
    }

    // ── ScrollView content clip (RFC-0005) ───────────────────────────────

    fn box_at(x: f32, y: f32) -> BoxInstance {
        BoxInstance {
            rect: [x, y, 10.0, 10.0],
            color: [1.0; 4],
            radii: [0.0; 4],
            transform: Transform::IDENTITY,
        }
    }

    #[test]
    fn drawables_are_stamped_with_the_active_clip_and_none_outside() {
        let mut f = RenderFrame::new();
        f.push_instance(box_at(0.0, 0.0)); // unclipped
        let c = f.begin_clip(Rect::new(20.0, 20.0, 100.0, 80.0));
        f.push_instance(box_at(30.0, 30.0)); // inside the clip
        f.end_clip();
        f.push_instance(box_at(5.0, 5.0)); // unclipped again

        assert_eq!(f.solid_clips(), &[None, Some(c), None]);
        assert_eq!(f.clips().len(), 1);
        assert_eq!(f.clips()[0].rect, Rect::new(20.0, 20.0, 100.0, 80.0));
    }

    #[test]
    fn a_nested_clip_stores_its_intersection_with_the_parent() {
        let mut f = RenderFrame::new();
        let outer = f.begin_clip(Rect::new(0.0, 0.0, 100.0, 100.0));
        let inner = f.begin_clip(Rect::new(50.0, 50.0, 100.0, 100.0));
        f.push_instance(box_at(60.0, 60.0));
        f.end_clip();
        f.end_clip();

        // Inner clip is the intersection: (50,50) .. (100,100) → 50×50.
        assert_eq!(
            f.clips()[inner as usize].rect,
            Rect::new(50.0, 50.0, 50.0, 50.0)
        );
        assert_eq!(
            f.clips()[outer as usize].rect,
            Rect::new(0.0, 0.0, 100.0, 100.0)
        );
        assert_eq!(f.solid_clips(), &[Some(inner)]);
    }

    #[test]
    fn disjoint_nested_clips_intersect_to_an_empty_rect() {
        let mut f = RenderFrame::new();
        f.begin_clip(Rect::new(0.0, 0.0, 40.0, 40.0));
        let inner = f.begin_clip(Rect::new(100.0, 100.0, 40.0, 40.0)); // no overlap
        f.end_clip();
        f.end_clip();
        let r = f.clips()[inner as usize].rect;
        assert!(r.width == 0.0 || r.height == 0.0, "empty: {r:?}");
    }

    #[test]
    fn every_pool_carries_a_clip_slice_parallel_to_it() {
        let mut f = RenderFrame::new();
        let c = f.begin_clip(Rect::new(0.0, 0.0, 50.0, 50.0));
        f.push_instance(box_at(1.0, 1.0));
        f.push_vector(VectorInstance::new(
            Rect::new(0.0, 0.0, 10.0, 10.0),
            Rect::new(0.0, 0.0, 1.0, 1.0),
            [1.0; 4],
            4.0,
            0,
        ));
        f.end_clip();
        assert_eq!(f.solid_clips(), &[Some(c)]);
        assert_eq!(f.vector_clips(), &[Some(c)]);
    }

    #[test]
    fn clear_resets_the_clip_table_stack_and_slices() {
        let mut f = RenderFrame::new();
        f.begin_clip(Rect::new(0.0, 0.0, 10.0, 10.0));
        f.push_instance(box_at(0.0, 0.0));
        // Deliberately leave the clip open, then clear.
        f.clear();
        assert!(f.clips().is_empty());
        assert!(f.solid_clips().is_empty());
        // The stack is empty, so a fresh push is unclipped.
        f.push_instance(box_at(0.0, 0.0));
        assert_eq!(f.solid_clips(), &[None]);
    }

    // ── Z-layer marks (RFC-0017 layered draw batches) ──────────────────────────

    #[test]
    fn begin_layer_records_every_pool_cursor() {
        let mut f = RenderFrame::new();
        f.push_instance(box_at(0.0, 0.0));
        f.push_instance(box_at(1.0, 1.0));
        f.push_text(TextLine {
            x: 0.0,
            y: 0.0,
            text: "hi".to_string(),
            font_size: 12.0,
            color: [1.0; 4],
            dirty: true,
        });
        f.begin_layer();
        assert_eq!(
            f.layer_marks(),
            &[LayerMark {
                solid: 2,
                decorated: 0,
                texture: 0,
                vector: 0,
                text: 1,
                canvas: 0,
            }]
        );
    }

    // ── Canvas shape pool (RFC-0020) ────────────────────────────────────────

    #[test]
    fn push_canvas_shape_stamps_depth_clip_and_layer_cursor() {
        let mut f = RenderFrame::new();
        let c = f.begin_clip(Rect::new(0.0, 0.0, 100.0, 100.0));
        f.push_canvas_shape(CanvasShape {
            kind: CANVAS_SHAPE_ARC,
            params: [24.0, 24.0, 20.0, 0.0, 3.14, 0.0, 0.0, 0.0],
            ..CanvasShape::default()
        });
        f.end_clip();
        assert_eq!(f.canvas_shapes().len(), 1);
        assert_eq!(f.canvas_depths().len(), 1);
        assert_eq!(f.canvas_clips(), &[Some(c)]);
        // The shape advanced the global emission counter: its depth is nearer
        // than the cleared far plane.
        assert!(f.canvas_depths()[0] < DRAW_DEPTH_CLEAR);
        f.begin_layer();
        assert_eq!(f.layer_marks()[0].canvas, 1);
        f.clear();
        assert!(f.canvas_shapes().is_empty());
        assert!(f.canvas_depths().is_empty());
        assert!(f.canvas_clips().is_empty());
    }

    #[test]
    fn canvas_shape_bounds_cover_stroke_and_caps() {
        // Circle r=20 at (24,24), stroke 4 → box must reach at least ±22 from
        // the centre on every side.
        let s = CanvasShape {
            kind: CANVAS_SHAPE_CIRCLE,
            params: [24.0, 24.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            stroke_width: 4.0,
            ..CanvasShape::default()
        };
        let b = s.bounds();
        assert!(b.x <= 24.0 - 22.0 && b.y <= 24.0 - 22.0);
        assert!(b.x + b.width >= 24.0 + 22.0 && b.y + b.height >= 24.0 + 22.0);

        // Line with round caps: endpoints overhang by half the stroke width.
        let l = CanvasShape {
            kind: CANVAS_SHAPE_LINE,
            params: [10.0, 10.0, 90.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            stroke_width: 6.0,
            cap: CANVAS_CAP_ROUND,
            ..CanvasShape::default()
        };
        let lb = l.bounds();
        assert!(lb.x <= 7.0 && lb.x + lb.width >= 93.0);
        assert!(lb.y <= 7.0 && lb.y + lb.height >= 13.0);
    }

    #[test]
    fn begin_layer_dedups_an_empty_layer() {
        let mut f = RenderFrame::new();
        f.push_instance(box_at(0.0, 0.0));
        f.begin_layer();
        f.begin_layer(); // nothing emitted in between — must not add a mark
        assert_eq!(f.layer_marks().len(), 1);
        f.push_instance(box_at(1.0, 1.0));
        f.begin_layer(); // pool advanced — a real new layer
        assert_eq!(f.layer_marks().len(), 2);
    }

    #[test]
    fn a_frame_with_no_layers_has_no_marks_and_clear_resets_them() {
        let mut f = RenderFrame::new();
        f.push_instance(box_at(0.0, 0.0));
        assert!(f.layer_marks().is_empty());
        f.begin_layer();
        f.push_instance(box_at(1.0, 1.0));
        f.clear();
        assert!(f.layer_marks().is_empty());
    }
}
