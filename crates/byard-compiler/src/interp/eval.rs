//! The eval driver: walk the AST, wiring declarations to the reactive core
//! (RFC-0002 §"Dev-mode interpreter"; RFC-0004 §3/§11).
//!
//! Each `byld` expression is **lowered** to a reactive computation — a
//! `FnMut(&mut ReactiveCtx) -> Value` closure that resolves identifiers against
//! the [`Env`] *at lowering time* (capturing their `SignalId`/`ScopeId`
//! handles) and performs its `Signal`/memo reads through the context at run
//! time, so read-tracking stays dynamic (RFC-0004 §3). This is the concrete
//! form of RFC-0004's `walk_expr(scope.expr)`:
//!
//! - `var x = init` ⇒ a reactive source: `init` is evaluated once and a signal
//!   is created from it; `x` binds to `Value::Signal`.
//! - `let y` / `fn f` ⇒ a [`ReactiveCtx::open_memo`]; `y`/`f` binds to
//!   `Value::Memo`. Whether it is actually reactive is observed by the tracker
//!   (D3), not declared.
//! - Reading a `var`/memo identifier routes through `read_signal`/`read_memo`.
//! - `untrack(expr)` is a reserved-name call dispatched to [`untrack`].
//! - A mutation (`=`, `+=`, `++`, `--`) on a `var` marks it; on anything else
//!   it is [`CompileError::NotAssignable`].

use super::env::{Env, SignalId, Value};
use super::events::Action;
use super::intrinsics::validate_element;
use super::reactive::{FrameTarget, ReactiveCtx, ScopeId, untrack};
use crate::diagnostics::{CompileError, Span};
use crate::parser::ast::{
    Arg, AssignOp, Attr, AttrKind, BinOp, ElementNode, Expr, Member, Param, PostfixOp, StateBlock,
    StrPart, StyleStateKind, Type, ViewDecl,
};
use crate::symbol::Symbol;
use crate::util::closest_match;

/// Decimal places a `Slider` without an explicit `step` quantises its value to.
///
/// A continuous slider otherwise emits the full `f64` precision of a
/// pixel-derived ratio (e.g. `0.6035294…`); rounding keeps the bound value
/// readable. Authors who need a specific granularity set `step:` instead.
const SLIDER_DEFAULT_DECIMALS: i32 = 3;

/// Maximum user-`View` instantiation depth before lowering truncates with a
/// diagnostic rather than risking a native stack overflow (RFC-0007 §4, D-C).
/// Far beyond any hand-written nesting, shallow enough to never
/// approach the stack limit. The static cycle check (`load_views`) catches
/// *unguarded* cycles at load; this bound is the backstop for a guarded
/// recursion whose runtime guard never terminates at lower time.
const MAX_INSTANCE_DEPTH: u32 = 64;

/// Decimal places implied by `step`, via its shortest round-trip form
/// (`0.1 → 1`, `0.25 → 2`, `1.0 → 0`). Used so a stepped slider never emits a
/// value with more decimal places than the step itself — e.g. `step: 0.1`
/// landing on `6 × 0.1 = 0.6000000000000001` is rounded back to `0.6`. Capped
/// at 10 places (any real step is far coarser).
fn step_decimals(step: f64) -> i32 {
    match format!("{}", step.abs()).split_once('.') {
        Some((_, frac)) => i32::try_from(frac.len().min(10)).unwrap_or(0),
        None => 0,
    }
}

/// Settling thresholds for the CPU-sampled animation path (RFC-0010).
///
/// `eval_pure` animates opacity, scale, rotate, and translate through one
/// generic path that doesn't carry the property's unit, so the epsilons must be
/// tight enough to be correct for the *tightest* unit (ratios, ~0..1) — which is
/// simply conservative (settles a hair later) for pixels and radians. Position
/// is the final-value accuracy gate; a tight velocity gate keeps a spring's
/// overshoot alive rather than freezing it at the first crossing of the target.
const ANIM_SETTLE_EPS_POS: f32 = 0.002;
const ANIM_SETTLE_EPS_VEL: f32 = 0.02;

/// Rounds `val` to `decimals` decimal places (half-away-from-zero).
fn round_to_decimals(val: f64, decimals: i32) -> f64 {
    let factor = 10f64.powi(decimals);
    (val * factor).round() / factor
}

/// A resolved first-class style value (RFC-0016): a flat base attribute set
/// plus its `on <state> { … }` interaction-state blocks. Produced by
/// [`Interpreter::resolve_style_expr`] from a `style { … }` value, a `let`-bound
/// style name, or a `merge` of two styles. Static and view-scoped — no cascade.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StyleDef {
    /// The base attributes, last-write-wins in written order.
    pub base: Vec<Attr>,
    /// The state blocks, in written order (a later block of the same state
    /// wins, which is how `merge` layers the right operand over the left).
    pub states: Vec<StateBlock>,
}

/// A lowered render-tree node: the interpreter's plan for one element. Reactive
/// fields are reactive-scope ids the engine reads each tick (M14).
#[derive(Debug, Clone, PartialEq)]
pub enum RenderNode {
    /// A box-like container.
    Box {
        /// The element intrinsic name.
        name: Symbol,
        /// Styling attributes.
        attrs: Vec<Attr>,
        /// `on <state> { … }` blocks (RFC-0016), overlaid onto `attrs` at render
        /// time when their engine state is active. Empty for the common case.
        state_blocks: Vec<StateBlock>,
        /// Child render nodes.
        children: Vec<RenderNode>,
        /// Event shorthand action.
        action: Option<Expr>,
        /// The `var` signal bound via `bind:` or `value:` (M16: value widgets).
        bound_sig: Option<super::env::SignalId>,
        /// The instance environment captured at lower time (RFC-0019 §2), or
        /// empty at the top level. Event attrs and the `action` are re-lowered
        /// each frame during the render walk; for a box lowered inside a
        /// user-view instance this snapshot restores the callee's `Fn` params
        /// and argument bindings, so a forwarded callback (`tap => on_tap()`)
        /// resolves against the scope it was instantiated in.
        env_snapshot: Vec<(Symbol, super::env::Value)>,
    },
    /// A text run.
    Text {
        /// Styling attributes.
        attrs: Vec<Attr>,
        /// `on <state> { … }` blocks (RFC-0016) overlaid at render time.
        state_blocks: Vec<StateBlock>,
        /// The reactive scope projecting the text content.
        content: ScopeId,
    },
    /// A flexible gap (layout-only).
    Spacer,
    /// A texture-sampled image (M21).
    Image {
        /// Styling attributes (width, height, fit, radii, opacity, …).
        attrs: Vec<Attr>,
        /// `on <state> { … }` blocks (RFC-0016) overlaid at render time.
        state_blocks: Vec<StateBlock>,
        /// The reactive scope that evaluates to the image source path/URL.
        src: ScopeId,
    },
    /// An MSDF vector glyph — the `VectorIcon` intrinsic (RFC-0009 §1)
    /// routed to the `VectorMSDF` pipeline.
    Vector {
        /// Styling attributes (`size`, `color`, `m`, `opacity`, `style`).
        attrs: Vec<Attr>,
        /// The reactive scope evaluating to the asset handle (a `Str` path).
        src: ScopeId,
    },
    /// A `Canvas` — the RFC-0020 programmatic drawing surface. A fixed-size
    /// leaf whose children are *shape commands*, not views: each render tick
    /// re-evaluates every shape's parameter expressions (so a reactive
    /// `sweep: percent * 3.6` animates for free) and lowers Tier-1 shapes to
    /// the `CanvasShape` pipeline, `path` commands to `VectorMSDF` (Tier 2),
    /// and `text` commands to `TextLine`s.
    Canvas {
        /// Styling attributes (`width`/`height`, `bg`, `opacity`, `style`).
        attrs: Vec<Attr>,
        /// `on <state> { … }` blocks (RFC-0016) overlaid at render time.
        state_blocks: Vec<StateBlock>,
        /// The validated shape-command elements, in declaration order. Kept
        /// as AST elements — their named args are ordinary `Expr`s evaluated
        /// per tick through `eval_pure`, which is what makes every parameter
        /// reactive and `with`-animatable (RFC-0010) without extra plumbing.
        shapes: Vec<ElementNode>,
        /// The `=> action` tap shorthand.
        action: Option<Expr>,
        /// The instance environment captured at lower time (RFC-0019 §2), so
        /// shape parameters and event actions referencing instance vars
        /// resolve against the scope the canvas was instantiated in.
        env_snapshot: Vec<(Symbol, super::env::Value)>,
    },
    /// An `Overlay` — the RFC-0017 escape-hatch. Its children leave the normal
    /// layout flow and render in the overlay layer, above all main content and
    /// laid out against the viewport. In the parent tree the node occupies zero
    /// space (a 0×0 layout leaf); the render walk collects it into the overlay
    /// stack and emits it in a deferred second phase.
    Overlay {
        /// Styling/behaviour attributes: `modal`, `dismiss_on_outside`, and the
        /// `dismiss =>` event action.
        attrs: Vec<Attr>,
        /// The overlay's floating content subtree.
        children: Vec<RenderNode>,
        /// The instance environment captured at lower time (RFC-0019 §2), so a
        /// `dismiss` action or a child's forwarded callback resolves against the
        /// scope the overlay was instantiated in. Empty at the top level.
        env_snapshot: Vec<(Symbol, super::env::Value)>,
    },
    /// A reactive `when cond { … } else { … }` (RFC-0018 structural reactivity).
    /// The driver re-reads `cond` every frame and expands the taken branch, so a
    /// `var` flip mounts/unmounts the subtree with no re-lowering. Each branch is
    /// lowered **lazily** on first selection and cached (see [`WhenPool`]) — so a
    /// guarded recursion (`when done { … } else { Recurse() }`) only lowers the
    /// recursive branch when the guard actually reaches it, terminating finitely.
    When {
        /// The reactive predicate, re-read each frame.
        cond: ScopeId,
        /// Index into the interpreter's `when_pools`.
        pool: usize,
    },
    /// A reactive `for item in list { … }` (RFC-0018 structural reactivity).
    /// Coarse, positional reconciliation (RFC-0002 D7): the driver reads `list`
    /// each frame and renders one pooled body per element. Bodies are lowered
    /// lazily into a reusable pool (grown to the high-water length, never
    /// re-lowered per frame), each reading its element from a per-slot signal the
    /// driver updates — so list growth/shrink/value changes are reactive without
    /// re-lowering or churning scopes.
    For {
        /// Index into the interpreter's `for_pools`.
        pool: usize,
        /// The reactive list projection, re-read each frame.
        list: ScopeId,
    },
}

/// A borrowed view of both structural-reactivity caches (RFC-0018), passed
/// through the read-only build/paint phase so `when`/`for` expand consistently.
#[derive(Clone, Copy)]
struct Pools<'a> {
    fors: &'a [ForPool],
    whens: &'a [WhenPool],
}

/// A `when`'s lazily-lowered branch cache (RFC-0018). Each branch's AST is kept
/// and lowered only the first time the condition selects it, then reused — so an
/// untaken (possibly recursive) branch costs nothing until it is actually shown.
struct WhenPool {
    /// The `then` branch AST.
    then_ast: Vec<Member>,
    /// The `else` branch AST (empty when there is no `else`).
    els_ast: Vec<Member>,
    /// User-view names in scope at lower time.
    known_views: Vec<String>,
    /// Instance env captured at lower time (RFC-0019), restored when lowering.
    env_snapshot: Vec<(Symbol, super::env::Value)>,
    /// The lowered `then` branch, once first taken.
    then: Option<Vec<RenderNode>>,
    /// The lowered `else` branch, once first taken.
    els: Option<Vec<RenderNode>>,
}

/// A `for`'s reusable body pool (RFC-0018). Bodies are lowered once per slot and
/// reused across frames; each slot's element value lives in `item_slots[i]`,
/// which the driver rewrites from the current list before painting.
struct ForPool {
    /// The loop variable name, bound to each slot's signal when lowering a body.
    item_var: Symbol,
    /// The loop body AST, re-lowered only when the pool grows to a new index.
    body: Vec<Member>,
    /// User-view names in scope at lower time (for lowering new bodies).
    known_views: Vec<String>,
    /// The instance env captured at lower time (RFC-0019), restored when lowering
    /// a new body so it resolves against the scope the `for` was written in.
    env_snapshot: Vec<(Symbol, super::env::Value)>,
    /// One signal per pooled index, holding that slot's current element value.
    item_slots: Vec<super::env::SignalId>,
    /// One lowered body per pooled index (parallel to `item_slots`).
    bodies: Vec<Vec<RenderNode>>,
    /// How many bodies are live (painted) this frame — the current list length.
    len: usize,
}

/// A lowered reactive computation (see the module docs).
type Lowered = Box<dyn FnMut(&mut ReactiveCtx) -> Value>;

/// One scrollable axis of a [`ScrollTarget`]: the `var` behind `offset.x` or
/// `offset.y` and how far it may travel (content extent − viewport, ≥ 0).
#[derive(Clone, Copy)]
struct ScrollAxis {
    /// The signal backing this axis's offset component; the wheel/drag writes it.
    sig: SignalId,
    /// Maximum scroll distance on this axis (content − viewport), clamped ≥ 0.
    max: f32,
}

/// A wheel/drag-scrollable region recorded during render (RFC-0005
/// `ScrollView`). `dispatch_events` turns a wheel or a drag over `rect` into a
/// clamped write to whichever of `offset.x`/`offset.y` is a writable `var`.
#[derive(Clone, Copy)]
struct ScrollTarget {
    /// Viewport rect in logical screen px (the wheel/drag hit region).
    rect: crate::interp::intrinsics::Rect,
    /// Horizontal axis, present when `offset.x` is a writable `var`.
    x: Option<ScrollAxis>,
    /// Vertical axis, present when `offset.y` is a writable `var`.
    y: Option<ScrollAxis>,
    /// The `ScrollView` element index, for firing engine scroll events
    /// (`end_reached`/`page_change`/`scroll_end`) — RFC-0021.
    elem: Option<u32>,
    /// RFC-0021 `snap: page`: on release, snap the offset to the nearest
    /// viewport-sized page. (`snap: item` boundary snapping is a follow-up.)
    snap_page: bool,
    /// RFC-0021 reflected `page:` var — written the current page index on a
    /// page-snap settle; a `page_change` fires when it changes.
    page_sig: Option<SignalId>,
    /// RFC-0021 `end_threshold` (0..1): the fraction of the scrollable extent at
    /// which `end_reached` fires. `None` when no `end_reached` handler exists.
    end_threshold: Option<f32>,
}

/// An in-flight smooth snap (RFC-0021 §2): a spring driving one `ScrollView`
/// axis's offset signal to an exact page boundary. Seeded when scrolling settles
/// (drag release or scroll-quiet), advanced each `render` from the shared engine
/// clock, and dropped once [`Motion::is_settled_with_eps`] reports it has
/// arrived — at which point the offset is pinned to `target` and `scroll_end`
/// fires. Cancelled outright by any fresh scroll/drag on the same elem.
#[derive(Clone, Copy)]
struct SnapAnim {
    /// The offset-axis signal the spring writes each frame.
    sig: SignalId,
    /// The spring itself (`from` = offset at settle, `to` = `target`).
    motion: byard_core::frame::Motion,
    /// The exact page boundary to pin the offset to on settle.
    target: f32,
}

/// The visible slice of a windowed `ScrollView` list (RFC-0005 windowed layout):
/// only rows `start..end` of a uniform-height list are built, laid out, and
/// emitted, with a leading/trailing [`Spacer`] standing in for the elided rows so
/// the content extent (and thus the scroll clamp) and every visible row's
/// position stay exact. Computed identically in the build and render passes from
/// the same offset, so the parallel flat-id cursor stays aligned.
#[derive(Clone, Copy)]
struct WindowSpec {
    /// Index of the first materialised row.
    start: usize,
    /// One past the last materialised row (`start..end` is the live slice).
    end: usize,
    /// Fixed per-row extent in logical px (spacing folded in).
    row_height: f32,
    /// Total row count, so the trailing spacer covers `n − end` rows.
    n: usize,
}

/// One resolved drop shadow (RFC-0011 custom shadows): offset, blur, spread, and
/// resolved RGBA colour. A box may carry several — CSS-style layered shadows —
/// each emitted as its own shadow-only `DecoratedBox` beneath the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ShadowSpec {
    dx: f32,
    dy: f32,
    blur: f32,
    spread: f32,
    color: [f32; 4],
}

/// Default drop-shadow colour (`0xAARRGGBB`, ~33% black) when a shadow omits its
/// own `color`.
const DEFAULT_SHADOW_COLOR: i64 = 0x5500_0000;

/// Pushes a single rounded stroke quad from `a` to `b` of thickness `t`,
/// rotated to the segment angle about its midpoint and composed under
/// `transform` (so an element/group transform carries the mark with it). Backs
/// the RFC-0018 `Checkbox` checkmark. Emitted on the **decorated** pipeline so it
/// paints above the checkbox's decorated (bordered) container, which is pushed
/// first.
fn push_stroke_quad(
    frame: &mut byard_core::frame::RenderFrame,
    a: [f32; 2],
    b: [f32; 2],
    t: f32,
    color: [f32; 4],
    transform: byard_core::frame::Transform,
) {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len = (dx * dx + dy * dy).sqrt();
    let cx = (a[0] + b[0]) * 0.5;
    let cy = (a[1] + b[1]) * 0.5;
    let seg = byard_core::frame::Transform {
        rotate: dy.atan2(dx),
        origin: [cx, cy],
        ..byard_core::frame::Transform::IDENTITY
    };
    frame.push_decorated(byard_core::frame::DecoratedBox {
        base: byard_core::BoxInstance {
            rect: [cx - len / 2.0, cy - t / 2.0, len, t],
            color,
            radii: [t / 2.0; 4],
            transform: transform.compose(&seg),
        },
        dirty: true,
        ..Default::default()
    });
}

/// A shadow-only [`DecoratedBox`](byard_core::frame::DecoratedBox): the box's
/// geometry (rect/radii/transform) with a transparent fill and no border, so it
/// casts `sh` beneath the surface. Emitted per shadow (RFC-0011 layered shadows).
fn shadow_decorated(
    base: byard_core::BoxInstance,
    opacity: f32,
    sh: &ShadowSpec,
) -> byard_core::frame::DecoratedBox {
    byard_core::frame::DecoratedBox {
        base: byard_core::BoxInstance {
            color: [0.0; 4],
            ..base
        },
        border_width: 0.0,
        border_color: [0.0; 4],
        shadow_dx: sh.dx,
        shadow_dy: sh.dy,
        shadow_blur: sh.blur,
        shadow_spread: sh.spread,
        shadow_color: sh.color,
        opacity,
        dirty: true,
    }
}

/// The per-instance parameter bindings produced by binding a user-view call's
/// arguments to the callee's declared parameters (RFC-0007 §3). Each entry
/// is a reactive memo projecting the argument expression over the *parent*
/// scope, so a parameter fed a parent `var` stays live (RFC-0004); a literal
/// argument lowers to a constant memo with no dirty edges.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct InstanceBindings {
    /// Successfully bound `(param name, projecting memo)` pairs, in parameter
    /// declaration order.
    pub bindings: Vec<(Symbol, ScopeId)>,
}

/// The default-value expression of a parameter, if it declares one (RFC-0007
/// D-B). A parameter with a default is not required at the call site;
/// the default is evaluated in the callee scope when the argument is omitted.
fn param_default(param: &Param) -> Option<&Expr> {
    param.default.as_ref()
}

/// Whether a parameter is a callback prop — declared with a function type
/// `Fn(...)` (RFC-0019). Callback params bind a caller-supplied action block
/// rather than a projected value, so they take a separate binding path and are
/// skipped by the ordinary value-argument machinery in [`Interpreter::bind_args`].
fn is_callback_param(param: &Param) -> bool {
    matches!(param.ty, Some(Type::Function { .. }))
}

/// The reserved parameter/element name for a user view's child-block slot
/// (RFC-0007 D-A). A `View` declaring a `content` parameter accepts a
/// `{ ... }` block at its call sites; referencing `content` as an element inside
/// the body splices the caller-supplied block.
const RESERVED_CONTENT: &str = "content";

thread_local! {
    /// Thread-local storage holding the active payload of the event currently being processed.
    pub static CURRENT_PAYLOAD: std::cell::RefCell<Option<Value>> = const { std::cell::RefCell::new(None) };
}

/// The Dev-mode interpreter for one `View` instance: a reactive context plus
/// the View's binding environment.
#[derive(Default)]
pub struct Interpreter {
    ctx: ReactiveCtx,
    env: Env<'static>,
    next_target: u32,
    errors: Vec<CompileError>,
    /// `var` name → its `Signal`, so a hot-reload can preserve state by
    /// rebinding instead of re-initializing (RFC-0004 §11).
    var_sigs: std::collections::HashMap<Symbol, SignalId>,
    /// Incremental LayoutAtlas.
    pub atlas: byard_core::atlas::layout::LayoutAtlas,
    /// Interactive events router.
    pub router: crate::interp::events::EventRouter,
    /// Glyph-accurate text measurer, created lazily on first layout so the
    /// non-rendering paths (parsing, reactivity tests) never load fonts.
    text_measurer: Option<byard_core::text::TextMeasurer>,
    /// Active design-token theme (RFC-0022; the theme-default layer).
    pub theme: super::theme::Theme,
    /// The reactive `Bool` signal backing the theme's active scheme (`true` ⇒
    /// dark), created by [`set_theme`](Self::set_theme). `theme.primary` reads it
    /// (tracked) and `theme.dark = …` / `bind: theme.dark` writes it, so a scheme
    /// flip drives Mark-and-Pull across every token reference (RFC-0022 §1).
    theme_scheme: Option<SignalId>,
    /// Parameterized `fn` definitions (`fn f(params) => body`, M25) *and*
    /// callback-prop bindings (RFC-0019): stored as `(param names, body expr,
    /// is_callback)` and indexed by `AstId`. Both share the invocation path in
    /// [`Self::lower_call`] — a callback is a caller-supplied action block
    /// inlined at the callee's invocation site; the `is_callback` flag turns on
    /// the RFC-0019 §4 arity/invocability diagnostics that plain `fn`s don't
    /// want.
    fn_table: Vec<(Vec<Symbol>, Expr, bool)>,
    /// The resolved user-`View` registry for this program (RFC-0007 §1).
    /// Built once from `ParsedFile::views` via [`Interpreter::load_views`]; a
    /// call whose name resolves here is a user-view instantiation, not a
    /// container.
    view_table: super::views::ViewTable,
    /// Current user-view instantiation depth, bounded by [`MAX_INSTANCE_DEPTH`]
    /// to guard against runaway guarded recursion (RFC-0007 §4).
    instance_depth: u32,
    /// Stack of caller-supplied child-block slots, one frame per active
    /// user-view instance (RFC-0007 D-A). The block is
    /// pre-lowered in the *caller* scope so a `content` element reference inside
    /// the callee body splices nodes that capture the caller's environment.
    slot_stack: Vec<Vec<RenderNode>>,
    /// Reactive `for` body pools (RFC-0018), indexed by [`RenderNode::For::pool`].
    /// Grows as `for` loops are lowered; each pool holds that loop's reusable
    /// per-slot bodies and element signals. Reconciled once per frame before the
    /// layout/paint walk.
    for_pools: Vec<ForPool>,
    /// Reactive `when` branch caches (RFC-0018), indexed by
    /// [`RenderNode::When::pool`]. Each branch is lowered lazily on first
    /// selection so an untaken (recursive) branch never lowers until shown.
    when_pools: Vec<WhenPool>,
    /// Current engine time (ms since the runner's epoch), set once per frame by
    /// the runner via [`set_now_ms`](Self::set_now_ms). Drives `with`
    /// animations (RFC-0010).
    now_ms: u32,
    /// Whether a host has ever advanced the clock. Distinguishes a real
    /// `set_now_ms(0)` start from "the clock was never set" — without it, a host
    /// that never ticks the clock would pin an animation at `t = 0` (never
    /// settling, `has_active_animations` latched true, an infinite redraw loop
    /// on a wait-based runner). Unset ⇒ animations resolve to their target
    /// instantly.
    clock_set: bool,
    /// Persisted per-property animation state (RFC-0010), keyed by the `with`
    /// node's source span so it survives the whole-tree re-render each frame.
    /// A mid-flight target change reseeds `from` to the current sampled value
    /// (interruptible springs).
    animations: std::collections::HashMap<Span, byard_core::frame::Motion>,
    /// Persisted colour-animation state (RFC-0010 A3): one `Motion` per OKLab
    /// channel (`L`, `a`, `b`), so a `bg`/`color`/`border` transition
    /// interpolates in a perceptually-uniform space — no muddy mid-points — and
    /// is interruptible like the scalar props. Keyed by the `with` node's span.
    color_animations: std::collections::HashMap<Span, [byard_core::frame::Motion; 3]>,
    /// Set true during a render whenever an animation sampled this frame has not
    /// yet settled — the runner reads it (via [`has_active_animations`]) to keep
    /// requesting frames until motion stops (idle → 0 frames).
    ///
    /// [`has_active_animations`]: Self::has_active_animations
    any_active: bool,
    /// First-class style values (RFC-0016): `let name = style { … }` registers
    /// its base attributes and `on <state>` blocks here, and a `..name` spread
    /// on an element splices them in at lower time. Static and view-scoped — no
    /// cascade.
    styles: std::collections::HashMap<Symbol, StyleDef>,
    /// Dev-mode MSDF generation cache/dispatcher for `VectorIcon` (RFC-0009
    /// §2). Drained once per [`render`](Self::render) call, before the tree
    /// walk, so a freshly-resident glyph is visible the same tick it lands.
    vector_jit: crate::vector::VectorJit,
    /// Wheel-scroll targets recorded during the last render (RFC-0005): one per
    /// `ScrollView` whose `offset.y` is a writable signal. `dispatch_events`
    /// reads this to convert a wheel into a clamped scroll — the same
    /// render-then-dispatch handshake the router's hit rects use.
    scroll_targets: Vec<ScrollTarget>,
    /// The drag-to-scroll gesture in flight, if any (RFC-0005). Set when a
    /// pointer press lands on inert `ScrollView` content; each move writes the
    /// offset so the content tracks the pointer; cleared on release.
    scroll_drag: Option<ScrollDrag>,
    /// RFC-0021 `on_end_reached` debounce: `ScrollView` element indices currently
    /// past their `end_threshold` and already fired. An elem re-fires only after
    /// its offset falls back below the threshold (removed here), so appending
    /// items — which lowers the fraction — re-arms it. Persists across ticks
    /// (gesture-like state), keyed by the stable element index.
    end_reached_fired: std::collections::HashSet<u32>,
    /// RFC-0021 reflected `page:` — the last page value synced to the offset per
    /// `ScrollView` elem. Edge-triggered: when the `page` var differs from this
    /// (the app set it), the offset scrolls to `page × viewport`; a drag never
    /// changes `page` mid-gesture, so this never fights scrolling.
    scroll_page_last: std::collections::HashMap<u32, i64>,
    /// RFC-0021 snap settle: the [`frame_seq`](Self::frame_seq) of the last
    /// wheel/trackpad scroll input per `snap`-enabled `ScrollView` elem. Snapping
    /// waits until an elem has been *quiet* (no scroll input) for a few frames, so
    /// trackpad momentum — a stream of ever-smaller deltas that leaves the offset
    /// looking briefly "still" — cannot trigger a snap mid-fling that then fights
    /// the next momentum event. Clock-independent (frame-counted), so it settles
    /// identically whether or not the host advances `now_ms`.
    scroll_quiet: std::collections::HashMap<u32, u64>,
    /// RFC-0021 smooth snap: the in-flight spring driving a `snap: page` view's
    /// offset to its target page, per elem. Seeded on drag release / scroll-quiet
    /// settle and advanced each `render` until it settles (then removed and the
    /// offset pinned exactly on the page). A fresh scroll/drag on the elem cancels
    /// it so the user always takes over cleanly (interruptible).
    snap_anims: std::collections::HashMap<u32, SnapAnim>,
    /// Monotonic render counter (RFC-0021 snap timing): bumped once at the top of
    /// every [`render`](Self::render). Drives the frame-counted "scroll has gone
    /// quiet" test in [`scroll_quiet`](Self::scroll_quiet) without depending on an
    /// advancing wall clock.
    frame_seq: u64,
    /// RFC-0018 `RadioButton` groups: the ordered `value`s of every radio sharing
    /// a `bind:` group var, keyed by that var's [`SignalId`]. Rebuilt each render
    /// (cleared at the top of [`render`](Self::render), appended as each radio is
    /// painted, in declaration order). Each group's ordering is shared into that
    /// group's radios' arrow-key handlers via a cheap `Rc` clone, so the handlers
    /// — which fire *after* the full render has populated the vector — can move
    /// selection to the next/previous value with wrap-around (WAI-ARIA radio
    /// group pattern). `Rc`/`RefCell` are sound here: the interpreter and its
    /// event closures are single-threaded logic-thread state (`!Send`).
    radio_groups: std::collections::HashMap<SignalId, std::rc::Rc<std::cell::RefCell<Vec<String>>>>,
}

/// One axis of a live [`ScrollDrag`]: the signal to write and its value at the
/// press, so the live offset is a pure function of the pointer travel.
#[derive(Clone, Copy)]
struct ScrollDragAxis {
    /// The signal backing this axis's offset component, written as it moves.
    sig: SignalId,
    /// The offset at the press; the live offset is this minus the pointer travel.
    start_offset: f32,
    /// Maximum scroll distance on this axis (content − viewport), clamped ≥ 0.
    max: f32,
    /// Whether `sig` holds an `Int` (write back rounded) or a `Float`.
    is_int: bool,
}

/// A live drag-to-scroll gesture (RFC-0005 `ScrollView`): the content follows
/// the pointer between press and release. Captured at press so the offset is a
/// pure function of the pointer travel — no accumulated drift (IMPL-10).
#[derive(Clone, Copy)]
struct ScrollDrag {
    /// Pointer position at the press, in logical screen px.
    start_pos: (f32, f32),
    /// Horizontal axis, present when `offset.x` is a writable `var`.
    x: Option<ScrollDragAxis>,
    /// Vertical axis, present when `offset.y` is a writable `var`.
    y: Option<ScrollDragAxis>,
    /// The dragged `ScrollView` element (for RFC-0021 snap-settle activity).
    elem: Option<u32>,
}

impl Interpreter {
    /// Creates an empty interpreter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The reactive context (for tests and the engine bridge).
    #[must_use]
    pub fn ctx(&self) -> &ReactiveCtx {
        &self.ctx
    }

    /// Diagnostics accumulated while evaluating.
    #[must_use]
    pub fn errors(&self) -> &[CompileError] {
        &self.errors
    }

    /// Builds the user-`View` registry (RFC-0007 §1) from a whole file's
    /// views and stores it on the interpreter, so subsequent `lower_view`/
    /// `lower_element` calls can recognize and expand user-view calls.
    /// Returns the load-time diagnostics — `IntrinsicShadowed` and any
    /// unguarded-cycle `RecursiveView` (RFC-0007 §4) — which are also
    /// recorded in [`Interpreter::errors`].
    pub fn load_views(&mut self, views: &[ViewDecl]) -> Vec<CompileError> {
        let (table, mut diags) = super::views::ViewTable::build(views);
        // Static cycle detection over the call graph.
        let graph = super::views::CallGraph::build(&table);
        if let Some((view, path)) = graph.unguarded_cycle(&table) {
            diags.push(CompileError::RecursiveView {
                span: table.decl(view).span,
                path,
            });
        }
        self.view_table = table;
        self.errors.extend(diags.iter().cloned());
        diags
    }

    /// The resolved user-`View` registry (for the reload pass and tests).
    #[must_use]
    pub fn view_table(&self) -> &super::views::ViewTable {
        &self.view_table
    }

    /// Runs one tick: begins an epoch and pulls all dirty scopes.
    pub fn tick(&mut self) {
        let epoch = self.ctx.begin_tick();
        self.ctx.pull(epoch);
    }

    /// Sets the current engine time (ms since the runner's epoch) that `with`
    /// animations sample against (RFC-0010). The runner calls this once per
    /// frame, before [`render`](Self::render).
    pub fn set_now_ms(&mut self, ms: u32) {
        self.now_ms = ms;
        self.clock_set = true;
    }

    /// Wires in the channel the render thread reports applied vector-atlas
    /// upload ids through (RFC-0009 §2-C), so the dev JIT cache stops
    /// re-sending an upload once it knows the GPU actually received it. Call
    /// once, right after construction, before the first [`render`](Self::render).
    pub fn set_vector_ack_receiver(&mut self, rx: crossbeam_channel::Receiver<u64>) {
        self.vector_jit.set_ack_receiver(rx);
    }

    /// Invalidates any cached MSDF field generated from the asset at `path`, so
    /// a saved `.svg` regenerates live (RFC-0009 §3, M47). The dev runner calls
    /// this on the logic thread when the file watcher reports an SVG change; the
    /// regenerated field reuses the same atlas cell, so the consuming `View`
    /// never remounts. Returns `true` if a cached asset matched `path`.
    pub fn invalidate_vector_asset(&mut self, path: &std::path::Path) -> bool {
        self.vector_jit.invalidate_path(path)
    }

    /// Points the vector JIT at a persistent on-disk field cache (RFC-0009 §5,
    /// M52), so cold `byard dev` starts load previously generated fields instead
    /// of regenerating them. The dev runner passes `.byard/cache/vectors/`.
    pub fn set_vector_cache_dir(&mut self, dir: std::path::PathBuf) {
        self.vector_jit.set_cache_dir(dir);
    }

    /// Whether any `with` animation was still in flight as of the last
    /// [`render`](Self::render). The runner keeps requesting frames while this
    /// is true and lets the app idle (0 frames) once every animation settles.
    #[must_use]
    pub fn has_active_animations(&self) -> bool {
        self.any_active
    }

    /// The most recently projected value of a value binding (for tests).
    #[must_use]
    pub fn binding_value(&self, s: ScopeId) -> Option<Value> {
        self.ctx.binding_value(s)
    }

    /// Allocates the next frame-target id for a value binding.
    fn next_target(&mut self) -> FrameTarget {
        let t = FrameTarget(self.next_target);
        self.next_target += 1;
        t
    }

    /// Glyph-accurate `(width, height)` of `text` at `font_size`, lazily
    /// initializing the font system on first use.
    fn measure_text(&mut self, text: &str, font_size: f32) -> (f32, f32) {
        self.measure_text_wrapped(text, font_size, None)
    }

    /// Measures `text`, wrapping to `max_width` logical pixels when `Some`
    /// (RFC-0018 text wrap). Returns the wrapped `(width, height)`.
    fn measure_text_wrapped(
        &mut self,
        text: &str,
        font_size: f32,
        max_width: Option<f32>,
    ) -> (f32, f32) {
        self.text_measurer
            .get_or_insert_with(byard_core::text::TextMeasurer::new)
            .measure_wrapped(text, font_size, max_width)
    }

    // ── declarations ────────────────────────────────────────────────────

    /// Processes the declaration-level members of a `View` body (`var`/`let`/
    /// `fn`/`inject`/bare expression). Elements are lowered by the intrinsics
    /// layer (M10).
    pub fn eval_view_decls(&mut self, view: &ViewDecl) {
        for member in &view.body {
            self.eval_member(member);
        }
    }

    fn eval_member(&mut self, member: &Member) {
        match member {
            Member::Var { name, init, .. } => {
                self.define_var(name.clone(), init);
            }
            Member::Let { name, init, .. } => {
                // `let x = style { … }` / `let x = a merge b` (RFC-0016) register
                // a style value in the view-scoped table rather than a reactive
                // memo; a `..x` spread splices its attributes at lower time.
                if matches!(init, Expr::StyleValue { .. } | Expr::Merge { .. }) {
                    match self.resolve_style_expr(init) {
                        Some(def) => {
                            self.styles.insert(name.clone(), def);
                        }
                        None => self
                            .errors
                            .push(CompileError::NotAStyle { span: init.span() }),
                    }
                } else {
                    self.define_let(name.clone(), init);
                }
            }
            Member::Fn {
                name, params, body, ..
            } => {
                if params.is_empty() {
                    // No-param fn: lower body to a memo (existing behavior).
                    self.define_let(name.clone(), body);
                } else {
                    // Parameterized fn (M25): store params+body in fn_table,
                    // bind Value::Fn(AstId) in env.
                    let id = crate::interp::env::AstId(
                        u32::try_from(self.fn_table.len()).unwrap_or(u32::MAX),
                    );
                    let param_names: Vec<Symbol> = params.iter().map(|p| p.name.clone()).collect();
                    self.fn_table.push((param_names, body.clone(), false));
                    self.env.push(name.clone(), Value::Fn(id));
                }
            }
            Member::Expr(e) => {
                if let Err(err) = self.eval_action(e) {
                    self.errors.push(err);
                }
            }
            Member::Inject { ty, name, span } => {
                // Resolve `inject T as name` from the ambient environment chain (M23).
                let ty_name = match ty {
                    crate::parser::ast::Type::Named { name: n, .. } => n.clone(),
                    crate::parser::ast::Type::Function { .. } => Symbol::intern("?"),
                };
                match self.env.resolve_inject(&ty_name).cloned() {
                    Some(val) => self.env.push(name.clone(), val),
                    None => self.errors.push(CompileError::UnresolvedInject {
                        span: *span,
                        name: ty_name.as_str().to_string(),
                    }),
                }
            }
            // elements / control flow / style handled in lower_members.
            _ => {}
        }
    }

    /// `var x = init` — evaluate `init` once, create a reactive source from it.
    pub fn define_var(&mut self, name: Symbol, init: &Expr) -> SignalId {
        let initial = self.eval_pure(init);
        let sig = self.ctx.create_signal(initial);
        self.env.push(name.clone(), Value::Signal(sig));
        self.var_sigs.insert(name, sig);
        sig
    }

    /// The `Signal` backing the `var` named `name`, if any.
    #[must_use]
    pub fn var_signal(&self, name: &Symbol) -> Option<SignalId> {
        self.var_sigs.get(name).copied()
    }

    /// Writes a value to a `Signal` (a controller result or test driver), running
    /// the mark cascade.
    pub fn write_var(&mut self, sig: SignalId, value: Value) {
        self.ctx.write_signal(sig, value);
    }

    /// Reads a `Signal`'s current value without tracking.
    #[must_use]
    pub fn peek(&self, sig: SignalId) -> Value {
        self.ctx.peek_signal(sig)
    }

    /// The number of nodes in the last-computed layout atlas — the direct
    /// witness that a windowed `ScrollView` lays out O(visible), not O(list)
    /// (RFC-0005 windowed layout).
    #[cfg(test)]
    #[must_use]
    fn atlas_node_count(&self) -> usize {
        self.atlas.node_count()
    }

    // ── M23: Controller boundary ─────────────────────────────────────────

    /// Provides an ambient value keyed by `ty` to this view and its
    /// descendants (`inject T as name` resolution, RFC-0002 §inject).
    /// Call before [`lower_view`](Self::lower_view) so the environment is
    /// ready when the view body is evaluated.
    pub fn inject_provider(&mut self, ty: &str, value: Value) {
        self.env.provide(Symbol::intern(ty), value);
    }

    /// Installs `theme` as the active design-token theme and provides it as the
    /// ambient `Theme` so `inject Theme as t` resolves in every view (RFC-0022).
    ///
    /// Creates the reactive scheme signal (a `Bool`, `true` ⇒ dark) seeded from
    /// the theme's active scheme, then provides a [`Value::Theme`] carrying it.
    /// Call once, before [`lower_view`](Self::lower_view). Idempotent: re-calling
    /// reuses the existing scheme signal (so a hot-reload keeps the toggle state).
    pub fn set_theme(&mut self, theme: super::theme::Theme) {
        let dark = theme.active_dark;
        self.theme = theme;
        let sig = if let Some(sig) = self.theme_scheme {
            sig
        } else {
            let sig = self.ctx.create_signal(Value::Bool(dark));
            self.theme_scheme = Some(sig);
            sig
        };
        self.env.provide(Symbol::intern("Theme"), Value::Theme(sig));
    }

    /// Flips the active color scheme (RFC-0022 §1): writes the reactive scheme
    /// signal — marking every binding that reads a theme token dirty — and
    /// mirrors the flag into the theme's non-reactive default accessors. The
    /// next [`tick`](Self::tick) recomputes; the next [`render`](Self::render)
    /// paints the new scheme. A no-op if no theme has been installed.
    ///
    /// This is the programmatic entry point (a controller, or a future OS
    /// dark-mode observer) equivalent to `theme.dark = <dark>` in `byld`.
    pub fn set_theme_dark(&mut self, dark: bool) {
        self.theme.active_dark = dark;
        if let Some(sig) = self.theme_scheme {
            self.ctx.write_signal(sig, Value::Bool(dark));
        }
    }

    /// Whether the active theme scheme is currently dark (RFC-0022 §1).
    #[must_use]
    pub fn theme_is_dark(&self) -> bool {
        self.theme_scheme.map_or(self.theme.active_dark, |sig| {
            self.ctx.peek_signal(sig).as_bool().unwrap_or(false)
        })
    }

    /// Applies a batch of pending I/O results from the async controller pool
    /// (RFC-0001 §5.1). Each callback receives a mutable reference to `self`
    /// and writes to whatever `var` signals it needs via [`write_var`](Self::write_var).
    /// Results are drained before the next [`tick`](Self::tick).
    pub fn apply_io_results(
        &mut self,
        results: impl IntoIterator<Item = Box<dyn FnOnce(&mut Self) + Send>>,
    ) {
        for f in results {
            f(self);
        }
    }

    /// Applies a hot-reload patch (RFC-0002 §"Hot-reload boundary", RFC-0004
    /// §11). On a [`reactive-compatible`](super::reload::ReloadKind) patch the
    /// existing `Signal`s are **kept** (matched by name) so state survives; on a
    /// structure-incompatible patch every `var` is re-initialized from the new
    /// AST (state resets). The reactive scopes are rebuilt from the new AST
    /// either way — read-tracking re-derives the dependency graph (§11).
    pub fn reload(&mut self, new_view: &ViewDecl, kind: super::reload::ReloadKind) {
        use super::reload::ReloadKind;
        let old = std::mem::take(&mut self.var_sigs);
        self.env = Env::new();
        for member in &new_view.body {
            match member {
                Member::Var { name, init, .. } => {
                    if matches!(kind, ReloadKind::ReactiveCompatible) {
                        if let Some(&sig) = old.get(name) {
                            // Keep the live Signal (and its value).
                            self.env.push(name.clone(), Value::Signal(sig));
                            self.var_sigs.insert(name.clone(), sig);
                            continue;
                        }
                    }
                    self.define_var(name.clone(), init);
                }
                Member::Let { name, init, .. }
                | Member::Fn {
                    name, body: init, ..
                } => {
                    self.define_let(name.clone(), init);
                }
                _ => {}
            }
        }
    }

    /// `let y = init` (and `fn`) — open a computed memo.
    pub fn define_let(&mut self, name: Symbol, init: &Expr) -> ScopeId {
        let compute = self.lower_expr(init, None);
        let scope = self.ctx.open_memo(compute);
        self.env.push(name, Value::Memo(scope));
        scope
    }

    /// Opens a value binding projecting `expr` into a fresh frame target
    /// (used by intrinsics, M10, and by tests).
    pub fn bind_value(&mut self, expr: &Expr) -> ScopeId {
        let target = self.next_target();
        let compute = self.lower_expr(expr, None);
        self.ctx.open_value_binding(target, compute)
    }

    /// Reads a memo's current value (for the engine bridge and tests). Pulls it
    /// on demand if dirty.
    pub fn read_memo(&mut self, scope: ScopeId) -> Value {
        self.ctx.read_memo(scope)
    }

    // ── argument → parameter binding (RFC-0007 §3) ──────────────────

    /// Projects one call argument into a reactive memo over the **parent** scope
    /// (the env active at the call site), so a parameter fed a parent `var` stays
    /// live (RFC-0004); a literal lowers to a constant memo with no dirty edges.
    fn project_arg(&mut self, expr: &Expr) -> ScopeId {
        let compute = self.lower_expr(expr, None);
        self.ctx.open_memo(compute)
    }

    /// Binds a single named argument (`name: value`, from `(...)` or `#[...]`) to
    /// the callee parameter of the same `Symbol`, filling `slots[i]` and emitting
    /// `UnknownParam`/`DuplicateParam` as needed (RFC-0007 §3/§6).
    fn bind_named_arg(
        &mut self,
        params: &[Param],
        callee: &str,
        name: &Symbol,
        value: &Expr,
        slots: &mut [Option<ScopeId>],
    ) {
        // The reserved `content` slot is filled by the child block, never a
        // named value.
        if name.as_str() == RESERVED_CONTENT {
            return;
        }
        match params.iter().position(|p| &p.name == name) {
            // A callback prop is bound separately (RFC-0019): it captures the
            // caller's action block, not a projected value, so leave its slot
            // empty here and let `bind_callbacks` handle it.
            Some(i) if is_callback_param(&params[i]) => {}
            Some(i) if slots[i].is_none() => {
                slots[i] = Some(self.project_arg(value));
            }
            Some(_) => self.errors.push(CompileError::DuplicateParam {
                span: value.span(),
                name: name.as_str().to_string(),
                callee: callee.to_string(),
            }),
            None => {
                let hint = closest_match(name.as_str(), params.iter().map(|p| p.name.as_str()))
                    .map(str::to_string);
                self.errors.push(CompileError::UnknownParam {
                    span: value.span(),
                    name: name.as_str().to_string(),
                    callee: callee.to_string(),
                    hint,
                });
            }
        }
    }

    /// Binds a user-view call's positional `content` and named `content`/`attrs`
    /// arguments to the callee's declared parameters, producing one reactive memo
    /// per bound parameter (RFC-0007 §3) and the §6 diagnostics
    /// (`ViewArityMismatch`/`UnknownParam`/`MissingParam`/`DuplicateParam`).
    ///
    /// Positional arguments (unnamed `(...)` entries) match by declaration order;
    /// named arguments (`name:` in `(...)` or `#[name: value]`) match by symbol.
    pub fn bind_args(&mut self, callee: &ViewDecl, call: &ElementNode) -> InstanceBindings {
        let params = &callee.params;
        let callee_name = callee.name.as_str().to_string();
        let mut slots: Vec<Option<ScopeId>> = vec![None; params.len()];
        // Positional arguments map only to *value* parameters; the reserved
        // `content` slot is filled by the child block, not a value.
        let value_param_idx: Vec<usize> = params
            .iter()
            .enumerate()
            .filter(|(_, p)| p.name.as_str() != RESERVED_CONTENT && !is_callback_param(p))
            .map(|(i, _)| i)
            .collect();
        let mut positional_count = 0usize;
        let mut next_positional = 0usize;

        // 1) `(...)` content: unnamed → positional by order; named → by symbol.
        for arg in &call.content {
            if let Some(name) = &arg.name {
                self.bind_named_arg(params, &callee_name, name, &arg.value, &mut slots);
            } else {
                positional_count += 1;
                if let Some(&i) = value_param_idx.get(next_positional) {
                    next_positional += 1;
                    let scope = self.project_arg(&arg.value);
                    if slots[i].is_some() {
                        self.errors.push(CompileError::DuplicateParam {
                            span: arg.value.span(),
                            name: params[i].name.as_str().to_string(),
                            callee: callee_name.clone(),
                        });
                    } else {
                        slots[i] = Some(scope);
                    }
                }
                // Excess positional args are reported once via the arity check
                // below.
            }
        }

        // 2) `#[name: value]` attrs: named arguments (events are not parameters).
        for attr in &call.attrs {
            if let AttrKind::Prop { value } = &attr.kind {
                self.bind_named_arg(params, &callee_name, &attr.name, value, &mut slots);
            }
        }

        // 3) Arity: more positional args than the callee declares value
        //    parameters (RFC-0007 §6).
        if positional_count > value_param_idx.len() {
            self.errors.push(CompileError::ViewArityMismatch {
                span: call.span,
                name: callee_name.clone(),
                expected: value_param_idx.len(),
                found: positional_count,
            });
        }

        // 4) Missing required parameters: an unbound parameter with no default
        //   . The reserved `content` slot is never required — it
        //    defaults to an empty block.
        for (i, slot) in slots.iter().enumerate() {
            if slot.is_none()
                && param_default(&params[i]).is_none()
                && params[i].name.as_str() != RESERVED_CONTENT
                // Callback params are checked for presence in `bind_callbacks`.
                && !is_callback_param(&params[i])
            {
                self.errors.push(CompileError::MissingParam {
                    span: call.span,
                    name: params[i].name.as_str().to_string(),
                    callee: callee_name.clone(),
                });
            }
        }

        InstanceBindings {
            bindings: slots
                .into_iter()
                .enumerate()
                .filter_map(|(i, s)| s.map(|sc| (params[i].name.clone(), sc)))
                .collect(),
        }
    }

    // ── callback props (RFC-0019) ───────────────────────────────────────

    /// Registers a caller-supplied callback body in the shared `fn_table`,
    /// returning its [`AstId`]. The body is the caller's action block; it is
    /// lowered later, at the callee's invocation site, against the shared flat
    /// env — which still holds the caller's `var` bindings below the callee
    /// frame — so writes route to the caller's signals (RFC-0019 §2/§3).
    fn register_callback(&mut self, params: &[Symbol], body: &Expr) -> super::env::AstId {
        let id = super::env::AstId(u32::try_from(self.fn_table.len()).unwrap_or(u32::MAX));
        self.fn_table.push((params.to_vec(), body.clone(), true));
        id
    }

    /// Registers an arity-matched no-op callback (an empty action block with
    /// `arity` ignored parameters), used when a bare-identifier forward cannot
    /// be resolved to a live callback in the current lowering context. Matching
    /// the declared arity keeps the invocation-site arity check (§4) quiet.
    fn noop_callback(&mut self, arity: usize, span: Span) -> super::env::AstId {
        let params: Vec<Symbol> = (0..arity)
            .map(|i| Symbol::intern(&format!("__cb_arg{i}")))
            .collect();
        self.register_callback(&params, &Expr::Block(Vec::new(), span))
    }

    /// The caller-supplied argument expression for a named parameter — a `name:`
    /// entry in the `(...)` content or a `#[name: value]` attribute. Callback
    /// props are always passed by name.
    fn find_named_arg<'a>(&self, call: &'a ElementNode, name: &Symbol) -> Option<&'a Expr> {
        call.content
            .iter()
            .find(|a| a.name.as_ref() == Some(name))
            .map(|a| &a.value)
            .or_else(|| {
                call.attrs.iter().find_map(|attr| match &attr.kind {
                    AttrKind::Prop { value } if &attr.name == name => Some(value),
                    _ => None,
                })
            })
    }

    /// Binds a callback-prop parameter (RFC-0019): pushes a `Value::Fn` naming
    /// the caller's action block (or the `= { … }` default, or a forwarded
    /// callback already in scope). Emits the §4 diagnostics — arity mismatch
    /// between the `Fn(...)` type and the block's `|params|`, a non-callback
    /// argument, or a missing required callback.
    fn bind_callback_param(&mut self, param: &Param, call: &ElementNode) {
        let arg_ty_count = match &param.ty {
            Some(Type::Function { params, .. }) => params.len(),
            _ => 0,
        };
        if let Some(arg) = self.find_named_arg(call, &param.name) {
            match arg {
                Expr::Lambda {
                    params, body, span, ..
                } => {
                    if params.len() != arg_ty_count {
                        self.errors.push(CompileError::CallbackArityMismatch {
                            span: *span,
                            name: param.name.as_str().to_string(),
                            expected: arg_ty_count,
                            found: params.len(),
                        });
                    }
                    let id = self.register_callback(params, body);
                    self.env.push(param.name.clone(), Value::Fn(id));
                }
                // Forwarding: `on_tap: outer_on_tap` re-binds a callback already
                // in scope (a wrapper forwarding its own callback prop inward).
                // A bare identifier that does *not* currently resolve to a
                // callback is bound to an arity-matched no-op rather than a hard
                // type error — a wrapper checked in isolation has its own
                // callback params unbound, and that must not false-positive.
                Expr::Ident(other, span) => {
                    if let Some(&Value::Fn(id)) = self.env.lookup(other) {
                        self.env.push(param.name.clone(), Value::Fn(id));
                    } else {
                        let id = self.noop_callback(arg_ty_count, *span);
                        self.env.push(param.name.clone(), Value::Fn(id));
                    }
                }
                other => self.errors.push(CompileError::CallbackTypeMismatch {
                    span: other.span(),
                    callee: call.name.as_str().to_string(),
                    name: param.name.as_str().to_string(),
                }),
            }
        } else if let Some(Expr::Lambda { params, body, .. }) = param_default(param) {
            // The default is an action block (`= {}` / `= {|_|}`); register it.
            let id = self.register_callback(params, body);
            self.env.push(param.name.clone(), Value::Fn(id));
        } else {
            // A required callback with no default and no argument.
            self.errors.push(CompileError::MissingParam {
                span: call.span,
                name: param.name.as_str().to_string(),
                callee: call.name.as_str().to_string(),
            });
        }
    }

    // ── element lowering (RFC-0005) ─────────────────────────────────────

    /// Resolves the `bind:` or `value:` attribute of a value widget to a
    /// `SignalId`. Returns `None` if no such attribute exists or it doesn't
    /// name a `var` (M16).
    fn resolve_bind_sig(&self, attrs: &[Attr]) -> Option<super::env::SignalId> {
        use crate::parser::ast::Expr;
        for attr in attrs {
            if matches!(attr.name.as_str(), "bind" | "value") {
                match &attr.kind {
                    AttrKind::Prop {
                        value: Expr::Ident(name, _),
                    } => {
                        if let Some(super::env::Value::Signal(sig)) = self.env.lookup(name) {
                            return Some(*sig);
                        }
                    }
                    // `bind: theme.dark` binds a toggle straight to the reactive
                    // scheme flag (RFC-0022 §1) — tapping it recolors the tree.
                    AttrKind::Prop { value } => {
                        if let Some(sig) = self.resolve_theme_scheme_target(value) {
                            return Some(sig);
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Resolves *only* the `bind:` attribute of a `RadioButton` (RFC-0018) to the
    /// group `var`'s `SignalId`. Unlike [`resolve_bind_sig`], it never inspects
    /// `value:` — for a radio, `value` is the button's literal identity string,
    /// not a signal binding. Returns `None` if `bind:` is absent or doesn't name
    /// a `var`.
    fn resolve_group_bind_sig(&self, attrs: &[Attr]) -> Option<super::env::SignalId> {
        use crate::parser::ast::Expr;
        for attr in attrs {
            if attr.name.as_str() == "bind" {
                if let AttrKind::Prop {
                    value: Expr::Ident(name, _),
                } = &attr.kind
                {
                    if let Some(super::env::Value::Signal(sig)) = self.env.lookup(name) {
                        return Some(*sig);
                    }
                }
            }
        }
        None
    }

    /// Resolves the attribute `name`'s value to a writable `var`'s `SignalId`
    /// when it is a bare identifier bound to a `var` (else `None`). Backs
    /// RFC-0021's reflected `page:` prop; a small generalization of
    /// [`resolve_group_bind_sig`](Self::resolve_group_bind_sig).
    fn resolve_named_var_sig(&self, attrs: &[Attr], name: &str) -> Option<super::env::SignalId> {
        use crate::parser::ast::Expr;
        for attr in attrs {
            if attr.name.as_str() == name {
                if let AttrKind::Prop {
                    value: Expr::Ident(n, _),
                } = &attr.kind
                {
                    if let Some(super::env::Value::Signal(sig)) = self.env.lookup(n) {
                        return Some(*sig);
                    }
                }
            }
        }
        None
    }

    /// The prop/value-driven style states an element contributes (RFC-0024), on
    /// top of the router's pointer/focus/drag states: `checked` (a value-widget's
    /// bound value is true), `selected` (the `selected:` prop, or a `RadioButton`
    /// whose `bind == value`), `invalid` (the `invalid:` prop), and
    /// `indeterminate` (a `Checkbox`'s mixed prop). `checked` and `indeterminate`
    /// are mutually exclusive — `indeterminate` clears `checked` (RFC-0024).
    fn prop_style_state(
        &mut self,
        attrs: &[Attr],
        bound_sig: Option<super::env::SignalId>,
        name: &str,
    ) -> crate::interp::events::StyleState {
        use crate::interp::events::StyleState;
        let mut s = StyleState::empty();
        if self.eval_bool_prop(attrs, "selected") == Some(true) {
            s = s.union(StyleState::SELECTED);
        }
        if self.eval_bool_prop(attrs, "invalid") == Some(true) {
            s = s.union(StyleState::INVALID);
        }
        let indeterminate =
            name == "Checkbox" && self.eval_bool_prop(attrs, "indeterminate") == Some(true);
        if indeterminate {
            s = s.union(StyleState::INDETERMINATE);
        }
        // `checked` from a value-widget's bound bool — suppressed while mixed.
        if matches!(name, "Checkbox" | "Toggle") && !indeterminate {
            let checked =
                bound_sig.is_some_and(|sig| self.ctx.peek_signal(sig).as_bool().unwrap_or(false));
            if checked {
                s = s.union(StyleState::CHECKED);
            }
        }
        // `selected` from a `RadioButton` whose group var equals its value.
        if name == "RadioButton" {
            let value = self.eval_str_prop(attrs, "value").unwrap_or_default();
            let selected = bound_sig.is_some_and(
                |sig| matches!(self.ctx.peek_signal(sig), Value::Str(v) if v == value),
            );
            if selected {
                s = s.union(StyleState::SELECTED);
            }
        }
        s
    }

    /// The signals backing a `ScrollView`'s `offset.x` and `offset.y` (RFC-0005),
    /// each present when that tuple component is a writable `var` — e.g.
    /// `offset: (panX, scrollY)` yields both, `offset: (0, scrollY)` only the y.
    /// A component that is a literal or computed value yields `None` (that axis
    /// is inert to wheel/drag; the app drives it). Returned as `(x, y)`.
    fn resolve_offset_sigs(
        &self,
        attrs: &[Attr],
    ) -> (Option<super::env::SignalId>, Option<super::env::SignalId>) {
        use crate::parser::ast::Expr;
        let Some(value) = attrs.iter().find_map(|a| match (&a.name, &a.kind) {
            (n, AttrKind::Prop { value }) if n.as_str() == "offset" => Some(value),
            _ => None,
        }) else {
            return (None, None);
        };
        // `offset: (x, y)` — a component is scrollable iff it names a `var`.
        let sig_at = |i: usize| -> Option<super::env::SignalId> {
            let Expr::Tuple(args, _) = value else {
                return None;
            };
            let Some(Expr::Ident(name, _)) = args.get(i).map(|a| &a.value) else {
                return None;
            };
            match self.env.lookup(name) {
                Some(super::env::Value::Signal(sig)) => Some(*sig),
                _ => None,
            }
        };
        (sig_at(0), sig_at(1))
    }

    /// The visible row window of a windowed `ScrollView` (RFC-0005), or `None`
    /// when it is not `windowed`, its `row_height` is unset/≤ 0, or it has no
    /// uniform list child. The window brackets the viewport with a couple of
    /// overscan rows so a partially-scrolled row is always materialised. Computed
    /// from the *current* `offset.y`, and — because both passes read the same
    /// offset within one render — identically in build and render.
    fn scroll_window(&mut self, sv_attrs: &[Attr], child_count: usize) -> Option<WindowSpec> {
        // Overscan rows on each side keep a row that is only partly scrolled into
        // view fully materialised, and hide the one-frame lag between an input
        // and the re-render that follows it.
        const OVERSCAN: usize = 2;
        if self.eval_bool_prop(sv_attrs, "windowed") != Some(true) {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let row_height = self.eval_int_prop(sv_attrs, "row_height").unwrap_or(0) as f32;
        if row_height <= 0.0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let viewport_h = self.eval_int_prop(sv_attrs, "height").unwrap_or(0) as f32;
        let (_, offset_y) = self.resolve_axis_pair(sv_attrs, "offset", (0.0, 0.0));

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let first = (offset_y / row_height).floor().max(0.0) as usize;
        let start = first.saturating_sub(OVERSCAN);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let span = (viewport_h / row_height).ceil() as usize + 2 * OVERSCAN + 1;
        let end = start.saturating_add(span).min(child_count);
        Some(WindowSpec {
            start,
            end,
            row_height,
            n: child_count,
        })
    }

    /// Whether a `ScrollView` child's laid-out rectangle, mapped through the
    /// scroll-shifted `transform`, falls entirely outside `clip` — the emission-
    /// culling test (RFC-0005 §3.3). All four corners are transformed so a scaled
    /// ancestor is handled; an unknown rect is conservatively kept (never culled).
    fn child_fully_clipped(
        &self,
        child_id: byard_core::atlas::layout::AtlasNodeId,
        transform: byard_core::frame::Transform,
        clip: byard_core::frame::Rect,
    ) -> bool {
        let Ok(Some(r)) = self.atlas.resolved_rect(child_id) else {
            return false;
        };
        let corners = [
            transform.apply_point([r.x, r.y]),
            transform.apply_point([r.x + r.width, r.y]),
            transform.apply_point([r.x, r.y + r.height]),
            transform.apply_point([r.x + r.width, r.y + r.height]),
        ];
        let min_x = corners.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min);
        let max_x = corners
            .iter()
            .map(|c| c[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = corners.iter().map(|c| c[1]).fold(f32::INFINITY, f32::min);
        let max_y = corners
            .iter()
            .map(|c| c[1])
            .fold(f32::NEG_INFINITY, f32::max);
        max_x <= clip.x
            || min_x >= clip.x + clip.width
            || max_y <= clip.y
            || min_y >= clip.y + clip.height
    }

    /// Lowers an element to a [`RenderNode`], validating it against the §5
    /// attribute contract first (diagnostics accumulate in [`Interpreter::errors`]).
    /// `known_views` are user `ViewDecl` names in scope.
    pub fn lower_element(&mut self, el: &ElementNode, known_views: &[&str]) -> RenderNode {
        // RFC-0016: expand `..style` spreads into a flat attribute set *before*
        // validating or lowering, so everything downstream sees ordinary
        // resolved attributes (and a spread can never leak into the checker).
        let (attrs, state_blocks) = self.expand_style_spreads(&el.attrs);
        // Validate the base *and* every state block's attributes against the
        // intrinsic's contract (an `on hover { bg: … }` must obey the same §5
        // rules as an inline `bg:`); the state attrs are validation-only and do
        // not affect the emitted base set.
        let to_validate = attrs_with_states(&attrs, &state_blocks);
        self.errors
            .extend(validate_element(el, &to_validate, known_views));
        match el.name.as_str() {
            "Text" | "Button" if !el.content.is_empty() => {
                let content = self.bind_value(&el.content[0].value);
                if el.name.as_str() == "Button" {
                    // A Button is a decorated box wrapping its label.
                    RenderNode::Box {
                        name: Symbol::intern("Button"),
                        attrs,
                        state_blocks,
                        children: vec![RenderNode::Text {
                            attrs: Vec::new(),
                            state_blocks: Vec::new(),
                            content,
                        }],
                        action: el.action.clone(),
                        bound_sig: None,
                        env_snapshot: self.capture_env_snapshot(),
                    }
                } else {
                    RenderNode::Text {
                        attrs,
                        state_blocks,
                        content,
                    }
                }
            }
            "Spacer" => RenderNode::Spacer,
            // Image intrinsic → TextureSampler pipeline (M21).
            // Syntax: Image("path.jpg") #[fit: .cover, width: 200, height: 150]
            "Image" => {
                let src_expr = el.content.first().map_or_else(
                    || Expr::StrLit(vec![], crate::diagnostics::Span::new(0, 0)),
                    |c| c.value.clone(),
                );
                let src = self.bind_value(&src_expr);
                RenderNode::Image {
                    attrs,
                    state_blocks,
                    src,
                }
            }
            // VectorIcon intrinsic → VectorMSDF pipeline. Content is an
            // asset handle (a `Str` path), like Image's source.
            "VectorIcon" => {
                let src_expr = el.content.first().map_or_else(
                    || Expr::StrLit(vec![], crate::diagnostics::Span::new(0, 0)),
                    |c| c.value.clone(),
                );
                let src = self.bind_value(&src_expr);
                RenderNode::Vector {
                    attrs: el.attrs.clone(),
                    src,
                }
            }
            // RFC-0020: the `Canvas` drawing surface. Its children are shape
            // commands, validated here (never silently ignored) and carried as
            // AST elements — the render walk re-evaluates their parameter
            // expressions every tick, which is what makes them reactive.
            "Canvas" => {
                self.errors
                    .extend(super::intrinsics::validate_canvas(el, &to_validate));
                let shapes = el
                    .children
                    .iter()
                    .filter_map(|m| match m {
                        Member::Element(c)
                            if super::intrinsics::is_shape_command(c.name.as_str()) =>
                        {
                            Some(c.clone())
                        }
                        _ => None, // rejected by validate_canvas above
                    })
                    .collect();
                RenderNode::Canvas {
                    attrs,
                    state_blocks,
                    shapes,
                    action: el.action.clone(),
                    env_snapshot: self.capture_env_snapshot(),
                }
            }
            // RFC-0017: the `Overlay` escape-hatch. Its children are lowered
            // normally, but the node itself carries them out of the parent flow
            // — the render walk defers them to the overlay layer.
            "Overlay" => {
                let children = self.lower_members(&el.children, known_views);
                RenderNode::Overlay {
                    attrs,
                    children,
                    env_snapshot: self.capture_env_snapshot(),
                }
            }
            // Value widgets: resolve bound signal and keep as leaf nodes (M16/M19).
            // `Checkbox` (RFC-0018) joins them: a `bind: Bool` leaf that owns its
            // square-plus-checkmark visual and flips on tap/Space.
            "Toggle" | "Slider" | "TextField" | "Checkbox" => {
                let bound_sig = self.resolve_bind_sig(&attrs);
                RenderNode::Box {
                    name: el.name.clone(),
                    attrs,
                    state_blocks,
                    children: Vec::new(),
                    action: el.action.clone(),
                    bound_sig,
                    env_snapshot: self.capture_env_snapshot(),
                }
            }
            // RFC-0018 `RadioButton`: like the value widgets, but its `value:` is a
            // literal identity (a `Str`), not a bound signal — only `bind:` names
            // the shared group `var`. Resolve *just* `bind` to the group signal;
            // `value` is read from the attrs at render time.
            "RadioButton" => {
                let bound_sig = self.resolve_group_bind_sig(&attrs);
                RenderNode::Box {
                    name: el.name.clone(),
                    attrs,
                    state_blocks,
                    children: Vec::new(),
                    action: el.action.clone(),
                    bound_sig,
                    env_snapshot: self.capture_env_snapshot(),
                }
            }
            _ => {
                // Box / Column / Row / ScrollView and any other container.
                let children = self.lower_members(&el.children, known_views);
                RenderNode::Box {
                    name: el.name.clone(),
                    attrs,
                    state_blocks,
                    children,
                    action: el.action.clone(),
                    bound_sig: None,
                    env_snapshot: self.capture_env_snapshot(),
                }
            }
        }
    }

    /// Captures the instance environment for a box being lowered (RFC-0019 §2),
    /// or an empty snapshot at the top level. Only boxes lowered *inside* a
    /// user-view instance need one — a top-level box's event actions re-lower
    /// against the persistent root env, exactly as before, so its snapshot stays
    /// empty and render behaviour is unchanged.
    fn capture_env_snapshot(&self) -> Vec<(Symbol, Value)> {
        if self.instance_depth == 0 {
            Vec::new()
        } else {
            self.env.snapshot()
        }
    }

    /// Whether `el` is a user-`View` call: a name that resolves in the view
    /// table and is not an RFC-0005 intrinsic, which always wins.
    fn is_user_view_call(&self, el: &ElementNode) -> bool {
        super::intrinsics::lookup(el.name.as_str()).is_none() && self.view_table.contains(&el.name)
    }

    /// Expands a user-`View` call site into its instantiated subtree, spliced as
    /// siblings at `out` (RFC-0007 §2). Opens a fresh instance scope holding the
    /// argument bindings plus the callee's own local `var`/`let`/`fn`
    /// (isolated per instance), lowers the callee body, then truncates the scope
    /// so the parent environment is untouched.
    fn lower_user_view_call(
        &mut self,
        el: &ElementNode,
        known_views: &[&str],
        out: &mut Vec<RenderNode>,
    ) {
        // A user view is not validated by the intrinsic contract; its argument
        // diagnostics come from `bind_args` (RFC-0007 §3/§6).
        let Some(id) = self.view_table.resolve(&el.name) else {
            // Unreachable in practice (caller checked), but degrade gracefully.
            out.push(self.lower_element(el, known_views));
            return;
        };
        // Own the callee so the `&self.view_table` borrow does not conflict with
        // the `&mut self` lowering below (the table is `Send`/owned, INV-3).
        let callee = self.view_table.decl(id).clone();

        // Runtime depth bound (RFC-0007 §4): a guarded recursion whose
        // guard never terminates at lower time is truncated with a diagnostic
        // rather than overflowing the native stack.
        if self.instance_depth >= MAX_INSTANCE_DEPTH {
            self.errors.push(CompileError::RecursiveView {
                span: el.span,
                path: format!(
                    "{} (instantiation depth bound {MAX_INSTANCE_DEPTH} exceeded)",
                    el.name.as_str()
                ),
            });
            return; // truncate the subtree; never recurse past the bound
        }
        self.instance_depth += 1;

        // Slot (RFC-0007 D-A): a `{ ... }` block is allowed only when
        // the callee declares a `content` parameter; the block is pre-lowered in
        // the *caller* scope (capturing caller `var`s) and pushed as this
        // instance's slot. A block passed to a slot-less callee is
        // `UnexpectedChildren`.
        let has_content_param = callee
            .params
            .iter()
            .any(|p| p.name.as_str() == RESERVED_CONTENT);
        let slot_nodes = if el.children.is_empty() {
            Vec::new()
        } else if has_content_param {
            self.lower_members(&el.children, known_views)
        } else {
            self.errors.push(CompileError::UnexpectedChildren {
                span: el.span,
                name: el.name.as_str().to_string(),
            });
            Vec::new()
        };

        // 1) Bind arguments → parameters in the *parent* scope (RFC-0007 §3).
        let bindings = self.bind_args(&callee, el);

        // 2) Open the per-instance lexical frame (D-D): push each
        //    parameter in declaration order — a bound argument's memo, else a
        //    default evaluated in the callee scope — then the callee's
        //    own declarations.
        let snapshot = self.env.len();
        for param in &callee.params {
            // A callback prop (RFC-0019) binds the caller's action block as a
            // `Value::Fn`, resolved at invocation in `lower_call`, rather than a
            // projected value memo.
            if is_callback_param(param) {
                self.bind_callback_param(param, el);
                continue;
            }
            if let Some((_, scope)) = bindings.bindings.iter().find(|(n, _)| n == &param.name) {
                self.env.push(param.name.clone(), Value::Memo(*scope));
            } else if let Some(default) = param_default(param) {
                // Lowered in the current (callee) frame, so a default may
                // reference earlier parameters.
                let scope = self.project_arg(default);
                self.env.push(param.name.clone(), Value::Memo(scope));
            }
            // An unbound, defaultless parameter already produced a `MissingParam`
            // diagnostic in `bind_args`; leave it unbound.
        }
        // Local `var`/`let`/`fn`/`inject` open in the instance frame, so two
        // instances of the same view keep independent state.
        self.eval_view_decls(&callee);

        // 3) Lower the callee body and splice the roots as siblings (RFC-0007
        //    §2 step 5; reuses the multi-node `when`/`for` splice shape). The
        //    slot is live for `content` references within the body.
        self.slot_stack.push(slot_nodes);
        let nodes = self.lower_members(&callee.body, known_views);
        self.slot_stack.pop();
        out.extend(nodes);

        // 4) Close the instance scope (RFC-0007 §2 step 6).
        self.env.truncate(snapshot);
        self.instance_depth -= 1;
    }

    /// Expands `..style` spreads (RFC-0016) in an attribute list into a flat
    /// set: each spread splices the referenced style's attributes in written
    /// order (a later spread overrides an earlier one), then inline attributes
    /// override every spread. The common, spread-free case returns a plain
    /// clone with no work. A spread that doesn't resolve to a known style is a
    /// [`CompileError::NotAStyle`].
    fn expand_style_spreads(&mut self, attrs: &[Attr]) -> (Vec<Attr>, Vec<StateBlock>) {
        if !attrs
            .iter()
            .any(|a| matches!(a.kind, AttrKind::Spread { .. }))
        {
            return (attrs.to_vec(), Vec::new());
        }
        let mut resolved: Vec<Attr> = Vec::new();
        let mut states: Vec<StateBlock> = Vec::new();
        // 1) Spreads first, in written order. Each spread contributes its base
        //    attributes (last-write-wins) and appends its `on <state>` blocks
        //    (a later spread's block of the same state wins at resolve time).
        for a in attrs {
            if let AttrKind::Spread { value } = &a.kind {
                match self.resolve_style_expr(value) {
                    Some(def) => {
                        for sa in def.base {
                            override_attr(&mut resolved, sa);
                        }
                        states.extend(def.states);
                    }
                    None => self.errors.push(CompileError::NotAStyle { span: a.span }),
                }
            }
        }
        // 2) Inline attributes win over the spreads.
        for a in attrs {
            if !matches!(a.kind, AttrKind::Spread { .. }) {
                override_attr(&mut resolved, a.clone());
            }
        }
        (resolved, states)
    }

    /// Resolves a style expression to a [`StyleDef`] (base attributes + state
    /// blocks): a `let`-bound style name, an inline `style { … }` value, or a
    /// `merge` of two styles.
    fn resolve_style_expr(&self, value: &Expr) -> Option<StyleDef> {
        match value {
            Expr::Ident(name, _) => self.styles.get(name).cloned(),
            Expr::StyleValue { attrs, states, .. } => Some(StyleDef {
                base: attrs.clone(),
                states: states.clone(),
            }),
            // `a merge b` (RFC-0016): the right style overrides the left — its
            // base attributes overlay last-write-wins, and its state blocks are
            // appended so a later block of the same state wins at resolve time.
            Expr::Merge { left, right, .. } => {
                let mut def = self.resolve_style_expr(left)?;
                let over = self.resolve_style_expr(right)?;
                for a in over.base {
                    override_attr(&mut def.base, a);
                }
                def.states.extend(over.states);
                Some(def)
            }
            _ => None,
        }
    }

    /// Lowers a slice of `Member`s into child `RenderNode`s, handling
    /// `Element`, `When`, and `For` (M20).
    fn lower_members(&mut self, members: &[Member], known_views: &[&str]) -> Vec<RenderNode> {
        let mut nodes = Vec::new();
        for m in members {
            self.lower_member_into(m, known_views, &mut nodes);
        }
        nodes
    }

    fn lower_member_into(
        &mut self,
        member: &Member,
        known_views: &[&str],
        out: &mut Vec<RenderNode>,
    ) {
        match member {
            // A `content` reference inside a user-view body splices the slot the
            // current instance was called with (RFC-0007 D-A). The slot
            // nodes were pre-lowered in the caller scope.
            Member::Element(e)
                if e.name.as_str() == RESERVED_CONTENT && !self.slot_stack.is_empty() =>
            {
                if let Some(slot) = self.slot_stack.last() {
                    out.extend(slot.clone());
                }
            }
            Member::Element(e) if self.is_user_view_call(e) => {
                // A user-view call expands into its instantiated subtree, spliced
                // as siblings here (RFC-0007 §2).
                self.lower_user_view_call(e, known_views, out);
            }
            Member::Element(e) => {
                out.push(self.lower_element(e, known_views));
            }
            // RFC-0018 reactive `when`: bind the condition and register a branch
            // cache. Branches are lowered lazily on first selection (see
            // [`WhenPool`]) so an untaken recursive branch never lowers; the
            // driver re-reads the condition each frame and expands the taken one.
            Member::When {
                cond, then, els, ..
            } => {
                let cond_scope = self.bind_value(cond);
                let pool = self.when_pools.len();
                let env_snapshot = self.capture_env_snapshot();
                self.when_pools.push(WhenPool {
                    then_ast: then.clone(),
                    els_ast: els.clone().unwrap_or_default(),
                    known_views: known_views.iter().map(|s| (*s).to_string()).collect(),
                    env_snapshot,
                    then: None,
                    els: None,
                });
                out.push(RenderNode::When {
                    cond: cond_scope,
                    pool,
                });
            }
            // RFC-0018 reactive `for`: bind the list as a reactive projection and
            // register a body pool. Bodies are lowered lazily per slot during
            // reconciliation (never per frame); the driver renders one pooled body
            // per current element.
            Member::For {
                var, iter, body, ..
            } => {
                let list_scope = self.bind_value(iter);
                let pool = self.for_pools.len();
                let env_snapshot = self.capture_env_snapshot();
                self.for_pools.push(ForPool {
                    item_var: var.clone(),
                    body: body.clone(),
                    known_views: known_views.iter().map(|s| (*s).to_string()).collect(),
                    env_snapshot,
                    item_slots: Vec::new(),
                    bodies: Vec::new(),
                    len: 0,
                });
                out.push(RenderNode::For {
                    pool,
                    list: list_scope,
                });
            }
            _ => {}
        }
    }

    /// Walks a render tree, projecting it into a `byard-core` [`RenderFrame`]
    /// using Taffy layout via `byard-core`'s [`LayoutAtlas`].
    #[allow(clippy::similar_names)]
    pub fn render(
        &mut self,
        tree: &[RenderNode],
        frame: &mut byard_core::frame::RenderFrame,
        width: f32,
        height: f32,
    ) {
        use byard_core::frame::Viewport;

        // Recomputed every frame: an animation re-marks itself active below if it
        // sampled without having settled this tick (RFC-0010).
        self.any_active = false;
        // One monotonic tick per render — the clock-independent basis for the
        // RFC-0021 "scroll has gone quiet" snap settle.
        self.frame_seq = self.frame_seq.wrapping_add(1);
        self.atlas.clear();
        // Rebuild the handler set from the fresh layout, but keep the in-flight
        // gesture state (a pending `down`, the focused element) so a tap that
        // spans this re-render is still recognized (RFC-0003 E4).
        self.router.clear_handlers();
        // RFC-0021, over the previous frame's scroll targets (before they are
        // dropped, so the offset writes below are what *this* frame paints):
        //   • reverse `page:` — honour an app-driven `page` change (edge-triggered,
        //     never fights a drag);
        //   • snap settle — snap a `snap: page` view once its scroll has gone quiet
        //     (works for wheel/trackpad, which has no release event).
        self.sync_page_offsets();
        self.advance_snap_anims();
        self.settle_snaps();
        // Wheel-scroll targets are re-recorded each render (RFC-0005), like the
        // router's hit rects.
        self.scroll_targets.clear();
        // RFC-0018 radio groups are rebuilt from the fresh layout each render.
        self.radio_groups.clear();

        // Drain any MSDF generations that finished since the last tick,
        // before the tree walk below, so a freshly-resident glyph is visible
        // the same tick it lands (RFC-0009 §2, INV-2: logic-thread only).
        for upload in self.vector_jit.drain_ready() {
            frame.push_atlas_upload(upload);
        }
        // RFC-0018 structural reactivity, phase A: reconcile the `for` pools
        // (grow to the current list lengths, rewrite element slots) so the tree's
        // reactive structure reflects this frame's state. If anything changed,
        // re-pull so the freshly-mounted/updated bindings project before paint.
        // Iterate to a fixpoint: a branch/body lowered *this* pass creates fresh
        // bindings (its own `when` condition, its `for` list) that are not
        // projected until the next pull, so a newly-mounted nested `when`/`for`
        // would otherwise read stale (false/empty) for a frame. Re-pull and
        // re-reconcile until nothing new mounts. Bounded by the reconcile depth
        // guard, so a runaway recursion still terminates (with a diagnostic).
        let mut passes = 0;
        while self.reconcile_structure(tree, 0) && passes <= MAX_INSTANCE_DEPTH {
            let epoch = self.ctx.begin_tick();
            self.ctx.pull(epoch);
            passes += 1;
        }
        // Phase B is read-only over the pools: take them out so `&mut self`
        // (atlas, router, …) stays free while build/paint borrow them.
        let for_pools = std::mem::take(&mut self.for_pools);
        let when_pools = std::mem::take(&mut self.when_pools);
        let pools = Pools {
            fors: &for_pools,
            whens: &when_pools,
        };

        let mut flat_ids = Vec::new();
        // Expand reactive `when`/`for` at the root, then build each concrete node.
        let root_children = self.build_children(tree, pools, &mut flat_ids);

        // RFC-0017: collect every mounted `Overlay` (pre-order = declaration =
        // mount order) and build each into the *same* atlas as an absolutely
        // positioned wrapper floating over the main tree. Nothing is built when
        // no overlay is mounted, so the overlay path is truly zero-cost — the
        // render root stays the plain main container it always was.
        let mut overlays: Vec<&RenderNode> = Vec::new();
        self.collect_overlays(tree, pools, &mut overlays);
        let mut overlay_layouts: Vec<OverlayLayout<'_>> = Vec::new();
        for ov in overlays {
            if let Some(layout) = self.build_overlay_layout(ov, pools) {
                overlay_layouts.push(layout);
            }
        }

        // The main content container (viewport-sized, column). `None` when the
        // whole view is nothing but overlays.
        let main_id = if root_children.is_empty() {
            None
        } else {
            let root_style =
                byard_core::atlas::layout::ContainerStyle::new(Some(width), Some(height))
                    .with_direction(byard_core::atlas::layout::FlexDir::Column);
            self.atlas.add_container(root_style, &root_children).ok()
        };

        // The render root: with no overlay it is the main container itself (the
        // pre-RFC-0017 shape, unchanged). With overlays it is a super-root
        // holding the main content plus each overlay wrapper as an absolute
        // sibling that neither displaces nor is displaced by the main tree.
        let root_id = if overlay_layouts.is_empty() {
            main_id
        } else {
            let mut super_children = Vec::new();
            if let Some(m) = main_id {
                super_children.push(m);
            }
            for ol in &overlay_layouts {
                super_children.push(ol.wrapper_id);
            }
            let super_style =
                byard_core::atlas::layout::ContainerStyle::new(Some(width), Some(height))
                    .with_direction(byard_core::atlas::layout::FlexDir::Column);
            self.atlas.add_container(super_style, &super_children).ok()
        };

        let Some(root_id) = root_id else {
            // Nothing to lay out (an empty tree). Still restore the pools taken
            // out above, or the next frame would see them empty (RFC-0018).
            self.for_pools = for_pools;
            self.when_pools = when_pools;
            return;
        };
        self.atlas.set_root(root_id).unwrap();
        // Drive layout with the shared text measurer so wrapping `Text` leaves
        // reflow to their parent's width (RFC-0005 default wrap). Disjoint field
        // borrows: `self.atlas` and `self.text_measurer`.
        let measurer = self
            .text_measurer
            .get_or_insert_with(byard_core::text::TextMeasurer::new);
        self.atlas
            .compute_with_text(Viewport::new(width, height), measurer)
            .unwrap();
        self.atlas.populate_frame(frame, &[]);

        let parent_rect = crate::interp::intrinsics::Rect::new(0.0, 0.0, width, height);

        // Emit the main tree (below every overlay in painter's order). Iterate
        // the same expanded concrete node sequence `build_children` laid out, so
        // the flat-id cursor stays in lockstep (RFC-0018).
        if main_id.is_some() {
            let mut flat_idx = 0;
            for node in self.expand_concrete(tree, pools) {
                let node_id = flat_ids[flat_idx];
                self.render_node_with_atlas(
                    node,
                    node_id,
                    frame,
                    &flat_ids,
                    &mut flat_idx,
                    parent_rect,
                    1.0,
                    byard_core::frame::Transform::IDENTITY,
                    None,
                    None,
                    pools,
                );
            }
        }

        // RFC-0017 overlay phase: emit each overlay's children *after* the main
        // tree, so their emission-order depth is nearer and they composite on
        // top (the shared depth buffer resolves cross-layer order — no separate
        // GPU pass needed). Emitted in mount order, so a later overlay stacks
        // over an earlier one. A modal overlay installs a scrim first.
        //
        // `begin_layer` marks the z-layer boundary: the Encoder draws each
        // layer's pools — including its *text* — as one interleaved batch
        // inside the single render pass, so this overlay's transparent
        // geometry (scrim, shadow) alpha-blends over the text and images of
        // everything beneath it instead of being drawn before a frame-final
        // text batch. With no overlay, no mark is recorded and the frame
        // renders through the exact single-layer draw stream.
        for ol in &overlay_layouts {
            frame.begin_layer();
            self.emit_overlay(ol, frame, width, height, pools);
        }

        // RFC-0018: return the (possibly grown) pools taken out for the read-only
        // build/paint phase.
        self.for_pools = for_pools;
        self.when_pools = when_pools;
    }

    /// Flattens a node slice into the concrete nodes to lay out/paint this frame
    /// (RFC-0018 structural reactivity): a `When` expands to its taken branch
    /// (condition re-read live), a `For` to its live pooled bodies, recursively.
    /// A concrete node (`Box`/`Text`/…) passes through unchanged — its *own*
    /// children are expanded when it is built/walked, not here. Build, paint,
    /// `flat_len`, and overlay collection all funnel through this one function, so
    /// they agree on the exact node sequence and the flat-id cursor stays aligned.
    fn expand_concrete<'a>(
        &self,
        nodes: &'a [RenderNode],
        pools: Pools<'a>,
    ) -> Vec<&'a RenderNode> {
        let mut out = Vec::new();
        for n in nodes {
            match n {
                RenderNode::When { cond, pool } => {
                    let take = self
                        .binding_value(*cond)
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    // The taken branch was lowered by `reconcile_structure`; an
                    // as-yet-unselected branch is `None` and expands to nothing.
                    if let Some(p) = pools.whens.get(*pool) {
                        let branch = if take {
                            p.then.as_ref()
                        } else {
                            p.els.as_ref()
                        };
                        if let Some(branch) = branch {
                            out.extend(self.expand_concrete(branch, pools));
                        }
                    }
                }
                RenderNode::For { pool, .. } => {
                    if let Some(p) = pools.fors.get(*pool) {
                        for body in p.bodies.iter().take(p.len) {
                            out.extend(self.expand_concrete(body, pools));
                        }
                    }
                }
                other => out.push(other),
            }
        }
        out
    }

    /// Reconciles the reactive `for` pools before the paint walk (RFC-0018,
    /// coarse D7): reads each live `for`'s list, grows its pool to the list length
    /// (lowering a body the first time an index is needed), rewrites each slot's
    /// element signal, and records the live count. Returns `true` if any slot or
    /// pool changed, so the caller re-pulls to project the new values. Descends
    /// through `when` (taken branch) and `for` (live bodies) so nested loops
    /// reconcile too.
    fn reconcile_structure(&mut self, nodes: &[RenderNode], depth: u32) -> bool {
        // Bound the reconcile recursion: a guarded recursion whose guard never
        // terminates (`when go { Recurse() }` with `go` always true) lowers a new
        // level each descent, so cap it here — the same role `instance_depth`
        // plays at lower time (RFC-0007 §4), but for the reconcile-time expansion.
        // Truncate with a diagnostic rather than overflow the stack (D4: never a
        // silent failure); dedup so a re-render doesn't spam the error list.
        if depth >= MAX_INSTANCE_DEPTH {
            let already = self
                .errors
                .iter()
                .any(|e| matches!(e, CompileError::RecursiveView { .. }));
            if !already {
                self.errors.push(CompileError::RecursiveView {
                    span: crate::diagnostics::Span::new(0, 0),
                    path: format!(
                        "(reactive `when`/`for` recursion exceeded {MAX_INSTANCE_DEPTH})"
                    ),
                });
            }
            return false;
        }
        let mut dirtied = false;
        for n in nodes {
            match n {
                RenderNode::When { cond, pool } => {
                    let take = self
                        .binding_value(*cond)
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    // Lazily lower the taken branch on first selection (so an
                    // untaken recursive branch never lowers), then descend into it
                    // to reconcile any nested `when`/`for`.
                    let already = if take {
                        self.when_pools[*pool].then.is_some()
                    } else {
                        self.when_pools[*pool].els.is_some()
                    };
                    if !already {
                        let ast = if take {
                            self.when_pools[*pool].then_ast.clone()
                        } else {
                            self.when_pools[*pool].els_ast.clone()
                        };
                        let known: Vec<String> = self.when_pools[*pool].known_views.clone();
                        let env_snap = self.when_pools[*pool].env_snapshot.clone();
                        let env_base = self.env.len();
                        for (k, v) in &env_snap {
                            self.env.push(k.clone(), v.clone());
                        }
                        let known_refs: Vec<&str> = known.iter().map(String::as_str).collect();
                        let nodes = self.lower_members(&ast, &known_refs);
                        self.env.truncate(env_base);
                        if take {
                            self.when_pools[*pool].then = Some(nodes);
                        } else {
                            self.when_pools[*pool].els = Some(nodes);
                        }
                        dirtied = true;
                    }
                    // Descend into the (now-lowered) taken branch. Take it out to
                    // avoid aliasing `self.when_pools` during nested reconcile.
                    let branch = if take {
                        self.when_pools[*pool].then.take()
                    } else {
                        self.when_pools[*pool].els.take()
                    };
                    if let Some(branch) = branch {
                        dirtied |= self.reconcile_structure(&branch, depth + 1);
                        if take {
                            self.when_pools[*pool].then = Some(branch);
                        } else {
                            self.when_pools[*pool].els = Some(branch);
                        }
                    }
                }
                RenderNode::For { pool, list } => {
                    let items = match self.binding_value(*list) {
                        Some(Value::List(items)) => items,
                        _ => Vec::new(),
                    };
                    let new_len = items.len();
                    // Grow the pool: lower one body per newly-needed index, each
                    // reading its element from a fresh per-slot signal.
                    while self.for_pools[*pool].bodies.len() < new_len {
                        let slot = self.ctx.create_signal(Value::Unit);
                        // Clone the lowering inputs out first so the borrow on
                        // `self.for_pools` is released before `lower_members`
                        // (which may append *nested* pools to `self.for_pools`).
                        let item_var = self.for_pools[*pool].item_var.clone();
                        let body_ast = self.for_pools[*pool].body.clone();
                        let known: Vec<String> = self.for_pools[*pool].known_views.clone();
                        let env_snap = self.for_pools[*pool].env_snapshot.clone();
                        let env_base = self.env.len();
                        for (k, v) in &env_snap {
                            self.env.push(k.clone(), v.clone());
                        }
                        self.env.push(item_var, Value::Signal(slot));
                        let known_refs: Vec<&str> = known.iter().map(String::as_str).collect();
                        let body_nodes = self.lower_members(&body_ast, &known_refs);
                        self.env.truncate(env_base);
                        // Index `*pool` is still valid (nested lowering only
                        // appended higher indices).
                        self.for_pools[*pool].item_slots.push(slot);
                        self.for_pools[*pool].bodies.push(body_nodes);
                        dirtied = true;
                    }
                    // Update each live slot's element value (value-deduped).
                    for (i, item) in items.iter().enumerate() {
                        let slot = self.for_pools[*pool].item_slots[i];
                        if self.ctx.peek_signal(slot) != *item {
                            self.ctx.write_signal(slot, item.clone());
                            dirtied = true;
                        }
                    }
                    self.for_pools[*pool].len = new_len;
                    // Reconcile nested loops inside the live bodies. Take the
                    // bodies out so nested growth (which mutates `self.for_pools`)
                    // can't alias this pool's own vector.
                    let bodies = std::mem::take(&mut self.for_pools[*pool].bodies);
                    for body in bodies.iter().take(new_len) {
                        dirtied |= self.reconcile_structure(body, depth + 1);
                    }
                    self.for_pools[*pool].bodies = bodies;
                }
                RenderNode::Box { children, .. } | RenderNode::Overlay { children, .. } => {
                    dirtied |= self.reconcile_structure(children, depth + 1);
                }
                _ => {}
            }
        }
        dirtied
    }

    /// Builds the atlas layout for a node slice, expanding reactive `when`/`for`
    /// (RFC-0018) into their concrete children first. Returns the child atlas ids
    /// in paint order, appending each subtree's flat-id list to `flat_ids`.
    fn build_children(
        &mut self,
        nodes: &[RenderNode],
        pools: Pools<'_>,
        flat_ids: &mut Vec<byard_core::atlas::layout::AtlasNodeId>,
    ) -> Vec<byard_core::atlas::layout::AtlasNodeId> {
        // The expansion refs borrow `nodes`/`pools`, never `self`, so `&mut self`
        // stays free for `build_layout_tree` below.
        let concrete = self.expand_concrete(nodes, pools);
        let mut ids = Vec::with_capacity(concrete.len());
        for node in concrete {
            if let Ok(id) = self.build_layout_tree(node, pools, flat_ids) {
                ids.push(id);
            }
        }
        ids
    }

    /// The number of flattened layout nodes a concrete [`RenderNode`] subtree
    /// contributes, mirroring [`build_children`](Self::build_children)/
    /// [`build_layout_tree`](Self::build_layout_tree) exactly (one entry plus its
    /// expanded children). Used to advance the flat-id cursor past a culled
    /// `ScrollView` child without walking it (RFC-0005). `when`/`for` are
    /// expanded, so their live subtree is counted (RFC-0018).
    fn flat_len(&self, node: &RenderNode, pools: Pools<'_>) -> usize {
        match node {
            RenderNode::Box { children, .. } => {
                1 + self
                    .expand_concrete(children, pools)
                    .iter()
                    .map(|c| self.flat_len(c, pools))
                    .sum::<usize>()
            }
            _ => 1,
        }
    }

    /// Collects every mounted `Overlay` in `nodes` in pre-order (RFC-0017 mount =
    /// declaration order), expanding reactive `when`/`for` (RFC-0018) so an
    /// overlay inside a live branch/body is found. Recurses through `Box` and an
    /// overlay's own children, so a nested overlay is collected as its own later
    /// — hence higher — stack entry.
    fn collect_overlays<'a>(
        &self,
        nodes: &'a [RenderNode],
        pools: Pools<'a>,
        out: &mut Vec<&'a RenderNode>,
    ) {
        for node in self.expand_concrete(nodes, pools) {
            match node {
                RenderNode::Overlay { children, .. } => {
                    out.push(node);
                    self.collect_overlays(children, pools, out);
                }
                RenderNode::Box { children, .. } => {
                    self.collect_overlays(children, pools, out);
                }
                _ => {}
            }
        }
    }

    /// Builds one `Overlay`'s layout into the atlas (RFC-0017): each child is
    /// laid out at its natural size, then wrapped in an absolute, inset-0
    /// container whose `justify`/`align` realise the child's `anchor` within the
    /// viewport. All the anchor wrappers hang off one absolute overlay wrapper.
    /// Returns the wrapper id and per-child emission slots. `None` if `ov` is not
    /// an `Overlay` or the atlas rejects the nodes.
    fn build_overlay_layout<'a>(
        &mut self,
        ov: &'a RenderNode,
        pools: Pools<'a>,
    ) -> Option<OverlayLayout<'a>> {
        let RenderNode::Overlay { children, .. } = ov else {
            return None;
        };
        // RFC-0018: an overlay's direct children may be reactive `when`/`for`;
        // expand them to concrete anchor targets before laying each out.
        let concrete = self.expand_concrete(children, pools);
        let mut anchor_ids = Vec::with_capacity(concrete.len());
        let mut slots = Vec::with_capacity(concrete.len());
        for child in concrete {
            let mut cflat = Vec::new();
            let Ok(cid) = self.build_layout_tree(child, pools, &mut cflat) else {
                continue;
            };
            let anchor = self.anchor_token(child);
            let style = anchor_wrapper_style(anchor.as_deref());
            let Ok(anchor_id) = self.atlas.add_container(style, &[cid]) else {
                continue;
            };
            anchor_ids.push(anchor_id);
            slots.push(OverlayChildSlot {
                node: child,
                id: cid,
                flat_ids: cflat,
            });
        }
        let wrapper_style =
            byard_core::atlas::layout::ContainerStyle::default().with_absolute(true);
        let wrapper_id = self.atlas.add_container(wrapper_style, &anchor_ids).ok()?;
        Some(OverlayLayout {
            node: ov,
            wrapper_id,
            children: slots,
        })
    }

    /// The `anchor:` token of an overlay child (RFC-0017), or `None` for an
    /// unanchored child (a scrim, which fills the viewport via `grow`).
    fn anchor_token(&mut self, child: &RenderNode) -> Option<String> {
        match child {
            RenderNode::Box { attrs, .. } => Self::enum_prop(attrs, "anchor").map(str::to_string),
            _ => None,
        }
    }

    /// Emits one overlay's children on top of the main scene (RFC-0017 overlay
    /// phase). Clips them to the viewport, installs a modal scrim first when
    /// `modal` (the input barrier + `dismiss` target), then walks each child
    /// through the ordinary render path so it uses every existing pipeline.
    fn emit_overlay(
        &mut self,
        ol: &OverlayLayout<'_>,
        frame: &mut byard_core::frame::RenderFrame,
        width: f32,
        height: f32,
        pools: Pools<'_>,
    ) {
        let RenderNode::Overlay {
            attrs,
            env_snapshot,
            ..
        } = ol.node
        else {
            return;
        };
        let viewport = crate::interp::intrinsics::Rect::new(0.0, 0.0, width, height);
        // `modal` defaults true (RFC-0017 §Modality); `dismiss_on_outside`
        // defaults to whatever `modal` is.
        let modal = self.eval_bool_prop(attrs, "modal").unwrap_or(true);
        let dismiss_on_outside = self
            .eval_bool_prop(attrs, "dismiss_on_outside")
            .unwrap_or(modal);

        // Clamp everything the overlay paints to the viewport (RFC-0017
        // resolved-question: scissor interaction).
        frame.begin_clip(byard_core::frame::Rect::new(0.0, 0.0, width, height));

        // A modal overlay installs its scrim *before* its content so the content
        // wins hit-testing where it overlaps, while the scrim blocks (and
        // optionally dismisses) everything beneath the overlay.
        if modal {
            // Restore the overlay's instance environment so a `dismiss` action
            // referencing an instance `var`/param resolves correctly (RFC-0019).
            let env_base = self.env.len();
            for (k, v) in env_snapshot {
                self.env.push(k.clone(), v.clone());
            }
            let dismiss = if dismiss_on_outside {
                self.lower_overlay_dismiss(attrs)
            } else {
                None
            };
            self.env.truncate(env_base);
            let elem = self.atlas.node_index(ol.wrapper_id).unwrap_or(u32::MAX);
            self.router.push_modal_scrim(elem, viewport, dismiss);
        }

        for slot in &ol.children {
            let mut flat_idx = 0;
            self.render_node_with_atlas(
                slot.node,
                slot.id,
                frame,
                &slot.flat_ids,
                &mut flat_idx,
                viewport,
                1.0,
                byard_core::frame::Transform::IDENTITY,
                None,
                None,
                pools,
            );
        }

        frame.end_clip();
    }

    /// Lowers an `Overlay`'s `dismiss =>` action to a router [`Action`], if
    /// present (RFC-0017 §Dismissal). The action runs on scrim tap and on
    /// `Escape`.
    ///
    /// [`Action`]: super::events::Action
    fn lower_overlay_dismiss(&mut self, attrs: &[Attr]) -> Option<super::events::Action> {
        for attr in attrs {
            if attr.name.as_str() == "dismiss" {
                if let AttrKind::Event { payload, action } = &attr.kind {
                    return self.lower_action(action, payload.clone()).ok();
                }
            }
        }
        None
    }

    // ── Canvas shape lowering (RFC-0020) ────────────────────────────────────

    /// The named argument `name` of a shape command, if present.
    fn shape_arg<'e>(el: &'e ElementNode, name: &str) -> Option<&'e Expr> {
        el.content
            .iter()
            .find(|a| a.name.as_ref().is_some_and(|n| n.as_str() == name))
            .map(|a| &a.value)
    }

    /// Evaluates a numeric shape parameter (reactive + `with`-animatable via
    /// `eval_pure`'s animation chokepoint, RFC-0010).
    fn shape_num(&mut self, el: &ElementNode, name: &str) -> Option<f32> {
        Self::shape_arg(el, name).map(|e| {
            let e = e.clone();
            self.eval_num(&e)
        })
    }

    /// Evaluates a shape color parameter. The alpha byte is auto-detected:
    /// values above `0xFFFFFF` are `0xAARRGGBB`, else `0xRRGGBB` at full
    /// alpha — matching how shadow colors already read (RFC-0011).
    fn shape_color(&mut self, el: &ElementNode, name: &str) -> Option<[f32; 4]> {
        let e = Self::shape_arg(el, name)?.clone();
        let packed = self.eval_pure(&e).as_int()?;
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let has_alpha = (packed as u64) > 0x00FF_FFFF;
        Some(super::intrinsics::color_to_rgba(packed, has_alpha))
    }

    /// Evaluates a `(a, b)` shape parameter (the `dash` pattern).
    fn shape_vec2(&mut self, el: &ElementNode, name: &str) -> Option<[f32; 2]> {
        let e = Self::shape_arg(el, name)?.clone();
        match self.eval_pure(&e) {
            Value::Tuple(items) if items.len() == 2 => {
                #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                let f = |v: &Value| match v {
                    Value::Int(n) => *n as f32,
                    Value::Float(x) => *x as f32,
                    _ => 0.0,
                };
                Some([f(&items[0].1), f(&items[1].1)])
            }
            _ => None,
        }
    }

    /// Reads a bare-token shape parameter (`cap: round`) — an enum token is a
    /// syntactic identifier, never an env lookup, mirroring how `align:` and
    /// `fit:` tokens read elsewhere.
    fn shape_token(el: &ElementNode, name: &str) -> Option<String> {
        match Self::shape_arg(el, name)? {
            Expr::Ident(sym, _) | Expr::ClassRef(sym, _) => Some(sym.as_str().to_string()),
            _ => None,
        }
    }

    /// Evaluates a string shape parameter (`path`'s `d`).
    fn shape_string(&mut self, el: &ElementNode, name: &str) -> Option<String> {
        let e = Self::shape_arg(el, name)?.clone();
        match self.eval_pure(&e) {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Emits one shape command into the frame (RFC-0020). `canvas` is the
    /// canvas's resolved rect: shape coordinates are canvas-local and offset
    /// by its origin here. Tier 1 (`arc`/`circle`/`line`/`rect`, plus
    /// `bezier` flattened to line segments) goes to the `CanvasShape`
    /// pipeline; `path` rasterizes through `VectorMSDF` (Tier 2); `text`
    /// lowers to a `TextLine`.
    fn emit_canvas_shape(
        &mut self,
        el: &ElementNode,
        canvas: crate::interp::intrinsics::Rect,
        opacity: f32,
        transform: byard_core::frame::Transform,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        use byard_core::frame::{
            CANVAS_CAP_BUTT, CANVAS_CAP_ROUND, CANVAS_CAP_SQUARE, CANVAS_SHAPE_ARC,
            CANVAS_SHAPE_CIRCLE, CANVAS_SHAPE_LINE, CANVAS_SHAPE_RECT, CanvasShape,
        };

        let name = el.name.as_str();
        if name == "text" {
            self.emit_canvas_text(el, canvas, opacity, transform, frame);
            return;
        }
        if name == "path" {
            self.emit_canvas_path(el, canvas, opacity, transform, frame);
            return;
        }

        // Shared paint parameters (RFC-0020 §"Stroke and fill"). A shape with
        // neither stroke nor fill paints nothing — skip it entirely.
        let stroke_color = self.shape_color(el, "stroke").unwrap_or([0.0; 4]);
        let fill_color = self.shape_color(el, "fill").unwrap_or([0.0; 4]);
        if stroke_color[3] <= 0.0 && fill_color[3] <= 0.0 {
            return;
        }
        let stroke_width = self.shape_num(el, "stroke_width").unwrap_or(1.0);
        let cap = match Self::shape_token(el, "cap").as_deref() {
            Some("round") => CANVAS_CAP_ROUND,
            Some("square") => CANVAS_CAP_SQUARE,
            _ => CANVAS_CAP_BUTT,
        };
        let dash = self.shape_vec2(el, "dash").unwrap_or([0.0, 0.0]);
        let dash_offset = self.shape_num(el, "dash_offset").unwrap_or(0.0);
        let shape_opacity = opacity * self.shape_num(el, "opacity").unwrap_or(1.0);
        let (ox, oy) = (canvas.x, canvas.y);

        let base = CanvasShape {
            kind: CANVAS_SHAPE_CIRCLE,
            params: [0.0; 8],
            stroke_color,
            fill_color,
            stroke_width,
            cap,
            dash,
            dash_offset,
            opacity: shape_opacity,
            transform,
            dirty: true,
        };

        match name {
            "arc" | "circle" => {
                let cx = ox + self.shape_num(el, "cx").unwrap_or(0.0);
                let cy = oy + self.shape_num(el, "cy").unwrap_or(0.0);
                let r = self.shape_num(el, "r").unwrap_or(0.0);
                // Angles are authored in degrees (RFC-0020 examples:
                // `start: -90, sweep: 270`); the GPU wants radians. An
                // unswept `arc` defaults to a full circle — `circle` is the
                // explicit sugar for exactly that (RFC-0020 §"Shape commands").
                let start = self.shape_num(el, "start").unwrap_or(0.0);
                let sweep = if name == "circle" {
                    360.0
                } else {
                    self.shape_num(el, "sweep").unwrap_or(360.0)
                };
                let full = sweep.abs() >= 360.0;
                frame.push_canvas_shape(CanvasShape {
                    kind: if full {
                        CANVAS_SHAPE_CIRCLE
                    } else {
                        CANVAS_SHAPE_ARC
                    },
                    params: [
                        cx,
                        cy,
                        r,
                        start.to_radians(),
                        sweep.to_radians(),
                        0.0,
                        0.0,
                        0.0,
                    ],
                    ..base
                });
            }
            "line" => {
                frame.push_canvas_shape(CanvasShape {
                    kind: CANVAS_SHAPE_LINE,
                    params: [
                        ox + self.shape_num(el, "x1").unwrap_or(0.0),
                        oy + self.shape_num(el, "y1").unwrap_or(0.0),
                        ox + self.shape_num(el, "x2").unwrap_or(0.0),
                        oy + self.shape_num(el, "y2").unwrap_or(0.0),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    ..base
                });
            }
            "rect" => {
                frame.push_canvas_shape(CanvasShape {
                    kind: CANVAS_SHAPE_RECT,
                    params: [
                        ox + self.shape_num(el, "x").unwrap_or(0.0),
                        oy + self.shape_num(el, "y").unwrap_or(0.0),
                        self.shape_num(el, "w").unwrap_or(0.0),
                        self.shape_num(el, "h").unwrap_or(0.0),
                        self.shape_num(el, "radius").unwrap_or(0.0),
                        0.0,
                        0.0,
                        0.0,
                    ],
                    ..base
                });
            }
            "bezier" => {
                // Flattened CPU-side into round-capped line segments on the
                // same Tier-1 pipeline — cheaper and *fully animatable*,
                // unlike an MSDF re-rasterization (see the RFC-0020 notes in
                // the design record). Round caps hide the joints; the curve
                // has no fill.
                if let Some(c) = self.bezier_coords(el) {
                    let p = |t: f32| -> [f32; 2] {
                        let u = 1.0 - t;
                        let b0 = u * u * u;
                        let b1 = 3.0 * u * u * t;
                        let b2 = 3.0 * u * t * t;
                        let b3 = t * t * t;
                        [
                            ox + b0 * c[0] + b1 * c[2] + b2 * c[4] + b3 * c[6],
                            oy + b0 * c[1] + b1 * c[3] + b2 * c[5] + b3 * c[7],
                        ]
                    };
                    // Segment count scales with the control polygon's length:
                    // ~one segment per 6 logical px, clamped to [8, 48].
                    let poly_len = ((c[2] - c[0]).hypot(c[3] - c[1])
                        + (c[4] - c[2]).hypot(c[5] - c[3])
                        + (c[6] - c[4]).hypot(c[7] - c[5]))
                    .max(1.0);
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let segments = ((poly_len / 6.0) as usize).clamp(8, 48);
                    let mut prev = p(0.0);
                    #[allow(clippy::cast_precision_loss)]
                    for i in 1..=segments {
                        let next = p(i as f32 / segments as f32);
                        frame.push_canvas_shape(CanvasShape {
                            kind: CANVAS_SHAPE_LINE,
                            params: [prev[0], prev[1], next[0], next[1], 0.0, 0.0, 0.0, 0.0],
                            cap: CANVAS_CAP_ROUND,
                            fill_color: [0.0; 4],
                            ..base.clone()
                        });
                        prev = next;
                    }
                }
            }
            _ => {}
        }
    }

    /// The 8 cubic-bezier coordinates, from either the terse positional form
    /// (`bezier(10, 90, 40, 10, …)`) or the named form (`x1:`, `cy2:`, …).
    fn bezier_coords(&mut self, el: &ElementNode) -> Option<[f32; 8]> {
        const NAMES: [&str; 8] = ["x1", "y1", "cx1", "cy1", "cx2", "cy2", "x2", "y2"];
        let positional: Vec<Expr> = el
            .content
            .iter()
            .filter(|a| a.name.is_none())
            .map(|a| a.value.clone())
            .collect();
        let mut out = [0.0f32; 8];
        if positional.len() == 8 {
            for (slot, expr) in out.iter_mut().zip(&positional) {
                *slot = self.eval_num(expr);
            }
            return Some(out);
        }
        for (slot, name) in out.iter_mut().zip(NAMES) {
            *slot = self.shape_num(el, name)?;
        }
        Some(out)
    }

    /// RFC-0020 §2 Tier 2: a `path(d: …)` command rasterized through the
    /// MSDF pipeline. The synthetic SVG's viewBox equals the canvas size, so
    /// `d` coordinates are canvas-local 1:1; the resulting glyph is drawn
    /// over the whole canvas rect and tinted by `fill`. Content-keyed, so a
    /// re-render of an unchanged path is a pure cache hit; only a genuinely
    /// new `d` (or canvas size) dispatches a generation.
    fn emit_canvas_path(
        &mut self,
        el: &ElementNode,
        canvas: crate::interp::intrinsics::Rect,
        opacity: f32,
        transform: byard_core::frame::Transform,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        let Some(fill) = self.shape_color(el, "fill") else {
            return; // no fill → nothing to rasterize (stroke is rejected upstream)
        };
        let Some(d) = self.shape_string(el, "d") else {
            return;
        };
        let shape_opacity = opacity * self.shape_num(el, "opacity").unwrap_or(1.0);
        let (w, h) = (canvas.w.max(1.0), canvas.h.max(1.0));

        let key = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            d.hash(&mut hasher);
            w.to_bits().hash(&mut hasher);
            h.to_bits().hash(&mut hasher);
            format!("canvas-path:{:016x}", hasher.finish())
        };
        let glyph = self.vector_jit.lookup_or_dispatch_svg(&key, || {
            format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path d="{d}" fill="#000000"/></svg>"##
            )
            .into_bytes()
        });
        // Cache miss: skip this tick (INV-9 — the frame ships without
        // stalling); the generated field lands via the ordinary JIT drain.
        let Some(glyph) = glyph else { return };

        // A `VectorInstance` carries no transform: bake translate/scale into
        // the rect like `Image` does (rotation stays a box-primitive feature).
        let tl = transform.apply_point([canvas.x, canvas.y]);
        let rgb = [fill[0], fill[1], fill[2], fill[3] * shape_opacity];
        frame.push_vector(byard_core::frame::VectorInstance::new(
            byard_core::frame::Rect::new(
                tl[0],
                tl[1],
                canvas.w * transform.scale[0],
                canvas.h * transform.scale[1],
            ),
            glyph.uv_rect,
            rgb,
            glyph.px_range,
            glyph.layer,
        ));
    }

    /// A canvas `text(…)` command: a `TextLine` anchored at `(x, y)` with
    /// optional `align` (start/center/end around `x`) — `y` is the vertical
    /// center of the run, matching the RFC's centred-label example.
    fn emit_canvas_text(
        &mut self,
        el: &ElementNode,
        canvas: crate::interp::intrinsics::Rect,
        opacity: f32,
        transform: byard_core::frame::Transform,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        let Some(content) = el.content.iter().find(|a| a.name.is_none()) else {
            return;
        };
        let expr = content.value.clone();
        let text = match self.eval_pure(&expr) {
            Value::Str(s) => s,
            other => format!("{other:?}"),
        };
        if text.is_empty() {
            return;
        }
        let size = self.shape_num(el, "size").unwrap_or(self.theme.font_size);
        let color = self
            .shape_color(el, "color")
            .unwrap_or_else(|| super::intrinsics::color_to_rgba(self.theme.on_surface(), false));
        let x = canvas.x + self.shape_num(el, "x").unwrap_or(0.0);
        let y = canvas.y + self.shape_num(el, "y").unwrap_or(0.0);
        let measured = self.measure_text(&text, size).0;
        let tx = match Self::shape_token(el, "align").as_deref() {
            Some("center") => x - measured / 2.0,
            Some("end") => x - measured,
            _ => x,
        };
        // Anchor `y` at the run's vertical center (≈0.6em above the top of
        // the em box reads optically centred for Latin text).
        let ty = y - size * 0.6;
        let anchor = transform.apply_point([tx, ty]);
        frame.push_text(byard_core::TextLine {
            x: anchor[0],
            y: anchor[1],
            text,
            font_size: size * transform.uniform_scale(),
            color: dim_alpha(color, opacity),
            dirty: true,
        });
    }

    fn build_layout_tree(
        &mut self,
        node: &RenderNode,
        pools: Pools<'_>,
        flat_ids: &mut Vec<byard_core::atlas::layout::AtlasNodeId>,
    ) -> Result<byard_core::atlas::layout::AtlasNodeId, byard_core::atlas::AtlasError> {
        use byard_core::atlas::layout::LeafSize;
        match node {
            // Reactive `when`/`for` are expanded to their concrete children by
            // `build_children` before reaching here (RFC-0018), so they never
            // arrive as a single layout node.
            RenderNode::When { .. } | RenderNode::For { .. } => {
                unreachable!("when/for are expanded by build_children before build_layout_tree")
            }
            RenderNode::Spacer => {
                let id = self.atlas.add_leaf(LeafSize::new(0.0, 12.0))?;
                flat_ids.push(id);
                Ok(id)
            }
            // RFC-0017: an `Overlay` occupies zero space in its parent's flow —
            // its children are laid out separately against the viewport in the
            // deferred overlay phase. A 0×0 leaf keeps the parallel flat-id
            // cursor aligned without displacing any sibling.
            RenderNode::Overlay { .. } => {
                let id = self.atlas.add_leaf(LeafSize::new(0.0, 0.0))?;
                flat_ids.push(id);
                Ok(id)
            }
            RenderNode::Image { attrs, .. } => {
                let w = self.eval_int_prop(attrs, "width").unwrap_or(100) as f32;
                let h = self.eval_int_prop(attrs, "height").unwrap_or(100) as f32;
                let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                flat_ids.push(id);
                Ok(id)
            }
            // A `Canvas` is a fixed-size drawing surface (RFC-0020 §1): both
            // dimensions are required (enforced by `validate_canvas`); the 0
            // fallback keeps a mis-declared canvas laid out (collapsed) rather
            // than aborting the whole tree.
            RenderNode::Canvas { attrs, .. } => {
                let w = self.eval_int_prop(attrs, "width").unwrap_or(0) as f32;
                let h = self.eval_int_prop(attrs, "height").unwrap_or(0) as f32;
                let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                flat_ids.push(id);
                Ok(id)
            }
            // A VectorIcon is a square leaf sized by its `size` prop (default 24),
            // RFC-0009 §1.
            RenderNode::Vector { attrs, .. } => {
                let s = self.eval_int_prop(attrs, "size").unwrap_or(24) as f32;
                let id = self.atlas.add_leaf(LeafSize::new(s, s))?;
                flat_ids.push(id);
                Ok(id)
            }
            RenderNode::Text { attrs, content, .. } => {
                let text = match self.binding_value(*content) {
                    Some(Value::Str(s)) => s,
                    other => other.map_or_else(String::new, |v| format!("{v:?}")),
                };
                let typo_size = self.eval_typo_size(attrs);
                #[allow(clippy::cast_precision_loss)]
                let font_size = self
                    .eval_int_prop(attrs, "size")
                    .or(typo_size)
                    .unwrap_or(self.theme.font_size as i64) as f32;
                // RFC-0005 default text wrap: `wrap` defaults to `true`. A
                // wrapping `Text` becomes a measured leaf that the atlas sizes to
                // the width its parent offers during layout (via the shared
                // `TextMeasurer`), so it reflows without an explicit `width`. An
                // explicit `width` fixes the wrap width; `wrap: false` opts out to
                // a fixed natural single-line leaf (may overflow — the caller's
                // choice). `fallback` is the natural size for the no-sizer path.
                let (nat_w, nat_h) = self.measure_text_wrapped(&text, font_size, None);
                if self.eval_bool_prop(attrs, "wrap") == Some(false) {
                    let id = self.atlas.add_leaf(LeafSize::new(nat_w, nat_h))?;
                    flat_ids.push(id);
                    return Ok(id);
                }
                #[allow(clippy::cast_precision_loss)]
                let explicit_w = self.eval_int_prop(attrs, "width").map(|w| w as f32);
                let id = self.atlas.add_text_leaf(byard_core::atlas::TextLeaf {
                    content: text,
                    font_size,
                    width: explicit_w,
                    fallback: (nat_w, nat_h),
                })?;
                flat_ids.push(id);
                Ok(id)
            }
            RenderNode::Box {
                name,
                attrs,
                children,
                ..
            } => {
                // Value widgets are leaf nodes with intrinsic default sizes (M16/M19).
                match name.as_str() {
                    "Toggle" => {
                        let w = self.eval_int_prop(attrs, "width").unwrap_or(50) as f32;
                        let h = self.eval_int_prop(attrs, "height").unwrap_or(30) as f32;
                        let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                        flat_ids.push(id);
                        return Ok(id);
                    }
                    "Slider" => {
                        let w = self.eval_int_prop(attrs, "width").unwrap_or(200) as f32;
                        let h = self.eval_int_prop(attrs, "height").unwrap_or(24) as f32;
                        let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                        flat_ids.push(id);
                        return Ok(id);
                    }
                    // RFC-0018 Checkbox: an 18×18 square by default; `width`/
                    // `height` override. Sized square unless overridden so the
                    // container and checkmark geometry stay proportional.
                    "Checkbox" => {
                        let w = self.eval_int_prop(attrs, "width").unwrap_or(18) as f32;
                        let h = self.eval_int_prop(attrs, "height").unwrap_or(18) as f32;
                        let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                        flat_ids.push(id);
                        return Ok(id);
                    }
                    // RFC-0018 RadioButton: a 20×20 circle by default.
                    "RadioButton" => {
                        let w = self.eval_int_prop(attrs, "width").unwrap_or(20) as f32;
                        let h = self.eval_int_prop(attrs, "height").unwrap_or(20) as f32;
                        let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                        flat_ids.push(id);
                        return Ok(id);
                    }
                    "TextField" => {
                        let w = self.eval_int_prop(attrs, "width").unwrap_or(200) as f32;
                        let h = self.eval_int_prop(attrs, "height").unwrap_or(36) as f32;
                        let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                        flat_ids.push(id);
                        return Ok(id);
                    }
                    _ => {}
                }
                // RFC-0005 windowed ScrollView: when opted in, build only the
                // visible slice of a single uniform-height list child, bracketed
                // by spacer leaves for the elided rows — so layout is O(visible),
                // not O(list). The same window is recomputed in the render pass.
                if name.as_str() == "ScrollView" {
                    if let [
                        RenderNode::Box {
                            name: list_name,
                            attrs: list_attrs,
                            children: rows_raw,
                            ..
                        },
                    ] = children.as_slice()
                    {
                        // RFC-0018: expand a reactive `for` (or literal rows) to
                        // the concrete row nodes, then window over them — so
                        // virtualization still lays out only the visible slice.
                        let rows = self.expand_concrete(rows_raw, pools);
                        if let Some(win) = self.scroll_window(attrs, rows.len()) {
                            let mut temp_flat = Vec::new();
                            let list_id = self.build_windowed_list(
                                list_name,
                                list_attrs,
                                &rows,
                                win,
                                pools,
                                &mut temp_flat,
                            )?;
                            let style = self.eval_container_style(name.as_str(), attrs);
                            let id = self.atlas.add_container(style, &[list_id])?;
                            flat_ids.push(id);
                            flat_ids.extend(temp_flat);
                            return Ok(id);
                        }
                    }
                }
                // RFC-0018 `Grid`: a CSS-grid container. Built via its own path so
                // the parent gets grid tracks/gaps and each child can carry an
                // explicit placement.
                if name.as_str() == "Grid" {
                    return self.build_grid(attrs, children, pools, flat_ids);
                }
                // RFC-0018 `ZStack`: overlapping children — a single-cell grid.
                if name.as_str() == "ZStack" {
                    return self.build_zstack(attrs, children, pools, flat_ids);
                }
                let mut temp_flat = Vec::new();
                // RFC-0018: expand reactive `when`/`for` children before layout.
                let child_ids = self.build_children(children, pools, &mut temp_flat);
                let style = self.eval_container_style(name.as_str(), attrs);
                let id = self.atlas.add_container(style, &child_ids)?;
                flat_ids.push(id);
                flat_ids.extend(temp_flat);
                Ok(id)
            }
        }
    }

    /// Builds the atlas subtree for a `Grid` (RFC-0018). Children are expanded
    /// (reactive `when`/`for` first), each built and — if it carries `col`/`row`/
    /// `col_span`/`row_span` — placed explicitly; the rest auto-place. The
    /// container is emitted in the same parent-then-children `flat_ids` order the
    /// generic container path uses, so the render walk's parallel cursor stays
    /// aligned.
    fn build_grid(
        &mut self,
        attrs: &[Attr],
        children: &[RenderNode],
        pools: Pools<'_>,
        flat_ids: &mut Vec<byard_core::atlas::layout::AtlasNodeId>,
    ) -> Result<byard_core::atlas::layout::AtlasNodeId, byard_core::atlas::AtlasError> {
        let concrete = self.expand_concrete(children, pools);
        let mut child_ids = Vec::with_capacity(concrete.len());
        let mut placements: Vec<(
            byard_core::atlas::layout::AtlasNodeId,
            byard_core::atlas::GridItemPlacement,
        )> = Vec::new();
        let mut temp_flat = Vec::new();
        for child in concrete {
            if let Ok(cid) = self.build_layout_tree(child, pools, &mut temp_flat) {
                if let Some(p) = self.grid_child_placement(child) {
                    placements.push((cid, p));
                }
                child_ids.push(cid);
            }
        }
        let base = self.eval_container_style("Grid", attrs);
        let (cols, rows) = self.eval_grid_templates(attrs);
        let (col_gap, row_gap) = self.eval_grid_gaps(attrs);
        let id = self
            .atlas
            .add_grid_container(base, &cols, &rows, col_gap, row_gap, &child_ids)?;
        for (cid, p) in placements {
            // A rejected placement (e.g. a foreign node) is non-fatal — the child
            // simply auto-places — so it never aborts the frame.
            let _ = self.atlas.set_grid_item(cid, p);
        }
        flat_ids.push(id);
        flat_ids.extend(temp_flat);
        Ok(id)
    }

    /// Reads a grid child's explicit placement (`col`/`row`/`col_span`/
    /// `row_span`) from its attrs, or `None` if it carries none (→ auto-placed).
    /// Only `Box`-family children (Box/Column/Row/Grid/…) carry placement.
    fn grid_child_placement(
        &mut self,
        child: &RenderNode,
    ) -> Option<byard_core::atlas::GridItemPlacement> {
        let RenderNode::Box { attrs, .. } = child else {
            return None;
        };
        let col = self.eval_int_prop(attrs, "col");
        let row = self.eval_int_prop(attrs, "row");
        let col_span = self.eval_int_prop(attrs, "col_span");
        let row_span = self.eval_int_prop(attrs, "row_span");
        if col.is_none() && row.is_none() && col_span.is_none() && row_span.is_none() {
            return None;
        }
        Some(byard_core::atlas::GridItemPlacement {
            col_start: col.and_then(|n| i16::try_from(n).ok()),
            col_span: col_span
                .and_then(|n| u16::try_from(n.max(1)).ok())
                .unwrap_or(1),
            row_start: row.and_then(|n| i16::try_from(n).ok()),
            row_span: row_span
                .and_then(|n| u16::try_from(n.max(1)).ok())
                .unwrap_or(1),
        })
    }

    /// Parses a `Grid`'s `columns:`/`rows:` templates (RFC-0018). A missing or
    /// malformed `columns` defaults to a single `auto` track (one column);
    /// missing `rows` leaves rows implicit (Taffy auto-creates them). A malformed
    /// template also pushes a [`CompileError::InvalidGridTemplate`].
    ///
    /// [`CompileError::InvalidGridTemplate`]: crate::diagnostics::CompileError::InvalidGridTemplate
    fn eval_grid_templates(
        &mut self,
        attrs: &[Attr],
    ) -> (
        Vec<byard_core::atlas::GridTrack>,
        Vec<byard_core::atlas::GridTrack>,
    ) {
        let cols = self
            .eval_grid_axis(attrs, "columns")
            .unwrap_or_else(|| vec![byard_core::atlas::GridTrack::Auto]);
        let rows = self.eval_grid_axis(attrs, "rows").unwrap_or_default();
        (cols, rows)
    }

    /// Resolves one grid-template axis attribute to tracks, pushing an
    /// `InvalidGridTemplate` diagnostic on a malformed string. `None` = the
    /// attribute is absent (or not a string).
    fn eval_grid_axis(
        &mut self,
        attrs: &[Attr],
        name: &str,
    ) -> Option<Vec<byard_core::atlas::GridTrack>> {
        let attr = attrs.iter().find(|a| a.name.as_str() == name)?;
        let AttrKind::Prop { value } = &attr.kind else {
            return None;
        };
        let Value::Str(s) = self.eval_pure(value) else {
            return None;
        };
        let parsed = super::intrinsics::parse_grid_template(&s);
        if parsed.is_none() {
            self.errors.push(CompileError::InvalidGridTemplate {
                span: attr.span,
                template: s,
            });
        }
        parsed
    }

    /// Resolves a `Grid`'s per-axis gaps: `col_gap`/`row_gap` each fall back to
    /// the shared `gap` (default 0). Returns `(col_gap, row_gap)`.
    fn eval_grid_gaps(&mut self, attrs: &[Attr]) -> (f32, f32) {
        let gap = self.eval_int_prop(attrs, "gap").map_or(0.0, |n| n as f32);
        let col_gap = self
            .eval_int_prop(attrs, "col_gap")
            .map_or(gap, |n| n as f32);
        let row_gap = self
            .eval_int_prop(attrs, "row_gap")
            .map_or(gap, |n| n as f32);
        (col_gap, row_gap)
    }

    /// Builds the atlas subtree for a `ZStack` (RFC-0018): a single-cell grid in
    /// which every child overlaps. Emitted in the same parent-then-children
    /// `flat_ids` order as the generic container path so the render walk's cursor
    /// stays aligned, and rendered through the ordinary Box paint path (bg +
    /// children in declaration order, last on top).
    fn build_zstack(
        &mut self,
        attrs: &[Attr],
        children: &[RenderNode],
        pools: Pools<'_>,
        flat_ids: &mut Vec<byard_core::atlas::layout::AtlasNodeId>,
    ) -> Result<byard_core::atlas::layout::AtlasNodeId, byard_core::atlas::AtlasError> {
        let concrete = self.expand_concrete(children, pools);
        let mut child_ids = Vec::with_capacity(concrete.len());
        let mut temp_flat = Vec::new();
        for child in concrete {
            if let Ok(cid) = self.build_layout_tree(child, pools, &mut temp_flat) {
                child_ids.push(cid);
            }
        }
        let base = self.eval_container_style("ZStack", attrs);
        let align = self.eval_stack_align(attrs);
        let id = self.atlas.add_stack_container(base, align, &child_ids)?;
        flat_ids.push(id);
        flat_ids.extend(temp_flat);
        Ok(id)
    }

    /// Resolves a `ZStack`'s `alignment` prop to a [`StackAlign`], default
    /// `Center`.
    ///
    /// [`StackAlign`]: byard_core::atlas::StackAlign
    fn eval_stack_align(&mut self, attrs: &[Attr]) -> byard_core::atlas::StackAlign {
        use byard_core::atlas::StackAlign;
        for attr in attrs {
            if attr.name.as_str() != "alignment" {
                continue;
            }
            let AttrKind::Prop { value } = &attr.kind else {
                continue;
            };
            if let Some(s) = Self::enum_token(value) {
                return match s {
                    "top_start" => StackAlign::TopStart,
                    "top_end" => StackAlign::TopEnd,
                    "bottom_start" => StackAlign::BottomStart,
                    "bottom_end" => StackAlign::BottomEnd,
                    "top" => StackAlign::Top,
                    "bottom" => StackAlign::Bottom,
                    "start" => StackAlign::Start,
                    "end" => StackAlign::End,
                    _ => StackAlign::Center,
                };
            }
        }
        StackAlign::Center
    }

    /// Builds the atlas subtree for a windowed `ScrollView`'s list child
    /// (RFC-0005): a leading spacer sized to the rows scrolled off the top, the
    /// materialised rows `win.start..win.end`, then a trailing spacer for the
    /// rows below the window. The two spacers preserve the container's content
    /// extent (so the scroll clamp is exact) and every visible row's position,
    /// while only `end − start` rows are ever laid out. `flat_ids` receives the
    /// same `[container, top-spacer, rows…, bottom-spacer]` order the render pass
    /// walks, keeping the parallel cursor aligned.
    fn build_windowed_list(
        &mut self,
        list_name: &Symbol,
        list_attrs: &[Attr],
        rows: &[&RenderNode],
        win: WindowSpec,
        pools: Pools<'_>,
        flat_ids: &mut Vec<byard_core::atlas::layout::AtlasNodeId>,
    ) -> Result<byard_core::atlas::layout::AtlasNodeId, byard_core::atlas::AtlasError> {
        use byard_core::atlas::layout::LeafSize;
        #[allow(clippy::cast_precision_loss)]
        let top_h = win.start as f32 * win.row_height;
        #[allow(clippy::cast_precision_loss)]
        let bottom_h = (win.n - win.end) as f32 * win.row_height;

        let mut child_ids = Vec::with_capacity(win.end - win.start + 2);
        let mut temp = Vec::new();
        let top = self.atlas.add_leaf(LeafSize::new(0.0, top_h))?;
        temp.push(top);
        child_ids.push(top);
        for &row in &rows[win.start..win.end] {
            let id = self.build_layout_tree(row, pools, &mut temp)?;
            child_ids.push(id);
        }
        let bottom = self.atlas.add_leaf(LeafSize::new(0.0, bottom_h))?;
        temp.push(bottom);
        child_ids.push(bottom);

        let mut style = self.eval_container_style(list_name.as_str(), list_attrs);
        // Rows are positioned purely by `row_height` (spacing folded in); a flex
        // gap would add phantom space around the spacers and desync the window.
        style.gap = 0.0;
        let id = self.atlas.add_container(style, &child_ids)?;
        flat_ids.push(id);
        flat_ids.extend(temp);
        Ok(id)
    }

    #[allow(clippy::similar_names)]
    #[allow(clippy::too_many_arguments)]
    fn render_node_with_atlas(
        &mut self,
        node: &RenderNode,
        atlas_node: byard_core::atlas::layout::AtlasNodeId,
        frame: &mut byard_core::frame::RenderFrame,
        flat_ids: &[byard_core::atlas::layout::AtlasNodeId],
        flat_idx: &mut usize,
        parent_rect: crate::interp::intrinsics::Rect,
        // Opacity inherited from ancestors (RFC-0011 T4 approximation): folded
        // into this element's own `opacity` and multiplied into the alpha of
        // every primitive it emits, so a translucent parent dims its text and
        // widgets too — not only its own background.
        inherited_opacity: f32,
        // Paint-time transform inherited from ancestors (RFC-0011 group
        // transforms): composed with this element's own transform so a scaled or
        // translated container carries its children, text, and widgets with it —
        // not only its own background box. `IDENTITY` at the root.
        inherited_transform: byard_core::frame::Transform,
        // The nearest enclosing `ScrollView` viewport, in screen space (RFC-0005
        // emission culling). A node whose scroll-shifted rect falls entirely
        // outside it is skipped — the scissor already hides such fragments, so
        // this only spares the CPU the emission. `None` outside any scroll
        // container (the whole viewport is live).
        cull_clip: Option<byard_core::frame::Rect>,
        // Set only on a windowed `ScrollView`'s list child (RFC-0005 windowed
        // layout): this node renders just rows `start..end`, bracketed by the two
        // spacer leaves the build pass emitted, so the flat-id cursor stays
        // aligned. `None` everywhere else — the ordinary full child walk.
        window: Option<WindowSpec>,
        // RFC-0018: the reactive `for` pools, for expanding `when`/`for` children.
        pools: Pools<'_>,
    ) {
        debug_assert_eq!(flat_ids[*flat_idx], atlas_node);
        *flat_idx += 1;

        match node {
            // Reactive `when`/`for` are expanded to concrete children before the
            // walk reaches them (RFC-0018), so they never arrive as a paint node.
            RenderNode::When { .. } | RenderNode::For { .. } => {
                unreachable!("when/for are expanded before render_node_with_atlas")
            }
            // A `Spacer` is layout-only. An `Overlay` renders nothing in the main
            // flow — its 0×0 leaf holds a slot in the flat-id cursor (already
            // advanced above) while its children are emitted separately in the
            // deferred overlay phase (RFC-0017).
            RenderNode::Spacer | RenderNode::Overlay { .. } => {}
            RenderNode::Text {
                attrs,
                state_blocks,
                content,
            } => {
                if let Ok(Some(rect)) = self.atlas.resolved_rect(atlas_node) {
                    // RFC-0016: overlay any active `on <state>` block against the
                    // live engine mask before reading paint properties. RFC-0024:
                    // fold in the universal `selected:`/`invalid:` states.
                    let elem_idx = self.atlas.node_index(atlas_node);
                    let state = elem_idx
                        .map_or_else(crate::interp::events::StyleState::empty, |i| {
                            self.router.style_state(i)
                        })
                        .union(self.prop_style_state(attrs, None, ""));
                    let attrs = resolve_state_attrs(attrs, state_blocks, state);
                    let attrs = attrs.as_ref();
                    let text = match self.binding_value(*content) {
                        Some(Value::Str(s)) => s,
                        other => other.map_or_else(String::new, |v| format!("{v:?}")),
                    };
                    // M22: fall back to theme on-surface color when unset.
                    let color = self
                        .eval_color_prop(attrs, "color")
                        .unwrap_or(self.theme.on_surface());
                    // Resolve `typo:` token to font size; inline `size:` overrides
                    // (RFC-0005 `Typo`, completed by RFC-0022).
                    let typo_size = self.eval_typo_size(attrs);
                    let size =
                        self.eval_int_prop(attrs, "size")
                            .or(typo_size)
                            .unwrap_or(self.theme.font_size as i64) as f32;
                    let mut rgba = super::intrinsics::color_to_rgba(color, false);
                    rgba[3] *= inherited_opacity;
                    // RFC-0011 group transforms: a `Text` carries no transform of
                    // its own, so an ancestor's scale/translate is baked into the
                    // baseline anchor and the font size (glyph extents scale from
                    // the anchor, so this scales the run about the ancestor pivot).
                    // Rotation can't be baked per-glyph and is left to box
                    // primitives (shader-applied) — a documented limitation.
                    let anchor = inherited_transform.apply_point([rect.x, rect.y]);
                    let scaled_size = size * inherited_transform.uniform_scale();
                    // RFC-0005 default text wrap: shape the run to the width layout
                    // resolved for this leaf (its parent-offered width), scaled by
                    // any ancestor scale (the run's glyphs scale about the pivot).
                    // `wrap: false` opts out to a single-line run. This mirrors the
                    // atlas's measure pass, so the rendered line breaks match the
                    // laid-out height.
                    let wrap_w = if self.eval_bool_prop(attrs, "wrap") == Some(false) {
                        None
                    } else {
                        Some(rect.width * inherited_transform.uniform_scale())
                    };
                    frame.push_text_wrapped(
                        byard_core::TextLine {
                            x: anchor[0],
                            y: anchor[1],
                            text,
                            font_size: scaled_size,
                            color: rgba,
                            dirty: true,
                        },
                        wrap_w,
                    );

                    let has_events = attrs
                        .iter()
                        .any(|a| matches!(a.kind, AttrKind::Event { .. }));
                    if has_events {
                        let self_rect = crate::interp::intrinsics::Rect::new(
                            rect.x,
                            rect.y,
                            rect.width,
                            rect.height,
                        );
                        let hit_rect =
                            crate::interp::intrinsics::inflate_hit_rect(self_rect, parent_rect);
                        self.register_event_attrs(attrs, hit_rect, elem_idx);
                    }
                }
            }
            RenderNode::Box {
                name,
                attrs,
                state_blocks,
                children,
                action,
                bound_sig,
                env_snapshot,
            } => {
                // RFC-0019 §2: restore the instance environment captured at lower
                // time so event actions re-lowered below (a forwarded callback,
                // or any param reference in `attrs`) resolve against the scope
                // this box was instantiated in. Empty at the top level (no-op).
                let env_base = self.env.len();
                for (k, v) in env_snapshot {
                    self.env.push(k.clone(), v.clone());
                }
                let mut current_rect = parent_rect;
                // Opacity children inherit from this box: its effective opacity
                // when it has a resolved rect (set below), else whatever it
                // inherited unchanged.
                let mut child_opacity = inherited_opacity;
                // Likewise the composed paint transform children inherit (RFC-0011
                // group transforms): this box's own transform ∘ its ancestors',
                // set once the rect is known, else passed through unchanged.
                let mut child_transform = inherited_transform;
                if let Ok(Some(rect)) = self.atlas.resolved_rect(atlas_node) {
                    current_rect = crate::interp::intrinsics::Rect::new(
                        rect.x,
                        rect.y,
                        rect.width,
                        rect.height,
                    );
                    let elem_idx = self.atlas.node_index(atlas_node);
                    // RFC-0012 S5: a `disabled:` element still lays out and paints,
                    // but the router gates every handler it registers below and
                    // reports the `DISABLED` interaction state. Marked here, before
                    // resolving state styles, so an `on disabled { … }` block takes
                    // effect on the very frame the element becomes disabled.
                    if self.eval_bool_prop(attrs, "disabled") == Some(true) {
                        if let Some(idx) = elem_idx {
                            self.router.set_disabled(idx);
                        }
                    }
                    // RFC-0016: overlay any active `on <state>` block over the
                    // base attributes against the live engine `StyleState` mask
                    // *before* reading paint properties. Stateless boxes borrow
                    // `attrs` unchanged (no clone). The base `attrs` still drive
                    // event/handler registration below so hit targets are stable.
                    // RFC-0024: fold the prop/value-driven states (checked,
                    // selected, invalid, indeterminate) into the router's
                    // pointer/focus/drag mask so `on checked { … }` etc. resolve.
                    let paint_state = elem_idx
                        .map_or_else(crate::interp::events::StyleState::empty, |i| {
                            self.router.style_state(i)
                        })
                        .union(self.prop_style_state(attrs, *bound_sig, name.as_str()));
                    let paint_attrs = resolve_state_attrs(attrs, state_blocks, paint_state);
                    let paint_attrs = paint_attrs.as_ref();
                    // Resolve the paint-time transform once, up front, so it can
                    // be applied both to a plain container's `bg` fill *and* to
                    // the self-owned visuals of `Toggle`/`Slider`/`TextField`
                    // (their track/fill/thumb/underline/caret are the element's
                    // own quads, so RFC-0011's element-local transform applies to
                    // them exactly as it does to a `Box` fill).
                    // The element's own transform, then composed with the one
                    // inherited from its ancestors (RFC-0011 group transforms) so
                    // this box's fill, its widget visuals, its children, and its
                    // text all move/scale as a group. Passed on to children below.
                    let own_transform = self.resolve_transform(paint_attrs, current_rect);
                    let transform = inherited_transform.compose(&own_transform);
                    child_transform = transform;
                    let bg = self.eval_color_prop(paint_attrs, "bg");
                    let radii = self.resolve_radii(paint_attrs, "radius");
                    // `border` is a Color (catalog DECORATION); a present border
                    // draws a 2px ring of that colour.
                    let border_color = self.eval_color_prop(paint_attrs, "border");
                    // `border_width` is an animatable paint prop (RFC-0010): it
                    // resolves through `eval_pure`, so `border_width: n with
                    // anim.*` interpolates like any other scalar. Defaults to 2px
                    // when a border colour is present, 0 when there is no border.
                    let border_width = if border_color.is_some() {
                        self.eval_float_prop(paint_attrs, "border_width")
                            .map_or(2.0, |v| v as f32)
                    } else {
                        0.0
                    };
                    // `shadow` is a token (`sm`/`md`/`lg`) → an offset+blur drop
                    // shadow; any other non-empty value falls back to `md`.
                    // `shadow` (RFC-0011 custom shadows): a preset token, a
                    // single (named/positional) tuple, or an array of tuples for
                    // CSS-style layered shadows. Each becomes its own shadow-only
                    // decorated box beneath the surface.
                    let shadows = self.resolve_shadows(paint_attrs);
                    // The element's *effective* opacity: its own `opacity` prop
                    // folded with whatever it inherited (RFC-0011 T4). Used for
                    // this box's own fill and passed down so children (a Button's
                    // label, a widget's visuals) dim with it.
                    let opacity = inherited_opacity
                        * self
                            .eval_float_prop(paint_attrs, "opacity")
                            .map_or(1.0, |v| v as f32);
                    child_opacity = opacity;
                    let translucent = (opacity - 1.0).abs() > f32::EPSILON;
                    // `Toggle`/`Slider` own their visuals (track/fill/thumb) and
                    // treat `bg` as the *accent* colour, not a full-rect fill —
                    // painting the rect here would draw a slab behind the control.
                    let owns_visuals = matches!(
                        name.as_str(),
                        "Toggle" | "Slider" | "Checkbox" | "RadioButton"
                    );
                    if let (false, Some(color)) = (owns_visuals, bg) {
                        let base = byard_core::BoxInstance {
                            rect: [rect.x, rect.y, rect.width, rect.height],
                            color: super::intrinsics::color_to_rgba(color, false),
                            radii,
                            transform,
                        };
                        let border_rgba = border_color
                            .map_or([0.0; 4], |c| super::intrinsics::color_to_rgba(c, false));
                        // Cast the shadows first so they sit *beneath* the fill.
                        // Reversed: first-listed is pushed last → nearest z → on
                        // top of later shadows (CSS box-shadow order), all still
                        // behind the surface pushed after them.
                        for sh in shadows.iter().rev() {
                            frame.push_decorated(shadow_decorated(base, opacity, sh));
                        }
                        if translucent {
                            // A translucent box blends its fill as one unit on the
                            // decorated pipeline (leaf showcase boxes); keep it whole.
                            frame.push_decorated(byard_core::frame::DecoratedBox {
                                base,
                                border_width,
                                border_color: border_rgba,
                                opacity,
                                // Re-walked and re-emitted every tick;
                                // mirror Text's always-dirty lowering.
                                dirty: true,
                                ..Default::default()
                            });
                        } else if border_color.is_some() {
                            // Paint the opaque fill on the SolidBox pass so it stays
                            // *behind* this container's children (they also paint as
                            // solids, pushed after it — and the decorated pass runs
                            // after every solid). Then add the border as a decorated
                            // overlay whose interior is transparent: it only strokes
                            // the edge, so it can never occlude the children drawn
                            // beneath it (fixes the parent-card-over-child-widget
                            // z-order bug).
                            frame.push_instance(base);
                            frame.push_decorated(byard_core::frame::DecoratedBox {
                                base: byard_core::BoxInstance {
                                    color: [0.0; 4],
                                    ..base
                                },
                                border_width,
                                border_color: border_rgba,
                                opacity: 1.0,
                                dirty: true,
                                ..Default::default()
                            });
                        } else {
                            frame.push_instance(base);
                        }
                    }

                    let element_name = name.as_str();
                    let hit_rect =
                        crate::interp::intrinsics::inflate_hit_rect(current_rect, parent_rect);

                    // RFC-0016: an element that styles `on hover`/`on pressed` but
                    // registers no handler still needs the engine to track the
                    // pointer over it, so register a bare hover/press hit region.
                    if let Some(idx) = elem_idx {
                        if state_blocks.iter().any(|sb| {
                            sb.states.iter().any(|s| {
                                matches!(s, StyleStateKind::Hover | StyleStateKind::Pressed)
                            })
                        }) {
                            self.router.track_region(idx, hit_rect);
                        }
                    }

                    // ── Widget-specific visual lowering & handler registration (M16/M19) ──
                    match element_name {
                        "Toggle" => {
                            self.render_toggle(
                                *bound_sig,
                                paint_attrs,
                                current_rect,
                                hit_rect,
                                elem_idx,
                                transform,
                                opacity,
                                frame,
                            );
                        }
                        "Slider" => {
                            self.render_slider(
                                *bound_sig,
                                paint_attrs,
                                current_rect,
                                hit_rect,
                                elem_idx,
                                transform,
                                opacity,
                                frame,
                            );
                        }
                        "Checkbox" => {
                            self.render_checkbox(
                                *bound_sig,
                                paint_attrs,
                                current_rect,
                                hit_rect,
                                elem_idx,
                                transform,
                                opacity,
                                frame,
                            );
                        }
                        "RadioButton" => {
                            self.render_radio(
                                *bound_sig,
                                paint_attrs,
                                current_rect,
                                hit_rect,
                                elem_idx,
                                transform,
                                opacity,
                                frame,
                            );
                        }
                        "TextField" => {
                            self.render_text_field(
                                *bound_sig,
                                paint_attrs,
                                current_rect,
                                hit_rect,
                                elem_idx,
                                transform,
                                opacity,
                                frame,
                            );
                        }
                        _ => {
                            // General interactive elements: register event-attr handlers.
                            let has_event_attrs = attrs
                                .iter()
                                .any(|a| matches!(a.kind, AttrKind::Event { .. }));
                            let is_interactive = matches!(element_name, "Button")
                                || has_event_attrs
                                || action.is_some();

                            if is_interactive {
                                self.register_event_attrs(attrs, hit_rect, elem_idx);

                                if let Some(action_expr) = action {
                                    if let Ok(action_closure) = self.lower_action(action_expr, None)
                                    {
                                        if let Some(idx) = elem_idx {
                                            self.router.on(
                                                idx,
                                                hit_rect,
                                                crate::interp::events::EventKind::Tap,
                                                action_closure,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // ── `focused:` reflected prop → register as focusable (M16/M18) ──
                    // TextField, Checkbox, and RadioButton register their own
                    // focusable inside their render fns (they are focusable *by
                    // default*, RFC-0018), so exclude them here to avoid
                    // double-registration.
                    if !matches!(element_name, "TextField" | "Checkbox" | "RadioButton") {
                        self.register_focusable(attrs, hit_rect, elem_idx);
                    }
                }
                // RFC-0005 `ScrollView`: clip children to this viewport and
                // translate the content by `−offset`. The overflow is scissored
                // by the encoder (an off-viewport child costs no fragments), and
                // the content was measured unbounded (layout `scroll`). `offset`
                // is a two-way `Vec2` the app can read or drive on either axis;
                // wheel and drag write it below. Rotation of a scroll viewport is
                // out of scope (the clip is an axis-aligned screen rect).
                let scroll_clip = if name.as_str() == "ScrollView" {
                    let (ox, oy) = self.resolve_axis_pair(attrs, "offset", (0.0, 0.0));
                    let tl = inherited_transform.apply_point([current_rect.x, current_rect.y]);
                    let clip = byard_core::frame::Rect::new(
                        tl[0],
                        tl[1],
                        current_rect.w * inherited_transform.scale[0],
                        current_rect.h * inherited_transform.scale[1],
                    );
                    frame.begin_clip(clip);
                    child_transform.translate[0] -= ox * inherited_transform.scale[0];
                    child_transform.translate[1] -= oy * inherited_transform.scale[1];
                    // Record a wheel/drag scroll target for whichever of
                    // `offset.x`/`offset.y` is a writable signal (e.g.
                    // `offset: (panX, scrollY)`): the input scrolls by writing it,
                    // clamped to the content extent. `dispatch_events` consumes
                    // these next tick (render-then-dispatch handshake).
                    let (sig_x, sig_y) = self.resolve_offset_sigs(attrs);
                    if sig_x.is_some() || sig_y.is_some() {
                        let (content_w, content_h) = self
                            .atlas
                            .content_size(atlas_node)
                            .ok()
                            .flatten()
                            .unwrap_or((current_rect.w, current_rect.h));
                        // RFC-0021 behaviours resolved from props.
                        let snap_page = matches!(Self::enum_prop(attrs, "snap"), Some("page"));
                        let page_sig = self.resolve_named_var_sig(attrs, "page");
                        let has_end = attrs.iter().any(|a| {
                            a.name.as_str() == "end_reached"
                                && matches!(a.kind, AttrKind::Event { .. })
                        });
                        #[allow(clippy::cast_possible_truncation)]
                        let end_threshold = has_end.then(|| {
                            self.eval_float_prop(attrs, "end_threshold")
                                .map_or(0.8, |v| v as f32)
                        });
                        let sv_elem = self.atlas.node_index(atlas_node);
                        self.scroll_targets.push(ScrollTarget {
                            rect: crate::interp::intrinsics::Rect::new(
                                clip.x,
                                clip.y,
                                clip.width,
                                clip.height,
                            ),
                            x: sig_x.map(|sig| ScrollAxis {
                                sig,
                                max: (content_w - current_rect.w).max(0.0),
                            }),
                            y: sig_y.map(|sig| ScrollAxis {
                                sig,
                                max: (content_h - current_rect.h).max(0.0),
                            }),
                            elem: sv_elem,
                            snap_page,
                            page_sig,
                            end_threshold,
                        });
                    }
                    Some(clip)
                } else {
                    None
                };
                // Children cull against this box's own scroll viewport when it is
                // a `ScrollView`, otherwise against whatever viewport an ancestor
                // `ScrollView` established — so rows nested under an inner `Column`
                // are culled too, not just the `ScrollView`'s direct child.
                let child_clip = scroll_clip.or(cull_clip);
                // A windowed `ScrollView` hands its computed row window to its
                // single list child (mirrors the build pass); nothing else
                // propagates one.
                let child_window = if name.as_str() == "ScrollView" {
                    match children.as_slice() {
                        [RenderNode::Box { children: rows, .. }] => {
                            // RFC-0018: count the expanded rows (a reactive `for`
                            // is one node that expands to N), mirroring the build.
                            let n = self.expand_concrete(rows, pools).len();
                            self.scroll_window(attrs, n)
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                let render_child = |this: &mut Self,
                                    child: &RenderNode,
                                    frame: &mut byard_core::frame::RenderFrame,
                                    flat_idx: &mut usize| {
                    let child_id = flat_ids[*flat_idx];
                    // RFC-0005 emission culling (north star): a child the
                    // scroll has pushed entirely out of the viewport is never
                    // pushed to the frame — a long list costs only its visible
                    // slice. Advance the cursor past the skipped subtree so the
                    // remaining children stay aligned.
                    if let Some(clip) = child_clip {
                        if this.child_fully_clipped(child_id, child_transform, clip) {
                            *flat_idx += this.flat_len(child, pools);
                            return;
                        }
                    }
                    this.render_node_with_atlas(
                        child,
                        child_id,
                        frame,
                        flat_ids,
                        flat_idx,
                        current_rect,
                        child_opacity,
                        child_transform,
                        child_clip,
                        child_window,
                        pools,
                    );
                };
                if let Some(win) = window {
                    // This box is a windowed list child (RFC-0005): the build pass
                    // wrapped its rows in a leading + trailing spacer leaf. Consume
                    // the leading spacer, render only rows `start..end`, then the
                    // trailing spacer — keeping the flat-id cursor in lockstep.
                    *flat_idx += 1;
                    // RFC-0018: expand a reactive `for` (or literal rows) and
                    // paint only the windowed slice, mirroring the build pass.
                    let rows = self.expand_concrete(children, pools);
                    for &row in &rows[win.start..win.end] {
                        render_child(self, row, frame, flat_idx);
                    }
                    *flat_idx += 1;
                } else {
                    // RFC-0018: expand reactive `when`/`for` into concrete children
                    // in the same order the layout pass did.
                    for child in self.expand_concrete(children, pools) {
                        render_child(self, child, frame, flat_idx);
                    }
                }
                if scroll_clip.is_some() {
                    frame.end_clip();
                }
                // Close the RFC-0019 instance-env scope opened at the top of this
                // arm (balanced with `env_base`), restoring the caller's env for
                // the remaining siblings.
                self.env.truncate(env_base);
            }
            RenderNode::Image {
                attrs,
                state_blocks,
                src,
            } => {
                if let Ok(Some(rect)) = self.atlas.resolved_rect(atlas_node) {
                    // RFC-0016: overlay active `on <state>` blocks before reading
                    // paint properties (fit/radius/opacity). RFC-0024: fold in the
                    // universal `selected:`/`invalid:` states.
                    let state = self
                        .atlas
                        .node_index(atlas_node)
                        .map_or_else(crate::interp::events::StyleState::empty, |i| {
                            self.router.style_state(i)
                        })
                        .union(self.prop_style_state(attrs, None, ""));
                    let attrs = resolve_state_attrs(attrs, state_blocks, state);
                    let attrs = attrs.as_ref();
                    let src_val = self
                        .binding_value(*src)
                        .and_then(|v| if let Value::Str(s) = v { Some(s) } else { None })
                        .unwrap_or_default();
                    let fit = self.eval_fit_prop(attrs);
                    let radii = self.resolve_radii(attrs, "radius");
                    let opacity = inherited_opacity
                        * self
                            .eval_float_prop(attrs, "opacity")
                            .map_or(1.0, |v| v as f32);
                    // RFC-0011 group transforms: an `Image` carries no transform
                    // field, so an ancestor's scale/translate is baked into its
                    // rect (top-left through the transform, extents scaled per
                    // axis). Rotation isn't representable here and is left to box
                    // primitives — same limitation as `Text`.
                    let tl = inherited_transform.apply_point([rect.x, rect.y]);
                    let tw = rect.width * inherited_transform.scale[0];
                    let th = rect.height * inherited_transform.scale[1];
                    frame.push_texture(byard_core::frame::TextureSampler {
                        rect: [tl[0], tl[1], tw, th],
                        src: src_val,
                        fit,
                        radii,
                        opacity,
                        // Re-emitted every tick; mirror Text's
                        // always-dirty lowering.
                        dirty: true,
                    });
                }
            }
            RenderNode::Vector { attrs, src } => {
                if let Ok(Some(rect)) = self.atlas.resolved_rect(atlas_node) {
                    let handle = self
                        .binding_value(*src)
                        .and_then(|v| if let Value::Str(s) = v { Some(s) } else { None })
                        .unwrap_or_default();
                    let base_rgb = self
                        .eval_color_prop(attrs, "color")
                        .map_or([1.0, 1.0, 1.0, 1.0], |c| {
                            super::intrinsics::color_to_rgba(c, false)
                        });
                    let opacity = inherited_opacity
                        * self
                            .eval_float_prop(attrs, "opacity")
                            .map_or(1.0, |v| v as f32);

                    // Cache hit: a resident glyph, tinted and opacity-applied.
                    // Cache miss: a zero-opacity placeholder so the frame ships
                    // without stalling (INV-9); the dispatch itself happened
                    // inside `lookup_or_dispatch`.
                    let (uv_rect, layer, px_range, alpha) =
                        match self.vector_jit.lookup_or_dispatch(&handle) {
                            Some(glyph) => (
                                glyph.uv_rect,
                                glyph.layer,
                                glyph.px_range,
                                base_rgb[3] * opacity,
                            ),
                            None => (
                                byard_core::frame::Rect::new(0.0, 0.0, 0.0, 0.0),
                                0,
                                super::intrinsics::VECTOR_DEFAULT_PX_RANGE,
                                0.0,
                            ),
                        };
                    let rgb = [base_rgb[0], base_rgb[1], base_rgb[2], alpha];

                    frame.push_vector(byard_core::frame::VectorInstance::new(
                        byard_core::frame::Rect::new(rect.x, rect.y, rect.width, rect.height),
                        uv_rect,
                        rgb,
                        px_range,
                        layer,
                    ));
                }
            }
            // RFC-0020: the `Canvas` drawing surface. Shape parameters are
            // re-evaluated every tick through `eval_pure`, so a reactive or
            // `with`-animated `sweep`/`dash_offset`/color animates with zero
            // extra plumbing (RFC-0010's single evaluation chokepoint).
            RenderNode::Canvas {
                attrs,
                state_blocks,
                shapes,
                action,
                env_snapshot,
            } => {
                if let Ok(Some(rect)) = self.atlas.resolved_rect(atlas_node) {
                    let elem_idx = self.atlas.node_index(atlas_node);
                    // RFC-0016: overlay active `on <state>` blocks before
                    // reading paint properties. RFC-0024: fold in universal states.
                    let state = elem_idx
                        .map_or_else(crate::interp::events::StyleState::empty, |i| {
                            self.router.style_state(i)
                        })
                        .union(self.prop_style_state(attrs, None, ""));
                    let paint_attrs = resolve_state_attrs(attrs, state_blocks, state);
                    let paint_attrs = paint_attrs.as_ref();

                    // RFC-0019 §2: restore the instance environment so shape
                    // parameter expressions and event actions resolve against
                    // the scope the canvas was instantiated in.
                    let env_base = self.env.len();
                    for (k, v) in env_snapshot {
                        self.env.push(k.clone(), v.clone());
                    }

                    let opacity = inherited_opacity
                        * self
                            .eval_float_prop(paint_attrs, "opacity")
                            .map_or(1.0, |v| v as f32);
                    let canvas_rect = crate::interp::intrinsics::Rect::new(
                        rect.x,
                        rect.y,
                        rect.width,
                        rect.height,
                    );

                    // Background fill: a plain solid behind every shape.
                    if let Some(bg) = self.eval_color_prop(paint_attrs, "bg") {
                        frame.push_instance(byard_core::BoxInstance {
                            rect: [rect.x, rect.y, rect.width, rect.height],
                            color: dim_alpha(super::intrinsics::color_to_rgba(bg, false), opacity),
                            radii: [0.0; 4],
                            transform: inherited_transform,
                        });
                    }

                    // Shape commands, in declaration order (painter's order —
                    // each `push_canvas_shape` advances the global emission
                    // depth, RFC-0011).
                    for shape in shapes {
                        self.emit_canvas_shape(
                            shape,
                            canvas_rect,
                            opacity,
                            inherited_transform,
                            frame,
                        );
                    }

                    // Events: the canvas rect only — individual shapes are not
                    // hit-testable (RFC-0020 resolved question).
                    let hit_rect =
                        crate::interp::intrinsics::inflate_hit_rect(canvas_rect, parent_rect);
                    // RFC-0016: an `on hover`/`on pressed` block with no handler
                    // still needs pointer tracking, mirroring the Box path.
                    if let Some(idx) = elem_idx {
                        if state_blocks.iter().any(|sb| {
                            sb.states.iter().any(|s| {
                                matches!(s, StyleStateKind::Hover | StyleStateKind::Pressed)
                            })
                        }) {
                            self.router.track_region(idx, hit_rect);
                        }
                    }
                    let has_events = attrs
                        .iter()
                        .any(|a| matches!(a.kind, AttrKind::Event { .. }))
                        || action.is_some();
                    if has_events {
                        self.register_event_attrs(attrs, hit_rect, elem_idx);
                        if let Some(action_expr) = action {
                            if let Ok(closure) = self.lower_action(action_expr, None) {
                                if let Some(idx) = elem_idx {
                                    self.router.on(
                                        idx,
                                        hit_rect,
                                        crate::interp::events::EventKind::Tap,
                                        closure,
                                    );
                                }
                            }
                        }
                    }
                    self.env.truncate(env_base);
                }
            }
        }
    }

    // ── Widget rendering helpers (M16/M19) ─────────────────────────────

    /// Renders a `Toggle` widget: track + thumb (M19), and registers a Tap
    /// handler to flip the bound bool (M16).
    #[allow(clippy::too_many_arguments)]
    fn render_toggle(
        &mut self,
        bound_sig: Option<super::env::SignalId>,
        attrs: &[Attr],
        rect: crate::interp::intrinsics::Rect,
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
        transform: byard_core::frame::Transform,
        opacity: f32,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        let is_on = bound_sig.is_some_and(|s| self.ctx.peek_signal(s).as_bool().unwrap_or(false));

        // The full-height pill track. `bg` is the ON accent (default: theme
        // primary); OFF is a muted surface tint.
        let accent = self
            .eval_color_prop(attrs, "bg")
            .unwrap_or(self.theme.primary());
        let track_color = if is_on {
            super::intrinsics::color_to_rgba(accent, false)
        } else {
            [0.40_f32, 0.42, 0.48, 1.0]
        };
        let radius = rect.h / 2.0;
        frame.push_instance(byard_core::BoxInstance {
            rect: [rect.x, rect.y, rect.w, rect.h],
            color: dim_alpha(track_color, opacity),
            radii: [radius; 4],
            transform,
        });

        // Thumb: a white circle inset from the track edges, sliding L↔R.
        let pad = (rect.h * 0.12).max(2.0);
        let thumb_size = (rect.h - pad * 2.0).max(2.0);
        let thumb_y = rect.y + pad;
        let thumb_x = if is_on {
            rect.x + rect.w - thumb_size - pad
        } else {
            rect.x + pad
        };
        frame.push_instance(byard_core::BoxInstance {
            rect: [thumb_x, thumb_y, thumb_size, thumb_size],
            color: dim_alpha([1.0, 1.0, 1.0, 1.0], opacity),
            radii: [thumb_size / 2.0; 4],
            transform,
        });

        // Tap handler to flip the bool (M16).
        if let (Some(sig), Some(idx)) = (bound_sig, elem_idx) {
            let flip: super::events::Action = Box::new(move |ctx, _| {
                let cur = ctx.peek_signal(sig).as_bool().unwrap_or(false);
                ctx.write_signal(sig, Value::Bool(!cur));
            });
            self.router
                .on(idx, hit_rect, super::events::EventKind::Tap, flip);
        }
    }

    /// Renders a `Checkbox` widget (RFC-0018): a rounded square that fills with
    /// the accent colour and shows an engine-drawn checkmark when checked, a
    /// muted filled slot when unchecked, and a horizontal dash for the
    /// `indeterminate` mixed state — all borderless SolidBoxes (a 2px ring reads
    /// as a heavy dark outline at control sizes). Registers Tap + Space (KeyDown)
    /// to flip the bound bool, a `change` write-back (RFC-0003 E1), and a focus
    /// target so Tab and click reach it. Like `Toggle`, it owns its visuals —
    /// `bg` is the checked accent, never a full-rect slab.
    ///
    /// The checkmark is drawn as two rounded stroke quads rotated to each
    /// segment's angle through the paint-transform system, so it stays crisp at
    /// any size/DPI and needs no atlas asset for the leaf mark. (RFC-0018's
    /// resolved MSDF-baked checkmark is a rendering refinement tracked for the
    /// vector subsystem; the geometry and interaction contract here are final.)
    #[allow(clippy::too_many_arguments)]
    fn render_checkbox(
        &mut self,
        bound_sig: Option<super::env::SignalId>,
        attrs: &[Attr],
        rect: crate::interp::intrinsics::Rect,
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
        transform: byard_core::frame::Transform,
        opacity: f32,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        let is_on = bound_sig.is_some_and(|s| self.ctx.peek_signal(s).as_bool().unwrap_or(false));
        let indeterminate = self.eval_bool_prop(attrs, "indeterminate").unwrap_or(false);
        // A checkbox is square: use the smaller side so a non-square rect still
        // yields crisp, proportional geometry, centred in the laid-out box.
        let side = rect.w.min(rect.h);
        let ox = rect.x + (rect.w - side) / 2.0;
        let oy = rect.y + (rect.h - side) / 2.0;
        let radius = (side * 0.18).max(2.0);
        let filled = is_on || indeterminate;

        // The container is a rounded `DecoratedBox` so it can carry a `border`
        // (RFC-0024 lets a style set `border`/`on checked { border }`). `bg` is
        // the checked accent (default: theme primary); when unchecked, a styled
        // `border` yields an outlined box with a transparent interior (the M3
        // look), otherwise a muted filled slot (`Toggle`'s OFF tint). The
        // container is pushed *before* the mark — both on the decorated pipeline —
        // so the white mark lands on top.
        let bg = self.eval_color_prop(attrs, "bg");
        let border = self.eval_color_prop(attrs, "border");
        let border_width = if border.is_some() {
            self.eval_float_prop(attrs, "border_width")
                .map_or(2.0, |v| v as f32)
        } else {
            0.0
        };
        let border_rgba = border.map_or([0.0; 4], |c| {
            dim_alpha(super::intrinsics::color_to_rgba(c, false), opacity)
        });
        let accent = bg.unwrap_or(self.theme.primary());
        let fill = if filled {
            dim_alpha(super::intrinsics::color_to_rgba(accent, false), opacity)
        } else if border.is_some() {
            [0.0; 4]
        } else {
            dim_alpha([0.40, 0.42, 0.48, 1.0], opacity)
        };
        frame.push_decorated(byard_core::frame::DecoratedBox {
            base: byard_core::BoxInstance {
                rect: [ox, oy, side, side],
                color: fill,
                radii: [radius; 4],
                transform,
            },
            border_width,
            border_color: border_rgba,
            dirty: true,
            ..Default::default()
        });

        // The mark, in white, on the filled square (also decorated, so it paints
        // above the container).
        let mark = dim_alpha([1.0, 1.0, 1.0, 1.0], opacity);
        if indeterminate {
            // Mixed state: a single horizontal bar (checked overrides mixed only
            // in the container fill; the dash renders whenever `indeterminate`).
            let bar_w = side * 0.5;
            let bar_h = (side * 0.12).max(2.0);
            frame.push_decorated(byard_core::frame::DecoratedBox {
                base: byard_core::BoxInstance {
                    rect: [
                        ox + (side - bar_w) / 2.0,
                        oy + (side - bar_h) / 2.0,
                        bar_w,
                        bar_h,
                    ],
                    color: mark,
                    radii: [bar_h / 2.0; 4],
                    transform,
                },
                dirty: true,
                ..Default::default()
            });
        } else if is_on {
            // A two-stroke checkmark: canonical vertices in the unit square,
            // each segment a rounded quad rotated to its angle.
            let t = (side * 0.13).max(2.0);
            let pts = [
                [ox + side * 0.26, oy + side * 0.52], // short-stroke start
                [ox + side * 0.44, oy + side * 0.70], // bottom vertex
                [ox + side * 0.76, oy + side * 0.32], // long-stroke end
                [ox + side * 0.41, oy + side * 0.70], // long-stroke start
            ];
            for (a, b) in [(pts[0], pts[1]), (pts[3], pts[2])] {
                push_stroke_quad(frame, a, b, t, mark, transform);
            }
        }

        // Handlers: Tap + Space flip the bool; `change` write-back; focusable.
        if let (Some(sig), Some(idx)) = (bound_sig, elem_idx) {
            let flip: super::events::Action = Box::new(move |ctx, _| {
                let cur = ctx.peek_signal(sig).as_bool().unwrap_or(false);
                ctx.write_signal(sig, Value::Bool(!cur));
            });
            self.router
                .on(idx, hit_rect, super::events::EventKind::Tap, flip);

            // Space toggles when focused (WAI-ARIA checkbox keyboard pattern).
            let key_flip: super::events::Action = Box::new(move |ctx, payload| {
                if let Some(Value::Str(key)) = payload {
                    if matches!(key.as_str(), " " | "Space" | "Spacebar") {
                        let cur = ctx.peek_signal(sig).as_bool().unwrap_or(false);
                        ctx.write_signal(sig, Value::Bool(!cur));
                    }
                }
            });
            self.router
                .on(idx, hit_rect, super::events::EventKind::KeyDown, key_flip);

            // Change write-back from the platform (RFC-0003 E1).
            self.router.on(
                idx,
                hit_rect,
                super::events::EventKind::Change,
                super::events::write_back_action(sig),
            );

            // Focus target so Tab and click reach the box. Uses the element's own
            // `focused:` var when given, else a private signal for the registry.
            let focused_sig = self.resolve_focused_sig(attrs);
            let fsig = focused_sig.unwrap_or_else(|| self.ctx.create_signal(Value::Bool(false)));
            self.router.focusable(idx, hit_rect, fsig);
        }
    }

    /// Renders a `RadioButton` widget (RFC-0018): an outer ring plus an inner
    /// filled dot when selected. Selection is `bind == value`: the bound group
    /// `var`'s current string equals this button's `value`. Tapping writes this
    /// button's `value` to the group var, so the previously selected sibling
    /// deselects reactively (automatic mutual exclusion — every radio in the
    /// group reads the same var). Registers the group ordering for arrow-key
    /// navigation, a Tap handler, arrow KeyDown handlers (move selection within
    /// the group, wrapping), a `change` write-back (RFC-0003 E1), and a focus
    /// target so Tab/click reach it. Owns its visuals — `bg` is the selected
    /// accent. The ring is the radio's defining affordance (unlike `Checkbox`,
    /// whose square border was dropped); its interior is transparent so the dot
    /// shows through and it composes over any background.
    #[allow(clippy::too_many_arguments)]
    fn render_radio(
        &mut self,
        bound_sig: Option<super::env::SignalId>,
        attrs: &[Attr],
        rect: crate::interp::intrinsics::Rect,
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
        transform: byard_core::frame::Transform,
        opacity: f32,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        let value = self.eval_str_prop(attrs, "value").unwrap_or_default();
        let selected = bound_sig.is_some_and(|s| match self.ctx.peek_signal(s) {
            Value::Str(v) => v == value,
            _ => false,
        });

        // A radio is circular: use the smaller side, centred in the laid-out box.
        let side = rect.w.min(rect.h);
        let ox = rect.x + (rect.w - side) / 2.0;
        let oy = rect.y + (rect.h - side) / 2.0;
        let r = side / 2.0;

        let accent = self
            .eval_color_prop(attrs, "bg")
            .unwrap_or(self.theme.primary());
        let accent_rgba = super::intrinsics::color_to_rgba(accent, false);
        let ring_color = if selected {
            accent_rgba
        } else {
            [0.55, 0.57, 0.62, 1.0]
        };

        // Inner dot FIRST (solid pipeline) so it sits beneath the ring; the ring's
        // interior is transparent, so the dot shows through regardless. Only drawn
        // when selected.
        if selected {
            let dot = side * 0.5;
            let inset = (side - dot) / 2.0;
            frame.push_instance(byard_core::BoxInstance {
                rect: [ox + inset, oy + inset, dot, dot],
                color: dim_alpha(accent_rgba, opacity),
                radii: [dot / 2.0; 4],
                transform,
            });
        }

        // Outer ring: a transparent-interior DecoratedBox with a full-radius
        // border. Width scales with size so it reads as a ring, not a hairline.
        let ring_w = (side * 0.12).max(2.0);
        frame.push_decorated(byard_core::frame::DecoratedBox {
            base: byard_core::BoxInstance {
                rect: [ox, oy, side, side],
                color: [0.0; 4],
                radii: [r; 4],
                transform,
            },
            border_width: ring_w,
            border_color: dim_alpha(ring_color, opacity),
            dirty: true,
            ..Default::default()
        });

        // ── Group registration + handlers ────────────────────────────────────
        if let (Some(sig), Some(idx)) = (bound_sig, elem_idx) {
            // Record this button's value in its group's ordered list (shared via
            // an Rc so the arrow handlers below see the whole group once the
            // render has finished populating it).
            let group = self.radio_groups.entry(sig).or_default().clone();
            group.borrow_mut().push(value.clone());

            // Tap → select this button (write its value to the group var).
            let val = value.clone();
            let select: super::events::Action = Box::new(move |ctx, _| {
                ctx.write_signal(sig, Value::Str(val.clone()));
            });
            self.router
                .on(idx, hit_rect, super::events::EventKind::Tap, select);

            // Arrow keys → move selection to the next/previous value in the group,
            // wrapping at both ends (WAI-ARIA radio group). Down/Right advance;
            // Up/Left retreat.
            let grp = group.clone();
            let arrows: super::events::Action = Box::new(move |ctx, payload| {
                let Some(Value::Str(key)) = payload else {
                    return;
                };
                let forward = match key.as_str() {
                    "ArrowDown" | "ArrowRight" => true,
                    "ArrowUp" | "ArrowLeft" => false,
                    _ => return,
                };
                let vals = grp.borrow();
                let n = vals.len();
                if n == 0 {
                    return;
                }
                let cur = match ctx.peek_signal(sig) {
                    Value::Str(s) => s,
                    _ => String::new(),
                };
                let here = vals.iter().position(|v| *v == cur).unwrap_or(0);
                // Wrap-around in unsigned arithmetic (no signed casts): forward is
                // `+1`, backward is `+ (n − 1)`, both mod `n`.
                let next = if forward {
                    (here + 1) % n
                } else {
                    (here + n - 1) % n
                };
                ctx.write_signal(sig, Value::Str(vals[next].clone()));
            });
            self.router
                .on(idx, hit_rect, super::events::EventKind::KeyDown, arrows);

            // Change write-back from the platform (RFC-0003 E1).
            self.router.on(
                idx,
                hit_rect,
                super::events::EventKind::Change,
                super::events::write_back_action(sig),
            );

            // Focus target so Tab and click reach the button.
            let focused_sig = self.resolve_focused_sig(attrs);
            let fsig = focused_sig.unwrap_or_else(|| self.ctx.create_signal(Value::Bool(false)));
            self.router.focusable(idx, hit_rect, fsig);
        }
    }

    /// Renders a `Slider` widget: track + fill + thumb (M19), and registers
    /// PointerDown + PointerDrag handlers to write the value (M16).
    #[allow(clippy::too_many_arguments)]
    fn render_slider(
        &mut self,
        bound_sig: Option<super::env::SignalId>,
        attrs: &[Attr],
        rect: crate::interp::intrinsics::Rect,
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
        transform: byard_core::frame::Transform,
        opacity: f32,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        // Keep the authored `f64` values for the value-write path: computing the
        // emitted value in `f64` avoids the `f32`→`f64` widening artifact (a drag
        // landing on 0.6 was stored as `f64::from(0.6_f32)` =
        // 0.6000000238418579). The `f32` casts below are only for pixel-space
        // visual layout (track/fill/thumb), where the noise is invisible.
        let min_f = self.eval_float_prop(attrs, "min").unwrap_or(0.0);
        let max_f = self.eval_float_prop(attrs, "max").unwrap_or(1.0);
        let step_f = self.eval_float_prop(attrs, "step");
        let min = min_f as f32;
        let max = max_f as f32;
        let cur_val = bound_sig.map_or(min, |s| match self.ctx.peek_signal(s) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => min,
        });
        let t = if (max - min).abs() > f32::EPSILON {
            ((cur_val - min) / (max - min)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // `bg` is the fill accent (default: theme primary); the unfilled track
        // is a muted tint.
        let accent = self
            .eval_color_prop(attrs, "bg")
            .unwrap_or(self.theme.primary());
        let accent_rgba = super::intrinsics::color_to_rgba(accent, false);

        // Track (unfilled remainder).
        let track_h = (rect.h * 0.28).clamp(4.0, 8.0);
        let track_y = rect.y + (rect.h - track_h) / 2.0;
        let track_r = track_h / 2.0;
        frame.push_instance(byard_core::BoxInstance {
            rect: [rect.x, track_y, rect.w, track_h],
            color: dim_alpha([0.40, 0.42, 0.48, 1.0], opacity),
            radii: [track_r; 4],
            transform,
        });

        // Fill up to the thumb.
        let fill_w = t * rect.w;
        if fill_w > 0.0 {
            frame.push_instance(byard_core::BoxInstance {
                rect: [rect.x, track_y, fill_w, track_h],
                color: dim_alpha(accent_rgba, opacity),
                radii: [track_r; 4],
                transform,
            });
        }

        // Thumb: white circle with a thin accent ring (drawn as accent disc
        // under a slightly smaller white disc).
        let thumb_size = (rect.h * 0.85).clamp(14.0, 22.0);
        let thumb_x = rect.x + t * (rect.w - thumb_size);
        let thumb_y = rect.y + (rect.h - thumb_size) / 2.0;
        frame.push_instance(byard_core::BoxInstance {
            rect: [thumb_x, thumb_y, thumb_size, thumb_size],
            color: dim_alpha(accent_rgba, opacity),
            radii: [thumb_size / 2.0; 4],
            transform,
        });
        let inner = thumb_size - 5.0;
        frame.push_instance(byard_core::BoxInstance {
            rect: [thumb_x + 2.5, thumb_y + 2.5, inner, inner],
            color: dim_alpha([1.0, 1.0, 1.0, 1.0], opacity),
            radii: [inner / 2.0; 4],
            transform,
        });

        // Handlers: PointerDown + PointerDrag (M16).
        if let (Some(sig), Some(idx)) = (bound_sig, elem_idx) {
            let track_x = rect.x;
            let track_w = rect.w;
            let make_drag_action =
                |min: f64, max: f64, step: Option<f64>| -> super::events::Action {
                    Box::new(move |ctx, _| {
                        let pos = super::events::CURRENT_EVENT_POS.with(std::cell::Cell::get);
                        // Pixel positions are `f32`; widen before the value math so
                        // the stored value never carries `f32` rounding noise.
                        let t = ((f64::from(pos.0) - f64::from(track_x)) / f64::from(track_w))
                            .clamp(0.0, 1.0);
                        let raw = min + t * (max - min);
                        // Quantise so the value never carries more decimals than
                        // the step (or, with no step, a readable default) — see
                        // `step_decimals`/`SLIDER_DEFAULT_DECIMALS`.
                        let val = match step {
                            Some(s) => round_to_decimals((raw / s).round() * s, step_decimals(s)),
                            None => round_to_decimals(raw, SLIDER_DEFAULT_DECIMALS),
                        };
                        ctx.write_signal(sig, Value::Float(val));
                    })
                };
            self.router.on(
                idx,
                hit_rect,
                super::events::EventKind::PointerDown,
                make_drag_action(min_f, max_f, step_f),
            );
            self.router.on(
                idx,
                hit_rect,
                super::events::EventKind::PointerDrag,
                make_drag_action(min_f, max_f, step_f),
            );
        }
    }

    /// Renders a `TextField` widget: background box + text/placeholder (M19),
    /// and registers keyboard handlers for text input (M16/M17).
    #[allow(clippy::too_many_arguments)]
    fn render_text_field(
        &mut self,
        bound_sig: Option<super::env::SignalId>,
        attrs: &[Attr],
        rect: crate::interp::intrinsics::Rect,
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
        transform: byard_core::frame::Transform,
        opacity: f32,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        let placeholder = self.eval_str_prop(attrs, "placeholder").unwrap_or_default();
        let cur_text = bound_sig
            .map(|s| match self.ctx.peek_signal(s) {
                Value::Str(t) => t,
                _ => String::new(),
            })
            .unwrap_or_default();

        let (display_text, is_placeholder) = if cur_text.is_empty() {
            (placeholder, true)
        } else {
            (cur_text, false)
        };

        let text_color = if is_placeholder {
            0x0088_8888_i64
        } else {
            0x00ff_ffff_i64
        };
        let font_size = self.eval_int_prop(attrs, "size").unwrap_or(16) as f32;
        let is_focused = elem_idx.is_some_and(|i| self.router.is_focused(i));

        // Focus underline (Material-style): a thin accent bar along the bottom
        // edge when the field holds focus.
        if is_focused {
            let bar_h = 2.0_f32;
            frame.push_instance(byard_core::BoxInstance {
                rect: [rect.x, rect.y + rect.h - bar_h, rect.w, bar_h],
                color: dim_alpha(
                    super::intrinsics::color_to_rgba(self.theme.primary(), false),
                    opacity,
                ),
                radii: [0.0; 4],
                transform,
            });
        }

        let pad_x = 10.0_f32;
        let text_x = rect.x + pad_x;
        let text_y = rect.y + (rect.h - font_size) / 2.0;
        // NOTE: `TextLine` carries no `Transform` field (RFC-0011 engine slice:
        // only box primitives were given one), so the field's *text* does not
        // follow `translate`/`scale`/`rotate` — the box visuals below (underline,
        // caret) and its `bg` fill do. Same limitation as the `Text` intrinsic.
        if !display_text.is_empty() {
            frame.push_text(byard_core::TextLine {
                x: text_x,
                y: text_y,
                text: display_text.clone(),
                font_size,
                color: dim_alpha(super::intrinsics::color_to_rgba(text_color, false), opacity),
                dirty: true,
            });
        }

        // Caret at the end of the entered text while focused (M17/M19).
        if is_focused {
            let measured = if is_placeholder {
                0.0
            } else {
                self.measure_text(&display_text, font_size).0
            };
            frame.push_instance(byard_core::BoxInstance {
                rect: [text_x + measured + 1.0, text_y, 1.5, font_size],
                color: dim_alpha([1.0, 1.0, 1.0, 1.0], opacity),
                radii: [0.0; 4],
                transform,
            });
        }

        // Handlers: TextInput appends, KeyDown handles Backspace/Enter/Tab (M16/M17).
        if let (Some(sig), Some(idx)) = (bound_sig, elem_idx) {
            // TextInput: append typed text
            let text_input: super::events::Action = Box::new(move |ctx, payload| {
                if let Some(Value::Str(ch)) = payload {
                    let cur = match ctx.peek_signal(sig) {
                        Value::Str(s) => s,
                        _ => String::new(),
                    };
                    ctx.write_signal(sig, Value::Str(cur + ch.as_str()));
                }
            });
            self.router.on(
                idx,
                hit_rect,
                super::events::EventKind::TextInput,
                text_input,
            );

            // KeyDown: Backspace deletes, Enter/Escape handled (submit fires via Change)
            let key_down: super::events::Action = Box::new(move |ctx, payload| {
                if let Some(Value::Str(key)) = payload {
                    match key.as_str() {
                        "Backspace" => {
                            let cur = match ctx.peek_signal(sig) {
                                Value::Str(s) => s,
                                _ => String::new(),
                            };
                            let mut s = cur;
                            s.pop();
                            ctx.write_signal(sig, Value::Str(s));
                        }
                        "Delete" => {
                            ctx.write_signal(sig, Value::Str(String::new()));
                        }
                        _ => {}
                    }
                }
            });
            self.router
                .on(idx, hit_rect, super::events::EventKind::KeyDown, key_down);

            // Change event: write-back from platform (E1).
            self.router.on(
                idx,
                hit_rect,
                super::events::EventKind::Change,
                super::events::write_back_action(sig),
            );

            // Register as focusable so Tab and click steal focus (M18).
            // TextField uses its own focused-var if provided via `focused:` attr;
            // otherwise we create a dummy signal just for the focusable registry.
            let focused_sig = self.resolve_focused_sig(attrs);
            let fsig = focused_sig.unwrap_or_else(|| self.ctx.create_signal(Value::Bool(false)));
            self.router.focusable(idx, hit_rect, fsig);
        }
    }

    /// Resolves the `focused:` attribute to a `SignalId`, if present.
    fn resolve_focused_sig(&self, attrs: &[Attr]) -> Option<super::env::SignalId> {
        use crate::parser::ast::Expr;
        for attr in attrs {
            if attr.name.as_str() == "focused" {
                if let AttrKind::Prop {
                    value: Expr::Ident(name, _),
                } = &attr.kind
                {
                    if let Some(Value::Signal(sig)) = self.env.lookup(name) {
                        return Some(*sig);
                    }
                }
            }
        }
        None
    }

    /// Registers handlers for all event-kind attrs (`#[tap => …]`, etc.).
    fn register_event_attrs(
        &mut self,
        attrs: &[Attr],
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
    ) {
        for attr in attrs {
            if let AttrKind::Event { payload, action } = &attr.kind {
                let event_kind = match attr.name.as_str() {
                    "tap" | "click" => super::events::EventKind::Tap, // "click" is an alias (RFC-0012 §A)
                    "pointer_down" => super::events::EventKind::PointerDown,
                    "pointer_up" => super::events::EventKind::PointerUp,
                    "pointer_move" => super::events::EventKind::PointerMove,
                    "scroll" => super::events::EventKind::Scroll,
                    "wheel" => super::events::EventKind::Wheel,
                    // RFC-0021 advanced scroll behaviours (engine-fired).
                    "end_reached" => super::events::EventKind::EndReached,
                    "page_change" => super::events::EventKind::PageChange,
                    "scroll_end" => super::events::EventKind::ScrollEnd,
                    "refresh" => super::events::EventKind::Refresh,
                    "change" => super::events::EventKind::Change,
                    "key_down" => super::events::EventKind::KeyDown,
                    "key_up" => super::events::EventKind::KeyUp,
                    "text_input" => super::events::EventKind::TextInput,
                    // RFC-0012 §A: the six modeled-but-previously-unexposed events.
                    "hover" => super::events::EventKind::Hover,
                    "pointer_enter" => super::events::EventKind::PointerEnter,
                    "pointer_exit" => super::events::EventKind::PointerExit,
                    "long_press" => super::events::EventKind::LongPress,
                    "double_tap" => super::events::EventKind::DoubleTap,
                    "secondary" => super::events::EventKind::Secondary,
                    // RFC-0012 S2: `focus =>`/`blur =>` sugar over `focused_sig`'s
                    // edges — registered as ordinary handlers here; `steal_focus`
                    // fires them directly (see `interp::events::EventKind::Focus`).
                    "focus" => super::events::EventKind::Focus,
                    "blur" => super::events::EventKind::Blur,
                    _ => continue,
                };
                if let Ok(closure) = self.lower_action(action, payload.clone()) {
                    if let Some(idx) = elem_idx {
                        self.router.on(idx, hit_rect, event_kind, closure);
                    }
                }
            }
        }
    }

    /// Registers an element as focusable if it has a `focused:` prop attr
    /// (M16/M18), **or** a `focus =>`/`blur =>` handler (RFC-0012 S2) — the
    /// sugar rides `focused_sig`'s edges, so an element that only wants the
    /// one-shot event (no bound `var`) still needs a signal for
    /// `steal_focus` to flip. That signal is a fresh internal one when
    /// `focused:` wasn't given, mirroring `render_text_field`'s same
    /// bind-or-create pattern for its own `focused_sig`.
    fn register_focusable(
        &mut self,
        attrs: &[Attr],
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
    ) {
        // Without an index there is nowhere to register the focusable, so a
        // freshly created internal signal below would just be dropped —
        // bail out first rather than allocate one for nothing.
        let Some(idx) = elem_idx else {
            return;
        };
        let has_focus_sugar = attrs.iter().any(|a| {
            matches!(a.kind, AttrKind::Event { .. }) && matches!(a.name.as_str(), "focus" | "blur")
        });
        let sig = self
            .resolve_focused_sig(attrs)
            .or_else(|| has_focus_sugar.then(|| self.ctx.create_signal(Value::Bool(false))));
        if let Some(sig) = sig {
            self.router.focusable(idx, hit_rect, sig);
        }
    }

    fn eval_color_prop(&mut self, attrs: &[Attr], name: &str) -> Option<i64> {
        // Resolve the matching attribute value; a `with` colour animation
        // (RFC-0010 A3) is driven through the OKLab path rather than the scalar
        // one, since a packed `0xRRGGBB` can't be interpolated component-wise.
        let value = attrs.iter().find_map(|a| match (&a.name, &a.kind) {
            (n, AttrKind::Prop { value }) if n.as_str() == name => Some(value),
            _ => None,
        })?;
        if let Expr::Animated {
            value: target,
            anim,
            span,
        } = value
        {
            return Some(self.eval_animated_color(target, anim, *span));
        }
        self.eval_pure(value).as_int()
    }

    /// Drives one colour `with` animation (RFC-0010 A3): interpolates from the
    /// current colour to the target in OKLab (one [`Motion`] per channel), so
    /// the transition is perceptually uniform and interruptible. Returns the
    /// current colour packed as `0xRRGGBB`.
    ///
    /// [`Motion`]: byard_core::frame::Motion
    fn eval_animated_color(&mut self, target: &Expr, anim: &Expr, key: Span) -> i64 {
        let target_int = self.eval_pure(target).as_int().unwrap_or(0);
        // Without an advancing clock, jump straight to the target (mirrors the
        // scalar path — never latch `has_active_animations` on t=0).
        if !self.clock_set {
            return target_int;
        }
        let Ok(curve) = crate::interp::anim::resolve_curve(anim) else {
            return target_int;
        };
        let packed = pack_curve(curve);
        let now = self.now_ms;
        let target_oklab = oklab_from_hex(target_int);
        let motions = self.color_animations.entry(key).or_insert_with(|| {
            [0, 1, 2].map(|i| byard_core::frame::Motion {
                from: target_oklab[i],
                to: target_oklab[i],
                start_ms: now,
                curve: packed,
            })
        });
        let mut current = [0.0_f32; 3];
        let mut all_settled = true;
        for (i, m) in motions.iter_mut().enumerate() {
            if (m.to - target_oklab[i]).abs() > 1e-5 {
                let here = m.sample(now);
                m.from = here;
                m.to = target_oklab[i];
                m.start_ms = now;
            }
            m.curve = packed;
            current[i] = m.sample(now);
            if !m.is_settled_with_eps(now, ANIM_SETTLE_EPS_POS, ANIM_SETTLE_EPS_VEL) {
                all_settled = false;
            }
        }
        if !all_settled {
            self.any_active = true;
        }
        hex_from_oklab(current)
    }

    fn eval_int_prop(&mut self, attrs: &[Attr], name: &str) -> Option<i64> {
        attrs.iter().find_map(|a| {
            if a.name.as_str() == name {
                if let AttrKind::Prop { value } = &a.kind {
                    let val = self.eval_pure(value);
                    return val.as_int();
                }
            }
            None
        })
    }

    fn eval_float_prop(&mut self, attrs: &[Attr], name: &str) -> Option<f64> {
        attrs.iter().find_map(|a| {
            if a.name.as_str() == name {
                if let AttrKind::Prop { value } = &a.kind {
                    let val = self.eval_pure(value);
                    return match val {
                        Value::Float(f) => Some(f),
                        Value::Int(n) => Some(n as f64),
                        _ => None,
                    };
                }
            }
            None
        })
    }

    fn eval_bool_prop(&mut self, attrs: &[Attr], name: &str) -> Option<bool> {
        attrs.iter().find_map(|a| {
            if a.name.as_str() == name {
                if let AttrKind::Prop { value } = &a.kind {
                    if let Value::Bool(b) = self.eval_pure(value) {
                        return Some(b);
                    }
                }
            }
            None
        })
    }

    /// Resolves the `typo:` prop to a font size in logical pixels (RFC-0005
    /// `Typo`, completed by RFC-0022). Accepts either a bare token
    /// (`typo: titleLarge` → a `Str`, resolved against the theme's typography
    /// then the built-in M3 scale) or a theme accessor (`typo: t.titleLarge` →
    /// an `Int` size projected by [`lower_theme_member`](Self::lower_theme_member)).
    fn eval_typo_size(&mut self, attrs: &[Attr]) -> Option<i64> {
        let value = attrs.iter().find_map(|a| match (&a.name, &a.kind) {
            (n, AttrKind::Prop { value }) if n.as_str() == "typo" => Some(value),
            _ => None,
        })?;
        match self.eval_pure(value) {
            // A theme accessor already resolved to a concrete pixel size.
            Value::Int(px) => Some(px),
            Value::Float(px) =>
            {
                #[allow(clippy::cast_possible_truncation)]
                Some(px as i64)
            }
            // A bare token name → theme typography, falling back to M3 sizes.
            Value::Str(token) => self
                .theme
                .typo_size(&token)
                .or_else(|| super::theme::resolve_typo(&token))
                .map(|s| {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        s as i64
                    }
                }),
            _ => None,
        }
    }

    fn eval_str_prop(&mut self, attrs: &[Attr], name: &str) -> Option<String> {
        attrs.iter().find_map(|a| {
            if a.name.as_str() == name {
                if let AttrKind::Prop { value } = &a.kind {
                    let val = self.eval_pure(value);
                    return match val {
                        Value::Str(s) => Some(s),
                        _ => None,
                    };
                }
            }
            None
        })
    }

    /// The literal keyword token of an *enum* (keyword) prop value, read straight
    /// from the AST bareword — e.g. the `page` in `snap: page`.
    ///
    /// Enum props (`PropType::Enum`) are a closed keyword set validated by the
    /// checker as a bare [`Expr::Ident`] (`intrinsics.rs`, `check_attr_value`),
    /// never as a reactive expression. Reading them with [`eval_pure`] would
    /// instead resolve the identifier through the environment, so a same-named
    /// `var` silently *shadows* the keyword: `snap: page` next to a `var page`
    /// evaluates to the variable's `Int`, not the token `"page"`, and the
    /// behaviour turns off (RFC-0021). Taking the token from the AST matches the
    /// checker exactly, can never be shadowed, and skips lowering an expression
    /// for a value that is always a compile-time keyword.
    fn enum_token(value: &Expr) -> Option<&str> {
        match value {
            Expr::Ident(sym, _) => Some(sym.as_str()),
            // A keyword may also be written as a plain string literal
            // (`fit: "cover"`). A single, non-interpolated run is a token; an
            // interpolated string is not.
            Expr::StrLit(parts, _) => match parts.as_slice() {
                [StrPart::Text(s)] => Some(s.as_str()),
                _ => None,
            },
            _ => None,
        }
    }

    /// [`enum_token`](Self::enum_token) for a named attribute: the keyword token
    /// of enum prop `name`, or `None` if absent or not a bareword keyword.
    fn enum_prop<'a>(attrs: &'a [Attr], name: &str) -> Option<&'a str> {
        attrs.iter().find_map(|a| match &a.kind {
            AttrKind::Prop { value } if a.name.as_str() == name => Self::enum_token(value),
            _ => None,
        })
    }

    fn eval_fit_prop(&mut self, attrs: &[Attr]) -> byard_core::frame::ImageFit {
        match Self::enum_prop(attrs, "fit") {
            Some("contain") => byard_core::frame::ImageFit::Contain,
            Some("cover") => byard_core::frame::ImageFit::Cover,
            Some("none") => byard_core::frame::ImageFit::None,
            _ => byard_core::frame::ImageFit::Fill,
        }
    }

    fn eval_container_style(
        &mut self,
        element_name: &str,
        attrs: &[Attr],
    ) -> byard_core::atlas::layout::ContainerStyle {
        use byard_core::atlas::layout::{Align, ContainerStyle, FlexDir, Justify};

        let val_to_f32 = |v: &Value| -> Option<f32> {
            match v {
                Value::Int(n) => Some(*n as f32),
                Value::Float(f) => Some(*f as f32),
                _ => None,
            }
        };

        let mut style = ContainerStyle::default();
        style.direction = match element_name {
            "Row" => FlexDir::Row,
            _ => FlexDir::Column,
        };
        // RFC-0005 `ScrollView`: a scroll container — content is measured at
        // natural size and overflows the fixed viewport (clipped + scrolled by
        // the renderer), rather than flex-shrunk to fit. `axis` (default
        // `vertical`) picks the overflowing axes; `both` scrolls in 2D.
        if element_name == "ScrollView" {
            style = style.with_scroll_axes(false, true);
        }
        for attr in attrs {
            if let AttrKind::Prop { value } = &attr.kind {
                let val = self.eval_pure(value);
                match attr.name.as_str() {
                    "axis" if element_name == "ScrollView" => {
                        if let Some(s) = Self::enum_token(value) {
                            let (x, y) = match s {
                                "horizontal" => (true, false),
                                "both" => (true, true),
                                _ => (false, true),
                            };
                            style = style.with_scroll_axes(x, y);
                            // A ScrollView is `Column`, so its cross axis is x. To
                            // scroll horizontally, content must keep its natural
                            // width instead of being stretched to the viewport, or
                            // Taffy caps the content extent at the viewport and
                            // there is nothing to scroll. A `stretch` on the block
                            // axis (vertical-only) is still what fills row width.
                            if x {
                                style.align = Align::Start;
                            }
                        }
                    }
                    "width" => style.width = val.as_int().map(|n| n as f32),
                    "height" => style.height = val.as_int().map(|n| n as f32),
                    "direction" => {
                        if let Some(s) = Self::enum_token(value) {
                            style.direction = match s {
                                "column" => FlexDir::Column,
                                _ => FlexDir::Row,
                            };
                        }
                    }
                    "gap" => {
                        if let Some(n) = val.as_int() {
                            style.gap = n as f32;
                        }
                    }
                    "p" | "padding" => {
                        style.padding = self.resolve_spacing(value, "p");
                    }
                    "pt" | "padding_top" | "padding-top" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.padding.top = v;
                        }
                    }
                    "pr" | "padding_right" | "padding-right" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.padding.right = v;
                        }
                    }
                    "pb" | "padding_bottom" | "padding-bottom" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.padding.bottom = v;
                        }
                    }
                    "pl" | "padding_left" | "padding-left" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.padding.left = v;
                        }
                    }
                    "m" | "margin" => {
                        style.margin = self.resolve_spacing(value, "m");
                    }
                    "mt" | "margin_top" | "margin-top" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.top = v;
                        }
                    }
                    "mr" | "margin_right" | "margin-right" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.right = v;
                        }
                    }
                    "mb" | "margin_bottom" | "margin-bottom" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.bottom = v;
                        }
                    }
                    "ml" | "margin_left" | "margin-left" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.left = v;
                        }
                    }
                    "mx" | "margin_x" | "margin_horizontal" | "margin-horizontal" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.left = v;
                            style.margin.right = v;
                        }
                    }
                    "my" | "margin_y" | "margin_vertical" | "margin-vertical" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.top = v;
                            style.margin.bottom = v;
                        }
                    }
                    "align" => {
                        if let Some(s) = Self::enum_token(value) {
                            style.align = match s {
                                "start" => Align::Start,
                                "center" => Align::Center,
                                "end" => Align::End,
                                _ => Align::Stretch,
                            };
                        }
                    }
                    "justify" => {
                        if let Some(s) = Self::enum_token(value) {
                            style.justify = match s {
                                "center" => Justify::Center,
                                "end" => Justify::End,
                                "between" => Justify::Between,
                                "around" => Justify::Around,
                                "evenly" => Justify::Evenly,
                                _ => Justify::Start,
                            };
                        }
                    }
                    "grow" => {
                        if let Some(n) = val.as_int() {
                            style.grow = n as f32;
                        }
                    }
                    _ => {}
                }
            }
        }
        style
    }

    /// Resolves a `Len`-typed `p`/`m` attribute value into a `Spacing` quad
    /// (RFC-0005 §1 erratum), emitting span-anchored `CompileError`s
    /// for the four error classes:
    ///
    /// - an unknown side name → [`CompileError::UnknownAttribute`] with a hint;
    /// - a side set twice, an axis shorthand plus one of its component sides, or
    ///   a tuple mixing named and positional fields →
    ///   [`CompileError::ConflictingSpacingField`];
    /// - a non-numeric side value → [`CompileError::AttributeTypeMismatch`];
    /// - a positional tuple of arity 3 or > 4 → [`CompileError::ArityMismatch`].
    ///
    /// Accepted forms: scalar (`p: 5`), inferred pair (`p: (vertical, horizontal)`),
    /// inferred quad CSS `(top, right, bottom, left)`, and the verbose named form
    /// (`p: (top: 4, horizontal: 8)`). A single parenthesized value parses to the
    /// inner expression, so it arrives as a scalar.
    fn resolve_spacing(&mut self, expr: &Expr, prop: &str) -> byard_core::atlas::layout::Spacing {
        use byard_core::atlas::layout::Spacing;
        match expr {
            Expr::Tuple(args, span) => {
                let any_named = args.iter().any(|a| a.name.is_some());
                let all_named = args.iter().all(|a| a.name.is_some());
                if any_named && !all_named {
                    self.errors.push(CompileError::ConflictingSpacingField {
                        span: *span,
                        message: "a spacing tuple cannot mix named and positional fields"
                            .to_string(),
                    });
                    return Spacing::default();
                }
                if all_named {
                    self.resolve_named_spacing(args)
                } else {
                    self.resolve_positional_spacing(args, *span, prop)
                }
            }
            other => {
                let val = self.eval_pure(other);
                if let Some(v) = spacing_value(&val) {
                    Spacing::all(v)
                } else {
                    self.errors.push(CompileError::AttributeTypeMismatch {
                        span: other.span(),
                        expected: "a length (an integer)".to_string(),
                    });
                    Spacing::default()
                }
            }
        }
    }

    /// Verbose named spacing form (`p: (top: 4, horizontal: 8)`).
    fn resolve_named_spacing(&mut self, args: &[Arg]) -> byard_core::atlas::layout::Spacing {
        use byard_core::atlas::layout::Spacing;
        const SIDES: &[&str] = &["top", "bottom", "left", "right", "horizontal", "vertical"];

        let (mut top, mut right, mut bottom, mut left) = (None, None, None, None);
        for arg in args {
            // `all_named` guarantees a name is present.
            let Some(name) = &arg.name else { continue };
            let span = arg.value.span();
            let val = self.eval_pure(&arg.value);
            let Some(v) = spacing_value(&val) else {
                self.errors.push(CompileError::AttributeTypeMismatch {
                    span,
                    expected: "a length (an integer)".to_string(),
                });
                continue;
            };
            match name.as_str() {
                "top" => assign_side(&mut top, v, "top", span, &mut self.errors),
                "bottom" => assign_side(&mut bottom, v, "bottom", span, &mut self.errors),
                "left" => assign_side(&mut left, v, "left", span, &mut self.errors),
                "right" => assign_side(&mut right, v, "right", span, &mut self.errors),
                "horizontal" => {
                    assign_side(&mut left, v, "left", span, &mut self.errors);
                    assign_side(&mut right, v, "right", span, &mut self.errors);
                }
                "vertical" => {
                    assign_side(&mut top, v, "top", span, &mut self.errors);
                    assign_side(&mut bottom, v, "bottom", span, &mut self.errors);
                }
                unknown => {
                    let hint = crate::util::closest_match(unknown, SIDES.iter().copied())
                        .map(str::to_string);
                    self.errors.push(CompileError::UnknownAttribute {
                        span,
                        name: unknown.to_string(),
                        hint,
                    });
                }
            }
        }
        Spacing {
            top: top.unwrap_or(0.0),
            right: right.unwrap_or(0.0),
            bottom: bottom.unwrap_or(0.0),
            left: left.unwrap_or(0.0),
        }
    }

    /// Inferred positional spacing forms: pair `(vertical, horizontal)` or quad
    /// CSS `(top, right, bottom, left)`. Any other arity is an error.
    fn resolve_positional_spacing(
        &mut self,
        args: &[Arg],
        span: Span,
        prop: &str,
    ) -> byard_core::atlas::layout::Spacing {
        use byard_core::atlas::layout::Spacing;
        let mut vals = Vec::with_capacity(args.len());
        for arg in args {
            let val = self.eval_pure(&arg.value);
            if let Some(v) = spacing_value(&val) {
                vals.push(v);
            } else {
                self.errors.push(CompileError::AttributeTypeMismatch {
                    span: arg.value.span(),
                    expected: "a length (an integer)".to_string(),
                });
                vals.push(0.0);
            }
        }
        match vals.len() {
            2 => Spacing::symmetric(vals[0], vals[1]),
            4 => Spacing {
                top: vals[0],
                right: vals[1],
                bottom: vals[2],
                left: vals[3],
            },
            n => {
                self.errors.push(CompileError::ArityMismatch {
                    span,
                    name: prop.to_string(),
                    expected: 4,
                    found: n,
                });
                Spacing::default()
            }
        }
    }

    /// Resolves a `radius`-typed attribute into per-corner radii
    /// `[top_left, top_right, bottom_right, bottom_left]` — the exact order
    /// `BoxInstance::radii`/`TextureSampler::radii` expect (`frame.rs`).
    ///
    /// RFC-0005 §"Decoration" documents `radius: Len` as "scalar = all, quad =
    /// per-corner". Accepted forms: a scalar (`radius: 16`, all four
    /// corners) and the positional CSS-order quad (`radius: (4, 8, 12, 16)`).
    /// Unlike `p`/`m`'s generic `Len` contract, there is no pair shorthand and
    /// no named-field form for `radius` — the RFC documents only scalar/quad,
    /// so this resolver doesn't invent additional surface. A non-4 tuple
    /// arity is a `CompileError::ArityMismatch`; a non-numeric corner is an
    /// `AttributeTypeMismatch`; a named field is a `ConflictingSpacingField`
    /// (reusing the existing diagnostic — the message states the real cause).
    /// Resolves the `shadow` attribute into zero or more drop shadows
    /// (RFC-0011 custom shadows). Accepts a preset token (`sm`/`md`/`lg`/`none`),
    /// a single tuple — named `(y: 4, blur: 8, spread: 0, color: 0x…)` or
    /// positional `(x, y, blur, spread, color)` — or an array of tuples for
    /// CSS-style layered shadows.
    fn resolve_shadows(&mut self, attrs: &[Attr]) -> Vec<ShadowSpec> {
        let Some(value) = attrs.iter().find_map(|a| match (&a.name, &a.kind) {
            (n, AttrKind::Prop { value }) if n.as_str() == "shadow" => Some(value),
            _ => None,
        }) else {
            return Vec::new();
        };
        match value {
            // Layered shadows: first-listed paints on top (CSS order), so the
            // caller emits them reversed to sit nearest.
            Expr::Array(items, _) => items
                .iter()
                .filter_map(|e| self.shadow_from_expr(e))
                .collect(),
            other => self.shadow_from_expr(other).into_iter().collect(),
        }
    }

    /// One shadow from a tuple, or a preset token; `None` for `none`/unknown.
    fn shadow_from_expr(&mut self, value: &Expr) -> Option<ShadowSpec> {
        if let Expr::Tuple(args, _) = value {
            return Some(self.shadow_from_tuple(args));
        }
        // A preset token (`sm`/`md`/`lg`); `none`/anything else → no shadow.
        let (dy, blur) = match self.eval_pure(value) {
            Value::Str(t) => match t.as_str() {
                "sm" => (1.0, 3.0),
                "md" => (3.0, 8.0),
                "lg" => (6.0, 16.0),
                _ => return None,
            },
            _ => return None,
        };
        // Preset alpha scales gently with size (sm 0x44 → lg 0x66).
        #[allow(clippy::cast_possible_truncation)]
        let alpha = (0x44 + (blur as i64 - 3) * 2).clamp(0x44, 0x66);
        Some(ShadowSpec {
            dx: 0.0,
            dy,
            blur,
            spread: 0.0,
            color: super::intrinsics::color_to_rgba(alpha << 24, true),
        })
    }

    /// Builds a [`ShadowSpec`] from a `shadow` tuple. Named fields (`x`/`dx`,
    /// `y`/`dy`, `blur`, `spread`, `color`) take any order; a positional tuple
    /// maps by slot `(x, y, blur, spread, color)`, each optional (later slots
    /// default), with `color` always the fifth slot so it is unambiguous.
    fn shadow_from_tuple(&mut self, args: &[Arg]) -> ShadowSpec {
        let mut s = ShadowSpec {
            dx: 0.0,
            dy: 0.0,
            blur: 0.0,
            spread: 0.0,
            color: super::intrinsics::color_to_rgba(DEFAULT_SHADOW_COLOR, true),
        };
        if args.iter().any(|a| a.name.is_some()) {
            for a in args {
                let Some(field) = a.name.as_ref().map(crate::Symbol::as_str) else {
                    continue;
                };
                match field {
                    "x" | "dx" => s.dx = self.eval_num(&a.value),
                    "y" | "dy" => s.dy = self.eval_num(&a.value),
                    "blur" => s.blur = self.eval_num(&a.value),
                    "spread" => s.spread = self.eval_num(&a.value),
                    "color" => s.color = self.eval_shadow_color(&a.value),
                    _ => {}
                }
            }
        } else {
            for (i, a) in args.iter().enumerate() {
                match i {
                    0 => s.dx = self.eval_num(&a.value),
                    1 => s.dy = self.eval_num(&a.value),
                    2 => s.blur = self.eval_num(&a.value),
                    3 => s.spread = self.eval_num(&a.value),
                    4 => s.color = self.eval_shadow_color(&a.value),
                    _ => {}
                }
            }
        }
        s
    }

    /// Evaluates a numeric shadow field (offset/blur/spread) to `f32`.
    #[allow(clippy::cast_possible_truncation)]
    fn eval_num(&mut self, e: &Expr) -> f32 {
        match self.eval_pure(e) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => 0.0,
        }
    }

    /// Evaluates a shadow `color` field (a `0xAARRGGBB` literal) to RGBA.
    fn eval_shadow_color(&mut self, e: &Expr) -> [f32; 4] {
        let packed = self.eval_pure(e).as_int().unwrap_or(DEFAULT_SHADOW_COLOR);
        super::intrinsics::color_to_rgba(packed, true)
    }

    fn resolve_radii(&mut self, attrs: &[Attr], name: &str) -> [f32; 4] {
        let Some(attr) = attrs.iter().find(|a| a.name.as_str() == name) else {
            return [0.0; 4];
        };
        let AttrKind::Prop { value } = &attr.kind else {
            return [0.0; 4];
        };
        match value {
            Expr::Tuple(args, span) => {
                if args.iter().any(|a| a.name.is_some()) {
                    self.errors.push(CompileError::ConflictingSpacingField {
                        span: *span,
                        message: format!(
                            "`{name}` does not accept named corner fields; use a \
                             positional quad (top_left, top_right, bottom_right, \
                             bottom_left)"
                        ),
                    });
                    return [0.0; 4];
                }
                if args.len() != 4 {
                    self.errors.push(CompileError::ArityMismatch {
                        span: *span,
                        name: name.to_string(),
                        expected: 4,
                        found: args.len(),
                    });
                    return [0.0; 4];
                }
                let mut radii = [0.0_f32; 4];
                for (slot, arg) in radii.iter_mut().zip(args) {
                    let val = self.eval_pure(&arg.value);
                    if let Some(v) = spacing_value(&val) {
                        *slot = v;
                    } else {
                        self.errors.push(CompileError::AttributeTypeMismatch {
                            span: arg.value.span(),
                            expected: "a length (an integer)".to_string(),
                        });
                    }
                }
                radii
            }
            other => {
                let val = self.eval_pure(other);
                if let Some(v) = spacing_value(&val) {
                    [v; 4]
                } else {
                    self.errors.push(CompileError::AttributeTypeMismatch {
                        span: other.span(),
                        expected: "a length (an integer)".to_string(),
                    });
                    [0.0; 4]
                }
            }
        }
    }

    /// Resolves the paint-time transform attributes (RFC-0011:
    /// `translate`/`scale`/`rotate`/`origin`; `opacity` stays on its own
    /// existing path — see the doc comment on `DecoratedBox`/`Transform` in
    /// `frame.rs` for why). `rect` is the element's own laid-out rect,
    /// logical pixels, needed to resolve a token/fractional `origin` into an
    /// absolute pivot.
    fn resolve_transform(
        &mut self,
        attrs: &[Attr],
        rect: crate::interp::intrinsics::Rect,
    ) -> byard_core::frame::Transform {
        let translate = self.resolve_axis_pair(attrs, "translate", (0.0, 0.0));
        let scale = self.resolve_axis_pair(attrs, "scale", (1.0, 1.0));
        let rotate = self.resolve_rotate(attrs).unwrap_or(0.0);
        let origin = self.resolve_origin(attrs, rect);
        byard_core::frame::Transform {
            translate: [translate.0, translate.1],
            scale: [scale.0, scale.1],
            rotate,
            origin: [origin.0, origin.1],
            opacity: 1.0,
        }
    }

    /// Resolves a two-axis prop (`translate`/`scale`) to `(x, y)` — RFC-0011's
    /// dual surface: a bare scalar fills both axes; `(a, b)` binds positionally;
    /// `(x: a, y: b)` sets any subset, order-independent, leaving the rest at
    /// `default`. The sub-property form (`name.x: v` / `name.y: v`, a separate
    /// `Attr` with `axis: Some(_)`) then overrides individual axes on top of
    /// whatever the base `name: value` attribute (if any) already resolved —
    /// so `translate.y: 2` alone is exactly `translate: (y: 2)`.
    fn resolve_axis_pair(&mut self, attrs: &[Attr], name: &str, default: (f32, f32)) -> (f32, f32) {
        let mut result = default;
        if let Some(attr) = attrs
            .iter()
            .find(|a| a.name.as_str() == name && a.axis.is_none())
        {
            if let AttrKind::Prop { value } = &attr.kind {
                result = self.resolve_axis_pair_value(value, default);
            }
        }
        for attr in attrs
            .iter()
            .filter(|a| a.name.as_str() == name && a.axis.is_some())
        {
            let AttrKind::Prop { value } = &attr.kind else {
                continue;
            };
            let span = value.span();
            let val = self.eval_pure(value);
            let Some(v) = spacing_value(&val) else {
                self.errors.push(CompileError::AttributeTypeMismatch {
                    span,
                    expected: "a number".to_string(),
                });
                continue;
            };
            let Some(axis) = attr.axis.as_ref() else {
                continue;
            };
            match axis.as_str() {
                "x" => result.0 = v,
                "y" => result.1 = v,
                unknown => {
                    let hint = crate::util::closest_match(unknown, ["x", "y"]).map(String::from);
                    self.errors.push(CompileError::UnknownAttribute {
                        span: attr.span,
                        name: format!("{name}.{unknown}"),
                        hint: hint.map(|h| format!("{name}.{h}")),
                    });
                }
            }
        }
        result
    }

    /// Parses one `translate`/`scale`-shaped [`Expr`] (scalar, positional
    /// tuple, or named tuple) into `(x, y)` — the value-shape half of
    /// [`Self::resolve_axis_pair`], factored out so [`Self::resolve_origin`]
    /// can reuse the exact same tuple grammar for its own fractional pair.
    fn resolve_axis_pair_value(&mut self, value: &Expr, default: (f32, f32)) -> (f32, f32) {
        match value {
            Expr::Tuple(args, span) => {
                let any_named = args.iter().any(|a| a.name.is_some());
                let all_named = args.iter().all(|a| a.name.is_some());
                if any_named && !all_named {
                    self.errors.push(CompileError::ConflictingSpacingField {
                        span: *span,
                        message: "cannot mix named and positional fields".to_string(),
                    });
                    return default;
                }
                if all_named {
                    let (mut x, mut y) = (None, None);
                    for arg in args {
                        let Some(name) = &arg.name else { continue };
                        let span = arg.value.span();
                        let val = self.eval_pure(&arg.value);
                        let Some(v) = spacing_value(&val) else {
                            self.errors.push(CompileError::AttributeTypeMismatch {
                                span,
                                expected: "a number".to_string(),
                            });
                            continue;
                        };
                        match name.as_str() {
                            "x" => assign_side(&mut x, v, "x", span, &mut self.errors),
                            "y" => assign_side(&mut y, v, "y", span, &mut self.errors),
                            unknown => {
                                let hint = crate::util::closest_match(unknown, ["x", "y"])
                                    .map(String::from);
                                self.errors.push(CompileError::UnknownAttribute {
                                    span,
                                    name: unknown.to_string(),
                                    hint,
                                });
                            }
                        }
                    }
                    (x.unwrap_or(default.0), y.unwrap_or(default.1))
                } else if args.len() == 2 {
                    let x = self.eval_pure(&args[0].value);
                    let y = self.eval_pure(&args[1].value);
                    let x = spacing_value(&x).unwrap_or_else(|| {
                        self.errors.push(CompileError::AttributeTypeMismatch {
                            span: args[0].value.span(),
                            expected: "a number".to_string(),
                        });
                        default.0
                    });
                    let y = spacing_value(&y).unwrap_or_else(|| {
                        self.errors.push(CompileError::AttributeTypeMismatch {
                            span: args[1].value.span(),
                            expected: "a number".to_string(),
                        });
                        default.1
                    });
                    (x, y)
                } else {
                    self.errors.push(CompileError::ArityMismatch {
                        span: *span,
                        name: "translate/scale/origin".to_string(),
                        expected: 2,
                        found: args.len(),
                    });
                    default
                }
            }
            other => {
                let val = self.eval_pure(other);
                if let Some(v) = spacing_value(&val) {
                    (v, v)
                } else {
                    self.errors.push(CompileError::AttributeTypeMismatch {
                        span: other.span(),
                        expected: "a number".to_string(),
                    });
                    default
                }
            }
        }
    }

    /// Resolves `rotate` (RFC-0011): the terse `rotate: 90deg` form or the
    /// verbose `rotate: (angle: 90deg)` single-field tuple — both already
    /// canonicalized to radians by the lexer's `Expr::AngleLit`. Absent →
    /// `None` (caller defaults to `0.0`, no rotation).
    fn resolve_rotate(&mut self, attrs: &[Attr]) -> Option<f32> {
        let attr = attrs
            .iter()
            .find(|a| a.name.as_str() == "rotate" && a.axis.is_none())?;
        let AttrKind::Prop { value } = &attr.kind else {
            return None;
        };
        let inner = match value {
            Expr::Tuple(args, _)
                if args.len() == 1
                    && args[0].name.as_ref().map(Symbol::as_str) == Some("angle") =>
            {
                &args[0].value
            }
            other => other,
        };
        let val = self.eval_pure(inner);
        let Some(rad) = spacing_value(&val) else {
            // A non-numeric `rotate` (e.g. `rotate: center`, or a reactive var
            // that didn't resolve to a number) is a real mistake, not a no-op —
            // flag it the same way `translate`/`scale` flag theirs instead of
            // silently painting with no rotation.
            self.errors.push(CompileError::AttributeTypeMismatch {
                span: inner.span(),
                expected: "an angle (e.g. 90deg, 1.5rad)".to_string(),
            });
            return None;
        };
        Some(rad)
    }

    /// Resolves `origin` (RFC-0011 T2) to an absolute logical-pixel pivot in
    /// the same coordinate space as `rect`: a named token (`center` and the
    /// four corners/edges), or a fractional `(fx, fy)` tuple relative to
    /// `rect` (positional or named, reusing [`Self::resolve_axis_pair_value`]'s
    /// tuple grammar). Absent, or an unrecognized token, defaults to `center`
    /// — RFC-0011's own stated default — rather than hard-failing.
    ///
    /// Deliberately out of scope for now: the `px` absolute-origin suffix
    /// (T2's third form) needs a new lexer literal this slice doesn't add;
    /// only the token and fractional forms are implemented.
    fn resolve_origin(
        &mut self,
        attrs: &[Attr],
        rect: crate::interp::intrinsics::Rect,
    ) -> (f32, f32) {
        let center = (rect.x + rect.w * 0.5, rect.y + rect.h * 0.5);
        let Some(attr) = attrs
            .iter()
            .find(|a| a.name.as_str() == "origin" && a.axis.is_none())
        else {
            return center;
        };
        let AttrKind::Prop { value } = &attr.kind else {
            return center;
        };
        if let Expr::Ident(sym, span) = value {
            const TOKENS: &[&str] = &[
                "center",
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
                "top",
                "bottom",
                "left",
                "right",
            ];
            return match sym.as_str() {
                "center" => center,
                "top_left" => (rect.x, rect.y),
                "top_right" => (rect.x + rect.w, rect.y),
                "bottom_left" => (rect.x, rect.y + rect.h),
                "bottom_right" => (rect.x + rect.w, rect.y + rect.h),
                "top" => (rect.x + rect.w * 0.5, rect.y),
                "bottom" => (rect.x + rect.w * 0.5, rect.y + rect.h),
                "left" => (rect.x, rect.y + rect.h * 0.5),
                "right" => (rect.x + rect.w, rect.y + rect.h * 0.5),
                unknown => {
                    let hint = crate::util::closest_match(unknown, TOKENS.iter().copied())
                        .map(String::from);
                    self.errors.push(CompileError::UnknownAttribute {
                        span: *span,
                        name: format!("origin: {unknown}"),
                        hint,
                    });
                    center
                }
            };
        }
        let (fx, fy) = self.resolve_axis_pair_value(value, (0.5, 0.5));
        (rect.x + fx * rect.w, rect.y + fy * rect.h)
    }

    /// Processes a whole `View`: its declarations first (so bindings can resolve
    /// names), then lowers its top-level elements into a render tree, handling
    /// `when`/`for` structural members (M20).
    pub fn lower_view(&mut self, view: &ViewDecl, known_views: &[&str]) -> Vec<RenderNode> {
        // RFC-0018: a fresh tree gets fresh `when`/`for` pools; the previous
        // tree's pool ids are discarded with it (hot-reload re-lowers the tree).
        self.for_pools.clear();
        self.when_pools.clear();
        self.eval_view_decls(view);
        // A view that declares a `content` slot (RFC-0007 D-A) may reference it in
        // its body. When the view is lowered *standalone* — e.g. `byard check`
        // validates each `ViewDecl` independently, or a slot view is a root —
        // there is no calling instance, so push an empty slot frame: the bare
        // `content` reference then splices nothing instead of being mistaken for
        // an `UnknownView`. A real call (`lower_user_view_call`) pushes the
        // caller's block over this before lowering the body.
        let has_content_slot = view
            .params
            .iter()
            .any(|p| p.name.as_str() == RESERVED_CONTENT);
        if has_content_slot {
            self.slot_stack.push(Vec::new());
        }
        let nodes = self.lower_members(&view.body, known_views);
        if has_content_slot {
            self.slot_stack.pop();
        }
        nodes
    }

    // ── lowering ────────────────────────────────────────────────────────

    /// Lowers `expr` to a reactive computation against the current environment.
    fn lower_expr(&mut self, expr: &Expr, payload_name: Option<&Symbol>) -> Lowered {
        match expr {
            Expr::IntLit(n, _) => {
                let n = *n;
                Box::new(move |_| Value::Int(n))
            }
            Expr::FloatLit(f, _) => {
                let f = *f;
                Box::new(move |_| Value::Float(f))
            }
            // Already canonicalized to radians by the lexer (RFC-0011 T1) —
            // from here on an angle is just a plain `Float`.
            Expr::AngleLit(rad, _) => {
                let rad = *rad;
                Box::new(move |_| Value::Float(rad))
            }
            Expr::StrLit(parts, _) => self.lower_strlit(parts, payload_name),
            Expr::Ident(name, span) => {
                // A bare reference to a callback prop (RFC-0019 §4) — reached only
                // when it is *not* the callee of a call (`on_tap()` is handled in
                // `lower_call`) — is invalid: callbacks are fire-and-forget, not
                // first-class values.
                if let Some(&Value::Fn(id)) = self.env.lookup(name) {
                    if self.fn_table.get(id.0 as usize).is_some_and(|e| e.2) {
                        self.errors.push(CompileError::CallbackNotInvocable {
                            span: *span,
                            name: name.as_str().to_string(),
                        });
                    }
                }
                self.lower_ident(name, payload_name)
            }
            Expr::Array(elems, _) => {
                let mut cs: Vec<Lowered> = elems
                    .iter()
                    .map(|e| self.lower_expr(e, payload_name))
                    .collect();
                Box::new(move |ctx| Value::List(cs.iter_mut().map(|c| c(ctx)).collect()))
            }
            Expr::Tuple(elems, _) => {
                let mut cs: Vec<(Option<Symbol>, Lowered)> = elems
                    .iter()
                    .map(|arg| (arg.name.clone(), self.lower_expr(&arg.value, payload_name)))
                    .collect();
                Box::new(move |ctx| {
                    Value::Tuple(
                        cs.iter_mut()
                            .map(|(name, c)| (name.clone(), c(ctx)))
                            .collect(),
                    )
                })
            }
            Expr::Ternary {
                cond, then, els, ..
            } => {
                let mut cc = self.lower_expr(cond, payload_name);
                let mut tc = self.lower_expr(then, payload_name);
                let mut ec = self.lower_expr(els, payload_name);
                Box::new(move |ctx| {
                    if cc(ctx).as_bool().unwrap_or(false) {
                        tc(ctx)
                    } else {
                        ec(ctx)
                    }
                })
            }
            // Binary arithmetic (`+ - * /`, RFC-0020 enabler). Numeric
            // promotion: Int∘Int stays Int (division truncates, like Rust);
            // any Float operand promotes the result to Float. Division by
            // zero yields 0 — the logic thread never panics on user input
            // (INV-4 spirit); a non-numeric operand yields `Unit`, which
            // downstream prop readers treat as "unset".
            Expr::Binary { op, lhs, rhs, .. } => {
                let op = *op;
                let mut lc = self.lower_expr(lhs, payload_name);
                let mut rc = self.lower_expr(rhs, payload_name);
                Box::new(move |ctx| eval_binary(op, lc(ctx), rc(ctx)))
            }
            Expr::Call { callee, args, .. } => self.lower_call(callee, args, payload_name),
            Expr::ClassRef(class, _) => {
                let s = format!(".{class}");
                Box::new(move |_| Value::Str(s.clone()))
            }
            Expr::Postfix { target, op, span } => {
                if let Ok(sig) = self.resolve_var(target, *span) {
                    let op = *op;
                    Box::new(move |ctx| {
                        let cur = ctx.peek_signal(sig).as_int().unwrap_or(0);
                        let new = match op {
                            PostfixOp::Inc => cur + 1,
                            PostfixOp::Dec => cur - 1,
                        };
                        ctx.write_signal(sig, Value::Int(new));
                        Value::Unit
                    })
                } else {
                    Box::new(|_| Value::Unit)
                }
            }
            Expr::Assign {
                target,
                op,
                value,
                span,
            } => {
                if let Ok(sig) = self.resolve_var(target, *span) {
                    let op = *op;
                    let mut rhs = self.lower_expr(value, payload_name);
                    Box::new(move |ctx| {
                        let val = rhs(ctx);
                        let new = match op {
                            AssignOp::Assign => val,
                            AssignOp::Add => {
                                let cur = ctx.peek_signal(sig).as_int().unwrap_or(0);
                                Value::Int(cur + val.as_int().unwrap_or(0))
                            }
                            AssignOp::Sub => {
                                let cur = ctx.peek_signal(sig).as_int().unwrap_or(0);
                                Value::Int(cur - val.as_int().unwrap_or(0))
                            }
                        };
                        ctx.write_signal(sig, new);
                        Value::Unit
                    })
                } else {
                    Box::new(|_| Value::Unit)
                }
            }
            // A `style { … }` value (RFC-0016) is consumed structurally — bound
            // via `let` into the style table and spliced by `..` at lower time
            // (see `register_style`/`expand_style_spreads`) — never projected as
            // a scalar. Reaching here means it was used where a value was
            // expected, which has no meaning; yield Unit.
            Expr::StyleValue { .. } | Expr::Merge { .. } => Box::new(|_| Value::Unit),
            // `value with anim.*(…)` (RFC-0010): lower to the *target* value.
            // The curve is validated by the checker; the `Motion` runtime that
            // actually drives the on-screen transition lands in the follow-up
            // slice, so for now the target resolves instantly (as it did before
            // any `with` was written), which is a safe, correct fallback.
            Expr::Animated { value, .. } => self.lower_expr(value, payload_name),
            // A callback-prop action block (RFC-0019): lower each statement in
            // order and run them in sequence when the callback fires, returning
            // the last statement's value (`Unit` for the no-op default `{}`).
            // Mutations inside route through the reactive system exactly like any
            // event-handler action, because each `Assign`/`Postfix` statement
            // lowers to a signal write.
            Expr::Block(stmts, _) => {
                let mut cs: Vec<Lowered> = stmts
                    .iter()
                    .map(|s| self.lower_expr(s, payload_name))
                    .collect();
                Box::new(move |ctx| {
                    let mut last = Value::Unit;
                    for c in &mut cs {
                        last = c(ctx);
                    }
                    last
                })
            }
            // A `theme.<token>` access (RFC-0022): reads the reactive scheme
            // signal and projects the token's value for the active scheme. Any
            // other member access needs controller metadata (not modeled in
            // Phase 2); lambdas/assignments are actions, not projected values.
            Expr::Member { base, field, span } => self
                .lower_theme_member(base, field, *span)
                .unwrap_or_else(|| Box::new(|_| Value::Unit)),
            Expr::Lambda { .. } | Expr::Error(_) => Box::new(|_| Value::Unit),
        }
    }

    /// Lowers a `theme.<field>` access to a reactive projection (RFC-0022 §1),
    /// or returns `None` when `base` does not resolve to an injected
    /// [`Value::Theme`] (leaving the caller to fall back to `Unit`).
    ///
    /// The returned closure reads the active-scheme signal *tracked*, so any
    /// binding that projects a token re-runs when the scheme flips. Token data
    /// is resolved once, here, and captured by value — the closure never borrows
    /// the interpreter.
    fn lower_theme_member(&mut self, base: &Expr, field: &Symbol, span: Span) -> Option<Lowered> {
        let Expr::Ident(base_name, _) = base else {
            return None;
        };
        let sig = match self.env.lookup(base_name) {
            Some(Value::Theme(sig)) => *sig,
            _ => return None,
        };
        let f = field.as_str();

        // Reserved reactive members: the scheme flag itself.
        if f == "dark" {
            return Some(Box::new(move |ctx| ctx.read_signal(sig)));
        }
        if f == "mode" {
            return Some(Box::new(move |ctx| {
                let dark = ctx.read_signal(sig).as_bool().unwrap_or(false);
                Value::Str(if dark { "dark" } else { "light" }.to_string())
            }));
        }

        // Color tokens differ per scheme → capture both resolved values.
        let light = self.theme.color(f, false);
        let dark = self.theme.color(f, true);
        if light.is_some() || dark.is_some() {
            return Some(Box::new(move |ctx| {
                let is_dark = ctx.read_signal(sig).as_bool().unwrap_or(false);
                let v = if is_dark {
                    dark.or(light)
                } else {
                    light.or(dark)
                };
                Value::Int(v.unwrap_or(0))
            }));
        }

        // Typography tokens: project the size (the current `typo:`/`size:`
        // pipeline is size-only; weight/family land with font byte-loading).
        if let Some(size) = self.theme.typo_size(f) {
            #[allow(clippy::cast_possible_truncation)]
            let size = size as i64;
            // Still read the signal so the binding is theme-scoped and re-runs on
            // a scheme flip (typography can differ per scheme in future themes).
            return Some(Box::new(move |ctx| {
                let _ = ctx.read_signal(sig);
                Value::Int(size)
            }));
        }

        // Shape (corner-radius) tokens.
        if let Some(radius) = self.theme.shape(f) {
            #[allow(clippy::cast_possible_truncation)]
            let radius = radius.round() as i64;
            return Some(Box::new(move |ctx| {
                let _ = ctx.read_signal(sig);
                Value::Int(radius)
            }));
        }

        // A member of a theme that names no known token is a hard error.
        self.errors.push(CompileError::UnknownThemeToken {
            span,
            field: f.to_string(),
            theme: self.theme.name.clone(),
        });
        Some(Box::new(|_| Value::Unit))
    }

    fn lower_ident(&self, name: &Symbol, payload_name: Option<&Symbol>) -> Lowered {
        if let Some(pname) = payload_name {
            if pname == name {
                return Box::new(move |_| {
                    CURRENT_PAYLOAD.with(|cell| cell.borrow().clone().unwrap_or(Value::Unit))
                });
            }
        }
        match name.as_str() {
            "true" => return Box::new(|_| Value::Bool(true)),
            "false" => return Box::new(|_| Value::Bool(false)),
            _ => {}
        }
        match self.env.lookup(name) {
            Some(Value::Signal(sig)) => {
                let sig = *sig;
                Box::new(move |ctx| ctx.read_signal(sig))
            }
            Some(Value::Memo(scope)) => {
                let m = *scope;
                Box::new(move |ctx| ctx.read_memo(m))
            }
            Some(v) => {
                let v = v.clone();
                Box::new(move |_| v.clone())
            }
            // An unresolved identifier is treated as an enum/style token
            // (e.g. `center`, `cover`); intrinsics validate it (M10).
            None => {
                let token = name.as_str().to_string();
                Box::new(move |_| Value::Str(token.clone()))
            }
        }
    }

    fn lower_strlit(&mut self, parts: &[StrPart], payload_name: Option<&Symbol>) -> Lowered {
        enum Part {
            Text(String),
            Interp(Lowered),
        }
        let mut lowered: Vec<Part> = parts
            .iter()
            .map(|p| match p {
                StrPart::Text(t) => Part::Text(t.clone()),
                StrPart::Interp(e) => Part::Interp(self.lower_expr(e, payload_name)),
            })
            .collect();
        Box::new(move |ctx| {
            let mut s = String::new();
            for part in &mut lowered {
                match part {
                    Part::Text(t) => s.push_str(t),
                    Part::Interp(c) => s.push_str(&display_value(&c(ctx))),
                }
            }
            Value::Str(s)
        })
    }

    fn lower_call(
        &mut self,
        callee: &Expr,
        args: &[crate::parser::ast::Arg],
        payload_name: Option<&Symbol>,
    ) -> Lowered {
        // `untrack(expr)` — the reserved escape hatch (D2).
        if let Expr::Ident(name, _) = callee {
            if name.as_str() == "untrack" {
                if let Some(arg) = args.first() {
                    let mut inner = self.lower_expr(&arg.value, payload_name);
                    return Box::new(move |ctx| untrack(|| inner(ctx)));
                }
            }
            // A zero-arg call to a `fn`/`let` memo reads that memo.
            if let Some(Value::Memo(scope)) = self.env.lookup(name) {
                let m = *scope;
                return Box::new(move |ctx| ctx.read_memo(m));
            }
            // Parameterized fn call (M25) *or* callback-prop invocation
            // (RFC-0019 §3): inline the body with args bound as memos. For a
            // callback, the body is the *caller's* action block — still resolved
            // here, where the caller's `var`s remain live below the callee frame
            // in the shared flat env, so `count++` routes to the caller's signal.
            if let Some(Value::Fn(id)) = self.env.lookup(name).cloned() {
                if (id.0 as usize) < self.fn_table.len() {
                    let (params, body, is_callback) = self.fn_table[id.0 as usize].clone();
                    // A callback invoked with the wrong arity is a hard error
                    // (RFC-0019 §4); a plain `fn` keeps the historical lenient
                    // zip (extra args ignored, missing bound to nothing).
                    if is_callback && params.len() != args.len() {
                        self.errors.push(CompileError::CallbackArityMismatch {
                            span: callee.span(),
                            name: name.as_str().to_string(),
                            expected: params.len(),
                            found: args.len(),
                        });
                    }
                    // Bind each arg as a reactive memo so signal reads inside the
                    // body are tracked by the enclosing scope.
                    let snapshot = self.env.len();
                    for (param, arg) in params.iter().zip(args.iter()) {
                        let arg_lowered = self.lower_expr(&arg.value, payload_name);
                        let scope = self.ctx.open_memo(arg_lowered);
                        self.env.push(param.clone(), Value::Memo(scope));
                    }
                    // Lower the body with arg bindings in scope.
                    let body_lowered = self.lower_expr(&body, payload_name);
                    // Restore env.
                    self.env.truncate(snapshot);
                    return body_lowered;
                }
            }
        }
        Box::new(|_| Value::Unit)
    }

    // ── actions (mutations & bare expressions) ──────────────────────────

    /// Evaluates an expression with no reactive scope active (an *action*, not a
    /// projection). Mutations route through the mark cascade; a mutation on a
    /// non-`var` l-value is [`CompileError::NotAssignable`].
    ///
    /// # Errors
    ///
    /// Returns [`CompileError::NotAssignable`] if a mutation targets something
    /// other than a `var`.
    pub fn eval_action(&mut self, expr: &Expr) -> Result<Value, CompileError> {
        match expr {
            Expr::Postfix { target, op, span } => {
                let sig = self.resolve_var(target, *span)?;
                let cur = self.ctx.peek_signal(sig).as_int().unwrap_or(0);
                let new = match op {
                    PostfixOp::Inc => cur + 1,
                    PostfixOp::Dec => cur - 1,
                };
                self.ctx.write_signal(sig, Value::Int(new));
                Ok(Value::Unit)
            }
            Expr::Assign {
                target,
                op,
                value,
                span,
            } => {
                let sig = self.resolve_var(target, *span)?;
                let rhs = self.eval_pure(value);
                let new = match op {
                    AssignOp::Assign => rhs,
                    AssignOp::Add => {
                        let cur = self.ctx.peek_signal(sig).as_int().unwrap_or(0);
                        Value::Int(cur + rhs.as_int().unwrap_or(0))
                    }
                    AssignOp::Sub => {
                        let cur = self.ctx.peek_signal(sig).as_int().unwrap_or(0);
                        Value::Int(cur - rhs.as_int().unwrap_or(0))
                    }
                };
                self.ctx.write_signal(sig, new);
                Ok(Value::Unit)
            }
            other => Ok(self.eval_pure(other)),
        }
    }

    /// Evaluates `expr` once, immediately, with no scope active (so nothing
    /// subscribes). Used to seed `var`s and to evaluate action operands.
    fn eval_pure(&mut self, expr: &Expr) -> Value {
        // A `with` animation (RFC-0010) is driven here, at the single evaluation
        // chokepoint, so every animatable scalar prop (opacity/scale/translate/
        // rotate — all of which resolve through `eval_pure`) animates without
        // per-prop plumbing. A non-animated value takes the ordinary path.
        if let Expr::Animated { value, anim, span } = expr {
            return self.eval_animated(value, anim, *span);
        }
        let mut compute = self.lower_expr(expr, None);
        compute(&mut self.ctx)
    }

    /// Drives one `with` animation (RFC-0010): resolves the target and curve,
    /// advances (or seeds) the persisted [`Motion`](byard_core::frame::Motion)
    /// keyed by `key`, and returns the value sampled at the current engine time.
    /// A target change reseeds `from` to the current on-screen value so a
    /// mid-flight reversal is continuous.
    fn eval_animated(&mut self, target: &Expr, anim: &Expr, key: Span) -> Value {
        let target_val = match self.eval_pure(target) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            // A non-numeric target can't be interpolated — pass it through
            // untouched (the checker already restricts `with` to numeric props).
            other => return other,
        };
        // No advancing clock (a host that never calls `set_now_ms`, e.g. a
        // non-animating test path): resolve straight to the target so an
        // animation can never latch `has_active_animations` on `t = 0` forever.
        if !self.clock_set {
            return Value::Float(f64::from(target_val));
        }
        let Ok(curve) = crate::interp::anim::resolve_curve(anim) else {
            // The checker already reported this; render the target inertly.
            return Value::Float(f64::from(target_val));
        };
        let packed = pack_curve(curve);
        let now = self.now_ms;
        let motion = self
            .animations
            .entry(key)
            .or_insert_with(|| byard_core::frame::Motion {
                from: target_val,
                to: target_val,
                start_ms: now,
                curve: packed,
            });
        // Retarget on a goal change: reseed `from` to where the property
        // actually is right now (interruptible spring), restart the clock.
        if (motion.to - target_val).abs() > f32::EPSILON {
            let current = motion.sample(now);
            motion.from = current;
            motion.to = target_val;
            motion.start_ms = now;
        }
        // Keep the curve in sync (a hot-reload may have edited it).
        motion.curve = packed;
        let sampled = motion.sample(now);
        // `Motion::DEFAULT_EPS_*` are pixel-scaled (0.5), far too loose for the
        // ratio/opacity/radian props that also animate through this one generic
        // path — with them an ease-out could read "settled" while still visibly
        // short of the target. Use tight, unit-agnostic epsilons: position is
        // the final-value accuracy gate; the velocity gate keeps a spring's
        // overshoot alive instead of freezing it at the first target crossing.
        let settled = motion.is_settled_with_eps(now, ANIM_SETTLE_EPS_POS, ANIM_SETTLE_EPS_VEL);
        if !settled {
            self.any_active = true;
        }
        Value::Float(f64::from(sampled))
    }

    fn resolve_var(&self, target: &Expr, span: Span) -> Result<SignalId, CompileError> {
        if let Expr::Ident(name, _) = target {
            if let Some(Value::Signal(sig)) = self.env.lookup(name) {
                return Ok(*sig);
            }
        }
        // `theme.dark = …` writes the reactive scheme signal (RFC-0022 §1), so a
        // scheme flip drives Mark-and-Pull across every token reference.
        if let Some(sig) = self.resolve_theme_scheme_target(target) {
            return Ok(sig);
        }
        Err(CompileError::NotAssignable { span })
    }

    /// Resolves `theme.dark` (the assignable/bindable scheme flag) to its backing
    /// scheme signal, or `None` if `target` is not that member (RFC-0022 §1).
    fn resolve_theme_scheme_target(&self, target: &Expr) -> Option<SignalId> {
        let Expr::Member { base, field, .. } = target else {
            return None;
        };
        if field.as_str() != "dark" {
            return None;
        }
        let Expr::Ident(base_name, _) = base.as_ref() else {
            return None;
        };
        match self.env.lookup(base_name) {
            Some(Value::Theme(sig)) => Some(*sig),
            _ => None,
        }
    }

    /// Lowers an action expression to an event handler closure, capturing any optional payload bindings.
    ///
    /// # Errors
    ///
    /// Returns a [`CompileError`] if variable resolution or assignment validation fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn lower_action(
        &mut self,
        expr: &Expr,
        payload_name: Option<Symbol>,
    ) -> Result<Action, CompileError> {
        let mut compute = self.lower_expr(expr, payload_name.as_ref());
        Ok(Box::new(move |ctx, payload| {
            CURRENT_PAYLOAD.with(|cell| {
                *cell.borrow_mut() = payload.cloned();
            });
            let _ = compute(ctx);
            CURRENT_PAYLOAD.with(|cell| {
                *cell.borrow_mut() = None;
            });
        }))
    }

    /// Reads a scroll offset signal as an `f32` (an `Int` or `Float` `var`);
    /// anything else reads as the origin.
    fn peek_scroll(&self, sig: SignalId) -> f32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        match self.peek(sig) {
            Value::Int(n) => n as f32,
            Value::Float(f) => f as f32,
            _ => 0.0,
        }
    }

    /// Writes `value` back to a scroll offset signal, preserving its `Int`/`Float`
    /// kind so a whole-pixel `var off: Int` never becomes a `Float` mid-scroll.
    fn write_scroll(&mut self, sig: SignalId, value: f32) {
        #[allow(clippy::cast_possible_truncation)]
        let v = match self.peek(sig) {
            Value::Int(_) => Value::Int(value.round() as i64),
            _ => Value::Float(f64::from(value)),
        };
        self.write_var(sig, v);
    }

    /// Nudges one scroll axis by `delta` logical px (wheel/trackpad), clamped to
    /// `[0, max]`. A forward delta reveals earlier content, so the offset shrinks.
    fn nudge_scroll(&mut self, axis: ScrollAxis, delta: f32) {
        let next = (self.peek_scroll(axis.sig) - delta).clamp(0.0, axis.max);
        self.write_scroll(axis.sig, next);
    }

    /// The scrollable axis of a target and its viewport extent (the one with
    /// travel — `max > 0`), vertical preferred. RFC-0021 snap/pagination helper.
    fn scrollable_axis(t: &ScrollTarget) -> Option<(ScrollAxis, f32)> {
        t.y.filter(|a| a.max > 0.0)
            .map(|a| (a, t.rect.h))
            .or_else(|| t.x.filter(|a| a.max > 0.0).map(|a| (a, t.rect.w)))
    }

    /// Reflects the current page for `t` from its offset (`round(offset /
    /// viewport)`), writing the `page:` var and firing `page_change` on a change.
    /// Runs continuously (every scroll) so pagination tracks wheel/trackpad
    /// scrolling, not just snap settles (RFC-0021).
    fn reflect_page(&mut self, t: &ScrollTarget) {
        let (Some(psig), Some(elem)) = (t.page_sig, t.elem) else {
            return;
        };
        let Some((axis, vp)) = Self::scrollable_axis(t) else {
            return;
        };
        if vp <= 0.0 {
            return;
        }
        #[allow(clippy::cast_possible_truncation)]
        let page = (self.peek_scroll(axis.sig) / vp).round() as i64;
        if self.ctx.peek_signal(psig).as_int() != Some(page) {
            self.ctx.write_signal(psig, Value::Int(page));
            self.router.fire_event(
                &mut self.ctx,
                elem,
                super::events::EventKind::PageChange,
                Some(&Value::Int(page)),
            );
        }
        // Treat our reflection as the synced value so `page`→offset never fights.
        self.scroll_page_last.insert(elem, page);
    }

    /// Begins a smooth snap of `t`'s scrollable axis to its nearest page: reflects
    /// `page` right away (the indicator jumps to the destination as the glide
    /// starts) and seeds a spring from the current offset to the page boundary
    /// (RFC-0021 §2, RFC-0010 spring). [`advance_snap_anims`](Self::advance_snap_anims)
    /// drives it to rest and fires `scroll_end`. When no clock is advancing (a
    /// non-animating host or a test), it resolves to the boundary instantly so the
    /// behaviour is identical minus the animation. Shared by drag-release and the
    /// scroll-quiet settle; a no-op if already resting on a page.
    fn begin_snap(&mut self, t: &ScrollTarget) {
        /// Sub-pixel tolerance for "already on a page".
        const EPS: f32 = 0.5;
        if !t.snap_page {
            return;
        }
        let (Some((axis, vp)), Some(elem)) = (Self::scrollable_axis(t), t.elem) else {
            return;
        };
        if vp <= 0.0 {
            return;
        }
        let cur = self.peek_scroll(axis.sig);
        let target = ((cur / vp).round() * vp).clamp(0.0, axis.max);
        if (cur - target).abs() < EPS {
            self.snap_anims.remove(&elem);
            return; // already on a page — nothing to glide
        }
        // Destination page is known now — reflect it immediately (fires
        // `page_change`) so pagination leads the glide, and settle it exactly once
        // the spring (or the instant fallback) arrives.
        self.reflect_page(t);
        if self.clock_set {
            let curve = pack_curve(crate::interp::anim::Curve::DEFAULT_SPRING);
            self.snap_anims.insert(
                elem,
                SnapAnim {
                    sig: axis.sig,
                    motion: byard_core::frame::Motion {
                        from: cur,
                        to: target,
                        start_ms: self.now_ms,
                        curve,
                    },
                    target,
                },
            );
            self.any_active = true;
        } else {
            self.finish_snap(elem, axis.sig, target);
        }
    }

    /// Pins `sig` to the exact page `target`, clears any pending glide, and fires
    /// `scroll_end` for `elem` (RFC-0021 snap completion).
    fn finish_snap(&mut self, elem: u32, sig: SignalId, target: f32) {
        self.snap_anims.remove(&elem);
        self.write_scroll(sig, target);
        self.router.fire_event(
            &mut self.ctx,
            elem,
            super::events::EventKind::ScrollEnd,
            None,
        );
    }

    /// Advances every in-flight snap spring one `render` (RFC-0021 smooth snap):
    /// samples each [`SnapAnim`](SnapAnim) at the engine clock, writes the offset,
    /// and — once the spring settles — pins the offset exactly on the page and
    /// fires `scroll_end`. A live drag on an elem cancels its glide (the finger
    /// takes over). Keeps `any_active` set while any spring is still moving so the
    /// host keeps presenting frames until it rests.
    fn advance_snap_anims(&mut self) {
        /// Pixel/velocity settle gates for a scroll offset (sub-pixel is
        /// imperceptible; the exact target is pinned on settle regardless).
        const EPS_POS: f32 = 0.5;
        const EPS_VEL: f32 = 2.0;
        if self.snap_anims.is_empty() {
            return;
        }
        let drag_elem = self.scroll_drag.and_then(|d| d.elem);
        let now = self.now_ms;
        for (elem, anim) in self
            .snap_anims
            .iter()
            .map(|(e, a)| (*e, *a))
            .collect::<Vec<_>>()
        {
            if drag_elem == Some(elem) {
                self.snap_anims.remove(&elem); // the finger reclaimed this view
                continue;
            }
            if anim.motion.is_settled_with_eps(now, EPS_POS, EPS_VEL) {
                self.finish_snap(elem, anim.sig, anim.target);
            } else {
                self.write_scroll(anim.sig, anim.motion.sample(now));
                self.any_active = true;
            }
        }
    }

    /// On drag release, snap the `ScrollView` under the press point (RFC-0021).
    fn snap_scroll_on_release(&mut self, start_pos: (f32, f32)) {
        let (px, py) = start_pos;
        let Some(t) = self
            .scroll_targets
            .iter()
            .rev()
            .find(|t| {
                px >= t.rect.x
                    && px < t.rect.x + t.rect.w
                    && py >= t.rect.y
                    && py < t.rect.y + t.rect.h
            })
            .copied()
        else {
            return;
        };
        self.begin_snap(&t);
    }

    /// Reflects `page` for every scroll target after this tick's scroll writes
    /// (RFC-0021 continuous pagination).
    fn reflect_pages(&mut self) {
        let targets: Vec<ScrollTarget> = self.scroll_targets.clone();
        for t in targets {
            self.reflect_page(&t);
        }
    }

    /// RFC-0021 snap settle: once a `snap`-enabled `ScrollView` has gone quiet —
    /// no wheel/trackpad scroll input for [`SETTLE_FRAMES`] renders (so trackpad
    /// momentum, a stream of shrinking deltas, cannot trigger a snap mid-fling
    /// that fights the next event) — glide its offset to the nearest page via
    /// [`begin_snap`](Self::begin_snap). Frame-counted, not clock-based, so it
    /// settles identically whether or not the host advances `now_ms`. A live drag
    /// never settles on stillness (it snaps on release). Runs each `render`, over
    /// the previous frame's targets.
    fn settle_snaps(&mut self) {
        /// Renders of quiet (no scroll input) before a `snap: page` view settles.
        const SETTLE_FRAMES: u64 = 4;
        /// Sub-pixel tolerance for "already on a page".
        const EPS: f32 = 0.5;
        let drag_elem = self.scroll_drag.and_then(|d| d.elem);
        let targets: Vec<ScrollTarget> = self.scroll_targets.clone();
        for t in targets {
            let Some(elem) = t.elem else { continue };
            // A drag in progress, or a glide already running, owns the offset.
            if !t.snap_page || drag_elem == Some(elem) || self.snap_anims.contains_key(&elem) {
                continue;
            }
            let Some((axis, vp)) = Self::scrollable_axis(&t) else {
                continue;
            };
            if vp <= 0.0 {
                continue;
            }
            let cur = self.peek_scroll(axis.sig);
            let boundary = ((cur / vp).round() * vp).clamp(0.0, axis.max);
            if (cur - boundary).abs() < EPS {
                continue; // already resting on a page
            }
            // Off a boundary: wait for the scroll to go quiet, then glide. `quiet`
            // is how many renders since the last scroll input touched this elem;
            // momentum keeps resetting it, so we only snap once the fling ends.
            let quiet = self
                .scroll_quiet
                .get(&elem)
                .map_or(SETTLE_FRAMES, |last| self.frame_seq.saturating_sub(*last));
            if quiet >= SETTLE_FRAMES {
                self.begin_snap(&t);
            } else {
                self.any_active = true; // keep frames coming until it goes quiet
            }
        }
    }

    /// RFC-0021 reflected `page:` (the reverse direction): when the app sets the
    /// `page` var, scroll the `ScrollView`'s offset to that page. Edge-triggered
    /// against [`scroll_page_last`](Self::scroll_page_last) so it fires only on an
    /// external change (a drag never writes `page` mid-gesture, and our own snap
    /// updates the tracker), never level-triggered against the live offset — so it
    /// can't fight scrolling. Runs at the top of `render` over the previous
    /// frame's targets.
    fn sync_page_offsets(&mut self) {
        let targets: Vec<ScrollTarget> = self.scroll_targets.clone();
        for t in targets {
            let (Some(psig), Some(elem)) = (t.page_sig, t.elem) else {
                continue;
            };
            let Some(page) = self.ctx.peek_signal(psig).as_int() else {
                continue;
            };
            if self.scroll_page_last.get(&elem) == Some(&page) {
                continue; // no external change since we last synced
            }
            let axis_vp =
                t.y.filter(|a| a.max > 0.0)
                    .map(|a| (a, t.rect.h))
                    .or_else(|| t.x.filter(|a| a.max > 0.0).map(|a| (a, t.rect.w)));
            if let Some((axis, vp)) = axis_vp {
                #[allow(clippy::cast_precision_loss)]
                let target = (page.max(0) as f32 * vp).clamp(0.0, axis.max);
                self.write_scroll(axis.sig, target);
            }
            self.scroll_page_last.insert(elem, page);
        }
    }

    /// RFC-0021 `on_end_reached`: fires `end_reached` once for any `ScrollView`
    /// whose visible bottom has crossed `end_threshold` of its content, debounced
    /// via [`end_reached_fired`](Self::end_reached_fired) until the offset falls
    /// back below the threshold (so appending items re-arms it).
    fn fire_end_reached(&mut self) {
        let targets: Vec<ScrollTarget> = self.scroll_targets.clone();
        for t in targets {
            let (Some(threshold), Some(elem)) = (t.end_threshold, t.elem) else {
                continue;
            };
            let axis_vp =
                t.y.filter(|a| a.max > 0.0)
                    .map(|a| (a, t.rect.h))
                    .or_else(|| t.x.filter(|a| a.max > 0.0).map(|a| (a, t.rect.w)));
            let Some((axis, vp)) = axis_vp else { continue };
            if axis.max <= 0.0 {
                continue;
            }
            let frac = (self.peek_scroll(axis.sig) + vp) / (axis.max + vp);
            if frac >= threshold {
                if self.end_reached_fired.insert(elem) {
                    self.router.fire_event(
                        &mut self.ctx,
                        elem,
                        super::events::EventKind::EndReached,
                        None,
                    );
                }
            } else {
                self.end_reached_fired.remove(&elem);
            }
        }
    }

    /// Snapshots one axis of a drag at the press: its live offset becomes the
    /// baseline the pointer travel is subtracted from (RFC-0005, IMPL-10).
    fn capture_drag_axis(&self, axis: ScrollAxis) -> ScrollDragAxis {
        let is_int = matches!(self.peek(axis.sig), Value::Int(_));
        ScrollDragAxis {
            sig: axis.sig,
            start_offset: self.peek_scroll(axis.sig),
            max: axis.max,
            is_int,
        }
    }

    /// Applies a drag's pointer `travel` (current − press) to one axis: the
    /// content follows the pointer, so the offset is the press offset minus the
    /// travel, clamped to `[0, max]` (RFC-0005 drag-to-scroll).
    fn write_drag_axis(&mut self, axis: ScrollDragAxis, travel: f32) {
        let next = (axis.start_offset - travel).clamp(0.0, axis.max);
        let value = if axis.is_int {
            #[allow(clippy::cast_possible_truncation)]
            Value::Int(next.round() as i64)
        } else {
            Value::Float(f64::from(next))
        };
        self.write_var(axis.sig, value);
    }

    /// Converts winit-sourced input events to interpreter event payloads and dispatches them to the `EventRouter`.
    pub fn dispatch_events(&mut self, events: &[byard_core::InputEvent]) {
        use crate::interp::events::{EventKind as CompKind, InputEvent as CompEvent};
        use byard_core::platform::{EventKind as CoreKind, InputPayload};

        /// Logical pixels a `ScrollView` scrolls per wheel line (RFC-0005).
        const WHEEL_LINE_PX: f32 = 40.0;

        let comp_events: Vec<CompEvent> = events
            .iter()
            .map(|ev| {
                let kind = match ev.kind {
                    CoreKind::PointerDown => CompKind::PointerDown,
                    CoreKind::PointerUp => CompKind::PointerUp,
                    CoreKind::Tap => CompKind::Tap,
                    CoreKind::PointerMove => CompKind::PointerMove,
                    CoreKind::Scroll => CompKind::Scroll,
                    CoreKind::Wheel => CompKind::Wheel,
                    CoreKind::Change => CompKind::Change,
                    CoreKind::KeyDown => CompKind::KeyDown,
                    CoreKind::KeyUp => CompKind::KeyUp,
                    CoreKind::TextInput => CompKind::TextInput,
                    CoreKind::PointerEnter => CompKind::PointerEnter,
                    CoreKind::PointerExit => CompKind::PointerExit,
                    CoreKind::Hover => CompKind::Hover,
                    CoreKind::LongPress => CompKind::LongPress,
                    CoreKind::DoubleTap => CompKind::DoubleTap,
                    CoreKind::Secondary => CompKind::Secondary,
                };
                let value = ev.payload.as_ref().map(|p| match p {
                    InputPayload::Str(s) => Value::Str(s.clone()),
                    InputPayload::Bool(b) => Value::Bool(*b),
                    InputPayload::Float(f) => Value::Float(f64::from(*f)),
                    InputPayload::Key(k) => Value::Str(k.clone()),
                });
                CompEvent {
                    kind,
                    pos: ev.pos,
                    delta: ev.delta,
                    value,
                    time_ms: ev.time_ms,
                }
            })
            .collect();

        // RFC-0005 `ScrollView` wheel: a wheel/scroll over a recorded scroll
        // target nudges whichever of `offset.x`/`offset.y` is writable, each
        // clamped to `[0, content − viewport]`. Wheel deltas are line-based (× a
        // per-line step); trackpad `Scroll` deltas are already pixels. Done here,
        // before the render, so the same tick paints the new offset (paint-time
        // translate, no relayout — INV-8).
        for ev in events {
            let step = match ev.kind {
                CoreKind::Wheel => WHEEL_LINE_PX,
                CoreKind::Scroll => 1.0,
                _ => continue,
            };
            let (px, py) = ev.pos;
            let Some(t) = self
                .scroll_targets
                .iter()
                .rev()
                .find(|t| {
                    px >= t.rect.x
                        && px < t.rect.x + t.rect.w
                        && py >= t.rect.y
                        && py < t.rect.y + t.rect.h
                })
                .copied()
            else {
                continue;
            };
            // Wheel forward (delta > 0) reveals earlier content → offset shrinks.
            if let Some(axis) = t.x {
                self.nudge_scroll(axis, ev.delta.0 * step);
            }
            if let Some(axis) = t.y {
                self.nudge_scroll(axis, ev.delta.1 * step);
            }
            // RFC-0021: mark this elem freshly scrolled and cancel any in-flight
            // snap glide — the user is driving again, so `settle_snaps` restarts
            // its quiet countdown and only snaps once the fling truly ends.
            if let Some(elem) = t.elem {
                self.scroll_quiet.insert(elem, self.frame_seq);
                self.snap_anims.remove(&elem);
            }
        }

        // RFC-0005 `ScrollView` drag-to-scroll: a pointer press on inert scroll
        // content starts a drag; each move slides the offset (on every writable
        // axis) so the content tracks the pointer — a pure function of the
        // press-relative travel, no accumulated drift (IMPL-10); release ends it.
        // The press defers to interactive children via `claims_pointer`, so a
        // button or slider inside the list still wins its own gesture.
        for ev in events {
            match ev.kind {
                CoreKind::PointerDown => {
                    let (px, py) = ev.pos;
                    let target = if self.router.claims_pointer(ev.pos) {
                        None
                    } else {
                        self.scroll_targets
                            .iter()
                            .rev()
                            .find(|t| {
                                px >= t.rect.x
                                    && px < t.rect.x + t.rect.w
                                    && py >= t.rect.y
                                    && py < t.rect.y + t.rect.h
                            })
                            .copied()
                    };
                    // A press reclaims the view — cancel any in-flight snap glide
                    // so the finger, not the spring, owns the offset (RFC-0021).
                    if let Some(elem) = target.and_then(|t| t.elem) {
                        self.snap_anims.remove(&elem);
                    }
                    self.scroll_drag = target.map(|t| ScrollDrag {
                        start_pos: (px, py),
                        x: t.x.map(|a| self.capture_drag_axis(a)),
                        y: t.y.map(|a| self.capture_drag_axis(a)),
                        elem: t.elem,
                    });
                }
                CoreKind::PointerMove => {
                    if let Some(d) = self.scroll_drag {
                        if let Some(a) = d.x {
                            let travel = ev.pos.0 - d.start_pos.0;
                            self.write_drag_axis(a, travel);
                        }
                        if let Some(a) = d.y {
                            let travel = ev.pos.1 - d.start_pos.1;
                            self.write_drag_axis(a, travel);
                        }
                    }
                }
                CoreKind::PointerUp | CoreKind::Tap => {
                    // RFC-0021 snap: on release, settle the offset to the nearest
                    // page for a `snap: page` ScrollView (before clearing the drag).
                    if let Some(d) = self.scroll_drag {
                        self.snap_scroll_on_release(d.start_pos);
                    }
                    self.scroll_drag = None;
                }
                _ => {}
            }
        }

        // RFC-0021: after this tick's scroll writes, reflect `page` continuously
        // (so pagination tracks wheel/trackpad scrolling, not just snap settles)
        // and fire `on_end_reached` for anything past its `end_threshold`.
        self.reflect_pages();
        self.fire_end_reached();

        self.router
            .dispatch_tick(&mut self.ctx, Some(&self.atlas), comp_events);
    }
}

/// One `Overlay`'s built layout (RFC-0017): its absolute wrapper node plus a
/// per-child emission slot. Holds borrows into the frozen render tree, so its
/// lifetime is scoped to a single [`Interpreter::render`] call.
struct OverlayLayout<'a> {
    /// The `RenderNode::Overlay` this describes (source of `attrs`/`children`).
    node: &'a RenderNode,
    /// The absolute wrapper container in the atlas; its node index doubles as
    /// the modal scrim's element id.
    wrapper_id: byard_core::atlas::layout::AtlasNodeId,
    /// One slot per built child, in declaration order.
    children: Vec<OverlayChildSlot<'a>>,
}

/// A single overlay child ready to emit (RFC-0017): the child render node, its
/// atlas id, and the flat-id list its render walk consumes.
struct OverlayChildSlot<'a> {
    node: &'a RenderNode,
    id: byard_core::atlas::layout::AtlasNodeId,
    flat_ids: Vec<byard_core::atlas::layout::AtlasNodeId>,
}

/// The absolute, inset-0 anchor wrapper style for an overlay child (RFC-0017
/// §Positioning). Direction is `Column`, so `justify` drives the vertical edge
/// and `align` the horizontal one. An unanchored child keeps the default
/// (`Start`/`Stretch`), so a `grow` scrim fills the viewport; an anchored child
/// is pinned to the requested edge/centre.
fn anchor_wrapper_style(anchor: Option<&str>) -> byard_core::atlas::layout::ContainerStyle {
    use byard_core::atlas::layout::{Align, ContainerStyle, FlexDir, Justify};
    let mut style = ContainerStyle::default()
        .with_absolute(true)
        .with_direction(FlexDir::Column);
    let (justify, align) = match anchor {
        Some("center") => (Some(Justify::Center), Some(Align::Center)),
        Some("top") => (Some(Justify::Start), Some(Align::Center)),
        Some("bottom") => (Some(Justify::End), Some(Align::Center)),
        Some("start") => (Some(Justify::Center), Some(Align::Start)),
        Some("end") => (Some(Justify::Center), Some(Align::End)),
        // No anchor (a scrim): keep flow defaults so `grow` fills the viewport.
        _ => (None, None),
    };
    if let Some(j) = justify {
        style = style.with_justify(j);
    }
    if let Some(a) = align {
        style = style.with_align(a);
    }
    style
}

/// Renders a value for string interpolation (`"Count: {count}"`).
/// Coerces a spacing side/scalar value to `f32`; only numeric values are valid
/// `Len`s (a non-numeric side is a `TypeMismatch`).
fn spacing_value(v: &Value) -> Option<f32> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    match v {
        Value::Int(n) => Some(*n as f32),
        Value::Float(f) => Some(*f as f32),
        _ => None,
    }
}

/// Inserts `attr` into a resolved style set, replacing any existing attribute
/// with the same name and sub-property axis (last-wins) or appending it — so a
/// spread/inline override cleanly supersedes an earlier value (RFC-0016).
fn override_attr(set: &mut Vec<Attr>, attr: Attr) {
    if let Some(existing) = set
        .iter_mut()
        .find(|a| a.name == attr.name && a.axis == attr.axis)
    {
        *existing = attr;
    } else {
        set.push(attr);
    }
}

/// Builds a flat attribute list for *validation only* (RFC-0016): the base
/// attributes followed by every `on <state>` block's attributes, so a state
/// block's `bg:`/`scale:`/… is checked against the intrinsic's §5 contract just
/// like an inline attribute. Never emitted — rendering keeps base and states
/// separate so states resolve per-frame against the live mask.
fn attrs_with_states(base: &[Attr], states: &[StateBlock]) -> Vec<Attr> {
    if states.is_empty() {
        return base.to_vec();
    }
    let mut all = base.to_vec();
    for sb in states {
        all.extend(sb.attrs.iter().cloned());
    }
    all
}

/// The `StyleState` bit a single [`StyleStateKind`] maps to (RFC-0024).
fn state_bit(kind: StyleStateKind) -> crate::interp::events::StyleState {
    use crate::interp::events::StyleState;
    match kind {
        StyleStateKind::Hover => StyleState::HOVER,
        StyleStateKind::Pressed => StyleState::PRESSED,
        StyleStateKind::Focused => StyleState::FOCUSED,
        StyleStateKind::Disabled => StyleState::DISABLED,
        StyleStateKind::Checked => StyleState::CHECKED,
        StyleStateKind::Selected => StyleState::SELECTED,
        StyleStateKind::Invalid => StyleState::INVALID,
        StyleStateKind::Indeterminate => StyleState::INDETERMINATE,
        StyleStateKind::Dragging => StyleState::DRAGGING,
    }
}

/// The combined-selector mask a state block requires (RFC-0024): every state in
/// `states` must be active for the block to apply.
fn state_block_mask(sb: &StateBlock) -> crate::interp::events::StyleState {
    sb.states
        .iter()
        .fold(crate::interp::events::StyleState::empty(), |m, &k| {
            m.union(state_bit(k))
        })
}

/// Resolves an element's effective attributes for the current interaction state
/// (RFC-0016 §"Resolution order", extended by RFC-0024 §2): a block applies when
/// its required mask is a **subset** of the live `StyleState` (all its states are
/// active). Matching blocks overlay the base last-wins, ordered by **specificity**
/// (number of states — a combined `on focused+hover` beats a single `on hover`)
/// then **declaration order** for equal specificity.
///
/// The common stateless case (no blocks) borrows the base with no allocation.
fn resolve_state_attrs<'a>(
    base: &'a [Attr],
    state_blocks: &[StateBlock],
    active: crate::interp::events::StyleState,
) -> std::borrow::Cow<'a, [Attr]> {
    if state_blocks.is_empty() {
        return std::borrow::Cow::Borrowed(base);
    }
    // Collect blocks whose full mask is active, tagged with (specificity, order).
    let mut matching: Vec<(u32, usize)> = state_blocks
        .iter()
        .enumerate()
        .filter_map(|(i, sb)| {
            let required = state_block_mask(sb);
            active.contains(required).then_some((required.count(), i))
        })
        .collect();
    if matching.is_empty() {
        return std::borrow::Cow::Borrowed(base);
    }
    // Apply lowest-specificity first, then declaration order, so a more specific
    // (or later) block wins on conflicting properties — the `(spec, idx)` tuples
    // sort lexicographically.
    matching.sort_unstable();
    let mut resolved = base.to_vec();
    for (_, idx) in matching {
        for a in &state_blocks[idx].attrs {
            override_attr(&mut resolved, a.clone());
        }
    }
    std::borrow::Cow::Owned(resolved)
}

/// Multiplies a colour's alpha by `opacity` — folds an element's effective
/// opacity into the widget/text primitives it emits so a translucent control
/// dims as a whole, not just its background (RFC-0011 T4 approximation).
/// Evaluates one binary arithmetic operation (`+ - * /`, RFC-0020 enabler)
/// with numeric promotion: Int∘Int → Int (division truncates), any Float
/// operand → Float. Division by zero yields the zero of the promoted type and
/// a non-numeric operand yields [`Value::Unit`] — the logic thread never
/// panics on user expressions. Pure and unit-testable.
fn eval_binary(op: BinOp, lhs: Value, rhs: Value) -> Value {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => Value::Int(match op {
            BinOp::Add => a.wrapping_add(b),
            BinOp::Sub => a.wrapping_sub(b),
            BinOp::Mul => a.wrapping_mul(b),
            BinOp::Div => {
                if b == 0 {
                    0
                } else {
                    a.wrapping_div(b)
                }
            }
        }),
        (Value::Int(a), Value::Float(b)) => eval_binary_f(op, a as f64, b),
        (Value::Float(a), Value::Int(b)) => eval_binary_f(op, a, b as f64),
        (Value::Float(a), Value::Float(b)) => eval_binary_f(op, a, b),
        _ => Value::Unit,
    }
}

/// The Float leg of [`eval_binary`]. `x / 0.0` yields `0.0`, not an
/// IEEE infinity/NaN — a NaN sweep or width would poison layout and paint.
fn eval_binary_f(op: BinOp, a: f64, b: f64) -> Value {
    Value::Float(match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => {
            if b == 0.0 {
                0.0
            } else {
                a / b
            }
        }
    })
}

fn dim_alpha(mut color: [f32; 4], opacity: f32) -> [f32; 4] {
    color[3] *= opacity;
    color
}

/// Converts a packed `0xRRGGBB` colour to OKLab `[L, a, b]` for perceptually
/// uniform interpolation (RFC-0010 A3).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)] // standard colour-space notation
fn oklab_from_hex(hex: i64) -> [f32; 3] {
    let r = srgb_to_linear(((hex >> 16) & 0xFF) as f32 / 255.0);
    let g = srgb_to_linear(((hex >> 8) & 0xFF) as f32 / 255.0);
    let b = srgb_to_linear((hex & 0xFF) as f32 / 255.0);
    // Björn Ottosson's linear-sRGB → OKLab.
    let l = 0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
    let (l_, m_, s_) = (l.cbrt(), m.cbrt(), s.cbrt());
    [
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    ]
}

/// Converts OKLab `[L, a, b]` back to a packed `0xRRGGBB` colour, clamping any
/// out-of-gamut result (a spring can overshoot a channel).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)] // standard colour-space notation
fn hex_from_oklab(lab: [f32; 3]) -> i64 {
    let [big_l, a, b] = lab;
    let l_ = big_l + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = big_l - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_ = big_l - 0.089_484_18 * a - 1.291_485_5 * b;
    let (l, m, s) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    let r = 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s;
    let g = -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s;
    let bl = -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s;
    let to_byte = |c: f32| -> i64 { (linear_to_srgb(c).clamp(0.0, 1.0) * 255.0).round() as i64 };
    (to_byte(r) << 16) | (to_byte(g) << 8) | to_byte(bl)
}

/// sRGB gamma → linear (per channel).
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear → sRGB gamma (per channel).
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Packs the compiler's typed [`Curve`](crate::interp::anim::Curve) into the
/// engine's POD [`MotionCurve`](byard_core::frame::MotionCurve) (RFC-0010), so
/// a resolved curve crosses the frame boundary as plain data.
fn pack_curve(curve: crate::interp::anim::Curve) -> byard_core::frame::MotionCurve {
    use crate::interp::anim::{Curve, EaseKind};
    use byard_core::frame::MotionCurve;
    #[allow(clippy::cast_precision_loss)]
    match curve {
        Curve::Linear { ms } => MotionCurve {
            kind: MotionCurve::LINEAR,
            params: [ms as f32, 0.0, 0.0],
        },
        Curve::Ease { ms, kind } => MotionCurve {
            kind: match kind {
                EaseKind::In => MotionCurve::EASE_IN,
                EaseKind::Out => MotionCurve::EASE_OUT,
                EaseKind::InOut => MotionCurve::EASE_IN_OUT,
            },
            params: [ms as f32, 0.0, 0.0],
        },
        Curve::Spring {
            stiffness,
            damping,
            v0,
        } => MotionCurve {
            kind: MotionCurve::SPRING,
            params: [stiffness, damping, v0],
        },
    }
}

/// Assigns one resolved side of a named spacing tuple, recording a
/// [`CompileError::ConflictingSpacingField`] if the side was already set (either
/// directly or via an axis shorthand).
fn assign_side(
    slot: &mut Option<f32>,
    v: f32,
    side: &str,
    span: Span,
    errors: &mut Vec<CompileError>,
) {
    if slot.is_some() {
        errors.push(CompileError::ConflictingSpacingField {
            span,
            message: format!("spacing side `{side}` was set more than once"),
        });
    } else {
        *slot = Some(v);
    }
}

fn display_value(v: &Value) -> String {
    match v {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Str(s) => s.clone(),
        _ => String::new(),
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::parser::ast::ElementNode;
    use crate::parser::parse;

    fn element(m: &Member) -> &ElementNode {
        match m {
            Member::Element(e) => e,
            _ => panic!("expected element"),
        }
    }

    // ── user-view registry & call-site recognition ──────────────────

    /// Loads a multi-view file and lowers the named view to a render tree.
    fn lower_named(src: &str, name: &str) -> (Interpreter, Vec<RenderNode>) {
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        interp.load_views(&parsed.views);
        let known: Vec<&str> = parsed.views.iter().map(|v| v.name.as_str()).collect();
        let view = parsed
            .views
            .iter()
            .find(|v| v.name.as_str() == name)
            .unwrap();
        let tree = interp.lower_view(view, &known);
        (interp, tree)
    }

    #[test]
    fn vector_icon_lowers_to_a_vector_node() {
        let (_interp, tree) = lower_named(
            "View App() { VectorIcon(\"icons/gear.svg\") #[size: 24, color: 0xFFFFFF] }",
            "App",
        );
        assert!(
            matches!(&tree[0], RenderNode::Vector { .. }),
            "VectorIcon lowers to RenderNode::Vector, got {:?}",
            tree[0]
        );
    }

    #[test]
    fn vector_icon_starts_as_a_placeholder_then_becomes_resident() {
        // Uses the real gear fixture from the M45 generator PR so this proves
        // the JIT dispatch end to end, not just the cache bookkeeping.
        let svg_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/svg/gear.svg");
        let src =
            format!("View App() {{ VectorIcon(\"{svg_path}\") #[size: 24, color: 0xFFFFFF] }}");
        let (mut interp, tree) = lower_named(&src, "App");

        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let first = frame.vector_instances()[0];
        assert!(
            first.color[3] < f32::EPSILON,
            "first tick must be a zero-opacity placeholder (INV-9), got alpha {}",
            first.color[3]
        );

        // Poll subsequent ticks until the background generation lands.
        let mut resident = None;
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            interp.tick();
            let mut frame = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut frame, 400.0, 300.0);
            let inst = frame.vector_instances()[0];
            if inst.color[3] > 0.0 {
                resident = Some(inst);
                break;
            }
        }
        let inst = resident.expect("the glyph must become resident within the poll window");
        assert!(
            (inst.color[3] - 1.0).abs() < f32::EPSILON,
            "full opacity once resident"
        );
        assert!(
            (inst.color[0] - 1.0).abs() < f32::EPSILON,
            "color: 0xFFFFFF tints white"
        );
    }

    // ── Binary arithmetic (`+ - * /`, RFC-0020 enabler) ─────────────

    #[test]
    fn eval_binary_promotes_and_never_panics() {
        // Int ∘ Int stays Int; division truncates.
        assert_eq!(
            eval_binary(BinOp::Add, Value::Int(2), Value::Int(3)),
            Value::Int(5)
        );
        assert_eq!(
            eval_binary(BinOp::Div, Value::Int(7), Value::Int(2)),
            Value::Int(3)
        );
        // Any Float operand promotes.
        assert_eq!(
            eval_binary(BinOp::Mul, Value::Int(25), Value::Float(3.6)),
            Value::Float(90.0)
        );
        // Division by zero is 0, not a panic or an IEEE infinity.
        assert_eq!(
            eval_binary(BinOp::Div, Value::Int(1), Value::Int(0)),
            Value::Int(0)
        );
        assert_eq!(
            eval_binary(BinOp::Div, Value::Float(1.0), Value::Float(0.0)),
            Value::Float(0.0)
        );
        // Non-numeric operands degrade to Unit.
        assert_eq!(
            eval_binary(BinOp::Add, Value::Str("a".into()), Value::Int(1)),
            Value::Unit
        );
    }

    #[test]
    fn arithmetic_expressions_evaluate_through_bindings() {
        // `let` chains with arithmetic reach paint properties: a Box whose
        // width is `base * 2 + 10`.
        let (mut interp, tree) = lower_named(
            "View App() { let base = 45 let w = base * 2 + 10 \
               Box #[width: w, height: 20, bg: 0xFF0000] {} }",
            "App",
        );
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let inst = frame.instances()[0];
        assert!(
            (inst.rect[2] - 100.0).abs() < 0.5,
            "width = 45 * 2 + 10 = 100, got {}",
            inst.rect[2]
        );
    }

    // ── Canvas & shape commands (RFC-0020) ──────────────────────────

    #[test]
    fn canvas_lowers_to_a_canvas_node_carrying_its_shapes() {
        let (interp, tree) = lower_named(
            "View App() { Canvas #[width: 48, height: 48] { \
               arc(cx: 24, cy: 24, r: 20, start: -90, sweep: 270, \
                   stroke: 0x6750A4, stroke_width: 4, cap: round) \
               circle(cx: 24, cy: 24, r: 8, fill: 0xE8DEF8) } }",
            "App",
        );
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let RenderNode::Canvas { shapes, .. } = &tree[0] else {
            panic!("Canvas lowers to RenderNode::Canvas, got {:?}", tree[0]);
        };
        assert_eq!(shapes.len(), 2);
        assert_eq!(shapes[0].name.as_str(), "arc");
        assert_eq!(shapes[1].name.as_str(), "circle");
    }

    #[test]
    fn canvas_shapes_render_into_the_canvas_pool_with_evaluated_params() {
        // The sweep is an expression over a view binding — proving shape
        // params run through the ordinary evaluator, so they are reactive.
        let (mut interp, tree) = lower_named(
            "View App() { let p = 0.5 \
               Canvas #[width: 100, height: 100] { \
                 arc(cx: 50, cy: 50, r: 40, start: 0, sweep: p * 360, \
                     stroke: 0xFF0000, stroke_width: 4) \
                 line(x1: 0, y1: 0, x2: 100, y2: 100, stroke: 0x00FF00) \
                 rect(x: 10, y: 10, w: 30, h: 20, radius: 4, fill: 0x0000FF) \
                 text(\"hi\", x: 50, y: 50, align: center, color: 0xFFFFFF) } }",
            "App",
        );
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let shapes = frame.canvas_shapes();
        assert_eq!(shapes.len(), 3, "arc + line + rect on the CanvasShape pool");
        // `p * 360` with p = 0.5 → a 180° sweep, stored in radians.
        let arc = &shapes[0];
        assert_eq!(arc.kind, byard_core::frame::CANVAS_SHAPE_ARC);
        assert!(
            (arc.params[4] - std::f32::consts::PI).abs() < 1e-3,
            "sweep 180° → π rad, got {}",
            arc.params[4]
        );
        assert!(
            (arc.stroke_color[0] - 1.0).abs() < 1e-6 && arc.stroke_color[3] > 0.99,
            "stroke: 0xFF0000 is opaque red"
        );
        // Shape coordinates are canvas-local + canvas origin (canvas at 0,0
        // in this single-child layout).
        assert!((arc.params[0] - 50.0).abs() < 0.5);
        // The `text(…)` command lowers to an ordinary TextLine, centred
        // around x=50 (its left edge sits before the anchor).
        assert_eq!(frame.texts().len(), 1);
        assert!(frame.texts()[0].x < 50.0);
        // Depths are parallel and strictly ordered (later = nearer).
        let d = frame.canvas_depths();
        assert_eq!(d.len(), 3);
        assert!(d[0] > d[1] && d[1] > d[2]);
    }

    #[test]
    fn full_sweep_arcs_collapse_to_the_cheaper_circle_kind() {
        let (mut interp, tree) = lower_named(
            "View App() { Canvas #[width: 48, height: 48] { \
               arc(cx: 24, cy: 24, r: 20, start: 0, sweep: 360, stroke: 0xFFFFFF) } }",
            "App",
        );
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(
            frame.canvas_shapes()[0].kind,
            byard_core::frame::CANVAS_SHAPE_CIRCLE
        );
    }

    #[test]
    fn bezier_flattens_to_a_contiguous_round_capped_polyline() {
        let (mut interp, tree) = lower_named(
            "View App() { Canvas #[width: 100, height: 100] { \
               bezier(x1: 0, y1: 80, cx1: 30, cy1: 0, cx2: 70, cy2: 0, x2: 100, y2: 80, \
                      stroke: 0xFFFFFF, stroke_width: 2) } }",
            "App",
        );
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let shapes = frame.canvas_shapes();
        assert!(shapes.len() >= 8, "flattening yields several segments");
        for s in shapes {
            assert_eq!(s.kind, byard_core::frame::CANVAS_SHAPE_LINE);
            assert_eq!(s.cap, byard_core::frame::CANVAS_CAP_ROUND);
        }
        // Endpoints chain: each segment starts where the previous ended, and
        // the whole polyline spans the curve's anchors.
        for pair in shapes.windows(2) {
            assert!((pair[0].params[2] - pair[1].params[0]).abs() < 1e-4);
            assert!((pair[0].params[3] - pair[1].params[1]).abs() < 1e-4);
        }
        assert!((shapes[0].params[0] - 0.0).abs() < 0.5);
        assert!((shapes[shapes.len() - 1].params[2] - 100.0).abs() < 0.5);
    }

    #[test]
    fn a_shapeless_paintless_command_emits_nothing() {
        // No stroke and no fill → invisible → skipped entirely.
        let (mut interp, tree) = lower_named(
            "View App() { Canvas #[width: 48, height: 48] { circle(cx: 24, cy: 24, r: 20) } }",
            "App",
        );
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(frame.canvas_shapes().is_empty());
    }

    #[test]
    fn canvas_validation_errors_surface_through_lowering() {
        let (interp, _tree) = lower_named(
            "View App() { Canvas #[width: 48] { arc(cx: 1, cy: 1) Text(\"no\") } }",
            "App",
        );
        let errs = interp.errors();
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::CanvasMissingSize { .. })),
            "{errs:?}"
        );
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::MissingShapeParam { .. })),
            "{errs:?}"
        );
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::UnknownShapeCommand { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn user_view_call_is_recognized_and_no_unknown_view_fires() {
        // `App` calls `Card` (a user view); no `UnknownView` diagnostic fires.
        let (interp, _tree) =
            lower_named("View Card() { Text(\"hi\") }\nView App() { Card() }", "App");
        assert!(
            !interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::UnknownView { .. })),
            "no UnknownView expected: {:?}",
            interp.errors()
        );
    }

    #[test]
    fn intrinsic_named_view_reports_shadowed_at_load() {
        let parsed = parse("View Row() { Text(\"x\") }");
        let mut interp = Interpreter::new();
        let diags = interp.load_views(&parsed.views);
        assert!(
            diags
                .iter()
                .any(|d| matches!(d, CompileError::IntrinsicShadowed { .. })),
            "expected IntrinsicShadowed, got {diags:?}"
        );
    }

    // ── argument → parameter binding ────────────────────────────────

    /// Parses `callee_src` (a single view) and a call element from `call_src`'s
    /// first view body, returning `(callee, call_element)`.
    fn callee_and_call(callee_src: &str, call_src: &str) -> (ViewDecl, ElementNode) {
        let callee = parse(callee_src).views.into_iter().next().unwrap();
        let host = parse(call_src).views.into_iter().next().unwrap();
        let Member::Element(call) = host.body.into_iter().next().unwrap() else {
            panic!("expected element")
        };
        (callee, call)
    }

    #[test]
    fn named_positional_and_mixed_binding() {
        let (callee, _) = callee_and_call(
            "View Avatar(url, size) { Text(url) }",
            "View H() { Text(\"x\") }",
        );
        // Named.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call(
            "View Avatar(url, size) { Text(url) }",
            "View H() { Avatar(url: \"a.png\", size: 40) }",
        );
        let b = interp.bind_args(&callee, &call);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert_eq!(b.bindings.len(), 2);
        assert_eq!(b.bindings[0].0.as_str(), "url");
        assert_eq!(b.bindings[1].0.as_str(), "size");

        // Positional.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call(
            "View Avatar(url, size) { Text(url) }",
            "View H() { Avatar(\"a.png\", 40) }",
        );
        let b = interp.bind_args(&callee, &call);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert_eq!(b.bindings.len(), 2);

        // Mixed: positional then named.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call(
            "View Avatar(url, size) { Text(url) }",
            "View H() { Avatar(\"a.png\") #[size: 40] }",
        );
        let b = interp.bind_args(&callee, &call);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert_eq!(b.bindings.len(), 2);
    }

    #[test]
    fn arity_unknown_duplicate_and_missing_diagnostics() {
        let (callee, _) = callee_and_call("View A(x, y) { Text(x) }", "View H() { Text(\"_\") }");

        // Over-arity: 3 positional for 2 params.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View A(x, y) { Text(x) }", "View H() { A(1, 2, 3) }");
        interp.bind_args(&callee, &call);
        assert!(interp.errors().iter().any(|e| matches!(
            e,
            CompileError::ViewArityMismatch {
                expected: 2,
                found: 3,
                ..
            }
        )));

        // Unknown named param.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View A(x, y) { Text(x) }", "View H() { A(z: 1) }");
        interp.bind_args(&callee, &call);
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::UnknownParam { .. }))
        );

        // Duplicate: positional + named bind the same param.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View A(x, y) { Text(x) }", "View H() { A(1) #[x: 2] }");
        interp.bind_args(&callee, &call);
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::DuplicateParam { .. }))
        );

        // Missing required.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View A(x, y) { Text(x) }", "View H() { A(1) }");
        interp.bind_args(&callee, &call);
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::MissingParam { name, .. } if name == "y"))
        );
    }

    #[test]
    fn parent_var_arg_is_a_live_memo_literal_is_constant() {
        // The parent declares `var n = 1`; the call passes `n` to a parameter.
        // The projecting memo tracks the parent signal: writing `n` and ticking
        // changes the memo's value (dirty edge preserved).
        let callee = parse("View Foo(v) { Text(\"{v}\") }")
            .views
            .into_iter()
            .next()
            .unwrap();
        let mut interp = Interpreter::new();
        let init = Expr::IntLit(1, crate::diagnostics::Span::new(0, 1));
        let n = interp.define_var(Symbol::intern("n"), &init);

        let (_, call) = callee_and_call("View Foo(v) { Text(\"{v}\") }", "View H() { Foo(n) }");
        let b = interp.bind_args(&callee, &call);
        let memo = b.bindings[0].1;
        interp.tick();
        assert_eq!(interp.read_memo(memo), Value::Int(1));
        interp.write_var(n, Value::Int(7));
        interp.tick();
        assert_eq!(
            interp.read_memo(memo),
            Value::Int(7),
            "memo tracks the parent var"
        );

        // A literal argument is a constant memo: it never changes.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View Foo(v) { Text(\"{v}\") }", "View H() { Foo(5) }");
        let b = interp.bind_args(&callee, &call);
        let memo = b.bindings[0].1;
        interp.tick();
        assert_eq!(interp.read_memo(memo), Value::Int(5));
    }

    // ── body expansion & per-instance scope ─────────────────────────

    /// The string value of a `Text` node's content scope, after a tick.
    fn text_value(interp: &mut Interpreter, node: &RenderNode) -> String {
        let RenderNode::Text { content, .. } = node else {
            panic!("expected Text node, got {node:?}");
        };
        interp.tick();
        match interp.binding_value(*content) {
            Some(Value::Str(s)) => s,
            other => panic!("expected Str binding, got {other:?}"),
        }
    }

    #[test]
    fn user_view_expands_body_and_binds_a_parameter() {
        // `App` calls `Greet("Ada")`; the call expands to the callee body with
        // `name` bound, projecting "Hi Ada".
        let (mut interp, tree) = lower_named(
            "View Greet(name) { Text(\"Hi {name}\") }\nView App() { Greet(\"Ada\") }",
            "App",
        );
        assert_eq!(tree.len(), 1, "one spliced root");
        assert_eq!(text_value(&mut interp, &tree[0]), "Hi Ada");
    }

    #[test]
    fn user_view_passes_value_to_an_intrinsic() {
        // `Avatar(url, size)` lowers `Image(url) #[width: size]`; the call's
        // arguments flow through to the intrinsic node.
        let (_interp, tree) = lower_named(
            "View Avatar(url, size) { Image(url) #[width: size, height: size] }\n\
             View App() { Avatar(\"ada.png\", 40) }",
            "App",
        );
        assert_eq!(tree.len(), 1);
        assert!(
            matches!(&tree[0], RenderNode::Image { .. }),
            "expected an Image node, got {:?}",
            tree[0]
        );
    }

    #[test]
    fn a_call_yielding_multiple_roots_splices_as_siblings() {
        let (_interp, tree) = lower_named(
            "View Pair() { Text(\"a\")\n Text(\"b\") }\nView App() { Pair() }",
            "App",
        );
        assert_eq!(tree.len(), 2, "both callee roots spliced as siblings");
    }

    #[test]
    fn nested_user_view_calls_expand() {
        // App → Outer → Inner → Text("x").
        let (mut interp, tree) = lower_named(
            "View Inner() { Text(\"x\") }\n\
             View Outer() { Inner() }\n\
             View App() { Outer() }",
            "App",
        );
        assert_eq!(tree.len(), 1);
        assert_eq!(text_value(&mut interp, &tree[0]), "x");
    }

    #[test]
    fn two_instances_keep_independent_local_state() {
        // Two `Counter()` instances each lower their own `var n`; their content
        // scopes are distinct bindings (independent per-instance state).
        let (_interp, tree) = lower_named(
            "View Counter() { var n = 0\n Text(\"{n}\") }\n\
             View App() { Column { Counter()\n Counter() } }",
            "App",
        );
        // App → Column(Box) containing two expanded Counters.
        let RenderNode::Box { children, .. } = &tree[0] else {
            panic!("expected a Column box, got {:?}", tree[0]);
        };
        let texts: Vec<&RenderNode> = children
            .iter()
            .filter(|c| matches!(c, RenderNode::Text { .. }))
            .collect();
        assert_eq!(texts.len(), 2, "two independent Counter texts");
        let scopes: Vec<ScopeId> = texts
            .iter()
            .map(|t| match t {
                RenderNode::Text { content, .. } => *content,
                _ => unreachable!(),
            })
            .collect();
        assert_ne!(scopes[0], scopes[1], "each instance has its own binding");
    }

    #[test]
    fn two_level_composition_golden_shape() {
        // UserRow composes Avatar + Text inside a Row; App stacks two UserRows.
        let (_interp, tree) = lower_named(
            "View Avatar(url) { Image(url) }\n\
             View UserRow(name, avatar) { Row { Avatar(avatar)\n Text(name) } }\n\
             View App() { Column { UserRow(\"Ada\", \"ada.png\")\n UserRow(\"Alan\", \"alan.png\") } }",
            "App",
        );
        // App → Column(Box) → [Row(Box)[Image, Text], Row(Box)[Image, Text]].
        let RenderNode::Box { children, .. } = &tree[0] else {
            panic!("expected Column");
        };
        assert_eq!(children.len(), 2, "two UserRow instances");
        for row in children {
            let RenderNode::Box { children: rc, .. } = row else {
                panic!("expected Row");
            };
            assert!(matches!(rc[0], RenderNode::Image { .. }));
            assert!(matches!(rc[1], RenderNode::Text { .. }));
        }
    }

    // ── recursion & cycle protection ────────────────────────────────

    #[test]
    fn unguarded_self_call_is_recursive_view_at_load() {
        let parsed = parse("View A() { A() }");
        let mut interp = Interpreter::new();
        let diags = interp.load_views(&parsed.views);
        assert!(
            diags
                .iter()
                .any(|d| matches!(d, CompileError::RecursiveView { .. })),
            "expected RecursiveView at load, got {diags:?}"
        );
    }

    #[test]
    fn guarded_recursion_that_terminates_is_legal() {
        // RFC-0018: `Tree` recurses only in the `else` of a guard that is true, so
        // the recursive branch is never lowered (lazy `when`) — it renders to a
        // finite depth with no diagnostic.
        let (mut interp, tree) = lower_named(
            "View Tree() { var leaf = true\n when leaf { Text(\"x\") } else { Tree() } }\n\
             View App() { Tree() }",
            "App",
        );
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(
            interp.errors().is_empty(),
            "guarded terminating recursion is legal: {:?}",
            interp.errors()
        );
        assert_eq!(frame.texts().len(), 1);
        assert_eq!(frame.texts()[0].text, "x");
    }

    #[test]
    fn runaway_guarded_recursion_hits_depth_bound_without_panicking() {
        // `go` is always true, so the guard never terminates at lower time. The
        // static check does not flag it (the cycle is guarded), so the runtime
        // depth bound must stop it with a diagnostic — not a stack overflow.
        let parsed =
            parse("View Loop() { var go = true\n when go { Loop() } }\nView App() { Loop() }");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let load_diags = interp.load_views(&parsed.views);
        assert!(
            !load_diags
                .iter()
                .any(|d| matches!(d, CompileError::RecursiveView { .. })),
            "a guarded cycle is not a static error"
        );
        let known: Vec<&str> = parsed.views.iter().map(|v| v.name.as_str()).collect();
        let app = parsed
            .views
            .iter()
            .find(|v| v.name.as_str() == "App")
            .unwrap();
        let tree = interp.lower_view(app, &known); // lazy: no recursion at lower
        // RFC-0018: the recursion now unrolls at render (reconcile) time — one
        // level per frame (a freshly-lowered `go` reads false until the next
        // pull), so each render is finite (no stack overflow). Over enough frames
        // the reconcile depth bound stops it with a diagnostic.
        let mut hit = false;
        for _ in 0..(MAX_INSTANCE_DEPTH + 8) {
            interp.tick();
            let mut frame = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut frame, 400.0, 300.0); // must not overflow
            if interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::RecursiveView { .. }))
            {
                hit = true;
                break;
            }
        }
        assert!(
            hit,
            "the reconcile depth bound must stop the runaway recursion"
        );
    }

    // ── hot-reload across instances ─────────────────────────────────

    #[test]
    fn reloading_a_leaf_view_updates_all_its_instances() {
        use crate::interp::reload::{affected_views, diff_view};

        let old = parse("View Leaf() { Text(\"old\") }\nView App() { Column { Leaf()\n Leaf() } }");
        let new = parse("View Leaf() { Text(\"new\") }\nView App() { Column { Leaf()\n Leaf() } }");

        let mut interp = Interpreter::new();
        interp.load_views(&old.views);
        let known_old: Vec<&str> = old.views.iter().map(|v| v.name.as_str()).collect();
        let app_old = old.views.iter().find(|v| v.name.as_str() == "App").unwrap();
        let tree = interp.lower_view(app_old, &known_old);
        let RenderNode::Box { children, .. } = &tree[0] else {
            panic!("expected Column");
        };
        assert_eq!(text_value(&mut interp, &children[0]), "old");

        // The edit to the leaf transitively affects App (RFC-0007 §5).
        let affected = affected_views(&old.views, &new.views);
        assert!(affected.contains(&Symbol::intern("App")));

        // Rebuild the registry and re-derive App; both Leaf instances update.
        interp.load_views(&new.views);
        let app_new = new.views.iter().find(|v| v.name.as_str() == "App").unwrap();
        interp.reload(app_new, diff_view(app_old, app_new));
        let known_new: Vec<&str> = new.views.iter().map(|v| v.name.as_str()).collect();
        let tree = interp.lower_view(app_new, &known_new);
        let RenderNode::Box { children, .. } = &tree[0] else {
            panic!("expected Column");
        };
        assert_eq!(text_value(&mut interp, &children[0]), "new");
        assert_eq!(text_value(&mut interp, &children[1]), "new");
    }

    // ── slots & parameter defaults ──────────────────────────────────

    #[test]
    fn omitted_defaulted_param_uses_its_default() {
        // `label` is omitted; the default "?" is evaluated in the callee scope.
        let (mut interp, tree) = lower_named(
            "View Tag(label = \"?\") { Text(label) }\nView App() { Tag() }",
            "App",
        );
        assert!(
            interp.errors().is_empty(),
            "a defaulted param is not required: {:?}",
            interp.errors()
        );
        assert_eq!(text_value(&mut interp, &tree[0]), "?");
    }

    #[test]
    fn supplied_argument_overrides_the_default() {
        let (mut interp, tree) = lower_named(
            "View Tag(label = \"?\") { Text(label) }\nView App() { Tag(\"hi\") }",
            "App",
        );
        assert_eq!(text_value(&mut interp, &tree[0]), "hi");
    }

    #[test]
    fn missing_param_only_fires_for_required_params() {
        // `a` is required, `b` is defaulted; omitting both reports only `a`.
        let (callee, _) =
            callee_and_call("View V(a, b = 1) { Text(a) }", "View H() { Text(\"_\") }");
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View V(a, b = 1) { Text(a) }", "View H() { V() }");
        interp.bind_args(&callee, &call);
        let missing: Vec<&str> = interp
            .errors()
            .iter()
            .filter_map(|e| match e {
                CompileError::MissingParam { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(missing, vec!["a"], "only the required param is missing");
    }

    #[test]
    fn content_slot_renders_the_passed_block() {
        // `Card` declares a `content` slot; `App` passes a `Text` block, which is
        // spliced where `content` appears inside the card body.
        let (mut interp, tree) = lower_named(
            "View Card(content) { Column { content } }\n\
             View App() { Card { Text(\"inside\") } }",
            "App",
        );
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let RenderNode::Box { children, .. } = &tree[0] else {
            panic!("expected the card's Column, got {:?}", tree[0]);
        };
        assert_eq!(children.len(), 1, "the passed block is spliced");
        assert_eq!(text_value(&mut interp, &children[0]), "inside");
    }

    #[test]
    fn block_passed_to_a_slotless_view_is_unexpected_children() {
        let (interp, _tree) = lower_named(
            "View Plain() { Text(\"x\") }\nView App() { Plain { Text(\"no\") } }",
            "App",
        );
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::UnexpectedChildren { .. })),
            "expected UnexpectedChildren, got {:?}",
            interp.errors()
        );
    }

    #[test]
    fn slot_block_captures_the_caller_scope() {
        // The block passed to `Card` reads the *caller's* `name` var, proving the
        // slot is lowered in the caller scope (not the callee's).
        let (mut interp, tree) = lower_named(
            "View Card(content) { content }\n\
             View App() { var name = \"Ada\"\n Card { Text(\"Hi {name}\") } }",
            "App",
        );
        assert_eq!(text_value(&mut interp, &tree[0]), "Hi Ada");
    }

    #[test]
    fn var_text_binding_updates_after_mutation_and_tick() {
        let parsed =
            parse("View C() {\n var count = 0\n Text(\"{count}\")\n Button(\"+\") => count++\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];

        let mut interp = Interpreter::new();
        let Member::Var { name, init, .. } = &view.body[0] else {
            panic!("expected var");
        };
        interp.define_var(name.clone(), init);

        let text = element(&view.body[1]);
        let bind = interp.bind_value(&text.content[0].value);
        interp.tick();
        assert_eq!(
            interp.binding_value(bind),
            Some(Value::Str("0".to_string()))
        );

        // The Button's `=> count++` action.
        let action = element(&view.body[2]).action.as_ref().unwrap();
        interp.eval_action(action).unwrap();
        interp.tick();
        assert_eq!(
            interp.binding_value(bind),
            Some(Value::Str("1".to_string()))
        );
    }

    #[test]
    fn let_memo_recomputes_when_its_source_changes() {
        let parsed = parse(
            "View C() {\n var count = 0\n let doubled = count\n Text(\"{doubled}\")\n Button(\"+\") => count++\n}",
        );
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();

        let Member::Var { name, init, .. } = &view.body[0] else {
            panic!()
        };
        interp.define_var(name.clone(), init);
        let Member::Let { name, init, .. } = &view.body[1] else {
            panic!()
        };
        let memo = interp.define_let(name.clone(), init);

        let text = element(&view.body[2]);
        let bind = interp.bind_value(&text.content[0].value);
        interp.tick();
        assert_eq!(
            interp.binding_value(bind),
            Some(Value::Str("0".to_string()))
        );
        let evals = interp.ctx().eval_count(memo);

        let action = element(&view.body[3]).action.as_ref().unwrap();
        interp.eval_action(action).unwrap();
        interp.tick();
        assert_eq!(
            interp.binding_value(bind),
            Some(Value::Str("1".to_string()))
        );
        assert!(interp.ctx().eval_count(memo) > evals, "memo recomputed");
    }

    #[test]
    fn assignment_to_a_let_is_not_assignable() {
        let parsed =
            parse("View C() {\n var count = 0\n let y = count\n Button(\"x\") => y = 5\n}");
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();

        let Member::Var { name, init, .. } = &view.body[0] else {
            panic!()
        };
        interp.define_var(name.clone(), init);
        let Member::Let { name, init, .. } = &view.body[1] else {
            panic!()
        };
        interp.define_let(name.clone(), init);

        let action = element(&view.body[2]).action.as_ref().unwrap();
        let err = interp.eval_action(action).unwrap_err();
        assert!(matches!(err, CompileError::NotAssignable { .. }));
    }

    #[test]
    fn lower_view_emits_expected_render_tree() {
        let parsed = parse(
            "View C() {\n var count = 0\n Column #[bg: 0x222222, radius: 16] {\n Text(\"Count: {count}\")\n }\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];

        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        // One top-level Column box with the literal bg/radius and one Text child.
        assert_eq!(tree.len(), 1);
        let RenderNode::Box {
            attrs, children, ..
        } = &tree[0]
        else {
            panic!("expected a Box, got {:?}", tree[0]);
        };
        let bg = interp.eval_color_prop(attrs, "bg");
        let radius = interp.eval_int_prop(attrs, "radius");
        assert_eq!(bg, Some(0x0022_2222));
        assert_eq!(radius, Some(16));
        assert_eq!(children.len(), 1);
        let RenderNode::Text { content, .. } = &children[0] else {
            panic!("expected a Text child");
        };

        // The Text projects the reactive count.
        interp.tick();
        assert_eq!(
            interp.binding_value(*content),
            Some(Value::Str("Count: 0".to_string()))
        );
    }

    #[test]
    fn lowering_an_unknown_element_records_unknown_view() {
        let parsed = parse("View C() { Colunm #[gap: 8] {} }");
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let _ = interp.lower_view(view, &[]);
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::UnknownView { .. }))
        );
    }

    #[test]
    fn mutation_on_an_undeclared_name_is_not_assignable() {
        let parsed = parse("View C() { Button(\"x\") => ghost++ }");
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let action = element(&view.body[0]).action.as_ref().unwrap();
        assert!(matches!(
            interp.eval_action(action).unwrap_err(),
            CompileError::NotAssignable { .. }
        ));
    }

    #[test]
    fn spacing_convenience_parses_correctly() {
        use byard_core::atlas::layout::Spacing;

        let test_cases = vec![
            // 1-value positional
            ("View C() { Column #[p: (10)] {} }", Spacing::all(10.0)),
            // 2-value positional
            (
                "View C() { Column #[p: (2, 3)] {} }",
                Spacing::symmetric(2.0, 3.0),
            ),
            // 4-value positional: CSS order top, right, bottom, left
            (
                "View C() { Column #[p: (1, 2, 3, 4)] {} }",
                Spacing {
                    top: 1.0,
                    right: 2.0,
                    bottom: 3.0,
                    left: 4.0,
                },
            ),
            // Named top only — unspecified sides default to 0
            (
                "View C() { Column #[p: (top: 10)] {} }",
                Spacing {
                    top: 10.0,
                    right: 0.0,
                    bottom: 0.0,
                    left: 0.0,
                },
            ),
            // Named bottom only
            (
                "View C() { Column #[p: (bottom: 7)] {} }",
                Spacing {
                    top: 0.0,
                    right: 0.0,
                    bottom: 7.0,
                    left: 0.0,
                },
            ),
            // Named mixed sides
            (
                "View C() { Column #[p: (left: 5, bottom: 3)] {} }",
                Spacing {
                    top: 0.0,
                    right: 0.0,
                    bottom: 3.0,
                    left: 5.0,
                },
            ),
            // Verbose axis shorthands (the only accepted shorthands)
            (
                "View C() { Column #[p: (horizontal: 10, vertical: 5)] {} }",
                Spacing {
                    top: 5.0,
                    right: 10.0,
                    bottom: 5.0,
                    left: 10.0,
                },
            ),
        ];

        for (source, expected_spacing) in test_cases {
            let parsed = parse(source);
            assert!(
                parsed.errors.is_empty(),
                "Failed to parse: {}\nErrors: {:?}",
                source,
                parsed.errors
            );
            let view = &parsed.views[0];
            let mut interp = Interpreter::new();
            let tree = interp.lower_view(view, &[]);
            let RenderNode::Box { name, attrs, .. } = &tree[0] else {
                panic!("expected a Box");
            };
            let style = interp.eval_container_style(name.as_str(), attrs);
            assert_eq!(
                style.padding.top, expected_spacing.top,
                "top mismatch for source: {}",
                source
            );
            assert_eq!(
                style.padding.right, expected_spacing.right,
                "right mismatch for source: {}",
                source
            );
            assert_eq!(
                style.padding.bottom, expected_spacing.bottom,
                "bottom mismatch for source: {}",
                source
            );
            assert_eq!(
                style.padding.left, expected_spacing.left,
                "left mismatch for source: {}",
                source
            );
        }
    }

    #[test]
    fn individual_margin_padding_properties_override() {
        use byard_core::atlas::layout::Spacing;

        let parsed = parse("View C() { Column #[p: (10), pt: 2, pb: 4, ml: 5, mt: 1] {} }");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        let RenderNode::Box { name, attrs, .. } = &tree[0] else {
            panic!("expected a Box");
        };
        let style = interp.eval_container_style(name.as_str(), attrs);
        // padding.top overridden to 2, padding.bottom overridden to 4, others stay 10
        assert_eq!(
            style.padding,
            Spacing {
                top: 2.0,
                right: 10.0,
                bottom: 4.0,
                left: 10.0
            }
        );
        // margins
        assert_eq!(
            style.margin,
            Spacing {
                top: 1.0,
                right: 0.0,
                bottom: 0.0,
                left: 5.0
            }
        );
    }

    // ── M25: `Len` padding/margin forms ──────────────────────────────────

    /// Lowers a single-`Box` view and returns the resolved padding plus any
    /// errors raised during style resolution.
    fn resolve_padding(src: &str) -> (byard_core::atlas::layout::Spacing, Vec<CompileError>) {
        let parsed = parse(src);
        assert!(
            parsed.errors.is_empty(),
            "parse errors: {:?}",
            parsed.errors
        );
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        let RenderNode::Box { name, attrs, .. } = &tree[0] else {
            panic!("expected a Box");
        };
        let style = interp.eval_container_style(name.as_str(), attrs);
        (style.padding, interp.errors().to_vec())
    }

    #[test]
    fn impl30_scalar_sets_all_sides() {
        use byard_core::atlas::layout::Spacing;
        let (p, errs) = resolve_padding("View C() { Column #[p: 5] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(p, Spacing::all(5.0));
    }

    #[test]
    fn impl30_pair_is_vertical_horizontal() {
        use byard_core::atlas::layout::Spacing;
        let (p, errs) = resolve_padding("View C() { Column #[p: (10, 5)] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        // (vertical, horizontal): top=bottom=10, left=right=5.
        assert_eq!(
            p,
            Spacing {
                top: 10.0,
                right: 5.0,
                bottom: 10.0,
                left: 5.0
            }
        );
    }

    #[test]
    fn impl30_quad_is_css_order() {
        use byard_core::atlas::layout::Spacing;
        let (p, errs) = resolve_padding("View C() { Column #[p: (4, 6, 8, 7)] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(
            p,
            Spacing {
                top: 4.0,
                right: 6.0,
                bottom: 8.0,
                left: 7.0
            }
        );
    }

    #[test]
    fn impl30_named_single_side_defaults_rest_to_zero() {
        use byard_core::atlas::layout::Spacing;
        let (p, errs) = resolve_padding("View C() { Column #[p: (bottom: 7)] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(
            p,
            Spacing {
                top: 0.0,
                right: 0.0,
                bottom: 7.0,
                left: 0.0
            }
        );
    }

    #[test]
    fn impl30_named_axis_shorthands() {
        use byard_core::atlas::layout::Spacing;
        let (p, errs) =
            resolve_padding("View C() { Column #[p: (horizontal: 10, vertical: 5)] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(
            p,
            Spacing {
                top: 5.0,
                right: 10.0,
                bottom: 5.0,
                left: 10.0
            }
        );
    }

    #[test]
    fn impl30_unknown_side_is_unknown_attribute_with_hint() {
        let (_p, errs) = resolve_padding("View C() { Column #[p: (tpo: 4)] {} }");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                CompileError::UnknownAttribute { name, hint: Some(h), .. }
                    if name == "tpo" && h == "top"
            )),
            "expected UnknownAttribute(tpo)->top, got {errs:?}"
        );
    }

    #[test]
    fn impl30_axis_plus_component_conflicts() {
        let (_p, errs) = resolve_padding("View C() { Column #[p: (horizontal: 10, left: 3)] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::ConflictingSpacingField { .. })),
            "expected ConflictingSpacingField, got {errs:?}"
        );
    }

    #[test]
    fn impl30_non_int_side_is_type_mismatch() {
        let (_p, errs) = resolve_padding("View C() { Column #[p: (top: 4, left: \"x\")] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::AttributeTypeMismatch { .. })),
            "expected AttributeTypeMismatch, got {errs:?}"
        );
    }

    #[test]
    fn impl30_wrong_positional_arity_is_arity_mismatch() {
        let (_p, errs) = resolve_padding("View C() { Column #[p: (1, 2, 3)] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::ArityMismatch { .. })),
            "expected ArityMismatch for a 3-tuple, got {errs:?}"
        );
    }

    #[test]
    fn impl30_mixing_named_and_positional_errors() {
        let (_p, errs) = resolve_padding("View C() { Column #[p: (10, top: 4)] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::ConflictingSpacingField { .. })),
            "expected a conflict for mixed named/positional, got {errs:?}"
        );
    }

    #[test]
    fn impl30_px_py_are_now_unknown_attributes() {
        let parsed = parse("View C() { Column #[px: 5] {} }");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let _ = interp.lower_view(view, &[]);
        assert!(
            interp.errors().iter().any(|e| matches!(
                e,
                CompileError::UnknownAttribute { name, .. } if name == "px"
            )),
            "px must now be UnknownAttribute, got {:?}",
            interp.errors()
        );
    }

    // ── Per-corner `radius` ──────────────────────────────────────────────

    /// Lowers a single-element view and returns `resolve_radii`'s result for
    /// its `radius` attribute alongside any errors raised.
    fn resolve_radius_test(src: &str) -> ([f32; 4], Vec<CompileError>) {
        let parsed = parse(src);
        assert!(
            parsed.errors.is_empty(),
            "parse errors: {:?}",
            parsed.errors
        );
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let attrs = element(&view.body[0]).attrs.clone();
        let radii = interp.resolve_radii(&attrs, "radius");
        (radii, interp.errors().to_vec())
    }

    #[test]
    fn impl44_radius_scalar_broadcasts_to_all_four_corners() {
        let (radii, errs) = resolve_radius_test("View C() { Column #[radius: 16] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(radii, [16.0, 16.0, 16.0, 16.0]);
    }

    #[test]
    fn impl44_radius_quad_sets_independent_corners_in_css_order() {
        // top_left, top_right, bottom_right, bottom_left (frame.rs / WGSL convention).
        let (radii, errs) = resolve_radius_test("View C() { Column #[radius: (4, 8, 12, 16)] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(radii, [4.0, 8.0, 12.0, 16.0]);
    }

    #[test]
    fn impl44_radius_missing_attribute_defaults_to_zero() {
        let (radii, errs) = resolve_radius_test("View C() { Column {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(radii, [0.0; 4]);
    }

    #[test]
    fn impl44_radius_wrong_arity_is_arity_mismatch() {
        let (radii, errs) = resolve_radius_test("View C() { Column #[radius: (4, 8)] {} }");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                CompileError::ArityMismatch {
                    expected: 4,
                    found: 2,
                    ..
                }
            )),
            "expected ArityMismatch(4, found 2), got {errs:?}"
        );
        assert_eq!(radii, [0.0; 4]);
    }

    #[test]
    fn impl44_radius_named_corner_field_is_rejected() {
        let (radii, errs) = resolve_radius_test(
            "View C() { Column #[radius: (top_left: 4, top_right: 8, bottom_right: 12, bottom_left: 16)] {} }",
        );
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::ConflictingSpacingField { .. })),
            "expected ConflictingSpacingField for named corners, got {errs:?}"
        );
        assert_eq!(radii, [0.0; 4]);
    }

    #[test]
    fn impl44_radius_non_numeric_corner_is_type_mismatch() {
        let (radii, errs) =
            resolve_radius_test("View C() { Column #[radius: (4, \"x\", 12, 16)] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::AttributeTypeMismatch { .. })),
            "expected AttributeTypeMismatch, got {errs:?}"
        );
        // Valid corners still resolve; the bad one is left at the [0.0;4]
        // default rather than aborting the whole tuple.
        assert_eq!(radii, [4.0, 0.0, 12.0, 16.0]);
    }

    #[test]
    fn impl44_decorated_box_carries_independent_corner_radii_into_box_instance() {
        // End-to-end: a quad `radius` on a Box that also has `bg` (so it's a
        // plain BoxInstance push, not a DecoratedBox) reaches the GPU instance
        // with all four corners intact rather than being collapsed to a scalar.
        let parsed = parse(
            "View C() { Box #[bg: 0xFF0000, radius: (4, 8, 12, 16), width: 50, height: 50] }",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let instances = frame.instances();
        assert_eq!(instances.len(), 1, "expected exactly one BoxInstance");
        assert_eq!(instances[0].radii, [4.0, 8.0, 12.0, 16.0]);
    }

    // ── RFC-0011: paint-time transform attribute surface ─────────────────

    #[test]
    fn transform_attrs_reach_the_box_instance() {
        let parsed = parse(
            "View C() { Box #[bg: 0xFF0000, width: 50, height: 50, \
             translate: (5, 10), scale: 1.5, rotate: 90deg] }",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let instances = frame.instances();
        assert_eq!(instances.len(), 1);
        let t = instances[0].transform;
        assert_eq!(t.translate, [5.0, 10.0]);
        assert_eq!(t.scale, [1.5, 1.5]);
        assert!((t.rotate - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        // Unset `origin` defaults to the element's own center, not (0,0).
        assert_eq!(t.origin, [25.0, 25.0]);
    }

    #[test]
    fn with_animation_interpolates_toward_the_target_and_settles() {
        // A linear ramp gives deterministic sample points to assert on.
        let parsed = parse(
            "View V() { var on: Bool = false \
             Box #[bg: 0x808080, width: 10, height: 10, \
             scale: on ? 2.0 : 1.0 with anim.linear(1000)] }",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();

        let render_scale = |interp: &mut Interpreter, now: u32| -> f32 {
            interp.set_now_ms(now);
            let mut frame = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut frame, 400.0, 300.0);
            frame.instances()[0].transform.scale[0]
        };

        // At rest: target is 1.0, nothing is animating.
        assert!((render_scale(&mut interp, 0) - 1.0).abs() < 1e-3);
        assert!(!interp.has_active_animations());

        // Flip the target to 2.0 at t=0 — the motion retargets from the current
        // value (~1.0) and is now active.
        let sig = interp.var_signal(&Symbol::intern("on")).unwrap();
        interp.write_var(sig, Value::Bool(true));
        interp.tick();
        assert!(
            (render_scale(&mut interp, 0) - 1.0).abs() < 1e-2,
            "starts where it was"
        );
        assert!(
            interp.has_active_animations(),
            "a just-retargeted motion is active"
        );

        // Halfway through the 1000 ms ramp → ~1.5.
        assert!((render_scale(&mut interp, 500) - 1.5).abs() < 5e-2);

        // Past the end → arrived at 2.0 and settled (idle again).
        assert!((render_scale(&mut interp, 1000) - 2.0).abs() < 1e-3);
        assert!(
            !interp.has_active_animations(),
            "settles once the ramp completes"
        );
    }

    /// Drives a `on ? … : …` paint prop through a 1000 ms linear ramp and
    /// returns the value `sample` reads from the rendered frame at t = 0 (just
    /// after the flip), 500, and 1000 ms — the shared body of the coverage tests
    /// below, which each assert a different paint prop interpolates.
    fn ramp_paint_prop(
        src: &str,
        sample: impl Fn(&byard_core::frame::RenderFrame) -> f32,
    ) -> [f32; 3] {
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        // Seed at rest, then flip the target on at t = 0 so the motion retargets.
        interp.set_now_ms(0);
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let sig = interp.var_signal(&Symbol::intern("on")).unwrap();
        interp.write_var(sig, Value::Bool(true));
        interp.tick();
        let at = |interp: &mut Interpreter, now: u32| {
            interp.set_now_ms(now);
            let mut frame = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut frame, 400.0, 300.0);
            sample(&frame)
        };
        [
            at(&mut interp, 0),
            at(&mut interp, 500),
            at(&mut interp, 1000),
        ]
    }

    #[test]
    fn radius_animates_as_a_paint_prop() {
        let [a, b, c] = ramp_paint_prop(
            "View V() { var on: Bool = false \
             Box #[bg: 0x808080, width: 40, height: 40, radius: on ? 20 : 4 with anim.linear(1000)] }",
            |f| f.instances()[0].radii[0],
        );
        assert!((a - 4.0).abs() < 0.5, "starts near 4, got {a}");
        assert!((b - 12.0).abs() < 1.5, "~halfway, got {b}");
        assert!((c - 20.0).abs() < 0.5, "arrives at 20, got {c}");
    }

    #[test]
    fn border_width_animates_as_a_paint_prop() {
        let sample = |f: &byard_core::frame::RenderFrame| {
            f.decorated()
                .iter()
                .map(|d| d.border_width)
                .fold(0.0_f32, f32::max)
        };
        let [a, b, c] = ramp_paint_prop(
            "View V() { var on: Bool = false \
             Box #[bg: 0x808080, border: 0xFFFFFF, width: 40, height: 40, \
             border_width: on ? 8 : 2 with anim.linear(1000)] }",
            sample,
        );
        assert!((a - 2.0).abs() < 0.5, "starts near 2, got {a}");
        assert!((b - 5.0).abs() < 1.0, "~halfway, got {b}");
        assert!((c - 8.0).abs() < 0.5, "arrives at 8, got {c}");
    }

    #[test]
    fn a_shadow_field_animates() {
        let sample = |f: &byard_core::frame::RenderFrame| {
            f.decorated()
                .iter()
                .map(|d| d.shadow_dy)
                .fold(0.0_f32, f32::max)
        };
        let [a, b, c] = ramp_paint_prop(
            "View V() { var on: Bool = false \
             Box #[bg: 0x808080, width: 40, height: 40, \
             shadow: (y: (on ? 12 : 2) with anim.linear(1000), blur: 8, color: 0x80000000)] }",
            sample,
        );
        assert!((a - 2.0).abs() < 0.6, "starts near 2, got {a}");
        assert!((b - 7.0).abs() < 1.5, "~halfway, got {b}");
        assert!((c - 12.0).abs() < 0.6, "arrives at 12, got {c}");
    }

    /// RFC-0005 `ScrollView`: content is clipped to the viewport and translated
    /// by `−offset`, so scrolling moves the content up without relayout.
    #[test]
    fn scrollview_clips_and_translates_content_by_offset() {
        let src = "View V() { var off: Int = 0 \
             ScrollView #[width: 200, height: 100, offset: (0, off)] { \
                 Column { \
                     Box #[bg: 0xFF0000, width: 180, height: 60] {} \
                     Box #[bg: 0x00FF00, width: 180, height: 60] {} \
                     Box #[bg: 0x0000FF, width: 180, height: 60] {} \
                 } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();

        // The red content box's paint-time translate.y (where the scroll offset
        // lives — the shader applies it, so the layout rect is untouched, i.e.
        // no relayout on scroll), plus whether a content clip was emitted.
        let sample = |interp: &mut Interpreter| -> (f32, f32, usize) {
            let mut frame = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut frame, 400.0, 300.0);
            let red = *frame
                .instances()
                .iter()
                .find(|b| b.color[0] > 0.8 && b.color[1] < 0.3 && b.color[2] < 0.3)
                .expect("the red content box is emitted");
            (red.rect[1], red.transform.translate[1], frame.clips().len())
        };

        let (rect_y0, tx0, clips0) = sample(&mut interp);
        assert!(clips0 >= 1, "the ScrollView must emit a content clip");

        // Scroll down by 40 logical px → the content's paint translate moves up
        // by 40, while its layout rect is unchanged (INV-8: no relayout).
        let off = interp.var_signal(&Symbol::intern("off")).unwrap();
        interp.write_var(off, Value::Int(40));
        interp.tick();
        let (rect_y1, tx1, clips1) = sample(&mut interp);
        assert!(clips1 >= 1);
        assert!(
            (rect_y0 - rect_y1).abs() < 0.01,
            "layout rect must not move on scroll (no relayout): {rect_y0} vs {rect_y1}"
        );
        assert!(
            (tx0 - tx1 - 40.0).abs() < 0.5,
            "content must translate up by the offset: tx0={tx0} tx1={tx1}"
        );
    }

    /// RFC-0005: the mouse wheel over a `ScrollView` scrolls it by writing the
    /// signal behind `offset.y`, clamped to `[0, content − viewport]`.
    #[test]
    fn wheel_over_a_scrollview_scrolls_and_clamps_the_offset() {
        // Content = 4 × 60px = 240 tall in a 100px viewport → max scroll 140.
        let src = "View V() { var off: Float = 0.0 \
             ScrollView #[width: 200, height: 100, offset: (0, off)] { \
                 Column { \
                     Box #[bg: 0xFF0000, width: 180, height: 60] {} \
                     Box #[bg: 0x00FF00, width: 180, height: 60] {} \
                     Box #[bg: 0x0000FF, width: 180, height: 60] {} \
                     Box #[bg: 0xFFFF00, width: 180, height: 60] {} \
                 } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();

        // Render once to record the scroll target (viewport at the top-left).
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let off = interp.var_signal(&Symbol::intern("off")).unwrap();
        let peek_f = |interp: &Interpreter| -> f32 {
            match interp.peek(off) {
                Value::Float(f) => f as f32,
                Value::Int(n) => n as f32,
                _ => panic!("offset must be numeric"),
            }
        };
        let wheel = |interp: &mut Interpreter, dy: f32| {
            interp.dispatch_events(&[byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::Wheel,
                pos: (100.0, 50.0), // inside the 200×100 viewport
                delta: (0.0, dy),
                payload: None,
                time_ms: 0,
            }]);
            interp.tick();
            let mut f = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut f, 400.0, 300.0);
        };

        // Wheel forward by 2 lines (× 40px) → scroll down by 80.
        wheel(&mut interp, -2.0);
        let after_one = peek_f(&interp);
        assert!(
            (after_one - 80.0).abs() < 1.0,
            "one wheel notch scrolls by lines×40, got {after_one}"
        );

        // A big wheel clamps to the content extent (max = 240 − 100 = 140).
        wheel(&mut interp, -20.0);
        let clamped = peek_f(&interp);
        assert!(
            (clamped - 140.0).abs() < 1.0,
            "scroll must clamp to content−viewport, got {clamped}"
        );

        // Wheel back up past the top clamps at 0.
        wheel(&mut interp, 20.0);
        let top = peek_f(&interp);
        assert!(top.abs() < 1.0, "scroll must clamp at 0, got {top}");
    }

    /// RFC-0005 emission culling: a `ScrollView` child scrolled entirely out of
    /// the viewport is never pushed to the frame (only its visible slice costs
    /// anything), while the flat-id cursor stays aligned so siblings still paint.
    #[test]
    fn scrollview_culls_children_scrolled_out_of_view() {
        let src = "View V() { var off: Int = 0 \
             ScrollView #[width: 200, height: 100, offset: (0, off)] { \
                 Column { \
                     Box #[bg: 0xFF0000, width: 180, height: 60] {} \
                     Box #[bg: 0x00FF00, width: 180, height: 60] {} \
                     Box #[bg: 0x0000FF, width: 180, height: 60] {} \
                 } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();

        // Present iff a box with the given dominant colour channel was emitted.
        let has = |interp: &mut Interpreter, chan: usize| -> bool {
            let mut frame = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut frame, 400.0, 300.0);
            frame
                .instances()
                .iter()
                .any(|b| b.color[chan] > 0.8 && b.color[(chan + 1) % 3] < 0.3)
        };

        // At rest, the third box (y 120..180) sits fully below the 100px
        // viewport → culled. The first two overlap it → kept.
        assert!(has(&mut interp, 0), "red (top) is visible at rest");
        assert!(has(&mut interp, 1), "green (straddling) is visible at rest");
        assert!(!has(&mut interp, 2), "blue (below) is culled at rest");

        // Scroll down 120px: the first box (now y -120..-60) leaves the top →
        // culled; the third box scrolls into view.
        let off = interp.var_signal(&Symbol::intern("off")).unwrap();
        interp.write_var(off, Value::Int(120));
        interp.tick();
        assert!(
            !has(&mut interp, 0),
            "red is culled once scrolled past the top"
        );
        assert!(has(&mut interp, 2), "blue scrolls into view");
    }

    /// RFC-0005: dragging on inert `ScrollView` content scrolls it — the content
    /// tracks the pointer between press and release, clamped to the extent.
    #[test]
    fn drag_on_scrollview_content_scrolls_and_clamps() {
        use byard_core::platform::EventKind as K;
        // Content = 4 × 60px = 240 tall in a 100px viewport → max scroll 140.
        let src = "View V() { var off: Float = 0.0 \
             ScrollView #[width: 200, height: 100, offset: (0, off)] { \
                 Column { \
                     Box #[bg: 0xFF0000, width: 180, height: 60] {} \
                     Box #[bg: 0x00FF00, width: 180, height: 60] {} \
                     Box #[bg: 0x0000FF, width: 180, height: 60] {} \
                     Box #[bg: 0xFFFF00, width: 180, height: 60] {} \
                 } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let off = interp.var_signal(&Symbol::intern("off")).unwrap();
        let peek_f = |interp: &Interpreter| match interp.peek(off) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => panic!("offset must be numeric"),
        };
        let ev = |kind, x: f32, y: f32| byard_core::platform::InputEvent {
            kind,
            pos: (x, y),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        };

        // Press on inert content, then drag up 50px → content scrolls down 50.
        interp.dispatch_events(&[ev(K::PointerDown, 100.0, 80.0)]);
        interp.dispatch_events(&[ev(K::PointerMove, 100.0, 30.0)]);
        assert!(
            (peek_f(&interp) - 50.0).abs() < 1.0,
            "drag up 50px scrolls down 50, got {}",
            peek_f(&interp)
        );

        // Dragging further up clamps at the content extent (140).
        interp.dispatch_events(&[ev(K::PointerMove, 100.0, -200.0)]);
        assert!(
            (peek_f(&interp) - 140.0).abs() < 1.0,
            "drag clamps to content−viewport, got {}",
            peek_f(&interp)
        );

        // Releasing ends the gesture: a later stray move no longer scrolls.
        interp.dispatch_events(&[ev(K::PointerUp, 100.0, -200.0)]);
        let held = peek_f(&interp);
        interp.dispatch_events(&[ev(K::PointerMove, 100.0, 300.0)]);
        assert!(
            (peek_f(&interp) - held).abs() < 0.01,
            "no drag is in flight after release, got {} (was {held})",
            peek_f(&interp)
        );
    }

    // ── RFC-0021: snap + pagination + on_end_reached ─────────────────────

    #[test]
    fn page_snap_settles_to_nearest_page_on_release() {
        use byard_core::platform::EventKind as K;
        // Horizontal `snap: page`: three 100px pages in a 100px viewport (max
        // 200). Drag left 60px → offset 60; release snaps to page 1 (offset 100),
        // reflects `page`, and would fire `page_change`.
        let src = "View V() { var offX: Float = 0.0 var offY: Float = 0.0 var pg: Int = 0 \
             ScrollView #[axis: horizontal, snap: page, offset: (offX, offY), page: pg, \
                          width: 100, height: 100] { \
                 Row { Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let off = interp.var_signal(&Symbol::intern("offX")).unwrap();
        let pg = interp.var_signal(&Symbol::intern("pg")).unwrap();
        let peek_f = |interp: &Interpreter| match interp.peek(off) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => 0.0,
        };
        let ev = |kind, x: f32, y: f32| byard_core::platform::InputEvent {
            kind,
            pos: (x, y),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        };

        interp.dispatch_events(&[ev(K::PointerDown, 50.0, 50.0)]);
        interp.dispatch_events(&[ev(K::PointerMove, -10.0, 50.0)]); // drag left 60 → offX 60
        assert!(
            (peek_f(&interp) - 60.0).abs() < 2.0,
            "mid-drag offset ~60, got {}",
            peek_f(&interp)
        );
        interp.dispatch_events(&[ev(K::PointerUp, -10.0, 50.0)]); // release → snap
        assert!(
            (peek_f(&interp) - 100.0).abs() < 1.0,
            "snapped to page 1 (offset 100), got {}",
            peek_f(&interp)
        );
        assert_eq!(interp.peek(pg), Value::Int(1), "the reflected page is 1");
    }

    #[test]
    fn on_end_reached_fires_once_past_threshold() {
        use byard_core::platform::EventKind as K;
        // 400px of content in a 100px viewport (max 300); `end_threshold: 0.8`.
        // Drag up 250 → offset 250 → visible bottom (250+100)/400 = 0.875 ≥ 0.8,
        // so `end_reached` fires and sets `loaded`.
        let src = "View V() { var offX: Float = 0.0 var offY: Float = 0.0 var loaded = false \
             ScrollView #[offset: (offX, offY), width: 100, height: 100, \
                          end_threshold: 0.8, end_reached => loaded = true] { \
                 Column { Box #[width: 100, height: 400] {} } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let loaded = interp.var_signal(&Symbol::intern("loaded")).unwrap();
        let ev = |kind, x: f32, y: f32| byard_core::platform::InputEvent {
            kind,
            pos: (x, y),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        };
        assert_eq!(
            interp.peek(loaded),
            Value::Bool(false),
            "not loaded at rest"
        );

        interp.dispatch_events(&[ev(K::PointerDown, 50.0, 50.0)]);
        interp.dispatch_events(&[ev(K::PointerMove, 50.0, -200.0)]); // drag up 250 → offY 250
        interp.tick();
        assert_eq!(
            interp.peek(loaded),
            Value::Bool(true),
            "end_reached fired past the 0.8 threshold"
        );
    }

    #[test]
    fn setting_the_page_var_scrolls_to_that_page() {
        // Reflected `page:` (reverse): writing `page = 2` scrolls the horizontal
        // offset to page 2 (offset 200) on the next render.
        let src = "View V() { var offX: Float = 0.0 var offY: Float = 0.0 var pg: Int = 0 \
             ScrollView #[axis: horizontal, snap: page, offset: (offX, offY), page: pg, \
                          width: 100, height: 100] { \
                 Row { Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0); // records the scroll target

        let off = interp.var_signal(&Symbol::intern("offX")).unwrap();
        let pg = interp.var_signal(&Symbol::intern("pg")).unwrap();

        interp.write_var(pg, Value::Int(2));
        interp.tick();
        interp.render(&tree, &mut frame, 400.0, 300.0); // sync scrolls to page 2

        let got = match interp.peek(off) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => 0.0,
        };
        assert!(
            (got - 200.0).abs() < 1.0,
            "page = 2 scrolled the offset to 200, got {got}"
        );
    }

    #[test]
    fn page_reflects_on_wheel_scroll_without_a_release() {
        use byard_core::platform::EventKind as K;
        // A trackpad/wheel `Scroll` updates the reflected `page` continuously —
        // no drag release needed (the desktop scrolling case).
        let src = "View V() { var offX: Float = 0.0 var offY: Float = 0.0 var pg: Int = 0 \
             ScrollView #[axis: horizontal, snap: page, offset: (offX, offY), page: pg, \
                          width: 100, height: 100] { \
                 Row { Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let pg = interp.var_signal(&Symbol::intern("pg")).unwrap();
        // Scroll right ~120px (a `Scroll` delta is pixels): offX → 120, page → 1.
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: K::Scroll,
            pos: (50.0, 50.0),
            delta: (-120.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        assert_eq!(
            interp.peek(pg),
            Value::Int(1),
            "page reflects the wheel scroll"
        );
    }

    #[test]
    fn wheel_scroll_snaps_to_a_page_after_settling() {
        use byard_core::platform::EventKind as K;
        // Wheel-scroll 60px, then hold still: once the offset stops moving for a
        // few frames the settle fires and snaps to page 1 — no release event, no
        // wall clock, just observed stillness (clock-independent settle).
        let src = "View V() { var offX: Float = 0.0 var offY: Float = 0.0 \
             ScrollView #[axis: horizontal, snap: page, offset: (offX, offY), \
                          width: 100, height: 100] { \
                 Row { Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let off = interp.var_signal(&Symbol::intern("offX")).unwrap();
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: K::Scroll,
            pos: (50.0, 50.0),
            delta: (-60.0, 0.0), // offX → 60 (not yet a page boundary)
            payload: None,
            time_ms: 0,
        }]);
        // Render successive idle frames with the offset held still; the settle
        // counts stable frames and snaps once it reaches its threshold.
        for _ in 0..8 {
            interp.render(&tree, &mut frame, 400.0, 300.0);
        }
        let got = match interp.peek(off) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => 0.0,
        };
        assert!(
            (got - 100.0).abs() < 1.0,
            "wheel scroll settled and snapped to page 1 (offset 100), got {got}"
        );
    }

    /// RFC-0021: the stillness settle must not fire while a page is *actively*
    /// being scrolled — the offset moving each frame restarts the settle count,
    /// so a mid-scroll frame never snaps out from under the motion.
    #[test]
    fn wheel_scroll_does_not_snap_while_still_moving() {
        use byard_core::platform::EventKind as K;
        let src = "View V() { var offX: Float = 0.0 var offY: Float = 0.0 \
             ScrollView #[axis: horizontal, snap: page, offset: (offX, offY), \
                          width: 100, height: 100] { \
                 Row { Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let off = interp.var_signal(&Symbol::intern("offX")).unwrap();
        let wheel = |dx: f32| byard_core::platform::InputEvent {
            kind: K::Scroll,
            pos: (50.0, 50.0),
            delta: (dx, 0.0),
            payload: None,
            time_ms: 0,
        };
        // Several small scrolls, each followed by a render (offset keeps moving),
        // so the settle count resets every frame and never snaps mid-motion.
        for _ in 0..3 {
            interp.dispatch_events(&[wheel(-20.0)]);
            interp.render(&tree, &mut frame, 400.0, 300.0);
        }
        let got = match interp.peek(off) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => 0.0,
        };
        assert!(
            (got - 60.0).abs() < 1.0,
            "offset must track the in-progress scroll (60), not snap early, got {got}"
        );
    }

    /// RFC-0021: an enum keyword prop (`snap: page`) must keep working even when
    /// the view declares a `var` of the *same name* as the keyword — the token is
    /// read from the AST, so the variable can never shadow it. This is exactly the
    /// `scroll_snap` example's shape (`var page` + `snap: page`), which silently
    /// disabled snapping before enum props stopped resolving through the env.
    #[test]
    fn snap_page_keyword_is_not_shadowed_by_a_same_named_var() {
        use byard_core::platform::EventKind as K;
        let src = "View V() { var offX = 0.0 var offY = 0.0 var page = 0 \
             ScrollView #[axis: horizontal, snap: page, offset: (offX, offY), \
                          page: page, width: 100, height: 100] { \
                 Row { Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let off = interp.var_signal(&Symbol::intern("offX")).unwrap();
        let pg = interp.var_signal(&Symbol::intern("page")).unwrap();
        // Wheel two-thirds of a page (past the snap midpoint) then hold still: the
        // stillness settle must snap forward to page 1, not stay put.
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: K::Scroll,
            pos: (50.0, 50.0),
            delta: (-70.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        for _ in 0..8 {
            interp.render(&tree, &mut frame, 400.0, 300.0);
        }
        let got = match interp.peek(off) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => 0.0,
        };
        assert!(
            (got - 100.0).abs() < 1.0,
            "`snap: page` must engage despite the `var page`; snapped to {got}, want 100"
        );
        assert_eq!(interp.peek(pg), Value::Int(1), "reflected page is 1");
    }

    fn page_snap_view() -> &'static str {
        "View V() { var offX = 0.0 var offY = 0.0 \
             ScrollView #[axis: horizontal, snap: page, offset: (offX, offY), \
                          width: 100, height: 100] { \
                 Row { Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} \
                       Box #[width: 100, height: 100] {} } \
             } }"
    }

    /// RFC-0021 smooth snap: with an advancing clock the offset *glides* to the
    /// page over several frames (a spring), rather than hard-jumping — some frame
    /// must land strictly between the release offset and the page boundary.
    #[test]
    fn page_snap_glides_smoothly_when_a_clock_is_advancing() {
        use byard_core::platform::EventKind as K;
        let parsed = parse(page_snap_view());
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.set_now_ms(0);
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let off = interp.var_signal(&Symbol::intern("offX")).unwrap();
        let peek_f = |i: &Interpreter| match i.peek(off) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => 0.0,
        };
        // Wheel two-thirds of a page toward page 1, then let it settle: past the
        // quiet threshold it springs from 70 → 100.
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: K::Scroll,
            pos: (50.0, 50.0),
            delta: (-70.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        let mut saw_intermediate = false;
        for step in 1..=60 {
            interp.set_now_ms(step * 16);
            interp.render(&tree, &mut frame, 400.0, 300.0);
            let v = peek_f(&interp);
            if v > 70.5 && v < 99.5 {
                saw_intermediate = true; // mid-glide, neither the start nor the page
            }
        }
        assert!(
            saw_intermediate,
            "the offset must glide through intermediate positions, not hard-jump"
        );
        assert!(
            (peek_f(&interp) - 100.0).abs() < 1.0,
            "the glide settles exactly on page 1 (100), got {}",
            peek_f(&interp)
        );
    }

    /// RFC-0021: a stream of shrinking scroll deltas (trackpad momentum) must not
    /// trigger a snap while the fling is still delivering events — the offset
    /// tracks the input and only snaps once the scroll goes quiet, so the snap and
    /// the scroll never fight.
    #[test]
    fn momentum_scroll_does_not_snap_until_it_goes_quiet() {
        use byard_core::platform::EventKind as K;
        let parsed = parse(page_snap_view());
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let off = interp.var_signal(&Symbol::intern("offX")).unwrap();
        let peek_f = |i: &Interpreter| match i.peek(off) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => 0.0,
        };
        let wheel = |dx: f32| byard_core::platform::InputEvent {
            kind: K::Scroll,
            pos: (50.0, 50.0),
            delta: (dx, 0.0),
            payload: None,
            time_ms: 0,
        };
        // Five momentum frames (input every frame) accumulate to offset 75 without
        // ever snapping — each event restarts the quiet countdown.
        let mut acc = 0.0;
        for _ in 0..5 {
            interp.dispatch_events(&[wheel(-15.0)]);
            interp.render(&tree, &mut frame, 400.0, 300.0);
            acc += 15.0;
            assert!(
                (peek_f(&interp) - acc).abs() < 1.0,
                "offset must track momentum ({acc}), never snap mid-fling; got {}",
                peek_f(&interp)
            );
        }
        // Fling over: a few quiet renders and it snaps to the nearest page (75 → 100).
        for _ in 0..8 {
            interp.render(&tree, &mut frame, 400.0, 300.0);
        }
        assert!(
            (peek_f(&interp) - 100.0).abs() < 1.0,
            "once quiet, it snaps to page 1 (100), got {}",
            peek_f(&interp)
        );
    }

    /// RFC-0005: a press that lands on an interactive child (here a `Button`)
    /// is that child's gesture — drag-to-scroll defers and the list stays put.
    #[test]
    fn drag_defers_to_interactive_children() {
        use byard_core::platform::EventKind as K;
        let src = "View V() { var off: Float = 0.0 var c: Int = 0 \
             ScrollView #[width: 200, height: 100, offset: (0, off)] { \
                 Column { \
                     Button(\"tap\") #[width: 180, height: 60] => c++ \
                     Box #[bg: 0x00FF00, width: 180, height: 60] {} \
                     Box #[bg: 0x0000FF, width: 180, height: 60] {} \
                 } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let off = interp.var_signal(&Symbol::intern("off")).unwrap();
        let ev = |kind, x: f32, y: f32| byard_core::platform::InputEvent {
            kind,
            pos: (x, y),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        };

        // Press on the Button (top 60px), then drag: the button owns the press,
        // so the list must not scroll.
        interp.dispatch_events(&[ev(K::PointerDown, 100.0, 30.0)]);
        interp.dispatch_events(&[ev(K::PointerMove, 100.0, -60.0)]);
        let scrolled = match interp.peek(off) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => panic!("numeric"),
        };
        assert!(
            scrolled.abs() < 0.01,
            "a press on a Button must not drag-scroll the list, got {scrolled}"
        );
    }

    /// RFC-0005 `axis: horizontal`: content overflows on the inline axis and the
    /// wheel's x delta scrolls `offset.x`, clamped to the horizontal extent.
    #[test]
    fn horizontal_scrollview_scrolls_offset_x_by_wheel() {
        // A Row of 4 × 100px cards = 400 wide in a 200px viewport → max x 200.
        let src = "View V() { var panX: Float = 0.0 \
             ScrollView #[width: 200, height: 80, axis: horizontal, offset: (panX, 0)] { \
                 Row { \
                     Box #[bg: 0xFF0000, width: 100, height: 60] {} \
                     Box #[bg: 0x00FF00, width: 100, height: 60] {} \
                     Box #[bg: 0x0000FF, width: 100, height: 60] {} \
                     Box #[bg: 0xFFFF00, width: 100, height: 60] {} \
                 } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let pan = interp.var_signal(&Symbol::intern("panX")).unwrap();
        let peek = |interp: &Interpreter| match interp.peek(pan) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => panic!("numeric"),
        };
        // Sample the red card's paint translate.x, so we prove the content
        // actually shifts left (−offset), not just that the signal moved.
        let red_tx = |interp: &mut Interpreter| -> f32 {
            let mut f = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut f, 400.0, 300.0);
            f.instances()
                .iter()
                .find(|b| b.color[0] > 0.8 && b.color[1] < 0.3 && b.color[2] < 0.3)
                .map_or(0.0, |b| b.transform.translate[0])
        };
        let tx0 = red_tx(&mut interp);

        let wheel_x = |interp: &mut Interpreter, dx: f32| {
            interp.dispatch_events(&[byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::Wheel,
                pos: (100.0, 40.0),
                delta: (dx, 0.0),
                payload: None,
                time_ms: 0,
            }]);
            interp.tick();
        };

        // Wheel 2 lines right (×40) → scroll right 80; red card shifts left 80.
        wheel_x(&mut interp, -2.0);
        assert!(
            (peek(&interp) - 80.0).abs() < 1.0,
            "x wheel scrolls, got {}",
            peek(&interp)
        );
        assert!(
            (tx0 - red_tx(&mut interp) - 80.0).abs() < 0.5,
            "content shifts left by the x offset"
        );

        // A big wheel clamps to content−viewport (400 − 200 = 200).
        wheel_x(&mut interp, -20.0);
        assert!(
            (peek(&interp) - 200.0).abs() < 1.0,
            "x clamps to extent, got {}",
            peek(&interp)
        );
    }

    /// RFC-0005 `axis: both`: a single drag pans the content in 2D, each axis
    /// clamped independently.
    #[test]
    fn both_axis_scrollview_pans_in_two_dimensions_by_drag() {
        use byard_core::platform::EventKind as K;
        // A 400×400 content grid in a 200×200 viewport → max 200 on each axis.
        let src = "View V() { var panX: Float = 0.0 var panY: Float = 0.0 \
             ScrollView #[width: 200, height: 200, axis: both, offset: (panX, panY)] { \
                 Column { \
                     Row { Box #[bg: 0xFF0000, width: 200, height: 200] {} \
                           Box #[bg: 0x00FF00, width: 200, height: 200] {} } \
                     Row { Box #[bg: 0x0000FF, width: 200, height: 200] {} \
                           Box #[bg: 0xFFFF00, width: 200, height: 200] {} } \
                 } \
             } }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let px = interp.var_signal(&Symbol::intern("panX")).unwrap();
        let py = interp.var_signal(&Symbol::intern("panY")).unwrap();
        let peek = |interp: &Interpreter, s| match interp.peek(s) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => panic!("numeric"),
        };
        let ev = |kind, x: f32, y: f32| byard_core::platform::InputEvent {
            kind,
            pos: (x, y),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        };

        // Press mid-viewport, drag up-left 60px each → pan right 60, down 60.
        interp.dispatch_events(&[ev(K::PointerDown, 100.0, 100.0)]);
        interp.dispatch_events(&[ev(K::PointerMove, 40.0, 40.0)]);
        assert!(
            (peek(&interp, px) - 60.0).abs() < 1.0,
            "panX, got {}",
            peek(&interp, px)
        );
        assert!(
            (peek(&interp, py) - 60.0).abs() < 1.0,
            "panY, got {}",
            peek(&interp, py)
        );
    }

    /// RFC-0005 windowed layout: a `windowed` ScrollView lays out only the
    /// visible slice of a long uniform list — O(visible), not O(list) — while a
    /// plain ScrollView over the same list lays out every row.
    #[test]
    fn windowed_scrollview_lays_out_only_the_visible_window() {
        // 1000 rows × 20px in a 100px viewport. A windowed pass should build only
        // a handful of rows (viewport/row + overscan) + 2 spacers + containers.
        let list = |windowed: &str| {
            format!(
                "View V() {{ var y: Float = 0.0 \
                 ScrollView #[width: 200, height: 100, row_height: 20, {windowed} offset: (0, y)] {{ \
                     Column {{ \
                         for i in [{}] {{ \
                             Box #[bg: 0x6495ED, width: 180, height: 20] {{}} \
                         }} \
                     }} \
                 }} }}",
                (0..1000)
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let node_count = |src: String| {
            let parsed = parse(&src);
            assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
            let mut interp = Interpreter::new();
            let tree = interp.lower_view(&parsed.views[0], &[]);
            assert!(interp.errors().is_empty(), "{:?}", interp.errors());
            interp.tick();
            let mut frame = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut frame, 400.0, 300.0);
            (interp.atlas_node_count(), frame.instances().len())
        };
        let (windowed_nodes, windowed_boxes) = node_count(list("windowed: true,"));
        let (plain_nodes, _) = node_count(list(""));

        assert!(
            windowed_nodes < 40,
            "a windowed 1000-row list lays out O(visible), got {windowed_nodes} nodes"
        );
        assert!(
            plain_nodes > 1000,
            "a plain list lays out every row, got {plain_nodes} nodes"
        );
        assert!(
            windowed_boxes < 30,
            "only the visible rows are emitted, got {windowed_boxes}"
        );
    }

    /// RFC-0005 windowed layout: the two spacer leaves preserve the full content
    /// extent, so the scroll clamp still reaches the true bottom of the list.
    #[test]
    fn windowed_scrollview_preserves_scroll_extent() {
        // 500 rows × 20 = 10 000 tall in a 100px viewport → max scroll 9 900.
        let src = format!(
            "View V() {{ var y: Float = 0.0 \
             ScrollView #[width: 200, height: 100, row_height: 20, windowed: true, offset: (0, y)] {{ \
                 Column {{ for i in [{}] {{ Box #[bg: 0x6495ED, width: 180, height: 20] {{}} }} }} \
             }} }}",
            (0..500)
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let parsed = parse(&src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let y = interp.var_signal(&Symbol::intern("y")).unwrap();
        // A huge wheel must clamp to content − viewport = 9 900, proving the
        // elided rows still count toward the extent (spacers, not shrinkage).
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::Wheel,
            pos: (100.0, 50.0),
            delta: (0.0, -10_000.0),
            payload: None,
            time_ms: 0,
        }]);
        let clamped = match interp.peek(y) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => panic!("numeric"),
        };
        assert!(
            (clamped - 9_900.0).abs() < 1.0,
            "windowed extent must span the whole list, clamped at {clamped}"
        );
    }

    /// RFC-0005 windowed layout: as the offset grows the window slides, so a row
    /// deep in the list becomes visible while the atlas stays O(visible).
    #[test]
    fn windowed_scrollview_slides_the_window_on_scroll() {
        let src = format!(
            "View V() {{ var y: Float = 0.0 \
             ScrollView #[width: 200, height: 100, row_height: 20, windowed: true, offset: (0, y)] {{ \
                 Column {{ for i in [{}] {{ Box #[bg: 0x6495ED, width: 180, height: 20] {{}} }} }} \
             }} }}",
            (0..500)
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let parsed = parse(&src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();

        // The emitted rows' *layout* Y band (their true list positions, not the
        // scrolled screen position), and the atlas size, at the current offset.
        let sample = |interp: &mut Interpreter| -> (f32, f32, usize) {
            let mut frame = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut frame, 400.0, 300.0);
            let ys: Vec<f32> = frame.instances().iter().map(|b| b.rect[1]).collect();
            let min = ys.iter().copied().fold(f32::INFINITY, f32::min);
            let max = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (min, max, interp.atlas_node_count())
        };
        let (_, at_rest_bottom, nodes0) = sample(&mut interp);
        assert!(
            at_rest_bottom < 300.0,
            "at rest the window sits at the top of the list, got bottom {at_rest_bottom}"
        );

        // Jump near the bottom (offset 8000 → row ~400 of 500). The window must
        // slide to the deep rows: they lay out at y ≈ 8000 (row_index × 20).
        let y = interp.var_signal(&Symbol::intern("y")).unwrap();
        interp.write_var(y, Value::Float(8000.0));
        interp.tick();
        let (deep_top, _, nodes1) = sample(&mut interp);
        assert!(
            deep_top > 7000.0,
            "the window slid to the deep rows (laid out near y≈8000), got top {deep_top}"
        );
        assert!(nodes0 < 40, "the window is O(visible) at rest: {nodes0}");
        assert!(
            nodes1 < 40,
            "the window stays O(visible) after scrolling deep: {nodes1}"
        );
    }

    /// RFC-0005 windowed layout regression: with uniform rows whose stride equals
    /// `row_height`, the materialised rows must stay on an exact `row_height` grid
    /// at every offset — including across a window-slide boundary. A spacer sized
    /// off-grid would shift the whole content when `start` ticks (the "small
    /// jumps" bug), so this pins the invariant that a scroll of 1px moves the
    /// content by exactly 1px, never a row.
    #[test]
    fn windowed_rows_stay_on_an_exact_grid_across_slides() {
        // 500 rows laid out at exactly row_height (height 20, no gap → stride 20).
        let src = format!(
            "View V() {{ var y: Float = 0.0 \
             ScrollView #[width: 200, height: 100, windowed: true, row_height: 20, offset: (0, y)] {{ \
                 Column {{ for i in [{}] {{ Box #[bg: 0x6495ED, width: 180, height: 20] {{}} }} }} \
             }} }}",
            (0..500)
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let parsed = parse(&src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let y = interp.var_signal(&Symbol::intern("y")).unwrap();

        // Sweep offsets straddling several window-slide boundaries (start ticks
        // every 20px). At each, the emitted rows must be exactly 20px apart.
        for off in [0.0, 19.0, 20.0, 21.0, 79.0, 80.0, 81.0, 200.0, 205.0] {
            interp.write_var(y, Value::Float(off));
            interp.tick();
            let mut frame = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut frame, 400.0, 300.0);
            let mut ys: Vec<f32> = frame.instances().iter().map(|b| b.rect[1]).collect();
            ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for w in ys.windows(2) {
                let stride = w[1] - w[0];
                assert!(
                    (stride - 20.0).abs() < 0.01,
                    "rows must stay on the 20px grid at offset {off}, got stride {stride} in {ys:?}"
                );
            }
        }
    }

    #[test]
    fn oklab_hex_round_trips_within_one_lsb() {
        for hex in [
            0x00_0000_i64,
            0xFF_FFFF,
            0x64_95ED,
            0xEF_4444,
            0x10_B981,
            0x80_8080,
        ] {
            let back = hex_from_oklab(oklab_from_hex(hex));
            for shift in [16, 8, 0] {
                let a = (hex >> shift) & 0xFF;
                let b = (back >> shift) & 0xFF;
                assert!(
                    (a - b).abs() <= 1,
                    "channel drift for {hex:#08x}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn with_animation_lerps_color_in_oklab_and_settles() {
        let parsed = parse(
            "View V() { var on: Bool = false \
             Box #[width: 10, height: 10, \
             bg: on ? 0x000000 : 0xFFFFFF with anim.linear(1000)] }",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();

        let render_r = |interp: &mut Interpreter, now: u32| -> f32 {
            interp.set_now_ms(now);
            let mut frame = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut frame, 400.0, 300.0);
            frame.instances()[0].color[0]
        };

        // At rest the target is white; nothing is animating.
        assert!((render_r(&mut interp, 0) - 1.0).abs() < 1e-2);
        assert!(!interp.has_active_animations());

        // Flip toward black: starts near white and is active.
        let sig = interp.var_signal(&Symbol::intern("on")).unwrap();
        interp.write_var(sig, Value::Bool(true));
        interp.tick();
        let start = render_r(&mut interp, 0);
        assert!(start > 0.9, "starts near white, got {start}");
        assert!(interp.has_active_animations());

        // Mid-flight it's a grey between the endpoints, still moving.
        let mid = render_r(&mut interp, 500);
        assert!((0.05..0.95).contains(&mid), "mid-flight grey, got {mid}");

        // Arrives at black and settles (idle again).
        assert!(render_r(&mut interp, 1000) < 1e-2, "arrives black");
        assert!(!interp.has_active_animations());
    }

    #[test]
    fn animation_is_inert_until_the_clock_is_advanced() {
        // A host that never advances the clock must resolve the value to its
        // target and never mark it active — otherwise a wait-based runner would
        // spin forever redrawing a motion pinned at t=0.
        let parsed = parse(
            "View V() { var on: Bool = true \
             Box #[bg: 0x808080, width: 10, height: 10, \
             scale: on ? 2.0 : 1.0 with anim.spring()] }",
        );
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(
            (frame.instances()[0].transform.scale[0] - 2.0).abs() < 1e-6,
            "with no clock the value jumps straight to its target"
        );
        assert!(
            !interp.has_active_animations(),
            "an un-advanced clock must never leave an animation active"
        );
    }

    #[test]
    fn opacity_dims_descendant_text_not_only_the_background() {
        // Regression: a translucent Button dims its *label* too, not just its
        // background — `opacity` folds into the alpha of every primitive the
        // element and its descendants emit.
        let parsed = parse(
            "View V() { var c: Int = 0 \
             Button(\"x\") #[bg: 0x6495ED, opacity: 0.4, width: 100, height: 44] => c++ }",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let label = frame
            .texts()
            .iter()
            .find(|t| t.text == "x")
            .expect("the button's label was emitted");
        assert!(
            (label.color[3] - 0.4).abs() < 1e-3,
            "label alpha should inherit the 0.4 opacity, got {}",
            label.color[3]
        );
    }

    #[test]
    fn style_value_spreads_onto_an_element_and_inline_overrides() {
        // RFC-0016: a `let`-bound style is spliced by `..`, and inline attrs win.
        let parsed = parse(
            "View V() { \
             let btn = style { bg: 0x112233, radius: 8 } \
             Box #[..btn, width: 10, height: 10] {} \
             Box #[..btn, bg: 0x445566, width: 10, height: 10] {} }",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let insts = frame.instances();
        // First box takes `bg` from the spread (0x11 red channel).
        assert!(
            (insts[0].color[0] - f32::from(0x11u8) / 255.0).abs() < 1e-3,
            "spread bg reaches the box, got {:?}",
            insts[0].color
        );
        // Second box: inline `bg` overrides the spread (0x44 red channel).
        assert!(
            (insts[1].color[0] - f32::from(0x44u8) / 255.0).abs() < 1e-3,
            "inline bg overrides the spread, got {:?}",
            insts[1].color
        );
    }

    #[test]
    fn merge_composes_two_styles_right_wins() {
        // RFC-0016: `base merge overrides` — the right style wins on conflicts,
        // the left's non-conflicting attributes survive.
        let parsed = parse(
            "View V() { \
             let base = style { bg: 0x111111, radius: 8 } \
             let hot = base merge style { bg: 0x445566 } \
             Box #[..hot, width: 10, height: 10] {} }",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let inst = frame.instances()[0];
        // `bg` comes from the right side of the merge (0x44 red channel)…
        assert!(
            (inst.color[0] - f32::from(0x44u8) / 255.0).abs() < 1e-3,
            "right style's bg wins, got {:?}",
            inst.color
        );
        // …while `radius` (only on the base) survives (radii != 0).
        assert!(inst.radii[0] > 0.0, "base radius survives the merge");
    }

    #[test]
    fn parent_scale_is_inherited_by_child_text_and_boxes() {
        // RFC-0011 group transforms: a scaled container carries its descendants —
        // the reported bug was that a scaled parent's *text* stayed the same size.
        let parsed = parse(
            "View V() {\n Column #[scale: 2.0, width: 100, height: 100, bg: 0x111111] {\n \
             Text(\"hi\") #[size: 10]\n Box #[bg: 0x222222, width: 20, height: 20]\n }\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // The child text's font size doubled with the parent scale (the fix).
        let line = frame
            .texts()
            .iter()
            .find(|t| t.text == "hi")
            .expect("the child text line is emitted");
        assert!(
            (line.font_size - 20.0).abs() < 1e-3,
            "child text scaled 2× with its parent (10 → 20), got {}",
            line.font_size
        );

        // Both boxes carry a 2× scale: the parent's own, and the child's inherited.
        for inst in frame.instances() {
            assert!(
                (inst.transform.scale[0] - 2.0).abs() < 1e-3
                    && (inst.transform.scale[1] - 2.0).abs() < 1e-3,
                "every box in the group inherits the 2× scale, got {:?}",
                inst.transform.scale
            );
        }
    }

    #[test]
    fn resolve_state_attrs_applies_specificity_then_declaration_order() {
        // RFC-0024 §2: single-state blocks apply in declaration order (equal
        // specificity → later wins); a combined `on hover+focused` (higher
        // specificity) beats both single-state blocks.
        use crate::interp::events::StyleState;
        let sp = crate::diagnostics::Span::new(0, 0);
        let prop = |name: &str, v: i64| Attr {
            name: Symbol::intern(name),
            axis: None,
            kind: AttrKind::Prop {
                value: Expr::IntLit(v, sp),
            },
            span: sp,
        };
        let block = |states: Vec<StyleStateKind>, v: i64| StateBlock {
            states,
            attrs: vec![prop("bg", v)],
            span: sp,
        };
        let base = vec![prop("bg", 1)];
        let blocks = vec![
            block(vec![StyleStateKind::Hover], 2),
            block(vec![StyleStateKind::Disabled], 3),
            block(vec![StyleStateKind::Hover, StyleStateKind::Focused], 4),
        ];

        // No state active → base survives, and the borrow is cheap (no clone).
        let none = resolve_state_attrs(&base, &blocks, StyleState::empty());
        assert!(matches!(none, std::borrow::Cow::Borrowed(_)));
        assert_eq!(find_int(&none, "bg"), Some(1));

        // Hover alone → the hover block overlays (the combined block needs focus).
        let hov = resolve_state_attrs(&base, &blocks, StyleState::HOVER);
        assert_eq!(find_int(&hov, "bg"), Some(2));

        // Hover + disabled (equal specificity) → disabled wins by declaration
        // order (declared after hover).
        let both = resolve_state_attrs(
            &base,
            &blocks,
            StyleState::HOVER.union(StyleState::DISABLED),
        );
        assert_eq!(find_int(&both, "bg"), Some(3));

        // Hover + focused → the combined `hover+focused` block (specificity 2)
        // beats the single-state `hover` block regardless of declaration order.
        let combined =
            resolve_state_attrs(&base, &blocks, StyleState::HOVER.union(StyleState::FOCUSED));
        assert_eq!(find_int(&combined, "bg"), Some(4));
    }

    #[test]
    fn checkbox_on_checked_state_recolours_the_accent() {
        // RFC-0024: a checked value drives the `checked` state, so `on checked`
        // overlays reach the checkbox's own filled-accent visual.
        let src = "View C() {\n var c = true\n \
                   let chk = style { bg: 0x111111 on checked { bg: 0x00FF00 } }\n \
                   Checkbox #[..chk, bind: c, width: 20, height: 20]\n}";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // The filled square is the first decorated box; its accent is the
        // `on checked` green.
        let fill = frame.decorated()[0].base.color;
        assert!(
            fill[1] > 0.9 && fill[0] < 0.1,
            "checked accent is the `on checked` green, got {fill:?}"
        );
    }

    #[test]
    fn universal_selected_prop_drives_the_selected_state() {
        // RFC-0024: `selected: true` on any element activates the `selected`
        // state, so `on selected { bg }` recolours it.
        let src = "View C() {\n \
                   let s = style { bg: 0x111111 on selected { bg: 0x00FF00 } }\n \
                   Box #[..s, selected: true, width: 20, height: 20] {}\n}";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let fill = frame.instances()[0].color;
        assert!(
            fill[1] > 0.9 && fill[0] < 0.1,
            "selected box uses the `on selected` green, got {fill:?}"
        );
    }

    fn find_int(attrs: &[Attr], name: &str) -> Option<i64> {
        attrs
            .iter()
            .find(|a| a.name == Symbol::intern(name))
            .and_then(|a| match &a.kind {
                AttrKind::Prop {
                    value: Expr::IntLit(n, _),
                } => Some(*n),
                _ => None,
            })
    }

    #[test]
    fn disabled_state_block_recolours_in_the_same_frame() {
        // A `disabled:` box with an `on disabled { bg }` block resolves the
        // DISABLED state on the very frame it renders (the router is marked
        // before state styles resolve), so the disabled bg wins immediately.
        let parsed = parse(
            "View V() { \
             let btn = style { bg: 0x111111 on disabled { bg: 0x445566 } } \
             Box #[..btn, disabled: true, width: 40, height: 20] {} }",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let inst = frame.instances()[0];
        assert!(
            (inst.color[0] - f32::from(0x44u8) / 255.0).abs() < 1e-3,
            "disabled bg overlays the base, got {:?}",
            inst.color
        );
    }

    #[test]
    fn hover_state_block_recolours_after_pointer_enters() {
        // RFC-0016: an `on hover { bg }` block lights up once the pointer moves
        // over the element — even though the element registers no handler of its
        // own (it is tracked as a bare hover region).
        let parsed = parse(
            "View V() { \
             let btn = style { bg: 0x111111 on hover { bg: 0x445566 } } \
             Box #[..btn, width: 40, height: 20] {} }",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        // First frame: pointer hasn't entered, base bg (0x11 red channel).
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(
            (frame.instances()[0].color[0] - f32::from(0x11u8) / 255.0).abs() < 1e-3,
            "base bg before hover, got {:?}",
            frame.instances()[0].color
        );

        // Move the pointer inside the box, then re-render.
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::PointerMove,
            pos: (10.0, 10.0),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        interp.tick();
        let mut frame2 = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame2, 400.0, 300.0);
        assert!(
            (frame2.instances()[0].color[0] - f32::from(0x44u8) / 255.0).abs() < 1e-3,
            "hover bg overlays after the pointer enters, got {:?}",
            frame2.instances()[0].color
        );
    }

    #[test]
    fn unknown_state_name_is_an_error_with_a_hint() {
        let parsed =
            parse("View V() { let s = style { bg: 1 on hoover { bg: 2 } } Box #[..s] {} }");
        assert!(
            parsed
                .errors
                .iter()
                .any(|e| matches!(e, CompileError::UnknownStyleState { .. })),
            "an unknown state name must be an UnknownStyleState error, got {:?}",
            parsed.errors
        );
    }

    #[test]
    fn spreading_a_non_style_is_an_error() {
        let parsed = parse("View V() { let x = 5 Box #[..x, width: 10, height: 10] {} }");
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let _ = interp.lower_view(view, &[]);
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::NotAStyle { .. })),
            "spreading a non-style must be a NotAStyle error, got {:?}",
            interp.errors()
        );
    }

    #[test]
    fn no_transform_attrs_produces_identity() {
        let parsed = parse("View C() { Box #[bg: 0xFF0000, width: 50, height: 50] }");
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // `origin` alone isn't checked against `Transform::IDENTITY`'s [0,0]:
        // the compiler defaults an unset `origin` to the element's own
        // center (RFC-0011's stated default), which is a real but *inert*
        // difference from the engine's raw identity — pivot is irrelevant
        // when scale = 1 and rotate = 0, so the render is pixel-identical.
        let t = frame.instances()[0].transform;
        assert_eq!(t.translate, [0.0, 0.0]);
        assert_eq!(t.scale, [1.0, 1.0]);
        assert_eq!(t.rotate, 0.0);
        assert_eq!(t.opacity, 1.0);
    }

    #[test]
    fn sub_property_axis_sets_one_axis_and_leaves_the_other_default() {
        let parsed =
            parse("View C() { Box #[bg: 0xFF0000, width: 50, height: 50, translate.y: 7] }");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert_eq!(frame.instances()[0].transform.translate, [0.0, 7.0]);
    }

    #[test]
    fn named_tuple_scale_sets_one_axis_and_leaves_the_other_at_one() {
        let parsed =
            parse("View C() { Box #[bg: 0xFF0000, width: 50, height: 50, scale: (y: 2.0)] }");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert_eq!(frame.instances()[0].transform.scale, [1.0, 2.0]);
    }

    #[test]
    fn origin_token_resolves_relative_to_the_laid_out_rect() {
        let parsed =
            parse("View C() { Box #[bg: 0xFF0000, width: 40, height: 20, origin: top_left] }");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        // The Box lays out at the view's origin (0,0) by default.
        assert_eq!(frame.instances()[0].transform.origin, [0.0, 0.0]);
    }

    #[test]
    fn unknown_origin_token_is_a_compile_error_with_a_hint() {
        let parsed =
            parse("View C() { Box #[bg: 0xFF0000, width: 50, height: 50, origin: centre] }");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(matches!(
            &interp.errors()[0],
            CompileError::UnknownAttribute { hint: Some(h), .. } if h.contains("center")
        ));
    }

    // ── M16: Toggle/Slider/TextField write-back ──────────────────────────

    #[test]
    fn toggle_with_bg_has_no_background_slab() {
        // Regression: `bg` on a Toggle is the ON accent, not a full-rect fill
        // painted behind the control (that stray slab made widgets look "off").
        let parsed = parse(
            "View C() {\n var on = true\n Toggle #[bind: on, bg: 0x10B981, width: 52, height: 30]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // Exactly track + thumb — no extra background rectangle, no DecoratedBox.
        assert_eq!(
            frame.instances().len(),
            2,
            "toggle should emit only track + thumb"
        );
        assert_eq!(frame.decorated().len(), 0);
    }

    #[test]
    fn slider_with_bg_has_no_background_slab() {
        // Regression: `bg` on a Slider is the fill accent, not a full-rect fill.
        let parsed = parse(
            "View C() {\n var v = 0.5\n Slider #[bind: v, bg: 0xEF4444, min: 0.0, max: 1.0, width: 200, height: 24]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // track + fill + thumb(accent disc) + thumb(white inner) = 4; no slab.
        assert_eq!(
            frame.instances().len(),
            4,
            "slider should emit track + fill + thumb (2 discs), no slab"
        );
    }

    #[test]
    fn toggle_tap_flips_bound_var() {
        let parsed = parse("View C() {\n var on = false\n Toggle #[bind: on]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("on"))
            .unwrap();
        interp.tick();
        assert_eq!(interp.peek(sig), Value::Bool(false));

        // Simulate a render so handlers are registered.
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Tap inside the Toggle rect.
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 50,
            },
        ]);
        interp.tick();
        assert_eq!(
            interp.peek(sig),
            Value::Bool(true),
            "toggle flipped to true"
        );

        // Second tap flips back — gap > DOUBLE_TAP_MS (300ms) so it's a plain tap.
        interp.render(&tree, &mut frame, 400.0, 300.0);
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 400,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 450,
            },
        ]);
        interp.tick();
        assert_eq!(interp.peek(sig), Value::Bool(false), "toggle flipped back");
    }

    // ── RFC-0018: Checkbox ────────────────────────────────────────────────

    /// Renders `src`'s first view and returns `(instances, decorated)` counts,
    /// plus the first decorated box, for the Checkbox visual tests.
    fn checkbox_frame(src: &str) -> byard_core::frame::RenderFrame {
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        frame
    }

    #[test]
    fn unchecked_checkbox_is_a_single_borderless_slot() {
        // Unchecked with no styled border: one decorated slot, no mark. (The
        // container is a DecoratedBox so it *can* carry a border when styled; with
        // none, it's a borderless muted fill.)
        let frame = checkbox_frame("View C() {\n var c = false\n Checkbox #[bind: c]\n}");
        assert_eq!(frame.instances().len(), 0, "no solid instances");
        assert_eq!(frame.decorated().len(), 1, "just the muted slot, no mark");
        let slot = frame.decorated()[0];
        assert!(
            slot.base.color[3] > 0.0,
            "the slot has an opaque muted fill"
        );
        assert_eq!(slot.border_width, 0.0, "no border when none is styled");
    }

    #[test]
    fn checked_checkbox_fills_and_draws_a_two_stroke_check() {
        // Checked: a filled accent square + two checkmark stroke quads on top —
        // three decorated boxes (all on the decorated pipeline, in push order, so
        // the check is never hidden behind the fill).
        let frame = checkbox_frame("View C() {\n var c = true\n Checkbox #[bind: c]\n}");
        assert_eq!(frame.instances().len(), 0, "no solid instances");
        assert_eq!(
            frame.decorated().len(),
            3,
            "filled square + two rotated stroke quads"
        );
        assert!(
            frame.decorated()[0].base.color[3] > 0.0,
            "the checked square has an opaque accent fill"
        );
        // The two strokes rotate in opposite senses about their midpoints — proof
        // the mark is angled geometry, not two axis-aligned bars.
        let r1 = frame.decorated()[1].base.transform.rotate;
        let r2 = frame.decorated()[2].base.transform.rotate;
        assert!(
            (r1 - r2).abs() > 0.1,
            "the two strokes are at different angles"
        );
    }

    #[test]
    fn indeterminate_checkbox_fills_and_draws_a_single_dash() {
        // Mixed state: filled square + one horizontal bar, no checkmark.
        let frame = checkbox_frame(
            "View C() {\n var c = false\n Checkbox #[bind: c, indeterminate: true]\n}",
        );
        assert_eq!(frame.instances().len(), 0, "no solid instances");
        assert_eq!(frame.decorated().len(), 2, "filled square + one dash");
        assert!(
            frame.decorated()[0].base.color[3] > 0.0,
            "filled accent square"
        );
    }

    #[test]
    fn checkbox_bg_is_the_accent_not_a_full_rect_slab() {
        // Regression parity with Toggle/Slider: `bg` on a Checkbox is the checked
        // accent (the box fill), never a background slab — a checked box is the
        // filled square plus the two mark strokes, nothing more.
        let frame = checkbox_frame(
            "View C() {\n var c = true\n Checkbox #[bind: c, bg: 0x10B981, width: 24, height: 24]\n}",
        );
        assert_eq!(
            frame.decorated().len(),
            3,
            "filled square + the two checkmark strokes"
        );
        let fill = frame.decorated()[0].base.color;
        let accent = crate::interp::intrinsics::color_to_rgba(0x0010_B981, false);
        assert!(
            (fill[0] - accent[0]).abs() < 0.01 && (fill[1] - accent[1]).abs() < 0.01,
            "the filled square carries the `bg` accent"
        );
    }

    #[test]
    fn checkbox_tap_flips_bound_var() {
        let parsed = parse("View C() {\n var c = false\n Checkbox #[bind: c]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("c"))
            .unwrap();
        interp.tick();
        assert_eq!(interp.peek(sig), Value::Bool(false));

        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 50,
            },
        ]);
        interp.tick();
        assert_eq!(interp.peek(sig), Value::Bool(true), "tap checked the box");
    }

    #[test]
    fn checkbox_space_key_toggles_when_focused() {
        let parsed = parse("View C() {\n var c = false\n Checkbox #[bind: c]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("c"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // Click to focus the box, then press Space.
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::KeyDown,
                pos: (0.0, 0.0),
                delta: (0.0, 0.0),
                payload: Some(byard_core::platform::InputPayload::Key(" ".to_string())),
                time_ms: 10,
            },
        ]);
        interp.tick();
        assert_eq!(
            interp.peek(sig),
            Value::Bool(true),
            "Space toggled the focused checkbox"
        );
    }

    // ── RFC-0018: RadioButton ─────────────────────────────────────────────

    #[test]
    fn unselected_radio_is_a_ring_only() {
        // `choice != value` → the ring only, no inner dot.
        let frame = checkbox_frame(
            "View C() {\n var choice = \"work\"\n RadioButton #[value: \"home\", bind: choice]\n}",
        );
        assert_eq!(frame.instances().len(), 0, "no inner dot when unselected");
        assert_eq!(frame.decorated().len(), 1, "just the outer ring");
        assert!(
            frame.decorated()[0].border_width > 0.0,
            "the ring is a bordered decorated box"
        );
        assert_eq!(
            frame.decorated()[0].base.color[3],
            0.0,
            "the ring interior is transparent"
        );
    }

    #[test]
    fn selected_radio_draws_ring_plus_inner_dot() {
        // `choice == value` → the ring plus a filled inner dot.
        let frame = checkbox_frame(
            "View C() {\n var choice = \"home\"\n RadioButton #[value: \"home\", bind: choice]\n}",
        );
        assert_eq!(frame.decorated().len(), 1, "the outer ring");
        assert_eq!(frame.instances().len(), 1, "the inner dot");
        assert!(
            frame.instances()[0].color[3] > 0.0,
            "the dot has an opaque accent fill"
        );
    }

    #[test]
    fn radio_tap_selects_its_value() {
        let parsed = parse(
            "View C() {\n var choice = \"home\"\n RadioButton #[value: \"work\", bind: choice]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("choice"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 50,
            },
        ]);
        interp.tick();
        assert_eq!(
            interp.peek(sig),
            Value::Str("work".to_string()),
            "tapping the radio wrote its value to the group var"
        );
    }

    #[test]
    fn radio_group_is_mutually_exclusive_via_the_shared_var() {
        // Two radios on one `var`: tapping the second selects it, which
        // deselects the first (they read the same var — no explicit exclusion).
        let parsed = parse(
            "View C() {\n var choice = \"home\"\n \
             Column #[gap: 40] {\n \
               RadioButton #[value: \"home\", bind: choice, width: 44, height: 44]\n \
               RadioButton #[value: \"work\", bind: choice, width: 44, height: 44]\n \
             }\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("choice"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // Tap the SECOND radio (below the first, gap 40 keeps hit rects apart).
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (20.0, 106.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (20.0, 106.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 50,
            },
        ]);
        interp.tick();
        assert_eq!(
            interp.peek(sig),
            Value::Str("work".to_string()),
            "the group var now holds the second radio's value"
        );
    }

    #[test]
    fn radio_arrow_keys_move_selection_with_wrap() {
        let parsed = parse(
            "View C() {\n var choice = \"home\"\n \
             Column #[gap: 40] {\n \
               RadioButton #[value: \"home\", bind: choice, width: 44, height: 44]\n \
               RadioButton #[value: \"work\", bind: choice, width: 44, height: 44]\n \
               RadioButton #[value: \"other\", bind: choice, width: 44, height: 44]\n \
             }\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("choice"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();

        // Focus the FIRST radio with a press (no release → no tap, so `choice`
        // stays "home").
        interp.render(&tree, &mut frame, 400.0, 300.0);
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::PointerDown,
            pos: (20.0, 20.0),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        interp.tick();
        assert_eq!(interp.peek(sig), Value::Str("home".to_string()));

        // A helper: re-render (repopulates the group ordering + handlers), press
        // the arrow, tick, and read back the group var.
        let press = |interp: &mut Interpreter, key: &str| -> String {
            let mut f = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut f, 400.0, 300.0);
            interp.dispatch_events(&[byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::KeyDown,
                pos: (0.0, 0.0),
                delta: (0.0, 0.0),
                payload: Some(byard_core::platform::InputPayload::Key(key.to_string())),
                time_ms: 10,
            }]);
            interp.tick();
            match interp.peek(sig) {
                Value::Str(s) => s,
                other => panic!("expected Str, got {other:?}"),
            }
        };

        assert_eq!(press(&mut interp, "ArrowDown"), "work", "home → work");
        assert_eq!(press(&mut interp, "ArrowDown"), "other", "work → other");
        assert_eq!(
            press(&mut interp, "ArrowDown"),
            "home",
            "other → home (forward wrap)"
        );
        assert_eq!(
            press(&mut interp, "ArrowUp"),
            "other",
            "home → other (backward wrap)"
        );
    }

    // ── RFC-0018: Grid ────────────────────────────────────────────────────

    #[test]
    fn grid_auto_places_children_into_columns() {
        // Two 1fr columns in a 200px-wide grid → 100px each; two auto-flowed
        // children land one per column (x≈0 and x≈100).
        let frame = checkbox_frame(
            "View C() {\n Grid #[columns: \"1fr 1fr\", width: 200, height: 100] {\n \
               Box #[bg: 0xFF0000, height: 40] {}\n \
               Box #[bg: 0x00FF00, height: 40] {}\n \
             }\n}",
        );
        assert_eq!(frame.instances().len(), 2, "two child fills");
        let mut xs: Vec<f32> = frame.instances().iter().map(|i| i.rect[0]).collect();
        xs.sort_by(f32::total_cmp);
        assert!(xs[0] < 5.0, "first column near x=0, got {xs:?}");
        assert!(
            (xs[1] - 100.0).abs() < 5.0,
            "second column near x=100, got {xs:?}"
        );
    }

    #[test]
    fn grid_explicit_col_placement_pins_a_child() {
        // A lone child pinned to column 2 sits in the right column (x≈100),
        // proving `set_grid_item` wires `col:` through to Taffy.
        let frame = checkbox_frame(
            "View C() {\n Grid #[columns: \"1fr 1fr\", width: 200, height: 100] {\n \
               Box #[bg: 0xFF0000, height: 40, col: 2] {}\n \
             }\n}",
        );
        assert_eq!(frame.instances().len(), 1, "one child fill");
        assert!(
            (frame.instances()[0].rect[0] - 100.0).abs() < 5.0,
            "pinned to column 2 (x≈100), got {:?}",
            frame.instances()[0].rect
        );
    }

    #[test]
    fn grid_row_gap_offsets_the_second_row() {
        // One 1fr column, two explicit 40px rows, row_gap 20 → the second row
        // starts at y = 40 + 20 = 60. (Explicit row tracks so the assertion is
        // independent of grid `align-content`, which stretches *auto* rows to
        // fill a fixed-height container.)
        let frame = checkbox_frame(
            "View C() {\n Grid #[columns: \"1fr\", rows: \"40 40\", row_gap: 20, width: 100] {\n \
               Box #[bg: 0xFF0000] {}\n \
               Box #[bg: 0x00FF00] {}\n \
             }\n}",
        );
        assert_eq!(frame.instances().len(), 2);
        let mut ys: Vec<f32> = frame.instances().iter().map(|i| i.rect[1]).collect();
        ys.sort_by(f32::total_cmp);
        assert!(ys[0] < 5.0, "first row at top, got {ys:?}");
        assert!(
            (ys[1] - 60.0).abs() < 5.0,
            "second row after 40px row + 20px gap, got {ys:?}"
        );
    }

    #[test]
    fn grid_invalid_template_reports_a_diagnostic_and_still_renders() {
        let parsed =
            parse("View C() {\n Grid #[columns: \"1fr bogus\"] { Box #[bg: 0xFF0000] {} }\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::InvalidGridTemplate { .. })),
            "expected InvalidGridTemplate, got {:?}",
            interp.errors()
        );
        // Non-fatal: the grid still lays out (falls back to a single auto column).
        assert_eq!(frame.instances().len(), 1, "the child still renders");
    }

    // ── RFC-0018: ZStack ──────────────────────────────────────────────────

    #[test]
    fn zstack_overlaps_children_at_the_same_origin() {
        // Two children with bg: the small one centres over the big one (they
        // overlap), unlike a Column which would stack them vertically.
        let frame = checkbox_frame(
            "View C() {\n ZStack #[width: 100, height: 100] {\n \
               Box #[bg: 0xFF0000, width: 100, height: 100] {}\n \
               Box #[bg: 0x00FF00, width: 40, height: 40] {}\n \
             }\n}",
        );
        assert_eq!(frame.instances().len(), 2, "two child fills");
        // Declaration order: big first (bottom), small second (on top).
        let big = frame.instances()[0].rect;
        let small = frame.instances()[1].rect;
        assert!(
            big[0] < 5.0 && big[1] < 5.0,
            "big child at origin, got {big:?}"
        );
        // Small (40) centred in the 100 stack → (100 − 40) / 2 = 30.
        assert!(
            (small[0] - 30.0).abs() < 5.0 && (small[1] - 30.0).abs() < 5.0,
            "small child centred, got {small:?}"
        );
    }

    #[test]
    fn zstack_alignment_pins_child_to_corner() {
        // `bottom_end` puts the small child at the bottom-right corner.
        let frame = checkbox_frame(
            "View C() {\n ZStack #[width: 100, height: 100, alignment: bottom_end] {\n \
               Box #[bg: 0xFF0000, width: 100, height: 100] {}\n \
               Box #[bg: 0x00FF00, width: 20, height: 20] {}\n \
             }\n}",
        );
        assert_eq!(frame.instances().len(), 2);
        let small = frame.instances()[1].rect;
        assert!(
            (small[0] - 80.0).abs() < 5.0 && (small[1] - 80.0).abs() < 5.0,
            "small child at bottom-right (80,80), got {small:?}"
        );
    }

    // ── RFC-0005: default text wrap ───────────────────────────────────────

    #[test]
    fn text_wraps_to_parent_width_by_default() {
        // A long line in a narrow fixed-width column wraps with NO explicit
        // `wrap`/`width` — default wrap reflows it to the column's width.
        let parsed = parse(
            "View C() {\n Column #[width: 120] {\n \
               Text(\"This is a fairly long sentence that must wrap within a narrow column.\")\n \
             }\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert_eq!(frame.texts().len(), 1, "one text run");
        let wrap = frame.text_wraps()[0];
        assert!(wrap.is_some(), "default wrap is on");
        assert!(
            wrap.unwrap() <= 130.0,
            "wrapped to ~the 120px column (not the natural width), got {wrap:?}"
        );
    }

    #[test]
    fn wrap_false_opts_out_to_a_single_line() {
        let parsed = parse(
            "View C() {\n Column #[width: 120] {\n \
               Text(\"A long unwrapped line that overflows the column.\") #[wrap: false]\n \
             }\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert_eq!(frame.texts().len(), 1);
        assert!(
            frame.text_wraps()[0].is_none(),
            "wrap: false → single line (no wrap width), got {:?}",
            frame.text_wraps()[0]
        );
    }

    #[test]
    fn slider_drag_sets_float_value() {
        let parsed = parse(
            "View C() {\n var vol = 0.0\n Slider #[bind: vol, min: 0, max: 1, width: 100]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("vol"))
            .unwrap();
        interp.tick();

        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // PointerDown at ~50% of track (x=50 on a 100px track starting at x=0).
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::PointerDown,
            pos: (50.0, 5.0),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        interp.tick();

        let val = match interp.peek(sig) {
            Value::Float(f) => f,
            other => panic!("expected Float, got {other:?}"),
        };
        assert!(
            (val - 0.5).abs() < 0.1,
            "slider at 50% should be ~0.5, got {val}"
        );
    }

    #[test]
    fn slider_value_has_no_f32_widening_tail() {
        // Regression: a drag landing on 0.6 used to be stored as
        // `f64::from(0.6_f32)` = 0.6000000238418579 because the value math ran
        // in f32 and was only widened at the end. The value path now stays in
        // f64, so a pixel-aligned 60% drag round-trips to a clean "0.6".
        let parsed = parse(
            "View C() {\n var vol = 0.0\n Slider #[bind: vol, min: 0, max: 1, width: 100]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("vol"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // x = 60 on a 100px track starting at x = 0 → exactly 60%.
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::PointerDown,
            pos: (60.0, 5.0),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        interp.tick();

        let val = match interp.peek(sig) {
            Value::Float(f) => f,
            other => panic!("expected Float, got {other:?}"),
        };
        assert_eq!(
            format!("{val}"),
            "0.6",
            "slider value must not carry an f32 widening tail"
        );
    }

    #[test]
    fn slider_with_step_does_not_emit_more_decimals_than_the_step() {
        // step: 0.1 landing on 60% used to store 6 * 0.1 = 0.6000000000000001.
        // The value is now rounded to the step's precision → a clean "0.6".
        let parsed = parse(
            "View C() {\n var vol = 0.0\n Slider #[bind: vol, min: 0, max: 1, step: 0.1, width: 100]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("vol"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::PointerDown,
            pos: (60.0, 5.0),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        interp.tick();

        let val = match interp.peek(sig) {
            Value::Float(f) => f,
            other => panic!("expected Float, got {other:?}"),
        };
        assert_eq!(format!("{val}"), "0.6");
    }

    #[test]
    fn text_field_change_event_round_trips() {
        let parsed = parse("View C() {\n var query = \"\"\n TextField #[bind: query]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("query"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Change event with new value.
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::Change,
            pos: (5.0, 5.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Str("hello".to_string())),
            time_ms: 0,
        }]);
        assert_eq!(interp.peek(sig), Value::Str("hello".to_string()));

        // Re-delivering the same value is deduped (E1).
        let before = interp.peek(sig);
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::Change,
            pos: (5.0, 5.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Str("hello".to_string())),
            time_ms: 1,
        }]);
        assert_eq!(interp.peek(sig), before, "equal value deduped");
    }

    #[test]
    fn bind_to_non_var_produces_no_bound_sig() {
        // `let y = 0` is not a var → resolve_bind_sig returns None.
        let parsed = parse("View C() {\n let y = 0\n Toggle #[bind: y]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        // No error expected at lowering; just bound_sig is None (non-var silently ignored).
        let RenderNode::Box { bound_sig, .. } = &tree[0] else {
            panic!("expected Box");
        };
        assert!(bound_sig.is_none(), "let binding yields no bound_sig");
    }

    // ── M17: Keyboard delivery ───────────────────────────────────────────

    #[test]
    fn text_field_receives_keyboard_text_input() {
        let parsed2 = parse("View C() {\n var text = \"\"\n TextField #[bind: text]\n}");
        assert!(parsed2.errors.is_empty(), "{:?}", parsed2.errors);
        let view2 = &parsed2.views[0];
        let mut interp2 = Interpreter::new();
        let tree2 = interp2.lower_view(view2, &[]);
        let sig = interp2
            .var_signal(&crate::symbol::Symbol::intern("text"))
            .unwrap();
        interp2.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp2.render(&tree2, &mut frame, 400.0, 300.0);

        // Focus the TextField by tapping it first.
        interp2.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 50,
            },
        ]);
        interp2.tick();
        interp2.render(&tree2, &mut frame, 400.0, 300.0);

        // Type "ab".
        interp2.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::TextInput,
            pos: (5.0, 5.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key("a".to_string())),
            time_ms: 100,
        }]);
        interp2.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::TextInput,
            pos: (5.0, 5.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key("b".to_string())),
            time_ms: 200,
        }]);
        interp2.tick();
        assert_eq!(
            interp2.peek(sig),
            Value::Str("ab".to_string()),
            "typed 'ab'"
        );

        // Backspace removes last char.
        interp2.render(&tree2, &mut frame, 400.0, 300.0);
        interp2.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::KeyDown,
            pos: (5.0, 5.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key(
                "Backspace".to_string(),
            )),
            time_ms: 300,
        }]);
        interp2.tick();
        assert_eq!(
            interp2.peek(sig),
            Value::Str("a".to_string()),
            "backspace deleted 'b'"
        );
    }

    // ── M18: Tab focus traversal ─────────────────────────────────────────

    #[test]
    fn tab_key_advances_focus_through_text_fields() {
        // Two TextFields — Tab should cycle between them.
        let parsed = parse(
            "View C() {\n var fa = false\n var fb = false\n TextField #[bind: fa, focused: fa]\n TextField #[bind: fb, focused: fb]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let fa = interp
            .var_signal(&crate::symbol::Symbol::intern("fa"))
            .unwrap();
        let fb = interp
            .var_signal(&crate::symbol::Symbol::intern("fb"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Tab: should focus the first field (none focused yet → index 0).
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::KeyDown,
            pos: (0.0, 0.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key("Tab".to_string())),
            time_ms: 0,
        }]);
        interp.tick();
        assert_eq!(
            interp.peek(fa),
            Value::Bool(true),
            "first field focused after Tab"
        );
        assert_eq!(interp.peek(fb), Value::Bool(false));

        // Second Tab: advances to second field.
        interp.render(&tree, &mut frame, 400.0, 300.0);
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::KeyDown,
            pos: (0.0, 0.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key("Tab".to_string())),
            time_ms: 100,
        }]);
        interp.tick();
        assert_eq!(interp.peek(fa), Value::Bool(false), "first field blurred");
        assert_eq!(interp.peek(fb), Value::Bool(true), "second field focused");
    }

    // ── M20: Structural for/when in render tree ──────────────────────────

    #[test]
    fn when_true_includes_then_branch() {
        // RFC-0018: `when` lowers to one reactive `When` node; its taken branch is
        // expanded at paint time. With the condition true, the branch paints.
        let parsed = parse("View C() {\n var show = true\n when show { Text(\"visible\") }\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert_eq!(tree.len(), 1);
        assert!(
            matches!(tree[0], RenderNode::When { .. }),
            "when → When node"
        );
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(frame.texts().len(), 1, "then-branch paints when true");
        assert_eq!(frame.texts()[0].text, "visible");
    }

    #[test]
    fn when_false_emits_nothing_without_else() {
        let parsed = parse("View C() {\n var hide = false\n when hide { Text(\"hidden\") }\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(frame.texts().is_empty(), "false, no else → nothing paints");
    }

    #[test]
    fn when_reacts_to_a_var_flip_at_runtime() {
        // RFC-0018: the whole point — flipping the guard `var` mounts/unmounts the
        // subtree at runtime, with no re-lowering.
        let parsed = parse(
            "View C() {\n var show = false\n when show { Text(\"hi\") } else { Text(\"bye\") }\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        let show = interp
            .var_signal(&crate::symbol::Symbol::intern("show"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(frame.texts()[0].text, "bye", "else branch first");

        interp.write_var(show, Value::Bool(true));
        interp.tick();
        let mut frame2 = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame2, 400.0, 300.0);
        assert_eq!(frame2.texts()[0].text, "hi", "then branch after flip");
    }

    #[test]
    fn for_loop_emits_one_node_per_item() {
        // RFC-0018: `for` lowers to one reactive `For` node; the driver renders one
        // pooled body per element at paint time.
        let parsed =
            parse("View C() {\n var items = [1, 2, 3]\n for item in items { Text(\"{item}\") }\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert!(matches!(tree[0], RenderNode::For { .. }), "for → For node");
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        let texts: Vec<&str> = frame.texts().iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, ["1", "2", "3"], "one node per item, in order");
    }

    #[test]
    fn for_reacts_to_list_growth_and_element_change() {
        // RFC-0018: growing the list mounts more rows; changing an element updates
        // its row — all without re-lowering.
        let parsed = parse("View C() {\n var xs = [10, 20]\n for x in xs { Text(\"{x}\") }\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        let xs = interp
            .var_signal(&crate::symbol::Symbol::intern("xs"))
            .unwrap();
        interp.tick();
        let mut f = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut f, 400.0, 300.0);
        let t: Vec<&str> = f.texts().iter().map(|t| t.text.as_str()).collect();
        assert_eq!(t, ["10", "20"]);

        // Grow + change: [10, 20] → [10, 99, 30].
        interp.write_var(
            xs,
            Value::List(vec![Value::Int(10), Value::Int(99), Value::Int(30)]),
        );
        interp.tick();
        let mut f2 = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut f2, 400.0, 300.0);
        let t2: Vec<&str> = f2.texts().iter().map(|t| t.text.as_str()).collect();
        assert_eq!(t2, ["10", "99", "30"], "grew to 3 rows, element updated");

        // Shrink: [10, 99, 30] → [7].
        interp.write_var(xs, Value::List(vec![Value::Int(7)]));
        interp.tick();
        let mut f3 = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut f3, 400.0, 300.0);
        let t3: Vec<&str> = f3.texts().iter().map(|t| t.text.as_str()).collect();
        assert_eq!(t3, ["7"], "shrank to 1 row");
    }

    // ── M23: Controller boundary ─────────────────────────────────────────

    #[test]
    fn inject_provider_is_visible_to_view() {
        let parsed = parse("View C() {\n inject AppEnv as env\n Text(\"{env}\")\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        // Provide the ambient value before lowering.
        interp.inject_provider("AppEnv", Value::Str("prod".to_string()));
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // The Text should contain the injected value.
        assert_eq!(frame.texts()[0].text, "prod");
    }

    #[test]
    fn apply_io_results_writes_to_var_and_ticks() {
        let parsed = parse("View C() {\n var data = \"\"\n Text(\"{data}\")\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let _tree = interp.lower_view(view, &[]);
        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("data"))
            .unwrap();
        interp.tick();

        // Simulate an async I/O result writing to the `data` var.
        interp.apply_io_results([Box::new(move |interp: &mut Interpreter| {
            interp.write_var(sig, Value::Str("loaded".to_string()));
        }) as Box<dyn FnOnce(&mut Interpreter) + Send>]);
        interp.tick();
        assert_eq!(interp.peek(sig), Value::Str("loaded".to_string()));
    }

    // ── M25: Parameterized fn call sites ─────────────────────────────────

    #[test]
    fn parameterized_fn_call_binds_args() {
        // fn identity(n: Int) => n  →  let y = identity(42)  →  Text renders "42"
        let src = "View C() {\n fn identity(n: Int) => n\n var x = 42\n let y = identity(x)\n Text(\"{y}\")\n}";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(frame.texts()[0].text, "42", "identity(42) == 42");
    }

    #[test]
    fn parameterized_fn_reacts_to_signal_arg() {
        // fn greet(name: Str) => "Hi {name}"  →  reactive on `greeting` signal
        let src = "View C() {\n fn greet(name: Str) => \"Hi {name}\"\n var greeting = \"Alice\"\n let msg = greet(greeting)\n Text(\"{msg}\")\n}";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("greeting"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(frame.texts()[0].text, "Hi Alice");

        // Change greeting → "Bob": msg should become "Hi Bob".
        interp.write_var(sig, Value::Str("Bob".into()));
        interp.tick();
        frame.clear();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(
            frame.texts()[0].text,
            "Hi Bob",
            "greet reacts to signal change"
        );
    }

    // ── M21: DecoratedBox / TextureSampler ───────────────────────────────

    #[test]
    fn image_lowers_to_texture_sampler_in_frame() {
        let parsed = parse(
            "View C() {\n Image(\"photo.jpg\") #[fit: \"cover\", width: 200, height: 150]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert!(
            matches!(tree[0], RenderNode::Image { .. }),
            "Image element lowers to RenderNode::Image"
        );

        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert_eq!(frame.textures().len(), 1, "one TextureSampler in frame");
        let tex = &frame.textures()[0];
        assert_eq!(tex.src, "photo.jpg");
        assert_eq!(tex.fit, byard_core::frame::ImageFit::Cover);
    }

    #[test]
    fn image_fit_defaults_to_fill() {
        let parsed = parse("View C() {\n Image(\"img.png\")\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(frame.textures()[0].fit, byard_core::frame::ImageFit::Fill);
    }

    #[test]
    fn box_with_border_becomes_decorated_box() {
        // `border` is the catalog Color attr; it yields a 2px ring.
        let parsed = parse("View C() {\n Box #[bg: 0xffffff, border: 0x000000]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // A bordered container splits into an opaque SolidBox fill
        // (so it stays behind its children, which also paint as solids) plus a
        // decorated *overlay* whose interior is transparent and only strokes the
        // 2px border — it can't occlude the children drawn beneath it.
        assert_eq!(
            frame.instances().len(),
            1,
            "opaque fill on the SolidBox pass"
        );
        assert_eq!(
            frame.instances()[0].color,
            [1.0, 1.0, 1.0, 1.0],
            "the fill carries the bg colour"
        );
        assert_eq!(frame.decorated().len(), 1, "border overlay → DecoratedBox");
        assert!((frame.decorated()[0].border_width - 2.0).abs() < f32::EPSILON);
        assert_eq!(
            frame.decorated()[0].base.color,
            [0.0; 4],
            "the overlay interior is transparent so children stay visible"
        );
    }

    #[test]
    fn bordered_container_paints_fill_before_its_child_widget() {
        // The regression behind the "widgets invisible" report: an opaque,
        // bordered card must NOT paint over the solid boxes of the widgets it
        // contains. The card's fill is a SolidBox pushed *before*
        // the child's, and the only decorated primitive is a transparent-interior
        // border overlay — so the child's fill is never occluded.
        let parsed = parse(
            "View C() {\n Column #[bg: 0x222233, border: 0x445566] {\n Box #[bg: 0xFF0000, width: 20, height: 20]\n }\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Two solid fills: the card, then the child — in that paint order, so
        // the child (drawn second) lands on top of the card, not under it.
        assert_eq!(frame.instances().len(), 2, "card fill + child fill");
        assert_ne!(
            frame.instances()[0].color,
            [1.0, 0.0, 0.0, 1.0],
            "the first solid fill is the card, not the child"
        );
        assert_eq!(
            frame.instances()[1].color,
            [1.0, 0.0, 0.0, 1.0],
            "the child's red fill paints last (on top)"
        );
        // Every decorated primitive in this frame is a transparent-interior
        // overlay, so nothing opaque is layered above the child.
        assert!(
            frame.decorated().iter().all(|d| d.base.color[3] == 0.0),
            "all decorated overlays have transparent interiors"
        );
    }

    #[test]
    fn box_with_shadow_token_becomes_decorated_box() {
        let parsed = parse("View C() {\n Box #[bg: 0x222222, shadow: \"md\"]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert_eq!(frame.decorated().len(), 1, "shadowed box → DecoratedBox");
        assert!(frame.decorated()[0].shadow_blur > 0.0);
        assert!(
            frame.decorated()[0].shadow_color[3] > 0.0,
            "shadow is translucent"
        );
    }

    /// Renders `src`'s first view and returns the frame's decorated boxes'
    /// shadow triples `(dy, blur, spread)`, for the custom-shadow tests below.
    fn shadow_params(src: &str) -> Vec<(f32, f32, f32)> {
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        frame
            .decorated()
            .iter()
            .map(|d| (d.shadow_dy, d.shadow_blur, d.shadow_spread))
            .collect()
    }

    #[test]
    fn named_custom_shadow_sets_offset_blur_and_spread() {
        let got = shadow_params(
            "View C() { Box #[bg: 0x222222, shadow: (y: 6, blur: 12, spread: 3, color: 0x80000000)] {} }",
        );
        assert_eq!(got.len(), 1, "one shadow instance beneath the fill");
        let (dy, blur, spread) = got[0];
        assert!((dy - 6.0).abs() < 0.01, "dy={dy}");
        assert!((blur - 12.0).abs() < 0.01, "blur={blur}");
        assert!((spread - 3.0).abs() < 0.01, "spread={spread}");
    }

    #[test]
    fn positional_shadow_maps_x_y_blur_spread_color_by_slot() {
        let got =
            shadow_params("View C() { Box #[bg: 0x222222, shadow: (0, 4, 8, 2, 0x80000000)] {} }");
        assert_eq!(got.len(), 1);
        let (dy, blur, spread) = got[0];
        assert!(
            (dy - 4.0).abs() < 0.01 && (blur - 8.0).abs() < 0.01 && (spread - 2.0).abs() < 0.01
        );
    }

    #[test]
    fn layered_shadows_emit_one_instance_each() {
        let got = shadow_params(
            "View C() { Box #[bg: 0x222222, shadow: [(y: 2, blur: 4), (y: 8, blur: 16)]] {} }",
        );
        assert_eq!(got.len(), 2, "two layered shadows → two instances");
        let mut blurs: Vec<f32> = got.iter().map(|s| s.1).collect();
        blurs.sort_by(f32::total_cmp);
        assert!((blurs[0] - 4.0).abs() < 0.01 && (blurs[1] - 16.0).abs() < 0.01);
    }

    #[test]
    fn shadow_none_and_absent_emit_no_shadow() {
        assert!(shadow_params("View C() { Box #[bg: 0x222222] {} }").is_empty());
        assert!(shadow_params("View C() { Box #[bg: 0x222222, shadow: \"none\"] {} }").is_empty());
    }

    // ── M22: Theme system ────────────────────────────────────────────────

    #[test]
    fn text_without_color_uses_theme_on_surface() {
        let parsed = parse("View C() {\n Text(\"hi\")\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let expected_color =
            crate::interp::intrinsics::color_to_rgba(interp.theme.on_surface(), false);
        assert_eq!(
            frame.texts()[0].color,
            expected_color,
            "no-color Text gets theme on_surface"
        );
    }

    #[test]
    fn typo_token_resolves_to_concrete_size() {
        let parsed = parse("View C() {\n Text(\"hi\") #[typo: \"titleLarge\"]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert!(
            (frame.texts()[0].font_size - 22.0).abs() < f32::EPSILON,
            "titleLarge → 22pt, got {}",
            frame.texts()[0].font_size
        );
    }

    // ── RFC-0022: theme runtime (injected reactive tokens) ────────────────

    /// Lowers `src`'s first view against `theme` (installed as the ambient
    /// `Theme`, RFC-0022), ticks, and renders one frame.
    fn theme_render(interp: &mut Interpreter, src: &str) -> byard_core::frame::RenderFrame {
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        frame
    }

    #[test]
    fn injected_theme_color_token_paints_and_flips_with_scheme() {
        let mut interp = Interpreter::new();
        interp.set_theme(super::super::theme::Theme::byard_base());
        let src = "View C() {\n inject Theme as t\n Column #[bg: t.primary] {}\n}";

        let frame = theme_render(&mut interp, src);
        let light = frame.instances()[0].color;
        let expected_light = crate::interp::intrinsics::color_to_rgba(
            interp.theme.color("primary", false).unwrap(),
            false,
        );
        assert_eq!(
            light, expected_light,
            "light scheme paints the light primary"
        );

        // Flip the scheme — a single reactive write — and re-render.
        interp.set_theme_dark(true);
        let mut frame2 = byard_core::frame::RenderFrame::new();
        // Re-lower against the same env so the injected `t` still resolves.
        let tree = interp.lower_view(&parse(src).views[0], &[]);
        interp.tick();
        interp.render(&tree, &mut frame2, 400.0, 300.0);
        let dark = frame2.instances()[0].color;
        let expected_dark = crate::interp::intrinsics::color_to_rgba(
            interp.theme.color("primary", true).unwrap(),
            false,
        );
        assert_eq!(dark, expected_dark, "dark scheme paints the dark primary");
        assert_ne!(light, dark, "flipping the scheme recolours the box");
    }

    #[test]
    fn theme_typo_accessor_sets_font_size() {
        let mut interp = Interpreter::new();
        interp.set_theme(super::super::theme::Theme::byard_base());
        let frame = theme_render(
            &mut interp,
            "View C() {\n inject Theme as t\n Text(\"hi\") #[typo: t.titleLarge]\n}",
        );
        assert!(
            (frame.texts()[0].font_size - 22.0).abs() < f32::EPSILON,
            "t.titleLarge → 22pt, got {}",
            frame.texts()[0].font_size
        );
    }

    #[test]
    fn theme_shape_accessor_sets_corner_radius() {
        let mut interp = Interpreter::new();
        interp.set_theme(super::super::theme::Theme::byard_base());
        let frame = theme_render(
            &mut interp,
            "View C() {\n inject Theme as t\n Box #[bg: 0x222222, radius: t.cornerLg] {}\n}",
        );
        assert!(
            (frame.instances()[0].radii[0] - 16.0).abs() < f32::EPSILON,
            "t.cornerLg → 16px radius, got {:?}",
            frame.instances()[0].radii
        );
    }

    #[test]
    fn manifest_custom_token_resolves_over_base() {
        let mut theme = super::super::theme::Theme::byard_base();
        theme.set_color("light", "primary", 0x0012_3456);
        let mut interp = Interpreter::new();
        interp.set_theme(theme);
        let frame = theme_render(
            &mut interp,
            "View C() {\n inject Theme as t\n Column #[bg: t.primary] {}\n}",
        );
        assert_eq!(
            frame.instances()[0].color,
            crate::interp::intrinsics::color_to_rgba(0x0012_3456, false),
            "a manifest-overridden token wins over byard-base"
        );
    }

    #[test]
    fn unknown_theme_token_is_a_compile_error() {
        let mut interp = Interpreter::new();
        interp.set_theme(super::super::theme::Theme::byard_base());
        let parsed = parse("View C() {\n inject Theme as t\n Column #[bg: t.nope] {}\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let tree = interp.lower_view(&parsed.views[0], &[]);
        interp.tick();
        // The bad token surfaces when the `bg` prop is evaluated at render time.
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(
            interp.errors().iter().any(
                |e| matches!(e, CompileError::UnknownThemeToken { field, .. } if field == "nope")
            ),
            "t.nope → UnknownThemeToken, got {:?}",
            interp.errors()
        );
    }

    #[test]
    fn theme_dark_is_assignable_and_bindable() {
        // `t.dark = …` (assign) and `bind: t.dark` (Toggle) must both resolve to
        // the scheme signal — neither is `NotAssignable` (RFC-0022 §1).
        let mut interp = Interpreter::new();
        interp.set_theme(super::super::theme::Theme::byard_base());
        let _ = theme_render(
            &mut interp,
            "View C() {\n inject Theme as t\n \
             Column {\n Button(\"x\") => t.dark = true\n \
             Toggle #[bind: t.dark]\n }\n}",
        );
        assert!(
            interp.errors().is_empty(),
            "assignable/bindable theme.dark should not error: {:?}",
            interp.errors()
        );
    }

    #[test]
    fn theme_mode_string_reflects_active_scheme() {
        let mut interp = Interpreter::new();
        interp.set_theme(super::super::theme::Theme::byard_base());
        let frame = theme_render(
            &mut interp,
            "View C() {\n inject Theme as t\n Text(\"{t.mode}\")\n}",
        );
        assert_eq!(frame.texts()[0].text, "light");
        assert!(!interp.theme_is_dark());
        interp.set_theme_dark(true);
        assert!(interp.theme_is_dark());
    }

    #[test]
    fn inline_size_overrides_typo_token() {
        let parsed = parse("View C() {\n Text(\"hi\") #[typo: \"titleLarge\", size: 30]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert!(
            (frame.texts()[0].font_size - 30.0).abs() < f32::EPSILON,
            "inline size: 30 overrides typo token"
        );
    }

    #[test]
    fn plain_box_stays_as_box_instance() {
        let parsed = parse("View C() {\n Box #[bg: 0x111111]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert_eq!(frame.instances().len(), 1, "plain box → BoxInstance");
        assert_eq!(frame.decorated().len(), 0);
    }

    // ── RFC-0005: default text wrap ──────────────────────────────────────

    #[test]
    fn explicit_width_pins_wrap_and_yields_a_taller_leaf() {
        let long = "the quick brown fox jumps over the lazy dog again and again";
        // Same text: the first wraps to the 400px root by default, the second is
        // pinned to width 120 (both wrap — wrap is default now).
        let src = format!("View C() {{\n Text(\"{long}\")\n Text(\"{long}\") #[width: 120]\n}}");
        let parsed = parse(&src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert_eq!(frame.texts().len(), 2);
        assert_eq!(frame.text_wraps().len(), 2, "wrap slice parallel to texts");
        let w0 = frame.text_wraps()[0].expect("default wrap on the first line");
        let w1 = frame.text_wraps()[1].expect("explicit-width line still wraps");
        assert!(w0 > 200.0, "first wraps to the wide root, got {w0}");
        assert!(
            (w1 - 120.0).abs() < 1.0,
            "second wraps to its 120 width, got {w1}"
        );

        // The 120-wide leaf is narrower and at least as tall (more lines) as the
        // one that wrapped to the wide root.
        let wide = frame.rects()[1]; // C root is rects[0]; first Text is rects[1]
        let narrow = frame.rects()[2];
        assert!(
            (narrow.width - 120.0).abs() < 1.0,
            "the pinned leaf is 120 wide, got {}",
            narrow.width
        );
        assert!(
            narrow.height >= wide.height,
            "the narrower leaf wraps onto ≥ lines: {} vs {}",
            narrow.height,
            wide.height
        );
    }

    #[test]
    fn reactive_demo_example_reacts_live() {
        // Ties the shipped RFC-0018 example to the suite: `when` toggles a subtree
        // and `for` grows a list, both at runtime.
        let src = include_str!("../../examples/reactive_demo.byd");
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let known: Vec<&str> = parsed.views.iter().map(|v| v.name.as_str()).collect();
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &known);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let show = interp
            .var_signal(&crate::symbol::Symbol::intern("showList"))
            .unwrap();
        let names = interp
            .var_signal(&crate::symbol::Symbol::intern("names"))
            .unwrap();
        let render = |interp: &mut Interpreter| {
            interp.tick();
            let mut f = byard_core::frame::RenderFrame::new();
            interp.render(&tree, &mut f, 640.0, 520.0);
            f.texts().iter().map(|t| t.text.clone()).collect::<Vec<_>>()
        };

        // Initially: list shown → both names present.
        let t0 = render(&mut interp);
        assert!(t0.contains(&"Ada".to_string()) && t0.contains(&"Alan".to_string()));
        assert!(!t0.iter().any(|s| s.starts_with("List hidden")));

        // Hide the list live.
        interp.write_var(show, Value::Bool(false));
        let t1 = render(&mut interp);
        assert!(!t1.contains(&"Ada".to_string()), "list unmounted live");
        assert!(
            t1.iter().any(|s| s.starts_with("List hidden")),
            "else branch"
        );

        // Show again + grow the list live → four rows.
        interp.write_var(show, Value::Bool(true));
        interp.write_var(
            names,
            Value::List(vec![
                Value::Str("Ada".into()),
                Value::Str("Alan".into()),
                Value::Str("Grace".into()),
                Value::Str("Katherine".into()),
            ]),
        );
        let t2 = render(&mut interp);
        for n in ["Ada", "Alan", "Grace", "Katherine"] {
            assert!(t2.contains(&n.to_string()), "{n} row mounted after growth");
        }
    }

    // ── RFC-0017: Overlay & z-layer system ───────────────────────────────

    /// Finds the emitted solid box closest to the given colour channels.
    fn find_solid_by_red(
        frame: &byard_core::frame::RenderFrame,
        red: f32,
    ) -> Option<(usize, byard_core::BoxInstance)> {
        frame
            .instances()
            .iter()
            .enumerate()
            .find(|(_, b)| (b.color[0] - red).abs() < 0.05)
            .map(|(i, b)| (i, *b))
    }

    #[test]
    fn overlay_takes_no_flow_space_and_paints_above_main() {
        // A main box (red) followed by an overlay whose scrim (blue, grow:1)
        // fills the viewport. The overlay must not displace the main box, and
        // its scrim must paint *above* the main box (nearer draw-order depth).
        let src = "View C() {\n \
            Box #[bg: 0xFF0000, width: 40, height: 40] {}\n \
            Overlay #[modal: false] {\n \
                Box #[bg: 0x0000FF, grow: 1] {}\n \
            }\n\
        }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let (red_i, red) = find_solid_by_red(&frame, 1.0).expect("main red box emitted");
        let (blue_i, blue) = frame
            .instances()
            .iter()
            .enumerate()
            .find(|(_, b)| b.color[2] > 0.95 && b.color[0] < 0.05)
            .map(|(i, b)| (i, *b))
            .expect("overlay scrim emitted");

        // Main box keeps its natural 40×40 at the origin — the overlay's 0×0
        // flow leaf did not push it down.
        assert!((red.rect[0]).abs() < 0.01 && (red.rect[1]).abs() < 0.01);
        assert!((red.rect[2] - 40.0).abs() < 0.01);
        // The scrim fills the whole viewport.
        assert!((blue.rect[2] - 400.0).abs() < 0.5 && (blue.rect[3] - 300.0).abs() < 0.5);
        // Painter's order: the overlay is emitted after the main tree, so its
        // depth is strictly nearer (smaller NDC-z) → it composites on top.
        assert!(
            frame.solid_depths()[blue_i] < frame.solid_depths()[red_i],
            "overlay scrim must paint above the main box"
        );
    }

    #[test]
    fn overlay_center_anchor_positions_content_in_the_viewport() {
        let src = "View C() {\n \
            Overlay {\n \
                Column #[anchor: center, bg: 0x222222, width: 100, height: 60] {}\n \
            }\n\
        }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let dialog = frame
            .instances()
            .iter()
            .find(|b| (b.color[0] - 0.133).abs() < 0.05)
            .expect("dialog emitted");
        // Centred: (400−100)/2 = 150, (300−60)/2 = 120.
        assert!(
            (dialog.rect[0] - 150.0).abs() < 1.0,
            "x centred, got {}",
            dialog.rect[0]
        );
        assert!(
            (dialog.rect[1] - 120.0).abs() < 1.0,
            "y centred, got {}",
            dialog.rect[1]
        );
    }

    #[test]
    fn overlay_bottom_anchor_pins_content_to_the_viewport_bottom() {
        let src = "View C() {\n \
            Overlay {\n \
                Column #[anchor: bottom, bg: 0x333333, width: 200, height: 80] {}\n \
            }\n\
        }";
        let parsed = parse(src);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let sheet = frame
            .instances()
            .iter()
            .find(|b| (b.color[0] - 0.2).abs() < 0.05)
            .expect("sheet emitted");
        // Pinned to the bottom: y = 300 − 80 = 220; centred x = (400−200)/2 = 100.
        assert!(
            (sheet.rect[1] - 220.0).abs() < 1.0,
            "y bottom, got {}",
            sheet.rect[1]
        );
        assert!(
            (sheet.rect[0] - 100.0).abs() < 1.0,
            "x centred, got {}",
            sheet.rect[0]
        );
    }

    #[test]
    fn modal_overlay_blocks_the_main_tree_and_dismisses_on_outside_tap() {
        // A main button sits behind a modal overlay. Its scrim fills the
        // viewport; a small confirm button is centred. Tapping the scrim (an
        // outside tap) fires `dismiss` and must NOT reach the button behind.
        let src = "View C() {\n \
            var open = true\n \
            var behind = false\n \
            var confirmed = false\n \
            Button(\"behind\") #[width: 400, height: 300] => behind = true\n \
            Overlay #[modal: true, dismiss => open = false] {\n \
                Box #[bg: 0x000000, opacity: 0.3, grow: 1] {}\n \
                Column #[anchor: center, width: 80, height: 40] {\n \
                    Button(\"ok\") #[width: 80, height: 40] => confirmed = true\n \
                }\n \
            }\n\
        }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let open = interp
            .var_signal(&crate::symbol::Symbol::intern("open"))
            .unwrap();
        let behind = interp
            .var_signal(&crate::symbol::Symbol::intern("behind"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Tap the top-left corner — over the scrim, outside the centred content.
        let tap = |t: u64, p: (f32, f32)| {
            [
                byard_core::platform::InputEvent {
                    kind: byard_core::platform::EventKind::PointerDown,
                    pos: p,
                    delta: (0.0, 0.0),
                    payload: None,
                    time_ms: t,
                },
                byard_core::platform::InputEvent {
                    kind: byard_core::platform::EventKind::PointerUp,
                    pos: p,
                    delta: (0.0, 0.0),
                    payload: None,
                    time_ms: t + 20,
                },
            ]
        };
        interp.dispatch_events(&tap(0, (10.0, 10.0)));
        interp.tick();

        assert_eq!(
            interp.peek(open),
            Value::Bool(false),
            "outside tap dismissed"
        );
        assert_eq!(
            interp.peek(behind),
            Value::Bool(false),
            "modal scrim blocked the button behind it"
        );
    }

    #[test]
    fn modal_overlay_content_wins_over_the_scrim() {
        // A tap on the centred confirm button fires its action, not the scrim's
        // dismiss — the content is registered after the scrim, so it wins.
        let src = "View C() {\n \
            var open = true\n \
            var confirmed = false\n \
            Overlay #[modal: true, dismiss => open = false] {\n \
                Box #[bg: 0x000000, grow: 1] {}\n \
                Column #[anchor: center, width: 80, height: 40] {\n \
                    Button(\"ok\") #[width: 80, height: 40] => confirmed = true\n \
                }\n \
            }\n\
        }";
        let parsed = parse(src);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        let open = interp
            .var_signal(&crate::symbol::Symbol::intern("open"))
            .unwrap();
        let confirmed = interp
            .var_signal(&crate::symbol::Symbol::intern("confirmed"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Centre of the viewport = centre of the confirm button.
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (200.0, 150.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (200.0, 150.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 20,
            },
        ]);
        interp.tick();

        assert_eq!(
            interp.peek(confirmed),
            Value::Bool(true),
            "content button fired"
        );
        assert_eq!(
            interp.peek(open),
            Value::Bool(true),
            "scrim dismiss did NOT fire"
        );
    }

    #[test]
    fn escape_dismisses_the_topmost_modal_overlay() {
        let src = "View C() {\n \
            var open = true\n \
            Overlay #[modal: true, dismiss => open = false] {\n \
                Box #[grow: 1] {}\n \
            }\n\
        }";
        let parsed = parse(src);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        let open = interp
            .var_signal(&crate::symbol::Symbol::intern("open"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::KeyDown,
            pos: (0.0, 0.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key("Escape".into())),
            time_ms: 0,
        }]);
        interp.tick();
        assert_eq!(
            interp.peek(open),
            Value::Bool(false),
            "Escape dismissed the modal"
        );
    }

    #[test]
    fn when_gated_overlay_unmounts_live_on_dismiss() {
        // RFC-0018 × RFC-0017: a `when`-gated modal overlay now dismisses at
        // runtime — tapping the scrim flips the guard `var`, and the very next
        // render unmounts the overlay (no hot-reload needed). This is the headline
        // reactivity win: overlays are live.
        let src = "View C() {\n \
            var open = true\n \
            when open {\n \
                Overlay #[modal: true, dismiss => open = false] {\n \
                    Box #[bg: 0x000000, opacity: 0.4, grow: 1] {}\n \
                    Column #[anchor: center, bg: 0xFFFFFF, width: 100, height: 60] {}\n \
                }\n \
            }\n\
        }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // The overlay is mounted: the white dialog surface is present.
        assert!(
            frame.instances().iter().any(|b| b.color[0] > 0.95),
            "overlay mounted initially"
        );

        // Tap the scrim (top-left, outside the centred content).
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 20,
            },
        ]);
        interp.tick();
        let mut frame2 = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame2, 400.0, 300.0);
        // `open` flipped false → the `when` unmounts the overlay entirely.
        assert!(
            !frame2.instances().iter().any(|b| b.color[0] > 0.95),
            "overlay unmounted live after dismiss"
        );
        assert!(frame2.instances().is_empty(), "nothing left on screen");
    }

    #[test]
    fn non_modal_overlay_lets_taps_fall_through() {
        // A non-modal overlay (a snackbar-style surface) must not block the main
        // tree: a tap on the button behind still fires.
        let src = "View C() {\n \
            var behind = false\n \
            Button(\"behind\") #[width: 400, height: 300] => behind = true\n \
            Overlay #[modal: false] {\n \
                Column #[anchor: bottom, width: 100, height: 20] {}\n \
            }\n\
        }";
        let parsed = parse(src);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        let behind = interp
            .var_signal(&crate::symbol::Symbol::intern("behind"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Tap top-left, away from the bottom-anchored surface.
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (10.0, 10.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (10.0, 10.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 20,
            },
        ]);
        interp.tick();
        assert_eq!(
            interp.peek(behind),
            Value::Bool(true),
            "non-modal overlay let the tap fall through"
        );
    }

    #[test]
    fn overlay_demo_example_renders_dialog_above_the_base_app() {
        // Ties the shipped visual example to the test suite: it must parse,
        // lower, and render, with the modal dialog surface compositing above the
        // base app (RFC-0017). Guards the demo against silent breakage.
        let src = include_str!("../../examples/overlay_demo.byd");
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let known: Vec<&str> = parsed.views.iter().map(|v| v.name.as_str()).collect();
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &known);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 900.0, 560.0);

        // The base app background (0x14141C, red≈0.078) is emitted early; the
        // dialog surface (0xECE6F0, red≈0.925) is an overlay emitted later, so it
        // sits at a nearer depth than the base app.
        let base = frame
            .instances()
            .iter()
            .enumerate()
            .find(|(_, b)| (b.color[0] - 0.078).abs() < 0.02)
            .map(|(i, _)| i)
            .expect("base app background emitted");
        let (dialog, dialog_box) = frame
            .instances()
            .iter()
            .enumerate()
            .find(|(_, b)| b.color[0] > 0.9 && b.color[1] > 0.85)
            .map(|(i, b)| (i, *b))
            .expect("dialog surface emitted");
        assert!(
            frame.solid_depths()[dialog] < frame.solid_depths()[base],
            "the modal dialog must composite above the base app"
        );

        // No dialog text line may overflow the dialog surface — line wrap is not
        // built yet, so the example is authored to fit. Guards the reported
        // overflow against regression: every dark-on-light label painted inside
        // the surface must end before the surface's right edge.
        let mut measurer = byard_core::text::TextMeasurer::new();
        let surf_left = dialog_box.rect[0];
        let surf_right = dialog_box.rect[0] + dialog_box.rect[2];
        for (line, wrap) in frame.texts().iter().zip(frame.text_wraps()) {
            let inside = line.x >= surf_left && line.x < surf_right && line.color[0] < 0.5;
            if inside {
                // Honour the wrap width (RFC-0018): a wrapped label's laid-out
                // width is bounded to it, not the full one-line measurement.
                let (w, _) = measurer.measure_wrapped(&line.text, line.font_size, *wrap);
                assert!(
                    line.x + w <= surf_right + 0.5,
                    "dialog text {:?} overflows the surface: {} + {} > {}",
                    line.text,
                    line.x,
                    w,
                    surf_right
                );
            }
        }
    }

    #[test]
    fn nested_overlays_stack_in_mount_order() {
        // An overlay whose content mounts a second overlay: both are collected,
        // and the inner one is emitted later (on top).
        let src = "View C() {\n \
            Overlay #[modal: false] {\n \
                Box #[bg: 0x111111, grow: 1] {}\n \
                Overlay #[modal: false] {\n \
                    Box #[bg: 0x222222, grow: 1] {}\n \
                }\n \
            }\n\
        }";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(&parsed.views[0], &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let outer = frame
            .instances()
            .iter()
            .enumerate()
            .find(|(_, b)| (b.color[0] - 0.066).abs() < 0.02)
            .map(|(i, _)| i)
            .expect("outer overlay box");
        let inner = frame
            .instances()
            .iter()
            .enumerate()
            .find(|(_, b)| (b.color[0] - 0.133).abs() < 0.02)
            .map(|(i, _)| i)
            .expect("inner overlay box");
        assert!(
            frame.solid_depths()[inner] < frame.solid_depths()[outer],
            "nested overlay stacks above its parent overlay"
        );
    }
}
