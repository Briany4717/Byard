//! The intrinsic catalog (eleven Phase-2 + `VectorIcon` RFC-0009 + `Overlay`
//! RFC-0017 + `Canvas` RFC-0020) and the RFC-0005 §5 attribute contract, plus
//! the RFC-0020 shape-command contract (`validate_canvas`/`validate_shape`).
//!
//! A closed table maps each reserved intrinsic name to its content arity,
//! accepted property/event vocabulary, focusability, and children policy.
//! [`validate_element`] applies the eight §5 rules in order, each producing a
//! precise span-anchored [`CompileError`] — no failure is ever silent (D4,
//! INV-4). Interactive elements register a hit rect inflated to a 44×44 minimum
//! (RFC-0003 E8), computed by [`inflate_hit_rect`].

use std::collections::{HashMap, HashSet};

use crate::diagnostics::CompileError;
use crate::parser::ast::{Attr, AttrKind, ElementNode, Expr, Member};
use crate::util::closest_match;

/// The closed set of intrinsic names (RFC-0005 §4).
pub const INTRINSIC_NAMES: &[&str] = &[
    "Box",
    "Column",
    "Row",
    "Spacer",
    "Text",
    "Button",
    "TextField",
    "Toggle",
    "Slider",
    "Image",
    "ScrollView",
    "VectorIcon",
    "Overlay",
    "Canvas",
];

/// The scalar type an attribute value must have (RFC-0005 §1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropType {
    /// `Int` (logical pixels, counts).
    Int,
    /// `Float` (opacities, slider values).
    Float,
    /// `Bool`.
    Bool,
    /// `Str`.
    Str,
    /// A hex `Color` (`0xRRGGBB` / `0xAARRGGBB`).
    Color,
    /// A `Len`: scalar, pair, or quad.
    Len,
    /// A typography token (or a themed member like `m3.titleLarge`).
    Typo,
    /// An enum token validated against a fixed set.
    Enum(&'static [&'static str]),
    /// A scoped style class reference (`.title`).
    Class,
    /// A `Vec2` `(Float, Float)`.
    Vec2,
    /// An angle (`360deg`/`1.5rad`, RFC-0011 T1) — canonicalized to radians
    /// by the lexer.
    Angle,
    /// A function-valued callback prop.
    Fn,
}

const ALIGN: &[&str] = &["start", "center", "end", "stretch", "justify"];
const JUSTIFY: &[&str] = &["start", "center", "end", "between", "around", "evenly"];
const WEIGHT: &[&str] = &["thin", "regular", "medium", "bold"];
const FIT: &[&str] = &["fill", "contain", "cover", "none"];
const DIRECTION: &[&str] = &["row", "column"];
const AXIS: &[&str] = &["vertical", "horizontal", "both"];
/// Overlay child placement within the full-viewport coordinate space
/// (RFC-0017 §"Positioning"). `center` centres; the edge tokens pin the child
/// to that viewport edge, centred on the cross axis. Absolute `(x, y)` and
/// `relative(ref)` anchoring are deferred (RFC-0017 Future possibilities) —
/// coordinate-passing covers the gap in the interim.
const ANCHOR: &[&str] = &["center", "top", "bottom", "start", "end"];

const LAYOUT: &[(&str, PropType)] = &[
    ("width", PropType::Int),
    ("height", PropType::Int),
    ("gap", PropType::Int),
    ("p", PropType::Len),
    ("m", PropType::Len),
    ("pt", PropType::Len),
    ("pr", PropType::Len),
    ("pb", PropType::Len),
    ("pl", PropType::Len),
    ("mx", PropType::Len),
    ("my", PropType::Len),
    ("mt", PropType::Len),
    ("mr", PropType::Len),
    ("mb", PropType::Len),
    ("ml", PropType::Len),
    ("align", PropType::Enum(ALIGN)),
    ("justify", PropType::Enum(JUSTIFY)),
    ("grow", PropType::Int),
    ("basis", PropType::Int),
];
const DECORATION: &[(&str, PropType)] = &[
    ("bg", PropType::Color),
    ("radius", PropType::Len),
    ("opacity", PropType::Float),
    ("border", PropType::Color),
    ("border_width", PropType::Int),
    ("shadow", PropType::Str),
];
/// Paint-time transform props (RFC-0011). `opacity` is deliberately **not**
/// repeated here — it already lives in [`DECORATION`] and is wired end to
/// end (a non-1.0 `opacity` already promotes a box to the `DecoratedBox`
/// pipeline); this group is only the four props that are new with this RFC.
///
/// Attached everywhere [`DECORATION`] is (every intrinsic sharing the
/// generic container/`Box` render path: `Box`/`Column`/`Row`/`Button`/
/// `TextField`/`Toggle`/`Slider`/`ScrollView`) — **not** `Text`/`Image`,
/// whose engine primitives (`TextLine`/`TextureSampler`) have no `Transform`
/// field yet (see the RFC-0011 engine-slice decision log).
const TRANSFORM: &[(&str, PropType)] = &[
    ("translate", PropType::Vec2),
    ("scale", PropType::Vec2),
    ("rotate", PropType::Angle),
    ("origin", PropType::Vec2),
];
const TEXT_PROPS: &[(&str, PropType)] = &[
    ("typo", PropType::Typo),
    ("color", PropType::Color),
    ("size", PropType::Int),
    ("weight", PropType::Enum(WEIGHT)),
    ("align", PropType::Enum(ALIGN)),
    ("lines", PropType::Int),
    ("wrap", PropType::Bool),
];

const POINTER_EVENTS: &[&str] = &[
    "tap",
    "click", // alias of "tap" (RFC-0012 §A)
    "pointer_down",
    "pointer_up",
    "pointer_move",
    "pointer_enter",
    "pointer_exit",
    "hover",
    "long_press",
    "double_tap",
    "secondary",
    "wheel",
    // `focus =>`/`blur =>` sugar (RFC-0012 S2) makes *any* interactive
    // element focusable on demand (`register_focusable` creates a fresh
    // internal `focused_sig` when `focused:` wasn't given) — so, unlike
    // `key_down`/`key_up` below, these aren't gated behind an intrinsic's
    // *default* focusability.
    "focus",
    "blur",
];
const KEY_EVENTS: &[&str] = &["key_down", "key_up"];

/// The accepted vocabulary of one intrinsic.
pub struct Intrinsic {
    /// Number of positional `(...)` content arguments.
    pub arity: usize,
    /// The content type, when `arity > 0`.
    pub content: Option<PropType>,
    /// Whether the intrinsic accepts a `{ … }` children block.
    pub children: bool,
    /// Whether the intrinsic is focusable by default.
    pub focusable: bool,
    /// Whether attaching a pointer/keyboard listener registers a hit rect.
    pub interactive: bool,
    props: HashMap<&'static str, PropType>,
    events: HashSet<&'static str>,
}

impl Intrinsic {
    /// Returns the type of property `name`, if recognized.
    #[must_use]
    pub fn property_type(&self, name: &str) -> Option<PropType> {
        self.props.get(name).copied()
    }

    /// Returns `true` if `name` is a recognized event.
    #[must_use]
    pub fn has_event(&self, name: &str) -> bool {
        self.events.contains(name)
    }

    /// Returns an iterator over all recognized property names and their types.
    pub fn properties(&self) -> impl Iterator<Item = (&'static str, PropType)> + '_ {
        self.props.iter().map(|(&k, &v)| (k, v))
    }

    /// Returns an iterator over all recognized event names.
    pub fn events(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.events.iter().copied()
    }
}

fn props_from(groups: &[&[(&'static str, PropType)]]) -> HashMap<&'static str, PropType> {
    let mut m = HashMap::new();
    for g in groups {
        for &(k, v) in *g {
            m.insert(k, v);
        }
    }
    // Universal props.
    m.insert("style", PropType::Class);
    m
}

fn events_from(focusable: bool, extra: &[&'static str]) -> HashSet<&'static str> {
    let mut s: HashSet<&'static str> = POINTER_EVENTS.iter().copied().collect();
    if focusable {
        s.extend(KEY_EVENTS.iter().copied());
    }
    s.extend(extra.iter().copied());
    s
}

/// Looks up the intrinsic named `name` (RFC-0005 §4 table).
#[must_use]
pub fn lookup(name: &str) -> Option<Intrinsic> {
    let container = |dir_default: bool| {
        let mut props = props_from(&[LAYOUT, DECORATION, TRANSFORM]);
        if dir_default {
            props.insert("direction", PropType::Enum(DIRECTION));
        }
        props.insert("focused", PropType::Bool);
        props.insert("disabled", PropType::Bool);
        // RFC-0017: a child of an `Overlay` may carry an `anchor` placing it
        // within the viewport. Harmless outside an overlay (no-op in normal
        // flow), so it lives on every container rather than a special case.
        props.insert("anchor", PropType::Enum(ANCHOR));
        Intrinsic {
            arity: 0,
            content: None,
            children: true,
            focusable: false,
            interactive: true,
            props,
            events: events_from(false, &[]),
        }
    };
    Some(match name {
        "Box" => container(true),
        "Column" | "Row" => container(false),
        "Spacer" => Intrinsic {
            arity: 0,
            content: None,
            children: false,
            focusable: false,
            interactive: false,
            props: props_from(&[&[("grow", PropType::Int), ("basis", PropType::Int)]]),
            events: HashSet::new(),
        },
        "Text" => Intrinsic {
            arity: 1,
            content: Some(PropType::Str),
            children: false,
            focusable: false,
            interactive: true,
            props: props_from(&[
                TEXT_PROPS,
                &[("m", PropType::Len), ("width", PropType::Int)],
            ]),
            events: events_from(false, &[]),
        },
        "Button" => {
            let mut props = props_from(&[LAYOUT, DECORATION, TEXT_PROPS, TRANSFORM]);
            props.insert("focused", PropType::Bool);
            props.insert("disabled", PropType::Bool);
            Intrinsic {
                arity: 1,
                content: Some(PropType::Str),
                children: false,
                focusable: true,
                interactive: true,
                props,
                events: events_from(true, &[]),
            }
        }
        "TextField" => {
            let mut props = props_from(&[LAYOUT, DECORATION, TEXT_PROPS, TRANSFORM]);
            props.insert("placeholder", PropType::Str);
            props.insert("value", PropType::Str);
            props.insert("bind", PropType::Str);
            props.insert("focused", PropType::Bool);
            props.insert("disabled", PropType::Bool);
            Intrinsic {
                arity: 0,
                content: None,
                children: false,
                focusable: true,
                interactive: true,
                props,
                events: events_from(true, &["change", "input", "submit"]),
            }
        }
        "Toggle" => {
            let mut props = props_from(&[LAYOUT, DECORATION, TRANSFORM]);
            props.insert("value", PropType::Bool);
            props.insert("bind", PropType::Bool);
            props.insert("focused", PropType::Bool);
            props.insert("disabled", PropType::Bool);
            Intrinsic {
                arity: 0,
                content: None,
                children: false,
                focusable: true,
                interactive: true,
                props,
                events: events_from(true, &["change"]),
            }
        }
        "Slider" => {
            let mut props = props_from(&[LAYOUT, DECORATION, TRANSFORM]);
            for k in ["min", "max", "step"] {
                props.insert(k, PropType::Float);
            }
            props.insert("value", PropType::Float);
            props.insert("bind", PropType::Float);
            props.insert("focused", PropType::Bool);
            props.insert("disabled", PropType::Bool);
            Intrinsic {
                arity: 0,
                content: None,
                children: false,
                focusable: true,
                interactive: true,
                props,
                events: events_from(true, &["change"]),
            }
        }
        "Image" => {
            let mut props = props_from(&[LAYOUT]);
            props.insert("radius", PropType::Len);
            props.insert("opacity", PropType::Float);
            props.insert("fit", PropType::Enum(FIT));
            Intrinsic {
                arity: 1,
                content: Some(PropType::Str),
                children: false,
                focusable: false,
                interactive: true,
                props,
                events: events_from(false, &[]),
            }
        }
        "ScrollView" => {
            let mut props = props_from(&[LAYOUT, DECORATION, TRANSFORM]);
            props.insert("axis", PropType::Enum(AXIS));
            props.insert("offset", PropType::Vec2);
            // RFC-0005 windowed layout: opt-in list virtualization. `windowed`
            // materialises only the visible slice of a uniform-height vertical
            // list; `row_height` is that fixed per-row **stride** the window math
            // indexes by. It MUST equal each row's laid-out outer height, because
            // windowing lays the list out gap-free (a flex `gap` can't survive
            // virtualization) — so fold any inter-row spacing into the row itself
            // (its `height` or a `mb` margin), not the container's `gap`. A
            // `row_height` that disagrees with the real stride makes the content
            // jump as rows scroll past the edge.
            props.insert("windowed", PropType::Bool);
            props.insert("row_height", PropType::Int);
            Intrinsic {
                arity: 0,
                content: None,
                children: true,
                focusable: false,
                interactive: true,
                props,
                events: events_from(false, &["scroll"]),
            }
        }
        // The twelfth intrinsic (RFC-0009 §1, RFC-0005 amendment): an MSDF vector
        // glyph. Content arity 1 = an asset handle (a `Str` path resolved against
        // the asset table, like `Image`). Props: `size`, `color`, `m`, `opacity`,
        // universal `style`; pointer events match `Image`. No children. Routes to
        // the `VectorMSDF` pipeline.
        "VectorIcon" => {
            let mut props: HashMap<&'static str, PropType> = HashMap::new();
            props.insert("size", PropType::Int);
            props.insert("color", PropType::Color);
            props.insert("m", PropType::Len);
            props.insert("opacity", PropType::Float);
            props.insert("style", PropType::Class);
            Intrinsic {
                arity: 1,
                content: Some(PropType::Str),
                children: false,
                focusable: false,
                interactive: true,
                props,
                events: events_from(false, &[]),
            }
        }
        // RFC-0020: a fixed-size programmatic drawing surface. Content: none.
        // Children: shape commands only (`arc`, `circle`, `line`, `rect`,
        // `path`, `bezier`, `text`) — validated by [`validate_canvas`], not the
        // generic children rule. Props: `width`/`height` (required — a canvas
        // never sizes to content), `bg` (background fill), `grow`, margins,
        // `opacity`, universal `style`. Events: all pointer events, hit-tested
        // against the canvas rect only (individual shapes are not hit-testable;
        // RFC-0020 resolved question).
        "Canvas" => {
            let mut props: HashMap<&'static str, PropType> = HashMap::new();
            props.insert("width", PropType::Int);
            props.insert("height", PropType::Int);
            props.insert("bg", PropType::Color);
            props.insert("grow", PropType::Int);
            props.insert("m", PropType::Len);
            props.insert("opacity", PropType::Float);
            props.insert("style", PropType::Class);
            Intrinsic {
                arity: 0,
                content: None,
                children: true,
                focusable: false,
                interactive: true,
                props,
                events: events_from(false, &[]),
            }
        }
        // RFC-0017: the overlay escape-hatch. Content: none. Children: the
        // overlay's floating subtree, laid out against the viewport rather than
        // the parent's flow. Props: `modal` (default true — captures all input
        // behind a scrim) and `dismiss_on_outside` (default true when modal). It
        // is layout-only itself (occupies zero space in its parent); its children
        // route to their own pipelines. The `dismiss` event fires when a modal
        // overlay's scrim is tapped or `Escape` is pressed.
        "Overlay" => {
            let mut props: HashMap<&'static str, PropType> = HashMap::new();
            props.insert("modal", PropType::Bool);
            props.insert("dismiss_on_outside", PropType::Bool);
            props.insert("style", PropType::Class);
            Intrinsic {
                arity: 0,
                content: None,
                children: true,
                focusable: false,
                interactive: true,
                props,
                events: events_from(false, &["dismiss"]),
            }
        }
        _ => return None,
    })
}

/// Validates `el` against the §5 contract, returning every diagnostic it
/// produces (possibly several). `known_views` are the user `ViewDecl` names in
/// scope, so a non-intrinsic element that resolves to a view is not an error.
#[must_use]
pub fn validate_element(
    el: &ElementNode,
    attrs: &[Attr],
    known_views: &[&str],
) -> Vec<CompileError> {
    let mut errs = Vec::new();
    let name = el.name.as_str();

    // Rule 1 — name resolution.
    let Some(info) = lookup(name) else {
        if !known_views.contains(&name) {
            // A shape command reached the ordinary element path — it was
            // written outside a `Canvas` body (RFC-0020 §3). More precise
            // than the generic unknown-view diagnostic.
            if is_shape_command(name) {
                errs.push(CompileError::ShapeOutsideCanvas {
                    span: el.span,
                    name: name.to_string(),
                });
                return errs;
            }
            let hint = closest_match(
                name,
                INTRINSIC_NAMES
                    .iter()
                    .copied()
                    .chain(known_views.iter().copied()),
            )
            .map(str::to_string);
            errs.push(CompileError::UnknownView {
                span: el.span,
                name: name.to_string(),
                hint,
            });
        }
        // A user view: its own body is validated when that view is checked.
        return errs;
    };

    // Rule 2 — content arity.
    if el.content.len() != info.arity {
        errs.push(CompileError::ArityMismatch {
            span: el.span,
            name: name.to_string(),
            expected: info.arity,
            found: el.content.len(),
        });
    } else if let Some(ty) = info.content {
        // Rule 3 — content type.
        for arg in &el.content {
            if let Some(err) = check_value_type(ty, &arg.value) {
                errs.push(err);
            }
        }
    }

    // Rule 8 — children on a childless intrinsic.
    if !info.children && !el.children.is_empty() {
        errs.push(CompileError::UnexpectedChildren {
            span: el.span,
            name: name.to_string(),
        });
    }

    // Rules 4–6 — per attribute.
    for attr in attrs {
        let an = attr.name.as_str();
        let is_prop = matches!(attr.kind, AttrKind::Prop { .. });
        let prop_ty = info.props.get(an).copied();
        let is_event = info.events.contains(an);

        if prop_ty.is_none() && !is_event {
            // Rule 4 — unknown attribute.
            let hint = closest_match(
                an,
                info.props
                    .keys()
                    .copied()
                    .chain(info.events.iter().copied()),
            )
            .map(str::to_string);
            errs.push(CompileError::UnknownAttribute {
                span: attr.span,
                name: an.to_string(),
                hint,
            });
            continue;
        }

        // Rule 5 — separator/kind.
        if is_prop && prop_ty.is_none() && is_event {
            errs.push(CompileError::WrongAttributeSeparator {
                span: attr.span,
                name: an.to_string(),
                expected_property: false,
            });
            continue;
        }
        if !is_prop && prop_ty.is_some() && !is_event {
            errs.push(CompileError::WrongAttributeSeparator {
                span: attr.span,
                name: an.to_string(),
                expected_property: true,
            });
            continue;
        }

        // Rule 6 — attribute value type.
        if let (AttrKind::Prop { value }, Some(ty)) = (&attr.kind, prop_ty) {
            // RFC-0010: `value with anim.*(…)` — reject an animation on a layout
            // property (it can't animate on the GPU), otherwise validate every
            // curve in the (possibly nested) chain and type-check the innermost
            // target value. The chain walk matters: `(x with a) with b` must not
            // let its inner curve or value slip past unchecked.
            if let Expr::Animated { span, .. } = value {
                if is_layout_prop(an) {
                    errs.push(CompileError::LayoutPropNotAnimatable {
                        span: *span,
                        prop: an.to_string(),
                    });
                } else {
                    let mut target = value;
                    while let Expr::Animated {
                        value: inner, anim, ..
                    } = target
                    {
                        if let Err(err) = crate::interp::anim::resolve_curve(anim) {
                            errs.push(err);
                        }
                        target = inner;
                    }
                    if let Some(err) = check_value_type(ty, target) {
                        errs.push(err);
                    }
                }
            } else if let Some(err) = check_value_type(ty, value) {
                errs.push(err);
            }
        }
    }

    errs
}

/// Whether `name` is a layout-affecting attribute — one whose value feeds Taffy
/// and so cannot be GPU-animated (RFC-0010 §"Layout properties"). Covers the
/// [`LAYOUT`] group plus the container `direction`.
fn is_layout_prop(name: &str) -> bool {
    name == "direction" || LAYOUT.iter().any(|(k, _)| *k == name)
}

/// Light, false-positive-averse type check: only clear scalar-literal
/// mismatches and unknown enum tokens are flagged; identifiers/members (which
/// may resolve to a reactive `var`) are accepted.
fn check_value_type(ty: PropType, value: &Expr) -> Option<CompileError> {
    let span = value.span();
    let mismatch = |what: &str| {
        Some(CompileError::AttributeTypeMismatch {
            span,
            expected: what.to_string(),
        })
    };
    match (ty, value) {
        (PropType::Color, Expr::StrLit(..) | Expr::FloatLit(..)) => mismatch("a color (0xRRGGBB)"),
        (PropType::Int | PropType::Len, Expr::StrLit(..)) => mismatch("an integer length"),
        (PropType::Angle, Expr::StrLit(..) | Expr::IntLit(..) | Expr::FloatLit(..)) => {
            mismatch("an angle (e.g. 90deg, 1.5rad)")
        }
        (PropType::Angle, Expr::Tuple(args, _)) => {
            // Verbose `rotate: (angle: <expr>)` — recurse into the field so a
            // bare number can't slip past the terse-form rejection by hiding in
            // the tuple wrapper (which is otherwise not type-checked).
            match args.as_slice() {
                [arg] if arg.name.as_ref().is_some_and(|n| n.as_str() == "angle") => {
                    check_value_type(PropType::Angle, &arg.value)
                }
                _ => mismatch("an angle (e.g. 90deg) or the verbose form `(angle: 90deg)`"),
            }
        }
        (PropType::Str, Expr::IntLit(..) | Expr::FloatLit(..)) => mismatch("a string"),
        (PropType::Bool, Expr::IntLit(..) | Expr::StrLit(..) | Expr::FloatLit(..)) => {
            mismatch("a boolean")
        }
        (PropType::Class, e) if !matches!(e, Expr::ClassRef(..)) => {
            mismatch("a style class (.name)")
        }
        (PropType::Enum(set), Expr::Ident(sym, _)) => {
            let tok = sym.as_str();
            if tok == "true" || tok == "false" || set.contains(&tok) {
                None
            } else {
                let hint = closest_match(tok, set.iter().copied()).map(str::to_string);
                Some(CompileError::AttributeTypeMismatch {
                    span,
                    expected: hint.map_or_else(
                        || format!("one of {set:?}"),
                        |h| format!("one of {set:?} (did you mean `{h}`?)"),
                    ),
                })
            }
        }
        _ => None,
    }
}

// ── Canvas shape commands (RFC-0020) ─────────────────────────────────────────

/// The closed set of shape-command names valid inside a `Canvas` body
/// (RFC-0020 §"Shape commands").
pub const SHAPE_COMMAND_NAMES: &[&str] =
    &["arc", "circle", "line", "rect", "path", "bezier", "text"];

/// Whether `name` is one of the RFC-0020 shape commands.
#[must_use]
pub fn is_shape_command(name: &str) -> bool {
    SHAPE_COMMAND_NAMES.contains(&name)
}

/// Line-cap tokens (RFC-0020 §"Stroke and fill").
const CAP: &[&str] = &["butt", "round", "square"];
/// Line-join tokens. Accepted for forward-compatibility; v1's shape set has
/// no polyline joints (`rect` corners are exact SDF, `bezier` flattens to
/// round-capped segments), so the value does not yet change rendering.
const JOIN: &[&str] = &["miter", "round", "bevel"];

/// The stroke/fill/paint parameters every geometric shape accepts
/// (RFC-0020 §"Stroke and fill").
const SHAPE_PAINT_PARAMS: &[(&str, PropType)] = &[
    ("stroke", PropType::Color),
    ("stroke_width", PropType::Float),
    ("cap", PropType::Enum(CAP)),
    ("join", PropType::Enum(JOIN)),
    ("fill", PropType::Color),
    ("dash", PropType::Vec2),
    ("dash_offset", PropType::Float),
    ("opacity", PropType::Float),
];

/// A static table of shape-parameter `(name, type)` pairs.
type ShapeParams = &'static [(&'static str, PropType)];

/// Geometry parameters per shape command: `(required, optional)` name/type
/// pairs, not counting the shared [`SHAPE_PAINT_PARAMS`].
fn shape_geometry(name: &str) -> (ShapeParams, ShapeParams) {
    match name {
        "arc" => (
            &[
                ("cx", PropType::Float),
                ("cy", PropType::Float),
                ("r", PropType::Float),
            ],
            // `start`/`sweep` default to 0°/360° — an unswept arc is a circle.
            &[("start", PropType::Float), ("sweep", PropType::Float)],
        ),
        "circle" => (
            &[
                ("cx", PropType::Float),
                ("cy", PropType::Float),
                ("r", PropType::Float),
            ],
            &[],
        ),
        "line" => (
            &[
                ("x1", PropType::Float),
                ("y1", PropType::Float),
                ("x2", PropType::Float),
                ("y2", PropType::Float),
            ],
            &[],
        ),
        "rect" => (
            &[
                ("x", PropType::Float),
                ("y", PropType::Float),
                ("w", PropType::Float),
                ("h", PropType::Float),
            ],
            &[("radius", PropType::Float)],
        ),
        "path" => (&[("d", PropType::Str)], &[]),
        "bezier" => (
            &[
                ("x1", PropType::Float),
                ("y1", PropType::Float),
                ("cx1", PropType::Float),
                ("cy1", PropType::Float),
                ("cx2", PropType::Float),
                ("cy2", PropType::Float),
                ("x2", PropType::Float),
                ("y2", PropType::Float),
            ],
            &[],
        ),
        // Canvas `text`: positional content (the string) is handled by the
        // caller; these are its named parameters.
        "text" => (
            &[("x", PropType::Float), ("y", PropType::Float)],
            &[
                ("color", PropType::Color),
                ("size", PropType::Float),
                ("align", PropType::Enum(&["start", "center", "end"])),
            ],
        ),
        _ => (&[], &[]),
    }
}

/// Validates a `Canvas` element (RFC-0020 §1): required `width`/`height`
/// props, and a body of shape commands only — each checked against its
/// geometry/paint parameter contract with the same precision as RFC-0005 §5's
/// attribute rules. Call alongside [`validate_element`] (which covers the
/// canvas's own attrs/events through the ordinary intrinsic contract).
#[must_use]
pub fn validate_canvas(el: &ElementNode, attrs: &[Attr]) -> Vec<CompileError> {
    let mut errs = Vec::new();

    // A canvas is a fixed-size surface: it never sizes to content, so both
    // dimensions are required up front.
    let has_prop = |name: &str| {
        attrs
            .iter()
            .any(|a| a.name.as_str() == name && matches!(a.kind, AttrKind::Prop { .. }))
    };
    if !has_prop("width") || !has_prop("height") {
        errs.push(CompileError::CanvasMissingSize { span: el.span });
    }

    for member in &el.children {
        match member {
            Member::Element(child) if is_shape_command(child.name.as_str()) => {
                errs.extend(validate_shape(child));
            }
            Member::Element(child) => {
                let name = child.name.as_str();
                errs.push(CompileError::UnknownShapeCommand {
                    span: child.span,
                    name: name.to_string(),
                    hint: closest_match(name, SHAPE_COMMAND_NAMES.iter().copied())
                        .map(str::to_string),
                });
            }
            // Declarations, control flow, and style blocks are not shape
            // commands (RFC-0020 §1). Reported with the member keyword so the
            // message reads naturally.
            Member::Var { span, .. } => push_non_shape(&mut errs, *span, "var"),
            Member::Let { span, .. } => push_non_shape(&mut errs, *span, "let"),
            Member::Fn { span, .. } => push_non_shape(&mut errs, *span, "fn"),
            Member::Inject { span, .. } => push_non_shape(&mut errs, *span, "inject"),
            Member::For { span, .. } => push_non_shape(&mut errs, *span, "for"),
            Member::When { span, .. } => push_non_shape(&mut errs, *span, "when"),
            Member::Style { span, .. } => push_non_shape(&mut errs, *span, "style"),
            Member::Expr(e) => push_non_shape(&mut errs, e.span(), "an expression"),
        }
    }
    errs
}

/// Helper for [`validate_canvas`]: a non-element member inside a `Canvas`.
fn push_non_shape(errs: &mut Vec<CompileError>, span: crate::diagnostics::Span, what: &str) {
    errs.push(CompileError::UnknownShapeCommand {
        span,
        name: what.to_string(),
        hint: None,
    });
}

/// Validates one shape command against its parameter contract (RFC-0020):
/// unknown parameters (with a Levenshtein hint), missing required geometry,
/// scalar-literal type mismatches, no attribute block, no children — and the
/// `path`-is-fill-only rule ([`CompileError::PathStrokeUnsupported`]).
#[must_use]
pub fn validate_shape(el: &ElementNode) -> Vec<CompileError> {
    let mut errs = Vec::new();
    let shape = el.name.as_str();
    let (required, optional) = shape_geometry(shape);

    // Shape commands carry everything in their `(...)` argument list: an
    // attribute block or a children block has no meaning on one.
    for attr in &el.attrs {
        errs.push(CompileError::UnknownShapeParam {
            span: attr.span,
            shape: shape.to_string(),
            name: attr.name.as_str().to_string(),
            hint: None,
        });
    }
    if !el.children.is_empty() {
        errs.push(CompileError::UnexpectedChildren {
            span: el.span,
            name: shape.to_string(),
        });
    }

    let param_type = |name: &str| -> Option<PropType> {
        required
            .iter()
            .chain(optional)
            .chain(SHAPE_PAINT_PARAMS)
            .find(|(k, _)| *k == name)
            .map(|(_, t)| *t)
    };
    // `text` is fill-rendered glyphs, not a stroked path: it takes only its
    // own geometry/typography params, never the paint set.
    let paint_allowed = shape != "text";

    // `bezier` accepts the terse positional form (8 numbers, RFC-0020 table)
    // as well as the named form; `text` takes its content string positionally.
    let positional_budget = match shape {
        "bezier" => required.len(),
        "text" => 1,
        _ => 0,
    };
    let mut positional_seen = 0usize;

    for arg in &el.content {
        // A positional arg only spends the shape's positional budget
        // (`bezier`'s 8 coordinates, canvas `text`'s content string).
        let Some(name) = &arg.name else {
            positional_seen += 1;
            if positional_seen > positional_budget {
                errs.push(CompileError::UnknownShapeParam {
                    span: arg.value.span(),
                    shape: shape.to_string(),
                    name: "<positional>".to_string(),
                    hint: None,
                });
            }
            continue;
        };
        let pname = name.as_str();
        let known = param_type(pname)
            .is_some_and(|_| paint_allowed || !SHAPE_PAINT_PARAMS.iter().any(|(k, _)| *k == pname));
        if !known {
            let candidates =
                required
                    .iter()
                    .chain(optional)
                    .map(|(k, _)| *k)
                    .chain(if paint_allowed {
                        SHAPE_PAINT_PARAMS
                            .iter()
                            .map(|(k, _)| *k)
                            .collect::<Vec<_>>()
                    } else {
                        Vec::new()
                    });
            errs.push(CompileError::UnknownShapeParam {
                span: arg.value.span(),
                shape: shape.to_string(),
                name: pname.to_string(),
                hint: closest_match(pname, candidates).map(str::to_string),
            });
        } else if let Some(ty) = param_type(pname) {
            // Same literal-level check as attribute values, including the
            // RFC-0010 `with` animation chain walk.
            let mut target = &arg.value;
            while let Expr::Animated { value, anim, .. } = target {
                if let Err(err) = crate::interp::anim::resolve_curve(anim) {
                    errs.push(err);
                }
                target = value;
            }
            if let Some(err) = check_value_type(ty, target) {
                errs.push(err);
            }
        }
    }

    // Required geometry. A fully-positional `bezier` (all 8 coordinates in
    // order) satisfies its required set; canvas `text`'s positional content
    // string is checked separately below.
    let named_has = |name: &str| {
        el.content
            .iter()
            .any(|a| a.name.as_ref().is_some_and(|n| n.as_str() == name))
    };
    let bezier_positional = shape == "bezier" && positional_seen == required.len();
    if !bezier_positional {
        for (name, _) in required {
            if !named_has(name) {
                errs.push(CompileError::MissingShapeParam {
                    span: el.span,
                    shape: shape.to_string(),
                    name: (*name).to_string(),
                });
            }
        }
    }
    if shape == "text" && positional_seen == 0 {
        errs.push(CompileError::MissingShapeParam {
            span: el.span,
            shape: shape.to_string(),
            name: "content".to_string(),
        });
    }

    // RFC-0020 §2 Tier 2 is fill-only in v1: stroking a `path` is rejected
    // rather than silently ignored (never-silent, RFC-0002 D4 spirit).
    if shape == "path" {
        let strokes = [
            "stroke",
            "stroke_width",
            "cap",
            "join",
            "dash",
            "dash_offset",
        ];
        if el.content.iter().any(|a| {
            a.name
                .as_ref()
                .is_some_and(|n| strokes.contains(&n.as_str()))
        }) {
            errs.push(CompileError::PathStrokeUnsupported { span: el.span });
        }
    }

    errs
}

/// An axis-aligned rectangle in logical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl Rect {
    /// Creates a rectangle.
    #[must_use]
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }
}

/// Minimum hit-target size in logical pixels (RFC-0003 E8).
pub const HIT_MIN: f32 = 44.0;

/// Default MSDF distance range in atlas texels for a `VectorIcon` (RFC-0009
/// §2-E), used by the placeholder lowering until the generator bakes a
/// per-glyph value. Ties to the generation grid (a 32² grid with a 4-texel
/// range gives a clean edge under heavy magnification).
pub const VECTOR_DEFAULT_PX_RANGE: f32 = 4.0;

/// Inflates an interactive element's collision rect to at least 44×44, centered
/// on the original rect and clamped to `parent` (RFC-0003 E8).
#[must_use]
pub fn inflate_hit_rect(rect: Rect, parent: Rect) -> Rect {
    let w = rect.w.max(HIT_MIN);
    let h = rect.h.max(HIT_MIN);
    let cx = rect.x + rect.w / 2.0;
    let cy = rect.y + rect.h / 2.0;
    let x = (cx - w / 2.0).clamp(parent.x, (parent.x + parent.w - w).max(parent.x));
    let y = (cy - h / 2.0).clamp(parent.y, (parent.y + parent.h - h).max(parent.y));
    Rect { x, y, w, h }
}

/// Parses a `Color` integer into RGBA `[f32; 4]` (6-digit ⇒ opaque, 8-digit ⇒
/// alpha-first `0xAARRGGBB`) — RFC-0005 §1.
#[must_use]
pub fn color_to_rgba(hex: i64, alpha_byte: bool) -> [f32; 4] {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let v = hex as u32;
    let f = |b: u32| (b & 0xFF) as f32 / 255.0;
    if alpha_byte {
        [f(v >> 16), f(v >> 8), f(v), f(v >> 24)]
    } else {
        [f(v >> 16), f(v >> 8), f(v), 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::Member;
    use crate::parser::parse;

    fn first_element(src: &str) -> ElementNode {
        let parsed = parse(src);
        assert!(
            parsed.errors.is_empty(),
            "parse errors: {:?}",
            parsed.errors
        );
        match parsed
            .views
            .into_iter()
            .next()
            .unwrap()
            .body
            .into_iter()
            .next()
            .unwrap()
        {
            Member::Element(e) => e,
            _ => panic!("expected element"),
        }
    }

    fn errs(src: &str) -> Vec<CompileError> {
        let el = first_element(src);
        validate_element(&el, &el.attrs, &[])
    }

    #[test]
    fn valid_intrinsics_pass() {
        assert!(errs("View V() { Text(\"hi\") #[color: 0xFFFFFF, align: center] }").is_empty());
        assert!(errs("View V() { Column #[gap: 8, p: 16] { } }").is_empty());
        assert!(errs("View V() { Button(\"+\") #[bg: 0x3B82F6] => x }").is_empty());
    }

    #[test]
    fn transform_props_are_accepted_on_containers_but_not_text_or_image() {
        assert!(
            errs(
                "View V() { Box #[translate: (0, 2), scale: 1.05, rotate: 90deg, origin: center] {} }"
            )
            .is_empty()
        );
        assert!(
            errs("View V() { Row #[scale.y: 1.2] {} }").is_empty(),
            "sub-property axis form"
        );

        // `Text`/`Image` don't have a `Transform` field on their engine
        // primitives yet (RFC-0011 engine-slice decision log) — these must
        // still report `UnknownAttribute`, not silently accept and drop.
        let e = errs("View V() { Text(\"hi\") #[rotate: 90deg] }");
        assert!(matches!(&e[0], CompileError::UnknownAttribute { .. }));
        let e = errs("View V() { Image(\"x\") #[translate: (0, 2)] }");
        assert!(matches!(&e[0], CompileError::UnknownAttribute { .. }));
    }

    #[test]
    fn rotate_rejects_a_bare_number_without_a_deg_or_rad_suffix() {
        let e = errs("View V() { Box #[rotate: 90] {} }");
        assert!(matches!(&e[0], CompileError::AttributeTypeMismatch { .. }));
    }

    #[test]
    fn rotate_verbose_form_still_rejects_a_bare_number() {
        // The verbose `(angle: N)` wrapper must not let a bare number bypass the
        // deg/rad requirement — recurse into the field.
        let e = errs("View V() { Box #[rotate: (angle: 90)] {} }");
        assert!(matches!(&e[0], CompileError::AttributeTypeMismatch { .. }));
        // …but the properly-suffixed verbose form is accepted.
        assert!(errs("View V() { Box #[rotate: (angle: 90deg)] {} }").is_empty());
        // A verbose tuple with the wrong field name is a mismatch too.
        let e = errs("View V() { Box #[rotate: (deg: 90deg)] {} }");
        assert!(matches!(&e[0], CompileError::AttributeTypeMismatch { .. }));
    }

    #[test]
    fn with_animation_on_a_paint_prop_is_accepted() {
        // RFC-0010: paint-time animatable props accept a `with` curve.
        assert!(errs("View V() { Box #[scale: 1 with anim.spring()] {} }").is_empty());
        assert!(errs("View V() { Box #[opacity: 0.5 with anim.linear(200ms)] {} }").is_empty());
    }

    #[test]
    fn with_animation_unknown_curve_is_an_error_with_a_hint() {
        let e = errs("View V() { Box #[scale: 1 with anim.sprng()] {} }");
        assert!(matches!(
            &e[0],
            CompileError::UnknownAnimation { hint: Some(h), .. } if h == "spring"
        ));
    }

    #[test]
    fn with_animation_on_a_layout_prop_is_rejected() {
        // Animating `width` would relayout every frame — a compile error, not a
        // silent slowdown (RFC-0010 §"Layout properties").
        let e = errs("View V() { Box #[width: 100 with anim.spring()] {} }");
        assert!(matches!(
            &e[0],
            CompileError::LayoutPropNotAnimatable { .. }
        ));
    }

    #[test]
    fn nested_animated_values_still_check_the_innermost_value_and_every_curve() {
        // A parenthesised `(x with a) with b` must not let its inner value or
        // curve slip past the checker.
        let e = errs(
            "View V() { Box #[radius: (\"hi\" with anim.spring()) with anim.linear(200ms)] {} }",
        );
        assert!(
            e.iter()
                .any(|err| matches!(err, CompileError::AttributeTypeMismatch { .. })),
            "innermost `\"hi\"` must be type-checked against `radius`"
        );
        let e =
            errs("View V() { Box #[radius: (3 with anim.sprng()) with anim.linear(200ms)] {} }");
        assert!(
            e.iter()
                .any(|err| matches!(err, CompileError::UnknownAnimation { .. })),
            "a bad nested curve must still be reported"
        );
    }

    #[test]
    fn rule1_unknown_view_suggests() {
        let e = errs("View V() { Colunm #[gap: 8] {} }");
        assert!(matches!(
            &e[0],
            CompileError::UnknownView { hint: Some(h), .. } if h == "Column"
        ));
    }

    #[test]
    fn rule1_known_user_view_is_ok() {
        // `Card` is not an intrinsic but is a known view in scope.
        let el = first_element("View V() { Card #[gap: 8] {} }");
        assert!(validate_element(&el, &el.attrs, &["Card"]).is_empty());
    }

    #[test]
    fn rule2_arity_mismatch() {
        // Text takes exactly one content arg.
        let e = errs("View V() { Text(\"a\", \"b\") }");
        assert!(matches!(
            &e[0],
            CompileError::ArityMismatch {
                expected: 1,
                found: 2,
                ..
            }
        ));
        // Column takes none.
        let e = errs("View V() { Column(\"x\") }");
        assert!(
            e.iter()
                .any(|d| matches!(d, CompileError::ArityMismatch { expected: 0, .. }))
        );
    }

    #[test]
    fn rule4_unknown_attribute_suggests_gap() {
        let e = errs("View V() { Column #[gp: 1] {} }");
        assert!(matches!(
            &e[0],
            CompileError::UnknownAttribute { hint: Some(h), .. } if h == "gap"
        ));
    }

    #[test]
    fn rule4_value_on_box_is_unknown_attribute() {
        let e = errs("View V() { Box #[value: 1] {} }");
        assert!(matches!(&e[0], CompileError::UnknownAttribute { .. }));
    }

    #[test]
    fn rule5_wrong_separator() {
        // `gap` is a property; using `=>` is a separator error.
        let e = errs("View V() { Column #[gap => 1] {} }");
        assert!(matches!(
            &e[0],
            CompileError::WrongAttributeSeparator {
                expected_property: true,
                ..
            }
        ));
        // `tap` is an event; using `:` is a separator error.
        let e = errs("View V() { Button(\"x\") #[tap: 1] }");
        assert!(matches!(
            &e[0],
            CompileError::WrongAttributeSeparator {
                expected_property: false,
                ..
            }
        ));
    }

    #[test]
    fn rule6_type_and_enum_token() {
        // A string where a color is expected.
        let e = errs("View V() { Column #[bg: \"red\"] {} }");
        assert!(matches!(&e[0], CompileError::AttributeTypeMismatch { .. }));
        // An unknown enum token.
        let e = errs("View V() { Column #[align: centr] {} }");
        assert!(matches!(&e[0], CompileError::AttributeTypeMismatch { .. }));
    }

    #[test]
    fn rule8_children_on_childless_intrinsic() {
        let e = errs("View V() { Text(\"hi\") { Text(\"no\") } }");
        assert!(
            e.iter()
                .any(|d| matches!(d, CompileError::UnexpectedChildren { .. }))
        );
    }

    #[test]
    fn hit_rect_inflates_small_button_clamped_to_parent() {
        let parent = Rect::new(0.0, 0.0, 200.0, 200.0);
        let inflated = inflate_hit_rect(Rect::new(0.0, 0.0, 10.0, 10.0), parent);
        assert!(inflated.w >= HIT_MIN && inflated.h >= HIT_MIN);
        // Stays within the parent scissor.
        assert!(inflated.x >= parent.x && inflated.y >= parent.y);
        assert!(inflated.x + inflated.w <= parent.x + parent.w);
        assert!(inflated.y + inflated.h <= parent.y + parent.h);
    }

    #[test]
    fn vector_icon_validates_like_an_asset_handle_intrinsic() {
        // Valid: arity-1 asset handle + size/color props.
        assert!(
            errs("View V() { VectorIcon(\"icons/gear.svg\") #[size: 24, color: 0xFFFFFF] }")
                .is_empty()
        );
        // Arity 0 and 2 → ArityMismatch.
        assert!(
            errs("View V() { VectorIcon() }")
                .iter()
                .any(|e| matches!(e, CompileError::ArityMismatch { expected: 1, .. }))
        );
        assert!(
            errs("View V() { VectorIcon(\"a.svg\", \"b.svg\") }")
                .iter()
                .any(|e| matches!(
                    e,
                    CompileError::ArityMismatch {
                        expected: 1,
                        found: 2,
                        ..
                    }
                ))
        );
        // A child block → UnexpectedChildren.
        assert!(
            errs("View V() { VectorIcon(\"a.svg\") { Text(\"no\") } }")
                .iter()
                .any(|e| matches!(e, CompileError::UnexpectedChildren { .. }))
        );
        // An unknown attribute (e.g. gradient) → UnknownAttribute.
        assert!(
            errs("View V() { VectorIcon(\"a.svg\") #[gradient: 0x00FF00] }")
                .iter()
                .any(|e| matches!(e, CompileError::UnknownAttribute { .. }))
        );
    }

    #[test]
    fn overlay_validates_as_a_childful_layout_intrinsic() {
        // Valid: modal overlay with a scrim + content, and a `dismiss` event.
        assert!(
            errs(
                "View V() { Overlay #[modal: true] { Box #[bg: 0x000000, opacity: 0.3, grow: 1] {} \
                 Column #[anchor: center, bg: 0xFFFFFF] { Text(\"hi\") } } }"
            )
            .is_empty()
        );
        // `dismiss` is an event, so `=>` is correct.
        assert!(errs("View V() { Overlay #[dismiss => x] { Box {} } }").is_empty());
        // Content args are rejected (arity 0).
        assert!(
            errs("View V() { Overlay(\"x\") { Box {} } }")
                .iter()
                .any(|e| matches!(e, CompileError::ArityMismatch { expected: 0, .. }))
        );
        // A stray prop → UnknownAttribute.
        assert!(
            errs("View V() { Overlay #[z_index: 3] { Box {} } }")
                .iter()
                .any(|e| matches!(e, CompileError::UnknownAttribute { .. }))
        );
        // `dismiss` with `:` instead of `=>` is a separator error (it's an event).
        assert!(
            errs("View V() { Overlay #[dismiss: 1] { Box {} } }")
                .iter()
                .any(|e| matches!(e, CompileError::WrongAttributeSeparator { .. }))
        );
    }

    #[test]
    fn anchor_enum_is_accepted_on_containers_and_rejects_unknown_tokens() {
        assert!(errs("View V() { Column #[anchor: bottom] {} }").is_empty());
        assert!(errs("View V() { Box #[anchor: center] {} }").is_empty());
        let e = errs("View V() { Box #[anchor: middle] {} }");
        assert!(matches!(&e[0], CompileError::AttributeTypeMismatch { .. }));
    }

    #[test]
    fn color_parsing() {
        let green = color_to_rgba(0x00_FF_00, false);
        assert!(green[0] < 0.01 && green[1] > 0.99 && green[2] < 0.01 && green[3] > 0.99);
        let c = color_to_rgba(0x80_00_00_00, true);
        assert!((c[3] - 0.5019).abs() < 0.01, "alpha-first 0x80… ≈ 0.5");
    }

    // ── Canvas & shape commands (RFC-0020) ─────────────────────────────────────

    /// `validate_element` + `validate_canvas` on the first element of `src` —
    /// the exact pair the evaluator's `Canvas` lowering runs.
    fn canvas_errs(src: &str) -> Vec<CompileError> {
        let el = first_element(src);
        let mut e = validate_element(&el, &el.attrs, &[]);
        e.extend(validate_canvas(&el, &el.attrs));
        e
    }

    #[test]
    fn valid_canvas_with_shapes_passes() {
        let e = canvas_errs(
            "View V() { Canvas #[width: 48, height: 48, bg: 0x1E1E2A] { \
               arc(cx: 24, cy: 24, r: 20, start: -90, sweep: 270, \
                   stroke: 0x6750A4, stroke_width: 4, cap: round) \
               circle(cx: 24, cy: 24, r: 8, fill: 0xE8DEF8) \
               line(x1: 0, y1: 0, x2: 48, y2: 48, stroke: 0xFFFFFF, dash: (4, 4)) \
               rect(x: 4, y: 4, w: 12, h: 8, radius: 2, fill: 0x334155) \
               bezier(x1: 0, y1: 40, cx1: 16, cy1: 0, cx2: 32, cy2: 0, x2: 48, y2: 40, \
                      stroke: 0x00FF00) \
               path(d: \"M4 4 L20 4 L20 20 Z\", fill: 0xFF0000) \
               text(\"75%\", x: 24, y: 24, align: center, size: 12) } }",
        );
        assert!(e.is_empty(), "{e:?}");
    }

    #[test]
    fn canvas_requires_width_and_height() {
        let e = canvas_errs("View V() { Canvas #[width: 48] { circle(cx: 1, cy: 1, r: 1) } }");
        assert!(
            e.iter()
                .any(|x| matches!(x, CompileError::CanvasMissingSize { .. })),
            "{e:?}"
        );
    }

    #[test]
    fn shape_command_outside_canvas_is_a_precise_error() {
        let e = errs("View V() { arc(cx: 24, cy: 24, r: 20) }");
        assert!(
            matches!(&e[0], CompileError::ShapeOutsideCanvas { name, .. } if name == "arc"),
            "{e:?}"
        );
    }

    #[test]
    fn non_shape_children_inside_canvas_are_rejected() {
        // An intrinsic view child is not a shape command…
        let e = canvas_errs("View V() { Canvas #[width: 10, height: 10] { Text(\"no\") } }");
        assert!(
            e.iter().any(
                |x| matches!(x, CompileError::UnknownShapeCommand { name, .. } if name == "Text")
            ),
            "{e:?}"
        );
        // …and neither is control flow.
        let e = canvas_errs(
            "View V() { Canvas #[width: 10, height: 10] { when x { circle(cx: 1, cy: 1, r: 1) } } }",
        );
        assert!(
            e.iter().any(
                |x| matches!(x, CompileError::UnknownShapeCommand { name, .. } if name == "when")
            ),
            "{e:?}"
        );
    }

    #[test]
    fn unknown_shape_param_gets_a_levenshtein_hint() {
        let e = canvas_errs(
            "View V() { Canvas #[width: 10, height: 10] { \
               arc(cx: 1, cy: 1, r: 5, stroke_widht: 2) } }",
        );
        assert!(
            e.iter().any(|x| matches!(
                x,
                CompileError::UnknownShapeParam { name, hint: Some(h), .. }
                    if name == "stroke_widht" && h == "stroke_width"
            )),
            "{e:?}"
        );
    }

    #[test]
    fn missing_required_geometry_is_reported() {
        let e = canvas_errs("View V() { Canvas #[width: 10, height: 10] { arc(cx: 1, cy: 1) } }");
        assert!(
            e.iter()
                .any(|x| matches!(x, CompileError::MissingShapeParam { name, .. } if name == "r")),
            "{e:?}"
        );
    }

    #[test]
    fn stroking_a_path_is_rejected_in_v1() {
        let e = canvas_errs(
            "View V() { Canvas #[width: 10, height: 10] { \
               path(d: \"M0 0 L5 5\", stroke: 0xFF0000) } }",
        );
        assert!(
            e.iter()
                .any(|x| matches!(x, CompileError::PathStrokeUnsupported { .. })),
            "{e:?}"
        );
    }

    #[test]
    fn bezier_accepts_the_terse_positional_form() {
        let e = canvas_errs(
            "View V() { Canvas #[width: 10, height: 10] { \
               bezier(0, 40, 16, 0, 32, 0, 48, 40, stroke: 0xFFFFFF) } }",
        );
        assert!(e.is_empty(), "{e:?}");
    }

    #[test]
    fn bad_cap_token_is_flagged_with_a_hint() {
        let e = canvas_errs(
            "View V() { Canvas #[width: 10, height: 10] { \
               circle(cx: 1, cy: 1, r: 1, stroke: 0xFFFFFF, cap: rounded) } }",
        );
        assert!(
            e.iter()
                .any(|x| matches!(x, CompileError::AttributeTypeMismatch { .. })),
            "{e:?}"
        );
    }
}
