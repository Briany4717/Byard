//! The 11 Phase-2 intrinsics and the RFC-0005 §5 attribute contract.
//!
//! A closed table maps each reserved intrinsic name to its content arity,
//! accepted property/event vocabulary, focusability, and children policy.
//! [`validate_element`] applies the eight §5 rules in order, each producing a
//! precise span-anchored [`CompileError`] — no failure is ever silent (D4,
//! INV-4). Interactive elements register a hit rect inflated to a 44×44 minimum
//! (RFC-0003 E8), computed by [`inflate_hit_rect`].

use std::collections::{HashMap, HashSet};

use crate::diagnostics::CompileError;
use crate::parser::ast::{AttrKind, ElementNode, Expr};
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
        _ => return None,
    })
}

/// Validates `el` against the §5 contract, returning every diagnostic it
/// produces (possibly several). `known_views` are the user `ViewDecl` names in
/// scope, so a non-intrinsic element that resolves to a view is not an error.
#[must_use]
pub fn validate_element(el: &ElementNode, known_views: &[&str]) -> Vec<CompileError> {
    let mut errs = Vec::new();
    let name = el.name.as_str();

    // Rule 1 — name resolution.
    let Some(info) = lookup(name) else {
        if !known_views.contains(&name) {
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
    for attr in &el.attrs {
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
            // RFC-0010: `value with anim.*(…)` — validate the curve and reject
            // an animation on a layout property (it can't animate on the GPU),
            // then fall through to type-check the target `value` itself.
            if let Expr::Animated {
                value: target,
                anim,
                span,
            } = value
            {
                if is_layout_prop(an) {
                    errs.push(CompileError::LayoutPropNotAnimatable {
                        span: *span,
                        prop: an.to_string(),
                    });
                } else if let Err(err) = crate::interp::anim::resolve_curve(anim) {
                    errs.push(err);
                } else if let Some(err) = check_value_type(ty, target) {
                    errs.push(err);
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
        validate_element(&first_element(src), &[])
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
        assert!(
            validate_element(&first_element("View V() { Card #[gap: 8] {} }"), &["Card"])
                .is_empty()
        );
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
    fn color_parsing() {
        let green = color_to_rgba(0x00_FF_00, false);
        assert!(green[0] < 0.01 && green[1] > 0.99 && green[2] < 0.01 && green[3] > 0.99);
        let c = color_to_rgba(0x80_00_00_00, true);
        assert!((c[3] - 0.5019).abs() < 0.01, "alpha-first 0x80… ≈ 0.5");
    }
}
