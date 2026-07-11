//! The typed, fully-owned AST (RFC-0002 §"Data structures"; RFC-0003 attrs/`Fn`).
//!
//! Every node owns all of its data — no borrows into the source text (INV-3) —
//! so hot-reload can re-parse and structurally diff a new tree against the
//! running one without lifetime entanglement, and a `CompiledView` carrying
//! this AST is `Send` (INV-6) for the file-watcher → logic-thread channel.
//!
//! The AST is **immutable after parse**: reactivity/`is_reactive` metadata lives
//! in side-tables (RFC-0002 D3, RFC-0004 §10), never on these nodes.

use crate::diagnostics::Span;
use crate::symbol::Symbol;

/// A type annotation (`type := IDENT ("<" type ("," type)* ">")?`), extended
/// with the function type `Fn(...)` for callback props (RFC-0003 E2).
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    /// A named (optionally generic) type, e.g. `Str`, `Int`, `List<Str>`.
    Named {
        /// The type's name.
        name: Symbol,
        /// Generic arguments, if any (`List<Str>` ⇒ `[Str]`).
        args: Vec<Type>,
        /// Source span.
        span: Span,
    },
    /// A function type `Fn(P0, P1, ...) -> R` (RFC-0003 E2). `ret` is `None`
    /// for a callback with no declared return.
    Function {
        /// Parameter types.
        params: Vec<Type>,
        /// Optional return type.
        ret: Option<Box<Type>>,
        /// Source span.
        span: Span,
    },
}

impl Type {
    /// The source span of this type.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Named { span, .. } | Self::Function { span, .. } => *span,
        }
    }
}

/// A `View`/`fn` parameter (`param := IDENT (":" type)?`). The annotation is
/// optional in the AST; D9's "annotation required here" rule is enforced by the
/// checker (M5), not the parser.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// Parameter name.
    pub name: Symbol,
    /// Declared type, if written.
    pub ty: Option<Type>,
    /// Default value expression (`= expr`), if written (RFC-0007 D-B). A
    /// defaulted parameter omitted at a user-view call site evaluates this in
    /// the callee scope.
    pub default: Option<Expr>,
    /// Source span.
    pub span: Span,
}

/// A top-of-file package import (RFC-0008 Pillar A/B, decisions D-F/D-G).
///
/// Three surface forms, all resolved against the manifest-declared dependency
/// set (never a path string — the two-layer rule, RFC-0001 §1):
///
/// - `use material` — qualified access as `material.Card`;
/// - `use material as m` — qualified access under an explicit alias, `m.Card`;
/// - `use material.{Card, Chip}` — selective bare imports (`Card`, `Chip`),
///   legal only while unambiguous (a collision is a `NameCollision` demanding
///   an alias, D-G).
///
/// `alias` and `symbols` are grammatically exclusive: the selective form has
/// no `as` clause.
#[derive(Clone, Debug, PartialEq)]
pub struct UseDecl {
    /// The imported package name, as declared in `[dependencies]`.
    pub package: Symbol,
    /// The explicit alias (`as m`), if written.
    pub alias: Option<Symbol>,
    /// Selective bare imports (`.{A, B}`), each with its own span for precise
    /// per-symbol diagnostics. `None` for the whole-package forms.
    pub symbols: Option<Vec<(Symbol, Span)>>,
    /// Source span of the whole declaration.
    pub span: Span,
}

/// A whole `.byd` file is a list of [`ViewDecl`]s (D11: multiple `View`s per
/// file are allowed).
#[derive(Clone, Debug, PartialEq)]
pub struct ViewDecl {
    /// View name.
    pub name: Symbol,
    /// Declared parameters.
    pub params: Vec<Param>,
    /// The view body — declarations, elements, control flow, a style block.
    pub body: Vec<Member>,
    /// Source span.
    pub span: Span,
}

/// A member of a `View` body. Replaces the prior draft's flat `Stmt`: a View
/// body holds declarations, elements, control flow, and a style block.
#[derive(Clone, Debug, PartialEq)]
pub enum Member {
    /// `var x = init` — a reactive source (lowers to `Signal::new_in`).
    Var {
        /// Binding name.
        name: Symbol,
        /// Declared type, if written (else inferred from `init`; D9).
        ty: Option<Type>,
        /// Initializer expression.
        init: Expr,
        /// Source span.
        span: Span,
    },
    /// `let y = expr` — a computed/constant binding (lowers to a memo).
    Let {
        /// Binding name.
        name: Symbol,
        /// Declared type, if written.
        ty: Option<Type>,
        /// Initializer expression.
        init: Expr,
        /// Source span.
        span: Span,
    },
    /// `fn f(params) -> ret => body` — a computed helper (memo).
    Fn {
        /// Function name.
        name: Symbol,
        /// Parameters.
        params: Vec<Param>,
        /// Declared return type, if written.
        ret: Option<Type>,
        /// Body expression.
        body: Expr,
        /// Source span.
        span: Span,
    },
    /// `inject T as name` — ambient lookup at the controller boundary.
    Inject {
        /// The injected type.
        ty: Type,
        /// Local binding name.
        name: Symbol,
        /// Source span.
        span: Span,
    },
    /// An intrinsic or user-`View` element.
    Element(ElementNode),
    /// `for item in iter { ... }` — structural reactivity.
    For {
        /// Loop variable.
        var: Symbol,
        /// Iterable expression.
        iter: Expr,
        /// Loop body.
        body: Vec<Member>,
        /// Source span.
        span: Span,
    },
    /// `when cond { ... } else { ... }` — structural reactivity.
    When {
        /// Condition.
        cond: Expr,
        /// Then-branch members.
        then: Vec<Member>,
        /// Optional else-branch members.
        els: Option<Vec<Member>>,
        /// Source span.
        span: Span,
    },
    /// `style { .class #[...] ... }` — scoped style rules (static; D5).
    Style {
        /// The style rules.
        rules: Vec<StyleRule>,
        /// Source span.
        span: Span,
    },
    /// A bare expression statement (e.g. a call).
    Expr(Expr),
}

/// An element: `IDENT ("(" content ")")? attr_block? ("{" children "}" | "=>" action)`.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementNode {
    /// Intrinsic (`Column`, `Text`, …) or user-`View` name.
    pub name: Symbol,
    /// Positional `(...)` content (a `Text`'s string, a `Button`'s label).
    pub content: Vec<Arg>,
    /// `#[...]` properties / config / events.
    pub attrs: Vec<Attr>,
    /// The `=> action` shorthand (the hoisted primary `tap` event), if present.
    pub action: Option<Expr>,
    /// The `{ ... }` children block.
    pub children: Vec<Member>,
    /// Source span.
    pub span: Span,
}

/// One `#[...]` attribute: either a property (`name: expr`) or an engine event
/// (`name(payload)? => expr`) — RFC-0003 D4-bis. The kind is decided
/// syntactically by the separator; a mismatch against the intrinsic's contract
/// is a *checker* error (M10), not a parse error.
#[derive(Clone, Debug, PartialEq)]
pub struct Attr {
    /// Attribute name.
    pub name: Symbol,
    /// The sub-property axis, if written as `name.axis: value` (RFC-0011
    /// §"Dual surface", e.g. `translate.y: 2`). `None` for the ordinary
    /// `name: value` / `name(payload)? => action` forms.
    pub axis: Option<Symbol>,
    /// Whether this is a property binding or an engine event.
    pub kind: AttrKind,
    /// Source span.
    pub span: Span,
}

/// The two attribute flavors distinguished by the `:` vs `=>` separator.
#[derive(Clone, Debug, PartialEq)]
pub enum AttrKind {
    /// `name: value` — binds a value (including reactive props and callback
    /// props, since a function *value* is still a value).
    Prop {
        /// The bound value expression.
        value: Expr,
    },
    /// `name(payload)? => action` — maps an engine event to an action; the
    /// optional `payload` binds the event record (e.g. `pointer_move(e)`).
    Event {
        /// The optional payload binding.
        payload: Option<Symbol>,
        /// The action expression.
        action: Expr,
    },
    /// `..expr` — a style spread (RFC-0016): splice the attributes of the
    /// [`StyleValue`](Expr::StyleValue) `expr` resolves to into this list, in
    /// written order, before any inline attributes override them. The owning
    /// [`Attr`]'s `name` is empty for a spread.
    Spread {
        /// The style expression being spread (an identifier bound to a style,
        /// or an inline `style { … }`).
        value: Expr,
    },
}

/// One of the four engine-owned interaction states an `on <state> { }` block
/// (RFC-0016) can target. The engine reports these via `StyleState` (RFC-0012);
/// when several are active at once the highest-priority block wins, in the fixed
/// order `Disabled > Pressed > Focused > Hover` (RFC-0016 §"Resolution order").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StyleStateKind {
    /// The pointer is over the element.
    Hover,
    /// The element is being pressed (pointer down inside it).
    Pressed,
    /// The element holds keyboard focus.
    Focused,
    /// The element is disabled (also gates event dispatch, RFC-0012 §S5).
    Disabled,
}

impl StyleStateKind {
    /// Parses a state name; `None` for anything not one of the four states (an
    /// unknown name is a compile error, [`CompileError::UnknownStyleState`]).
    ///
    /// [`CompileError::UnknownStyleState`]: crate::diagnostics::CompileError::UnknownStyleState
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "hover" => Some(Self::Hover),
            "pressed" => Some(Self::Pressed),
            "focused" => Some(Self::Focused),
            "disabled" => Some(Self::Disabled),
            _ => None,
        }
    }

    /// The canonical spelling of this state, for diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hover => "hover",
            Self::Pressed => "pressed",
            Self::Focused => "focused",
            Self::Disabled => "disabled",
        }
    }

    /// The four state names, for `closest_match` suggestions.
    pub const NAMES: [&'static str; 4] = ["hover", "pressed", "focused", "disabled"];
}

/// An `on <state> { attr* }` block inside a `style { }` value (RFC-0016): the
/// attributes that apply only while the element is in the engine-owned `state`.
/// Resolved at render time against the live `StyleState` mask — the *only*
/// sanctioned dynamism in an otherwise-static style (D8).
#[derive(Clone, Debug, PartialEq)]
pub struct StateBlock {
    /// Which interaction state activates this block.
    pub state: StyleStateKind,
    /// The attributes overlaid onto the base while `state` is active.
    pub attrs: Vec<Attr>,
    /// Source span.
    pub span: Span,
}

/// A style rule: `. IDENT #[ attrs ]` (D5).
#[derive(Clone, Debug, PartialEq)]
pub struct StyleRule {
    /// The class name (after the `.`).
    pub class: Symbol,
    /// The class's attributes.
    pub attrs: Vec<Attr>,
    /// Source span.
    pub span: Span,
}

/// A call / content argument: `(IDENT ":")? expr`.
#[derive(Clone, Debug, PartialEq)]
pub struct Arg {
    /// The optional `name:` label.
    pub name: Option<Symbol>,
    /// The argument value.
    pub value: Expr,
}

/// Assignment operators (`= += -=`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    /// `=`
    Assign,
    /// `+=`
    Add,
    /// `-=`
    Sub,
}

/// Postfix mutation operators (`++ --`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostfixOp {
    /// `++`
    Inc,
    /// `--`
    Dec,
}

/// A piece of an interpolated string literal.
#[derive(Clone, Debug, PartialEq)]
pub enum StrPart {
    /// A literal text run.
    Text(String),
    /// An interpolated `{ expr }`.
    Interp(Box<Expr>),
}

/// An expression. Every variant carries its own [`Span`].
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// An integer literal (`i64`; D9).
    IntLit(i64, Span),
    /// A float literal (`f64`; D9).
    FloatLit(f64, Span),
    /// An angle literal (`360deg`/`1.5rad`, RFC-0011 T1), already
    /// canonicalized to radians by the lexer.
    AngleLit(f64, Span),
    /// A string literal, possibly interpolated.
    StrLit(Vec<StrPart>, Span),
    /// An identifier reference.
    Ident(Symbol, Span),
    /// An array literal `[a, b, ...]`.
    Array(Vec<Expr>, Span),
    /// A parenthesized tuple `(a, b, ...)` — used for `Len` pairs/quads such as
    /// `p: (8, 16)` (RFC-0005 §1). A single parenthesized expression is *not* a
    /// tuple; it parses to the inner expression directly.
    Tuple(Vec<Arg>, Span),
    /// A leading-dot class reference, e.g. the `.title` in `#[style: .title]`
    /// (RFC-0002 §"Grammar" `style_rule`; resolved against the View's style map
    /// in M11).
    ClassRef(Symbol, Span),
    /// Member access `base.field`.
    Member {
        /// The receiver expression.
        base: Box<Expr>,
        /// The field name.
        field: Symbol,
        /// Source span.
        span: Span,
    },
    /// A call `callee(args)`.
    Call {
        /// The callee.
        callee: Box<Expr>,
        /// The arguments.
        args: Vec<Arg>,
        /// Source span.
        span: Span,
    },
    /// A lambda `|p| e` or `(p) => e`.
    Lambda {
        /// Parameter names (types inferred from the use site; E2).
        params: Vec<Symbol>,
        /// The body expression.
        body: Box<Expr>,
        /// Source span.
        span: Span,
    },
    /// A brace-delimited action block `{ stmt* }` (RFC-0019): the body of a
    /// callback-prop literal (`on_tap: { count++ }`). Holds zero or more action
    /// statements evaluated in order; the value is the last statement's (or
    /// [`Value::Unit`] for the empty no-op default `{}`). Distinct from a
    /// `style { … }` value and from a View/`when`/`for` body — those consume
    /// their braces structurally and never reach expression position.
    ///
    /// [`Value::Unit`]: crate::interp::env::Value::Unit
    Block(Vec<Expr>, Span),
    /// An assignment `target op value` (`= += -=`).
    Assign {
        /// The l-value target.
        target: Box<Expr>,
        /// The operator.
        op: AssignOp,
        /// The new value.
        value: Box<Expr>,
        /// Source span.
        span: Span,
    },
    /// A postfix mutation `target++` / `target--`.
    Postfix {
        /// The l-value target.
        target: Box<Expr>,
        /// The operator.
        op: PostfixOp,
        /// Source span.
        span: Span,
    },
    /// A ternary `cond ? then : els`.
    Ternary {
        /// The condition.
        cond: Box<Expr>,
        /// The then-branch.
        then: Box<Expr>,
        /// The else-branch.
        els: Box<Expr>,
        /// Source span.
        span: Span,
    },
    /// An animated attribute value `value with anim.*(…)` (RFC-0010): `value`
    /// is the (usually ternary) target and `anim` is the `anim.*` curve call,
    /// resolved to a typed `Curve` at lowering. `with` binds below the ternary,
    /// so `a ? b : c with k` parses as `(a ? b : c) with k`.
    Animated {
        /// The target value expression (scalar/ternary).
        value: Box<Expr>,
        /// The `anim.*` curve call, resolved to a typed `Curve` at lower time.
        anim: Box<Expr>,
        /// Source span.
        span: Span,
    },
    /// A first-class style value `style { name: value, … }` (RFC-0016): an
    /// ordered bundle of attributes, `let`-bound and applied to an element with
    /// the `..` spread. Static and composable; no cascade.
    StyleValue {
        /// The style's base attributes, in written order.
        attrs: Vec<Attr>,
        /// `on <state> { … }` interaction-state blocks (RFC-0016), applied at
        /// render time over the base when their state is active.
        states: Vec<StateBlock>,
        /// Source span.
        span: Span,
    },
    /// `left merge right` (RFC-0016 M3): composes two style values into one; on
    /// a conflicting attribute the right operand wins. Both operands resolve to
    /// styles at lower time.
    Merge {
        /// The base style.
        left: Box<Expr>,
        /// The overriding style.
        right: Box<Expr>,
        /// Source span.
        span: Span,
    },
    /// A parse-error placeholder, so recovery can continue and collect more
    /// diagnostics (RFC-0002 §"Parser").
    Error(Span),
}

impl Expr {
    /// The source span of this expression.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::IntLit(_, span)
            | Self::FloatLit(_, span)
            | Self::AngleLit(_, span)
            | Self::StrLit(_, span)
            | Self::Ident(_, span)
            | Self::Array(_, span)
            | Self::Tuple(_, span)
            | Self::ClassRef(_, span)
            | Self::Member { span, .. }
            | Self::Call { span, .. }
            | Self::Lambda { span, .. }
            | Self::Block(_, span)
            | Self::Assign { span, .. }
            | Self::Postfix { span, .. }
            | Self::Ternary { span, .. }
            | Self::Animated { span, .. }
            | Self::StyleValue { span, .. }
            | Self::Merge { span, .. }
            | Self::Error(span) => *span,
        }
    }
}

// INV-6: the AST must be `Send` so a `CompiledView` built from it can cross the
// file-watcher → logic-thread channel. If any node grew a non-`Send` field
// (e.g. an `Rc`), this would stop compiling.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<ViewDecl>();
    assert_send::<Member>();
    assert_send::<Expr>();
    assert_send::<Type>();
    assert_send::<UseDecl>();
};

#[cfg(test)]
mod tests {
    use super::*;

    fn sp() -> Span {
        Span::new(0, 1)
    }

    /// Build a small tree by hand — exercises every owning node and proves the
    /// AST can represent `Button("+") #[bg: 1] => count++`.
    #[test]
    fn hand_built_tree_round_trips() {
        let count = Symbol::intern("count");
        let element = ElementNode {
            name: Symbol::intern("Button"),
            content: vec![Arg {
                name: None,
                value: Expr::StrLit(vec![StrPart::Text("+".to_string())], sp()),
            }],
            attrs: vec![Attr {
                name: Symbol::intern("bg"),
                axis: None,
                kind: AttrKind::Prop {
                    value: Expr::IntLit(1, sp()),
                },
                span: sp(),
            }],
            action: Some(Expr::Postfix {
                target: Box::new(Expr::Ident(count.clone(), sp())),
                op: PostfixOp::Inc,
                span: sp(),
            }),
            children: Vec::new(),
            span: sp(),
        };
        let view = ViewDecl {
            name: Symbol::intern("Counter"),
            params: Vec::new(),
            body: vec![
                Member::Var {
                    name: count,
                    ty: None,
                    init: Expr::IntLit(0, sp()),
                    span: sp(),
                },
                Member::Element(element),
            ],
            span: sp(),
        };

        assert_eq!(view.body.len(), 2);
        let Member::Element(el) = &view.body[1] else {
            panic!("expected element");
        };
        assert!(matches!(el.action, Some(Expr::Postfix { .. })));
        assert_eq!(el.attrs[0].name, Symbol::intern("bg"));
    }

    #[test]
    fn event_attr_carries_optional_payload() {
        let attr = Attr {
            name: Symbol::intern("pointer_move"),
            axis: None,
            kind: AttrKind::Event {
                payload: Some(Symbol::intern("e")),
                action: Expr::Error(sp()),
            },
            span: sp(),
        };
        let AttrKind::Event { payload, .. } = &attr.kind else {
            panic!("expected event");
        };
        assert_eq!(*payload, Some(Symbol::intern("e")));
    }

    #[test]
    fn function_type_is_representable() {
        let ty = Type::Function {
            params: vec![Type::Named {
                name: Symbol::intern("ChangeEvent"),
                args: vec![Type::Named {
                    name: Symbol::intern("Str"),
                    args: Vec::new(),
                    span: sp(),
                }],
                span: sp(),
            }],
            ret: None,
            span: sp(),
        };
        assert_eq!(ty.span(), sp());
    }

    /// Runtime echo of the compile-time `assert_send` above (keeps the
    /// intent visible in the test report).
    #[test]
    fn ast_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ViewDecl>();
    }
}
