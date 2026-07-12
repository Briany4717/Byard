//! Golden parses and targeted parser unit tests (RFC-0002 §"Grammar").

use super::ast::*;
use super::parse;
use crate::symbol::Symbol;

fn sym(s: &str) -> Symbol {
    Symbol::intern(s)
}

/// Parses `src`, asserting it produced exactly one view and no diagnostics.
fn one_view(src: &str) -> ViewDecl {
    let parsed = parse(src);
    assert!(
        parsed.errors.is_empty(),
        "unexpected diagnostics: {:#?}",
        parsed.errors
    );
    assert_eq!(parsed.views.len(), 1, "expected exactly one view");
    parsed.views.into_iter().next().unwrap()
}

fn as_element(member: &Member) -> &ElementNode {
    match member {
        Member::Element(e) => e,
        other => panic!("expected element, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Golden parses (the four canonical examples).
// ---------------------------------------------------------------------------

#[test]
fn golden_counter() {
    // RFC-0003 §"Mutating the view".
    let src = r#"
View Counter() {
    var count = 0
    Column #[gap: 8, p: 16] {
        Text("Count: {count}")
        Button("+") => count++
        Button("−") #[tap => count--]
        Button("Reset") #[tap => count = 0]
    }
}
"#;
    let view = one_view(src);
    assert_eq!(view.name, sym("Counter"));
    assert_eq!(view.body.len(), 2);
    assert!(matches!(&view.body[0], Member::Var { name, .. } if *name == sym("count")));

    let column = as_element(&view.body[1]);
    assert_eq!(column.name, sym("Column"));
    assert_eq!(column.attrs.len(), 2);
    assert_eq!(column.children.len(), 4);

    // Text("Count: {count}") → [Text("Count: "), Interp(count)]
    let text = as_element(&column.children[0]);
    let Expr::StrLit(parts, _) = &text.content[0].value else {
        panic!("expected string literal content");
    };
    assert_eq!(parts.len(), 2);
    assert!(matches!(&parts[0], StrPart::Text(t) if t == "Count: "));
    assert!(
        matches!(&parts[1], StrPart::Interp(e) if matches!(**e, Expr::Ident(ref s, _) if *s == sym("count")))
    );

    // Button("+") => count++   (action shorthand)
    let plus = as_element(&column.children[1]);
    assert!(matches!(
        &plus.action,
        Some(Expr::Postfix {
            op: PostfixOp::Inc,
            ..
        })
    ));

    // Button("−") #[tap => count--]   (explicit event)
    let minus = as_element(&column.children[2]);
    assert_eq!(minus.attrs.len(), 1);
    assert_eq!(minus.attrs[0].name, sym("tap"));
    assert!(matches!(
        &minus.attrs[0].kind,
        AttrKind::Event {
            payload: None,
            action: Expr::Postfix {
                op: PostfixOp::Dec,
                ..
            }
        }
    ));

    // Button("Reset") #[tap => count = 0]   (assignment action)
    let reset = as_element(&column.children[3]);
    assert!(matches!(
        &reset.attrs[0].kind,
        AttrKind::Event {
            action: Expr::Assign {
                op: AssignOp::Assign,
                ..
            },
            ..
        }
    ));
}

#[test]
fn golden_user_card() {
    // Erratum canonical / RFC-0002 §"at a glance".
    let src = r#"
View UserCard() {
    var clicks = 0
    inject AppEnvironment as env

    Column #[gap: 12, bg: env.theme.surface, radius: 16, p: 20] {
        Text("Clicks: {clicks}") #[typo: m3.titleLarge]
        Button("Action") => clicks++
    }
}
"#;
    let view = one_view(src);
    assert_eq!(view.name, sym("UserCard"));
    assert_eq!(view.body.len(), 3);
    assert!(matches!(&view.body[0], Member::Var { name, .. } if *name == sym("clicks")));
    assert!(matches!(
        &view.body[1],
        Member::Inject { ty: Type::Named { name, .. }, name: bind, .. }
            if *name == sym("AppEnvironment") && *bind == sym("env")
    ));

    let column = as_element(&view.body[2]);
    assert_eq!(column.attrs.len(), 4);
    // bg: env.theme.surface  → nested member access
    let bg = &column.attrs[1];
    assert_eq!(bg.name, sym("bg"));
    assert!(matches!(
        &bg.kind,
        AttrKind::Prop {
            value: Expr::Member { .. }
        }
    ));

    let button = as_element(&column.children[1]);
    assert!(matches!(
        &button.action,
        Some(Expr::Postfix {
            op: PostfixOp::Inc,
            ..
        })
    ));
}

#[test]
fn golden_search() {
    // RFC-0002 §"State, derived values" — exercises typed var, let/fn memos,
    // lambdas, for/when, and a scoped style block.
    let src = r#"
View Search() {
    var query = ""
    var items: List<Str> = ["apple", "pear", "plum"]

    let filtered = items.filter(|x| x.starts_with(query))
    fn greeting() -> Str => filtered.is_empty() ? "No matches" : "Results"

    Column #[gap: 8, p: 16] {
        Text(greeting()) #[style: .title]
        TextField #[bind: query, placeholder: "Filter…"]

        for item in filtered {
            Text(item) #[style: .row]
        }
        when filtered.is_empty() {
            Text("Nothing here") #[style: .muted]
        }
    }

    style {
        .title #[size: 20, weight: bold]
        .row   #[p: (4, 8)]
        .muted #[color: 0x888888]
    }
}
"#;
    let view = one_view(src);
    assert_eq!(view.name, sym("Search"));

    // var items: List<Str> = [...]
    let Member::Var {
        ty: Some(Type::Named { name, args, .. }),
        init: Expr::Array(elems, _),
        ..
    } = &view.body[1]
    else {
        panic!("expected typed array var, got {:?}", view.body[1]);
    };
    assert_eq!(*name, sym("List"));
    assert_eq!(args.len(), 1);
    assert_eq!(elems.len(), 3);

    // let filtered = items.filter(|x| ...)
    assert!(matches!(
        &view.body[2],
        Member::Let { init: Expr::Call { .. }, name, .. } if *name == sym("filtered")
    ));

    // fn greeting() -> Str => ... ? ... : ...
    let Member::Fn {
        ret: Some(Type::Named { name: ret, .. }),
        body,
        ..
    } = &view.body[3]
    else {
        panic!("expected fn with return type");
    };
    assert_eq!(*ret, sym("Str"));
    assert!(matches!(body, Expr::Ternary { .. }));

    // The Column has Text, TextField, a `for`, and a `when`.
    let column = as_element(&view.body[4]);
    assert_eq!(column.children.len(), 4);
    assert!(matches!(&column.children[2], Member::For { var, .. } if *var == sym("item")));
    assert!(matches!(
        &column.children[3],
        Member::When { els: None, .. }
    ));

    // #[style: .title] resolves to a class reference value.
    let text = as_element(&column.children[0]);
    assert!(matches!(
        &text.attrs[0].kind,
        AttrKind::Prop { value: Expr::ClassRef(c, _) } if *c == sym("title")
    ));

    // The scoped style block has three rules.
    let Member::Style { rules, .. } = &view.body[5] else {
        panic!("expected style block");
    };
    assert_eq!(rules.len(), 3);
    assert_eq!(rules[0].class, sym("title"));
}

#[test]
fn golden_profile_card() {
    // RFC-0005 §"Guide-level" — params with a type, hex colors, a Len pair,
    // a nested interpolated string, and a `=> follow()` action.
    let src = r#"
View ProfileCard(name: Str) {
    var liked = false

    Column #[gap: 12, p: 16, bg: 0x1E1E1E, radius: 16, width: 280] {
        Row #[gap: 8, align: center] {
            Image("avatar.png") #[width: 40, height: 40, radius: 20, fit: cover]
            Text(name) #[typo: titleMedium]
            Spacer #[grow: 1]
            Toggle #[bind: liked]
        }
        Text("{name} {liked ? \"♥ liked\" : \"\"}") #[color: 0xAAAAAA, lines: 1]
        Button("Follow") #[bg: 0x3B82F6, radius: 8, p: (8, 16)] => follow()
    }
}
"#;
    let view = one_view(src);
    assert_eq!(view.name, sym("ProfileCard"));
    assert_eq!(view.params.len(), 1);
    assert!(matches!(
        &view.params[0],
        Param { name, ty: Some(Type::Named { name: ty, .. }), .. }
            if *name == sym("name") && *ty == sym("Str")
    ));

    let column = as_element(&view.body[1]);
    // bg: 0x1E1E1E lexed as a hex int.
    assert!(matches!(
        &column.attrs[2].kind,
        AttrKind::Prop { value: Expr::IntLit(v, _) } if *v == 0x001E_1E1E
    ));

    let row = as_element(&column.children[0]);
    assert_eq!(row.children.len(), 4);
    // fit: cover → enum token as an identifier.
    let image = as_element(&row.children[0]);
    assert!(matches!(
        &image.attrs[3].kind,
        AttrKind::Prop { value: Expr::Ident(c, _) } if *c == sym("cover")
    ));

    // The interpolated string with nested escaped strings inside the ternary.
    let text = as_element(&column.children[1]);
    let Expr::StrLit(parts, _) = &text.content[0].value else {
        panic!("expected interpolated string");
    };
    // [Interp(name), Text(" "), Interp(ternary)]
    assert!(matches!(&parts[0], StrPart::Interp(_)));
    assert!(
        parts
            .iter()
            .any(|p| matches!(p, StrPart::Interp(e) if matches!(**e, Expr::Ternary { .. })))
    );

    // Button("Follow") #[... p: (8, 16)] => follow()
    let button = as_element(&column.children[2]);
    assert!(matches!(
        &button.attrs[2].kind,
        AttrKind::Prop { value: Expr::Tuple(items, _) } if items.len() == 2
    ));
    assert!(matches!(&button.action, Some(Expr::Call { .. })));
}

// ---------------------------------------------------------------------------
// Targeted unit tests.
// ---------------------------------------------------------------------------

#[test]
fn prop_vs_event_attributes() {
    let view = one_view("View V() { Box #[gap: 12, tap => x++, move(e) => y = e.pos] }");
    let el = as_element(&view.body[0]);
    assert_eq!(el.attrs.len(), 3);
    assert!(matches!(&el.attrs[0].kind, AttrKind::Prop { .. }));
    assert!(matches!(
        &el.attrs[1].kind,
        AttrKind::Event { payload: None, .. }
    ));
    assert!(matches!(
        &el.attrs[2].kind,
        AttrKind::Event { payload: Some(p), .. } if *p == sym("e")
    ));
}

#[test]
fn sub_property_axis_parses_and_carries_the_base_name_plus_axis() {
    // RFC-0011 `translate.y: 2` — one axis of a two-axis prop, set inline
    // without a tuple.
    let view = one_view("View V() { Box #[translate.y: 2, gap: 12] }");
    let el = as_element(&view.body[0]);
    assert_eq!(el.attrs.len(), 2);
    assert_eq!(el.attrs[0].name, sym("translate"));
    assert_eq!(el.attrs[0].axis, Some(sym("y")));
    assert!(matches!(
        &el.attrs[0].kind,
        AttrKind::Prop {
            value: Expr::IntLit(2, _)
        }
    ));

    // An ordinary attribute (no dot) always has `axis: None`.
    assert_eq!(el.attrs[1].name, sym("gap"));
    assert_eq!(el.attrs[1].axis, None);
}

#[test]
fn angle_literal_parses_as_an_angle_lit_expr() {
    let view = one_view("View V() { Box #[rotate: 90deg] }");
    let el = as_element(&view.body[0]);
    assert!(matches!(
        &el.attrs[0].kind,
        AttrKind::Prop { value: Expr::AngleLit(rad, _) }
            if (*rad - std::f64::consts::FRAC_PI_2).abs() < 1e-9
    ));
}

#[test]
fn negative_numeric_literals_parse() {
    // Byld has no binary arithmetic; a leading `-` is the sign of a numeric
    // literal. `translate: (-8, 4)` must parse, not raise a parse error.
    let view = one_view("View V() { Box #[translate: (-8, 4)] }");
    let el = as_element(&view.body[0]);
    let AttrKind::Prop {
        value: Expr::Tuple(args, _),
    } = &el.attrs[0].kind
    else {
        panic!("expected a tuple value");
    };
    assert!(matches!(&args[0].value, Expr::IntLit(-8, _)));
    assert!(matches!(&args[1].value, Expr::IntLit(4, _)));

    // Negative float and negative angle literals too.
    let view = one_view("View V() { Box #[scale: -1.5, rotate: -90deg] }");
    let el = as_element(&view.body[0]);
    assert!(matches!(
        &el.attrs[0].kind,
        AttrKind::Prop { value: Expr::FloatLit(f, _) } if (*f + 1.5).abs() < 1e-9
    ));
    assert!(matches!(
        &el.attrs[1].kind,
        AttrKind::Prop { value: Expr::AngleLit(rad, _) }
            if (*rad + std::f64::consts::FRAC_PI_2).abs() < 1e-9
    ));
}

#[test]
fn bare_minus_without_a_number_is_a_targeted_error() {
    // `-` not followed by a number must report the specific diagnostic, not a
    // silent drop or a generic "expected an expression".
    let parsed = parse("View V() { Box #[translate: (-, 4)] }");
    assert!(!parsed.errors.is_empty());
}

#[test]
fn with_animation_binds_the_whole_ternary_as_the_value() {
    // RFC-0010: `a ? b : c with k` groups as `(a ? b : c) with k` — the whole
    // conditional is the animated value, not just the else-branch.
    let view = one_view("View V() { Box #[radius: pressed ? 3 : 10 with anim.spring()] }");
    let el = as_element(&view.body[0]);
    let AttrKind::Prop {
        value: Expr::Animated { value, anim, .. },
    } = &el.attrs[0].kind
    else {
        panic!("expected an Animated value, got {:?}", el.attrs[0].kind);
    };
    assert!(
        matches!(value.as_ref(), Expr::Ternary { .. }),
        "the ternary must be the animated value"
    );
    assert!(
        matches!(anim.as_ref(), Expr::Call { .. }),
        "the anim side must be the `anim.spring()` call"
    );
}

#[test]
fn with_animation_optional_parens_and_named_args_parse() {
    // Bare call, named-arg call, and a `200ms` duration literal all parse.
    one_view("View V() { Box #[scale: hovered ? 1.05 : 1.0 with anim.spring()] }");
    one_view(
        "View V() { Box #[scale: hovered ? 1.05 : 1.0 with anim.spring(stiffness: 210, damping: 20)] }",
    );
    one_view("View V() { Box #[opacity: shown ? 1.0 : 0.0 with anim.linear(200ms)] }");
}

#[test]
fn style_value_and_spread_parse() {
    // RFC-0016: `let s = style { … }` binds a style value; `#[..s]` spreads it.
    let view = one_view(
        "View V() { let s = style { bg: 0x111111, radius: 4 } Box #[..s, color: 0xFFFFFF] }",
    );
    let Member::Let {
        init: Expr::StyleValue { attrs, .. },
        ..
    } = &view.body[0]
    else {
        panic!("expected `let = style {{}}`, got {:?}", view.body[0]);
    };
    assert_eq!(attrs.len(), 2, "the style holds two attributes");

    let el = as_element(&view.body[1]);
    assert!(
        matches!(&el.attrs[0].kind, AttrKind::Spread { .. }),
        "the first element attribute is a `..` spread"
    );
    assert!(
        matches!(&el.attrs[1].kind, AttrKind::Prop { .. }),
        "the inline attribute follows the spread"
    );
}

#[test]
fn function_types_parse() {
    let view = one_view("View V(onPick: Fn(ChangeEvent<Str>), test: Fn(Int) -> Bool) {}");
    let Type::Function { params, ret, .. } = view.params[0].ty.as_ref().unwrap() else {
        panic!("expected Fn type for onPick");
    };
    assert_eq!(params.len(), 1);
    assert!(ret.is_none());

    let Type::Function { params, ret, .. } = view.params[1].ty.as_ref().unwrap() else {
        panic!("expected Fn type for test");
    };
    assert_eq!(params.len(), 1);
    assert!(matches!(ret.as_deref(), Some(Type::Named { name, .. }) if *name == sym("Bool")));
}

#[test]
fn multiple_views_per_file() {
    let parsed = parse("View A() {}\nView B() {}");
    assert!(parsed.errors.is_empty());
    assert_eq!(parsed.views.len(), 2);
    assert_eq!(parsed.views[0].name, sym("A"));
    assert_eq!(parsed.views[1].name, sym("B"));
}

#[test]
fn error_recovery_collects_multiple_diagnostics() {
    // Two independent malformed bindings must each be reported, and the view is
    // still returned (single-pass multi-diagnostic recovery).
    let parsed = parse("View Bad() {\n    var = 1\n    let = 2\n}");
    assert!(
        parsed.errors.len() >= 2,
        "expected ≥2 diagnostics, got {:#?}",
        parsed.errors
    );
    assert_eq!(parsed.views.len(), 1);
    assert_eq!(parsed.views[0].body.len(), 2);
}

#[test]
fn callback_param_type_is_a_function() {
    // RFC-0019: `on_tap: Fn()` declares a callback parameter; `Fn(Str)` carries
    // its argument types.
    let view = one_view("View W(on_tap: Fn(), on_change: Fn(Str)) { Text(\"x\") }");
    assert_eq!(view.params.len(), 2);
    assert!(matches!(
        view.params[0].ty,
        Some(Type::Function { ref params, .. }) if params.is_empty()
    ));
    let Some(Type::Function { params, .. }) = &view.params[1].ty else {
        panic!("expected Fn(Str), got {:?}", view.params[1].ty);
    };
    assert_eq!(params.len(), 1);
}

#[test]
fn callback_block_parses_as_lambda_over_block() {
    // RFC-0019: a `{ … }` action block in expression position parses as a
    // parameterless lambda whose body is an `Expr::Block` of statements.
    let view = one_view("View V() { Card(on_tap: { count++ x = 0 }) }");
    let card = as_element(&view.body[0]);
    let value = &card.content[0].value;
    let Expr::Lambda { params, body, .. } = value else {
        panic!("expected a Lambda, got {value:?}");
    };
    assert!(params.is_empty(), "no-arg callback");
    let Expr::Block(stmts, _) = body.as_ref() else {
        panic!("expected a Block body, got {body:?}");
    };
    assert_eq!(stmts.len(), 2, "two statements run in order");
}

#[test]
fn callback_block_with_params_and_empty_default() {
    // A `{|text| … }` header names the callback's arguments; `{}` is the empty
    // no-op default.
    let view = one_view("View V(on_change: Fn(Str) = {}) { Field(on_change: {|text| q = text}) }");
    // The default `{}` is a Lambda over an empty Block.
    let Some(Expr::Lambda { params, body, .. }) = &view.params[0].default else {
        panic!(
            "expected a Lambda default, got {:?}",
            view.params[0].default
        );
    };
    assert!(params.is_empty());
    assert!(matches!(body.as_ref(), Expr::Block(s, _) if s.is_empty()));
    // The call-site block names its parameter.
    let field = as_element(&view.body[0]);
    let value = &field.content[0].value;
    let Expr::Lambda { params, .. } = value else {
        panic!("expected a Lambda, got {value:?}");
    };
    assert_eq!(params, &[sym("text")]);
}

#[test]
fn style_value_captures_base_attrs_and_state_blocks() {
    // RFC-0016: `style { … on <state> { … } }` collects base attributes and
    // interaction-state blocks into `Expr::StyleValue`.
    let view = one_view(
        "View V() {\n let b = style { bg: 1 on hover { bg: 2 } on pressed { scale: 0.97 } }\n}",
    );
    let Member::Let { init, .. } = &view.body[0] else {
        panic!("expected a let binding, got {:?}", view.body[0]);
    };
    let Expr::StyleValue { attrs, states, .. } = init else {
        panic!("expected a StyleValue, got {init:?}");
    };
    assert_eq!(attrs.len(), 1, "one base attribute");
    assert_eq!(states.len(), 2, "two state blocks");
    assert_eq!(states[0].state, StyleStateKind::Hover);
    assert_eq!(states[1].state, StyleStateKind::Pressed);
}

// ---------------------------------------------------------------------------
// Binary arithmetic (`+ - * /`) — the minimal surface RFC-0020's reactive
// shape parameters need (`sweep: percent * 3.6`).
// ---------------------------------------------------------------------------

/// Parses `View V() { let x = <src> }` and returns the initializer.
fn init_expr(src: &str) -> Expr {
    let view = one_view(&format!("View V() {{ let x = {src} }}"));
    let Member::Let { init, .. } = &view.body[0] else {
        panic!("expected a let binding, got {:?}", view.body[0]);
    };
    init.clone()
}

#[test]
fn multiplication_binds_tighter_than_addition() {
    // `a + b * c` groups as `a + (b * c)`.
    let e = init_expr("a + b * c");
    let Expr::Binary {
        op: BinOp::Add,
        rhs,
        ..
    } = &e
    else {
        panic!("expected top-level Add, got {e:?}");
    };
    assert!(
        matches!(rhs.as_ref(), Expr::Binary { op: BinOp::Mul, .. }),
        "rhs must be the product, got {rhs:?}"
    );
}

#[test]
fn same_precedence_is_left_associative() {
    // `a - b + c` groups as `(a - b) + c`; `a / b * c` as `(a / b) * c`.
    let e = init_expr("a - b + c");
    let Expr::Binary {
        op: BinOp::Add,
        lhs,
        ..
    } = &e
    else {
        panic!("expected top-level Add, got {e:?}");
    };
    assert!(matches!(lhs.as_ref(), Expr::Binary { op: BinOp::Sub, .. }));

    let e = init_expr("a / b * c");
    let Expr::Binary {
        op: BinOp::Mul,
        lhs,
        ..
    } = &e
    else {
        panic!("expected top-level Mul, got {e:?}");
    };
    assert!(matches!(lhs.as_ref(), Expr::Binary { op: BinOp::Div, .. }));
}

#[test]
fn arithmetic_groups_below_with_and_inside_ternary() {
    // `p * 360 with anim.spring()` animates the whole product (RFC-0010).
    let e = init_expr("p * 360 with anim.spring()");
    let Expr::Animated { value, .. } = &e else {
        panic!("expected Animated, got {e:?}");
    };
    assert!(matches!(
        value.as_ref(),
        Expr::Binary { op: BinOp::Mul, .. }
    ));

    // Arithmetic is available inside ternary branches.
    let e = init_expr("cond ? a + 1 : b * 2");
    let Expr::Ternary { then, els, .. } = &e else {
        panic!("expected Ternary, got {e:?}");
    };
    assert!(matches!(then.as_ref(), Expr::Binary { op: BinOp::Add, .. }));
    assert!(matches!(els.as_ref(), Expr::Binary { op: BinOp::Mul, .. }));
}

#[test]
fn unary_minus_still_parses_and_binary_minus_needs_a_left_operand() {
    // The numeric-sign form is untouched: `(-8, 0)` is a tuple of literals.
    let view = one_view("View V() { Box #[translate: (-8, 0)] {} }");
    let el = as_element(&view.body[0]);
    let AttrKind::Prop { value } = &el.attrs[0].kind else {
        panic!("expected a prop");
    };
    let Expr::Tuple(items, _) = value else {
        panic!("expected a tuple, got {value:?}");
    };
    assert!(matches!(items[0].value, Expr::IntLit(-8, _)));

    // With a left operand the same token is subtraction.
    let e = init_expr("a - 8");
    assert!(matches!(e, Expr::Binary { op: BinOp::Sub, .. }));
}

#[test]
fn shape_command_args_accept_arithmetic() {
    // The RFC-0020 headline: a reactive sweep expression inside a shape
    // command's argument list parses as ordinary named args.
    let view = one_view(
        "View V() { Canvas #[width: 48, height: 48] { \
           arc(cx: 24, cy: 24, r: 20, sweep: percent * 3.6, stroke: 0xFFFFFF) } }",
    );
    let canvas = as_element(&view.body[0]);
    let arc = as_element(&canvas.children[0]);
    let sweep = arc
        .content
        .iter()
        .find(|a| a.name.as_ref().is_some_and(|n| n.as_str() == "sweep"))
        .expect("sweep arg");
    assert!(matches!(sweep.value, Expr::Binary { op: BinOp::Mul, .. }));
}
