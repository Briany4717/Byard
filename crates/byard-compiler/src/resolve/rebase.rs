//! Span rebasing: shifts every [`Span`] in a parsed AST by a fixed byte
//! delta, so per-file spans become offsets into the program-wide
//! [`SourceMap`](super::SourceMap) (RFC-0008 Pillar A).
//!
//! Each file is parsed in isolation (spans start at 0); the resolver then
//! rebases the whole tree by the file's base offset. Every match here is
//! **exhaustive on purpose** — when the AST grows a new node the compiler
//! forces this walker to acknowledge it, so a span can never silently stay
//! file-relative and point diagnostics at the wrong file.

use crate::diagnostics::Span;
use crate::parser::ast::{
    Arg, Attr, AttrKind, ElementNode, Expr, Member, Param, StateBlock, StrPart, StyleRule, Type,
    UseDecl, ViewDecl,
};

fn shift(span: &mut Span, delta: u32) {
    span.start += delta;
    span.end += delta;
}

/// Rebases a whole `View` declaration (params, body, every expression) by
/// `delta` bytes.
pub(super) fn shift_view(view: &mut ViewDecl, delta: u32) {
    shift(&mut view.span, delta);
    for param in &mut view.params {
        shift_param(param, delta);
    }
    for member in &mut view.body {
        shift_member(member, delta);
    }
}

/// Rebases a `use` declaration by `delta` bytes.
pub(super) fn shift_use(decl: &mut UseDecl, delta: u32) {
    shift(&mut decl.span, delta);
    if let Some(symbols) = &mut decl.symbols {
        for (_, span) in symbols {
            shift(span, delta);
        }
    }
}

fn shift_param(param: &mut Param, delta: u32) {
    shift(&mut param.span, delta);
    if let Some(ty) = &mut param.ty {
        shift_type(ty, delta);
    }
    if let Some(default) = &mut param.default {
        shift_expr(default, delta);
    }
}

fn shift_type(ty: &mut Type, delta: u32) {
    match ty {
        Type::Named { args, span, .. } => {
            shift(span, delta);
            for arg in args {
                shift_type(arg, delta);
            }
        }
        Type::Function { params, ret, span } => {
            shift(span, delta);
            for p in params {
                shift_type(p, delta);
            }
            if let Some(r) = ret {
                shift_type(r, delta);
            }
        }
    }
}

fn shift_member(member: &mut Member, delta: u32) {
    match member {
        Member::Var { ty, init, span, .. } | Member::Let { ty, init, span, .. } => {
            shift(span, delta);
            if let Some(ty) = ty {
                shift_type(ty, delta);
            }
            shift_expr(init, delta);
        }
        Member::Fn {
            params,
            ret,
            body,
            span,
            ..
        } => {
            shift(span, delta);
            for p in params {
                shift_param(p, delta);
            }
            if let Some(r) = ret {
                shift_type(r, delta);
            }
            shift_expr(body, delta);
        }
        Member::Inject { ty, span, .. } => {
            shift(span, delta);
            shift_type(ty, delta);
        }
        Member::Element(el) => shift_element(el, delta),
        Member::For {
            iter, body, span, ..
        } => {
            shift(span, delta);
            shift_expr(iter, delta);
            for m in body {
                shift_member(m, delta);
            }
        }
        Member::When {
            cond,
            then,
            els,
            span,
        } => {
            shift(span, delta);
            shift_expr(cond, delta);
            for m in then {
                shift_member(m, delta);
            }
            if let Some(els) = els {
                for m in els {
                    shift_member(m, delta);
                }
            }
        }
        Member::Style { rules, span } => {
            shift(span, delta);
            for rule in rules {
                shift_style_rule(rule, delta);
            }
        }
        Member::Expr(expr) => shift_expr(expr, delta),
    }
}

fn shift_element(el: &mut ElementNode, delta: u32) {
    shift(&mut el.span, delta);
    for arg in &mut el.content {
        shift_arg(arg, delta);
    }
    for attr in &mut el.attrs {
        shift_attr(attr, delta);
    }
    if let Some(action) = &mut el.action {
        shift_expr(action, delta);
    }
    for child in &mut el.children {
        shift_member(child, delta);
    }
}

fn shift_attr(attr: &mut Attr, delta: u32) {
    shift(&mut attr.span, delta);
    match &mut attr.kind {
        AttrKind::Prop { value } | AttrKind::Spread { value } => shift_expr(value, delta),
        AttrKind::Event { action, .. } => shift_expr(action, delta),
    }
}

fn shift_state_block(block: &mut StateBlock, delta: u32) {
    shift(&mut block.span, delta);
    for attr in &mut block.attrs {
        shift_attr(attr, delta);
    }
}

fn shift_style_rule(rule: &mut StyleRule, delta: u32) {
    shift(&mut rule.span, delta);
    for attr in &mut rule.attrs {
        shift_attr(attr, delta);
    }
}

fn shift_arg(arg: &mut Arg, delta: u32) {
    shift_expr(&mut arg.value, delta);
}

fn shift_expr(expr: &mut Expr, delta: u32) {
    match expr {
        Expr::IntLit(_, span)
        | Expr::FloatLit(_, span)
        | Expr::AngleLit(_, span)
        | Expr::Ident(_, span)
        | Expr::ClassRef(_, span)
        | Expr::Error(span) => shift(span, delta),
        Expr::StrLit(parts, span) => {
            shift(span, delta);
            for part in parts {
                if let StrPart::Interp(inner) = part {
                    shift_expr(inner, delta);
                }
            }
        }
        Expr::Array(items, span) => {
            shift(span, delta);
            for item in items {
                shift_expr(item, delta);
            }
        }
        Expr::Tuple(args, span) => {
            shift(span, delta);
            for arg in args {
                shift_arg(arg, delta);
            }
        }
        Expr::Member { base, span, .. } => {
            shift(span, delta);
            shift_expr(base, delta);
        }
        Expr::Call { callee, args, span } => {
            shift(span, delta);
            shift_expr(callee, delta);
            for arg in args {
                shift_arg(arg, delta);
            }
        }
        Expr::Lambda { body, span, .. } => {
            shift(span, delta);
            shift_expr(body, delta);
        }
        Expr::Block(stmts, span) => {
            shift(span, delta);
            for stmt in stmts {
                shift_expr(stmt, delta);
            }
        }
        Expr::Assign {
            target,
            value,
            span,
            ..
        } => {
            shift(span, delta);
            shift_expr(target, delta);
            shift_expr(value, delta);
        }
        Expr::Postfix { target, span, .. } => {
            shift(span, delta);
            shift_expr(target, delta);
        }
        Expr::Binary { lhs, rhs, span, .. } => {
            shift(span, delta);
            shift_expr(lhs, delta);
            shift_expr(rhs, delta);
        }
        Expr::Ternary {
            cond,
            then,
            els,
            span,
        } => {
            shift(span, delta);
            shift_expr(cond, delta);
            shift_expr(then, delta);
            shift_expr(els, delta);
        }
        Expr::Animated { value, anim, span } => {
            shift(span, delta);
            shift_expr(value, delta);
            shift_expr(anim, delta);
        }
        Expr::StyleValue {
            attrs,
            states,
            span,
        } => {
            shift(span, delta);
            for attr in attrs {
                shift_attr(attr, delta);
            }
            for state in states {
                shift_state_block(state, delta);
            }
        }
        Expr::Merge { left, right, span } => {
            shift(span, delta);
            shift_expr(left, delta);
            shift_expr(right, delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn every_span_in_a_rich_view_moves_by_the_delta() {
        let src = "View A(x: Int = 1) {\n  var n = 0\n  style { .card #[bg: 0xFF0000] }\n  \
                   when n { Text(\"hi {n}\") #[size: 12] } else { B() }\n  \
                   for i in items { Box #[..s, translate.y: 2] => n++ }\n}";
        let parsed = parse(src);
        let mut moved = parsed.views[0].clone();
        shift_view(&mut moved, 100);
        assert_eq!(moved.span.start, parsed.views[0].span.start + 100);
        assert_eq!(
            moved.params[0].span.start,
            parsed.views[0].params[0].span.start + 100
        );
        // Spot-check a deeply nested span: the `when` member.
        let (Member::When { span: old, .. }, Member::When { span: new, .. }) =
            (&parsed.views[0].body[2], &moved.body[2])
        else {
            panic!("expected when at body[2]");
        };
        assert_eq!(new.start, old.start + 100);
    }
}
