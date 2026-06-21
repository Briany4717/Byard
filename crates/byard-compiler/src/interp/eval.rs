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
use super::intrinsics::validate_element;
use super::reactive::{FrameTarget, ReactiveCtx, ScopeId, untrack};
use crate::diagnostics::{CompileError, Span};
use crate::parser::ast::{
    AssignOp, Attr, AttrKind, ElementNode, Expr, Member, PostfixOp, StrPart, ViewDecl,
};
use crate::symbol::Symbol;

/// A lowered render-tree node: the interpreter's plan for one element. Reactive
/// fields are reactive-scope ids the engine reads each tick (M14); static
/// fields (a literal `bg`, `radius`) are resolved at lowering time.
#[derive(Debug, Clone, PartialEq)]
pub enum RenderNode {
    /// A box-like container (`Box`/`Column`/`Row`/`Button` background).
    Box {
        /// Static background color (`0xRRGGBB`), if a literal `bg:` was given.
        bg: Option<i64>,
        /// Static corner radius.
        radius: i64,
        /// Child render nodes.
        children: Vec<RenderNode>,
    },
    /// A text run; `content` is the value binding projecting its string.
    Text {
        /// The reactive scope projecting the text content.
        content: ScopeId,
    },
    /// A flexible gap (layout-only).
    Spacer,
}

/// A lowered reactive computation (see the module docs).
type Lowered = Box<dyn FnMut(&mut ReactiveCtx) -> Value>;

/// The Dev-mode interpreter for one `View` instance: a reactive context plus
/// the View's binding environment.
#[derive(Default)]
pub struct Interpreter {
    ctx: ReactiveCtx,
    env: Env<'static>,
    next_target: u32,
    errors: Vec<CompileError>,
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

    /// Runs one tick: begins an epoch and pulls all dirty scopes.
    pub fn tick(&mut self) {
        let epoch = self.ctx.begin_tick();
        self.ctx.pull(epoch);
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
                self.define_let(name.clone(), init);
            }
            Member::Fn { name, body, .. } => {
                // Phase 2: a `fn` lowers to a memo of its body (params are bound
                // at call sites in a later milestone).
                self.define_let(name.clone(), body);
            }
            Member::Expr(e) => {
                if let Err(err) = self.eval_action(e) {
                    self.errors.push(err);
                }
            }
            // inject / elements / control flow / style are wired in M9.x/M10+.
            _ => {}
        }
    }

    /// `var x = init` — evaluate `init` once, create a reactive source from it.
    pub fn define_var(&mut self, name: Symbol, init: &Expr) -> SignalId {
        let initial = self.eval_pure(init);
        let sig = self.ctx.create_signal(initial);
        self.env.push(name, Value::Signal(sig));
        sig
    }

    /// `let y = init` (and `fn`) — open a computed memo.
    pub fn define_let(&mut self, name: Symbol, init: &Expr) -> ScopeId {
        let compute = self.lower_expr(init);
        let scope = self.ctx.open_memo(compute);
        self.env.push(name, Value::Memo(scope));
        scope
    }

    /// Opens a value binding projecting `expr` into a fresh frame target
    /// (used by intrinsics, M10, and by tests).
    pub fn bind_value(&mut self, expr: &Expr) -> ScopeId {
        let target = self.next_target();
        let compute = self.lower_expr(expr);
        self.ctx.open_value_binding(target, compute)
    }

    // ── element lowering (RFC-0005) ─────────────────────────────────────

    /// Lowers an element to a [`RenderNode`], validating it against the §5
    /// attribute contract first (diagnostics accumulate in [`Interpreter::errors`]).
    /// `known_views` are user `ViewDecl` names in scope.
    pub fn lower_element(&mut self, el: &ElementNode, known_views: &[&str]) -> RenderNode {
        self.errors.extend(validate_element(el, known_views));
        match el.name.as_str() {
            "Text" | "Button" if !el.content.is_empty() => {
                let content = self.bind_value(&el.content[0].value);
                if el.name.as_str() == "Button" {
                    // A Button is a decorated box wrapping its label.
                    RenderNode::Box {
                        bg: static_color(&el.attrs, "bg"),
                        radius: static_len(&el.attrs, "radius"),
                        children: vec![RenderNode::Text { content }],
                    }
                } else {
                    RenderNode::Text { content }
                }
            }
            "Spacer" => RenderNode::Spacer,
            _ => {
                // Box / Column / Row / ScrollView and any other container.
                let children = el
                    .children
                    .iter()
                    .filter_map(|m| match m {
                        Member::Element(e) => Some(self.lower_element(e, known_views)),
                        _ => None,
                    })
                    .collect();
                RenderNode::Box {
                    bg: static_color(&el.attrs, "bg"),
                    radius: static_len(&el.attrs, "radius"),
                    children,
                }
            }
        }
    }

    /// Processes a whole `View`: its declarations first (so bindings can resolve
    /// names), then lowers its top-level elements into a render tree.
    pub fn lower_view(&mut self, view: &ViewDecl, known_views: &[&str]) -> Vec<RenderNode> {
        self.eval_view_decls(view);
        view.body
            .iter()
            .filter_map(|m| match m {
                Member::Element(e) => Some(self.lower_element(e, known_views)),
                _ => None,
            })
            .collect()
    }

    // ── lowering ────────────────────────────────────────────────────────

    /// Lowers `expr` to a reactive computation against the current environment.
    fn lower_expr(&self, expr: &Expr) -> Lowered {
        match expr {
            Expr::IntLit(n, _) => {
                let n = *n;
                Box::new(move |_| Value::Int(n))
            }
            Expr::FloatLit(f, _) => {
                let f = *f;
                Box::new(move |_| Value::Float(f))
            }
            Expr::StrLit(parts, _) => self.lower_strlit(parts),
            Expr::Ident(name, _) => self.lower_ident(name),
            Expr::Array(elems, _) | Expr::Tuple(elems, _) => {
                let mut cs: Vec<Lowered> = elems.iter().map(|e| self.lower_expr(e)).collect();
                Box::new(move |ctx| Value::List(cs.iter_mut().map(|c| c(ctx)).collect()))
            }
            Expr::Ternary {
                cond, then, els, ..
            } => {
                let mut cc = self.lower_expr(cond);
                let mut tc = self.lower_expr(then);
                let mut ec = self.lower_expr(els);
                Box::new(move |ctx| {
                    if cc(ctx).as_bool().unwrap_or(false) {
                        tc(ctx)
                    } else {
                        ec(ctx)
                    }
                })
            }
            Expr::Call { callee, args, .. } => self.lower_call(callee, args),
            Expr::ClassRef(class, _) => {
                let s = format!(".{class}");
                Box::new(move |_| Value::Str(s.clone()))
            }
            // Member access needs controller metadata (not modeled in Phase 2);
            // lambdas/assignments are actions, not projected values.
            Expr::Member { .. } | Expr::Lambda { .. } | Expr::Error(_) => Box::new(|_| Value::Unit),
            Expr::Assign { .. } | Expr::Postfix { .. } => Box::new(|_| Value::Unit),
        }
    }

    fn lower_ident(&self, name: &Symbol) -> Lowered {
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

    fn lower_strlit(&self, parts: &[StrPart]) -> Lowered {
        enum Part {
            Text(String),
            Interp(Lowered),
        }
        let mut lowered: Vec<Part> = parts
            .iter()
            .map(|p| match p {
                StrPart::Text(t) => Part::Text(t.clone()),
                StrPart::Interp(e) => Part::Interp(self.lower_expr(e)),
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

    fn lower_call(&self, callee: &Expr, args: &[crate::parser::ast::Arg]) -> Lowered {
        // `untrack(expr)` — the reserved escape hatch (D2).
        if let Expr::Ident(name, _) = callee {
            if name.as_str() == "untrack" {
                if let Some(arg) = args.first() {
                    let mut inner = self.lower_expr(&arg.value);
                    return Box::new(move |ctx| untrack(|| inner(ctx)));
                }
            }
            // A zero-arg call to a `fn`/`let` memo reads that memo.
            if let Some(Value::Memo(scope)) = self.env.lookup(name) {
                let m = *scope;
                return Box::new(move |ctx| ctx.read_memo(m));
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
        let mut compute = self.lower_expr(expr);
        compute(&mut self.ctx)
    }

    fn resolve_var(&self, target: &Expr, span: Span) -> Result<SignalId, CompileError> {
        if let Expr::Ident(name, _) = target {
            if let Some(Value::Signal(sig)) = self.env.lookup(name) {
                return Ok(*sig);
            }
        }
        Err(CompileError::NotAssignable { span })
    }
}

/// Finds a static (literal) `Color` attribute value, if present.
fn static_color(attrs: &[Attr], name: &str) -> Option<i64> {
    attrs
        .iter()
        .find_map(|a| match (&a.kind, a.name.as_str() == name) {
            (
                AttrKind::Prop {
                    value: Expr::IntLit(n, _),
                },
                true,
            ) => Some(*n),
            _ => None,
        })
}

/// Finds a static (literal) scalar `Len` attribute value, defaulting to `0`.
fn static_len(attrs: &[Attr], name: &str) -> i64 {
    attrs
        .iter()
        .find_map(|a| match (&a.kind, a.name.as_str() == name) {
            (
                AttrKind::Prop {
                    value: Expr::IntLit(n, _),
                },
                true,
            ) => Some(*n),
            _ => None,
        })
        .unwrap_or(0)
}

/// Renders a value for string interpolation (`"Count: {count}"`).
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
            bg,
            radius,
            children,
        } = &tree[0]
        else {
            panic!("expected a Box, got {:?}", tree[0]);
        };
        assert_eq!(*bg, Some(0x0022_2222));
        assert_eq!(*radius, 16);
        assert_eq!(children.len(), 1);
        let RenderNode::Text { content } = &children[0] else {
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
}
