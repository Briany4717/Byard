//! Local type inference and the D9 type checks (RFC-0002 D9; RFC-0003 E2;
//! RFC-0005 §1).
//!
//! This is deliberately **not** Hindley-Milner. The rules:
//!
//! - **`View` parameters and `fn` signatures** (params + return) require
//!   explicit annotations; a missing one is [`CompileError::MissingAnnotation`].
//! - **`var`/`let`** infer locally from their initializer (int→`Int`,
//!   float→`Float`, string→`Str`, bool→`Bool`, homogeneous array→`List<T>`,
//!   call→callee return type). A non-inferable initializer (the empty array, a
//!   heterogeneous array) without an annotation is [`CompileError::CannotInfer`].
//! - **Lambda parameters are exempt** (E2): their types come from the expected
//!   `Fn` type at the use site, so they are never flagged here.
//! - **`Text` is a view, not a type** (INV-7): using it in any annotation is
//!   [`CompileError::TextUsedAsType`]; the scalar string type is `Str`.

use std::collections::HashMap;

use crate::diagnostics::{CompileError, Span};
use crate::interp::style::check_static;
use crate::parser::ast::{Expr, Member, Param, Type, ViewDecl};
use crate::symbol::Symbol;

/// An inferred / resolved type. Distinct from the syntactic [`Type`] AST node:
/// `Ty` is normalized (e.g. `List<Str>` ⇒ `List(Str)`), and `Unknown` covers
/// the cases Phase 2 cannot resolve without controller metadata or a method
/// catalog (member access, most method calls) — those are not errors, since the
/// Dev interpreter is dynamically evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    /// `Int` (`i64`).
    Int,
    /// `Float` (`f64`).
    Float,
    /// `Str` — the scalar string type.
    Str,
    /// `Bool`.
    Bool,
    /// `List<T>`.
    List(Box<Ty>),
    /// `Fn(params) -> ret`.
    Fn(Vec<Ty>, Option<Box<Ty>>),
    /// A named type the checker does not model further (e.g. `AppEnvironment`).
    Named(Symbol),
    /// A type Phase 2 cannot determine; permitted, never an error by itself.
    Unknown,
}

/// The result of checking a set of views: any diagnostics, plus the inferred
/// types of top-level `var`/`let` bindings (exposed for tests and the future
/// transpiler).
#[derive(Debug, Default)]
pub struct Inference {
    /// Diagnostics produced by the checks.
    pub errors: Vec<CompileError>,
    /// Inferred type of each top-level `var`/`let`, in declaration order.
    pub bindings: Vec<(Symbol, Ty)>,
}

/// Runs the D9 checks and local inference over `views`.
#[must_use]
pub fn check_views(views: &[ViewDecl]) -> Inference {
    let mut out = Inference::default();
    for view in views {
        let mut checker = Checker {
            errors: &mut out.errors,
            bindings: &mut out.bindings,
            env: HashMap::new(),
            fns: HashMap::new(),
        };
        checker.check_view(view);
    }
    out
}

struct Checker<'a> {
    errors: &'a mut Vec<CompileError>,
    bindings: &'a mut Vec<(Symbol, Ty)>,
    /// Known binding types in scope (params, vars, lets).
    env: HashMap<Symbol, Ty>,
    /// Declared return types of named `fn`s, for call inference.
    fns: HashMap<Symbol, Option<Ty>>,
}

impl Checker<'_> {
    fn check_view(&mut self, view: &ViewDecl) {
        // View parameters must be annotated (D9).
        for param in &view.params {
            self.check_param(param, "view parameter");
        }
        // Pre-collect `fn` return types so calls in any order can resolve them.
        for member in &view.body {
            if let Member::Fn { name, ret, .. } = member {
                let ty = ret.as_ref().map(|t| self.resolve_type(t));
                self.fns.insert(name.clone(), ty);
            }
        }
        // Collect var names for static style checks (M11)
        let mut vars = Vec::new();
        for member in &view.body {
            if let Member::Var { name, .. } = member {
                vars.push(name.clone());
            }
        }
        self.check_members(&view.body, true);
        // Run style check for all style blocks (M11)
        for member in &view.body {
            if let Member::Style { rules, .. } = member {
                let style_errors = check_static(rules, &vars);
                self.errors.extend(style_errors);
            }
        }
    }

    fn check_param(&mut self, param: &Param, what: &str) {
        if let Some(ty) = &param.ty {
            let resolved = self.resolve_type(ty);
            self.env.insert(param.name.clone(), resolved);
        } else {
            self.errors.push(CompileError::MissingAnnotation {
                span: param.span,
                what: what.to_string(),
            });
            self.env.insert(param.name.clone(), Ty::Unknown);
        }
    }

    fn check_members(&mut self, members: &[Member], top_level: bool) {
        for member in members {
            self.check_member(member, top_level);
        }
    }

    fn check_member(&mut self, member: &Member, top_level: bool) {
        match member {
            Member::Var {
                name,
                ty,
                init,
                span,
            }
            | Member::Let {
                name,
                ty,
                init,
                span,
            } => {
                let inferred = self.infer_binding(ty.as_ref(), init, *span);
                self.env.insert(name.clone(), inferred.clone());
                if top_level {
                    self.bindings.push((name.clone(), inferred));
                }
            }
            Member::Fn { params, ret, .. } => {
                for param in params {
                    self.check_param(param, "function parameter");
                }
                if ret.is_none() {
                    self.errors.push(CompileError::MissingAnnotation {
                        span: member_span(member),
                        what: "function return".to_string(),
                    });
                }
            }
            Member::Inject { ty, name, .. } => {
                let resolved = self.resolve_type(ty);
                self.env.insert(name.clone(), resolved);
            }
            Member::Element(el) => {
                // Nested members in children keep the same View scope.
                self.check_members(&el.children, false);
            }
            Member::For { body, .. } => self.check_members(body, false),
            Member::When { then, els, .. } => {
                self.check_members(then, false);
                if let Some(els) = els {
                    self.check_members(els, false);
                }
            }
            Member::Style { .. } | Member::Expr(_) => {}
        }
    }

    /// Infers a `var`/`let` type from its annotation (if any) or initializer.
    fn infer_binding(&mut self, annot: Option<&Type>, init: &Expr, span: Span) -> Ty {
        if let Some(ty) = annot {
            return self.resolve_type(ty);
        }
        match init {
            Expr::Array(elems, _) => self.infer_array(elems, span),
            other => self.infer_expr(other),
        }
    }

    fn infer_array(&mut self, elems: &[Expr], span: Span) -> Ty {
        if elems.is_empty() {
            self.errors.push(CompileError::CannotInfer { span });
            return Ty::Unknown;
        }
        let mut element_ty: Option<Ty> = None;
        for e in elems {
            let t = self.infer_expr(e);
            if t == Ty::Unknown {
                // Can't confirm homogeneity from an unknown element; stay lenient.
                continue;
            }
            match &element_ty {
                None => element_ty = Some(t),
                Some(prev) if *prev == t => {}
                Some(_) => {
                    // Heterogeneous concrete types ⇒ require an annotation.
                    self.errors.push(CompileError::CannotInfer { span });
                    return Ty::Unknown;
                }
            }
        }
        element_ty.map_or(Ty::Unknown, |t| Ty::List(Box::new(t)))
    }

    fn infer_expr(&self, expr: &Expr) -> Ty {
        match expr {
            Expr::IntLit(..) => Ty::Int,
            Expr::FloatLit(..) => Ty::Float,
            Expr::StrLit(..) => Ty::Str,
            Expr::Ident(sym, _) => {
                // `true` / `false` are contextual boolean literals (the grammar
                // has no bool token; they lex as identifiers).
                if sym.as_str() == "true" || sym.as_str() == "false" {
                    Ty::Bool
                } else {
                    self.env.get(sym).cloned().unwrap_or(Ty::Unknown)
                }
            }
            Expr::Call { callee, .. } => match callee.as_ref() {
                // A call to a named `fn` resolves to its declared return type.
                Expr::Ident(name, _) => {
                    self.fns.get(name).cloned().flatten().unwrap_or(Ty::Unknown)
                }
                // Method calls need a catalog Phase 2 does not have.
                _ => Ty::Unknown,
            },
            // Member access needs controller metadata (not modeled in Phase 2).
            _ => Ty::Unknown,
        }
    }

    /// Maps a syntactic [`Type`] to a [`Ty`], enforcing INV-7 (`Text` is not a
    /// type) and normalizing the known scalar/`List`/`Fn` forms.
    fn resolve_type(&mut self, ty: &Type) -> Ty {
        match ty {
            Type::Named { name, args, span } => {
                if name.as_str() == "Text" {
                    self.errors
                        .push(CompileError::TextUsedAsType { span: *span });
                    return Ty::Unknown;
                }
                match name.as_str() {
                    "Int" => Ty::Int,
                    "Float" => Ty::Float,
                    "Str" => Ty::Str,
                    "Bool" => Ty::Bool,
                    "List" => {
                        let inner = args.first().map_or(Ty::Unknown, |a| self.resolve_type(a));
                        Ty::List(Box::new(inner))
                    }
                    _ => {
                        // Still recurse into args to catch a nested `Text`.
                        for a in args {
                            let _ = self.resolve_type(a);
                        }
                        Ty::Named(name.clone())
                    }
                }
            }
            Type::Function { params, ret, .. } => {
                let params = params.iter().map(|p| self.resolve_type(p)).collect();
                let ret = ret.as_ref().map(|r| Box::new(self.resolve_type(r)));
                Ty::Fn(params, ret)
            }
        }
    }
}

/// The span of a member (used for member-level diagnostics).
fn member_span(member: &Member) -> Span {
    match member {
        Member::Var { span, .. }
        | Member::Let { span, .. }
        | Member::Fn { span, .. }
        | Member::Inject { span, .. }
        | Member::For { span, .. }
        | Member::When { span, .. }
        | Member::Style { span, .. } => *span,
        Member::Element(el) => el.span,
        Member::Expr(e) => e.span(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn sym(s: &str) -> Symbol {
        Symbol::intern(s)
    }

    /// Infers the top-level bindings of a single-view source, asserting no
    /// diagnostics.
    fn infer_ok(src: &str) -> Vec<(Symbol, Ty)> {
        let parsed = parse(src);
        assert!(
            parsed.errors.is_empty(),
            "parse errors: {:?}",
            parsed.errors
        );
        let inf = check_views(&parsed.views);
        assert!(
            inf.errors.is_empty(),
            "unexpected type errors: {:?}",
            inf.errors
        );
        inf.bindings
    }

    fn errors_of(src: &str) -> Vec<CompileError> {
        let parsed = parse(src);
        check_views(&parsed.views).errors
    }

    #[test]
    fn infers_scalar_literals() {
        let b = infer_ok(
            "View V() { var i = 12\n var f = 0.5\n var s = \"hi\"\n var t = true\n var u = false }",
        );
        assert_eq!(b[0], (sym("i"), Ty::Int));
        assert_eq!(b[1], (sym("f"), Ty::Float));
        assert_eq!(b[2], (sym("s"), Ty::Str));
        assert_eq!(b[3], (sym("t"), Ty::Bool));
        assert_eq!(b[4], (sym("u"), Ty::Bool));
    }

    #[test]
    fn infers_homogeneous_array() {
        let b = infer_ok("View V() { var xs = [\"a\", \"b\", \"c\"] }");
        assert_eq!(b[0], (sym("xs"), Ty::List(Box::new(Ty::Str))));
    }

    #[test]
    fn annotation_overrides_inference() {
        let b = infer_ok("View V() { var xs: List<Str> = [] }");
        assert_eq!(b[0], (sym("xs"), Ty::List(Box::new(Ty::Str))));
    }

    #[test]
    fn empty_array_without_annotation_cannot_infer() {
        let errs = errors_of("View V() { var items = [] }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::CannotInfer { .. })),
            "got {errs:?}"
        );
    }

    #[test]
    fn heterogeneous_array_cannot_infer() {
        let errs = errors_of("View V() { var x = [1, \"a\"] }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::CannotInfer { .. }))
        );
    }

    #[test]
    fn unannotated_fn_param_is_missing_annotation() {
        let errs = errors_of("View V() { fn f(x) -> Int => x }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::MissingAnnotation { what, .. } if what == "function parameter")),
            "got {errs:?}"
        );
    }

    #[test]
    fn fn_without_return_type_is_missing_annotation() {
        let errs = errors_of("View V() { fn f(x: Int) => x }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::MissingAnnotation { what, .. } if what == "function return"))
        );
    }

    #[test]
    fn unannotated_view_param_is_missing_annotation() {
        let errs = errors_of("View V(name) {}");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::MissingAnnotation { what, .. } if what == "view parameter"))
        );
    }

    #[test]
    fn text_as_a_type_is_rejected() {
        // `fn ... -> Text` and `var x: Text` both use the view name as a type.
        let errs = errors_of("View V() { fn g() -> Text => 1 }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::TextUsedAsType { .. }))
        );

        let errs2 = errors_of("View V(x: Text) {}");
        assert!(
            errs2
                .iter()
                .any(|e| matches!(e, CompileError::TextUsedAsType { .. }))
        );
    }

    #[test]
    fn call_to_named_fn_infers_its_return_type() {
        let b = infer_ok("View V() { fn make() -> Int => 1\n var n = make() }");
        assert_eq!(b.iter().find(|(s, _)| *s == sym("n")).unwrap().1, Ty::Int);
    }

    #[test]
    fn golden_examples_type_check_cleanly() {
        // The four canonical examples must produce no D9 diagnostics.
        let goldens = [
            "View Counter() { var count = 0\n Text(\"{count}\") }",
            "View UserCard() { var clicks = 0\n inject AppEnvironment as env\n Text(\"{clicks}\") }",
            "View Search() { var query = \"\"\n var items: List<Str> = [\"a\"]\n let filtered = items.filter(|x| x)\n fn greeting() -> Str => \"x\" }",
            "View ProfileCard(name: Str) { var liked = false }",
        ];
        for src in goldens {
            let errs = errors_of(src);
            assert!(errs.is_empty(), "{src}\n→ {errs:?}");
        }
    }

    #[test]
    fn dynamic_style_in_style_block_is_forbidden_during_type_checking() {
        let errs = errors_of("View V() {\n var c = 1\n style { .a #[bg: c] }\n}");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::DynamicStyleForbidden { .. })),
            "expected DynamicStyleForbidden, got {errs:?}"
        );
    }
}
