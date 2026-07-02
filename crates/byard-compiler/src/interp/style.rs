//! Scoped `style { }` resolution with three-layer precedence (RFC-0002 D5).
//!
//! A `style { .class #[…] }` block builds a per-`View` [`StyleMap`] — there is
//! **no global cascade**. An element's attributes are resolved in three layers,
//! last write wins:
//!
//! 1. the framework/theme **default** base style (D5 layer 1),
//! 2. each **class** the element references via `#[style: .class]`, merged in
//!    element-list order (later classes override earlier),
//! 3. the element's **inline** `#[...]` attributes (override everything).
//!
//! Phase 2 styles are **static** (D5/D8): reading a `var` inside a `style` block
//! is [`CompileError::DynamicStyleForbidden`].

use std::collections::HashMap;

use crate::diagnostics::CompileError;
use crate::parser::ast::{Attr, AttrKind, Expr, StyleRule};
use crate::symbol::Symbol;

/// A `View`'s scoped style table: class name → its attributes.
#[derive(Debug, Default)]
pub struct StyleMap {
    classes: HashMap<Symbol, Vec<Attr>>,
}

impl StyleMap {
    /// Builds a style map from a `style { }` block's rules.
    #[must_use]
    pub fn from_rules(rules: &[StyleRule]) -> Self {
        let mut classes = HashMap::new();
        for rule in rules {
            classes.insert(rule.class.clone(), rule.attrs.clone());
        }
        Self { classes }
    }

    /// Resolves an element's effective attributes by applying the three D5
    /// layers in order. `defaults` is the intrinsic's default base style.
    #[must_use]
    pub fn resolve(&self, defaults: &[Attr], element_attrs: &[Attr]) -> Vec<Attr> {
        let style_sym = Symbol::intern("style");
        let mut merged: Vec<Attr> = Vec::new();

        // Layer 1 — defaults.
        for a in defaults {
            set_attr(&mut merged, a.clone());
        }
        // Layer 2 — referenced classes, in element-list order.
        for a in element_attrs {
            if a.name == style_sym {
                if let AttrKind::Prop {
                    value: Expr::ClassRef(class, _),
                } = &a.kind
                {
                    if let Some(class_attrs) = self.classes.get(class) {
                        for ca in class_attrs {
                            set_attr(&mut merged, ca.clone());
                        }
                    }
                }
            }
        }
        // Layer 3 — inline attributes (everything except the `style:` selector).
        for a in element_attrs {
            if a.name != style_sym {
                set_attr(&mut merged, a.clone());
            }
        }
        merged
    }
}

/// Inserts `attr` into `attrs`, replacing any existing attribute of the same
/// name (last-write-wins).
fn set_attr(attrs: &mut Vec<Attr>, attr: Attr) {
    attrs.retain(|x| x.name != attr.name);
    attrs.push(attr);
}

/// Checks a `style` block for `var` reads (forbidden in Phase 2; D5/D8).
/// `var_names` is the set of reactive sources declared in the enclosing View.
#[must_use]
pub fn check_static(rules: &[StyleRule], var_names: &[Symbol]) -> Vec<CompileError> {
    let mut errs = Vec::new();
    for rule in rules {
        for attr in &rule.attrs {
            if let AttrKind::Prop { value } = &attr.kind {
                if let Some(span) = reads_var(value, var_names) {
                    errs.push(CompileError::DynamicStyleForbidden { span });
                }
            }
        }
    }
    errs
}

/// Returns the span of the first identifier in `expr` that names a `var`.
fn reads_var(expr: &Expr, vars: &[Symbol]) -> Option<crate::diagnostics::Span> {
    match expr {
        Expr::Ident(name, span) if vars.contains(name) => Some(*span),
        Expr::Member { base, .. } => reads_var(base, vars),
        Expr::Array(items, _) => items.iter().find_map(|e| reads_var(e, vars)),
        Expr::Tuple(items, _) => items.iter().find_map(|a| reads_var(&a.value, vars)),
        Expr::Call { callee, args, .. } => {
            reads_var(callee, vars).or_else(|| args.iter().find_map(|a| reads_var(&a.value, vars)))
        }
        Expr::Ternary {
            cond, then, els, ..
        } => reads_var(cond, vars)
            .or_else(|| reads_var(then, vars))
            .or_else(|| reads_var(els, vars)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{Member, ViewDecl};
    use crate::parser::parse;

    fn sym(s: &str) -> Symbol {
        Symbol::intern(s)
    }

    fn view(src: &str) -> ViewDecl {
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        parsed.views.into_iter().next().unwrap()
    }

    /// Extracts the style rules and the first element's attrs from a View.
    fn rules_and_element(view: &ViewDecl) -> (Vec<StyleRule>, Vec<Attr>) {
        let mut rules = Vec::new();
        let mut attrs = Vec::new();
        for m in &view.body {
            match m {
                Member::Style { rules: r, .. } => rules = r.clone(),
                Member::Element(e) if attrs.is_empty() => attrs = e.attrs.clone(),
                _ => {}
            }
        }
        (rules, attrs)
    }

    fn find_int(attrs: &[Attr], name: &str) -> Option<i64> {
        attrs
            .iter()
            .find(|a| a.name == sym(name))
            .and_then(|a| match &a.kind {
                AttrKind::Prop {
                    value: Expr::IntLit(n, _),
                } => Some(*n),
                _ => None,
            })
    }

    #[test]
    fn inline_overrides_class() {
        let v = view("View V() {\n Box #[style: .b, bg: 2] {}\n style { .b #[bg: 1] }\n}");
        let (rules, attrs) = rules_and_element(&v);
        let map = StyleMap::from_rules(&rules);
        let resolved = map.resolve(&[], &attrs);
        assert_eq!(
            find_int(&resolved, "bg"),
            Some(2),
            "inline bg wins over class"
        );
    }

    #[test]
    fn later_class_overrides_earlier() {
        let v = view(
            "View V() {\n Box #[style: .a, style: .b] {}\n style { .a #[bg: 1] .b #[bg: 2] }\n}",
        );
        let (rules, attrs) = rules_and_element(&v);
        let map = StyleMap::from_rules(&rules);
        let resolved = map.resolve(&[], &attrs);
        assert_eq!(find_int(&resolved, "bg"), Some(2), "later class .b wins");
    }

    #[test]
    fn default_applies_when_unset() {
        let v = view("View V() {\n Box #[bg: 5] {}\n}");
        let (_rules, attrs) = rules_and_element(&v);
        let defaults = vec![Attr {
            name: sym("p"),
            axis: None,
            kind: AttrKind::Prop {
                value: Expr::IntLit(16, crate::diagnostics::Span::new(0, 0)),
            },
            span: crate::diagnostics::Span::new(0, 0),
        }];
        let map = StyleMap::default();
        let resolved = map.resolve(&defaults, &attrs);
        assert_eq!(find_int(&resolved, "p"), Some(16), "default p survives");
        assert_eq!(find_int(&resolved, "bg"), Some(5), "inline bg present");
    }

    #[test]
    fn var_read_in_style_block_is_forbidden() {
        let v = view("View V() {\n var c = 1\n style { .a #[bg: c] }\n}");
        let mut rules = Vec::new();
        for m in &v.body {
            if let Member::Style { rules: r, .. } = m {
                rules = r.clone();
            }
        }
        let errs = check_static(&rules, &[sym("c")]);
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::DynamicStyleForbidden { .. })),
            "got {errs:?}"
        );
    }

    #[test]
    fn static_token_in_style_block_is_allowed() {
        // `center` is an enum token, not a var — no error.
        let v = view("View V() {\n var c = 1\n style { .a #[align: center] }\n}");
        let mut rules = Vec::new();
        for m in &v.body {
            if let Member::Style { rules: r, .. } = m {
                rules = r.clone();
            }
        }
        assert!(check_static(&rules, &[sym("c")]).is_empty());
    }
}
