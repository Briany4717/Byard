//! Per-`View` binding environment and `inject` resolution (RFC-0002 §"Dev-mode
//! interpreter — values").
//!
//! [`Env`] is a flat `Vec<(Symbol, Value)>` walked in **reverse** so shadowing
//! is free. It is allocated once per `View` instance and never truncated until
//! the View's `ViewArena` drops — nothing below the `View` level introduces a
//! scope (nested elements share the View's single `Env`), so a binding pushed
//! while evaluating one element stays visible to the next.
//!
//! `inject T as name` is a runtime, React-Context-style lookup: it walks a
//! parent-pointer chain of ancestor `Env` references, returning the nearest
//! ambient value provided for `T`, or [`CompileError::UnresolvedInject`] if no
//! ancestor provides one.

use super::reactive::ScopeId;
use crate::diagnostics::{CompileError, Span};
use crate::symbol::Symbol;

/// An opaque handle to a lambda's AST subtree, held in a side-table by the eval
/// driver (M9). Lambdas are **never** Rust closures (RFC-0002): a `Value::Fn`
/// names an AST node, re-walked when the bound event fires or an iterator
/// applies it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AstId(pub u32);

/// An opaque handle to a reactive source (`Signal`) allocated in the View's
/// arena and stored in the reactive context (M8). Kept as an id rather than an
/// embedded `Signal<'a, _>` so [`Value`] carries no arena lifetime and the
/// environment stays simple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SignalId(pub u32);

/// A runtime value in the Dev interpreter.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// `Int` (`i64`).
    Int(i64),
    /// `Float` (`f64`).
    Float(f64),
    /// `Bool`.
    Bool(bool),
    /// `Str`.
    Str(String),
    /// `List<T>`.
    List(Vec<Value>),
    /// A lambda, referenced by its AST id (not a Rust closure).
    Fn(AstId),
    /// A reactive source (`var`), referenced by its `Signal` id.
    Signal(SignalId),
    /// A computed memo (`let`/`fn`), referenced by its reactive scope id.
    Memo(ScopeId),
    /// The unit value (e.g. the result of a mutation).
    Unit,
}

impl Value {
    /// Returns the `i64` if this is an [`Value::Int`].
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns the `bool` if this is a [`Value::Bool`].
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the elements if this is a [`Value::List`].
    #[must_use]
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(xs) => Some(xs),
            _ => None,
        }
    }
}

/// A per-`View` binding environment, optionally linked to its enclosing View's
/// environment for `inject` resolution.
///
/// The `'p` lifetime is the parent chain: a child `Env` borrows its parent for
/// as long as it lives, which is sound because an ancestor View is fully
/// constructed (and outlives) the descendants that look up through it.
#[derive(Debug, Default)]
pub struct Env<'p> {
    /// Ordinary bindings (`var`/`let`/`fn`/params), in push order; shadowing is
    /// resolved by reverse scan.
    bindings: Vec<(Symbol, Value)>,
    /// Ambient values provided to descendants, keyed by their type name — the
    /// provider side of `inject` (React-Context).
    ambient: Vec<(Symbol, Value)>,
    /// The enclosing View's environment, if any.
    parent: Option<&'p Env<'p>>,
}

impl<'p> Env<'p> {
    /// Creates a root environment with no parent.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            ambient: Vec::new(),
            parent: None,
        }
    }

    /// Creates a child environment linked to `parent` for `inject` resolution.
    #[must_use]
    pub fn child(parent: &'p Env<'p>) -> Self {
        Self {
            bindings: Vec::new(),
            ambient: Vec::new(),
            parent: Some(parent),
        }
    }

    /// Pushes a binding. Re-pushing a name shadows the previous one (the later
    /// push wins on lookup); the environment is never truncated below a View.
    pub fn push(&mut self, name: Symbol, value: Value) {
        self.bindings.push((name, value));
    }

    /// Looks up `name` in this View's bindings, most-recent first (shadowing).
    /// Does **not** walk the parent chain — only `inject` crosses View
    /// boundaries (RFC-0003 §6: no shared mutable state across Views).
    #[must_use]
    pub fn lookup(&self, name: &Symbol) -> Option<&Value> {
        self.bindings
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
    }

    /// Provides an ambient `ty` value to descendant Views (the provider side of
    /// `inject`).
    pub fn provide(&mut self, ty: Symbol, value: Value) {
        self.ambient.push((ty, value));
    }

    /// Resolves `inject ty` by walking this environment and its ancestors,
    /// nearest first, returning the first ambient value provided for `ty`.
    #[must_use]
    pub fn resolve_inject(&self, ty: &Symbol) -> Option<&Value> {
        let mut env: Option<&Env> = Some(self);
        while let Some(e) = env {
            if let Some(v) = e
                .ambient
                .iter()
                .rev()
                .find(|(k, _)| k == ty)
                .map(|(_, v)| v)
            {
                return Some(v);
            }
            env = e.parent;
        }
        None
    }

    /// Like [`Env::resolve_inject`], but turns a missing ambient into a
    /// [`CompileError::UnresolvedInject`] anchored at `span` (INV-4).
    ///
    /// # Errors
    ///
    /// Returns [`CompileError::UnresolvedInject`] if no ancestor provides `ty`.
    pub fn require_inject(&self, ty: &Symbol, span: Span) -> Result<&Value, CompileError> {
        self.resolve_inject(ty)
            .ok_or_else(|| CompileError::UnresolvedInject {
                span,
                name: ty.as_str().to_string(),
            })
    }

    /// The number of bindings currently in this environment (not counting
    /// ancestors).
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether this environment has no bindings of its own.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(s: &str) -> Symbol {
        Symbol::intern(s)
    }

    #[test]
    fn lookup_resolves_to_the_latest_shadow() {
        let mut env = Env::new();
        env.push(sym("x"), Value::Int(1));
        env.push(sym("y"), Value::Str("hi".to_string()));
        env.push(sym("x"), Value::Int(2));

        assert_eq!(env.lookup(&sym("x")), Some(&Value::Int(2)));
        assert_eq!(env.lookup(&sym("y")), Some(&Value::Str("hi".to_string())));
        assert_eq!(env.len(), 3);
    }

    #[test]
    fn lookup_missing_is_none() {
        let env = Env::new();
        assert!(env.is_empty());
        assert_eq!(env.lookup(&sym("nope")), None);
    }

    #[test]
    fn lookup_does_not_cross_view_boundaries() {
        let mut parent = Env::new();
        parent.push(sym("secret"), Value::Int(7));
        let child = Env::child(&parent);
        // A plain binding in the parent View is NOT visible to the child View.
        assert_eq!(child.lookup(&sym("secret")), None);
    }

    #[test]
    fn inject_finds_an_ancestor_value() {
        let mut grandparent = Env::new();
        grandparent.provide(sym("AppEnvironment"), Value::Str("theme".to_string()));
        let parent = Env::child(&grandparent);
        let child = Env::child(&parent);

        assert_eq!(
            child.resolve_inject(&sym("AppEnvironment")),
            Some(&Value::Str("theme".to_string()))
        );
    }

    #[test]
    fn inject_resolves_to_the_nearest_provider() {
        let mut grandparent = Env::new();
        grandparent.provide(sym("Theme"), Value::Int(1));
        let mut parent = Env::child(&grandparent);
        parent.provide(sym("Theme"), Value::Int(2));
        let child = Env::child(&parent);

        assert_eq!(child.resolve_inject(&sym("Theme")), Some(&Value::Int(2)));
    }

    #[test]
    fn missing_inject_is_unresolved_inject_error() {
        let env = Env::new();
        let err = env
            .require_inject(&sym("Missing"), Span::new(0, 7))
            .unwrap_err();
        assert!(matches!(
            err,
            CompileError::UnresolvedInject { ref name, .. } if name == "Missing"
        ));
    }

    #[test]
    fn fn_and_signal_values_round_trip() {
        let mut env = Env::new();
        env.push(sym("handler"), Value::Fn(AstId(3)));
        env.push(sym("count"), Value::Signal(SignalId(0)));
        assert_eq!(env.lookup(&sym("handler")), Some(&Value::Fn(AstId(3))));
        assert_eq!(env.lookup(&sym("count")), Some(&Value::Signal(SignalId(0))));
    }
}
