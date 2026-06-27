//! The resolved user-`View` registry (RFC-0007 §1) and the call graph used for
//! recursion/cycle detection (RFC-0007 §4) and reload blast-radius (RFC-0007 §5).
//!
//! Lowering needs more than the names-only `known_views: &[&str]` slice the
//! interpreter historically threaded for validation — to *instantiate* a call it
//! needs the callee's [`ViewDecl`]. [`ViewTable`] owns its `ViewDecl`s (INV-3,
//! `Send`), maps each name to a dense [`ViewId`], and is built once per program
//! load from `ParsedFile::views`. The intrinsic catalog (RFC-0005) stays closed
//! and takes precedence: a `ViewDecl` named like an intrinsic is reported as
//! [`CompileError::IntrinsicShadowed`] (IMPL-50) and is never reachable, because
//! intrinsic dispatch runs first.

use std::collections::HashMap;

use super::intrinsics::lookup;
use crate::diagnostics::CompileError;
use crate::parser::ast::{Member, ViewDecl};
use crate::symbol::Symbol;

/// A dense index into [`ViewTable::decls`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewId(pub u32);

/// A resolved registry of the user `View`s declared in one `.byd` file
/// (RFC-0007 §1). Owns its `ViewDecl`s so the table is fully `Send` (INV-3/6).
#[derive(Clone, Debug, Default)]
pub struct ViewTable {
    by_name: HashMap<Symbol, ViewId>,
    decls: Vec<ViewDecl>,
}

impl ViewTable {
    /// Builds the table from a file's views, returning it alongside the
    /// [`CompileError::IntrinsicShadowed`] diagnostics for any view that
    /// collides with an RFC-0005 intrinsic (IMPL-50).
    #[must_use]
    pub fn build(views: &[ViewDecl]) -> (Self, Vec<CompileError>) {
        let mut by_name = HashMap::with_capacity(views.len());
        let mut decls = Vec::with_capacity(views.len());
        let mut errors = Vec::new();
        for view in views {
            if lookup(view.name.as_str()).is_some() {
                // The intrinsic always wins (IMPL-50); the view is unreachable.
                errors.push(CompileError::IntrinsicShadowed {
                    span: view.span,
                    name: view.name.as_str().to_string(),
                });
                continue;
            }
            let id = ViewId(u32::try_from(decls.len()).unwrap_or(u32::MAX));
            by_name.insert(view.name.clone(), id);
            decls.push(view.clone());
        }
        (Self { by_name, decls }, errors)
    }

    /// Resolves a name to its [`ViewId`], if it is a (non-shadowed) user view.
    #[must_use]
    pub fn resolve(&self, name: &Symbol) -> Option<ViewId> {
        self.by_name.get(name).copied()
    }

    /// Whether `name` resolves to a user view in this table.
    #[must_use]
    pub fn contains(&self, name: &Symbol) -> bool {
        self.by_name.contains_key(name)
    }

    /// The `ViewDecl` behind a [`ViewId`].
    #[must_use]
    pub fn decl(&self, id: ViewId) -> &ViewDecl {
        &self.decls[id.0 as usize]
    }

    /// Resolves a name directly to its `ViewDecl`, if present.
    #[must_use]
    pub fn get(&self, name: &Symbol) -> Option<&ViewDecl> {
        self.resolve(name).map(|id| self.decl(id))
    }

    /// The number of user views registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.decls.len()
    }

    /// Whether the table holds no user views.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.decls.is_empty()
    }

    /// The names of all registered views (for deriving the `known_views`
    /// validation slice without re-walking `ParsedFile`).
    pub fn names(&self) -> impl Iterator<Item = &str> + '_ {
        self.decls.iter().map(|v| v.name.as_str())
    }
}

/// Collects the user-view names referenced anywhere in a member subtree,
/// recording for each whether the reference is *guarded* by a `when`/`for`
/// structural boundary (RFC-0007 §4: a guarded self-reference is legal
/// recursion; an unguarded one diverges). `guarded` carries the guard state of
/// the enclosing context.
fn collect_refs(members: &[Member], guarded: bool, out: &mut Vec<(Symbol, bool)>) {
    for member in members {
        match member {
            Member::Element(el) => {
                out.push((el.name.clone(), guarded));
                collect_refs(&el.children, guarded, out);
            }
            Member::When { then, els, .. } => {
                collect_refs(then, true, out);
                if let Some(els) = els {
                    collect_refs(els, true, out);
                }
            }
            Member::For { body, .. } => collect_refs(body, true, out),
            _ => {}
        }
    }
}

/// A directed call graph over a [`ViewTable`]: an edge `A → B` (with a `guarded`
/// flag) exists when `A`'s body references user view `B` (RFC-0007 §4). Used for
/// static cycle detection (M33) and reload blast-radius (M34).
#[derive(Clone, Debug)]
pub struct CallGraph {
    /// `edges[a]` is the list of `(callee, guarded)` references in view `a`.
    edges: Vec<Vec<(ViewId, bool)>>,
}

impl CallGraph {
    /// Builds the call graph for `table`.
    #[must_use]
    pub fn build(table: &ViewTable) -> Self {
        let mut edges = vec![Vec::new(); table.len()];
        for (idx, edge_list) in edges.iter_mut().enumerate() {
            let decl = &table.decls[idx];
            let mut refs = Vec::new();
            collect_refs(&decl.body, false, &mut refs);
            for (name, guarded) in refs {
                if let Some(callee) = table.resolve(&name) {
                    edge_list.push((callee, guarded));
                }
            }
        }
        Self { edges }
    }

    /// The direct callees of `view` (each with its guarded flag).
    #[must_use]
    pub fn callees(&self, view: ViewId) -> &[(ViewId, bool)] {
        &self.edges[view.0 as usize]
    }

    /// Detects an **unguarded** call cycle reachable from any view, returning the
    /// offending view and the human-readable cycle path (RFC-0007 §4, IMPL-48).
    /// A cycle that passes through at least one `when`/`for` guard is legal
    /// recursion and is not reported.
    #[must_use]
    pub fn unguarded_cycle(&self, table: &ViewTable) -> Option<(ViewId, String)> {
        // DFS over only unguarded edges; a back-edge to a node on the current
        // stack is an unguarded cycle.
        let n = self.edges.len();
        let mut state = vec![0u8; n]; // 0 = unvisited, 1 = on-stack, 2 = done
        let mut stack: Vec<ViewId> = Vec::new();
        for start in 0..n {
            if state[start] == 0 {
                if let Some(cycle) =
                    self.dfs_unguarded(ViewId(start as u32), &mut state, &mut stack, table)
                {
                    return Some(cycle);
                }
            }
        }
        None
    }

    fn dfs_unguarded(
        &self,
        node: ViewId,
        state: &mut [u8],
        stack: &mut Vec<ViewId>,
        table: &ViewTable,
    ) -> Option<(ViewId, String)> {
        state[node.0 as usize] = 1;
        stack.push(node);
        for &(callee, guarded) in self.callees(node) {
            if guarded {
                continue; // a guarded edge cannot diverge unconditionally
            }
            match state[callee.0 as usize] {
                1 => {
                    // Back-edge: build the cycle path from the stack.
                    let pos = stack.iter().position(|&v| v == callee).unwrap_or(0);
                    let mut names: Vec<&str> = stack[pos..]
                        .iter()
                        .map(|&v| table.decl(v).name.as_str())
                        .collect();
                    names.push(table.decl(callee).name.as_str());
                    return Some((callee, names.join(" → ")));
                }
                0 => {
                    if let Some(found) = self.dfs_unguarded(callee, state, stack, table) {
                        return Some(found);
                    }
                }
                _ => {}
            }
        }
        stack.pop();
        state[node.0 as usize] = 2;
        None
    }

    /// The set of views transitively affected by a change to `changed`: the view
    /// itself plus every (transitive) **caller** of it (RFC-0007 §5). Used by
    /// reload to recompute the instances that must be re-derived.
    #[must_use]
    pub fn affected_by(&self, changed: ViewId) -> Vec<ViewId> {
        // Reverse reachability over the edge set.
        let n = self.edges.len();
        let mut affected = vec![false; n];
        affected[changed.0 as usize] = true;
        // Fixed-point: repeatedly mark any view that calls an affected view.
        let mut changed_any = true;
        while changed_any {
            changed_any = false;
            for (caller, callees) in self.edges.iter().enumerate() {
                if affected[caller] {
                    continue;
                }
                if callees.iter().any(|&(c, _)| affected[c.0 as usize]) {
                    affected[caller] = true;
                    changed_any = true;
                }
            }
        }
        (0..n)
            .filter(|&i| affected[i])
            .map(|i| ViewId(i as u32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn views(src: &str) -> Vec<ViewDecl> {
        let parsed = parse(src);
        assert!(
            parsed.errors.is_empty(),
            "parse errors: {:?}",
            parsed.errors
        );
        parsed.views
    }

    #[test]
    fn table_resolves_all_names_in_a_three_view_file() {
        let vs =
            views("View A() { Text(\"a\") }\nView B() { Text(\"b\") }\nView C() { Text(\"c\") }");
        let (table, errs) = ViewTable::build(&vs);
        assert!(errs.is_empty());
        assert_eq!(table.len(), 3);
        for name in ["A", "B", "C"] {
            assert!(table.contains(&Symbol::intern(name)), "missing {name}");
        }
    }

    #[test]
    fn intrinsic_named_view_is_shadowed() {
        let vs = views("View Column() { Text(\"x\") }");
        let (table, errs) = ViewTable::build(&vs);
        assert!(matches!(errs[0], CompileError::IntrinsicShadowed { .. }));
        assert!(!table.contains(&Symbol::intern("Column")));
    }

    #[test]
    fn call_graph_edges_and_affected_set() {
        // App → UserRow → Avatar (a leaf).
        let vs = views(
            "View Avatar() { Text(\"a\") }\n\
             View UserRow() { Avatar() }\n\
             View App() { UserRow() }",
        );
        let (table, _) = ViewTable::build(&vs);
        let g = CallGraph::build(&table);
        let avatar = table.resolve(&Symbol::intern("Avatar")).unwrap();
        // Editing Avatar affects Avatar, UserRow and App.
        let affected = g.affected_by(avatar);
        assert_eq!(affected.len(), 3);
    }

    #[test]
    fn unguarded_cycle_detected_guarded_is_legal() {
        let unguarded = views("View A() { A() }");
        let (t, _) = ViewTable::build(&unguarded);
        let g = CallGraph::build(&t);
        assert!(g.unguarded_cycle(&t).is_some());

        let guarded = views("View Tree() { when cond { Tree() } }");
        let (t2, _) = ViewTable::build(&guarded);
        let g2 = CallGraph::build(&t2);
        assert!(g2.unguarded_cycle(&t2).is_none());
    }
}
