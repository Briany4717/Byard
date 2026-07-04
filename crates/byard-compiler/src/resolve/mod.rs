//! The module resolver (RFC-0008 Pillars A & B).
//!
//! Turns a set of `.byd` files — the root project plus every package reachable
//! through `use` — into **one validated program**: a flat, deterministic
//! `Vec<ViewDecl>` with package-qualified canonical names, ready for
//! [`ViewTable`](crate::interp::views::ViewTable) so instantiation (RFC-0007)
//! is identical for single-file and multi-file programs.
//!
//! Design invariants:
//!
//! - **No I/O decisions in `byld`** (RFC-0001 §1, D-F): a `use` names a
//!   package; *where* that package lives is answered by the
//!   [`PackageProvider`] the CLI/LSP injects (manifest + lockfile + cache).
//! - **A package is one namespace**: every `View` declared across a package's
//!   files shares one flat export table (like a Dart library / Rust crate
//!   root). Views of the *root* project keep their bare names; views of a
//!   dependency get the canonical name `<package>.<View>`. Duplicates are a
//!   [`CompileError::DuplicateViewName`].
//! - **Explicit namespacing (D-G)**: `use pkg as m` → `m.Card`;
//!   `use pkg` → `pkg.Card`; `use pkg.{Card}` → bare `Card`, legal only while
//!   unambiguous — any collision is a [`CompileError::NameCollision`]
//!   demanding an alias. Resolution is order-independent.
//! - **Determinism**: files are processed in the order the provider returns
//!   them (sorted by the CLI), packages in first-`use` order, and the merged
//!   view list is root-first — so the same inputs always produce the same
//!   program and the same diagnostics.
//!
//! Spans: each file is parsed in isolation, then every span is rebased by the
//! file's byte offset in the program-wide [`SourceMap`], so any diagnostic —
//! parse-time or lower-time — can be located back to `file:line:col`.

mod rebase;

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::diagnostics::{CompileError, Span};
use crate::interp::intrinsics;
use crate::parser::ast::{ElementNode, Member, UseDecl, ViewDecl};
use crate::parser::parse;
use crate::symbol::Symbol;
use crate::util::closest_match;

/// The reserved name of the root package (the project being built). Displayed
/// as "this project" in diagnostics.
pub const ROOT_PACKAGE: &str = "";

/// One `.byd` source file handed to the resolver: a display name (for
/// diagnostics; typically a path relative to the project root) plus its text.
#[derive(Clone, Debug)]
pub struct SourceFile {
    /// Display name, e.g. `src/sidebar.byd` or `material/src/buttons.byd`.
    pub name: String,
    /// The file's source text.
    pub source: String,
}

/// Supplies package sources to the resolver. Implemented by the CLI (manifest,
/// lockfile, and cache) and the LSP; tests use an in-memory map. The resolver
/// itself never touches the filesystem — acquisition policy lives one layer up
/// (RFC-0008 Pillar C).
pub trait PackageProvider {
    /// Returns the `.byd` files of package `package`, requested by
    /// `dependent` ([`ROOT_PACKAGE`] for the project). Order must be
    /// deterministic (the CLI sorts by path). An `Err` explains why the
    /// package cannot be resolved (not declared, not fetched, …).
    fn package_files(&mut self, dependent: &str, package: &str) -> Result<Vec<SourceFile>, String>;
}

/// One file's slice of the program-wide source map.
#[derive(Clone, Debug)]
pub struct MapEntry {
    /// Display name of the file.
    pub file: String,
    /// Byte offset of this file's first byte in program-wide span space.
    pub base: u32,
    /// The file's source text.
    pub source: String,
}

/// Maps program-wide spans back to `(file, local span)` — the multi-file
/// equivalent of handing `CompileError::render` the single source string.
#[derive(Clone, Debug, Default)]
pub struct SourceMap {
    entries: Vec<MapEntry>,
}

impl SourceMap {
    /// Registers `source` under `file`, returning the base offset its spans
    /// were rebased by.
    fn add(&mut self, file: String, source: String) -> u32 {
        let base = self
            .entries
            .last()
            .map_or(0, |e| e.base + e.source.len() as u32 + 1);
        self.entries.push(MapEntry { file, base, source });
        base
    }

    /// The entry containing `span`, plus the span rebased back to file-local
    /// offsets. `None` for a span outside every file (e.g. the zero span).
    #[must_use]
    pub fn locate(&self, span: Span) -> Option<(&MapEntry, Span)> {
        let entry = self
            .entries
            .iter()
            .rev()
            .find(|e| span.start >= e.base && span.start <= e.base + e.source.len() as u32)?;
        Some((
            entry,
            Span::new(span.start - entry.base, span.end.saturating_sub(entry.base)),
        ))
    }

    /// Renders `err` as `file:line:col: error[kind]: headline` (the C7
    /// rustc-compatible shape `byard check` prints).
    #[must_use]
    pub fn render_line(&self, err: &CompileError) -> String {
        match self.locate(err.span()) {
            Some((entry, local)) => {
                let (line, col) = line_col(&entry.source, local.start as usize);
                format!(
                    "{}:{line}:{col}: error[{}]: {}",
                    entry.file,
                    err.kind(),
                    err.headline()
                )
            }
            None => format!("error[{}]: {}", err.kind(), err.headline()),
        }
    }

    /// Renders `err` with caret-anchored source context against its own file.
    #[must_use]
    pub fn render(&self, err: &CompileError) -> String {
        match self.locate(err.span()) {
            Some((entry, _)) => {
                let mut local = err.clone();
                local.shift_span(-i64::from(entry.base));
                format!("{}: {}", entry.file, local.render(&entry.source))
            }
            None => format!("error: {}", err.headline()),
        }
    }

    /// The registered files, in resolution order.
    pub fn files(&self) -> impl Iterator<Item = &MapEntry> + '_ {
        self.entries.iter()
    }
}

/// Converts a byte offset to 1-based (line, col) within `src`.
#[must_use]
pub fn line_col(src: &str, byte: usize) -> (usize, usize) {
    let safe = byte.min(src.len());
    let line = src[..safe].bytes().filter(|&b| b == b'\n').count() + 1;
    let line_start = src[..safe].rfind('\n').map_or(0, |i| i + 1);
    (line, safe - line_start + 1)
}

/// The resolver's output: one flat program plus everything needed to report
/// on it.
#[derive(Debug, Default)]
pub struct ResolvedProgram {
    /// Every view in the program, canonically named (`material.Card`), spans
    /// rebased into [`Self::source_map`] space. Root-project views come first,
    /// in file order, so `views.first()` is still the dev-runner's root.
    pub views: Vec<ViewDecl>,
    /// All diagnostics, in deterministic order.
    pub errors: Vec<CompileError>,
    /// Maps program-wide spans back to files.
    pub source_map: SourceMap,
    /// Loaded packages in load order; [`ROOT_PACKAGE`] first.
    pub packages: Vec<String>,
}

/// Human name of a package for diagnostics.
fn pkg_display(name: &str) -> String {
    if name == ROOT_PACKAGE {
        "this project".to_string()
    } else {
        format!("package `{name}`")
    }
}

/// One parsed file, spans already rebased.
struct ParsedModule {
    imports: Vec<UseDecl>,
    views: Vec<ViewDecl>,
}

/// A loaded package: its parsed files and flat export table.
#[derive(Default)]
struct PackageData {
    modules: Vec<ParsedModule>,
    /// Bare view name → span of its declaration.
    exports: HashMap<Symbol, Span>,
}

struct Resolver<'p> {
    provider: &'p mut dyn PackageProvider,
    map: SourceMap,
    errors: Vec<CompileError>,
    packages: HashMap<String, PackageData>,
    order: Vec<String>,
}

/// Resolves the whole program: `root_files` (entry first, siblings sorted by
/// the caller) plus every package reachable through `use`, recursively.
#[must_use]
pub fn resolve_program(
    root_files: Vec<SourceFile>,
    provider: &mut dyn PackageProvider,
) -> ResolvedProgram {
    let mut r = Resolver {
        provider,
        map: SourceMap::default(),
        errors: Vec::new(),
        packages: HashMap::new(),
        order: Vec::new(),
    };

    let mut stack = Vec::new();
    r.load_package(ROOT_PACKAGE, root_files, &mut stack);
    r.rewrite_all();

    // Merge: root first (file order), then packages in load order.
    let mut views = Vec::new();
    for name in &r.order {
        if let Some(pkg) = r.packages.get(name) {
            for module in &pkg.modules {
                views.extend(module.views.iter().cloned());
            }
        }
    }

    ResolvedProgram {
        views,
        errors: r.errors,
        source_map: r.map,
        packages: r.order,
    }
}

impl Resolver<'_> {
    /// Parses `files` into package `name`, registers its exports, and loads
    /// its dependencies depth-first. `stack` is the active dependency chain
    /// for cycle detection.
    fn load_package(&mut self, name: &str, files: Vec<SourceFile>, stack: &mut Vec<String>) {
        self.order.push(name.to_string());
        self.packages
            .insert(name.to_string(), PackageData::default());
        stack.push(name.to_string());

        let mut modules = Vec::new();
        let mut exports: HashMap<Symbol, Span> = HashMap::new();
        let mut dep_requests: Vec<(String, Span)> = Vec::new();

        for file in files {
            let parsed = parse(&file.source);
            let base = self.map.add(file.name, file.source);

            for mut err in parsed.errors {
                err.shift_span(i64::from(base));
                self.errors.push(err);
            }
            let mut imports = parsed.imports;
            for import in &mut imports {
                rebase::shift_use(import, base);
                dep_requests.push((import.package.as_str().to_string(), import.span));
            }
            let mut views = parsed.views;
            for view in &mut views {
                rebase::shift_view(view, base);
                match exports.entry(view.name.clone()) {
                    Entry::Vacant(v) => {
                        v.insert(view.span);
                    }
                    Entry::Occupied(_) => self.errors.push(CompileError::DuplicateViewName {
                        span: view.span,
                        name: view.name.as_str().to_string(),
                        package: pkg_display(name),
                    }),
                }
            }
            modules.push(ParsedModule { imports, views });
        }

        if let Some(pkg) = self.packages.get_mut(name) {
            pkg.modules = modules;
            pkg.exports = exports;
        }

        // Load dependencies depth-first, in first-`use` order (deterministic).
        for (dep, span) in dep_requests {
            if self.packages.contains_key(&dep) {
                // Already loaded — an edge back into the active chain is a cycle.
                if stack.contains(&dep) {
                    let mut path: Vec<&str> = stack
                        .iter()
                        .skip_while(|p| **p != dep)
                        .map(String::as_str)
                        .collect();
                    path.push(dep.as_str());
                    let path = path
                        .iter()
                        .map(|p| if p.is_empty() { "this project" } else { p })
                        .collect::<Vec<_>>()
                        .join(" → ");
                    self.errors.push(CompileError::PackageCycle { span, path });
                }
                continue;
            }
            match self.provider.package_files(name, &dep) {
                Ok(files) => self.load_package(&dep, files, stack),
                Err(detail) => self.errors.push(CompileError::UnknownPackage {
                    span,
                    name: dep,
                    detail,
                }),
            }
        }

        stack.pop();
    }

    /// Second pass: for every file, build its import maps, validate them, and
    /// rewrite element references + declared names to canonical form.
    fn rewrite_all(&mut self) {
        let pkg_names: Vec<String> = self.order.clone();
        for pkg_name in &pkg_names {
            let n_modules = self.packages.get(pkg_name).map_or(0, |p| p.modules.len());
            for module_idx in 0..n_modules {
                self.rewrite_module(pkg_name, module_idx);
            }
        }
    }

    fn rewrite_module(&mut self, pkg_name: &str, module_idx: usize) {
        // ---- build this file's import maps ----
        let imports: Vec<UseDecl> = self.packages[pkg_name].modules[module_idx].imports.clone();
        // alias → target package
        let mut aliases: HashMap<Symbol, String> = HashMap::new();
        // bare imported view name → target package
        let mut bare: HashMap<Symbol, String> = HashMap::new();
        let local_exports: Vec<(Symbol, Span)> = self.packages[pkg_name]
            .exports
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        for import in &imports {
            let target = import.package.as_str().to_string();
            if !self.packages.contains_key(&target) {
                continue; // already reported as UnknownPackage / cycle
            }
            if let Some(symbols) = &import.symbols {
                // Selective form: `use pkg.{A, B}` — binds bare names only.
                for (sym, span) in symbols {
                    self.check_selective_symbol(&target, sym, *span, &local_exports, &mut bare);
                }
            } else {
                let alias = import
                    .alias
                    .clone()
                    .unwrap_or_else(|| import.package.clone());
                match aliases.entry(alias.clone()) {
                    Entry::Vacant(v) => {
                        v.insert(target);
                    }
                    Entry::Occupied(o) => self.errors.push(CompileError::NameCollision {
                        span: import.span,
                        name: alias.as_str().to_string(),
                        first: pkg_display(o.get()),
                        second: pkg_display(&target),
                    }),
                }
            }
        }

        // ---- canonicalize declared names + rewrite bodies ----
        let local: HashMap<Symbol, Span> = self.packages[pkg_name].exports.clone();
        let mut views =
            std::mem::take(&mut self.packages.get_mut(pkg_name).unwrap().modules[module_idx].views);
        let mut errors = Vec::new();
        for view in &mut views {
            if pkg_name != ROOT_PACKAGE {
                view.name = Symbol::intern(&format!("{pkg_name}.{}", view.name.as_str()));
            }
            for member in &mut view.body {
                self.rewrite_member(member, pkg_name, &local, &aliases, &bare, &mut errors);
            }
        }
        self.packages.get_mut(pkg_name).unwrap().modules[module_idx].views = views;
        self.errors.extend(errors);
    }

    /// Validates one `use pkg.{sym}` entry and records the bare binding.
    fn check_selective_symbol(
        &mut self,
        target: &str,
        sym: &Symbol,
        span: Span,
        local_exports: &[(Symbol, Span)],
        bare: &mut HashMap<Symbol, String>,
    ) {
        let exports = &self.packages[target].exports;
        if !exports.contains_key(sym) {
            let hint =
                closest_match(sym.as_str(), exports.keys().map(Symbol::as_str)).map(str::to_string);
            self.errors.push(CompileError::UnknownImportSymbol {
                span,
                package: target.to_string(),
                name: sym.as_str().to_string(),
                hint,
            });
            // Still record the binding so downstream references resolve to
            // *something* and don't cascade extra UnknownView noise.
        }
        if intrinsics::lookup(sym.as_str()).is_some() {
            self.errors.push(CompileError::IntrinsicShadowed {
                span,
                name: sym.as_str().to_string(),
            });
            return; // the intrinsic always wins; don't bind
        }
        if local_exports.iter().any(|(name, _)| name == sym) {
            self.errors.push(CompileError::NameCollision {
                span,
                name: sym.as_str().to_string(),
                first: pkg_display(target),
                second: "a view declared here".to_string(),
            });
            return; // the local view wins deterministically
        }
        match bare.entry(sym.clone()) {
            Entry::Vacant(v) => {
                v.insert(target.to_string());
            }
            Entry::Occupied(o) => {
                if o.get() != target {
                    self.errors.push(CompileError::NameCollision {
                        span,
                        name: sym.as_str().to_string(),
                        first: pkg_display(o.get()),
                        second: pkg_display(target),
                    });
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rewrite_member(
        &self,
        member: &mut Member,
        pkg_name: &str,
        local: &HashMap<Symbol, Span>,
        aliases: &HashMap<Symbol, String>,
        bare: &HashMap<Symbol, String>,
        errors: &mut Vec<CompileError>,
    ) {
        match member {
            Member::Element(el) => {
                self.rewrite_element(el, pkg_name, local, aliases, bare, errors);
            }
            Member::For { body, .. } => {
                for m in body {
                    self.rewrite_member(m, pkg_name, local, aliases, bare, errors);
                }
            }
            Member::When { then, els, .. } => {
                for m in then {
                    self.rewrite_member(m, pkg_name, local, aliases, bare, errors);
                }
                if let Some(els) = els {
                    for m in els {
                        self.rewrite_member(m, pkg_name, local, aliases, bare, errors);
                    }
                }
            }
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rewrite_element(
        &self,
        el: &mut ElementNode,
        pkg_name: &str,
        local: &HashMap<Symbol, Span>,
        aliases: &HashMap<Symbol, String>,
        bare: &HashMap<Symbol, String>,
        errors: &mut Vec<CompileError>,
    ) {
        let name = el.name.as_str();
        if let Some((head, rest)) = name.split_once('.') {
            // Qualified reference `alias.View` (Pillar B).
            if let Some(target) = aliases.get(&Symbol::intern(head)) {
                if !self.packages[target]
                    .exports
                    .contains_key(&Symbol::intern(rest))
                {
                    let hint = closest_match(
                        rest,
                        self.packages[target].exports.keys().map(Symbol::as_str),
                    )
                    .map(str::to_string);
                    errors.push(CompileError::UnknownImportSymbol {
                        span: el.span,
                        package: target.clone(),
                        name: rest.to_string(),
                        hint,
                    });
                }
                el.name = Symbol::intern(&format!("{target}.{rest}"));
            } else {
                errors.push(CompileError::UnknownPackage {
                    span: el.span,
                    name: head.to_string(),
                    detail: format!("no `use {head}` in this file"),
                });
            }
        } else if intrinsics::lookup(name).is_some() {
            // The closed intrinsic catalog always wins (RFC-0005).
        } else if local.contains_key(&el.name) {
            if pkg_name != ROOT_PACKAGE {
                el.name = Symbol::intern(&format!("{pkg_name}.{name}"));
            }
        } else if let Some(target) = bare.get(&el.name) {
            el.name = Symbol::intern(&format!("{target}.{name}"));
        }
        // Anything else stays as written; ViewTable validation reports
        // UnknownView with hints over the canonical name set.

        for child in &mut el.children {
            self.rewrite_member(child, pkg_name, local, aliases, bare, errors);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::eval::Interpreter;
    use std::collections::BTreeMap;

    /// An in-memory provider: package name → files.
    struct MemProvider(BTreeMap<&'static str, Vec<(&'static str, &'static str)>>);

    impl PackageProvider for MemProvider {
        fn package_files(
            &mut self,
            _dependent: &str,
            package: &str,
        ) -> Result<Vec<SourceFile>, String> {
            self.0
                .get(package)
                .map(|files| {
                    files
                        .iter()
                        .map(|(name, src)| SourceFile {
                            name: (*name).to_string(),
                            source: (*src).to_string(),
                        })
                        .collect()
                })
                .ok_or_else(|| "not declared in `[dependencies]`".to_string())
        }
    }

    fn root(src: &str) -> Vec<SourceFile> {
        vec![SourceFile {
            name: "main.byd".to_string(),
            source: src.to_string(),
        }]
    }

    fn material() -> (&'static str, Vec<(&'static str, &'static str)>) {
        (
            "material",
            vec![
                ("material/buttons.byd", "View Card() { Text(\"card\") }"),
                (
                    "material/chips.byd",
                    "View Chip(label: Str = \"chip\") { Text(label) }",
                ),
            ],
        )
    }

    fn names(program: &ResolvedProgram) -> Vec<&str> {
        program.views.iter().map(|v| v.name.as_str()).collect()
    }

    /// The element names of a view's body, flattened.
    fn body_names(view: &ViewDecl) -> Vec<&str> {
        view.body
            .iter()
            .filter_map(|m| match m {
                Member::Element(el) => Some(el.name.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn alias_import_rewrites_to_canonical_names() {
        let mut p = MemProvider(BTreeMap::from([material()]));
        let program = resolve_program(root("use material as m\nView App() { m.Card() }"), &mut p);
        assert!(program.errors.is_empty(), "{:?}", program.errors);
        assert_eq!(names(&program), ["App", "material.Card", "material.Chip"]);
        assert_eq!(body_names(&program.views[0]), ["material.Card"]);
    }

    #[test]
    fn whole_package_import_uses_the_package_name_as_namespace() {
        let mut p = MemProvider(BTreeMap::from([material()]));
        let program = resolve_program(root("use material\nView App() { material.Chip() }"), &mut p);
        assert!(program.errors.is_empty(), "{:?}", program.errors);
        assert_eq!(body_names(&program.views[0]), ["material.Chip"]);
    }

    #[test]
    fn selective_import_binds_bare_names() {
        let mut p = MemProvider(BTreeMap::from([material()]));
        let program = resolve_program(
            root("use material.{Card, Chip}\nView App() { Card()\nChip() }"),
            &mut p,
        );
        assert!(program.errors.is_empty(), "{:?}", program.errors);
        assert_eq!(
            body_names(&program.views[0]),
            ["material.Card", "material.Chip"]
        );
    }

    #[test]
    fn unknown_package_is_reported_with_the_provider_detail() {
        let mut p = MemProvider(BTreeMap::new());
        let program = resolve_program(root("use nope\nView App() { Text(\"x\") }"), &mut p);
        assert!(matches!(
            &program.errors[0],
            CompileError::UnknownPackage { name, detail, .. }
                if name == "nope" && detail.contains("dependencies")
        ));
    }

    #[test]
    fn unknown_import_symbol_gets_a_hint() {
        let mut p = MemProvider(BTreeMap::from([material()]));
        let program = resolve_program(
            root("use material.{Cardd}\nView App() { Text(\"x\") }"),
            &mut p,
        );
        assert!(matches!(
            &program.errors[0],
            CompileError::UnknownImportSymbol { name, hint: Some(h), .. }
                if name == "Cardd" && h == "Card"
        ));
    }

    #[test]
    fn qualified_reference_to_a_missing_view_gets_a_hint() {
        let mut p = MemProvider(BTreeMap::from([material()]));
        let program = resolve_program(root("use material as m\nView App() { m.Cardz() }"), &mut p);
        assert!(matches!(
            &program.errors[0],
            CompileError::UnknownImportSymbol { package, hint: Some(h), .. }
                if package == "material" && h == "Card"
        ));
    }

    #[test]
    fn bare_name_collision_across_two_packages_demands_an_alias() {
        let mut p = MemProvider(BTreeMap::from([
            ("a", vec![("a/x.byd", "View Card() { Text(\"a\") }")]),
            ("b", vec![("b/x.byd", "View Card() { Text(\"b\") }")]),
        ]));
        let program = resolve_program(
            root("use a.{Card}\nuse b.{Card}\nView App() { Card() }"),
            &mut p,
        );
        assert!(
            program
                .errors
                .iter()
                .any(|e| matches!(e, CompileError::NameCollision { name, .. } if name == "Card")),
            "{:?}",
            program.errors
        );
        // Deterministic winner: the first binding (package `a`).
        assert_eq!(body_names(&program.views[0]), ["a.Card"]);
    }

    #[test]
    fn duplicate_alias_is_a_collision() {
        let mut p = MemProvider(BTreeMap::from([
            ("a", vec![("a/x.byd", "View X() { Text(\"a\") }")]),
            ("b", vec![("b/x.byd", "View Y() { Text(\"b\") }")]),
        ]));
        let program = resolve_program(root("use a as m\nuse b as m\nView App() { m.X() }"), &mut p);
        assert!(
            program
                .errors
                .iter()
                .any(|e| matches!(e, CompileError::NameCollision { name, .. } if name == "m"))
        );
    }

    #[test]
    fn selective_import_shadowing_an_intrinsic_is_rejected() {
        let mut p = MemProvider(BTreeMap::from([(
            "a",
            vec![("a/x.byd", "View Button() { Text(\"a\") }")],
        )]));
        let program = resolve_program(
            root("use a.{Button}\nView App() { Button(\"ok\") }"),
            &mut p,
        );
        assert!(program.errors.iter().any(
            |e| matches!(e, CompileError::IntrinsicShadowed { name, .. } if name == "Button")
        ));
        // The intrinsic wins: the reference stays bare.
        assert_eq!(body_names(&program.views[0]), ["Button"]);
    }

    #[test]
    fn qualified_access_reaches_an_intrinsic_named_package_view() {
        // `Button` as a *package* view is unreachable bare, but reachable
        // qualified — namespacing makes it unambiguous.
        let mut p = MemProvider(BTreeMap::from([(
            "a",
            vec![("a/x.byd", "View Button() { Text(\"a\") }")],
        )]));
        let program = resolve_program(root("use a\nView App() { a.Button() }"), &mut p);
        assert!(program.errors.is_empty(), "{:?}", program.errors);
        assert_eq!(body_names(&program.views[0]), ["a.Button"]);
    }

    #[test]
    fn package_cycle_is_detected() {
        let mut p = MemProvider(BTreeMap::from([
            ("a", vec![("a/x.byd", "use b\nView X() { Text(\"a\") }")]),
            ("b", vec![("b/x.byd", "use a\nView Y() { Text(\"b\") }")]),
        ]));
        let program = resolve_program(root("use a\nView App() { a.X() }"), &mut p);
        assert!(
            program
                .errors
                .iter()
                .any(|e| matches!(e, CompileError::PackageCycle { path, .. } if path.contains("a → b → a"))),
            "{:?}",
            program.errors
        );
    }

    #[test]
    fn duplicate_view_across_package_files_is_reported() {
        let mut p = MemProvider(BTreeMap::from([(
            "a",
            vec![
                ("a/one.byd", "View X() { Text(\"1\") }"),
                ("a/two.byd", "View X() { Text(\"2\") }"),
            ],
        )]));
        let program = resolve_program(root("use a\nView App() { a.X() }"), &mut p);
        assert!(
            program
                .errors
                .iter()
                .any(|e| matches!(e, CompileError::DuplicateViewName { name, .. } if name == "X"))
        );
    }

    #[test]
    fn root_project_is_one_namespace_across_files() {
        // Two root files; sibling views resolve with no `use` at all.
        let mut p = MemProvider(BTreeMap::new());
        let program = resolve_program(
            vec![
                SourceFile {
                    name: "main.byd".into(),
                    source: "View App() { Header() }".into(),
                },
                SourceFile {
                    name: "header.byd".into(),
                    source: "View Header() { Text(\"h\") }".into(),
                },
            ],
            &mut p,
        );
        assert!(program.errors.is_empty(), "{:?}", program.errors);
        assert_eq!(names(&program), ["App", "Header"]);
    }

    #[test]
    fn transitive_dependencies_resolve() {
        let mut p = MemProvider(BTreeMap::from([
            (
                "material",
                vec![(
                    "material/card.byd",
                    "use icons\nView Card() { icons.Star() }",
                )],
            ),
            (
                "icons",
                vec![("icons/star.byd", "View Star() { Text(\"*\") }")],
            ),
        ]));
        let program = resolve_program(root("use material as m\nView App() { m.Card() }"), &mut p);
        assert!(program.errors.is_empty(), "{:?}", program.errors);
        assert_eq!(names(&program), ["App", "material.Card", "icons.Star"]);
        // The package-internal reference was rewritten too.
        assert_eq!(body_names(&program.views[1]), ["icons.Star"]);
    }

    #[test]
    fn source_map_locates_errors_in_the_right_file() {
        let mut p = MemProvider(BTreeMap::from([(
            "a",
            vec![("a/x.byd", "View X() { Text(\"ok\") }\nView Y() { Bogus() }")],
        )]));
        let program = resolve_program(root("use a\nView App() { a.X() }"), &mut p);
        // Lower the program so UnknownView for `Bogus` fires with a rebased span.
        let mut interp = Interpreter::new();
        interp.load_views(&program.views);
        let known: Vec<&str> = program.views.iter().map(|v| v.name.as_str()).collect();
        for view in &program.views {
            let _ = interp.lower_view(view, &known);
        }
        let unknown = interp
            .errors()
            .iter()
            .find(|e| matches!(e, CompileError::UnknownView { name, .. } if name == "Bogus"))
            .expect("Bogus must be reported");
        let (entry, local) = program.source_map.locate(unknown.span()).unwrap();
        assert_eq!(entry.file, "a/x.byd");
        let (line, _col) = line_col(&entry.source, local.start as usize);
        assert_eq!(line, 2, "Bogus is on line 2 of a/x.byd");
    }

    #[test]
    fn resolved_program_lowers_end_to_end() {
        let mut p = MemProvider(BTreeMap::from([material()]));
        let program = resolve_program(
            root("use material.{Chip}\nView App() { Column #[gap: 8] { Chip()\nChip(\"two\") } }"),
            &mut p,
        );
        assert!(program.errors.is_empty(), "{:?}", program.errors);
        let mut interp = Interpreter::new();
        interp.load_views(&program.views);
        let known: Vec<&str> = program.views.iter().map(|v| v.name.as_str()).collect();
        let tree = interp.lower_view(&program.views[0], &known);
        assert!(
            interp.errors().is_empty(),
            "lowering must be clean: {:?}",
            interp.errors()
        );
        assert!(!tree.is_empty());
    }

    #[test]
    fn import_after_view_is_reported() {
        let parsed = parse("View A() { Text(\"x\") }\nuse material");
        assert!(
            parsed
                .errors
                .iter()
                .any(|e| matches!(e, CompileError::ImportAfterView { .. }))
        );
        // The decl is still recorded for resolver recovery.
        assert_eq!(parsed.imports.len(), 1);
    }
}
