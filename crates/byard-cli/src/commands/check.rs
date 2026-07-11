//! `byard check [file]` — parse and validate without opening a window
//! (RFC-0006 §5, decision C7; RFC-0008: checks the *whole module graph*,
//! project siblings and packages included).

use crate::deps::resolve_project;
use crate::manifest::Manifest;
use byard_compiler::CompileError;
use byard_compiler::interp::eval::Interpreter;
use byard_compiler::resolve::{ResolvedProgram, SourceMap};
use std::path::Path;

pub fn run(file: Option<&Path>) -> Result<(), String> {
    let manifest = Manifest::discover(file)?;

    println!("Checking {}…", manifest.entry.display());

    let (program, _provider) = resolve_project(&manifest)?;
    let n_files = program.source_map.files().count();
    let n_pkgs = program.packages.len().saturating_sub(1);
    if n_files > 1 || n_pkgs > 0 {
        println!("  {n_files} file(s), {n_pkgs} package(s)");
    }

    let errors = check_program_with_theme(&program, manifest.theme);

    if errors.is_empty() {
        println!("  ok (0 errors)");
        Ok(())
    } else {
        for err in &errors {
            // rustc-compatible format: file:line:col: error[kind]: message (C7),
            // located through the program-wide source map (RFC-0008).
            eprintln!("{}", program.source_map.render_line(err));
        }
        let n = errors.len();
        eprintln!("{n} error{}.", if n == 1 { "" } else { "s" });
        // Signal failure to the caller (main.rs maps Err → exit 1).
        Err(String::new())
    }
}

/// Headless semantic validation of a resolved program (no `wgpu`/`winit`).
///
/// Resolve/parse errors short-circuit; otherwise every `View` is lowered and
/// rendered into a throwaway [`RenderFrame`] so attribute-contract and
/// `Len`-form validation checks — which run during lowering and render — are
/// exercised across the whole module graph.
#[must_use]
pub fn check_program(program: &ResolvedProgram) -> Vec<CompileError> {
    check_program_with_theme(program, byard_compiler::interp::theme::Theme::byard_base())
}

/// Like [`check_program`], but validates against a specific design-token theme
/// (RFC-0022) so `inject Theme as t` resolves and `t.token` references are
/// checked against the project's *actual* declared tokens — a custom manifest
/// token must not be flagged `UnknownThemeToken`.
#[must_use]
pub fn check_program_with_theme(
    program: &ResolvedProgram,
    theme: byard_compiler::interp::theme::Theme,
) -> Vec<CompileError> {
    if !program.errors.is_empty() {
        return program.errors.clone();
    }

    let known: Vec<&str> = program.views.iter().map(|v| v.name.as_str()).collect();
    let mut interp = Interpreter::new();
    interp.set_theme(theme);
    // Build the user-`View` registry once for the whole program so user-view
    // calls resolve and expand during lowering (RFC-0007 §1).
    interp.load_views(&program.views);
    let mut frame = byard_core::frame::RenderFrame::new();
    for view in &program.views {
        let tree = interp.lower_view(view, &known);
        interp.tick();
        interp.render(&tree, &mut frame, 1024.0, 768.0);
        frame.clear();
    }
    interp.errors().to_vec()
}

/// Headless parse + semantic validation of one `.byd` source — the
/// single-file path (bare `byard check file.byd`, unit tests). Any `use` in a
/// bare file is an `UnknownPackage` error: dependencies need a manifest.
// Exercised by the unit tests below; the `run` entry point always goes through
// the manifest/module-graph path, so the bin build sees this as unused.
#[allow(dead_code)]
#[must_use]
pub fn check_source(src: &str) -> Vec<CompileError> {
    struct NoPackages;
    impl byard_compiler::resolve::PackageProvider for NoPackages {
        fn package_files(
            &mut self,
            _dependent: &str,
            _package: &str,
        ) -> Result<Vec<byard_compiler::resolve::SourceFile>, String> {
            Err("a bare `.byd` file has no `[dependencies]`; create a byard.toml".to_string())
        }
    }
    let program = byard_compiler::resolve::resolve_program(
        vec![byard_compiler::resolve::SourceFile {
            name: "main.byd".to_string(),
            source: src.to_string(),
        }],
        &mut NoPackages,
    );
    check_program(&program)
}

/// Pretty-prints all errors with caret-anchored source context.
#[allow(dead_code)]
pub fn print_verbose(errors: &[CompileError], map: &SourceMap) {
    for err in errors {
        eprintln!("{}", map.render(err));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_source_has_no_errors() {
        let errs =
            check_source("View Main() { Column #[gap: 8, p: (horizontal: 8)] { Text(\"hi\") } }");
        assert!(errs.is_empty(), "{errs:?}");
    }

    #[test]
    fn bad_attr_is_reported() {
        let errs = check_source("View Main() { Column #[bogus: 1] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::UnknownAttribute { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn removed_px_attr_is_reported() {
        // px/py are no longer accepted attributes.
        let errs = check_source("View Main() { Column #[px: 4] {} }");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                CompileError::UnknownAttribute { name, .. } if name == "px"
            )),
            "{errs:?}"
        );
    }

    #[test]
    fn conflicting_spacing_is_reported() {
        let errs = check_source("View Main() { Column #[p: (horizontal: 4, left: 2)] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::ConflictingSpacingField { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn parse_error_short_circuits() {
        let errs = check_source("View Main() { Column #[gap: ");
        assert!(!errs.is_empty());
    }

    #[test]
    fn use_in_a_bare_file_explains_it_needs_a_manifest() {
        let errs = check_source("use material\nView Main() { Text(\"x\") }");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                CompileError::UnknownPackage { detail, .. } if detail.contains("byard.toml")
            )),
            "{errs:?}"
        );
    }
}
