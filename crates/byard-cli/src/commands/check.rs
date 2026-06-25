//! `byard check [file]` — parse and validate without opening a window
//! (RFC-0006 §5, decision C7).

use crate::manifest::Manifest;
use byard_compiler::CompileError;
use byard_compiler::interp::eval::Interpreter;
use std::path::Path;

pub fn run(file: Option<&Path>) -> Result<(), String> {
    let manifest = Manifest::discover(file)?;

    let src = std::fs::read_to_string(&manifest.entry)
        .map_err(|e| format!("{}: {e}", manifest.entry.display()))?;

    let entry_display = manifest.entry.display().to_string();
    println!("Checking {entry_display}…");

    let errors = check_source(&src);

    if errors.is_empty() {
        println!("  ok (0 errors)");
        Ok(())
    } else {
        for err in &errors {
            // rustc-compatible format: file:line:col: error[kind]: message (C7).
            let (line, col) = byte_to_line_col(&src, err.span().start as usize);
            eprintln!(
                "{entry_display}:{line}:{col}: error[{}]: {}",
                err.kind(),
                err.headline(),
            );
        }
        let n = errors.len();
        eprintln!("{n} error{}.", if n == 1 { "" } else { "s" });
        // Signal failure to the caller (main.rs maps Err → exit 1).
        Err(String::new())
    }
}

/// Headless parse + semantic validation of one `.byd` source (no `wgpu`/`winit`).
///
/// Parse/lex errors short-circuit; otherwise every `View` is lowered and rendered
/// into a throwaway [`RenderFrame`] so attribute-contract and `Len`-form
/// validation checks — which run during lowering and render — are exercised.
#[must_use]
pub fn check_source(src: &str) -> Vec<CompileError> {
    let parsed = byard_compiler::parser::parse(src);
    if !parsed.errors.is_empty() {
        return parsed.errors;
    }

    let known: Vec<&str> = parsed.views.iter().map(|v| v.name.as_str()).collect();
    let mut interp = Interpreter::new();
    let mut frame = byard_core::frame::RenderFrame::new();
    for view in &parsed.views {
        let tree = interp.lower_view(view, &known);
        interp.tick();
        interp.render(&tree, &mut frame, 1024.0, 768.0);
        frame.clear();
    }
    interp.errors().to_vec()
}

/// Converts a byte offset to 1-based (line, col).
fn byte_to_line_col(src: &str, byte: usize) -> (usize, usize) {
    let safe = byte.min(src.len());
    let line = src[..safe].bytes().filter(|&b| b == b'\n').count() + 1;
    let line_start = src[..safe].rfind('\n').map_or(0, |i| i + 1);
    let col = safe - line_start + 1;
    (line, col)
}

/// Pretty-prints all errors with source context (uses `CompileError::render`).
#[allow(dead_code)]
pub fn print_verbose(errors: &[CompileError], src: &str, path: &str) {
    for err in errors {
        eprintln!("{path}: {}", err.render(src));
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
}
