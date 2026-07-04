//! Guards the committed RFC-0008 example workspace end-to-end through the real
//! `byard` binary: a project that `use`s a `path` package and a sibling file
//! must resolve into one module graph and check clean.

use std::path::PathBuf;
use std::process::Command;

fn example(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/workspace")
        .join(sub)
}

/// `byard check <project-dir>` resolves the whole graph (project files +
/// the `kit` path package) and reports no errors.
#[test]
fn workspace_example_checks_clean() {
    let out = Command::new(env!("CARGO_BIN_EXE_byard"))
        .arg("check")
        .arg(example("app"))
        .output()
        .expect("run `byard check`");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "check failed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("1 package(s)") && stdout.contains("ok (0 errors)"),
        "expected a 1-package clean check, got:\n{stdout}"
    );
}

/// Pointing at the manifest resolves the same way as pointing at the directory.
#[test]
fn checking_the_manifest_path_is_equivalent() {
    let out = Command::new(env!("CARGO_BIN_EXE_byard"))
        .arg("check")
        .arg(example("app/byard.toml"))
        .output()
        .expect("run `byard check`");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("ok (0 errors)"));
}
