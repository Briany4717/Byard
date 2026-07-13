//! Guards the committed RFC-0005 default-text-wrap example through the real
//! `byard` binary: the `text_wrap` project — long paragraphs that wrap to their
//! parent's width with no explicit `width`, plus a `wrap: false` opt-out — must
//! check clean (parse, type-check, lower, validate with no diagnostics).

use std::path::PathBuf;
use std::process::Command;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/text_wrap")
}

/// `byard check <project-dir>` on the text-wrap example reports no errors.
#[test]
fn text_wrap_example_checks_clean() {
    let out = Command::new(env!("CARGO_BIN_EXE_byard"))
        .arg("check")
        .arg(example_dir())
        .output()
        .expect("run `byard check`");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "check failed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("ok (0 errors)"),
        "expected a clean check, got:\n{stdout}"
    );
}
