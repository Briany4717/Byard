//! Guards the committed RFC-0021 collapsing-header example through the real
//! `byard` binary: the `collapse_header` project — a `collapse_header: true`
//! `ScrollView` whose header reads the implicit `scroll_fraction` binding — must
//! check clean (parse, type-check, lower, validate with no diagnostics).

use std::path::PathBuf;
use std::process::Command;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/collapse_header")
}

/// `byard check <project-dir>` on the collapse-header example reports no errors.
#[test]
fn collapse_header_example_checks_clean() {
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
