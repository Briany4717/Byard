//! Guards the committed RFC-0018 `Grid` example through the real `byard` binary:
//! the `grid` project — a 3-column dashboard with `col_span`/`row` placement —
//! must check clean (parse, type-check, lower, validate the intrinsic contract
//! and the grid templates with no diagnostics).

use std::path::PathBuf;
use std::process::Command;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/grid")
}

/// `byard check <project-dir>` on the grid example reports no errors.
#[test]
fn grid_example_checks_clean() {
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
