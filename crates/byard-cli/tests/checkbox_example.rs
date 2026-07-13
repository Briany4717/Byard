//! Guards the committed RFC-0018 `Checkbox` example through the real `byard`
//! binary: the `checkbox` project — a mixed-state sample plus three `bind:`ed
//! boolean controls and a reactive `when` status line — must check clean
//! (parse, type-check, lower, and validate the intrinsic contract with no
//! diagnostics).

use std::path::PathBuf;
use std::process::Command;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/checkbox")
}

/// `byard check <project-dir>` on the checkbox example reports no errors.
#[test]
fn checkbox_example_checks_clean() {
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
