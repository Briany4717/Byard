//! Guards the committed RFC-0018 `RadioButton` example through the real `byard`
//! binary: the `radio_button` project — a three-option group sharing one
//! `bind:` var, with a reactive "Selected: …" line — must check clean (parse,
//! type-check, lower, and validate the intrinsic contract with no diagnostics).

use std::path::PathBuf;
use std::process::Command;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/radio_button")
}

/// `byard check <project-dir>` on the radio-button example reports no errors.
#[test]
fn radio_button_example_checks_clean() {
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
