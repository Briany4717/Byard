//! Guards the committed RFC-0019 callback-props example through the real
//! `byard` binary: the `callbacks` project — reusable interactive wrappers that
//! forward their inner intrinsic's `tap` to a caller-supplied `Fn()` action
//! block — must check clean (parse, type-check, lower, and validate the
//! callback bindings with no diagnostics).

use std::path::PathBuf;
use std::process::Command;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/callbacks")
}

/// `byard check <project-dir>` on the callbacks example reports no errors —
/// including the forwarding case (`on_tap: on_up`) where one wrapper passes a
/// callback it received down into another.
#[test]
fn callbacks_example_checks_clean() {
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
