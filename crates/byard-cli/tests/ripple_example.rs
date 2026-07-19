//! Guards the committed RFC-0023 ripple example through the real `byard`
//! binary: the `ripple` project — `ripple`/`ripple_active`/`ripple_radius`/
//! `ripple_duration` props across styles with `on pressed` triggers — must
//! check clean (parse, type-check, lower, validate with no diagnostics).

use std::path::PathBuf;
use std::process::Command;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/ripple")
}

/// `byard check <project-dir>` on the ripple example reports no errors.
#[test]
fn ripple_example_checks_clean() {
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
