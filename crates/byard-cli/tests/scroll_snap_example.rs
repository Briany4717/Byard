//! Guards the committed RFC-0021 advanced-scroll example through the real
//! `byard` binary: the `scroll_snap` project — a `snap: page` carousel with a
//! reflected `page:` var and an infinite-scroll list with `end_reached` — must
//! check clean (parse, type-check, lower, validate with no diagnostics).

use std::path::PathBuf;
use std::process::Command;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/scroll_snap")
}

/// `byard check <project-dir>` on the scroll-snap example reports no errors.
#[test]
fn scroll_snap_example_checks_clean() {
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
