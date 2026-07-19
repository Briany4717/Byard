//! Guards the committed RFC-0023 §2 backdrop-blur example through the real
//! `byard` binary: the `frosted_glass` project — `blur`/`backdrop_tint`/
//! `blur_saturation`/`blur_quality` across styles, animated glass via
//! `with anim.spring()` + `on hover`, and the tint-only cheap path — must
//! check clean (parse, type-check, lower, validate with no diagnostics).

use std::path::PathBuf;
use std::process::Command;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/frosted_glass")
}

/// `byard check <project-dir>` on the frosted-glass example reports no errors.
#[test]
fn frosted_glass_example_checks_clean() {
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
