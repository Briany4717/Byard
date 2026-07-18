//! Verifies the RFC-0027 live data-ops showcase: it must `byard check` clean and
//! render the *correct* derived values (comparison, logic, concat, list ops,
//! records) — the same values a human sees running `-- dev`. Also taps a control
//! to prove the readouts recompute reactively.

use std::path::PathBuf;
use std::process::Command;

use byard_compiler::interp::eval::{Interpreter, RenderNode};
use byard_compiler::interp::events::EventKind as CompKind;
use byard_compiler::parser::parse;
use byard_core::frame::RenderFrame;
use byard_core::{EventKind, InputEvent};

const SRC: &str = include_str!("../examples/data_ops/src/main.byd");
const W: f32 = 900.0;
const H: f32 = 1000.0;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/data_ops")
}

#[test]
fn data_ops_example_checks_clean() {
    let out = Command::new(env!("CARGO_BIN_EXE_byard"))
        .arg("check")
        .arg(example_dir())
        .output()
        .expect("run `byard check`");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "check failed:\n{stdout}\n{stderr}");
    assert!(stdout.contains("ok (0 errors)"), "got:\n{stdout}");
}

fn texts(interp: &mut Interpreter, tree: &[RenderNode]) -> Vec<String> {
    let mut f = RenderFrame::new();
    interp.render(tree, &mut f, W, H);
    f.texts().iter().map(|t| t.text.clone()).collect()
}

/// A rendered line must exist that both starts with `label` and ends with
/// `value` — i.e. the `LABEL → {derived}` readout resolved to `value`.
fn has_readout(lines: &[String], label: &str, value: &str) {
    assert!(
        lines
            .iter()
            .any(|l| l.contains(label) && l.trim_end().ends_with(value)),
        "expected a `{label} … {value}` readout, got:\n{lines:#?}"
    );
}

#[test]
fn derived_readouts_are_correct_and_reactive() {
    let parsed = parse(SRC);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let view = parsed.views[0].clone();
    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&view, &[]);
    interp.tick();

    // Initial state: n = 3, xs = [1, 2, 3], user = { name: "Ada", score: 0 }.
    let lines = texts(&mut interp, &tree);

    // Comparison → Bool.
    has_readout(&lines, "n > 5", "false");
    has_readout(&lines, "n == 3", "true");
    has_readout(&lines, "n <= 3", "true");
    // Logic (short-circuit).
    has_readout(&lines, "0 < n < 10", "true");
    has_readout(&lines, "!(n > 5)", "true");
    has_readout(&lines, "n<0 || n>0", "true");
    // String concat + scalar coercion.
    has_readout(&lines, "\"n = \" + n", "n = 3");
    has_readout(&lines, "count message", "you have 3 items");
    // List ops.
    has_readout(&lines, "xs.len", "3");
    has_readout(&lines, "xs[0]", "1");
    has_readout(&lines, "xs.contains(n)", "true");
    has_readout(&lines, "filter(x>2).len", "1");
    // Records.
    has_readout(&lines, "Ada", "score 0");
    // The `map(x => x*2)` list renders 2, 4, 6.
    for want in ["• 2", "• 4", "• 6"] {
        assert!(
            lines.iter().any(|l| l == want),
            "missing mapped item {want}"
        );
    }

    // ── prove reactivity: tap "n +" three times → n = 6, flipping `n > 5`. ──
    // The two top-row buttons are "n −" (left) and "n +" (right); pick the
    // right-most Tap handler in the top band as "n +".
    let np = interp
        .router
        .handler_rects()
        .into_iter()
        .filter(|(_, k, r)| matches!(k, CompKind::Tap) && r.y < 200.0)
        .max_by(|a, b| a.2.x.partial_cmp(&b.2.x).unwrap())
        .map(|(_, _, r)| (r.x + r.w / 2.0, r.y + r.h / 2.0))
        .expect("an `n +` button");

    let mut t = 100;
    for _ in 0..3 {
        interp.dispatch_events(&[InputEvent {
            kind: EventKind::PointerDown,
            pos: np,
            delta: (0.0, 0.0),
            payload: None,
            time_ms: t,
        }]);
        interp.tick();
        let _ = texts(&mut interp, &tree);
        interp.dispatch_events(&[InputEvent {
            kind: EventKind::PointerUp,
            pos: np,
            delta: (0.0, 0.0),
            payload: None,
            time_ms: t + 30,
        }]);
        interp.tick();
        let _ = texts(&mut interp, &tree);
        t += 1000;
    }

    // n is now 6, so `n > 5` flips to true and the concat updates live.
    let lines = texts(&mut interp, &tree);
    has_readout(&lines, "n > 5", "true");
    has_readout(&lines, "\"n = \" + n", "n = 6");
    assert!(
        lines.iter().any(|l| l == "n = 6"),
        "the `n = {{n}}` readout updated to 6"
    );
}
