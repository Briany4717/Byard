//! Guards the committed RFC-0027 todo example: the whole app is pure `byld`
//! (no `.rs`), so it must (1) `byard check` clean and (2) actually run — add 3
//! tasks, toggle 1, remove 1 — driven headlessly through the real interpreter,
//! proving `push`/`filter`/`map` + records reach the reactive tree end to end.

use std::path::PathBuf;
use std::process::Command;

use byard_compiler::interp::env::Value;
use byard_compiler::interp::eval::{Interpreter, RenderNode};
use byard_compiler::interp::events::EventKind as CompKind;
use byard_compiler::parser::parse;
use byard_compiler::symbol::Symbol;
use byard_core::frame::RenderFrame;
use byard_core::{EventKind, InputEvent};

const TODO: &str = include_str!("../examples/todo/src/main.byd");
const W: f32 = 800.0;
const H: f32 = 900.0;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/todo")
}

#[test]
fn todo_example_checks_clean() {
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

fn pointer(kind: EventKind, pos: (f32, f32), t: u64) -> InputEvent {
    InputEvent {
        kind,
        pos,
        delta: (0.0, 0.0),
        payload: None,
        time_ms: t,
    }
}

/// One runner frame: dispatch inputs, tick, render (re-registering hit rects).
fn frame(interp: &mut Interpreter, tree: &[RenderNode], inputs: &[InputEvent]) {
    interp.dispatch_events(inputs);
    interp.tick();
    let mut f = RenderFrame::new();
    interp.render(tree, &mut f, W, H);
}

/// A tap = down on one frame, up on the next (as in the real app).
fn tap(interp: &mut Interpreter, tree: &[RenderNode], center: (f32, f32), t: u64) {
    frame(interp, tree, &[pointer(EventKind::PointerDown, center, t)]);
    frame(
        interp,
        tree,
        &[pointer(EventKind::PointerUp, center, t + 30)],
    );
}

/// All live `Tap` handler rects (center points), in registration = document
/// order.
fn tap_centers(interp: &Interpreter) -> Vec<(f32, f32)> {
    interp
        .router
        .handler_rects()
        .into_iter()
        .filter(|(_, k, _)| matches!(k, CompKind::Tap))
        .map(|(_, _, r)| (r.x + r.w / 2.0, r.y + r.h / 2.0))
        .collect()
}

fn todos_len(interp: &Interpreter, sig: byard_compiler::interp::env::SignalId) -> usize {
    match interp.peek(sig) {
        Value::List(xs) => xs.len(),
        other => panic!("todos should be a List, got {other:?}"),
    }
}

#[test]
fn todo_add_toggle_remove_drives_the_reactive_list() {
    let parsed = parse(TODO);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let view = parsed.views[0].clone();
    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&view, &[]);
    interp.tick();

    // Frame 0: register hit rects (list empty ⇒ the only Tap handler is "Add").
    frame(&mut interp, &tree, &[]);
    let todos = interp.var_signal(&Symbol::intern("todos")).unwrap();
    let draft = interp.var_signal(&Symbol::intern("draft")).unwrap();

    let add = tap_centers(&interp);
    assert_eq!(add.len(), 1, "empty list ⇒ only the Add button is tappable");
    let add_center = add[0];

    // Add three tasks. `draft` is written directly (as a TextField would), then
    // the Add button appends `{ id, text, done: false }` and clears the draft.
    // Taps are spaced > DOUBLE_TAP_MS (300ms) apart so each is a distinct
    // single tap, not a double-tap.
    let mut t = 100;
    for label in ["a", "b", "c"] {
        interp.write_var(draft, Value::Str(label.to_string()));
        tap(&mut interp, &tree, add_center, t);
        t += 1000;
    }
    assert_eq!(todos_len(&interp, todos), 3, "three pushes grew the list");
    // The derived `remaining` memo (a `let`) is observable through its readout.
    assert!(
        count_readout(&mut interp, &tree).contains("3 of 3 left"),
        "3 tasks, none done"
    );

    // The tap handlers are now: [Add, toggle0, remove0, toggle1, …]. The Add
    // button sits at the top (smallest y); the first row's two buttons are the
    // next-lowest y, toggle left of remove.
    let handlers: Vec<(u32, CompKind, byard_compiler::interp::intrinsics::Rect)> = interp
        .router
        .handler_rects()
        .into_iter()
        .filter(|(_, k, _)| matches!(k, CompKind::Tap))
        .collect();
    assert_eq!(handlers.len(), 7, "Add + three rows × (toggle, remove)");

    // Row buttons = everyone below the Add button, sorted by (y, x).
    let mut rows: Vec<byard_compiler::interp::intrinsics::Rect> = handlers
        .iter()
        .map(|(_, _, r)| *r)
        .filter(|r| r.y > add_center.1 + 1.0)
        .collect();
    rows.sort_by(|a, b| {
        a.y.partial_cmp(&b.y)
            .unwrap()
            .then(a.x.partial_cmp(&b.x).unwrap())
    });
    let toggle0 = (rows[0].x + rows[0].w / 2.0, rows[0].y + rows[0].h / 2.0);
    let remove0 = (rows[1].x + rows[1].w / 2.0, rows[1].y + rows[1].h / 2.0);

    // Toggle the first task done → remaining drops to 2 (map flips one record).
    tap(&mut interp, &tree, toggle0, t);
    t += 1000;
    assert_eq!(todos_len(&interp, todos), 3, "toggle does not remove");
    assert!(
        count_readout(&mut interp, &tree).contains("2 of 3 left"),
        "toggling one task done leaves 2 remaining, got {:?}",
        count_readout(&mut interp, &tree)
    );

    // Remove the first task → list shrinks to 2 (filter drops one record).
    tap(&mut interp, &tree, remove0, t);
    assert_eq!(todos_len(&interp, todos), 2, "removing one leaves 2 tasks");
}

/// Renders and returns the "N of M left · K done" readout line (the second
/// `Text`), so the `let remaining`/`let done` memos can be observed.
fn count_readout(interp: &mut Interpreter, tree: &[RenderNode]) -> String {
    let mut f = RenderFrame::new();
    interp.render(tree, &mut f, W, H);
    f.texts()
        .iter()
        .map(|t| t.text.clone())
        .find(|s| s.contains(" left"))
        .unwrap_or_default()
}
