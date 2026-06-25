//! Reproduces the real runner's per-tick loop (dispatch → tick → render every
//! frame) to catch cross-frame gesture bugs that a render-once test misses.
//!
//! A `tap` is `PointerDown` on one frame and `PointerUp` on a later frame; the
//! engine renders (and re-registers hit rects) between them, so the router's
//! gesture state must survive a re-render — otherwise the button's `=> count++`
//! (a tap) never fires.

use byard_compiler::interp::env::Value;
use byard_compiler::interp::eval::{Interpreter, RenderNode};
use byard_compiler::interp::events::EventKind as CompKind;
use byard_compiler::parser::ast::ViewDecl;
use byard_compiler::parser::parse;
use byard_compiler::symbol::Symbol;
use byard_core::frame::RenderFrame;
use byard_core::{EventKind, InputEvent};

const HELLO_WORLD: &str = include_str!("../examples/hello_world.byd");

const W: f32 = 800.0;
const H: f32 = 600.0;

/// One frame of the runner loop: drain inputs, dispatch, tick, render.
fn frame(interp: &mut Interpreter, tree: &[RenderNode], inputs: &[InputEvent]) {
    interp.dispatch_events(inputs);
    interp.tick();
    let mut f = RenderFrame::new();
    interp.render(tree, &mut f, W, H);
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

fn build() -> (Interpreter, Vec<RenderNode>, ViewDecl) {
    let parsed = parse(HELLO_WORLD);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let view = parsed.views[0].clone();
    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&view, &[]);
    interp.tick();
    (interp, tree, view)
}

/// The tap's down and up arrive on **separate** frames (as in the real app),
/// with a render in between — the gesture must still be recognized.
#[test]
fn tap_across_frames_increments_count() {
    let (mut interp, tree, _view) = build();

    // Frame 0: register hit rects, no input.
    frame(&mut interp, &tree, &[]);

    let count = interp.var_signal(&Symbol::intern("count")).unwrap();
    let (_, _, br) = interp
        .router
        .handler_rects()
        .into_iter()
        .find(|(_, k, _)| matches!(k, CompKind::Tap))
        .expect("button registered a Tap handler");
    let center = (br.x + br.w / 2.0, br.y + br.h / 2.0);

    // Frame 1: press. Frame 2: release. (A render happens after each.)
    frame(
        &mut interp,
        &tree,
        &[pointer(EventKind::PointerDown, center, 100)],
    );
    frame(
        &mut interp,
        &tree,
        &[pointer(EventKind::PointerUp, center, 150)],
    );

    // The first Tap button in the demo mutates `count` by a fixed delta
    // (sign irrelevant); what matters is the cross-frame gesture is recognized.
    let delta = match interp.peek(count) {
        Value::Int(n) => n,
        other => panic!("count should be an Int, got {other:?}"),
    };
    assert_ne!(
        delta, 0,
        "a tap split across frames (with a re-render between) must fire the action"
    );

    // A second tap, also split across frames.
    // Gap from first UP (150ms) must be > DOUBLE_TAP_MS (300ms) so it's a
    // plain single tap, not a double-tap.
    frame(
        &mut interp,
        &tree,
        &[pointer(EventKind::PointerDown, center, 500)],
    );
    frame(
        &mut interp,
        &tree,
        &[pointer(EventKind::PointerUp, center, 550)],
    );
    assert_eq!(
        interp.peek(count),
        Value::Int(delta * 2),
        "the second cross-frame tap fires the action again"
    );
}

/// A press inside and release outside is not a tap (no increment), even across
/// frames.
#[test]
fn press_inside_release_outside_is_not_a_tap() {
    let (mut interp, tree, _view) = build();
    frame(&mut interp, &tree, &[]);
    let count = interp.var_signal(&Symbol::intern("count")).unwrap();
    let (_, _, br) = interp
        .router
        .handler_rects()
        .into_iter()
        .find(|(_, k, _)| matches!(k, CompKind::Tap))
        .unwrap();
    let center = (br.x + br.w / 2.0, br.y + br.h / 2.0);

    frame(
        &mut interp,
        &tree,
        &[pointer(EventKind::PointerDown, center, 100)],
    );
    frame(
        &mut interp,
        &tree,
        &[pointer(EventKind::PointerUp, (2.0, 2.0), 150)],
    );
    assert_eq!(interp.peek(count), Value::Int(0));
}
