//! Regression: a click must respect the button's bounds — clicking outside the
//! button does nothing; clicking inside it fires `count++` (RFC-0003 §4.2/E8,
//! topmost-wins, no ancestor bubbling).

use byard_compiler::interp::env::Value;
use byard_compiler::interp::eval::Interpreter;
use byard_compiler::interp::events::EventKind as CompKind;
use byard_compiler::parser::parse;
use byard_compiler::symbol::Symbol;
use byard_core::frame::RenderFrame;
use byard_core::platform::{EventKind, InputEvent};

const HELLO_WORLD: &str = include_str!("../examples/hello_world.byd");

fn down(pos: (f32, f32), t: u64) -> InputEvent {
    InputEvent {
        kind: EventKind::PointerDown,
        pos,
        delta: (0.0, 0.0),
        payload: None,
        time_ms: t,
    }
}

fn up(pos: (f32, f32), t: u64) -> InputEvent {
    InputEvent {
        kind: EventKind::PointerUp,
        pos,
        delta: (0.0, 0.0),
        payload: None,
        time_ms: t,
    }
}

#[test]
fn click_respects_button_bounds() {
    let parsed = parse(HELLO_WORLD);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let view = parsed.views[0].clone();

    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&view, &[]);
    interp.tick();

    // Render once so the event router registers the button's hit rect (the
    // engine dispatches tick T's input against tick T−1's geometry).
    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, 800.0, 600.0);

    let count = interp.var_signal(&Symbol::intern("count")).unwrap();
    assert_eq!(interp.peek(count), Value::Int(0));

    // The button's registered Tap hit rect.
    let rects = interp.router.handler_rects();
    let (_, _, br) = rects
        .iter()
        .copied()
        .find(|(_, k, _)| matches!(k, CompKind::Tap))
        .expect("the Button registered a Tap handler");

    // The hit rect must be a real sub-region, not the whole window.
    assert!(
        br.w < 800.0 || br.h < 600.0,
        "button hit rect should not cover the whole window: {br:?}"
    );

    // A click well outside the button (top-left corner) fires nothing.
    let outside = (2.0, 2.0);
    assert!(
        !contains(br, outside),
        "sanity: corner must be outside the button rect {br:?}"
    );
    interp.dispatch_events(&[down(outside, 0), up(outside, 50)]);
    interp.tick();
    assert_eq!(
        interp.peek(count),
        Value::Int(0),
        "a click outside the button must NOT increment (rect was {br:?})"
    );

    // A click at the button's center fires `count++`.
    let center = (br.x + br.w / 2.0, br.y + br.h / 2.0);
    interp.dispatch_events(&[down(center, 100), up(center, 150)]);
    interp.tick();
    assert_eq!(
        interp.peek(count),
        Value::Int(1),
        "a click inside the button must increment"
    );
}

fn contains(r: byard_compiler::interp::intrinsics::Rect, p: (f32, f32)) -> bool {
    p.0 >= r.x && p.0 <= r.x + r.w && p.1 >= r.y && p.1 <= r.y + r.h
}
