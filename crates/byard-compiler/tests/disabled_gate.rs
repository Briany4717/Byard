//! Regression: a `disabled:` element still lays out and registers a hit rect,
//! but the router gates its handlers so a tap fires nothing (RFC-0012 S5).

use byard_compiler::interp::eval::Interpreter;
use byard_compiler::interp::events::EventKind as CompKind;
use byard_compiler::parser::parse;
use byard_compiler::symbol::Symbol;
use byard_core::frame::RenderFrame;
use byard_core::platform::{EventKind, InputEvent};

fn ev(kind: EventKind, pos: (f32, f32), t: u64) -> InputEvent {
    InputEvent {
        kind,
        pos,
        delta: (0.0, 0.0),
        payload: None,
        time_ms: t,
    }
}

/// Renders `src`, taps the centre of its first registered `Tap` hit rect, and
/// returns the resulting `count` value.
fn tap_count(src: &str) -> i64 {
    let parsed = parse(src);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let view = parsed.views[0].clone();
    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&view, &[]);
    interp.tick();

    // Render once so the router registers the button's hit rect (input for tick
    // T is dispatched against tick T−1's geometry).
    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, 800.0, 600.0);

    let (_, _, br) = interp
        .router
        .handler_rects()
        .into_iter()
        .find(|(_, k, _)| matches!(k, CompKind::Tap))
        .expect("a Tap handler was registered");
    let centre = (br.x + br.w / 2.0, br.y + br.h / 2.0);

    // A full tap: down then up at the same point, within the tap thresholds.
    interp.dispatch_events(&[ev(EventKind::PointerDown, centre, 0)]);
    interp.tick();
    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, 800.0, 600.0);
    interp.dispatch_events(&[ev(EventKind::PointerUp, centre, 10)]);
    interp.tick();

    let count = interp.var_signal(&Symbol::intern("count")).unwrap();
    interp.peek(count).as_int().unwrap()
}

#[test]
fn a_disabled_button_does_not_fire_its_tap() {
    // Control: an ordinary button increments on tap.
    let enabled = "View V() { var count: Int = 0 \
         Button(\"go\") #[bg: 0x6495ED, width: 120, height: 44] => count++ }";
    assert_eq!(tap_count(enabled), 1, "an enabled button taps");

    // A `disabled: true` button lays out and registers its handler, but the tap
    // is gated — `count` stays 0.
    let disabled = "View V() { var count: Int = 0 \
         Button(\"nope\") #[bg: 0x6495ED, width: 120, height: 44, disabled: true] => count++ }";
    assert_eq!(tap_count(disabled), 0, "a disabled button ignores the tap");
}
