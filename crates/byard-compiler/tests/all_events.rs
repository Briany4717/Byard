//! End-to-end reactivity for the full wired event set (RFC-0003 §4 catalog):
//! every supported event, driven through the exact path the winit runner uses
//! (`lower_view` → `render` → `dispatch_events` → `tick`), must mutate its
//! `var` when it lands inside the element and do nothing when it lands outside.

use byard_compiler::interp::env::Value;
use byard_compiler::interp::eval::Interpreter;
use byard_compiler::interp::events::EventKind as CompKind;
use byard_compiler::parser::parse;
use byard_compiler::symbol::Symbol;
use byard_core::frame::RenderFrame;
use byard_core::{EventKind, InputEvent, InputPayload};

const SRC: &str = "
View AllEvents() {
    var downs = 0
    var ups = 0
    var moves = 0
    var scrolls = 0
    var wheels = 0
    var changes = 0
    var taps = 0

    Box #[
        width: 120,
        height: 120,
        pointer_down => downs++,
        pointer_up => ups++,
        pointer_move => moves++,
        scroll => scrolls++,
        wheel => wheels++,
        change(e) => changes++,
    ] => taps++
}
";

fn ev(kind: EventKind, pos: (f32, f32), t: u64, payload: Option<InputPayload>) -> InputEvent {
    InputEvent {
        kind,
        pos,
        delta: (1.0, 1.0),
        payload,
        time_ms: t,
    }
}

#[test]
fn every_supported_event_reacts_and_respects_bounds() {
    let parsed = parse(SRC);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let view = parsed.views[0].clone();

    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&view, &[]);
    // Note: `scroll`/`change` are not in `Box`'s accepted attribute set (they
    // belong to `ScrollView`/value intrinsics), so the §5 checker reports them;
    // that does not block handler registration. This test exercises event
    // *dispatch* reactivity, so it deliberately crams every event onto one Box.
    interp.tick();

    // Render once so the router registers the Box's hit rects.
    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, 400.0, 400.0);

    let sig = |interp: &Interpreter, name: &str| interp.var_signal(&Symbol::intern(name)).unwrap();
    let counters = [
        "downs", "ups", "moves", "scrolls", "wheels", "changes", "taps",
    ];
    let sigs: Vec<_> = counters.iter().map(|n| sig(&interp, n)).collect();

    // The element center (any registered handler shares the inflated box rect).
    let rects = interp.router.handler_rects();
    let any = rects
        .iter()
        .find(|(_, k, _)| matches!(k, CompKind::PointerDown))
        .expect("pointer_down handler registered");
    let r = any.2;
    let center = (r.x + r.w / 2.0, r.y + r.h / 2.0);
    let outside = (r.x + r.w + 200.0, r.y + r.h + 200.0);

    // ── 1. Events that land OUTSIDE must do nothing ────────────────────
    interp.dispatch_events(&[
        ev(EventKind::PointerDown, outside, 0, None),
        ev(EventKind::PointerUp, outside, 10, None),
        ev(EventKind::PointerMove, outside, 20, None),
        ev(EventKind::Scroll, outside, 30, None),
        ev(EventKind::Wheel, outside, 40, None),
        ev(
            EventKind::Change,
            outside,
            50,
            Some(InputPayload::Str("x".into())),
        ),
    ]);
    interp.tick();
    for (n, s) in counters.iter().zip(&sigs) {
        assert_eq!(
            interp.peek(*s),
            Value::Int(0),
            "`{n}` must stay 0 for clicks outside"
        );
    }

    // ── 2. Each event INSIDE mutates its var ───────────────────────────
    // tap = pointer_down then pointer_up (also fires down/up per E4).
    interp.dispatch_events(&[
        ev(EventKind::PointerDown, center, 100, None),
        ev(EventKind::PointerUp, center, 150, None),
    ]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "downs")),
        Value::Int(1),
        "pointer_down"
    );
    assert_eq!(
        interp.peek(sig(&interp, "ups")),
        Value::Int(1),
        "pointer_up"
    );
    assert_eq!(
        interp.peek(sig(&interp, "taps")),
        Value::Int(1),
        "tap (E4: up→tap)"
    );

    interp.dispatch_events(&[ev(EventKind::PointerMove, center, 200, None)]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "moves")),
        Value::Int(1),
        "pointer_move"
    );

    interp.dispatch_events(&[ev(EventKind::Scroll, center, 300, None)]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "scrolls")),
        Value::Int(1),
        "scroll"
    );

    interp.dispatch_events(&[ev(EventKind::Wheel, center, 400, None)]);
    interp.tick();
    assert_eq!(interp.peek(sig(&interp, "wheels")), Value::Int(1), "wheel");

    interp.dispatch_events(&[ev(
        EventKind::Change,
        center,
        500,
        Some(InputPayload::Str("hello".into())),
    )]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "changes")),
        Value::Int(1),
        "change"
    );
}
