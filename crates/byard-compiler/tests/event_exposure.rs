//! RFC-0012 §A — the six modeled-but-previously-unexposed events (`hover`,
//! `pointer_enter`, `pointer_exit`, `long_press`, `double_tap`, `secondary`),
//! the `click` alias for `tap`, and the `focus =>`/`blur =>` sugar (S2).
//!
//! Mirrors `tests/all_events.rs`'s structure and driving path
//! (`lower_view` → `render` → `dispatch_events` → `tick`): each event must
//! mutate its `var` when it lands inside the element and do nothing when it
//! lands outside.

use byard_compiler::interp::env::Value;
use byard_compiler::interp::eval::Interpreter;
use byard_compiler::parser::parse;
use byard_compiler::symbol::Symbol;
use byard_core::frame::RenderFrame;
use byard_core::{EventKind, InputEvent};

const SRC: &str = "
View EventExposure() {
    var hovers = 0
    var enters = 0
    var exits = 0
    var longs = 0
    var doubles = 0
    var secondaries = 0
    var clicks = 0
    var aFocuses = 0
    var aBlurs = 0
    var bFocuses = 0
    var bBlurs = 0

    Column {
        Box #[
            width: 100,
            height: 100,
            hover => hovers++,
            pointer_enter => enters++,
            pointer_exit => exits++,
            long_press => longs++,
            double_tap => doubles++,
            secondary => secondaries++,
            click => clicks++,
            focus => aFocuses++,
            blur => aBlurs++,
        ]
        Box #[
            width: 100,
            height: 100,
            focus => bFocuses++,
            blur => bBlurs++,
        ]
    }
}
";

fn ev(kind: EventKind, pos: (f32, f32), t: u64) -> InputEvent {
    InputEvent {
        kind,
        pos,
        delta: (0.0, 0.0),
        payload: None,
        time_ms: t,
    }
}

#[test]
fn newly_exposed_events_react_and_respect_bounds() {
    let parsed = parse(SRC);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let view = parsed.views[0].clone();

    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&view, &[]);
    interp.tick();

    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, 400.0, 400.0);

    let sig = |interp: &Interpreter, name: &str| interp.var_signal(&Symbol::intern(name)).unwrap();
    let counters = [
        "hovers",
        "enters",
        "exits",
        "longs",
        "doubles",
        "secondaries",
        "clicks",
    ];
    let sigs: Vec<_> = counters.iter().map(|n| sig(&interp, n)).collect();

    // Box A is the first (topmost) registered handler's rect.
    let rects = interp.router.handler_rects();
    let a = rects
        .iter()
        .find(|(_, k, _)| matches!(k, byard_compiler::interp::events::EventKind::Hover))
        .expect("hover handler registered on Box A")
        .2;
    let center = (a.x + a.w / 2.0, a.y + a.h / 2.0);
    let outside = (a.x + a.w + 300.0, a.y + a.h + 300.0);

    // ── 1. Every new event landing OUTSIDE must do nothing ─────────────────
    interp.dispatch_events(&[
        ev(EventKind::Hover, outside, 0),
        ev(EventKind::PointerEnter, outside, 10),
        ev(EventKind::PointerExit, outside, 20),
        ev(EventKind::LongPress, outside, 30),
        ev(EventKind::DoubleTap, outside, 40),
        ev(EventKind::Secondary, outside, 50),
        ev(EventKind::Tap, outside, 60), // "click" is an alias of "tap"
    ]);
    interp.tick();
    for (n, s) in counters.iter().zip(&sigs) {
        assert_eq!(
            interp.peek(*s),
            Value::Int(0),
            "`{n}` must stay 0 for events outside the element"
        );
    }

    // ── 2. Each new event INSIDE mutates its var ────────────────────────────
    interp.dispatch_events(&[ev(EventKind::Hover, center, 100)]);
    interp.tick();
    assert_eq!(interp.peek(sig(&interp, "hovers")), Value::Int(1), "hover");

    interp.dispatch_events(&[ev(EventKind::PointerEnter, center, 200)]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "enters")),
        Value::Int(1),
        "pointer_enter"
    );

    interp.dispatch_events(&[ev(EventKind::PointerExit, center, 300)]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "exits")),
        Value::Int(1),
        "pointer_exit"
    );

    interp.dispatch_events(&[ev(EventKind::LongPress, center, 400)]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "longs")),
        Value::Int(1),
        "long_press"
    );

    interp.dispatch_events(&[ev(EventKind::DoubleTap, center, 500)]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "doubles")),
        Value::Int(1),
        "double_tap"
    );

    interp.dispatch_events(&[ev(EventKind::Secondary, center, 600)]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "secondaries")),
        Value::Int(1),
        "secondary"
    );

    // `click` is an alias of `tap`; a real `Tap` core event fires it.
    interp.dispatch_events(&[ev(EventKind::Tap, center, 700)]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "clicks")),
        Value::Int(1),
        "click (alias of tap)"
    );
}

#[test]
fn focus_and_blur_sugar_fire_on_the_focused_signal_edges() {
    let parsed = parse(SRC);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let view = parsed.views[0].clone();

    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&view, &[]);
    interp.tick();

    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, 400.0, 400.0);

    let sig = |interp: &Interpreter, name: &str| interp.var_signal(&Symbol::intern(name)).unwrap();

    let rects = interp.router.handler_rects();
    let a = rects
        .iter()
        .find(|(_, k, _)| matches!(k, byard_compiler::interp::events::EventKind::Hover))
        .expect("Box A's hover handler is registered")
        .2;
    // Box B registers no pointer handlers, but `focus =>`/`blur =>` are
    // ordinary `Handler`s too — its `Blur` handler's rect locates it
    // directly, rather than inferring the position from Column layout.
    // Box A also has a `blur =>` handler (registered first, earlier in the
    // `Vec`); `.rev()` picks B's, the most-recently-registered one.
    let b = rects
        .iter()
        .rev()
        .find(|(_, k, _)| matches!(k, byard_compiler::interp::events::EventKind::Blur))
        .expect("Box B's blur handler is registered")
        .2;
    let b_center = (b.x + b.w / 2.0, b.y + b.h / 2.0);
    let a_center = (a.x + a.w / 2.0, a.y + a.h / 2.0);

    // Tapping A steals focus onto it: A focuses, nothing blurs yet (nothing
    // was focused before).
    interp.dispatch_events(&[
        ev(EventKind::PointerDown, a_center, 0),
        ev(EventKind::PointerUp, a_center, 10),
    ]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "aFocuses")),
        Value::Int(1),
        "focus => on A"
    );
    assert_eq!(
        interp.peek(sig(&interp, "aBlurs")),
        Value::Int(0),
        "no prior focus to blur"
    );
    assert_eq!(interp.peek(sig(&interp, "bFocuses")), Value::Int(0));

    // Tapping B steals focus away from A: A blurs, B focuses.
    interp.dispatch_events(&[
        ev(EventKind::PointerDown, b_center, 100),
        ev(EventKind::PointerUp, b_center, 110),
    ]);
    interp.tick();
    assert_eq!(
        interp.peek(sig(&interp, "aBlurs")),
        Value::Int(1),
        "blur => on A"
    );
    assert_eq!(
        interp.peek(sig(&interp, "bFocuses")),
        Value::Int(1),
        "focus => on B"
    );
    assert_eq!(
        interp.peek(sig(&interp, "aFocuses")),
        Value::Int(1),
        "A's focus count is unchanged"
    );
}
