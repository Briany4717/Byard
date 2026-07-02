//! M14 — the end-to-end Phase-2 thesis (RFC-0001 milestone, now driven by
//! `byld`): a `.byd` view renders through a `byard-core` `RenderFrame`, reacts
//! to a click, and survives a hot-reload with its state intact.

use byard_compiler::interp::env::Value;
use byard_compiler::interp::eval::Interpreter;
use byard_compiler::interp::reload::diff_view;
use byard_compiler::parser::ast::{Expr, Member, ViewDecl};
use byard_compiler::parser::parse;
use byard_compiler::symbol::Symbol;
use byard_core::frame::RenderFrame;

const HELLO_WORLD: &str = include_str!("../examples/hello_world.byd");

fn texts(frame: &RenderFrame) -> Vec<String> {
    frame.texts().iter().map(|t| t.text.clone()).collect()
}

/// Finds the action on the `+` `Button` (the increment) in the view.
fn button_action(view: &ViewDecl) -> &Expr {
    fn is_plus(e: &byard_compiler::parser::ast::ElementNode) -> bool {
        e.content.first().is_some_and(|arg| {
            matches!(
                &arg.value,
                Expr::StrLit(parts, _)
                    if matches!(parts.first(),
                        Some(byard_compiler::parser::ast::StrPart::Text(t)) if t == "+")
            )
        })
    }
    fn search<'a>(
        members: &'a [Member],
        pred: &dyn Fn(&byard_compiler::parser::ast::ElementNode) -> bool,
    ) -> Option<&'a Expr> {
        for m in members {
            if let Member::Element(e) = m {
                if e.name.as_str() == "Button" && pred(e) {
                    if let Some(action) = &e.action {
                        return Some(action);
                    }
                }
                if let Some(a) = search(&e.children, pred) {
                    return Some(a);
                }
            }
        }
        None
    }
    search(&view.body, &is_plus).expect("the example has a `+` Button with an action")
}

#[test]
fn hello_world_renders_reacts_and_hot_reloads() {
    // ── parse + build ──────────────────────────────────────────────────
    let parsed = parse(HELLO_WORLD);
    assert!(
        parsed.errors.is_empty(),
        "parse errors: {:?}",
        parsed.errors
    );
    let view = parsed.views[0].clone();

    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&view, &[]);
    assert!(
        interp.errors().is_empty(),
        "lowering errors: {:?}",
        interp.errors()
    );
    interp.tick();

    // ── initial render ─────────────────────────────────────────────────
    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, 800.0, 600.0);
    let snapshot = texts(&frame);
    assert!(
        snapshot.iter().any(|s| s == "the counter is 0"),
        "initial text: {snapshot:?}"
    );
    assert!(snapshot.iter().any(|s| s == "+"), "button label present");
    assert!(
        !frame.instances().is_empty(),
        "the Column's rounded background box was emitted"
    );

    // ── click `+` → count++ → re-render ────────────────────────────────
    interp.eval_action(button_action(&view)).unwrap();
    interp.tick();
    let mut frame2 = RenderFrame::new();
    interp.render(&tree, &mut frame2, 800.0, 600.0);
    assert!(
        texts(&frame2).iter().any(|s| s == "the counter is 1"),
        "after the click the reactive text re-projected: {:?}",
        texts(&frame2)
    );

    // ── hot-reload (case-1 body edit) preserves `count` ────────────────
    // Identical declaration shape as the demo (var×11, fn, let) → a
    // reactive-compatible (body-only) edit; only the element tree changes.
    let edited = parse(
        "View Main() {\n var count = 0\n var liked = true\n var volume = 0.5\n var name = \"\"\n var note = \"\"\n var hovered = false\n var eventFocused = false\n var pokes = 0\n var wheelTicks = 0\n var scrollTicks = 0\n var cardHovered = false\n fn describe(n: Int) => \"the counter is {n}\"\n let summary = describe(count)\n Column #[bg: 0x222222] {\n Text(\"Total: {count}\")\n }\n}",
    );
    let new_view = &edited.views[0];
    let kind = diff_view(&view, new_view);
    interp.reload(new_view, kind);

    let count_sig = interp.var_signal(&Symbol::intern("count")).unwrap();
    assert_eq!(
        interp.peek(count_sig),
        Value::Int(1),
        "the counter survived a reactive-compatible hot-reload"
    );
}

/// The demo must exercise every render pipeline so a regression in any of them
/// (text, solid boxes, the M21 `DecoratedBox`/`TextureSampler`) is caught here.
#[test]
fn demo_exercises_all_pipelines() {
    let parsed = parse(HELLO_WORLD);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let view = parsed.views[0].clone();

    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&view, &[]);
    assert!(interp.errors().is_empty(), "{:?}", interp.errors());
    interp.tick();

    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, 800.0, 600.0);

    let t = texts(&frame);
    assert!(
        t.iter().any(|s| s == "the counter is 0"),
        "fn+let summary: {t:?}"
    );
    assert!(t.iter().any(|s| s == "favourited"), "ternary branch: {t:?}");
    assert!(t.iter().any(|s| s == "reactive"), "for-loop item: {t:?}");
    assert!(
        t.iter().any(|s| s.contains("thanks for the love")),
        "when/else branch: {t:?}"
    );

    assert!(
        !frame.instances().is_empty(),
        "solid boxes (widgets/buttons)"
    );
    assert!(
        !frame.decorated().is_empty(),
        "DecoratedBox pipeline (border/shadow/opacity)"
    );
    assert_eq!(frame.textures().len(), 1, "one Image → TextureSampler");
    assert_eq!(frame.textures()[0].src, "logo.png");
}
