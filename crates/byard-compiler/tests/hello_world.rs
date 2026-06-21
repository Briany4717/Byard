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

/// Finds the `=> count++` action on the first `Button` in the view.
fn button_action(view: &ViewDecl) -> &Expr {
    fn search(members: &[Member]) -> Option<&Expr> {
        for m in members {
            if let Member::Element(e) = m {
                if e.name.as_str() == "Button" {
                    if let Some(action) = &e.action {
                        return Some(action);
                    }
                }
                if let Some(a) = search(&e.children) {
                    return Some(a);
                }
            }
        }
        None
    }
    search(&view.body).expect("the example has a Button with an action")
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
        snapshot.iter().any(|s| s == "Count: 0"),
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
        texts(&frame2).iter().any(|s| s == "Count: 1"),
        "after the click the reactive text re-projected: {:?}",
        texts(&frame2)
    );

    // ── hot-reload (case-1 body edit) preserves `count` ────────────────
    let edited = parse(
        "View HelloWorld() {\n var count = 0\n var textVal = \"Initial\"\n var isToggled = true\n var sliderVal = 0.5\n Column #[bg: 0x222222] {\n Text(\"Total: {count}\")\n }\n}",
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
