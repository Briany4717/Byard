//! RFC-0007 — user-view instantiation, exercised end-to-end through the
//! `user_views.byd` example: parameterized views compose, arguments bind
//! (named/positional/defaulted), a reactive param flows across the call
//! boundary, and a `content` slot splices the caller's block.

use byard_compiler::interp::eval::Interpreter;
use byard_compiler::parser::parse;
use byard_core::frame::RenderFrame;

const USER_VIEWS: &str = include_str!("../examples/user_views.byd");

fn texts(frame: &RenderFrame) -> Vec<String> {
    frame.texts().iter().map(|t| t.text.clone()).collect()
}

/// Parses, loads the view registry, lowers `Main`, and renders one frame.
fn render() -> (Interpreter, RenderFrame) {
    let parsed = parse(USER_VIEWS);
    assert!(
        parsed.errors.is_empty(),
        "parse errors: {:?}",
        parsed.errors
    );

    let mut interp = Interpreter::new();
    let load_errs = interp.load_views(&parsed.views);
    assert!(load_errs.is_empty(), "load errors: {load_errs:?}");

    let known: Vec<&str> = parsed.views.iter().map(|v| v.name.as_str()).collect();
    let main = parsed
        .views
        .iter()
        .find(|v| v.name.as_str() == "Main")
        .expect("the example declares a `Main` view");
    let tree = interp.lower_view(main, &known);
    assert!(
        interp.errors().is_empty(),
        "lowering errors: {:?}",
        interp.errors()
    );
    interp.tick();

    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, 800.0, 900.0);
    (interp, frame)
}

#[test]
fn example_composes_views_and_binds_arguments() {
    let (_interp, frame) = render();
    let t = texts(&frame);

    // Both `StatCard` instances expanded, each with its own bound arguments.
    assert!(
        t.iter().any(|s| s == "Sessions"),
        "first StatCard title: {t:?}"
    );
    assert!(t.iter().any(|s| s == "128"), "first StatCard value: {t:?}");
    assert!(
        t.iter().any(|s| s == "Errors"),
        "second StatCard title: {t:?}"
    );

    // A reactive `var` (errors = 7) flowed through a positional argument and was
    // interpolated inside the child instance — arguments cross the call boundary
    // as live values, not snapshots (RFC-0007 §3).
    assert!(
        t.iter().any(|s| s == "7"),
        "reactive param crossed the call boundary: {t:?}"
    );

    // Nested `Badge` calls expanded inside `StatCard`, including the defaulted one.
    assert!(
        t.iter().any(|s| s == "live"),
        "nested Badge (named arg): {t:?}"
    );
    assert!(
        t.iter().any(|s| s == "default"),
        "nested Badge with omitted defaulted `tone`: {t:?}"
    );

    // The `content` slot spliced the caller's block into the `Panel`.
    assert!(
        t.iter().any(|s| s == "Slotted content"),
        "panel heading: {t:?}"
    );
    for tag in ["reactive", "typed", "native"] {
        assert!(
            t.iter().any(|s| s == tag),
            "slotted Badge `{tag}` rendered: {t:?}"
        );
    }
}

/// Every user-view name resolves without a spurious `UnknownView`, and the
/// registry recognizes the four declared views.
#[test]
fn example_registry_loads_all_views() {
    let parsed = parse(USER_VIEWS);
    let mut interp = Interpreter::new();
    assert!(interp.load_views(&parsed.views).is_empty());
    for name in ["Badge", "StatCard", "Panel", "Main"] {
        assert!(
            interp
                .view_table()
                .contains(&byard_compiler::symbol::Symbol::intern(name)),
            "`{name}` registered in the view table"
        );
    }
}
