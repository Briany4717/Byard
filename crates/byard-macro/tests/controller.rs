//! Integration tests for `#[byard_controller]`. Proc-macro crates can only
//! exercise their macro from an external test target.
//!
//! The impl-form test controllers do sync work in `async fn`s (no real I/O), so
//! the `unused_async` lint is expected and allowed here.
#![allow(clippy::unused_async)]

use byard_macro::byard_controller;

#[byard_controller]
struct CounterController {
    count: i64,
    label: String,
    ratio: f64,
    enabled: bool,
    tries: u32,
}

#[test]
fn struct_is_emitted_unchanged() {
    // The original struct still exists and is constructible/usable.
    let c = CounterController {
        count: 3,
        label: "hi".to_string(),
        ratio: 0.5,
        enabled: true,
        tries: 2,
    };
    assert_eq!(c.count, 3);
    assert_eq!(c.label, "hi");
    assert!(c.enabled);
    assert_eq!(c.tries, 2);
    assert!((c.ratio - 0.5).abs() < f64::EPSILON);
}

#[test]
fn metadata_maps_fields_to_byld_types() {
    // The interpreter's `member → field type` inference (M5) consumes this.
    assert_eq!(CounterController::byard_field_type("count"), Some("Int"));
    assert_eq!(CounterController::byard_field_type("tries"), Some("Int"));
    assert_eq!(CounterController::byard_field_type("ratio"), Some("Float"));
    assert_eq!(CounterController::byard_field_type("label"), Some("Str"));
    assert_eq!(CounterController::byard_field_type("enabled"), Some("Bool"));
    assert_eq!(CounterController::byard_field_type("missing"), None);
}

#[test]
fn fields_are_in_declaration_order() {
    assert_eq!(
        CounterController::BYARD_FIELDS,
        &[
            ("count", "Int"),
            ("label", "Str"),
            ("ratio", "Float"),
            ("enabled", "Bool"),
            ("tries", "Int"),
        ]
    );
}

#[byard_controller]
struct StrRefHolder<'a> {
    name: &'a str,
}

#[test]
fn handles_references_and_generics() {
    assert_eq!(StrRefHolder::byard_field_type("name"), Some("Str"));
    let h = StrRefHolder { name: "x" };
    assert_eq!(h.name, "x");
}

// ── The impl-block form: `impl Controller` dispatch (RFC-0028 §2) ────────
//
// These compile the emitted `impl Controller` against the real `byard` façade,
// confirming the generated `::byard::bridge::*` references resolve (a
// dev-dependency cycle byard-macro ↔ byard, permitted for dev-deps).

use byard::bridge::{Controller, HostValue};

#[byard_controller]
#[derive(Clone)]
struct MathController {
    base: i64,
    label: String,
}

#[byard_controller]
impl MathController {
    async fn add(&self, n: i64) -> Result<i64, ()> {
        Ok(self.base + n)
    }

    async fn greet(&self, who: String) -> Result<String, String> {
        if who.is_empty() {
            Err("empty name".to_string())
        } else {
            Ok(format!("{}: {who}", self.label))
        }
    }
}

#[test]
fn impl_form_dispatches_async_methods_by_name() {
    let c = MathController {
        base: 40,
        label: "hi".to_string(),
    };
    assert_eq!(c.type_name(), "MathController");

    let ok = pollster::block_on(c.invoke("add", vec![HostValue::Int(2)]));
    assert_eq!(ok, Ok(HostValue::Int(42)));

    let greet = pollster::block_on(c.invoke("greet", vec![HostValue::Str("Ada".into())]));
    assert_eq!(greet, Ok(HostValue::Str("hi: Ada".into())));

    // A method's `Err` maps through the `err` arm to an error `HostValue`.
    let err = pollster::block_on(c.invoke("greet", vec![HostValue::Str(String::new())]));
    assert_eq!(err, Err(HostValue::Str("empty name".into())));
}

#[test]
fn impl_form_unknown_method_is_an_error_reply_not_a_panic() {
    let c = MathController {
        base: 0,
        label: String::new(),
    };
    let r = pollster::block_on(c.invoke("frobnicate", vec![]));
    assert!(matches!(r, Err(HostValue::Str(_))));
}
