//! Integration tests for `#[byard_controller]` (M23). Proc-macro crates can only
//! exercise their macro from an external test target.

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
