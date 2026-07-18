//! `Value ⇄ HostValue` conversions at the controller boundary (RFC-0028 §1).
//!
//! These live in `byard-compiler` — which depends on `byard-core`, never the
//! reverse (INV-1) — because they touch the interpreter's `!Send` [`Value`],
//! which cannot live in core. The boundary type ([`HostValue`]) is the neutral,
//! `Send` data enum defined in `byard-core::bridge`; only the data subset of
//! `Value` converts. A `Signal`/`Memo`/`Fn`/`Theme`/`Controller` has no
//! `HostValue` form — passing one as a controller argument is
//! [`CompileError::NonDataControllerArg`] (INV-13: the boundary holds only data).

use byard_core::bridge::HostValue;

use super::env::Value;
use crate::symbol::Symbol;

/// Converts a [`Value`] to a [`HostValue`] for the crossing to the Tokio pool,
/// or `None` if it holds a non-data variant (`Signal`/`Memo`/`Fn`/`Theme`/
/// `Controller`) — the caller turns that into
/// [`CompileError::NonDataControllerArg`](crate::diagnostics::CompileError::NonDataControllerArg).
#[must_use]
pub fn value_to_host(value: &Value) -> Option<HostValue> {
    Some(match value {
        Value::Unit => HostValue::Unit,
        Value::Bool(b) => HostValue::Bool(*b),
        Value::Int(n) => HostValue::Int(*n),
        Value::Float(f) => HostValue::Float(*f),
        Value::Str(s) => HostValue::Str(s.clone()),
        Value::List(xs) => HostValue::List(xs.iter().map(value_to_host).collect::<Option<_>>()?),
        Value::Record(fields) => HostValue::Record(
            fields
                .iter()
                .map(|(k, v)| Some((k.as_str().to_string(), value_to_host(v)?)))
                .collect::<Option<_>>()?,
        ),
        // A tuple is positional layout data, not a controller data shape; map it
        // to a list of its values (names dropped) so it still crosses lossily
        // rather than erroring — controllers rarely receive one.
        Value::Tuple(items) => HostValue::List(
            items
                .iter()
                .map(|(_, v)| value_to_host(v))
                .collect::<Option<_>>()?,
        ),
        // Reactive/handle variants have no data form (INV-13).
        Value::Signal(_)
        | Value::Memo(_)
        | Value::Fn(_)
        | Value::Theme(_)
        | Value::Controller(_) => return None,
    })
}

/// Converts a [`HostValue`] returned from a controller back into a [`Value`] so
/// it can be written to a `var` and rendered (RFC-0028 §5). Total (lossless over
/// the data subset), so it never fails.
#[must_use]
pub fn host_to_value(host: &HostValue) -> Value {
    match host {
        HostValue::Unit => Value::Unit,
        HostValue::Bool(b) => Value::Bool(*b),
        HostValue::Int(n) => Value::Int(*n),
        HostValue::Float(f) => Value::Float(*f),
        HostValue::Str(s) => Value::Str(s.clone()),
        HostValue::List(xs) => Value::List(xs.iter().map(host_to_value).collect()),
        HostValue::Record(fields) => Value::Record(
            fields
                .iter()
                .map(|(k, v)| (Symbol::intern(k), host_to_value(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_values_round_trip_through_host() {
        let v = Value::Record(vec![
            (Symbol::intern("id"), Value::Int(3)),
            (
                Symbol::intern("tags"),
                Value::List(vec![Value::Str("a".into()), Value::Bool(true)]),
            ),
        ]);
        let host = value_to_host(&v).expect("data converts");
        assert_eq!(host_to_value(&host), v);
    }

    #[test]
    fn non_data_values_are_rejected() {
        use super::super::env::{AstId, SignalId};
        assert!(value_to_host(&Value::Signal(SignalId(0))).is_none());
        assert!(value_to_host(&Value::Fn(AstId(0))).is_none());
        // Nesting a signal inside a list still fails (recursive check).
        assert!(value_to_host(&Value::List(vec![Value::Signal(SignalId(1))])).is_none());
    }
}
