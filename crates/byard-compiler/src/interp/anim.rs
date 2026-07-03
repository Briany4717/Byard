//! RFC-0010 typed animation curves and their resolution from an `anim.*(…)`
//! call written after a `with` operator.
//!
//! The parser produces an [`Expr::Animated`](crate::parser::ast::Expr::Animated)
//! whose `anim` side is an ordinary call/member expression; [`resolve_curve`]
//! turns that surface into a typed, argument-validated [`Curve`] at lowering
//! time. The parser stays free of any knowledge of the curve catalog (mirrors
//! D6: the surface is generic; meaning is assigned later).
//!
//! The [`Motion`] runtime that actually drives an on-screen transition is a
//! separate slice; this module is only the *grammar → typed spec* half.

use crate::diagnostics::{CompileError, Span};
use crate::parser::ast::{Arg, Expr};
use crate::symbol::Symbol;
use crate::util::closest_match;

/// The easing family for `anim.ease(…)`. `InOut` is the default when no family
/// is named (the most common, symmetric ease).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EaseKind {
    /// Accelerate from rest.
    In,
    /// Decelerate to rest.
    Out,
    /// Symmetric ease in then out.
    InOut,
}

/// A resolved, typed animation curve (RFC-0010 §"Typed animation spec").
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Curve {
    /// A fixed-duration linear ramp.
    Linear {
        /// Duration in milliseconds.
        ms: u32,
    },
    /// A fixed-duration eased ramp.
    Ease {
        /// Duration in milliseconds.
        ms: u32,
        /// The easing family.
        kind: EaseKind,
    },
    /// A duration-free damped spring, continuous under interruption.
    Spring {
        /// Spring stiffness.
        stiffness: f32,
        /// Damping coefficient.
        damping: f32,
        /// Initial velocity.
        v0: f32,
    },
}

impl Curve {
    /// The default spring (RFC-0010 A2): a snappy, iOS-feel `210/20` with no
    /// initial velocity — what a bare `anim.spring()` resolves to.
    pub const DEFAULT_SPRING: Curve = Curve::Spring {
        stiffness: 210.0,
        damping: 20.0,
        v0: 0.0,
    };
}

/// The closed set of curve names, for the unknown-curve suggestion.
const CURVE_NAMES: &[&str] = &["linear", "ease", "spring"];

/// Resolves the `anim` side of an [`Expr::Animated`](crate::parser::ast::Expr)
/// into a typed [`Curve`], validating its arguments (RFC-0010).
///
/// Accepts both the parenthesised form (`anim.spring(stiffness: 210)`) and the
/// bare form (`anim.spring`), the latter using every default. Any name other
/// than `linear`/`ease`/`spring` is [`CompileError::UnknownAnimation`] with a
/// Levenshtein suggestion; malformed arguments are
/// [`CompileError::InvalidAnimation`].
///
/// # Errors
///
/// Returns the first diagnostic encountered (not an `anim.*` call, unknown
/// curve, or a bad argument list).
pub fn resolve_curve(anim: &Expr) -> Result<Curve, CompileError> {
    let (name, name_span, args) = destructure_anim_call(anim)?;
    match name.as_str() {
        "spring" => resolve_spring(args, name_span),
        "linear" => Ok(Curve::Linear {
            ms: single_duration(args, name_span, "linear")?,
        }),
        "ease" => Ok(Curve::Ease {
            ms: single_duration(args, name_span, "ease")?,
            kind: EaseKind::InOut,
        }),
        other => Err(CompileError::UnknownAnimation {
            span: name_span,
            name: other.to_string(),
            hint: closest_match(other, CURVE_NAMES.iter().copied()).map(str::to_string),
        }),
    }
}

/// Splits `anim.<name>(<args>)` (or the bare `anim.<name>`) into its curve
/// name, that name's span, and the argument list. Anything not shaped like a
/// call on the `anim` namespace is an [`CompileError::InvalidAnimation`].
fn destructure_anim_call(anim: &Expr) -> Result<(Symbol, Span, &[Arg]), CompileError> {
    // `anim.spring(...)` — a call whose callee is `anim.<name>`.
    if let Expr::Call { callee, args, .. } = anim {
        if let Expr::Member { base, field, span } = callee.as_ref() {
            if is_anim_base(base) {
                return Ok((field.clone(), *span, args));
            }
        }
    }
    // `anim.spring` — the bare member form, no parens, all defaults.
    if let Expr::Member { base, field, span } = anim {
        if is_anim_base(base) {
            return Ok((field.clone(), *span, &[]));
        }
    }
    Err(CompileError::InvalidAnimation {
        span: anim.span(),
        message: "expected an animation curve, e.g. `anim.spring(...)` or `anim.linear(200ms)`"
            .to_string(),
    })
}

/// Whether `base` is the `anim` namespace identifier.
fn is_anim_base(base: &Expr) -> bool {
    matches!(base, Expr::Ident(sym, _) if sym.as_str() == "anim")
}

/// Resolves `anim.spring(stiffness: …, damping: …, initial_velocity: …)`. All
/// three are optional named arguments; omitted ones take the A2 defaults.
fn resolve_spring(args: &[Arg], call_span: Span) -> Result<Curve, CompileError> {
    let Curve::Spring {
        mut stiffness,
        mut damping,
        mut v0,
    } = Curve::DEFAULT_SPRING
    else {
        unreachable!("DEFAULT_SPRING is a Spring")
    };
    for arg in args {
        let Some(name) = &arg.name else {
            return Err(CompileError::InvalidAnimation {
                span: arg.value.span(),
                message: "`anim.spring` takes named arguments \
                          (stiffness / damping / initial_velocity)"
                    .to_string(),
            });
        };
        let value = literal_f32(&arg.value)?;
        match name.as_str() {
            "stiffness" => stiffness = value,
            "damping" => damping = value,
            "initial_velocity" => v0 = value,
            other => {
                let hint = closest_match(other, ["stiffness", "damping", "initial_velocity"])
                    .map_or_else(String::new, |h| format!(" (did you mean `{h}`?)"));
                return Err(CompileError::InvalidAnimation {
                    span: arg.value.span(),
                    message: format!("unknown `anim.spring` argument `{other}`{hint}"),
                });
            }
        }
    }
    let _ = call_span;
    Ok(Curve::Spring {
        stiffness,
        damping,
        v0,
    })
}

/// Extracts the single positional duration (whole milliseconds) from
/// `anim.linear`/`ease`. A duration is a non-negative *integer* count of
/// milliseconds — a fractional value (`200.5`) is rejected rather than silently
/// truncated, and a value beyond `u32` is a range error.
fn single_duration(args: &[Arg], call_span: Span, curve: &str) -> Result<u32, CompileError> {
    let [arg] = args else {
        return Err(CompileError::InvalidAnimation {
            span: call_span,
            message: format!(
                "`anim.{curve}` takes exactly one duration, e.g. `anim.{curve}(200ms)`"
            ),
        });
    };
    if arg.name.is_some() {
        return Err(CompileError::InvalidAnimation {
            span: arg.value.span(),
            message: format!("`anim.{curve}`'s duration is positional, e.g. `anim.{curve}(200ms)`"),
        });
    }
    match &arg.value {
        Expr::IntLit(ms, span) => u32::try_from(*ms).map_err(|_| CompileError::InvalidAnimation {
            span: *span,
            message: format!(
                "`anim.{curve}` duration must be between 0 and {} milliseconds",
                u32::MAX
            ),
        }),
        // `200.5` folds to a `FloatLit`; a duration must be whole ms.
        Expr::FloatLit(_, span) => Err(CompileError::InvalidAnimation {
            span: *span,
            message: format!("`anim.{curve}` duration must be a whole number of milliseconds"),
        }),
        other => Err(CompileError::InvalidAnimation {
            span: other.span(),
            message: format!("`anim.{curve}` duration must be a millisecond literal, e.g. `200ms`"),
        }),
    }
}

/// Reads a compile-time numeric literal (`IntLit`/`FloatLit`) as `f32`. Curve
/// parameters are constants, so a non-literal (e.g. a `var`) is rejected.
fn literal_f32(expr: &Expr) -> Result<f32, CompileError> {
    match expr {
        #[allow(clippy::cast_precision_loss)]
        Expr::IntLit(n, _) => Ok(*n as f32),
        #[allow(clippy::cast_possible_truncation)]
        Expr::FloatLit(f, _) => Ok(*f as f32),
        other => Err(CompileError::InvalidAnimation {
            span: other.span(),
            message: "animation arguments must be numeric literals".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{AttrKind, Member};
    use crate::parser::parse;

    /// Parses `scale: 0 with <anim>` and resolves the curve, so tests can write
    /// the curve in surface syntax rather than hand-building an `Expr`.
    fn curve_of(anim_src: &str) -> Result<Curve, CompileError> {
        let src = format!("View V() {{ Box #[scale: 0 with {anim_src}] {{}} }}");
        let parsed = parse(&src);
        assert!(
            parsed.errors.is_empty(),
            "parse errors: {:?}",
            parsed.errors
        );
        let Member::Element(el) = &parsed.views[0].body[0] else {
            panic!("expected an element");
        };
        let AttrKind::Prop {
            value: Expr::Animated { anim, .. },
        } = &el.attrs[0].kind
        else {
            panic!("expected an animated attribute");
        };
        resolve_curve(anim)
    }

    #[test]
    fn bare_spring_uses_the_default_constants() {
        assert_eq!(curve_of("anim.spring()").unwrap(), Curve::DEFAULT_SPRING);
        // The parenless member form resolves identically.
        assert_eq!(curve_of("anim.spring").unwrap(), Curve::DEFAULT_SPRING);
    }

    #[test]
    fn spring_named_args_override_only_what_is_given() {
        assert_eq!(
            curve_of("anim.spring(stiffness: 300, damping: 25)").unwrap(),
            Curve::Spring {
                stiffness: 300.0,
                damping: 25.0,
                v0: 0.0,
            }
        );
    }

    #[test]
    fn linear_and_ease_take_a_duration_in_ms() {
        assert_eq!(
            curve_of("anim.linear(200ms)").unwrap(),
            Curve::Linear { ms: 200 }
        );
        // A bare integer is accepted as milliseconds too.
        assert_eq!(
            curve_of("anim.ease(150)").unwrap(),
            Curve::Ease {
                ms: 150,
                kind: EaseKind::InOut,
            }
        );
    }

    #[test]
    fn unknown_curve_suggests_the_closest_name() {
        let err = curve_of("anim.sprng()").unwrap_err();
        assert!(matches!(
            err,
            CompileError::UnknownAnimation { hint: Some(h), .. } if h == "spring"
        ));
    }

    #[test]
    fn spring_rejects_an_unknown_named_argument() {
        assert!(matches!(
            curve_of("anim.spring(stifness: 300)").unwrap_err(),
            CompileError::InvalidAnimation { .. }
        ));
    }

    #[test]
    fn linear_without_a_duration_is_an_error() {
        assert!(matches!(
            curve_of("anim.linear()").unwrap_err(),
            CompileError::InvalidAnimation { .. }
        ));
    }

    #[test]
    fn fractional_duration_is_rejected_not_truncated() {
        // `200.5` must be a hard error, not silently floored to 200ms.
        assert!(matches!(
            curve_of("anim.linear(200.5)").unwrap_err(),
            CompileError::InvalidAnimation { .. }
        ));
    }

    #[test]
    fn negative_duration_is_a_range_error() {
        assert!(matches!(
            curve_of("anim.ease(-5)").unwrap_err(),
            CompileError::InvalidAnimation { .. }
        ));
    }
}
