//! Lexer: source text → `Token` stream (RFC-0002 §"Lexer" + full `Token` set;
//! RFC-0003 §"Attribute syntax"; D12 nesting cap).
//!
//! A single `#[derive(Logos)]` enum compiles to one DFA. Whitespace and line
//! comments are skipped at the enum level. String interpolation is handled by a
//! manual callback ([`lex_string`]) that emits one raw `StrLit` token whose end
//! is located with a fixed-depth two-state stack (`InString` / `InBrace`); the
//! *parser* later re-invokes the pipeline on each `{...}` span (the PEP 701
//! model). Nesting interpolated strings deeper than 3 levels is rejected as
//! [`CompileError::StringNestingTooDeep`] (D12).
//!
//! The driver ([`lex`]) wraps every `logos` lex failure as a [`CompileError`]
//! with a [`Span`] — there are no silent failures (INV-4).

use logos::{Lexer, Logos};

use crate::diagnostics::{CompileError, Span};
use crate::symbol::Symbol;

/// Lexer-internal error, carried by `logos` and mapped to a [`CompileError`]
/// (with a span) by the [`lex`] driver. It carries no span itself; the driver
/// attaches `lexer.span()` at the point of failure.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum LexError {
    /// A byte that begins no valid token (the `logos` default error).
    #[default]
    Unexpected,
    /// More than 3 levels of interpolated-string nesting (D12).
    StringNestingTooDeep,
    /// A `"` opened a string that never closed before end of input.
    UnterminatedString,
}

/// The terminal set of the `byld` Lume surface (RFC-0002 §"Data structures",
/// RFC-0003 §"Attribute syntax").
///
/// `#[` is a single token ([`Token::HashBracket`]), never `#` followed by `[`,
/// so it can never lex ambiguously against a future standalone `#`.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(error = LexError)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip r"//[^\n]*")]
pub enum Token {
    // ---- keywords (reserved; INV-7) ----
    /// `View`
    #[token("View")]
    View,
    /// `var`
    #[token("var")]
    Var,
    /// `let`
    #[token("let")]
    Let,
    /// `fn`
    #[token("fn")]
    Fn,
    /// `inject`
    #[token("inject")]
    Inject,
    /// `as`
    #[token("as")]
    As,
    /// `for`
    #[token("for")]
    For,
    /// `in`
    #[token("in")]
    In,
    /// `when`
    #[token("when")]
    When,
    /// `else`
    #[token("else")]
    Else,
    /// `style`
    #[token("style")]
    Style,
    /// `untrack` (reserved intrinsic; D2 — parsed as a call, dispatched in the
    /// interpreter).
    #[token("untrack")]
    Untrack,

    // ---- identifiers & literals ----
    /// An identifier (interned).
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| Symbol::intern(lex.slice()))]
    Ident(Symbol),
    /// An angle literal with a unit suffix (RFC-0011 T1: `360deg`, `1.5rad`).
    /// Listed before [`Token::FloatLit`]/[`Token::IntLit`] so `logos`'s
    /// longest-match prefers the suffixed form over a bare number followed by
    /// an `Ident` — same principle as hex-before-decimal below. The value is
    /// canonicalized to **radians** right here at lex time: the deg→rad
    /// conversion is a pure, infallible numeric transform (nothing here can
    /// fail the way string/number parsing can), so there is no benefit to
    /// deferring it to a later compiler pass the way D9 defers *type*
    /// inference — this is unit normalization, not semantic analysis.
    #[regex(r"[0-9]+(\.[0-9]+)?deg", |lex| {
        let s = lex.slice();
        s[..s.len() - 3].parse::<f64>().ok().map(f64::to_radians)
    })]
    #[regex(r"[0-9]+(\.[0-9]+)?rad", |lex| {
        let s = lex.slice();
        s[..s.len() - 3].parse::<f64>().ok()
    })]
    AngleLit(f64),
    /// A floating-point literal. Listed before [`Token::IntLit`] because
    /// `logos` longest-match makes `3.14` a float, `3` an int.
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    FloatLit(f64),
    /// An integer literal (`i64`; D9), decimal or hex (`0xRRGGBB` colors,
    /// RFC-0005 §1). Hex is listed first so `0x1E` is one hex int, not `0`
    /// followed by an identifier.
    #[regex(r"0x[0-9a-fA-F]+", |lex| i64::from_str_radix(&lex.slice()[2..], 16).ok())]
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    IntLit(i64),
    /// A raw string literal (possibly interpolated). The slice (via its span)
    /// holds the source including quotes; interpolations are re-lexed by the
    /// parser.
    #[token("\"", lex_string)]
    StrLit,

    // ---- brackets & grouping ----
    /// `(`
    #[token("(")]
    LParen,
    /// `)`
    #[token(")")]
    RParen,
    /// `{`
    #[token("{")]
    LBrace,
    /// `}`
    #[token("}")]
    RBrace,
    /// `[` (array literal)
    #[token("[")]
    LBrack,
    /// `]` (closes an attribute block or array)
    #[token("]")]
    RBracket,
    /// `#[` (opens an attribute block) — one token.
    #[token("#[")]
    HashBracket,

    // ---- punctuation & operators ----
    /// `,`
    #[token(",")]
    Comma,
    /// `:`
    #[token(":")]
    Colon,
    /// `.`
    #[token(".")]
    Dot,
    /// `=>` (event / lambda arrow)
    #[token("=>")]
    Arrow,
    /// `->` (function return type)
    #[token("->")]
    ThinArrow,
    /// `=` (assignment)
    #[token("=")]
    Eq,
    /// `+=`
    #[token("+=")]
    PlusEq,
    /// `-=`
    #[token("-=")]
    MinusEq,
    /// `++`
    #[token("++")]
    PlusPlus,
    /// `--`
    #[token("--")]
    MinusMinus,
    /// `<` (opens generic type arguments)
    #[token("<")]
    Lt,
    /// `>` (closes generic type arguments)
    #[token(">")]
    Gt,
    /// `|` (lambda parameter delimiter)
    #[token("|")]
    Pipe,
    /// `?` (ternary)
    #[token("?")]
    Question,
}

/// State of the two-state string-scanning stack.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StrState {
    /// Scanning the characters of a string literal.
    InString,
    /// Scanning an interpolation expression inside `{ ... }`.
    InBrace,
}

/// Hard cap on interpolated-string nesting (D12): the outer string is level 1,
/// so at most 3 `InString` frames may exist at once.
const MAX_STRING_DEPTH: u8 = 3;

/// Locates the end of a string literal opened by `"`, accounting for escapes
/// and interpolation (`{ expr }`, where `expr` may itself contain strings).
///
/// On success bumps the lexer past the closing quote and returns `Ok(())`.
/// Returns [`LexError::StringNestingTooDeep`] if nesting would exceed
/// [`MAX_STRING_DEPTH`] (D12), or [`LexError::UnterminatedString`] if input
/// ends first.
fn lex_string(lex: &mut Lexer<Token>) -> Result<(), LexError> {
    // Capacity for depth 4 (7 frames) so the 4th level is *observed* and
    // rejected cleanly rather than overflowing the array (D12).
    let mut stack = [StrState::InString; 8];
    let mut sp: usize = 1; // stack[0] is the opening quote's InString frame
    let mut string_depth: u8 = 1; // number of InString frames currently open
    let mut consumed = 0usize;
    let mut escaped = false;

    for c in lex.remainder().chars() {
        consumed += c.len_utf8();
        // Escapes are honored in *both* states: a `\"` inside an interpolation
        // (which is itself inside this string) is an escaped quote, a literal —
        // it must not toggle string/brace state. D12's nesting cap still counts
        // genuinely nested *un-escaped* strings via the two-state stack below.
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        match stack[sp - 1] {
            StrState::InString => {
                if c == '"' {
                    sp -= 1;
                    string_depth -= 1;
                    if sp == 0 {
                        lex.bump(consumed);
                        return Ok(());
                    }
                } else if c == '{' {
                    if sp >= stack.len() {
                        return Err(LexError::StringNestingTooDeep);
                    }
                    stack[sp] = StrState::InBrace;
                    sp += 1;
                }
            }
            StrState::InBrace => match c {
                '{' => {
                    if sp >= stack.len() {
                        return Err(LexError::StringNestingTooDeep);
                    }
                    stack[sp] = StrState::InBrace;
                    sp += 1;
                }
                '}' => sp -= 1,
                '"' => {
                    if string_depth >= MAX_STRING_DEPTH || sp >= stack.len() {
                        return Err(LexError::StringNestingTooDeep);
                    }
                    stack[sp] = StrState::InString;
                    sp += 1;
                    string_depth += 1;
                }
                _ => {}
            },
        }
    }
    Err(LexError::UnterminatedString)
}

/// A token paired with its source span.
pub type SpannedToken = (Token, Span);

/// The result of lexing a source file: the token stream plus any diagnostics.
///
/// Lexing never aborts on the first error — it records a [`CompileError`] and
/// continues, so one pass surfaces multiple problems (INV-4).
#[derive(Debug, Default)]
pub struct LexedFile {
    /// Successfully lexed tokens, in source order.
    pub tokens: Vec<SpannedToken>,
    /// Diagnostics collected during lexing.
    pub errors: Vec<CompileError>,
}

/// Lexes `source` into a [`LexedFile`]. Every `logos` failure becomes a
/// span-carrying [`CompileError`]; the driver never panics on malformed input.
#[must_use]
pub fn lex(source: &str) -> LexedFile {
    let mut lexer = Token::lexer(source);
    let mut out = LexedFile::default();
    while let Some(result) = lexer.next() {
        let span: Span = lexer.span().into();
        match result {
            Ok(token) => out.tokens.push((token, span)),
            Err(err) => out.errors.push(match err {
                LexError::StringNestingTooDeep => CompileError::StringNestingTooDeep { span },
                LexError::UnterminatedString => CompileError::UnterminatedString { span },
                LexError::Unexpected => CompileError::UnexpectedChar { span },
            }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collects just the token kinds (dropping spans) for terse assertions.
    fn kinds(source: &str) -> Vec<Token> {
        let lexed = lex(source);
        assert!(
            lexed.errors.is_empty(),
            "unexpected lex errors: {:?}",
            lexed.errors
        );
        lexed.tokens.into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn keywords_lex_to_their_tokens() {
        assert_eq!(
            kinds("View var let fn inject as for in when else style untrack"),
            vec![
                Token::View,
                Token::Var,
                Token::Let,
                Token::Fn,
                Token::Inject,
                Token::As,
                Token::For,
                Token::In,
                Token::When,
                Token::Else,
                Token::Style,
                Token::Untrack,
            ]
        );
    }

    #[test]
    fn operators_and_brackets_lex() {
        assert_eq!(
            kinds("( ) { } [ ] #[ , : . => -> = += -= ++ -- < > | ?"),
            vec![
                Token::LParen,
                Token::RParen,
                Token::LBrace,
                Token::RBrace,
                Token::LBrack,
                Token::RBracket,
                Token::HashBracket,
                Token::Comma,
                Token::Colon,
                Token::Dot,
                Token::Arrow,
                Token::ThinArrow,
                Token::Eq,
                Token::PlusEq,
                Token::MinusEq,
                Token::PlusPlus,
                Token::MinusMinus,
                Token::Lt,
                Token::Gt,
                Token::Pipe,
                Token::Question,
            ]
        );
    }

    #[test]
    fn hash_bracket_is_one_token() {
        // `#[gap: 12]` — `#[` must be a single HashBracket, not `#` + `[`.
        assert_eq!(
            kinds("#[gap: 12]"),
            vec![
                Token::HashBracket,
                Token::Ident(Symbol::intern("gap")),
                Token::Colon,
                Token::IntLit(12),
                Token::RBracket,
            ]
        );
    }

    #[test]
    fn int_and_float_literals() {
        assert_eq!(
            kinds("42 1.5"),
            vec![Token::IntLit(42), Token::FloatLit(1.5)]
        );
    }

    #[test]
    fn angle_literals_lex_as_one_token_and_canonicalize_to_radians() {
        let toks = kinds("360deg 180deg 1.5rad");
        let Token::AngleLit(full_turn) = &toks[0] else {
            panic!("expected AngleLit, got {:?}", toks[0]);
        };
        assert!((full_turn - std::f64::consts::TAU).abs() < 1e-9);

        let Token::AngleLit(half_turn) = &toks[1] else {
            panic!("expected AngleLit, got {:?}", toks[1]);
        };
        assert!((half_turn - std::f64::consts::PI).abs() < 1e-9);

        // `rad` is already radians — passed through unchanged.
        assert_eq!(toks[2], Token::AngleLit(1.5));
    }

    #[test]
    fn angle_suffix_wins_over_a_bare_number_then_identifier() {
        // Longest-match must prefer `360deg` as one AngleLit, never
        // `IntLit(360)` followed by `Ident("deg")`.
        assert_eq!(
            kinds("360deg"),
            vec![Token::AngleLit(std::f64::consts::TAU)]
        );
    }

    #[test]
    fn arrow_wins_over_eq() {
        // longest-match: `=>` is Arrow, a following `=` is Eq.
        assert_eq!(kinds("=> ="), vec![Token::Arrow, Token::Eq]);
    }

    #[test]
    fn interpolated_string_is_a_single_strlit_with_correct_span() {
        let lexed = lex("\"Clicks: {clicks}\"");
        assert!(lexed.errors.is_empty());
        assert_eq!(lexed.tokens.len(), 1);
        let (tok, span) = &lexed.tokens[0];
        assert_eq!(*tok, Token::StrLit);
        assert_eq!(*span, Span::new(0, 18));
    }

    #[test]
    fn nested_literal_finds_the_correct_end() {
        // The inner `"b"` must not terminate the outer string early.
        let src = "\"a {f(\"b\")}\" x";
        let lexed = lex(src);
        assert!(lexed.errors.is_empty(), "{:?}", lexed.errors);
        assert_eq!(lexed.tokens[0].0, Token::StrLit);
        let end = lexed.tokens[0].1.end as usize;
        assert_eq!(&src[..end], "\"a {f(\"b\")}\"");
        // The trailing `x` lexes as a separate identifier.
        assert_eq!(lexed.tokens[1].0, Token::Ident(Symbol::intern("x")));
    }

    #[test]
    fn three_levels_of_interpolation_lex_ok() {
        // "L1 { "L2 { "L3" }" }" — exactly the D12 boundary, must pass.
        let src = "\"L1 { \"L2 { \"L3\" }\" }\"";
        let lexed = lex(src);
        assert!(
            lexed.errors.is_empty(),
            "3 levels must lex: {:?}",
            lexed.errors
        );
        assert_eq!(lexed.tokens.len(), 1);
        assert_eq!(lexed.tokens[0].0, Token::StrLit);
    }

    #[test]
    fn four_levels_of_interpolation_is_too_deep() {
        let src = "\"L1 { \"L2 { \"L3 { \"L4\" }\" }\" }\"";
        let lexed = lex(src);
        assert!(
            lexed
                .errors
                .iter()
                .any(|e| matches!(e, CompileError::StringNestingTooDeep { .. })),
            "4 levels must be rejected: {:?}",
            lexed.errors
        );
    }

    #[test]
    fn unterminated_string_is_reported_not_panicked() {
        let lexed = lex("\"open");
        assert!(
            lexed
                .errors
                .iter()
                .any(|e| matches!(e, CompileError::UnterminatedString { .. }))
        );
    }

    #[test]
    fn unexpected_char_becomes_a_diagnostic_never_a_panic() {
        // `@`, `$`, `~`, `` ` `` begin no token.
        let lexed = lex("View @ A");
        assert!(
            lexed
                .errors
                .iter()
                .any(|e| matches!(e, CompileError::UnexpectedChar { .. })),
            "got {:?}",
            lexed.errors
        );
        // Lexing still recovers and yields the surrounding tokens.
        assert_eq!(lexed.tokens[0].0, Token::View);
    }

    #[test]
    fn fuzz_junk_never_panics() {
        for junk in [
            "@#$%", "\"\"\"\"", "{{{{", "}}}}", "....", ">>>><<<<", "\\\\", "#",
        ] {
            let _ = lex(junk); // must not panic; errors are fine
        }
    }
}
