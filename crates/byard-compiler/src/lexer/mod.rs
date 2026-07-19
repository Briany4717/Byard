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

/// Bit 32 of a [`Token::IntLit`], set at lex time on a hex literal written
/// with **more than six digits** — the RFC-0005 §1 alpha-first `0xAARRGGBB`
/// colour form. The tag is what lets a colour consumer distinguish
/// `0x00FFFFFF` (transparent white, alpha byte explicitly written as zero)
/// from `0xFFFFFF` (opaque white): the two are the same `i64` value.
/// Channel extraction always truncates to `u32`, so the tag never bleeds
/// into a colour; computed (untagged) values keep the magnitude heuristic
/// (`> 0xFFFFFF` ⇒ alpha present).
pub const COLOR_HAS_ALPHA_TAG: i64 = 1 << 32;

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
    /// `with` (RFC-0010 A1): the infix animation operator inside `#[...]`
    /// values (`radius: pressed ? 3 : 10 with anim.spring()`). Reserved as its
    /// own token so it parses cleanly and can never be read as an identifier in
    /// value position.
    #[token("with")]
    With,
    /// `merge` (RFC-0016 M3): the infix style-composition operator —
    /// `base merge overrides` produces a new `Style` whose right operand wins.
    #[token("merge")]
    Merge,
    /// `use` (RFC-0008 D-F): a package-qualified symbol import at file top —
    /// `use material`, `use material as m`, `use material.{Card}`. The name
    /// is resolved by the module resolver against the *manifest-declared*
    /// dependency set; no path string ever appears in `byld` source (the
    /// two-layer rule, RFC-0001 §1).
    #[token("use")]
    Use,

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
    /// A duration literal with a `ms` suffix (RFC-0010: `anim.linear(200ms)`),
    /// value in **milliseconds**. Listed before the plain number rules so
    /// `logos`'s longest-match prefers `200ms` over `200` + `ms`. The parser
    /// lowers it to an `Expr::IntLit` node (see `parser/expr.rs`): a duration is
    /// only meaningful inside an `anim.*` curve call, read there as milliseconds.
    #[regex(r"[0-9]+ms", |lex| lex.slice()[..lex.slice().len() - 2].parse::<u32>().ok())]
    DurationLit(u32),
    /// A floating-point literal. Listed before [`Token::IntLit`] because
    /// `logos` longest-match makes `3.14` a float, `3` an int.
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    FloatLit(f64),
    /// An integer literal (`i64`; D9), decimal or hex (`0xRRGGBB` colors,
    /// RFC-0005 §1). Hex is listed first so `0x1E` is one hex int, not `0`
    /// followed by an identifier.
    ///
    /// A hex literal with **more than six digits** is, per the RFC-0005 §1
    /// colour contract, alpha-first `0xAARRGGBB` — and the written width is
    /// semantic: `0x00FFFFFF` (transparent white) and `0xFFFFFF` (opaque
    /// white) are the same `i64`, so the value alone cannot carry "the alpha
    /// byte was written". Such literals are tagged with
    /// [`COLOR_HAS_ALPHA_TAG`] (bit 32) here at lex time. Colour consumers
    /// read the tag (or fall back to the magnitude heuristic for computed
    /// values) and truncate to `u32` for the channels, so the tag never
    /// reaches a channel; a >6-digit hex literal used as a plain *number* is
    /// the one degenerate case this trades away (RFC-0005 reserves that
    /// width for colours).
    #[regex(r"0x[0-9a-fA-F]+", |lex| {
        let digits = &lex.slice()[2..];
        i64::from_str_radix(digits, 16)
            .ok()
            .map(|v| if digits.len() > 6 { v | COLOR_HAS_ALPHA_TAG } else { v })
    })]
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
    /// `..` (RFC-0016 spread: `#[..style]`). Longest-match keeps this distinct
    /// from a single `.` (member access / sub-property axis).
    #[token("..")]
    DotDot,
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
    /// `-` — the sign of a negative numeric literal (`translate: (-8, 0)`)
    /// *or* binary subtraction (part of the minimal arithmetic surface, see
    /// [`Token::Star`]). Longest-match keeps `->`, `-=` and `--` as their own
    /// tokens.
    #[token("-")]
    Minus,
    /// `+` — binary addition. Part of the minimal arithmetic surface
    /// (`+ - * /`) required by RFC-0020's reactive shape parameters
    /// (`sweep: percent * 3.6`). Longest-match keeps `+=` and `++` as their
    /// own tokens.
    #[token("+")]
    Plus,
    /// `*` — binary multiplication (see [`Token::Plus`]).
    #[token("*")]
    Star,
    /// `/` — binary division (see [`Token::Plus`]). A `//` comment still wins
    /// over two `/` tokens through the skip rule's longest match.
    #[token("/")]
    Slash,
    /// `==` — equality comparison (RFC-0027 §1). Longest-match keeps this
    /// distinct from a single `=` (assignment).
    #[token("==")]
    EqEq,
    /// `!=` — inequality comparison (RFC-0027 §1).
    #[token("!=")]
    BangEq,
    /// `!` — boolean negation (RFC-0027 §2). Longest-match keeps `!=` its own
    /// token.
    #[token("!")]
    Bang,
    /// `<=` — less-than-or-equal (RFC-0027 §1). Longest-match prefers this over
    /// `<` (generic open / less-than).
    #[token("<=")]
    LtEq,
    /// `>=` — greater-than-or-equal (RFC-0027 §1).
    #[token(">=")]
    GtEq,
    /// `&&` — short-circuiting logical AND (RFC-0027 §2).
    #[token("&&")]
    AmpAmp,
    /// `||` — short-circuiting logical OR (RFC-0027 §2). Longest-match keeps a
    /// single `|` (lambda delimiter) distinct.
    #[token("||")]
    PipePipe,
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
            kinds("View var let fn inject as for in when else style untrack with merge use"),
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
                Token::With,
                Token::Merge,
                Token::Use,
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
    fn comparison_and_logic_operators_lex() {
        // Longest-match must keep `==`/`!=`/`<=`/`>=`/`&&`/`||` distinct from
        // their single-char prefixes (`=`, `!`, `<`, `>`, `|`).
        assert_eq!(
            kinds("== != < <= > >= && || ! |"),
            vec![
                Token::EqEq,
                Token::BangEq,
                Token::Lt,
                Token::LtEq,
                Token::Gt,
                Token::GtEq,
                Token::AmpAmp,
                Token::PipePipe,
                Token::Bang,
                Token::Pipe,
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
