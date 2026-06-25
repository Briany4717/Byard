//! M15 fuzz: the lexer and parser must never panic on arbitrary input — every
//! failure is a `CompileError`, never an abort (INV-4). `proptest` reports any
//! panic as a test failure.

use byard_compiler::lexer::lex;
use byard_compiler::parser::parse;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4000))]

    /// Arbitrary (possibly non-`byld`) text of any shape.
    #[test]
    fn lex_and_parse_never_panic_on_arbitrary_text(src in ".{0,256}") {
        let lexed = lex(&src);
        // Lexing yields tokens and/or diagnostics, never a panic.
        let _ = (lexed.tokens.len(), lexed.errors.len());
        let parsed = parse(&src);
        let _ = (parsed.views.len(), parsed.errors.len());
    }

    /// Dense `byld`-flavored junk: the tokens and brackets the grammar uses,
    /// shuffled into nonsense, to stress recovery paths.
    #[test]
    fn parser_never_panics_on_byld_like_junk(
        src in r#"[A-Za-z0-9_ \t\n(){}\[\]#:=>.,|?"+\-/<*]{0,384}"#
    ) {
        let _ = parse(&src);
    }

    /// Pathological string/interpolation/escape soup (exercises the StrLit
    /// scanner and the interpolation splitter).
    #[test]
    fn string_scanner_never_panics(src in r#""[^"]{0,64}|[{}"\\]{0,64}"#) {
        let _ = parse(&src);
    }
}
