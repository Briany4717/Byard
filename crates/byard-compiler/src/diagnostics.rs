//! Shared error and span primitives (RFC-0002 §"Data structures", D6; INV-4/5).
//!
//! [`CompileError`] lives **only** here. Per D6 (and INV-1/INV-5), `byard-core`'s
//! `ByardError` gains no compiler variant — unifying the two is the job of the
//! application crate one layer up, so the dependency edge stays
//! `byard-compiler → byard-core` and never the reverse.
//!
//! Every error path in the compiler produces a `CompileError` carrying a
//! [`Span`] (INV-4: no silent failures). The variant set starts small and grows
//! one milestone at a time as later passes need to report new conditions.

/// A byte-offset range into the source text, `[start, end)`.
///
/// `Copy`; every token and AST node carries one for diagnostics and the future
/// LSP. Offsets are byte indices (not char indices), matching `logos`'s spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    /// Start byte offset (inclusive).
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
}

impl Span {
    /// Creates a span from a start and end byte offset.
    #[must_use]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }
}

impl From<std::ops::Range<usize>> for Span {
    fn from(range: std::ops::Range<usize>) -> Self {
        Self {
            start: range.start as u32,
            end: range.end as u32,
        }
    }
}

const _: () = {
    assert!(
        std::mem::size_of::<Span>() <= 8,
        "Span exceeded its 8-byte budget"
    );
};

/// A structural compilation error. Each variant carries the [`Span`] of the
/// offending source range so [`CompileError::render`] can anchor a caret under
/// it (INV-4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// The lexer could not turn a byte into any token (driver-level fallback
    /// for a `logos` lex error).
    UnexpectedChar {
        /// Source range of the offending byte(s).
        span: Span,
    },
    /// The parser expected one thing and found another.
    UnexpectedToken {
        /// Source range of the unexpected token.
        span: Span,
        /// Human-readable description of what was expected.
        expected: String,
    },
}

impl CompileError {
    /// The source span this error points at.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::UnexpectedChar { span } | Self::UnexpectedToken { span, .. } => *span,
        }
    }

    /// The one-line headline for this error (no source context).
    #[must_use]
    pub fn headline(&self) -> String {
        match self {
            Self::UnexpectedChar { .. } => "unexpected character".to_string(),
            Self::UnexpectedToken { expected, .. } => {
                format!("unexpected token, expected {expected}")
            }
        }
    }

    /// Renders a caret-anchored, human-readable diagnostic against `source`.
    ///
    /// The caret line underlines exactly the byte range `self.span()` covers
    /// (at least one `^`), with `line`/`column` reported 1-based.
    #[must_use]
    pub fn render(&self, source: &str) -> String {
        let span = self.span();
        let start = (span.start as usize).min(source.len());
        let end = (span.end as usize).min(source.len()).max(start);

        // Locate the line containing `start`.
        let line_start = source[..start].rfind('\n').map_or(0, |i| i + 1);
        let line_end = source[start..]
            .find('\n')
            .map_or(source.len(), |i| start + i);
        let line_no = source[..start].bytes().filter(|&b| b == b'\n').count() + 1;
        let col = start - line_start; // byte column, 0-based

        let line_text = &source[line_start..line_end];

        // The caret underlines the span, clamped to this line's end.
        let caret_len = (end.min(line_end) - start).max(1);
        let pad: String = line_text[..col].chars().map(|_| ' ').collect();
        let carets: String = std::iter::repeat_n('^', caret_len).collect();

        format!(
            "error: {headline}\n --> line {line_no}, column {col1}\n  |\n{line_no} | {line_text}\n  | {pad}{carets}",
            headline = self.headline(),
            col1 = col + 1,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// INV-5: `CompileError` must be `Send` so a failed parse can be shipped
    /// from the watcher thread to the logic thread (RFC-0002 §"Hot-reload").
    #[test]
    fn compile_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CompileError>();
    }

    #[test]
    fn render_points_at_the_right_byte_range() {
        // The '@' is at byte offset 2..3.
        let source = "ab@cd";
        let err = CompileError::UnexpectedChar {
            span: Span::new(2, 3),
        };
        let out = err.render(source);

        // The caret line is the last line; its '^' must sit under column 3.
        let caret_line = out.lines().next_back().unwrap();
        let caret_col = caret_line.find('^').unwrap();
        // "  | " prefix is 4 chars, then 2 spaces of padding for cols 0..2.
        assert_eq!(caret_col, 4 + 2, "caret must align under the '@'");
        assert_eq!(caret_line.matches('^').count(), 1);
        assert!(out.contains("line 1, column 3"));
    }

    #[test]
    fn render_underlines_a_multi_byte_span() {
        let source = "View Bad(";
        let err = CompileError::UnexpectedToken {
            span: Span::new(5, 8), // "Bad"
            expected: "a known token".to_string(),
        };
        let out = err.render(source);
        let caret_line = out.lines().next_back().unwrap();
        assert_eq!(caret_line.matches('^').count(), 3, "underline all of 'Bad'");
        assert!(out.contains("expected a known token"));
    }

    #[test]
    fn render_reports_line_number_on_later_lines() {
        let source = "View A() {}\nView B@\n";
        // '@' is on line 2.
        let at = source.find('@').unwrap() as u32;
        let err = CompileError::UnexpectedChar {
            span: Span::new(at, at + 1),
        };
        let out = err.render(source);
        assert!(out.contains("line 2"), "got: {out}");
    }

    #[test]
    fn span_from_range() {
        let s: Span = (3usize..7usize).into();
        assert_eq!(s, Span::new(3, 7));
    }
}
