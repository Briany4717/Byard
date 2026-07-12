//! Pratt (precedence-climbing) expression parser + string-interpolation
//! splitting (RFC-0002 §"Parser", §"Grammar"; the PEP 701 interpolation model).
//!
//! `nud` handles expression *starts* (literals, identifiers, arrays, parens,
//! lambdas, leading-dot class references); `led` handles infix/postfix
//! continuations (`.`, calls, `++`/`--`, `= += -=`, the ternary `?:`). Binding
//! powers encode precedence; right-associative forms (assignment, ternary) use
//! `r_bp < l_bp`.

use super::Parser;
use super::ast::{Arg, AssignOp, BinOp, Expr, PostfixOp, StrPart};
use crate::diagnostics::Span;
use crate::lexer::Token;
use crate::symbol::Symbol;

/// Left/right binding powers for a `led` operator.
struct Bp {
    left: u8,
    right: u8,
}

impl Parser<'_> {
    /// Parses an expression whose operators bind at least as tightly as
    /// `min_bp`. Always returns a node: on error it records a [`CompileError`]
    /// and yields [`Expr::Error`] so recovery can continue (multi-diagnostic).
    pub(super) fn parse_expr(&mut self, min_bp: u8) -> Expr {
        let mut lhs = self.parse_nud();
        while let Some(bp) = self.peek_led() {
            if bp.left < min_bp {
                break;
            }
            lhs = self.parse_led(lhs, bp.right);
        }
        lhs
    }

    /// Null-denotation: parses an expression that needs no left operand.
    fn parse_nud(&mut self) -> Expr {
        let span = self.cur_span();
        match self.cur().cloned() {
            Some(Token::IntLit(n)) => {
                self.advance();
                Expr::IntLit(n, span)
            }
            Some(Token::FloatLit(f)) => {
                self.advance();
                Expr::FloatLit(f, span)
            }
            Some(Token::AngleLit(rad)) => {
                self.advance();
                Expr::AngleLit(rad, span)
            }
            // A `200ms` duration folds to a plain integer count of milliseconds
            // (RFC-0010) — only read inside an `anim.*` curve call, as ms.
            Some(Token::DurationLit(ms)) => {
                self.advance();
                Expr::IntLit(i64::from(ms), span)
            }
            Some(Token::StrLit) => {
                self.advance();
                let parts = self.parse_string_literal(span);
                Expr::StrLit(parts, span)
            }
            Some(Token::Ident(sym)) => {
                self.advance();
                Expr::Ident(sym, span)
            }
            Some(Token::Minus) => self.parse_negative(),
            Some(Token::Style) => self.parse_style_value(),
            Some(Token::LBrack) => self.parse_array(),
            Some(Token::LParen) => self.parse_paren_or_lambda(),
            Some(Token::Pipe) => self.parse_pipe_lambda(),
            Some(Token::LBrace) => self.parse_callback_block(),
            Some(Token::Dot) => self.parse_class_ref(),
            _ => {
                self.error("an expression");
                // Do not consume: let the caller's recovery decide.
                Expr::Error(span)
            }
        }
    }

    /// A first-class style value `style { name: value, … }` (RFC-0016) in
    /// expression position (e.g. `let btn = style { … }`). Attributes are the
    /// same `name: value` / `..spread` forms as an element's `#[…]`, separated
    /// by commas or newlines. (The member-level `style { .class … }` rules
    /// block is a separate, statement-level form.)
    fn parse_style_value(&mut self) -> Expr {
        let start = self.cur_span();
        self.advance(); // `style`
        self.expect(&Token::LBrace, "'{' after `style`");
        let mut attrs = Vec::new();
        let mut states = Vec::new();
        while !matches!(self.cur(), Some(Token::RBrace) | None) {
            let before = self.pos;
            if self.at_state_block() {
                if let Some(sb) = self.parse_state_block() {
                    states.push(sb);
                }
            } else if let Some(a) = self.parse_attr() {
                attrs.push(a);
            }
            if self.pos == before {
                self.advance(); // guarantee progress on a malformed attr
            }
            // Commas are optional (newlines separate too, but aren't tokens).
            self.eat(&Token::Comma);
        }
        self.expect(&Token::RBrace, "'}' to close the style");
        Expr::StyleValue {
            attrs,
            states,
            span: self.span_from(start),
        }
    }

    /// True when the cursor is on an `on <state> { … }` interaction-state block
    /// (RFC-0016). `on` is a *contextual* keyword: it opens a state block only
    /// here (followed by an identifier), so nothing else that spells `on` breaks.
    fn at_state_block(&self) -> bool {
        matches!(self.cur(), Some(Token::Ident(s)) if s.as_str() == "on")
            && matches!(self.peek2(), Some(Token::Ident(_)))
    }

    /// Parses `on <state> { attr* }` (RFC-0016). An unknown state name records
    /// an [`UnknownStyleState`](crate::diagnostics::CompileError::UnknownStyleState)
    /// but the block body is still consumed so parsing recovers cleanly.
    fn parse_state_block(&mut self) -> Option<super::ast::StateBlock> {
        use super::ast::StyleStateKind;
        let start = self.cur_span();
        self.advance(); // `on`
        let name_span = self.cur_span();
        let name = self.expect_ident("an interaction state (hover/pressed/focused/disabled)")?;
        let kind = StyleStateKind::from_name(name.as_str());
        if kind.is_none() {
            let hint =
                crate::util::closest_match(name.as_str(), StyleStateKind::NAMES.iter().copied())
                    .map(str::to_string);
            self.errors
                .push(crate::diagnostics::CompileError::UnknownStyleState {
                    span: name_span,
                    name: name.as_str().to_string(),
                    hint,
                });
        }
        self.expect(&Token::LBrace, "'{' after the interaction state");
        let mut attrs = Vec::new();
        while !matches!(self.cur(), Some(Token::RBrace) | None) {
            let before = self.pos;
            if let Some(a) = self.parse_attr() {
                attrs.push(a);
            }
            if self.pos == before {
                self.advance();
            }
            self.eat(&Token::Comma);
        }
        self.expect(&Token::RBrace, "'}' to close the state block");
        // Only yield a block for a known state — an unknown one is dropped (the
        // error is already recorded) rather than silently mis-applied.
        Some(super::ast::StateBlock {
            state: kind?,
            attrs,
            span: self.span_from(start),
        })
    }

    /// A negative numeric literal `-<number>`. Byld has no binary arithmetic
    /// operators, so a leading `-` is only meaningful as the sign of a numeric
    /// literal (e.g. `translate: (-8, 0)`, `rotate: -90deg`). Anything else
    /// gets a targeted "a number after `-`" diagnostic instead of the generic
    /// "expected an expression", so the dev-overlay message is actionable.
    fn parse_negative(&mut self) -> Expr {
        let start = self.cur_span();
        self.advance(); // '-'
        match self.cur().cloned() {
            Some(Token::IntLit(n)) => {
                self.advance();
                Expr::IntLit(-n, self.span_from(start))
            }
            Some(Token::FloatLit(f)) => {
                self.advance();
                Expr::FloatLit(-f, self.span_from(start))
            }
            Some(Token::AngleLit(rad)) => {
                self.advance();
                Expr::AngleLit(-rad, self.span_from(start))
            }
            _ => {
                self.error("a number after `-`");
                Expr::Error(self.span_from(start))
            }
        }
    }

    /// Peeks the binding power of the upcoming `led` operator, if any.
    fn peek_led(&self) -> Option<Bp> {
        match self.cur()? {
            // Member access and calls bind tightest.
            Token::Dot => Some(Bp {
                left: 22,
                right: 23,
            }),
            Token::LParen => Some(Bp {
                left: 20,
                right: 21,
            }),
            // Postfix mutation.
            Token::PlusPlus | Token::MinusMinus => Some(Bp {
                left: 18,
                right: 19,
            }),
            // Binary arithmetic (RFC-0020 enabler): standard precedence —
            // `* /` over `+ -`, both left-associative (`right = left + 1`),
            // and both above the `merge`/ternary/`with`/assignment band so
            // `p * 360 with anim.spring()` animates the product.
            Token::Star | Token::Slash => Some(Bp { left: 9, right: 10 }),
            Token::Plus | Token::Minus => Some(Bp { left: 7, right: 8 }),
            // Ternary (right-assoc). `right: 4` (not 3) so the else-branch is
            // parsed one power *above* the `with` operator (left: 3) below — this
            // is what makes `a ? b : c with k` group as `(a ? b : c) with k`
            // (RFC-0010): the whole conditional is the animated value, not just
            // the else-branch. Nothing else has left bp 3, so this is invisible
            // to every other expression.
            Token::Question => Some(Bp { left: 4, right: 4 }),
            // `with` animation operator (RFC-0010): below the ternary, above
            // assignment, so `(cond ? a : b) with anim.spring()` is the value.
            Token::With => Some(Bp { left: 3, right: 3 }),
            // `merge` style composition (RFC-0016): binds tighter than the
            // ternary so `a merge b` is a single composed style; right-assoc.
            Token::Merge => Some(Bp { left: 5, right: 6 }),
            // Assignment (right-assoc), lowest.
            Token::Eq | Token::PlusEq | Token::MinusEq => Some(Bp { left: 2, right: 1 }),
            _ => None,
        }
    }

    /// Left-denotation: extends `lhs` with the operator the parser is sitting
    /// on. `right_bp` is the right binding power for right-associative forms.
    fn parse_led(&mut self, lhs: Expr, right_bp: u8) -> Expr {
        let start = lhs.span();
        match self.cur().cloned() {
            Some(Token::Dot) => {
                self.advance();
                let field = self
                    .expect_ident("a field name")
                    .unwrap_or_else(|| Symbol::intern(""));
                Expr::Member {
                    base: Box::new(lhs),
                    field,
                    span: self.span_from(start),
                }
            }
            Some(Token::LParen) => {
                self.advance();
                let args = self.parse_arg_list(&Token::RParen);
                self.expect(&Token::RParen, "')'");
                Expr::Call {
                    callee: Box::new(lhs),
                    args,
                    span: self.span_from(start),
                }
            }
            Some(Token::PlusPlus | Token::MinusMinus) => {
                let op = if matches!(self.cur(), Some(Token::PlusPlus)) {
                    PostfixOp::Inc
                } else {
                    PostfixOp::Dec
                };
                self.advance();
                Expr::Postfix {
                    target: Box::new(lhs),
                    op,
                    span: self.span_from(start),
                }
            }
            // Binary arithmetic (RFC-0020 enabler). A `-` reaching `led` is
            // always subtraction — the numeric-sign form is consumed by
            // `parse_negative` in `nud`, before any left operand exists.
            Some(tok @ (Token::Plus | Token::Minus | Token::Star | Token::Slash)) => {
                let op = match tok {
                    Token::Plus => BinOp::Add,
                    Token::Minus => BinOp::Sub,
                    Token::Star => BinOp::Mul,
                    _ => BinOp::Div,
                };
                self.advance();
                let rhs = Box::new(self.parse_expr(right_bp));
                Expr::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs,
                    span: self.span_from(start),
                }
            }
            Some(Token::Question) => {
                self.advance();
                let then = Box::new(self.parse_expr(0));
                self.expect(&Token::Colon, "':' in ternary");
                let els = Box::new(self.parse_expr(right_bp));
                Expr::Ternary {
                    cond: Box::new(lhs),
                    then,
                    els,
                    span: self.span_from(start),
                }
            }
            // `value with anim.*(…)` (RFC-0010): `lhs` is the target value; the
            // RHS is the `anim.*` curve call, resolved to a typed `Curve` at
            // lowering. Parsed at `right_bp` (3) so it stops before an assignment.
            Some(Token::With) => {
                self.advance();
                let anim = Box::new(self.parse_expr(right_bp));
                Expr::Animated {
                    value: Box::new(lhs),
                    anim,
                    span: self.span_from(start),
                }
            }
            // `left merge right` (RFC-0016): compose two styles, right wins.
            Some(Token::Merge) => {
                self.advance();
                let right = Box::new(self.parse_expr(right_bp));
                Expr::Merge {
                    left: Box::new(lhs),
                    right,
                    span: self.span_from(start),
                }
            }
            Some(tok @ (Token::Eq | Token::PlusEq | Token::MinusEq)) => {
                let op = match tok {
                    Token::Eq => AssignOp::Assign,
                    Token::PlusEq => AssignOp::Add,
                    _ => AssignOp::Sub,
                };
                self.advance();
                let value = Box::new(self.parse_expr(right_bp));
                Expr::Assign {
                    target: Box::new(lhs),
                    op,
                    value,
                    span: self.span_from(start),
                }
            }
            _ => lhs,
        }
    }

    /// `array := "[" (expr ("," expr)*)? "]"`.
    fn parse_array(&mut self) -> Expr {
        let start = self.cur_span();
        self.advance(); // [
        let mut items = Vec::new();
        while !matches!(self.cur(), Some(Token::RBracket) | None) {
            items.push(self.parse_expr(0));
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect(&Token::RBracket, "']'");
        Expr::Array(items, self.span_from(start))
    }

    /// A leading-dot class reference `.title` (RFC-0002 `style_rule`-style use
    /// in `#[style: .title]`).
    fn parse_class_ref(&mut self) -> Expr {
        let start = self.cur_span();
        self.advance(); // .
        let class = self
            .expect_ident("a class name after '.'")
            .unwrap_or_else(|| Symbol::intern(""));
        Expr::ClassRef(class, self.span_from(start))
    }

    /// Disambiguates `(` between a grouped expression, a tuple (`Len` pair/quad),
    /// and a parenthesized lambda `(params) => body`.
    fn parse_paren_or_lambda(&mut self) -> Expr {
        let start = self.cur_span();
        self.advance(); // (

        // Empty parens: only valid as a zero-arg lambda `() => body`.
        if self.eat(&Token::RParen) {
            if self.eat(&Token::Arrow) {
                let body = Box::new(self.parse_expr(0));
                return Expr::Lambda {
                    params: Vec::new(),
                    body,
                    span: self.span_from(start),
                };
            }
            return Expr::Tuple(Vec::new(), self.span_from(start));
        }

        let args = self.parse_arg_list(&Token::RParen);
        self.expect(&Token::RParen, "')'");

        // `(a, b) => body` — the items were really lambda parameters.
        if self.eat(&Token::Arrow) {
            let params = args
                .iter()
                .map(|a| {
                    if a.name.is_some() {
                        self.error_at(a.value.span(), "a lambda parameter name");
                    }
                    match &a.value {
                        Expr::Ident(sym, _) => sym.clone(),
                        other => {
                            self.error_at(other.span(), "a lambda parameter name");
                            Symbol::intern("")
                        }
                    }
                })
                .collect();
            let body = Box::new(self.parse_expr(0));
            return Expr::Lambda {
                params,
                body,
                span: self.span_from(start),
            };
        }

        if args.len() == 1 && args[0].name.is_none() {
            // Grouped expression — unwrap to the inner node.
            args[0].value.clone()
        } else {
            Expr::Tuple(args, self.span_from(start))
        }
    }

    /// `lambda := "|" param_list? "|" expr` (the `filter(|x| …)` form).
    fn parse_pipe_lambda(&mut self) -> Expr {
        let start = self.cur_span();
        self.advance(); // |
        let mut params = Vec::new();
        while !matches!(self.cur(), Some(Token::Pipe) | None) {
            if let Some(sym) = self.expect_ident("a lambda parameter") {
                params.push(sym);
            } else {
                break;
            }
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect(&Token::Pipe, "'|'");
        let body = Box::new(self.parse_expr(0));
        Expr::Lambda {
            params,
            body,
            span: self.span_from(start),
        }
    }

    /// A callback-prop literal `{ (|params|)? stmt* }` (RFC-0019). The optional
    /// `|params|` header names the callback's arguments (`{|text| … }`, `{|_|}`);
    /// the body is a sequence of self-delimiting action statements (byld has no
    /// statement separator — each `count++` / `x = e` / `f()` is its own node),
    /// run in order when the callback fires. `{}` is the no-op default.
    ///
    /// Represented as an [`Expr::Lambda`] over an [`Expr::Block`] so invocation
    /// reuses the parameterized-`fn` inlining path: the caller's block is
    /// evaluated in the caller's scope, its `var` writes routed through the
    /// reactive system by `SignalId` (RFC-0019 §2).
    fn parse_callback_block(&mut self) -> Expr {
        let start = self.cur_span();
        self.advance(); // {
        // Optional `|params|` header.
        let mut params = Vec::new();
        if self.eat(&Token::Pipe) {
            while !matches!(self.cur(), Some(Token::Pipe) | None) {
                if let Some(sym) = self.expect_ident("a callback parameter") {
                    params.push(sym);
                } else {
                    break;
                }
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
            self.expect(&Token::Pipe, "'|' to close the callback parameter list");
        }
        // Body statements up to `}`.
        let mut stmts = Vec::new();
        while !matches!(self.cur(), Some(Token::RBrace) | None) {
            let before = self.pos;
            stmts.push(self.parse_expr(0));
            // Guard against a non-advancing parse (a stray token that is neither
            // a statement start nor `}`) so recovery never spins.
            if self.pos == before {
                self.advance();
            }
        }
        self.expect(&Token::RBrace, "'}' to close the callback block");
        let span = self.span_from(start);
        Expr::Lambda {
            params,
            body: Box::new(Expr::Block(stmts, span)),
            span,
        }
    }

    /// `arg_list := arg ("," arg)*`, `arg := (IDENT ":")? expr`. Stops at
    /// `close`. A trailing comma is allowed.
    pub(super) fn parse_arg_list(&mut self, close: &Token) -> Vec<Arg> {
        let mut args = Vec::new();
        while self.cur().is_some() && self.cur() != Some(close) {
            // Named argument `name: expr` (two-token lookahead).
            let name = if let (Some(Token::Ident(sym)), Some(Token::Colon)) =
                (self.cur().cloned(), self.peek2())
            {
                self.advance(); // name
                self.advance(); // :
                Some(sym)
            } else {
                None
            };
            let value = self.parse_expr(0);
            args.push(Arg { name, value });
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        args
    }

    /// Splits a raw `StrLit` (whose `span` covers the quoted source) into text
    /// and interpolation parts, recursively parsing each `{ expr }` (PEP 701).
    fn parse_string_literal(&mut self, span: Span) -> Vec<StrPart> {
        let raw = &self.source[span.start as usize..span.end as usize];
        // Strip the surrounding quotes; a malformed (unclosed) literal is
        // already reported by the lexer, so guard the slice defensively.
        let inner = raw
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or("")
            .to_string();
        self.split_interpolations(&inner, span)
    }

    /// The interpolation splitter, operating on the already-quote-stripped
    /// inner text. Text runs are unescaped; each `{ … }` becomes an
    /// [`StrPart::Interp`] parsed as an expression.
    fn split_interpolations(&mut self, inner: &str, outer: Span) -> Vec<StrPart> {
        let cs: Vec<char> = inner.chars().collect();
        let mut parts = Vec::new();
        let mut text = String::new();
        let mut i = 0;
        while i < cs.len() {
            let c = cs[i];
            if c == '\\' && i + 1 < cs.len() {
                push_unescaped(&mut text, cs[i + 1]);
                i += 2;
                continue;
            }
            if c == '{' {
                if !text.is_empty() {
                    parts.push(StrPart::Text(std::mem::take(&mut text)));
                }
                let (frag, next) = collect_interpolation(&cs, i + 1);
                let expr = self.parse_fragment_expr(&frag, outer);
                parts.push(StrPart::Interp(Box::new(expr)));
                i = next;
                continue;
            }
            text.push(c);
            i += 1;
        }
        if !text.is_empty() || parts.is_empty() {
            parts.push(StrPart::Text(text));
        }
        parts
    }

    /// Parses an interpolation fragment as a standalone expression on a fresh
    /// sub-parser. Inner spans are relative to the fragment (coarse), which is
    /// acceptable for Phase 2 Dev diagnostics; the outer span anchors errors.
    fn parse_fragment_expr(&mut self, frag: &str, _outer: Span) -> Expr {
        let mut sub = Parser::new(frag);
        let expr = sub.parse_expr(0);
        self.errors.append(&mut sub.errors);
        expr
    }
}

/// Appends the unescaped form of the character after a `\` to `text`.
fn push_unescaped(text: &mut String, escaped: char) {
    match escaped {
        '"' => text.push('"'),
        '\\' => text.push('\\'),
        'n' => text.push('\n'),
        't' => text.push('\t'),
        '{' => text.push('{'),
        '}' => text.push('}'),
        other => {
            text.push('\\');
            text.push(other);
        }
    }
}

/// Collects the source of an interpolation starting at `cs[from]` (just after
/// the opening `{`), up to the matching `}`. Returns the fragment and the index
/// just past the closing `}`.
///
/// Because the interpolation lives inside the outer string literal, every quote
/// that delimits a *nested* string was escaped as `\"` in the source. Those
/// escapes are removed here so the fragment can be re-lexed as ordinary code:
/// an un-escaped `\"` becomes a real `"` and toggles in-string tracking, while
/// `{`/`}` outside a nested string balance the interpolation depth.
fn collect_interpolation(cs: &[char], from: usize) -> (String, usize) {
    let mut frag = String::new();
    let mut depth = 1usize;
    let mut in_str = false;
    let mut j = from;
    while j < cs.len() {
        let d = cs[j];
        if d == '\\' && j + 1 < cs.len() {
            let n = cs[j + 1];
            match n {
                // An escaped quote is a nested-string delimiter: un-escape and
                // toggle string state.
                '"' => {
                    frag.push('"');
                    in_str = !in_str;
                }
                '\\' => frag.push('\\'),
                other => {
                    frag.push('\\');
                    frag.push(other);
                }
            }
            j += 2;
            continue;
        }
        if in_str {
            frag.push(d);
        } else if d == '{' {
            depth += 1;
            frag.push(d);
        } else if d == '}' {
            depth -= 1;
            if depth == 0 {
                return (frag, j + 1);
            }
            frag.push(d);
        } else {
            frag.push(d);
        }
        j += 1;
    }
    (frag, j)
}
