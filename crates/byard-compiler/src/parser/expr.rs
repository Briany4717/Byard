//! Pratt (precedence-climbing) expression parser + string-interpolation
//! splitting (RFC-0002 §"Parser", §"Grammar"; the PEP 701 interpolation model).
//!
//! `nud` handles expression *starts* (literals, identifiers, arrays, parens,
//! lambdas, leading-dot class references); `led` handles infix/postfix
//! continuations (`.`, calls, `++`/`--`, `= += -=`, the ternary `?:`). Binding
//! powers encode precedence; right-associative forms (assignment, ternary) use
//! `r_bp < l_bp`.

use super::Parser;
use super::ast::{Arg, AssignOp, BinOp, Expr, PostfixOp, StrPart, UnOp};
use crate::diagnostics::Span;
use crate::lexer::Token;
use crate::symbol::Symbol;

/// Left/right binding powers for a `led` operator.
struct Bp {
    left: u8,
    right: u8,
}

/// Binding power a prefix unary operator (`!`/`-`) parses its operand at
/// (RFC-0027 §2 precedence): tighter than every binary operator, looser than
/// postfix/call/member so `!a.b` is `!(a.b)` and `-a * b` is `(-a) * b`.
const UNARY_BP: u8 = 17;

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
                // A bare single-parameter lambda `x => body` (RFC-0027 §5, used
                // by `map`/`filter`). `IDENT` not followed by `=>` is a plain
                // identifier reference.
                if self.eat(&Token::Arrow) {
                    let body = Box::new(self.parse_expr(0));
                    Expr::Lambda {
                        params: vec![sym],
                        body,
                        span: self.span_from(span),
                    }
                } else {
                    Expr::Ident(sym, span)
                }
            }
            Some(Token::Minus) => self.parse_negative(),
            Some(Token::Bang) => self.parse_not(),
            Some(Token::Style) => self.parse_style_value(),
            Some(Token::LBrack) => self.parse_array(),
            Some(Token::LParen) => self.parse_paren_or_lambda(),
            Some(Token::Pipe) => self.parse_pipe_lambda(),
            Some(Token::LBrace) => self.parse_brace_value(),
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

    /// Parses `on <state> ("+" <state>)* { attr* }` (RFC-0016, RFC-0024 combined
    /// selectors). Each unknown state name records an
    /// [`UnknownStyleState`](crate::diagnostics::CompileError::UnknownStyleState)
    /// but the block body is still consumed so parsing recovers cleanly.
    fn parse_state_block(&mut self) -> Option<super::ast::StateBlock> {
        use super::ast::StyleStateKind;
        let start = self.cur_span();
        self.advance(); // `on`

        // The `+`-joined state list. Every state must be known for the block to
        // apply; any unknown name is diagnosed (and the block dropped).
        let mut states = Vec::new();
        let mut all_known = true;
        loop {
            let name_span = self.cur_span();
            let name = self.expect_ident("an interaction state (e.g. hover, focused, checked)")?;
            if let Some(kind) = StyleStateKind::from_name(name.as_str()) {
                states.push(kind);
            } else {
                all_known = false;
                let hint = crate::util::closest_match(
                    name.as_str(),
                    StyleStateKind::NAMES.iter().copied(),
                )
                .map(str::to_string);
                self.errors
                    .push(crate::diagnostics::CompileError::UnknownStyleState {
                        span: name_span,
                        name: name.as_str().to_string(),
                        hint,
                    });
            }
            // A `+` continues the combined selector; anything else ends it.
            if matches!(self.cur(), Some(Token::Plus)) {
                self.advance();
            } else {
                break;
            }
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
        // Only yield a block when every state parsed — an unknown one is dropped
        // (the error is already recorded) rather than silently mis-applied.
        if !all_known || states.is_empty() {
            return None;
        }
        Some(super::ast::StateBlock {
            states,
            attrs,
            span: self.span_from(start),
        })
    }

    /// A leading `-`: the sign of a numeric literal (`translate: (-8, 0)`,
    /// `rotate: -90deg`) folded straight into the literal, or — for any other
    /// operand — a unary negation [`Expr::Unary`] (RFC-0027 §2, `-x`, `-count`).
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
                let rhs = Box::new(self.parse_expr(UNARY_BP));
                Expr::Unary {
                    op: UnOp::Neg,
                    rhs,
                    span: self.span_from(start),
                }
            }
        }
    }

    /// A boolean negation `!expr` (RFC-0027 §2). Binds tighter than every binary
    /// operator, so `!a && b` is `(!a) && b`.
    fn parse_not(&mut self) -> Expr {
        let start = self.cur_span();
        self.advance(); // '!'
        let rhs = Box::new(self.parse_expr(UNARY_BP));
        Expr::Unary {
            op: UnOp::Not,
            rhs,
            span: self.span_from(start),
        }
    }

    /// Disambiguates a leading `{` between a **record literal** (RFC-0027 §6:
    /// `{ k: v, .. }` / `{ ..spread, k: v }`) and a **callback action block**
    /// (RFC-0019: `{ count++ }`, `{|x| … }`). A record is signalled by a first
    /// token of `IDENT :` or a `..spread`; everything else is a callback block.
    /// The empty `{}` stays a no-op callback block (the established default).
    fn parse_brace_value(&mut self) -> Expr {
        if self.at_record_literal() {
            self.parse_record()
        } else {
            self.parse_callback_block()
        }
    }

    /// True when the cursor sits on a `{` opening a record literal (as opposed
    /// to a callback block): the token after `{` is `..` (spread), or an
    /// identifier immediately followed by `:`.
    fn at_record_literal(&self) -> bool {
        if !matches!(self.cur(), Some(Token::LBrace)) {
            return false;
        }
        match self.peek2() {
            Some(Token::DotDot) => true,
            Some(Token::Ident(_)) => matches!(self.peek3(), Some(Token::Colon)),
            _ => false,
        }
    }

    /// `record := "{" (".." expr ",")? (IDENT ":" expr ("," …)*)? "}"`
    /// (RFC-0027 §6). Fields keep written order; a single optional `..spread`
    /// base may appear anywhere and seeds the record before written fields.
    fn parse_record(&mut self) -> Expr {
        let start = self.cur_span();
        self.advance(); // {
        let mut fields = Vec::new();
        let mut spread = None;
        while !matches!(self.cur(), Some(Token::RBrace) | None) {
            let before = self.pos;
            if self.eat(&Token::DotDot) {
                let base = self.parse_expr(0);
                if spread.is_none() {
                    spread = Some(Box::new(base));
                } else {
                    self.error_at(base.span(), "a single `..spread` per record");
                }
            } else if let Some(name) = self.expect_ident("a record field name") {
                self.expect(&Token::Colon, "':' after the field name");
                let value = self.parse_expr(0);
                fields.push((name, value));
            }
            if self.pos == before {
                self.advance(); // guarantee progress on a malformed field
            }
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect(&Token::RBrace, "'}' to close the record");
        Expr::Record {
            fields,
            spread,
            span: self.span_from(start),
        }
    }

    /// Peeks the binding power of the upcoming `led` operator, if any.
    ///
    /// Precedence ladder, lowest → highest (RFC-0027 §1 extends the pre-existing
    /// arithmetic band with logic/comparison):
    ///
    /// ```text
    /// assignment (= += -=)   L2  R1
    /// with                   L3  R3
    /// ||                     L4  R5
    /// ternary ?:             L6  R6      (tighter than ||; looser than comparison)
    /// &&                     L7  R8
    /// comparison             L9  R10     (== != < <= > >=)
    /// merge                  L11 R12
    /// + -                    L13 R14
    /// * /                    L15 R16
    /// (unary prefix !/-      17, in `nud`)
    /// ++ --                  L18 R19
    /// call (                 L20 R21
    /// . [ (member/index)     L22 R23
    /// ```
    fn peek_led(&self) -> Option<Bp> {
        match self.cur()? {
            // Member access, indexing, and calls bind tightest.
            Token::Dot | Token::LBrack => Some(Bp {
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
            // `* /` over `+ -`, both left-associative (`right = left + 1`).
            Token::Star | Token::Slash => Some(Bp {
                left: 15,
                right: 16,
            }),
            Token::Plus | Token::Minus => Some(Bp {
                left: 13,
                right: 14,
            }),
            // `merge` style composition (RFC-0016): right-assoc, above arithmetic
            // is meaningless (styles never mix with `+`), so it sits just below.
            Token::Merge => Some(Bp {
                left: 11,
                right: 12,
            }),
            // Comparison (RFC-0027 §1): non-chaining in practice, left-assoc.
            // Above `&&`/`||` and the ternary so `a == b ? x : y` is `(a==b)?x:y`.
            Token::EqEq | Token::BangEq | Token::Lt | Token::LtEq | Token::Gt | Token::GtEq => {
                Some(Bp { left: 9, right: 10 })
            }
            // `&&` (RFC-0027 §2): short-circuit, tighter than `||`/ternary.
            Token::AmpAmp => Some(Bp { left: 7, right: 8 }),
            // Ternary (right-assoc): tighter than `||` (RFC-0027 §1 note), looser
            // than comparison. `right: 6` keeps the else-branch one power above
            // `with` (L3) so `a ? b : c with k` groups as `(a ? b : c) with k`.
            Token::Question => Some(Bp { left: 6, right: 6 }),
            // `||` (RFC-0027 §2): short-circuit, lowest of the logic band.
            Token::PipePipe => Some(Bp { left: 4, right: 5 }),
            // `with` animation operator (RFC-0010): below the ternary, above
            // assignment, so `(cond ? a : b) with anim.spring()` is the value.
            Token::With => Some(Bp { left: 3, right: 3 }),
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
            // Indexing `base[index]` (RFC-0027 §4).
            Some(Token::LBrack) => {
                self.advance();
                let index = Box::new(self.parse_expr(0));
                self.expect(&Token::RBracket, "']' to close the index");
                Expr::Index {
                    base: Box::new(lhs),
                    index,
                    span: self.span_from(start),
                }
            }
            // Binary arithmetic (RFC-0020), comparison and logic (RFC-0027 §1/§2).
            // A `-` reaching `led` is always subtraction — the numeric-sign form
            // is consumed by `parse_negative` in `nud`, before any left operand.
            Some(
                tok @ (Token::Plus
                | Token::Minus
                | Token::Star
                | Token::Slash
                | Token::EqEq
                | Token::BangEq
                | Token::Lt
                | Token::LtEq
                | Token::Gt
                | Token::GtEq
                | Token::AmpAmp
                | Token::PipePipe),
            ) => {
                let op = match tok {
                    Token::Plus => BinOp::Add,
                    Token::Minus => BinOp::Sub,
                    Token::Star => BinOp::Mul,
                    Token::Slash => BinOp::Div,
                    Token::EqEq => BinOp::Eq,
                    Token::BangEq => BinOp::Ne,
                    Token::Lt => BinOp::Lt,
                    Token::LtEq => BinOp::Le,
                    Token::Gt => BinOp::Gt,
                    Token::GtEq => BinOp::Ge,
                    Token::AmpAmp => BinOp::And,
                    _ => BinOp::Or,
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
