//! Parser: `Token` stream → `Vec<ViewDecl>` (RFC-0002 §"Parser" + §"Grammar";
//! RFC-0003 §"Attribute syntax").
//!
//! Declarations, elements, and control flow are ordinary recursive descent —
//! one function per production, following the EBNF. Expressions use the Pratt
//! parser in [`expr`]. Error recovery records a [`CompileError`], substitutes an
//! [`Expr::Error`] (or skips one token at member level), and continues, so a
//! single pass collects multiple diagnostics (INV-4).

pub mod ast;
mod expr;

use ast::{Attr, AttrKind, ElementNode, Member, Param, StyleRule, Type, ViewDecl};

use crate::diagnostics::{CompileError, Span};
use crate::lexer::{SpannedToken, Token, lex};
use crate::symbol::Symbol;

/// The result of parsing a whole `.byd` file.
#[derive(Debug, Default)]
pub struct ParsedFile {
    /// The parsed views (D11: a file may declare several).
    pub views: Vec<ViewDecl>,
    /// All diagnostics from lexing and parsing, in source order.
    pub errors: Vec<CompileError>,
}

/// Parses `source` into a [`ParsedFile`].
#[must_use]
pub fn parse(source: &str) -> ParsedFile {
    let mut parser = Parser::new(source);
    let views = parser.parse_file();
    ParsedFile {
        views,
        errors: parser.errors,
    }
}

/// A recursive-descent + Pratt parser over a lexed token stream.
pub(crate) struct Parser<'a> {
    source: &'a str,
    tokens: Vec<SpannedToken>,
    pos: usize,
    errors: Vec<CompileError>,
}

impl<'a> Parser<'a> {
    /// Lexes `source` and prepares a parser positioned at the first token. Any
    /// lex diagnostics are carried forward into [`Parser::errors`].
    fn new(source: &'a str) -> Self {
        let lexed = lex(source);
        Self {
            source,
            tokens: lexed.tokens,
            pos: 0,
            errors: lexed.errors,
        }
    }

    // ---- token cursor ----

    fn cur(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|(t, _)| t)
    }

    fn peek2(&self) -> Option<Token> {
        self.tokens.get(self.pos + 1).map(|(t, _)| t.clone())
    }

    fn cur_span(&self) -> Span {
        self.tokens.get(self.pos).map_or_else(
            || {
                let n = self.source.len() as u32;
                Span::new(n, n)
            },
            |(_, s)| *s,
        )
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    /// Consumes the current token iff it equals `want` (for unit tokens).
    fn eat(&mut self, want: &Token) -> bool {
        if self.cur() == Some(want) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consumes `want`, or records an "expected `what`" diagnostic.
    fn expect(&mut self, want: &Token, what: &str) -> bool {
        if self.eat(want) {
            true
        } else {
            self.error(what);
            false
        }
    }

    /// Consumes and returns the current identifier, or records a diagnostic.
    fn expect_ident(&mut self, what: &str) -> Option<Symbol> {
        if let Some(Token::Ident(sym)) = self.cur() {
            let sym = sym.clone();
            self.advance();
            Some(sym)
        } else {
            self.error(what);
            None
        }
    }

    /// Consumes a *name*: an identifier or a contextual keyword used as one
    /// (e.g. `style` in `#[style: .title]`). Keywords are recovered from their
    /// source slice and interned.
    fn expect_name(&mut self, what: &str) -> Option<Symbol> {
        match self.cur() {
            Some(Token::Ident(sym)) => {
                let sym = sym.clone();
                self.advance();
                Some(sym)
            }
            Some(tok) if is_keyword(tok) => {
                let span = self.cur_span();
                let sym = Symbol::intern(&self.source[span.start as usize..span.end as usize]);
                self.advance();
                Some(sym)
            }
            _ => {
                self.error(what);
                None
            }
        }
    }

    /// The span from `start.start` to the end of the most recently consumed
    /// token.
    fn span_from(&self, start: Span) -> Span {
        let end = if self.pos > 0 {
            self.tokens
                .get(self.pos - 1)
                .map_or(start.end, |(_, s)| s.end)
        } else {
            start.end
        };
        Span::new(start.start, end.max(start.start))
    }

    fn error(&mut self, expected: &str) {
        let span = self.cur_span();
        self.error_at(span, expected);
    }

    fn error_at(&mut self, span: Span, expected: &str) {
        self.errors.push(CompileError::UnexpectedToken {
            span,
            expected: expected.to_string(),
        });
    }

    // ---- file / view ----

    fn parse_file(&mut self) -> Vec<ViewDecl> {
        let mut views = Vec::new();
        while self.cur().is_some() {
            let before = self.pos;
            if matches!(self.cur(), Some(Token::View)) {
                views.push(self.parse_view());
            } else {
                self.error("'View'");
            }
            if self.pos == before {
                self.advance(); // guaranteed progress on unrecognized input
            }
        }
        views
    }

    fn parse_view(&mut self) -> ViewDecl {
        let start = self.cur_span();
        self.advance(); // View
        let name = self
            .expect_ident("a view name")
            .unwrap_or_else(|| Symbol::intern(""));
        self.expect(&Token::LParen, "'('");
        let params = self.parse_params();
        self.expect(&Token::RParen, "')'");
        self.expect(&Token::LBrace, "'{'");
        let body = self.parse_block_members();
        ViewDecl {
            name,
            params,
            body,
            span: self.span_from(start),
        }
    }

    /// Parameters after a consumed `(`, up to (not including) `)`.
    fn parse_params(&mut self) -> Vec<Param> {
        let mut params = Vec::new();
        while !matches!(self.cur(), Some(Token::RParen) | None) {
            let start = self.cur_span();
            let name = self
                .expect_ident("a parameter name")
                .unwrap_or_else(|| Symbol::intern(""));
            let ty = if self.eat(&Token::Colon) {
                Some(self.parse_type())
            } else {
                None
            };
            // Optional default value (`= expr`), RFC-0007 D-B / IMPL-47.
            let default = if self.eat(&Token::Eq) {
                Some(self.parse_expr(0))
            } else {
                None
            };
            params.push(Param {
                name,
                ty,
                default,
                span: self.span_from(start),
            });
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        params
    }

    // ---- members ----

    /// Members up to and including the closing `}` (the opening `{` must have
    /// been consumed by the caller).
    fn parse_block_members(&mut self) -> Vec<Member> {
        let mut members = Vec::new();
        while !matches!(self.cur(), Some(Token::RBrace) | None) {
            let before = self.pos;
            if let Some(m) = self.parse_member() {
                members.push(m);
            }
            if self.pos == before {
                self.advance(); // recovery: skip the offending token
            }
        }
        self.expect(&Token::RBrace, "'}'");
        members
    }

    fn parse_member(&mut self) -> Option<Member> {
        match self.cur() {
            Some(Token::Var) => Some(self.parse_binding(true)),
            Some(Token::Let) => Some(self.parse_binding(false)),
            Some(Token::Fn) => Some(self.parse_fn()),
            Some(Token::Inject) => Some(self.parse_inject()),
            Some(Token::For) => Some(self.parse_for()),
            Some(Token::When) => Some(self.parse_when()),
            Some(Token::Style) => Some(self.parse_style_block()),
            Some(Token::Ident(_)) => Some(Member::Element(self.parse_element())),
            _ => {
                self.error("a declaration or element");
                None
            }
        }
    }

    /// `var`/`let` share a shape: `kw IDENT (":" type)? "=" expr`.
    fn parse_binding(&mut self, is_var: bool) -> Member {
        let start = self.cur_span();
        self.advance(); // var | let
        let name = self
            .expect_ident("a binding name")
            .unwrap_or_else(|| Symbol::intern(""));
        let ty = if self.eat(&Token::Colon) {
            Some(self.parse_type())
        } else {
            None
        };
        self.expect(&Token::Eq, "'='");
        let init = self.parse_expr(0);
        let span = self.span_from(start);
        if is_var {
            Member::Var {
                name,
                ty,
                init,
                span,
            }
        } else {
            Member::Let {
                name,
                ty,
                init,
                span,
            }
        }
    }

    /// `fn_decl := "fn" IDENT "(" param_list? ")" ("->" type)? "=>" expr`.
    fn parse_fn(&mut self) -> Member {
        let start = self.cur_span();
        self.advance(); // fn
        let name = self
            .expect_ident("a function name")
            .unwrap_or_else(|| Symbol::intern(""));
        self.expect(&Token::LParen, "'('");
        let params = self.parse_params();
        self.expect(&Token::RParen, "')'");
        let ret = if self.eat(&Token::ThinArrow) {
            Some(self.parse_type())
        } else {
            None
        };
        self.expect(&Token::Arrow, "'=>'");
        let body = self.parse_expr(0);
        Member::Fn {
            name,
            params,
            ret,
            body,
            span: self.span_from(start),
        }
    }

    /// `inject_stmt := "inject" type "as" IDENT`.
    fn parse_inject(&mut self) -> Member {
        let start = self.cur_span();
        self.advance(); // inject
        let ty = self.parse_type();
        self.expect(&Token::As, "'as'");
        let name = self
            .expect_ident("an inject binding name")
            .unwrap_or_else(|| Symbol::intern(""));
        Member::Inject {
            ty,
            name,
            span: self.span_from(start),
        }
    }

    /// `for_stmt := "for" IDENT "in" expr "{" member* "}"`.
    fn parse_for(&mut self) -> Member {
        let start = self.cur_span();
        self.advance(); // for
        let var = self
            .expect_ident("a loop variable")
            .unwrap_or_else(|| Symbol::intern(""));
        self.expect(&Token::In, "'in'");
        let iter = self.parse_expr(0);
        self.expect(&Token::LBrace, "'{'");
        let body = self.parse_block_members();
        Member::For {
            var,
            iter,
            body,
            span: self.span_from(start),
        }
    }

    /// `when_stmt := "when" expr "{" member* "}" ("else" "{" member* "}")?`.
    fn parse_when(&mut self) -> Member {
        let start = self.cur_span();
        self.advance(); // when
        let cond = self.parse_expr(0);
        self.expect(&Token::LBrace, "'{'");
        let then = self.parse_block_members();
        let els = if self.eat(&Token::Else) {
            self.expect(&Token::LBrace, "'{'");
            Some(self.parse_block_members())
        } else {
            None
        };
        Member::When {
            cond,
            then,
            els,
            span: self.span_from(start),
        }
    }

    /// `style_block := "style" "{" style_rule* "}"`,
    /// `style_rule := "." IDENT attr_block`.
    fn parse_style_block(&mut self) -> Member {
        let start = self.cur_span();
        self.advance(); // style
        self.expect(&Token::LBrace, "'{'");
        let mut rules = Vec::new();
        while !matches!(self.cur(), Some(Token::RBrace) | None) {
            let before = self.pos;
            let rstart = self.cur_span();
            self.expect(&Token::Dot, "'.' to start a style rule");
            let class = self
                .expect_ident("a class name")
                .unwrap_or_else(|| Symbol::intern(""));
            let attrs = if matches!(self.cur(), Some(Token::HashBracket)) {
                self.parse_attr_block()
            } else {
                Vec::new()
            };
            rules.push(StyleRule {
                class,
                attrs,
                span: self.span_from(rstart),
            });
            if self.pos == before {
                self.advance();
            }
        }
        self.expect(&Token::RBrace, "'}'");
        Member::Style {
            rules,
            span: self.span_from(start),
        }
    }

    // ---- elements & attributes ----

    /// `element := IDENT ("(" arg_list? ")")? attr_block? element_tail?`.
    fn parse_element(&mut self) -> ElementNode {
        let start = self.cur_span();
        let name = self
            .expect_ident("an element name")
            .unwrap_or_else(|| Symbol::intern(""));

        let content = if matches!(self.cur(), Some(Token::LParen)) {
            self.advance();
            let args = self.parse_arg_list(&Token::RParen);
            self.expect(&Token::RParen, "')'");
            args
        } else {
            Vec::new()
        };

        let attrs = if matches!(self.cur(), Some(Token::HashBracket)) {
            self.parse_attr_block()
        } else {
            Vec::new()
        };

        // element_tail := "{" member* "}" | "=>" expr
        let mut action = None;
        let mut children = Vec::new();
        if self.eat(&Token::Arrow) {
            action = Some(self.parse_expr(0));
        } else if self.eat(&Token::LBrace) {
            children = self.parse_block_members();
        }

        ElementNode {
            name,
            content,
            attrs,
            action,
            children,
            span: self.span_from(start),
        }
    }

    /// `attr_block := "#[" attr_list? "]"`.
    fn parse_attr_block(&mut self) -> Vec<Attr> {
        self.advance(); // #[
        let mut attrs = Vec::new();
        while !matches!(self.cur(), Some(Token::RBracket) | None) {
            let before = self.pos;
            if let Some(a) = self.parse_attr() {
                attrs.push(a);
            }
            if self.pos == before {
                self.advance();
            }
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect(&Token::RBracket, "']'");
        attrs
    }

    /// `attr := prop_attr | event_attr`, distinguished by the separator:
    /// `IDENT ("." IDENT)? ":" expr` (property, the optional sub-property
    /// axis is RFC-0011's `translate.y: 2`) vs `IDENT ("(" IDENT ")")? "=>"
    /// expr` (event — no sub-property form).
    fn parse_attr(&mut self) -> Option<Attr> {
        let start = self.cur_span();
        // `..style` spread (RFC-0016): no name — it splices a style value's
        // attributes into this list.
        if self.eat(&Token::DotDot) {
            let value = self.parse_expr(0);
            return Some(Attr {
                name: Symbol::intern(""),
                axis: None,
                kind: AttrKind::Spread { value },
                span: self.span_from(start),
            });
        }
        let name = self.expect_name("an attribute name")?;
        if self.eat(&Token::LParen) {
            let payload = self.expect_ident("an event payload name");
            self.expect(&Token::RParen, "')'");
            self.expect(&Token::Arrow, "'=>'");
            let action = self.parse_expr(0);
            Some(Attr {
                name,
                axis: None,
                kind: AttrKind::Event { payload, action },
                span: self.span_from(start),
            })
        } else if self.eat(&Token::Dot) {
            // Sub-property access (RFC-0011): `translate.y: 2`. Only the
            // property form takes an axis — an event never does.
            let axis = self.expect_name("a sub-property axis (e.g. `x`, `y`)")?;
            self.expect(&Token::Colon, "':'");
            let value = self.parse_expr(0);
            Some(Attr {
                name,
                axis: Some(axis),
                kind: AttrKind::Prop { value },
                span: self.span_from(start),
            })
        } else if self.eat(&Token::Colon) {
            let value = self.parse_expr(0);
            Some(Attr {
                name,
                axis: None,
                kind: AttrKind::Prop { value },
                span: self.span_from(start),
            })
        } else if self.eat(&Token::Arrow) {
            let action = self.parse_expr(0);
            Some(Attr {
                name,
                axis: None,
                kind: AttrKind::Event {
                    payload: None,
                    action,
                },
                span: self.span_from(start),
            })
        } else {
            self.error("':' (property) or '=>' (event)");
            None
        }
    }

    // ---- types ----

    /// `type := IDENT ("<" type ("," type)* ">")?`, plus the function type
    /// `Fn(type,*) ("->" type)?` (RFC-0003 E2). `Fn` lexes as a plain `Ident`,
    /// so it is recognized here by name.
    fn parse_type(&mut self) -> Type {
        let start = self.cur_span();
        let name = self
            .expect_ident("a type name")
            .unwrap_or_else(|| Symbol::intern(""));

        if name.as_str() == "Fn" && matches!(self.cur(), Some(Token::LParen)) {
            self.advance(); // (
            let mut params = Vec::new();
            while !matches!(self.cur(), Some(Token::RParen) | None) {
                params.push(self.parse_type());
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
            self.expect(&Token::RParen, "')'");
            let ret = if self.eat(&Token::ThinArrow) {
                Some(Box::new(self.parse_type()))
            } else {
                None
            };
            return Type::Function {
                params,
                ret,
                span: self.span_from(start),
            };
        }

        let mut args = Vec::new();
        if self.eat(&Token::Lt) {
            while !matches!(self.cur(), Some(Token::Gt) | None) {
                args.push(self.parse_type());
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
            self.expect(&Token::Gt, "'>'");
        }
        Type::Named {
            name,
            args,
            span: self.span_from(start),
        }
    }
}

/// Whether `tok` is a reserved keyword that may also be used as a contextual
/// name (an attribute, field, or argument label).
fn is_keyword(tok: &Token) -> bool {
    matches!(
        tok,
        Token::View
            | Token::Var
            | Token::Let
            | Token::Fn
            | Token::Inject
            | Token::As
            | Token::For
            | Token::In
            | Token::When
            | Token::Else
            | Token::Style
            | Token::Untrack
    )
}

#[cfg(test)]
mod tests;
