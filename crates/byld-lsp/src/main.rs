//! Production-grade LSP server for the Byld (.byd) language files.
//!
//! Provides document synchronization, hover details (for variables and intrinsics),
//! and compile/type-check diagnostics in real time.
//!
//! `byld-lsp` is an in-progress crate, out of the Phase-3 milestone scope (it
//! participates only as a diagnostics consumer). These pedantic allows keep the
//! workspace clippy gate honest for the in-scope crates without churning WIP code.
#![allow(
    clippy::needless_pass_by_value,
    clippy::mutable_key_type,
    clippy::explicit_counter_loop,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::only_used_in_recursion,
    clippy::format_push_string
)]

use std::collections::HashMap;

use byard_compiler::diagnostics::Span;
use byard_compiler::interp::eval::Interpreter;
use byard_compiler::parser::ast::{AttrKind, Expr, Member, StrPart, ViewDecl};
use byard_compiler::parser::parse;
use lsp_server::{Connection, Message, Notification, Request, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification as _,
    PublishDiagnostics,
};
use lsp_types::request::{Completion, GotoDefinition, HoverRequest, Request as _};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionResponse, Diagnostic, DiagnosticSeverity,
    GotoDefinitionResponse, Hover, HoverContents, HoverProviderCapability, MarkupContent,
    MarkupKind, OneOf, Position, Range, ServerCapabilities, TextDocumentSyncCapability,
    TextDocumentSyncKind, Uri,
};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Set up communication channel
    let (connection, io_threads) = Connection::stdio();

    // Register capabilities
    let server_capabilities = serde_json::to_value(&ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        completion_provider: Some(lsp_types::CompletionOptions {
            resolve_provider: Some(false),
            trigger_characters: Some(vec![
                "#".to_string(),
                "[".to_string(),
                ":".to_string(),
                ".".to_string(),
                " ".to_string(),
            ]),
            ..Default::default()
        }),
        definition_provider: Some(OneOf::Left(true)),
        ..Default::default()
    })?;

    let initialization_params = connection.initialize(server_capabilities)?;
    main_loop(connection, initialization_params)?;

    io_threads.join()?;
    Ok(())
}

fn main_loop(
    connection: Connection,
    _params: serde_json::Value,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut documents: HashMap<Uri, String> = HashMap::new();

    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    return Ok(());
                }

                if req.method == HoverRequest::METHOD {
                    if let Ok((id, params)) = cast_request::<HoverRequest>(req) {
                        let uri = params.text_document_position_params.text_document.uri;
                        let pos = params.text_document_position_params.position;

                        let response = if let Some(content) = documents.get(&uri) {
                            let hover_info = handle_hover(content, pos);
                            Response::new_ok(id, hover_info)
                        } else {
                            Response::new_ok(id, None::<Hover>)
                        };
                        let _ = connection.sender.send(Message::Response(response));
                    }
                } else if req.method == Completion::METHOD {
                    if let Ok((id, params)) = cast_request::<Completion>(req) {
                        let uri = params.text_document_position.text_document.uri;
                        let pos = params.text_document_position.position;

                        let response = if let Some(content) = documents.get(&uri) {
                            let completion_info = handle_completion(content, pos);
                            Response::new_ok(id, completion_info)
                        } else {
                            Response::new_ok(id, None::<CompletionResponse>)
                        };
                        let _ = connection.sender.send(Message::Response(response));
                    }
                } else if req.method == GotoDefinition::METHOD {
                    if let Ok((id, params)) = cast_request::<GotoDefinition>(req) {
                        let uri = params
                            .text_document_position_params
                            .text_document
                            .uri
                            .clone();
                        let pos = params.text_document_position_params.position;

                        let response = if let Some(content) = documents.get(&uri) {
                            let definition_info = handle_definition(content, pos, uri);
                            Response::new_ok(id, definition_info)
                        } else {
                            Response::new_ok(id, None::<GotoDefinitionResponse>)
                        };
                        let _ = connection.sender.send(Message::Response(response));
                    }
                }
            }
            Message::Notification(not) => {
                if not.method == DidOpenTextDocument::METHOD {
                    if let Ok(params) = cast_notification::<DidOpenTextDocument>(not) {
                        let uri = params.text_document.uri;
                        let text = params.text_document.text;
                        validate_and_publish(&connection, uri.clone(), &text)?;
                        documents.insert(uri, text);
                    }
                } else if not.method == DidChangeTextDocument::METHOD {
                    if let Ok(params) = cast_notification::<DidChangeTextDocument>(not) {
                        let uri = params.text_document.uri;
                        if let Some(change) = params.content_changes.into_iter().next() {
                            validate_and_publish(&connection, uri.clone(), &change.text)?;
                            documents.insert(uri, change.text);
                        }
                    }
                } else if not.method == DidCloseTextDocument::METHOD {
                    if let Ok(params) = cast_notification::<DidCloseTextDocument>(not) {
                        let uri = params.text_document.uri;
                        // Clear diagnostics on document close
                        let clear_params = lsp_types::PublishDiagnosticsParams {
                            uri: uri.clone(),
                            diagnostics: Vec::new(),
                            version: None,
                        };
                        let notification =
                            Notification::new(PublishDiagnostics::METHOD.to_string(), clear_params);
                        let _ = connection.sender.send(Message::Notification(notification));
                        documents.remove(&uri);
                    }
                }
            }
            Message::Response(_) => {}
        }
    }

    Ok(())
}

fn cast_request<R>(req: Request) -> Result<(lsp_server::RequestId, R::Params), Request>
where
    R: lsp_types::request::Request,
    R::Params: serde::de::DeserializeOwned,
{
    if req.method == R::METHOD {
        let id = req.id.clone();
        match serde_json::from_value(req.params) {
            Ok(params) => Ok((id, params)),
            Err(_) => Err(Request {
                id,
                method: req.method,
                params: serde_json::Value::Null,
            }),
        }
    } else {
        Err(req)
    }
}

fn cast_notification<N>(not: Notification) -> Result<N::Params, Notification>
where
    N: lsp_types::notification::Notification,
    N::Params: serde::de::DeserializeOwned,
{
    if not.method == N::METHOD {
        match serde_json::from_value(not.params) {
            Ok(params) => Ok(params),
            Err(_) => Err(Notification {
                method: not.method,
                params: serde_json::Value::Null,
            }),
        }
    } else {
        Err(not)
    }
}

fn validate_and_publish(
    connection: &Connection,
    uri: Uri,
    content: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let parsed = parse(content);
    let mut errors = parsed.errors;

    // Type checking
    let inference = byard_compiler::infer::check_views(&parsed.views);
    errors.extend(inference.errors);

    // Element & Intrinsic validation
    let mut interp = Interpreter::new();
    interp.load_views(&parsed.views);
    let known_views: Vec<&str> = parsed.views.iter().map(|v| v.name.as_str()).collect();
    for view in &parsed.views {
        let _ = interp.lower_view(view, &known_views);
    }
    errors.extend(interp.errors().iter().cloned());

    let mut diagnostics = Vec::new();
    for err in errors {
        let span = err.span();
        let range = Range::new(
            byte_offset_to_lsp_pos(content, span.start as usize),
            byte_offset_to_lsp_pos(content, span.end as usize),
        );
        diagnostics.push(Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("byld-compiler".to_string()),
            message: err.headline(),
            related_information: None,
            tags: None,
            data: None,
        });
    }

    let params = lsp_types::PublishDiagnosticsParams {
        uri,
        diagnostics,
        version: None,
    };

    let notification = Notification::new(PublishDiagnostics::METHOD.to_string(), params);
    connection
        .sender
        .send(Message::Notification(notification))?;
    Ok(())
}

fn lsp_pos_to_byte_offset(source: &str, line: u32, character: u32) -> Option<usize> {
    let mut offset = 0;
    for (i, current_line) in source.lines().enumerate() {
        if i == line as usize {
            let mut char_count = 0;
            for (byte_idx, _) in current_line.char_indices() {
                if char_count == character as usize {
                    return Some(offset + byte_idx);
                }
                char_count += 1;
            }
            return Some(offset + current_line.len());
        }
        offset += current_line.len() + 1; // +1 for the newline
    }
    None
}

fn byte_offset_to_lsp_pos(source: &str, offset: usize) -> Position {
    let offset = offset.min(source.len());
    let mut line = 0;
    let mut char_idx = 0;
    let mut current_offset = 0;
    for c in source.chars() {
        if current_offset >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            char_idx = 0;
        } else {
            char_idx += 1;
        }
        current_offset += c.len_utf8();
    }
    Position::new(line, char_idx)
}

#[allow(dead_code)]
enum HoverTarget {
    Intrinsic {
        name: String,
    },
    Attribute {
        element_name: String,
        attr_name: String,
    },
    VarIdent {
        name: String,
        span: Span,
    },
}

fn handle_hover(content: &str, pos: Position) -> Option<Hover> {
    let offset = lsp_pos_to_byte_offset(content, pos.line, pos.character)?;
    let parsed = parse(content);

    let target = find_hover_target(&parsed.views, offset)?;
    let docs = match target {
        HoverTarget::Intrinsic { name } => intrinsic_hover_docs(&name)?,
        HoverTarget::Attribute {
            element_name,
            attr_name,
        } => attribute_hover_docs(&element_name, &attr_name)?,
        HoverTarget::VarIdent { name, .. } => {
            let inference = byard_compiler::infer::check_views(&parsed.views);
            var_hover_docs(&name, &inference)
        }
    };

    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: docs,
        }),
        range: None,
    })
}

fn find_hover_target(views: &[ViewDecl], offset: usize) -> Option<HoverTarget> {
    for view in views {
        if !span_contains(view.span, offset) {
            continue;
        }
        for param in &view.params {
            if span_contains(param.span, offset) {
                return Some(HoverTarget::VarIdent {
                    name: param.name.to_string(),
                    span: param.span,
                });
            }
        }
        if let Some(target) = find_in_members(&view.body, offset, None) {
            return Some(target);
        }
    }
    None
}

fn span_contains(span: Span, offset: usize) -> bool {
    offset >= span.start as usize && offset < span.end as usize
}

fn find_in_members(
    members: &[Member],
    offset: usize,
    parent_element: Option<&str>,
) -> Option<HoverTarget> {
    for member in members {
        match member {
            Member::Var {
                name, init, span, ..
            } if span_contains(*span, offset) => {
                if offset >= span.start as usize
                    && offset < span.start as usize + name.as_str().len()
                {
                    return Some(HoverTarget::VarIdent {
                        name: name.to_string(),
                        span: *span,
                    });
                }
                return find_in_expr(init, offset);
            }
            Member::Let {
                name, init, span, ..
            } if span_contains(*span, offset) => {
                if offset >= span.start as usize
                    && offset < span.start as usize + name.as_str().len()
                {
                    return Some(HoverTarget::VarIdent {
                        name: name.to_string(),
                        span: *span,
                    });
                }
                return find_in_expr(init, offset);
            }
            Member::Fn {
                name,
                params,
                body,
                span,
                ..
            } if span_contains(*span, offset) => {
                if offset >= span.start as usize
                    && offset < span.start as usize + name.as_str().len()
                {
                    return Some(HoverTarget::VarIdent {
                        name: name.to_string(),
                        span: *span,
                    });
                }
                for param in params {
                    if span_contains(param.span, offset) {
                        return Some(HoverTarget::VarIdent {
                            name: param.name.to_string(),
                            span: param.span,
                        });
                    }
                }
                if span_contains(body.span(), offset) {
                    return find_in_expr(body, offset);
                }
            }
            Member::Element(el) if span_contains(el.span, offset) => {
                let name_start = el.span.start as usize;
                let name_end = name_start + el.name.as_str().len();
                if offset >= name_start && offset < name_end {
                    return Some(HoverTarget::Intrinsic {
                        name: el.name.to_string(),
                    });
                }
                for arg in &el.content {
                    if let Some(target) = find_in_expr(&arg.value, offset) {
                        return Some(target);
                    }
                }
                for attr in &el.attrs {
                    if span_contains(attr.span, offset) {
                        let attr_name_start = attr.span.start as usize;
                        let attr_name_end = attr_name_start + attr.name.as_str().len();
                        if offset >= attr_name_start && offset < attr_name_end {
                            return Some(HoverTarget::Attribute {
                                element_name: el.name.to_string(),
                                attr_name: attr.name.to_string(),
                            });
                        }
                        match &attr.kind {
                            AttrKind::Prop { value } => {
                                if let Some(target) = find_in_expr(value, offset) {
                                    return Some(target);
                                }
                            }
                            AttrKind::Event { action, .. } => {
                                if let Some(target) = find_in_expr(action, offset) {
                                    return Some(target);
                                }
                            }
                        }
                    }
                }
                if let Some(action) = &el.action {
                    if let Some(target) = find_in_expr(action, offset) {
                        return Some(target);
                    }
                }
                if let Some(target) = find_in_members(&el.children, offset, Some(el.name.as_str()))
                {
                    return Some(target);
                }
            }
            Member::For {
                var,
                iter,
                body,
                span,
            } if span_contains(*span, offset) => {
                let var_start = span.start as usize + 4; // after "for "
                let var_end = var_start + var.as_str().len();
                if offset >= var_start && offset < var_end {
                    return Some(HoverTarget::VarIdent {
                        name: var.to_string(),
                        span: Span::new(var_start as u32, var_end as u32),
                    });
                }
                if let Some(target) = find_in_expr(iter, offset) {
                    return Some(target);
                }
                if let Some(target) = find_in_members(body, offset, parent_element) {
                    return Some(target);
                }
            }
            Member::When {
                cond,
                then,
                els,
                span,
            } if span_contains(*span, offset) => {
                if let Some(target) = find_in_expr(cond, offset) {
                    return Some(target);
                }
                if let Some(target) = find_in_members(then, offset, parent_element) {
                    return Some(target);
                }
                if let Some(els_members) = els {
                    if let Some(target) = find_in_members(els_members, offset, parent_element) {
                        return Some(target);
                    }
                }
            }
            Member::Expr(expr) => {
                if let Some(target) = find_in_expr(expr, offset) {
                    return Some(target);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_in_expr(expr: &Expr, offset: usize) -> Option<HoverTarget> {
    if !span_contains(expr.span(), offset) {
        return None;
    }
    match expr {
        Expr::Ident(sym, span) => Some(HoverTarget::VarIdent {
            name: sym.to_string(),
            span: *span,
        }),
        Expr::Array(items, _) => {
            for item in items {
                if let Some(target) = find_in_expr(item, offset) {
                    return Some(target);
                }
            }
            None
        }
        Expr::Tuple(args, _) => {
            for arg in args {
                if let Some(target) = find_in_expr(&arg.value, offset) {
                    return Some(target);
                }
            }
            None
        }
        Expr::Member { base, field, span } => {
            if span_contains(base.span(), offset) {
                return find_in_expr(base, offset);
            }
            let field_start = span.end as usize - field.as_str().len();
            if offset >= field_start && offset < span.end as usize {
                return Some(HoverTarget::VarIdent {
                    name: field.to_string(),
                    span: Span::new(field_start as u32, span.end),
                });
            }
            None
        }
        Expr::Call { callee, args, .. } => {
            if span_contains(callee.span(), offset) {
                return find_in_expr(callee, offset);
            }
            for arg in args {
                if span_contains(arg.value.span(), offset) {
                    return find_in_expr(&arg.value, offset);
                }
            }
            None
        }
        Expr::Lambda { body, .. } => {
            if span_contains(body.span(), offset) {
                return find_in_expr(body, offset);
            }
            None
        }
        Expr::StrLit(parts, _) => {
            for part in parts {
                if let StrPart::Interp(expr) = part {
                    if let Some(target) = find_in_expr(expr, offset) {
                        return Some(target);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn intrinsic_hover_docs(name: &str) -> Option<String> {
    let info = byard_compiler::interp::intrinsics::lookup(name)?;
    let mut doc = format!("### Intrinsic `{name}`\n\n");
    if info.children {
        doc.push_str("- **Children**: Yes (expects `{ ... }`)\n");
    } else {
        doc.push_str("- **Children**: No\n");
    }
    if info.focusable {
        doc.push_str("- **Focusable**: Yes\n");
    }
    if info.interactive {
        doc.push_str("- **Interactive**: Yes\n");
    }
    doc.push_str("\n#### Accepted Properties:\n");
    let mut props: Vec<_> = info.properties().collect();
    props.sort_by_key(|(k, _)| *k);
    for (prop, ty) in props {
        doc.push_str(&format!("* `{prop}`: `{ty:?}`\n"));
    }
    doc.push_str("\n#### Accepted Events:\n");
    let mut events: Vec<_> = info.events().collect();
    events.sort_unstable();
    for event in events {
        doc.push_str(&format!("* `{event}`\n"));
    }
    Some(doc)
}

fn attribute_hover_docs(element_name: &str, attr_name: &str) -> Option<String> {
    let info = byard_compiler::interp::intrinsics::lookup(element_name)?;
    let mut doc = format!("### Attribute `{attr_name}` on `{element_name}`\n\n");
    if let Some(prop_ty) = info.property_type(attr_name) {
        doc.push_str(&format!("Type: Property (`{prop_ty:?}`)\n"));
    } else if info.has_event(attr_name) {
        doc.push_str("Type: Event Callback (`=>`)\n");
    } else {
        return None;
    }
    Some(doc)
}

fn var_hover_docs(name: &str, inference: &byard_compiler::infer::Inference) -> String {
    let var_symbol = byard_compiler::Symbol::intern(name);
    if let Some((_, ty)) = inference
        .bindings
        .iter()
        .find(|(sym, _)| *sym == var_symbol)
    {
        format!("```byld\nvar {name}: {}\n```", format_ty(ty))
    } else {
        format!("```byld\nvar {name}\n```")
    }
}

fn format_ty(ty: &byard_compiler::infer::Ty) -> String {
    use byard_compiler::infer::Ty;
    match ty {
        Ty::Int => "Int".to_string(),
        Ty::Float => "Float".to_string(),
        Ty::Bool => "Bool".to_string(),
        Ty::Str => "Str".to_string(),
        Ty::List(inner) => format!("List<{}>", format_ty(inner)),
        Ty::Fn(params, ret) => {
            let params_str: Vec<String> = params.iter().map(format_ty).collect();
            let ret_str = match ret {
                Some(r) => format!(" -> {}", format_ty(r)),
                None => String::new(),
            };
            format!("Fn({}){}", params_str.join(", "), ret_str)
        }
        Ty::Named(sym) => sym.as_str().to_string(),
        Ty::Unknown => "Unknown".to_string(),
    }
}

const STYLE_PROPS: &[(&str, &str)] = &[
    ("width", "Int (logical pixels)"),
    ("height", "Int (logical pixels)"),
    ("gap", "Int (logical pixels spacing)"),
    ("p", "Len (padding on all sides)"),
    ("m", "Len (margin on all sides)"),
    ("px", "Len (horizontal padding)"),
    ("py", "Len (vertical padding)"),
    ("pt", "Len (top padding)"),
    ("pr", "Len (right padding)"),
    ("pb", "Len (bottom padding)"),
    ("pl", "Len (left padding)"),
    ("mx", "Len (horizontal margin)"),
    ("my", "Len (vertical margin)"),
    ("mt", "Len (top margin)"),
    ("mr", "Len (right margin)"),
    ("mb", "Len (bottom margin)"),
    ("ml", "Len (left margin)"),
    ("align", "Enum: start | center | end | stretch | justify"),
    (
        "justify",
        "Enum: start | center | end | between | around | evenly",
    ),
    ("grow", "Int (flex grow factor)"),
    ("basis", "Int (flex basis size)"),
    ("bg", "Color (background color hex)"),
    ("radius", "Len (border radius)"),
    ("opacity", "Float (0.0 to 1.0)"),
    ("border", "Color (border color)"),
    ("shadow", "Str (shadow specification)"),
];

#[derive(Debug, Clone, PartialEq, Eq)]
enum AttributeContext {
    StyleRule,
    Element(String),
}

fn is_style_rule_before_attr(content: &str, attr_start: usize) -> bool {
    let bytes = content.as_bytes();
    let mut i = attr_start;
    while i > 0 && bytes[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    while i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_') {
        i -= 1;
    }
    if i > 0 && bytes[i - 1] == b'.' {
        return true;
    }
    false
}

fn find_element_name_before_attr(content: &str, attr_start: usize) -> Option<String> {
    let bytes = content.as_bytes();
    let mut i = attr_start;
    while i > 0 && bytes[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    if i == 0 {
        return None;
    }
    if bytes[i - 1] == b')' {
        i -= 1;
        let mut paren_depth = 1;
        while i > 0 && paren_depth > 0 {
            i -= 1;
            if i >= bytes.len() {
                break;
            }
            if bytes[i] == b')' {
                paren_depth += 1;
            } else if bytes[i] == b'(' {
                paren_depth -= 1;
            }
        }
        while i > 0 && bytes[i - 1].is_ascii_whitespace() {
            i -= 1;
        }
    }
    if i == 0 {
        return None;
    }
    let name_end = i;
    while i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_') {
        i -= 1;
    }
    if i < name_end {
        let name = &content[i..name_end];
        if name.starts_with(|c: char| c.is_ascii_alphabetic()) {
            return Some(name.to_string());
        }
    }
    None
}

fn find_active_attribute_context(content: &str, offset: usize) -> Option<AttributeContext> {
    let mut bracket_depth = 0;
    let mut found_attr_start = None;
    let mut i = offset;
    let bytes = content.as_bytes();
    while i > 0 {
        i -= 1;
        if i >= bytes.len() {
            continue;
        }
        let c = bytes[i];
        if c == b']' {
            bracket_depth += 1;
        } else if c == b'[' && i > 0 && bytes[i - 1] == b'#' {
            if bracket_depth == 0 {
                found_attr_start = Some(i - 1);
                break;
            }
            bracket_depth -= 1;
        }
    }

    let start_offset = found_attr_start?;

    if is_style_rule_before_attr(content, start_offset) {
        return Some(AttributeContext::StyleRule);
    }

    if let Some(el_name) = find_element_name_before_attr(content, start_offset) {
        return Some(AttributeContext::Element(el_name));
    }

    let parsed = parse(content);
    for view in &parsed.views {
        if !span_contains(view.span, start_offset) {
            continue;
        }
        for member in &view.body {
            if let Member::Style { rules, span } = member {
                if span_contains(*span, start_offset) {
                    for rule in rules {
                        if span_contains(rule.span, start_offset) {
                            return Some(AttributeContext::StyleRule);
                        }
                    }
                }
            }
        }
        if let Some(name) = find_element_at_offset(&view.body, start_offset) {
            return Some(AttributeContext::Element(name));
        }
    }
    None
}

fn find_element_at_offset(members: &[Member], offset: usize) -> Option<String> {
    for member in members {
        match member {
            Member::Element(el) if span_contains(el.span, offset) => {
                if let Some(child_name) = find_element_at_offset(&el.children, offset) {
                    return Some(child_name);
                }
                return Some(el.name.to_string());
            }
            Member::For { body, span, .. } if span_contains(*span, offset) => {
                if let Some(name) = find_element_at_offset(body, offset) {
                    return Some(name);
                }
            }
            Member::When {
                then, els, span, ..
            } if span_contains(*span, offset) => {
                if let Some(name) = find_element_at_offset(then, offset) {
                    return Some(name);
                }
                if let Some(els_members) = els {
                    if let Some(name) = find_element_at_offset(els_members, offset) {
                        return Some(name);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn collect_locals_in_members(
    members: &[Member],
    offset: usize,
    locals: &mut Vec<(String, CompletionItemKind, String)>,
) {
    for member in members {
        match member {
            Member::Var { name, .. } => {
                locals.push((
                    name.to_string(),
                    CompletionItemKind::VARIABLE,
                    "Variable (Signal)".to_string(),
                ));
            }
            Member::Let { name, .. } => {
                locals.push((
                    name.to_string(),
                    CompletionItemKind::VARIABLE,
                    "Computed Value (Memo)".to_string(),
                ));
            }
            Member::Fn { name, .. } => {
                locals.push((
                    name.to_string(),
                    CompletionItemKind::FUNCTION,
                    "Helper Function".to_string(),
                ));
            }
            Member::Inject { name, ty, .. } => {
                locals.push((
                    name.to_string(),
                    CompletionItemKind::VARIABLE,
                    format!("Injected Value: {ty:?}"),
                ));
            }
            Member::For {
                var, body, span, ..
            } if span_contains(*span, offset) => {
                locals.push((
                    var.to_string(),
                    CompletionItemKind::VARIABLE,
                    "Loop Item".to_string(),
                ));
                collect_locals_in_members(body, offset, locals);
            }
            Member::When {
                then, els, span, ..
            } if span_contains(*span, offset) => {
                collect_locals_in_members(then, offset, locals);
                if let Some(els_members) = els {
                    collect_locals_in_members(els_members, offset, locals);
                }
            }
            Member::Element(el) if span_contains(el.span, offset) => {
                collect_locals_in_members(&el.children, offset, locals);
            }
            _ => {}
        }
    }
}

fn handle_completion(content: &str, pos: Position) -> Option<CompletionResponse> {
    let offset = lsp_pos_to_byte_offset(content, pos.line, pos.character)?;

    if let Some(attr_context) = find_active_attribute_context(content, offset) {
        let mut items = Vec::new();
        match attr_context {
            AttributeContext::StyleRule => {
                for &(prop, desc) in STYLE_PROPS {
                    items.push(CompletionItem {
                        label: prop.to_string(),
                        kind: Some(CompletionItemKind::PROPERTY),
                        detail: Some(desc.to_string()),
                        insert_text: Some(format!("{prop}: ")),
                        ..Default::default()
                    });
                }
            }
            AttributeContext::Element(el_name) => {
                if let Some(info) = byard_compiler::interp::intrinsics::lookup(&el_name) {
                    for (prop, ty) in info.properties() {
                        items.push(CompletionItem {
                            label: prop.to_string(),
                            kind: Some(CompletionItemKind::PROPERTY),
                            detail: Some(format!("Property ({ty:?})")),
                            insert_text: Some(format!("{prop}: ")),
                            ..Default::default()
                        });
                    }
                    for event in info.events() {
                        items.push(CompletionItem {
                            label: event.to_string(),
                            kind: Some(CompletionItemKind::EVENT),
                            detail: Some("Event Callback".to_string()),
                            insert_text: Some(format!("{event} => ")),
                            ..Default::default()
                        });
                    }
                } else {
                    let parsed = parse(content);
                    if let Some(view) = parsed.views.iter().find(|v| v.name.as_str() == el_name) {
                        for param in &view.params {
                            items.push(CompletionItem {
                                label: param.name.to_string(),
                                kind: Some(CompletionItemKind::PROPERTY),
                                detail: Some("View Parameter".to_string()),
                                insert_text: Some(format!("{}: ", param.name)),
                                ..Default::default()
                            });
                        }
                    }
                }
            }
        }
        return Some(CompletionResponse::Array(items));
    }

    let mut items = Vec::new();

    let keywords = &[
        "var", "let", "fn", "inject", "for", "in", "when", "else", "style",
    ];
    for kw in keywords {
        items.push(CompletionItem {
            label: kw.to_string(),
            kind: Some(CompletionItemKind::KEYWORD),
            ..Default::default()
        });
    }

    for intrinsic in byard_compiler::interp::intrinsics::INTRINSIC_NAMES {
        items.push(CompletionItem {
            label: intrinsic.to_string(),
            kind: Some(CompletionItemKind::CLASS),
            detail: Some("Intrinsic Component".to_string()),
            ..Default::default()
        });
    }

    let parsed = parse(content);

    for view in &parsed.views {
        items.push(CompletionItem {
            label: view.name.to_string(),
            kind: Some(CompletionItemKind::INTERFACE),
            detail: Some("User View".to_string()),
            ..Default::default()
        });
    }

    if let Some(active_view) = parsed.views.iter().find(|v| span_contains(v.span, offset)) {
        let mut locals = Vec::new();
        for param in &active_view.params {
            locals.push((
                param.name.to_string(),
                CompletionItemKind::VARIABLE,
                "Parameter".to_string(),
            ));
        }
        collect_locals_in_members(&active_view.body, offset, &mut locals);

        for (name, kind, detail) in locals {
            items.push(CompletionItem {
                label: name,
                kind: Some(kind),
                detail: Some(detail),
                ..Default::default()
            });
        }
    }

    Some(CompletionResponse::Array(items))
}

fn find_element_ref_at_offset(views: &[ViewDecl], offset: usize) -> Option<String> {
    for view in views {
        if !span_contains(view.span, offset) {
            continue;
        }
        if let Some(name) = find_element_ref_in_members(&view.body, offset) {
            return Some(name);
        }
    }
    None
}

fn find_element_ref_in_members(members: &[Member], offset: usize) -> Option<String> {
    for member in members {
        match member {
            Member::Element(el) => {
                let name_len = el.name.as_str().len();
                let name_span = Span::new(el.span.start, el.span.start + name_len as u32);
                if span_contains(name_span, offset) {
                    return Some(el.name.to_string());
                }
                if let Some(name) = find_element_ref_in_members(&el.children, offset) {
                    return Some(name);
                }
            }
            Member::For { body, span, .. } if span_contains(*span, offset) => {
                if let Some(name) = find_element_ref_in_members(body, offset) {
                    return Some(name);
                }
            }
            Member::When {
                then, els, span, ..
            } if span_contains(*span, offset) => {
                if let Some(name) = find_element_ref_in_members(then, offset) {
                    return Some(name);
                }
                if let Some(els_members) = els {
                    if let Some(name) = find_element_ref_in_members(els_members, offset) {
                        return Some(name);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn find_local_declaration_span(members: &[Member], var_name: &str, offset: usize) -> Option<Span> {
    for member in members {
        match member {
            Member::Var { name, span, .. } if name.as_str() == var_name => {
                return Some(*span);
            }
            Member::Let { name, span, .. } if name.as_str() == var_name => {
                return Some(*span);
            }
            Member::Fn { name, span, .. } if name.as_str() == var_name => {
                return Some(*span);
            }
            Member::Inject { name, span, .. } if name.as_str() == var_name => {
                return Some(*span);
            }
            Member::For {
                var, body, span, ..
            } if span_contains(*span, offset) => {
                if var.as_str() == var_name {
                    return Some(Span::new(
                        span.start,
                        span.start + 4 + var.as_str().len() as u32,
                    ));
                }
                if let Some(s) = find_local_declaration_span(body, var_name, offset) {
                    return Some(s);
                }
            }
            Member::When {
                then, els, span, ..
            } if span_contains(*span, offset) => {
                if let Some(s) = find_local_declaration_span(then, var_name, offset) {
                    return Some(s);
                }
                if let Some(els_members) = els {
                    if let Some(s) = find_local_declaration_span(els_members, var_name, offset) {
                        return Some(s);
                    }
                }
            }
            Member::Element(el) if span_contains(el.span, offset) => {
                if let Some(s) = find_local_declaration_span(&el.children, var_name, offset) {
                    return Some(s);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_class_ref_at_offset(views: &[ViewDecl], offset: usize) -> Option<String> {
    for view in views {
        if !span_contains(view.span, offset) {
            continue;
        }
        if let Some(name) = find_class_ref_in_members(&view.body, offset) {
            return Some(name);
        }
    }
    None
}

fn find_class_ref_in_members(members: &[Member], offset: usize) -> Option<String> {
    for member in members {
        match member {
            Member::Element(el) => {
                for attr in &el.attrs {
                    match &attr.kind {
                        AttrKind::Prop { value } => {
                            if let Some(name) = find_class_ref_in_expr(value, offset) {
                                return Some(name);
                            }
                        }
                        AttrKind::Event { action, .. } => {
                            if let Some(name) = find_class_ref_in_expr(action, offset) {
                                return Some(name);
                            }
                        }
                    }
                }
                if let Some(action) = &el.action {
                    if let Some(name) = find_class_ref_in_expr(action, offset) {
                        return Some(name);
                    }
                }
                if let Some(name) = find_class_ref_in_members(&el.children, offset) {
                    return Some(name);
                }
            }
            Member::Var { init, .. } | Member::Let { init, .. } => {
                if let Some(name) = find_class_ref_in_expr(init, offset) {
                    return Some(name);
                }
            }
            Member::Fn { body, .. } => {
                if let Some(name) = find_class_ref_in_expr(body, offset) {
                    return Some(name);
                }
            }
            Member::For {
                var: _,
                iter,
                body,
                span,
            } if span_contains(*span, offset) => {
                if let Some(name) = find_class_ref_in_expr(iter, offset) {
                    return Some(name);
                }
                if let Some(name) = find_class_ref_in_members(body, offset) {
                    return Some(name);
                }
            }
            Member::When {
                cond,
                then,
                els,
                span,
            } if span_contains(*span, offset) => {
                if let Some(name) = find_class_ref_in_expr(cond, offset) {
                    return Some(name);
                }
                if let Some(name) = find_class_ref_in_members(then, offset) {
                    return Some(name);
                }
                if let Some(els_members) = els {
                    if let Some(name) = find_class_ref_in_members(els_members, offset) {
                        return Some(name);
                    }
                }
            }
            Member::Expr(expr) => {
                if let Some(name) = find_class_ref_in_expr(expr, offset) {
                    return Some(name);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_class_ref_in_expr(expr: &Expr, offset: usize) -> Option<String> {
    if !span_contains(expr.span(), offset) {
        return None;
    }
    match expr {
        Expr::ClassRef(sym, span) => {
            if span_contains(*span, offset) {
                return Some(sym.as_str().to_string());
            }
            None
        }
        Expr::Array(items, _) => {
            for item in items {
                if let Some(name) = find_class_ref_in_expr(item, offset) {
                    return Some(name);
                }
            }
            None
        }
        Expr::Tuple(args, _) => {
            for arg in args {
                if let Some(name) = find_class_ref_in_expr(&arg.value, offset) {
                    return Some(name);
                }
            }
            None
        }
        Expr::Member { base, .. } => find_class_ref_in_expr(base, offset),
        Expr::Call { callee, args, .. } => {
            if let Some(name) = find_class_ref_in_expr(callee, offset) {
                return Some(name);
            }
            for arg in args {
                if let Some(name) = find_class_ref_in_expr(&arg.value, offset) {
                    return Some(name);
                }
            }
            None
        }
        Expr::Lambda { body, .. } => find_class_ref_in_expr(body, offset),
        Expr::Assign { target, value, .. } => {
            if let Some(name) = find_class_ref_in_expr(target, offset) {
                return Some(name);
            }
            find_class_ref_in_expr(value, offset)
        }
        Expr::Postfix { target, .. } => find_class_ref_in_expr(target, offset),
        Expr::Ternary {
            cond, then, els, ..
        } => {
            if let Some(name) = find_class_ref_in_expr(cond, offset) {
                return Some(name);
            }
            if let Some(name) = find_class_ref_in_expr(then, offset) {
                return Some(name);
            }
            find_class_ref_in_expr(els, offset)
        }
        _ => None,
    }
}

fn handle_definition(
    content: &str,
    pos: Position,
    uri: Uri,
) -> Option<lsp_types::GotoDefinitionResponse> {
    let offset = lsp_pos_to_byte_offset(content, pos.line, pos.character)?;
    let parsed = parse(content);

    if let Some(view_ref_name) = find_element_ref_at_offset(&parsed.views, offset) {
        if let Some(target_view) = parsed
            .views
            .iter()
            .find(|v| v.name.as_str() == view_ref_name)
        {
            let range = Range::new(
                byte_offset_to_lsp_pos(content, target_view.span.start as usize),
                byte_offset_to_lsp_pos(content, target_view.span.end as usize),
            );
            return Some(lsp_types::GotoDefinitionResponse::Scalar(
                lsp_types::Location::new(uri, range),
            ));
        }
    }

    let target = find_hover_target(&parsed.views, offset)?;
    match target {
        HoverTarget::VarIdent { name, .. } => {
            let enclosing_view = parsed
                .views
                .iter()
                .find(|v| span_contains(v.span, offset))?;

            for param in &enclosing_view.params {
                if param.name.as_str() == name {
                    let range = Range::new(
                        byte_offset_to_lsp_pos(content, param.span.start as usize),
                        byte_offset_to_lsp_pos(content, param.span.end as usize),
                    );
                    return Some(lsp_types::GotoDefinitionResponse::Scalar(
                        lsp_types::Location::new(uri, range),
                    ));
                }
            }

            if let Some(def_span) = find_local_declaration_span(&enclosing_view.body, &name, offset)
            {
                let range = Range::new(
                    byte_offset_to_lsp_pos(content, def_span.start as usize),
                    byte_offset_to_lsp_pos(content, def_span.end as usize),
                );
                return Some(lsp_types::GotoDefinitionResponse::Scalar(
                    lsp_types::Location::new(uri, range),
                ));
            }

            if let Some(target_view) = parsed.views.iter().find(|v| v.name.as_str() == name) {
                let range = Range::new(
                    byte_offset_to_lsp_pos(content, target_view.span.start as usize),
                    byte_offset_to_lsp_pos(content, target_view.span.end as usize),
                );
                return Some(lsp_types::GotoDefinitionResponse::Scalar(
                    lsp_types::Location::new(uri, range),
                ));
            }
        }
        HoverTarget::Attribute {
            element_name,
            attr_name,
        } => {
            let target_view = parsed
                .views
                .iter()
                .find(|v| v.name.as_str() == element_name)?;
            let param = target_view
                .params
                .iter()
                .find(|p| p.name.as_str() == attr_name)?;
            let range = Range::new(
                byte_offset_to_lsp_pos(content, param.span.start as usize),
                byte_offset_to_lsp_pos(content, param.span.end as usize),
            );
            return Some(lsp_types::GotoDefinitionResponse::Scalar(
                lsp_types::Location::new(uri, range),
            ));
        }
        HoverTarget::Intrinsic { .. } => {}
    }

    if let Some(class_name) = find_class_ref_at_offset(&parsed.views, offset) {
        let enclosing_view = parsed
            .views
            .iter()
            .find(|v| span_contains(v.span, offset))?;
        for member in &enclosing_view.body {
            if let Member::Style { rules, .. } = member {
                for rule in rules {
                    if rule.class.as_str() == class_name {
                        let range = Range::new(
                            byte_offset_to_lsp_pos(content, rule.span.start as usize),
                            byte_offset_to_lsp_pos(content, rule.span.end as usize),
                        );
                        return Some(lsp_types::GotoDefinitionResponse::Scalar(
                            lsp_types::Location::new(uri, range),
                        ));
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_keywords_and_intrinsics() {
        let content = "View Main {\n  Column {\n  }\n}";
        // Position at line 2, character 2 (inside the Column children area)
        let pos = Position::new(2, 2);
        let resp = handle_completion(content, pos).unwrap();
        if let CompletionResponse::Array(items) = resp {
            assert!(items.iter().any(|item| item.label == "Column"));
            assert!(items.iter().any(|item| item.label == "Button"));
            assert!(items.iter().any(|item| item.label == "var"));
            assert!(items.iter().any(|item| item.label == "Main"));
        } else {
            panic!("Expected Array completion response");
        }
    }

    #[test]
    fn test_completion_attributes_intrinsic() {
        let content = "View Main {\n  Column #[wi] {\n  }\n}";
        // Position inside `#[wi]`
        let pos = Position::new(1, 11);
        let resp = handle_completion(content, pos).unwrap();
        if let CompletionResponse::Array(items) = resp {
            assert!(items.iter().any(|item| item.label == "width"));
            assert!(items.iter().any(|item| item.label == "height"));
            assert!(items.iter().any(|item| item.label == "align"));
            assert!(!items.iter().any(|item| item.label == "var")); // shouldn't show keywords
        } else {
            panic!("Expected Array completion response");
        }
    }

    #[test]
    fn test_completion_attributes_style_rule() {
        let content = "View Main {\n  style {\n    .title #[ra] {}\n  }\n}";
        // Position inside `#[ra]`
        let pos = Position::new(2, 13);
        let resp = handle_completion(content, pos).unwrap();
        if let CompletionResponse::Array(items) = resp {
            assert!(items.iter().any(|item| item.label == "radius"));
            assert!(items.iter().any(|item| item.label == "bg"));
            assert!(items.iter().any(|item| item.label == "width"));
            assert!(!items.iter().any(|item| item.label == "Button")); // no elements/intrinsics in styling rules
        } else {
            panic!("Expected Array completion response");
        }
    }

    #[test]
    fn test_completion_locals() {
        let content = "View Main(title: Str) {\n  var my_var = 10\n  let my_let = \"hello\"\n  \n}";
        // Position in view body
        let pos = Position::new(3, 2);
        let resp = handle_completion(content, pos).unwrap();
        if let CompletionResponse::Array(items) = resp {
            assert!(items.iter().any(|item| item.label == "title"));
            assert!(items.iter().any(|item| item.label == "my_var"));
            assert!(items.iter().any(|item| item.label == "my_let"));
        } else {
            panic!("Expected Array completion response");
        }
    }

    #[test]
    fn test_definition_local_variable() {
        let content = "View Main {\n  var my_var = 10\n  Text(my_var)\n}";
        // Position on the usage of my_var inside Text (line 2, char 8)
        let pos = Position::new(2, 8);
        let uri: Uri = "file:///dummy.byd".parse().unwrap();
        let resp = handle_definition(content, pos, uri.clone()).unwrap();
        if let GotoDefinitionResponse::Scalar(loc) = resp {
            assert_eq!(loc.uri, uri);
            // Definition of my_var is on line 1: `var my_var = 10`
            assert_eq!(loc.range.start.line, 1);
        } else {
            panic!("Expected Scalar definition response");
        }
    }

    #[test]
    fn test_definition_user_view() {
        let content = "View Child {}\nView Main {\n  Child {}\n}";
        // Position on the Child tag inside Main (line 2, char 3)
        let pos = Position::new(2, 3);
        let uri: Uri = "file:///dummy.byd".parse().unwrap();
        let resp = handle_definition(content, pos, uri.clone()).unwrap();
        if let GotoDefinitionResponse::Scalar(loc) = resp {
            assert_eq!(loc.uri, uri);
            // Definition of View Child is on line 0
            assert_eq!(loc.range.start.line, 0);
        } else {
            panic!("Expected Scalar definition response");
        }
    }
}
