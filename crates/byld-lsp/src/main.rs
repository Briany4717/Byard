//! Production-grade LSP server for the Byld (.byd) language files.
//!
//! Provides document synchronization, hover details (for variables and intrinsics),
//! and compile/type-check diagnostics in real time.

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
use lsp_types::request::{HoverRequest, Request as _};
use lsp_types::{
    Diagnostic, DiagnosticSeverity, Hover, HoverContents, HoverProviderCapability, MarkupContent,
    MarkupKind, Position, Range, ServerCapabilities, TextDocumentSyncCapability,
    TextDocumentSyncKind, Uri,
};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Set up communication channel
    let (connection, io_threads) = Connection::stdio();

    // Register capabilities
    let server_capabilities = serde_json::to_value(&ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
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
                        connection.sender.send(Message::Response(response))?;
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
            } => {
                if span_contains(*span, offset) {
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
            }
            Member::Let {
                name, init, span, ..
            } => {
                if span_contains(*span, offset) {
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
            }
            Member::Fn {
                name,
                params,
                body,
                span,
                ..
            } => {
                if span_contains(*span, offset) {
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
            }
            Member::Element(el) => {
                if span_contains(el.span, offset) {
                    let name_start = el.span.start as usize;
                    let name_end = name_start + el.name.as_str().len();
                    if offset >= name_start && offset < name_end {
                        return Some(HoverTarget::Intrinsic {
                            name: el.name.to_string(),
                        });
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
                    if let Some(target) =
                        find_in_members(&el.children, offset, Some(el.name.as_str()))
                    {
                        return Some(target);
                    }
                }
            }
            Member::For {
                var,
                iter,
                body,
                span,
            } => {
                if span_contains(*span, offset) {
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
            }
            Member::When {
                cond,
                then,
                els,
                span,
            } => {
                if span_contains(*span, offset) {
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
    events.sort();
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
                None => "".to_string(),
            };
            format!("Fn({}){}", params_str.join(", "), ret_str)
        }
        Ty::Named(sym) => sym.as_str().to_string(),
        Ty::Unknown => "Unknown".to_string(),
    }
}
