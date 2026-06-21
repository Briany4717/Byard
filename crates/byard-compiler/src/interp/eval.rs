//! The eval driver: walk the AST, wiring declarations to the reactive core
//! (RFC-0002 §"Dev-mode interpreter"; RFC-0004 §3/§11).
//!
//! Each `byld` expression is **lowered** to a reactive computation — a
//! `FnMut(&mut ReactiveCtx) -> Value` closure that resolves identifiers against
//! the [`Env`] *at lowering time* (capturing their `SignalId`/`ScopeId`
//! handles) and performs its `Signal`/memo reads through the context at run
//! time, so read-tracking stays dynamic (RFC-0004 §3). This is the concrete
//! form of RFC-0004's `walk_expr(scope.expr)`:
//!
//! - `var x = init` ⇒ a reactive source: `init` is evaluated once and a signal
//!   is created from it; `x` binds to `Value::Signal`.
//! - `let y` / `fn f` ⇒ a [`ReactiveCtx::open_memo`]; `y`/`f` binds to
//!   `Value::Memo`. Whether it is actually reactive is observed by the tracker
//!   (D3), not declared.
//! - Reading a `var`/memo identifier routes through `read_signal`/`read_memo`.
//! - `untrack(expr)` is a reserved-name call dispatched to [`untrack`].
//! - A mutation (`=`, `+=`, `++`, `--`) on a `var` marks it; on anything else
//!   it is [`CompileError::NotAssignable`].

use super::env::{Env, SignalId, Value};
use super::events::Action;
use super::intrinsics::validate_element;
use super::reactive::{FrameTarget, ReactiveCtx, ScopeId, untrack};
use crate::diagnostics::{CompileError, Span};
use crate::parser::ast::{
    AssignOp, Attr, AttrKind, ElementNode, Expr, Member, PostfixOp, StrPart, ViewDecl,
};
use crate::symbol::Symbol;

/// A lowered render-tree node: the interpreter's plan for one element. Reactive
/// fields are reactive-scope ids the engine reads each tick (M14).
#[derive(Debug, Clone, PartialEq)]
pub enum RenderNode {
    /// A box-like container.
    Box {
        /// The element intrinsic name.
        name: Symbol,
        /// Styling attributes.
        attrs: Vec<Attr>,
        /// Child render nodes.
        children: Vec<RenderNode>,
        /// Event shorthand action.
        action: Option<Expr>,
    },
    /// A text run.
    Text {
        /// Styling attributes.
        attrs: Vec<Attr>,
        /// The reactive scope projecting the text content.
        content: ScopeId,
    },
    /// A flexible gap (layout-only).
    Spacer,
}

/// A lowered reactive computation (see the module docs).
type Lowered = Box<dyn FnMut(&mut ReactiveCtx) -> Value>;

thread_local! {
    /// Thread-local storage holding the active payload of the event currently being processed.
    pub static CURRENT_PAYLOAD: std::cell::RefCell<Option<Value>> = const { std::cell::RefCell::new(None) };
}

/// The Dev-mode interpreter for one `View` instance: a reactive context plus
/// the View's binding environment.
#[derive(Default)]
pub struct Interpreter {
    ctx: ReactiveCtx,
    env: Env<'static>,
    next_target: u32,
    errors: Vec<CompileError>,
    /// `var` name → its `Signal`, so a hot-reload can preserve state by
    /// rebinding instead of re-initializing (RFC-0004 §11).
    var_sigs: std::collections::HashMap<Symbol, SignalId>,
    /// Incremental LayoutAtlas.
    pub atlas: byard_core::atlas::layout::LayoutAtlas,
    /// Interactive events router.
    pub router: crate::interp::events::EventRouter,
    /// Glyph-accurate text measurer, created lazily on first layout so the
    /// non-rendering paths (parsing, reactivity tests) never load fonts.
    text_measurer: Option<byard_core::text::TextMeasurer>,
}

impl Interpreter {
    /// Creates an empty interpreter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The reactive context (for tests and the engine bridge).
    #[must_use]
    pub fn ctx(&self) -> &ReactiveCtx {
        &self.ctx
    }

    /// Diagnostics accumulated while evaluating.
    #[must_use]
    pub fn errors(&self) -> &[CompileError] {
        &self.errors
    }

    /// Runs one tick: begins an epoch and pulls all dirty scopes.
    pub fn tick(&mut self) {
        let epoch = self.ctx.begin_tick();
        self.ctx.pull(epoch);
    }

    /// The most recently projected value of a value binding (for tests).
    #[must_use]
    pub fn binding_value(&self, s: ScopeId) -> Option<Value> {
        self.ctx.binding_value(s)
    }

    /// Allocates the next frame-target id for a value binding.
    fn next_target(&mut self) -> FrameTarget {
        let t = FrameTarget(self.next_target);
        self.next_target += 1;
        t
    }

    /// Glyph-accurate `(width, height)` of `text` at `font_size`, lazily
    /// initializing the font system on first use.
    fn measure_text(&mut self, text: &str, font_size: f32) -> (f32, f32) {
        self.text_measurer
            .get_or_insert_with(byard_core::text::TextMeasurer::new)
            .measure(text, font_size)
    }

    // ── declarations ────────────────────────────────────────────────────

    /// Processes the declaration-level members of a `View` body (`var`/`let`/
    /// `fn`/`inject`/bare expression). Elements are lowered by the intrinsics
    /// layer (M10).
    pub fn eval_view_decls(&mut self, view: &ViewDecl) {
        for member in &view.body {
            self.eval_member(member);
        }
    }

    fn eval_member(&mut self, member: &Member) {
        match member {
            Member::Var { name, init, .. } => {
                self.define_var(name.clone(), init);
            }
            Member::Let { name, init, .. } => {
                self.define_let(name.clone(), init);
            }
            Member::Fn { name, body, .. } => {
                // Phase 2: a `fn` lowers to a memo of its body (params are bound
                // at call sites in a later milestone).
                self.define_let(name.clone(), body);
            }
            Member::Expr(e) => {
                if let Err(err) = self.eval_action(e) {
                    self.errors.push(err);
                }
            }
            // inject / elements / control flow / style are wired in M9.x/M10+.
            _ => {}
        }
    }

    /// `var x = init` — evaluate `init` once, create a reactive source from it.
    pub fn define_var(&mut self, name: Symbol, init: &Expr) -> SignalId {
        let initial = self.eval_pure(init);
        let sig = self.ctx.create_signal(initial);
        self.env.push(name.clone(), Value::Signal(sig));
        self.var_sigs.insert(name, sig);
        sig
    }

    /// The `Signal` backing the `var` named `name`, if any.
    #[must_use]
    pub fn var_signal(&self, name: &Symbol) -> Option<SignalId> {
        self.var_sigs.get(name).copied()
    }

    /// Writes a value to a `Signal` (a controller result or test driver), running
    /// the mark cascade.
    pub fn write_var(&mut self, sig: SignalId, value: Value) {
        self.ctx.write_signal(sig, value);
    }

    /// Reads a `Signal`'s current value without tracking.
    #[must_use]
    pub fn peek(&self, sig: SignalId) -> Value {
        self.ctx.peek_signal(sig)
    }

    /// Applies a hot-reload patch (RFC-0002 §"Hot-reload boundary", RFC-0004
    /// §11). On a [`reactive-compatible`](super::reload::ReloadKind) patch the
    /// existing `Signal`s are **kept** (matched by name) so state survives; on a
    /// structure-incompatible patch every `var` is re-initialized from the new
    /// AST (state resets). The reactive scopes are rebuilt from the new AST
    /// either way — read-tracking re-derives the dependency graph (§11).
    pub fn reload(&mut self, new_view: &ViewDecl, kind: super::reload::ReloadKind) {
        use super::reload::ReloadKind;
        let old = std::mem::take(&mut self.var_sigs);
        self.env = Env::new();
        for member in &new_view.body {
            match member {
                Member::Var { name, init, .. } => {
                    if matches!(kind, ReloadKind::ReactiveCompatible) {
                        if let Some(&sig) = old.get(name) {
                            // Keep the live Signal (and its value).
                            self.env.push(name.clone(), Value::Signal(sig));
                            self.var_sigs.insert(name.clone(), sig);
                            continue;
                        }
                    }
                    self.define_var(name.clone(), init);
                }
                Member::Let { name, init, .. }
                | Member::Fn {
                    name, body: init, ..
                } => {
                    self.define_let(name.clone(), init);
                }
                _ => {}
            }
        }
    }

    /// `let y = init` (and `fn`) — open a computed memo.
    pub fn define_let(&mut self, name: Symbol, init: &Expr) -> ScopeId {
        let compute = self.lower_expr(init, None);
        let scope = self.ctx.open_memo(compute);
        self.env.push(name, Value::Memo(scope));
        scope
    }

    /// Opens a value binding projecting `expr` into a fresh frame target
    /// (used by intrinsics, M10, and by tests).
    pub fn bind_value(&mut self, expr: &Expr) -> ScopeId {
        let target = self.next_target();
        let compute = self.lower_expr(expr, None);
        self.ctx.open_value_binding(target, compute)
    }

    // ── element lowering (RFC-0005) ─────────────────────────────────────

    /// Lowers an element to a [`RenderNode`], validating it against the §5
    /// attribute contract first (diagnostics accumulate in [`Interpreter::errors`]).
    /// `known_views` are user `ViewDecl` names in scope.
    pub fn lower_element(&mut self, el: &ElementNode, known_views: &[&str]) -> RenderNode {
        self.errors.extend(validate_element(el, known_views));
        match el.name.as_str() {
            "Text" | "Button" if !el.content.is_empty() => {
                let content = self.bind_value(&el.content[0].value);
                if el.name.as_str() == "Button" {
                    // A Button is a decorated box wrapping its label.
                    RenderNode::Box {
                        name: Symbol::intern("Button"),
                        attrs: el.attrs.clone(),
                        children: vec![RenderNode::Text {
                            attrs: Vec::new(),
                            content,
                        }],
                        action: el.action.clone(),
                    }
                } else {
                    RenderNode::Text {
                        attrs: el.attrs.clone(),
                        content,
                    }
                }
            }
            "Spacer" => RenderNode::Spacer,
            _ => {
                // Box / Column / Row / ScrollView and any other container.
                let children = el
                    .children
                    .iter()
                    .filter_map(|m| match m {
                        Member::Element(e) => Some(self.lower_element(e, known_views)),
                        _ => None,
                    })
                    .collect();
                RenderNode::Box {
                    name: el.name.clone(),
                    attrs: el.attrs.clone(),
                    children,
                    action: el.action.clone(),
                }
            }
        }
    }

    /// Walks a render tree, projecting it into a `byard-core` [`RenderFrame`]
    /// using Taffy layout via `byard-core`'s [`LayoutAtlas`].
    #[allow(clippy::similar_names)]
    pub fn render(
        &mut self,
        tree: &[RenderNode],
        frame: &mut byard_core::frame::RenderFrame,
        width: f32,
        height: f32,
    ) {
        use byard_core::frame::Viewport;

        self.atlas.clear();
        self.router = crate::interp::events::EventRouter::new();
        let mut flat_ids = Vec::new();

        let mut root_children = Vec::new();
        for node in tree {
            if let Ok(id) = self.build_layout_tree(node, &mut flat_ids) {
                root_children.push(id);
            }
        }

        if !root_children.is_empty() {
            let root_style =
                byard_core::atlas::layout::ContainerStyle::new(Some(width), Some(height))
                    .with_direction(byard_core::atlas::layout::FlexDir::Column);
            if let Ok(root_id) = self.atlas.add_container(root_style, &root_children) {
                self.atlas.set_root(root_id).unwrap();
                self.atlas.compute(Viewport::new(width, height)).unwrap();

                // Populate frame layout bounds
                self.atlas.populate_frame(frame, &[]);

                // Emit instances and text lines at computed positions
                let mut flat_idx = 0;
                let parent_rect = crate::interp::intrinsics::Rect::new(0.0, 0.0, width, height);
                for (i, node) in tree.iter().enumerate() {
                    let node_id = root_children[i];
                    self.render_node_with_atlas(
                        node,
                        node_id,
                        frame,
                        &flat_ids,
                        &mut flat_idx,
                        parent_rect,
                    );
                }
            }
        }
    }

    fn build_layout_tree(
        &mut self,
        node: &RenderNode,
        flat_ids: &mut Vec<byard_core::atlas::layout::AtlasNodeId>,
    ) -> Result<byard_core::atlas::layout::AtlasNodeId, byard_core::atlas::AtlasError> {
        use byard_core::atlas::layout::LeafSize;
        match node {
            RenderNode::Spacer => {
                let id = self.atlas.add_leaf(LeafSize::new(0.0, 12.0))?;
                flat_ids.push(id);
                Ok(id)
            }
            RenderNode::Text { attrs, content } => {
                let text = match self.binding_value(*content) {
                    Some(Value::Str(s)) => s,
                    other => other.map_or_else(String::new, |v| format!("{v:?}")),
                };
                #[allow(clippy::cast_precision_loss)]
                let font_size = self.eval_int_prop(attrs, "size").unwrap_or(16) as f32;
                let (w, h) = self.measure_text(&text, font_size);
                let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                flat_ids.push(id);
                Ok(id)
            }
            RenderNode::Box {
                name,
                attrs,
                children,
                ..
            } => {
                let mut child_ids = Vec::with_capacity(children.len());
                let mut temp_flat = Vec::new();
                for child in children {
                    let child_id = self.build_layout_tree(child, &mut temp_flat)?;
                    child_ids.push(child_id);
                }
                let style = self.eval_container_style(name.as_str(), attrs);
                let id = self.atlas.add_container(style, &child_ids)?;
                flat_ids.push(id);
                flat_ids.extend(temp_flat);
                Ok(id)
            }
        }
    }

    #[allow(clippy::similar_names)]
    fn render_node_with_atlas(
        &mut self,
        node: &RenderNode,
        atlas_node: byard_core::atlas::layout::AtlasNodeId,
        frame: &mut byard_core::frame::RenderFrame,
        flat_ids: &[byard_core::atlas::layout::AtlasNodeId],
        flat_idx: &mut usize,
        parent_rect: crate::interp::intrinsics::Rect,
    ) {
        debug_assert_eq!(flat_ids[*flat_idx], atlas_node);
        *flat_idx += 1;

        match node {
            RenderNode::Spacer => {}
            RenderNode::Text { attrs, content } => {
                if let Ok(Some(rect)) = self.atlas.resolved_rect(atlas_node) {
                    let text = match self.binding_value(*content) {
                        Some(Value::Str(s)) => s,
                        other => other.map_or_else(String::new, |v| format!("{v:?}")),
                    };
                    let color = self.eval_color_prop(attrs, "color").unwrap_or(0x00ff_ffff);
                    let size = self.eval_int_prop(attrs, "size").unwrap_or(16) as f32;
                    frame.push_text(byard_core::TextLine {
                        x: rect.x,
                        y: rect.y,
                        text,
                        font_size: size,
                        color: super::intrinsics::color_to_rgba(color, false),
                        dirty: true,
                    });

                    let has_events = attrs
                        .iter()
                        .any(|a| matches!(a.kind, AttrKind::Event { .. }));
                    if has_events {
                        let self_rect = crate::interp::intrinsics::Rect::new(
                            rect.x,
                            rect.y,
                            rect.width,
                            rect.height,
                        );
                        let hit_rect =
                            crate::interp::intrinsics::inflate_hit_rect(self_rect, parent_rect);
                        for attr in attrs {
                            if let AttrKind::Event { payload, action } = &attr.kind {
                                let event_kind = match attr.name.as_str() {
                                    "tap" => crate::interp::events::EventKind::Tap,
                                    "pointer_down" => crate::interp::events::EventKind::PointerDown,
                                    "pointer_up" => crate::interp::events::EventKind::PointerUp,
                                    "pointer_move" => crate::interp::events::EventKind::PointerMove,
                                    "scroll" => crate::interp::events::EventKind::Scroll,
                                    "wheel" => crate::interp::events::EventKind::Wheel,
                                    "change" => crate::interp::events::EventKind::Change,
                                    _ => continue,
                                };
                                if let Ok(action_closure) =
                                    self.lower_action(action, payload.clone())
                                {
                                    if let Some(elem_idx) = self.atlas.node_index(atlas_node) {
                                        self.router.on(
                                            elem_idx,
                                            hit_rect,
                                            event_kind,
                                            action_closure,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            RenderNode::Box {
                name,
                attrs,
                children,
                action,
            } => {
                let mut current_rect = parent_rect;
                if let Ok(Some(rect)) = self.atlas.resolved_rect(atlas_node) {
                    current_rect = crate::interp::intrinsics::Rect::new(
                        rect.x,
                        rect.y,
                        rect.width,
                        rect.height,
                    );
                    let bg = self.eval_color_prop(attrs, "bg");
                    let radius = self.eval_int_prop(attrs, "radius").unwrap_or(0) as f32;
                    if let Some(color) = bg {
                        frame.push_instance(byard_core::BoxInstance {
                            rect: [rect.x, rect.y, rect.width, rect.height],
                            color: super::intrinsics::color_to_rgba(color, false),
                            radii: [radius; 4],
                        });
                    }

                    let element_name = name.as_str();
                    let has_event_attrs = attrs
                        .iter()
                        .any(|a| matches!(a.kind, AttrKind::Event { .. }));
                    let is_interactive =
                        matches!(element_name, "Button" | "TextField" | "Toggle" | "Slider")
                            || has_event_attrs
                            || action.is_some();

                    if is_interactive {
                        let hit_rect =
                            crate::interp::intrinsics::inflate_hit_rect(current_rect, parent_rect);
                        for attr in attrs {
                            if let AttrKind::Event { payload, action } = &attr.kind {
                                let event_kind = match attr.name.as_str() {
                                    "tap" => crate::interp::events::EventKind::Tap,
                                    "pointer_down" => crate::interp::events::EventKind::PointerDown,
                                    "pointer_up" => crate::interp::events::EventKind::PointerUp,
                                    "pointer_move" => crate::interp::events::EventKind::PointerMove,
                                    "scroll" => crate::interp::events::EventKind::Scroll,
                                    "wheel" => crate::interp::events::EventKind::Wheel,
                                    "change" => crate::interp::events::EventKind::Change,
                                    _ => continue,
                                };
                                if let Ok(action_closure) =
                                    self.lower_action(action, payload.clone())
                                {
                                    if let Some(elem_idx) = self.atlas.node_index(atlas_node) {
                                        self.router.on(
                                            elem_idx,
                                            hit_rect,
                                            event_kind,
                                            action_closure,
                                        );
                                    }
                                }
                            }
                        }

                        if let Some(action_expr) = action {
                            if let Ok(action_closure) = self.lower_action(action_expr, None) {
                                if let Some(elem_idx) = self.atlas.node_index(atlas_node) {
                                    self.router.on(
                                        elem_idx,
                                        hit_rect,
                                        crate::interp::events::EventKind::Tap,
                                        action_closure,
                                    );
                                }
                            }
                        }
                    }
                }
                for child in children {
                    let child_id = flat_ids[*flat_idx];
                    self.render_node_with_atlas(
                        child,
                        child_id,
                        frame,
                        flat_ids,
                        flat_idx,
                        current_rect,
                    );
                }
            }
        }
    }

    fn eval_color_prop(&mut self, attrs: &[Attr], name: &str) -> Option<i64> {
        attrs.iter().find_map(|a| {
            if a.name.as_str() == name {
                if let AttrKind::Prop { value } = &a.kind {
                    let val = self.eval_pure(value);
                    return val.as_int();
                }
            }
            None
        })
    }

    fn eval_int_prop(&mut self, attrs: &[Attr], name: &str) -> Option<i64> {
        attrs.iter().find_map(|a| {
            if a.name.as_str() == name {
                if let AttrKind::Prop { value } = &a.kind {
                    let val = self.eval_pure(value);
                    return val.as_int();
                }
            }
            None
        })
    }

    fn eval_container_style(
        &mut self,
        element_name: &str,
        attrs: &[Attr],
    ) -> byard_core::atlas::layout::ContainerStyle {
        use byard_core::atlas::layout::{Align, ContainerStyle, FlexDir, Justify};

        let val_to_f32 = |v: &Value| -> Option<f32> {
            match v {
                Value::Int(n) => Some(*n as f32),
                Value::Float(f) => Some(*f as f32),
                _ => None,
            }
        };

        let mut style = ContainerStyle::default();
        style.direction = match element_name {
            "Row" => FlexDir::Row,
            _ => FlexDir::Column,
        };
        for attr in attrs {
            if let AttrKind::Prop { value } = &attr.kind {
                let val = self.eval_pure(value);
                match attr.name.as_str() {
                    "width" => style.width = val.as_int().map(|n| n as f32),
                    "height" => style.height = val.as_int().map(|n| n as f32),
                    "direction" => {
                        if let Value::Str(s) = &val {
                            style.direction = match s.as_str() {
                                "column" => FlexDir::Column,
                                _ => FlexDir::Row,
                            };
                        }
                    }
                    "gap" => {
                        if let Some(n) = val.as_int() {
                            style.gap = n as f32;
                        }
                    }
                    "p" | "padding" => {
                        style.padding = self.eval_spacing(&val);
                    }
                    "pt" | "padding_top" | "padding-top" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.padding.top = v;
                        }
                    }
                    "pr" | "padding_right" | "padding-right" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.padding.right = v;
                        }
                    }
                    "pb" | "padding_bottom" | "padding-bottom" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.padding.bottom = v;
                        }
                    }
                    "pl" | "padding_left" | "padding-left" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.padding.left = v;
                        }
                    }
                    "px" | "padding_x" | "padding_horizontal" | "padding-horizontal" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.padding.left = v;
                            style.padding.right = v;
                        }
                    }
                    "py" | "padding_y" | "padding_vertical" | "padding-vertical" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.padding.top = v;
                            style.padding.bottom = v;
                        }
                    }
                    "m" | "margin" => {
                        style.margin = self.eval_spacing(&val);
                    }
                    "mt" | "margin_top" | "margin-top" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.top = v;
                        }
                    }
                    "mr" | "margin_right" | "margin-right" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.right = v;
                        }
                    }
                    "mb" | "margin_bottom" | "margin-bottom" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.bottom = v;
                        }
                    }
                    "ml" | "margin_left" | "margin-left" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.left = v;
                        }
                    }
                    "mx" | "margin_x" | "margin_horizontal" | "margin-horizontal" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.left = v;
                            style.margin.right = v;
                        }
                    }
                    "my" | "margin_y" | "margin_vertical" | "margin-vertical" => {
                        if let Some(v) = val_to_f32(&val) {
                            style.margin.top = v;
                            style.margin.bottom = v;
                        }
                    }
                    "align" => {
                        if let Value::Str(s) = &val {
                            style.align = match s.as_str() {
                                "start" => Align::Start,
                                "center" => Align::Center,
                                "end" => Align::End,
                                _ => Align::Stretch,
                            };
                        }
                    }
                    "justify" => {
                        if let Value::Str(s) = &val {
                            style.justify = match s.as_str() {
                                "center" => Justify::Center,
                                "end" => Justify::End,
                                "between" => Justify::Between,
                                "around" => Justify::Around,
                                "evenly" => Justify::Evenly,
                                _ => Justify::Start,
                            };
                        }
                    }
                    "grow" => {
                        if let Some(n) = val.as_int() {
                            style.grow = n as f32;
                        }
                    }
                    _ => {}
                }
            }
        }
        style
    }

    fn eval_spacing(&mut self, val: &Value) -> byard_core::atlas::layout::Spacing {
        use byard_core::atlas::layout::Spacing;
        let val_to_f32 = |v: &Value| -> f32 {
            match v {
                Value::Int(n) => *n as f32,
                Value::Float(f) => *f as f32,
                _ => 0.0,
            }
        };

        match val {
            Value::Int(n) => Spacing::all(*n as f32),
            Value::Float(f) => Spacing::all(*f as f32),
            Value::Tuple(items) => {
                let has_names = items.iter().any(|(name, _)| name.is_some());
                if has_names {
                    let mut s = Spacing::default();
                    for (name, item_val) in items {
                        let v = val_to_f32(item_val);
                        if let Some(sym) = name {
                            match sym.as_str() {
                                "top" | "t" => s.top = v,
                                "right" | "r" => s.right = v,
                                "bottom" | "b" => s.bottom = v,
                                "left" | "l" => s.left = v,
                                "horizontal" | "h" | "x" | "px" | "mx" => {
                                    s.left = v;
                                    s.right = v;
                                }
                                "vertical" | "v" | "y" | "py" | "my" => {
                                    s.top = v;
                                    s.bottom = v;
                                }
                                "all" | "a" => {
                                    s.top = v;
                                    s.right = v;
                                    s.bottom = v;
                                    s.left = v;
                                }
                                _ => {}
                            }
                        }
                    }
                    s
                } else {
                    match items.len() {
                        1 => Spacing::all(val_to_f32(&items[0].1)),
                        2 => {
                            let vertical = val_to_f32(&items[0].1);
                            let horizontal = val_to_f32(&items[1].1);
                            Spacing::symmetric(vertical, horizontal)
                        }
                        3 => {
                            let top = val_to_f32(&items[0].1);
                            let horizontal = val_to_f32(&items[1].1);
                            let bottom = val_to_f32(&items[2].1);
                            Spacing {
                                top,
                                right: horizontal,
                                bottom,
                                left: horizontal,
                            }
                        }
                        4 => Spacing {
                            top: val_to_f32(&items[0].1),
                            right: val_to_f32(&items[1].1),
                            bottom: val_to_f32(&items[2].1),
                            left: val_to_f32(&items[3].1),
                        },
                        _ => Spacing::default(),
                    }
                }
            }
            Value::List(items) => match items.len() {
                1 => Spacing::all(val_to_f32(&items[0])),
                2 => {
                    let vertical = val_to_f32(&items[0]);
                    let horizontal = val_to_f32(&items[1]);
                    Spacing::symmetric(vertical, horizontal)
                }
                3 => {
                    let top = val_to_f32(&items[0]);
                    let horizontal = val_to_f32(&items[1]);
                    let bottom = val_to_f32(&items[2]);
                    Spacing {
                        top,
                        right: horizontal,
                        bottom,
                        left: horizontal,
                    }
                }
                4 => Spacing {
                    top: val_to_f32(&items[0]),
                    right: val_to_f32(&items[1]),
                    bottom: val_to_f32(&items[2]),
                    left: val_to_f32(&items[3]),
                },
                _ => Spacing::default(),
            },
            _ => Spacing::default(),
        }
    }

    /// Processes a whole `View`: its declarations first (so bindings can resolve
    /// names), then lowers its top-level elements into a render tree.
    pub fn lower_view(&mut self, view: &ViewDecl, known_views: &[&str]) -> Vec<RenderNode> {
        self.eval_view_decls(view);
        view.body
            .iter()
            .filter_map(|m| match m {
                Member::Element(e) => Some(self.lower_element(e, known_views)),
                _ => None,
            })
            .collect()
    }

    // ── lowering ────────────────────────────────────────────────────────

    /// Lowers `expr` to a reactive computation against the current environment.
    fn lower_expr(&self, expr: &Expr, payload_name: Option<&Symbol>) -> Lowered {
        match expr {
            Expr::IntLit(n, _) => {
                let n = *n;
                Box::new(move |_| Value::Int(n))
            }
            Expr::FloatLit(f, _) => {
                let f = *f;
                Box::new(move |_| Value::Float(f))
            }
            Expr::StrLit(parts, _) => self.lower_strlit(parts, payload_name),
            Expr::Ident(name, _) => self.lower_ident(name, payload_name),
            Expr::Array(elems, _) => {
                let mut cs: Vec<Lowered> = elems
                    .iter()
                    .map(|e| self.lower_expr(e, payload_name))
                    .collect();
                Box::new(move |ctx| Value::List(cs.iter_mut().map(|c| c(ctx)).collect()))
            }
            Expr::Tuple(elems, _) => {
                let mut cs: Vec<(Option<Symbol>, Lowered)> = elems
                    .iter()
                    .map(|arg| (arg.name.clone(), self.lower_expr(&arg.value, payload_name)))
                    .collect();
                Box::new(move |ctx| {
                    Value::Tuple(
                        cs.iter_mut()
                            .map(|(name, c)| (name.clone(), c(ctx)))
                            .collect(),
                    )
                })
            }
            Expr::Ternary {
                cond, then, els, ..
            } => {
                let mut cc = self.lower_expr(cond, payload_name);
                let mut tc = self.lower_expr(then, payload_name);
                let mut ec = self.lower_expr(els, payload_name);
                Box::new(move |ctx| {
                    if cc(ctx).as_bool().unwrap_or(false) {
                        tc(ctx)
                    } else {
                        ec(ctx)
                    }
                })
            }
            Expr::Call { callee, args, .. } => self.lower_call(callee, args, payload_name),
            Expr::ClassRef(class, _) => {
                let s = format!(".{class}");
                Box::new(move |_| Value::Str(s.clone()))
            }
            Expr::Postfix { target, op, span } => {
                if let Ok(sig) = self.resolve_var(target, *span) {
                    let op = *op;
                    Box::new(move |ctx| {
                        let cur = ctx.peek_signal(sig).as_int().unwrap_or(0);
                        let new = match op {
                            PostfixOp::Inc => cur + 1,
                            PostfixOp::Dec => cur - 1,
                        };
                        ctx.write_signal(sig, Value::Int(new));
                        Value::Unit
                    })
                } else {
                    Box::new(|_| Value::Unit)
                }
            }
            Expr::Assign {
                target,
                op,
                value,
                span,
            } => {
                if let Ok(sig) = self.resolve_var(target, *span) {
                    let op = *op;
                    let mut rhs = self.lower_expr(value, payload_name);
                    Box::new(move |ctx| {
                        let val = rhs(ctx);
                        let new = match op {
                            AssignOp::Assign => val,
                            AssignOp::Add => {
                                let cur = ctx.peek_signal(sig).as_int().unwrap_or(0);
                                Value::Int(cur + val.as_int().unwrap_or(0))
                            }
                            AssignOp::Sub => {
                                let cur = ctx.peek_signal(sig).as_int().unwrap_or(0);
                                Value::Int(cur - val.as_int().unwrap_or(0))
                            }
                        };
                        ctx.write_signal(sig, new);
                        Value::Unit
                    })
                } else {
                    Box::new(|_| Value::Unit)
                }
            }
            // Member access needs controller metadata (not modeled in Phase 2);
            // lambdas/assignments are actions, not projected values.
            Expr::Member { .. } | Expr::Lambda { .. } | Expr::Error(_) => Box::new(|_| Value::Unit),
        }
    }

    fn lower_ident(&self, name: &Symbol, payload_name: Option<&Symbol>) -> Lowered {
        if let Some(pname) = payload_name {
            if pname == name {
                return Box::new(move |_| {
                    CURRENT_PAYLOAD.with(|cell| cell.borrow().clone().unwrap_or(Value::Unit))
                });
            }
        }
        match name.as_str() {
            "true" => return Box::new(|_| Value::Bool(true)),
            "false" => return Box::new(|_| Value::Bool(false)),
            _ => {}
        }
        match self.env.lookup(name) {
            Some(Value::Signal(sig)) => {
                let sig = *sig;
                Box::new(move |ctx| ctx.read_signal(sig))
            }
            Some(Value::Memo(scope)) => {
                let m = *scope;
                Box::new(move |ctx| ctx.read_memo(m))
            }
            Some(v) => {
                let v = v.clone();
                Box::new(move |_| v.clone())
            }
            // An unresolved identifier is treated as an enum/style token
            // (e.g. `center`, `cover`); intrinsics validate it (M10).
            None => {
                let token = name.as_str().to_string();
                Box::new(move |_| Value::Str(token.clone()))
            }
        }
    }

    fn lower_strlit(&self, parts: &[StrPart], payload_name: Option<&Symbol>) -> Lowered {
        enum Part {
            Text(String),
            Interp(Lowered),
        }
        let mut lowered: Vec<Part> = parts
            .iter()
            .map(|p| match p {
                StrPart::Text(t) => Part::Text(t.clone()),
                StrPart::Interp(e) => Part::Interp(self.lower_expr(e, payload_name)),
            })
            .collect();
        Box::new(move |ctx| {
            let mut s = String::new();
            for part in &mut lowered {
                match part {
                    Part::Text(t) => s.push_str(t),
                    Part::Interp(c) => s.push_str(&display_value(&c(ctx))),
                }
            }
            Value::Str(s)
        })
    }

    fn lower_call(
        &self,
        callee: &Expr,
        args: &[crate::parser::ast::Arg],
        payload_name: Option<&Symbol>,
    ) -> Lowered {
        // `untrack(expr)` — the reserved escape hatch (D2).
        if let Expr::Ident(name, _) = callee {
            if name.as_str() == "untrack" {
                if let Some(arg) = args.first() {
                    let mut inner = self.lower_expr(&arg.value, payload_name);
                    return Box::new(move |ctx| untrack(|| inner(ctx)));
                }
            }
            // A zero-arg call to a `fn`/`let` memo reads that memo.
            if let Some(Value::Memo(scope)) = self.env.lookup(name) {
                let m = *scope;
                return Box::new(move |ctx| ctx.read_memo(m));
            }
        }
        Box::new(|_| Value::Unit)
    }

    // ── actions (mutations & bare expressions) ──────────────────────────

    /// Evaluates an expression with no reactive scope active (an *action*, not a
    /// projection). Mutations route through the mark cascade; a mutation on a
    /// non-`var` l-value is [`CompileError::NotAssignable`].
    ///
    /// # Errors
    ///
    /// Returns [`CompileError::NotAssignable`] if a mutation targets something
    /// other than a `var`.
    pub fn eval_action(&mut self, expr: &Expr) -> Result<Value, CompileError> {
        match expr {
            Expr::Postfix { target, op, span } => {
                let sig = self.resolve_var(target, *span)?;
                let cur = self.ctx.peek_signal(sig).as_int().unwrap_or(0);
                let new = match op {
                    PostfixOp::Inc => cur + 1,
                    PostfixOp::Dec => cur - 1,
                };
                self.ctx.write_signal(sig, Value::Int(new));
                Ok(Value::Unit)
            }
            Expr::Assign {
                target,
                op,
                value,
                span,
            } => {
                let sig = self.resolve_var(target, *span)?;
                let rhs = self.eval_pure(value);
                let new = match op {
                    AssignOp::Assign => rhs,
                    AssignOp::Add => {
                        let cur = self.ctx.peek_signal(sig).as_int().unwrap_or(0);
                        Value::Int(cur + rhs.as_int().unwrap_or(0))
                    }
                    AssignOp::Sub => {
                        let cur = self.ctx.peek_signal(sig).as_int().unwrap_or(0);
                        Value::Int(cur - rhs.as_int().unwrap_or(0))
                    }
                };
                self.ctx.write_signal(sig, new);
                Ok(Value::Unit)
            }
            other => Ok(self.eval_pure(other)),
        }
    }

    /// Evaluates `expr` once, immediately, with no scope active (so nothing
    /// subscribes). Used to seed `var`s and to evaluate action operands.
    fn eval_pure(&mut self, expr: &Expr) -> Value {
        let mut compute = self.lower_expr(expr, None);
        compute(&mut self.ctx)
    }

    fn resolve_var(&self, target: &Expr, span: Span) -> Result<SignalId, CompileError> {
        if let Expr::Ident(name, _) = target {
            if let Some(Value::Signal(sig)) = self.env.lookup(name) {
                return Ok(*sig);
            }
        }
        Err(CompileError::NotAssignable { span })
    }

    /// Lowers an action expression to an event handler closure, capturing any optional payload bindings.
    ///
    /// # Errors
    ///
    /// Returns a [`CompileError`] if variable resolution or assignment validation fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn lower_action(
        &self,
        expr: &Expr,
        payload_name: Option<Symbol>,
    ) -> Result<Action, CompileError> {
        let mut compute = self.lower_expr(expr, payload_name.as_ref());
        Ok(Box::new(move |ctx, payload| {
            CURRENT_PAYLOAD.with(|cell| {
                *cell.borrow_mut() = payload.cloned();
            });
            let _ = compute(ctx);
            CURRENT_PAYLOAD.with(|cell| {
                *cell.borrow_mut() = None;
            });
        }))
    }

    /// Converts winit-sourced input events to interpreter event payloads and dispatches them to the `EventRouter`.
    pub fn dispatch_events(&mut self, events: &[byard_core::InputEvent]) {
        use crate::interp::events::{EventKind as CompKind, InputEvent as CompEvent};
        use byard_core::platform::{EventKind as CoreKind, InputPayload};

        let comp_events: Vec<CompEvent> = events
            .iter()
            .map(|ev| {
                let kind = match ev.kind {
                    CoreKind::PointerDown => CompKind::PointerDown,
                    CoreKind::PointerUp => CompKind::PointerUp,
                    CoreKind::Tap => CompKind::Tap,
                    CoreKind::PointerMove => CompKind::PointerMove,
                    CoreKind::Scroll => CompKind::Scroll,
                    CoreKind::Wheel => CompKind::Wheel,
                    CoreKind::Change => CompKind::Change,
                };
                let value = ev.payload.as_ref().map(|p| match p {
                    InputPayload::Str(s) => Value::Str(s.clone()),
                    InputPayload::Bool(b) => Value::Bool(*b),
                    InputPayload::Float(f) => Value::Float(f64::from(*f)),
                });
                CompEvent {
                    kind,
                    pos: ev.pos,
                    delta: ev.delta,
                    value,
                    time_ms: ev.time_ms,
                }
            })
            .collect();

        self.router
            .dispatch_tick(&mut self.ctx, Some(&self.atlas), comp_events);
    }
}

/// Renders a value for string interpolation (`"Count: {count}"`).
fn display_value(v: &Value) -> String {
    match v {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Str(s) => s.clone(),
        _ => String::new(),
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::parser::ast::ElementNode;
    use crate::parser::parse;

    fn element(m: &Member) -> &ElementNode {
        match m {
            Member::Element(e) => e,
            _ => panic!("expected element"),
        }
    }

    #[test]
    fn var_text_binding_updates_after_mutation_and_tick() {
        let parsed =
            parse("View C() {\n var count = 0\n Text(\"{count}\")\n Button(\"+\") => count++\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];

        let mut interp = Interpreter::new();
        let Member::Var { name, init, .. } = &view.body[0] else {
            panic!("expected var");
        };
        interp.define_var(name.clone(), init);

        let text = element(&view.body[1]);
        let bind = interp.bind_value(&text.content[0].value);
        interp.tick();
        assert_eq!(
            interp.binding_value(bind),
            Some(Value::Str("0".to_string()))
        );

        // The Button's `=> count++` action.
        let action = element(&view.body[2]).action.as_ref().unwrap();
        interp.eval_action(action).unwrap();
        interp.tick();
        assert_eq!(
            interp.binding_value(bind),
            Some(Value::Str("1".to_string()))
        );
    }

    #[test]
    fn let_memo_recomputes_when_its_source_changes() {
        let parsed = parse(
            "View C() {\n var count = 0\n let doubled = count\n Text(\"{doubled}\")\n Button(\"+\") => count++\n}",
        );
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();

        let Member::Var { name, init, .. } = &view.body[0] else {
            panic!()
        };
        interp.define_var(name.clone(), init);
        let Member::Let { name, init, .. } = &view.body[1] else {
            panic!()
        };
        let memo = interp.define_let(name.clone(), init);

        let text = element(&view.body[2]);
        let bind = interp.bind_value(&text.content[0].value);
        interp.tick();
        assert_eq!(
            interp.binding_value(bind),
            Some(Value::Str("0".to_string()))
        );
        let evals = interp.ctx().eval_count(memo);

        let action = element(&view.body[3]).action.as_ref().unwrap();
        interp.eval_action(action).unwrap();
        interp.tick();
        assert_eq!(
            interp.binding_value(bind),
            Some(Value::Str("1".to_string()))
        );
        assert!(interp.ctx().eval_count(memo) > evals, "memo recomputed");
    }

    #[test]
    fn assignment_to_a_let_is_not_assignable() {
        let parsed =
            parse("View C() {\n var count = 0\n let y = count\n Button(\"x\") => y = 5\n}");
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();

        let Member::Var { name, init, .. } = &view.body[0] else {
            panic!()
        };
        interp.define_var(name.clone(), init);
        let Member::Let { name, init, .. } = &view.body[1] else {
            panic!()
        };
        interp.define_let(name.clone(), init);

        let action = element(&view.body[2]).action.as_ref().unwrap();
        let err = interp.eval_action(action).unwrap_err();
        assert!(matches!(err, CompileError::NotAssignable { .. }));
    }

    #[test]
    fn lower_view_emits_expected_render_tree() {
        let parsed = parse(
            "View C() {\n var count = 0\n Column #[bg: 0x222222, radius: 16] {\n Text(\"Count: {count}\")\n }\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];

        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        // One top-level Column box with the literal bg/radius and one Text child.
        assert_eq!(tree.len(), 1);
        let RenderNode::Box {
            attrs, children, ..
        } = &tree[0]
        else {
            panic!("expected a Box, got {:?}", tree[0]);
        };
        let bg = interp.eval_color_prop(attrs, "bg");
        let radius = interp.eval_int_prop(attrs, "radius");
        assert_eq!(bg, Some(0x0022_2222));
        assert_eq!(radius, Some(16));
        assert_eq!(children.len(), 1);
        let RenderNode::Text { content, .. } = &children[0] else {
            panic!("expected a Text child");
        };

        // The Text projects the reactive count.
        interp.tick();
        assert_eq!(
            interp.binding_value(*content),
            Some(Value::Str("Count: 0".to_string()))
        );
    }

    #[test]
    fn lowering_an_unknown_element_records_unknown_view() {
        let parsed = parse("View C() { Colunm #[gap: 8] {} }");
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let _ = interp.lower_view(view, &[]);
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::UnknownView { .. }))
        );
    }

    #[test]
    fn mutation_on_an_undeclared_name_is_not_assignable() {
        let parsed = parse("View C() { Button(\"x\") => ghost++ }");
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let action = element(&view.body[0]).action.as_ref().unwrap();
        assert!(matches!(
            interp.eval_action(action).unwrap_err(),
            CompileError::NotAssignable { .. }
        ));
    }

    #[test]
    fn spacing_convenience_parses_correctly() {
        use byard_core::atlas::layout::Spacing;

        let test_cases = vec![
            // 1-value positional
            ("View C() { Column #[p: (10)] {} }", Spacing::all(10.0)),
            // 2-value positional
            (
                "View C() { Column #[p: (2, 3)] {} }",
                Spacing::symmetric(2.0, 3.0),
            ),
            // 3-value positional: top, horizontal, bottom
            (
                "View C() { Column #[p: (10, 20, 30)] {} }",
                Spacing {
                    top: 10.0,
                    right: 20.0,
                    bottom: 30.0,
                    left: 20.0,
                },
            ),
            // 4-value positional: top, right, bottom, left
            (
                "View C() { Column #[p: (1, 2, 3, 4)] {} }",
                Spacing {
                    top: 1.0,
                    right: 2.0,
                    bottom: 3.0,
                    left: 4.0,
                },
            ),
            // Named top
            (
                "View C() { Column #[p: (top: 10)] {} }",
                Spacing {
                    top: 10.0,
                    right: 0.0,
                    bottom: 0.0,
                    left: 0.0,
                },
            ),
            // Named bottom
            (
                "View C() { Column #[p: (bottom: 2)] {} }",
                Spacing {
                    top: 0.0,
                    right: 0.0,
                    bottom: 2.0,
                    left: 0.0,
                },
            ),
            // Named mixed
            (
                "View C() { Column #[p: (left: 5, bottom: 3)] {} }",
                Spacing {
                    top: 0.0,
                    right: 0.0,
                    bottom: 3.0,
                    left: 5.0,
                },
            ),
            // Named abbreviations (t, r, b, l)
            (
                "View C() { Column #[p: (t: 10, r: 8, b: 6, l: 4)] {} }",
                Spacing {
                    top: 10.0,
                    right: 8.0,
                    bottom: 6.0,
                    left: 4.0,
                },
            ),
            // Named horizontal / vertical / all / x / y shorthands
            (
                "View C() { Column #[p: (x: 5, y: 7)] {} }",
                Spacing {
                    top: 7.0,
                    right: 5.0,
                    bottom: 7.0,
                    left: 5.0,
                },
            ),
            (
                "View C() { Column #[p: (h: 12, v: 14)] {} }",
                Spacing {
                    top: 14.0,
                    right: 12.0,
                    bottom: 14.0,
                    left: 12.0,
                },
            ),
            ("View C() { Column #[p: (all: 42)] {} }", Spacing::all(42.0)),
            ("View C() { Column #[p: (a: 9)] {} }", Spacing::all(9.0)),
        ];

        for (source, expected_spacing) in test_cases {
            let parsed = parse(source);
            assert!(
                parsed.errors.is_empty(),
                "Failed to parse: {}\nErrors: {:?}",
                source,
                parsed.errors
            );
            let view = &parsed.views[0];
            let mut interp = Interpreter::new();
            let tree = interp.lower_view(view, &[]);
            let RenderNode::Box { name, attrs, .. } = &tree[0] else {
                panic!("expected a Box");
            };
            let style = interp.eval_container_style(name.as_str(), attrs);
            assert_eq!(
                style.padding.top, expected_spacing.top,
                "top mismatch for source: {}",
                source
            );
            assert_eq!(
                style.padding.right, expected_spacing.right,
                "right mismatch for source: {}",
                source
            );
            assert_eq!(
                style.padding.bottom, expected_spacing.bottom,
                "bottom mismatch for source: {}",
                source
            );
            assert_eq!(
                style.padding.left, expected_spacing.left,
                "left mismatch for source: {}",
                source
            );
        }
    }

    #[test]
    fn individual_margin_padding_properties_override() {
        use byard_core::atlas::layout::Spacing;

        let parsed = parse("View C() { Column #[p: (10), pt: 2, pb: 4, ml: 5, mt: 1] {} }");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        let RenderNode::Box { name, attrs, .. } = &tree[0] else {
            panic!("expected a Box");
        };
        let style = interp.eval_container_style(name.as_str(), attrs);
        // padding.top overridden to 2, padding.bottom overridden to 4, others stay 10
        assert_eq!(
            style.padding,
            Spacing {
                top: 2.0,
                right: 10.0,
                bottom: 4.0,
                left: 10.0
            }
        );
        // margins
        assert_eq!(
            style.margin,
            Spacing {
                top: 1.0,
                right: 0.0,
                bottom: 0.0,
                left: 5.0
            }
        );
    }
}
