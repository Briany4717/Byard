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
    Arg, AssignOp, Attr, AttrKind, ElementNode, Expr, Member, Param, PostfixOp, StrPart, ViewDecl,
};
use crate::symbol::Symbol;
use crate::util::closest_match;

/// Decimal places a `Slider` without an explicit `step` quantises its value to.
///
/// A continuous slider otherwise emits the full `f64` precision of a
/// pixel-derived ratio (e.g. `0.6035294…`); rounding keeps the bound value
/// readable. Authors who need a specific granularity set `step:` instead.
const SLIDER_DEFAULT_DECIMALS: i32 = 3;

/// Decimal places implied by `step`, via its shortest round-trip form
/// (`0.1 → 1`, `0.25 → 2`, `1.0 → 0`). Used so a stepped slider never emits a
/// value with more decimal places than the step itself — e.g. `step: 0.1`
/// landing on `6 × 0.1 = 0.6000000000000001` is rounded back to `0.6`. Capped
/// at 10 places (any real step is far coarser).
fn step_decimals(step: f64) -> i32 {
    match format!("{}", step.abs()).split_once('.') {
        Some((_, frac)) => i32::try_from(frac.len().min(10)).unwrap_or(0),
        None => 0,
    }
}

/// Rounds `val` to `decimals` decimal places (half-away-from-zero).
fn round_to_decimals(val: f64, decimals: i32) -> f64 {
    let factor = 10f64.powi(decimals);
    (val * factor).round() / factor
}

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
        /// The `var` signal bound via `bind:` or `value:` (M16: value widgets).
        bound_sig: Option<super::env::SignalId>,
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
    /// A texture-sampled image (M21).
    Image {
        /// Styling attributes (width, height, fit, radii, opacity, …).
        attrs: Vec<Attr>,
        /// The reactive scope that evaluates to the image source path/URL.
        src: ScopeId,
    },
}

/// A lowered reactive computation (see the module docs).
type Lowered = Box<dyn FnMut(&mut ReactiveCtx) -> Value>;

/// The per-instance parameter bindings produced by binding a user-view call's
/// arguments to the callee's declared parameters (RFC-0007 §3, M31). Each entry
/// is a reactive memo projecting the argument expression over the *parent*
/// scope, so a parameter fed a parent `var` stays live (RFC-0004); a literal
/// argument lowers to a constant memo with no dirty edges.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct InstanceBindings {
    /// Successfully bound `(param name, projecting memo)` pairs, in parameter
    /// declaration order.
    pub bindings: Vec<(Symbol, ScopeId)>,
}

/// The default-value expression of a parameter, if it declares one (RFC-0007
/// D-B / IMPL-47). Defaults are surfaced in M35; until then this is always
/// `None`, so every unbound parameter is required.
fn param_default(_param: &Param) -> Option<&Expr> {
    None
}

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
    /// Active design-token theme (M22, D5 layer 1).
    pub theme: super::theme::Theme,
    /// Parameterized fn definitions: `fn f(params) => body` stored as
    /// `(param names, body expr)`, indexed by `AstId` (M25).
    fn_table: Vec<(Vec<Symbol>, Expr)>,
    /// The resolved user-`View` registry for this program (RFC-0007 §1, M30).
    /// Built once from `ParsedFile::views` via [`Interpreter::load_views`]; a
    /// call whose name resolves here is a user-view instantiation, not a
    /// container.
    view_table: super::views::ViewTable,
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

    /// Builds the user-`View` registry (RFC-0007 §1, M30) from a whole file's
    /// views and stores it on the interpreter, so subsequent `lower_view`/
    /// `lower_element` calls can recognize and (M32+) expand user-view calls.
    /// Returns the load-time diagnostics — `IntrinsicShadowed` (IMPL-50) and any
    /// unguarded-cycle `RecursiveView` (RFC-0007 §4, M33) — which are also
    /// recorded in [`Interpreter::errors`].
    pub fn load_views(&mut self, views: &[ViewDecl]) -> Vec<CompileError> {
        let (table, mut diags) = super::views::ViewTable::build(views);
        // Static cycle detection over the call graph (M33).
        let graph = super::views::CallGraph::build(&table);
        if let Some((view, path)) = graph.unguarded_cycle(&table) {
            diags.push(CompileError::RecursiveView {
                span: table.decl(view).span,
                path,
            });
        }
        self.view_table = table;
        self.errors.extend(diags.iter().cloned());
        diags
    }

    /// The resolved user-`View` registry (for the reload pass and tests).
    #[must_use]
    pub fn view_table(&self) -> &super::views::ViewTable {
        &self.view_table
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
            Member::Fn {
                name, params, body, ..
            } => {
                if params.is_empty() {
                    // No-param fn: lower body to a memo (existing behavior).
                    self.define_let(name.clone(), body);
                } else {
                    // Parameterized fn (M25): store params+body in fn_table,
                    // bind Value::Fn(AstId) in env.
                    let id = crate::interp::env::AstId(
                        u32::try_from(self.fn_table.len()).unwrap_or(u32::MAX),
                    );
                    let param_names: Vec<Symbol> = params.iter().map(|p| p.name.clone()).collect();
                    self.fn_table.push((param_names, body.clone()));
                    self.env.push(name.clone(), Value::Fn(id));
                }
            }
            Member::Expr(e) => {
                if let Err(err) = self.eval_action(e) {
                    self.errors.push(err);
                }
            }
            Member::Inject { ty, name, span } => {
                // Resolve `inject T as name` from the ambient environment chain (M23).
                let ty_name = match ty {
                    crate::parser::ast::Type::Named { name: n, .. } => n.clone(),
                    crate::parser::ast::Type::Function { .. } => Symbol::intern("?"),
                };
                match self.env.resolve_inject(&ty_name).cloned() {
                    Some(val) => self.env.push(name.clone(), val),
                    None => self.errors.push(CompileError::UnresolvedInject {
                        span: *span,
                        name: ty_name.as_str().to_string(),
                    }),
                }
            }
            // elements / control flow / style handled in lower_members.
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

    // ── M23: Controller boundary ─────────────────────────────────────────

    /// Provides an ambient value keyed by `ty` to this view and its
    /// descendants (`inject T as name` resolution, RFC-0002 §inject).
    /// Call before [`lower_view`](Self::lower_view) so the environment is
    /// ready when the view body is evaluated.
    pub fn inject_provider(&mut self, ty: &str, value: Value) {
        self.env.provide(Symbol::intern(ty), value);
    }

    /// Applies a batch of pending I/O results from the async controller pool
    /// (RFC-0001 §5.1). Each callback receives a mutable reference to `self`
    /// and writes to whatever `var` signals it needs via [`write_var`](Self::write_var).
    /// Results are drained before the next [`tick`](Self::tick).
    pub fn apply_io_results(
        &mut self,
        results: impl IntoIterator<Item = Box<dyn FnOnce(&mut Self) + Send>>,
    ) {
        for f in results {
            f(self);
        }
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

    /// Reads a memo's current value (for the engine bridge and tests). Pulls it
    /// on demand if dirty.
    pub fn read_memo(&mut self, scope: ScopeId) -> Value {
        self.ctx.read_memo(scope)
    }

    // ── M31: argument → parameter binding (RFC-0007 §3) ──────────────────

    /// Projects one call argument into a reactive memo over the **parent** scope
    /// (the env active at the call site), so a parameter fed a parent `var` stays
    /// live (RFC-0004); a literal lowers to a constant memo with no dirty edges.
    fn project_arg(&mut self, expr: &Expr) -> ScopeId {
        let compute = self.lower_expr(expr, None);
        self.ctx.open_memo(compute)
    }

    /// Binds a single named argument (`name: value`, from `(...)` or `#[...]`) to
    /// the callee parameter of the same `Symbol`, filling `slots[i]` and emitting
    /// `UnknownParam`/`DuplicateParam` as needed (RFC-0007 §3/§6).
    fn bind_named_arg(
        &mut self,
        params: &[Param],
        callee: &str,
        name: &Symbol,
        value: &Expr,
        slots: &mut [Option<ScopeId>],
    ) {
        match params.iter().position(|p| &p.name == name) {
            Some(i) if slots[i].is_none() => {
                slots[i] = Some(self.project_arg(value));
            }
            Some(_) => self.errors.push(CompileError::DuplicateParam {
                span: value.span(),
                name: name.as_str().to_string(),
                callee: callee.to_string(),
            }),
            None => {
                let hint = closest_match(name.as_str(), params.iter().map(|p| p.name.as_str()))
                    .map(str::to_string);
                self.errors.push(CompileError::UnknownParam {
                    span: value.span(),
                    name: name.as_str().to_string(),
                    callee: callee.to_string(),
                    hint,
                });
            }
        }
    }

    /// Binds a user-view call's positional `content` and named `content`/`attrs`
    /// arguments to the callee's declared parameters, producing one reactive memo
    /// per bound parameter (RFC-0007 §3) and the §6 diagnostics
    /// (`ViewArityMismatch`/`UnknownParam`/`MissingParam`/`DuplicateParam`).
    ///
    /// Positional arguments (unnamed `(...)` entries) match by declaration order;
    /// named arguments (`name:` in `(...)` or `#[name: value]`) match by symbol.
    pub fn bind_args(&mut self, callee: &ViewDecl, call: &ElementNode) -> InstanceBindings {
        let params = &callee.params;
        let callee_name = callee.name.as_str().to_string();
        let mut slots: Vec<Option<ScopeId>> = vec![None; params.len()];
        let mut positional_count = 0usize;
        let mut next_positional = 0usize;

        // 1) `(...)` content: unnamed → positional by order; named → by symbol.
        for arg in &call.content {
            if let Some(name) = &arg.name {
                self.bind_named_arg(params, &callee_name, name, &arg.value, &mut slots);
            } else {
                positional_count += 1;
                if next_positional < params.len() {
                    let i = next_positional;
                    next_positional += 1;
                    let scope = self.project_arg(&arg.value);
                    if slots[i].is_some() {
                        self.errors.push(CompileError::DuplicateParam {
                            span: arg.value.span(),
                            name: params[i].name.as_str().to_string(),
                            callee: callee_name.clone(),
                        });
                    } else {
                        slots[i] = Some(scope);
                    }
                }
                // Excess positional args are reported once via the arity check
                // below.
            }
        }

        // 2) `#[name: value]` attrs: named arguments (events are not parameters).
        for attr in &call.attrs {
            if let AttrKind::Prop { value } = &attr.kind {
                self.bind_named_arg(params, &callee_name, &attr.name, value, &mut slots);
            }
        }

        // 3) Arity: more positional args than the callee declares (RFC-0007 §6).
        if positional_count > params.len() {
            self.errors.push(CompileError::ViewArityMismatch {
                span: call.span,
                name: callee_name.clone(),
                expected: params.len(),
                found: positional_count,
            });
        }

        // 4) Missing required parameters. Defaults (IMPL-47) land in M35; until
        //    then every unbound parameter is required.
        for (i, slot) in slots.iter().enumerate() {
            if slot.is_none() && param_default(&params[i]).is_none() {
                self.errors.push(CompileError::MissingParam {
                    span: call.span,
                    name: params[i].name.as_str().to_string(),
                    callee: callee_name.clone(),
                });
            }
        }

        InstanceBindings {
            bindings: slots
                .into_iter()
                .enumerate()
                .filter_map(|(i, s)| s.map(|sc| (params[i].name.clone(), sc)))
                .collect(),
        }
    }

    // ── element lowering (RFC-0005) ─────────────────────────────────────

    /// Resolves the `bind:` or `value:` attribute of a value widget to a
    /// `SignalId`. Returns `None` if no such attribute exists or it doesn't
    /// name a `var` (M16).
    fn resolve_bind_sig(&self, attrs: &[Attr]) -> Option<super::env::SignalId> {
        use crate::parser::ast::Expr;
        for attr in attrs {
            if matches!(attr.name.as_str(), "bind" | "value") {
                if let AttrKind::Prop {
                    value: Expr::Ident(name, _),
                } = &attr.kind
                {
                    if let Some(super::env::Value::Signal(sig)) = self.env.lookup(name) {
                        return Some(*sig);
                    }
                }
            }
        }
        None
    }

    /// Lowers an element to a [`RenderNode`], validating it against the §5
    /// attribute contract first (diagnostics accumulate in [`Interpreter::errors`]).
    /// `known_views` are user `ViewDecl` names in scope.
    pub fn lower_element(&mut self, el: &ElementNode, known_views: &[&str]) -> RenderNode {
        self.errors.extend(validate_element(el, known_views));
        // A name that resolves in the view table — and is not an intrinsic, which
        // always wins (IMPL-50) — is a user-view call, expanded in its own
        // instance scope rather than lowered as a generic container (RFC-0007 §2).
        if super::intrinsics::lookup(el.name.as_str()).is_none()
            && self.view_table.contains(&el.name)
        {
            return self.lower_user_view_call(el, known_views);
        }
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
                        bound_sig: None,
                    }
                } else {
                    RenderNode::Text {
                        attrs: el.attrs.clone(),
                        content,
                    }
                }
            }
            "Spacer" => RenderNode::Spacer,
            // Image intrinsic → TextureSampler pipeline (M21).
            // Syntax: Image("path.jpg") #[fit: .cover, width: 200, height: 150]
            "Image" => {
                let src_expr = el.content.first().map_or_else(
                    || Expr::StrLit(vec![], crate::diagnostics::Span::new(0, 0)),
                    |c| c.value.clone(),
                );
                let src = self.bind_value(&src_expr);
                RenderNode::Image {
                    attrs: el.attrs.clone(),
                    src,
                }
            }
            // Value widgets: resolve bound signal and keep as leaf nodes (M16/M19).
            "Toggle" | "Slider" | "TextField" => {
                let bound_sig = self.resolve_bind_sig(&el.attrs);
                RenderNode::Box {
                    name: el.name.clone(),
                    attrs: el.attrs.clone(),
                    children: Vec::new(),
                    action: el.action.clone(),
                    bound_sig,
                }
            }
            _ => {
                // Box / Column / Row / ScrollView and any other container.
                let children = self.lower_members(&el.children, known_views);
                RenderNode::Box {
                    name: el.name.clone(),
                    attrs: el.attrs.clone(),
                    children,
                    action: el.action.clone(),
                    bound_sig: None,
                }
            }
        }
    }

    /// Lowers a user-`View` call site into its instantiated subtree (RFC-0007
    /// §2). M30 establishes the single hook; M31 binds arguments → parameters,
    /// M32 expands the callee body in a fresh instance scope, M33 bounds
    /// recursion.
    fn lower_user_view_call(&mut self, el: &ElementNode, known_views: &[&str]) -> RenderNode {
        // M31: bind arguments → parameters here.
        // M32: open instance scope, lower `callee.body`, splice, truncate.
        // For M30 this reproduces the historical generic-container behavior so
        // the gate stays green with no semantic change yet.
        let children = self.lower_members(&el.children, known_views);
        RenderNode::Box {
            name: el.name.clone(),
            attrs: el.attrs.clone(),
            children,
            action: el.action.clone(),
            bound_sig: None,
        }
    }

    /// Lowers a slice of `Member`s into child `RenderNode`s, handling
    /// `Element`, `When`, and `For` (M20).
    fn lower_members(&mut self, members: &[Member], known_views: &[&str]) -> Vec<RenderNode> {
        let mut nodes = Vec::new();
        for m in members {
            self.lower_member_into(m, known_views, &mut nodes);
        }
        nodes
    }

    fn lower_member_into(
        &mut self,
        member: &Member,
        known_views: &[&str],
        out: &mut Vec<RenderNode>,
    ) {
        match member {
            Member::Element(e) => {
                out.push(self.lower_element(e, known_views));
            }
            Member::When {
                cond, then, els, ..
            } => {
                let val = self.eval_pure(cond);
                let body = if val.as_bool().unwrap_or(false) {
                    then.as_slice()
                } else {
                    match els {
                        Some(els) => els.as_slice(),
                        None => return,
                    }
                };
                for m in body {
                    self.lower_member_into(m, known_views, out);
                }
            }
            Member::For {
                var, iter, body, ..
            } => {
                let list = self.eval_pure(iter);
                if let Value::List(items) = list {
                    for item in items {
                        let snapshot = self.env.len();
                        // Create a one-tick signal to hold the item value.
                        let item_sig = self.ctx.create_signal(item);
                        self.env.push(var.clone(), Value::Signal(item_sig));
                        for m in body.as_slice() {
                            self.lower_member_into(m, known_views, out);
                        }
                        self.env.truncate(snapshot);
                    }
                }
            }
            _ => {}
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
        // Rebuild the handler set from the fresh layout, but keep the in-flight
        // gesture state (a pending `down`, the focused element) so a tap that
        // spans this re-render is still recognized (RFC-0003 E4).
        self.router.clear_handlers();
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
            RenderNode::Image { attrs, .. } => {
                let w = self.eval_int_prop(attrs, "width").unwrap_or(100) as f32;
                let h = self.eval_int_prop(attrs, "height").unwrap_or(100) as f32;
                let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                flat_ids.push(id);
                Ok(id)
            }
            RenderNode::Text { attrs, content } => {
                let text = match self.binding_value(*content) {
                    Some(Value::Str(s)) => s,
                    other => other.map_or_else(String::new, |v| format!("{v:?}")),
                };
                let typo_size = self
                    .eval_str_prop(attrs, "typo")
                    .and_then(|t| super::theme::resolve_typo(&t))
                    .map(|s| s as i64);
                #[allow(clippy::cast_precision_loss)]
                let font_size = self
                    .eval_int_prop(attrs, "size")
                    .or(typo_size)
                    .unwrap_or(self.theme.font_size as i64) as f32;
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
                // Value widgets are leaf nodes with intrinsic default sizes (M16/M19).
                match name.as_str() {
                    "Toggle" => {
                        let w = self.eval_int_prop(attrs, "width").unwrap_or(50) as f32;
                        let h = self.eval_int_prop(attrs, "height").unwrap_or(30) as f32;
                        let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                        flat_ids.push(id);
                        return Ok(id);
                    }
                    "Slider" => {
                        let w = self.eval_int_prop(attrs, "width").unwrap_or(200) as f32;
                        let h = self.eval_int_prop(attrs, "height").unwrap_or(24) as f32;
                        let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                        flat_ids.push(id);
                        return Ok(id);
                    }
                    "TextField" => {
                        let w = self.eval_int_prop(attrs, "width").unwrap_or(200) as f32;
                        let h = self.eval_int_prop(attrs, "height").unwrap_or(36) as f32;
                        let id = self.atlas.add_leaf(LeafSize::new(w, h))?;
                        flat_ids.push(id);
                        return Ok(id);
                    }
                    _ => {}
                }
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
                    // M22: fall back to theme on-surface color when unset.
                    let color = self
                        .eval_color_prop(attrs, "color")
                        .unwrap_or(self.theme.on_surface);
                    // M22: resolve `typo:` token to font size; inline `size:` overrides.
                    let typo_size = self
                        .eval_str_prop(attrs, "typo")
                        .and_then(|t| super::theme::resolve_typo(&t))
                        .map(|s| s as i64);
                    let size =
                        self.eval_int_prop(attrs, "size")
                            .or(typo_size)
                            .unwrap_or(self.theme.font_size as i64) as f32;
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
                        let elem_idx = self.atlas.node_index(atlas_node);
                        self.register_event_attrs(attrs, hit_rect, elem_idx);
                    }
                }
            }
            RenderNode::Box {
                name,
                attrs,
                children,
                action,
                bound_sig,
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
                    let radii = self.resolve_radii(attrs, "radius");
                    // `border` is a Color (catalog DECORATION); a present border
                    // draws a 2px ring of that colour.
                    let border_color = self.eval_color_prop(attrs, "border");
                    let border_width = if border_color.is_some() { 2.0 } else { 0.0 };
                    // `shadow` is a token (`sm`/`md`/`lg`) → an offset+blur drop
                    // shadow; any other non-empty value falls back to `md`.
                    let (shadow_dy, shadow_blur, shadow_color) =
                        match self.eval_str_prop(attrs, "shadow").as_deref() {
                            Some("sm") => (1.0_f32, 3.0_f32, Some(0x4400_0000_i64)),
                            Some("lg") => (6.0, 16.0, Some(0x6600_0000)),
                            Some("none") | None => (0.0, 0.0, None),
                            Some(_) => (3.0, 8.0, Some(0x5500_0000)),
                        };
                    let shadow_dx = 0.0_f32;
                    let opacity = self
                        .eval_float_prop(attrs, "opacity")
                        .map_or(1.0, |v| v as f32);
                    let is_decorated = border_width > 0.0
                        || shadow_blur > 0.0
                        || (opacity - 1.0).abs() > f32::EPSILON;
                    // `Toggle`/`Slider` own their visuals (track/fill/thumb) and
                    // treat `bg` as the *accent* colour, not a full-rect fill —
                    // painting the rect here would draw a slab behind the control.
                    let owns_visuals = matches!(name.as_str(), "Toggle" | "Slider");
                    if let (false, Some(color)) = (owns_visuals, bg) {
                        let base = byard_core::BoxInstance {
                            rect: [rect.x, rect.y, rect.width, rect.height],
                            color: super::intrinsics::color_to_rgba(color, false),
                            radii,
                        };
                        let border_rgba = border_color
                            .map_or([0.0; 4], |c| super::intrinsics::color_to_rgba(c, false));
                        let shadow_rgba = shadow_color
                            .map_or([0.0; 4], |c| super::intrinsics::color_to_rgba(c, true));
                        let translucent = (opacity - 1.0).abs() > f32::EPSILON;
                        if translucent {
                            // A translucent box blends its fill as one unit on the
                            // decorated pipeline (leaf showcase boxes); keep it whole.
                            frame.push_decorated(byard_core::frame::DecoratedBox {
                                base,
                                border_width,
                                border_color: border_rgba,
                                shadow_dx,
                                shadow_dy,
                                shadow_blur,
                                shadow_color: shadow_rgba,
                                opacity,
                                // Re-walked and re-emitted every tick;
                                // mirror Text's always-dirty lowering.
                                dirty: true,
                            });
                        } else if is_decorated || border_color.is_some() {
                            // Paint the opaque fill on the SolidBox pass so it stays
                            // *behind* this container's children (they also paint as
                            // solids, pushed after it — and the decorated pass runs
                            // after every solid). Then add the border/shadow as a
                            // decorated overlay whose interior is transparent: it only
                            // strokes the edge and casts the shadow, so it can never
                            // occlude the children drawn beneath it (fixes the
                            // parent-card-over-child-widget z-order bug).
                            frame.push_instance(base);
                            frame.push_decorated(byard_core::frame::DecoratedBox {
                                base: byard_core::BoxInstance {
                                    color: [0.0; 4],
                                    ..base
                                },
                                border_width,
                                border_color: border_rgba,
                                shadow_dx,
                                shadow_dy,
                                shadow_blur,
                                shadow_color: shadow_rgba,
                                opacity: 1.0,
                                dirty: true,
                            });
                        } else {
                            frame.push_instance(base);
                        }
                    }

                    let element_name = name.as_str();
                    let hit_rect =
                        crate::interp::intrinsics::inflate_hit_rect(current_rect, parent_rect);
                    let elem_idx = self.atlas.node_index(atlas_node);

                    // ── Widget-specific visual lowering & handler registration (M16/M19) ──
                    match element_name {
                        "Toggle" => {
                            self.render_toggle(
                                *bound_sig,
                                attrs,
                                current_rect,
                                hit_rect,
                                elem_idx,
                                frame,
                            );
                        }
                        "Slider" => {
                            self.render_slider(
                                *bound_sig,
                                attrs,
                                current_rect,
                                hit_rect,
                                elem_idx,
                                frame,
                            );
                        }
                        "TextField" => {
                            self.render_text_field(
                                *bound_sig,
                                attrs,
                                current_rect,
                                hit_rect,
                                elem_idx,
                                frame,
                            );
                        }
                        _ => {
                            // General interactive elements: register event-attr handlers.
                            let has_event_attrs = attrs
                                .iter()
                                .any(|a| matches!(a.kind, AttrKind::Event { .. }));
                            let is_interactive = matches!(element_name, "Button")
                                || has_event_attrs
                                || action.is_some();

                            if is_interactive {
                                self.register_event_attrs(attrs, hit_rect, elem_idx);

                                if let Some(action_expr) = action {
                                    if let Ok(action_closure) = self.lower_action(action_expr, None)
                                    {
                                        if let Some(idx) = elem_idx {
                                            self.router.on(
                                                idx,
                                                hit_rect,
                                                crate::interp::events::EventKind::Tap,
                                                action_closure,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // ── `focused:` reflected prop → register as focusable (M16/M18) ──
                    // TextFields register their own focusable inside render_text_field
                    // to avoid double-registration.
                    if element_name != "TextField" {
                        self.register_focusable(attrs, hit_rect, elem_idx);
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
            RenderNode::Image { attrs, src } => {
                if let Ok(Some(rect)) = self.atlas.resolved_rect(atlas_node) {
                    let src_val = self
                        .binding_value(*src)
                        .and_then(|v| if let Value::Str(s) = v { Some(s) } else { None })
                        .unwrap_or_default();
                    let fit = self.eval_fit_prop(attrs);
                    let radii = self.resolve_radii(attrs, "radius");
                    let opacity = self
                        .eval_float_prop(attrs, "opacity")
                        .map_or(1.0, |v| v as f32);
                    frame.push_texture(byard_core::frame::TextureSampler {
                        rect: [rect.x, rect.y, rect.width, rect.height],
                        src: src_val,
                        fit,
                        radii,
                        opacity,
                        // Re-emitted every tick; mirror Text's
                        // always-dirty lowering.
                        dirty: true,
                    });
                }
            }
        }
    }

    // ── Widget rendering helpers (M16/M19) ─────────────────────────────

    /// Renders a `Toggle` widget: track + thumb (M19), and registers a Tap
    /// handler to flip the bound bool (M16).
    #[allow(clippy::too_many_arguments)]
    fn render_toggle(
        &mut self,
        bound_sig: Option<super::env::SignalId>,
        attrs: &[Attr],
        rect: crate::interp::intrinsics::Rect,
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        let is_on = bound_sig.is_some_and(|s| self.ctx.peek_signal(s).as_bool().unwrap_or(false));

        // The full-height pill track. `bg` is the ON accent (default: theme
        // primary); OFF is a muted surface tint.
        let accent = self
            .eval_color_prop(attrs, "bg")
            .unwrap_or(self.theme.primary);
        let track_color = if is_on {
            super::intrinsics::color_to_rgba(accent, false)
        } else {
            [0.40_f32, 0.42, 0.48, 1.0]
        };
        let radius = rect.h / 2.0;
        frame.push_instance(byard_core::BoxInstance {
            rect: [rect.x, rect.y, rect.w, rect.h],
            color: track_color,
            radii: [radius; 4],
        });

        // Thumb: a white circle inset from the track edges, sliding L↔R.
        let pad = (rect.h * 0.12).max(2.0);
        let thumb_size = (rect.h - pad * 2.0).max(2.0);
        let thumb_y = rect.y + pad;
        let thumb_x = if is_on {
            rect.x + rect.w - thumb_size - pad
        } else {
            rect.x + pad
        };
        frame.push_instance(byard_core::BoxInstance {
            rect: [thumb_x, thumb_y, thumb_size, thumb_size],
            color: [1.0, 1.0, 1.0, 1.0],
            radii: [thumb_size / 2.0; 4],
        });

        // Tap handler to flip the bool (M16).
        if let (Some(sig), Some(idx)) = (bound_sig, elem_idx) {
            let flip: super::events::Action = Box::new(move |ctx, _| {
                let cur = ctx.peek_signal(sig).as_bool().unwrap_or(false);
                ctx.write_signal(sig, Value::Bool(!cur));
            });
            self.router
                .on(idx, hit_rect, super::events::EventKind::Tap, flip);
        }
    }

    /// Renders a `Slider` widget: track + fill + thumb (M19), and registers
    /// PointerDown + PointerDrag handlers to write the value (M16).
    #[allow(clippy::too_many_arguments)]
    fn render_slider(
        &mut self,
        bound_sig: Option<super::env::SignalId>,
        attrs: &[Attr],
        rect: crate::interp::intrinsics::Rect,
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        // Keep the authored `f64` values for the value-write path: computing the
        // emitted value in `f64` avoids the `f32`→`f64` widening artifact (a drag
        // landing on 0.6 was stored as `f64::from(0.6_f32)` =
        // 0.6000000238418579). The `f32` casts below are only for pixel-space
        // visual layout (track/fill/thumb), where the noise is invisible.
        let min_f = self.eval_float_prop(attrs, "min").unwrap_or(0.0);
        let max_f = self.eval_float_prop(attrs, "max").unwrap_or(1.0);
        let step_f = self.eval_float_prop(attrs, "step");
        let min = min_f as f32;
        let max = max_f as f32;
        let cur_val = bound_sig.map_or(min, |s| match self.ctx.peek_signal(s) {
            Value::Float(f) => f as f32,
            Value::Int(n) => n as f32,
            _ => min,
        });
        let t = if (max - min).abs() > f32::EPSILON {
            ((cur_val - min) / (max - min)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // `bg` is the fill accent (default: theme primary); the unfilled track
        // is a muted tint.
        let accent = self
            .eval_color_prop(attrs, "bg")
            .unwrap_or(self.theme.primary);
        let accent_rgba = super::intrinsics::color_to_rgba(accent, false);

        // Track (unfilled remainder).
        let track_h = (rect.h * 0.28).clamp(4.0, 8.0);
        let track_y = rect.y + (rect.h - track_h) / 2.0;
        let track_r = track_h / 2.0;
        frame.push_instance(byard_core::BoxInstance {
            rect: [rect.x, track_y, rect.w, track_h],
            color: [0.40, 0.42, 0.48, 1.0],
            radii: [track_r; 4],
        });

        // Fill up to the thumb.
        let fill_w = t * rect.w;
        if fill_w > 0.0 {
            frame.push_instance(byard_core::BoxInstance {
                rect: [rect.x, track_y, fill_w, track_h],
                color: accent_rgba,
                radii: [track_r; 4],
            });
        }

        // Thumb: white circle with a thin accent ring (drawn as accent disc
        // under a slightly smaller white disc).
        let thumb_size = (rect.h * 0.85).clamp(14.0, 22.0);
        let thumb_x = rect.x + t * (rect.w - thumb_size);
        let thumb_y = rect.y + (rect.h - thumb_size) / 2.0;
        frame.push_instance(byard_core::BoxInstance {
            rect: [thumb_x, thumb_y, thumb_size, thumb_size],
            color: accent_rgba,
            radii: [thumb_size / 2.0; 4],
        });
        let inner = thumb_size - 5.0;
        frame.push_instance(byard_core::BoxInstance {
            rect: [thumb_x + 2.5, thumb_y + 2.5, inner, inner],
            color: [1.0, 1.0, 1.0, 1.0],
            radii: [inner / 2.0; 4],
        });

        // Handlers: PointerDown + PointerDrag (M16).
        if let (Some(sig), Some(idx)) = (bound_sig, elem_idx) {
            let track_x = rect.x;
            let track_w = rect.w;
            let make_drag_action =
                |min: f64, max: f64, step: Option<f64>| -> super::events::Action {
                    Box::new(move |ctx, _| {
                        let pos = super::events::CURRENT_EVENT_POS.with(std::cell::Cell::get);
                        // Pixel positions are `f32`; widen before the value math so
                        // the stored value never carries `f32` rounding noise.
                        let t = ((f64::from(pos.0) - f64::from(track_x)) / f64::from(track_w))
                            .clamp(0.0, 1.0);
                        let raw = min + t * (max - min);
                        // Quantise so the value never carries more decimals than
                        // the step (or, with no step, a readable default) — see
                        // `step_decimals`/`SLIDER_DEFAULT_DECIMALS`.
                        let val = match step {
                            Some(s) => round_to_decimals((raw / s).round() * s, step_decimals(s)),
                            None => round_to_decimals(raw, SLIDER_DEFAULT_DECIMALS),
                        };
                        ctx.write_signal(sig, Value::Float(val));
                    })
                };
            self.router.on(
                idx,
                hit_rect,
                super::events::EventKind::PointerDown,
                make_drag_action(min_f, max_f, step_f),
            );
            self.router.on(
                idx,
                hit_rect,
                super::events::EventKind::PointerDrag,
                make_drag_action(min_f, max_f, step_f),
            );
        }
    }

    /// Renders a `TextField` widget: background box + text/placeholder (M19),
    /// and registers keyboard handlers for text input (M16/M17).
    #[allow(clippy::too_many_arguments)]
    fn render_text_field(
        &mut self,
        bound_sig: Option<super::env::SignalId>,
        attrs: &[Attr],
        rect: crate::interp::intrinsics::Rect,
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
        frame: &mut byard_core::frame::RenderFrame,
    ) {
        let placeholder = self.eval_str_prop(attrs, "placeholder").unwrap_or_default();
        let cur_text = bound_sig
            .map(|s| match self.ctx.peek_signal(s) {
                Value::Str(t) => t,
                _ => String::new(),
            })
            .unwrap_or_default();

        let (display_text, is_placeholder) = if cur_text.is_empty() {
            (placeholder, true)
        } else {
            (cur_text, false)
        };

        let text_color = if is_placeholder {
            0x0088_8888_i64
        } else {
            0x00ff_ffff_i64
        };
        let font_size = self.eval_int_prop(attrs, "size").unwrap_or(16) as f32;
        let is_focused = elem_idx.is_some_and(|i| self.router.is_focused(i));

        // Focus underline (Material-style): a thin accent bar along the bottom
        // edge when the field holds focus.
        if is_focused {
            let bar_h = 2.0_f32;
            frame.push_instance(byard_core::BoxInstance {
                rect: [rect.x, rect.y + rect.h - bar_h, rect.w, bar_h],
                color: super::intrinsics::color_to_rgba(self.theme.primary, false),
                radii: [0.0; 4],
            });
        }

        let pad_x = 10.0_f32;
        let text_x = rect.x + pad_x;
        let text_y = rect.y + (rect.h - font_size) / 2.0;
        if !display_text.is_empty() {
            frame.push_text(byard_core::TextLine {
                x: text_x,
                y: text_y,
                text: display_text.clone(),
                font_size,
                color: super::intrinsics::color_to_rgba(text_color, false),
                dirty: true,
            });
        }

        // Caret at the end of the entered text while focused (M17/M19).
        if is_focused {
            let measured = if is_placeholder {
                0.0
            } else {
                self.measure_text(&display_text, font_size).0
            };
            frame.push_instance(byard_core::BoxInstance {
                rect: [text_x + measured + 1.0, text_y, 1.5, font_size],
                color: [1.0, 1.0, 1.0, 1.0],
                radii: [0.0; 4],
            });
        }

        // Handlers: TextInput appends, KeyDown handles Backspace/Enter/Tab (M16/M17).
        if let (Some(sig), Some(idx)) = (bound_sig, elem_idx) {
            // TextInput: append typed text
            let text_input: super::events::Action = Box::new(move |ctx, payload| {
                if let Some(Value::Str(ch)) = payload {
                    let cur = match ctx.peek_signal(sig) {
                        Value::Str(s) => s,
                        _ => String::new(),
                    };
                    ctx.write_signal(sig, Value::Str(cur + ch.as_str()));
                }
            });
            self.router.on(
                idx,
                hit_rect,
                super::events::EventKind::TextInput,
                text_input,
            );

            // KeyDown: Backspace deletes, Enter/Escape handled (submit fires via Change)
            let key_down: super::events::Action = Box::new(move |ctx, payload| {
                if let Some(Value::Str(key)) = payload {
                    match key.as_str() {
                        "Backspace" => {
                            let cur = match ctx.peek_signal(sig) {
                                Value::Str(s) => s,
                                _ => String::new(),
                            };
                            let mut s = cur;
                            s.pop();
                            ctx.write_signal(sig, Value::Str(s));
                        }
                        "Delete" => {
                            ctx.write_signal(sig, Value::Str(String::new()));
                        }
                        _ => {}
                    }
                }
            });
            self.router
                .on(idx, hit_rect, super::events::EventKind::KeyDown, key_down);

            // Change event: write-back from platform (E1).
            self.router.on(
                idx,
                hit_rect,
                super::events::EventKind::Change,
                super::events::write_back_action(sig),
            );

            // Register as focusable so Tab and click steal focus (M18).
            // TextField uses its own focused-var if provided via `focused:` attr;
            // otherwise we create a dummy signal just for the focusable registry.
            let focused_sig = self.resolve_focused_sig(attrs);
            let fsig = focused_sig.unwrap_or_else(|| self.ctx.create_signal(Value::Bool(false)));
            self.router.focusable(idx, hit_rect, fsig);
        }
    }

    /// Resolves the `focused:` attribute to a `SignalId`, if present.
    fn resolve_focused_sig(&self, attrs: &[Attr]) -> Option<super::env::SignalId> {
        use crate::parser::ast::Expr;
        for attr in attrs {
            if attr.name.as_str() == "focused" {
                if let AttrKind::Prop {
                    value: Expr::Ident(name, _),
                } = &attr.kind
                {
                    if let Some(Value::Signal(sig)) = self.env.lookup(name) {
                        return Some(*sig);
                    }
                }
            }
        }
        None
    }

    /// Registers handlers for all event-kind attrs (`#[tap => …]`, etc.).
    fn register_event_attrs(
        &mut self,
        attrs: &[Attr],
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
    ) {
        for attr in attrs {
            if let AttrKind::Event { payload, action } = &attr.kind {
                let event_kind = match attr.name.as_str() {
                    "tap" => super::events::EventKind::Tap,
                    "pointer_down" => super::events::EventKind::PointerDown,
                    "pointer_up" => super::events::EventKind::PointerUp,
                    "pointer_move" => super::events::EventKind::PointerMove,
                    "scroll" => super::events::EventKind::Scroll,
                    "wheel" => super::events::EventKind::Wheel,
                    "change" => super::events::EventKind::Change,
                    "key_down" => super::events::EventKind::KeyDown,
                    "key_up" => super::events::EventKind::KeyUp,
                    "text_input" => super::events::EventKind::TextInput,
                    _ => continue,
                };
                if let Ok(closure) = self.lower_action(action, payload.clone()) {
                    if let Some(idx) = elem_idx {
                        self.router.on(idx, hit_rect, event_kind, closure);
                    }
                }
            }
        }
    }

    /// Registers an element as focusable if it has a `focused:` prop attr (M16/M18).
    fn register_focusable(
        &mut self,
        attrs: &[Attr],
        hit_rect: crate::interp::intrinsics::Rect,
        elem_idx: Option<u32>,
    ) {
        if let Some(sig) = self.resolve_focused_sig(attrs) {
            if let Some(idx) = elem_idx {
                self.router.focusable(idx, hit_rect, sig);
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

    fn eval_float_prop(&mut self, attrs: &[Attr], name: &str) -> Option<f64> {
        attrs.iter().find_map(|a| {
            if a.name.as_str() == name {
                if let AttrKind::Prop { value } = &a.kind {
                    let val = self.eval_pure(value);
                    return match val {
                        Value::Float(f) => Some(f),
                        Value::Int(n) => Some(n as f64),
                        _ => None,
                    };
                }
            }
            None
        })
    }

    fn eval_str_prop(&mut self, attrs: &[Attr], name: &str) -> Option<String> {
        attrs.iter().find_map(|a| {
            if a.name.as_str() == name {
                if let AttrKind::Prop { value } = &a.kind {
                    let val = self.eval_pure(value);
                    return match val {
                        Value::Str(s) => Some(s),
                        _ => None,
                    };
                }
            }
            None
        })
    }

    fn eval_fit_prop(&mut self, attrs: &[Attr]) -> byard_core::frame::ImageFit {
        match self.eval_str_prop(attrs, "fit").as_deref() {
            Some("contain") => byard_core::frame::ImageFit::Contain,
            Some("cover") => byard_core::frame::ImageFit::Cover,
            Some("none") => byard_core::frame::ImageFit::None,
            _ => byard_core::frame::ImageFit::Fill,
        }
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
                        style.padding = self.resolve_spacing(value, "p");
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
                    "m" | "margin" => {
                        style.margin = self.resolve_spacing(value, "m");
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

    /// Resolves a `Len`-typed `p`/`m` attribute value into a `Spacing` quad
    /// (RFC-0005 §1 erratum), emitting span-anchored `CompileError`s
    /// for the four error classes:
    ///
    /// - an unknown side name → [`CompileError::UnknownAttribute`] with a hint;
    /// - a side set twice, an axis shorthand plus one of its component sides, or
    ///   a tuple mixing named and positional fields →
    ///   [`CompileError::ConflictingSpacingField`];
    /// - a non-numeric side value → [`CompileError::AttributeTypeMismatch`];
    /// - a positional tuple of arity 3 or > 4 → [`CompileError::ArityMismatch`].
    ///
    /// Accepted forms: scalar (`p: 5`), inferred pair (`p: (vertical, horizontal)`),
    /// inferred quad CSS `(top, right, bottom, left)`, and the verbose named form
    /// (`p: (top: 4, horizontal: 8)`). A single parenthesized value parses to the
    /// inner expression, so it arrives as a scalar.
    fn resolve_spacing(&mut self, expr: &Expr, prop: &str) -> byard_core::atlas::layout::Spacing {
        use byard_core::atlas::layout::Spacing;
        match expr {
            Expr::Tuple(args, span) => {
                let any_named = args.iter().any(|a| a.name.is_some());
                let all_named = args.iter().all(|a| a.name.is_some());
                if any_named && !all_named {
                    self.errors.push(CompileError::ConflictingSpacingField {
                        span: *span,
                        message: "a spacing tuple cannot mix named and positional fields"
                            .to_string(),
                    });
                    return Spacing::default();
                }
                if all_named {
                    self.resolve_named_spacing(args)
                } else {
                    self.resolve_positional_spacing(args, *span, prop)
                }
            }
            other => {
                let val = self.eval_pure(other);
                if let Some(v) = spacing_value(&val) {
                    Spacing::all(v)
                } else {
                    self.errors.push(CompileError::AttributeTypeMismatch {
                        span: other.span(),
                        expected: "a length (an integer)".to_string(),
                    });
                    Spacing::default()
                }
            }
        }
    }

    /// Verbose named spacing form (`p: (top: 4, horizontal: 8)`).
    fn resolve_named_spacing(&mut self, args: &[Arg]) -> byard_core::atlas::layout::Spacing {
        use byard_core::atlas::layout::Spacing;
        const SIDES: &[&str] = &["top", "bottom", "left", "right", "horizontal", "vertical"];

        let (mut top, mut right, mut bottom, mut left) = (None, None, None, None);
        for arg in args {
            // `all_named` guarantees a name is present.
            let Some(name) = &arg.name else { continue };
            let span = arg.value.span();
            let val = self.eval_pure(&arg.value);
            let Some(v) = spacing_value(&val) else {
                self.errors.push(CompileError::AttributeTypeMismatch {
                    span,
                    expected: "a length (an integer)".to_string(),
                });
                continue;
            };
            match name.as_str() {
                "top" => assign_side(&mut top, v, "top", span, &mut self.errors),
                "bottom" => assign_side(&mut bottom, v, "bottom", span, &mut self.errors),
                "left" => assign_side(&mut left, v, "left", span, &mut self.errors),
                "right" => assign_side(&mut right, v, "right", span, &mut self.errors),
                "horizontal" => {
                    assign_side(&mut left, v, "left", span, &mut self.errors);
                    assign_side(&mut right, v, "right", span, &mut self.errors);
                }
                "vertical" => {
                    assign_side(&mut top, v, "top", span, &mut self.errors);
                    assign_side(&mut bottom, v, "bottom", span, &mut self.errors);
                }
                unknown => {
                    let hint = crate::util::closest_match(unknown, SIDES.iter().copied())
                        .map(str::to_string);
                    self.errors.push(CompileError::UnknownAttribute {
                        span,
                        name: unknown.to_string(),
                        hint,
                    });
                }
            }
        }
        Spacing {
            top: top.unwrap_or(0.0),
            right: right.unwrap_or(0.0),
            bottom: bottom.unwrap_or(0.0),
            left: left.unwrap_or(0.0),
        }
    }

    /// Inferred positional spacing forms: pair `(vertical, horizontal)` or quad
    /// CSS `(top, right, bottom, left)`. Any other arity is an error.
    fn resolve_positional_spacing(
        &mut self,
        args: &[Arg],
        span: Span,
        prop: &str,
    ) -> byard_core::atlas::layout::Spacing {
        use byard_core::atlas::layout::Spacing;
        let mut vals = Vec::with_capacity(args.len());
        for arg in args {
            let val = self.eval_pure(&arg.value);
            if let Some(v) = spacing_value(&val) {
                vals.push(v);
            } else {
                self.errors.push(CompileError::AttributeTypeMismatch {
                    span: arg.value.span(),
                    expected: "a length (an integer)".to_string(),
                });
                vals.push(0.0);
            }
        }
        match vals.len() {
            2 => Spacing::symmetric(vals[0], vals[1]),
            4 => Spacing {
                top: vals[0],
                right: vals[1],
                bottom: vals[2],
                left: vals[3],
            },
            n => {
                self.errors.push(CompileError::ArityMismatch {
                    span,
                    name: prop.to_string(),
                    expected: 4,
                    found: n,
                });
                Spacing::default()
            }
        }
    }

    /// Resolves a `radius`-typed attribute into per-corner radii
    /// `[top_left, top_right, bottom_right, bottom_left]` — the exact order
    /// `BoxInstance::radii`/`TextureSampler::radii` expect (`frame.rs`).
    ///
    /// RFC-0005 §"Decoration" documents `radius: Len` as "scalar = all, quad =
    /// per-corner". Accepted forms: a scalar (`radius: 16`, all four
    /// corners) and the positional CSS-order quad (`radius: (4, 8, 12, 16)`).
    /// Unlike `p`/`m`'s generic `Len` contract, there is no pair shorthand and
    /// no named-field form for `radius` — the RFC documents only scalar/quad,
    /// so this resolver doesn't invent additional surface. A non-4 tuple
    /// arity is a `CompileError::ArityMismatch`; a non-numeric corner is an
    /// `AttributeTypeMismatch`; a named field is a `ConflictingSpacingField`
    /// (reusing the existing diagnostic — the message states the real cause).
    fn resolve_radii(&mut self, attrs: &[Attr], name: &str) -> [f32; 4] {
        let Some(attr) = attrs.iter().find(|a| a.name.as_str() == name) else {
            return [0.0; 4];
        };
        let AttrKind::Prop { value } = &attr.kind else {
            return [0.0; 4];
        };
        match value {
            Expr::Tuple(args, span) => {
                if args.iter().any(|a| a.name.is_some()) {
                    self.errors.push(CompileError::ConflictingSpacingField {
                        span: *span,
                        message: format!(
                            "`{name}` does not accept named corner fields; use a \
                             positional quad (top_left, top_right, bottom_right, \
                             bottom_left)"
                        ),
                    });
                    return [0.0; 4];
                }
                if args.len() != 4 {
                    self.errors.push(CompileError::ArityMismatch {
                        span: *span,
                        name: name.to_string(),
                        expected: 4,
                        found: args.len(),
                    });
                    return [0.0; 4];
                }
                let mut radii = [0.0_f32; 4];
                for (slot, arg) in radii.iter_mut().zip(args) {
                    let val = self.eval_pure(&arg.value);
                    if let Some(v) = spacing_value(&val) {
                        *slot = v;
                    } else {
                        self.errors.push(CompileError::AttributeTypeMismatch {
                            span: arg.value.span(),
                            expected: "a length (an integer)".to_string(),
                        });
                    }
                }
                radii
            }
            other => {
                let val = self.eval_pure(other);
                if let Some(v) = spacing_value(&val) {
                    [v; 4]
                } else {
                    self.errors.push(CompileError::AttributeTypeMismatch {
                        span: other.span(),
                        expected: "a length (an integer)".to_string(),
                    });
                    [0.0; 4]
                }
            }
        }
    }

    /// Processes a whole `View`: its declarations first (so bindings can resolve
    /// names), then lowers its top-level elements into a render tree, handling
    /// `when`/`for` structural members (M20).
    pub fn lower_view(&mut self, view: &ViewDecl, known_views: &[&str]) -> Vec<RenderNode> {
        self.eval_view_decls(view);
        self.lower_members(&view.body, known_views)
    }

    // ── lowering ────────────────────────────────────────────────────────

    /// Lowers `expr` to a reactive computation against the current environment.
    fn lower_expr(&mut self, expr: &Expr, payload_name: Option<&Symbol>) -> Lowered {
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

    fn lower_strlit(&mut self, parts: &[StrPart], payload_name: Option<&Symbol>) -> Lowered {
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
        &mut self,
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
            // Parameterized fn call (M25): inline the body with args bound as memos.
            if let Some(Value::Fn(id)) = self.env.lookup(name).cloned() {
                if (id.0 as usize) < self.fn_table.len() {
                    let (params, body) = self.fn_table[id.0 as usize].clone();
                    // Bind each arg as a reactive memo so signal reads inside the
                    // fn body are tracked by the enclosing scope.
                    let snapshot = self.env.len();
                    for (param, arg) in params.iter().zip(args.iter()) {
                        let arg_lowered = self.lower_expr(&arg.value, payload_name);
                        let scope = self.ctx.open_memo(arg_lowered);
                        self.env.push(param.clone(), Value::Memo(scope));
                    }
                    // Lower the body with arg bindings in scope.
                    let body_lowered = self.lower_expr(&body, payload_name);
                    // Restore env.
                    self.env.truncate(snapshot);
                    return body_lowered;
                }
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
        &mut self,
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
                    CoreKind::KeyDown => CompKind::KeyDown,
                    CoreKind::KeyUp => CompKind::KeyUp,
                    CoreKind::TextInput => CompKind::TextInput,
                    CoreKind::PointerEnter => CompKind::PointerEnter,
                    CoreKind::PointerExit => CompKind::PointerExit,
                    CoreKind::Hover => CompKind::Hover,
                    CoreKind::LongPress => CompKind::LongPress,
                    CoreKind::DoubleTap => CompKind::DoubleTap,
                    CoreKind::Secondary => CompKind::Secondary,
                };
                let value = ev.payload.as_ref().map(|p| match p {
                    InputPayload::Str(s) => Value::Str(s.clone()),
                    InputPayload::Bool(b) => Value::Bool(*b),
                    InputPayload::Float(f) => Value::Float(f64::from(*f)),
                    InputPayload::Key(k) => Value::Str(k.clone()),
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
/// Coerces a spacing side/scalar value to `f32`; only numeric values are valid
/// `Len`s (a non-numeric side is a `TypeMismatch`).
fn spacing_value(v: &Value) -> Option<f32> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    match v {
        Value::Int(n) => Some(*n as f32),
        Value::Float(f) => Some(*f as f32),
        _ => None,
    }
}

/// Assigns one resolved side of a named spacing tuple, recording a
/// [`CompileError::ConflictingSpacingField`] if the side was already set (either
/// directly or via an axis shorthand).
fn assign_side(
    slot: &mut Option<f32>,
    v: f32,
    side: &str,
    span: Span,
    errors: &mut Vec<CompileError>,
) {
    if slot.is_some() {
        errors.push(CompileError::ConflictingSpacingField {
            span,
            message: format!("spacing side `{side}` was set more than once"),
        });
    } else {
        *slot = Some(v);
    }
}

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

    // ── M30: user-view registry & call-site recognition ──────────────────

    #[test]
    fn user_view_call_is_recognized_without_behavior_change() {
        // `App` calls `Card` (a user view). M30 recognizes the call as a
        // user-view instantiation but reproduces the historical container
        // behavior: a `Box` named `Card`. No `UnknownView` diagnostic fires.
        let parsed = parse("View Card() { Text(\"hi\") }\nView App() { Card() }");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let mut interp = Interpreter::new();
        interp.load_views(&parsed.views);
        let known: Vec<&str> = parsed.views.iter().map(|v| v.name.as_str()).collect();
        let app = parsed
            .views
            .iter()
            .find(|v| v.name.as_str() == "App")
            .unwrap();
        let tree = interp.lower_view(app, &known);
        assert!(
            interp.errors().is_empty(),
            "no diagnostics expected: {:?}",
            interp.errors()
        );
        assert_eq!(tree.len(), 1);
        assert!(matches!(&tree[0], RenderNode::Box { name, .. } if name.as_str() == "Card"));
    }

    #[test]
    fn intrinsic_named_view_reports_shadowed_at_load() {
        let parsed = parse("View Row() { Text(\"x\") }");
        let mut interp = Interpreter::new();
        let diags = interp.load_views(&parsed.views);
        assert!(
            diags
                .iter()
                .any(|d| matches!(d, CompileError::IntrinsicShadowed { .. })),
            "expected IntrinsicShadowed, got {diags:?}"
        );
    }

    // ── M31: argument → parameter binding ────────────────────────────────

    /// Parses `callee_src` (a single view) and a call element from `call_src`'s
    /// first view body, returning `(callee, call_element)`.
    fn callee_and_call(callee_src: &str, call_src: &str) -> (ViewDecl, ElementNode) {
        let callee = parse(callee_src).views.into_iter().next().unwrap();
        let host = parse(call_src).views.into_iter().next().unwrap();
        let Member::Element(call) = host.body.into_iter().next().unwrap() else {
            panic!("expected element")
        };
        (callee, call)
    }

    #[test]
    fn named_positional_and_mixed_binding() {
        let (callee, _) = callee_and_call(
            "View Avatar(url, size) { Text(url) }",
            "View H() { Text(\"x\") }",
        );
        // Named.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call(
            "View Avatar(url, size) { Text(url) }",
            "View H() { Avatar(url: \"a.png\", size: 40) }",
        );
        let b = interp.bind_args(&callee, &call);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert_eq!(b.bindings.len(), 2);
        assert_eq!(b.bindings[0].0.as_str(), "url");
        assert_eq!(b.bindings[1].0.as_str(), "size");

        // Positional.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call(
            "View Avatar(url, size) { Text(url) }",
            "View H() { Avatar(\"a.png\", 40) }",
        );
        let b = interp.bind_args(&callee, &call);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert_eq!(b.bindings.len(), 2);

        // Mixed: positional then named.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call(
            "View Avatar(url, size) { Text(url) }",
            "View H() { Avatar(\"a.png\") #[size: 40] }",
        );
        let b = interp.bind_args(&callee, &call);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert_eq!(b.bindings.len(), 2);
    }

    #[test]
    fn arity_unknown_duplicate_and_missing_diagnostics() {
        let (callee, _) = callee_and_call("View A(x, y) { Text(x) }", "View H() { Text(\"_\") }");

        // Over-arity: 3 positional for 2 params.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View A(x, y) { Text(x) }", "View H() { A(1, 2, 3) }");
        interp.bind_args(&callee, &call);
        assert!(interp.errors().iter().any(|e| matches!(
            e,
            CompileError::ViewArityMismatch {
                expected: 2,
                found: 3,
                ..
            }
        )));

        // Unknown named param.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View A(x, y) { Text(x) }", "View H() { A(z: 1) }");
        interp.bind_args(&callee, &call);
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::UnknownParam { .. }))
        );

        // Duplicate: positional + named bind the same param.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View A(x, y) { Text(x) }", "View H() { A(1) #[x: 2] }");
        interp.bind_args(&callee, &call);
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::DuplicateParam { .. }))
        );

        // Missing required.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View A(x, y) { Text(x) }", "View H() { A(1) }");
        interp.bind_args(&callee, &call);
        assert!(
            interp
                .errors()
                .iter()
                .any(|e| matches!(e, CompileError::MissingParam { name, .. } if name == "y"))
        );
    }

    #[test]
    fn parent_var_arg_is_a_live_memo_literal_is_constant() {
        // The parent declares `var n = 1`; the call passes `n` to a parameter.
        // The projecting memo tracks the parent signal: writing `n` and ticking
        // changes the memo's value (dirty edge preserved).
        let callee = parse("View Foo(v) { Text(\"{v}\") }")
            .views
            .into_iter()
            .next()
            .unwrap();
        let mut interp = Interpreter::new();
        let init = Expr::IntLit(1, crate::diagnostics::Span::new(0, 1));
        let n = interp.define_var(Symbol::intern("n"), &init);

        let (_, call) = callee_and_call("View Foo(v) { Text(\"{v}\") }", "View H() { Foo(n) }");
        let b = interp.bind_args(&callee, &call);
        let memo = b.bindings[0].1;
        interp.tick();
        assert_eq!(interp.read_memo(memo), Value::Int(1));
        interp.write_var(n, Value::Int(7));
        interp.tick();
        assert_eq!(
            interp.read_memo(memo),
            Value::Int(7),
            "memo tracks the parent var"
        );

        // A literal argument is a constant memo: it never changes.
        let mut interp = Interpreter::new();
        let (_, call) = callee_and_call("View Foo(v) { Text(\"{v}\") }", "View H() { Foo(5) }");
        let b = interp.bind_args(&callee, &call);
        let memo = b.bindings[0].1;
        interp.tick();
        assert_eq!(interp.read_memo(memo), Value::Int(5));
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
            // 4-value positional: CSS order top, right, bottom, left
            (
                "View C() { Column #[p: (1, 2, 3, 4)] {} }",
                Spacing {
                    top: 1.0,
                    right: 2.0,
                    bottom: 3.0,
                    left: 4.0,
                },
            ),
            // Named top only — unspecified sides default to 0
            (
                "View C() { Column #[p: (top: 10)] {} }",
                Spacing {
                    top: 10.0,
                    right: 0.0,
                    bottom: 0.0,
                    left: 0.0,
                },
            ),
            // Named bottom only
            (
                "View C() { Column #[p: (bottom: 7)] {} }",
                Spacing {
                    top: 0.0,
                    right: 0.0,
                    bottom: 7.0,
                    left: 0.0,
                },
            ),
            // Named mixed sides
            (
                "View C() { Column #[p: (left: 5, bottom: 3)] {} }",
                Spacing {
                    top: 0.0,
                    right: 0.0,
                    bottom: 3.0,
                    left: 5.0,
                },
            ),
            // Verbose axis shorthands (the only accepted shorthands)
            (
                "View C() { Column #[p: (horizontal: 10, vertical: 5)] {} }",
                Spacing {
                    top: 5.0,
                    right: 10.0,
                    bottom: 5.0,
                    left: 10.0,
                },
            ),
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

    // ── M25: `Len` padding/margin forms ──────────────────────────────────

    /// Lowers a single-`Box` view and returns the resolved padding plus any
    /// errors raised during style resolution.
    fn resolve_padding(src: &str) -> (byard_core::atlas::layout::Spacing, Vec<CompileError>) {
        let parsed = parse(src);
        assert!(
            parsed.errors.is_empty(),
            "parse errors: {:?}",
            parsed.errors
        );
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        let RenderNode::Box { name, attrs, .. } = &tree[0] else {
            panic!("expected a Box");
        };
        let style = interp.eval_container_style(name.as_str(), attrs);
        (style.padding, interp.errors().to_vec())
    }

    #[test]
    fn impl30_scalar_sets_all_sides() {
        use byard_core::atlas::layout::Spacing;
        let (p, errs) = resolve_padding("View C() { Column #[p: 5] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(p, Spacing::all(5.0));
    }

    #[test]
    fn impl30_pair_is_vertical_horizontal() {
        use byard_core::atlas::layout::Spacing;
        let (p, errs) = resolve_padding("View C() { Column #[p: (10, 5)] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        // (vertical, horizontal): top=bottom=10, left=right=5.
        assert_eq!(
            p,
            Spacing {
                top: 10.0,
                right: 5.0,
                bottom: 10.0,
                left: 5.0
            }
        );
    }

    #[test]
    fn impl30_quad_is_css_order() {
        use byard_core::atlas::layout::Spacing;
        let (p, errs) = resolve_padding("View C() { Column #[p: (4, 6, 8, 7)] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(
            p,
            Spacing {
                top: 4.0,
                right: 6.0,
                bottom: 8.0,
                left: 7.0
            }
        );
    }

    #[test]
    fn impl30_named_single_side_defaults_rest_to_zero() {
        use byard_core::atlas::layout::Spacing;
        let (p, errs) = resolve_padding("View C() { Column #[p: (bottom: 7)] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(
            p,
            Spacing {
                top: 0.0,
                right: 0.0,
                bottom: 7.0,
                left: 0.0
            }
        );
    }

    #[test]
    fn impl30_named_axis_shorthands() {
        use byard_core::atlas::layout::Spacing;
        let (p, errs) =
            resolve_padding("View C() { Column #[p: (horizontal: 10, vertical: 5)] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(
            p,
            Spacing {
                top: 5.0,
                right: 10.0,
                bottom: 5.0,
                left: 10.0
            }
        );
    }

    #[test]
    fn impl30_unknown_side_is_unknown_attribute_with_hint() {
        let (_p, errs) = resolve_padding("View C() { Column #[p: (tpo: 4)] {} }");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                CompileError::UnknownAttribute { name, hint: Some(h), .. }
                    if name == "tpo" && h == "top"
            )),
            "expected UnknownAttribute(tpo)->top, got {errs:?}"
        );
    }

    #[test]
    fn impl30_axis_plus_component_conflicts() {
        let (_p, errs) = resolve_padding("View C() { Column #[p: (horizontal: 10, left: 3)] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::ConflictingSpacingField { .. })),
            "expected ConflictingSpacingField, got {errs:?}"
        );
    }

    #[test]
    fn impl30_non_int_side_is_type_mismatch() {
        let (_p, errs) = resolve_padding("View C() { Column #[p: (top: 4, left: \"x\")] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::AttributeTypeMismatch { .. })),
            "expected AttributeTypeMismatch, got {errs:?}"
        );
    }

    #[test]
    fn impl30_wrong_positional_arity_is_arity_mismatch() {
        let (_p, errs) = resolve_padding("View C() { Column #[p: (1, 2, 3)] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::ArityMismatch { .. })),
            "expected ArityMismatch for a 3-tuple, got {errs:?}"
        );
    }

    #[test]
    fn impl30_mixing_named_and_positional_errors() {
        let (_p, errs) = resolve_padding("View C() { Column #[p: (10, top: 4)] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::ConflictingSpacingField { .. })),
            "expected a conflict for mixed named/positional, got {errs:?}"
        );
    }

    #[test]
    fn impl30_px_py_are_now_unknown_attributes() {
        let parsed = parse("View C() { Column #[px: 5] {} }");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let _ = interp.lower_view(view, &[]);
        assert!(
            interp.errors().iter().any(|e| matches!(
                e,
                CompileError::UnknownAttribute { name, .. } if name == "px"
            )),
            "px must now be UnknownAttribute, got {:?}",
            interp.errors()
        );
    }

    // ── Per-corner `radius` ──────────────────────────────────────────────

    /// Lowers a single-element view and returns `resolve_radii`'s result for
    /// its `radius` attribute alongside any errors raised.
    fn resolve_radius_test(src: &str) -> ([f32; 4], Vec<CompileError>) {
        let parsed = parse(src);
        assert!(
            parsed.errors.is_empty(),
            "parse errors: {:?}",
            parsed.errors
        );
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let attrs = element(&view.body[0]).attrs.clone();
        let radii = interp.resolve_radii(&attrs, "radius");
        (radii, interp.errors().to_vec())
    }

    #[test]
    fn impl44_radius_scalar_broadcasts_to_all_four_corners() {
        let (radii, errs) = resolve_radius_test("View C() { Column #[radius: 16] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(radii, [16.0, 16.0, 16.0, 16.0]);
    }

    #[test]
    fn impl44_radius_quad_sets_independent_corners_in_css_order() {
        // top_left, top_right, bottom_right, bottom_left (frame.rs / WGSL convention).
        let (radii, errs) = resolve_radius_test("View C() { Column #[radius: (4, 8, 12, 16)] {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(radii, [4.0, 8.0, 12.0, 16.0]);
    }

    #[test]
    fn impl44_radius_missing_attribute_defaults_to_zero() {
        let (radii, errs) = resolve_radius_test("View C() { Column {} }");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(radii, [0.0; 4]);
    }

    #[test]
    fn impl44_radius_wrong_arity_is_arity_mismatch() {
        let (radii, errs) = resolve_radius_test("View C() { Column #[radius: (4, 8)] {} }");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                CompileError::ArityMismatch {
                    expected: 4,
                    found: 2,
                    ..
                }
            )),
            "expected ArityMismatch(4, found 2), got {errs:?}"
        );
        assert_eq!(radii, [0.0; 4]);
    }

    #[test]
    fn impl44_radius_named_corner_field_is_rejected() {
        let (radii, errs) = resolve_radius_test(
            "View C() { Column #[radius: (top_left: 4, top_right: 8, bottom_right: 12, bottom_left: 16)] {} }",
        );
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::ConflictingSpacingField { .. })),
            "expected ConflictingSpacingField for named corners, got {errs:?}"
        );
        assert_eq!(radii, [0.0; 4]);
    }

    #[test]
    fn impl44_radius_non_numeric_corner_is_type_mismatch() {
        let (radii, errs) =
            resolve_radius_test("View C() { Column #[radius: (4, \"x\", 12, 16)] {} }");
        assert!(
            errs.iter()
                .any(|e| matches!(e, CompileError::AttributeTypeMismatch { .. })),
            "expected AttributeTypeMismatch, got {errs:?}"
        );
        // Valid corners still resolve; the bad one is left at the [0.0;4]
        // default rather than aborting the whole tuple.
        assert_eq!(radii, [4.0, 0.0, 12.0, 16.0]);
    }

    #[test]
    fn impl44_decorated_box_carries_independent_corner_radii_into_box_instance() {
        // End-to-end: a quad `radius` on a Box that also has `bg` (so it's a
        // plain BoxInstance push, not a DecoratedBox) reaches the GPU instance
        // with all four corners intact rather than being collapsed to a scalar.
        let parsed = parse(
            "View C() { Box #[bg: 0xFF0000, radius: (4, 8, 12, 16), width: 50, height: 50] }",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        let instances = frame.instances();
        assert_eq!(instances.len(), 1, "expected exactly one BoxInstance");
        assert_eq!(instances[0].radii, [4.0, 8.0, 12.0, 16.0]);
    }

    // ── M16: Toggle/Slider/TextField write-back ──────────────────────────

    #[test]
    fn toggle_with_bg_has_no_background_slab() {
        // Regression: `bg` on a Toggle is the ON accent, not a full-rect fill
        // painted behind the control (that stray slab made widgets look "off").
        let parsed = parse(
            "View C() {\n var on = true\n Toggle #[bind: on, bg: 0x10B981, width: 52, height: 30]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // Exactly track + thumb — no extra background rectangle, no DecoratedBox.
        assert_eq!(
            frame.instances().len(),
            2,
            "toggle should emit only track + thumb"
        );
        assert_eq!(frame.decorated().len(), 0);
    }

    #[test]
    fn slider_with_bg_has_no_background_slab() {
        // Regression: `bg` on a Slider is the fill accent, not a full-rect fill.
        let parsed = parse(
            "View C() {\n var v = 0.5\n Slider #[bind: v, bg: 0xEF4444, min: 0.0, max: 1.0, width: 200, height: 24]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // track + fill + thumb(accent disc) + thumb(white inner) = 4; no slab.
        assert_eq!(
            frame.instances().len(),
            4,
            "slider should emit track + fill + thumb (2 discs), no slab"
        );
    }

    #[test]
    fn toggle_tap_flips_bound_var() {
        let parsed = parse("View C() {\n var on = false\n Toggle #[bind: on]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("on"))
            .unwrap();
        interp.tick();
        assert_eq!(interp.peek(sig), Value::Bool(false));

        // Simulate a render so handlers are registered.
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Tap inside the Toggle rect.
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 50,
            },
        ]);
        interp.tick();
        assert_eq!(
            interp.peek(sig),
            Value::Bool(true),
            "toggle flipped to true"
        );

        // Second tap flips back — gap > DOUBLE_TAP_MS (300ms) so it's a plain tap.
        interp.render(&tree, &mut frame, 400.0, 300.0);
        interp.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 400,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 450,
            },
        ]);
        interp.tick();
        assert_eq!(interp.peek(sig), Value::Bool(false), "toggle flipped back");
    }

    #[test]
    fn slider_drag_sets_float_value() {
        let parsed = parse(
            "View C() {\n var vol = 0.0\n Slider #[bind: vol, min: 0, max: 1, width: 100]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("vol"))
            .unwrap();
        interp.tick();

        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // PointerDown at ~50% of track (x=50 on a 100px track starting at x=0).
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::PointerDown,
            pos: (50.0, 5.0),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        interp.tick();

        let val = match interp.peek(sig) {
            Value::Float(f) => f,
            other => panic!("expected Float, got {other:?}"),
        };
        assert!(
            (val - 0.5).abs() < 0.1,
            "slider at 50% should be ~0.5, got {val}"
        );
    }

    #[test]
    fn slider_value_has_no_f32_widening_tail() {
        // Regression: a drag landing on 0.6 used to be stored as
        // `f64::from(0.6_f32)` = 0.6000000238418579 because the value math ran
        // in f32 and was only widened at the end. The value path now stays in
        // f64, so a pixel-aligned 60% drag round-trips to a clean "0.6".
        let parsed = parse(
            "View C() {\n var vol = 0.0\n Slider #[bind: vol, min: 0, max: 1, width: 100]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("vol"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // x = 60 on a 100px track starting at x = 0 → exactly 60%.
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::PointerDown,
            pos: (60.0, 5.0),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        interp.tick();

        let val = match interp.peek(sig) {
            Value::Float(f) => f,
            other => panic!("expected Float, got {other:?}"),
        };
        assert_eq!(
            format!("{val}"),
            "0.6",
            "slider value must not carry an f32 widening tail"
        );
    }

    #[test]
    fn slider_with_step_does_not_emit_more_decimals_than_the_step() {
        // step: 0.1 landing on 60% used to store 6 * 0.1 = 0.6000000000000001.
        // The value is now rounded to the step's precision → a clean "0.6".
        let parsed = parse(
            "View C() {\n var vol = 0.0\n Slider #[bind: vol, min: 0, max: 1, step: 0.1, width: 100]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("vol"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::PointerDown,
            pos: (60.0, 5.0),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        }]);
        interp.tick();

        let val = match interp.peek(sig) {
            Value::Float(f) => f,
            other => panic!("expected Float, got {other:?}"),
        };
        assert_eq!(format!("{val}"), "0.6");
    }

    #[test]
    fn text_field_change_event_round_trips() {
        let parsed = parse("View C() {\n var query = \"\"\n TextField #[bind: query]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("query"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Change event with new value.
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::Change,
            pos: (5.0, 5.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Str("hello".to_string())),
            time_ms: 0,
        }]);
        assert_eq!(interp.peek(sig), Value::Str("hello".to_string()));

        // Re-delivering the same value is deduped (E1).
        let before = interp.peek(sig);
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::Change,
            pos: (5.0, 5.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Str("hello".to_string())),
            time_ms: 1,
        }]);
        assert_eq!(interp.peek(sig), before, "equal value deduped");
    }

    #[test]
    fn bind_to_non_var_produces_no_bound_sig() {
        // `let y = 0` is not a var → resolve_bind_sig returns None.
        let parsed = parse("View C() {\n let y = 0\n Toggle #[bind: y]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        // No error expected at lowering; just bound_sig is None (non-var silently ignored).
        let RenderNode::Box { bound_sig, .. } = &tree[0] else {
            panic!("expected Box");
        };
        assert!(bound_sig.is_none(), "let binding yields no bound_sig");
    }

    // ── M17: Keyboard delivery ───────────────────────────────────────────

    #[test]
    fn text_field_receives_keyboard_text_input() {
        let parsed2 = parse("View C() {\n var text = \"\"\n TextField #[bind: text]\n}");
        assert!(parsed2.errors.is_empty(), "{:?}", parsed2.errors);
        let view2 = &parsed2.views[0];
        let mut interp2 = Interpreter::new();
        let tree2 = interp2.lower_view(view2, &[]);
        let sig = interp2
            .var_signal(&crate::symbol::Symbol::intern("text"))
            .unwrap();
        interp2.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp2.render(&tree2, &mut frame, 400.0, 300.0);

        // Focus the TextField by tapping it first.
        interp2.dispatch_events(&[
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 0,
            },
            byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerUp,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: 50,
            },
        ]);
        interp2.tick();
        interp2.render(&tree2, &mut frame, 400.0, 300.0);

        // Type "ab".
        interp2.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::TextInput,
            pos: (5.0, 5.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key("a".to_string())),
            time_ms: 100,
        }]);
        interp2.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::TextInput,
            pos: (5.0, 5.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key("b".to_string())),
            time_ms: 200,
        }]);
        interp2.tick();
        assert_eq!(
            interp2.peek(sig),
            Value::Str("ab".to_string()),
            "typed 'ab'"
        );

        // Backspace removes last char.
        interp2.render(&tree2, &mut frame, 400.0, 300.0);
        interp2.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::KeyDown,
            pos: (5.0, 5.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key(
                "Backspace".to_string(),
            )),
            time_ms: 300,
        }]);
        interp2.tick();
        assert_eq!(
            interp2.peek(sig),
            Value::Str("a".to_string()),
            "backspace deleted 'b'"
        );
    }

    // ── M18: Tab focus traversal ─────────────────────────────────────────

    #[test]
    fn tab_key_advances_focus_through_text_fields() {
        // Two TextFields — Tab should cycle between them.
        let parsed = parse(
            "View C() {\n var fa = false\n var fb = false\n TextField #[bind: fa, focused: fa]\n TextField #[bind: fb, focused: fb]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());

        let fa = interp
            .var_signal(&crate::symbol::Symbol::intern("fa"))
            .unwrap();
        let fb = interp
            .var_signal(&crate::symbol::Symbol::intern("fb"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Tab: should focus the first field (none focused yet → index 0).
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::KeyDown,
            pos: (0.0, 0.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key("Tab".to_string())),
            time_ms: 0,
        }]);
        interp.tick();
        assert_eq!(
            interp.peek(fa),
            Value::Bool(true),
            "first field focused after Tab"
        );
        assert_eq!(interp.peek(fb), Value::Bool(false));

        // Second Tab: advances to second field.
        interp.render(&tree, &mut frame, 400.0, 300.0);
        interp.dispatch_events(&[byard_core::platform::InputEvent {
            kind: byard_core::platform::EventKind::KeyDown,
            pos: (0.0, 0.0),
            delta: (0.0, 0.0),
            payload: Some(byard_core::platform::InputPayload::Key("Tab".to_string())),
            time_ms: 100,
        }]);
        interp.tick();
        assert_eq!(interp.peek(fa), Value::Bool(false), "first field blurred");
        assert_eq!(interp.peek(fb), Value::Bool(true), "second field focused");
    }

    // ── M20: Structural for/when in render tree ──────────────────────────

    #[test]
    fn when_true_includes_then_branch() {
        let parsed = parse("View C() {\n var show = true\n when show { Text(\"visible\") }\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        // `when true { ... }` → one Text node in the tree
        assert_eq!(tree.len(), 1, "when=true emits one node");
        assert!(matches!(tree[0], RenderNode::Text { .. }));
    }

    #[test]
    fn when_false_emits_nothing_without_else() {
        let parsed = parse("View C() {\n var hide = false\n when hide { Text(\"hidden\") }\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert!(tree.is_empty(), "when=false emits no nodes");
    }

    #[test]
    fn for_loop_emits_one_node_per_item() {
        let parsed =
            parse("View C() {\n var items = [1, 2, 3]\n for item in items { Text(\"{item}\") }\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert_eq!(tree.len(), 3, "for over 3 items emits 3 nodes");
    }

    // ── M23: Controller boundary ─────────────────────────────────────────

    #[test]
    fn inject_provider_is_visible_to_view() {
        let parsed = parse("View C() {\n inject AppEnv as env\n Text(\"{env}\")\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        // Provide the ambient value before lowering.
        interp.inject_provider("AppEnv", Value::Str("prod".to_string()));
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        // The Text should contain the injected value.
        assert_eq!(frame.texts()[0].text, "prod");
    }

    #[test]
    fn apply_io_results_writes_to_var_and_ticks() {
        let parsed = parse("View C() {\n var data = \"\"\n Text(\"{data}\")\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let _tree = interp.lower_view(view, &[]);
        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("data"))
            .unwrap();
        interp.tick();

        // Simulate an async I/O result writing to the `data` var.
        interp.apply_io_results([Box::new(move |interp: &mut Interpreter| {
            interp.write_var(sig, Value::Str("loaded".to_string()));
        }) as Box<dyn FnOnce(&mut Interpreter) + Send>]);
        interp.tick();
        assert_eq!(interp.peek(sig), Value::Str("loaded".to_string()));
    }

    // ── M25: Parameterized fn call sites ─────────────────────────────────

    #[test]
    fn parameterized_fn_call_binds_args() {
        // fn identity(n: Int) => n  →  let y = identity(42)  →  Text renders "42"
        let src = "View C() {\n fn identity(n: Int) => n\n var x = 42\n let y = identity(x)\n Text(\"{y}\")\n}";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(frame.texts()[0].text, "42", "identity(42) == 42");
    }

    #[test]
    fn parameterized_fn_reacts_to_signal_arg() {
        // fn greet(name: Str) => "Hi {name}"  →  reactive on `greeting` signal
        let src = "View C() {\n fn greet(name: Str) => \"Hi {name}\"\n var greeting = \"Alice\"\n let msg = greet(greeting)\n Text(\"{msg}\")\n}";
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        let sig = interp
            .var_signal(&crate::symbol::Symbol::intern("greeting"))
            .unwrap();
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(frame.texts()[0].text, "Hi Alice");

        // Change greeting → "Bob": msg should become "Hi Bob".
        interp.write_var(sig, Value::Str("Bob".into()));
        interp.tick();
        frame.clear();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(
            frame.texts()[0].text,
            "Hi Bob",
            "greet reacts to signal change"
        );
    }

    // ── M21: DecoratedBox / TextureSampler ───────────────────────────────

    #[test]
    fn image_lowers_to_texture_sampler_in_frame() {
        let parsed = parse(
            "View C() {\n Image(\"photo.jpg\") #[fit: \"cover\", width: 200, height: 150]\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        assert!(
            matches!(tree[0], RenderNode::Image { .. }),
            "Image element lowers to RenderNode::Image"
        );

        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert_eq!(frame.textures().len(), 1, "one TextureSampler in frame");
        let tex = &frame.textures()[0];
        assert_eq!(tex.src, "photo.jpg");
        assert_eq!(tex.fit, byard_core::frame::ImageFit::Cover);
    }

    #[test]
    fn image_fit_defaults_to_fill() {
        let parsed = parse("View C() {\n Image(\"img.png\")\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);
        assert_eq!(frame.textures()[0].fit, byard_core::frame::ImageFit::Fill);
    }

    #[test]
    fn box_with_border_becomes_decorated_box() {
        // `border` is the catalog Color attr; it yields a 2px ring.
        let parsed = parse("View C() {\n Box #[bg: 0xffffff, border: 0x000000]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // A bordered container splits into an opaque SolidBox fill
        // (so it stays behind its children, which also paint as solids) plus a
        // decorated *overlay* whose interior is transparent and only strokes the
        // 2px border — it can't occlude the children drawn beneath it.
        assert_eq!(
            frame.instances().len(),
            1,
            "opaque fill on the SolidBox pass"
        );
        assert_eq!(
            frame.instances()[0].color,
            [1.0, 1.0, 1.0, 1.0],
            "the fill carries the bg colour"
        );
        assert_eq!(frame.decorated().len(), 1, "border overlay → DecoratedBox");
        assert!((frame.decorated()[0].border_width - 2.0).abs() < f32::EPSILON);
        assert_eq!(
            frame.decorated()[0].base.color,
            [0.0; 4],
            "the overlay interior is transparent so children stay visible"
        );
    }

    #[test]
    fn bordered_container_paints_fill_before_its_child_widget() {
        // The regression behind the "widgets invisible" report: an opaque,
        // bordered card must NOT paint over the solid boxes of the widgets it
        // contains. The card's fill is a SolidBox pushed *before*
        // the child's, and the only decorated primitive is a transparent-interior
        // border overlay — so the child's fill is never occluded.
        let parsed = parse(
            "View C() {\n Column #[bg: 0x222233, border: 0x445566] {\n Box #[bg: 0xFF0000, width: 20, height: 20]\n }\n}",
        );
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        // Two solid fills: the card, then the child — in that paint order, so
        // the child (drawn second) lands on top of the card, not under it.
        assert_eq!(frame.instances().len(), 2, "card fill + child fill");
        assert_ne!(
            frame.instances()[0].color,
            [1.0, 0.0, 0.0, 1.0],
            "the first solid fill is the card, not the child"
        );
        assert_eq!(
            frame.instances()[1].color,
            [1.0, 0.0, 0.0, 1.0],
            "the child's red fill paints last (on top)"
        );
        // Every decorated primitive in this frame is a transparent-interior
        // overlay, so nothing opaque is layered above the child.
        assert!(
            frame.decorated().iter().all(|d| d.base.color[3] == 0.0),
            "all decorated overlays have transparent interiors"
        );
    }

    #[test]
    fn box_with_shadow_token_becomes_decorated_box() {
        let parsed = parse("View C() {\n Box #[bg: 0x222222, shadow: \"md\"]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        assert!(interp.errors().is_empty(), "{:?}", interp.errors());
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert_eq!(frame.decorated().len(), 1, "shadowed box → DecoratedBox");
        assert!(frame.decorated()[0].shadow_blur > 0.0);
        assert!(
            frame.decorated()[0].shadow_color[3] > 0.0,
            "shadow is translucent"
        );
    }

    // ── M22: Theme system ────────────────────────────────────────────────

    #[test]
    fn text_without_color_uses_theme_on_surface() {
        let parsed = parse("View C() {\n Text(\"hi\")\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        let expected_color =
            crate::interp::intrinsics::color_to_rgba(interp.theme.on_surface, false);
        assert_eq!(
            frame.texts()[0].color,
            expected_color,
            "no-color Text gets theme on_surface"
        );
    }

    #[test]
    fn typo_token_resolves_to_concrete_size() {
        let parsed = parse("View C() {\n Text(\"hi\") #[typo: \"titleLarge\"]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert!(
            (frame.texts()[0].font_size - 22.0).abs() < f32::EPSILON,
            "titleLarge → 22pt, got {}",
            frame.texts()[0].font_size
        );
    }

    #[test]
    fn inline_size_overrides_typo_token() {
        let parsed = parse("View C() {\n Text(\"hi\") #[typo: \"titleLarge\", size: 30]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert!(
            (frame.texts()[0].font_size - 30.0).abs() < f32::EPSILON,
            "inline size: 30 overrides typo token"
        );
    }

    #[test]
    fn plain_box_stays_as_box_instance() {
        let parsed = parse("View C() {\n Box #[bg: 0x111111]\n}");
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        let view = &parsed.views[0];
        let mut interp = Interpreter::new();
        let tree = interp.lower_view(view, &[]);
        interp.tick();
        let mut frame = byard_core::frame::RenderFrame::new();
        interp.render(&tree, &mut frame, 400.0, 300.0);

        assert_eq!(frame.instances().len(), 1, "plain box → BoxInstance");
        assert_eq!(frame.decorated().len(), 0);
    }
}
