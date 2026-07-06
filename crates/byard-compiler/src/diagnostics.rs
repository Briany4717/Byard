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
    /// A string literal nested interpolated strings deeper than 3 levels (D12).
    StringNestingTooDeep {
        /// Source range at the point the limit was exceeded.
        span: Span,
    },
    /// A string literal was opened but never closed before end of input.
    UnterminatedString {
        /// Source range of the offending string literal.
        span: Span,
    },
    /// A required type annotation is missing (D9: `View` params and `fn`
    /// signatures must be annotated).
    MissingAnnotation {
        /// Source range of the under-annotated item.
        span: Span,
        /// What needs an annotation (e.g. "view parameter", "function return").
        what: String,
    },
    /// A `var`/`let` initializer cannot be inferred and has no annotation
    /// (D9: the empty array `[]`, or a heterogeneous array).
    CannotInfer {
        /// Source range of the un-inferable initializer.
        span: Span,
    },
    /// `Text` was used where a type is expected; `Text` is the text *view*, the
    /// scalar string type is `Str` (D9, INV-7).
    TextUsedAsType {
        /// Source range of the offending annotation.
        span: Span,
    },
    /// An `inject T as name` found no ambient `T` in any ancestor scope.
    UnresolvedInject {
        /// Source range of the `inject` statement.
        span: Span,
        /// The injected type name that could not be resolved.
        name: String,
    },
    /// A mutation (`=`, `+=`, `++`, …) targeted something that is not an
    /// assignable `var` l-value (M9; RFC-0003 §6).
    NotAssignable {
        /// Source range of the offending target.
        span: Span,
    },
    /// An element name is neither a known intrinsic nor a `ViewDecl` in scope
    /// (RFC-0002 D4); `hint` carries a Levenshtein "did you mean …?".
    UnknownView {
        /// Source range of the element name.
        span: Span,
        /// The unknown name.
        name: String,
        /// The closest known name, if any.
        hint: Option<String>,
    },
    /// An intrinsic did not recognize an attribute name (RFC-0002 D4, never
    /// silently ignored).
    UnknownAttribute {
        /// Source range of the attribute.
        span: Span,
        /// The unknown attribute name.
        name: String,
        /// The closest recognized attribute, if any.
        hint: Option<String>,
    },
    /// An attribute used the wrong separator for its kind — a property given
    /// `=>`, or an event given `:` (RFC-0003 D4-bis).
    WrongAttributeSeparator {
        /// Source range of the attribute.
        span: Span,
        /// The attribute name.
        name: String,
        /// Whether the attribute *should* be a property (`:`) — else an event.
        expected_property: bool,
    },
    /// An intrinsic received the wrong number of positional `(...)` content
    /// arguments (RFC-0005 §5).
    ArityMismatch {
        /// Source range of the element.
        span: Span,
        /// The intrinsic name.
        name: String,
        /// How many content arguments were expected.
        expected: usize,
        /// How many were supplied.
        found: usize,
    },
    /// An attribute value (or content value) had the wrong scalar type for the
    /// intrinsic's contract (RFC-0005 §5).
    AttributeTypeMismatch {
        /// Source range of the value.
        span: Span,
        /// A short description of what was expected (e.g. "a length", "a color").
        expected: String,
    },
    /// Children were given to a childless intrinsic (`Text`, `Spacer`, `Image`,
    /// `TextField`, `Toggle`, `Slider`) — RFC-0005 §5 rule 8.
    UnexpectedChildren {
        /// Source range of the element.
        span: Span,
        /// The intrinsic name.
        name: String,
    },
    /// A `style { }` block read a `var`; Phase 2 styles are static (RFC-0002
    /// D5/D8).
    DynamicStyleForbidden {
        /// Source range of the offending value.
        span: Span,
    },
    /// A `p`/`m` spacing tuple named a side more than once, combined an axis
    /// shorthand (`horizontal`/`vertical`) with one of its component sides, or
    /// mixed named and positional fields (RFC-0005 §1 `Len` erratum).
    ConflictingSpacingField {
        /// Source range of the offending tuple or element.
        span: Span,
        /// Human-readable description of the conflict.
        message: String,
    },
    /// A user-`View` call supplied more positional content arguments than the
    /// callee declares parameters (RFC-0007 §6).
    ViewArityMismatch {
        /// Source range of the call site.
        span: Span,
        /// The callee view name.
        name: String,
        /// How many parameters the callee declares.
        expected: usize,
        /// How many positional arguments were supplied.
        found: usize,
    },
    /// A named argument at a user-`View` call site does not match any declared
    /// parameter (RFC-0007 §6); `hint` carries a Levenshtein "did you mean …?".
    UnknownParam {
        /// Source range of the argument.
        span: Span,
        /// The unknown parameter name.
        name: String,
        /// The callee view name.
        callee: String,
        /// The closest declared parameter, if any.
        hint: Option<String>,
    },
    /// A required callee parameter (no default) received no argument at the call
    /// site (RFC-0007 §6, interacts with D-B defaults).
    MissingParam {
        /// Source range of the call site.
        span: Span,
        /// The missing parameter name.
        name: String,
        /// The callee view name.
        callee: String,
    },
    /// The same callee parameter was bound twice — positional + named, or named
    /// twice (RFC-0007 §6).
    DuplicateParam {
        /// Source range of the offending argument.
        span: Span,
        /// The doubly-bound parameter name.
        name: String,
        /// The callee view name.
        callee: String,
    },
    /// A `ViewDecl` is named like an RFC-0005 intrinsic; the intrinsic always
    /// wins and the user view is unreachable (RFC-0007 §6).
    IntrinsicShadowed {
        /// Source range of the offending view declaration.
        span: Span,
        /// The shadowing view name (an intrinsic name).
        name: String,
    },
    /// A user-`View` call cycle is not guarded by a `when`/`for` structural
    /// boundary, so instantiation would diverge (RFC-0007 §4).
    RecursiveView {
        /// Source range of the offending view declaration.
        span: Span,
        /// The cycle path, e.g. `A → B → A`.
        path: String,
    },
    /// A `with` clause named a curve that is not `anim.linear`/`ease`/`spring`
    /// (RFC-0010); `hint` carries a Levenshtein "did you mean …?".
    UnknownAnimation {
        /// Source range of the offending `anim.*` call.
        span: Span,
        /// The unknown curve name.
        name: String,
        /// The closest known curve name, if any.
        hint: Option<String>,
    },
    /// A `with` clause attached an animation to a layout-affecting property
    /// (`width`/`height`/`p`/`m`/`gap`/…), which cannot animate on the GPU
    /// because it would require a per-frame relayout (RFC-0010 §"Layout
    /// properties", INV-8).
    LayoutPropNotAnimatable {
        /// Source range of the animated attribute.
        span: Span,
        /// The offending layout property name.
        prop: String,
    },
    /// A curve call had a malformed argument list (RFC-0010): a missing
    /// duration, a non-numeric value, or an unknown parameter name.
    InvalidAnimation {
        /// Source range of the offending call or argument.
        span: Span,
        /// Human-readable description of the problem.
        message: String,
    },
    /// A `..` spread's operand did not resolve to a `style { … }` value
    /// (RFC-0016) — e.g. `..x` where `x` is not a `let`-bound style.
    NotAStyle {
        /// Source range of the offending spread.
        span: Span,
    },
    /// An `on <state> { … }` block named a state that isn't one of the four
    /// engine-owned interaction states (RFC-0016) — `hover`/`pressed`/
    /// `focused`/`disabled`.
    UnknownStyleState {
        /// Source range of the offending state name.
        span: Span,
        /// The name as written.
        name: String,
        /// The closest known state, if one is near (D4-style suggestion).
        hint: Option<String>,
    },
    /// A `use` declaration appeared after the first `View` — imports are legal
    /// only at file top (RFC-0008 Pillar A).
    ImportAfterView {
        /// Source range of the offending `use` declaration.
        span: Span,
    },
    /// A `use` (or a qualified `alias.View` reference) named a package that is
    /// not in the resolved dependency set (RFC-0008 Pillar A).
    UnknownPackage {
        /// Source range of the reference.
        span: Span,
        /// The unresolved package or alias name.
        name: String,
        /// Why resolution failed (e.g. "not declared in `[dependencies]`").
        detail: String,
    },
    /// A selective import or qualified reference named a `View` the package
    /// does not export (RFC-0008 Pillar B); `hint` carries a Levenshtein
    /// "did you mean …?" over the package's exports.
    UnknownImportSymbol {
        /// Source range of the symbol.
        span: Span,
        /// The package searched.
        package: String,
        /// The missing view name.
        name: String,
        /// The closest exported name, if any.
        hint: Option<String>,
    },
    /// A name became ambiguous — two imports (or an import and a local view)
    /// bind the same bare name. Resolution is deterministic and
    /// order-independent, so ambiguity is an error demanding an explicit
    /// alias (RFC-0008 D-G).
    NameCollision {
        /// Source range of the later binding.
        span: Span,
        /// The colliding name.
        name: String,
        /// Where the two bindings come from (package or "this file").
        first: String,
        /// The second origin.
        second: String,
    },
    /// The package dependency graph contains a cycle (RFC-0008 Pillar A —
    /// module-graph cycle detection, one level above RFC-0007 §4's intra-file
    /// call cycles).
    PackageCycle {
        /// Source range of the `use` that closed the cycle.
        span: Span,
        /// The cycle path, e.g. `a → b → a`.
        path: String,
    },
    /// The same `View` name is declared twice within one package namespace
    /// (across its files) — exports must be unambiguous (RFC-0008 Pillar B).
    DuplicateViewName {
        /// Source range of the second declaration.
        span: Span,
        /// The duplicated view name.
        name: String,
        /// The package whose namespace is ambiguous ("this project" for the
        /// root package).
        package: String,
    },
    /// A project-level failure outside any single source file — a broken
    /// `byard.toml`, an unreadable dependency, a corrupt lockfile. Carried as
    /// a `CompileError` so the dev overlay and `check` report it through the
    /// same channel as source diagnostics (RFC-0008 Pillar C).
    Project {
        /// Anchor span (typically zero — there is no source to point at).
        span: Span,
        /// The failure, human-readable.
        message: String,
    },
    /// A `VectorIcon`'s SVG paints with a gradient, pattern, or filter — the
    /// MSDF pipeline supports flat monochrome fills only (RFC-0009 §2).
    SvgUnsupportedFeatures {
        /// Anchor span of the `VectorIcon` content argument.
        span: Span,
    },
    /// A `VectorIcon`'s SVG exceeds the path-segment complexity budget for MSDF
    /// generation (RFC-0009 §2; default 500 segments).
    SvgTooComplexForMssdf {
        /// Anchor span of the `VectorIcon` content argument.
        span: Span,
        /// The number of path segments actually found.
        found_nodes: usize,
    },
    /// A `VectorIcon`'s asset argument is not a compile-time-constant handle, so
    /// the AOT (`byard build`) packer cannot close the icon set statically
    /// (RFC-0009 §4). The fix is to make the handle a string literal, or to
    /// declare an explicit `[assets.vectors] include = [...]` list in
    /// `byard.toml`. A partial atlas is never shipped silently.
    VectorAssetNotStatic {
        /// Anchor span of the `VectorIcon` call site.
        span: Span,
    },
}

impl CompileError {
    /// The source span this error points at.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::UnexpectedChar { span }
            | Self::UnexpectedToken { span, .. }
            | Self::StringNestingTooDeep { span }
            | Self::UnterminatedString { span }
            | Self::MissingAnnotation { span, .. }
            | Self::CannotInfer { span }
            | Self::TextUsedAsType { span }
            | Self::UnresolvedInject { span, .. }
            | Self::NotAssignable { span }
            | Self::UnknownView { span, .. }
            | Self::UnknownAttribute { span, .. }
            | Self::WrongAttributeSeparator { span, .. }
            | Self::ArityMismatch { span, .. }
            | Self::AttributeTypeMismatch { span, .. }
            | Self::UnexpectedChildren { span, .. }
            | Self::DynamicStyleForbidden { span }
            | Self::ConflictingSpacingField { span, .. }
            | Self::ViewArityMismatch { span, .. }
            | Self::UnknownParam { span, .. }
            | Self::MissingParam { span, .. }
            | Self::DuplicateParam { span, .. }
            | Self::IntrinsicShadowed { span, .. }
            | Self::RecursiveView { span, .. }
            | Self::UnknownAnimation { span, .. }
            | Self::LayoutPropNotAnimatable { span, .. }
            | Self::InvalidAnimation { span, .. }
            | Self::NotAStyle { span }
            | Self::UnknownStyleState { span, .. }
            | Self::ImportAfterView { span }
            | Self::UnknownPackage { span, .. }
            | Self::UnknownImportSymbol { span, .. }
            | Self::NameCollision { span, .. }
            | Self::PackageCycle { span, .. }
            | Self::DuplicateViewName { span, .. }
            | Self::Project { span, .. }
            | Self::SvgUnsupportedFeatures { span }
            | Self::SvgTooComplexForMssdf { span, .. }
            | Self::VectorAssetNotStatic { span } => *span,
        }
    }

    /// Shifts this error's span by `delta` bytes (may be negative) — used by
    /// the module resolver to rebase per-file spans into the program-wide
    /// source map and back (RFC-0008).
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    pub fn shift_span(&mut self, delta: i64) {
        match self {
            Self::UnexpectedChar { span }
            | Self::UnexpectedToken { span, .. }
            | Self::StringNestingTooDeep { span }
            | Self::UnterminatedString { span }
            | Self::MissingAnnotation { span, .. }
            | Self::CannotInfer { span }
            | Self::TextUsedAsType { span }
            | Self::UnresolvedInject { span, .. }
            | Self::NotAssignable { span }
            | Self::UnknownView { span, .. }
            | Self::UnknownAttribute { span, .. }
            | Self::WrongAttributeSeparator { span, .. }
            | Self::ArityMismatch { span, .. }
            | Self::AttributeTypeMismatch { span, .. }
            | Self::UnexpectedChildren { span, .. }
            | Self::DynamicStyleForbidden { span }
            | Self::ConflictingSpacingField { span, .. }
            | Self::ViewArityMismatch { span, .. }
            | Self::UnknownParam { span, .. }
            | Self::MissingParam { span, .. }
            | Self::DuplicateParam { span, .. }
            | Self::IntrinsicShadowed { span, .. }
            | Self::RecursiveView { span, .. }
            | Self::UnknownAnimation { span, .. }
            | Self::LayoutPropNotAnimatable { span, .. }
            | Self::InvalidAnimation { span, .. }
            | Self::NotAStyle { span }
            | Self::UnknownStyleState { span, .. }
            | Self::ImportAfterView { span }
            | Self::UnknownPackage { span, .. }
            | Self::UnknownImportSymbol { span, .. }
            | Self::NameCollision { span, .. }
            | Self::PackageCycle { span, .. }
            | Self::DuplicateViewName { span, .. }
            | Self::Project { span, .. }
            | Self::SvgUnsupportedFeatures { span }
            | Self::SvgTooComplexForMssdf { span, .. }
            | Self::VectorAssetNotStatic { span } => {
                span.start = (i64::from(span.start) + delta).max(0) as u32;
                span.end = (i64::from(span.end) + delta).max(0) as u32;
            }
        }
    }

    /// A short, stable slug naming this error class, used for the rustc-style
    /// `error[kind]:` prefix in `byard check` output (RFC-0006 §5, C7).
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::UnexpectedChar { .. } => "UnexpectedChar",
            Self::UnexpectedToken { .. } => "UnexpectedToken",
            Self::StringNestingTooDeep { .. } => "StringNestingTooDeep",
            Self::UnterminatedString { .. } => "UnterminatedString",
            Self::MissingAnnotation { .. } => "MissingAnnotation",
            Self::CannotInfer { .. } => "CannotInfer",
            Self::TextUsedAsType { .. } => "TextUsedAsType",
            Self::UnresolvedInject { .. } => "UnresolvedInject",
            Self::NotAssignable { .. } => "NotAssignable",
            Self::UnknownView { .. } => "UnknownView",
            Self::UnknownAttribute { .. } => "UnknownAttribute",
            Self::WrongAttributeSeparator { .. } => "WrongAttributeSeparator",
            Self::ArityMismatch { .. } => "ArityMismatch",
            Self::AttributeTypeMismatch { .. } => "AttributeTypeMismatch",
            Self::UnexpectedChildren { .. } => "UnexpectedChildren",
            Self::DynamicStyleForbidden { .. } => "DynamicStyleForbidden",
            Self::ConflictingSpacingField { .. } => "ConflictingSpacingField",
            Self::ViewArityMismatch { .. } => "ViewArityMismatch",
            Self::UnknownParam { .. } => "UnknownParam",
            Self::MissingParam { .. } => "MissingParam",
            Self::DuplicateParam { .. } => "DuplicateParam",
            Self::IntrinsicShadowed { .. } => "IntrinsicShadowed",
            Self::RecursiveView { .. } => "RecursiveView",
            Self::UnknownAnimation { .. } => "UnknownAnimation",
            Self::LayoutPropNotAnimatable { .. } => "LayoutPropNotAnimatable",
            Self::InvalidAnimation { .. } => "InvalidAnimation",
            Self::NotAStyle { .. } => "NotAStyle",
            Self::UnknownStyleState { .. } => "UnknownStyleState",
            Self::ImportAfterView { .. } => "ImportAfterView",
            Self::UnknownPackage { .. } => "UnknownPackage",
            Self::UnknownImportSymbol { .. } => "UnknownImportSymbol",
            Self::NameCollision { .. } => "NameCollision",
            Self::PackageCycle { .. } => "PackageCycle",
            Self::DuplicateViewName { .. } => "DuplicateViewName",
            Self::Project { .. } => "Project",
            Self::SvgUnsupportedFeatures { .. } => "SvgUnsupportedFeatures",
            Self::SvgTooComplexForMssdf { .. } => "SvgTooComplexForMssdf",
            Self::VectorAssetNotStatic { .. } => "VectorAssetNotStatic",
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
            Self::StringNestingTooDeep { .. } => {
                "string interpolation nested deeper than 3 levels".to_string()
            }
            Self::UnterminatedString { .. } => "unterminated string literal".to_string(),
            Self::MissingAnnotation { what, .. } => {
                format!("missing type annotation on {what}")
            }
            Self::CannotInfer { .. } => {
                "cannot infer a type; add an explicit annotation".to_string()
            }
            Self::TextUsedAsType { .. } => {
                "`Text` is a view, not a type; use `Str` for strings".to_string()
            }
            Self::UnresolvedInject { name, .. } => {
                format!("no ambient `{name}` is provided by any ancestor view")
            }
            Self::NotAssignable { .. } => "this is not an assignable `var`".to_string(),
            Self::UnknownView { name, hint, .. } => {
                with_hint(format!("unknown view `{name}`"), hint.as_deref())
            }
            Self::UnknownAttribute { name, hint, .. } => {
                with_hint(format!("unknown attribute `{name}`"), hint.as_deref())
            }
            Self::WrongAttributeSeparator {
                name,
                expected_property,
                ..
            } => {
                if *expected_property {
                    format!("`{name}` is a property; use `:` not `=>`")
                } else {
                    format!("`{name}` is an event; use `=>` not `:`")
                }
            }
            Self::ArityMismatch {
                name,
                expected,
                found,
                ..
            } => format!("`{name}` takes {expected} content argument(s), found {found}"),
            Self::AttributeTypeMismatch { expected, .. } => {
                format!("expected {expected}")
            }
            Self::UnexpectedChildren { name, .. } => {
                format!("`{name}` takes no children")
            }
            Self::DynamicStyleForbidden { .. } => {
                "a `style` block cannot read a `var` (styles are static in Phase 2)".to_string()
            }
            Self::ConflictingSpacingField { message, .. }
            | Self::InvalidAnimation { message, .. }
            | Self::Project { message, .. } => message.clone(),
            Self::ViewArityMismatch {
                name,
                expected,
                found,
                ..
            } => format!(
                "view `{name}` declares {expected} parameter(s), found {found} positional argument(s)"
            ),
            Self::UnknownParam {
                name, callee, hint, ..
            } => with_hint(
                format!("`{callee}` has no parameter `{name}`"),
                hint.as_deref(),
            ),
            Self::MissingParam { name, callee, .. } => {
                format!("missing required argument `{name}` for view `{callee}`")
            }
            Self::DuplicateParam { name, callee, .. } => {
                format!("parameter `{name}` of view `{callee}` is bound more than once")
            }
            Self::IntrinsicShadowed { name, .. } => {
                format!("view `{name}` shadows the built-in intrinsic of the same name")
            }
            Self::RecursiveView { path, .. } => {
                format!("recursive view cycle without a `when`/`for` guard: {path}")
            }
            Self::UnknownAnimation { name, hint, .. } => {
                with_hint(format!("unknown animation curve `{name}`"), hint.as_deref())
            }
            Self::LayoutPropNotAnimatable { prop, .. } => format!(
                "`{prop}` is a layout property and cannot be animated with `with` \
                 (it would relayout every frame); animate a `scale` transform instead"
            ),
            Self::NotAStyle { .. } => "`..` can only spread a `style { … }` value".to_string(),
            Self::UnknownStyleState { name, hint, .. } => with_hint(
                format!(
                    "unknown interaction state `{name}` (expected hover/pressed/focused/disabled)"
                ),
                hint.as_deref(),
            ),
            Self::ImportAfterView { .. } => {
                "`use` imports must appear at the top of the file, before any `View`".to_string()
            }
            Self::UnknownPackage { name, detail, .. } => {
                format!("unknown package `{name}`: {detail}")
            }
            Self::UnknownImportSymbol {
                package,
                name,
                hint,
                ..
            } => with_hint(
                format!("package `{package}` does not export a view `{name}`"),
                hint.as_deref(),
            ),
            Self::NameCollision {
                name,
                first,
                second,
                ..
            } => format!(
                "`{name}` is ambiguous — bound by both {first} and {second}; \
                 disambiguate with an explicit alias (`use <pkg> as <alias>`)"
            ),
            Self::PackageCycle { path, .. } => {
                format!("package dependency cycle: {path}")
            }
            Self::DuplicateViewName { name, package, .. } => {
                format!("view `{name}` is declared more than once in {package}")
            }
            Self::SvgUnsupportedFeatures { .. } => {
                "SVG uses a gradient, pattern, or filter; the MSDF vector pipeline only \
                 supports flat monochrome fills — use `Image` instead"
                    .to_string()
            }
            Self::SvgTooComplexForMssdf { found_nodes, .. } => format!(
                "SVG has {found_nodes} path segments, exceeding the MSDF complexity budget \
                 (500) — simplify the artwork or use `Image` instead"
            ),
            Self::VectorAssetNotStatic { .. } => {
                "VectorIcon asset is not a compile-time-constant path; `byard build` cannot \
                 statically close the icon set — use a string literal, or list the assets in \
                 `byard.toml` under `[assets.vectors] include`"
                    .to_string()
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

/// Appends a "did you mean …?" suffix to a headline if a hint is present.
fn with_hint(message: String, hint: Option<&str>) -> String {
    match hint {
        Some(h) => format!("{message} (did you mean `{h}`?)"),
        None => message,
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
