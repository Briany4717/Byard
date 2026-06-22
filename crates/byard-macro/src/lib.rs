//! Procedural macros for byard (M23 — controller boundary).
//!
//! `#[byard_controller]` marks a Rust struct as a byard controller: it can
//! provide ambient values via `inject`, expose async methods, and deliver
//! results back to the logic thread through the relay's I/O channel.
//!
//! Phase 2 implementation: the macro is an identity transform (passes the item
//! through unchanged). Future phases will generate the metadata the interpreter
//! needs for type-directed member-access (`M5`) and hook up the async I/O
//! plumbing automatically.

use proc_macro::TokenStream;

/// Marks a Rust struct as a byard controller.
///
/// Phase 2: identity transform — the struct is emitted unchanged.
/// Future phases will derive `ControllerMeta` and wire up async I/O.
///
/// # Example
///
/// ```ignore
/// #[byard_controller]
/// struct CounterController {
///     count: i64,
/// }
/// ```
#[proc_macro_attribute]
pub fn byard_controller(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
