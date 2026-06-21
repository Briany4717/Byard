//! Parser: `Token` stream → `Vec<ViewDecl>` (RFC-0002 §"Parser" + §"Grammar").
//!
//! The recursive-descent driver and the Pratt expression parser land in M4; for
//! now this module exposes the typed [`ast`] definitions everything downstream
//! builds on.

pub mod ast;
