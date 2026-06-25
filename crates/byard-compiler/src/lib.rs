//! # byard-compiler
//!
//! Compiler and developer-mode interpreter pipeline for the Byard UI framework.

#![allow(
    clippy::must_use_candidate,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::unused_self,
    clippy::elidable_lifetime_names,
    clippy::module_name_repetitions,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::ptr_as_ptr,
    clippy::wildcard_imports,
    clippy::uninlined_format_args
)]

pub mod diagnostics;
pub mod infer;
pub mod interp;
pub mod lexer;
pub mod parser;
pub mod symbol;
pub mod util;

pub use diagnostics::{CompileError, Span};
pub use lexer::Token;
pub use symbol::Symbol;
