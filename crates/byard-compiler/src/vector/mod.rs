//! The MSDF vector/icon generator (RFC-0009 §2, §5).
//!
//! Compiler-side only — `byard-core` never depends on this module (INV-1). It
//! only ever produces a finished [`generate::MsdfGlyph`] that crosses
//! `frame.rs` as data; the actual multi-channel signed-distance-field math is
//! delegated to the vendored `bymsdfgen-core` generator.

pub mod generate;
pub mod validate;

pub use generate::{EDGE_ANGLE_DEGREES, GRID_SIZE, MsdfGlyph, PX_RANGE, generate};
pub use validate::{MAX_NODES, validate_vector_complexity};
