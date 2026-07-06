//! The MSDF vector/icon generator (RFC-0009 §2, §5).
//!
//! Compiler-side only — `byard-core` never depends on this module (INV-1). It
//! only ever produces a finished [`generate::MsdfGlyph`] that crosses
//! `frame.rs` as data; the actual multi-channel signed-distance-field math is
//! delegated to the vendored `bymsdfgen-core` generator.

pub mod aot;
pub mod cache;
pub mod generate;
pub mod jit;
pub mod pack;
pub mod validate;

pub use aot::{BakedGlyph, BakedVectorAtlas, bake_atlas, collect_static_vector_refs};
pub use cache::{cache_key, generate_cached};
pub use generate::{
    EDGE_ANGLE_DEGREES, GENERATOR_VERSION, GRID_SIZE, MsdfGlyph, PX_RANGE, generate,
};
pub use jit::{ResidentGlyph, VectorJit};
pub use validate::{MAX_NODES, validate_vector_complexity};
