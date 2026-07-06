//! Persistent content-addressed field cache (RFC-0009 §5, M52).
//!
//! Generating an MSDF field parses the SVG and runs the distance-field math;
//! doing it on every cold `byard dev` start (or `byard build`) re-pays a cost
//! whose result is deterministic (the M45 guarantee). This cache keys a
//! generated field by `hash(svg-bytes ‖ grid ‖ px_range ‖ generator-version)`
//! and stores its bytes under `.byard/cache/vectors/<key>.msdf`, so a second run
//! over unchanged input loads from disk and skips generation entirely.
//!
//! The generator version is part of the key, so a toolchain/algorithm bump
//! invalidates every entry with no explicit purge (and `byard clean` wipes the
//! directory). A corrupt or truncated file is treated as a miss and safely
//! regenerated — never a panic (INV-3: the loaded payload is fully owned).

use std::path::{Path, PathBuf};

use crate::diagnostics::{CompileError, Span};

use super::generate::{GENERATOR_VERSION, MsdfGlyph, generate};

/// FNV-1a 64-bit over `bytes` — a small, dependency-free, fully deterministic
/// hash (identical on every platform and run, unlike `DefaultHasher`), which is
/// exactly what a cross-run disk key needs.
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// The content-address (hex) of a field generated from `svg_bytes` at `grid` /
/// `px_range` with the current [`GENERATOR_VERSION`]. Any change to the SVG,
/// either parameter, or the generator version yields a different key.
#[must_use]
pub fn cache_key(svg_bytes: &[u8], grid: u32, px_range: f32) -> String {
    let mut buf = Vec::with_capacity(svg_bytes.len() + 16);
    buf.extend_from_slice(svg_bytes);
    buf.extend_from_slice(&grid.to_le_bytes());
    buf.extend_from_slice(&px_range.to_bits().to_le_bytes());
    buf.extend_from_slice(&GENERATOR_VERSION.to_le_bytes());
    format!("{:016x}", fnv1a(&buf))
}

/// On-disk record: a tiny header (`width`, `height`, `px_range`) then the RGBA
/// field bytes. The header lets a load validate the payload length and reject a
/// truncated/corrupt file as a miss.
fn encode(glyph: &MsdfGlyph) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + glyph.bitmap.len());
    out.extend_from_slice(&glyph.width.to_le_bytes());
    out.extend_from_slice(&glyph.height.to_le_bytes());
    out.extend_from_slice(&glyph.px_range.to_bits().to_le_bytes());
    out.extend_from_slice(&glyph.bitmap);
    out
}

/// Parses a cache record, returning `None` for anything malformed (short header,
/// wrong payload length) so the caller regenerates instead of trusting garbage.
fn decode(bytes: &[u8]) -> Option<MsdfGlyph> {
    if bytes.len() < 12 {
        return None;
    }
    let width = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
    let height = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    let px_range = f32::from_bits(u32::from_le_bytes(bytes[8..12].try_into().ok()?));
    let expected = (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)?;
    let bitmap = &bytes[12..];
    if bitmap.len() != expected {
        return None;
    }
    Some(MsdfGlyph {
        bitmap: bitmap.to_vec(),
        width,
        height,
        px_range,
    })
}

/// The file a `key` maps to within `cache_dir`.
fn entry_path(cache_dir: &Path, key: &str) -> PathBuf {
    cache_dir.join(format!("{key}.msdf"))
}

/// Loads a cached field for `key`, or `None` on a miss or any read/parse error.
#[must_use]
pub fn load(cache_dir: &Path, key: &str) -> Option<MsdfGlyph> {
    let bytes = std::fs::read(entry_path(cache_dir, key)).ok()?;
    decode(&bytes)
}

/// Writes `glyph` under `key`. A cache write is best-effort: a failure (e.g. a
/// read-only `.byard/`) must never fail the build/generation, only forgo the
/// speedup, so the error is intentionally swallowed.
pub fn store(cache_dir: &Path, key: &str, glyph: &MsdfGlyph) {
    if std::fs::create_dir_all(cache_dir).is_err() {
        return;
    }
    let _ = std::fs::write(entry_path(cache_dir, key), encode(glyph));
}

/// Generates a field, consulting `cache_dir` first when one is provided: a hit
/// returns the stored bytes without generating; a miss generates and writes
/// through. With `cache_dir == None` this is exactly [`generate`] (the path unit
/// tests and cache-less callers take).
///
/// # Errors
///
/// Propagates any [`CompileError`] from [`generate`] on a miss. A cache I/O
/// failure is never an error — it just forgoes caching.
pub fn generate_cached(
    svg_bytes: &[u8],
    grid: u32,
    px_range: f32,
    span: Span,
    cache_dir: Option<&Path>,
) -> Result<MsdfGlyph, CompileError> {
    let Some(dir) = cache_dir else {
        return generate(svg_bytes, grid, px_range, span);
    };
    let key = cache_key(svg_bytes, grid, px_range);
    if let Some(glyph) = load(dir, &key) {
        return Ok(glyph);
    }
    let glyph = generate(svg_bytes, grid, px_range, span)?;
    store(dir, &key, &glyph);
    Ok(glyph)
}

#[cfg(test)]
mod tests {
    use super::super::generate::{GRID_SIZE, PX_RANGE};
    use super::*;

    const SQUARE: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <path d="M4 4 L20 4 L20 20 L4 20 Z" fill="#000000"/></svg>"##;
    const RING: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <path d="M4 4 L20 4 L20 20 L4 20 Z M8 8 L16 8 L16 16 L8 16 Z" fill="#000000" fill-rule="evenodd"/></svg>"##;

    fn tmp(tag: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("byard_vcache_{tag}_{n}"))
    }

    #[test]
    fn key_changes_with_content_params_and_version() {
        let a = cache_key(SQUARE, GRID_SIZE, PX_RANGE);
        assert_eq!(
            a,
            cache_key(SQUARE, GRID_SIZE, PX_RANGE),
            "stable for same input"
        );
        assert_ne!(a, cache_key(RING, GRID_SIZE, PX_RANGE), "content matters");
        assert_ne!(a, cache_key(SQUARE, 64, PX_RANGE), "grid matters");
        assert_ne!(a, cache_key(SQUARE, GRID_SIZE, 2.0), "px_range matters");
    }

    #[test]
    fn a_second_generate_is_a_hit_that_matches_the_first() {
        let dir = tmp("hit");
        let first =
            generate_cached(SQUARE, GRID_SIZE, PX_RANGE, Span::new(0, 0), Some(&dir)).unwrap();
        // The entry now exists on disk.
        let key = cache_key(SQUARE, GRID_SIZE, PX_RANGE);
        assert!(
            entry_path(&dir, &key).exists(),
            "the miss must write through"
        );
        // A load returns a byte-identical field (the determinism the cache relies on).
        let hit = load(&dir, &key).expect("must hit");
        assert_eq!(hit, first);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn editing_the_svg_produces_a_new_key_and_regenerates() {
        let dir = tmp("edit");
        let _ = generate_cached(SQUARE, GRID_SIZE, PX_RANGE, Span::new(0, 0), Some(&dir)).unwrap();
        let _ = generate_cached(RING, GRID_SIZE, PX_RANGE, Span::new(0, 0), Some(&dir)).unwrap();
        // Two distinct keys → two files.
        let n = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(n, 2, "a changed SVG must not overwrite the old entry");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_cache_file_is_a_miss_not_a_panic() {
        let dir = tmp("corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        let key = cache_key(SQUARE, GRID_SIZE, PX_RANGE);
        std::fs::write(entry_path(&dir, &key), b"not a valid record").unwrap();
        assert!(load(&dir, &key).is_none(), "garbage must decode to None");
        // generate_cached still succeeds (regenerates, overwrites the bad file).
        let glyph =
            generate_cached(SQUARE, GRID_SIZE, PX_RANGE, Span::new(0, 0), Some(&dir)).unwrap();
        assert_eq!(glyph.width, GRID_SIZE);
        assert!(
            load(&dir, &key).is_some(),
            "the bad file must be overwritten"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_cache_dir_is_a_plain_generate() {
        let glyph = generate_cached(SQUARE, GRID_SIZE, PX_RANGE, Span::new(0, 0), None).unwrap();
        assert_eq!(glyph.width, GRID_SIZE);
    }
}
