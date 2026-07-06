//! AOT vector-atlas baking (RFC-0009 §4, M49).
//!
//! `byard build` closes the set of icons an app actually instantiates, bakes
//! only those into one **tightest-fit** immutable atlas, and emits a fixed
//! coordinate table — so a shipped binary uploads one small texture and indexes
//! a `[BakedGlyph; N]` with no runtime SVG parsing or coordinate math. The atlas
//! is sized to the packing (theoretical-minimum VRAM), not to a fixed sheet.
//! Dev/prod render parity (INV-7) holds because the baked **field bytes** are
//! identical to dev's — the UV only addresses them, and a self-describing atlas
//! `size` travels with the table, so a smaller sheet changes addressing, never
//! the sampled texels.
//!
//! Three stages, each independently testable:
//!
//! 1. [`collect_static_vector_refs`] — a static traversal of the resolved view
//!    graph collecting every `VectorIcon("literal")`, with the RFC-0009 §4
//!    dynamic-reference guard ([`CompileError::VectorAssetNotStatic`]).
//! 2. generation (reusing [`super::generate`]) + dedup of identical handles.
//! 3. [`super::pack`] MaxRects packing into layered cells.
//!
//! [`bake_atlas`] runs 2–3 over the closed set from stage 1.

use std::collections::BTreeSet;
use std::path::Path;

use byard_core::encoder::vector_msdf::ATLAS_SIZE;

use crate::diagnostics::{CompileError, Span};
use crate::parser::ast::{ElementNode, Expr, Member, StrPart, ViewDecl};

use super::generate::{GRID_SIZE, PX_RANGE};
use super::pack::{Placement, Size, pack_layers};

/// A statically resolved `VectorIcon` reference: the literal handle string as
/// written in source, plus the call-site span for diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticVectorRef {
    /// The asset handle (a path string) exactly as written.
    pub handle: String,
    /// The `VectorIcon` call site.
    pub span: Span,
}

/// One baked glyph's coordinate-table entry — the AOT counterpart of a dev
/// [`super::ResidentGlyph`]. Serialized into the generated `[BakedGlyph; N]`.
#[derive(Clone, Debug, PartialEq)]
pub struct BakedGlyph {
    /// The source handle this entry resolves (the runtime lookup key).
    pub handle: String,
    /// Normalized UV rect `(u0, v0, u1, v1)` within this atlas ([`BakedVectorAtlas::size`]).
    /// Addresses the same field bytes the dev path samples, so the render matches
    /// (INV-7) even though the corners differ from dev's larger sheet.
    pub uv_rect: [f32; 4],
    /// Array-texture layer.
    pub layer: u32,
    /// Baked distance range (RFC-0009 §2-E).
    pub px_range: f32,
}

/// The result of an AOT bake: the packed atlas bytes plus the coordinate table.
#[derive(Clone, Debug, PartialEq)]
pub struct BakedVectorAtlas {
    /// Atlas edge length in texels (one layer is `size × size`).
    pub size: u32,
    /// Number of array layers the packing used.
    pub layers: u32,
    /// RGBA8 texels, `size * size * layers * 4` bytes, layer-major then
    /// row-major within each layer (top row first).
    pub atlas: Vec<u8>,
    /// One entry per unique baked handle, in a stable (sorted) order.
    pub table: Vec<BakedGlyph>,
}

/// Traverses `views` and collects every statically resolvable `VectorIcon`
/// reference (RFC-0009 §4 tree-shake). A non-literal asset argument (derived
/// from a `var`, a `match`, or string interpolation) cannot be closed at build
/// time: unless the developer declared an explicit inclusion list (`include`
/// non-empty — the `byard.toml` `[assets.vectors] include` escape hatch), each
/// such site is a [`CompileError::VectorAssetNotStatic`], never a silently
/// partial atlas.
///
/// Duplicate handles are preserved here (the same icon used in two places); the
/// bake stage dedups them.
#[must_use]
pub fn collect_static_vector_refs(
    views: &[ViewDecl],
    include: &[String],
) -> (Vec<StaticVectorRef>, Vec<CompileError>) {
    let mut refs: Vec<StaticVectorRef> = include
        .iter()
        .map(|h| StaticVectorRef {
            handle: h.clone(),
            span: Span::new(0, 0),
        })
        .collect();
    let mut errors = Vec::new();
    // A declared inclusion list is the author taking responsibility for the
    // closed set, so a dynamic site is then permitted (its assets are expected
    // to be in `include`); with no list, a dynamic site is a hard error.
    let allow_dynamic = !include.is_empty();

    for view in views {
        walk_members(&view.body, &mut |el| {
            if el.name.as_str() != "VectorIcon" {
                return;
            }
            let Some(arg) = el.content.first() else {
                return; // arity is validated elsewhere; nothing to collect.
            };
            match static_handle(&arg.value) {
                Some(handle) if !handle.is_empty() => refs.push(StaticVectorRef {
                    handle,
                    span: el.span,
                }),
                Some(_) => {} // empty literal — the INV-9 placeholder, not an asset.
                None if allow_dynamic => {}
                None => errors.push(CompileError::VectorAssetNotStatic { span: el.span }),
            }
        });
    }

    (refs, errors)
}

/// Generates, dedups, and MaxRects-packs `refs` into a [`BakedVectorAtlas`].
/// Handles are resolved relative to `base` (the project directory) for reading,
/// exactly as the dev runner resolves them relative to the working directory.
/// Every generation/read failure is collected — the whole set is reported at
/// once rather than aborting on the first bad icon.
///
/// # Errors
///
/// Returns every [`CompileError`] encountered (unreadable file, invalid SVG,
/// unsupported features, over-complex path), or a packing failure folded into a
/// [`CompileError::Project`].
pub fn bake_atlas(
    refs: &[StaticVectorRef],
    base: &Path,
    cache_dir: Option<&Path>,
) -> Result<BakedVectorAtlas, Vec<CompileError>> {
    // Dedup identical handles (RFC-0009 §4) while keeping a stable order.
    let mut seen = BTreeSet::new();
    let mut unique: Vec<&StaticVectorRef> = Vec::new();
    for r in refs {
        if seen.insert(r.handle.as_str()) {
            unique.push(r);
        }
    }
    unique.sort_by(|a, b| a.handle.cmp(&b.handle));

    // Generate each field.
    let mut errors = Vec::new();
    let mut glyphs = Vec::new();
    for r in &unique {
        let path = base.join(&r.handle);
        match std::fs::read(&path) {
            // RFC-0009 §5 (M52): a build reuses the same on-disk field cache as
            // the dev JIT, so an unchanged icon is not re-generated on rebuild.
            Ok(bytes) => {
                match super::cache::generate_cached(&bytes, GRID_SIZE, PX_RANGE, r.span, cache_dir)
                {
                    Ok(glyph) => glyphs.push((r.handle.clone(), glyph)),
                    Err(e) => errors.push(e),
                }
            }
            Err(e) => errors.push(CompileError::Project {
                span: r.span,
                message: format!("cannot read vector asset {}: {e}", r.handle),
            }),
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Pack into the *tightest* square atlas (the north star: theoretical-minimum
    // VRAM). Unlike the dev atlas, the AOT set is fully known, so we size the
    // texture to the packing instead of always allocating `ATLAS_SIZE`² — a
    // 5-icon app ships a 64² atlas (~16 KB), not a 2048² one (~16 MB). The baked
    // atlas is self-describing (`size` travels with the table), so a smaller
    // atlas changes the UV *addressing*, never the sampled texels: dev/prod
    // render parity (INV-7) is preserved by the identical field bytes, not by a
    // byte-identical UV.
    let sizes: Vec<Size> = glyphs
        .iter()
        .map(|(_, g)| Size {
            w: g.width,
            h: g.height,
        })
        .collect();
    let Some((atlas_size, placements, layers)) = pack_minimal(&sizes) else {
        return Err(vec![CompileError::Project {
            span: Span::new(0, 0),
            message: format!(
                "vector atlas packing failed: an icon exceeds the {ATLAS_SIZE}px atlas cap"
            ),
        }]);
    };

    // Blit each field into its cell and record its table entry.
    let plane = (atlas_size as usize) * (atlas_size as usize) * 4;
    let mut atlas = vec![0u8; plane * layers as usize];
    let mut table = Vec::with_capacity(glyphs.len());
    for p in &placements {
        let (handle, glyph) = &glyphs[p.index];
        blit(&mut atlas, atlas_size, glyph, p.x, p.y, p.layer);
        table.push(BakedGlyph {
            handle: handle.clone(),
            uv_rect: cell_uv(atlas_size, p.x, p.y, glyph.width, glyph.height),
            layer: p.layer,
            px_range: glyph.px_range,
        });
    }
    table.sort_by(|a, b| a.handle.cmp(&b.handle));

    Ok(BakedVectorAtlas {
        size: atlas_size,
        layers,
        atlas,
        table,
    })
}

/// Smallest power-of-two square atlas that packs `sizes` at minimum total VRAM
/// (`edge² · layers`). Grows the bin by powers of two up to the [`ATLAS_SIZE`]
/// cap and keeps whichever edge minimises the allocated texels — a small icon
/// set collapses to a tiny sheet instead of always paying for `ATLAS_SIZE`².
/// Returns `(edge, placements, layers)`, or `None` if a glyph exceeds the cap.
fn pack_minimal(sizes: &[Size]) -> Option<(u32, Vec<Placement>, u32)> {
    /// Floor edge: below a full-HD-ish sheet there's no reason to go smaller
    /// than one grid cell, but a 64² floor keeps tiny atlases GPU-friendly.
    const MIN_EDGE: u32 = 64;
    let mut best: Option<(u64, u32, Vec<Placement>, u32)> = None;
    let mut edge = MIN_EDGE;
    while edge <= ATLAS_SIZE {
        if let Some((placements, layers)) = pack_layers(sizes, edge, edge) {
            let vram = u64::from(edge) * u64::from(edge) * u64::from(layers);
            if best.as_ref().is_none_or(|b| vram < b.0) {
                best = Some((vram, edge, placements, layers));
            }
        }
        edge = edge.saturating_mul(2);
    }
    best.map(|(_, edge, placements, layers)| (edge, placements, layers))
}

/// The normalized `(u0, v0, u1, v1)` corners of a cell at pixel `(x, y)` sized
/// `w × h` within an `atlas_size`² sheet. Self-consistent with the baked atlas
/// (the runtime uploads `atlas_size` and indexes these UVs together), so the
/// sampled texels — and thus the render — match dev regardless of the size.
#[must_use]
#[allow(clippy::many_single_char_names)] // x/y/w/h/s are the natural rect names
fn cell_uv(atlas_size: u32, x: u32, y: u32, w: u32, h: u32) -> [f32; 4] {
    #[allow(clippy::cast_precision_loss)]
    let s = atlas_size as f32;
    #[allow(clippy::cast_precision_loss)]
    let (x, y, w, h) = (x as f32, y as f32, w as f32, h as f32);
    [x / s, y / s, (x + w) / s, (y + h) / s]
}

/// Copies a glyph's RGBA rows into the `atlas_size`² atlas at `(x, y, layer)`.
fn blit(
    atlas: &mut [u8],
    atlas_size: u32,
    glyph: &super::generate::MsdfGlyph,
    x: u32,
    y: u32,
    layer: u32,
) {
    let size = atlas_size as usize;
    let plane = size * size * 4;
    let (gw, gh) = (glyph.width as usize, glyph.height as usize);
    let (x, y, layer) = (x as usize, y as usize, layer as usize);
    for row in 0..gh {
        let dst = layer * plane + ((y + row) * size + x) * 4;
        let src = row * gw * 4;
        atlas[dst..dst + gw * 4].copy_from_slice(&glyph.bitmap[src..src + gw * 4]);
    }
}

/// The static string a `VectorIcon` argument resolves to, or `None` if it is a
/// non-literal (interpolated or computed) expression that cannot be closed at
/// build time.
fn static_handle(expr: &Expr) -> Option<String> {
    let Expr::StrLit(parts, _) = expr else {
        return None;
    };
    match parts.as_slice() {
        [] => Some(String::new()),
        [StrPart::Text(s)] => Some(s.clone()),
        // More than one part, or any `Interp`, means runtime-dependent content.
        _ => None,
    }
}

/// Visits every [`ElementNode`] in `members`, descending through elements,
/// `for`, and `when` bodies (the members that can contain further elements).
fn walk_members(members: &[Member], visit: &mut impl FnMut(&ElementNode)) {
    for m in members {
        match m {
            Member::Element(el) => {
                visit(el);
                walk_members(&el.children, visit);
            }
            Member::For { body, .. } => walk_members(body, visit),
            Member::When { then, els, .. } => {
                walk_members(then, visit);
                if let Some(els) = els {
                    walk_members(els, visit);
                }
            }
            Member::Var { .. }
            | Member::Let { .. }
            | Member::Fn { .. }
            | Member::Inject { .. }
            | Member::Style { .. }
            | Member::Expr(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    const SQUARE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <path d="M4 4 L20 4 L20 20 L4 20 Z" fill="#000000"/></svg>"##;
    const RING: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <path d="M4 4 L20 4 L20 20 L4 20 Z M8 8 L16 8 L16 16 L8 16 Z" fill="#000000" fill-rule="evenodd"/></svg>"##;

    fn views(src: &str) -> Vec<ViewDecl> {
        let parsed = parse(src);
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        parsed.views
    }

    #[test]
    fn collects_literal_refs_and_tree_shakes_unused_views() {
        let vs = views(
            r#"
            View Main() {
                Column {
                    VectorIcon("a.svg") #[size: 24]
                    VectorIcon("b.svg") #[size: 24]
                    VectorIcon("a.svg") #[size: 48]
                }
            }
            View Unused() { Text("no icons here") }
            "#,
        );
        let (refs, errs) = collect_static_vector_refs(&vs, &[]);
        assert!(errs.is_empty(), "{errs:?}");
        let handles: Vec<&str> = refs.iter().map(|r| r.handle.as_str()).collect();
        // Two `a.svg` (dedup happens at bake), one `b.svg`, nothing from `Unused`.
        assert_eq!(handles, vec!["a.svg", "b.svg", "a.svg"]);
    }

    #[test]
    fn a_dynamic_asset_without_an_include_list_is_a_hard_error() {
        let vs = views(r#"View Main() { var p = "x.svg" VectorIcon(p) #[size: 24] }"#);
        let (_refs, errs) = collect_static_vector_refs(&vs, &[]);
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], CompileError::VectorAssetNotStatic { .. }));
    }

    #[test]
    fn an_include_list_permits_dynamic_assets_and_seeds_the_set() {
        let vs = views(r#"View Main() { var p = "x.svg" VectorIcon(p) #[size: 24] }"#);
        let include = vec!["x.svg".to_string()];
        let (refs, errs) = collect_static_vector_refs(&vs, &include);
        assert!(errs.is_empty(), "an inclusion list must silence the guard");
        assert!(refs.iter().any(|r| r.handle == "x.svg"));
    }

    #[test]
    fn an_interpolated_string_is_not_static() {
        let vs = views(r#"View Main() { var n = "gear" VectorIcon("icons/{n}.svg") #[size: 24] }"#);
        let (_refs, errs) = collect_static_vector_refs(&vs, &[]);
        assert_eq!(errs.len(), 1, "an interpolated handle must be dynamic");
    }

    #[test]
    fn bake_dedups_generates_and_packs_the_closed_set() {
        let dir = std::env::temp_dir().join(format!("byard_aot_bake_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("square.svg"), SQUARE).unwrap();
        std::fs::write(dir.join("ring.svg"), RING).unwrap();

        // `square.svg` referenced twice → one baked entry (dedup).
        let refs = vec![
            StaticVectorRef {
                handle: "square.svg".into(),
                span: Span::new(0, 0),
            },
            StaticVectorRef {
                handle: "ring.svg".into(),
                span: Span::new(0, 0),
            },
            StaticVectorRef {
                handle: "square.svg".into(),
                span: Span::new(0, 0),
            },
        ];
        let baked = bake_atlas(&refs, &dir, None).expect("bake must succeed");

        assert_eq!(baked.table.len(), 2, "identical handles must dedup");
        assert_eq!(baked.layers, 1, "two 32px cells fit one layer");
        assert_eq!(
            baked.atlas.len(),
            (baked.size as usize).pow(2) * 4 * baked.layers as usize
        );
        // Distinct cells for distinct icons (bit comparison — exact by cell math).
        assert_ne!(
            baked.table[0].uv_rect.map(f32::to_bits),
            baked.table[1].uv_rect.map(f32::to_bits)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn baked_field_bytes_match_dev_and_uv_is_self_consistent() {
        // INV-7 (dev/prod render parity) after the tightest-fit change: the baked
        // UV corners are relative to the *baked* atlas size (not dev's 2048²), so
        // they no longer equal dev's corners. Parity now rests on two facts,
        // both asserted here: (a) the baked field texels are byte-identical to
        // what the generator produces for the dev path, and (b) the UV addresses
        // exactly that field's cell within the baked atlas — so the sampled
        // texels, and thus the render, are identical.
        let dir = std::env::temp_dir().join(format!("byard_aot_parity_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("square.svg"), SQUARE).unwrap();

        let refs = vec![StaticVectorRef {
            handle: "square.svg".into(),
            span: Span::new(0, 0),
        }];
        let baked = bake_atlas(&refs, &dir, None).unwrap();
        assert_eq!(baked.table.len(), 1);
        let entry = &baked.table[0];

        // (a) Field parity: the atlas region at the glyph's cell equals the raw
        // generated field bytes (the dev path uploads exactly these).
        let dev_field = crate::vector::generate::generate(
            SQUARE.as_bytes(),
            GRID_SIZE,
            PX_RANGE,
            Span::new(0, 0),
        )
        .unwrap()
        .bitmap;
        let size = baked.size as usize;
        let (cx, cy) = (0usize, 0usize); // first packed cell
        for row in 0..GRID_SIZE as usize {
            let dst = ((cy + row) * size + cx) * 4;
            let src = row * GRID_SIZE as usize * 4;
            let n = GRID_SIZE as usize * 4;
            assert_eq!(
                &baked.atlas[dst..dst + n],
                &dev_field[src..src + n],
                "baked texels must equal the dev-generated field"
            );
        }

        // (b) UV self-consistency: corners = cell / baked-atlas-size.
        #[allow(clippy::cast_precision_loss)]
        let s = baked.size as f32;
        #[allow(clippy::cast_precision_loss)]
        let c = GRID_SIZE as f32;
        assert_eq!(
            entry.uv_rect.map(f32::to_bits),
            [0.0, 0.0, c / s, c / s].map(f32::to_bits),
            "UV must address the glyph's cell within the baked atlas"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_small_icon_set_bakes_a_tiny_atlas_not_the_full_sheet() {
        // The north star: theoretical-minimum VRAM. Two 32px icons must not ship
        // a 2048² (~16 MB) sheet — they collapse to a 64² one (~16 KB).
        let dir = std::env::temp_dir().join(format!("byard_aot_vram_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("square.svg"), SQUARE).unwrap();
        std::fs::write(dir.join("ring.svg"), RING).unwrap();

        let refs = vec![
            StaticVectorRef {
                handle: "square.svg".into(),
                span: Span::new(0, 0),
            },
            StaticVectorRef {
                handle: "ring.svg".into(),
                span: Span::new(0, 0),
            },
        ];
        let baked = bake_atlas(&refs, &dir, None).unwrap();
        assert_eq!(baked.size, 64, "two 32px cells fit a 64² atlas");
        assert_eq!(baked.layers, 1);
        assert!(
            baked.atlas.len() <= 64 * 64 * 4,
            "atlas bytes track the tight size, not ATLAS_SIZE²"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
