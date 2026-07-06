//! AOT vector-atlas baking (RFC-0009 §4, M49).
//!
//! `byard build` closes the set of icons an app actually instantiates, bakes
//! only those into one immutable atlas, and emits a fixed coordinate table — so
//! a shipped binary uploads one texture and indexes a `[BakedGlyph; N]` with no
//! runtime SVG parsing or coordinate math (the dev/prod parity invariant INV-7:
//! the baked UV rect is byte-for-byte what the dev JIT produces for the same
//! cell).
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

use super::generate::{GRID_SIZE, PX_RANGE, generate};
use super::pack::{Size, pack_layers};

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
    /// Normalized UV rect `(u0, v0, u1, v1)` — identical to the corners a dev
    /// [`byard_core::frame::VectorInstance`] carries for the same atlas cell.
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
            Ok(bytes) => match generate(&bytes, GRID_SIZE, PX_RANGE, r.span) {
                Ok(glyph) => glyphs.push((r.handle.clone(), glyph)),
                Err(e) => errors.push(e),
            },
            Err(e) => errors.push(CompileError::Project {
                span: r.span,
                message: format!("cannot read vector asset {}: {e}", r.handle),
            }),
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Pack (all fields are GRID_SIZE² today, but the packer is size-general).
    let sizes: Vec<Size> = glyphs
        .iter()
        .map(|(_, g)| Size {
            w: g.width,
            h: g.height,
        })
        .collect();
    let Some((placements, layers)) = pack_layers(&sizes, ATLAS_SIZE, ATLAS_SIZE) else {
        return Err(vec![CompileError::Project {
            span: Span::new(0, 0),
            message: format!(
                "vector atlas packing failed: an icon exceeds the {ATLAS_SIZE}px atlas"
            ),
        }]);
    };

    // Blit each field into its cell and record its table entry.
    let plane = (ATLAS_SIZE as usize) * (ATLAS_SIZE as usize) * 4;
    let mut atlas = vec![0u8; plane * layers as usize];
    let mut table = Vec::with_capacity(glyphs.len());
    for p in &placements {
        let (handle, glyph) = &glyphs[p.index];
        blit(&mut atlas, glyph, p.x, p.y, p.layer);
        table.push(BakedGlyph {
            handle: handle.clone(),
            uv_rect: cell_uv(p.x, p.y, glyph.width, glyph.height),
            layer: p.layer,
            px_range: glyph.px_range,
        });
    }
    table.sort_by(|a, b| a.handle.cmp(&b.handle));

    Ok(BakedVectorAtlas {
        size: ATLAS_SIZE,
        layers,
        atlas,
        table,
    })
}

/// The normalized `(u0, v0, u1, v1)` corners of a cell at pixel `(x, y)` sized
/// `w × h` — the exact corners a dev `VectorInstance` carries (INV-7 parity).
#[must_use]
#[allow(clippy::many_single_char_names)] // x/y/w/h/s are the natural rect names
fn cell_uv(x: u32, y: u32, w: u32, h: u32) -> [f32; 4] {
    #[allow(clippy::cast_precision_loss)]
    let s = ATLAS_SIZE as f32;
    #[allow(clippy::cast_precision_loss)]
    let (x, y, w, h) = (x as f32, y as f32, w as f32, h as f32);
    [x / s, y / s, (x + w) / s, (y + h) / s]
}

/// Copies a glyph's RGBA rows into the atlas at `(x, y, layer)`.
fn blit(atlas: &mut [u8], glyph: &super::generate::MsdfGlyph, x: u32, y: u32, layer: u32) {
    let size = ATLAS_SIZE as usize;
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
        let baked = bake_atlas(&refs, &dir).expect("bake must succeed");

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
    fn baked_uv_rect_matches_the_dev_vector_instance_for_the_same_cell() {
        // INV-7 (dev/prod render parity): the AOT table's UV corners must be
        // byte-identical to what the dev JIT → `VectorInstance` produces for a
        // glyph in the same atlas cell.
        let dir = std::env::temp_dir().join(format!("byard_aot_parity_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("square.svg"), SQUARE).unwrap();

        let refs = vec![StaticVectorRef {
            handle: "square.svg".into(),
            span: Span::new(0, 0),
        }];
        let baked = bake_atlas(&refs, &dir).unwrap();
        assert_eq!(baked.table.len(), 1);
        let entry = &baked.table[0];

        // Reproduce the dev path's cell → UV corners for the same cell (0,0).
        #[allow(clippy::cast_precision_loss)]
        let s = baked.size as f32;
        #[allow(clippy::cast_precision_loss)]
        let cell = GRID_SIZE as f32;
        let dev_uv = byard_core::frame::Rect::new(0.0, 0.0, cell / s, cell / s);
        let dev_instance = byard_core::frame::VectorInstance::new(
            byard_core::frame::Rect::new(0.0, 0.0, 24.0, 24.0),
            dev_uv,
            [1.0, 1.0, 1.0, 1.0],
            entry.px_range,
            entry.layer,
        );
        // Bit-exact comparison: the two paths compute the corners with identical
        // integer→f32 math, so they must be byte-for-byte equal, not merely close.
        assert_eq!(
            entry.uv_rect.map(f32::to_bits),
            dev_instance.atlas_uv_rect.map(f32::to_bits),
            "AOT and dev must agree on the UV corners for the same cell"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
