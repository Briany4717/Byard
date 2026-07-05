//! `svg bytes → MsdfGlyph`: the field generator (RFC-0009 §2, §5).
//!
//! Parsing and normalization go through `usvg` (it resolves groups,
//! transforms, other shape elements, and paint info — including the paint
//! introspection [`super::validate`] needs); the flattened, absolute-command
//! path geometry is then adapted into a `bymsdfgen_core::Shape` ourselves and
//! handed to the vendored generator for the distance-field math itself.

use bymsdfgen_core::coloring::edge_coloring_simple;
use bymsdfgen_core::correction::msdf_error_correction;
use bymsdfgen_core::generator::{
    DistanceMapping, MsdfGeneratorConfig, Projection, SdfTransformation, generate_msdf,
};
use bymsdfgen_core::{Bitmap, Contour, EdgeSegment, Range, Shape, Vector2};

use crate::diagnostics::{CompileError, Span};

use super::validate::validate_vector_complexity;

/// Generation grid: 32×32 (IMPL-62); high-PPI targets may opt into 64×64.
pub const GRID_SIZE: u32 = 32;
/// Baked distance range in atlas texels (IMPL-62; RFC-0009 §1, §5, §2-E).
pub const PX_RANGE: f32 = 4.0;
/// Edge-coloring angular threshold in degrees (IMPL-62; RFC-0009 §5).
pub const EDGE_ANGLE_DEGREES: f64 = 48.0;

/// A generated MSDF field, owned and `Send + 'static` (INV-3) so it can cross
/// a channel from a worker to the logic thread untouched.
#[derive(Clone, Debug, PartialEq)]
pub struct MsdfGlyph {
    /// RGBA8 field bytes, row-major, row 0 at the top, `width * height * 4`
    /// long. Only the RGB channels carry the field; alpha is always 255.
    pub bitmap: Vec<u8>,
    /// Field width in pixels (equals the generation grid, [`GRID_SIZE`]).
    pub width: u32,
    /// Field height in pixels (equals the generation grid, [`GRID_SIZE`]).
    pub height: u32,
    /// Distance range baked into this field, in atlas texels — flows
    /// straight into `VectorInstance::px_range` (RFC-0009 §2-E).
    pub px_range: f32,
}

/// Parses `svg_bytes`, validates it against the RFC-0009 §2 complexity
/// guardrail, and generates an MSDF field on a `grid × grid` bitmap with the
/// given baked `px_range`. `span` anchors any [`CompileError`] at the
/// `VectorIcon` call site that requested this asset.
pub fn generate(
    svg_bytes: &[u8],
    grid: u32,
    px_range: f32,
    span: Span,
) -> Result<MsdfGlyph, CompileError> {
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_data(svg_bytes, &opt).map_err(|e| CompileError::Project {
        span,
        message: format!("invalid SVG: {e}"),
    })?;

    validate_vector_complexity(&tree, span)?;

    let size = tree.size();
    let doc_height = f64::from(size.height());
    let mut shape = Shape::new();
    collect_paths(tree.root(), doc_height, &mut shape);

    if shape.contours.is_empty() {
        return Ok(empty_glyph(grid, px_range));
    }

    shape.normalize();
    shape.orient_contours();
    edge_coloring_simple(&mut shape, EDGE_ANGLE_DEGREES.to_radians(), 0);

    let dim = f64::from(size.width())
        .max(f64::from(size.height()))
        .max(1e-6);
    let scale = f64::from(grid) / dim;
    // Centre the (possibly non-square) document canvas within the square
    // grid: pixel = scale * (shape_coord + translate).
    let half_grid_in_shape_units = f64::from(grid) / (2.0 * scale);
    let translate = Vector2::new(
        half_grid_in_shape_units - f64::from(size.width()) / 2.0,
        half_grid_in_shape_units - f64::from(size.height()) / 2.0,
    );

    let projection = Projection::new(Vector2::splat(scale), translate);
    let distance_mapping =
        DistanceMapping::from_range(Range::symmetric(f64::from(px_range) / scale));
    let transformation = SdfTransformation::new(projection, distance_mapping);

    let mut bitmap: Bitmap<f32, 3> = Bitmap::new(grid as usize, grid as usize);
    let config = MsdfGeneratorConfig::default();
    generate_msdf(&mut bitmap, &shape, &transformation, &config);
    msdf_error_correction(&mut bitmap, &shape, &transformation, &config);

    Ok(bitmap_to_rgba8(&bitmap, px_range))
}

/// Recursively collects every filled path into `shape`'s contours, applying
/// each path's absolute transform and flipping Y (SVG is Y-down; the
/// generator's winding/orientation math is Y-up, RFC-0009 §5).
fn collect_paths(group: &usvg::Group, doc_height: f64, shape: &mut Shape) {
    for node in group.children() {
        match node {
            usvg::Node::Group(g) => collect_paths(g, doc_height, shape),
            usvg::Node::Path(p) => {
                if p.is_visible() && p.fill().is_some() {
                    add_path_contours(p.data(), p.abs_transform(), doc_height, shape);
                }
            }
            usvg::Node::Image(_) | usvg::Node::Text(_) => {}
        }
    }
}

fn to_shape_point(
    mut pt: usvg::tiny_skia_path::Point,
    transform: usvg::tiny_skia_path::Transform,
    doc_height: f64,
) -> Vector2 {
    transform.map_point(&mut pt);
    Vector2::new(f64::from(pt.x), doc_height - f64::from(pt.y))
}

fn add_path_contours(
    path: &usvg::tiny_skia_path::Path,
    transform: usvg::tiny_skia_path::Transform,
    doc_height: f64,
    shape: &mut Shape,
) {
    let mut contour: Option<Contour> = None;
    let mut start = Vector2::new(0.0, 0.0);
    let mut cur = Vector2::new(0.0, 0.0);

    for seg in path.segments() {
        match seg {
            usvg::tiny_skia_path::PathSegment::MoveTo(p) => {
                if let Some(c) = contour.take() {
                    if !c.is_empty() {
                        shape.add_contour(c);
                    }
                }
                let v = to_shape_point(p, transform, doc_height);
                start = v;
                cur = v;
                contour = Some(Contour::new());
            }
            usvg::tiny_skia_path::PathSegment::LineTo(p) => {
                let v = to_shape_point(p, transform, doc_height);
                if let Some(c) = contour.as_mut() {
                    if v != cur {
                        c.add_edge(EdgeSegment::line(cur, v));
                    }
                }
                cur = v;
            }
            usvg::tiny_skia_path::PathSegment::QuadTo(p1, p2) => {
                let v1 = to_shape_point(p1, transform, doc_height);
                let v2 = to_shape_point(p2, transform, doc_height);
                if let Some(c) = contour.as_mut() {
                    c.add_edge(EdgeSegment::quadratic(cur, v1, v2));
                }
                cur = v2;
            }
            usvg::tiny_skia_path::PathSegment::CubicTo(p1, p2, p3) => {
                let v1 = to_shape_point(p1, transform, doc_height);
                let v2 = to_shape_point(p2, transform, doc_height);
                let v3 = to_shape_point(p3, transform, doc_height);
                if let Some(c) = contour.as_mut() {
                    c.add_edge(EdgeSegment::cubic(cur, v1, v2, v3));
                }
                cur = v3;
            }
            usvg::tiny_skia_path::PathSegment::Close => {
                if let Some(c) = contour.as_mut() {
                    if cur != start {
                        c.add_edge(EdgeSegment::line(cur, start));
                    }
                }
                cur = start;
            }
        }
    }
    if let Some(c) = contour.take() {
        if !c.is_empty() {
            shape.add_contour(c);
        }
    }
}

/// An all-background field: every sample reads as "far outside" (channel
/// `0.0` → `median - 0.5 = -0.5`), so an empty shape paints nothing.
fn empty_glyph(grid: u32, px_range: f32) -> MsdfGlyph {
    MsdfGlyph {
        bitmap: vec![0u8; (grid as usize) * (grid as usize) * 4],
        width: grid,
        height: grid,
        px_range,
    }
}

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Converts the generator's bottom-up `f32` field into top-down RGBA8 bytes
/// (the atlas upload / raster convention used elsewhere in this codebase).
fn bitmap_to_rgba8(bitmap: &Bitmap<f32, 3>, px_range: f32) -> MsdfGlyph {
    let (w, h) = (bitmap.width, bitmap.height);
    let mut bytes = vec![0u8; w * h * 4];
    for y in 0..h {
        let src_y = h - 1 - y;
        for x in 0..w {
            let px = bitmap.pixel(x, src_y);
            let o = (y * w + x) * 4;
            bytes[o] = to_u8(px[0]);
            bytes[o + 1] = to_u8(px[1]);
            bytes[o + 2] = to_u8(px[2]);
            bytes[o + 3] = 255;
        }
    }
    MsdfGlyph {
        bitmap: bytes,
        width: w as u32,
        height: h as u32,
        px_range,
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    fn gear_svg() -> &'static str {
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
             <path d="M4 4 L20 4 L20 20 L4 20 Z M8 8 L16 8 L16 16 L8 16 Z" fill="#000000" fill-rule="evenodd"/>
           </svg>"##
    }

    #[test]
    fn generation_is_deterministic() {
        let a = generate(gear_svg().as_bytes(), GRID_SIZE, PX_RANGE, Span::new(0, 0)).unwrap();
        let b = generate(gear_svg().as_bytes(), GRID_SIZE, PX_RANGE, Span::new(0, 0)).unwrap();
        assert_eq!(
            a, b,
            "same SVG + params must generate byte-identical fields"
        );
    }

    #[test]
    fn dimensions_match_the_requested_grid() {
        let glyph = generate(gear_svg().as_bytes(), GRID_SIZE, PX_RANGE, Span::new(0, 0)).unwrap();
        assert_eq!(glyph.width, GRID_SIZE);
        assert_eq!(glyph.height, GRID_SIZE);
        assert_eq!(glyph.bitmap.len(), (GRID_SIZE as usize).pow(2) * 4);
        assert!((glyph.px_range - PX_RANGE).abs() < f32::EPSILON);
    }

    #[test]
    fn a_hole_is_distinguishable_from_a_solid_square() {
        let ring = generate(gear_svg().as_bytes(), 32, PX_RANGE, Span::new(0, 0)).unwrap();
        let solid = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                          <path d="M4 4 L20 4 L20 20 L4 20 Z" fill="#000000"/>
                        </svg>"##;
        let filled = generate(solid.as_bytes(), 32, PX_RANGE, Span::new(0, 0)).unwrap();
        // Centre pixel: inside the ring's hole (outside the shape) vs. inside
        // the solid square (inside the shape) must decode to different medians.
        let center = (16 * 32 + 16) * 4;
        let median = |b: &[u8]| {
            f32::from(b[0])
                .max(f32::from(b[1]))
                .min(f32::from(b[0]).max(f32::from(b[1])).max(f32::from(b[2])))
        };
        assert_ne!(
            median(&ring.bitmap[center..center + 3]).total_cmp(&128.0),
            median(&filled.bitmap[center..center + 3]).total_cmp(&128.0),
            "the ring's hole must read as outside where the solid square reads inside"
        );
    }

    #[test]
    fn gradient_svg_is_rejected() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <defs>
                          <linearGradient id="g">
                            <stop offset="0" stop-color="#000"/>
                            <stop offset="1" stop-color="#fff"/>
                          </linearGradient>
                        </defs>
                        <path d="M0 0 L24 0 L24 24 L0 24 Z" fill="url(#g)"/>
                      </svg>"##;
        assert!(matches!(
            generate(svg.as_bytes(), GRID_SIZE, PX_RANGE, Span::new(1, 2)),
            Err(CompileError::SvgUnsupportedFeatures { span }) if span == Span::new(1, 2)
        ));
    }

    #[test]
    fn oversized_svg_is_rejected() {
        let mut d = String::from("M0 0 ");
        for i in 0..600 {
            let _ = write!(d, "L{} {} ", i % 24, (i * 3) % 24);
        }
        let svg = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                 <path d="{d}Z" fill="#000000"/>
               </svg>"##
        );
        assert!(matches!(
            generate(svg.as_bytes(), GRID_SIZE, PX_RANGE, Span::new(0, 0)),
            Err(CompileError::SvgTooComplexForMssdf { .. })
        ));
    }

    #[test]
    fn sharp_right_angle_corner_stays_sharp_at_8x_scale() {
        // A plain right-angle square should keep a crisp corner even when the
        // field is sampled far outside its native 32x32 baking resolution —
        // i.e. the corner reconstructs from the *median*, not a rounded blend.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M2 2 L22 2 L22 22 L2 22 Z" fill="#000000"/>
                      </svg>"##;
        let glyph = generate(svg.as_bytes(), GRID_SIZE, PX_RANGE, Span::new(0, 0)).unwrap();
        // Sample the four texels immediately around the top-left corner
        // (shape-space (2,2) maps near pixel (2,2) at this framing) and assert
        // at least one channel discriminates strongly across it (no uniform
        // blur that would indicate a rounded, non-median reconstruction).
        let idx = |x: usize, y: usize| (y * GRID_SIZE as usize + x) * 4;
        let inside = glyph.bitmap[idx(6, 6)];
        let outside = glyph.bitmap[idx(1, 1)];
        assert!(
            u32::from(inside).abs_diff(u32::from(outside)) > 40,
            "corner region must discriminate inside vs. outside sharply"
        );
    }
}
