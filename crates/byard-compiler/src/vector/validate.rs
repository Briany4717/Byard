//! The structural complexity guardrail (RFC-0009 §2, hard error).
//!
//! Keeps the JIT bounded: an SVG that paints with a gradient, pattern, or
//! filter — or one whose path segment count exceeds [`MAX_NODES`] — cannot go
//! through the monochrome MSDF pipeline and must route through `Image`
//! (`TextureSampler`) instead.

use crate::diagnostics::{CompileError, Span};

/// Default path-segment budget before a shape is rejected (IMPL-62, tunable).
pub const MAX_NODES: usize = 500;

/// Walks a parsed SVG tree, rejecting gradients/filters/patterns and counting
/// path segments; a total over [`MAX_NODES`] is a hard error.
pub fn validate_vector_complexity(tree: &usvg::Tree, span: Span) -> Result<(), CompileError> {
    let mut total_nodes = 0usize;
    check_group(tree.root(), span, &mut total_nodes)?;
    if total_nodes > MAX_NODES {
        return Err(CompileError::SvgTooComplexForMssdf {
            span,
            found_nodes: total_nodes,
        });
    }
    Ok(())
}

fn check_group(
    group: &usvg::Group,
    span: Span,
    total_nodes: &mut usize,
) -> Result<(), CompileError> {
    if !group.filters().is_empty() {
        return Err(CompileError::SvgUnsupportedFeatures { span });
    }
    for node in group.children() {
        match node {
            usvg::Node::Group(g) => check_group(g, span, total_nodes)?,
            usvg::Node::Path(p) => {
                *total_nodes += p.data().len();
                if let Some(fill) = p.fill() {
                    if !matches!(fill.paint(), usvg::Paint::Color(_)) {
                        return Err(CompileError::SvgUnsupportedFeatures { span });
                    }
                }
                if let Some(stroke) = p.stroke() {
                    if !matches!(stroke.paint(), usvg::Paint::Color(_)) {
                        return Err(CompileError::SvgUnsupportedFeatures { span });
                    }
                }
            }
            // Raster images and text-as-drawn glyphs have no monochrome-path
            // representation the MSDF pipeline can consume.
            usvg::Node::Image(_) | usvg::Node::Text(_) => {
                return Err(CompileError::SvgUnsupportedFeatures { span });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    fn parse(svg: &str) -> usvg::Tree {
        let opt = usvg::Options::default();
        usvg::Tree::from_data(svg.as_bytes(), &opt).expect("valid test SVG")
    }

    #[test]
    fn flat_fill_passes() {
        let tree = parse(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                 <path d="M0 0 L24 0 L24 24 L0 24 Z" fill="#000000"/>
               </svg>"##,
        );
        assert!(validate_vector_complexity(&tree, Span::new(0, 0)).is_ok());
    }

    #[test]
    fn linear_gradient_fill_is_rejected() {
        let tree = parse(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                 <defs>
                   <linearGradient id="g">
                     <stop offset="0" stop-color="#000000"/>
                     <stop offset="1" stop-color="#ffffff"/>
                   </linearGradient>
                 </defs>
                 <path d="M0 0 L24 0 L24 24 L0 24 Z" fill="url(#g)"/>
               </svg>"##,
        );
        assert_eq!(
            validate_vector_complexity(&tree, Span::new(0, 0)),
            Err(CompileError::SvgUnsupportedFeatures {
                span: Span::new(0, 0)
            })
        );
    }

    #[test]
    fn filter_is_rejected() {
        let tree = parse(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                 <defs>
                   <filter id="f"><feGaussianBlur stdDeviation="2"/></filter>
                 </defs>
                 <g filter="url(#f)">
                   <path d="M0 0 L24 0 L24 24 L0 24 Z" fill="#000000"/>
                 </g>
               </svg>"##,
        );
        assert_eq!(
            validate_vector_complexity(&tree, Span::new(0, 0)),
            Err(CompileError::SvgUnsupportedFeatures {
                span: Span::new(0, 0)
            })
        );
    }

    #[test]
    fn too_many_segments_is_rejected() {
        let mut d = String::from("M0 0 ");
        for i in 0..600 {
            let _ = write!(d, "L{} {} ", i % 24, (i * 3) % 24);
        }
        let svg = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                 <path d="{d}Z" fill="#000000"/>
               </svg>"##
        );
        let tree = parse(&svg);
        match validate_vector_complexity(&tree, Span::new(0, 0)) {
            Err(CompileError::SvgTooComplexForMssdf { found_nodes, .. }) => {
                assert!(found_nodes > MAX_NODES);
            }
            other => panic!("expected SvgTooComplexForMssdf, got {other:?}"),
        }
    }
}
