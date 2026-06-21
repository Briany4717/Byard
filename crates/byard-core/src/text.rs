//! Glyph-accurate text measurement for layout.
//!
//! [`TextMeasurer`] shapes a string with the same `glyphon`/`cosmic-text`
//! engine the renderer uses (`encoder::text_glyph`), so the intrinsic size a
//! `Text`/`Button` reports to Taffy matches what is actually drawn — which is
//! what lets text be aligned and justified correctly within its box (rather
//! than estimated from a character count). The owning `FontSystem` is created
//! once (it loads the system fonts) and reused for every measurement.

use glyphon::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping};

/// Measures shaped text sizes, reusing one [`FontSystem`].
pub struct TextMeasurer {
    font_system: FontSystem,
}

impl Default for TextMeasurer {
    fn default() -> Self {
        Self::new()
    }
}

impl TextMeasurer {
    /// Creates a measurer with a fresh font system (loads system fonts once).
    #[must_use]
    pub fn new() -> Self {
        Self {
            font_system: FontSystem::new(),
        }
    }

    /// Returns the shaped `(width, height)` of `text` at `font_size` logical
    /// pixels, using a `1.2×` line height. Width is the widest laid-out line;
    /// height is `lines × line_height`. Empty text still reports one line's
    /// height so an empty label keeps its baseline.
    #[must_use]
    pub fn measure(&mut self, text: &str, font_size: f32) -> (f32, f32) {
        let line_height = font_size * 1.2;
        let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(font_size, line_height));
        // Unbounded so the natural (single-line) width is measured.
        buffer.set_size(&mut self.font_system, None, None);
        buffer.set_text(
            &mut self.font_system,
            text,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        buffer.shape_until_scroll(&mut self.font_system, false);

        let (width, lines) = buffer
            .layout_runs()
            .fold((0.0_f32, 0u32), |(w, n), run| (w.max(run.line_w), n + 1));
        #[allow(clippy::cast_precision_loss)]
        let height = lines.max(1) as f32 * line_height;
        (width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wider_text_measures_wider() {
        let mut m = TextMeasurer::new();
        let (w_short, h) = m.measure("i", 16.0);
        let (w_long, _) = m.measure("wwwwwwwwww", 16.0);
        assert!(
            w_long > w_short,
            "more glyphs ⇒ wider: {w_short} vs {w_long}"
        );
        assert!(w_short > 0.0 && h > 0.0);
    }

    #[test]
    fn larger_font_is_taller() {
        let mut m = TextMeasurer::new();
        let (_, h_small) = m.measure("Ag", 12.0);
        let (_, h_big) = m.measure("Ag", 48.0);
        assert!(h_big > h_small);
    }

    #[test]
    fn empty_text_keeps_one_line_height() {
        let mut m = TextMeasurer::new();
        let (w, h) = m.measure("", 16.0);
        assert!(w.abs() < 1e-6, "empty text has zero width, got {w}");
        assert!(h > 0.0);
    }
}
