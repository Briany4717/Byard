//! Design-token theme (M22, RFC-0005 §6, D5 layer 1).
//!
//! A `Theme` bundles color and typography tokens. It is the default base style
//! (D5 layer 1): any intrinsic prop not set by a class or inline attribute falls
//! back to the theme. The theme is accessible in the interpreter and can be
//! overridden via `inject` at mount (M23 controller boundary).

/// Material 3 typography scale token names and their resolved point sizes.
/// Matches the M3 spec at the sizes recommended for dense UIs.
pub const TYPO_TOKENS: &[(&str, f32)] = &[
    ("displayLarge", 57.0),
    ("displayMedium", 45.0),
    ("displaySmall", 36.0),
    ("headlineLarge", 32.0),
    ("headlineMedium", 28.0),
    ("headlineSmall", 24.0),
    ("titleLarge", 22.0),
    ("titleMedium", 16.0),
    ("titleSmall", 14.0),
    ("bodyLarge", 16.0),
    ("bodyMedium", 14.0),
    ("bodySmall", 12.0),
    ("labelLarge", 14.0),
    ("labelMedium", 12.0),
    ("labelSmall", 11.0),
    // Short aliases used in byld samples.
    ("display", 36.0),
    ("headline", 24.0),
    ("title", 20.0),
    ("body", 14.0),
    ("caption", 12.0),
    ("label", 11.0),
];

/// Resolves a typography token name (e.g. `"titleLarge"`, `"m3.titleLarge"`) to
/// a font size in logical pixels. Returns `None` for unknown tokens.
#[must_use]
pub fn resolve_typo(token: &str) -> Option<f32> {
    let key = token.strip_prefix("m3.").unwrap_or(token);
    TYPO_TOKENS
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, size)| *size)
}

/// Color and typography design tokens for a view tree.
///
/// `Theme` provides the D5 layer-1 defaults for every intrinsic. When an
/// element does not set a prop inline or via a class, the theme value is used.
#[derive(Clone, Debug)]
pub struct Theme {
    /// Primary brand color (0xRRGGBB packed).
    pub primary: i64,
    /// Color of text/icons on the primary color.
    pub on_primary: i64,
    /// Surface (background) color.
    pub surface: i64,
    /// Color of text/icons on surfaces.
    pub on_surface: i64,
    /// Secondary / accent color.
    pub secondary: i64,
    /// Color of text/icons on the secondary color.
    pub on_secondary: i64,
    /// Error color.
    pub error: i64,
    /// Color of text/icons on the error color.
    pub on_error: i64,
    /// Default font size in logical pixels.
    pub font_size: f32,
}

impl Theme {
    /// A sensible Material 3-inspired light theme.
    #[must_use]
    pub fn light() -> Self {
        Self {
            primary: 0x0000_6495,
            on_primary: 0x00ff_ffff,
            surface: 0x00ff_ffff,
            on_surface: 0x001c_1b1f,
            secondary: 0x004f_5d75,
            on_secondary: 0x00ff_ffff,
            error: 0x00b3_261e,
            on_error: 0x00ff_ffff,
            font_size: 14.0,
        }
    }

    /// A Material 3-inspired dark theme.
    #[must_use]
    pub fn dark() -> Self {
        Self {
            primary: 0x0090_caf9,
            on_primary: 0x0000_3258,
            surface: 0x001c_1b1f,
            on_surface: 0x00e6_e1e5,
            secondary: 0x00b0_bec5,
            on_secondary: 0x0022_333b,
            error: 0x00f2_b8b5,
            on_error: 0x0060_1410,
            font_size: 14.0,
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::light()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_typo_known_tokens() {
        assert_eq!(resolve_typo("titleLarge"), Some(22.0));
        assert_eq!(resolve_typo("bodyMedium"), Some(14.0));
        assert_eq!(resolve_typo("display"), Some(36.0));
    }

    #[test]
    fn resolve_typo_m3_prefix_stripped() {
        assert_eq!(resolve_typo("m3.titleLarge"), Some(22.0));
        assert_eq!(resolve_typo("m3.headlineLarge"), Some(32.0));
    }

    #[test]
    fn resolve_typo_unknown_returns_none() {
        assert_eq!(resolve_typo("notAToken"), None);
    }

    #[test]
    fn light_and_dark_differ_on_surface() {
        let light = Theme::light();
        let dark = Theme::dark();
        assert_ne!(light.on_surface, dark.on_surface);
        assert_ne!(light.surface, dark.surface);
    }

    #[test]
    fn default_is_light() {
        let t = Theme::default();
        assert_eq!(t.on_surface, Theme::light().on_surface);
    }
}
