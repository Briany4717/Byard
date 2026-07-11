//! Design-token theme runtime (RFC-0022, completing RFC-0008 Pillar D and
//! RFC-0005 §6 `Typo` resolution).
//!
//! A [`Theme`] is a package/project-declared, engine-resolved bundle of **color**
//! (per named scheme — `light`/`dark`), **typography**, and **shape** tokens. It
//! serves two roles:
//!
//! 1. **Theme-default layer** (RFC-0005 §6) — any intrinsic prop not set by a class or inline
//!    attribute falls back to the theme (the pre-existing `Theme::light()` role).
//! 2. **Reactive token references** — `inject Theme as t` then `t.primary`,
//!    `t.titleLarge`, `t.cornerLg` in `.byd` resolve against the *active* scheme.
//!    Flipping the scheme (`t.dark = true`) is a single reactive write that
//!    marks every token-dependent binding dirty (RFC-0022 §1).
//!
//! Tokens are declared `snake_case` in `byard.toml` (TOML convention) and
//! referenced `camelCase` in `byld` (language convention); [`to_camel`] performs
//! the mapping at parse time so both maps share one canonical `camelCase` key.
//!
//! The engine ships a built-in [`Theme::byard_base`] (RFC-0022 §6): a neutral
//! gray palette, the M3 type scale, and the 0/4/8/12/16/28 shape scale. A
//! manifest theme layers on top — it overrides only the tokens it redefines;
//! everything else falls through to `byard-base`.

use std::collections::BTreeMap;

/// A font weight, resolved from a manifest `weight = "…"` string or the M3
/// defaults. The numeric value is the CSS/OpenType weight axis (100–900).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FontWeight {
    /// 100.
    Thin,
    /// 300.
    Light,
    /// 400 — the default.
    #[default]
    Regular,
    /// 500.
    Medium,
    /// 600.
    SemiBold,
    /// 700.
    Bold,
    /// 900.
    Black,
}

impl FontWeight {
    /// Parses a manifest weight string (case-insensitive). Unknown strings map
    /// to [`FontWeight::Regular`] — the caller reports the diagnostic.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "thin" | "100" => Self::Thin,
            "light" | "300" => Self::Light,
            "regular" | "normal" | "400" => Self::Regular,
            "medium" | "500" => Self::Medium,
            "semibold" | "semi_bold" | "600" => Self::SemiBold,
            "bold" | "700" => Self::Bold,
            "black" | "900" => Self::Black,
            _ => return None,
        })
    }

    /// The numeric `OpenType` weight axis value (100–900).
    #[must_use]
    pub fn axis(self) -> u16 {
        match self {
            Self::Thin => 100,
            Self::Light => 300,
            Self::Regular => 400,
            Self::Medium => 500,
            Self::SemiBold => 600,
            Self::Bold => 700,
            Self::Black => 900,
        }
    }
}

/// A resolved typography token (RFC-0022 §1, RFC-0005 `Typo`).
#[derive(Clone, Debug, PartialEq)]
pub struct TypoToken {
    /// The font family name (e.g. `"Roboto"`). `None` ⇒ the system font. A
    /// declared-but-unregistered family falls back to the system font with a
    /// `FontNotFound` warning (RFC-0022 §3-§4).
    pub family: Option<String>,
    /// Font size in logical pixels.
    pub size: f32,
    /// Font weight.
    pub weight: FontWeight,
    /// Letter tracking (spacing) in logical pixels.
    pub tracking: f32,
    /// Line height in logical pixels.
    pub line_height: f32,
}

impl TypoToken {
    /// A plain token: a size, regular weight, and a line height ≈ 1.25× size.
    #[must_use]
    pub fn plain(size: f32) -> Self {
        Self {
            family: None,
            size,
            weight: FontWeight::Regular,
            tracking: 0.0,
            line_height: (size * 1.25).round(),
        }
    }
}

/// Maps a `snake_case` manifest key to the `camelCase` byld reference form
/// (RFC-0022 unresolved-question resolution: snake in TOML, camel in byld).
///
/// `primary_container` → `primaryContainer`; `title_large` → `titleLarge`. A key
/// already in `camelCase` (no underscores) is returned unchanged.
#[must_use]
pub fn to_camel(snake: &str) -> String {
    let mut out = String::with_capacity(snake.len());
    let mut upper_next = false;
    for ch in snake.chars() {
        if ch == '_' {
            upper_next = true;
        } else if upper_next {
            out.extend(ch.to_uppercase());
            upper_next = false;
        } else {
            out.push(ch);
        }
    }
    out
}

/// The Material 3 typography scale: `camelCase` token → (size, weight, tracking).
/// Line height is derived (`plain`) unless a manifest overrides the whole token.
/// Used both for `byard-base` typography and as the fallback for a bare
/// `typo: titleLarge` size (RFC-0005 §6).
const M3_TYPO: &[(&str, f32, FontWeight, f32)] = &[
    ("displayLarge", 57.0, FontWeight::Regular, -0.25),
    ("displayMedium", 45.0, FontWeight::Regular, 0.0),
    ("displaySmall", 36.0, FontWeight::Regular, 0.0),
    ("headlineLarge", 32.0, FontWeight::Regular, 0.0),
    ("headlineMedium", 28.0, FontWeight::Regular, 0.0),
    ("headlineSmall", 24.0, FontWeight::Regular, 0.0),
    ("titleLarge", 22.0, FontWeight::Regular, 0.0),
    ("titleMedium", 16.0, FontWeight::Medium, 0.15),
    ("titleSmall", 14.0, FontWeight::Medium, 0.1),
    ("bodyLarge", 16.0, FontWeight::Regular, 0.5),
    ("bodyMedium", 14.0, FontWeight::Regular, 0.25),
    ("bodySmall", 12.0, FontWeight::Regular, 0.4),
    ("labelLarge", 14.0, FontWeight::Medium, 0.1),
    ("labelMedium", 12.0, FontWeight::Medium, 0.5),
    ("labelSmall", 11.0, FontWeight::Medium, 0.5),
    // Short aliases used across byld samples.
    ("display", 36.0, FontWeight::Regular, 0.0),
    ("headline", 24.0, FontWeight::Regular, 0.0),
    ("title", 20.0, FontWeight::Regular, 0.0),
    ("body", 14.0, FontWeight::Regular, 0.0),
    ("caption", 12.0, FontWeight::Regular, 0.0),
    ("label", 11.0, FontWeight::Medium, 0.0),
];

/// The M3 shape scale: `camelCase` corner token → radius (RFC-0022 §6).
const BASE_SHAPES: &[(&str, f32)] = &[
    ("cornerNone", 0.0),
    ("cornerXs", 4.0),
    ("cornerSm", 8.0),
    ("cornerMd", 12.0),
    ("cornerLg", 16.0),
    ("cornerXl", 28.0),
    ("cornerFull", 9999.0),
];

/// The `byard-base` light color scheme (`camelCase` token → 0xRRGGBB).
///
/// The core tokens (`primary`/`surface`/`onSurface`/`secondary`/`error`) keep
/// the historical `Theme::light()` values so the *default* look — every
/// intrinsic that falls back to the theme, and the pixel-coupled render tests —
/// is unchanged; RFC-0022 adds the container/outline tokens and the dark scheme.
const BASE_LIGHT: &[(&str, i64)] = &[
    ("primary", 0x0000_6495),
    ("onPrimary", 0x00FF_FFFF),
    ("primaryContainer", 0x00D6_E3FF),
    ("onPrimaryContainer", 0x0000_1B3E),
    ("secondary", 0x004F_5D75),
    ("onSecondary", 0x00FF_FFFF),
    ("secondaryContainer", 0x00D9_E3F8),
    ("onSecondaryContainer", 0x0012_1C2B),
    ("surface", 0x00FF_FFFF),
    ("surfaceContainerLow", 0x00F7_F2FA),
    ("surfaceContainer", 0x00F1_ECF4),
    ("surfaceContainerHigh", 0x00EC_E6EE),
    ("onSurface", 0x001C_1B1F),
    ("onSurfaceVariant", 0x0043_474E),
    ("outline", 0x0073_777F),
    ("outlineVariant", 0x00C3_C7CF),
    ("background", 0x00FF_FFFF),
    ("onBackground", 0x001C_1B1F),
    ("error", 0x00B3_261E),
    ("onError", 0x00FF_FFFF),
];

/// The `byard-base` dark color scheme (`camelCase` token → 0xRRGGBB). Core
/// tokens keep the historical `Theme::dark()` values.
const BASE_DARK: &[(&str, i64)] = &[
    ("primary", 0x0090_CAF9),
    ("onPrimary", 0x0000_3258),
    ("primaryContainer", 0x002E_4578),
    ("onPrimaryContainer", 0x00D6_E3FF),
    ("secondary", 0x00B0_BEC5),
    ("onSecondary", 0x0022_333B),
    ("secondaryContainer", 0x003D_4758),
    ("onSecondaryContainer", 0x00D9_E3F8),
    ("surface", 0x001C_1B1F),
    ("surfaceContainerLow", 0x001A_1C1E),
    ("surfaceContainer", 0x001E_2022),
    ("surfaceContainerHigh", 0x0029_2A2D),
    ("onSurface", 0x00E6_E1E5),
    ("onSurfaceVariant", 0x00C3_C7CF),
    ("outline", 0x008D_9199),
    ("outlineVariant", 0x0043_474E),
    ("background", 0x001C_1B1F),
    ("onBackground", 0x00E6_E1E5),
    ("error", 0x00F2_B8B5),
    ("onError", 0x0060_1410),
];

/// The default logical-pixel font size when no `titleLarge`-style token applies.
pub const DEFAULT_FONT_SIZE: f32 = 14.0;

/// The scheme name used for the light color scheme.
pub const SCHEME_LIGHT: &str = "light";
/// The scheme name used for the dark color scheme.
pub const SCHEME_DARK: &str = "dark";

/// Color, typography, and shape design tokens for a view tree (RFC-0022 §1).
///
/// The runtime *active scheme* is not stored here — it lives in a reactive
/// signal owned by the interpreter, so a scheme flip participates in
/// Mark-and-Pull. [`active_dark`](Self::active_dark) mirrors it for the
/// non-reactive theme-default accessors ([`primary`](Self::primary) etc.).
#[derive(Clone, Debug)]
pub struct Theme {
    /// Theme name (`[theme].name`), for diagnostics.
    pub name: String,
    /// `scheme name → (camelCase token → 0xRRGGBB)`.
    schemes: BTreeMap<String, BTreeMap<String, i64>>,
    /// `camelCase token → typography`.
    typography: BTreeMap<String, TypoToken>,
    /// `camelCase token → corner radius`.
    shapes: BTreeMap<String, f32>,
    /// Declared font families available for `TypoToken.family` resolution
    /// (RFC-0022 §3): `family name → asset path`. Registration of the bytes
    /// themselves is deferred; presence here suppresses the `FontNotFound`
    /// warning.
    fonts: BTreeMap<String, String>,
    /// The active scheme mirror for default resolution (`true` ⇒ `dark`).
    pub active_dark: bool,
    /// Default font size in logical pixels (the theme-default layer).
    pub font_size: f32,
}

impl Theme {
    /// The engine's built-in `byard-base` theme (RFC-0022 §6): a neutral gray
    /// palette, the M3 type scale, and the 0/4/8/12/16/28 shape scale.
    #[must_use]
    pub fn byard_base() -> Self {
        let mut schemes = BTreeMap::new();
        schemes.insert(
            SCHEME_LIGHT.to_string(),
            BASE_LIGHT
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
        );
        schemes.insert(
            SCHEME_DARK.to_string(),
            BASE_DARK
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
        );
        let typography = M3_TYPO
            .iter()
            .map(|(name, size, weight, tracking)| {
                (
                    (*name).to_string(),
                    TypoToken {
                        family: None,
                        size: *size,
                        weight: *weight,
                        tracking: *tracking,
                        line_height: (size * 1.25).round(),
                    },
                )
            })
            .collect();
        let shapes = BASE_SHAPES
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect();
        Self {
            name: "byard-base".to_string(),
            schemes,
            typography,
            shapes,
            fonts: BTreeMap::new(),
            active_dark: false,
            font_size: DEFAULT_FONT_SIZE,
        }
    }

    /// A Material 3-inspired light-first theme — `byard-base` is close enough to
    /// serve as the historical `Theme::light()` default. Kept for the interpreter
    /// default and existing call sites.
    #[must_use]
    pub fn light() -> Self {
        Self::byard_base()
    }

    /// `byard-base` with the dark scheme active.
    #[must_use]
    pub fn dark() -> Self {
        let mut t = Self::byard_base();
        t.active_dark = true;
        t
    }

    /// Whether this theme declares a `dark` color scheme with any tokens.
    #[must_use]
    pub fn has_dark(&self) -> bool {
        self.schemes.get(SCHEME_DARK).is_some_and(|m| !m.is_empty())
    }

    /// The scheme name for the given dark flag, falling back to `light` when the
    /// theme has no dark scheme.
    #[must_use]
    pub fn scheme_name(&self, dark: bool) -> &str {
        if dark && self.has_dark() {
            SCHEME_DARK
        } else {
            SCHEME_LIGHT
        }
    }

    /// Resolves a `camelCase` color token in the given scheme, falling back to
    /// the light scheme, then to `byard-base`. Returns `None` for an unknown
    /// token.
    #[must_use]
    pub fn color(&self, token: &str, dark: bool) -> Option<i64> {
        let scheme = self.scheme_name(dark);
        self.schemes
            .get(scheme)
            .and_then(|m| m.get(token))
            .or_else(|| self.schemes.get(SCHEME_LIGHT).and_then(|m| m.get(token)))
            .copied()
    }

    /// Resolves a `camelCase` typography token (RFC-0005 `typo:`).
    #[must_use]
    pub fn typo(&self, token: &str) -> Option<&TypoToken> {
        let key = token.strip_prefix("m3.").unwrap_or(token);
        self.typography.get(key)
    }

    /// The size of a typography token, for the `typo:` size-only fast path.
    #[must_use]
    pub fn typo_size(&self, token: &str) -> Option<f32> {
        self.typo(token).map(|t| t.size)
    }

    /// Resolves a `camelCase` shape (corner-radius) token.
    #[must_use]
    pub fn shape(&self, token: &str) -> Option<f32> {
        self.shapes.get(token).copied()
    }

    /// Whether a font family has been declared in `[assets.fonts]` (RFC-0022 §3).
    #[must_use]
    pub fn has_font(&self, family: &str) -> bool {
        self.fonts.contains_key(family) || self.fonts.keys().any(|k| to_camel(k) == family)
    }

    /// Declared font families (`name → asset path`).
    #[must_use]
    pub fn fonts(&self) -> &BTreeMap<String, String> {
        &self.fonts
    }

    /// Registers a font family declared in `[assets.fonts]`.
    pub fn add_font(&mut self, name: impl Into<String>, path: impl Into<String>) {
        self.fonts.insert(name.into(), path.into());
    }

    /// Sets (or overrides) a color token in a scheme. Keys are canonicalized to
    /// `camelCase`. Creates the scheme if absent.
    pub fn set_color(&mut self, scheme: &str, token: &str, rgb: i64) {
        self.schemes
            .entry(scheme.to_string())
            .or_default()
            .insert(to_camel(token), rgb);
    }

    /// Sets (or overrides) a typography token. The key is canonicalized to
    /// `camelCase`.
    pub fn set_typo(&mut self, token: &str, tok: TypoToken) {
        self.typography.insert(to_camel(token), tok);
    }

    /// Sets (or overrides) a shape token. The key is canonicalized to
    /// `camelCase`.
    pub fn set_shape(&mut self, token: &str, radius: f32) {
        self.shapes.insert(to_camel(token), radius);
    }

    // ── Theme-default accessors (non-reactive; read `active_dark`) ──────────

    /// Primary brand color for the active scheme.
    #[must_use]
    pub fn primary(&self) -> i64 {
        self.color("primary", self.active_dark)
            .unwrap_or(0x0000_6495)
    }

    /// On-surface (default text/icon) color for the active scheme.
    #[must_use]
    pub fn on_surface(&self) -> i64 {
        self.color("onSurface", self.active_dark)
            .unwrap_or(0x001C_1B1F)
    }

    /// Surface (default background) color for the active scheme.
    #[must_use]
    pub fn surface(&self) -> i64 {
        self.color("surface", self.active_dark)
            .unwrap_or(0x00FF_FFFF)
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::byard_base()
    }
}

/// Resolves a typography token name to a font size using the built-in M3 scale
/// (no theme override). Retained for callers without a [`Theme`] in hand.
#[must_use]
pub fn resolve_typo(token: &str) -> Option<f32> {
    let key = token.strip_prefix("m3.").unwrap_or(token);
    M3_TYPO
        .iter()
        .find(|(name, ..)| *name == key)
        .map(|(_, size, ..)| *size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_camel_maps_snake_and_leaves_camel() {
        assert_eq!(to_camel("primary_container"), "primaryContainer");
        assert_eq!(to_camel("title_large"), "titleLarge");
        assert_eq!(to_camel("primary"), "primary");
        assert_eq!(to_camel("onSurface"), "onSurface");
    }

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
    fn base_light_and_dark_differ_on_surface() {
        let t = Theme::byard_base();
        assert_ne!(t.color("onSurface", false), t.color("onSurface", true));
        assert_ne!(t.color("surface", false), t.color("surface", true));
    }

    #[test]
    fn color_falls_back_to_light_when_dark_token_missing() {
        let mut t = Theme::byard_base();
        // A custom token defined only in light resolves in dark via fallback.
        t.set_color("light", "brand", 0x0012_3456);
        assert_eq!(t.color("brand", true), Some(0x0012_3456));
    }

    #[test]
    fn shape_and_typo_tokens_resolve() {
        let t = Theme::byard_base();
        assert_eq!(t.shape("cornerLg"), Some(16.0));
        assert_eq!(t.typo_size("titleLarge"), Some(22.0));
        assert!(t.shape("cornerNope").is_none());
    }

    #[test]
    fn set_color_canonicalizes_snake_keys() {
        let mut t = Theme::byard_base();
        t.set_color("light", "primary_container", 0x00AB_CDEF);
        assert_eq!(t.color("primaryContainer", false), Some(0x00AB_CDEF));
    }

    #[test]
    fn default_is_base_light() {
        let t = Theme::default();
        assert!(!t.active_dark);
        assert_eq!(
            t.on_surface(),
            Theme::byard_base().color("onSurface", false).unwrap()
        );
    }

    #[test]
    fn font_weight_parse_and_axis() {
        assert_eq!(FontWeight::parse("medium"), Some(FontWeight::Medium));
        assert_eq!(FontWeight::parse("700"), Some(FontWeight::Bold));
        assert_eq!(FontWeight::parse("nope"), None);
        assert_eq!(FontWeight::Medium.axis(), 500);
    }
}
