# RFC-0022: Theme Runtime & Asset Distribution — Pillar D complete

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-10
- **Last updated:** 2026-07-10
- **Depends on:** RFC-0001 (§3.1 text pipeline, `inject`, two-layer rule), RFC-0002 (D5 style resolution layers, D8 dynamic styles — this lifts the theme slice), RFC-0005 (§6 theme-resolved defaults, `Typo` tokens), RFC-0008 (Pillar D design proposal — this RFC implements it), RFC-0010 (animated token transitions), RFC-0016 (style system — tokens referenced in `style {}`).
- **Completes:** RFC-0008 Pillar D (asset distribution), which was left as a design proposal. Also closes RFC-0005's `Typo` token resolution and `inject`-based theme provision.
- **Enables:** Dark/light mode switching, dynamic color (Material You), custom font loading (Roboto, SF Pro), design-system theme tokens in `byard-material` and `byard-cupertino`.

---

## Summary

Implement the **theme runtime**: a reactive, package-namespaced token system
where design systems declare color, typography, shape, and motion tokens in their
manifest; the engine resolves them at mount time; and the developer references
them in `style {}` blocks. Theme switching (light ↔ dark, dynamic color) is a
single `var` mutation that triggers Mark-and-Pull across all token-dependent
bindings — with animated transitions for every affected property (RFC-0010).

Separately, implement **asset distribution**: packages declare fonts and icon
sets in their manifest; the engine loads, caches, and registers them so that
`typo: m.titleLarge` resolves to a concrete font family + size + weight without
the `byld` author touching a file path.

---

## Motivation

Today, `byard-material` hardcodes hex color values (`0x6750A4`) throughout 29
`.byd` files. There is no dark mode, no dynamic color, no way to switch themes,
and no way to load Roboto (Material's typeface) or SF Pro (Cupertino's). The
README's gap analysis lists this as the #1 systemic blocker:

- **No theme tokens runtime** — colors are literals, not semantic references.
- **No dark scheme** — would require duplicating every component with different
  hex values.
- **No custom fonts** — `Text` uses whatever system font the engine loads.
- **No `Typo` resolution** — `typo: titleLarge` is in the spec but never
  implemented.

RFC-0008 Pillar D described the *model* (packages declare assets, the engine
loads them as policy) but deferred implementation. RFC-0005 §6 described
theme-resolved defaults but noted them as Phase 3. This RFC implements both.

---

## Guide-level explanation

### Theme tokens in a package manifest

```toml
# byard-material/byard.toml
[project]
name = "material"
entry = "src/lib.byd"

[theme]
name = "material3"
extends = "byard-base"  # the engine's built-in default theme

[theme.color.light]
primary = "#6750A4"
on_primary = "#FFFFFF"
primary_container = "#EADDFF"
surface = "#FEF7FF"
on_surface = "#1D1B20"
outline = "#79747E"
error = "#B3261E"
# ... full M3 baseline

[theme.color.dark]
primary = "#D0BCFF"
on_primary = "#381E72"
surface = "#141218"
on_surface = "#E6E0E9"
# ... full M3 dark

[theme.typography]
display_large = { family = "Roboto", size = 57, weight = "regular", tracking = -0.25 }
headline_large = { family = "Roboto", size = 32, weight = "regular" }
title_large = { family = "Roboto", size = 22, weight = "regular" }
body_large = { family = "Roboto", size = 16, weight = "regular", tracking = 0.5 }
label_large = { family = "Roboto", size = 14, weight = "medium", tracking = 0.1 }
# ... full M3 type scale

[theme.shape]
corner_xs = 4
corner_sm = 8
corner_md = 12
corner_lg = 16
corner_xl = 28

[assets.fonts]
roboto_regular = "assets/fonts/Roboto-Regular.ttf"
roboto_medium = "assets/fonts/Roboto-Medium.ttf"
roboto_bold = "assets/fonts/Roboto-Bold.ttf"

[assets.icons]
# Handled by VectorIcon + RFC-0009; listed here for AOT collection
icon_set = "assets/icons/"
```

### Using theme tokens in `byld`

```byld
use material as m

View App() {
    inject m.Theme as theme   // the package's theme provider

    Column #[bg: theme.surface, p: 16, gap: 12] {
        Text("Hello") #[typo: theme.titleLarge, color: theme.onSurface]
        Button("Action") #[bg: theme.primary, color: theme.onPrimary,
                           radius: theme.cornerXl]
    }
}
```

Or, in a `style {}` block:

```byld
let card = style {
    bg: theme.surfaceContainer
    radius: theme.cornerMd
    on hover { bg: theme.surfaceContainerHigh }
}
```

### Theme switching

```byld
View App() {
    var isDark = false
    inject m.Theme(mode: isDark ? "dark" : "light") as theme

    Column #[bg: theme.surface with anim.spring()] {
        Toggle #[bind: isDark]
        Text("Theme: {isDark ? 'Dark' : 'Light'}") #[color: theme.onSurface]
    }
}
```

When `isDark` flips, the `Theme` provider switches its token set. Every binding
that reads a theme token is marked dirty → Mark-and-Pull recomputes → `bg`,
`color`, etc. smoothly transition via their `with anim` declarations. The entire
app recolors in one frame with spring-animated transitions.

### Font loading

Packages declare fonts in `[assets.fonts]`. The engine loads them during
`byard get` (resolved alongside the package), registers them in the text
subsystem (`cosmic-text` / `glyphon`), and makes them available by family name.
A `Typo` token resolving to `family = "Roboto"` renders in Roboto if the package
has loaded it; otherwise falls back to the system font with a
`CompileWarning::FontNotFound`.

---

## Reference-level explanation

### 1. Theme token resolution

A `Theme` is a two-level map:

```rust
struct Theme {
    name: String,
    colors: HashMap<String, ColorScheme>,  // "light", "dark", ...
    typography: HashMap<String, TypoToken>,
    shapes: HashMap<String, f32>,          // corner radii
    active_scheme: String,                 // "light" or "dark"
}

struct ColorScheme {
    tokens: HashMap<String, Color>,  // "primary" → 0x6750A4
}

struct TypoToken {
    family: String,
    size: f32,
    weight: FontWeight,
    tracking: f32,
    line_height: f32,
}
```

Theme is provided via `inject` (RFC-0002). At mount, `theme.primary` resolves
to a `Signal<Color>` whose value is the current scheme's `primary` token. When
the scheme switches, all token signals are written → Mark-and-Pull.

### 2. The provider model

Each package can declare a `ThemeProvider` View:

```byld
// material/src/theme.byd (generated from byard.toml [theme])
View Theme(mode: Str = "light") {
    provide color_primary = mode == "dark" ? 0xD0BCFF : 0x6750A4
    provide color_on_primary = mode == "dark" ? 0x381E72 : 0xFFFFFF
    // ... all tokens as provide bindings
    provide typo_title_large = TypoToken("Roboto", 22, regular, 0, 28)
    // ...
    content  // children receive these via inject
}
```

The `provide` / `inject` mechanism (RFC-0002) already exists. This RFC adds:

1. **Automatic theme generation** from `byard.toml [theme]` → a `Theme` View
   with all `provide` bindings.
2. **Token accessor syntax** — `theme.primary` desugars to `inject color_primary`
   with the package namespace prefix.
3. **Scheme reactivity** — changing `mode` rewrites all `provide` values in one
   tick.

### 3. Asset loading

Fonts declared in `[assets.fonts]` are loaded during dependency resolution:

1. `byard get` / `byard install` copies font files to `~/.byard/cache/<pkg>/`.
2. At app start, the engine scans the resolved dependency graph's asset manifests
   and registers fonts with the text subsystem.
3. `TypoToken.family` references are resolved against registered font families.
4. Missing fonts fall back to the system font with a runtime warning.

This is **engine policy** (RFC-0001 §1, RFC-0008): the `.byd` author never
writes a font path. The manifest declares; the engine loads.

### 4. Typo token resolution

The `typo` prop (RFC-0005) is currently unresolved. This RFC completes it:

- `typo: theme.titleLarge` resolves to a `TypoToken` → sets `size`, `weight`,
  `line_height`, and `font_family` on the `TextGlyph` pipeline.
- Inline overrides win: `#[typo: theme.bodyLarge, size: 18]` uses bodyLarge's
  weight/family but overrides size to 18.
- A bare `typo: titleLarge` (without package prefix) resolves against the
  nearest `inject`ed theme. If ambiguous (two themes provide `titleLarge`),
  it's `CompileError::AmbiguousToken`.

### 5. Dynamic color (Material You)

Material You derives a color scheme from a user-provided seed color. This is
a controller concern:

```rust
#[byard_controller]
impl MaterialTheme {
    fn from_seed(seed: Color) -> ColorScheme {
        // HCT (Hue-Chroma-Tone) algorithm → M3 tonal palettes
        // Returns a full ColorScheme
    }
}
```

The controller computes the scheme; the theme provider receives it as a `var`
and updates all `provide` bindings. The byld side is unchanged.

### 6. `byard-base` — the engine's built-in theme

The engine ships a minimal `byard-base` theme with sensible defaults:

- **Colors:** a neutral gray palette (no brand color).
- **Typography:** system font, M3-compatible size scale.
- **Shapes:** 0/4/8/12/16/28 corner radius scale.

Every intrinsic's default base style (RFC-0005 §6, D5 layer 1) resolves against
`byard-base`. A package theme that `extends = "byard-base"` overrides only the
tokens it redefines; unset tokens fall through.

---

## Drawbacks

- **Manifest complexity.** `byard.toml` grows a `[theme]` and `[assets]` section
  that the CLI and resolver must parse and validate. Malformed themes need clear
  diagnostics.
- **Font loading at startup.** Loading TTF/OTF files adds startup latency. The
  font cache mitigates this for subsequent runs, but first-run cold starts are
  slower. Lazy loading (load only when first referenced) is an optimization.
- **Token explosion.** M3 has ~80 color tokens, ~15 typography tokens, ~6 shape
  tokens. Each is a reactive signal. For a large app with many theme consumers,
  this is ~100 signals marked dirty on a theme switch — all recomputed in one
  tick. Profiling needed.
- **Cross-package theme conflicts.** Two packages defining `primary` → the
  namespace prefix (`m.primary` vs `c.primary`) resolves this, but unaliased
  bare references are ambiguous.

---

## Rationale and alternatives

**Why manifest-declared tokens, not `.byd`-level `let` constants?** `let`
constants (what `byard-material` does today with inline hex) are not reactive,
not switchable, and not accessible to the engine for default resolution. Manifest
tokens are engine-managed reactive signals — the only model that supports live
theme switching.

**Why `provide/inject`, not a global theme object?** Global state conflicts with
RFC-0001's strict dependency graph and makes testing harder. `provide/inject` is
scoped, composable (different subtrees can have different themes), and already
exists.

**Why generate a `Theme` View from the manifest instead of a runtime API?** The
two-layer rule: theme *configuration* is engine policy (manifest); theme *use*
is declarative (`inject` in `.byd`). A runtime API would put theme logic in `.rs`
files — correct by the two-layer rule, but unnecessary for the common case.
The generated View is the ergonomic path; controllers handle dynamic color.

---

## Prior art

- **Flutter `ThemeData` / `Theme` widget:** a context-scoped theme object.
  Byard's `provide/inject` is the same pattern.
- **Material Design token system:** M3's color/typography/shape tokens. Direct
  source for the manifest schema.
- **SwiftUI `@Environment(\.colorScheme)`:** system-level dark/light. Byard
  makes it a package-level concern (Material's dark ≠ Cupertino's dark).
- **CSS custom properties / design tokens:** `--color-primary: #6750A4`. The
  closest web analogue to manifest-declared tokens.
- **Tailwind `theme.extend`:** static config-file tokens compiled into utility
  classes. Byard's are runtime-reactive.

---

## Resolved questions

- **Before merge:**
  - [x] **Token naming convention.** **Auto-map.** TOML manifests use
    `snake_case` (TOML convention): `primary_container`, `title_large`. Byld
    references use `camelCase` (byld convention): `theme.primaryContainer`,
    `theme.titleLarge`. The compiler maps automatically:
    `primary_container` ↔ `primaryContainer` via the standard snake→camel
    conversion. A `CompileError::UnknownThemeToken` with a Levenshtein hint
    fires if neither casing matches. This preserves idiomatic conventions in
    both languages without forcing developers to remember a foreign casing
    rule.
  - [x] **Theme inheritance.** Yes, multi-level chains are supported.
    `A extends B extends byard-base` resolves linearly: A's tokens override
    B's, B's override byard-base's. Resolution is a simple HashMap merge in
    dependency order (deepest ancestor first). Circular `extends` is a
    `CompileError::CircularThemeInheritance`.
  - [x] **Font subsetting.** Deferred. Full TTF/OTF files are loaded in v1.
    Font subsetting (stripping unused glyphs) is a build-time optimization
    that requires glyph usage analysis across all `.byd` files — significant
    toolchain work for marginal memory savings on desktop. Revisit when
    targeting WASM or memory-constrained mobile devices.

- **During implementation:**
  - [x] **Hot-reload of theme tokens.** Yes. When `byard.toml`'s `[theme]`
    section changes during `byard dev`, the engine reloads the token set and
    writes new values to all theme signals → Mark-and-Pull propagates → all
    themed elements update with animated transitions (if `with anim` is on
    the properties). Consistent with Pillar E's hot-reload model.
  - [x] **System dark mode detection.** Yes. The engine exposes the OS
    appearance preference as an injectable:
    ```byld
    inject SystemAppearance as appearance  // "light" | "dark"
    ```
    This is a reactive `Signal<Str>` that updates when the OS switches mode.
    The developer can wire it to the theme: `inject m.Theme(mode: appearance)`.
    On platforms without system dark mode (some Linux DEs), defaults to
    `"light"`.

---

## Future possibilities

- **Color harmonization** — auto-adjusting a package's theme to harmonize with
  another package's colors (Material You's color harmonization API).
- **Animated theme tokens** — springs on token transitions (already supported
  via `with anim` on properties that read tokens; this is about making the
  *token transition itself* tunable).
- **Typography responsive variants** — different type scales for phone/tablet/
  desktop, selected by viewport size.
- **Icon theme** — a theme-level color applied to all `VectorIcon`s from a
  package (e.g., `theme.iconColor` that all icons inherit unless overridden).
