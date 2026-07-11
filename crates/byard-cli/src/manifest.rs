//! `byard.toml` discovery and parsing (RFC-0006 §2, decision C1/C2;
//! RFC-0008 Pillar C — the `[dependencies]` table).
//!
//! Dependency entries are parsed **strictly**: a malformed entry is an error,
//! never a warning-and-drop (RFC-0008 explicitly reverses the C2
//! warn-on-unknown policy for this table — silently ignoring a dependency is
//! a reproducibility hazard).

use byard_compiler::interp::theme::{FontWeight, Theme, TypoToken};
use std::path::{Path, PathBuf};

/// How a dependency is acquired (RFC-0008 D-H: git + path first; a hosted
/// registry is deferred and layers onto the same lockfile later).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DepSource {
    /// A local package directory, relative to the declaring manifest.
    Path(PathBuf),
    /// A git repository pinned to an exact ref (`rev` or `tag` — D-H requires
    /// a pin; floating branches are not reproducible).
    Git {
        /// Clone URL.
        url: String,
        /// The pinned ref.
        reference: GitRef,
    },
}

/// The pin of a git dependency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GitRef {
    /// An exact commit hash.
    Rev(String),
    /// A tag name.
    Tag(String),
}

impl GitRef {
    /// The ref as written (for display and lockfile source strings).
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Rev(s) | Self::Tag(s) => s,
        }
    }
}

/// One `[dependencies]` entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dependency {
    /// The package name (the `use` name in `byld` source).
    pub name: String,
    /// Where it comes from.
    pub source: DepSource,
}

/// Parsed project manifest (or a synthetic one for bare-file usage).
pub struct Manifest {
    pub project_root: PathBuf,
    /// Absolute path to the `.byd` entry file.
    pub entry: PathBuf,
    pub name: String,
    /// Declared dependencies (RFC-0008 Pillar C). Empty for bare-file usage.
    pub dependencies: Vec<Dependency>,
    /// True when this is a lone `.byd` entry (`byard check foo.byd`), not a
    /// project: only the entry file is compiled, sibling `.byd`s are ignored.
    /// A real project (a `byard.toml`) treats every sibling as one namespace.
    pub single_file: bool,
    /// `[assets.vectors] include` — the RFC-0009 §4 escape hatch: handles the
    /// AOT packer must bake even though no `VectorIcon("literal")` names them
    /// (e.g. a `VectorIcon(someVar)` resolved at runtime). Empty by default.
    pub vector_includes: Vec<String>,
    /// The resolved design-token theme (RFC-0022): the built-in `byard-base`
    /// with any `[theme]` / `[assets.fonts]` declarations from `byard.toml`
    /// layered on top. Bare-file usage gets `byard-base` unchanged.
    pub theme: Theme,
}

impl Manifest {
    /// Discover the manifest by walking up from `CWD`, or fall back to
    /// `main.byd` in the current directory (C1).  If `override_path` is given,
    /// it is used as the entry file directly (no manifest required).
    pub fn discover(override_path: Option<&Path>) -> Result<Self, String> {
        if let Some(p) = override_path {
            let p = p
                .canonicalize()
                .map_err(|e| format!("{}: {e}", p.display()))?;
            // Pointing at a project (a directory, or its `byard.toml`) reads the
            // manifest — including `[dependencies]` — rather than treating the
            // path as a lone entry file. Only a `.byd` path is a bare entry.
            if p.is_dir() {
                let manifest = p.join("byard.toml");
                if manifest.exists() {
                    return Self::from_toml(&manifest);
                }
                let main = p.join("main.byd");
                if main.exists() {
                    return Ok(Self::bare_entry(main));
                }
                return Err(format!("`{}` has no byard.toml or main.byd", p.display()));
            }
            if p.file_name().is_some_and(|n| n == "byard.toml") {
                return Self::from_toml(&p);
            }
            return Ok(Self::bare_entry(p));
        }

        let cwd = std::env::current_dir().map_err(|e| e.to_string())?;

        // Walk upward looking for byard.toml (C1).
        let mut dir = cwd.as_path();
        loop {
            let candidate = dir.join("byard.toml");
            if candidate.exists() {
                return Self::from_toml(&candidate);
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }

        // CWD fallback: use main.byd if it exists.
        let fallback = cwd.join("main.byd");
        if fallback.exists() {
            let name = cwd
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("byard-project")
                .to_string();
            return Ok(Self {
                project_root: cwd.clone(),
                entry: fallback,
                name,
                dependencies: Vec::new(),
                single_file: false,
                vector_includes: Vec::new(),
                theme: Theme::byard_base(),
            });
        }

        Err("no byard.toml found and no main.byd in current directory\n\
             hint: run `byard new <name>` to create a project, or pass a file path"
            .to_string())
    }

    /// A manifest for a lone `.byd` entry file with no `[dependencies]` — the
    /// bare single-file path (`byard check foo.byd`). `entry` is assumed to
    /// already be canonicalized.
    fn bare_entry(entry: PathBuf) -> Self {
        let project_root = entry.parent().unwrap_or(Path::new(".")).to_path_buf();
        let name = project_root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("byard-project")
            .to_string();
        Self {
            project_root,
            entry,
            name,
            dependencies: Vec::new(),
            single_file: true,
            vector_includes: Vec::new(),
            theme: Theme::byard_base(),
        }
    }

    fn from_toml(path: &Path) -> Result<Self, String> {
        let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let table: toml::Table = src.parse().map_err(|e: toml::de::Error| e.to_string())?;

        let project_root = path.parent().unwrap_or(Path::new(".")).to_path_buf();

        let project = table.get("project");

        // Warn on unknown top-level keys (C2: forward-compatible) — except
        // `[dependencies]`, which is parsed strictly below, `[package]`,
        // reserved for package manifests, and `[assets]` (RFC-0009 §4).
        for key in table.keys() {
            if !matches!(
                key.as_str(),
                "project" | "dependencies" | "package" | "assets" | "theme"
            ) {
                eprintln!("byard.toml: warning: unknown key `{key}` (ignored)");
            }
        }

        // `[assets.vectors] include = ["a.svg", ...]` — the AOT escape hatch.
        let vector_includes = table
            .get("assets")
            .and_then(|a| a.get("vectors"))
            .and_then(|v| v.get("include"))
            .and_then(toml::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        let name = project
            .and_then(|p| p.get("name"))
            .and_then(|v| v.as_str())
            .map_or_else(
                || {
                    project_root
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("byard-project")
                        .to_string()
                },
                str::to_string,
            );

        let entry_rel = project
            .and_then(|p| p.get("entry"))
            .and_then(|v| v.as_str())
            .unwrap_or("main.byd");

        let entry = project_root.join(entry_rel);
        if !entry.exists() {
            return Err(format!(
                "entry file `{}` not found (set in byard.toml [project].entry)",
                entry.display()
            ));
        }

        let dependencies = match table.get("dependencies") {
            Some(deps) => parse_dependencies(deps)?,
            None => Vec::new(),
        };

        // RFC-0022: `[theme]` tokens + `[assets.fonts]` layer onto `byard-base`.
        let theme = parse_theme(&table)?;

        Ok(Self {
            project_root,
            entry,
            name,
            dependencies,
            single_file: false,
            vector_includes,
            theme,
        })
    }
}

/// Parses the `[theme]` table and `[assets.fonts]` into a [`Theme`] layered over
/// the built-in `byard-base` (RFC-0022). Every malformed token is an **error**
/// (a silently-dropped theme token is a hard-to-debug visual regression); the
/// message names the exact `token`/scheme so the fix is obvious.
///
/// Tokens are declared `snake_case` here and canonicalized to the `camelCase`
/// byld reference form by [`Theme::set_color`] et al.
fn parse_theme(table: &toml::Table) -> Result<Theme, String> {
    let mut theme = Theme::byard_base();

    if let Some(theme_tbl) = table.get("theme").and_then(toml::Value::as_table) {
        if let Some(name) = theme_tbl.get("name").and_then(toml::Value::as_str) {
            theme.name = name.to_string();
        }
        // `extends` beyond the built-in `byard-base` (multi-level, cross-package)
        // is deferred (RFC-0022 unresolved question); anything else is accepted
        // and simply layers onto `byard-base`, the only built-in base today.

        // [theme.color.<scheme>] — a table of `token = "#RRGGBB"`.
        if let Some(colors) = theme_tbl.get("color").and_then(toml::Value::as_table) {
            for (scheme, tokens) in colors {
                let tokens = tokens.as_table().ok_or_else(|| {
                    format!("byard.toml: [theme.color.{scheme}] must be a table of `token = \"#RRGGBB\"`")
                })?;
                for (token, value) in tokens {
                    let hex = value.as_str().ok_or_else(|| {
                        format!("byard.toml: theme color `{scheme}.{token}` must be a string like \"#6750A4\"")
                    })?;
                    let rgb = parse_hex_color(hex).ok_or_else(|| {
                        format!("byard.toml: theme color `{scheme}.{token} = {hex:?}` is not a valid `#RRGGBB` / `#RGB` hex color")
                    })?;
                    theme.set_color(scheme, token, rgb);
                }
            }
        }

        // [theme.typography] — `token = { size, family?, weight?, tracking?, line_height? }`.
        if let Some(typo) = theme_tbl.get("typography").and_then(toml::Value::as_table) {
            for (token, value) in typo {
                theme.set_typo(token, parse_typo_token(token, value)?);
            }
        }

        // [theme.shape] — `token = <radius>`.
        if let Some(shapes) = theme_tbl.get("shape").and_then(toml::Value::as_table) {
            for (token, value) in shapes {
                let radius = as_number(value).ok_or_else(|| {
                    format!("byard.toml: theme shape `{token}` must be a number (corner radius)")
                })?;
                #[allow(clippy::cast_possible_truncation)]
                theme.set_shape(token, radius as f32);
            }
        }
    }

    // [assets.fonts] — `family = "path/to/font.ttf"` (RFC-0022 §3). The bytes are
    // not loaded yet (deferred); declaring a family here makes it resolvable by
    // `TypoToken.family` and suppresses the `FontNotFound` fallback warning.
    if let Some(fonts) = table
        .get("assets")
        .and_then(|a| a.get("fonts"))
        .and_then(toml::Value::as_table)
    {
        for (family, path) in fonts {
            let path = path.as_str().ok_or_else(|| {
                format!(
                    "byard.toml: [assets.fonts] `{family}` must be a string path to a font file"
                )
            })?;
            theme.add_font(family.clone(), path.to_string());
        }
    }

    Ok(theme)
}

/// Parses one `[theme.typography]` entry: an inline table `{ size, family?,
/// weight?, tracking?, line_height? }`. `size` is required.
fn parse_typo_token(token: &str, value: &toml::Value) -> Result<TypoToken, String> {
    let tbl = value.as_table().ok_or_else(|| {
        format!("byard.toml: typography token `{token}` must be a table like `{{ size = 22, weight = \"regular\" }}`")
    })?;
    for key in tbl.keys() {
        if !matches!(
            key.as_str(),
            "size" | "family" | "weight" | "tracking" | "line_height"
        ) {
            return Err(format!(
                "byard.toml: typography token `{token}`: unknown field `{key}`"
            ));
        }
    }
    let size = tbl
        .get("size")
        .and_then(as_number)
        .ok_or_else(|| format!("byard.toml: typography token `{token}` needs a numeric `size`"))?;
    #[allow(clippy::cast_possible_truncation)]
    let size = size as f32;
    let family = tbl
        .get("family")
        .and_then(toml::Value::as_str)
        .map(str::to_string);
    let weight = match tbl.get("weight") {
        Some(v) => {
            let s = v.as_str().ok_or_else(|| {
                format!("byard.toml: typography token `{token}`: `weight` must be a string")
            })?;
            FontWeight::parse(s).ok_or_else(|| {
                format!("byard.toml: typography token `{token}`: unknown weight `{s}` (use thin/light/regular/medium/semibold/bold/black)")
            })?
        }
        None => FontWeight::Regular,
    };
    #[allow(clippy::cast_possible_truncation)]
    let tracking = tbl.get("tracking").and_then(as_number).unwrap_or(0.0) as f32;
    #[allow(clippy::cast_possible_truncation)]
    let line_height = tbl
        .get("line_height")
        .and_then(as_number)
        .map_or_else(|| (size * 1.25).round(), |lh| lh as f32);
    Ok(TypoToken {
        family,
        size,
        weight,
        tracking,
        line_height,
    })
}

/// Reads a TOML value as a number, accepting either a float or an integer
/// (TOML distinguishes `12` from `12.0`; a token radius/size may be written
/// either way).
fn as_number(value: &toml::Value) -> Option<f64> {
    #[allow(clippy::cast_precision_loss)]
    value
        .as_float()
        .or_else(|| value.as_integer().map(|i| i as f64))
}

/// Parses a CSS-style hex color (`#RRGGBB` or `#RGB`, `#` optional) to a packed
/// `0x00RRGGBB` `i64`. Returns `None` for any malformed input.
fn parse_hex_color(s: &str) -> Option<i64> {
    let h = s.strip_prefix('#').unwrap_or(s);
    let expanded = match h.len() {
        // `#RGB` shorthand → duplicate each nibble.
        3 => h.chars().flat_map(|c| [c, c]).collect::<String>(),
        6 => h.to_string(),
        _ => return None,
    };
    i64::from_str_radix(&expanded, 16).ok()
}

/// Parses a `[dependencies]` table (RFC-0008 Pillar C). Every malformed entry
/// is an **error** — this table is never warn-and-ignore.
pub fn parse_dependencies(deps: &toml::Value) -> Result<Vec<Dependency>, String> {
    let table = deps.as_table().ok_or("`[dependencies]` must be a table")?;

    let mut out = Vec::with_capacity(table.len());
    for (name, value) in table {
        out.push(parse_dependency(name, value)?);
    }
    Ok(out)
}

fn parse_dependency(name: &str, value: &toml::Value) -> Result<Dependency, String> {
    let err = |msg: &str| format!("byard.toml: dependency `{name}`: {msg}");

    if value.as_str().is_some() {
        return Err(err(
            "bare version strings need a package registry, which is deferred (RFC-0008 D-H); \
             use `{ git = \"…\", tag|rev = \"…\" }` or `{ path = \"…\" }`",
        ));
    }
    let spec = value.as_table().ok_or_else(|| {
        err("must be a table like `{ path = \"…\" }` or `{ git = \"…\", tag = \"…\" }`")
    })?;

    for key in spec.keys() {
        if !matches!(key.as_str(), "path" | "git" | "rev" | "tag") {
            return Err(err(&format!("unknown key `{key}`")));
        }
    }

    let path = spec.get("path").map(|v| {
        v.as_str()
            .map(PathBuf::from)
            .ok_or_else(|| err("`path` must be a string"))
    });
    let git = spec.get("git").map(|v| {
        v.as_str()
            .map(str::to_string)
            .ok_or_else(|| err("`git` must be a string"))
    });

    match (path, git) {
        (Some(_), Some(_)) => Err(err("`path` and `git` are mutually exclusive")),
        (Some(path), None) => {
            if spec.contains_key("rev") || spec.contains_key("tag") {
                return Err(err("`rev`/`tag` only apply to `git` sources"));
            }
            Ok(Dependency {
                name: name.to_string(),
                source: DepSource::Path(path?),
            })
        }
        (None, Some(url)) => {
            let rev = spec.get("rev").map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| err("`rev` must be a string"))
            });
            let tag = spec.get("tag").map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| err("`tag` must be a string"))
            });
            let reference = match (rev, tag) {
                (Some(_), Some(_)) => return Err(err("`rev` and `tag` are mutually exclusive")),
                (Some(rev), None) => GitRef::Rev(rev?),
                (None, Some(tag)) => GitRef::Tag(tag?),
                (None, None) => {
                    return Err(err(
                        "a git dependency must pin `rev = \"…\"` or `tag = \"…\"` (D-H: \
                         floating refs are not reproducible)",
                    ));
                }
            };
            Ok(Dependency {
                name: name.to_string(),
                source: DepSource::Git {
                    url: url?,
                    reference,
                },
            })
        }
        (None, None) => Err(err("needs a `path` or a `git` source")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deps(src: &str) -> Result<Vec<Dependency>, String> {
        let table: toml::Table = src.parse().unwrap();
        parse_dependencies(table.get("dependencies").unwrap())
    }

    #[test]
    fn path_and_pinned_git_dependencies_parse() {
        let parsed = deps(
            "[dependencies]\n\
             local = { path = \"../kit\" }\n\
             mat = { git = \"https://example.com/mat\", tag = \"v1.0.0\" }\n\
             exact = { git = \"https://example.com/x\", rev = \"abc123\" }\n",
        )
        .unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].source, DepSource::Path(PathBuf::from("../kit")));
        assert!(matches!(
            &parsed[2].source,
            DepSource::Git { reference: GitRef::Rev(r), .. } if r == "abc123"
        ));
    }

    #[test]
    fn unpinned_git_is_an_error_not_a_warning() {
        let err =
            deps("[dependencies]\nmat = { git = \"https://example.com/mat\" }\n").unwrap_err();
        assert!(err.contains("rev") && err.contains("tag"), "{err}");
    }

    #[test]
    fn bare_version_string_points_at_the_deferred_registry() {
        let err = deps("[dependencies]\nmat = \"1.0\"\n").unwrap_err();
        assert!(err.contains("registry"), "{err}");
    }

    #[test]
    fn unknown_dependency_key_is_an_error() {
        let err = deps("[dependencies]\nmat = { path = \"x\", branch = \"main\" }\n").unwrap_err();
        assert!(err.contains("unknown key `branch`"), "{err}");
    }

    #[test]
    fn path_plus_git_is_rejected() {
        let err =
            deps("[dependencies]\nmat = { path = \"x\", git = \"https://e.com\" }\n").unwrap_err();
        assert!(err.contains("mutually exclusive"), "{err}");
    }

    // ── RFC-0022: [theme] / [assets.fonts] parsing ────────────────────────

    fn theme_of(src: &str) -> Result<Theme, String> {
        let table: toml::Table = src.parse().unwrap();
        parse_theme(&table)
    }

    #[test]
    fn hex_colors_parse_full_and_shorthand() {
        assert_eq!(parse_hex_color("#6750A4"), Some(0x0067_50A4));
        assert_eq!(parse_hex_color("6750A4"), Some(0x0067_50A4));
        assert_eq!(parse_hex_color("#FFF"), Some(0x00FF_FFFF));
        assert_eq!(parse_hex_color("#12"), None);
        assert_eq!(parse_hex_color("#GGGGGG"), None);
    }

    #[test]
    fn theme_colors_map_snake_to_camel_and_layer_on_base() {
        let theme = theme_of(
            "[theme]\nname = \"demo\"\n\
             [theme.color.light]\nprimary = \"#6750A4\"\nprimary_container = \"#EADDFF\"\n\
             [theme.color.dark]\nprimary = \"#D0BCFF\"\n",
        )
        .unwrap();
        assert_eq!(theme.name, "demo");
        assert_eq!(theme.color("primary", false), Some(0x0067_50A4));
        assert_eq!(theme.color("primaryContainer", false), Some(0x00EA_DDFF));
        assert_eq!(theme.color("primary", true), Some(0x00D0_BCFF));
        // An untouched base token still resolves (layering, not replacement).
        assert!(theme.color("onSurface", false).is_some());
    }

    #[test]
    fn malformed_theme_color_is_an_error_not_a_drop() {
        let err = theme_of("[theme.color.light]\nprimary = \"not-a-hex\"\n").unwrap_err();
        assert!(err.contains("primary") && err.contains("hex"), "{err}");
    }

    #[test]
    fn typography_token_parses_size_weight_and_defaults() {
        let theme = theme_of(
            "[theme.typography]\n\
             title_large = { size = 22, weight = \"medium\", tracking = 0.1 }\n",
        )
        .unwrap();
        let tok = theme.typo("titleLarge").unwrap();
        assert!((tok.size - 22.0).abs() < f32::EPSILON);
        assert_eq!(tok.weight, FontWeight::Medium);
        assert!((tok.tracking - 0.1).abs() < 1e-6);
        // Derived line height ≈ 1.25× size when unset.
        assert!((tok.line_height - 28.0).abs() < f32::EPSILON);
    }

    #[test]
    fn typography_without_size_is_an_error() {
        let err =
            theme_of("[theme.typography]\ntitle_large = { weight = \"regular\" }\n").unwrap_err();
        assert!(err.contains("size"), "{err}");
    }

    #[test]
    fn typography_unknown_weight_is_an_error() {
        let err = theme_of("[theme.typography]\ntitle_large = { size = 22, weight = \"ultra\" }\n")
            .unwrap_err();
        assert!(err.contains("weight") && err.contains("ultra"), "{err}");
    }

    #[test]
    fn shape_tokens_parse_and_canonicalize() {
        let theme = theme_of("[theme.shape]\ncorner_lg = 16\ncorner_xl = 28\n").unwrap();
        assert_eq!(theme.shape("cornerLg"), Some(16.0));
        assert_eq!(theme.shape("cornerXl"), Some(28.0));
    }

    #[test]
    fn font_assets_register_families() {
        let theme =
            theme_of("[assets.fonts]\nroboto = \"assets/fonts/Roboto-Regular.ttf\"\n").unwrap();
        assert!(theme.has_font("roboto"));
        assert!(!theme.has_font("sfpro"));
    }

    #[test]
    fn no_theme_section_yields_byard_base() {
        let theme = theme_of("[project]\nname = \"x\"\n").unwrap();
        assert_eq!(theme.name, "byard-base");
        assert!(theme.color("primary", false).is_some());
    }
}
