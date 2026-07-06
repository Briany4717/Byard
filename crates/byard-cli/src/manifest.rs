//! `byard.toml` discovery and parsing (RFC-0006 §2, decision C1/C2;
//! RFC-0008 Pillar C — the `[dependencies]` table).
//!
//! Dependency entries are parsed **strictly**: a malformed entry is an error,
//! never a warning-and-drop (RFC-0008 explicitly reverses the C2
//! warn-on-unknown policy for this table — silently ignoring a dependency is
//! a reproducibility hazard).

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
                "project" | "dependencies" | "package" | "assets"
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

        Ok(Self {
            project_root,
            entry,
            name,
            dependencies,
            single_file: false,
            vector_includes,
        })
    }
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
}
