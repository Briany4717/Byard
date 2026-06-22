//! `byard.toml` discovery and parsing (RFC-0006 §2, decision C1/C2).

use std::path::{Path, PathBuf};

/// Parsed project manifest (or a synthetic one for bare-file usage).
pub struct Manifest {
    #[allow(dead_code)]
    pub project_root: PathBuf,
    /// Absolute path to the `.byd` entry file.
    pub entry: PathBuf,
    pub name: String,
}

impl Manifest {
    /// Discover the manifest by walking up from `CWD`, or fall back to
    /// `main.byd` in the current directory (C1).  If `override_path` is given,
    /// it is used as the entry file directly (no manifest required).
    pub fn discover(override_path: Option<&Path>) -> Result<Self, String> {
        if let Some(p) = override_path {
            let entry = p
                .canonicalize()
                .map_err(|e| format!("{}: {e}", p.display()))?;
            let project_root = entry.parent().unwrap_or(Path::new(".")).to_path_buf();
            let name = project_root
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("byard-project")
                .to_string();
            return Ok(Self {
                project_root,
                entry,
                name,
            });
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
            });
        }

        Err("no byard.toml found and no main.byd in current directory\n\
             hint: run `byard new <name>` to create a project, or pass a file path"
            .to_string())
    }

    fn from_toml(path: &Path) -> Result<Self, String> {
        let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let table: toml::Table = src.parse().map_err(|e: toml::de::Error| e.to_string())?;

        let project_root = path.parent().unwrap_or(Path::new(".")).to_path_buf();

        let project = table.get("project");

        // Warn on unknown top-level keys (C2: forward-compatible).
        for key in table.keys() {
            if key != "project" {
                eprintln!("byard.toml: warning: unknown key `{key}` (ignored)");
            }
        }

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

        Ok(Self {
            project_root,
            entry,
            name,
        })
    }
}
