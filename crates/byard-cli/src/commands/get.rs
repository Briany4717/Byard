//! `byard get` — fetch dependencies and write `byard.lock`
//! (RFC-0008 Pillar C, decisions D-H/D-I).
//!
//! The **only** command that may change the lockfile. Walks the dependency
//! graph transitively (path deps in place, git deps into `~/.byard/cache`
//! pinned by `rev`/`tag`), content-hashes every package, and writes the pins.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::deps::{
    LockedPackage, Lockfile, dep_root, fetch_git, git_cache_path, package_checksum, source_string,
};
use crate::manifest::{DepSource, Dependency, Manifest, parse_dependencies};

pub fn run() -> Result<(), String> {
    let manifest = Manifest::discover(None)?;
    println!("Resolving dependencies of `{}`…", manifest.name);

    let mut locked: Vec<LockedPackage> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut queue: Vec<(PathBuf, Vec<Dependency>)> =
        vec![(manifest.project_root.clone(), manifest.dependencies.clone())];

    while let Some((declarer_root, deps)) = queue.pop() {
        for dep in deps {
            if !seen.insert(dep.name.clone()) {
                continue; // first resolution wins (flat namespace, RFC-0008)
            }
            let (root, commit) = ensure_present(&declarer_root, &dep)?;
            let checksum = package_checksum(&root)?;
            println!(
                "  {} {} ({})",
                if commit.is_empty() { "path" } else { "git " },
                dep.name,
                checksum.split(':').nth(1).map_or("", |h| &h[..12])
            );
            locked.push(LockedPackage {
                name: dep.name.clone(),
                source: source_string(&dep),
                commit,
                checksum,
            });
            queue.push((root.clone(), package_deps(&root)?));
        }
    }

    let n = locked.len();
    Lockfile { packages: locked }.write(&manifest.project_root)?;
    println!(
        "  Locked {n} package{} → byard.lock",
        if n == 1 { "" } else { "s" }
    );
    Ok(())
}

/// Makes sure a dependency is on disk (fetching git sources into the cache),
/// returning its root and the exact commit (empty for path deps).
fn ensure_present(declarer_root: &Path, dep: &Dependency) -> Result<(PathBuf, String), String> {
    match &dep.source {
        DepSource::Path(_) => Ok((dep_root(declarer_root, dep)?, String::new())),
        DepSource::Git { url, reference } => {
            let dest = git_cache_path(&dep.name, url, reference);
            let commit = if dest.is_dir() {
                // Already cached — pinned refs are immutable, no refetch.
                let out = std::process::Command::new("git")
                    .args(["rev-parse", "HEAD"])
                    .current_dir(&dest)
                    .output()
                    .map_err(|e| format!("failed to run git: {e}"))?;
                String::from_utf8_lossy(&out.stdout).trim().to_string()
            } else {
                println!("  Fetching {} ({url}#{})…", dep.name, reference.as_str());
                fetch_git(url, reference, &dest)?
            };
            Ok((dest, commit))
        }
    }
}

/// The `[dependencies]` of a fetched package's own manifest, if any.
fn package_deps(root: &Path) -> Result<Vec<Dependency>, String> {
    let manifest_path = root.join("byard.toml");
    if !manifest_path.exists() {
        return Ok(Vec::new());
    }
    let src = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let table: toml::Table = src
        .parse()
        .map_err(|e: toml::de::Error| format!("{}: {e}", manifest_path.display()))?;
    match table.get("dependencies") {
        Some(d) => parse_dependencies(d),
        None => Ok(Vec::new()),
    }
}
