//! `byard clean [file]` — remove generated build artifacts and caches
//! (RFC-0009 §5, M52): the persistent vector-field cache and the AOT bake
//! output under `.byard/`. Lock-pinned package checkouts are left alone (they
//! are immutable and expensive to re-fetch).

use std::path::Path;

use crate::manifest::Manifest;

pub fn run(file: Option<&Path>) -> Result<(), String> {
    // Locate the project's `.byard/`; fall back to the current directory so a
    // broken/absent manifest still lets you clean.
    let root = Manifest::discover(file).map_or_else(
        |_| std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf()),
        |m| m.project_root,
    );
    let byard = root.join(".byard");

    let mut removed = 0usize;
    for sub in ["cache", "generated"] {
        let dir = byard.join(sub);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            println!("  removed {}", dir.display());
            removed += 1;
        }
    }
    if removed == 0 {
        println!("  nothing to clean");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_removes_cache_and_generated_but_not_the_manifest() {
        let dir = std::env::temp_dir().join(format!("byard_clean_it_{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".byard/cache/vectors")).unwrap();
        std::fs::create_dir_all(dir.join(".byard/generated")).unwrap();
        std::fs::write(dir.join(".byard/cache/vectors/x.msdf"), b"stale").unwrap();
        std::fs::write(dir.join("byard.toml"), "[project]\nname = \"t\"\n").unwrap();
        std::fs::write(dir.join("main.byd"), "View Main() { Text(\"hi\") }").unwrap();

        run(Some(&dir)).expect("clean must succeed");

        assert!(!dir.join(".byard/cache").exists(), "the cache must be gone");
        assert!(
            !dir.join(".byard/generated").exists(),
            "generated must be gone"
        );
        assert!(dir.join("byard.toml").exists(), "source must be untouched");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
