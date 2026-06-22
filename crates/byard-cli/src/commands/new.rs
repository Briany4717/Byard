//! `byard new <name>` — scaffold a new Byard project (RFC-0006 §4, decision C6).

use std::path::{Path, PathBuf};

pub fn run(name: &str) -> Result<(), String> {
    validate_name(name)?;

    let dir = PathBuf::from(name);
    if dir.exists() {
        return Err(format!("directory `{name}` already exists"));
    }

    // Create the directory first so we can track files for atomic rollback (C6).
    std::fs::create_dir(&dir).map_err(|e| format!("cannot create `{name}/`: {e}"))?;

    let mut created: Vec<PathBuf> = vec![];

    let result = write_project_files(&dir, name, &mut created);
    if let Err(e) = result {
        // Atomic rollback: remove everything we touched (C6).
        for f in &created {
            let _ = std::fs::remove_file(f);
        }
        let _ = std::fs::remove_dir(&dir);
        return Err(e);
    }

    println!("  Created {name}/");
    for f in &created {
        println!("  Created {}", f.display());
    }
    println!("\nRun `cd {name} && byard dev` to start.");
    Ok(())
}

fn write_project_files(dir: &Path, name: &str, created: &mut Vec<PathBuf>) -> Result<(), String> {
    let write = |rel: &str, contents: String, created: &mut Vec<PathBuf>| -> Result<(), String> {
        let path = dir.join(rel);
        std::fs::write(&path, contents)
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
        created.push(path);
        Ok(())
    };

    write(
        "byard.toml",
        format!("[project]\nname  = \"{name}\"\nentry = \"main.byd\"\n"),
        created,
    )?;

    write("main.byd", starter_view(name), created)?;

    write(".gitignore", "/target\n".to_string(), created)?;

    Ok(())
}

fn starter_view(name: &str) -> String {
    format!(
        r#"// {name} — starter view
// Edit and save; the window updates instantly.
View Main() {{
    var count = 0
    var label = "hello"

    Column #[gap: 20, p: 32, align: center, justify: center] {{
        Text("{{label}} — tapped {{count}} times") #[size: 24, color: 0xFFFFFF]

        Button("Tap me") #[bg: 0x3B82F6, radius: 8, p: (vertical: 10, horizontal: 20),
                           color: 0xFFFFFF, weight: bold] => count++

        TextField #[bg: 0x374151, radius: 6, p: (horizontal: 12), height: 36,
                    color: 0xFFFFFF, bind: label]
    }}
}}
"#
    )
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("project name cannot be empty".to_string());
    }
    if name.starts_with('.') || name.starts_with('-') {
        return Err(format!(
            "project name `{name}` must not start with `.` or `-`"
        ));
    }
    if name
        .chars()
        .any(|c| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
    {
        return Err(format!(
            "project name `{name}` must contain only ASCII letters, digits, `-`, or `_`"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starter_view_compiles_clean() {
        // RFC-0006 §4 / IMPL-30: the scaffolded view must validate with no errors
        // and use the new `Len` forms (no `px`/`py`).
        let src = starter_view("demo");
        assert!(
            !src.contains("px:") && !src.contains("py:"),
            "starter still uses px/py"
        );
        let errs = crate::commands::check::check_source(&src);
        assert!(errs.is_empty(), "starter view must compile clean: {errs:?}");
    }

    #[test]
    fn name_validation_rejects_bad_names() {
        assert!(validate_name("").is_err());
        assert!(validate_name(".hidden").is_err());
        assert!(validate_name("-dash").is_err());
        assert!(validate_name("has space").is_err());
        assert!(validate_name("good_name-1").is_ok());
    }
}
