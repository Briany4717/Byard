//! `byard add <name>` (alias: `byard install`) — record a dependency in
//! `byard.toml`, then fetch and lock it (RFC-0008 Pillar C).
//!
//! Sources, in order of preference:
//!
//! - `byard add kit --path ../kit` — a local package (cooperative dev);
//! - `byard add material --git <url> --tag v0.1.0` (or `--rev <hash>`) —
//!   a pinned git source (D-H);
//! - `byard add material` — bare names resolve through a small **built-in
//!   index** of well-known packages, a deliberate stopgap until the hosted
//!   registry (deferred by D-H) exists. The resolved git source is written
//!   into `byard.toml` explicitly, so the manifest never depends on the index.

use std::path::Path;

use crate::manifest::Manifest;

/// Well-known package names → git URL. The stopgap registry (see module docs).
const BUILTIN_INDEX: &[(&str, &str)] = &[(
    "material",
    "https://github.com/byard-framework/byard-material",
)];

pub struct AddArgs<'a> {
    pub name: &'a str,
    pub path: Option<&'a Path>,
    pub git: Option<&'a str>,
    pub tag: Option<&'a str>,
    pub rev: Option<&'a str>,
}

pub fn run(args: &AddArgs) -> Result<(), String> {
    let manifest = Manifest::discover(None)?;
    let manifest_path = manifest.project_root.join("byard.toml");
    if !manifest_path.exists() {
        return Err(
            "`byard add` needs a byard.toml (bare-file projects have no dependencies)\n\
             hint: run `byard new <name>` to create a project"
                .to_string(),
        );
    }

    validate_dep_name(args.name)?;

    // ── Decide the source ─────────────────────────────────────────────────────
    let entry_toml = match (args.path, args.git) {
        (Some(_), Some(_)) => return Err("`--path` and `--git` are mutually exclusive".into()),
        (Some(path), None) => {
            if args.tag.is_some() || args.rev.is_some() {
                return Err("`--tag`/`--rev` only apply to `--git` sources".into());
            }
            format!("{{ path = {:?} }}", path.to_string_lossy())
        }
        (None, Some(url)) => git_entry(url, args.tag, args.rev)?,
        (None, None) => {
            let Some((_, url)) = BUILTIN_INDEX.iter().find(|(n, _)| *n == args.name) else {
                return Err(format!(
                    "`{}` is not in the built-in index and no source was given\n\
                     hint: byard add {} --path <dir>\n\
                     hint: byard add {} --git <url> --tag <tag>",
                    args.name, args.name, args.name
                ));
            };
            println!("  `{}` resolved via the built-in index → {url}", args.name);
            git_entry(url, args.tag, args.rev)?
        }
    };

    // ── Edit byard.toml preserving formatting ─────────────────────────────────
    let src = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let mut doc: toml_edit::DocumentMut = src
        .parse()
        .map_err(|e: toml_edit::TomlError| format!("byard.toml: {e}"))?;

    let dep_value: toml_edit::DocumentMut = format!("dep = {entry_toml}")
        .parse()
        .map_err(|e: toml_edit::TomlError| format!("internal: {e}"))?;
    let dep_item = dep_value["dep"].clone();

    if doc.get("dependencies").is_none() {
        doc["dependencies"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    let already = doc["dependencies"].get(args.name).is_some();
    doc["dependencies"][args.name] = dep_item;

    std::fs::write(&manifest_path, doc.to_string())
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    println!(
        "  {} `{}` in byard.toml → {entry_toml}",
        if already { "Updated" } else { "Added" },
        args.name
    );

    // ── Fetch + lock (the `get` pass) ─────────────────────────────────────────
    super::get::run()
}

fn git_entry(url: &str, tag: Option<&str>, rev: Option<&str>) -> Result<String, String> {
    match (tag, rev) {
        (Some(_), Some(_)) => Err("`--tag` and `--rev` are mutually exclusive".into()),
        (Some(tag), None) => Ok(format!("{{ git = {url:?}, tag = {tag:?} }}")),
        (None, Some(rev)) => Ok(format!("{{ git = {url:?}, rev = {rev:?} }}")),
        (None, None) => {
            // No pin given: resolve the remote's default-branch HEAD to an
            // exact commit *now*, and record that (D-H: never a floating ref).
            let out = std::process::Command::new("git")
                .args(["ls-remote", url, "HEAD"])
                .output()
                .map_err(|e| format!("failed to run git: {e}"))?;
            if !out.status.success() {
                return Err(format!(
                    "git ls-remote {url} failed:\n{}\n\
                     hint: pin explicitly with `--tag <tag>` or `--rev <hash>`",
                    String::from_utf8_lossy(&out.stderr)
                ));
            }
            let stdout = String::from_utf8_lossy(&out.stdout);
            let rev = stdout
                .split_whitespace()
                .next()
                .ok_or("git ls-remote returned nothing; pin with `--tag`/`--rev`")?;
            println!("  Pinned to current HEAD {rev} (D-H: refs are always exact)");
            Ok(format!("{{ git = {url:?}, rev = {rev:?} }}"))
        }
    }
}

fn validate_dep_name(name: &str) -> Result<(), String> {
    let ok = !name.is_empty()
        && name.chars().next().is_some_and(|c| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if ok {
        Ok(())
    } else {
        Err(format!(
            "invalid package name `{name}`: lowercase identifiers only \
             ([a-z][a-z0-9_]*), so it can be written after `use` in byld"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dep_names_must_be_byld_identifiers() {
        assert!(validate_dep_name("material").is_ok());
        assert!(validate_dep_name("liquid_glass2").is_ok());
        assert!(validate_dep_name("Material").is_err());
        assert!(validate_dep_name("my-kit").is_err());
        assert!(validate_dep_name("").is_err());
    }

    #[test]
    fn git_entry_requires_exclusive_pins() {
        assert!(git_entry("https://e.com", Some("v1"), Some("abc")).is_err());
        assert_eq!(
            git_entry("https://e.com", Some("v1"), None).unwrap(),
            "{ git = \"https://e.com\", tag = \"v1\" }"
        );
    }
}
