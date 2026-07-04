//! `byard` — the Byard UI framework CLI.
//!
//! See RFC-0006 for the full design rationale.

#![allow(clippy::missing_errors_doc)]

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod commands;
mod deps;
mod manifest;
mod telemetry_overlay;

#[derive(Parser)]
#[command(
    name = "byard",
    version,
    about = "The Byard UI framework CLI",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scaffold a new Byard project.
    New {
        /// Name of the project (used as the directory name).
        name: String,
    },
    /// Start the dev window with live hot-reload.
    Dev {
        /// Path to a `.byd` file. Defaults to `entry` in `byard.toml`.
        file: Option<PathBuf>,
    },
    /// Parse and validate without opening a window (CI-friendly).
    Check {
        /// Path to a `.byd` file. Defaults to `entry` in `byard.toml`.
        file: Option<PathBuf>,
    },
    /// (Phase 3+) Compile to a production binary.
    Build,
    /// Add a dependency to byard.toml, then fetch and lock it (RFC-0008).
    #[command(alias = "install")]
    Add {
        /// Package name (a byld identifier, e.g. `material`).
        name: String,
        /// Use a local directory as the package source.
        #[arg(long)]
        path: Option<PathBuf>,
        /// Use a git repository as the package source.
        #[arg(long)]
        git: Option<String>,
        /// Pin the git source to a tag.
        #[arg(long, conflicts_with = "rev")]
        tag: Option<String>,
        /// Pin the git source to an exact commit.
        #[arg(long)]
        rev: Option<String>,
    },
    /// Fetch dependencies and write byard.lock (the only lock writer).
    Get,
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::New { name } => commands::new::run(&name),
        Command::Dev { file } => commands::dev::run(file.as_deref()),
        Command::Check { file } => commands::check::run(file.as_deref()),
        Command::Build => commands::build::run(),
        Command::Add {
            name,
            path,
            git,
            tag,
            rev,
        } => commands::add::run(&commands::add::AddArgs {
            name: &name,
            path: path.as_deref(),
            git: git.as_deref(),
            tag: tag.as_deref(),
            rev: rev.as_deref(),
        }),
        Command::Get => commands::get::run(),
    };
    if let Err(e) = result {
        // An empty message is a silent failure sentinel (e.g. `check` already
        // printed rustc-style diagnostics) — just set the exit code.
        if !e.is_empty() {
            eprintln!("error: {e}");
        }
        std::process::exit(1);
    }
}
