//! `byard build` — Phase 3+ stub (RFC-0006 §1).

// The Result return type matches the other commands' signature so main.rs can
// dispatch uniformly; clippy's unnecessary_wraps warning is intentionally suppressed.
#[allow(clippy::unnecessary_wraps)]
pub fn run() -> Result<(), String> {
    eprintln!("byard build is not yet available (Phase 3+).");
    eprintln!("Track progress at https://github.com/Briany4717/byard/issues");
    Ok(())
}
