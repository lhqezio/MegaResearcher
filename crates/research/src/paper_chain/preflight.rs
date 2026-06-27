//! Pre-flight checks for the paper-drafting chain.
//! 1:1 port of `lib/paper_chain/preflight.py`. The chain runs only when:
//!   1. `output.md` exists at the run root (synthesist produced it)
//!   2. `swarm-state.yaml` exists at the run root
//!   3. the run's `novelty_target` is `hypothesis`
//!   4. each `eval-designer-*` subdir has its own `output.md`

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use regex::Regex;

static NOVELTY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^novelty_target:\s*(\S+)\s*$").unwrap());

/// Return `(ok, reason)`. `reason` is empty when `ok`. `Err` if `swarm-state.yaml`
/// exists but cannot be read.
pub fn preflight_check(run_dir: &Path) -> io::Result<(bool, String)> {
    let output_md = run_dir.join("output.md");
    if !output_md.exists() {
        return Ok((
            false,
            format!(
                "Pre-flight refusal: output.md not found at {}. Re-run /research-execute first to produce the synthesist's output.",
                output_md.display()
            ),
        ));
    }

    let state = run_dir.join("swarm-state.yaml");
    if !state.exists() {
        return Ok((
            false,
            format!(
                "Pre-flight refusal: swarm-state.yaml not found at {}.",
                state.display()
            ),
        ));
    }

    let text = fs::read_to_string(&state)?;
    let target = match NOVELTY_RE.captures(&text) {
        Some(c) => c.get(1).unwrap().as_str().to_string(),
        None => {
            return Ok((
                false,
                format!(
                    "Pre-flight refusal: novelty_target not found in {}.",
                    state.display()
                ),
            ));
        }
    };
    if target != "hypothesis" {
        return Ok((
            false,
            format!(
                "Pre-flight refusal: paper chain only runs on hypothesis-target outputs. This run's novelty_target is {} (expected hypothesis); gap-finding runs lack the eval-designer protocols the paper chain consumes.",
                target
            ),
        ));
    }

    let eval_dirs: Vec<PathBuf> = fs::read_dir(run_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().starts_with("eval-designer-"))
                .unwrap_or(false)
        })
        .collect();
    if eval_dirs.is_empty() {
        return Ok((
            false,
            format!(
                "Pre-flight refusal: no eval-designer-* subdirs in {}. Paper chain requires Phase 5 protocols as input.",
                run_dir.display()
            ),
        ));
    }
    for d in &eval_dirs {
        if !d.join("output.md").exists() {
            return Ok((
                false,
                format!(
                    "Pre-flight refusal: eval-designer subdir {} missing output.md.",
                    d.display()
                ),
            ));
        }
    }

    Ok((true, String::new()))
}

/// Extended preflight returning `(ok, reason, warnings)`. When `paper_mode` is
/// true, adds a non-blocking `VERCEL_TOKEN` warning if the env var is unset or
/// empty (mirrors Python's falsy `os.environ.get("VERCEL_TOKEN")`).
pub fn preflight_check_with_paper(
    run_dir: &Path,
    paper_mode: bool,
) -> io::Result<(bool, String, Vec<String>)> {
    let (ok, reason) = preflight_check(run_dir)?;
    let mut warnings: Vec<String> = Vec::new();
    let token_set = env::var_os("VERCEL_TOKEN")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    if ok && paper_mode && !token_set {
        warnings.push(
            "VERCEL_TOKEN not set — Phase 6.5 (experimentalist) will fail immediately when it tries to spin up a sandbox. Set the env var before invoking /research-execute --paper if you want experiments."
                .to_string(),
        );
    }
    Ok((ok, reason, warnings))
}
