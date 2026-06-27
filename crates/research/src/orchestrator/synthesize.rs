//! Phase 6 synthesist dispatch + run finalization (run-root output.md and
//! the spec-latest symlink).

use std::fs;
use std::io;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};
use crate::orchestrator::OrchestratorError;
use crate::worker::WorkerOutcome;
use crate::worker_tools::Tool;

/// Build and run the single synthesist worker, inlining the spec, the plan,
/// every scout output, and every gap-finder output. Returns the spec (so the
/// caller can gate it) and the worker outcome.
#[allow(clippy::too_many_arguments)]
pub async fn run_synthesist(
    run_dir: &Path,
    spec_text: &str,
    plan_text: &str,
    scout_dirs: &[PathBuf],
    gap_dirs: &[PathBuf],
    smith_dirs: &[PathBuf],
    redteam_dirs: &[PathBuf],
    eval_dirs: &[PathBuf],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<(WorkerSpec, WorkerOutcome), OrchestratorError> {
    let mut prior: Vec<(String, String)> = Vec::new();
    prior.push(("Plan".to_string(), plan_text.to_string()));
    for d in scout_dirs {
        let name = dir_name(d);
        let body = read_output(d);
        prior.push((format!("Scout {name}"), body));
    }
    for d in gap_dirs {
        let name = dir_name(d);
        let body = read_output(d);
        prior.push((format!("Gap-finder {name}"), body));
    }
    for d in smith_dirs {
        let name = dir_name(d);
        let body = read_output(d);
        prior.push((format!("Hypothesis-smith {name}"), body));
    }
    for d in redteam_dirs {
        let name = dir_name(d);
        let body = read_output(d);
        prior.push((format!("Red-team {name}"), body));
    }
    for d in eval_dirs {
        let name = dir_name(d);
        let body = read_output(d);
        prior.push((format!("Eval-designer {name}"), body));
    }
    // `prior` owns the (label, content) strings; `prior_refs` borrows them for
    // the single `build_prompt` call. No leak — `prior` lives for this scope.
    let prior_refs: Vec<(&str, &str)> = prior
        .iter()
        .map(|(l, c)| (l.as_str(), c.as_str()))
        .collect();

    let output_dir = run_dir.join("synthesist");
    fs::create_dir_all(&output_dir)?;
    let spec = WorkerSpec {
        name: "synthesist".to_string(),
        role: "synthesist".to_string(),
        output_dir: output_dir.clone(),
        shared_dir: run_dir.to_path_buf(),
        prompt: build_prompt(spec_text, &prior_refs, None, &output_dir),
    };
    let outcomes = dispatch_wave(
        vec![spec.clone()],
        agents_dir,
        provider,
        default_model,
        max_parallel,
        extra_tools,
    )
    .await?;
    let (_, outcome) = outcomes
        .into_iter()
        .next()
        .ok_or_else(|| OrchestratorError::Finalize("synthesist produced no outcome".into()))?;
    Ok((spec, outcome))
}

/// Copy `synthesist/output.md` to `run_dir/output.md` and create the
/// spec-latest symlink `research_base/specs/<spec-stem>-latest.md` →
/// `../runs/<run_id>/output.md` (relative, so the tree is relocatable).
pub fn finalize_run(run_dir: &Path, spec_path: &Path, research_base: &Path) -> io::Result<PathBuf> {
    let synth_output = run_dir.join("synthesist").join("output.md");
    fs::copy(&synth_output, run_dir.join("output.md"))?;

    let specs_dir = research_base.join("specs");
    fs::create_dir_all(&specs_dir)?;
    let stem = spec_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "spec".to_string());
    let run_id = run_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let link = specs_dir.join(format!("{stem}-latest.md"));
    let target = format!("../runs/{run_id}/output.md");
    if link.exists() || symlink_exists(&link) {
        fs::remove_file(&link)?;
    }
    symlink(&target, &link)?;
    Ok(link)
}

fn symlink_exists(p: &Path) -> bool {
    p.symlink_metadata().is_ok()
}

fn dir_name(d: &Path) -> String {
    d.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn read_output(d: &Path) -> String {
    fs::read_to_string(d.join("output.md")).unwrap_or_else(|_| "(no output.md)".to_string())
}
