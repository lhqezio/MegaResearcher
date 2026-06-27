//! Phase 3 hypothesis-smith dispatch (one smith per gap) + the revision
//! re-dispatch the Phase 4 red-team loop drives on a REJECT verdict.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};
use crate::orchestrator::dispatch_plan::Assignment;
use crate::orchestrator::gaps::Gap;
use crate::orchestrator::gate::{verify_wave, GateStatus};
use crate::orchestrator::OrchestratorError;
use crate::worker_tools::Tool;

/// A hypothesis forged in Phase 3 and carried through Phases 4 and 5. `name` is
/// `hypothesis-smith-<N>`; `dir` is `run_dir/hypothesis-smith-<N>`; `gap` is the
/// targeted gap (so red-team/eval-designer prompts can inline the gap-finder).
#[derive(Debug, Clone)]
pub struct Hypothesis {
    pub name: String,
    pub dir: PathBuf,
    pub gap: Gap,
}

/// Build the worker spec for a hypothesis-smith. On initial dispatch
/// (`revision_prior == None`) the prompt inlines the gap (as a titled
/// assignment) and the gap-finder's `output.md`. On revision, the previous
/// red-team critique is appended as an extra prior section.
pub fn build_smith_spec(
    spec_text: &str,
    gap: &Gap,
    hyp_name: &str,
    output_dir: &Path,
    shared_dir: &Path,
    revision_prior: Option<&str>,
) -> WorkerSpec {
    let finder_output = fs::read_to_string(gap.finder_dir.join("output.md"))
        .unwrap_or_else(|_| "(no output.md)".to_string());
    let mut prior: Vec<(&str, &str)> = Vec::new();
    prior.push(("Gap-finder output", &finder_output));
    if let Some(critique) = revision_prior {
        prior.push(("Previous red-team critique", critique));
    }
    let assignment = Assignment {
        id: hyp_name.to_string(),
        role: "hypothesis-smith".to_string(),
        title: "Forge a hypothesis for this gap".to_string(),
        body: format!(
            "Targeted gap (from {}): {}\nGap category: {}",
            gap.finder_name, gap.statement, gap.gap_type
        ),
    };
    WorkerSpec {
        name: hyp_name.to_string(),
        role: "hypothesis-smith".to_string(),
        output_dir: output_dir.to_path_buf(),
        shared_dir: shared_dir.to_path_buf(),
        prompt: build_prompt(spec_text, &prior, Some(&assignment), output_dir),
    }
}

/// Phase 3: dispatch one hypothesis-smith per gap, in aggregate order, bounded
/// by `max_parallel`. Run the verification gate. Any gate escalation halts the
/// run with `Err(Escalated)` (matching scout/gap-finder gate behavior). On
/// success returns one `Hypothesis` per gap.
#[allow(clippy::too_many_arguments)]
pub async fn dispatch_hypothesis_smiths(
    run_dir: &Path,
    spec_text: &str,
    gaps: &[Gap],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<Vec<Hypothesis>, OrchestratorError> {
    let specs: Vec<WorkerSpec> = gaps
        .iter()
        .enumerate()
        .map(|(i, gap)| {
            let name = format!("hypothesis-smith-{}", i + 1);
            let output_dir = run_dir.join(&name);
            fs::create_dir_all(&output_dir).ok();
            build_smith_spec(spec_text, gap, &name, &output_dir, run_dir, None)
        })
        .collect();
    let outcomes = dispatch_wave(
        specs.clone(),
        agents_dir,
        provider.clone(),
        default_model,
        max_parallel,
        extra_tools,
    )
    .await?;
    let gates = verify_wave(
        outcomes,
        &specs,
        agents_dir,
        provider,
        default_model,
        extra_tools,
    )
    .await?;
    let escalated: Vec<String> = gates
        .iter()
        .filter(|g| g.status == GateStatus::Escalated)
        .map(|g| g.name.clone())
        .collect();
    if !escalated.is_empty() {
        return Err(OrchestratorError::Escalated(escalated));
    }
    Ok(gaps
        .iter()
        .enumerate()
        .map(|(i, gap)| Hypothesis {
            name: format!("hypothesis-smith-{}", i + 1),
            dir: run_dir.join(format!("hypothesis-smith-{}", i + 1)),
            gap: gap.clone(),
        })
        .collect())
}

/// Phase 4 revision step: re-dispatch the hypothesis-smith for `hyp` to its own
/// dir (overwriting `output.md` with the revised version that absorbs the
/// red-team critique). Single-worker wave + gate. Escalation halts the run.
pub async fn redispatch_smith_revision(
    hyp: &Hypothesis,
    spec_text: &str,
    redteam_output: &str,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<(), OrchestratorError> {
    let spec = build_smith_spec(
        spec_text,
        &hyp.gap,
        &hyp.name,
        &hyp.dir,
        hyp.dir.parent().unwrap_or_else(|| Path::new(".")),
        Some(redteam_output),
    );
    let outcomes = dispatch_wave(
        vec![spec.clone()],
        agents_dir,
        provider.clone(),
        default_model,
        1,
        extra_tools,
    )
    .await?;
    let gates = verify_wave(
        outcomes,
        std::slice::from_ref(&spec),
        agents_dir,
        provider,
        default_model,
        extra_tools,
    )
    .await?;
    if gates[0].status == GateStatus::Escalated {
        return Err(OrchestratorError::Escalated(vec![hyp.name.clone()]));
    }
    Ok(())
}
