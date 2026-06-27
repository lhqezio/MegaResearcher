//! Phase 5 eval-designer fan-out: one eval-designer per surviving hypothesis.
//! If a designer flags intractable compute, the hypothesis is recorded in
//! `swarm.escalations` (the audit trail) but its eval-designer output is still
//! passed to the synthesist.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;
use serde::Deserialize;

use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};
use crate::orchestrator::gate::{verify_wave, GateStatus};
use crate::orchestrator::hypothesis::Hypothesis;
use crate::orchestrator::preflight::add_escalation;
use crate::orchestrator::OrchestratorError;
use crate::state::swarm_state::SwarmState;

/// The Phase 5 outcome: the eval-designer dirs (for synthesis) and the
/// per-hypothesis phase-worker status (`passed` / `intractable`).
#[derive(Debug, Clone)]
pub struct EvalDesignResult {
    pub eval_dirs: Vec<PathBuf>,
    pub phase_workers: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct EvalDesignerManifest {
    #[serde(default)]
    flagged_intractable: bool,
}

/// Build the eval-designer worker spec for one survivor.
pub fn build_evaldesigner_prompt(
    spec_text: &str,
    hypothesis_output: &str,
    output_dir: &Path,
) -> WorkerSpec {
    let prior: [(&str, &str); 1] = [("Hypothesis to test", hypothesis_output)];
    WorkerSpec {
        name: output_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "eval-designer".to_string()),
        role: "eval-designer".to_string(),
        output_dir: output_dir.to_path_buf(),
        shared_dir: output_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        prompt: build_prompt(spec_text, &prior, None, output_dir),
    }
}

/// Read `flagged_intractable` from an eval-designer manifest. Missing field or
/// unparseable manifest -> false (never treat a malformed manifest as intractable).
pub fn parse_intractable(manifest_yaml: &str) -> bool {
    serde_yml::from_str::<EvalDesignerManifest>(manifest_yaml)
        .map(|m| m.flagged_intractable)
        .unwrap_or(false)
}

/// Phase 5: one eval-designer per survivor. Mutates `swarm.escalations` for
/// intractable designs. Returns the eval dirs (all of them, including
/// intractable ones, for the synthesis audit trail) and phase-worker statuses.
#[allow(clippy::too_many_arguments)]
pub async fn run_eval_designers(
    run_dir: &Path,
    spec_text: &str,
    survivors: &[Hypothesis],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    swarm: &mut SwarmState,
) -> Result<EvalDesignResult, OrchestratorError> {
    let specs: Vec<WorkerSpec> = survivors
        .iter()
        .map(|hyp| {
            let n = hyp
                .name
                .trim_start_matches("hypothesis-smith-")
                .parse::<u32>()
                .unwrap_or(0);
            let output_dir = run_dir.join(format!("eval-designer-{n}"));
            fs::create_dir_all(&output_dir).ok();
            let hypothesis_output = fs::read_to_string(hyp.dir.join("output.md"))
                .unwrap_or_else(|_| "(no output.md)".to_string());
            build_evaldesigner_prompt(spec_text, &hypothesis_output, &output_dir)
        })
        .collect();
    let outcomes = dispatch_wave(
        specs.clone(),
        agents_dir,
        provider.clone(),
        default_model,
        max_parallel,
    )
    .await?;
    let gates = verify_wave(outcomes, &specs, agents_dir, provider, default_model).await?;

    let mut eval_dirs = Vec::with_capacity(specs.len());
    let mut phase_workers = Vec::with_capacity(specs.len());
    for (spec, gate) in specs.into_iter().zip(gates.iter()) {
        // The hypothesis name is hypothesis-smith-<N>; the eval-designer is
        // eval-designer-<N>. Recover N from the eval-designer dir name.
        let n = spec
            .name
            .trim_start_matches("eval-designer-")
            .parse::<u32>()
            .unwrap_or(0);
        let hyp_name = format!("hypothesis-smith-{n}");
        if gate.status == GateStatus::Escalated {
            return Err(OrchestratorError::Escalated(vec![hyp_name]));
        }
        let manifest =
            fs::read_to_string(spec.output_dir.join("manifest.yaml")).unwrap_or_default();
        if parse_intractable(&manifest) {
            add_escalation(
                swarm,
                &hyp_name,
                "eval-designer flagged intractable compute",
                0,
            );
            phase_workers.push((hyp_name, "intractable".to_string()));
        } else {
            phase_workers.push((hyp_name, "passed".to_string()));
        }
        eval_dirs.push(spec.output_dir);
    }
    Ok(EvalDesignResult {
        eval_dirs,
        phase_workers,
    })
}
