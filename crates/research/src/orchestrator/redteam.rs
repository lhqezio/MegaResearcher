//! Phase 4 red-team critique loop. For each hypothesis, dispatch a red-team
//! worker, parse its verdict, and either advance the hypothesis (APPROVE),
//! revise it (REJECT, up to 3 revisions), or kill it (KILL / cap / unparseable).
//! Killed/intractable hypotheses land in `swarm.escalations` for the audit
//! trail; survivors advance to Phase 5.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};
use crate::orchestrator::gate::{verify_wave, GateStatus};
use crate::orchestrator::hypothesis::{redispatch_smith_revision, Hypothesis};
use crate::orchestrator::verdict::{parse_redteam_verdict_file, RedTeamVerdict};
use crate::orchestrator::OrchestratorError;
use crate::state::swarm_state::SwarmState;

/// Cap on red-team revisions before a hypothesis is escalated.
pub const REVISION_CAP: u32 = 3;

/// The outcome of the Phase 4 loop.
#[derive(Debug, Clone)]
pub struct RedTeamResult {
    pub survivors: Vec<Hypothesis>,
    pub redteam_dirs: Vec<PathBuf>,
    pub killed: Vec<String>,
}

/// Build the red-team worker spec for one critique round.
pub fn build_redteam_prompt(
    spec_text: &str,
    hypothesis_output: &str,
    gap_finder_output: &str,
    output_dir: &Path,
) -> WorkerSpec {
    let prior: [(&str, &str); 2] = [
        ("Hypothesis under critique", hypothesis_output),
        ("Gap-finder output for the targeted gap", gap_finder_output),
    ];
    WorkerSpec {
        name: output_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "red-team".to_string()),
        role: "red-team".to_string(),
        output_dir: output_dir.to_path_buf(),
        shared_dir: output_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        prompt: build_prompt(spec_text, &prior, None, output_dir),
    }
}

/// Run the Phase 4 loop over `hypotheses`. Mutates `swarm.retry_counts` (per-
/// hypothesis revision count) and `swarm.escalations` (killed hypotheses).
#[allow(clippy::too_many_arguments)]
pub async fn run_redteam_loop(
    run_dir: &Path,
    spec_text: &str,
    hypotheses: Vec<Hypothesis>,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    swarm: &mut SwarmState,
) -> Result<RedTeamResult, OrchestratorError> {
    let mut survivors = Vec::new();
    let mut killed = Vec::new();
    let mut redteam_dirs = Vec::new();

    for hyp in hypotheses.into_iter() {
        let n = hyp
            .name
            .trim_start_matches("hypothesis-smith-")
            .parse::<u32>()
            .unwrap_or(0);
        let finder_output = fs::read_to_string(hyp.gap.finder_dir.join("output.md"))
            .unwrap_or_else(|_| "(no output.md)".to_string());

        let mut revision_count: u32 = 0;
        loop {
            // Cap check at the top of each round: once a hypothesis has had
            // `REVISION_CAP` revisions, escalate before dispatching another
            // red-team worker. This keeps the 3rd revision in the audit trail
            // (redteam_dirs holds r1/r2/r3) while preventing a 4th dispatch.
            if revision_count >= REVISION_CAP {
                let round = revision_count + 1;
                crate::orchestrator::preflight::add_escalation(
                    swarm,
                    &hyp.name,
                    "exceeded 3 red-team revisions",
                    round,
                );
                killed.push(hyp.name.clone());
                break;
            }
            // Read the hypothesis fresh each round so revisions (which overwrite
            // hypothesis-smith-<N>/output.md) are re-critiqued, not the stale original.
            let hypothesis_output = fs::read_to_string(hyp.dir.join("output.md"))
                .unwrap_or_else(|_| "(no output.md)".to_string());
            let round = revision_count + 1;
            let rt_dir = run_dir.join(format!("red-team-{n}-r{round}"));
            fs::create_dir_all(&rt_dir).ok();
            let spec = build_redteam_prompt(spec_text, &hypothesis_output, &finder_output, &rt_dir);
            let outcomes = dispatch_wave(
                vec![spec.clone()],
                agents_dir,
                provider.clone(),
                default_model,
                max_parallel.max(1),
            )
            .await?;
            let gates = verify_wave(
                outcomes,
                std::slice::from_ref(&spec),
                agents_dir,
                provider.clone(),
                default_model,
            )
            .await?;
            if gates[0].status == GateStatus::Escalated {
                crate::orchestrator::preflight::add_escalation(
                    swarm,
                    &hyp.name,
                    "red-team missing artifacts after retry",
                    round,
                );
                killed.push(hyp.name.clone());
                redteam_dirs.push(rt_dir);
                break;
            }

            let verdict = parse_redteam_verdict_file(&rt_dir.join("output.md"))
                .map_err(OrchestratorError::Io)?;
            redteam_dirs.push(rt_dir.clone());
            match verdict {
                Some(RedTeamVerdict::Approve) => {
                    survivors.push(hyp);
                    break;
                }
                Some(RedTeamVerdict::Kill) => {
                    crate::orchestrator::preflight::add_escalation(
                        swarm,
                        &hyp.name,
                        "red-team KILL (irrecoverable)",
                        round,
                    );
                    killed.push(hyp.name.clone());
                    break;
                }
                Some(RedTeamVerdict::Reject { revision: _ }) => {
                    revision_count += 1;
                    swarm.retry_counts.insert(hyp.name.clone(), revision_count);
                    let critique = fs::read_to_string(rt_dir.join("output.md"))
                        .unwrap_or_else(|_| "(no output.md)".to_string());
                    redispatch_smith_revision(
                        &hyp,
                        spec_text,
                        &critique,
                        agents_dir,
                        provider.clone(),
                        default_model,
                    )
                    .await?;
                    // Loop: next round dispatches red-team again on the revised hypothesis.
                    continue;
                }
                None => {
                    crate::orchestrator::preflight::add_escalation(
                        swarm,
                        &hyp.name,
                        "red-team produced no parseable verdict",
                        round,
                    );
                    killed.push(hyp.name.clone());
                    break;
                }
            }
        }
    }

    Ok(RedTeamResult {
        survivors,
        redteam_dirs,
        killed,
    })
}
