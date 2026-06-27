//! Verification gate (spec §11): each worker must produce output.md,
//! manifest.yaml, verification.md. One retry on a miss; a second miss
//! escalates (the caller records the escalation and halts the run).

use std::path::Path;
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::orchestrator::dispatch::{run_worker, WorkerSpec};
use crate::orchestrator::OrchestratorError;
use crate::worker::WorkerOutcome;
use crate::worker_tools::check_artifacts;

/// The three artifacts every worker must write.
pub const REQUIRED_ARTIFACTS: &[&str] = &["output.md", "manifest.yaml", "verification.md"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateStatus {
    Passed,
    Escalated,
}

#[derive(Debug, Clone)]
pub struct GateOutcome {
    pub name: String,
    pub status: GateStatus,
    pub retries: u32,
}

/// Verify a wave of workers. For each, check the three artifacts; on a miss,
/// redispatch once with the missing-artifacts note appended to the prompt;
/// escalate if the retry still misses.
pub async fn verify_wave(
    outcomes: Vec<(String, WorkerOutcome)>,
    specs: &[WorkerSpec],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
) -> Result<Vec<GateOutcome>, OrchestratorError> {
    let mut results = Vec::with_capacity(outcomes.len());
    for (name, _first) in outcomes {
        let spec = specs
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| OrchestratorError::Finalize(format!("no spec for worker {name}")))?;
        let missing = check_artifacts(&spec.output_dir, REQUIRED_ARTIFACTS);
        if missing.is_empty() {
            results.push(GateOutcome { name, status: GateStatus::Passed, retries: 0 });
            continue;
        }
        // One retry with the missing list appended.
        let retry = retry_spec(spec, &missing);
        run_worker(&retry, agents_dir, provider.clone(), default_model).await?;
        let still_missing = check_artifacts(&spec.output_dir, REQUIRED_ARTIFACTS);
        let status = if still_missing.is_empty() {
            GateStatus::Passed
        } else {
            GateStatus::Escalated
        };
        results.push(GateOutcome { name, status, retries: 1 });
    }
    Ok(results)
}

fn retry_spec(spec: &WorkerSpec, missing: &[String]) -> WorkerSpec {
    let mut prompt = spec.prompt.clone();
    prompt.push_str(&format!(
        "\n\n# Missing artifacts\n\nYour previous run did not write: {}. \
         Write them now to {}.\n",
        missing.join(", "),
        spec.output_dir.display()
    ));
    WorkerSpec {
        prompt,
        ..spec.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retry_spec_appends_missing() {
        let spec = WorkerSpec {
            name: "x".into(),
            role: "r".into(),
            output_dir: Path::new("/tmp/d").to_path_buf(),
            shared_dir: Path::new("/tmp").to_path_buf(),
            prompt: "ORIG".into(),
        };
        let r = retry_spec(&spec, &["verification.md".to_string()]);
        assert!(r.prompt.starts_with("ORIG"));
        assert!(r.prompt.contains("verification.md"));
        assert!(r.prompt.contains("/tmp/d"));
    }
}