//! Wave dispatch: build a worker's user prompt, construct a jailed `Worker`,
//! and run a phase's workers bounded by `max_parallel`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;
use futures::stream::{self, StreamExt};

use crate::orchestrator::dispatch_plan::Assignment;
use crate::orchestrator::OrchestratorError;
use crate::prompt_asset::load as load_asset;
use crate::worker::{Worker, WorkerConfig, WorkerOutcome};
use crate::worker_tools::{ScopedRead, ScopedWrite, Tool};

/// A fully-specified worker dispatch: its name, role, jailed output dir,
/// shared read dir (the run dir), and the assembled user prompt.
#[derive(Debug, Clone)]
pub struct WorkerSpec {
    pub name: String,
    pub role: String,
    pub output_dir: PathBuf,
    pub shared_dir: PathBuf,
    pub prompt: String,
}

/// Assemble a worker's user prompt: the spec, any prior-phase sections, the
/// assignment (if any), and the output directory the worker must write to.
pub fn build_prompt(
    spec_text: &str,
    prior: &[(&str, &str)],
    assignment: Option<&Assignment>,
    output_dir: &Path,
) -> String {
    let mut out = String::new();
    out.push_str("# Research spec\n\n");
    out.push_str(spec_text);
    out.push_str("\n\n");
    for (label, content) in prior {
        out.push_str(&format!("# {label}\n\n{content}\n\n"));
    }
    if let Some(a) = assignment {
        out.push_str(&format!("# Your assignment ({})\n\n", a.id));
        out.push_str(&format!("## {}\n\n{}\n\n", a.title, a.body));
    }
    out.push_str("# Output directory\n\n");
    out.push_str(&format!(
        "Write your three required artifacts (output.md, manifest.yaml, \
         verification.md) to this directory:\n{}\n",
        output_dir.display()
    ));
    out
}

/// Run a single worker: load its agent prompt asset, resolve the model,
/// wire jailed Read/Write tools, and drive the worker.
pub async fn run_worker(
    spec: &WorkerSpec,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
) -> Result<WorkerOutcome, OrchestratorError> {
    let asset_path = agents_dir.join(format!("{}.md", spec.role));
    let asset = load_asset(&asset_path).map_err(OrchestratorError::Io)?;
    let model = if asset.model == "inherit" {
        default_model.to_string()
    } else {
        asset.model.clone()
    };
    let read = Arc::new(ScopedRead::with_shared(&spec.output_dir, &spec.shared_dir)) as Arc<dyn Tool>;
    let write = Arc::new(ScopedWrite::new(&spec.output_dir)) as Arc<dyn Tool>;
    let worker = Worker::new(
        asset.body.clone(),
        vec![read, write],
        provider,
        WorkerConfig {
            max_turns: 50,
            max_tokens: 4096,
            model,
        },
        spec.output_dir.clone(),
    );
    worker.run(&spec.prompt).await.map_err(OrchestratorError::Worker)
}

/// Dispatch a wave of workers bounded by `max_parallel`. Returns outcomes
/// in spec order (sorted by submission index). With `max_parallel == 1`
/// dispatch is fully sequential, so a shared scripted provider's call order
/// is deterministic.
pub async fn dispatch_wave(
    specs: Vec<WorkerSpec>,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
) -> Result<Vec<(String, WorkerOutcome)>, OrchestratorError> {
    let n = specs.len();
    let indexed: Vec<(usize, WorkerSpec)> = specs.into_iter().enumerate().collect();
    let results: Vec<Result<(usize, String, WorkerOutcome), OrchestratorError>> = stream::iter(indexed)
        .map(|(i, spec)| {
            let provider = provider.clone();
            async move {
                run_worker(&spec, agents_dir, provider, default_model)
                    .await
                    .map(|o| (i, spec.name.clone(), o))
            }
        })
        .buffer_unordered(max_parallel.max(1) as usize)
        .collect()
        .await;

    let mut collected: Vec<(usize, String, WorkerOutcome)> = Vec::with_capacity(n);
    for r in results {
        collected.push(r?);
    }
    collected.sort_by_key(|(i, _, _)| *i);
    Ok(collected.into_iter().map(|(_, name, o)| (name, o)).collect())
}