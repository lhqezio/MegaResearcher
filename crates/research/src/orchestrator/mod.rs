//! The deterministic swarm orchestrator (Phase 4). Drives leaf `Worker`s
//! through the six phases, runs the verification gate, assembles
//! consolidations, and finalizes the run. See the design spec §4/§10/§11.

pub mod consolidate;
pub mod dispatch;
pub mod dispatch_plan;
pub mod evaldesign;
pub mod gaps;
pub mod gate;
pub mod hypothesis;
pub mod preflight;
pub mod redteam;
pub mod synthesize;
pub mod verdict;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::orchestrator::consolidate::{consolidate_bibliography, consolidate_gaps};
use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};
use crate::orchestrator::dispatch_plan::parse_plan;
use crate::orchestrator::gate::{verify_wave, GateStatus};
use crate::orchestrator::preflight::{
    add_escalation, build_initial_swarm_state, preflight_check, set_phase, write_swarm,
};
use crate::orchestrator::synthesize::{finalize_run, run_synthesist};
use crate::state::run_tree::create_run_tree;
use crate::worker::WorkerError;

/// Orchestrator-wide error.
#[derive(Debug)]
pub enum OrchestratorError {
    Preflight(String),
    Parse(String),
    Io(io::Error),
    Worker(WorkerError),
    Escalated(Vec<String>),
    Finalize(String),
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preflight(s) => write!(f, "pre-flight failed: {s}"),
            Self::Parse(s) => write!(f, "plan parse failed: {s}"),
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Worker(e) => write!(f, "worker error: {e:?}"),
            Self::Escalated(names) => write!(f, "workers escalated: {names:?}"),
            Self::Finalize(s) => write!(f, "finalize failed: {s}"),
        }
    }
}

impl std::error::Error for OrchestratorError {}

impl From<io::Error> for OrchestratorError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<WorkerError> for OrchestratorError {
    fn from(e: WorkerError) -> Self {
        Self::Worker(e)
    }
}

/// Orchestrator configuration: where runs live, where agent prompt assets
/// are, the model to resolve "inherit" to, and the wave concurrency bound.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub research_base: PathBuf,
    pub agents_dir: PathBuf,
    pub default_model: String,
    pub max_parallel: u32,
}

/// The orchestrator: config + a shared provider.
pub struct Orchestrator {
    pub config: OrchestratorConfig,
    pub provider: Arc<dyn LlmProvider>,
}

impl Orchestrator {
    pub fn new(config: OrchestratorConfig, provider: Arc<dyn LlmProvider>) -> Self {
        Self { config, provider }
    }

    /// Run the swarm. `run_id` is provided by the caller (the CLI calls
    /// `generate_run_id()`; tests pass a fixed string) so the run is
    /// deterministic. Task 6 implements Phases 1–2; Task 7 adds Phase 6.
    pub async fn execute(
        &self,
        spec_path: &Path,
        plan_path: &Path,
        run_id: &str,
    ) -> Result<RunOutcome, OrchestratorError> {
        let spec_text = std::fs::read_to_string(spec_path)?;
        let plan_text = std::fs::read_to_string(plan_path)?;
        let parsed = parse_plan(&plan_text).map_err(OrchestratorError::Parse)?;
        let target = parsed.novelty_target;

        preflight_check(spec_path, plan_path, &self.config.agents_dir, target)
            .map_err(OrchestratorError::Preflight)?;

        let run_dir = create_run_tree(&self.config.research_base, run_id)?;
        let mut swarm = build_initial_swarm_state(
            run_id,
            spec_path,
            plan_path,
            target,
            self.config.max_parallel,
        );
        write_swarm(&swarm, &run_dir)?;

        // Phase 1 — literature-scout.
        let scout_specs: Vec<WorkerSpec> = parsed
            .scouts
            .iter()
            .map(|a| {
                let output_dir = run_dir.join(&a.id);
                std::fs::create_dir_all(&output_dir).ok();
                WorkerSpec {
                    name: a.id.clone(),
                    role: "literature-scout".into(),
                    output_dir,
                    shared_dir: run_dir.clone(),
                    prompt: build_prompt(&spec_text, &[], Some(a), &run_dir.join(&a.id)),
                }
            })
            .collect();
        let scout_dirs: Vec<PathBuf> = scout_specs.iter().map(|s| s.output_dir.clone()).collect();
        set_phase(&mut swarm, "literature-scout", "running", vec![]);
        write_swarm(&swarm, &run_dir)?;
        let scout_outcomes = dispatch_wave(
            scout_specs.clone(),
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
            self.config.max_parallel,
        )
        .await?;
        let scout_gates = verify_wave(
            scout_outcomes,
            &scout_specs,
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
        )
        .await?;
        let scout_workers: Vec<(String, String)> = scout_gates
            .iter()
            .map(|g| (g.name.clone(), gate_status_str(g.status)))
            .collect();
        set_phase(
            &mut swarm,
            "literature-scout",
            "complete",
            scout_workers.clone(),
        );
        let escalated: Vec<String> = scout_gates
            .iter()
            .filter(|g| g.status == GateStatus::Escalated)
            .map(|g| g.name.clone())
            .collect();
        if !escalated.is_empty() {
            for name in &escalated {
                add_escalation(&mut swarm, name, "missing artifacts after retry", 1);
            }
            write_swarm(&swarm, &run_dir)?;
            return Err(OrchestratorError::Escalated(escalated));
        }
        consolidate_bibliography(&run_dir, &scout_dirs)?;
        write_swarm(&swarm, &run_dir)?;

        // Phase 2 — gap-finder.
        let bib_text = std::fs::read_to_string(run_dir.join("bibliography.md")).unwrap_or_default();
        let gap_specs: Vec<WorkerSpec> = parsed
            .gap_finders
            .iter()
            .map(|a| {
                let output_dir = run_dir.join(&a.id);
                std::fs::create_dir_all(&output_dir).ok();
                WorkerSpec {
                    name: a.id.clone(),
                    role: "gap-finder".into(),
                    output_dir,
                    shared_dir: run_dir.clone(),
                    prompt: build_prompt(
                        &spec_text,
                        &[("Consolidated bibliography", &bib_text)],
                        Some(a),
                        &run_dir.join(&a.id),
                    ),
                }
            })
            .collect();
        let gap_dirs: Vec<PathBuf> = gap_specs.iter().map(|s| s.output_dir.clone()).collect();
        set_phase(&mut swarm, "gap-finder", "running", vec![]);
        write_swarm(&swarm, &run_dir)?;
        let gap_outcomes = dispatch_wave(
            gap_specs.clone(),
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
            self.config.max_parallel,
        )
        .await?;
        let gap_gates = verify_wave(
            gap_outcomes,
            &gap_specs,
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
        )
        .await?;
        let gap_workers: Vec<(String, String)> = gap_gates
            .iter()
            .map(|g| (g.name.clone(), gate_status_str(g.status)))
            .collect();
        set_phase(&mut swarm, "gap-finder", "complete", gap_workers);
        let gap_escalated: Vec<String> = gap_gates
            .iter()
            .filter(|g| g.status == GateStatus::Escalated)
            .map(|g| g.name.clone())
            .collect();
        if !gap_escalated.is_empty() {
            for name in &gap_escalated {
                add_escalation(&mut swarm, name, "missing artifacts after retry", 1);
            }
            write_swarm(&swarm, &run_dir)?;
            return Err(OrchestratorError::Escalated(gap_escalated));
        }
        consolidate_gaps(&run_dir, &gap_dirs)?;
        write_swarm(&swarm, &run_dir)?;

        // Phase 6 — synthesist (gap-finding skips 3/4/5, already marked skipped).
        set_phase(&mut swarm, "synthesist", "running", vec![]);
        write_swarm(&swarm, &run_dir)?;
        let (synth_spec, synth_outcome) = run_synthesist(
            &run_dir,
            &spec_text,
            &plan_text,
            &scout_dirs,
            &gap_dirs,
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
            self.config.max_parallel,
        )
        .await?;
        let synth_gates = verify_wave(
            vec![("synthesist".to_string(), synth_outcome)],
            std::slice::from_ref(&synth_spec),
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
        )
        .await?;
        let synth_status = gate_status_str(synth_gates[0].status);
        set_phase(
            &mut swarm,
            "synthesist",
            "complete",
            vec![("synthesist".to_string(), synth_status.clone())],
        );
        if synth_gates[0].status == GateStatus::Escalated {
            add_escalation(&mut swarm, "synthesist", "missing artifacts after retry", 1);
            write_swarm(&swarm, &run_dir)?;
            return Err(OrchestratorError::Escalated(vec!["synthesist".to_string()]));
        }
        finalize_run(&run_dir, spec_path, &self.config.research_base)?;
        write_swarm(&swarm, &run_dir)?;

        Ok(RunOutcome {
            run_dir,
            run_id: run_id.to_string(),
            phase_statuses: swarm
                .phases
                .iter()
                .map(|p| (p.name.clone(), p.status.clone()))
                .collect(),
            escalations: swarm.escalations.iter().map(|e| e.worker.clone()).collect(),
        })
    }
}

/// The outcome of a run: run dir/id, per-phase final statuses, escalated workers.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub run_dir: PathBuf,
    pub run_id: String,
    pub phase_statuses: Vec<(String, String)>,
    pub escalations: Vec<String>,
}

fn gate_status_str(status: GateStatus) -> String {
    match status {
        GateStatus::Passed => "passed".to_string(),
        GateStatus::Escalated => "escalated".to_string(),
    }
}
