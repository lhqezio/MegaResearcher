//! Orchestrator pre-flight + initial swarm-state construction.
//!
//! Spec §10.2: before a run, verify the inputs are present and writable.
//! Phase 4a does structural checks only (spec/plan exist, agent files
//! present, runs dir createable). Provider-key reachability and ml-intern
//! reachability are deferred to Phase 5 (live runs) — they have no meaning
//! against a fake provider.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::orchestrator::dispatch_plan::NoveltyTarget;
use crate::state::swarm_state::{Escalation, Phase, SwarmState, Worker};

/// The six phase names in execution order.
pub const PHASE_NAMES: [&str; 6] = [
    "literature-scout",
    "gap-finder",
    "hypothesis-smith",
    "red-team",
    "eval-designer",
    "synthesist",
];

/// Agent roles required for a run with the given target.
pub fn required_agent_roles(target: NoveltyTarget) -> Vec<&'static str> {
    let base = ["literature-scout", "gap-finder", "synthesist"];
    match target {
        NoveltyTarget::GapFinding => base.to_vec(),
        NoveltyTarget::Hypothesis => {
            let mut v = base.to_vec();
            v.extend(["hypothesis-smith", "red-team", "eval-designer"]);
            v
        }
    }
}

/// Structural pre-flight: spec/plan files exist; `agents_dir` is a directory
/// containing the `<role>.md` files the target needs.
pub fn preflight_check(
    spec_path: &Path,
    plan_path: &Path,
    agents_dir: &Path,
    target: NoveltyTarget,
) -> Result<(), String> {
    if !spec_path.exists() {
        return Err(format!("spec not found: {}", spec_path.display()));
    }
    if !plan_path.exists() {
        return Err(format!("plan not found: {}", plan_path.display()));
    }
    if !agents_dir.is_dir() {
        return Err(format!("agents dir not found: {}", agents_dir.display()));
    }
    for role in required_agent_roles(target) {
        let f = agents_dir.join(format!("{role}.md"));
        if !f.exists() {
            return Err(format!("missing agent file: {}", f.display()));
        }
    }
    Ok(())
}

/// Build the initial `SwarmState` with all six phases present. For a
/// gap-finding run phases 3/4/5 are marked `skipped` (idle, never dispatched);
/// for a hypothesis run all six are `pending`.
pub fn build_initial_swarm_state(
    run_id: &str,
    spec_path: &Path,
    plan_path: &Path,
    target: NoveltyTarget,
    max_parallel: u32,
) -> SwarmState {
    let skip = target.skips_critique_phases();
    let phases = PHASE_NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let is_critique = matches!(i, 2..=4);
            let status = if skip && is_critique {
                "skipped".to_string()
            } else {
                "pending".to_string()
            };
            Phase {
                name: name.to_string(),
                status,
                workers: Vec::new(),
                hypotheses: Vec::new(),
            }
        })
        .collect();
    SwarmState {
        run_id: run_id.to_string(),
        spec_path: spec_path.to_string_lossy().to_string(),
        plan_path: plan_path.to_string_lossy().to_string(),
        novelty_target: target.as_str().to_string(),
        max_parallel,
        phases,
        escalations: Vec::new(),
        retry_counts: std::collections::HashMap::new(),
    }
}

/// Find the phase named `name` and set its status + workers. No-op (returns
/// without error) if the phase is absent — callers always pass a known name.
pub fn set_phase(swarm: &mut SwarmState, name: &str, status: &str, workers: Vec<(String, String)>) {
    let phase = swarm.phases.iter_mut().find(|p| p.name == name);
    if let Some(phase) = phase {
        phase.status = status.to_string();
        phase.workers = workers
            .into_iter()
            .map(|(name, status)| Worker { name, status })
            .collect();
    }
}

/// Append an escalation record (spec §11).
pub fn add_escalation(swarm: &mut SwarmState, worker: &str, reason: &str, retry_count: u32) {
    swarm.escalations.push(Escalation {
        worker: worker.to_string(),
        reason: reason.to_string(),
        retry_count,
    });
}

/// Write `swarm` to `run_dir/swarm-state.yaml`.
pub fn write_swarm(swarm: &SwarmState, run_dir: &Path) -> io::Result<()> {
    swarm.write(&run_dir.join("swarm-state.yaml"))
}

#[allow(dead_code)]
fn _unused(_p: PathBuf, _f: fs::File) {}
