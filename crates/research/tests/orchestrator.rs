//! Orchestrator tests: pre-flight, run setup, dispatch, gate, consolidate,
//! execute, and the gap-finding integration test.

// `mod common;` pulls in `fake_provider`, which later tasks (Task 3 dispatch
// tests) consume. Until then it is dead code in this test binary; allow it
// here rather than editing the shared helper.
#[allow(dead_code)]
mod common;

use std::fs;
use std::path::{Path, PathBuf};

use tempfile::tempdir;

use megaresearcher_research::orchestrator::dispatch_plan::NoveltyTarget;
use megaresearcher_research::orchestrator::preflight::{
    build_initial_swarm_state, preflight_check, required_agent_roles, set_phase, write_swarm,
};
use megaresearcher_research::state::run_tree::{create_run_tree, run_dir};
use megaresearcher_research::state::swarm_state::SwarmState;

fn write_agents(dir: &Path, roles: &[&str]) {
    for r in roles {
        fs::write(dir.join(format!("{r}.md")), format!("# {r}\n\nbody\n")).unwrap();
    }
}

#[test]
fn preflight_passes_when_files_and_agents_exist() {
    let tmp = tempdir().unwrap();
    let spec = tmp.path().join("spec.md");
    let plan = tmp.path().join("plan.md");
    let agents = tmp.path().join("agents");
    fs::create_dir_all(&agents).unwrap();
    fs::write(&spec, "spec").unwrap();
    fs::write(&plan, "plan").unwrap();
    write_agents(&agents, &required_agent_roles(NoveltyTarget::GapFinding));
    assert!(preflight_check(&spec, &plan, &agents, NoveltyTarget::GapFinding).is_ok());
}

#[test]
fn preflight_fails_on_missing_agent_file() {
    let tmp = tempdir().unwrap();
    let spec = tmp.path().join("spec.md");
    let plan = tmp.path().join("plan.md");
    let agents = tmp.path().join("agents");
    fs::create_dir_all(&agents).unwrap();
    fs::write(&spec, "spec").unwrap();
    fs::write(&plan, "plan").unwrap();
    // gap-finder.md intentionally missing.
    write_agents(&agents, &["literature-scout", "synthesist"]);
    let err = preflight_check(&spec, &plan, &agents, NoveltyTarget::GapFinding).unwrap_err();
    assert!(err.contains("gap-finder.md"), "err was: {err}");
}

#[test]
fn preflight_fails_on_missing_spec() {
    let tmp = tempdir().unwrap();
    let agents = tmp.path().join("agents");
    fs::create_dir_all(&agents).unwrap();
    write_agents(&agents, &required_agent_roles(NoveltyTarget::GapFinding));
    let err = preflight_check(
        &tmp.path().join("missing-spec.md"),
        &tmp.path().join("plan.md"),
        &agents,
        NoveltyTarget::GapFinding,
    )
    .unwrap_err();
    assert!(err.contains("spec"));
}

#[test]
fn initial_swarm_state_marks_critique_phases_skipped_for_gap_finding() {
    let swarm = build_initial_swarm_state(
        "2026-06-27-0315-a1b2c3",
        &PathBuf::from("specs/s.md"),
        &PathBuf::from("plans/p.md"),
        NoveltyTarget::GapFinding,
        4,
    );
    assert_eq!(swarm.run_id, "2026-06-27-0315-a1b2c3");
    assert_eq!(swarm.novelty_target, "gap-finding");
    assert_eq!(swarm.max_parallel, 4);
    assert_eq!(swarm.phases.len(), 6);
    assert_eq!(swarm.phases[0].name, "literature-scout");
    assert_eq!(swarm.phases[0].status, "pending");
    assert_eq!(swarm.phases[2].name, "hypothesis-smith");
    assert_eq!(swarm.phases[2].status, "skipped");
    assert_eq!(swarm.phases[3].status, "skipped");
    assert_eq!(swarm.phases[4].status, "skipped");
    assert_eq!(swarm.phases[5].name, "synthesist");
    assert_eq!(swarm.phases[5].status, "pending");
}

#[test]
fn initial_swarm_state_all_pending_for_hypothesis() {
    let swarm = build_initial_swarm_state(
        "r",
        &PathBuf::from("s"),
        &PathBuf::from("p"),
        NoveltyTarget::Hypothesis,
        2,
    );
    assert!(swarm.phases.iter().all(|p| p.status == "pending"));
}

#[test]
fn write_swarm_round_trips() {
    let tmp = tempdir().unwrap();
    let run_dir = create_run_tree(tmp.path(), "rid").unwrap();
    let mut swarm = build_initial_swarm_state(
        "rid",
        &PathBuf::from("s"),
        &PathBuf::from("p"),
        NoveltyTarget::GapFinding,
        4,
    );
    set_phase(
        &mut swarm,
        "literature-scout",
        "complete",
        vec![("literature-scout-1".into(), "passed".into())],
    );
    write_swarm(&swarm, &run_dir).unwrap();
    let read_back = SwarmState::read(&run_dir.join("swarm-state.yaml")).unwrap();
    assert_eq!(read_back, swarm);
    assert_eq!(read_back.phases[0].status, "complete");
    assert_eq!(read_back.phases[0].workers.len(), 1);
    assert_eq!(read_back.phases[0].workers[0].name, "literature-scout-1");
}

// Suppress unused import until dispatch tests (Task 3) land.
#[allow(dead_code)]
fn _touch_run_dir(base: &std::path::Path) {
    let _ = run_dir(base, "x");
}