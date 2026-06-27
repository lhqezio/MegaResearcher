//! Orchestrator tests: pre-flight, run setup, dispatch, gate, consolidate,
//! execute, and the gap-finding integration test.

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

use std::sync::Arc;

use claurst_api::{LlmProvider, StopReason, StreamEvent};
use claurst_core::types::{ContentBlock, UsageInfo};
use serde_json::json;

use common::fake_provider::FakeProvider;
use megaresearcher_research::orchestrator::dispatch::{
    build_prompt, dispatch_wave, run_worker, WorkerSpec,
};
use megaresearcher_research::worker::WorkerStop;

fn write_turn(file: &str, content: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart {
            id: "m".into(),
            model: "fake".into(),
            usage: UsageInfo::default(),
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text {
                text: String::new(),
            },
        },
        StreamEvent::TextDelta {
            index: 0,
            text: format!("writing {file}"),
        },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::ContentBlockStart {
            index: 1,
            content_block: ContentBlock::ToolUse {
                id: format!("tu_{file}"),
                name: "Write".into(),
                input: json!({ "file_path": file, "content": content }),
            },
        },
        StreamEvent::ContentBlockStop { index: 1 },
        StreamEvent::MessageDelta {
            stop_reason: Some(StopReason::ToolUse),
            usage: Some(UsageInfo::default()),
        },
        StreamEvent::MessageStop,
    ]
}

fn final_turn(text: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart {
            id: "m".into(),
            model: "fake".into(),
            usage: UsageInfo::default(),
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text {
                text: String::new(),
            },
        },
        StreamEvent::TextDelta {
            index: 0,
            text: text.into(),
        },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::MessageDelta {
            stop_reason: Some(StopReason::EndTurn),
            usage: Some(UsageInfo::default()),
        },
        StreamEvent::MessageStop,
    ]
}

/// The standard 4-turn "write three artifacts + final" sequence one worker
/// consumes. Repeated per worker for a multi-worker run.
fn three_artifact_turns() -> Vec<Vec<StreamEvent>> {
    vec![
        write_turn("output.md", "# Output\n\ncontent"),
        write_turn("manifest.yaml", "role: literature-scout\n"),
        write_turn("verification.md", "# Verification\n\nok"),
        final_turn("Done."),
    ]
}

fn fixture_agents_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/agents")
}

#[tokio::test]
async fn build_prompt_inlines_spec_and_prior_and_assignment() {
    let tmp = tempdir().unwrap();
    let p = build_prompt(
        "SPEC TEXT",
        &[("Prior phase", "PRIOR BODY")],
        Some(
            &megaresearcher_research::orchestrator::dispatch_plan::Assignment {
                id: "x".into(),
                role: "gap-finder".into(),
                title: "T".into(),
                body: "BODY".into(),
            },
        ),
        tmp.path(),
    );
    assert!(p.contains("SPEC TEXT"));
    assert!(p.contains("Prior phase"));
    assert!(p.contains("PRIOR BODY"));
    assert!(p.contains("T"));
    assert!(p.contains("BODY"));
    assert!(p.contains(tmp.path().to_string_lossy().as_ref()));
}

#[tokio::test]
async fn dispatch_wave_runs_two_scouts_writes_artifacts() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    fs::create_dir_all(&run_dir).unwrap();
    let dir1 = run_dir.join("literature-scout-1");
    let dir2 = run_dir.join("literature-scout-2");
    fs::create_dir_all(&dir1).unwrap();
    fs::create_dir_all(&dir2).unwrap();

    // 2 workers × 4 turns = 8 scripted turns, in dispatch order.
    let turns: Vec<Vec<StreamEvent>> = [three_artifact_turns(), three_artifact_turns()]
        .into_iter()
        .flatten()
        .collect();
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;

    let specs = vec![
        WorkerSpec {
            name: "literature-scout-1".into(),
            role: "literature-scout".into(),
            output_dir: dir1.clone(),
            shared_dir: run_dir.clone(),
            prompt: build_prompt("SPEC", &[], None, &dir1),
        },
        WorkerSpec {
            name: "literature-scout-2".into(),
            role: "literature-scout".into(),
            output_dir: dir2.clone(),
            shared_dir: run_dir.clone(),
            prompt: build_prompt("SPEC", &[], None, &dir2),
        },
    ];
    let outcomes = dispatch_wave(specs, &fixture_agents_dir(), provider, "fake-model", 1)
        .await
        .unwrap();
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0].0, "literature-scout-1");
    assert_eq!(outcomes[0].1.stop, WorkerStop::EndTurn);
    assert_eq!(outcomes[1].0, "literature-scout-2");
    assert!(dir1.join("output.md").exists());
    assert!(dir1.join("manifest.yaml").exists());
    assert!(dir1.join("verification.md").exists());
    assert!(dir2.join("output.md").exists());
    assert_eq!(fake.call_count(), 8);
}

#[tokio::test]
async fn run_worker_resolves_inherit_model() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("w");
    fs::create_dir_all(&dir).unwrap();
    let spec = WorkerSpec {
        name: "literature-scout-1".into(),
        role: "literature-scout".into(),
        output_dir: dir.clone(),
        shared_dir: tmp.path().to_path_buf(),
        prompt: build_prompt("S", &[], None, &dir),
    };
    // literature-scout.md has model: inherit (Phase 3 fixture). default_model applies.
    let fake = Arc::new(FakeProvider::new("fake", three_artifact_turns()));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let outcome = run_worker(&spec, &fixture_agents_dir(), provider, "resolved-model")
        .await
        .unwrap();
    assert_eq!(outcome.stop, WorkerStop::EndTurn);
    assert!(dir.join("output.md").exists());
}

use megaresearcher_research::orchestrator::gate::{verify_wave, GateStatus, REQUIRED_ARTIFACTS};

fn spec_for(
    name: &str,
    dir: &Path,
    run_dir: &Path,
) -> megaresearcher_research::orchestrator::dispatch::WorkerSpec {
    megaresearcher_research::orchestrator::dispatch::WorkerSpec {
        name: name.into(),
        role: "literature-scout".into(),
        output_dir: dir.to_path_buf(),
        shared_dir: run_dir.to_path_buf(),
        prompt: build_prompt("SPEC", &[], None, dir),
    }
}

#[tokio::test]
async fn gate_passes_when_all_artifacts_present_first_try() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let dir = run_dir.join("literature-scout-1");
    fs::create_dir_all(&dir).unwrap();
    let fake = Arc::new(FakeProvider::new("fake", three_artifact_turns()));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let spec = spec_for("literature-scout-1", &dir, &run_dir);
    let outcomes = [
        run_worker(&spec, &fixture_agents_dir(), provider.clone(), "fake-model")
            .await
            .unwrap(),
    ];
    let outcomes = vec![("literature-scout-1".to_string(), outcomes[0].clone())];
    let gate = verify_wave(
        outcomes,
        std::slice::from_ref(&spec),
        &fixture_agents_dir(),
        provider,
        "fake-model",
    )
    .await
    .unwrap();
    assert_eq!(gate.len(), 1);
    assert_eq!(gate[0].status, GateStatus::Passed);
    assert_eq!(gate[0].retries, 0);
}

#[tokio::test]
async fn gate_retries_then_passes_on_missing_artifact() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let dir = run_dir.join("literature-scout-1");
    fs::create_dir_all(&dir).unwrap();

    // First run writes output.md + manifest only (3 turns), then the retry
    // writes verification.md + final (2 turns). 5 scripted turns total.
    let turns: Vec<Vec<StreamEvent>> = vec![
        write_turn("output.md", "x"),
        write_turn("manifest.yaml", "y"),
        final_turn("partial"),
        write_turn("verification.md", "z"),
        final_turn("done"),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let spec = spec_for("literature-scout-1", &dir, &run_dir);
    let first = run_worker(&spec, &fixture_agents_dir(), provider.clone(), "fake-model")
        .await
        .unwrap();
    let outcomes = vec![("literature-scout-1".to_string(), first)];
    let gate = verify_wave(
        outcomes,
        std::slice::from_ref(&spec),
        &fixture_agents_dir(),
        provider,
        "fake-model",
    )
    .await
    .unwrap();
    assert_eq!(gate[0].status, GateStatus::Passed);
    assert_eq!(gate[0].retries, 1);
    assert!(dir.join("verification.md").exists());
    assert_eq!(fake.call_count(), 5); // 3 first run + 2 retry
}

#[tokio::test]
async fn gate_escalates_when_retry_still_missing() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let dir = run_dir.join("literature-scout-1");
    fs::create_dir_all(&dir).unwrap();

    // First run writes output + manifest only; retry writes nothing (final
    // immediately). verification.md never appears -> Escalated.
    let turns: Vec<Vec<StreamEvent>> = vec![
        write_turn("output.md", "x"),
        write_turn("manifest.yaml", "y"),
        final_turn("partial"),
        final_turn("still partial"),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let spec = spec_for("literature-scout-1", &dir, &run_dir);
    let first = run_worker(&spec, &fixture_agents_dir(), provider.clone(), "fake-model")
        .await
        .unwrap();
    let outcomes = vec![("literature-scout-1".to_string(), first)];
    let gate = verify_wave(
        outcomes,
        std::slice::from_ref(&spec),
        &fixture_agents_dir(),
        provider,
        "fake-model",
    )
    .await
    .unwrap();
    assert_eq!(gate[0].status, GateStatus::Escalated);
    assert_eq!(gate[0].retries, 1);
    assert!(!dir.join("verification.md").exists());
}

#[test]
fn required_artifacts_constant() {
    assert_eq!(
        REQUIRED_ARTIFACTS,
        &["output.md", "manifest.yaml", "verification.md"]
    );
}

use megaresearcher_research::orchestrator::consolidate::{
    consolidate_bibliography, consolidate_gaps,
};

#[test]
fn consolidate_bibliography_assembles_scout_outputs_in_order() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let s1 = run_dir.join("literature-scout-1");
    let s2 = run_dir.join("literature-scout-2");
    fs::create_dir_all(&s1).unwrap();
    fs::create_dir_all(&s2).unwrap();
    fs::write(s1.join("output.md"), "BIB A").unwrap();
    fs::write(s2.join("output.md"), "BIB B").unwrap();

    let out = consolidate_bibliography(&run_dir, &[s1.clone(), s2.clone()]).unwrap();
    assert_eq!(out, run_dir.join("bibliography.md"));
    let text = fs::read_to_string(&out).unwrap();
    assert!(text.starts_with("# Consolidated bibliography"));
    let a = text.find("## literature-scout-1").unwrap();
    let b = text.find("## literature-scout-2").unwrap();
    assert!(a < b);
    assert!(text.contains("BIB A"));
    assert!(text.contains("BIB B"));
}

#[test]
fn consolidate_gaps_writes_gaps_md() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let g = run_dir.join("gap-finder-1");
    fs::create_dir_all(&g).unwrap();
    fs::write(g.join("output.md"), "GAP LIST").unwrap();
    let out = consolidate_gaps(&run_dir, &[g]).unwrap();
    assert_eq!(out, run_dir.join("gaps.md"));
    let text = fs::read_to_string(&out).unwrap();
    assert!(text.starts_with("# Consolidated gaps"));
    assert!(text.contains("## gap-finder-1"));
    assert!(text.contains("GAP LIST"));
}

#[test]
fn consolidate_handles_missing_output_md() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let s = run_dir.join("literature-scout-1");
    fs::create_dir_all(&s).unwrap();
    // No output.md in the scout dir.
    let out = consolidate_bibliography(&run_dir, &[s]).unwrap();
    let text = fs::read_to_string(&out).unwrap();
    assert!(text.contains("(no output.md)"));
}

use megaresearcher_research::orchestrator::{
    Orchestrator, OrchestratorConfig, OrchestratorError, RunOutcome,
};
// SwarmState already imported at the top of this file.

fn fixture_plan_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plans/gap-finding-plan.md")
}
fn fixture_spec_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/specs/gap-finding-spec.md")
}

/// 4 workers (2 scouts, 1 gap-finder, 1 synthesist) × 4 turns = 16 turns.
/// Used by Task 6 (3 workers, 12 turns — no synthesist yet) and Task 7 (16).
fn run_turns(n_workers: usize) -> Vec<Vec<StreamEvent>> {
    (0..n_workers)
        .flat_map(|_| three_artifact_turns())
        .collect()
}

fn run_dir_of(out: &RunOutcome) -> &std::path::Path {
    &out.run_dir
}

#[tokio::test]
async fn execute_phases_1_and_2_for_gap_finding() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    let agents = fixture_agents_dir();
    fs::create_dir_all(&research_base).unwrap();

    // 4 workers: 2 scouts + 1 gap-finder + 1 synthesist. 4 × 4 = 16 turns.
    let fake = Arc::new(FakeProvider::new("fake", run_turns(4)));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let orch = Orchestrator::new(
        OrchestratorConfig {
            research_base: research_base.clone(),
            agents_dir: agents,
            default_model: "fake-model".into(),
            max_parallel: 1,
        },
        provider,
    );

    let out = orch
        .execute(
            &fixture_spec_path(),
            &fixture_plan_path(),
            "2026-06-27-0315-a1b2c3",
        )
        .await
        .unwrap();

    let run_dir = run_dir_of(&out).to_path_buf();
    assert_eq!(out.run_id, "2026-06-27-0315-a1b2c3");
    // Run tree.
    assert!(run_dir.join("swarm-state.yaml").exists());
    assert!(run_dir.join("literature-scout-1/output.md").exists());
    assert!(run_dir.join("literature-scout-2/output.md").exists());
    assert!(run_dir.join("gap-finder-1/output.md").exists());
    assert!(run_dir.join("bibliography.md").exists());
    assert!(run_dir.join("gaps.md").exists());

    // Swarm-state phase statuses.
    let swarm = SwarmState::read(&run_dir.join("swarm-state.yaml")).unwrap();
    let by_name: std::collections::HashMap<&str, &str> = swarm
        .phases
        .iter()
        .map(|p| (p.name.as_str(), p.status.as_str()))
        .collect();
    assert_eq!(by_name["literature-scout"], "complete");
    assert_eq!(by_name["gap-finder"], "complete");
    assert_eq!(by_name["hypothesis-smith"], "skipped");
    assert_eq!(by_name["red-team"], "skipped");
    assert_eq!(by_name["eval-designer"], "skipped");
    assert_eq!(by_name["synthesist"], "complete"); // Task 7 fills this in.
                                                   // Each completed phase has its workers recorded.
    let scouts = swarm
        .phases
        .iter()
        .find(|p| p.name == "literature-scout")
        .unwrap();
    assert_eq!(scouts.workers.len(), 2);
    assert!(scouts.workers.iter().all(|w| w.status == "passed"));
}

#[tokio::test]
async fn execute_halts_on_worker_escalation() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();

    // A single final-only turn: every worker call returns EndTurn with no
    // tool use, so no worker ever writes any artifact. Every worker's gate
    // misses all three artifacts, retries (still nothing), and escalates.
    // execute dispatches both scouts before gating the wave, then halts with
    // Err(Escalated) listing the escalated scout(s).
    let turns: Vec<Vec<StreamEvent>> = vec![final_turn("nothing written")];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let orch = Orchestrator::new(
        OrchestratorConfig {
            research_base,
            agents_dir: fixture_agents_dir(),
            default_model: "fake-model".into(),
            max_parallel: 1,
        },
        provider,
    );
    let err = orch
        .execute(&fixture_spec_path(), &fixture_plan_path(), "rid2")
        .await
        .unwrap_err();
    match err {
        OrchestratorError::Escalated(names) => {
            assert!(
                names.contains(&"literature-scout-1".to_string()),
                "expected literature-scout-1 in escalated names, got {names:?}"
            );
        }
        other => panic!("expected Escalated, got {other:?}"),
    }
}

use megaresearcher_research::orchestrator::synthesize::{finalize_run, run_synthesist};

#[tokio::test]
async fn run_synthesist_inlines_all_outputs_and_writes_artifacts() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let synth_dir = run_dir.join("synthesist");
    let s1 = run_dir.join("literature-scout-1");
    let g1 = run_dir.join("gap-finder-1");
    fs::create_dir_all(&s1).unwrap();
    fs::create_dir_all(&g1).unwrap();
    fs::write(s1.join("output.md"), "SCOUT1").unwrap();
    fs::write(g1.join("output.md"), "GAP1").unwrap();

    let fake = Arc::new(FakeProvider::new("fake", three_artifact_turns()));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let (spec, outcome) = run_synthesist(
        &run_dir,
        "SPEC",
        "PLAN",
        &[s1],
        &[g1],
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
    )
    .await
    .unwrap();
    assert_eq!(spec.name, "synthesist");
    assert!(synth_dir.join("output.md").exists());
    assert!(synth_dir.join("manifest.yaml").exists());
    assert!(synth_dir.join("verification.md").exists());
    assert_eq!(outcome.stop, WorkerStop::EndTurn);
    // The prompt inlined the scout + gap-finder outputs.
    assert!(spec.prompt.contains("SCOUT1"));
    assert!(spec.prompt.contains("GAP1"));
    assert!(spec.prompt.contains("SPEC"));
    assert!(spec.prompt.contains("PLAN"));
}

#[test]
fn finalize_run_copies_output_and_creates_symlink() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    let run_dir = research_base.join("runs/2026-06-27-0315-a1b2c3");
    let synth = run_dir.join("synthesist");
    fs::create_dir_all(&synth).unwrap();
    fs::write(synth.join("output.md"), "FINAL OUTPUT").unwrap();
    let spec_path = research_base.join("specs/my-spec.md");
    fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
    fs::write(&spec_path, "spec body").unwrap();

    let link = finalize_run(&run_dir, &spec_path, &research_base).unwrap();
    assert_eq!(link, research_base.join("specs/my-spec-latest.md"));
    // Run-root output.md is the copy of synthesist/output.md.
    assert_eq!(
        fs::read_to_string(run_dir.join("output.md")).unwrap(),
        "FINAL OUTPUT"
    );
    // Symlink resolves to the run-root output.md.
    assert_eq!(fs::read_to_string(&link).unwrap(), "FINAL OUTPUT");
    // Re-running replaces the existing symlink (no error).
    finalize_run(&run_dir, &spec_path, &research_base).unwrap();
    assert_eq!(fs::read_to_string(&link).unwrap(), "FINAL OUTPUT");
}

#[tokio::test]
async fn full_gap_finding_integration_test() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    // 4 workers (2 scouts, 1 gap-finder, 1 synthesist) × 4 turns = 16 turns.
    let fake = Arc::new(FakeProvider::new("fake", run_turns(4)));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let orch = Orchestrator::new(
        OrchestratorConfig {
            research_base: research_base.clone(),
            agents_dir: fixture_agents_dir(),
            default_model: "fake-model".into(),
            max_parallel: 1,
        },
        provider,
    );
    // Put the spec inside research_base/specs/ so finalize_run's symlink lands
    // next to it, mirroring the real layout.
    let spec_path = research_base.join("specs/gap-finding-spec.md");
    fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
    fs::copy(fixture_spec_path(), &spec_path).unwrap();

    let out = orch
        .execute(&spec_path, &fixture_plan_path(), "2026-06-27-0315-a1b2c3")
        .await
        .unwrap();
    let run_dir = out.run_dir.clone();

    // Full run tree.
    assert!(run_dir.join("swarm-state.yaml").exists());
    assert!(run_dir.join("literature-scout-1/output.md").exists());
    assert!(run_dir.join("literature-scout-2/output.md").exists());
    assert!(run_dir.join("gap-finder-1/output.md").exists());
    assert!(run_dir.join("synthesist/output.md").exists());
    assert!(run_dir.join("bibliography.md").exists());
    assert!(run_dir.join("gaps.md").exists());
    // Run-root output.md + spec-latest symlink.
    assert!(run_dir.join("output.md").exists());
    let link = research_base.join("specs/gap-finding-spec-latest.md");
    assert!(link.exists());
    assert_eq!(
        fs::read_to_string(&link).unwrap(),
        fs::read_to_string(run_dir.join("output.md")).unwrap()
    );

    // Swarm-state: all six phases accounted for.
    let swarm = SwarmState::read(&run_dir.join("swarm-state.yaml")).unwrap();
    let by_name: std::collections::HashMap<&str, &str> = swarm
        .phases
        .iter()
        .map(|p| (p.name.as_str(), p.status.as_str()))
        .collect();
    assert_eq!(by_name["literature-scout"], "complete");
    assert_eq!(by_name["gap-finder"], "complete");
    assert_eq!(by_name["hypothesis-smith"], "skipped");
    assert_eq!(by_name["red-team"], "skipped");
    assert_eq!(by_name["eval-designer"], "skipped");
    assert_eq!(by_name["synthesist"], "complete");
    assert!(out.escalations.is_empty());
}

use megaresearcher_research::orchestrator::gaps::{collect_gaps, parse_gaps};

#[test]
fn parse_gaps_reads_structured_gaps_list_from_manifest() {
    let manifest = "\
role: gap-finder
slice: scouts 1-2
gaps_count: 2
discarded_count: 1
gaps:
  - id: gap-1
    statement: Technique X never applied to A+B fusion.
    type: unexplored-intersection
  - id: gap-2
    statement: Paper P and paper Q disagree on metric M.
    type: contradiction
";
    let finder_dir = PathBuf::from("/tmp/run/gap-finder-1");
    let gaps = parse_gaps(manifest, "gap-finder-1", &finder_dir);
    assert_eq!(gaps.len(), 2);
    assert_eq!(gaps[0].id, "gap-1");
    assert_eq!(gaps[0].finder_name, "gap-finder-1");
    assert_eq!(gaps[0].finder_dir, finder_dir);
    assert_eq!(
        gaps[0].statement,
        "Technique X never applied to A+B fusion."
    );
    assert_eq!(gaps[0].gap_type, "unexplored-intersection");
    assert_eq!(gaps[1].id, "gap-2");
    assert_eq!(gaps[1].gap_type, "contradiction");
}

#[test]
fn parse_gaps_returns_empty_when_manifest_has_no_gaps_block() {
    // A v0-style manifest with only gaps_count (no structured `gaps:` list) yields
    // zero gaps — the orchestrator then dispatches no smiths. serde(default) makes
    // the missing field parse cleanly.
    let manifest = "role: gap-finder\ngaps_count: 3\ndiscarded_count: 1\n";
    let gaps = parse_gaps(manifest, "gap-finder-2", Path::new("/tmp/r/gap-finder-2"));
    assert!(gaps.is_empty());
}

#[test]
fn parse_gaps_skips_gaps_with_empty_statement() {
    let manifest = "\
gaps:
  - id: gap-1
    statement: A defensible gap.
    type: contradiction
  - id: gap-2
    statement: ''
    type: missing-baseline
";
    let gaps = parse_gaps(manifest, "gap-finder-1", Path::new("/tmp/r/gap-finder-1"));
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].id, "gap-1");
}

#[test]
fn collect_gaps_aggregates_across_gap_finders_in_order() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let g1 = run_dir.join("gap-finder-1");
    let g2 = run_dir.join("gap-finder-2");
    fs::create_dir_all(&g1).unwrap();
    fs::create_dir_all(&g2).unwrap();
    fs::write(
        g1.join("manifest.yaml"),
        "role: gap-finder\ngaps:\n  - id: gap-1\n    statement: A.\n    type: contradiction\n  - id: gap-2\n    statement: B.\n    type: missing-baseline\n",
    )
    .unwrap();
    fs::write(
        g2.join("manifest.yaml"),
        "role: gap-finder\ngaps:\n  - id: gap-1\n    statement: C.\n    type: unexplored-intersection\n",
    )
    .unwrap();
    let gaps = collect_gaps(&run_dir, &[g1.clone(), g2.clone()]).unwrap();
    assert_eq!(gaps.len(), 3);
    // finder-then-gap order; ids are finder-local, so "gap-1" repeats across finders.
    assert_eq!(gaps[0].finder_name, "gap-finder-1");
    assert_eq!(gaps[0].statement, "A.");
    assert_eq!(gaps[1].finder_name, "gap-finder-1");
    assert_eq!(gaps[1].statement, "B.");
    assert_eq!(gaps[2].finder_name, "gap-finder-2");
    assert_eq!(gaps[2].statement, "C.");
    assert_eq!(gaps[2].finder_dir, g2);
}

#[test]
fn collect_gaps_skips_finders_whose_manifest_is_unreadable() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let g1 = run_dir.join("gap-finder-1");
    let g2 = run_dir.join("gap-finder-2"); // no manifest written
    fs::create_dir_all(&g1).unwrap();
    fs::create_dir_all(&g2).unwrap();
    fs::write(
        g1.join("manifest.yaml"),
        "role: gap-finder\ngaps:\n  - id: gap-1\n    statement: A.\n    type: contradiction\n",
    )
    .unwrap();
    // g2 has no manifest.yaml -> collect_gaps skips it (no error, no gaps from it).
    let gaps = collect_gaps(&run_dir, &[g1, g2]).unwrap();
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].statement, "A.");
}
