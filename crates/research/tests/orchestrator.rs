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
    let outcomes = dispatch_wave(specs, &fixture_agents_dir(), provider, "fake-model", 1, &[])
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
    let outcome = run_worker(
        &spec,
        &fixture_agents_dir(),
        provider,
        "resolved-model",
        &[],
    )
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
    let outcomes = [run_worker(
        &spec,
        &fixture_agents_dir(),
        provider.clone(),
        "fake-model",
        &[],
    )
    .await
    .unwrap()];
    let outcomes = vec![("literature-scout-1".to_string(), outcomes[0].clone())];
    let gate = verify_wave(
        outcomes,
        std::slice::from_ref(&spec),
        &fixture_agents_dir(),
        provider,
        "fake-model",
        &[],
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
    let first = run_worker(
        &spec,
        &fixture_agents_dir(),
        provider.clone(),
        "fake-model",
        &[],
    )
    .await
    .unwrap();
    let outcomes = vec![("literature-scout-1".to_string(), first)];
    let gate = verify_wave(
        outcomes,
        std::slice::from_ref(&spec),
        &fixture_agents_dir(),
        provider,
        "fake-model",
        &[],
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
    let first = run_worker(
        &spec,
        &fixture_agents_dir(),
        provider.clone(),
        "fake-model",
        &[],
    )
    .await
    .unwrap();
    let outcomes = vec![("literature-scout-1".to_string(), first)];
    let gate = verify_wave(
        outcomes,
        std::slice::from_ref(&spec),
        &fixture_agents_dir(),
        provider,
        "fake-model",
        &[],
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
        &[],
        &[],
        &[],
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

use megaresearcher_research::orchestrator::verdict::{parse_redteam_verdict, RedTeamVerdict};

#[test]
fn parse_redteam_verdict_approve() {
    let md = "# Red-team critique\n\n1. **Verdict** — APPROVE\n\n2. Gap re-verification: ...\n";
    assert_eq!(parse_redteam_verdict(md), Some(RedTeamVerdict::Approve));
}

#[test]
fn parse_redteam_verdict_reject_with_revision_number() {
    let md = "## Verdict\n\n**Verdict** — REJECT (revision-2)\n\nThe mechanism is unsupported.\n";
    assert_eq!(
        parse_redteam_verdict(md),
        Some(RedTeamVerdict::Reject { revision: 2 })
    );
}

#[test]
fn parse_redteam_verdict_kill() {
    let md = "1. **Verdict** — KILL (irrecoverable)\n\nThe gap is not actually unexplored.\n";
    assert_eq!(parse_redteam_verdict(md), Some(RedTeamVerdict::Kill));
}

#[test]
fn parse_redteam_verdict_tolerates_spacing_and_dash_variants() {
    // Workers vary the em-dash / colon / spacing. Accept any of "—", "-", ":".
    let cases = [
        "Verdict: APPROVE",
        "Verdict - APPROVE",
        "**Verdict** — APPROVE",
        "1. **Verdict** — REJECT (revision-1)",
    ];
    assert_eq!(
        parse_redteam_verdict(cases[0]),
        Some(RedTeamVerdict::Approve)
    );
    assert_eq!(
        parse_redteam_verdict(cases[1]),
        Some(RedTeamVerdict::Approve)
    );
    assert_eq!(
        parse_redteam_verdict(cases[2]),
        Some(RedTeamVerdict::Approve)
    );
    assert_eq!(
        parse_redteam_verdict(cases[3]),
        Some(RedTeamVerdict::Reject { revision: 1 })
    );
}

#[test]
fn parse_redteam_verdict_returns_none_when_no_verdict_line() {
    let md = "# Red-team critique\n\nNo verdict here, just discussion.\n";
    assert_eq!(parse_redteam_verdict(md), None);
}

#[test]
fn parse_redteam_verdict_returns_none_for_format_description_lines() {
    // A worker that echoes the agent's format instruction ("exactly one of: APPROVE | ...")
    // must NOT be misread as an APPROVE verdict.
    let md =
        "The verdict is exactly one of: APPROVE | REJECT (revision-N) | KILL (irrecoverable).\n";
    assert_eq!(parse_redteam_verdict(md), None);
}

#[test]
fn parse_redteam_verdict_file_reads_disk() {
    let tmp = tempdir().unwrap();
    let p = tmp.path().join("red-team-1-r1").join("output.md");
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(&p, "1. **Verdict** — REJECT (revision-1)\n").unwrap();
    let v = megaresearcher_research::orchestrator::verdict::parse_redteam_verdict_file(&p).unwrap();
    assert_eq!(v, Some(RedTeamVerdict::Reject { revision: 1 }));
}

use megaresearcher_research::orchestrator::gaps::Gap;
use megaresearcher_research::orchestrator::hypothesis::{
    build_smith_spec, dispatch_hypothesis_smiths, redispatch_smith_revision, Hypothesis,
};

fn gap(id: &str, finder: &str, finder_dir: &Path, statement: &str) -> Gap {
    Gap {
        id: id.into(),
        finder_name: finder.into(),
        finder_dir: finder_dir.to_path_buf(),
        statement: statement.into(),
        gap_type: "contradiction".into(),
    }
}

#[test]
fn build_smith_spec_inlines_gap_and_finder_output() {
    let tmp = tempdir().unwrap();
    let finder_dir = tmp.path().join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER OUTPUT BODY").unwrap();
    let g = gap(
        "gap-1",
        "gap-finder-1",
        &finder_dir,
        "Technique X never applied to A+B.",
    );
    let spec = build_smith_spec(
        "SPEC TEXT",
        &g,
        "hypothesis-smith-1",
        &tmp.path().join("hypothesis-smith-1"),
        tmp.path(),
        None,
    );
    assert_eq!(spec.name, "hypothesis-smith-1");
    assert_eq!(spec.role, "hypothesis-smith");
    assert!(spec.prompt.contains("SPEC TEXT"));
    assert!(spec.prompt.contains("Technique X never applied to A+B."));
    assert!(spec.prompt.contains("FINDER OUTPUT BODY"));
    assert!(spec.prompt.contains("hypothesis-smith-1"));
    // No revision prior on initial dispatch.
    assert!(!spec.prompt.contains("Previous red-team critique"));
}

#[test]
fn build_smith_spec_revision_appends_redteam_prior() {
    let tmp = tempdir().unwrap();
    let finder_dir = tmp.path().join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER BODY").unwrap();
    let g = gap("gap-1", "gap-finder-1", &finder_dir, "The gap statement.");
    let spec = build_smith_spec(
        "SPEC",
        &g,
        "hypothesis-smith-2",
        &tmp.path().join("hypothesis-smith-2"),
        tmp.path(),
        Some("RED-TEAM CRITIQUE BODY"),
    );
    assert!(spec.prompt.contains("Previous red-team critique"));
    assert!(spec.prompt.contains("RED-TEAM CRITIQUE BODY"));
}

#[tokio::test]
async fn dispatch_hypothesis_smiths_one_per_gap_and_gates() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let finder_dir = run_dir.join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER BODY").unwrap();
    let gaps = vec![
        gap("gap-1", "gap-finder-1", &finder_dir, "Gap one statement."),
        gap("gap-2", "gap-finder-1", &finder_dir, "Gap two statement."),
    ];
    // 2 smiths x 4 scripted turns = 8 turns.
    let fake = Arc::new(FakeProvider::new("fake", run_turns(2)));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let hyps = dispatch_hypothesis_smiths(
        &run_dir,
        "SPEC",
        &gaps,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
    )
    .await
    .unwrap();
    assert_eq!(hyps.len(), 2);
    assert_eq!(hyps[0].name, "hypothesis-smith-1");
    assert_eq!(hyps[1].name, "hypothesis-smith-2");
    // Each smith wrote all three artifacts.
    for h in &hyps {
        assert!(h.dir.join("output.md").exists());
        assert!(h.dir.join("manifest.yaml").exists());
        assert!(h.dir.join("verification.md").exists());
    }
    assert_eq!(fake.call_count(), 8);
}

#[tokio::test]
async fn dispatch_hypothesis_smiths_halts_on_gate_escalation() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let finder_dir = run_dir.join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER BODY").unwrap();
    let gaps = vec![gap("gap-1", "gap-finder-1", &finder_dir, "Gap one.")];
    // A single final-only turn: the smith writes nothing, the gate retries, the
    // retry also writes nothing (.or_else(last) re-emits the same empty turn),
    // so the smith escalates.
    let turns: Vec<Vec<StreamEvent>> = vec![final_turn("nothing written")];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let err = dispatch_hypothesis_smiths(
        &run_dir,
        "SPEC",
        &gaps,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
    )
    .await
    .expect_err("should escalate");
    match err {
        OrchestratorError::Escalated(names) => {
            assert_eq!(names, vec!["hypothesis-smith-1".to_string()])
        }
        other => panic!("expected Escalated, got {other:?}"),
    }
}

#[tokio::test]
async fn redispatch_smith_revision_overwrites_output_and_gates() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let finder_dir = run_dir.join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER BODY").unwrap();
    let g = gap("gap-1", "gap-finder-1", &finder_dir, "The gap.");
    let hyp = Hypothesis {
        name: "hypothesis-smith-1".into(),
        dir: run_dir.join("hypothesis-smith-1"),
        gap: g,
    };
    fs::create_dir_all(&hyp.dir).unwrap();
    fs::write(hyp.dir.join("output.md"), "OLD INITIAL HYPOTHESIS").unwrap();
    // 4 turns for the revision dispatch.
    let fake = Arc::new(FakeProvider::new("fake", run_turns(1)));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    redispatch_smith_revision(
        &hyp,
        "SPEC",
        "RED-TEAM CRITIQUE",
        &fixture_agents_dir(),
        provider,
        "fake-model",
    )
    .await
    .unwrap();
    // The revised output overwrote the old one (the fake writes "# Output\n\ncontent").
    assert_ne!(
        fs::read_to_string(hyp.dir.join("output.md")).unwrap(),
        "OLD INITIAL HYPOTHESIS"
    );
    assert!(hyp.dir.join("manifest.yaml").exists());
    assert_eq!(fake.call_count(), 4);
}

use std::collections::HashMap;

use megaresearcher_research::orchestrator::redteam::{build_redteam_prompt, run_redteam_loop};

// Helper: scripted turns that write output.md with a given verdict line plus
// the standard manifest + verification, then EndTurn.
fn redteam_turns(verdict_line: &str) -> Vec<Vec<StreamEvent>> {
    let output =
        format!("# Red-team critique\n\n1. **Verdict** — {verdict_line}\n\n2. Discussion.\n");
    vec![
        write_turn("output.md", &output),
        write_turn(
            "manifest.yaml",
            "role: red-team\nverdict: APPROVE\nrevision_round: 1\n",
        ),
        write_turn("verification.md", "# Verification\n\nok"),
        final_turn("Done."),
    ]
}

#[test]
fn build_redteam_prompt_inlines_hypothesis_and_finder_output() {
    let tmp = tempdir().unwrap();
    let spec = build_redteam_prompt(
        "SPEC TEXT",
        "HYPOTHESIS OUTPUT BODY",
        "GAP-FINDER OUTPUT BODY",
        &tmp.path().join("red-team-1-r1"),
    );
    assert_eq!(spec.name, "red-team-1-r1");
    assert_eq!(spec.role, "red-team");
    assert!(spec.prompt.contains("SPEC TEXT"));
    assert!(spec.prompt.contains("HYPOTHESIS OUTPUT BODY"));
    assert!(spec.prompt.contains("GAP-FINDER OUTPUT BODY"));
}

#[tokio::test]
async fn redteam_loop_approves_on_first_round() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    // 1 hypothesis, 1 red-team round APPROVE -> 4 turns.
    let fake = Arc::new(FakeProvider::new("fake", redteam_turns("APPROVE")));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_redteam_loop(
        &run_dir,
        "SPEC",
        hyps,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
        &mut swarm,
    )
    .await
    .unwrap();
    assert_eq!(res.survivors.len(), 1);
    assert!(res.killed.is_empty());
    assert_eq!(res.redteam_dirs.len(), 1);
    assert!(res.redteam_dirs[0].ends_with("red-team-1-r1"));
    assert!(swarm.escalations.is_empty());
    assert_eq!(swarm.retry_counts.get("hypothesis-smith-1"), None);
}

#[tokio::test]
async fn redteam_loop_revises_then_approves() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    // Round 1: REJECT (revision-1) -> revise smith (4 turns) + round 2 red-team APPROVE (4 turns).
    // Plus the initial round-1 red-team (4 turns) = 12 turns total.
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = redteam_turns("REJECT (revision-1)");
        // revision smith: 4 artifact turns (writes a revised output.md).
        t.extend(run_turns(1));
        // round-2 red-team: APPROVE.
        t.extend(redteam_turns("APPROVE"));
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_redteam_loop(
        &run_dir,
        "SPEC",
        hyps,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
        &mut swarm,
    )
    .await
    .unwrap();
    assert_eq!(res.survivors.len(), 1);
    assert!(res.killed.is_empty());
    assert_eq!(res.redteam_dirs.len(), 2);
    assert!(res.redteam_dirs[0].ends_with("red-team-1-r1"));
    assert!(res.redteam_dirs[1].ends_with("red-team-1-r2"));
    assert_eq!(swarm.retry_counts.get("hypothesis-smith-1"), Some(&1));
    assert!(swarm.escalations.is_empty());
}

#[tokio::test]
async fn redteam_loop_kills_on_kill_verdict() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    let fake = Arc::new(FakeProvider::new(
        "fake",
        redteam_turns("KILL (irrecoverable)"),
    ));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_redteam_loop(
        &run_dir,
        "SPEC",
        hyps,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
        &mut swarm,
    )
    .await
    .unwrap();
    assert!(res.survivors.is_empty());
    assert_eq!(res.killed, vec!["hypothesis-smith-1".to_string()]);
    assert_eq!(res.redteam_dirs.len(), 1);
    assert_eq!(swarm.escalations.len(), 1);
    assert_eq!(swarm.escalations[0].worker, "hypothesis-smith-1");
    assert!(swarm.escalations[0].reason.contains("KILL"));
}

#[tokio::test]
async fn redteam_loop_escalates_after_three_revisions() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    // Rounds 1-3: REJECT (revision-1/2/3) each -> revise smith; round 4 would be
    // revision_count==3 -> escalate WITHOUT dispatching a 4th red-team.
    // Turn sequence: r1-redteam(4) + revise(4) + r2-redteam(4) + revise(4) +
    // r3-redteam(4) + revise(4) = 24 turns. (No r4 red-team: cap reached at
    // revision_count==3 after the 3rd REJECT.)
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = Vec::new();
        for n in 1..=3 {
            t.extend(redteam_turns(&format!("REJECT (revision-{n})")));
            t.extend(run_turns(1)); // revision smith
        }
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_redteam_loop(
        &run_dir,
        "SPEC",
        hyps,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
        &mut swarm,
    )
    .await
    .unwrap();
    assert!(res.survivors.is_empty());
    assert_eq!(res.killed, vec!["hypothesis-smith-1".to_string()]);
    assert_eq!(res.redteam_dirs.len(), 3); // r1, r2, r3
    assert_eq!(swarm.retry_counts.get("hypothesis-smith-1"), Some(&3));
    assert_eq!(swarm.escalations.len(), 1);
    assert!(swarm.escalations[0]
        .reason
        .contains("exceeded 3 red-team revisions"));
}

// Shared fixture builder for the loop tests: one hypothesis in a run dir.
fn redteam_loop_fixture(n: usize) -> (SwarmState, Vec<Hypothesis>, PathBuf) {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let finder_dir = run_dir.join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER BODY").unwrap();
    let hyps: Vec<Hypothesis> = (1..=n)
        .map(|i| {
            let dir = run_dir.join(format!("hypothesis-smith-{i}"));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("output.md"), format!("HYPOTHESIS {i}")).unwrap();
            Hypothesis {
                name: format!("hypothesis-smith-{i}"),
                dir,
                gap: gap("gap-1", "gap-finder-1", &finder_dir, "The gap."),
            }
        })
        .collect();
    let swarm = SwarmState {
        run_id: "rid".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
        novelty_target: "hypothesis".into(),
        max_parallel: 1,
        phases: vec![],
        escalations: vec![],
        retry_counts: HashMap::new(),
    };
    (swarm, hyps, run_dir)
}

use megaresearcher_research::orchestrator::evaldesign::{
    build_evaldesigner_prompt, parse_intractable, run_eval_designers,
};

// Scripted turns for an eval-designer that writes a manifest with a given
// flagged_intractable value.
fn evaldesign_turns(intractable: bool) -> Vec<Vec<StreamEvent>> {
    let manifest = format!(
        "role: eval-designer\nhypothesis: h\ndatasets_count: 1\nbaselines_count: 3\nfalsification_experiments_count: 2\nflagged_intractable: {}\n",
        intractable
    );
    vec![
        write_turn("output.md", "# Eval plan\n\nThe experiment design."),
        write_turn("manifest.yaml", &manifest),
        write_turn("verification.md", "# Verification\n\nok"),
        final_turn("Done."),
    ]
}

#[test]
fn build_evaldesigner_prompt_inlines_hypothesis() {
    let tmp = tempdir().unwrap();
    let spec =
        build_evaldesigner_prompt("SPEC TEXT", "HYP BODY", &tmp.path().join("eval-designer-1"));
    assert_eq!(spec.name, "eval-designer-1");
    assert_eq!(spec.role, "eval-designer");
    assert!(spec.prompt.contains("SPEC TEXT"));
    assert!(spec.prompt.contains("HYP BODY"));
}

#[test]
fn parse_intractable_reads_flag() {
    assert!(!parse_intractable(
        "role: eval-designer\nflagged_intractable: false\n"
    ));
    assert!(parse_intractable(
        "role: eval-designer\nflagged_intractable: true\n"
    ));
    // Missing field defaults to false.
    assert!(!parse_intractable("role: eval-designer\n"));
}

#[tokio::test]
async fn eval_designers_one_per_survivor_and_gates() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(2);
    let survivors: Vec<Hypothesis> = hyps;
    // 2 eval-designers x 4 turns = 8 turns; neither intractable.
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = evaldesign_turns(false);
        t.extend(evaldesign_turns(false));
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_eval_designers(
        &run_dir,
        "SPEC",
        &survivors,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
        &mut swarm,
    )
    .await
    .unwrap();
    assert_eq!(res.eval_dirs.len(), 2);
    assert!(res.eval_dirs[0].ends_with("eval-designer-1"));
    assert!(res.eval_dirs[1].ends_with("eval-designer-2"));
    assert_eq!(
        res.phase_workers,
        vec![
            ("hypothesis-smith-1".to_string(), "passed".to_string()),
            ("hypothesis-smith-2".to_string(), "passed".to_string()),
        ]
    );
    assert!(swarm.escalations.is_empty());
}

#[tokio::test]
async fn eval_designer_intractable_is_escalated_but_kept_in_dirs() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    let survivors: Vec<Hypothesis> = hyps;
    let fake = Arc::new(FakeProvider::new("fake", evaldesign_turns(true)));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_eval_designers(
        &run_dir,
        "SPEC",
        &survivors,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
        &mut swarm,
    )
    .await
    .unwrap();
    assert_eq!(res.eval_dirs.len(), 1); // still included for the audit trail
    assert_eq!(
        res.phase_workers,
        vec![("hypothesis-smith-1".to_string(), "intractable".to_string())]
    );
    assert_eq!(swarm.escalations.len(), 1);
    assert_eq!(swarm.escalations[0].worker, "hypothesis-smith-1");
    assert!(swarm.escalations[0].reason.contains("intractable"));
}

#[tokio::test]
async fn eval_designers_halt_on_gate_escalation() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    let survivors: Vec<Hypothesis> = hyps;
    let turns: Vec<Vec<StreamEvent>> = vec![final_turn("nothing written")];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let err = run_eval_designers(
        &run_dir,
        "SPEC",
        &survivors,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
        &mut swarm,
    )
    .await
    .expect_err("should escalate");
    match err {
        OrchestratorError::Escalated(names) => {
            assert_eq!(names, vec!["hypothesis-smith-1".to_string()])
        }
        other => panic!("expected Escalated, got {other:?}"),
    }
}

fn fixture_hypothesis_spec_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/specs/hypothesis-spec.md")
}
fn fixture_hypothesis_plan_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plans/hypothesis-plan.md")
}

// Scripted turns for a gap-finder that writes a manifest with a structured
// gaps: list of `count` gaps.
fn gapfinder_turns_with_gaps(count: usize) -> Vec<Vec<StreamEvent>> {
    let mut gaps_yaml = String::from("gaps:\n");
    for i in 1..=count {
        gaps_yaml.push_str(&format!(
            "  - id: gap-{i}\n    statement: Gap {i} statement.\n    type: contradiction\n"
        ));
    }
    let manifest = format!("role: gap-finder\ngaps_count: {count}\n{gaps_yaml}");
    vec![
        write_turn("output.md", "# Gaps\n\nSome gaps identified."),
        write_turn("manifest.yaml", &manifest),
        write_turn("verification.md", "# Verification\n\nok"),
        final_turn("Done."),
    ]
}

#[tokio::test]
async fn execute_runs_full_hypothesis_path_minimal() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    let spec_path = research_base.join("specs/hypothesis-spec.md");
    fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
    fs::copy(fixture_hypothesis_spec_path(), &spec_path).unwrap();

    // 1 scout (4) + 1 gap-finder with 1 gap (4) + 1 smith (4) + 1 red-team APPROVE (4)
    // + 1 eval-designer not-intractable (4) + 1 synthesist (4) = 24 turns.
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = run_turns(1); // scout-1
        t.extend(gapfinder_turns_with_gaps(1)); // gap-finder-1 (writes 1 gap)
        t.extend(run_turns(1)); // hypothesis-smith-1
        t.extend(redteam_turns("APPROVE")); // red-team-1-r1
        t.extend(evaldesign_turns(false)); // eval-designer-1
        t.extend(run_turns(1)); // synthesist
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
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
    let out = orch
        .execute(
            &spec_path,
            &fixture_hypothesis_plan_path(),
            "2026-06-27-0400-d1e2f3",
        )
        .await
        .unwrap();
    let run_dir = out.run_dir.clone();
    let by_name: std::collections::HashMap<String, String> =
        out.phase_statuses.into_iter().collect();
    assert_eq!(by_name["literature-scout"], "complete");
    assert_eq!(by_name["gap-finder"], "complete");
    assert_eq!(by_name["hypothesis-smith"], "complete");
    assert_eq!(by_name["red-team"], "complete");
    assert_eq!(by_name["eval-designer"], "complete");
    assert_eq!(by_name["synthesist"], "complete");
    assert!(out.escalations.is_empty());
    // Run-root output.md + symlink resolve.
    assert!(run_dir.join("output.md").exists());
    let link = research_base.join("specs/hypothesis-spec-latest.md");
    assert!(link.exists());
    assert_eq!(
        fs::read_to_string(&link).unwrap(),
        fs::read_to_string(run_dir.join("output.md")).unwrap()
    );
    // The smith + red-team + eval-designer dirs exist.
    assert!(run_dir
        .join("hypothesis-smith-1")
        .join("output.md")
        .exists());
    assert!(run_dir.join("red-team-1-r1").join("output.md").exists());
    assert!(run_dir.join("eval-designer-1").join("output.md").exists());
}

#[tokio::test]
async fn full_hypothesis_integration_test_with_revision_loop() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    let spec_path = research_base.join("specs/hypothesis-spec.md");
    fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
    fs::copy(fixture_hypothesis_spec_path(), &spec_path).unwrap();

    // 1 scout + 1 gap-finder(2 gaps) + 2 smiths (Phase 3 wave) +
    // red-team-1-r1 APPROVE + red-team-2-r1 REJECT + smith-2 revision +
    // red-team-2-r2 APPROVE + 2 eval-designers + synthesist.
    //
    // Turn sequence (max_parallel=1, deterministic call order):
    //  1 scout-1            (4)
    //  2 gap-finder-1       (4) — manifest with 2 gaps
    //  3 hypothesis-smith-1  (4)
    //  4 hypothesis-smith-2 (4)   [Phase 3: both smiths in one wave]
    //  5 red-team-1-r1      (4) — APPROVE
    //  6 red-team-2-r1      (4) — REJECT (revision-1)
    //  7 hypothesis-smith-2 revision (4)
    //  8 red-team-2-r2      (4) — APPROVE
    //  9 eval-designer-1    (4)  [for survivor hypothesis-smith-1]
    // 10 eval-designer-2    (4)  [for survivor hypothesis-smith-2]
    // 11 synthesist         (4)
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = run_turns(1); // scout-1
        t.extend(gapfinder_turns_with_gaps(2)); // gap-finder-1 (2 gaps)
        t.extend(run_turns(2)); // smith-1, smith-2
        t.extend(redteam_turns("APPROVE")); // red-team-1-r1
        t.extend(redteam_turns("REJECT (revision-1)")); // red-team-2-r1
        t.extend(run_turns(1)); // smith-2 revision
        t.extend(redteam_turns("APPROVE")); // red-team-2-r2
        t.extend(evaldesign_turns(false)); // eval-designer-1
        t.extend(evaldesign_turns(false)); // eval-designer-2
        t.extend(run_turns(1)); // synthesist
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
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
    let out = orch
        .execute(
            &spec_path,
            &fixture_hypothesis_plan_path(),
            "2026-06-27-0430-a2b3c4",
        )
        .await
        .unwrap();
    let run_dir = out.run_dir.clone();
    let by_name: std::collections::HashMap<String, String> =
        out.phase_statuses.into_iter().collect();
    // All six phases complete; no critique phase skipped.
    assert_eq!(by_name["literature-scout"], "complete");
    assert_eq!(by_name["gap-finder"], "complete");
    assert_eq!(by_name["hypothesis-smith"], "complete");
    assert_eq!(by_name["red-team"], "complete");
    assert_eq!(by_name["eval-designer"], "complete");
    assert_eq!(by_name["synthesist"], "complete");
    assert!(out.escalations.is_empty());

    // Phase 4 produced two red-team rounds for hypothesis 2 (r1 reject, r2 approve)
    // and one for hypothesis 1.
    assert!(run_dir.join("red-team-1-r1").join("output.md").exists());
    assert!(run_dir.join("red-team-2-r1").join("output.md").exists());
    assert!(run_dir.join("red-team-2-r2").join("output.md").exists());
    // Two eval-designer dirs (one per survivor).
    assert!(run_dir.join("eval-designer-1").join("output.md").exists());
    assert!(run_dir.join("eval-designer-2").join("output.md").exists());
    // Run-root output.md + symlink resolve to it.
    assert!(run_dir.join("output.md").exists());
    let link = research_base.join("specs/hypothesis-spec-latest.md");
    assert!(link.exists());
    assert_eq!(
        fs::read_to_string(&link).unwrap(),
        fs::read_to_string(run_dir.join("output.md")).unwrap()
    );
    // The synthesist wrote its three artifacts.
    assert!(run_dir.join("synthesist").join("output.md").exists());
    assert!(run_dir.join("synthesist").join("manifest.yaml").exists());
}

#[tokio::test]
async fn full_hypothesis_integration_test_kill_halts_run() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    let spec_path = research_base.join("specs/hypothesis-spec.md");
    fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
    fs::copy(fixture_hypothesis_spec_path(), &spec_path).unwrap();

    // 1 scout + 1 gap-finder(1 gap) + 1 smith + 1 red-team KILL.
    // The red-team KILL escalates hypothesis-smith-1; all survivors gone ->
    // execute returns Err(Escalated(["hypothesis-smith-1"])).
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = run_turns(1); // scout-1
        t.extend(gapfinder_turns_with_gaps(1)); // gap-finder-1
        t.extend(run_turns(1)); // hypothesis-smith-1
        t.extend(redteam_turns("KILL (irrecoverable)")); // red-team-1-r1
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
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
    let err = orch
        .execute(
            &spec_path,
            &fixture_hypothesis_plan_path(),
            "2026-06-27-0500-f4e5d6",
        )
        .await
        .expect_err("kill should escalate and halt");
    match err {
        OrchestratorError::Escalated(names) => {
            assert_eq!(names, vec!["hypothesis-smith-1".to_string()]);
        }
        other => panic!("expected Escalated, got {other:?}"),
    }
}
