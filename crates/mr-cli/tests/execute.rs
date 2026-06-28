//! Integration tests for `mr execute` (T7): the fake-provider gap-finding run
//! and the `HeadlessEscalationHandler` unit tests.

mod common;

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tempfile::tempdir;

use common::fake_provider::FakeProvider;
use common::turns::run_turns;

use claurst_api::LlmProvider;
use megaresearcher_research::orchestrator::escalation::{EscalationHandler, EscalationVerdict};
use megaresearcher_research::phases::UserIo;
use megaresearcher_research::state::swarm_state::Escalation;
use mr_cli::commands::execute::run_with;
use mr_cli::escalation::HeadlessEscalationHandler;
use mr_cli::OnEscalate;

/// Stage a temp cwd with the gap-finding fixtures: minimal agent stubs
/// (preflight only checks existence), plus the spec + plan from embedded
/// fixtures. Returns the staged cwd path.
fn stage_cwd(tmp: &tempfile::TempDir) -> PathBuf {
    let cwd = tmp.path().to_path_buf();
    let agents = cwd.join("agents");
    fs::create_dir_all(&agents).unwrap();
    // Minimal valid agent stubs: prompt_asset::load (called by dispatch) requires
    // YAML frontmatter with name/description/model. FakeProvider ignores the body.
    for role in ["literature-scout", "gap-finder", "synthesist"] {
        fs::write(
            agents.join(format!("{role}.md")),
            format!("---\nname: {role}\ndescription: stub\nmodel: inherit\n---\n\nbody\n"),
        )
        .unwrap();
    }
    let plans = cwd.join("docs/research/plans");
    let specs = cwd.join("docs/research/specs");
    fs::create_dir_all(&plans).unwrap();
    fs::create_dir_all(&specs).unwrap();
    fs::write(
        plans.join("gap-finding-plan.md"),
        include_str!("fixtures/plans/gap-finding-plan.md"),
    )
    .unwrap();
    fs::write(
        specs.join("gap-finding-spec.md"),
        include_str!("fixtures/specs/gap-finding-spec.md"),
    )
    .unwrap();
    cwd
}

#[tokio::test]
async fn execute_no_mcp_with_fake_provider_produces_output() {
    let tmp = tempdir().unwrap();
    let cwd = stage_cwd(&tmp);
    // 4 workers × 4 turns = 16 turns. FakeProvider re-emits the last turn if
    // more are requested, so a turn-count mismatch degrades rather than panics.
    let turns = run_turns(4);
    let provider = (
        Arc::new(FakeProvider::new("fake", turns)) as Arc<dyn LlmProvider>,
        "fake-model".to_string(),
    );
    run_with(
        &cwd,
        provider,
        /*plan=*/ None,
        /*paper=*/ false,
        /*headless=*/ false,
        /*no_mcp=*/ true,
        OnEscalate::Fail,
    )
    .await
    .unwrap();

    // run dir is docs/research/runs/<run_id> — discover it (run_id is a generated timestamp).
    let runs = cwd.join("docs/research/runs");
    let entries: Vec<_> = fs::read_dir(&runs)
        .unwrap()
        .flatten()
        .filter(|e| e.path().is_dir())
        .collect();
    assert_eq!(entries.len(), 1, "expected exactly one run dir");
    let run_dir = entries[0].path();
    assert!(
        run_dir.join("output.md").exists(),
        "output.md missing in {}",
        run_dir.display()
    );
    assert!(run_dir.join("swarm-state.yaml").exists());
    assert!(
        run_dir.join("verification-report.md").exists(),
        "auto-verify should write the report"
    );
}

/// A `UserIo` that returns "y" for every read_line — used to exercise the
/// Pause arm of the handler.
struct FakeUserIo;

#[async_trait]
impl UserIo for FakeUserIo {
    async fn print(&self, _t: &str) -> std::io::Result<()> {
        Ok(())
    }
    async fn read_line(&self) -> std::io::Result<String> {
        Ok("y".into())
    }
}

#[tokio::test]
async fn headless_handler_continue_always_continues() {
    let h = HeadlessEscalationHandler {
        mode: OnEscalate::Continue,
        io: Arc::new(FakeUserIo),
    };
    let v = h
        .adjudicate(&Escalation {
            worker: "w".into(),
            reason: "r".into(),
            retry_count: 1,
        })
        .await;
    assert_eq!(v, EscalationVerdict::Continue);
}

#[tokio::test]
async fn headless_handler_fail_always_fails() {
    let h = HeadlessEscalationHandler {
        mode: OnEscalate::Fail,
        io: Arc::new(FakeUserIo),
    };
    let v = h
        .adjudicate(&Escalation {
            worker: "w".into(),
            reason: "r".into(),
            retry_count: 1,
        })
        .await;
    assert_eq!(v, EscalationVerdict::Fail);
}
