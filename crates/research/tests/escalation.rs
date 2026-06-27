//! Escalation-adjudication seam tests (Task 3). The fixture — a single
//! `final_turn("nothing written")` — makes every scout miss all three
//! artifacts, so `verify_wave` escalates. We vary only `OrchestratorConfig`
//! `escalation`: `None` (byte-identical pre-6a behavior), `Some(ContinueAll)`
//! (record + partial `RunOutcome`), and `Some(FailAll)` (explicit fail).

mod common;

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use claurst_api::{LlmProvider, StopReason, StreamEvent};
use claurst_core::types::{ContentBlock, UsageInfo};
use tempfile::tempdir;

use common::fake_provider::FakeProvider;
use megaresearcher_research::orchestrator::escalation::{EscalationHandler, EscalationVerdict};
use megaresearcher_research::orchestrator::{Orchestrator, OrchestratorConfig, OrchestratorError};
use megaresearcher_research::state::swarm_state::Escalation;

fn fixture_agents_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/agents")
}
fn fixture_spec_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/specs/gap-finding-spec.md")
}
fn fixture_plan_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plans/gap-finding-plan.md")
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

struct ContinueAll;

#[async_trait]
impl EscalationHandler for ContinueAll {
    async fn adjudicate(&self, _e: &Escalation) -> EscalationVerdict {
        EscalationVerdict::Continue
    }
}

fn build_orch(
    research_base: PathBuf,
    escalation: Option<Arc<dyn EscalationHandler>>,
) -> (Orchestrator, Arc<FakeProvider>) {
    let turns: Vec<Vec<StreamEvent>> = vec![final_turn("nothing written")];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let orch = Orchestrator::new(
        OrchestratorConfig {
            research_base,
            agents_dir: fixture_agents_dir(),
            default_model: "fake-model".into(),
            max_parallel: 1,
            mcp: None,
            escalation,
        },
        provider,
    );
    (orch, fake)
}

#[tokio::test]
async fn none_handler_preserves_err_behavior() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    let (orch, fake) = build_orch(research_base, None);
    let err = orch
        .execute(&fixture_spec_path(), &fixture_plan_path(), "ridN")
        .await
        .unwrap_err();
    // The provider was actually exercised before the escalation halted.
    assert!(fake.call_count() > 0, "provider should have been called");
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

#[tokio::test]
async fn continue_handler_returns_ok_with_escalations_listed() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    let (orch, fake) = build_orch(research_base, Some(Arc::new(ContinueAll)));
    let out = orch
        .execute(&fixture_spec_path(), &fixture_plan_path(), "ridC")
        .await
        .expect("Continue handler should return Ok with partial RunOutcome");
    assert!(fake.call_count() > 0, "provider should have been called");
    assert!(
        out.escalations.contains(&"literature-scout-1".to_string()),
        "expected literature-scout-1 in escalations, got {:?}",
        out.escalations
    );
    assert!(
        out.phase_statuses
            .iter()
            .any(|(n, _)| n == "literature-scout"),
        "expected literature-scout phase in phase_statuses, got {:?}",
        out.phase_statuses
    );
}

#[tokio::test]
async fn fail_handler_returns_err() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    let (orch, fake) = build_orch(
        research_base,
        Some(Arc::new(
            megaresearcher_research::orchestrator::escalation::FailAll,
        )),
    );
    let err = orch
        .execute(&fixture_spec_path(), &fixture_plan_path(), "ridF")
        .await
        .unwrap_err();
    assert!(fake.call_count() > 0, "provider should have been called");
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
