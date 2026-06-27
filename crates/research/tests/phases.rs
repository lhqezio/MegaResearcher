//! Phase 6a front-half: `GuidedSession` + `drive_session` approval-gate driver.
//! Reuses the existing `FakeProvider` test infra (scripted `create_message_stream`).

mod common;

use std::sync::Arc;

use async_trait::async_trait;
use claurst_api::{StopReason, StreamEvent};
use claurst_core::types::{ContentBlock, UsageInfo};
use serde_json::json;
use tempfile::tempdir;

use common::fake_provider::FakeProvider;
use megaresearcher_research::phases::{drive_session, DriveOutcome, Gate, GuidedSession, UserIo};
use megaresearcher_research::worker_tools::ScopedWrite;

/// A turn that emits a `Write` tool-use for `file` with `content`, plus a text
/// block. Mirrors `tests/orchestrator.rs:write_turn` verbatim.
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

/// A turn that emits one assistant text block and ends the turn. Mirrors
/// `tests/orchestrator.rs:final_turn` verbatim.
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

/// A fake `UserIo` that pops scripted lines and records printed text.
struct FakeUserIo {
    lines: std::sync::Mutex<std::collections::VecDeque<String>>,
    printed: std::sync::Mutex<Vec<String>>,
}

impl FakeUserIo {
    fn new(lines: Vec<String>) -> Self {
        Self {
            lines: std::sync::Mutex::new(lines.into_iter().collect()),
            printed: std::sync::Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl UserIo for FakeUserIo {
    async fn print(&self, text: &str) -> std::io::Result<()> {
        self.printed.lock().unwrap().push(text.to_string());
        Ok(())
    }
    async fn read_line(&self) -> std::io::Result<String> {
        self.lines
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "no input"))
    }
}

#[tokio::test]
async fn brainstorm_session_converges_on_approve() {
    // No gates, no tools: two text turns, two user lines, approval ends it.
    let provider = Arc::new(FakeProvider::new(
        "fake",
        vec![
            final_turn("What is your novelty target?"),
            final_turn("Restating: hypothesis, EO/IR fusion, open datasets only. Approve?"),
        ],
    ));
    let mut session = GuidedSession::new("flow body", vec![], provider, "fake", 1024, 10);
    let io = FakeUserIo::new(vec!["hypothesis".into(), "approve".into()]);
    let outcome = drive_session(&mut session, &io, vec![], &["approve", "yes", "y"])
        .await
        .unwrap();
    assert!(matches!(
        outcome,
        DriveOutcome::Approved { gates_passed: 0 }
    ));
}

#[tokio::test]
async fn spec_session_writes_artifact_then_advances_gate() {
    let dir = tempdir().unwrap();
    // The ScopedWrite jail resolves relative paths under dir.path(), so the
    // gate artifact must point at the joined path.
    std::fs::create_dir_all(dir.path().join("specs")).unwrap();
    let spec_path = dir.path().join("specs").join("test-spec.md");
    // Turn 1: a ToolUse writing the spec file; Turn 2: assistant says done.
    let provider = Arc::new(FakeProvider::new(
        "fake",
        vec![
            write_turn("specs/test-spec.md", "# Spec\n"),
            final_turn("Spec written to specs/test-spec.md. Approve?"),
        ],
    ));
    let provider_handle = Arc::clone(&provider);
    let tools: Vec<Arc<dyn megaresearcher_research::worker_tools::Tool>> =
        vec![Arc::new(ScopedWrite::new(dir.path()))];
    let mut session = GuidedSession::new("write the spec", tools, provider, "fake", 1024, 10);
    let io = FakeUserIo::new(vec!["approve".into()]);
    let outcome = drive_session(
        &mut session,
        &io,
        vec![Gate {
            artifact: spec_path.clone(),
            label: "spec".into(),
        }],
        &["approve", "yes"],
    )
    .await
    .unwrap();
    assert!(matches!(
        outcome,
        DriveOutcome::Approved { gates_passed: 1 }
    ));
    assert!(spec_path.exists());
    // Two scripted turns consumed: the tool-use turn and the final text turn.
    assert_eq!(provider_handle.call_count(), 2);
}

#[tokio::test]
async fn approve_before_artifact_exists_is_refused() {
    let provider = Arc::new(FakeProvider::new(
        "fake",
        vec![
            final_turn("Drafting... approve?"),
            final_turn("OK, writing now. Approve?"),
        ],
    ));
    let mut session = GuidedSession::new("flow", vec![], provider, "fake", 1024, 10);
    let io = FakeUserIo::new(vec!["approve".into(), "approve".into()]);
    let dir = tempdir().unwrap();
    let missing = dir.path().join("nope.md");
    let outcome = drive_session(
        &mut session,
        &io,
        vec![Gate {
            artifact: missing.clone(),
            label: "x".into(),
        }],
        &["approve"],
    )
    .await;
    // The first "approve" is refused (artifact missing); the second also
    // refused, so the session runs out of scripted turns -> MaxTurns (or error).
    assert!(matches!(outcome, Ok(DriveOutcome::MaxTurns) | Err(_)));
    assert!(!missing.exists());
}

#[tokio::test]
async fn user_revision_feedback_is_injected_as_next_turn() {
    let provider = Arc::new(FakeProvider::new(
        "fake",
        vec![
            final_turn("Draft v1. Approve?"),
            final_turn("Applied your change. Approve?"),
        ],
    ));
    let dir = tempdir().unwrap();
    let artifact = dir.path().join("a.md");
    std::fs::write(&artifact, "v2").unwrap();
    let mut session = GuidedSession::new("flow", vec![], provider, "fake", 1024, 10);
    let io = FakeUserIo::new(vec!["make it shorter".into(), "approve".into()]);
    let outcome = drive_session(
        &mut session,
        &io,
        vec![Gate {
            artifact,
            label: "a".into(),
        }],
        &["approve"],
    )
    .await
    .unwrap();
    assert!(matches!(
        outcome,
        DriveOutcome::Approved { gates_passed: 1 }
    ));
}
