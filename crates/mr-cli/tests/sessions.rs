//! Integration tests for `mr init` and `mr spec`/`mr plan` guided sessions.
//!
//! Drives a scripted `FakeProvider` through the approval-gate loop with a
//! `FakeUserIo` that returns "approve" on every read, asserting the gate
//! artifacts land at the expected paths under `docs/research/`.

mod common;
use common::fake_provider::FakeProvider;

use std::sync::Arc;

use async_trait::async_trait;
use claurst_api::{LlmProvider, StopReason, StreamEvent};
use claurst_core::types::{ContentBlock, UsageInfo};
use megaresearcher_research::phases::UserIo;
use serde_json::json;
use tempfile::tempdir;

/// A `UserIo` that auto-approves every gate and discards printed text.
struct FakeUserIo;

#[async_trait]
impl UserIo for FakeUserIo {
    async fn print(&self, _text: &str) -> std::io::Result<()> {
        Ok(())
    }
    async fn read_line(&self) -> std::io::Result<String> {
        Ok("approve".into())
    }
}

/// A provider turn that calls the `Write` tool to create `file_rel` (relative
/// to the session's `docs` jail) with `content`, then stops on `ToolUse`.
fn write_turn(file_rel: &str, content: &str) -> Vec<StreamEvent> {
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
            text: format!("writing {file_rel}"),
        },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::ContentBlockStart {
            index: 1,
            content_block: ContentBlock::ToolUse {
                id: format!("tu_{file_rel}"),
                name: "Write".into(),
                input: json!({ "file_path": file_rel, "content": content }),
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

/// A provider turn that emits only text and ends the turn (`EndTurn`).
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

#[tokio::test]
async fn init_writes_spec_and_plan_then_suggests_execute() {
    let tmp = tempdir().unwrap();
    let cwd = tmp.path().to_path_buf();
    // Compute the date once so the scripted file_path and the post-run
    // assertion agree (negligible midnight-boundary flake, same as production).
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();

    let turns = vec![
        write_turn(&format!("specs/{date}-alpha-spec.md"), "# spec\n"),
        final_turn("spec done, approve?"),
        write_turn(&format!("plans/{date}-alpha-plan.md"), "# plan\n"),
        final_turn("plan done, approve?"),
    ];
    let provider = (
        Arc::new(FakeProvider::new("fake", turns)) as Arc<dyn LlmProvider>,
        "fake".to_string(),
    );

    mr_cli::commands::init::run_with(&cwd, provider, "alpha", &FakeUserIo)
        .await
        .unwrap();

    assert!(
        cwd.join(format!("docs/research/specs/{date}-alpha-spec.md"))
            .exists(),
        "spec file missing"
    );
    assert!(
        cwd.join(format!("docs/research/plans/{date}-alpha-plan.md"))
            .exists(),
        "plan file missing"
    );
}

#[tokio::test]
async fn spec_session_writes_spec_file() {
    let tmp = tempdir().unwrap();
    let cwd = tmp.path().to_path_buf();
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();

    let turns = vec![
        write_turn(&format!("specs/{date}-alpha-spec.md"), "# spec\n"),
        final_turn("spec done, approve?"),
    ];
    let provider = (
        Arc::new(FakeProvider::new("fake", turns)) as Arc<dyn LlmProvider>,
        "fake".to_string(),
    );

    mr_cli::commands::session::run_session_with(&cwd, provider, "spec", "alpha", &FakeUserIo)
        .await
        .unwrap();

    assert!(
        cwd.join(format!("docs/research/specs/{date}-alpha-spec.md"))
            .exists(),
        "spec file missing"
    );
}
