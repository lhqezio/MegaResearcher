//! Worker-contract tests: a deterministic fake provider drives the query loop.

mod common;

use std::sync::Arc;

use claurst_api::{LlmProvider, StopReason, StreamEvent};
use claurst_core::types::{ContentBlock, UsageInfo};
use serde_json::json;

use common::fake_provider::FakeProvider;
use megaresearcher_research::worker::{Worker, WorkerConfig, WorkerStop};
use megaresearcher_research::worker_tools::{ScopedWrite, Tool};

/// A turn that calls Write(file, content) and stops with ToolUse.
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
            text: format!("writing {file}").into(),
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

/// A turn with only a Text block, stopping with EndTurn.
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
async fn test_worker_writes_three_artifacts_and_ends() {
    let out = tempfile::tempdir().unwrap();
    let write = Arc::new(ScopedWrite::new(out.path())) as Arc<dyn Tool>;
    let turns = vec![
        write_turn("output.md", "# Output\n\nThe bibliography."),
        write_turn("manifest.yaml", "role: literature-scout\npapers_count: 8\n"),
        write_turn("verification.md", "# Verification\n\nAll checks passed."),
        final_turn("Done. All three artifacts written."),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let worker = Worker::new(
        "You are a literature scout.",
        vec![write],
        provider,
        WorkerConfig {
            max_turns: 10,
            max_tokens: 1024,
            model: "fake-model".into(),
        },
        out.path(),
    );
    let outcome = worker
        .run("Survey the topic. Write output.md, manifest.yaml, verification.md.")
        .await
        .unwrap();

    assert_eq!(outcome.stop, WorkerStop::EndTurn);
    assert_eq!(outcome.turns, 4);
    assert_eq!(outcome.final_text, "Done. All three artifacts written.");
    assert_eq!(
        std::fs::read_to_string(out.path().join("output.md")).unwrap(),
        "# Output\n\nThe bibliography."
    );
    assert_eq!(
        std::fs::read_to_string(out.path().join("manifest.yaml")).unwrap(),
        "role: literature-scout\npapers_count: 8\n"
    );
    assert_eq!(
        std::fs::read_to_string(out.path().join("verification.md")).unwrap(),
        "# Verification\n\nAll checks passed."
    );
    assert_eq!(fake.call_count(), 4);
}

#[tokio::test]
async fn test_worker_terminates_on_max_turns() {
    let out = tempfile::tempdir().unwrap();
    let write = Arc::new(ScopedWrite::new(out.path())) as Arc<dyn Tool>;
    // A single tool-use turn the fake repeats every call.
    let turn = write_turn("repeat.md", "x");
    let fake = Arc::new(FakeProvider::new("fake", vec![turn]));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let worker = Worker::new(
        "sys",
        vec![write],
        provider,
        WorkerConfig {
            max_turns: 2,
            max_tokens: 1024,
            model: "fake-model".into(),
        },
        out.path(),
    );
    let outcome = worker.run("write repeatedly").await.unwrap();
    assert_eq!(outcome.stop, WorkerStop::MaxTurns);
    assert_eq!(outcome.turns, 2);
    assert_eq!(fake.call_count(), 2);
    assert!(out.path().join("repeat.md").exists());
}

#[tokio::test]
async fn test_worker_unknown_tool_is_error_block_not_panic() {
    let out = tempfile::tempdir().unwrap();
    // No tools wired, but the fake requests a "Write" call.
    let turns = vec![
        vec![
            StreamEvent::MessageStart {
                id: "m".into(),
                model: "fake".into(),
                usage: UsageInfo::default(),
            },
            StreamEvent::ContentBlockStart {
                index: 0,
                content_block: ContentBlock::ToolUse {
                    id: "tu_1".into(),
                    name: "Write".into(),
                    input: json!({"file_path":"x.md","content":"x"}),
                },
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: Some(UsageInfo::default()),
            },
            StreamEvent::MessageStop,
        ],
        final_turn("Recovered after the missing tool."),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let worker = Worker::new(
        "sys",
        vec![],
        provider,
        WorkerConfig {
            max_turns: 10,
            max_tokens: 1024,
            model: "fake-model".into(),
        },
        out.path(),
    );
    let outcome = worker.run("use a tool i do not have").await.unwrap();
    // The worker did not panic; it dispatched an error tool_result and the next
    // turn ended cleanly.
    assert_eq!(outcome.stop, WorkerStop::EndTurn);
    assert_eq!(outcome.final_text, "Recovered after the missing tool.");
    // The unknown tool never wrote anything.
    assert!(!out.path().join("x.md").exists());
}
