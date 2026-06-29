//! Scripted `StreamEvent` turn sequences, copied verbatim (same pattern) from
//! `crates/research/tests/orchestrator.rs:150-220,539-543` (test infra,
//! GPL-3.0, same repo). Used by the fake-provider integration test for
//! `mr execute` to drive a 4-worker gap-finding run without a network.

// Each test file compiles its own `mod common;`, so a helper used only by
// execute.rs is dead code from sessions.rs's view. Allow it crate-wide here.
#![allow(dead_code)]

use claurst_api::{StopReason, StreamEvent};
use claurst_core::types::{ContentBlock, UsageInfo};
use serde_json::json;

pub fn write_turn(file: &str, content: &str) -> Vec<StreamEvent> {
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

pub fn final_turn(text: &str) -> Vec<StreamEvent> {
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
pub fn three_artifact_turns() -> Vec<Vec<StreamEvent>> {
    vec![
        write_turn("output.md", "# Output\n\ncontent"),
        write_turn("manifest.yaml", "role: literature-scout\n"),
        write_turn("verification.md", "# Verification\n\nok"),
        final_turn("Done."),
    ]
}

/// `n_workers` × 4 turns. 4 workers (2 scouts + 1 gap-finder + 1 synthesist)
/// ⇒ 16 turns, matching the proven gap-finding run in
/// `crates/research/tests/orchestrator.rs:549-611`.
pub fn run_turns(n_workers: usize) -> Vec<Vec<StreamEvent>> {
    (0..n_workers)
        .flat_map(|_| three_artifact_turns())
        .collect()
}
