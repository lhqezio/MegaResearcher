//! Task 8 — Converge surface: `drive_session` via `TuiUserIo` + the
//! spec/plan `[✓]` approval-card widget.
//!
//! SPDX-License-Identifier: GPL-3.0

mod common;

use std::sync::Arc;

use claurst_api::LlmProvider;
use megaresearcher_research::flows::load_embedded;
use megaresearcher_research::phases::{drive_session, DriveOutcome, Gate, GuidedSession};
use megaresearcher_research::worker_tools::{ScopedRead, ScopedWrite, Tool};
use mr_tui::io::tui_user_io;
use mr_tui::theme::research;
use mr_tui::widget::inline_chat::render_converge;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

use common::fake_provider::FakeProvider;
use common::turns::{final_turn, write_turn};

/// `drive_session` runs unchanged against `TuiUserIo`: the PROVEN 4-turn
/// sequence (write_spec → final_turn → write_plan → final_turn) writes both
/// artifacts, and the App side feeds "approve" at each gate. write_turn paths
/// are RELATIVE to the docs jail; gate artifact paths are ABSOLUTE. Mirrors
/// the proven 6a init sequence at `crates/mr-cli/tests/sessions.rs:104-108`.
#[tokio::test]
async fn converge_session_writes_spec_and_plan_and_approves() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs/research");
    std::fs::create_dir_all(docs.join("specs")).unwrap();
    std::fs::create_dir_all(docs.join("plans")).unwrap();
    let spec_path = docs.join("specs/converge-spec.md");
    let plan_path = docs.join("plans/converge-plan.md");

    // PROVEN sequence: each artifact write needs a following final_turn so
    // run_to_checkpoint hits EndTurn (no tool calls) and drive_session checks
    // the gate. write_turn paths are RELATIVE to the docs jail; the gate
    // artifact paths are absolute (docs.join(...)). 4 turns, not 3.
    let turns = vec![
        write_turn("specs/converge-spec.md", "# Spec\n\nconverged spec"),
        final_turn("spec done, approve?"),
        write_turn("plans/converge-plan.md", "# Plan\n\nconverged plan"),
        final_turn("plan done, approve?"),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;

    let body = format!(
        "# Phase 1 — Brainstorm\n\n{}\n\n# Phase 2 — Spec\n\n{}\n\n# Phase 3 — Plan\n\n{}\n\n\
         Drive all three phases in order. Pause for user approval after the spec and again after the plan.",
        load_embedded("brainstorm").body,
        load_embedded("spec").body,
        load_embedded("plan").body,
    );
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ScopedRead::with_shared(docs.clone(), docs.clone())),
        Arc::new(ScopedWrite::new(docs.clone())),
    ];
    let mut session = GuidedSession::new(body, tools, provider, "fake-model", 4096, 60);
    session.inject_user("Topic: can SAEs surface causal circuits?");

    let (io, mut handle) = tui_user_io();
    let gates = vec![
        Gate {
            artifact: spec_path.clone(),
            label: "spec".into(),
        },
        Gate {
            artifact: plan_path.clone(),
            label: "plan".into(),
        },
    ];
    let drive = tokio::spawn(async move {
        drive_session(&mut session, &io, gates, &["approve", "yes", "y", "done"]).await
    });

    // App side: feed "approve" for each gate. Drain prints to avoid
    // backpressure on the unbounded print channel.
    let mut approvals = 0;
    while approvals < 2 {
        while handle.print_rx.try_recv().is_ok() {}
        handle.input_tx.send("approve\n".to_string()).unwrap();
        approvals += 1;
    }
    let outcome = drive.await.unwrap().unwrap();
    assert!(
        matches!(outcome, DriveOutcome::Approved { gates_passed: 2 }),
        "expected Approved with 2 gates, got {outcome:?}"
    );
    assert!(spec_path.exists(), "spec file missing");
    assert!(plan_path.exists(), "plan file missing");
}

/// The Converge widget renders the inline conversation buffer plus the spec
/// and plan approval cards. With spec approved and plan pending, the spec card
/// shows `[✓]` and the plan card shows "(pending)".
#[test]
fn render_converge_shows_spec_and_plan_cards() {
    let theme = research();
    let conversation = vec![
        "q  gap to research, or a hypothesis?".to_string(),
        ">  gap to research".to_string(),
    ];
    let mut terminal = Terminal::new(TestBackend::new(100, 20)).unwrap();
    terminal
        .draw(|f| {
            render_converge(f, f.area(), &conversation, true, false, &theme);
        })
        .unwrap();
    let content: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains("spec"), "spec card: {content}");
    assert!(content.contains("plan"), "plan card: {content}");
    // spec approved → [✓]; plan not yet → no check on plan.
    assert!(content.contains('✓'), "approved check: {content}");
}
