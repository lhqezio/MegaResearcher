//! Integration tests for `mr_tui::io::TuiUserIo` — the channel-backed `UserIo`
//! seam that lets the Converge surface reuse 6a's `drive_session` engine.

mod common;

use std::sync::Arc;

use claurst_api::LlmProvider;
use megaresearcher_research::flows::load_embedded;
use megaresearcher_research::phases::{drive_session, DriveOutcome, Gate, GuidedSession, UserIo};
use megaresearcher_research::worker_tools::{ScopedRead, ScopedWrite, Tool};
use mr_tui::io::tui_user_io;

use common::fake_provider::FakeProvider;
use common::turns::{final_turn, write_turn};

/// The channel pair routes `print` into the App's receiver and resolves
/// `read_line` from the App's sender — a blocking round-trip per line.
#[tokio::test]
async fn tui_user_io_prints_and_reads_via_channels() {
    let (io, mut handle) = tui_user_io();
    let session = tokio::spawn(async move {
        io.print("hello from session").await.unwrap();
        let l1 = io.read_line().await.unwrap();
        assert_eq!(l1, "approve\n");
        io.print("second print").await.unwrap();
        let l2 = io.read_line().await.unwrap();
        assert_eq!(l2, "done\n");
    });
    let p1 = handle.print_rx.recv().await.unwrap();
    assert_eq!(p1, "hello from session");
    handle.input_tx.send("approve\n".to_string()).unwrap();
    let p2 = handle.print_rx.recv().await.unwrap();
    assert_eq!(p2, "second print");
    handle.input_tx.send("done\n".to_string()).unwrap();
    session.await.unwrap();
}

/// drive_session runs unchanged against `TuiUserIo`: a scripted 4-turn sequence
/// (write_spec → final_turn → write_plan → final_turn) writes both artifacts,
/// and the App side feeds "approve" at each gate. Mirrors the proven 6a init
/// sequence at `crates/mr-cli/tests/sessions.rs:104-108`.
#[tokio::test]
async fn drive_session_with_tui_user_io_writes_spec_and_plan() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_path_buf();
    let docs = cwd.join("docs/research");
    std::fs::create_dir_all(docs.join("specs")).unwrap();
    std::fs::create_dir_all(docs.join("plans")).unwrap();
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();

    // PROVEN sequence: each artifact write needs a following final_turn so
    // run_to_checkpoint hits EndTurn (no tool calls) and drive_session checks
    // the gate. write_turn paths are RELATIVE to the docs jail; the gate
    // artifact paths are absolute (docs.join(...)). 4 turns, not 3.
    let turns = vec![
        write_turn(&format!("specs/{date}-alpha-spec.md"), "# spec\n"),
        final_turn("spec done, approve?"),
        write_turn(&format!("plans/{date}-alpha-plan.md"), "# plan\n"),
        final_turn("plan done, approve?"),
    ];
    let provider = Arc::new(FakeProvider::new("fake", turns)) as Arc<dyn LlmProvider>;

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
    let mut session = GuidedSession::new(body, tools, provider, "fake", 4096, 60);
    session.inject_user("Topic: alpha");

    let spec_path = docs.join(format!("specs/{date}-alpha-spec.md"));
    let plan_path = docs.join(format!("plans/{date}-alpha-plan.md"));
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

    let (io, mut handle) = tui_user_io();
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
        "expected both gates approved, got {outcome:?}"
    );
    assert!(spec_path.exists(), "spec file missing");
    assert!(plan_path.exists(), "plan file missing");
}
