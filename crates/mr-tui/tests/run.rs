//! Task 9 — the make-or-break Run surface tests: a killed hypothesis renders
//! the greyed kill + its one-line kill reason + the round indicator, and the
//! escalation strip appears only when an escalation is pending (no defensive
//! UI).
//!
//! SPDX-License-Identifier: GPL-3.0

mod common;

use std::sync::Arc;

use megaresearcher_research::state::swarm_state::{
    Escalation, HypothesisNode, Phase, RoundVerdict, SwarmState, Verdict, Worker,
};
use mr_tui::app::{App, Surface};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

#[allow(dead_code)]
fn _unused_imports() {
    // Keep `common` linked so the shared test infra compiles alongside this
    // binary; T10+ consumes the fake provider + turns helpers.
    let _ = Arc::new(());
}

fn make_or_break_state() -> SwarmState {
    // A run with a killed hypothesis + kill reason — the thesis on one screen.
    SwarmState {
        run_id: "2026-06-28-1200-a1b2c3".into(),
        spec_path: "docs/research/specs/test-spec.md".into(),
        plan_path: "docs/research/plans/test-plan.md".into(),
        novelty_target: "hypothesis".into(),
        max_parallel: 4,
        phases: vec![
            Phase {
                name: "literature-scout".into(),
                status: "complete".into(),
                workers: vec![Worker {
                    name: "scout-1".into(),
                    status: "passed".into(),
                }],
                hypotheses: vec![],
            },
            Phase {
                name: "red-team".into(),
                status: "complete".into(),
                workers: vec![],
                hypotheses: vec![
                    HypothesisNode {
                        id: "hypothesis-smith-1".into(),
                        label: "causal-SAE-bridge".into(),
                        status: "killed".into(),
                        rounds: vec![
                            RoundVerdict {
                                round: 1,
                                critique: Verdict::Reject,
                                revised: true,
                            },
                            RoundVerdict {
                                round: 2,
                                critique: Verdict::Reject,
                                revised: true,
                            },
                        ],
                        kill_reason: Some(
                            "mechanism contradicts Marks et al. 2025 — effect is ablation, not causal"
                                .into(),
                        ),
                    },
                    HypothesisNode {
                        id: "hypothesis-smith-2".into(),
                        label: "logit-lens-circuits".into(),
                        status: "approved".into(),
                        rounds: vec![RoundVerdict {
                            round: 1,
                            critique: Verdict::Approve,
                            revised: false,
                        }],
                        kill_reason: None,
                    },
                ],
            },
        ],
        escalations: vec![],
        retry_counts: std::collections::HashMap::new(),
    }
}

#[test]
fn run_surface_renders_killed_hypothesis_with_kill_reason_and_round() {
    // THE make-or-break test: the run surface renders the greyed kill + its
    // one-line kill reason + the round indicator. This single test encodes
    // the thesis ("the audit trail is the interface").
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    app.surface = Surface::Run;
    app.run_state = Some(mr_tui::surface::run::RunState::from_swarm_for_test(
        make_or_break_state(),
    ));
    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let content: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains("KILLED"), "killed marker: {content}");
    assert!(
        content.contains("mechanism contradicts Marks et al. 2025"),
        "kill reason: {content}"
    );
    assert!(content.contains("round"), "round indicator: {content}");
}

#[test]
fn run_surface_escalation_strip_appears_only_when_pending() {
    // No-defensive-UI rule: the escalation strip is NOT drawn when there is
    // no pending escalation.
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    app.surface = Surface::Run;
    let mut state = mr_tui::surface::run::RunState::from_swarm_for_test(make_or_break_state());
    state.pending_escalation = None;
    app.run_state = Some(state);
    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let content_no_esc: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(
        !content_no_esc.contains("escalation"),
        "no escalation strip when none pending: {content_no_esc}"
    );

    // Now with a pending escalation, the strip appears.
    let mut app2 = App::new(std::path::PathBuf::from("/tmp"), None);
    app2.surface = Surface::Run;
    let mut state2 = mr_tui::surface::run::RunState::from_swarm_for_test(make_or_break_state());
    state2.pending_escalation = Some(Escalation {
        worker: "gap-finder".into(),
        reason: "no gaps in sub-topic 3".into(),
        retry_count: 1,
    });
    app2.run_state = Some(state2);
    let mut terminal2 = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal2.draw(|f| app2.render(f)).unwrap();
    let content_with_esc: String = terminal2
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(
        content_with_esc.contains("escalation"),
        "escalation strip when pending: {content_with_esc}"
    );
    assert!(
        content_with_esc.contains("continue"),
        "continue action: {content_with_esc}"
    );
    assert!(
        content_with_esc.contains("fail"),
        "fail action: {content_with_esc}"
    );
}
