mod common;

use megaresearcher_research::state::swarm_state::{
    HypothesisNode, Phase, RoundVerdict, SwarmState, Verdict, Worker,
};
use mr_tui::theme::research;
use mr_tui::widget::tree::render_tree;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn buf_content(terminal: &Terminal<TestBackend>) -> String {
    let buf = terminal.backend().buffer().clone();
    buf.content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect()
}

fn state_with_killed_hypothesis() -> SwarmState {
    SwarmState {
        run_id: "r".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
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
                status: "running".into(),
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
                        kill_reason: Some("mechanism contradicts Marks et al. 2025".into()),
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
fn tree_renders_killed_hypothesis_with_kill_reason() {
    let state = state_with_killed_hypothesis();
    let theme = research();
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal
        .draw(|f| {
            render_tree(f, f.area(), &state, &theme);
        })
        .unwrap();
    let content = buf_content(&terminal);
    assert!(
        content.contains("KILLED"),
        "killed status marker: {content}"
    );
    assert!(
        content.contains("mechanism contradicts Marks et al. 2025"),
        "kill reason: {content}"
    );
}

#[test]
fn tree_renders_redteam_round_count() {
    let state = state_with_killed_hypothesis();
    let theme = research();
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal
        .draw(|f| {
            render_tree(f, f.area(), &state, &theme);
        })
        .unwrap();
    let content = buf_content(&terminal);
    // The killed hypothesis went 2 rounds; the approved one went 1. Assert the
    // round indicator renders for at least one node.
    assert!(content.contains("round"), "round indicator: {content}");
}

#[test]
fn tree_renders_approved_hypothesis() {
    let state = state_with_killed_hypothesis();
    let theme = research();
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal
        .draw(|f| {
            render_tree(f, f.area(), &state, &theme);
        })
        .unwrap();
    let content = buf_content(&terminal);
    assert!(
        content.contains("APPROVE") || content.contains("approved"),
        "approved: {content}"
    );
}

#[test]
fn tree_renders_phases_without_hypotheses_for_gap_finding() {
    let state = SwarmState {
        run_id: "r".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
        novelty_target: "gap-finding".into(),
        max_parallel: 4,
        phases: vec![Phase {
            name: "literature-scout".into(),
            status: "complete".into(),
            workers: vec![Worker {
                name: "scout-1".into(),
                status: "passed".into(),
            }],
            hypotheses: vec![],
        }],
        escalations: vec![],
        retry_counts: std::collections::HashMap::new(),
    };
    let theme = research();
    let mut terminal = Terminal::new(TestBackend::new(100, 20)).unwrap();
    terminal
        .draw(|f| {
            render_tree(f, f.area(), &state, &theme);
        })
        .unwrap();
    let content = buf_content(&terminal);
    assert!(
        content.contains("literature-scout"),
        "phase name: {content}"
    );
    assert!(content.contains("scout-1"), "worker name: {content}");
    assert!(
        !content.contains("KILLED"),
        "no killed nodes in gap-finding"
    );
}
