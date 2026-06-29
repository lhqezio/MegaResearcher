use megaresearcher_research::state::swarm_state::{HypothesisNode, RoundVerdict, Verdict};
use mr_tui::app::{App, Surface};
use mr_tui::surface::artifact::ArtifactState;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn surviving() -> Vec<HypothesisNode> {
    vec![HypothesisNode {
        id: "hypothesis-smith-1".into(),
        label: "causal-SAE-bridge".into(),
        status: "approved".into(),
        rounds: vec![RoundVerdict {
            round: 1,
            critique: Verdict::Approve,
            revised: false,
        }],
        kill_reason: None,
    }]
}

fn rejected() -> Vec<HypothesisNode> {
    vec![HypothesisNode {
        id: "hypothesis-smith-2".into(),
        label: "activation-patching".into(),
        status: "killed".into(),
        rounds: vec![RoundVerdict {
            round: 2,
            critique: Verdict::Reject,
            revised: true,
        }],
        kill_reason: Some("effect is ablation, not causal".into()),
    }]
}

#[test]
fn artifact_renders_surviving_cards_and_rejected_fold() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    app.surface = Surface::Artifact;
    app.artifact = Some(ArtifactState {
        run_dir: "/tmp/run".into(),
        output_md: "# Research direction\n\nSurviving direction: SAEs for causal circuits.\n"
            .into(),
        surviving: surviving(),
        rejected: rejected(),
    });
    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let content: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(
        content.contains("causal-SAE-bridge"),
        "surviving card label: {content}"
    );
    assert!(content.contains("rejected"), "rejected fold: {content}");
    assert!(
        content.contains("effect is ablation, not causal"),
        "rejected kill reason: {content}"
    );
}

#[test]
fn parse_sections_splits_on_h2_headings() {
    let md =
        "# Title\n\n## Mechanism\n\nThe mechanism is X.\n\n## Predicted outcome\n\nY happens.\n";
    let sections = mr_tui::widget::cards::parse_sections(md);
    assert!(sections.contains_key("Mechanism"));
    assert!(sections.contains_key("Predicted outcome"));
    assert!(sections["Mechanism"].contains("The mechanism is X."));
}
