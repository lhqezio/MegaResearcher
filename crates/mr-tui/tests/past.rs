use megaresearcher_research::state::swarm_state::{
    HypothesisNode, Phase, RoundVerdict, SwarmState, Verdict,
};
use mr_tui::app::{App, Surface};
use mr_tui::surface::past::enumerate_runs;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

#[test]
fn enumerate_runs_lists_date_topic_headline() {
    let tmp = tempfile::tempdir().unwrap();
    let runs = tmp.path().join("docs/research/runs");
    let r1 = runs.join("2026-06-28-1200-a1b2c3");
    std::fs::create_dir_all(&r1).unwrap();
    std::fs::write(
        r1.join("output.md"),
        "# SAEs for causal circuits\n\nDirection body.\n",
    )
    .unwrap();
    let swarm = SwarmState {
        run_id: "2026-06-28-1200-a1b2c3".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
        novelty_target: "hypothesis".into(),
        max_parallel: 4,
        phases: vec![Phase {
            name: "red-team".into(),
            status: "complete".into(),
            workers: vec![],
            hypotheses: vec![HypothesisNode {
                id: "hypothesis-smith-1".into(),
                label: "causal-SAE-bridge".into(),
                status: "approved".into(),
                rounds: vec![RoundVerdict {
                    round: 1,
                    critique: Verdict::Approve,
                    revised: false,
                }],
                kill_reason: None,
            }],
        }],
        escalations: vec![],
        retry_counts: std::collections::HashMap::new(),
    };
    swarm.write(&r1.join("swarm-state.yaml")).unwrap();
    let past = enumerate_runs(tmp.path());
    assert_eq!(past.len(), 1);
    assert!(
        past[0].date.contains("2026-06-28"),
        "date: {:?}",
        past[0].date
    );
    assert!(past[0].topic.contains("SAEs"), "topic: {:?}", past[0].topic);
    assert!(
        past[0].headline.contains("causal-SAE-bridge"),
        "headline: {:?}",
        past[0].headline
    );
}

#[test]
fn past_surface_renders_list() {
    let tmp = tempfile::tempdir().unwrap();
    let runs = tmp.path().join("docs/research/runs");
    let r1 = runs.join("2026-06-28-1200-a1b2c3");
    std::fs::create_dir_all(&r1).unwrap();
    std::fs::write(r1.join("output.md"), "# SAEs for causal circuits\n\nbody\n").unwrap();
    let mut app = App::new(tmp.path().to_path_buf(), None);
    app.surface = Surface::Past;
    app.past = Some(mr_tui::surface::past::PastState::from_cwd(tmp.path()));
    let mut terminal = Terminal::new(TestBackend::new(100, 10)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let content: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains("SAEs"), "topic in list: {content}");
}
