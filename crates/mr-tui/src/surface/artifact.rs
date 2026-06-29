//! The Artifact surface — the payoff. Renders output.md (minimal markdown),
//! surviving hypothesis cards, and the rejected fold (the lessons). The tree
//! slides to a side rail as provenance (deferred visual polish to Phase 8;
//! 6b renders the direction taking the screen).
//!
//! SPDX-License-Identifier: GPL-3.0

use std::path::PathBuf;

use megaresearcher_research::state::swarm_state::HypothesisNode;

pub struct ArtifactState {
    pub run_dir: PathBuf,
    pub output_md: String,
    pub surviving: Vec<HypothesisNode>,
    pub rejected: Vec<HypothesisNode>,
}

impl ArtifactState {
    /// Build from a run dir: read output.md + the swarm's hypothesis nodes.
    pub fn from_run_dir(run_dir: &std::path::Path) -> Self {
        let output_md = std::fs::read_to_string(run_dir.join("output.md"))
            .unwrap_or_else(|_| "(no output.md)".to_string());
        let swarm = megaresearcher_research::state::swarm_state::SwarmState::read(
            &run_dir.join("swarm-state.yaml"),
        )
        .ok();
        let (surviving, rejected) = swarm
            .as_ref()
            .map(|s| {
                let mut surv = Vec::new();
                let mut rej = Vec::new();
                for p in &s.phases {
                    for h in &p.hypotheses {
                        if h.status == "approved" {
                            surv.push(h.clone());
                        } else if h.status == "killed" {
                            rej.push(h.clone());
                        }
                    }
                }
                (surv, rej)
            })
            .unwrap_or_default();
        Self {
            run_dir: run_dir.to_path_buf(),
            output_md,
            surviving,
            rejected,
        }
    }
}

pub fn render_artifact(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    state: &ArtifactState,
    theme: &crate::theme::ColorPalette,
) {
    let chunks = ratatui::layout::Layout::default()
        .constraints([
            ratatui::layout::Constraint::Min(5),
            ratatui::layout::Constraint::Min(3),
        ])
        .split(area);
    crate::widget::markdown::render_markdown(frame, chunks[0], &state.output_md, theme);
    crate::widget::cards::render_cards(frame, chunks[1], &state.surviving, &state.rejected, theme);
}
