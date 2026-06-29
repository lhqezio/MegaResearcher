//! Past runs — `mr` with no args when runs exist: a list by date + topic,
//! each with its headline surviving hypothesis. One tap reopens artifact+tree.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::path::{Path, PathBuf};

use megaresearcher_research::state::swarm_state::SwarmState;

#[derive(Debug, Clone)]
pub struct PastRun {
    pub date: String,
    pub topic: String,
    pub headline: String,
    pub run_dir: PathBuf,
}

/// Enumerate `docs/research/runs/` newest-first. `date` from the dir name
/// prefix; `topic` from the `output.md` H1; `headline` from the first
/// approved HypothesisNode (or "(gap-finding)" if none).
pub fn enumerate_runs(cwd: &Path) -> Vec<PastRun> {
    let runs = cwd.join("docs/research/runs");
    if !runs.is_dir() {
        return Vec::new();
    }
    let entries: Vec<_> = std::fs::read_dir(&runs)
        .map(|rd| rd.flatten().collect::<Vec<_>>())
        .unwrap_or_default();
    let mut entries: Vec<_> = entries.into_iter().filter(|e| e.path().is_dir()).collect();
    entries.sort_by_key(|e| e.file_name());
    let mut out = Vec::new();
    for e in entries.iter().rev() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        let date = name.split('-').take(3).collect::<Vec<_>>().join("-");
        let output_md = std::fs::read_to_string(path.join("output.md")).unwrap_or_default();
        let topic = output_md
            .lines()
            .find_map(|l| l.strip_prefix("# ").map(|s| s.trim().to_string()))
            .unwrap_or_else(|| name.clone());
        let headline = SwarmState::read(&path.join("swarm-state.yaml"))
            .ok()
            .and_then(|s| {
                s.phases
                    .iter()
                    .flat_map(|p| p.hypotheses.iter())
                    .find_map(|h| {
                        if h.status == "approved" {
                            Some(h.label.clone())
                        } else {
                            None
                        }
                    })
            })
            .unwrap_or_else(|| "(gap-finding)".to_string());
        out.push(PastRun {
            date,
            topic,
            headline,
            run_dir: path,
        });
    }
    out
}

pub struct PastState {
    pub runs: Vec<PastRun>,
    pub selected: usize,
}

impl PastState {
    pub fn from_cwd(cwd: &Path) -> Self {
        Self {
            runs: enumerate_runs(cwd),
            selected: 0,
        }
    }
}

pub fn render_past(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    state: &PastState,
    theme: &crate::theme::ColorPalette,
) {
    let mut lines: Vec<ratatui::text::Line> = Vec::new();
    lines.push(ratatui::text::Line::from(ratatui::text::Span::styled(
        "past runs",
        ratatui::style::Style::default().fg(theme.accent),
    )));
    for (i, run) in state.runs.iter().enumerate() {
        let marker = if i == state.selected { "▸" } else { " " };
        lines.push(ratatui::text::Line::from(vec![
            ratatui::text::Span::raw(format!("{marker} ")),
            ratatui::text::Span::styled(
                run.date.clone(),
                ratatui::style::Style::default().fg(theme.text_light),
            ),
            ratatui::text::Span::raw("  "),
            ratatui::text::Span::styled(
                run.topic.clone(),
                ratatui::style::Style::default().fg(theme.text_light),
            ),
            ratatui::text::Span::raw(" → "),
            ratatui::text::Span::styled(
                run.headline.clone(),
                ratatui::style::Style::default().fg(theme.success),
            ),
        ]));
    }
    frame.render_widget(ratatui::widgets::Paragraph::new(lines), area);
}
