//! Expandable hypothesis cards. Surviving: label/status + parsed
//! mechanism/predicted-outcome/falsification/experimental-design sections
//! from `hypothesis-smith-<N>/output.md`. Rejected: folded, with the kill
//! reason (the lessons). Falls back to label+status only if sections absent.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::collections::HashMap;

use megaresearcher_research::state::swarm_state::HypothesisNode;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::figures::{ARROW, COLLAPSE, CROSS};
use crate::theme::ColorPalette;

/// Parse `output.md`-style sections: split on `## ` headings into a map.
pub fn parse_sections(md: &str) -> HashMap<String, String> {
    let mut sections = HashMap::new();
    let mut current: Option<(String, String)> = None;
    for line in md.lines() {
        if let Some(h) = line.strip_prefix("## ") {
            if let Some((k, v)) = current.take() {
                sections.insert(k, v.trim().to_string());
            }
            current = Some((h.trim().to_string(), String::new()));
        } else if let Some((_, body)) = current.as_mut() {
            if !body.is_empty() || !line.is_empty() {
                body.push_str(line);
                body.push('\n');
            }
        }
    }
    if let Some((k, v)) = current.take() {
        sections.insert(k, v.trim().to_string());
    }
    sections
}

/// Render surviving hypothesis cards + the rejected fold.
pub fn render_cards(
    frame: &mut ratatui::Frame,
    area: Rect,
    surviving: &[HypothesisNode],
    rejected: &[HypothesisNode],
    theme: &ColorPalette,
) {
    let mut lines: Vec<Line> = Vec::new();
    for hyp in surviving {
        lines.push(Line::from(vec![
            Span::styled(ARROW, Style::default().fg(theme.accent)),
            Span::raw(" "),
            Span::styled(
                hyp.label.clone(),
                Style::default()
                    .fg(theme.alive)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        // Card fields are parsed from the smith output.md by the caller (T10
        // surface); the node itself carries id/label/status/rounds/kill_reason.
        // We render the status + rounds here as the card body.
        let rounds_label = format!("  status: {} · rounds: {}", hyp.status, hyp.rounds.len());
        lines.push(Line::from(Span::styled(
            rounds_label,
            Style::default().fg(theme.disabled),
        )));
    }
    if !rejected.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(COLLAPSE, Style::default().fg(theme.killed)),
            Span::styled(
                format!(" rejected ({}) — the lessons", rejected.len()),
                Style::default()
                    .fg(theme.killed)
                    .add_modifier(Modifier::DIM),
            ),
        ]));
        for hyp in rejected {
            let reason = hyp
                .kill_reason
                .clone()
                .unwrap_or_else(|| "(no reason)".to_string());
            lines.push(Line::from(vec![
                Span::raw("   "),
                Span::styled(CROSS, Style::default().fg(theme.killed)),
                Span::raw(" "),
                Span::styled(hyp.label.clone(), Style::default().fg(theme.killed)),
                Span::styled(
                    format!(" — \"{reason}\""),
                    Style::default().fg(theme.killed),
                ),
            ]));
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
}
