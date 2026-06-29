//! The signature widget: render the swarm as a tree that grows.
//! Phases → workers → hypothesis sub-nodes; the red-team row shows
//! `smith → critique → revise ↻ round N/3`; killed hypotheses dim with
//! their one-line kill reason indented beneath.
//!
//! SPDX-License-Identifier: GPL-3.0

use megaresearcher_research::state::swarm_state::{HypothesisNode, SwarmState, Verdict};
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::figures::{BRANCH, CHECK, CROSS, DASH, LAST, REVISE};
use crate::theme::ColorPalette;

/// Render `swarm` as the growing tree into `area`. Pure (no terminal side
/// effects) so it can be tested with `TestBackend`.
pub fn render_tree(
    frame: &mut ratatui::Frame,
    area: Rect,
    swarm: &SwarmState,
    theme: &ColorPalette,
) {
    let mut lines: Vec<Line> = Vec::new();
    for (pi, phase) in swarm.phases.iter().enumerate() {
        let phase_marker = if pi == swarm.phases.len().saturating_sub(1) {
            LAST
        } else {
            BRANCH
        };
        let status_style = phase_status_style(&phase.status, theme);
        lines.push(Line::from(vec![
            Span::styled(
                format!("{phase_marker} "),
                Style::default().fg(theme.border),
            ),
            Span::styled(
                format!("Phase · {}", phase.name),
                Style::default().fg(theme.text_light),
            ),
            Span::raw("  "),
            Span::styled(phase.status.clone(), status_style),
        ]));
        for (wi, worker) in phase.workers.iter().enumerate() {
            let connector = if wi == phase.workers.len().saturating_sub(1) {
                LAST
            } else {
                BRANCH
            };
            let wstatus_style = worker_status_style(&worker.status, theme);
            lines.push(Line::from(vec![
                Span::raw(format!("  {connector} ")),
                Span::styled(worker.name.clone(), Style::default().fg(theme.text_light)),
                Span::raw("  "),
                Span::styled(worker.status.clone(), wstatus_style),
            ]));
        }
        // Hypothesis sub-nodes (the signature).
        for (hi, hyp) in phase.hypotheses.iter().enumerate() {
            let connector = if hi == phase.hypotheses.len().saturating_sub(1) {
                LAST
            } else {
                BRANCH
            };
            lines.push(hypothesis_line(connector, hyp, theme));
            if let Some(reason) = hyp.kill_reason.as_ref() {
                let dash = DASH.to_string();
                lines.push(Line::from(vec![
                    Span::raw(format!("      {dash} ")),
                    Span::styled(format!("\"{reason}\""), Style::default().fg(theme.killed)),
                ]));
            }
        }
    }
    if swarm.phases.is_empty() {
        lines.push(Line::from(Span::styled(
            "waiting for run to start…",
            Style::default().fg(theme.disabled),
        )));
    }
    let para = Paragraph::new(lines).alignment(Alignment::Left);
    frame.render_widget(para, area);
}

fn hypothesis_line(connector: &str, hyp: &HypothesisNode, theme: &ColorPalette) -> Line<'static> {
    let id = hyp.id.replace("hypothesis-smith-", "H");
    let status_span = match hyp.status.as_str() {
        "killed" => Span::styled(
            format!("{}  KILLED", CROSS),
            Style::default()
                .fg(theme.killed)
                .add_modifier(Modifier::DIM),
        ),
        "approved" => Span::styled(
            format!("{}  APPROVE", CHECK),
            Style::default().fg(theme.success),
        ),
        _ => Span::styled(
            format!("{}  alive", REVISE),
            Style::default().fg(theme.running),
        ),
    };
    let mut spans = vec![
        Span::raw(format!("  {connector} ")),
        Span::styled(
            format!("{id}  {}", hyp.label),
            Style::default().fg(theme.alive),
        ),
        Span::raw("  "),
    ];
    // The red-team row: smith → critique → revise ↻ round N/3.
    let last_round = hyp.rounds.last().map(|r| r.round).unwrap_or(0);
    let round_label = format!("round {}/3", last_round.max(1));
    if hyp.status == "killed" || hyp.status == "approved" {
        // Final state — show the verdict sequence compactly.
        let mut seq = String::new();
        for (i, rv) in hyp.rounds.iter().enumerate() {
            if i > 0 {
                seq.push_str("  ");
            }
            seq.push_str("smith ");
            seq.push_str(if rv.critique == Verdict::Approve {
                CHECK
            } else {
                CROSS
            });
            seq.push_str(" critique ");
            if rv.revised {
                seq.push_str(REVISE);
            }
        }
        spans.push(Span::styled(seq, Style::default().fg(theme.alive)));
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            round_label,
            Style::default().fg(theme.disabled),
        ));
    } else {
        // In-progress — the animated cycling row.
        spans.push(Span::styled(
            format!("smith {} critique {} {}", CHECK, CROSS, REVISE),
            Style::default().fg(theme.running),
        ));
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            round_label,
            Style::default().fg(theme.running),
        ));
    }
    spans.push(Span::raw("  "));
    spans.push(status_span);
    Line::from(spans)
}

fn phase_status_style(status: &str, theme: &ColorPalette) -> Style {
    match status {
        "complete" | "done" => Style::default().fg(theme.success),
        "running" => Style::default().fg(theme.running),
        "skipped" | "pending" | "waiting" => Style::default().fg(theme.disabled),
        s if s.contains("fail") || s.contains("error") => Style::default().fg(theme.error),
        _ => Style::default().fg(theme.text_light),
    }
}

fn worker_status_style(status: &str, theme: &ColorPalette) -> Style {
    match status {
        "passed" | "complete" | "done" | "approved" => Style::default().fg(theme.success),
        "running" => Style::default().fg(theme.running),
        "killed" | "failed" => Style::default().fg(theme.killed),
        "escalated" => Style::default().fg(theme.escalation),
        _ => Style::default().fg(theme.disabled),
    }
}
