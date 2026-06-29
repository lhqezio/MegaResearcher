//! The bounded converge widget: renders the inline conversation buffer + the
//! spec/plan `[✓]` approval cards. Not free chat — the conversation is
//! bounded and converges the question.
//!
//! SPDX-License-Identifier: GPL-3.0

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::figures::{CHECK, PENCIL};
use crate::theme::ColorPalette;

/// Render the Converge surface into `frame`: the inline conversation buffer on
/// top, the spec + plan approval cards as a two-up row beneath. Pure
/// (TestBackend-testable). `spec_approved` / `plan_approved` toggle the `[✓]`.
pub fn render_converge(
    frame: &mut ratatui::Frame,
    area: Rect,
    conversation: &[String],
    spec_approved: bool,
    plan_approved: bool,
    theme: &ColorPalette,
) {
    let chunks = Layout::default()
        .constraints([Constraint::Min(3), Constraint::Length(4)])
        .split(area);
    let convo = conversation
        .iter()
        .map(|l| Line::from(Span::styled(l, Style::default().fg(theme.text_light))))
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(convo)
            .block(Block::default().borders(Borders::ALL).title(Span::styled(
                format!("{PENCIL} converge the question"),
                Style::default().fg(theme.accent),
            )))
            .alignment(Alignment::Left),
        chunks[0],
    );
    // Spec + plan cards.
    let cards = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);
    let spec_line = if spec_approved {
        Line::from(vec![
            Span::styled("spec  ", Style::default().fg(theme.text_light)),
            Span::styled(CHECK, Style::default().fg(theme.success)),
        ])
    } else {
        Line::from(Span::styled(
            "spec  (pending)",
            Style::default().fg(theme.disabled),
        ))
    };
    frame.render_widget(
        Paragraph::new(spec_line).block(Block::default().borders(Borders::ALL)),
        cards[0],
    );
    let plan_line = if plan_approved {
        Line::from(vec![
            Span::styled("plan  ", Style::default().fg(theme.text_light)),
            Span::styled(CHECK, Style::default().fg(theme.success)),
        ])
    } else {
        Line::from(Span::styled(
            "plan  (pending)",
            Style::default().fg(theme.disabled),
        ))
    };
    frame.render_widget(
        Paragraph::new(plan_line).block(Block::default().borders(Borders::ALL)),
        cards[1],
    );
}
