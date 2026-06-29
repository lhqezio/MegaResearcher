//! The Start surface — one input, a ghosted example. No wizard, no target
//! picker, no onboarding tour. Type, enter. (Jobs: first 60 seconds.)
//!
//! SPDX-License-Identifier: GPL-3.0

use ratatui::layout::{Alignment, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::theme::ColorPalette;

/// Render the Start surface into `frame`. `question` is the in-progress input;
/// when empty a ghosted example is shown. Pure (TestBackend-testable).
pub fn render_start_frame(
    frame: &mut ratatui::Frame,
    area: Rect,
    question: &str,
    theme: &ColorPalette,
) {
    let prompt = "What do you want to know?";
    let example = "e.g. can sparse autoencoders surface causal circuits in transformer attention?";
    let block = Block::default().borders(Borders::ALL).title(Span::styled(
        "MegaResearcher",
        Style::default().fg(theme.accent),
    ));
    let mut lines = vec![Line::from(Span::styled(
        prompt,
        Style::default().fg(theme.text_light),
    ))];
    let input_text = if question.is_empty() {
        format!("  {example}")
    } else {
        format!("  {question}")
    };
    let input_style = if question.is_empty() {
        Style::default().fg(theme.disabled)
    } else {
        Style::default().fg(theme.text_light)
    };
    lines.push(Line::from(Span::styled(input_text, input_style)));
    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .alignment(Alignment::Left),
        area,
    );
}
