//! Minimal hand-rolled markdown renderer: headings, paragraphs, lists, bold.
//! Deferred to Phase 8: a full markdown crate. For 6b this is enough to
//! render `output.md` readably on the artifact screen.
//!
//! SPDX-License-Identifier: GPL-3.0

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::theme::ColorPalette;

pub fn render_markdown(frame: &mut ratatui::Frame, area: Rect, md: &str, theme: &ColorPalette) {
    let mut lines: Vec<Line> = Vec::new();
    for raw in md.lines() {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.is_empty() {
            lines.push(Line::from(""));
            continue;
        }
        if let Some(h) = line.strip_prefix("# ") {
            lines.push(Line::from(Span::styled(
                h.to_string(),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            )));
        } else if let Some(h) = line.strip_prefix("## ") {
            lines.push(Line::from(Span::styled(
                h.to_string(),
                Style::default()
                    .fg(theme.text_light)
                    .add_modifier(Modifier::BOLD),
            )));
        } else if let Some(item) = line.strip_prefix("- ") {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("• {item}"), Style::default().fg(theme.text_light)),
            ]));
        } else {
            lines.push(render_inline_bold(line, theme));
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
}

/// Split on `**bold**` and render the bold spans with the accent color.
fn render_inline_bold(line: &str, theme: &ColorPalette) -> Line<'static> {
    let mut spans = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find("**") {
        if start > 0 {
            spans.push(Span::styled(
                rest[..start].to_string(),
                Style::default().fg(theme.text_light),
            ));
        }
        let after = &rest[start + 2..];
        if let Some(end) = after.find("**") {
            spans.push(Span::styled(
                after[..end].to_string(),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ));
            rest = &after[end + 2..];
        } else {
            spans.push(Span::styled(
                format!("**{after}"),
                Style::default().fg(theme.text_light),
            ));
            rest = "";
            break;
        }
    }
    if !rest.is_empty() {
        spans.push(Span::styled(
            rest.to_string(),
            Style::default().fg(theme.text_light),
        ));
    }
    if spans.is_empty() {
        spans.push(Span::raw(line.to_string()));
    }
    Line::from(spans)
}
