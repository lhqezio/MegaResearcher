//! Settings — a structured editor for MrConfig. One screen, no tabs;
//! opinionated defaults pre-filled; masked API key; test connection; save→file.
//! First-run auto-opens when no provider/key is configured (resolves the 6a
//! Important finding).
//!
//! SPDX-License-Identifier: GPL-3.0

use crate::config::MrConfig;

pub struct SettingsState {
    pub config: MrConfig,
    pub masked_key: String,
    pub selected_field: usize,
    pub test_status: Option<String>,
}

impl SettingsState {
    pub fn from_config(config: MrConfig) -> Self {
        let masked_key = config
            .api_key
            .as_ref()
            .map(|k| {
                if k.len() <= 8 {
                    "•".repeat(k.len().max(1))
                } else {
                    let prefix = &k[..8];
                    format!("{prefix}{}", "•".repeat(k.len() - 8))
                }
            })
            .unwrap_or_default();
        Self {
            config,
            masked_key,
            selected_field: 0,
            test_status: None,
        }
    }
}

pub fn render_settings(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    state: &SettingsState,
    theme: &crate::theme::ColorPalette,
) {
    let lines = vec![
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "settings",
            ratatui::style::Style::default().fg(theme.accent),
        )),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(
                "provider          ",
                ratatui::style::Style::default().fg(theme.text_light),
            ),
            ratatui::text::Span::raw(
                state
                    .config
                    .provider
                    .clone()
                    .unwrap_or_else(|| "(unset)".into()),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(
                "api key           ",
                ratatui::style::Style::default().fg(theme.text_light),
            ),
            ratatui::text::Span::styled(
                state.masked_key.clone(),
                ratatui::style::Style::default().fg(theme.disabled),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(
                "model             ",
                ratatui::style::Style::default().fg(theme.text_light),
            ),
            ratatui::text::Span::raw(
                state
                    .config
                    .model
                    .clone()
                    .unwrap_or_else(|| "claude-sonnet-4-6".into()),
            ),
        ]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "run",
            ratatui::style::Style::default().fg(theme.text_light),
        )),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(
                "max parallel      ",
                ratatui::style::Style::default().fg(theme.text_light),
            ),
            ratatui::text::Span::raw(state.config.max_parallel.to_string()),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(
                "on escalation     ",
                ratatui::style::Style::default().fg(theme.text_light),
            ),
            ratatui::text::Span::raw(state.config.on_escalate.clone()),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(
                "mcp (ml-intern)   ",
                ratatui::style::Style::default().fg(theme.text_light),
            ),
            ratatui::text::Span::raw(if state.config.mcp { "on" } else { "off" }),
        ]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(
                "cost ceiling      ",
                ratatui::style::Style::default().fg(theme.text_light),
            ),
            ratatui::text::Span::raw(format!(
                "${} sandbox · ${} api",
                state.config.cost_ceiling_sandbox, state.config.cost_ceiling_api
            )),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(
                "theme             ",
                ratatui::style::Style::default().fg(theme.text_light),
            ),
            ratatui::text::Span::raw(state.config.theme.clone()),
        ]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "[ save ]   [ cancel ]    saved to ~/.config/mr/config.toml",
            ratatui::style::Style::default().fg(theme.action),
        )),
    ];
    frame.render_widget(ratatui::widgets::Paragraph::new(lines), area);
}
