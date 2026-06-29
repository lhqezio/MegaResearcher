// MegaResearcher Phase 6b — interactive research TUI.
// The audit trail is the interface: a tree that grows, killed hypotheses
// stay on screen dimmed with a one-line kill reason, the red-team loop
// animates (smith → critique → revise ↻, round N/3).
//
// SPDX-License-Identifier: GPL-3.0

pub mod bootstrap;
pub mod figures;
pub mod guard;
pub mod theme;
pub mod widget;

use std::path::Path;

use ratatui::widgets::{Block, Borders, Paragraph};

/// Render the intro/start frame into `frame`. Pure (no terminal side effects)
/// so it can be tested with `TestBackend`.
pub fn render_intro(frame: &mut ratatui::Frame) {
    let area = frame.area();
    let text = "What do you want to know?";
    frame.render_widget(
        Paragraph::new(text).block(
            Block::default()
                .borders(Borders::ALL)
                .title("MegaResearcher"),
        ),
        area,
    );
}

/// Entry point. Phase 6b smoke: set up the terminal, render one intro frame,
/// restore, return. The full event loop arrives in T7.
pub async fn run(_cwd: &Path) -> anyhow::Result<()> {
    let mut terminal = bootstrap::setup_terminal()?;
    terminal.draw(render_intro)?;
    bootstrap::restore_terminal(&mut terminal)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn render_intro_shows_prompt() {
        let mut terminal = Terminal::new(TestBackend::new(60, 10)).unwrap();
        terminal.draw(render_intro).unwrap();
        let buf = terminal.backend().buffer().clone();
        let content: String = buf
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("What do you want to know?"));
        assert!(content.contains("MegaResearcher"));
    }
}
