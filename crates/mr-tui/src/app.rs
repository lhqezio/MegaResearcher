//! The surface state machine + async event loop. Factors a pure
//! `App::render(&self, frame)` so tests use TestBackend without a live
//! terminal, and a pure `App::handle_key` so transitions are testable.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::path::PathBuf;

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::surface::start::render_start_frame;
use crate::theme::{for_theme, ColorPalette};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    Start,
    Converge,
    Run,
    Artifact,
    Past,
    Settings,
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    Quit,
    ToSurface(Surface),
    SubmitQuestion,
    None,
}

pub struct App {
    pub surface: Surface,
    pub question: String,
    pub cwd: PathBuf,
    pub should_exit: bool,
    /// Per-frame counter for animation timing. Used in T9; kept here so the
    /// state machine has a stable shape.
    #[allow(dead_code)]
    pub frame_count: u64,
    pub theme: ColorPalette,
}

impl App {
    pub fn new(cwd: PathBuf) -> Self {
        Self {
            surface: Surface::Start,
            question: String::new(),
            cwd,
            should_exit: false,
            frame_count: 0,
            theme: for_theme("research"),
        }
    }

    /// Pure key handler. Returns the event the loop should act on. Testable
    /// without a terminal.
    pub fn handle_key(&mut self, key: KeyEvent) -> AppEvent {
        // Global keys: q / Ctrl-C quit, s -> Settings.
        if key.kind != KeyEventKind::Press {
            return AppEvent::None;
        }
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => AppEvent::Quit,
            KeyCode::Char('q') => AppEvent::Quit,
            KeyCode::Char('s') => AppEvent::ToSurface(Surface::Settings),
            KeyCode::Enter if self.surface == Surface::Start => {
                if !self.question.trim().is_empty() {
                    AppEvent::SubmitQuestion
                } else {
                    AppEvent::None
                }
            }
            KeyCode::Char(c) if self.surface == Surface::Start => {
                self.question.push(c);
                AppEvent::None
            }
            KeyCode::Backspace if self.surface == Surface::Start => {
                self.question.pop();
                AppEvent::None
            }
            _ => AppEvent::None,
        }
    }

    /// Pure render. Dispatches to the current surface. TestBackend-testable.
    pub fn render(&self, frame: &mut ratatui::Frame) {
        match self.surface {
            Surface::Start => {
                render_start_frame(frame, frame.area(), &self.question, &self.theme);
            }
            _ => {
                // Placeholder until the surface modules land (T8-T12).
                frame.render_widget(
                    ratatui::widgets::Paragraph::new(format!("surface: {:?}", self.surface)).block(
                        ratatui::widgets::Block::default()
                            .borders(ratatui::widgets::Borders::ALL)
                            .title("MegaResearcher"),
                    ),
                    frame.area(),
                );
            }
        }
    }

    /// The async event loop. Draw first each iter, then spawn_blocking poll
    /// 50ms, filter KeyEventKind::Press, act on AppEvent. `q`/Ctrl-C quit;
    /// `s` -> Settings. The session/orchestrator task wiring arrives in T8/T9.
    pub async fn run(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> anyhow::Result<()> {
        loop {
            self.frame_count = self.frame_count.wrapping_add(1);
            terminal.draw(|f| self.render(f))?;
            let event = tokio::task::spawn_blocking(|| {
                if crossterm::event::poll(std::time::Duration::from_millis(50)).ok()? {
                    crossterm::event::read().ok()
                } else {
                    None
                }
            })
            .await?;
            if let Some(Event::Key(key)) = event {
                let ev = self.handle_key(key);
                match ev {
                    AppEvent::Quit => {
                        self.should_exit = true;
                        return Ok(());
                    }
                    AppEvent::ToSurface(s) => self.surface = s,
                    AppEvent::SubmitQuestion => {
                        // T8 wires the Converge spawn here.
                        self.surface = Surface::Converge;
                    }
                    AppEvent::None => {}
                }
            }
            if self.should_exit {
                return Ok(());
            }
        }
    }
}
