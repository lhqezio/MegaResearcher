//! The surface state machine + async event loop. Factors a pure
//! `App::render(&self, frame)` so tests use TestBackend without a live
//! terminal, and a pure `App::handle_key` so transitions are testable.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::path::PathBuf;
use std::sync::Arc;

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures::FutureExt as _;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::surface::start::{render_start_frame, ConvergeOutcome, ConvergeState};
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
    pub provider: Option<(Arc<dyn claurst_api::LlmProvider>, String)>,
    pub converge: Option<ConvergeState>,
    pub converge_task: Option<
        tokio::task::JoinHandle<anyhow::Result<megaresearcher_research::phases::DriveOutcome>>,
    >,
    pub run_state: Option<crate::surface::run::RunState>,
}

impl App {
    pub fn new(
        cwd: PathBuf,
        provider: Option<(Arc<dyn claurst_api::LlmProvider>, String)>,
    ) -> Self {
        Self {
            surface: Surface::Start,
            question: String::new(),
            cwd,
            should_exit: false,
            frame_count: 0,
            theme: for_theme("research"),
            provider,
            converge: None,
            run_state: None,
            converge_task: None,
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
            KeyCode::Char('c') if self.surface == Surface::Run && self.run_pending_escalation() => {
                self.run_respond(
                    megaresearcher_research::orchestrator::escalation::EscalationVerdict::Continue,
                );
                AppEvent::None
            }
            KeyCode::Char('f') if self.surface == Surface::Run && self.run_pending_escalation() => {
                self.run_respond(
                    megaresearcher_research::orchestrator::escalation::EscalationVerdict::Fail,
                );
                AppEvent::None
            }
            _ => AppEvent::None,
        }
    }

    /// True when a Run surface escalation is awaiting the user's verdict.
    fn run_pending_escalation(&self) -> bool {
        self.run_state
            .as_ref()
            .is_some_and(|rs| rs.pending_escalation.is_some())
    }

    /// Send the user's verdict back to the orchestrator's escalation
    /// handler and clear the pending escalation. No-op if none pending.
    fn run_respond(
        &mut self,
        verdict: megaresearcher_research::orchestrator::escalation::EscalationVerdict,
    ) {
        if let Some(rs) = self.run_state.as_mut() {
            if let Some(tx) = rs.verdict_responder.take() {
                let _ = tx.send(verdict);
            }
            rs.pending_escalation = None;
        }
    }

    /// Pure render. Dispatches to the current surface. TestBackend-testable.
    pub fn render(&self, frame: &mut ratatui::Frame) {
        match self.surface {
            Surface::Start => {
                render_start_frame(frame, frame.area(), &self.question, &self.theme);
            }
            Surface::Converge => {
                if let Some(cs) = &self.converge {
                    crate::widget::inline_chat::render_converge(
                        frame,
                        frame.area(),
                        &cs.conversation,
                        cs.spec_approved,
                        cs.plan_approved,
                        &self.theme,
                    );
                }
            }
            Surface::Run => {
                if let Some(rs) = self.run_state.as_ref() {
                    crate::surface::run::render_run(frame, frame.area(), rs, &self.theme);
                }
            }
            _ => {
                // Placeholder until the surface modules land (T9-T12).
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
    /// `s` -> Settings. On `SubmitQuestion`, if a provider is set, spawn the
    /// Converge session and transition to the Converge surface; each frame
    /// drains the session's prints into the conversation buffer and
    /// non-blocking-probes the task — on `Approved` it transitions to Run,
    /// on `MaxTurns`/error it surfaces an inline message and stays.
    pub async fn run(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> anyhow::Result<()> {
        loop {
            self.frame_count = self.frame_count.wrapping_add(1);

            // Drain the Converge session's prints into the conversation buffer
            // before drawing, so the frame reflects the latest model output.
            if let Some(cs) = self.converge.as_mut() {
                while let Ok(msg) = cs.handle.print_rx.try_recv() {
                    cs.conversation.push(msg);
                }
            }
            // Run surface: drain a pending escalation non-blocking, re-read
            // swarm-state.yaml every ~5 frames (≈250ms), and probe the
            // orchestrator task for completion (-> Artifact). Drawn next.
            if self.surface == Surface::Run {
                if let Some(rs) = self.run_state.as_mut() {
                    // Pick up an escalation if one arrived and none is pending.
                    if rs.pending_escalation.is_none() {
                        if let Ok((esc, tx)) = rs.escalation_rx.try_recv() {
                            rs.pending_escalation = Some(esc);
                            rs.verdict_responder = Some(tx);
                        }
                    }
                    // Re-read swarm-state.yaml every ~5 frames so the tree
                    // re-renders as the run progresses.
                    if self.frame_count.is_multiple_of(5) {
                        let swarm_path = rs.run_dir.join("swarm-state.yaml");
                        if swarm_path.exists() {
                            if let Ok(swarm) =
                                megaresearcher_research::state::swarm_state::SwarmState::read(
                                    &swarm_path,
                                )
                            {
                                rs.swarm = Some(swarm);
                            }
                        }
                    }
                }
                // Non-blocking probe of the orchestrator task: on
                // completion, transition to the Artifact surface.
                if let Some(task) = self.run_state.as_mut().and_then(|rs| rs.orch_task.as_mut()) {
                    let mut done = std::pin::Pin::new(task).fuse();
                    tokio::select! {
                        biased;
                        _ = &mut done => {
                            if let Some(rs) = self.run_state.as_mut() {
                                rs.orch_task = None;
                            }
                            self.surface = Surface::Artifact;
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_millis(0)) => {}
                    }
                }
            }
            terminal.draw(|f| self.render(f))?;

            // Non-blocking probe of the Converge task: a biased select against
            // a 0ms sleep never blocks the event loop, so `q` still quits
            // while the session runs.
            if let Some(task) = self.converge_task.as_mut() {
                let mut done = std::pin::Pin::new(task).fuse();
                tokio::select! {
                    biased;
                    res = &mut done => {
                        let cs = self.converge.as_mut().expect("converge state missing");
                        match res {
                            Ok(Ok(megaresearcher_research::phases::DriveOutcome::Approved {
                                gates_passed: 2,
                            })) => {
                                cs.spec_approved = true;
                                cs.plan_approved = true;
                                cs.outcome = Some(ConvergeOutcome::Approved {
                                    spec_path: cs.spec_path.clone(),
                                    plan_path: cs.plan_path.clone(),
                                });
                                self.converge_task = None;
                                self.surface = Surface::Run;
                            }
                            Ok(Ok(megaresearcher_research::phases::DriveOutcome::Approved {
                                gates_passed,
                            })) => {
                                cs.conversation.push(format!(
                                    "converge approved with {gates_passed} gates (expected 2)"
                                ));
                                cs.outcome = Some(ConvergeOutcome::Failed(format!(
                                    "approved with {gates_passed} gates, expected 2"
                                )));
                                self.converge_task = None;
                            }
                            Ok(Ok(megaresearcher_research::phases::DriveOutcome::MaxTurns)) => {
                                cs.conversation.push(
                                    "converge hit the turn ceiling before both approvals.".into(),
                                );
                                cs.outcome = Some(ConvergeOutcome::MaxTurns);
                                self.converge_task = None;
                            }
                            Ok(Err(e)) => {
                                cs.conversation.push(format!("converge session failed: {e}"));
                                cs.outcome = Some(ConvergeOutcome::Failed(e.to_string()));
                                self.converge_task = None;
                            }
                            Err(e) => {
                                cs.conversation.push(format!("converge task panicked: {e}"));
                                cs.outcome = Some(ConvergeOutcome::Failed(e.to_string()));
                                self.converge_task = None;
                            }
                        }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(0)) => {}
                }
            }

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
                        if let Some((p, m)) = self.provider.clone() {
                            let q = std::mem::take(&mut self.question);
                            let (state, task) =
                                crate::surface::start::spawn_converge(&self.cwd, &q, p, m);
                            self.converge = Some(state);
                            self.converge_task = Some(task);
                            self.surface = Surface::Converge;
                        }
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
