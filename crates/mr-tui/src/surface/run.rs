// The Run surface — where the user lives. Spawns the orchestrator, watches
// swarm-state.yaml (250ms poll), renders the tree + cost meter + escalation
// strip (only when an escalation is pending — no defensive UI).
//
// SPDX-License-Identifier: GPL-3.0

use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;
use claurst_core::cost::CostTracker;
use megaresearcher_research::mcp::ml_intern_config;
use megaresearcher_research::orchestrator::escalation::{EscalationHandler, EscalationVerdict};
use megaresearcher_research::orchestrator::{Orchestrator, OrchestratorConfig, RunOutcome};
use megaresearcher_research::state::run_id::generate_run_id;
use megaresearcher_research::state::swarm_state::{Escalation, SwarmState};
use tokio::sync::{mpsc, oneshot};

use crate::cost::CountingProvider;
use crate::theme::ColorPalette;

pub struct RunState {
    pub run_dir: PathBuf,
    pub run_id: String,
    pub swarm: Option<SwarmState>,
    pub cost_tracker: Arc<CostTracker>,
    pub pending_escalation: Option<Escalation>,
    pub escalation_tx: mpsc::UnboundedSender<(Escalation, oneshot::Sender<EscalationVerdict>)>,
    pub escalation_rx: mpsc::UnboundedReceiver<(Escalation, oneshot::Sender<EscalationVerdict>)>,
    pub verdict_responder: Option<oneshot::Sender<EscalationVerdict>>,
    pub orch_task: Option<
        tokio::task::JoinHandle<
            Result<RunOutcome, megaresearcher_research::orchestrator::OrchestratorError>,
        >,
    >,
}

impl RunState {
    /// Test-only constructor: a RunState holding a fixture swarm, no
    /// orchestrator task. Used by the make-or-break render test.
    pub fn from_swarm_for_test(swarm: SwarmState) -> Self {
        let (esc_tx, esc_rx) = mpsc::unbounded_channel();
        Self {
            run_dir: PathBuf::from("/tmp/run"),
            run_id: "test".into(),
            swarm: Some(swarm),
            cost_tracker: CostTracker::with_model("claude-sonnet-4-6"),
            pending_escalation: None,
            escalation_tx: esc_tx,
            escalation_rx: esc_rx,
            verdict_responder: None,
            orch_task: None,
        }
    }
}

/// Spawn the orchestrator on a tokio task. Wraps the provider in a
/// CountingProvider so the cost meter reads live. The TuiEscalationHandler
/// routes escalations into the RunState's channel for inline adjudication.
pub fn spawn_run(
    cwd: &Path,
    spec_path: PathBuf,
    plan_path: PathBuf,
    provider: Arc<dyn LlmProvider>,
    model: String,
    max_parallel: u32,
    mcp_enabled: bool,
) -> anyhow::Result<(
    RunState,
    tokio::task::JoinHandle<
        Result<RunOutcome, megaresearcher_research::orchestrator::OrchestratorError>,
    >,
)> {
    let tracker = CostTracker::with_model(&model);
    let counting = Arc::new(CountingProvider::new(provider, tracker.clone()));
    let (esc_tx, esc_rx) =
        mpsc::unbounded_channel::<(Escalation, oneshot::Sender<EscalationVerdict>)>();
    let handler = Arc::new(crate::escalation::TuiEscalationHandler {
        escalation_tx: esc_tx.clone(),
    });
    let docs = cwd.join("docs/research");
    let mcp = if mcp_enabled {
        Some(ml_intern_config(cwd))
    } else {
        None
    };
    let cfg = OrchestratorConfig {
        research_base: docs.clone(),
        agents_dir: cwd.join("agents"),
        default_model: model,
        max_parallel,
        mcp,
        escalation: Some(handler as Arc<dyn EscalationHandler>),
    };
    let orch = Orchestrator::new(cfg, counting as Arc<dyn LlmProvider>);
    let run_id = generate_run_id().map_err(anyhow::Error::from)?;
    let run_dir = docs.join("runs").join(&run_id);
    let spec_c = spec_path.clone();
    let plan_c = plan_path.clone();
    let run_id_c = run_id.clone();
    let task = tokio::spawn(async move { orch.execute(&spec_c, &plan_c, &run_id_c).await });
    let state = RunState {
        run_dir,
        run_id,
        swarm: None,
        cost_tracker: tracker,
        pending_escalation: None,
        escalation_tx: esc_tx,
        escalation_rx: esc_rx,
        verdict_responder: None,
        orch_task: None,
    };
    // Fix 2: take the task out before returning state — avoiding the
    // borrow-after-move that `Ok((state, state.orch_task.unwrap()))` causes.
    Ok((state, task))
}

/// Render the Run surface: tree (top) + cost meter (top-right) + escalation
/// strip (bottom, only when an escalation is pending).
pub fn render_run(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    state: &RunState,
    theme: &ColorPalette,
) {
    let chunks = ratatui::layout::Layout::default()
        .constraints(if state.pending_escalation.is_some() {
            vec![
                ratatui::layout::Constraint::Min(3),
                ratatui::layout::Constraint::Length(3),
            ]
        } else {
            vec![ratatui::layout::Constraint::Min(1)]
        })
        .split(area);
    // Top: tree + cost meter. Split the top into tree (left) + cost (right, 16 cols).
    let top = ratatui::layout::Layout::default()
        .constraints([
            ratatui::layout::Constraint::Min(16),
            ratatui::layout::Constraint::Length(16),
        ])
        .split(chunks[0]);
    if let Some(swarm) = state.swarm.as_ref() {
        crate::widget::tree::render_tree(frame, top[0], swarm, theme);
    } else {
        frame.render_widget(
            ratatui::widgets::Paragraph::new("waiting for run to start…")
                .style(ratatui::style::Style::default().fg(theme.disabled)),
            top[0],
        );
    }
    crate::cost::render_cost_meter(frame, top[1], &state.cost_tracker, theme);
    // Escalation strip — only when pending (no empty "(none)" box).
    if let Some(esc) = state.pending_escalation.as_ref() {
        let line = ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(
                format!(
                    "{} escalation · {}: \"{}\"",
                    crate::figures::WARN,
                    esc.worker,
                    esc.reason
                ),
                ratatui::style::Style::default().fg(theme.escalation),
            ),
            ratatui::text::Span::raw("  "),
            ratatui::text::Span::styled(
                "[ continue ▸  fail ]",
                ratatui::style::Style::default().fg(theme.action),
            ),
        ]);
        frame.render_widget(
            ratatui::widgets::Paragraph::new(line)
                .block(ratatui::widgets::Block::default().borders(ratatui::widgets::Borders::TOP)),
            chunks[1],
        );
    }
}
