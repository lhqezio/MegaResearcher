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

use std::path::PathBuf;
use std::sync::Arc;

use megaresearcher_research::flows::load_embedded;
use megaresearcher_research::phases::{drive_session, DriveOutcome, Gate, GuidedSession};
use megaresearcher_research::worker_tools::{ScopedRead, ScopedWrite, Tool};

use crate::io::{tui_user_io, TuiUserIoHandle};

/// The outcome of a Converge session: the user approved both gates (artifacts
/// on disk), the turn ceiling was hit, or the session errored.
#[derive(Debug, Clone)]
pub enum ConvergeOutcome {
    Approved {
        spec_path: PathBuf,
        plan_path: PathBuf,
    },
    MaxTurns,
    Failed(String),
}

/// The App-side view of a running Converge session. The App drains
/// `handle.print_rx` into `conversation` each frame and feeds approvals into
/// `handle.input_tx`; `spawn_converge` owns the session side on a tokio task.
pub struct ConvergeState {
    pub handle: TuiUserIoHandle,
    pub conversation: Vec<String>,
    pub spec_path: PathBuf,
    pub plan_path: PathBuf,
    pub spec_approved: bool,
    pub plan_approved: bool,
    pub outcome: Option<ConvergeOutcome>,
}

/// Build + spawn the converge session on a tokio task. Returns the
/// `(ConvergeState, JoinHandle)` so the App can drain prints and feed
/// approvals. Mirrors `init.rs::run_with` exactly: concatenated flow bodies,
/// ScopedRead/ScopedWrite tools, gates spec+plan, approve_words, GuidedSession
/// args, inject_user "Topic: {question}".
pub fn spawn_converge(
    cwd: &std::path::Path,
    question: &str,
    provider: Arc<dyn claurst_api::LlmProvider>,
    model: String,
) -> (
    ConvergeState,
    tokio::task::JoinHandle<anyhow::Result<DriveOutcome>>,
) {
    let (io, handle) = tui_user_io();
    let docs = cwd.join("docs/research");
    std::fs::create_dir_all(docs.join("specs")).ok();
    std::fs::create_dir_all(docs.join("plans")).ok();
    let body = format!(
        "# Phase 1 — Brainstorm\n\n{}\n\n# Phase 2 — Spec\n\n{}\n\n# Phase 3 — Plan\n\n{}\n\n\
         Drive all three phases in order. Pause for user approval after the spec and again after the plan.",
        load_embedded("brainstorm").body,
        load_embedded("spec").body,
        load_embedded("plan").body,
    );
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ScopedRead::with_shared(docs.clone(), docs.clone())),
        Arc::new(ScopedWrite::new(docs.clone())),
    ];
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let slug = question
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    let spec_path = docs.join("specs").join(format!("{date}-{slug}-spec.md"));
    let plan_path = docs.join("plans").join(format!("{date}-{slug}-plan.md"));
    let mut session = GuidedSession::new(body, tools, provider, model, 4096, 60);
    session.inject_user(&format!("Topic: {question}"));
    let gates = vec![
        Gate {
            artifact: spec_path.clone(),
            label: "spec".into(),
        },
        Gate {
            artifact: plan_path.clone(),
            label: "plan".into(),
        },
    ];
    let task = tokio::spawn(async move {
        drive_session(&mut session, &io, gates, &["approve", "yes", "y", "done"])
            .await
            .map_err(anyhow::Error::from)
    });
    let state = ConvergeState {
        handle,
        conversation: Vec::new(),
        spec_path,
        plan_path,
        spec_approved: false,
        plan_approved: false,
        outcome: None,
    };
    (state, task)
}
