//! Headless escalation handler for `mr execute --on-escalate`. The TUI (Phase
//! 6b) supplies an interactive handler; the CLI (T7) supplies this one. It
//! adjudicates via the `OnEscalate` mode without spawning a full TUI.

use std::sync::Arc;

use async_trait::async_trait;
use megaresearcher_research::orchestrator::escalation::{EscalationHandler, EscalationVerdict};
use megaresearcher_research::phases::UserIo;
use megaresearcher_research::state::swarm_state::Escalation;

use crate::OnEscalate;

pub struct HeadlessEscalationHandler {
    pub mode: OnEscalate,
    pub io: Arc<dyn UserIo>,
}

#[async_trait]
impl EscalationHandler for HeadlessEscalationHandler {
    async fn adjudicate(&self, e: &Escalation) -> EscalationVerdict {
        match self.mode {
            OnEscalate::Continue => EscalationVerdict::Continue,
            OnEscalate::Fail => EscalationVerdict::Fail,
            OnEscalate::Pause => {
                let _ = self
                    .io
                    .print(&format!(
                        "\n[escalation] {} ({}): continue? [y/n] ",
                        e.worker, e.reason
                    ))
                    .await;
                let line = self.io.read_line().await.unwrap_or_default();
                let t = line.trim().to_lowercase();
                if t == "y" || t == "c" || t == "continue" {
                    EscalationVerdict::Continue
                } else {
                    EscalationVerdict::Fail
                }
            }
        }
    }
}
