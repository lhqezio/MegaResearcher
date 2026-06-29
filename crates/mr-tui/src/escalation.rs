// TuiEscalationHandler — the TUI counterpart to 6a's HeadlessEscalationHandler.
// On adjudicate, sends the Escalation into an in-App channel and blocks
// (awaits a oneshot) until the user adjudicates inline from the run strip.
//
// SPDX-License-Identifier: GPL-3.0

use async_trait::async_trait;
use megaresearcher_research::orchestrator::escalation::{EscalationHandler, EscalationVerdict};
use megaresearcher_research::state::swarm_state::Escalation;
use tokio::sync::{mpsc, oneshot};

pub struct TuiEscalationHandler {
    pub escalation_tx: mpsc::UnboundedSender<(Escalation, oneshot::Sender<EscalationVerdict>)>,
}

#[async_trait]
impl EscalationHandler for TuiEscalationHandler {
    async fn adjudicate(&self, e: &Escalation) -> EscalationVerdict {
        let (tx, rx) = oneshot::channel();
        if self.escalation_tx.send((e.clone(), tx)).is_err() {
            return EscalationVerdict::Fail;
        }
        rx.await.unwrap_or(EscalationVerdict::Fail)
    }
}
