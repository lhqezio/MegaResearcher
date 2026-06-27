//! The escalation-adjudication seam (design §112): when a worker escalates,
//! the orchestrator asks the handler whether to continue (record + return a
//! partial `RunOutcome`) or fail (return `Err`). `None` means fail —
//! preserving the pre-Phase-6a behavior the 52 orchestrator tests depend on.
//! The TUI (Phase 6b) supplies an interactive handler; the CLI (T7) supplies
//! `HeadlessEscalationHandler` for `--on-escalate`.

use async_trait::async_trait;

use crate::state::swarm_state::Escalation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationVerdict {
    Continue,
    Fail,
}

#[async_trait]
pub trait EscalationHandler: Send + Sync {
    async fn adjudicate(&self, escalation: &Escalation) -> EscalationVerdict;
}

pub struct FailAll;

#[async_trait]
impl EscalationHandler for FailAll {
    async fn adjudicate(&self, _e: &Escalation) -> EscalationVerdict {
        EscalationVerdict::Fail
    }
}
