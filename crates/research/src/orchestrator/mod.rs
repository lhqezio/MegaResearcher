//! The deterministic swarm orchestrator (Phase 4). Drives leaf `Worker`s
//! through the six phases, runs the verification gate, assembles
//! consolidations, and finalizes the run. See the design spec §4/§10/§11.

pub mod dispatch;
pub mod dispatch_plan;
pub mod preflight;

use std::io;

use crate::worker::WorkerError;

/// Orchestrator-wide error.
#[derive(Debug)]
pub enum OrchestratorError {
    Preflight(String),
    Parse(String),
    Io(io::Error),
    Worker(WorkerError),
    Escalated(Vec<String>),
    Finalize(String),
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preflight(s) => write!(f, "pre-flight failed: {s}"),
            Self::Parse(s) => write!(f, "plan parse failed: {s}"),
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Worker(e) => write!(f, "worker error: {e:?}"),
            Self::Escalated(names) => write!(f, "workers escalated: {names:?}"),
            Self::Finalize(s) => write!(f, "finalize failed: {s}"),
        }
    }
}

impl std::error::Error for OrchestratorError {}

impl From<io::Error> for OrchestratorError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<WorkerError> for OrchestratorError {
    fn from(e: WorkerError) -> Self {
        Self::Worker(e)
    }
}