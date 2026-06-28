//! `swarm-state.yaml` schema + (de)serialization. The orchestrator (Phase 4) is
//! the single writer; the TUI is a read-only view. Schema per the design spec and
//! the orchestrator skill: run_id, spec_path, plan_path, novelty_target,
//! max_parallel, phases (each status + workers), escalations, retry_counts.
//!
//! `#[serde(default)]` is applied to the optional/collection fields so Phase 4
//! can extend the structs without breaking deserialization of Phase-2-written
//! files, and vice versa.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

fn default_max_parallel() -> u32 {
    4
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwarmState {
    pub run_id: String,
    pub spec_path: String,
    pub plan_path: String,
    pub novelty_target: String,
    #[serde(default = "default_max_parallel")]
    pub max_parallel: u32,
    #[serde(default)]
    pub phases: Vec<Phase>,
    #[serde(default)]
    pub escalations: Vec<Escalation>,
    #[serde(default)]
    pub retry_counts: HashMap<String, u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phase {
    pub name: String,
    pub status: String,
    #[serde(default)]
    pub workers: Vec<Worker>,
    #[serde(default)]
    pub hypotheses: Vec<HypothesisNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Worker {
    pub name: String,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Escalation {
    pub worker: String,
    pub reason: String,
    #[serde(default)]
    pub retry_count: u32,
}

/// A red-team verdict on one round of a hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Approve,
    Reject,
}

/// One round of the red-team critique loop for a hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundVerdict {
    pub round: u32,
    pub critique: Verdict,
    pub revised: bool,
}

/// A hypothesis as persisted for the audit-trail tree (spec §8). Additive +
/// serde-defaulted empty so pre-6b `swarm-state.yaml` files deserialize unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypothesisNode {
    pub id: String,
    pub label: String,
    pub status: String,
    #[serde(default)]
    pub rounds: Vec<RoundVerdict>,
    pub kill_reason: Option<String>,
}

impl SwarmState {
    /// Read `swarm-state.yaml` from `path`.
    pub fn read(path: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        serde_yml::from_str(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Write `self` to `path` as YAML.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let text = serde_yml::to_string(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, text)
    }
}
