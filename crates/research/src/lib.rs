//! MegaResearcher v1 — the research layer.
//!
//! Phase 2 ports the pure-logic paper-chain modules and the run-state module
//! from the v0 Python plugin (`lib/paper_chain/*` plus the orchestrator skill's
//! `swarm-state.yaml` schema) to deterministic Rust. No LLM is involved yet —
//! these are 1:1 ports with 1:1 tests. Later phases add the orchestrator, the
//! worker primitive, the front-half guided phases, the HTML export, and the
//! doom-loop discipline trait.

pub mod flows;
pub mod mcp;
pub mod orchestrator;
pub mod paper_chain;
pub mod phases;
pub mod prompt_asset;
pub mod state;
pub mod verify;
pub mod worker;
pub mod worker_tools;

pub const CRATE_NAME: &str = "megaresearcher-research";
