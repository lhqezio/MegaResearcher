//! MegaResearcher v1 — the research layer.
//!
//! This crate holds the swarm orchestrator, the leaf worker primitive, the
//! front-half guided phases, the paper chain, run-state management, the HTML
//! export, and the doom-loop discipline trait. All of those are implemented in
//! later phases; Phase 1 only stands the crate up so the workspace includes it.

pub const CRATE_NAME: &str = "megaresearcher-research";
