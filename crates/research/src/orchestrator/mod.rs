//! The deterministic swarm orchestrator (Phase 4). Drives leaf `Worker`s
//! through the six phases, runs the verification gate, assembles
//! consolidations, and finalizes the run. See the design spec §4/§10/§11.

pub mod dispatch_plan;