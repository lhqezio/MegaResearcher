//! run_id, swarm_state serde, and run_tree tests.

use std::collections::HashMap;
use std::path::PathBuf;

use megaresearcher_research::state::run_id::{generate_run_id, run_id_from_parts};
use megaresearcher_research::state::run_tree::{create_run_tree, run_dir};
use megaresearcher_research::state::swarm_state::{Escalation, Phase, SwarmState, Worker};

// ---- run_id -------------------------------------------------------------

#[test]
fn test_run_id_from_parts() {
    assert_eq!(
        run_id_from_parts("2026-06-27-1430", "a1b2c3"),
        "2026-06-27-1430-a1b2c3"
    );
}

#[test]
fn test_generate_run_id_format() {
    let id = generate_run_id().unwrap();
    let parts: Vec<&str> = id.split('-').collect();
    assert_eq!(
        parts.len(),
        5,
        "run-id must have 5 dash-separated parts: {}",
        id
    );
    assert_eq!(parts[0].len(), 4, "year (4 digits): {}", id);
    assert!(
        parts[0].chars().all(|c| c.is_ascii_digit()),
        "year digits: {}",
        id
    );
    assert_eq!(parts[1].len(), 2, "month: {}", id);
    assert!(
        parts[1].chars().all(|c| c.is_ascii_digit()),
        "month digits: {}",
        id
    );
    assert_eq!(parts[2].len(), 2, "day: {}", id);
    assert!(
        parts[2].chars().all(|c| c.is_ascii_digit()),
        "day digits: {}",
        id
    );
    assert_eq!(parts[3].len(), 4, "HHMM: {}", id);
    assert!(
        parts[3].chars().all(|c| c.is_ascii_digit()),
        "HHMM digits: {}",
        id
    );
    assert_eq!(parts[4].len(), 6, "hex6: {}", id);
    assert!(
        parts[4]
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, 'a'..='f')),
        "hex6 must be 6 lowercase hex chars: {}",
        id
    );
}

// ---- swarm_state serde --------------------------------------------------

fn sample_state() -> SwarmState {
    SwarmState {
        run_id: "2026-06-27-1430-a1b2c3".to_string(),
        spec_path: "docs/research/specs/foo.md".to_string(),
        plan_path: "docs/research/plans/foo.md".to_string(),
        novelty_target: "hypothesis".to_string(),
        max_parallel: 4,
        phases: vec![Phase {
            name: "phase_1_literature_scout".to_string(),
            status: "pending".to_string(),
            workers: vec![Worker {
                name: "scout-1".to_string(),
                status: "pending".to_string(),
            }],
            hypotheses: vec![],
        }],
        escalations: vec![Escalation {
            worker: "hypothesis-smith-3".to_string(),
            reason: "red-team KILL after 3 rounds".to_string(),
            retry_count: 3,
        }],
        retry_counts: {
            let mut m = HashMap::new();
            m.insert("hypothesis-smith-3".to_string(), 3u32);
            m
        },
    }
}

#[test]
fn test_swarm_state_file_roundtrip() {
    let state = sample_state();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swarm-state.yaml");
    state.write(&path).unwrap();
    let loaded = SwarmState::read(&path).unwrap();
    assert_eq!(state, loaded);
}

#[test]
fn test_swarm_state_minimal_roundtrip() {
    // Empty phases/escalations/retry_counts must round-trip too.
    let state = SwarmState {
        run_id: "2026-06-27-0900-000000".to_string(),
        spec_path: "s".to_string(),
        plan_path: "p".to_string(),
        novelty_target: "gap-finding".to_string(),
        max_parallel: 4,
        phases: vec![],
        escalations: vec![],
        retry_counts: HashMap::new(),
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swarm-state.yaml");
    state.write(&path).unwrap();
    let loaded = SwarmState::read(&path).unwrap();
    assert_eq!(state, loaded);
}

#[test]
fn phase_with_hypotheses_round_trips() {
    use megaresearcher_research::state::swarm_state::{HypothesisNode, RoundVerdict, Verdict};
    let state = SwarmState {
        run_id: "r".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
        novelty_target: "hypothesis".into(),
        max_parallel: 4,
        phases: vec![Phase {
            name: "red-team".into(),
            status: "complete".into(),
            workers: vec![],
            hypotheses: vec![
                HypothesisNode {
                    id: "hypothesis-smith-1".into(),
                    label: "causal-SAE-bridge".into(),
                    status: "killed".into(),
                    rounds: vec![
                        RoundVerdict {
                            round: 1,
                            critique: Verdict::Reject,
                            revised: true,
                        },
                        RoundVerdict {
                            round: 2,
                            critique: Verdict::Reject,
                            revised: true,
                        },
                    ],
                    kill_reason: Some("red-team KILL (irrecoverable)".into()),
                },
                HypothesisNode {
                    id: "hypothesis-smith-2".into(),
                    label: "logit-lens-circuits".into(),
                    status: "approved".into(),
                    rounds: vec![RoundVerdict {
                        round: 1,
                        critique: Verdict::Approve,
                        revised: false,
                    }],
                    kill_reason: None,
                },
            ],
        }],
        escalations: vec![],
        retry_counts: HashMap::new(),
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swarm-state.yaml");
    state.write(&path).unwrap();
    let loaded = SwarmState::read(&path).unwrap();
    assert_eq!(state, loaded);
}

#[test]
fn old_phase_yaml_without_hypotheses_deserializes_to_empty() {
    // A pre-6b swarm-state.yaml has no `hypotheses:` key on Phase. It must
    // deserialize with hypotheses == [] so the 52 orchestrator tests stay green.
    let yaml = "\
run_id: r
spec_path: s
plan_path: p
novelty_target: gap-finding
max_parallel: 4
phases:
  - name: literature-scout
    status: complete
    workers:
      - name: scout-1
        status: passed
escalations: []
retry_counts: {}
";
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swarm-state.yaml");
    std::fs::write(&path, yaml).unwrap();
    let loaded = SwarmState::read(&path).unwrap();
    assert_eq!(loaded.phases.len(), 1);
    assert!(loaded.phases[0].hypotheses.is_empty());
}

// ---- run_tree -----------------------------------------------------------

#[test]
fn test_run_dir_path() {
    let base = tempfile::tempdir().unwrap();
    let dir: PathBuf = run_dir(base.path(), "2026-06-27-1430-a1b2c3");
    assert_eq!(dir, base.path().join("runs").join("2026-06-27-1430-a1b2c3"));
    assert!(!dir.exists(), "run_dir must not touch the filesystem");
}

#[test]
fn test_create_run_tree_makes_dir() {
    let base = tempfile::tempdir().unwrap();
    let created = create_run_tree(base.path(), "2026-06-27-1430-a1b2c3").unwrap();
    assert!(created.is_dir());
    assert_eq!(
        created,
        base.path().join("runs").join("2026-06-27-1430-a1b2c3")
    );
}
