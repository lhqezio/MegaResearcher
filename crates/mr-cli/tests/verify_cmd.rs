//! Task 8 — `mr verify <run_dir>` + top-level `--help` usage.
//!
//! Mirrors the T4 passing fixture (`crates/research/tests/verify.rs:38-90`):
//! a run dir with `output.md` + `swarm-state.yaml` (written via
//! `SwarmState::write`) + one worker subdir holding the 3 artifacts. The one
//! deliberate difference from T4 is that `spec_path` in the swarm state is the
//! ABSOLUTE path to the spec file the test writes, because `mr verify` reads
//! `spec_path` back from `swarm-state.yaml` (T4 passed the spec path directly
//! to `verify_run`).

mod common;

use common::fake_provider::FakeProvider;
use megaresearcher_research::flows::load_embedded;
use megaresearcher_research::state::swarm_state::SwarmState;
use mr_cli::commands::verify::run as verify_run_cmd;
use mr_cli::{run_cli, usage};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use tempfile::tempdir;

// Mirrors T4's `PASSING_OUTPUT` — the 8 synth sections, with "Identify gaps
// in prior art" appearing in the Executive summary so group E matches the
// spec's success criterion.
const PASSING_OUTPUT: &str = "\
# Research direction

## Executive summary
A short summary. Identify gaps in prior art via cross-reading.

## Surviving hypotheses
- S1: some hypothesis

## Rejected and killed hypotheses
- (none killed)

## Escalations
- (none)

## What we did NOT explore
- We did NOT explore modality Z.

## Recommended next actions
- Run the experiment for S1.

## Run metadata
- run_id: 2026-06-27-1430-abc123

## Sources
- arxiv 2401.12345 — Paper A
";

const SPEC: &str = "# Spec\n\n## Success criteria\n- Identify gaps in prior art\n\n## Other\n";

fn write_worker(dir: &std::path::Path, name: &str) {
    let w = dir.join(name);
    fs::create_dir_all(&w).unwrap();
    fs::write(w.join("output.md"), "worker output").unwrap();
    fs::write(w.join("manifest.yaml"), "name: worker\n").unwrap();
    fs::write(w.join("verification.md"), "ok\n").unwrap();
}

#[tokio::test]
async fn verify_command_writes_report_for_passing_run() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/2026-06-27-1430-abc123");
    fs::create_dir_all(&run_dir).unwrap();
    fs::write(run_dir.join("output.md"), PASSING_OUTPUT).unwrap();
    let spec_path = tmp.path().join("spec.md");
    fs::write(&spec_path, SPEC).unwrap();
    write_worker(&run_dir, "literature-scout-1");

    // swarm-state with ABSOLUTE spec_path (mr verify reads it back).
    let swarm = SwarmState {
        run_id: "2026-06-27-1430-abc123".into(),
        spec_path: spec_path.to_string_lossy().to_string(),
        plan_path: "plan.md".into(),
        novelty_target: "gap-finding".into(),
        max_parallel: 4,
        phases: vec![],
        escalations: vec![],
        retry_counts: HashMap::new(),
    };
    swarm.write(&run_dir.join("swarm-state.yaml")).unwrap();

    let cwd = tmp.path().to_path_buf();
    let provider = (
        Arc::new(FakeProvider::new("fake", vec![])) as Arc<dyn claurst_api::LlmProvider>,
        "fake".to_string(),
    );
    verify_run_cmd(&cwd, provider, run_dir.clone())
        .await
        .unwrap();
    assert!(
        run_dir.join("verification-report.md").exists(),
        "verify should write the report"
    );
}

#[test]
fn usage_lists_subcommands_and_flow_descriptions() {
    let u = usage();
    for sub in [
        "init",
        "brainstorm",
        "spec",
        "plan",
        "execute",
        "verify",
        "watch",
        "list",
    ] {
        assert!(u.contains(sub), "usage missing subcommand {sub}");
    }
    // brainstorm/spec/plan descriptions come from the embedded flow assets.
    assert!(
        u.contains(&load_embedded("brainstorm").description),
        "usage missing brainstorm description"
    );
    assert!(
        u.contains(&load_embedded("spec").description),
        "usage missing spec description"
    );
}

#[tokio::test]
async fn run_cli_help_returns_ok_without_provider() {
    // run_cli intercepts --help before resolve_provider → no API key needed.
    let res = run_cli(vec!["mr".into(), "--help".into()]).await;
    assert!(
        res.is_ok(),
        "run_cli --help should return Ok without a provider: {:?}",
        res
    );
}
