//! Task 4 — `research::verify` deterministic post-run checker.
//!
//! Fixtures are built from `tempfile::tempdir()` per Correction D: a run dir
//! with `output.md` + `swarm-state.yaml` (written via `SwarmState::write`) +
//! worker subdirs holding the 3 artifacts. A local `FakeMcp` resolves every
//! `hf_papers` call.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use claurst_mcp::{CallToolResult, McpContent};
use megaresearcher_research::mcp::{McpCaller, McpError};
use megaresearcher_research::state::swarm_state::{Escalation, Phase, SwarmState, Worker};
use megaresearcher_research::verify::{verify_run, Verdict};
use serde_json::json;
use tempfile::tempdir;

struct FakeMcp;
#[async_trait]
impl McpCaller for FakeMcp {
    async fn call_tool(
        &self,
        _name: &str,
        _args: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult {
            content: vec![McpContent::Text {
                text: "{\"title\":\"ok\"}".into(),
            }],
            is_error: false,
        })
    }
}

const PASSING_OUTPUT: &str = "\
# Research direction

## Executive summary
A short summary of the run. Identify gaps in prior art via cross-reading.

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
- arxiv 2402.23456 — Paper B
- arxiv 2403.34567 — Paper C
";

const SPEC: &str = "\
# Spec

## Success criteria
- Identify gaps in prior art

## Other
";

fn write_worker(dir: &std::path::Path, name: &str) {
    let wdir = dir.join(name);
    fs::create_dir_all(&wdir).unwrap();
    fs::write(wdir.join("output.md"), "worker output").unwrap();
    fs::write(wdir.join("manifest.yaml"), "name: worker\n").unwrap();
    fs::write(wdir.join("verification.md"), "ok\n").unwrap();
}

fn passing_swarm() -> SwarmState {
    SwarmState {
        run_id: "2026-06-27-1430-abc123".into(),
        spec_path: "spec.md".into(),
        plan_path: "plan.md".into(),
        novelty_target: "gap-finding".into(),
        max_parallel: 4,
        phases: vec![Phase {
            name: "scout".into(),
            status: "done".into(),
            workers: vec![Worker {
                name: "literature-scout-1".into(),
                status: "done".into(),
            }],
            hypotheses: vec![],
        }],
        escalations: vec![],
        retry_counts: HashMap::new(),
    }
}

fn write_passing_fixture(run_dir: &std::path::Path) -> PathBuf {
    fs::write(run_dir.join("output.md"), PASSING_OUTPUT).unwrap();
    write_worker(run_dir, "literature-scout-1");
    passing_swarm()
        .write(&run_dir.join("swarm-state.yaml"))
        .unwrap();
    let spec_path = run_dir.join("spec.md");
    fs::write(&spec_path, SPEC).unwrap();
    spec_path
}

#[tokio::test]
async fn passing_run_yields_pass() {
    let dir = tempdir().unwrap();
    let spec_path = write_passing_fixture(dir.path());

    let r = verify_run(dir.path(), &spec_path, None).await.unwrap();
    assert_eq!(r.verdict, Verdict::Pass);
}

#[tokio::test]
async fn missing_output_yields_fail() {
    let dir = tempdir().unwrap();
    let run_dir = dir.path();
    // No output.md at the run root.
    write_worker(run_dir, "literature-scout-1");
    passing_swarm()
        .write(&run_dir.join("swarm-state.yaml"))
        .unwrap();
    let spec_path = run_dir.join("spec.md");
    fs::write(&spec_path, SPEC).unwrap();

    let r = verify_run(run_dir, &spec_path, None).await.unwrap();
    assert_eq!(r.verdict, Verdict::Fail);
}

#[tokio::test]
async fn hidden_rejection_yields_fail() {
    let dir = tempdir().unwrap();
    let run_dir = dir.path();
    // output.md has all 8 sections but never names the escalated worker.
    fs::write(run_dir.join("output.md"), PASSING_OUTPUT).unwrap();
    write_worker(run_dir, "literature-scout-1");
    let mut swarm = passing_swarm();
    swarm.escalations.push(Escalation {
        worker: "literature-scout-1".into(),
        reason: "timeout".into(),
        retry_count: 1,
    });
    swarm.write(&run_dir.join("swarm-state.yaml")).unwrap();
    let spec_path = run_dir.join("spec.md");
    fs::write(&spec_path, SPEC).unwrap();

    let r = verify_run(run_dir, &spec_path, None).await.unwrap();
    assert_eq!(r.verdict, Verdict::Fail);
    let b_fail = r
        .checks
        .iter()
        .any(|c| c.group == 'B' && !c.passed && c.item.contains("literature-scout-1"));
    assert!(
        b_fail,
        "expected a group-B failure mentioning literature-scout-1, got: {:?}",
        r.checks
    );
}

#[tokio::test]
async fn no_mcp_skips_group_d_but_still_passes() {
    let dir = tempdir().unwrap();
    let spec_path = write_passing_fixture(dir.path());

    let r = verify_run(dir.path(), &spec_path, None).await.unwrap();
    assert_eq!(r.verdict, Verdict::Pass);
    let d_skip = r
        .checks
        .iter()
        .any(|c| c.group == 'D' && c.detail.as_deref() == Some("skipped (no MCP)"));
    assert!(
        d_skip,
        "expected a group-D skipped (no MCP) check, got: {:?}",
        r.checks
    );
}

#[tokio::test]
async fn with_mcp_three_spot_checks_run() {
    let dir = tempdir().unwrap();
    let spec_path = write_passing_fixture(dir.path());

    let r = verify_run(dir.path(), &spec_path, Some(Arc::new(FakeMcp)))
        .await
        .unwrap();
    assert_eq!(r.spot_checks.len(), 3);
    assert!(
        r.spot_checks.iter().all(|s| s.resolved),
        "expected all spot-checks resolved, got: {:?}",
        r.spot_checks
    );
}

// Keep `json` referenced for clarity of the MCP args shape used in production.
#[allow(dead_code)]
fn _mcp_args_shape() -> serde_json::Value {
    json!({ "operation": "paper_details", "arxiv_id": "0000.00000" })
}
