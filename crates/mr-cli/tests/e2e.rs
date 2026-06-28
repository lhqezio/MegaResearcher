//! Task 9 — end-to-end integration test for the `mr` CLI chain.
//!
//! Drives the full seam stack `mr init` -> `mr execute --no-mcp` ->
//! `mr verify` against a temp cwd with a fake provider and fixture
//! agents/, asserting that the chain produces the spec, the plan, a run
//! dir with `output.md`, and a `verification-report.md`.
//!
//! The subtle wiring point: `mr execute` (with `plan=None`) discovers the
//! plan that `mr init` wrote via `latest_plan`, then `parse_plan`s it to
//! learn the worker count. For the fake provider's scripted 16 turns
//! (`run_turns(4)` = 4 workers x 4 turns) to line up, the plan `mr init`
//! writes MUST be the gap-finding fixture (2 scouts + 1 gap-finder +
//! 1 synthesist = 4 workers). So `mr init`'s FakeProvider emits turns
//! that write the gap-finding fixture content into the spec + plan files.

mod common;
use common::fake_provider::FakeProvider;
use common::turns::{final_turn, run_turns, write_turn};

use std::fs;
use std::sync::Arc;

use async_trait::async_trait;
use tempfile::tempdir;

use claurst_api::LlmProvider;
use megaresearcher_research::phases::UserIo;
use mr_cli::commands::execute::run_with as execute_run_with;
use mr_cli::commands::init::run_with as init_run_with;
use mr_cli::commands::verify::run as verify_run_cmd;
use mr_cli::OnEscalate;

const SPEC_CONTENT: &str = include_str!("fixtures/specs/gap-finding-spec.md");
const PLAN_CONTENT: &str = include_str!("fixtures/plans/gap-finding-plan.md");

/// A `UserIo` that auto-approves every gate and discards printed text.
struct FakeUserIo;

#[async_trait]
impl UserIo for FakeUserIo {
    async fn print(&self, _text: &str) -> std::io::Result<()> {
        Ok(())
    }
    async fn read_line(&self) -> std::io::Result<String> {
        Ok("approve".into())
    }
}

/// Build the `(provider, model)` pair the CLI seams expect from a scripted
/// turn list.
fn fake_provider(turns: Vec<Vec<claurst_api::StreamEvent>>) -> (Arc<dyn LlmProvider>, String) {
    (
        Arc::new(FakeProvider::new("fake", turns)) as Arc<dyn LlmProvider>,
        "fake-model".to_string(),
    )
}

#[tokio::test]
async fn init_then_execute_then_verify_end_to_end() {
    let tmp = tempdir().unwrap();
    let cwd = tmp.path().to_path_buf();

    // Stage agents with valid YAML frontmatter — `dispatch::run_worker`
    // calls `prompt_asset::load`, which requires name/description/model
    // frontmatter (T7 finding). FakeProvider ignores the body.
    let agents = cwd.join("agents");
    fs::create_dir_all(&agents).unwrap();
    for role in ["literature-scout", "gap-finder", "synthesist"] {
        fs::write(
            agents.join(format!("{role}.md")),
            format!(
                "---\nname: {role}\ndescription: stub\nmodel: inherit\n---\n\n# {role}\n\nbody\n"
            ),
        )
        .unwrap();
    }

    // Compute the date once so the scripted file_path and the post-run
    // assertion agree (negligible midnight-boundary flake, same as T6).
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let slug = "alpha"; // question "alpha" -> slug "alpha"

    // --- mr init "alpha": write the gap-finding spec + plan via the guided session.
    // ScopedWrite is jailed to cwd/docs/research, so the scripted file_path
    // values are RELATIVE to docs/research (e.g. "specs/<date>-alpha-spec.md").
    let spec_rel = format!("specs/{date}-{slug}-spec.md");
    let plan_rel = format!("plans/{date}-{slug}-plan.md");
    let init_turns = vec![
        write_turn(&spec_rel, SPEC_CONTENT),
        final_turn("spec done, approve?"),
        write_turn(&plan_rel, PLAN_CONTENT),
        final_turn("plan done, approve?"),
    ];
    init_run_with(&cwd, fake_provider(init_turns), slug, &FakeUserIo)
        .await
        .unwrap();

    let spec_path = cwd
        .join("docs/research/specs")
        .join(format!("{date}-{slug}-spec.md"));
    let plan_path = cwd
        .join("docs/research/plans")
        .join(format!("{date}-{slug}-plan.md"));
    assert!(spec_path.exists(), "init should have written the spec");
    assert!(plan_path.exists(), "init should have written the plan");

    // --- mr execute --no-mcp --on-escalate=fail (plan=None -> discover the
    // plan init wrote, then sibling_spec -> the spec init wrote). The plan
    // content is the gap-finding fixture, so parse_plan yields 4 workers
    // and run_turns(4) = 16 scripted turns lines up.
    let exec_turns = run_turns(4);
    execute_run_with(
        &cwd,
        fake_provider(exec_turns),
        /*plan=*/ None,
        /*paper=*/ false,
        /*headless=*/ false,
        /*no_mcp=*/ true,
        OnEscalate::Fail,
    )
    .await
    .unwrap();

    // The run dir is docs/research/runs/<run_id>; run_id is a generated
    // timestamp, so discover it by listing (exactly one entry).
    let runs = cwd.join("docs/research/runs");
    let entries: Vec<_> = fs::read_dir(&runs)
        .unwrap()
        .flatten()
        .filter(|e| e.path().is_dir())
        .collect();
    assert_eq!(entries.len(), 1, "expected exactly one run dir");
    let run_dir = entries[0].path();
    assert!(
        run_dir.join("output.md").exists(),
        "execute should produce output.md"
    );
    assert!(
        run_dir.join("verification-report.md").exists(),
        "execute auto-verify should write the report"
    );

    // --- mr verify <run_dir>: re-run the deterministic checker. verify
    // reads spec_path back from swarm-state.yaml (stored as the absolute
    // path the orchestrator was given) and calls verify_run + write_report.
    verify_run_cmd(&cwd, fake_provider(vec![]), run_dir.clone())
        .await
        .unwrap();
    assert!(
        run_dir.join("verification-report.md").exists(),
        "verify should write the report"
    );

    // Final chain assertion: spec + plan + run output + report all present.
    assert!(
        spec_path.exists()
            && plan_path.exists()
            && run_dir.join("output.md").exists()
            && run_dir.join("verification-report.md").exists()
    );
}
