//! `mr execute`: build an `OrchestratorConfig`, run `Orchestrator::execute` in
//! a spawned task while a `tokio::select!` loop prints phase-status diffs from
//! `swarm-state.yaml`, print the `RunOutcome`, then auto-verify
//! (`verify_run` + `write_report`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use claurst_api::LlmProvider;
use megaresearcher_research::mcp::ml_intern_config;
use megaresearcher_research::orchestrator::{Orchestrator, OrchestratorConfig};
use megaresearcher_research::state::run_id::generate_run_id;
use megaresearcher_research::state::swarm_state::SwarmState;
use megaresearcher_research::verify::{verify_run, write_report};

use crate::escalation::HeadlessEscalationHandler;
use crate::io::StdinStdoutIo;
use crate::render::print_state;
use crate::OnEscalate;

/// Test seam: `run` delegates here so the integration test can inject a
/// `FakeProvider`. `dispatch` (commands/mod.rs) calls `execute::run(...)`.
pub async fn run(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    plan: Option<PathBuf>,
    paper: bool,
    headless: bool,
    no_mcp: bool,
    on_escalate: OnEscalate,
) -> anyhow::Result<()> {
    run_with(cwd, provider, plan, paper, headless, no_mcp, on_escalate).await
}

pub async fn run_with(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    plan: Option<PathBuf>,
    paper: bool,
    _headless: bool,
    no_mcp: bool,
    on_escalate: OnEscalate,
) -> anyhow::Result<()> {
    // Correction E: --paper is accepted + warned (6a does NOT run the paper chain).
    if paper {
        println!("note: --paper chain (Phases 7-9) arrives in a later phase; running the core swarm now.");
    }

    let docs = cwd.join("docs/research");
    let plans_dir = docs.join("plans");
    let plan_path = match plan {
        Some(p) => p,
        None => latest_plan(&plans_dir)?,
    };
    let spec_path = sibling_spec(&plan_path)?;
    println!("Spec : {}", spec_path.display());
    println!("Plan : {}", plan_path.display());

    // Correction G: MEGARESEARCHER_MAX_PARALLEL env, default 4.
    let max_parallel = std::env::var("MEGARESEARCHER_MAX_PARALLEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4u32);
    // Correction D: do NOT import McpServerConfig — the type is inferred here.
    let mcp = if no_mcp {
        None
    } else {
        Some(ml_intern_config(cwd))
    };
    let cfg = OrchestratorConfig {
        research_base: docs.clone(),
        agents_dir: cwd.join("agents"),
        default_model: provider.1.clone(),
        max_parallel,
        mcp,
        // Correction F: headless handler wired with the on-escalate mode + stdin/stdout.
        escalation: Some(Arc::new(HeadlessEscalationHandler {
            mode: on_escalate,
            io: Arc::new(StdinStdoutIo),
        })),
    };
    let orch = Orchestrator::new(cfg, provider.0);
    let run_id = generate_run_id().context("generate run id")?;
    println!("Run  : {run_id}");

    // The run dir is docs/research/runs/<run_id> — this matches what the
    // orchestrator's create_run_tree(&config.research_base, run_id) produces
    // (research_base = docs → cwd/docs/research, then runs/<run_id>).
    let run_dir = docs.join("runs").join(&run_id);

    // Correction B: spawn the execute task, then select! between a 250ms poll
    // for swarm-state diffs and the task's completion. The task owns `orch`,
    // `spec_path`, `plan_path`, `run_id` (moved in); `run_dir` stays in the
    // select! loop's scope.
    let spec_c = spec_path.clone();
    let plan_c = plan_path.clone();
    let run_id_c = run_id.clone();
    let mut task = tokio::spawn(async move { orch.execute(&spec_c, &plan_c, &run_id_c).await });
    let mut last: Option<SwarmState> = None;
    let outcome = loop {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(250)) => {
                if let Ok(state) = SwarmState::read(&run_dir.join("swarm-state.yaml")) {
                    let changed = last.as_ref().is_none_or(
                        |prev| prev.phases != state.phases || prev.escalations != state.escalations,
                    );
                    if changed {
                        print_state(&state);
                        last = Some(state);
                    }
                }
            }
            r = &mut task => {
                // r is Result<Result<RunOutcome, OrchestratorError>, JoinError>;
                // .context()? converts the JoinError to anyhow and unwraps the
                // outer Result, leaving Result<RunOutcome, OrchestratorError>.
                break r.context("orchestrator task")?;
            }
        }
    };

    match outcome {
        Ok(o) => {
            println!("\nRun complete: {run_id}");
            println!("output: {}", run_dir.join("output.md").display());
            for (phase, status) in &o.phase_statuses {
                println!("  {phase}: {status}");
            }
            if !o.escalations.is_empty() {
                println!("escalations: {:?}", o.escalations);
            }
        }
        Err(e) => {
            eprintln!("run failed: {e}");
            return Err(anyhow::anyhow!("{e}"));
        }
    }

    // Auto-verify (6a skips MCP spot-checks — pass None). verify_run is
    // defensive: it returns Ok(report) with Verdict::Fail for incomplete runs
    // rather than Err, so this won't break a successful run even if the
    // fixture spec lacks a Success-criteria section.
    let report = verify_run(&run_dir, &spec_path, None)
        .await
        .context("verify")?;
    write_report(&run_dir, &report).context("write report")?;
    println!("verdict: {:?}", report.verdict);
    Ok(())
}

/// Find the latest `.md` plan under `plans_dir` (sorted by path).
fn latest_plan(plans_dir: &Path) -> anyhow::Result<PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(plans_dir)
        .with_context(|| format!("read {}", plans_dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    entries.sort();
    entries.into_iter().last().ok_or_else(|| {
        anyhow::anyhow!(
            "no plan found in {}; run `mr init` first",
            plans_dir.display()
        )
    })
}

/// Resolve the sibling spec for a plan: `gap-finding-plan.md` →
/// `docs/research/specs/gap-finding-spec.md` (go up two from plans/ to
/// docs/research/, then into specs/).
fn sibling_spec(plan_path: &Path) -> anyhow::Result<PathBuf> {
    let stem = plan_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let date_topic = stem.trim_end_matches("-plan");
    let spec = plan_path
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("specs").join(format!("{date_topic}-spec.md")))
        .ok_or_else(|| anyhow::anyhow!("cannot resolve spec dir from {}", plan_path.display()))?;
    if spec.exists() {
        Ok(spec)
    } else {
        Err(anyhow::anyhow!(
            "spec not found for plan {}; expected {}",
            plan_path.display(),
            spec.display()
        ))
    }
}
