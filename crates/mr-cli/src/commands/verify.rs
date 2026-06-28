use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use claurst_api::LlmProvider;
use megaresearcher_research::state::swarm_state::SwarmState;
use megaresearcher_research::verify::{verify_run, write_report};

/// `mr verify <run_dir>` — re-run the deterministic post-run checker on a
/// completed run. Reads `swarm-state.yaml` for the spec path, runs
/// `verify_run`, writes `verification-report.md`, and prints the verdict.
///
/// The `provider` param is unused: verification is deterministic and does not
/// call the LLM. It is kept in the signature because `commands::dispatch`
/// passes it positionally.
pub async fn run(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    run_dir: PathBuf,
) -> anyhow::Result<()> {
    let _ = &cwd; // spec_path comes from swarm-state, not cwd
    let _ = &provider; // verify does not call the LLM

    let swarm_path = run_dir.join("swarm-state.yaml");
    let swarm =
        SwarmState::read(&swarm_path).with_context(|| format!("read {}", swarm_path.display()))?;
    let spec_path = PathBuf::from(&swarm.spec_path);

    println!("Verifying run: {}", swarm.run_id);
    println!("Spec : {}", spec_path.display());

    // 6a: pass `None` for mcp — citation spot-checks are skipped. Wiring a
    // connected McpCaller is deferred.
    let report = verify_run(&run_dir, &spec_path, None)
        .await
        .context("verify")?;
    write_report(&run_dir, &report).context("write verification report")?;

    let report_path = run_dir.join("verification-report.md");
    println!("Verdict: {:?}", report.verdict);
    println!("Report : {}", report_path.display());
    Ok(())
}
