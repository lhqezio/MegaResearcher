//! Text renderer: live phase-status diffs for `mr execute`, plus `mr list` and
//! `mr watch` (the full TUI arrives in Phase 6b).

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context as _;
use megaresearcher_research::state::swarm_state::SwarmState;

/// Print one snapshot of the swarm state — the run header, per-phase status
/// with worker sub-statuses, and any escalations. Shared between the
/// `mr execute` select! loop and `watch_state`.
pub(crate) fn print_state(state: &SwarmState) {
    println!(
        "--- {} (target: {}) ---",
        state.run_id, state.novelty_target
    );
    for p in &state.phases {
        let workers: Vec<String> = p
            .workers
            .iter()
            .map(|w| format!("{}={}", w.name, w.status))
            .collect();
        println!("  {:<16} {:<10} {}", p.name, p.status, workers.join(", "));
    }
    if !state.escalations.is_empty() {
        println!(
            "  escalations: {}",
            state
                .escalations
                .iter()
                .map(|e| e.worker.clone())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

/// Poll `run_dir/swarm-state.yaml` and print phase-status changes until the
/// caller signals via `done`. Returns the last state read. Used by `mr watch`
/// in Phase 6b.
#[allow(dead_code)] // used by mr watch in Phase 6b
pub async fn watch_state(
    run_dir: &Path,
    mut done: impl FnMut() -> bool,
) -> anyhow::Result<Option<SwarmState>> {
    let path = run_dir.join("swarm-state.yaml");
    let mut last: Option<SwarmState> = None;
    loop {
        if let Ok(state) = SwarmState::read(&path) {
            let changed = last.as_ref().is_none_or(|prev| {
                prev.phases != state.phases || prev.escalations != state.escalations
            });
            if changed {
                print_state(&state);
                last = Some(state);
            }
        }
        if done() {
            return Ok(last);
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// `mr list` — print every run dir newest-first with its output.md headline.
pub async fn list_runs(cwd: &Path) -> anyhow::Result<()> {
    let runs = cwd.join("docs/research/runs");
    if !runs.is_dir() {
        println!("No runs yet. Start with `mr init \"<question>\"`.");
        return Ok(());
    }
    let mut entries: Vec<_> = std::fs::read_dir(&runs)
        .with_context(|| format!("read {}", runs.display()))?
        .flatten()
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries.iter().rev() {
        let dir = e.path();
        let headline = std::fs::read_to_string(dir.join("output.md"))
            .unwrap_or_default()
            .lines()
            .find(|l| l.starts_with('#'))
            .unwrap_or("(no output.md)")
            .to_string();
        println!("{}\t{}", e.file_name().to_string_lossy(), headline);
    }
    Ok(())
}

/// `mr watch` — the full TUI arrives in Phase 6b. For 6a this is non-hanging:
/// print the placeholder, then either print the run dir or fall back to
/// `list_runs`. It does NOT poll `swarm-state.yaml` (that would loop forever
/// with `done = || false`); the live stream is `mr execute`'s job.
pub async fn watch(cwd: &Path, run_dir: Option<PathBuf>) -> anyhow::Result<()> {
    println!("`mr watch` (TUI) arrives in Phase 6b. For now, the run streams to stdout via `mr execute`.");
    if let Some(rd) = run_dir {
        println!("Run dir: {}", rd.display());
    } else {
        list_runs(cwd).await?;
    }
    Ok(())
}
