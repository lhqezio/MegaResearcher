use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;

pub async fn run(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    plan: Option<PathBuf>,
    paper: bool,
    headless: bool,
    no_mcp: bool,
    on_escalate: crate::OnEscalate,
) -> anyhow::Result<()> {
    let _ = (cwd, provider, plan, paper, headless, no_mcp, on_escalate);
    Ok(())
}
