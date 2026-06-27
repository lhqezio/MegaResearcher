use std::path::Path;
use std::sync::Arc;

use claurst_api::LlmProvider;

pub async fn run_session(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    flow: &str,
    topic: &str,
) -> anyhow::Result<()> {
    let _ = (cwd, provider, flow, topic);
    Ok(())
}
