use std::path::Path;
use std::sync::Arc;

use claurst_api::LlmProvider;

pub async fn run(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    question: &str,
) -> anyhow::Result<()> {
    let _ = (&cwd, &provider, question);
    Ok(())
}
