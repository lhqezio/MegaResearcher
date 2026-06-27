use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;

pub async fn run(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    run_dir: PathBuf,
) -> anyhow::Result<()> {
    let _ = (cwd, provider, run_dir);
    Ok(())
}
