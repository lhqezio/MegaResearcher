//! Render stubs. T7 fills these with the text renderer.

pub async fn list_runs(cwd: &std::path::Path) -> anyhow::Result<()> {
    let _ = cwd;
    Ok(())
}

pub async fn watch(
    cwd: &std::path::Path,
    run_dir: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    let _ = (cwd, run_dir);
    Ok(())
}
