//! Run-tree path management: `docs/research/runs/<run-id>/`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The run directory path under `base` (typically `docs/research`):
/// `base/runs/<run_id>`. Pure; does not touch the filesystem.
pub fn run_dir(base: &Path, run_id: &str) -> PathBuf {
    base.join("runs").join(run_id)
}

/// Create the run directory (mkdir -p) and return its path.
pub fn create_run_tree(base: &Path, run_id: &str) -> io::Result<PathBuf> {
    let dir = run_dir(base, run_id);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}
