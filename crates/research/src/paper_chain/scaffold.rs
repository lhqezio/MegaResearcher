//! Scaffold the `paper/` subdirectory under a swarm run dir.
//! 1:1 port of `lib/paper_chain/scaffold.py`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Create `<run_dir>/paper/` with an empty `revision-log.jsonl`.
///
/// Idempotent: safe to call multiple times; preserves existing files.
/// Returns the `paper/` path. `Err` if the directory cannot be created.
pub fn scaffold_paper_dir(run_dir: &Path) -> io::Result<PathBuf> {
    let paper = run_dir.join("paper");
    fs::create_dir_all(&paper)?;
    let log = paper.join("revision-log.jsonl");
    if !log.exists() {
        fs::write(&log, "")?;
    }
    Ok(paper)
}
