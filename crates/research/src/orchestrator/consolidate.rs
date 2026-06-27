//! Deterministic consolidations: assemble each worker's output.md into a
//! single run-root index. v0 used an LLM-synthesis step; the Rust port is a
//! plain file assembler (header + one section per worker, in dispatch order)
//! so consolidation is reproducible and testable.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn assemble(run_dir: &Path, dest: &str, header: &str, dirs: &[PathBuf]) -> io::Result<PathBuf> {
    let mut out = String::from(header);
    out.push_str("\n\n");
    for d in dirs {
        let name = d
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| d.display().to_string());
        let body = fs::read_to_string(d.join("output.md"))
            .unwrap_or_else(|_| "(no output.md)".to_string());
        out.push_str(&format!("## {name}\n\n{body}\n\n"));
    }
    let path = run_dir.join(dest);
    fs::write(&path, out)?;
    Ok(path)
}

/// Assemble `run_dir/bibliography.md` from scout output.md files.
pub fn consolidate_bibliography(run_dir: &Path, scout_dirs: &[PathBuf]) -> io::Result<PathBuf> {
    assemble(
        run_dir,
        "bibliography.md",
        "# Consolidated bibliography",
        scout_dirs,
    )
}

/// Assemble `run_dir/gaps.md` from gap-finder output.md files.
pub fn consolidate_gaps(run_dir: &Path, gap_finder_dirs: &[PathBuf]) -> io::Result<PathBuf> {
    assemble(run_dir, "gaps.md", "# Consolidated gaps", gap_finder_dirs)
}
