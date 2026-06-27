//! Deterministic gap enumeration (Phase 3 input). v0 read the consolidated
//! gaps.md and let the orchestrator LLM enumerate gaps; the Rust port reads a
//! structured `gaps:` list from each gap-finder's `manifest.yaml` so the
//! smith dispatch count and order are fixed and testable.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One gap-finder-emitted gap, as it appears in the finder's manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GapEntry {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub statement: String,
    #[serde(default, rename = "type")]
    pub gap_type: String,
}

/// The subset of a gap-finder's `manifest.yaml` this parser reads. Every field
/// defaults so partial manifests (e.g. a v0-style manifest with only
/// `gaps_count`) deserialize cleanly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GapFinderManifest {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub gaps_count: u32,
    #[serde(default)]
    pub gaps: Vec<GapEntry>,
}

/// A resolved gap: the finder-local id, the gap-finder that emitted it, the
/// finder's run dir (so smith/red-team prompts can inline its `output.md`),
/// the statement, and the gap category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    pub id: String,
    pub finder_name: String,
    pub finder_dir: PathBuf,
    pub statement: String,
    pub gap_type: String,
}

/// Parse the structured `gaps:` list out of one gap-finder's `manifest.yaml`.
/// Gaps with an empty statement are dropped (a statement is the minimum a
/// smith can act on). Unknown manifest fields are ignored.
pub fn parse_gaps(manifest_yaml: &str, finder_name: &str, finder_dir: &Path) -> Vec<Gap> {
    let manifest: GapFinderManifest = match serde_yml::from_str(manifest_yaml) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
    manifest
        .gaps
        .into_iter()
        .filter(|g| !g.statement.trim().is_empty())
        .map(|g| Gap {
            id: g.id,
            finder_name: finder_name.to_string(),
            finder_dir: finder_dir.to_path_buf(),
            statement: g.statement,
            gap_type: g.gap_type,
        })
        .collect()
}

/// Aggregate gaps across gap-finders, in `gap_dirs` order (finder-then-gap).
/// A gap-finder whose `manifest.yaml` is missing or unparseable contributes no
/// gaps (the run continues; the finder's output is still in the synthesis).
pub fn collect_gaps(run_dir: &Path, gap_dirs: &[PathBuf]) -> io::Result<Vec<Gap>> {
    let mut all = Vec::new();
    for d in gap_dirs {
        let manifest_path = d.join("manifest.yaml");
        let finder_name = d
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        if !manifest_path.exists() {
            continue;
        }
        let text = fs::read_to_string(&manifest_path)?;
        all.extend(parse_gaps(&text, &finder_name, d));
    }
    // `run_dir` is unused beyond anchoring the caller's intent; reference it so
    // the signature documents where the gaps live without forcing a redundant
    // read here. (Collect is per-finder; the run_dir is the parent of each.)
    let _ = run_dir;
    Ok(all)
}
