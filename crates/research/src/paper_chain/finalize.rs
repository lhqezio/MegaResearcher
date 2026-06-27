//! Phase 9 finalize: produce `paper.md` (latest draft) and `paper-history.md`.
//! 1:1 port of `lib/paper_chain/finalize.py`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use regex::Regex;

static DRAFT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^draft-v(\d+)\.md$").unwrap());
static REVIEW_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^review-v(\d+)\.md$").unwrap());

/// The highest-numbered `draft-vN.md` in `paper_dir`. `Err(NotFound)` if none.
fn latest_draft(paper_dir: &Path) -> io::Result<PathBuf> {
    let mut drafts: Vec<(u64, PathBuf)> = Vec::new();
    for entry in fs::read_dir(paper_dir)? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str() {
            if let Some(caps) = DRAFT_RE.captures(name) {
                let n: u64 = caps[1].parse().unwrap();
                drafts.push((n, entry.path()));
            }
        }
    }
    if drafts.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("No draft-vN.md in {}", paper_dir.display()),
        ));
    }
    drafts.sort_by_key(|(n, _)| *n);
    Ok(drafts.last().unwrap().1.clone())
}

/// `review-vN.md` files in `paper_dir`, ordered by N.
fn ordered_reviews(paper_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut reviews: Vec<(u64, PathBuf)> = Vec::new();
    for entry in fs::read_dir(paper_dir)? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str() {
            if let Some(caps) = REVIEW_RE.captures(name) {
                let n: u64 = caps[1].parse().unwrap();
                reviews.push((n, entry.path()));
            }
        }
    }
    reviews.sort_by_key(|(n, _)| *n);
    Ok(reviews.into_iter().map(|(_, p)| p).collect())
}

/// Produce `paper.md` (latest draft) and `paper-history.md`. Returns the
/// `paper.md` path. `Err(NotFound)` if no draft exists, `Err` on any I/O fault.
pub fn finalize_paper(paper_dir: &Path, final_verdict: &str) -> io::Result<PathBuf> {
    let latest = latest_draft(paper_dir)?;
    let paper_md = paper_dir.join("paper.md");
    let content = fs::read_to_string(&latest)?;
    fs::write(&paper_md, &content)?;

    let mut history = String::new();
    history.push_str(&format!(
        "# Paper history\n\nFinal verdict: {}\n",
        final_verdict
    ));
    for r in ordered_reviews(paper_dir)? {
        let rcontent = fs::read_to_string(&r)?;
        let name = r.file_name().unwrap().to_string_lossy().into_owned();
        history.push_str(&format!("\n---\n\n## {}\n\n{}", name, rcontent));
    }
    let log = paper_dir.join("revision-log.jsonl");
    if log.exists() && fs::metadata(&log)?.len() > 0 {
        let logcontent = fs::read_to_string(&log)?;
        history.push_str(&format!(
            "\n---\n\n## revision-log.jsonl\n\n```jsonl\n{}```\n",
            logcontent
        ));
    }
    fs::write(paper_dir.join("paper-history.md"), &history)?;
    Ok(paper_md)
}
