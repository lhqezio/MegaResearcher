//! Runaway-revision regression detector. 1:1 port of `lib/paper_chain/regression.py`.
//!
//! Compares two consecutive review files; flags regression when the count of
//! NEW weaknesses in v2 (tags not seen in v1) is >= the count of CLOSED
//! weaknesses (v1 tags absent from v2), and at least one new weakness exists.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;

use once_cell::sync::Lazy;
use regex::Regex;

// A weakness bullet line: `- W<int>:`. No multiline flag — the input is a single
// trimmed line, so `^` anchors at its start.
static WEAKNESS_LINE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^- (W\d+):").unwrap());

/// Return the full weakness bullet lines from a review's `## Weaknesses`
/// section, with the leading `- ` dropped. `Err` if the file cannot be read.
///
/// The section is found by the first `## Weaknesses` header line and runs to the
/// next line beginning with `## ` or end of file — a manual scan standing in for
/// the Python regex `^## Weaknesses\s*$\n(.*?)(?=^## |\Z)` (Rust's `regex` crate
/// has no lookahead).
pub fn extract_weaknesses(review_path: &Path) -> io::Result<Vec<String>> {
    let text = fs::read_to_string(review_path)?;
    Ok(weakness_lines(&text))
}

fn weakness_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_section = false;
    for raw in text.lines() {
        // Header/section boundaries use trim_end (strip trailing whitespace
        // only), so leading whitespace is preserved — matching Python's `^##`
        // which requires the line to start with `##` (no indent).
        let trailing_trim = raw.trim_end();
        if !in_section {
            if trailing_trim == "## Weaknesses" {
                in_section = true;
            }
            continue;
        }
        if trailing_trim.starts_with("## ") {
            break;
        }
        // Weakness check trims both ends — matches Python's `raw.strip()`.
        let s = raw.trim();
        if WEAKNESS_LINE_RE.is_match(s) {
            // Drop leading "- " (regex matched `^- `), then strip both ends.
            out.push(s[2..].trim().to_string());
        }
    }
    out
}

fn tag(line: &str) -> String {
    // Python: `line.lstrip("- ").strip()` then `split(":", 1)[0]`.
    // lstrip("- ") strips leading characters in the set {'-', ' '}.
    let body = line
        .trim_start_matches(|c: char| c == '-' || c == ' ')
        .trim();
    body.split(':').next().unwrap_or("").to_string()
}

/// Return `(flagged, closed_count, new_count)`. `closed` = v1 tags absent from
/// v2; `new` = v2 tags absent from v1; `flagged = new >= closed && new > 0`.
/// `Err` if either file cannot be read.
pub fn detect_regression(v1_path: &Path, v2_path: &Path) -> io::Result<(bool, usize, usize)> {
    let v1 = extract_weaknesses(v1_path)?;
    let v2 = extract_weaknesses(v2_path)?;
    let tags_v1: HashSet<String> = v1.iter().map(|w| tag(w)).collect();
    let tags_v2: HashSet<String> = v2.iter().map(|w| tag(w)).collect();
    let closed = tags_v1.difference(&tags_v2).count();
    let new = tags_v2.difference(&tags_v1).count();
    let flagged = new >= closed && new > 0;
    Ok((flagged, closed, new))
}
