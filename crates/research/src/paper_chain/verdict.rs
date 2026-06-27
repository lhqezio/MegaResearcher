//! Parse the VERDICT line from a review-vN.md file.
//!
//! 1:1 port of `lib/paper_chain/verdict.py`. A valid verdict line matches exactly
//! `^VERDICT: (APPROVE|REVISE|KILL)$` (multiline). Returns the verdict word, or
//! `None` if no valid verdict line is present. A missing/unreadable file is an
//! `Err` (the Python original raises `FileNotFoundError`); the Phase 4
//! orchestrator distinguishes "file missing" from "no verdict line".

use std::fs;
use std::io;
use std::path::Path;

use once_cell::sync::Lazy;
use regex::Regex;

/// The three verdict words a review may carry.
pub static VALID_VERDICTS: &[&str] = &["APPROVE", "REVISE", "KILL"];

static VERDICT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^VERDICT: (APPROVE|REVISE|KILL)$").unwrap());

/// Scan the whole review file; return the first verdict word, or `None` if no
/// valid verdict line is found. `Err` if the file cannot be read.
pub fn parse_verdict(review_path: &Path) -> io::Result<Option<String>> {
    let text = fs::read_to_string(review_path)?;
    Ok(VERDICT_RE
        .captures(&text)
        .map(|c| c.get(1).unwrap().as_str().to_string()))
}
