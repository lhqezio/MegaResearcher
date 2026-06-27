//! Parse the red-team verdict line from a `red-team-<N>-r<round>/output.md`.
//!
//! The red-team worker (see `agents/red-team.md`) ends its output with a
//! Verdict section: `1. **Verdict** — APPROVE | REJECT (revision-N) | KILL
//! (irrecoverable)`. The orchestrator parses this deterministically to decide
//! the Phase 4 loop (survive / revise / kill). This is distinct from
//! `paper_chain::verdict`, which parses the peer-reviewer's `VERDICT:` line.

use std::fs;
use std::io;
use std::path::Path;

use once_cell::sync::Lazy;
use regex::Regex;

/// The three verdict outcomes a red-team critique can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedTeamVerdict {
    Approve,
    Reject { revision: u32 },
    Kill,
}

/// Match a Verdict line. Tolerates leading `N.` numbering, `**Verdict**` or
/// bare `Verdict`, and `—` / `-` / `:` as the separator. Captures the verdict
/// token and (for REJECT) the revision number.
///
/// The negative lookahead-free design: a line counts as a verdict line only if
/// it starts with an optional number + `**Verdict**`/`Verdict`, a separator,
/// then one of the three tokens. Lines that merely *describe* the format
/// (containing "exactly one of" or "one of:") are rejected by the explicit
/// guard in `parse_redteam_verdict`.
static VERDICT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?m)^\s*(?:\d+\.\s*)?\**\s*Verdict\s*\**\s*[—\-:]\s*(APPROVE|REJECT\s*\(revision-(\d+)\)|KILL(?:\s*\(irrecoverable\))?)\s*$",
    )
    .unwrap()
});

/// Scan `output_md` for the verdict line; return `None` if no valid verdict line
/// is found, or if the only "verdict"-bearing lines are format-description lines.
pub fn parse_redteam_verdict(output_md: &str) -> Option<RedTeamVerdict> {
    for cap in VERDICT_RE.captures_iter(output_md) {
        let line = cap.get(0).unwrap().as_str();
        // Guard: skip lines that describe the format rather than state a verdict.
        let low = line.to_lowercase();
        if low.contains("exactly one of") || low.contains("one of:") {
            continue;
        }
        let token = cap.get(1).unwrap().as_str();
        if token.starts_with("APPROVE") {
            return Some(RedTeamVerdict::Approve);
        }
        if token.starts_with("KILL") {
            return Some(RedTeamVerdict::Kill);
        }
        if let Some(rev) = cap.get(2) {
            if let Ok(n) = rev.as_str().parse::<u32>() {
                return Some(RedTeamVerdict::Reject { revision: n });
            }
        }
    }
    None
}

/// Read `output.md` from `path` then parse the verdict. `Err` if the file
/// cannot be read; `Ok(None)` if it has no parseable verdict.
pub fn parse_redteam_verdict_file(path: &Path) -> io::Result<Option<RedTeamVerdict>> {
    let text = fs::read_to_string(path)?;
    Ok(parse_redteam_verdict(&text))
}
