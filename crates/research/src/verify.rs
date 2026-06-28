//! Deterministic post-run verification (design §5: verification is a tree node,
//! red/green; §67 lists verify in the front-half but §5 + the v0 skill make it
//! a deterministic checker, not a guided session — this module implements that
//! checker). Groups A–F mirror `skills/research-verification/SKILL.md`. Group D
//! (citation spot-checks) needs the ml-intern MCP and runs only when a caller is
//! supplied; otherwise it is skipped and the verdict derives from A/B/C/E/F.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::mcp::{McpCaller, McpError};
use crate::state::swarm_state::SwarmState;
use claurst_mcp::CallToolResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    PassWithCaveats,
}

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub group: char,
    pub item: String,
    pub passed: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SpotCheck {
    pub arxiv_id: String,
    pub claim: String,
    pub resolved: bool,
}

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub run_id: String,
    pub checks: Vec<CheckResult>,
    pub spot_checks: Vec<SpotCheck>,
    pub verdict: Verdict,
}

const SYNTH_SECTIONS: &[&str] = &[
    "Executive summary",
    "Surviving hypotheses",
    "Rejected and killed hypotheses",
    "Escalations",
    "What we did NOT explore",
    "Recommended next actions",
    "Run metadata",
    "Sources",
];
const REQUIRED_ARTIFACTS: &[&str] = &["output.md", "manifest.yaml", "verification.md"];

fn worker_subdirs(run_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(run_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Run the 6 check groups (+ optional spot-checks) over `run_dir`.
///
/// Returns `anyhow::Result` so callers can `?`-propagate unexpected IO/MCP
/// errors. In practice the body is defensive: missing files degrade to failed
/// checks rather than bubbling an error, so an `Ok` report with `Verdict::Fail`
/// is the normal signal for an incomplete run.
pub async fn verify_run(
    run_dir: &Path,
    spec_path: &Path,
    mcp: Option<Arc<dyn McpCaller>>,
) -> anyhow::Result<VerificationReport> {
    let swarm = SwarmState::read(&run_dir.join("swarm-state.yaml")).ok();
    let output = std::fs::read_to_string(run_dir.join("output.md")).unwrap_or_default();
    let mut checks = Vec::new();

    // A. Run completeness.
    checks.push(CheckResult {
        group: 'A',
        item: "output.md exists".into(),
        passed: !output.is_empty(),
        detail: None,
    });
    checks.push(CheckResult {
        group: 'A',
        item: "swarm-state.yaml exists".into(),
        passed: swarm.is_some(),
        detail: None,
    });
    for d in worker_subdirs(run_dir) {
        let name = d
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        for art in REQUIRED_ARTIFACTS {
            let ok = d.join(art).exists();
            checks.push(CheckResult {
                group: 'A',
                item: format!("{name} has {art}"),
                passed: ok,
                detail: if ok { None } else { Some("missing".into()) },
            });
        }
    }

    // B. Synthesis quality.
    for sec in SYNTH_SECTIONS {
        let ok = output.contains(sec);
        checks.push(CheckResult {
            group: 'B',
            item: format!("section {sec}"),
            passed: ok,
            detail: None,
        });
    }
    if let Some(s) = &swarm {
        // Each escalation in state must be reflected in the output (the worker
        // name appears somewhere — typically in the Rejected/killed or
        // Escalations section). A present-but-empty rejected section does NOT
        // count as reflection; the worker must be named.
        for e in &s.escalations {
            let reflected = output.contains(&e.worker);
            checks.push(CheckResult {
                group: 'B',
                item: format!("escalation {} reflected", e.worker),
                passed: reflected,
                detail: if reflected {
                    None
                } else {
                    Some("hidden rejection".into())
                },
            });
        }
    }
    let yagni_ok = output.lines().any(|l| l.contains("NOT explore"));
    checks.push(CheckResult {
        group: 'B',
        item: "YAGNI section non-empty".into(),
        passed: yagni_ok,
        detail: None,
    });
    let next_ok = !output.contains("more research is needed");
    checks.push(CheckResult {
        group: 'B',
        item: "next actions specific".into(),
        passed: next_ok,
        detail: None,
    });

    // C. Hypothesis discipline (hypothesis-target only).
    if let Some(s) = &swarm {
        if s.novelty_target == "hypothesis" {
            for d in worker_subdirs(run_dir) {
                let name = d
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if name.starts_with("hypothesis-smith") {
                    let out = std::fs::read_to_string(d.join("output.md")).unwrap_or_default();
                    let falsif = out.matches("falsif").count() >= 3;
                    checks.push(CheckResult {
                        group: 'C',
                        item: format!("{name} falsification criteria"),
                        passed: falsif,
                        detail: None,
                    });
                }
                if name.starts_with("red-team") {
                    let man = std::fs::read_to_string(d.join("manifest.yaml")).unwrap_or_default();
                    let approved = man.contains("verdict: APPROVE");
                    checks.push(CheckResult {
                        group: 'C',
                        item: format!("{name} APPROVE"),
                        passed: approved,
                        detail: None,
                    });
                }
            }
        }
    }

    // D. Citation spot-checks (needs MCP).
    let mut spot_checks = Vec::new();
    match &mcp {
        None => checks.push(CheckResult {
            group: 'D',
            item: "spot-checks".into(),
            passed: true,
            detail: Some("skipped (no MCP)".into()),
        }),
        Some(caller) => {
            let ids = collect_arxiv_ids(&output);
            let picks = pick_three(&ids);
            for (id, claim) in picks {
                let args = serde_json::json!({ "operation": "paper_details", "arxiv_id": id });
                let resolved = matches!(
                    spot_check_one(caller.as_ref(), &args).await,
                    Ok(CallToolResult {
                        is_error: false,
                        ..
                    })
                );
                spot_checks.push(SpotCheck {
                    arxiv_id: id.clone(),
                    claim,
                    resolved,
                });
                checks.push(CheckResult {
                    group: 'D',
                    item: format!("spot-check {id}"),
                    passed: resolved,
                    detail: None,
                });
            }
        }
    }

    // E. Success criteria (heuristic substring match against the spec).
    let spec = std::fs::read_to_string(spec_path).unwrap_or_default();
    if let Some(block) = extract_section(&spec, "Success criteria") {
        for line in block
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        {
            let key = line.trim().trim_start_matches("- ").to_string();
            let reflected = !key.is_empty() && output.contains(&key);
            checks.push(CheckResult {
                group: 'E',
                item: format!("criterion: {}", short(&key, 40)),
                passed: reflected,
                detail: None,
            });
        }
    }

    // F. Doom-loop: any worker at retry_count >= 3 must be in escalations.
    if let Some(s) = &swarm {
        for (worker, count) in &s.retry_counts {
            if *count >= 3 {
                let recorded = s.escalations.iter().any(|e| &e.worker == worker);
                checks.push(CheckResult {
                    group: 'F',
                    item: format!("{worker} retry cap recorded"),
                    passed: recorded,
                    detail: None,
                });
            }
        }
    }

    let any_fail = checks.iter().any(|c| !c.passed && c.group != 'D');
    let d_caveat = checks
        .iter()
        .any(|c| c.group == 'D' && !c.passed && c.detail.as_deref() != Some("skipped (no MCP)"));
    let verdict = if any_fail {
        Verdict::Fail
    } else if d_caveat {
        Verdict::PassWithCaveats
    } else {
        Verdict::Pass
    };

    Ok(VerificationReport {
        run_id: swarm.as_ref().map(|s| s.run_id.clone()).unwrap_or_default(),
        checks,
        spot_checks,
        verdict,
    })
}

/// Drive a single `hf_papers.paper_details` call through the MCP seam.
///
/// Errors propagate as `anyhow::Error` (via `From<McpError>`); the caller
/// treats any `Err` result — or `is_error: true` — as an unresolved citation.
async fn spot_check_one(
    caller: &dyn McpCaller,
    args: &serde_json::Value,
) -> Result<CallToolResult, McpError> {
    caller.call_tool("hf_papers", Some(args.clone())).await
}

fn collect_arxiv_ids(text: &str) -> Vec<String> {
    let re = regex::Regex::new(r"(?:arxiv[:\s/]|arXiv[:\s/]?)(\d{4}\.\d{4,5})").unwrap();
    let mut ids = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for c in re.captures_iter(text) {
        if let Some(m) = c.get(1) {
            let id = m.as_str().to_string();
            if seen.insert(id.clone()) {
                ids.push(id);
            }
        }
    }
    ids
}

fn pick_three(ids: &[String]) -> Vec<(String, String)> {
    if ids.is_empty() {
        return vec![];
    }
    let idx = |i: usize| (ids[i % ids.len()].clone(), String::new());
    match ids.len() {
        1 => vec![idx(0)],
        2 => vec![idx(0), idx(1)],
        _ => vec![idx(0), idx(ids.len() / 2), idx(ids.len() - 1)],
    }
}

fn extract_section(text: &str, header: &str) -> Option<String> {
    let mut in_sec = false;
    let mut out = String::new();
    for line in text.lines() {
        if line.starts_with("## ") {
            if in_sec {
                break;
            }
            in_sec = line.trim_start_matches("## ").starts_with(header);
        } else if in_sec {
            out.push_str(line);
            out.push('\n');
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn short(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

/// Write `verification-report.md` at the run root.
pub fn write_report(run_dir: &Path, report: &VerificationReport) -> io::Result<()> {
    let mut s = String::new();
    s.push_str(&format!(
        "# Verification Report — {}\n\n## Checks\n",
        report.run_id
    ));
    for c in &report.checks {
        let mark = if c.passed { "[x]" } else { "[ ]" };
        let detail = c
            .detail
            .as_deref()
            .map(|d| format!(" — {d}"))
            .unwrap_or_default();
        s.push_str(&format!("- {} {} ({}){}\n", mark, c.item, c.group, detail));
    }
    s.push_str("\n## Citation spot-checks\n");
    for sc in &report.spot_checks {
        s.push_str(&format!("- {}: resolved={}\n", sc.arxiv_id, sc.resolved));
    }
    s.push_str(&format!("\n## Verdict\n{:?}\n", report.verdict));
    std::fs::write(run_dir.join("verification-report.md"), s)
}
