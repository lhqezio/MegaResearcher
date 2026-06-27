//! Parse an eval-designer protocol markdown into a structured value.
//! 1:1 port of `lib/paper_chain/protocol_parser.py`. Returns an empty `Protocol`
//! when no recognizable structure (no `Substrate:` line) is found.

use std::fs;
use std::io;
use std::path::Path;

use once_cell::sync::Lazy;
use regex::Regex;

static SUBSTRATE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*[-*]\s*Substrate:\s*(.+?)\s*$").unwrap());
static SAMPLE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*[-*]\s*Sample size:\s*(\d+)").unwrap());
static SEED_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^\s*[-*]\s*Seed:\s*(\d+)").unwrap());
static BASELINES_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*[-*]\s*Baselines?:\s*(.+?)\s*$").unwrap());
static METRIC_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*[-*]\s*Metric[s]?:\s*(.+?)\s*$").unwrap());
static DECISION_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*[-*]\s*Decision rule[s]?:\s*(.+?)\s*$").unwrap());

/// A parsed decision-rule entry. Mirrors the Python `{"raw": ...}` dict shape.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRule {
    pub raw: String,
}

/// A parsed eval-designer protocol. `is_empty()` is true when no `Substrate:`
/// line was found (the Python original returns `{}` in that case).
#[derive(Debug, Clone, PartialEq)]
pub struct Protocol {
    pub substrate: Option<String>,
    pub sample_size: Option<i64>,
    pub seed: Option<i64>,
    pub baselines: Vec<String>,
    pub metrics: Vec<String>,
    pub decision_rules: Vec<DecisionRule>,
}

impl Protocol {
    /// The empty protocol, returned when no substrate is found.
    pub fn empty() -> Self {
        Protocol {
            substrate: None,
            sample_size: None,
            seed: None,
            baselines: Vec::new(),
            metrics: Vec::new(),
            decision_rules: Vec::new(),
        }
    }

    /// True when no substrate was found (mirrors `result == {}`).
    pub fn is_empty(&self) -> bool {
        self.substrate.is_none()
    }
}

/// Parse the protocol file. Returns an empty `Protocol` when no recognizable
/// structure is found. `Err` if the file cannot be read.
pub fn parse_protocol(protocol_path: &Path) -> io::Result<Protocol> {
    let text = fs::read_to_string(protocol_path)?;
    let substrate = match SUBSTRATE_RE.captures(&text) {
        Some(c) => Some(c.get(1).unwrap().as_str().trim().to_string()),
        None => return Ok(Protocol::empty()),
    };

    let sample_size = SAMPLE_RE
        .captures(&text)
        .map(|c| c.get(1).unwrap().as_str().parse::<i64>().unwrap());
    let seed = SEED_RE
        .captures(&text)
        .map(|c| c.get(1).unwrap().as_str().parse::<i64>().unwrap());

    let baselines = match BASELINES_RE.captures(&text) {
        Some(c) => c
            .get(1)
            .unwrap()
            .as_str()
            .split(',')
            .map(|b| b.trim().to_string())
            .collect(),
        None => Vec::new(),
    };

    let metrics: Vec<String> = METRIC_RE
        .captures_iter(&text)
        .map(|c| c.get(1).unwrap().as_str().trim().to_string())
        .collect();

    let decision_rules: Vec<DecisionRule> = DECISION_RE
        .captures_iter(&text)
        .map(|c| DecisionRule {
            raw: c.get(1).unwrap().as_str().trim().to_string(),
        })
        .collect();

    Ok(Protocol {
        substrate,
        sample_size,
        seed,
        baselines,
        metrics,
        decision_rules,
    })
}
