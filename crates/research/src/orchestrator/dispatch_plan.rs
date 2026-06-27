//! The structured dispatch contract parsed from a research plan's markdown.
//!
//! v0 read dispatch counts from plan prose via mid-run LLM judgment. The
//! Rust port parses the plan's `## Phase N — <role> dispatches` sections into
//! a fixed `Vec<Assignment>` so dispatch is deterministic. (Hypothesis-target
//! phases 3/4/5, which in v0 derived counts from prior outputs, are handled
//! by Phase 4b via worker manifests — not by this parser.)

use serde::{Deserialize, Serialize};

/// Which novelty target the run pursues. Drives the phase-skip rule: a
/// `GapFinding` run skips phases 3/4/5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NoveltyTarget {
    GapFinding,
    Hypothesis,
}

impl NoveltyTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GapFinding => "gap-finding",
            Self::Hypothesis => "hypothesis",
        }
    }

    /// Phases 3/4/5 (hypothesis-smith / red-team / eval-designer) are idle for
    /// a gap-finding run.
    pub fn skips_critique_phases(&self) -> bool {
        matches!(self, Self::GapFinding)
    }
}

/// One worker assignment extracted from a plan phase section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assignment {
    /// `<role>-<1-based index>`, e.g. `literature-scout-1`. Used as the
    /// worker name and the run-tree subdir.
    pub id: String,
    /// The agent role, e.g. `literature-scout`.
    pub role: String,
    /// The `### ` heading text — the sub-topic / assignment title.
    pub title: String,
    /// The paragraph body under the heading (the assignment instruction).
    pub body: String,
}

/// The full dispatch contract parsed from the plan markdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedPlan {
    pub novelty_target: NoveltyTarget,
    pub scouts: Vec<Assignment>,
    pub gap_finders: Vec<Assignment>,
}

/// Parse `markdown` into a `ParsedPlan`.
///
/// Frontmatter (YAML between the first two `---` lines) supplies
/// `novelty_target`. `## Phase N — <role> dispatches` sections supply
/// assignments; each `### <title>` under such a section is one assignment
/// whose body is the trimmed text up to the next `###` or `##`.
pub fn parse_plan(markdown: &str) -> Result<ParsedPlan, String> {
    let novelty_target = parse_novelty_target(markdown)?;
    let scouts = parse_phase(markdown, "literature-scout")?;
    let gap_finders = parse_phase(markdown, "gap-finder")?;
    Ok(ParsedPlan {
        novelty_target,
        scouts,
        gap_finders,
    })
}

fn parse_novelty_target(markdown: &str) -> Result<NoveltyTarget, String> {
    let fm = extract_frontmatter(markdown)
        .ok_or_else(|| "plan missing YAML frontmatter (--- … ---)".to_string())?;
    // A tiny key:value scan — avoids pulling a full YAML dep for one field.
    for line in fm.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("novelty_target:") {
            match rest.trim() {
                "gap-finding" => return Ok(NoveltyTarget::GapFinding),
                "hypothesis" => return Ok(NoveltyTarget::Hypothesis),
                other => {
                    return Err(format!(
                        "unknown novelty_target value: {other:?} (want gap-finding | hypothesis)"
                    ))
                }
            }
        }
    }
    Err("frontmatter missing novelty_target".to_string())
}

/// Return the text between the first two lines that are exactly `---`.
fn extract_frontmatter(markdown: &str) -> Option<String> {
    let mut lines = markdown.lines();
    let first = lines.next()?;
    if first.trim() != "---" {
        return None;
    }
    let mut body = String::new();
    for line in lines {
        if line.trim() == "---" {
            return Some(body);
        }
        body.push_str(line);
        body.push('\n');
    }
    None
}

/// Parse the `## Phase N — <role> dispatches` section into assignments.
fn parse_phase(markdown: &str, role: &str) -> Result<Vec<Assignment>, String> {
    let section_header = format!("— {role} dispatches");
    let mut in_section = false;
    let mut assignments = Vec::new();
    // Current assignment accumulator.
    let mut title: Option<String> = None;
    let mut body_lines: Vec<String> = Vec::new();
    let mut count = 0usize;

    let flush = |title: &mut Option<String>,
                 body_lines: &mut Vec<String>,
                 count: &mut usize,
                 assignments: &mut Vec<Assignment>| {
        if let Some(t) = title.take() {
            *count += 1;
            let body = body_lines.join("\n").trim().to_string();
            assignments.push(Assignment {
                id: format!("{role}-{count}"),
                role: role.to_string(),
                title: t,
                body,
            });
            body_lines.clear();
        }
    };

    for line in markdown.lines() {
        if line.starts_with("## ") {
            if in_section {
                // Leaving the section we were in.
                flush(&mut title, &mut body_lines, &mut count, &mut assignments);
                in_section = false;
            }
            if line.contains(&section_header) {
                in_section = true;
            }
            continue;
        }
        if !in_section {
            continue;
        }
        if let Some(heading) = line.strip_prefix("### ") {
            flush(&mut title, &mut body_lines, &mut count, &mut assignments);
            title = Some(heading.trim().to_string());
        } else if title.is_some() {
            body_lines.push(line.to_string());
        }
    }
    if in_section {
        flush(&mut title, &mut body_lines, &mut count, &mut assignments);
    }
    Ok(assignments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_extracted() {
        let m = "---\n novelty_target: gap-finding\n---\nbody";
        assert_eq!(extract_frontmatter(m).unwrap().trim(), "novelty_target: gap-finding");
    }
}