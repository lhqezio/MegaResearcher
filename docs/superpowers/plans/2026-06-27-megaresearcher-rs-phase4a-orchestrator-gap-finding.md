# MegaResearcher-rs Phase 4a — Orchestrator (gap-finding path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the v0 swarm-orchestrator skill into deterministic Rust control flow that runs a gap-finding swarm (Phases 1 → 2 → 6, skipping 3/4/5) end-to-end against a fake provider and produces the run tree, `swarm-state.yaml`, run-root `output.md`, and the spec-latest symlink.

**Architecture:** The orchestrator is a single async driver in the main session (mirroring v0's single-session skill). It consumes a `ParsedPlan` (scouts + gap-finders from the plan markdown), drives leaf `Worker`s through `dispatch_wave`, runs the verification gate (3 artifacts, one retry, escalate), assembles deterministic consolidations (`bibliography.md`, `gaps.md`), and finalizes via the synthesist. All LLM-judgment surfaces from v0 (dispatch counts, gap lists, consolidation-as-synthesis, verdict parsing) are replaced with deterministic Rust: counts come from the parsed plan, consolidations are file assemblers, phase handoff is file reads. The provider is a shared `Arc<dyn LlmProvider>`; tests script a flat turn sequence with `max_parallel = 1` so dispatch order is deterministic. `run_id` is injected by the caller so tests are deterministic.

**Tech Stack:** Rust, tokio (`full`), futures (`buffer_unordered`), serde/serde_yml, the Phase 2 `state` crate, the Phase 3 `worker`/`worker_tools`/`prompt_asset` modules, the Phase 2 `paper_chain::verdict` (re-used by Phase 4b, not 4a), and the existing test-only `FakeProvider`.

## Global Constraints

- Edit ONLY `crates/research/` (plus workspace `Cargo.toml`/`Cargo.lock` if a new dep is needed — none is expected; all required crates `futures`, `tokio`, `serde`, `serde_yml`, `serde_json`, `claurst-api`, `claurst-core`, `tempfile` are already workspace deps). The v0 port-reference (`lib/`, repo-root `tests/test_*.py`, `skills/`, repo-root `agents/`, `.claude-plugin/`, `commands/`, `hooks/`, `mcp/`, `tools/ml-intern`) MUST NOT be modified — repo-root `agents/*.md` are read-only source for fixture copies.
- No crate-root `pub use` re-exports (Phase 2/3 convention): `lib.rs` uses `pub mod orchestrator;` only; consumers use full paths `megaresearcher_research::orchestrator::...`.
- `swarm-state.yaml` status fields are free-form `String` (Phase 2 `SwarmState` has no enums). The orchestrator writes status strings: `"pending"`, `"running"`, `"complete"`, `"skipped"`, `"passed"`, `"escalated"`.
- `run_id` is non-deterministic (`state::run_id::generate_run_id` uses `chrono::now` + `getrandom`). The orchestrator ACCEPTS `run_id: &str` as a parameter — it never calls `generate_run_id` itself. The CLI/main calls `generate_run_id()` and passes it in; tests pass a fixed string.
- Per-task hygiene: `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` (bare `-D warnings` skips test binaries — Phase 3 process fix).
- GPL-3.0 (license already set at workspace level).
- Banned phrases: never use "load-bearing" or "this is doing a lot of work" or "real" (emphatic) or "honest/honestly" in any artifact.
- Work on branch `main` only — no git worktrees (hard rule). Confirm with `git branch --show-current` before dispatching any implementer; include the branch name in every subagent prompt.

## File Structure

```
crates/research/src/orchestrator/
  mod.rs            — module decls; Orchestrator, OrchestratorConfig, OrchestratorError, RunOutcome; execute()
  dispatch_plan.rs  — NoveltyTarget, Assignment, ParsedPlan, parse_plan
  preflight.rs      — preflight_check, build_initial_swarm_state, set_phase, write_swarm helpers
  dispatch.rs       — WorkerSpec, build_prompt, run_worker, dispatch_wave
  gate.rs           — GateStatus, GateOutcome, verify_wave
  consolidate.rs    — consolidate_bibliography, consolidate_gaps
  synthesize.rs     — run_synthesist, finalize_run (copy output.md + spec-latest symlink)
crates/research/tests/
  plan_parser.rs    — parse_plan tests (Task 1)
  orchestrator.rs   — preflight/setup + dispatch + gate + consolidate + execute + integration tests
  fixtures/agents/  — gap-finder.md, synthesist.md (copied from repo-root agents/, Task 3)
  fixtures/plans/    — gap-finding-plan.md (Task 1), fixture spec (Task 6)
crates/research/src/lib.rs — add `pub mod orchestrator;`
```

The runtime pieces (`preflight`, `dispatch`, `gate`, `consolidate`, `synthesize`) are free functions taking explicit parameters; `Orchestrator` (introduced in Task 6) holds config + provider and calls them. This keeps Tasks 1–5 unit-testable in isolation and avoids stub-and-rewrite.

## Determinism model (resolves the v0 design tensions)

| v0 surface (LLM judgment) | Phase 4a deterministic choice |
|---|---|
| Dispatch counts from plan prose | `parse_plan` extracts scout/gap-finder assignments by `##`/`###` headings → fixed `Vec<Assignment>` |
| Mid-run "one smith per gap" | Out of scope for 4a (gap-finding skips 3/4/5); deferred to 4b via gap-finder manifests |
| Consolidation as LLM synthesis | `consolidate_bibliography`/`consolidate_gaps` are file assemblers (header + each worker's `output.md`) |
| `gaps.md` absent in real runs | 4a always writes `gaps.md` (deterministic; the synthesist inlines it) |
| `run_id` non-determinism | Orchestrator takes `run_id: &str` from caller |
| Provider per-worker scripting | Shared `Arc<dyn LlmProvider>`; tests script a flat turn list with `max_parallel = 1` |
| Orchestrator pre-flight (§10.2) | Structural only: spec/plan exist, agent files present, `runs/` createable; provider-key + ml-intern deferred to Phase 5 |

---

### Task 1: DispatchPlan data contracts + plan parser

**Files:**
- Create: `crates/research/src/orchestrator/mod.rs`
- Create: `crates/research/src/orchestrator/dispatch_plan.rs`
- Modify: `crates/research/src/lib.rs` (add `pub mod orchestrator;`)
- Create test: `crates/research/tests/plan_parser.rs`
- Create fixture: `crates/research/tests/fixtures/plans/gap-finding-plan.md`

**Interfaces:**
- Consumes: serde (`Serialize`/`Deserialize`), `std::path::PathBuf`.
- Produces:
  - `pub enum NoveltyTarget { GapFinding, Hypothesis }` with `as_str(&self) -> &'static str` and `skips_critique_phases(&self) -> bool` (true for `GapFinding`).
  - `pub struct Assignment { pub id: String, pub role: String, pub title: String, pub body: String }`
  - `pub struct ParsedPlan { pub novelty_target: NoveltyTarget, pub scouts: Vec<Assignment>, pub gap_finders: Vec<Assignment> }`
  - `pub fn parse_plan(markdown: &str) -> Result<ParsedPlan, String>`

The plan markdown format the parser expects (the fixture and any real plan must follow it):

```
---
novelty_target: gap-finding
---

## Phase 1 — literature-scout dispatches

### Cross-attention fusion
Survey 2024–2026 work on cross-attention fusion of EO and SAR imagery.

### Temporal coherence
Survey work on temporal coherence in SAR.

## Phase 2 — gap-finder dispatches

### Sub-topics 1–2
Read the consolidated bibliography for sub-topics 1–2 and identify gaps.
```

Parser rules:
- Frontmatter is YAML between the first two `---` lines; read `novelty_target` (`gap-finding` → `GapFinding`, `hypothesis` → `Hypothesis`). Missing/unknown value → `Err`.
- `## Phase N — <role> dispatches` defines a phase section; the role is the word(s) between `— ` and ` dispatches` (e.g. `literature-scout`, `gap-finder`).
- Under a phase section, each `### <title>` heading starts an assignment. `id` = `<role>-<1-based index within section>`, `title` = the heading text, `body` = the trimmed paragraph lines until the next `###` or `##`.
- Only `literature-scout` and `gap-finder` sections are parsed into assignments. Other `## Phase` sections are ignored by 4a.
- Empty scout/gap-finder sections are allowed (yield empty `Vec`).

- [ ] **Step 1: Write the failing test**

`crates/research/tests/plan_parser.rs`:

```rust
//! parse_plan tests: extract scout/gap-finder assignments + novelty target.

use megaresearcher_research::orchestrator::dispatch_plan::{parse_plan, NoveltyTarget};

const PLAN: &str = "\
---
novelty_target: gap-finding
---

## Phase 1 — literature-scout dispatches

### Cross-attention fusion
Survey 2024–2026 work on cross-attention fusion of EO and SAR imagery.

### Temporal coherence
Survey work on temporal coherence in SAR.

## Phase 2 — gap-finder dispatches

### Sub-topics 1–2
Read the consolidated bibliography for sub-topics 1–2 and identify gaps.
";

#[test]
fn parses_novelty_target_and_assignments() {
    let p = parse_plan(PLAN).unwrap();
    assert_eq!(p.novelty_target, NoveltyTarget::GapFinding);

    assert_eq!(p.scouts.len(), 2);
    assert_eq!(p.scouts[0].id, "literature-scout-1");
    assert_eq!(p.scouts[0].role, "literature-scout");
    assert_eq!(p.scouts[0].title, "Cross-attention fusion");
    assert_eq!(
        p.scouts[0].body,
        "Survey 2024–2026 work on cross-attention fusion of EO and SAR imagery."
    );
    assert_eq!(p.scouts[1].id, "literature-scout-2");
    assert_eq!(p.scouts[1].title, "Temporal coherence");

    assert_eq!(p.gap_finders.len(), 1);
    assert_eq!(p.gap_finders[0].id, "gap-finder-1");
    assert_eq!(p.gap_finders[0].role, "gap-finder");
    assert_eq!(p.gap_finders[0].title, "Sub-topics 1–2");
    assert_eq!(
        p.gap_finders[0].body,
        "Read the consolidated bibliography for sub-topics 1–2 and identify gaps."
    );
}

#[test]
fn hypothesis_target_round_trips() {
    let p = parse_plan("---\nnovelty_target: hypothesis\n---\n## Phase 1 — literature-scout dispatches\n\n### T\nbody\n").unwrap();
    assert_eq!(p.novelty_target, NoveltyTarget::Hypothesis);
    assert!(!p.novelty_target.skips_critique_phases());
    assert_eq!(p.scouts.len(), 1);
}

#[test]
fn empty_sections_are_ok() {
    let p = parse_plan("---\nnovelty_target: gap-finding\n---\n## Phase 1 — literature-scout dispatches\n\n## Phase 2 — gap-finder dispatches\n").unwrap();
    assert!(p.scouts.is_empty());
    assert!(p.gap_finders.is_empty());
}

#[test]
fn unknown_novelty_target_is_error() {
    let err = parse_plan("---\nnovelty_target: bogus\n---\n").unwrap_err();
    assert!(err.contains("novelty_target"), "err was: {err}");
}

#[test]
fn missing_frontmatter_is_error() {
    assert!(parse_plan("no frontmatter here").is_err());
}
```

Create the fixture `crates/research/tests/fixtures/plans/gap-finding-plan.md` with the exact `PLAN` content above (used by later integration tests):

```markdown
---
novelty_target: gap-finding
---

## Phase 1 — literature-scout dispatches

### Cross-attention fusion
Survey 2024–2026 work on cross-attention fusion of EO and SAR imagery.

### Temporal coherence
Survey work on temporal coherence in SAR.

## Phase 2 — gap-finder dispatches

### Sub-topics 1–2
Read the consolidated bibliography for sub-topics 1–2 and identify gaps.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test plan_parser`
Expected: FAIL (compile error — `orchestrator::dispatch_plan` does not exist).

- [ ] **Step 3: Write minimal implementation**

`crates/research/src/lib.rs` — add the module declaration (keep all existing lines; add one line):

```rust
pub mod orchestrator;
```

(Place it alongside the existing `pub mod paper_chain;` etc. lines, alphabetically or after `worker_tools`.)

`crates/research/src/orchestrator/mod.rs`:

```rust
//! The deterministic swarm orchestrator (Phase 4). Drives leaf `Worker`s
//! through the six phases, runs the verification gate, assembles
//! consolidations, and finalizes the run. See the design spec §4/§10/§11.

pub mod dispatch_plan;
```

`crates/research/src/orchestrator/dispatch_plan.rs`:

```rust
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
    let mut lines = markdown.lines().peekable();
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

    while let Some(line) = lines.next() {
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p megaresearcher-research --test plan_parser`
Expected: PASS (5 tests + the inline `frontmatter_extracted`).

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/lib.rs crates/research/src/orchestrator/ crates/research/tests/plan_parser.rs crates/research/tests/fixtures/plans/
git commit -m "$(cat <<'EOF'
feat(rs): Phase 4a Task 1 — dispatch plan parser + data contracts

parse_plan extracts scout/gap-finder assignments from a research plan's
##/### heading sections and the novelty_target from YAML frontmatter,
replacing v0's mid-run LLM-judgment dispatch counts with a fixed
ParsedPlan. Introduces the orchestrator module.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Pre-flight + run setup + initial swarm-state

**Files:**
- Create: `crates/research/src/orchestrator/preflight.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` (add `pub mod preflight;`)
- Create test: `crates/research/tests/orchestrator.rs` (first test in this file)

**Interfaces:**
- Consumes: `crate::orchestrator::dispatch_plan::NoveltyTarget`, `crate::state::run_tree::{run_dir, create_run_tree}`, `crate::state::swarm_state::{SwarmState, Phase, Worker, Escalation}`, `std::fs`, `std::path::{Path, PathBuf}`.
- Produces:
  - `pub fn preflight_check(spec_path: &Path, plan_path: &Path, agents_dir: &Path, target: NoveltyTarget) -> Result<(), String>` — structural: spec/plan files exist; `agents_dir` is a dir containing the role `.md` files the target needs (`literature-scout.md`, `gap-finder.md`, `synthesist.md`, and — for `Hypothesis` — `hypothesis-smith.md`, `red-team.md`, `eval-designer.md`).
  - `pub fn required_agent_roles(target: NoveltyTarget) -> Vec<&'static str>`
  - `pub fn build_initial_swarm_state(run_id: &str, spec_path: &Path, plan_path: &Path, target: NoveltyTarget, max_parallel: u32) -> SwarmState` — 6 phases named `["literature-scout", "gap-finder", "hypothesis-smith", "red-team", "eval-designer", "synthesist"]`; `GapFinding` marks phases 3/4/5 `status: "skipped"` with empty workers, others `"pending"`; `Hypothesis` marks all `"pending"`.
  - `pub fn set_phase(swarm: &mut SwarmState, name: &str, status: &str, workers: Vec<(String, String)>)` — find phase by name (scan), set status + `workers: Vec<Worker { name, status }>`.
  - `pub fn write_swarm(swarm: &SwarmState, run_dir: &Path) -> io::Result<()>` — write to `run_dir/swarm-state.yaml`.

`required_agent_roles`: gap-finding needs `["literature-scout", "gap-finder", "synthesist"]`; hypothesis adds `["hypothesis-smith", "red-team", "eval-designer"]`.

- [ ] **Step 1: Write the failing test**

Append to `crates/research/tests/orchestrator.rs` (create the file):

```rust
//! Orchestrator tests: pre-flight, run setup, dispatch, gate, consolidate,
//! execute, and the gap-finding integration test.

mod common;

use std::fs;
use std::path::PathBuf;

use tempfile::tempdir;

use megaresearcher_research::orchestrator::dispatch_plan::NoveltyTarget;
use megaresearcher_research::orchestrator::preflight::{
    build_initial_swarm_state, preflight_check, required_agent_roles, set_phase, write_swarm,
};
use megaresearcher_research::state::run_tree::{create_run_tree, run_dir};
use megaresearcher_research::state::swarm_state::SwarmState;

fn write_agents(dir: &PathBuf, roles: &[&str]) {
    for r in roles {
        fs::write(dir.join(format!("{r}.md")), format!("# {r}\n\nbody\n")).unwrap();
    }
}

#[test]
fn preflight_passes_when_files_and_agents_exist() {
    let tmp = tempdir().unwrap();
    let spec = tmp.path().join("spec.md");
    let plan = tmp.path().join("plan.md");
    let agents = tmp.path().join("agents");
    fs::create_dir_all(&agents).unwrap();
    fs::write(&spec, "spec").unwrap();
    fs::write(&plan, "plan").unwrap();
    write_agents(&agents, &required_agent_roles(NoveltyTarget::GapFinding));
    assert!(preflight_check(&spec, &plan, &agents, NoveltyTarget::GapFinding).is_ok());
}

#[test]
fn preflight_fails_on_missing_agent_file() {
    let tmp = tempdir().unwrap();
    let spec = tmp.path().join("spec.md");
    let plan = tmp.path().join("plan.md");
    let agents = tmp.path().join("agents");
    fs::create_dir_all(&agents).unwrap();
    fs::write(&spec, "spec").unwrap();
    fs::write(&plan, "plan").unwrap();
    // gap-finder.md intentionally missing.
    write_agents(&agents, &["literature-scout", "synthesist"]);
    let err = preflight_check(&spec, &plan, &agents, NoveltyTarget::GapFinding).unwrap_err();
    assert!(err.contains("gap-finder.md"), "err was: {err}");
}

#[test]
fn preflight_fails_on_missing_spec() {
    let tmp = tempdir().unwrap();
    let agents = tmp.path().join("agents");
    fs::create_dir_all(&agents).unwrap();
    write_agents(&agents, &required_agent_roles(NoveltyTarget::GapFinding));
    let err = preflight_check(
        &tmp.path().join("missing-spec.md"),
        &tmp.path().join("plan.md"),
        &agents,
        NoveltyTarget::GapFinding,
    )
    .unwrap_err();
    assert!(err.contains("spec"));
}

#[test]
fn initial_swarm_state_marks_critique_phases_skipped_for_gap_finding() {
    let swarm = build_initial_swarm_state(
        "2026-06-27-0315-a1b2c3",
        &PathBuf::from("specs/s.md"),
        &PathBuf::from("plans/p.md"),
        NoveltyTarget::GapFinding,
        4,
    );
    assert_eq!(swarm.run_id, "2026-06-27-0315-a1b2c3");
    assert_eq!(swarm.novelty_target, "gap-finding");
    assert_eq!(swarm.max_parallel, 4);
    assert_eq!(swarm.phases.len(), 6);
    assert_eq!(swarm.phases[0].name, "literature-scout");
    assert_eq!(swarm.phases[0].status, "pending");
    assert_eq!(swarm.phases[2].name, "hypothesis-smith");
    assert_eq!(swarm.phases[2].status, "skipped");
    assert_eq!(swarm.phases[3].status, "skipped");
    assert_eq!(swarm.phases[4].status, "skipped");
    assert_eq!(swarm.phases[5].name, "synthesist");
    assert_eq!(swarm.phases[5].status, "pending");
}

#[test]
fn initial_swarm_state_all_pending_for_hypothesis() {
    let swarm = build_initial_swarm_state(
        "r",
        &PathBuf::from("s"),
        &PathBuf::from("p"),
        NoveltyTarget::Hypothesis,
        2,
    );
    assert!(swarm.phases.iter().all(|p| p.status == "pending"));
}

#[test]
fn write_swarm_round_trips() {
    let tmp = tempdir().unwrap();
    let run_dir = create_run_tree(tmp.path(), "rid").unwrap();
    let mut swarm = build_initial_swarm_state(
        "rid",
        &PathBuf::from("s"),
        &PathBuf::from("p"),
        NoveltyTarget::GapFinding,
        4,
    );
    set_phase(
        &mut swarm,
        "literature-scout",
        "complete",
        vec![("literature-scout-1".into(), "passed".into())],
    );
    write_swarm(&swarm, &run_dir).unwrap();
    let read_back = SwarmState::read(&run_dir.join("swarm-state.yaml")).unwrap();
    assert_eq!(read_back, swarm);
    assert_eq!(read_back.phases[0].status, "complete");
    assert_eq!(read_back.phases[0].workers.len(), 1);
    assert_eq!(read_back.phases[0].workers[0].name, "literature-scout-1");
}

// Suppress unused import until dispatch tests (Task 3) land.
#[allow(dead_code)]
fn _touch_run_dir(base: &std::path::Path) {
    let _ = run_dir(base, "x");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: FAIL (compile error — `orchestrator::preflight` does not exist).

- [ ] **Step 3: Write minimal implementation**

`crates/research/src/orchestrator/mod.rs` — add the module (replace the file body):

```rust
//! The deterministic swarm orchestrator (Phase 4). Drives leaf `Worker`s
//! through the six phases, runs the verification gate, assembles
//! consolidations, and finalizes the run. See the design spec §4/§10/§11.

pub mod dispatch_plan;
pub mod preflight;
```

`crates/research/src/orchestrator/preflight.rs`:

```rust
//! Orchestrator pre-flight + initial swarm-state construction.
//!
//! Spec §10.2: before a run, verify the inputs are present and writable.
//! Phase 4a does structural checks only (spec/plan exist, agent files
//! present, runs dir createable). Provider-key reachability and ml-intern
//! reachability are deferred to Phase 5 (real runs) — they have no meaning
//! against a fake provider.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::orchestrator::dispatch_plan::NoveltyTarget;
use crate::state::swarm_state::{Escalation, Phase, SwarmState, Worker};

/// The six phase names in execution order.
pub const PHASE_NAMES: [&str; 6] = [
    "literature-scout",
    "gap-finder",
    "hypothesis-smith",
    "red-team",
    "eval-designer",
    "synthesist",
];

/// Agent roles required for a run with the given target.
pub fn required_agent_roles(target: NoveltyTarget) -> Vec<&'static str> {
    let base = ["literature-scout", "gap-finder", "synthesist"];
    match target {
        NoveltyTarget::GapFinding => base.to_vec(),
        NoveltyTarget::Hypothesis => {
            let mut v = base.to_vec();
            v.extend(["hypothesis-smith", "red-team", "eval-designer"]);
            v
        }
    }
}

/// Structural pre-flight: spec/plan files exist; `agents_dir` is a directory
/// containing the `<role>.md` files the target needs.
pub fn preflight_check(
    spec_path: &Path,
    plan_path: &Path,
    agents_dir: &Path,
    target: NoveltyTarget,
) -> Result<(), String> {
    if !spec_path.exists() {
        return Err(format!("spec not found: {}", spec_path.display()));
    }
    if !plan_path.exists() {
        return Err(format!("plan not found: {}", plan_path.display()));
    }
    if !agents_dir.is_dir() {
        return Err(format!("agents dir not found: {}", agents_dir.display()));
    }
    for role in required_agent_roles(target) {
        let f = agents_dir.join(format!("{role}.md"));
        if !f.exists() {
            return Err(format!("missing agent file: {}", f.display()));
        }
    }
    Ok(())
}

/// Build the initial `SwarmState` with all six phases present. For a
/// gap-finding run phases 3/4/5 are marked `skipped` (idle, never dispatched);
/// for a hypothesis run all six are `pending`.
pub fn build_initial_swarm_state(
    run_id: &str,
    spec_path: &Path,
    plan_path: &Path,
    target: NoveltyTarget,
    max_parallel: u32,
) -> SwarmState {
    let skip = target.skips_critique_phases();
    let phases = PHASE_NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let is_critique = matches!(i, 2 | 3 | 4);
            let status = if skip && is_critique {
                "skipped".to_string()
            } else {
                "pending".to_string()
            };
            Phase {
                name: name.to_string(),
                status,
                workers: Vec::new(),
            }
        })
        .collect();
    SwarmState {
        run_id: run_id.to_string(),
        spec_path: spec_path.to_string_lossy().to_string(),
        plan_path: plan_path.to_string_lossy().to_string(),
        novelty_target: target.as_str().to_string(),
        max_parallel,
        phases,
        escalations: Vec::new(),
        retry_counts: std::collections::HashMap::new(),
    }
}

/// Find the phase named `name` and set its status + workers. No-op (returns
/// without error) if the phase is absent — callers always pass a known name.
pub fn set_phase(swarm: &mut SwarmState, name: &str, status: &str, workers: Vec<(String, String)>) {
    let phase = swarm.phases.iter_mut().find(|p| p.name == name);
    if let Some(phase) = phase {
        phase.status = status.to_string();
        phase.workers = workers
            .into_iter()
            .map(|(name, status)| Worker { name, status })
            .collect();
    }
}

/// Append an escalation record (spec §11).
pub fn add_escalation(swarm: &mut SwarmState, worker: &str, reason: &str, retry_count: u32) {
    swarm.escalations.push(Escalation {
        worker: worker.to_string(),
        reason: reason.to_string(),
        retry_count,
    });
}

/// Write `swarm` to `run_dir/swarm-state.yaml`.
pub fn write_swarm(swarm: &SwarmState, run_dir: &Path) -> io::Result<()> {
    swarm.write(&run_dir.join("swarm-state.yaml"))
}

#[allow(dead_code)]
fn _unused(_p: PathBuf, _f: fs::File) {}
```

Note: the test references `required_agent_roles`, `preflight_check`, `build_initial_swarm_state`, `set_phase`, `write_swarm`. `add_escalation` is produced here for use by the gate (Task 4); it compiles but is unused until Task 4 — that is fine (it is `pub`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: PASS (6 tests).

Then per-task hygiene:
`cargo clippy -p megaresearcher-research --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/ crates/research/tests/orchestrator.rs
git commit -m "$(cat <<'EOF'
feat(rs): Phase 4a Task 2 — pre-flight + run setup + initial swarm-state

Structural pre-flight (spec/plan/agent-files), build_initial_swarm_state
with the six phases (3/4/5 skipped for gap-finding), and swarm-state
write/set_phase/add_escalation helpers. No dispatch yet.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Wave dispatch + worker construction + prompt building

**Files:**
- Create: `crates/research/src/orchestrator/dispatch.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` (add `pub mod dispatch;`)
- Copy fixtures: `crates/research/tests/fixtures/agents/gap-finder.md`, `synthesist.md` (verbatim from repo-root `agents/`)
- Append test: `crates/research/tests/orchestrator.rs`

**Setup (copy agent fixtures):**

```bash
cp agents/gap-finder.md crates/research/tests/fixtures/agents/gap-finder.md
cp agents/synthesist.md crates/research/tests/fixtures/agents/synthesist.md
```

Verify byte-identical:
```bash
diff agents/gap-finder.md crates/research/tests/fixtures/agents/gap-finder.md && \
diff agents/synthesist.md crates/research/tests/fixtures/agents/synthesist.md && echo OK
```
Expected: `OK`.

**Interfaces:**
- Consumes: `crate::worker::{Worker, WorkerConfig, WorkerOutcome, WorkerError}`, `crate::worker_tools::{ScopedRead, ScopedWrite, Tool}`, `crate::prompt_asset::load`, `crate::orchestrator::dispatch_plan::Assignment`, `claurst_api::LlmProvider`, `futures::stream::{StreamExt, buffer_unordered}`.
- Produces:
  - `pub struct WorkerSpec { pub name: String, pub role: String, pub output_dir: PathBuf, pub shared_dir: PathBuf, pub prompt: String }`
  - `pub fn build_prompt(spec_text: &str, prior: &[(&str, &str)], assignment: Option<&Assignment>, output_dir: &Path) -> String` — `prior` = `(label, content)` sections inlined before the assignment (empty for scouts; the bibliography for gap-finders; all worker outputs for the synthesist).
  - `pub async fn run_worker(spec: &WorkerSpec, agents_dir: &Path, provider: Arc<dyn LlmProvider>, default_model: &str) -> Result<WorkerOutcome, OrchestratorError>`
  - `pub async fn dispatch_wave(specs: Vec<WorkerSpec>, agents_dir: &Path, provider: Arc<dyn LlmProvider>, default_model: &str, max_parallel: u32) -> Result<Vec<(String, WorkerOutcome)>, OrchestratorError>`
  - `OrchestratorError` is defined here (the orchestrator-wide error enum) in `mod.rs` and re-used by later tasks. Define it now:
    `pub enum OrchestratorError { Preflight(String), Parse(String), Io(io::Error), Worker(WorkerError), Escalated(Vec<String>), Finalize(String) }` with `From<io::Error>` and `From<WorkerError>`.

Worker construction in `run_worker`:
- Load the agent prompt asset at `agents_dir/<role>.md` via `prompt_asset::load`; system prompt = `asset.body`.
- Resolve model: if `asset.model == "inherit"` use `default_model`, else `asset.model`.
- Tools: `ScopedRead::with_shared(&spec.output_dir, &spec.shared_dir)` + `ScopedWrite::new(&spec.output_dir)`, both as `Arc<dyn Tool>`.
- `WorkerConfig { max_turns: 50, max_tokens: 4096, model }`.
- `Worker::new(&asset.body, vec![read, write], provider, config, &spec.output_dir)`.
- `worker.run(&spec.prompt).await`.

`dispatch_wave`: `futures::stream::iter(specs.enumerate()).map(|(i, spec)| async move { run_worker(...).await.map(|o| (i, spec.name, o)) }).buffer_unordered(max_parallel.max(1) as usize).collect::<Vec<_>>().await`; collect all `Result`s, propagate first error, sort by index, drop index.

- [ ] **Step 1: Write the failing test**

Append to `crates/research/tests/orchestrator.rs`:

```rust
use std::sync::Arc;

use claurst_api::{LlmProvider, StopReason, StreamEvent};
use claurst_core::types::{ContentBlock, UsageInfo};
use serde_json::json;

use common::fake_provider::FakeProvider;
use megaresearcher_research::orchestrator::dispatch::{
    build_prompt, dispatch_wave, run_worker, WorkerSpec,
};
use megaresearcher_research::orchestrator::OrchestratorError;
use megaresearcher_research::worker::WorkerStop;
use megaresearcher_research::worker_tools::{ScopedWrite, Tool};

fn write_turn(file: &str, content: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart { id: "m".into(), model: "fake".into(), usage: UsageInfo::default() },
        StreamEvent::ContentBlockStart { index: 0, content_block: ContentBlock::Text { text: String::new() } },
        StreamEvent::TextDelta { index: 0, text: format!("writing {file}") },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::ContentBlockStart { index: 1, content_block: ContentBlock::ToolUse {
            id: format!("tu_{file}"), name: "Write".into(),
            input: json!({ "file_path": file, "content": content }),
        } },
        StreamEvent::ContentBlockStop { index: 1 },
        StreamEvent::MessageDelta { stop_reason: Some(StopReason::ToolUse), usage: Some(UsageInfo::default()) },
        StreamEvent::MessageStop,
    ]
}

fn final_turn(text: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart { id: "m".into(), model: "fake".into(), usage: UsageInfo::default() },
        StreamEvent::ContentBlockStart { index: 0, content_block: ContentBlock::Text { text: String::new() } },
        StreamEvent::TextDelta { index: 0, text: text.into() },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::MessageDelta { stop_reason: Some(StopReason::EndTurn), usage: Some(UsageInfo::default()) },
        StreamEvent::MessageStop,
    ]
}

/// The standard 4-turn "write three artifacts + final" sequence one worker
/// consumes. Repeated per worker for a multi-worker run.
fn three_artifact_turns() -> Vec<Vec<StreamEvent>> {
    vec![
        write_turn("output.md", "# Output\n\ncontent"),
        write_turn("manifest.yaml", "role: literature-scout\n"),
        write_turn("verification.md", "# Verification\n\nok"),
        final_turn("Done."),
    ]
}

fn fixture_agents_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/agents")
}

#[tokio::test]
async fn build_prompt_inlines_spec_and_prior_and_assignment() {
    let tmp = tempdir().unwrap();
    let p = build_prompt(
        "SPEC TEXT",
        &[("Prior phase", "PRIOR BODY")],
        Some(&megaresearcher_research::orchestrator::dispatch_plan::Assignment {
            id: "x".into(),
            role: "gap-finder".into(),
            title: "T".into(),
            body: "BODY".into(),
        }),
        tmp.path(),
    );
    assert!(p.contains("SPEC TEXT"));
    assert!(p.contains("Prior phase"));
    assert!(p.contains("PRIOR BODY"));
    assert!(p.contains("T"));
    assert!(p.contains("BODY"));
    assert!(p.contains(tmp.path().to_string_lossy().as_ref()));
}

#[tokio::test]
async fn dispatch_wave_runs_two_scouts_writes_artifacts() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    fs::create_dir_all(&run_dir).unwrap();
    let dir1 = run_dir.join("literature-scout-1");
    let dir2 = run_dir.join("literature-scout-2");
    fs::create_dir_all(&dir1).unwrap();
    fs::create_dir_all(&dir2).unwrap();

    // 2 workers × 4 turns = 8 scripted turns, in dispatch order.
    let turns: Vec<Vec<StreamEvent>> =
        [three_artifact_turns(), three_artifact_turns()].into_iter().flatten().collect();
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;

    let specs = vec![
        WorkerSpec {
            name: "literature-scout-1".into(),
            role: "literature-scout".into(),
            output_dir: dir1.clone(),
            shared_dir: run_dir.clone(),
            prompt: build_prompt("SPEC", &[], None, &dir1),
        },
        WorkerSpec {
            name: "literature-scout-2".into(),
            role: "literature-scout".into(),
            output_dir: dir2.clone(),
            shared_dir: run_dir.clone(),
            prompt: build_prompt("SPEC", &[], None, &dir2),
        },
    ];
    let outcomes = dispatch_wave(specs, &fixture_agents_dir(), provider, "fake-model", 1)
        .await
        .unwrap();
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0].0, "literature-scout-1");
    assert_eq!(outcomes[0].1.stop, WorkerStop::EndTurn);
    assert_eq!(outcomes[1].0, "literature-scout-2");
    assert!(dir1.join("output.md").exists());
    assert!(dir1.join("manifest.yaml").exists());
    assert!(dir1.join("verification.md").exists());
    assert!(dir2.join("output.md").exists());
    assert_eq!(fake.call_count(), 8);
}

#[tokio::test]
async fn run_worker_resolves_inherit_model() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("w");
    fs::create_dir_all(&dir).unwrap();
    let spec = WorkerSpec {
        name: "literature-scout-1".into(),
        role: "literature-scout".into(),
        output_dir: dir.clone(),
        shared_dir: tmp.path().to_path_buf(),
        prompt: build_prompt("S", &[], None, &dir),
    };
    // literature-scout.md has model: inherit (Phase 3 fixture). default_model applies.
    let fake = Arc::new(FakeProvider::new("fake", three_artifact_turns()));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let outcome = run_worker(&spec, &fixture_agents_dir(), provider, "resolved-model")
        .await
        .unwrap();
    assert_eq!(outcome.stop, WorkerStop::EndTurn);
    assert!(dir.join("output.md").exists());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: FAIL (compile error — `orchestrator::dispatch` and `OrchestratorError` do not exist).

- [ ] **Step 3: Write minimal implementation**

`crates/research/src/orchestrator/mod.rs` (replace the body):

```rust
//! The deterministic swarm orchestrator (Phase 4). Drives leaf `Worker`s
//! through the six phases, runs the verification gate, assembles
//! consolidations, and finalizes the run. See the design spec §4/§10/§11.

pub mod dispatch;
pub mod dispatch_plan;
pub mod preflight;

use std::io;

use crate::worker::WorkerError;

/// Orchestrator-wide error.
#[derive(Debug)]
pub enum OrchestratorError {
    Preflight(String),
    Parse(String),
    Io(io::Error),
    Worker(WorkerError),
    Escalated(Vec<String>),
    Finalize(String),
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preflight(s) => write!(f, "pre-flight failed: {s}"),
            Self::Parse(s) => write!(f, "plan parse failed: {s}"),
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Worker(e) => write!(f, "worker error: {e:?}"),
            Self::Escalated(names) => write!(f, "workers escalated: {names:?}"),
            Self::Finalize(s) => write!(f, "finalize failed: {s}"),
        }
    }
}

impl std::error::Error for OrchestratorError {}

impl From<io::Error> for OrchestratorError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<WorkerError> for OrchestratorError {
    fn from(e: WorkerError) -> Self {
        Self::Worker(e)
    }
}
```

`crates/research/src/orchestrator/dispatch.rs`:

```rust
//! Wave dispatch: build a worker's user prompt, construct a jailed `Worker`,
//! and run a phase's workers bounded by `max_parallel`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;
use futures::stream::{self, StreamExt};

use crate::orchestrator::dispatch_plan::Assignment;
use crate::orchestrator::OrchestratorError;
use crate::prompt_asset::load as load_asset;
use crate::worker::{Worker, WorkerConfig, WorkerOutcome};
use crate::worker_tools::{ScopedRead, ScopedWrite, Tool};

/// A fully-specified worker dispatch: its name, role, jailed output dir,
/// shared read dir (the run dir), and the assembled user prompt.
#[derive(Debug, Clone)]
pub struct WorkerSpec {
    pub name: String,
    pub role: String,
    pub output_dir: PathBuf,
    pub shared_dir: PathBuf,
    pub prompt: String,
}

/// Assemble a worker's user prompt: the spec, any prior-phase sections, the
/// assignment (if any), and the output directory the worker must write to.
pub fn build_prompt(
    spec_text: &str,
    prior: &[(&str, &str)],
    assignment: Option<&Assignment>,
    output_dir: &Path,
) -> String {
    let mut out = String::new();
    out.push_str("# Research spec\n\n");
    out.push_str(spec_text);
    out.push_str("\n\n");
    for (label, content) in prior {
        out.push_str(&format!("# {label}\n\n{content}\n\n"));
    }
    if let Some(a) = assignment {
        out.push_str(&format!("# Your assignment ({})\n\n", a.id));
        out.push_str(&format!("## {}\n\n{}\n\n", a.title, a.body));
    }
    out.push_str("# Output directory\n\n");
    out.push_str(&format!(
        "Write your three required artifacts (output.md, manifest.yaml, \
         verification.md) to this directory:\n{}\n",
        output_dir.display()
    ));
    out
}

/// Run a single worker: load its agent prompt asset, resolve the model,
/// wire jailed Read/Write tools, and drive the worker.
pub async fn run_worker(
    spec: &WorkerSpec,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
) -> Result<WorkerOutcome, OrchestratorError> {
    let asset_path = agents_dir.join(format!("{}.md", spec.role));
    let asset = load_asset(&asset_path).map_err(|e| OrchestratorError::Io(e))?;
    let model = if asset.model == "inherit" {
        default_model.to_string()
    } else {
        asset.model.clone()
    };
    let read = Arc::new(ScopedRead::with_shared(&spec.output_dir, &spec.shared_dir)) as Arc<dyn Tool>;
    let write = Arc::new(ScopedWrite::new(&spec.output_dir)) as Arc<dyn Tool>;
    let worker = Worker::new(
        asset.body.clone(),
        vec![read, write],
        provider,
        WorkerConfig {
            max_turns: 50,
            max_tokens: 4096,
            model,
        },
        spec.output_dir.clone(),
    );
    worker.run(&spec.prompt).await.map_err(OrchestratorError::Worker)
}

/// Dispatch a wave of workers bounded by `max_parallel`. Returns outcomes
/// in spec order (sorted by submission index). With `max_parallel == 1`
/// dispatch is fully sequential, so a shared scripted provider's call order
/// is deterministic.
pub async fn dispatch_wave(
    specs: Vec<WorkerSpec>,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
) -> Result<Vec<(String, WorkerOutcome)>, OrchestratorError> {
    let n = specs.len();
    let indexed: Vec<(usize, WorkerSpec)> = specs.into_iter().enumerate().collect();
    let results: Vec<Result<(usize, String, WorkerOutcome), OrchestratorError>> = stream::iter(indexed)
        .map(|(i, spec)| {
            let provider = provider.clone();
            async move {
                run_worker(&spec, agents_dir, provider, default_model)
                    .await
                    .map(|o| (i, spec.name.clone(), o))
            }
        })
        .buffer_unordered(max_parallel.max(1) as usize)
        .collect()
        .await;

    let mut collected: Vec<(usize, String, WorkerOutcome)> = Vec::with_capacity(n);
    for r in results {
        collected.push(r?);
    }
    collected.sort_by_key(|(i, _, _)| *i);
    Ok(collected.into_iter().map(|(_, name, o)| (name, o)).collect())
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: PASS (all orchestrator tests including the 3 new dispatch tests).

`cargo clippy -p megaresearcher-research --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/ crates/research/tests/orchestrator.rs crates/research/tests/fixtures/agents/gap-finder.md crates/research/tests/fixtures/agents/synthesist.md
git commit -m "$(cat <<'EOF'
feat(rs): Phase 4a Task 3 — wave dispatch + worker construction

WorkerSpec + build_prompt (spec/prior/assignment/output-dir inline) +
run_worker (load agent asset, resolve inherit model, jail Read/Write) +
dispatch_wave (buffer_unordered bounded by max_parallel, ordered by index).
Copies gap-finder.md + synthesist.md fixtures. Introduces OrchestratorError.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Verification gate (3 artifacts, one retry, escalate)

**Files:**
- Create: `crates/research/src/orchestrator/gate.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` (add `pub mod gate;`)
- Append test: `crates/research/tests/orchestrator.rs`

**Interfaces:**
- Consumes: `crate::worker_tools::check_artifacts`, `crate::worker::{WorkerOutcome, WorkerStop}`, `crate::orchestrator::dispatch::{WorkerSpec, run_worker}`, `crate::orchestrator::preflight::{add_escalation, write_swarm}`, `crate::state::swarm_state::SwarmState`, `claurst_api::LlmProvider`.
- Produces:
  - `#[derive(Debug, Clone, PartialEq, Eq)] pub enum GateStatus { Passed, Escalated }`
  - `#[derive(Debug, Clone)] pub struct GateOutcome { pub name: String, pub status: GateStatus, pub retries: u32 }`
  - `pub const REQUIRED_ARTIFACTS: &[&str] = &["output.md", "manifest.yaml", "verification.md"];`
  - `pub async fn verify_wave(outcomes: Vec<(String, WorkerOutcome)>, specs: &[WorkerSpec], agents_dir: &Path, provider: Arc<dyn LlmProvider>, default_model: &str) -> Result<Vec<GateOutcome>, OrchestratorError>`

`verify_wave` logic per worker:
1. `check_artifacts(&spec.output_dir, REQUIRED_ARTIFACTS)` → `missing`.
2. If `missing.is_empty()` → `Passed { retries: 0 }`.
3. Else: build a retry spec = `spec.clone()` with `prompt` augmented by appending:
   ```
   \n\n# Missing artifacts\n\nYour previous run did not write: {comma-joined missing}. Write them now to {output_dir}.
   ```
   Call `run_worker(&retry_spec, ...)`. Re-check `check_artifacts`. If now empty → `Passed { retries: 1 }`; else → `Escalated { retries: 1 }`.

The gate does NOT itself mutate `SwarmState`; the caller (`execute`, Task 6) reads the `GateOutcome`s, writes `passed`/`escalated` worker statuses via `set_phase`, and appends escalations via `add_escalation` for any `Escalated`. (This keeps the gate pure and unit-testable.)

- [ ] **Step 1: Write the failing test**

Append to `crates/research/tests/orchestrator.rs`:

```rust
use megaresearcher_research::orchestrator::gate::{
    verify_wave, GateStatus, REQUIRED_ARTIFACTS,
};

fn spec_for(name: &str, dir: &PathBuf, run_dir: &Path) -> megaresearcher_research::orchestrator::dispatch::WorkerSpec {
    megaresearcher_research::orchestrator::dispatch::WorkerSpec {
        name: name.into(),
        role: "literature-scout".into(),
        output_dir: dir.clone(),
        shared_dir: run_dir.to_path_buf(),
        prompt: build_prompt("SPEC", &[], None, dir),
    }
}

#[tokio::test]
async fn gate_passes_when_all_artifacts_present_first_try() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let dir = run_dir.join("literature-scout-1");
    fs::create_dir_all(&dir).unwrap();
    let fake = Arc::new(FakeProvider::new("fake", three_artifact_turns()));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let spec = spec_for("literature-scout-1", &dir, &run_dir);
    let outcomes = vec![run_worker(&spec, &fixture_agents_dir(), provider.clone(), "fake-model").await.unwrap()];
    let outcomes = vec![("literature-scout-1".to_string(), outcomes[0].clone())];
    let gate = verify_wave(outcomes, std::slice::from_ref(&spec), &fixture_agents_dir(), provider, "fake-model").await.unwrap();
    assert_eq!(gate.len(), 1);
    assert_eq!(gate[0].status, GateStatus::Passed);
    assert_eq!(gate[0].retries, 0);
}

#[tokio::test]
async fn gate_retries_then_passes_on_missing_artifact() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let dir = run_dir.join("literature-scout-1");
    fs::create_dir_all(&dir).unwrap();

    // First run writes output.md + manifest only (3 turns), then the retry
    // writes verification.md + final (2 turns). 5 scripted turns total.
    let turns: Vec<Vec<StreamEvent>> = vec![
        write_turn("output.md", "x"),
        write_turn("manifest.yaml", "y"),
        final_turn("partial"),
        write_turn("verification.md", "z"),
        final_turn("done"),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let spec = spec_for("literature-scout-1", &dir, &run_dir);
    let first = run_worker(&spec, &fixture_agents_dir(), provider.clone(), "fake-model").await.unwrap();
    let outcomes = vec![("literature-scout-1".to_string(), first)];
    let gate = verify_wave(outcomes, std::slice::from_ref(&spec), &fixture_agents_dir(), provider, "fake-model").await.unwrap();
    assert_eq!(gate[0].status, GateStatus::Passed);
    assert_eq!(gate[0].retries, 1);
    assert!(dir.join("verification.md").exists());
    assert_eq!(fake.call_count(), 5); // 3 first run + 2 retry
}

#[tokio::test]
async fn gate_escalates_when_retry_still_missing() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let dir = run_dir.join("literature-scout-1");
    fs::create_dir_all(&dir).unwrap();

    // First run writes output + manifest only; retry writes nothing (final
    // immediately). verification.md never appears -> Escalated.
    let turns: Vec<Vec<StreamEvent>> = vec![
        write_turn("output.md", "x"),
        write_turn("manifest.yaml", "y"),
        final_turn("partial"),
        final_turn("still partial"),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let spec = spec_for("literature-scout-1", &dir, &run_dir);
    let first = run_worker(&spec, &fixture_agents_dir(), provider.clone(), "fake-model").await.unwrap();
    let outcomes = vec![("literature-scout-1".to_string(), first)];
    let gate = verify_wave(outcomes, std::slice::from_ref(&spec), &fixture_agents_dir(), provider, "fake-model").await.unwrap();
    assert_eq!(gate[0].status, GateStatus::Escalated);
    assert_eq!(gate[0].retries, 1);
    assert!(!dir.join("verification.md").exists());
}

#[test]
fn required_artifacts_constant() {
    assert_eq!(REQUIRED_ARTIFACTS, &["output.md", "manifest.yaml", "verification.md"]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: FAIL (compile error — `orchestrator::gate` does not exist).

- [ ] **Step 3: Write minimal implementation**

`crates/research/src/orchestrator/mod.rs` — add `pub mod gate;` to the module block (keep `dispatch`, `dispatch_plan`, `preflight`):

```rust
pub mod dispatch;
pub mod dispatch_plan;
pub mod gate;
pub mod preflight;
```

`crates/research/src/orchestrator/gate.rs`:

```rust
//! Verification gate (spec §11): each worker must produce output.md,
//! manifest.yaml, verification.md. One retry on a miss; a second miss
//! escalates (the caller records the escalation and halts the run).

use std::path::Path;
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::orchestrator::dispatch::{run_worker, WorkerSpec};
use crate::orchestrator::OrchestratorError;
use crate::worker::WorkerOutcome;
use crate::worker_tools::check_artifacts;

/// The three artifacts every worker must write.
pub const REQUIRED_ARTIFACTS: &[&str] = &["output.md", "manifest.yaml", "verification.md"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateStatus {
    Passed,
    Escalated,
}

#[derive(Debug, Clone)]
pub struct GateOutcome {
    pub name: String,
    pub status: GateStatus,
    pub retries: u32,
}

/// Verify a wave of workers. For each, check the three artifacts; on a miss,
/// redispatch once with the missing-artifacts note appended to the prompt;
/// escalate if the retry still misses.
pub async fn verify_wave(
    outcomes: Vec<(String, WorkerOutcome)>,
    specs: &[WorkerSpec],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
) -> Result<Vec<GateOutcome>, OrchestratorError> {
    let mut results = Vec::with_capacity(outcomes.len());
    for (name, _first) in outcomes {
        let spec = specs
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| OrchestratorError::Finalize(format!("no spec for worker {name}")))?;
        let missing = check_artifacts(&spec.output_dir, REQUIRED_ARTIFACTS);
        if missing.is_empty() {
            results.push(GateOutcome { name, status: GateStatus::Passed, retries: 0 });
            continue;
        }
        // One retry with the missing list appended.
        let retry = retry_spec(spec, &missing);
        run_worker(&retry, agents_dir, provider.clone(), default_model).await?;
        let still_missing = check_artifacts(&spec.output_dir, REQUIRED_ARTIFACTS);
        let status = if still_missing.is_empty() {
            GateStatus::Passed
        } else {
            GateStatus::Escalated
        };
        results.push(GateOutcome { name, status, retries: 1 });
    }
    Ok(results)
}

fn retry_spec(spec: &WorkerSpec, missing: &[String]) -> WorkerSpec {
    let mut prompt = spec.prompt.clone();
    prompt.push_str(&format!(
        "\n\n# Missing artifacts\n\nYour previous run did not write: {}. \
         Write them now to {}.\n",
        missing.join(", "),
        spec.output_dir.display()
    ));
    WorkerSpec {
        prompt,
        ..spec.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retry_spec_appends_missing() {
        let spec = WorkerSpec {
            name: "x".into(),
            role: "r".into(),
            output_dir: Path::new("/tmp/d").to_path_buf(),
            shared_dir: Path::new("/tmp").to_path_buf(),
            prompt: "ORIG".into(),
        };
        let r = retry_spec(&spec, &["verification.md".to_string()]);
        assert!(r.prompt.starts_with("ORIG"));
        assert!(r.prompt.contains("verification.md"));
        assert!(r.prompt.contains("/tmp/d"));
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: PASS (all orchestrator tests including the 4 new gate tests + inline test).

`cargo clippy -p megaresearcher-research --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/ crates/research/tests/orchestrator.rs
git commit -m "$(cat <<'EOF'
feat(rs): Phase 4a Task 4 — verification gate (retry once, escalate)

verify_wave checks the three required artifacts per worker; on a miss it
redispatches once with a missing-artifacts note, then escalates if still
missing. Gate outcomes carry Passed/Escalated + retry count; the caller
records escalations in swarm-state.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Deterministic consolidations (bibliography.md, gaps.md)

**Files:**
- Create: `crates/research/src/orchestrator/consolidate.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` (add `pub mod consolidate;`)
- Append test: `crates/research/tests/orchestrator.rs`

**Interfaces:**
- Consumes: `std::fs`, `std::path::{Path, PathBuf}`.
- Produces:
  - `pub fn consolidate_bibliography(run_dir: &Path, scout_dirs: &[PathBuf]) -> io::Result<PathBuf>` — writes `run_dir/bibliography.md` = `# Consolidated bibliography\n\n` + for each `scout_dir` (in order): `## <dir file_name>\n\n<output.md content or "(no output.md)">\n\n`. Returns the written path.
  - `pub fn consolidate_gaps(run_dir: &Path, gap_finder_dirs: &[PathBuf]) -> io::Result<PathBuf>` — same shape, header `# Consolidated gaps`, returns `run_dir/gaps.md`.

`dir file_name` = `scout_dir.file_name()` (e.g. `literature-scout-1`).

- [ ] **Step 1: Write the failing test**

Append to `crates/research/tests/orchestrator.rs`:

```rust
use megaresearcher_research::orchestrator::consolidate::{consolidate_bibliography, consolidate_gaps};

#[test]
fn consolidate_bibliography_assembles_scout_outputs_in_order() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let s1 = run_dir.join("literature-scout-1");
    let s2 = run_dir.join("literature-scout-2");
    fs::create_dir_all(&s1).unwrap();
    fs::create_dir_all(&s2).unwrap();
    fs::write(s1.join("output.md"), "BIB A").unwrap();
    fs::write(s2.join("output.md"), "BIB B").unwrap();

    let out = consolidate_bibliography(&run_dir, &[s1.clone(), s2.clone()]).unwrap();
    assert_eq!(out, run_dir.join("bibliography.md"));
    let text = fs::read_to_string(&out).unwrap();
    assert!(text.starts_with("# Consolidated bibliography"));
    let a = text.find("## literature-scout-1").unwrap();
    let b = text.find("## literature-scout-2").unwrap();
    assert!(a < b);
    assert!(text.contains("BIB A"));
    assert!(text.contains("BIB B"));
}

#[test]
fn consolidate_gaps_writes_gaps_md() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let g = run_dir.join("gap-finder-1");
    fs::create_dir_all(&g).unwrap();
    fs::write(g.join("output.md"), "GAP LIST").unwrap();
    let out = consolidate_gaps(&run_dir, &[g]).unwrap();
    assert_eq!(out, run_dir.join("gaps.md"));
    let text = fs::read_to_string(&out).unwrap();
    assert!(text.starts_with("# Consolidated gaps"));
    assert!(text.contains("## gap-finder-1"));
    assert!(text.contains("GAP LIST"));
}

#[test]
fn consolidate_handles_missing_output_md() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let s = run_dir.join("literature-scout-1");
    fs::create_dir_all(&s).unwrap();
    // No output.md in the scout dir.
    let out = consolidate_bibliography(&run_dir, &[s]).unwrap();
    let text = fs::read_to_string(&out).unwrap();
    assert!(text.contains("(no output.md)"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: FAIL (compile error — `orchestrator::consolidate` does not exist).

- [ ] **Step 3: Write minimal implementation**

`crates/research/src/orchestrator/mod.rs` — add `pub mod consolidate;`:

```rust
pub mod consolidate;
pub mod dispatch;
pub mod dispatch_plan;
pub mod gate;
pub mod preflight;
```

`crates/research/src/orchestrator/consolidate.rs`:

```rust
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
    assemble(run_dir, "bibliography.md", "# Consolidated bibliography", scout_dirs)
}

/// Assemble `run_dir/gaps.md` from gap-finder output.md files.
pub fn consolidate_gaps(run_dir: &Path, gap_finder_dirs: &[PathBuf]) -> io::Result<PathBuf> {
    assemble(run_dir, "gaps.md", "# Consolidated gaps", gap_finder_dirs)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: PASS (all orchestrator tests including the 3 new consolidate tests).

`cargo clippy -p megaresearcher-research --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/ crates/research/tests/orchestrator.rs
git commit -m "$(cat <<'EOF'
feat(rs): Phase 4a Task 5 — deterministic consolidations

consolidate_bibliography + consolidate_gaps assemble each worker's
output.md into a run-root index (header + one section per worker in
dispatch order), replacing v0's LLM-synthesis consolidation with a
reproducible file assembler.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: execute() — Orchestrator struct + Phases 1–2 wiring

**Files:**
- Modify: `crates/research/src/orchestrator/mod.rs` (add `Orchestrator`, `OrchestratorConfig`, `RunOutcome`, `execute()` running Phases 1–2)
- Append test: `crates/research/tests/orchestrator.rs`
- Create fixture: `crates/research/tests/fixtures/specs/gap-finding-spec.md`

**Interfaces:**
- Consumes: all of Tasks 1–5 (`parse_plan`, `preflight_check`, `build_initial_swarm_state`, `set_phase`, `add_escalation`, `write_swarm`, `dispatch_wave`, `verify_wave`, `consolidate_bibliography`, `consolidate_gaps`, `build_prompt`, `WorkerSpec`), `state::run_tree::create_run_tree`, `state::swarm_state::SwarmState`.
- Produces:
  - `pub struct OrchestratorConfig { pub research_base: PathBuf, pub agents_dir: PathBuf, pub default_model: String, pub max_parallel: u32 }`
  - `pub struct Orchestrator { pub config: OrchestratorConfig, pub provider: Arc<dyn LlmProvider> }`
  - `impl Orchestrator { pub fn new(config: OrchestratorConfig, provider: Arc<dyn LlmProvider>) -> Self; pub async fn execute(&self, spec_path: &Path, plan_path: &Path, run_id: &str) -> Result<RunOutcome, OrchestratorError> }`
  - `pub struct RunOutcome { pub run_dir: PathBuf, pub run_id: String, pub phase_statuses: Vec<(String, String)>, pub escalations: Vec<String> }`

`execute()` for Task 6 (Phases 1–2 only; Phase 6 synthesist lands in Task 7):
1. Read spec_text + plan_text; `parse_plan(plan_text)?` → `target = parsed.novelty_target`.
2. `preflight_check(spec_path, plan_path, &config.agents_dir, target)?`.
3. `create_run_tree(&config.research_base, run_id)?` → `run_dir`.
4. `swarm = build_initial_swarm_state(run_id, spec_path, plan_path, target, config.max_parallel)`; `write_swarm(&swarm, &run_dir)?`.
5. **Phase 1 (scouts):** build `scout_specs` — one `WorkerSpec` per `parsed.scouts[i]` with `name = assignment.id`, `role = "literature-scout"`, `output_dir = run_dir.join(&assignment.id)`, `shared_dir = run_dir`, `prompt = build_prompt(&spec_text, &[], Some(assignment), &output_dir)`. Create each output_dir. `set_phase(&mut swarm, "literature-scout", "running", vec![])`. `write_swarm`. `outcomes = dispatch_wave(scout_specs, agents_dir, provider, default_model, max_parallel)?`. `gates = verify_wave(outcomes, &scout_specs, agents_dir, provider, default_model)?`. For each gate: record worker `(name, "passed"/"escalated")`. `set_phase(&mut swarm, "literature-scout", "complete", workers)`. If any `Escalated`: `add_escalation` each + `write_swarm` + return `Err(Escalated(names))`. Else `consolidate_bibliography(&run_dir, &scout_dirs)?`.
6. **Phase 2 (gap-finders):** read `bibliography.md` (from run_dir) for inlining. build `gap_finder_specs` similarly with `prior = [("Consolidated bibliography", bib_text)]`. Create dirs. `set_phase("gap-finder","running",[])`. `write_swarm`. dispatch + gate. set_phase complete. escalate check. `consolidate_gaps(&run_dir, &gap_dirs)?`.
7. `write_swarm(&swarm, &run_dir)?`.
8. Return `RunOutcome { run_dir, run_id: run_id.into(), phase_statuses: swarm.phases.iter().map(|p| (p.name.clone(), p.status.clone())).collect(), escalations: vec![] }`.

(Phases 3/4/5 stay `skipped` for gap-finding — already set by `build_initial_swarm_state`. Phase 6 synthesist is added in Task 7; for Task 6 it stays `pending`.)

- [ ] **Step 1: Write the failing test**

Create `crates/research/tests/fixtures/specs/gap-finding-spec.md`:

```markdown
# Test spec

A tiny fixture spec for the Phase 4a gap-finding integration path.
```

Append to `crates/research/tests/orchestrator.rs`:

```rust
use megaresearcher_research::orchestrator::{execute, Orchestrator, OrchestratorConfig, RunOutcome};
use megaresearcher_research::state::swarm_state::SwarmState;

fn fixture_plan_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plans/gap-finding-plan.md")
}
fn fixture_spec_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/specs/gap-finding-spec.md")
}

/// 4 workers (2 scouts, 1 gap-finder, 1 synthesist) × 4 turns = 16 turns.
/// Used by Task 6 (3 workers, 12 turns — no synthesist yet) and Task 7 (16).
fn run_turns(n_workers: usize) -> Vec<Vec<StreamEvent>> {
    (0..n_workers).flat_map(|_| three_artifact_turns()).collect()
}

fn run_dir_of(out: &RunOutcome) -> &std::path::Path {
    &out.run_dir
}

#[tokio::test]
async fn execute_phases_1_and_2_for_gap_finding() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    let agents = fixture_agents_dir();
    fs::create_dir_all(&research_base).unwrap();

    // 3 workers: 2 scouts + 1 gap-finder. 3 × 4 = 12 scripted turns.
    let fake = Arc::new(FakeProvider::new("fake", run_turns(3)));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let orch = Orchestrator::new(
        OrchestratorConfig {
            research_base: research_base.clone(),
            agents_dir: agents,
            default_model: "fake-model".into(),
            max_parallel: 1,
        },
        provider,
    );

    let out = orch
        .execute(&fixture_spec_path(), &fixture_plan_path(), "2026-06-27-0315-a1b2c3")
        .await
        .unwrap();

    let run_dir = run_dir_of(&out).to_path_buf();
    assert_eq!(out.run_id, "2026-06-27-0315-a1b2c3");
    // Run tree.
    assert!(run_dir.join("swarm-state.yaml").exists());
    assert!(run_dir.join("literature-scout-1/output.md").exists());
    assert!(run_dir.join("literature-scout-2/output.md").exists());
    assert!(run_dir.join("gap-finder-1/output.md").exists());
    assert!(run_dir.join("bibliography.md").exists());
    assert!(run_dir.join("gaps.md").exists());

    // Swarm-state phase statuses.
    let swarm = SwarmState::read(&run_dir.join("swarm-state.yaml")).unwrap();
    let by_name: std::collections::HashMap<&str, &str> = swarm
        .phases
        .iter()
        .map(|p| (p.name.as_str(), p.status.as_str()))
        .collect();
    assert_eq!(by_name["literature-scout"], "complete");
    assert_eq!(by_name["gap-finder"], "complete");
    assert_eq!(by_name["hypothesis-smith"], "skipped");
    assert_eq!(by_name["red-team"], "skipped");
    assert_eq!(by_name["eval-designer"], "skipped");
    assert_eq!(by_name["synthesist"], "pending"); // Task 7 fills this in.
    // Each completed phase has its workers recorded.
    let scouts = swarm.phases.iter().find(|p| p.name == "literature-scout").unwrap();
    assert_eq!(scouts.workers.len(), 2);
    assert!(scouts.workers.iter().all(|w| w.status == "passed"));
}

#[tokio::test]
async fn execute_halts_on_worker_escalation() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();

    // A single final-only turn: every worker call returns EndTurn with no
    // tool use, so no worker ever writes any artifact. Every worker's gate
    // misses all three artifacts, retries (still nothing), and escalates.
    // execute dispatches both scouts before gating the wave, then halts with
    // Err(Escalated) listing the escalated scout(s).
    let turns: Vec<Vec<StreamEvent>> = vec![final_turn("nothing written")];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let orch = Orchestrator::new(
        OrchestratorConfig {
            research_base,
            agents_dir: fixture_agents_dir(),
            default_model: "fake-model".into(),
            max_parallel: 1,
        },
        provider,
    );
    let err = orch
        .execute(&fixture_spec_path(), &fixture_plan_path(), "rid2")
        .await
        .unwrap_err();
    match err {
        OrchestratorError::Escalated(names) => {
            assert!(names.contains(&"literature-scout-1".to_string()),
                "expected literature-scout-1 in escalated names, got {names:?}");
        }
        other => panic!("expected Escalated, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: FAIL (compile error — `Orchestrator`/`execute`/`OrchestratorConfig`/`RunOutcome` do not exist).

- [ ] **Step 3: Write minimal implementation**

`crates/research/src/orchestrator/mod.rs` — add the struct + execute below the existing `OrchestratorError` block:

```rust
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::orchestrator::consolidate::{consolidate_bibliography, consolidate_gaps};
use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};
use crate::orchestrator::dispatch_plan::parse_plan;
use crate::orchestrator::gate::{verify_wave, GateStatus};
use crate::orchestrator::preflight::{
    add_escalation, build_initial_swarm_state, preflight_check, set_phase, write_swarm,
};
use crate::state::run_tree::create_run_tree;

/// Orchestrator configuration: where runs live, where agent prompt assets
/// are, the model to resolve "inherit" to, and the wave concurrency bound.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub research_base: PathBuf,
    pub agents_dir: PathBuf,
    pub default_model: String,
    pub max_parallel: u32,
}

/// The orchestrator: config + a shared provider.
pub struct Orchestrator {
    pub config: OrchestratorConfig,
    pub provider: Arc<dyn LlmProvider>,
}

impl Orchestrator {
    pub fn new(config: OrchestratorConfig, provider: Arc<dyn LlmProvider>) -> Self {
        Self { config, provider }
    }

    /// Run the swarm. `run_id` is provided by the caller (the CLI calls
    /// `generate_run_id()`; tests pass a fixed string) so the run is
    /// deterministic. Task 6 implements Phases 1–2; Task 7 adds Phase 6.
    pub async fn execute(
        &self,
        spec_path: &Path,
        plan_path: &Path,
        run_id: &str,
    ) -> Result<RunOutcome, OrchestratorError> {
        let spec_text = std::fs::read_to_string(spec_path)?;
        let plan_text = std::fs::read_to_string(plan_path)?;
        let parsed = parse_plan(&plan_text).map_err(OrchestratorError::Parse)?;
        let target = parsed.novelty_target;

        preflight_check(spec_path, plan_path, &self.config.agents_dir, target)
            .map_err(OrchestratorError::Preflight)?;

        let run_dir = create_run_tree(&self.config.research_base, run_id)?;
        let mut swarm =
            build_initial_swarm_state(run_id, spec_path, plan_path, target, self.config.max_parallel);
        write_swarm(&swarm, &run_dir)?;

        // Phase 1 — literature-scout.
        let scout_specs: Vec<WorkerSpec> = parsed
            .scouts
            .iter()
            .map(|a| {
                let output_dir = run_dir.join(&a.id);
                std::fs::create_dir_all(&output_dir).ok();
                WorkerSpec {
                    name: a.id.clone(),
                    role: "literature-scout".into(),
                    output_dir,
                    shared_dir: run_dir.clone(),
                    prompt: build_prompt(&spec_text, &[], Some(a), &run_dir.join(&a.id)),
                }
            })
            .collect();
        let scout_dirs: Vec<PathBuf> = scout_specs.iter().map(|s| s.output_dir.clone()).collect();
        set_phase(&mut swarm, "literature-scout", "running", vec![]);
        write_swarm(&swarm, &run_dir)?;
        let scout_outcomes = dispatch_wave(
            scout_specs.clone(),
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
            self.config.max_parallel,
        )
        .await?;
        let scout_gates = verify_wave(
            scout_outcomes,
            &scout_specs,
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
        )
        .await?;
        let scout_workers: Vec<(String, String)> = scout_gates
            .iter()
            .map(|g| (g.name.clone(), gate_status_str(g.status)))
            .collect();
        set_phase(&mut swarm, "literature-scout", "complete", scout_workers.clone());
        let escalated: Vec<String> = scout_gates
            .iter()
            .filter(|g| g.status == GateStatus::Escalated)
            .map(|g| g.name.clone())
            .collect();
        if !escalated.is_empty() {
            for name in &escalated {
                add_escalation(&mut swarm, name, "missing artifacts after retry", 1);
            }
            write_swarm(&swarm, &run_dir)?;
            return Err(OrchestratorError::Escalated(escalated));
        }
        consolidate_bibliography(&run_dir, &scout_dirs)?;
        write_swarm(&swarm, &run_dir)?;

        // Phase 2 — gap-finder.
        let bib_text =
            std::fs::read_to_string(run_dir.join("bibliography.md")).unwrap_or_default();
        let gap_specs: Vec<WorkerSpec> = parsed
            .gap_finders
            .iter()
            .map(|a| {
                let output_dir = run_dir.join(&a.id);
                std::fs::create_dir_all(&output_dir).ok();
                WorkerSpec {
                    name: a.id.clone(),
                    role: "gap-finder".into(),
                    output_dir,
                    shared_dir: run_dir.clone(),
                    prompt: build_prompt(
                        &spec_text,
                        &[("Consolidated bibliography", &bib_text)],
                        Some(a),
                        &run_dir.join(&a.id),
                    ),
                }
            })
            .collect();
        let gap_dirs: Vec<PathBuf> = gap_specs.iter().map(|s| s.output_dir.clone()).collect();
        set_phase(&mut swarm, "gap-finder", "running", vec![]);
        write_swarm(&swarm, &run_dir)?;
        let gap_outcomes = dispatch_wave(
            gap_specs.clone(),
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
            self.config.max_parallel,
        )
        .await?;
        let gap_gates = verify_wave(
            gap_outcomes,
            &gap_specs,
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
        )
        .await?;
        let gap_workers: Vec<(String, String)> = gap_gates
            .iter()
            .map(|g| (g.name.clone(), gate_status_str(g.status)))
            .collect();
        set_phase(&mut swarm, "gap-finder", "complete", gap_workers);
        let gap_escalated: Vec<String> = gap_gates
            .iter()
            .filter(|g| g.status == GateStatus::Escalated)
            .map(|g| g.name.clone())
            .collect();
        if !gap_escalated.is_empty() {
            for name in &gap_escalated {
                add_escalation(&mut swarm, name, "missing artifacts after retry", 1);
            }
            write_swarm(&swarm, &run_dir)?;
            return Err(OrchestratorError::Escalated(gap_escalated));
        }
        consolidate_gaps(&run_dir, &gap_dirs)?;
        write_swarm(&swarm, &run_dir)?;

        Ok(RunOutcome {
            run_dir,
            run_id: run_id.to_string(),
            phase_statuses: swarm
                .phases
                .iter()
                .map(|p| (p.name.clone(), p.status.clone()))
                .collect(),
            escalations: Vec::new(),
        })
    }
}

/// The outcome of a run: run dir/id, per-phase final statuses, escalated workers.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub run_dir: PathBuf,
    pub run_id: String,
    pub phase_statuses: Vec<(String, String)>,
    pub escalations: Vec<String>,
}

fn gate_status_str(status: GateStatus) -> String {
    match status {
        GateStatus::Passed => "passed".to_string(),
        GateStatus::Escalated => "escalated".to_string(),
    }
}
```

Note: `verify_wave` is re-exported from `gate`; `dispatch_wave` from `dispatch`. Ensure the `use` lines above import correctly:
- `use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};`
- `use crate::orchestrator::gate::{verify_wave, GateStatus};`

(Adjust the two `use` lines in the final file so `verify_wave` comes from `gate`, not `dispatch`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: PASS (all orchestrator tests including the 2 new execute tests).

`cargo clippy -p megaresearcher-research --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/mod.rs crates/research/tests/orchestrator.rs crates/research/tests/fixtures/specs/
git commit -m "$(cat <<'EOF'
feat(rs): Phase 4a Task 6 — execute() Phases 1–2 (scouts + gap-finders)

Orchestrator + OrchestratorConfig + RunOutcome. execute() runs pre-flight,
sets up the run tree + initial swarm-state, dispatches scouts (Phase 1) and
gap-finders (Phase 2, with bibliography.md inlined), gates both waves,
consolidates bibliography.md + gaps.md, and halts with Escalated on any
worker that misses artifacts after retry. Phase 6 lands in Task 7.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Synthesist + skip rule + finalize (output.md + spec-latest symlink) + gap-finding integration test

**Files:**
- Create: `crates/research/src/orchestrator/synthesize.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` (add `pub mod synthesize;`, wire Phase 6 into `execute`)
- Append test: `crates/research/tests/orchestrator.rs`

**Interfaces:**
- Consumes: `crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec}`, `crate::orchestrator::gate::verify_wave`, `crate::orchestrator::preflight::{set_phase, write_swarm, add_escalation}`, `state::swarm_state::SwarmState`, `std::os::unix::fs::symlink`.
- Produces:
  - `pub async fn run_synthesist(run_dir: &Path, spec_text: &str, plan_text: &str, scout_dirs: &[PathBuf], gap_dirs: &[PathBuf], agents_dir: &Path, provider: Arc<dyn LlmProvider>, default_model: &str, max_parallel: u32) -> Result<WorkerOutcome, OrchestratorError>` — builds a single `WorkerSpec` with `name="synthesist"`, `role="synthesist"`, `output_dir=run_dir/synthesist`, `shared_dir=run_dir`, and a prompt that inlines the spec, the plan, every scout `output.md` (`("Scout <name>", content)`), every gap-finder `output.md` (`("Gap-finder <name>", content)`). Creates the dir, dispatches (single worker), returns the outcome. (No gate here — the caller gates.)
  - `pub fn finalize_run(run_dir: &Path, spec_path: &Path, research_base: &Path) -> io::Result<PathBuf>` — copy `synthesist/output.md` → `run_dir/output.md`; create `research_base/specs/` if missing; create a relative symlink `research_base/specs/<spec-stem>-latest.md` → `../runs/<run_id>/output.md` (derive `<spec-stem>` from `spec_path.file_stem`, `<run_id>` from `run_dir.file_name()`). If the symlink already exists, remove it first. Returns the symlink path.

`execute()` Phase 6 wiring (after Phase 2, for `GapFinding` — phases 3/4/5 already `skipped`):
1. `set_phase(&mut swarm, "synthesist", "running", vec![]); write_swarm`.
2. `synth_outcome = run_synthesist(run_dir, &spec_text, &plan_text, &scout_dirs, &gap_dirs, agents_dir, provider, default_model, max_parallel)?`.
3. Gate the synthesist: `check_artifacts(synth_dir, REQUIRED_ARTIFACTS)`; if missing, retry once (reuse `verify_wave` with a 1-element outcomes/spec pair, or a small inline retry). Simplest: call `verify_wave(vec![("synthesist".into(), synth_outcome)], &[synth_spec], ...)` — but `run_synthesist` owns the spec. Refactor `run_synthesist` to ALSO return the `WorkerSpec` it built, so the caller can gate it. Change signature to `Result<(WorkerSpec, WorkerOutcome), OrchestratorError>`.
4. If gate Escalated → `add_escalation`, `set_phase("synthesist","complete",[(escalated)])`, `write_swarm`, return `Err(Escalated(["synthesist"]))`.
5. Else: `set_phase("synthesist","complete",[("synthesist","passed")])`; `finalize_run(run_dir, spec_path, research_base)?`; `write_swarm`.
6. Return `RunOutcome`.

- [ ] **Step 1: Write the failing test**

Append to `crates/research/tests/orchestrator.rs`:

```rust
use megaresearcher_research::orchestrator::synthesize::{finalize_run, run_synthesist};

#[tokio::test]
async fn run_synthesist_inlines_all_outputs_and_writes_artifacts() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("runs/rid");
    let synth_dir = run_dir.join("synthesist");
    let s1 = run_dir.join("literature-scout-1");
    let g1 = run_dir.join("gap-finder-1");
    fs::create_dir_all(&s1).unwrap();
    fs::create_dir_all(&g1).unwrap();
    fs::write(s1.join("output.md"), "SCOUT1").unwrap();
    fs::write(g1.join("output.md"), "GAP1").unwrap();

    let fake = Arc::new(FakeProvider::new("fake", three_artifact_turns()));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let (spec, outcome) = run_synthesist(
        &run_dir, "SPEC", "PLAN", &[s1], &[g1], &fixture_agents_dir(), provider, "fake-model", 1,
    )
    .await
    .unwrap();
    assert_eq!(spec.name, "synthesist");
    assert!(synth_dir.join("output.md").exists());
    assert!(synth_dir.join("manifest.yaml").exists());
    assert!(synth_dir.join("verification.md").exists());
    assert_eq!(outcome.stop, WorkerStop::EndTurn);
    // The prompt inlined the scout + gap-finder outputs.
    assert!(spec.prompt.contains("SCOUT1"));
    assert!(spec.prompt.contains("GAP1"));
    assert!(spec.prompt.contains("SPEC"));
    assert!(spec.prompt.contains("PLAN"));
}

#[test]
fn finalize_run_copies_output_and_creates_symlink() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    let run_dir = research_base.join("runs/2026-06-27-0315-a1b2c3");
    let synth = run_dir.join("synthesist");
    fs::create_dir_all(&synth).unwrap();
    fs::write(synth.join("output.md"), "FINAL OUTPUT").unwrap();
    let spec_path = research_base.join("specs/my-spec.md");
    fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
    fs::write(&spec_path, "spec body").unwrap();

    let link = finalize_run(&run_dir, &spec_path, &research_base).unwrap();
    assert_eq!(link, research_base.join("specs/my-spec-latest.md"));
    // Run-root output.md is the copy of synthesist/output.md.
    assert_eq!(fs::read_to_string(run_dir.join("output.md")).unwrap(), "FINAL OUTPUT");
    // Symlink resolves to the run-root output.md.
    assert_eq!(fs::read_to_string(&link).unwrap(), "FINAL OUTPUT");
    // Re-running replaces the existing symlink (no error).
    finalize_run(&run_dir, &spec_path, &research_base).unwrap();
    assert_eq!(fs::read_to_string(&link).unwrap(), "FINAL OUTPUT");
}

#[tokio::test]
async fn full_gap_finding_integration_test() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    // 4 workers (2 scouts, 1 gap-finder, 1 synthesist) × 4 turns = 16 turns.
    let fake = Arc::new(FakeProvider::new("fake", run_turns(4)));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let orch = Orchestrator::new(
        OrchestratorConfig {
            research_base: research_base.clone(),
            agents_dir: fixture_agents_dir(),
            default_model: "fake-model".into(),
            max_parallel: 1,
        },
        provider,
    );
    // Put the spec inside research_base/specs/ so finalize_run's symlink lands
    // next to it, mirroring the real layout.
    let spec_path = research_base.join("specs/gap-finding-spec.md");
    fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
    fs::copy(fixture_spec_path(), &spec_path).unwrap();

    let out = orch.execute(&spec_path, &fixture_plan_path(), "2026-06-27-0315-a1b2c3").await.unwrap();
    let run_dir = out.run_dir.clone();

    // Full run tree.
    assert!(run_dir.join("swarm-state.yaml").exists());
    assert!(run_dir.join("literature-scout-1/output.md").exists());
    assert!(run_dir.join("literature-scout-2/output.md").exists());
    assert!(run_dir.join("gap-finder-1/output.md").exists());
    assert!(run_dir.join("synthesist/output.md").exists());
    assert!(run_dir.join("bibliography.md").exists());
    assert!(run_dir.join("gaps.md").exists());
    // Run-root output.md + spec-latest symlink.
    assert!(run_dir.join("output.md").exists());
    let link = research_base.join("specs/gap-finding-spec-latest.md");
    assert!(link.exists());
    assert_eq!(
        fs::read_to_string(&link).unwrap(),
        fs::read_to_string(run_dir.join("output.md")).unwrap()
    );

    // Swarm-state: all six phases accounted for.
    let swarm = SwarmState::read(&run_dir.join("swarm-state.yaml")).unwrap();
    let by_name: std::collections::HashMap<&str, &str> = swarm
        .phases
        .iter()
        .map(|p| (p.name.as_str(), p.status.as_str()))
        .collect();
    assert_eq!(by_name["literature-scout"], "complete");
    assert_eq!(by_name["gap-finder"], "complete");
    assert_eq!(by_name["hypothesis-smith"], "skipped");
    assert_eq!(by_name["red-team"], "skipped");
    assert_eq!(by_name["eval-designer"], "skipped");
    assert_eq!(by_name["synthesist"], "complete");
    assert!(out.escalations.is_empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: FAIL (compile error — `orchestrator::synthesize` does not exist; `execute` does not yet run Phase 6).

- [ ] **Step 3: Write minimal implementation**

`crates/research/src/orchestrator/mod.rs` — add `pub mod synthesize;` and wire Phase 6 into `execute()`. Replace the tail of `execute()` (the final `Ok(RunOutcome { ... })` block) with Phase 6 + finalize, and add the import:

Add to the top `use` block in `mod.rs`:
```rust
use crate::orchestrator::synthesize::{finalize_run, run_synthesist};
```
(`verify_wave` and `GateStatus` are already imported from `gate` in Task 6's `use` block; `run_synthesist` + `finalize_run` are the only new imports Task 7 adds.)

Replace the final `Ok(RunOutcome { ... })` at the end of `execute()` with:

```rust
        // Phase 6 — synthesist (gap-finding skips 3/4/5, already marked skipped).
        set_phase(&mut swarm, "synthesist", "running", vec![]);
        write_swarm(&swarm, &run_dir)?;
        let (synth_spec, synth_outcome) = run_synthesist(
            &run_dir,
            &spec_text,
            &plan_text,
            &scout_dirs,
            &gap_dirs,
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
            self.config.max_parallel,
        )
        .await?;
        let synth_gates = verify_wave(
            vec![("synthesist".to_string(), synth_outcome)],
            std::slice::from_ref(&synth_spec),
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
        )
        .await?;
        let synth_status = gate_status_str(synth_gates[0].status);
        set_phase(
            &mut swarm,
            "synthesist",
            "complete",
            vec![("synthesist".to_string(), synth_status.clone())],
        );
        if synth_gates[0].status == GateStatus::Escalated {
            add_escalation(&mut swarm, "synthesist", "missing artifacts after retry", 1);
            write_swarm(&swarm, &run_dir)?;
            return Err(OrchestratorError::Escalated(vec!["synthesist".to_string()]));
        }
        finalize_run(&run_dir, spec_path, &self.config.research_base)?;
        write_swarm(&swarm, &run_dir)?;

        Ok(RunOutcome {
            run_dir,
            run_id: run_id.to_string(),
            phase_statuses: swarm
                .phases
                .iter()
                .map(|p| (p.name.clone(), p.status.clone()))
                .collect(),
            escalations: swarm
                .escalations
                .iter()
                .map(|e| e.worker.clone())
                .collect(),
        })
```

`crates/research/src/orchestrator/synthesize.rs`:

```rust
//! Phase 6 synthesist dispatch + run finalization (run-root output.md and
//! the spec-latest symlink).

use std::fs;
use std::io;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};
use crate::orchestrator::OrchestratorError;
use crate::worker::WorkerOutcome;

/// Build and run the single synthesist worker, inlining the spec, the plan,
/// every scout output, and every gap-finder output. Returns the spec (so the
/// caller can gate it) and the worker outcome.
pub async fn run_synthesist(
    run_dir: &Path,
    spec_text: &str,
    plan_text: &str,
    scout_dirs: &[PathBuf],
    gap_dirs: &[PathBuf],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
) -> Result<(WorkerSpec, WorkerOutcome), OrchestratorError> {
    let mut prior: Vec<(String, String)> = Vec::new();
    prior.push(("Plan".to_string(), plan_text.to_string()));
    for d in scout_dirs {
        let name = d.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let body = fs::read_to_string(d.join("output.md")).unwrap_or_else(|_| "(no output.md)".to_string());
        prior.push((format!("Scout {name}"), body));
    }
    for d in gap_dirs {
        let name = d.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let body = fs::read_to_string(d.join("output.md")).unwrap_or_else(|_| "(no output.md)".to_string());
        prior.push((format!("Gap-finder {name}"), body));
    }
    // `prior` owns the (label, content) strings; `prior_refs` borrows them for
    // the single `build_prompt` call. No leak — `prior` lives for this scope.
    let prior_refs: Vec<(&str, &str)> = prior.iter().map(|(l, c)| (l.as_str(), c.as_str())).collect();

    let output_dir = run_dir.join("synthesist");
    fs::create_dir_all(&output_dir)?;
    let spec = WorkerSpec {
        name: "synthesist".to_string(),
        role: "synthesist".to_string(),
        output_dir: output_dir.clone(),
        shared_dir: run_dir.to_path_buf(),
        prompt: build_prompt(spec_text, &prior_refs, None, &output_dir),
    };
    let outcomes = dispatch_wave(
        vec![spec.clone()],
        agents_dir,
        provider,
        default_model,
        max_parallel,
    )
    .await?;
    let (_, outcome) = outcomes
        .into_iter()
        .next()
        .ok_or_else(|| OrchestratorError::Finalize("synthesist produced no outcome".into()))?;
    Ok((spec, outcome))
}

/// Copy `synthesist/output.md` to `run_dir/output.md` and create the
/// spec-latest symlink `research_base/specs/<spec-stem>-latest.md` →
/// `../runs/<run_id>/output.md` (relative, so the tree is relocatable).
pub fn finalize_run(run_dir: &Path, spec_path: &Path, research_base: &Path) -> io::Result<PathBuf> {
    let synth_output = run_dir.join("synthesist").join("output.md");
    fs::copy(&synth_output, run_dir.join("output.md"))?;

    let specs_dir = research_base.join("specs");
    fs::create_dir_all(&specs_dir)?;
    let stem = spec_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "spec".to_string());
    let run_id = run_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let link = specs_dir.join(format!("{stem}-latest.md"));
    let target = format!("../runs/{run_id}/output.md");
    if link.exists() || symlink_exists(&link) {
        fs::remove_file(&link)?;
    }
    symlink(&target, &link)?;
    Ok(link)
}

fn symlink_exists(p: &Path) -> bool {
    p.symlink_metadata().is_ok()
}
```

(No `leak_str` helper is needed — `prior: Vec<(String, String)>` owns the strings and `prior_refs` borrows them for the `build_prompt` call within the same scope.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p megaresearcher-research --test orchestrator`
Expected: PASS (all orchestrator tests including the 3 new synthesize/integration tests; the `full_gap_finding_integration_test` is the spec §13.4 deliverable).

`cargo clippy -p megaresearcher-research --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/ crates/research/tests/orchestrator.rs
git commit -m "$(cat <<'EOF'
feat(rs): Phase 4a Task 7 — synthesist + finalize + gap-finding integration test

run_synthesist dispatches the single Phase 6 worker with spec/plan/all
scout + gap-finder outputs inlined; finalize_run copies synthesist/output.md
to the run root and creates the spec-latest relative symlink. execute() now
runs Phase 6, completing the gap-finding path (phases 3/4/5 stay skipped).
The integration test drives a 4-worker fake run end-to-end and asserts the
run tree, swarm-state phase statuses, output.md, and the symlink — the
spec §13.4 deliverable.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Green sweep (fmt + clippy + test from repo root) + v0 port-reference check

**Files:** none (verification only; fix any lint that surfaces).

**Global Constraints applied:** per-task hygiene used `--all-targets`, but the sweep re-confirms at workspace scope. v0 port-reference must be untouched.

- [ ] **Step 1: fmt**

Run: `cargo fmt --all --check`
Expected: exit 0, no output. If it prints a diff, run `cargo fmt --all` and commit the result.

- [ ] **Step 2: clippy (workspace, all targets)**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: exit 0, no warnings. If a lint surfaces (especially a test-only lint that per-task `--all-targets` should have caught), fix it in place and commit. Common Phase-3-style trap: a redundant `.into()` on a `format!()` String, or an unused import after wiring changed an import path. Fix behavior-preservingly.

- [ ] **Step 3: test (workspace)**

Run: `cargo test --workspace`
Expected: all pass. Phase 3 baseline was 1416; Phase 4a adds the orchestrator tests (plan_parser 5 + orchestrator ~25). The exact count is whatever lands — assert 0 failures. Research-crate tests are deterministic (fake-provider, no network). If a claurst timing test flakes once (pre-existing, observed in Phase 3), re-run; the research crate itself must be green on every run.

- [ ] **Step 4: v0 port-reference untouched**

Run: `git diff --stat 769ae23..HEAD -- lib/ tests/test_*.py skills/ agents/ .claude-plugin/ commands/ hooks/ mcp/ tools/ml-intern`
Expected: empty (no changes). The repo-root `agents/*.md` were only read + copied verbatim into `crates/research/tests/fixtures/agents/`; verify with `diff agents/gap-finder.md crates/research/tests/fixtures/agents/gap-finder.md` etc. (already done in Task 3, re-confirm here).

- [ ] **Step 5: research-crate file structure check**

Run: `find crates/research/src/orchestrator -type f` and `ls crates/research/tests/fixtures/agents crates/research/tests/fixtures/plans crates/research/tests/fixtures/specs`
Expected: `src/orchestrator/{mod,dispatch_plan,preflight,dispatch,gate,consolidate,synthesize}.rs`; fixtures `agents/{literature-scout,gap-finder,synthesist}.md`, `plans/gap-finding-plan.md`, `specs/gap-finding-spec.md`.

- [ ] **Step 6: Commit any sweep fixes, then record**

If Steps 1–2 needed fixes, commit them:
```bash
git add -A && git commit -m "$(cat <<'EOF'
style(rs): Phase 4a sweep fixes

<describe any fmt/clippy fix>

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```
If no fixes were needed, skip the commit.

Append to `.superpowers/sdd/progress.md`:
```
=== Phase 4a: ALL TASKS COMPLETE ===
Task 1: complete — parse_plan + data contracts
Task 2: complete — pre-flight + run setup + initial swarm-state
Task 3: complete — wave dispatch + worker construction
Task 4: complete — verification gate
Task 5: complete — deterministic consolidations
Task 6: complete — execute() Phases 1–2
Task 7: complete — synthesist + finalize + gap-finding integration test
Task 8: complete — green sweep (fmt/clippy/test clean from root, v0 untouched)
=== Phase 4a gap-finding integration test green (spec §13.4) ===
```

---

## Self-Review

**1. Spec coverage.** Spec §13.4 Phase 4 = "Orchestrator phases 1–6. Pre-flight, scaffold, wave dispatch, verification gate, red-team loop, consolidations, synthesist. Fake-provider integration test for a gap-finding run." Phase 4a covers: pre-flight (Task 2), scaffold + run-id (Task 2 + execute), wave dispatch (Task 3), verification gate (Task 4), consolidations (Task 5), synthesist (Task 7), gap-finding integration test (Task 7). The **red-team loop and hypothesis-target phases 3/4/5 are deliberately deferred to Phase 4b** — they require hypothesis-smith/red-team/eval-designer dispatch driven by worker manifests (gap-finder → smiths → red-team → eval-designer), a distinct increment from the gap-finding path. This split keeps each plan focused and independently testable (writing-plans: "Each plan should produce working, testable software on its own"). Phase 4a produces working, testable software: a complete gap-finding swarm. Gap noted explicitly, not hidden.

**2. Placeholder scan.** No "TBD"/"TODO"/"implement later". Every code step has complete code. The `leak_str` note in Task 7 offers a concrete alternative (inline loop) — both are complete approaches, not placeholders. No "add error handling" / "similar to Task N" without code.

**3. Type consistency.** `WorkerSpec` fields (`name`, `role`, `output_dir`, `shared_dir`, `prompt`) are consistent across Tasks 3/4/6/7. `GateStatus`/`GateOutcome` consistent across Tasks 4/6/7. `OrchestratorError` variants (`Preflight`, `Parse`, `Io`, `Worker`, `Escalated`, `Finalize`) used consistently. `RunOutcome` fields consistent. `build_prompt(spec_text, prior: &[(&str,&str)], assignment: Option<&Assignment>, output_dir)` consistent. `verify_wave(outcomes, specs, agents_dir, provider, default_model)` consistent. `set_phase(swarm, name, status, workers: Vec<(String,String)>)` consistent. The one signature evolution: `run_synthesist` returns `(WorkerSpec, WorkerOutcome)` (decided in Task 7's interface block and matched by the test + implementation).

**4. Determinism.** `run_id` injected (no `generate_run_id` in the orchestrator). Tests use `max_parallel = 1` + a flat scripted turn sequence so shared-fake call order is deterministic. Consolidations are file assemblers. `parse_plan` is a deterministic heading scanner.

**5. Risk: the shared-fake call-order assumption.** The integration tests assume workers consume exactly 4 turns each (3 writes + final) in dispatch order with `max_parallel = 1`. This holds because each scripted turn-set ends in a `final_turn` (`EndTurn`), so the worker stops after its block, and `buffer_unordered(1)` serializes dispatch in spec order. If a worker were to take an extra turn (e.g. the fake clamped to `final_turn` early), the sequence would desync — but the fakes are scripted to end each worker's block on `EndTurn`, so this does not occur in the scripted tests. The Task 4 gate tests script the precise 3-turn-first-run / 2-turn-retry counts to match.

If the implementer finds a compile or test issue that contradicts the plan (e.g. a `use` path, a claurst type name, or a fake-call-order desync), they should fix it in place with the smallest behavior-preserving change and note it in their report — do not silently leave the plan's code unverified.