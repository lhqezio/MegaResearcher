# MegaResearcher-rs Phase 4b — Orchestrator Hypothesis-Target Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the hypothesis-target path (Phases 3/4/5) to the Phase 4a orchestrator so a `novelty_target: hypothesis` run drives hypothesis-smith → red-team critique loop → eval-designer → synthesist, deterministically.

**Architecture:** Phase 4a built the gap-finding path (Phases 1→2→6) and the shared infra (`dispatch_wave`, `verify_wave`, `swarm-state`, `execute`). Phase 4b inserts Phases 3/4/5 between Phase 2 and Phase 6, gated on `!NoveltyTarget::skips_critique_phases()`. New leaf modules under `crates/research/src/orchestrator/`: `gaps.rs` (deterministic gap enumeration from gap-finder manifests), `verdict.rs` (red-team verdict parser), `hypothesis.rs` (smith dispatch + revision), `redteam.rs` (the critique loop, cap 3 revisions), `evaldesign.rs` (eval-designer fan-out + intractable flagging). `synthesize.rs` extends `run_synthesist` to inline smith/red-team/eval-designer outputs. Same single-session orchestrator + leaf-worker pattern as 4a.

**Tech Stack:** Rust 2021, tokio, futures, serde + serde_yml, regex + once_cell, claurst `api::LlmProvider` (`Arc<dyn LlmProvider>`). Tests drive a shared `FakeProvider` with `max_parallel = 1` so the global `call_index` consumes scripted turns in dispatch order.

## Global Constraints

[Inherited verbatim from the Phase 4a plan + project CLAUDE.md — binding for every task:]

- Edit ONLY `crates/research/` (+ workspace `Cargo.toml`/`Cargo.lock` if a new dep is needed — none is expected; `serde_yml`, `regex`, `once_cell` are already workspace deps from Phases 2/3). The v0 plugin port-reference (`lib/`, repo-root `tests/test_*.py`, `skills/`, repo-root `agents/`, `.claude-plugin/`, `commands/`, `hooks/`, `mcp/`, `tools/ml-intern`) MUST NOT be modified — repo-root `agents/*.md` are read-only source for byte-identical fixture copies only.
- No crate-root `pub use` re-exports — `lib.rs` uses `pub mod` only; consumers use full paths `megaresearcher_research::orchestrator::...`.
- claurst `api` crate `LlmProvider` trait (object-safe, `Arc<dyn LlmProvider>`) is the provider seam — no api-crate changes.
- Per-task hygiene: `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` (bare `-D warnings` skips test binaries). Run `cargo fmt -p megaresearcher-research` before commit. Each feature task ends with `cargo test -p megaresearcher-research` green.
- Banned phrases in implementer-PRODUCED text: "load-bearing", "this is doing a lot of work", "real" (emphatic adjective), "honest"/"honestly"/"to be honest". Exception: byte-identical copies of port-reference source files (e.g. `red-team.md`, `eval-designer.md` which contain "honest"/"Honest" verbatim from v0) are NOT implementer-produced — ban does not apply to verbatim copies. Genuine technical terms ("real-time", "real number") are fine.
- Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`.
- Work directly on branch `main` (NO worktrees — hard non-negotiable rule). Before dispatching any implementer subagent, the controller confirms the branch with `git branch --show-current` and includes the explicit branch name in the prompt so the subagent never `git switch` mid-task.
- GPL-3.0.
- **Determinism model (continues from 4a):** the orchestrator never calls `generate_run_id` — `run_id` is injected by the caller. v0's LLM-judgment surfaces (gap enumeration, verdict parsing, intractable flagging) are replaced with deterministic Rust parsers over worker-written manifests/output.md. Tests use `max_parallel = 1` + flat scripted turn sequences so `FakeProvider`'s global `call_index` is deterministic. Each worker = exactly 4 scripted turns (3 writes + a final `EndTurn`) unless a gate retry fires (fakes write all 3 artifacts, so no gate retries).

---

## File Structure

New files (all under `crates/research/src/orchestrator/` and `crates/research/tests/`):

| File | Responsibility |
|---|---|
| `src/orchestrator/gaps.rs` | `Gap`/`GapEntry`/`GapFinderManifest` structs; `parse_gaps` (serde_yml manifest parse); `collect_gaps` (aggregate across gap-finders). Deterministic replacement for v0's "parse the consolidated gaps.md" LLM step. |
| `src/orchestrator/verdict.rs` | `RedTeamVerdict { Approve, Reject{revision}, Kill }`; `parse_redteam_verdict` (regex over the red-team `output.md` Verdict line). Distinct from `paper_chain/verdict.rs` (which parses the peer-reviewer `VERDICT:` line — different format). |
| `src/orchestrator/hypothesis.rs` | `Hypothesis` struct (carried 3→4→5); `build_smith_spec` (initial + revision); `dispatch_hypothesis_smiths` (Phase 3: one smith per gap, wave, gate); `redispatch_smith_revision` (single-worker revision re-dispatch used by the loop). |
| `src/orchestrator/redteam.rs` | `RedTeamResult`; `build_redteam_prompt`; `run_redteam_loop` (Phase 4: per-hypothesis dispatch red-team → parse verdict → APPROVE survives / REJECT revises (cap 3) / KILL or cap → escalate). Mutates `swarm.retry_counts` + `swarm.escalations`. |
| `src/orchestrator/evaldesign.rs` | `EvalDesignResult`; `build_evaldesigner_prompt`; `parse_intractable`; `run_eval_designers` (Phase 5: one eval-designer per survivor, gate, intractable → escalate but continue). Mutates `swarm.escalations`. |
| `src/orchestrator/mod.rs` | Add `pub mod gaps/verdict/hypothesis/redteam/evaldesign`; extend `execute()` with Phases 3/4/5 (gated on `!skips_critique_phases()`); import the new modules. |
| `src/orchestrator/synthesize.rs` | Extend `run_synthesist` signature: +`smith_dirs`, `redteam_dirs`, `eval_dirs` params (empty for gap-finding runs). |
| `tests/orchestrator.rs` | New tests for each phase + the hypothesis-target integration tests; update the 4a `run_synthesist` direct test to the new signature. |
| `tests/fixtures/agents/hypothesis-smith.md` | Byte-identical copy of repo-root `agents/hypothesis-smith.md`. |
| `tests/fixtures/agents/red-team.md` | Byte-identical copy of repo-root `agents/red-team.md`. |
| `tests/fixtures/agents/eval-designer.md` | Byte-identical copy of repo-root `agents/eval-designer.md`. |
| `tests/fixtures/plans/hypothesis-plan.md` | A `novelty_target: hypothesis` plan with scouts + gap-finders (smiths are derived from gaps, not listed in the plan — same as 4a). |
| `tests/fixtures/specs/hypothesis-spec.md` | A tiny `novelty_target: hypothesis`-bearing spec (the spec frontmatter is read by preflight? No — novelty_target lives in the PLAN frontmatter; the spec is just text passed to workers. So this is a trivial spec body). |

**4a files NOT modified except as explicitly noted:** `dispatch.rs`, `gate.rs`, `preflight.rs`, `consolidate.rs`, `dispatch_plan.rs`, `state/swarm_state.rs` are consumed unchanged. `mod.rs` and `synthesize.rs` are the only 4a files modified.

---

## Determinism model (Phase 4b additions)

| v0 LLM-judgment surface | Rust deterministic replacement |
|---|---|
| "parse the consolidated gaps.md; dispatch one smith per gap" | `collect_gaps` reads each gap-finder's `manifest.yaml` `gaps:` structured list (serde_yml); one smith per `Gap`, in aggregate (finder-then-gap) order. |
| "Read the red-team output.md. Parse the verdict line (`APPROVE \| REJECT (revision-N) \| KILL`)" | `parse_redteam_verdict` regex over `output.md`. |
| "If retry < 3, dispatch hypothesis-smith again … re-dispatch red-team … Continue the loop. If retry hits 3 OR KILL: escalate" | `run_redteam_loop` bounded loop, `revision_count` in `swarm.retry_counts[hyp]`, cap 3 revisions, KILL/cap/None → `add_escalation`. |
| "If a designer flags `flagged_intractable: true` in its manifest, surface to user before continuing" | `parse_intractable` reads `flagged_intractable` from the eval-designer `manifest.yaml`; true → `add_escalation` + continue (the eval-designer output still goes to synthesis — preserves the audit trail). |
| Synthesist inlines "all Phase-3 smith outputs (incl revisions), all Phase-4 red-team verdicts, all Phase-5 eval-designer outputs" | `run_synthesist` extended `prior` appends `"Hypothesis-smith <n>"`, `"Red-team <n>-r<round>"`, `"Eval-designer <n>"` sections (each worker's latest `output.md`). Smith revisions overwrite in-place so `output.md` carries all revision responses. |

**Red-team dir naming (audit trail):** `red-team-<N>-r<round>/` where `<N>` is the hypothesis number and `<round>` is the 1-based critique round (round 1 = first critique; round 2 = after the first revision). Each round's `output.md` is preserved (not overwritten), satisfying MegaResearcher's "audit trail is non-negotiable" discipline.

**Smith revision dir:** the smith writes to `hypothesis-smith-<N>/` (same dir, overwrites `output.md` with the revised version that includes the "Revision response (red-team round N)" section). The latest `output.md` is what the synthesist inlines.

**Hypothesis numbering:** smiths are numbered globally across all gaps in aggregate order: `hypothesis-smith-1` targets the first gap of the first gap-finder, etc. The hypothesis number is stable across Phases 3/4/5 (a survivor keeps its number into Phase 5; the eval-designer for hypothesis-N is `eval-designer-<N>`).

---

## Task 1: gaps.rs — deterministic gap enumeration

**Files:**
- Create: `crates/research/src/orchestrator/gaps.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` (add `pub mod gaps;` to the module block)
- Test: `crates/research/tests/orchestrator.rs` (append the gaps tests)

**Interfaces:**
- Consumes: `serde`/`serde_yml` (workspace deps from Phase 2), `std::fs`, `std::path`, `std::io`.
- Produces:
  - `pub struct Gap { pub id: String, pub finder_name: String, pub finder_dir: PathBuf, pub statement: String, pub gap_type: String }`
  - `pub fn parse_gaps(manifest_yaml: &str, finder_name: &str, finder_dir: &Path) -> Vec<Gap>`
  - `pub fn collect_gaps(run_dir: &Path, gap_dirs: &[PathBuf]) -> io::Result<Vec<Gap>>`

- [ ] **Step 1: Write the failing tests**

Append to `crates/research/tests/orchestrator.rs` (after the existing tests; add `use megaresearcher_research::orchestrator::gaps::{collect_gaps, parse_gaps, Gap};` at the top of the appended block — mid-file `use` is consistent with the 4a file's style):

```rust
use megaresearcher_research::orchestrator::gaps::{collect_gaps, parse_gaps, Gap};

#[test]
fn parse_gaps_reads_structured_gaps_list_from_manifest() {
    let manifest = "\
role: gap-finder
slice: scouts 1-2
gaps_count: 2
discarded_count: 1
gaps:
  - id: gap-1
    statement: Technique X never applied to A+B fusion.
    type: unexplored-intersection
  - id: gap-2
    statement: Paper P and paper Q disagree on metric M.
    type: contradiction
";
    let finder_dir = PathBuf::from("/tmp/run/gap-finder-1");
    let gaps = parse_gaps(manifest, "gap-finder-1", &finder_dir);
    assert_eq!(gaps.len(), 2);
    assert_eq!(gaps[0].id, "gap-1");
    assert_eq!(gaps[0].finder_name, "gap-finder-1");
    assert_eq!(gaps[0].finder_dir, finder_dir);
    assert_eq!(gaps[0].statement, "Technique X never applied to A+B fusion.");
    assert_eq!(gaps[0].gap_type, "unexplored-intersection");
    assert_eq!(gaps[1].id, "gap-2");
    assert_eq!(gaps[1].gap_type, "contradiction");
}

#[test]
fn parse_gaps_returns_empty_when_manifest_has_no_gaps_block() {
    // A v0-style manifest with only gaps_count (no structured `gaps:` list) yields
    // zero gaps — the orchestrator then dispatches no smiths. serde(default) makes
    // the missing field parse cleanly.
    let manifest = "role: gap-finder\ngaps_count: 3\ndiscarded_count: 1\n";
    let gaps = parse_gaps(manifest, "gap-finder-2", Path::new("/tmp/r/gap-finder-2"));
    assert!(gaps.is_empty());
}

#[test]
fn parse_gaps_skips_gaps_with_empty_statement() {
    let manifest = "\
gaps:
  - id: gap-1
    statement: A real gap.
    type: contradiction
  - id: gap-2
    statement: ''
    type: missing-baseline
";
    let gaps = parse_gaps(manifest, "gap-finder-1", Path::new("/tmp/r/gap-finder-1"));
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].id, "gap-1");
}

#[test]
fn collect_gaps_aggregates_across_gap_finders_in_order() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let g1 = run_dir.join("gap-finder-1");
    let g2 = run_dir.join("gap-finder-2");
    fs::create_dir_all(&g1).unwrap();
    fs::create_dir_all(&g2).unwrap();
    fs::write(
        g1.join("manifest.yaml"),
        "role: gap-finder\ngaps:\n  - id: gap-1\n    statement: A.\n    type: contradiction\n  - id: gap-2\n    statement: B.\n    type: missing-baseline\n",
    )
    .unwrap();
    fs::write(
        g2.join("manifest.yaml"),
        "role: gap-finder\ngaps:\n  - id: gap-1\n    statement: C.\n    type: unexplored-intersection\n",
    )
    .unwrap();
    let gaps = collect_gaps(&run_dir, &[g1.clone(), g2.clone()]).unwrap();
    assert_eq!(gaps.len(), 3);
    // finder-then-gap order; ids are finder-local, so "gap-1" repeats across finders.
    assert_eq!(gaps[0].finder_name, "gap-finder-1");
    assert_eq!(gaps[0].statement, "A.");
    assert_eq!(gaps[1].finder_name, "gap-finder-1");
    assert_eq!(gaps[1].statement, "B.");
    assert_eq!(gaps[2].finder_name, "gap-finder-2");
    assert_eq!(gaps[2].statement, "C.");
    assert_eq!(gaps[2].finder_dir, g2);
}

#[test]
fn collect_gaps_skips_finders_whose_manifest_is_unreadable() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let g1 = run_dir.join("gap-finder-1");
    let g2 = run_dir.join("gap-finder-2"); // no manifest written
    fs::create_dir_all(&g1).unwrap();
    fs::create_dir_all(&g2).unwrap();
    fs::write(
        g1.join("manifest.yaml"),
        "role: gap-finder\ngaps:\n  - id: gap-1\n    statement: A.\n    type: contradiction\n",
    )
    .unwrap();
    // g2 has no manifest.yaml -> collect_gaps skips it (no error, no gaps from it).
    let gaps = collect_gaps(&run_dir, &[g1, g2]).unwrap();
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].statement, "A.");
}
```

Note: the `A real gap.` literal in `parse_gaps_skips_gaps_with_empty_statement` is fixture content inside a test string, not implementer prose — but to respect the ban strictly, replace it with a non-banned phrase. Use statement `"A defensible gap."` instead of `"A real gap."`. (The implementer must use `"A defensible gap."` — do NOT write "real".)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p megaresearcher-research --test orchestrator parse_gaps collect_gaps 2>&1 | tail -20`
Expected: FAIL — `unresolved import megaresearcher_research::orchestrator::gaps` (module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `crates/research/src/orchestrator/gaps.rs`:

```rust
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
```

Modify `crates/research/src/orchestrator/mod.rs` module block — add `pub mod gaps;` (alphabetical-ish among the new 4b modules; the final 4b module block will read `consolidate / dispatch / dispatch_plan / gaps / gate / hypothesis / preflight / redteam / evaldesign / synthesize / verdict` — exact order is not load-bearing, but keep it readable). For Task 1 add only:

```rust
pub mod gaps;
```

alongside the existing `pub mod consolidate;` etc.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test orchestrator parse_gaps collect_gaps 2>&1 | tail -20`
Expected: PASS — 5 new tests green.

Then full crate hygiene:
```
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
cargo test -p megaresearcher-research
```
Expected: clippy exit 0; all research-crate tests pass (4a baseline + 5 new).

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/gaps.rs crates/research/src/orchestrator/mod.rs crates/research/tests/orchestrator.rs
git commit -m "feat(rs): Phase 4b Task 1 — deterministic gap enumeration from manifests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: verdict.rs — red-team verdict parser

**Files:**
- Create: `crates/research/src/orchestrator/verdict.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` (add `pub mod verdict;`)
- Test: `crates/research/tests/orchestrator.rs` (append verdict tests)

**Interfaces:**
- Consumes: `regex` + `once_cell` (workspace deps from Phase 2), `std::fs`, `std::path`.
- Produces:
  - `pub enum RedTeamVerdict { Approve, Reject { revision: u32 }, Kill }`
  - `pub fn parse_redteam_verdict(output_md: &str) -> Option<RedTeamVerdict>`
  - `pub fn parse_redteam_verdict_file(path: &Path) -> io::Result<Option<RedTeamVerdict>>`

**Design note:** This is distinct from `paper_chain/verdict.rs`, which parses the peer-reviewer's `^VERDICT: (APPROVE|REVISE|KILL)$` line (Phase 8 of the paper chain). The red-team worker emits a different line shape (see `agents/red-team.md`): a numbered section `1. **Verdict** — APPROVE | REJECT (revision-N) | KILL (irrecoverable)`. Do NOT reuse or modify `paper_chain/verdict.rs`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/research/tests/orchestrator.rs`:

```rust
use megaresearcher_research::orchestrator::verdict::{
    parse_redteam_verdict, RedTeamVerdict,
};

#[test]
fn parse_redteam_verdict_approve() {
    let md = "# Red-team critique\n\n1. **Verdict** — APPROVE\n\n2. Gap re-verification: ...\n";
    assert_eq!(parse_redteam_verdict(md), Some(RedTeamVerdict::Approve));
}

#[test]
fn parse_redteam_verdict_reject_with_revision_number() {
    let md = "## Verdict\n\n**Verdict** — REJECT (revision-2)\n\nThe mechanism is unsupported.\n";
    assert_eq!(
        parse_redteam_verdict(md),
        Some(RedTeamVerdict::Reject { revision: 2 })
    );
}

#[test]
fn parse_redteam_verdict_kill() {
    let md = "1. **Verdict** — KILL (irrecoverable)\n\nThe gap is not actually unexplored.\n";
    assert_eq!(parse_redteam_verdict(md), Some(RedTeamVerdict::Kill));
}

#[test]
fn parse_redteam_verdict_tolerates_spacing_and_dash_variants() {
    // Workers vary the em-dash / colon / spacing. Accept any of "—", "-", ":".
    let cases = [
        "Verdict: APPROVE",
        "Verdict - APPROVE",
        "**Verdict** — APPROVE",
        "1. **Verdict** — REJECT (revision-1)",
    ];
    assert_eq!(parse_redteam_verdict(cases[0]), Some(RedTeamVerdict::Approve));
    assert_eq!(parse_redteam_verdict(cases[1]), Some(RedTeamVerdict::Approve));
    assert_eq!(parse_redteam_verdict(cases[2]), Some(RedTeamVerdict::Approve));
    assert_eq!(
        parse_redteam_verdict(cases[3]),
        Some(RedTeamVerdict::Reject { revision: 1 })
    );
}

#[test]
fn parse_redteam_verdict_returns_none_when_no_verdict_line() {
    let md = "# Red-team critique\n\nNo verdict here, just discussion.\n";
    assert_eq!(parse_redteam_verdict(md), None);
}

#[test]
fn parse_redteam_verdict_returns_none_for_format_description_lines() {
    // A worker that echoes the agent's format instruction ("exactly one of: APPROVE | ...")
    // must NOT be misread as an APPROVE verdict.
    let md = "The verdict is exactly one of: APPROVE | REJECT (revision-N) | KILL (irrecoverable).\n";
    assert_eq!(parse_redteam_verdict(md), None);
}

#[test]
fn parse_redteam_verdict_file_reads_disk() {
    let tmp = tempdir().unwrap();
    let p = tmp.path().join("red-team-1-r1").join("output.md");
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(&p, "1. **Verdict** — REJECT (revision-1)\n").unwrap();
    let v = megaresearcher_research::orchestrator::verdict::parse_redteam_verdict_file(&p)
        .unwrap();
    assert_eq!(v, Some(RedTeamVerdict::Reject { revision: 1 }));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p megaresearcher-research --test orchestrator parse_redteam 2>&1 | tail -15`
Expected: FAIL — `unresolved import ...::verdict`.

- [ ] **Step 3: Write minimal implementation**

Create `crates/research/src/orchestrator/verdict.rs`:

```rust
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
```

Modify `crates/research/src/orchestrator/mod.rs`: add `pub mod verdict;`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test orchestrator parse_redteam 2>&1 | tail -15`
Expected: PASS — 7 new tests green.

Hygiene:
```
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
cargo test -p megaresearcher-research
```
Expected: clippy exit 0; all green.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/verdict.rs crates/research/src/orchestrator/mod.rs crates/research/tests/orchestrator.rs
git commit -m "feat(rs): Phase 4b Task 2 — red-team verdict parser

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: hypothesis.rs — Phase 3 smith dispatch + revision

**Files:**
- Create: `crates/research/src/orchestrator/hypothesis.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` (add `pub mod hypothesis;`)
- Create: `crates/research/tests/fixtures/agents/hypothesis-smith.md` (byte-identical copy of repo-root `agents/hypothesis-smith.md`)
- Test: `crates/research/tests/orchestrator.rs` (append hypothesis tests)

**Interfaces:**
- Consumes: `dispatch::{build_prompt, dispatch_wave, WorkerSpec}`, `gate::{verify_wave, GateStatus}`, `OrchestratorError`, `gaps::Gap`, `claurst_api::LlmProvider`, `worker::WorkerOutcome`.
- Produces:
  - `pub struct Hypothesis { pub name: String, pub dir: PathBuf, pub gap: Gap }`
  - `pub fn build_smith_spec(spec_text: &str, gap: &Gap, hyp_name: &str, output_dir: &Path, shared_dir: &Path, revision_prior: Option<&str>) -> WorkerSpec`
  - `pub async fn dispatch_hypothesis_smiths(run_dir: &Path, spec_text: &str, gaps: &[Gap], agents_dir: &Path, provider: Arc<dyn LlmProvider>, default_model: &str, max_parallel: u32) -> Result<Vec<Hypothesis>, OrchestratorError>`
  - `pub async fn redispatch_smith_revision(hyp: &Hypothesis, spec_text: &str, redteam_output: &str, agents_dir: &Path, provider: Arc<dyn LlmProvider>, default_model: &str) -> Result<(), OrchestratorError>`

**Behavior:**
- `dispatch_hypothesis_smiths` numbers smiths 1..=gaps.len() in aggregate order. Each smith's `output_dir = run_dir/hypothesis-smith-<N>`, `shared_dir = run_dir`. The prompt inlines the gap statement + the gap-finder's `output.md` (read from `gap.finder_dir`) as prior. dispatch_wave → verify_wave. On any `GateStatus::Escalated` → `Err(OrchestratorError::Escalated(names))` (halts the run, matching scout/gap-finder gate behavior). On all Passed → returns `Vec<Hypothesis>` (name, dir, gap).
- `redispatch_smith_revision` builds a revision smith spec to `hyp.dir` (overwrites `output.md`) whose prior includes `"Previous red-team critique"` = `redteam_output`. Single-worker dispatch_wave → verify_wave. On Escalated → `Err(OrchestratorError::Escalated([hyp.name]))`.
- `build_smith_spec` uses a synthetic `Assignment { id: hyp_name, role: "hypothesis-smith", title: "Forge a hypothesis for this gap", body: gap.statement }` so the existing `build_prompt` formats it as a titled assignment. Prior = `[("Gap-finder output", <finder output.md or "(no output.md)">)]` plus, on revision, `[("Previous red-team critique", redteam_output)]` appended.

- [ ] **Step 1: Write the failing tests**

Append to `crates/research/tests/orchestrator.rs`:

```rust
use megaresearcher_research::orchestrator::gaps::Gap;
use megaresearcher_research::orchestrator::hypothesis::{
    build_smith_spec, dispatch_hypothesis_smiths, redispatch_smith_revision, Hypothesis,
};

fn gap(id: &str, finder: &str, finder_dir: &Path, statement: &str) -> Gap {
    Gap {
        id: id.into(),
        finder_name: finder.into(),
        finder_dir: finder_dir.to_path_buf(),
        statement: statement.into(),
        gap_type: "contradiction".into(),
    }
}

#[test]
fn build_smith_spec_inlines_gap_and_finder_output() {
    let tmp = tempdir().unwrap();
    let finder_dir = tmp.path().join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER OUTPUT BODY").unwrap();
    let g = gap("gap-1", "gap-finder-1", &finder_dir, "Technique X never applied to A+B.");
    let spec = build_smith_spec(
        "SPEC TEXT",
        &g,
        "hypothesis-smith-1",
        &tmp.path().join("hypothesis-smith-1"),
        tmp.path(),
        None,
    );
    assert_eq!(spec.name, "hypothesis-smith-1");
    assert_eq!(spec.role, "hypothesis-smith");
    assert!(spec.prompt.contains("SPEC TEXT"));
    assert!(spec.prompt.contains("Technique X never applied to A+B."));
    assert!(spec.prompt.contains("FINDER OUTPUT BODY"));
    assert!(spec.prompt.contains("hypothesis-smith-1"));
    // No revision prior on initial dispatch.
    assert!(!spec.prompt.contains("Previous red-team critique"));
}

#[test]
fn build_smith_spec_revision_appends_redteam_prior() {
    let tmp = tempdir().unwrap();
    let finder_dir = tmp.path().join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER BODY").unwrap();
    let g = gap("gap-1", "gap-finder-1", &finder_dir, "The gap statement.");
    let spec = build_smith_spec(
        "SPEC",
        &g,
        "hypothesis-smith-2",
        &tmp.path().join("hypothesis-smith-2"),
        tmp.path(),
        Some("RED-TEAM CRITIQUE BODY"),
    );
    assert!(spec.prompt.contains("Previous red-team critique"));
    assert!(spec.prompt.contains("RED-TEAM CRITIQUE BODY"));
}

#[tokio::test]
async fn dispatch_hypothesis_smiths_one_per_gap_and_gates() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let finder_dir = run_dir.join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER BODY").unwrap();
    let gaps = vec![
        gap("gap-1", "gap-finder-1", &finder_dir, "Gap one statement."),
        gap("gap-2", "gap-finder-1", &finder_dir, "Gap two statement."),
    ];
    // 2 smiths x 4 scripted turns = 8 turns.
    let fake = Arc::new(FakeProvider::new("fake", run_turns(2)));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let hyps = dispatch_hypothesis_smiths(
        &run_dir,
        "SPEC",
        &gaps,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
    )
    .await
    .unwrap();
    assert_eq!(hyps.len(), 2);
    assert_eq!(hyps[0].name, "hypothesis-smith-1");
    assert_eq!(hyps[1].name, "hypothesis-smith-2");
    // Each smith wrote all three artifacts.
    for h in &hyps {
        assert!(h.dir.join("output.md").exists());
        assert!(h.dir.join("manifest.yaml").exists());
        assert!(h.dir.join("verification.md").exists());
    }
    assert_eq!(fake.call_count(), 8);
}

#[tokio::test]
async fn dispatch_hypothesis_smiths_halts_on_gate_escalation() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let finder_dir = run_dir.join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER BODY").unwrap();
    let gaps = vec![gap("gap-1", "gap-finder-1", &finder_dir, "Gap one.")];
    // A single final-only turn: the smith writes nothing, the gate retries, the
    // retry also writes nothing (.or_else(last) re-emits the same empty turn),
    // so the smith escalates.
    let turns: Vec<Vec<StreamEvent>> = vec![final_turn("nothing written")];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let err = dispatch_hypothesis_smiths(
        &run_dir,
        "SPEC",
        &gaps,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
    )
    .await
    .expect_err("should escalate");
    match err {
        OrchestratorError::Escalated(names) => assert_eq!(names, vec!["hypothesis-smith-1".to_string()]),
        other => panic!("expected Escalated, got {other:?}"),
    }
}

#[tokio::test]
async fn redispatch_smith_revision_overwrites_output_and_gates() {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let finder_dir = run_dir.join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER BODY").unwrap();
    let g = gap("gap-1", "gap-finder-1", &finder_dir, "The gap.");
    let hyp = Hypothesis {
        name: "hypothesis-smith-1".into(),
        dir: run_dir.join("hypothesis-smith-1"),
        gap: g,
    };
    fs::create_dir_all(&hyp.dir).unwrap();
    fs::write(hyp.dir.join("output.md"), "OLD INITIAL HYPOTHESIS").unwrap();
    // 4 turns for the revision dispatch.
    let fake = Arc::new(FakeProvider::new("fake", run_turns(1)));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    redispatch_smith_revision(&hyp, "SPEC", "RED-TEAM CRITIQUE", &fixture_agents_dir(), provider, "fake-model")
        .await
        .unwrap();
    // The revised output overwrote the old one (the fake writes "# Output\n\ncontent").
    assert_ne!(
        fs::read_to_string(hyp.dir.join("output.md")).unwrap(),
        "OLD INITIAL HYPOTHESIS"
    );
    assert!(hyp.dir.join("manifest.yaml").exists());
    assert_eq!(fake.call_count(), 4);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p megaresearcher-research --test orchestrator build_smith_spec dispatch_hypothesis_smiths redispatch_smith 2>&1 | tail -15`
Expected: FAIL — `unresolved import ...::hypothesis`.

- [ ] **Step 3: Create the fixture + write implementation**

Create the byte-identical fixture:
```bash
cp agents/hypothesis-smith.md crates/research/tests/fixtures/agents/hypothesis-smith.md
diff -q agents/hypothesis-smith.md crates/research/tests/fixtures/agents/hypothesis-smith.md && echo "byte-identical"
```
(The implementer runs this `cp` + `diff` from the repo root; the file must be byte-identical. The fixture's `model: inherit` frontmatter resolves to `default_model` via `run_worker`.)

Create `crates/research/src/orchestrator/hypothesis.rs`:

```rust
//! Phase 3 hypothesis-smith dispatch (one smith per gap) + the revision
//! re-dispatch the Phase 4 red-team loop drives on a REJECT verdict.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};
use crate::orchestrator::dispatch_plan::Assignment;
use crate::orchestrator::gate::{verify_wave, GateStatus};
use crate::orchestrator::gaps::Gap;
use crate::orchestrator::OrchestratorError;

/// A hypothesis forged in Phase 3 and carried through Phases 4 and 5. `name` is
/// `hypothesis-smith-<N>`; `dir` is `run_dir/hypothesis-smith-<N>`; `gap` is the
/// targeted gap (so red-team/eval-designer prompts can inline the gap-finder).
#[derive(Debug, Clone)]
pub struct Hypothesis {
    pub name: String,
    pub dir: PathBuf,
    pub gap: Gap,
}

/// Build the worker spec for a hypothesis-smith. On initial dispatch
/// (`revision_prior == None`) the prompt inlines the gap (as a titled
/// assignment) and the gap-finder's `output.md`. On revision, the previous
/// red-team critique is appended as an extra prior section.
pub fn build_smith_spec(
    spec_text: &str,
    gap: &Gap,
    hyp_name: &str,
    output_dir: &Path,
    shared_dir: &Path,
    revision_prior: Option<&str>,
) -> WorkerSpec {
    let finder_output = fs::read_to_string(gap.finder_dir.join("output.md"))
        .unwrap_or_else(|_| "(no output.md)".to_string());
    let mut prior: Vec<(&str, &str)> = Vec::new();
    prior.push(("Gap-finder output", &finder_output));
    if let Some(critique) = revision_prior {
        prior.push(("Previous red-team critique", critique));
    }
    let assignment = Assignment {
        id: hyp_name.to_string(),
        role: "hypothesis-smith".to_string(),
        title: "Forge a hypothesis for this gap".to_string(),
        body: format!(
            "Targeted gap (from {}): {}\nGap category: {}",
            gap.finder_name, gap.statement, gap.gap_type
        ),
    };
    WorkerSpec {
        name: hyp_name.to_string(),
        role: "hypothesis-smith".to_string(),
        output_dir: output_dir.to_path_buf(),
        shared_dir: shared_dir.to_path_buf(),
        prompt: build_prompt(spec_text, &prior, Some(&assignment), output_dir),
    }
}

/// Phase 3: dispatch one hypothesis-smith per gap, in aggregate order, bounded
/// by `max_parallel`. Run the verification gate. Any gate escalation halts the
/// run with `Err(Escalated)` (matching scout/gap-finder gate behavior). On
/// success returns one `Hypothesis` per gap.
pub async fn dispatch_hypothesis_smiths(
    run_dir: &Path,
    spec_text: &str,
    gaps: &[Gap],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
) -> Result<Vec<Hypothesis>, OrchestratorError> {
    let specs: Vec<WorkerSpec> = gaps
        .iter()
        .enumerate()
        .map(|(i, gap)| {
            let name = format!("hypothesis-smith-{}", i + 1);
            let output_dir = run_dir.join(&name);
            fs::create_dir_all(&output_dir).ok();
            build_smith_spec(spec_text, gap, &name, &output_dir, run_dir, None)
        })
        .collect();
    let outcomes = dispatch_wave(
        specs.clone(),
        agents_dir,
        provider.clone(),
        default_model,
        max_parallel,
    )
    .await?;
    let gates = verify_wave(outcomes, &specs, agents_dir, provider, default_model).await?;
    let escalated: Vec<String> = gates
        .iter()
        .filter(|g| g.status == GateStatus::Escalated)
        .map(|g| g.name.clone())
        .collect();
    if !escalated.is_empty() {
        return Err(OrchestratorError::Escalated(escalated));
    }
    Ok(specs
        .into_iter()
        .zip(gates.iter())
        .map(|(s, _)| Hypothesis {
            name: s.name,
            dir: s.output_dir,
            gap: gaps[specs_iter_index(&s.name)].clone(),
        })
        .collect())
}

/// Map a smith name `hypothesis-smith-<N>` back to its gap index. Used to
/// re-pair the `WorkerSpec` (which carries no gap) with the originating gap.
fn specs_iter_index(name: &str) -> usize {
    name.trim_start_matches("hypothesis-smith-")
        .parse::<usize>()
        .unwrap()
        .saturating_sub(1)
}
```

Wait — the `zip(gates.iter())` approach pairs specs with gates, but `gates` from `verify_wave` is returned in the order of `outcomes` which `dispatch_wave` already sorted by index, so it matches `specs` order. The `gaps[specs_iter_index(&s.name)]` re-lookup is awkward. Simplify: since specs were built in `gaps` order with `i+1` names, pair by position directly. Replace the `Ok(...)` block with:

```rust
    Ok(gaps
        .iter()
        .enumerate()
        .map(|(i, gap)| {
            let name = format!("hypothesis-smith-{}", i + 1);
            Hypothesis {
                name,
                dir: run_dir.join(format!("hypothesis-smith-{}", i + 1)),
                gap: gap.clone(),
            }
        })
        .collect())
```

and delete the `specs_iter_index` helper entirely. (The `specs` vec was already used for dispatch/gate; the `Hypothesis` vec is rebuilt from `gaps` after the gate passed.) The implementer MUST use this simplified form — do NOT include `specs_iter_index`.

`Gap` needs `Clone`. Add `#[derive(Debug, Clone, PartialEq, Eq)]` on `Gap` in `gaps.rs` — verify it is already `Clone` (it is, from Task 1). Good.

Full revised `dispatch_hypothesis_smiths` body (the implementer writes this exact version, no `specs_iter_index`):

```rust
pub async fn dispatch_hypothesis_smiths(
    run_dir: &Path,
    spec_text: &str,
    gaps: &[Gap],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
) -> Result<Vec<Hypothesis>, OrchestratorError> {
    let specs: Vec<WorkerSpec> = gaps
        .iter()
        .enumerate()
        .map(|(i, gap)| {
            let name = format!("hypothesis-smith-{}", i + 1);
            let output_dir = run_dir.join(&name);
            fs::create_dir_all(&output_dir).ok();
            build_smith_spec(spec_text, gap, &name, &output_dir, run_dir, None)
        })
        .collect();
    let outcomes = dispatch_wave(
        specs.clone(),
        agents_dir,
        provider.clone(),
        default_model,
        max_parallel,
    )
    .await?;
    let gates = verify_wave(outcomes, &specs, agents_dir, provider, default_model).await?;
    let escalated: Vec<String> = gates
        .iter()
        .filter(|g| g.status == GateStatus::Escalated)
        .map(|g| g.name.clone())
        .collect();
    if !escalated.is_empty() {
        return Err(OrchestratorError::Escalated(escalated));
    }
    Ok(gaps
        .iter()
        .enumerate()
        .map(|(i, gap)| Hypothesis {
            name: format!("hypothesis-smith-{}", i + 1),
            dir: run_dir.join(format!("hypothesis-smith-{}", i + 1)),
            gap: gap.clone(),
        })
        .collect())
}
```

Now `redispatch_smith_revision`:

```rust
/// Phase 4 revision step: re-dispatch the hypothesis-smith for `hyp` to its own
/// dir (overwriting `output.md` with the revised version that absorbs the
/// red-team critique). Single-worker wave + gate. Escalation halts the run.
pub async fn redispatch_smith_revision(
    hyp: &Hypothesis,
    spec_text: &str,
    redteam_output: &str,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
) -> Result<(), OrchestratorError> {
    let spec = build_smith_spec(
        spec_text,
        &hyp.gap,
        &hyp.name,
        &hyp.dir,
        &hyp.dir.parent().unwrap_or_else(|| Path::new(".")),
        Some(redteam_output),
    );
    let outcomes = dispatch_wave(
        vec![spec.clone()],
        agents_dir,
        provider.clone(),
        default_model,
        1,
    )
    .await?;
    let gates = verify_wave(outcomes, std::slice::from_ref(&spec), agents_dir, provider, default_model).await?;
    if gates[0].status == GateStatus::Escalated {
        return Err(OrchestratorError::Escalated(vec![hyp.name.clone()]));
    }
    Ok(())
}
```

Note: `shared_dir` for the revision smith is `hyp.dir.parent()` (the run_dir). `run_worker` wires `ScopedRead::with_shared(output_dir, shared_dir)` — the smith reads its own prior output + the run dir. That matches the initial dispatch's `shared_dir = run_dir`.

Modify `crates/research/src/orchestrator/mod.rs`: add `pub mod hypothesis;`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test orchestrator build_smith_spec dispatch_hypothesis_smiths redispatch_smith 2>&1 | tail -15`
Expected: PASS — 5 new tests green.

Hygiene:
```
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
cargo test -p megaresearcher-research
```
Expected: clippy exit 0; all green. Verify fixture byte-identical: `diff -q agents/hypothesis-smith.md crates/research/tests/fixtures/agents/hypothesis-smith.md` (no output).

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/hypothesis.rs crates/research/src/orchestrator/mod.rs crates/research/tests/orchestrator.rs crates/research/tests/fixtures/agents/hypothesis-smith.md
git commit -m "feat(rs): Phase 4b Task 3 — hypothesis-smith dispatch + revision

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: redteam.rs — Phase 4 critique loop

**Files:**
- Create: `crates/research/src/orchestrator/redteam.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` (add `pub mod redteam;`)
- Create: `crates/research/tests/fixtures/agents/red-team.md` (byte-identical copy of repo-root `agents/red-team.md`)
- Test: `crates/research/tests/orchestrator.rs` (append redteam tests)

**Interfaces:**
- Consumes: `dispatch::{build_prompt, dispatch_wave, WorkerSpec}`, `gate::{verify_wave, GateStatus}`, `verdict::{parse_redteam_verdict, RedTeamVerdict}`, `hypothesis::{Hypothesis, redispatch_smith_revision}`, `state::swarm_state::SwarmState`, `preflight::{add_escalation}`, `OrchestratorError`, `claurst_api::LlmProvider`.
- Produces:
  - `pub struct RedTeamResult { pub survivors: Vec<Hypothesis>, pub redteam_dirs: Vec<PathBuf>, pub killed: Vec<String> }`
  - `pub fn build_redteam_prompt(spec_text: &str, hypothesis_output: &str, gap_finder_output: &str, output_dir: &Path) -> WorkerSpec` (name `red-team-<N>-r<round>`, role `red-team`)
  - `pub async fn run_redteam_loop(run_dir: &Path, spec_text: &str, hypotheses: Vec<Hypothesis>, agents_dir: &Path, provider: Arc<dyn LlmProvider>, default_model: &str, max_parallel: u32, swarm: &mut SwarmState) -> Result<RedTeamResult, OrchestratorError>`

**Loop semantics (deterministic, documented):**
For each `hyp` in `hypotheses` (in order), with `revision_count` starting at 0:
- round = revision_count + 1; `rt_dir = run_dir/red-team-<N>-r<round>`.
- dispatch one red-team worker to `rt_dir`; verify_wave. If Escalated → `add_escalation(swarm, hyp.name, "red-team missing artifacts", round)`, push `hyp.name` to killed, continue to next hypothesis.
- read `rt_dir/output.md`; `parse_redteam_verdict`. Push `rt_dir` to `redteam_dirs`.
  - `Approve` → push `hyp` to survivors; break inner loop.
  - `Kill` → `add_escalation(swarm, hyp.name, "red-team KILL (irrecoverable)", round)`, push name to killed; break.
  - `None` (unparseable) → `add_escalation(swarm, hyp.name, "red-team produced no parseable verdict", round)`, push name to killed; break.
  - `Reject { revision }` → if `revision_count >= 3` → `add_escalation(swarm, hyp.name, "exceeded 3 red-team revisions", round)`, push name to killed; break. Else `revision_count += 1`; `swarm.retry_counts.insert(hyp.name, revision_count)`; call `redispatch_smith_revision(hyp, spec_text, <rt_dir/output.md>, agents_dir, provider.clone(), default_model)` (overwrites `hypothesis-smith-<N>/output.md`); continue inner loop (next round dispatches red-team again).

`max_parallel` is passed through but the loop dispatches red-team workers one hypothesis at a time (sequential per hypothesis; across hypotheses it's also sequential since we process `hypotheses` in order). This keeps the shared `FakeProvider` call order deterministic. The `max_parallel` arg is accepted for signature consistency with the other phase functions and used for the single-worker dispatch (value 1 is fine; pass `max_parallel.max(1)`).

- [ ] **Step 1: Write the failing tests**

Append to `crates/research/tests/orchestrator.rs`:

```rust
use megaresearcher_research::orchestrator::redteam::{
    build_redteam_prompt, run_redteam_loop, RedTeamResult,
};
use megaresearcher_research::state::swarm_state::SwarmState;

// Helper: scripted turns that write output.md with a given verdict line plus
// the standard manifest + verification, then EndTurn.
fn redteam_turns(verdict_line: &str) -> Vec<Vec<StreamEvent>> {
    let output = format!("# Red-team critique\n\n1. **Verdict** — {verdict_line}\n\n2. Discussion.\n");
    vec![
        write_turn("output.md", &output),
        write_turn("manifest.yaml", "role: red-team\nverdict: APPROVE\nrevision_round: 1\n"),
        write_turn("verification.md", "# Verification\n\nok"),
        final_turn("Done."),
    ]
}

#[test]
fn build_redteam_prompt_inlines_hypothesis_and_finder_output() {
    let tmp = tempdir().unwrap();
    let spec = build_redteam_prompt(
        "SPEC TEXT",
        "HYPOTHESIS OUTPUT BODY",
        "GAP-FINDER OUTPUT BODY",
        &tmp.path().join("red-team-1-r1"),
    );
    assert_eq!(spec.name, "red-team-1-r1");
    assert_eq!(spec.role, "red-team");
    assert!(spec.prompt.contains("SPEC TEXT"));
    assert!(spec.prompt.contains("HYPOTHESIS OUTPUT BODY"));
    assert!(spec.prompt.contains("GAP-FINDER OUTPUT BODY"));
}

#[tokio::test]
async fn redteam_loop_approves_on_first_round() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    // 1 hypothesis, 1 red-team round APPROVE -> 4 turns.
    let fake = Arc::new(FakeProvider::new("fake", redteam_turns("APPROVE")));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_redteam_loop(&run_dir, "SPEC", hyps, &fixture_agents_dir(), provider, "fake-model", 1, &mut swarm)
        .await
        .unwrap();
    assert_eq!(res.survivors.len(), 1);
    assert!(res.killed.is_empty());
    assert_eq!(res.redteam_dirs.len(), 1);
    assert!(res.redteam_dirs[0].ends_with("red-team-1-r1"));
    assert!(swarm.escalations.is_empty());
    assert_eq!(swarm.retry_counts.get("hypothesis-smith-1"), None);
}

#[tokio::test]
async fn redteam_loop_revises_then_approves() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    // Round 1: REJECT (revision-1) -> revise smith (4 turns) + round 2 red-team APPROVE (4 turns).
    // Plus the initial round-1 red-team (4 turns) = 12 turns total.
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = redteam_turns("REJECT (revision-1)");
        // revision smith: 4 artifact turns (writes a revised output.md).
        t.extend(run_turns(1));
        // round-2 red-team: APPROVE.
        t.extend(redteam_turns("APPROVE"));
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_redteam_loop(&run_dir, "SPEC", hyps, &fixture_agents_dir(), provider, "fake-model", 1, &mut swarm)
        .await
        .unwrap();
    assert_eq!(res.survivors.len(), 1);
    assert!(res.killed.is_empty());
    assert_eq!(res.redteam_dirs.len(), 2);
    assert!(res.redteam_dirs[0].ends_with("red-team-1-r1"));
    assert!(res.redteam_dirs[1].ends_with("red-team-1-r2"));
    assert_eq!(swarm.retry_counts.get("hypothesis-smith-1"), Some(&1));
    assert!(swarm.escalations.is_empty());
}

#[tokio::test]
async fn redteam_loop_kills_on_kill_verdict() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    let fake = Arc::new(FakeProvider::new("fake", redteam_turns("KILL (irrecoverable)")));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_redteam_loop(&run_dir, "SPEC", hyps, &fixture_agents_dir(), provider, "fake-model", 1, &mut swarm)
        .await
        .unwrap();
    assert!(res.survivors.is_empty());
    assert_eq!(res.killed, vec!["hypothesis-smith-1".to_string()]);
    assert_eq!(res.redteam_dirs.len(), 1);
    assert_eq!(swarm.escalations.len(), 1);
    assert_eq!(swarm.escalations[0].worker, "hypothesis-smith-1");
    assert!(swarm.escalations[0].reason.contains("KILL"));
}

#[tokio::test]
async fn redteam_loop_escalates_after_three_revisions() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    // Rounds 1-3: REJECT (revision-1/2/3) each -> revise smith; round 4 would be
    // revision_count==3 -> escalate WITHOUT dispatching a 4th red-team.
    // Turn sequence: r1-redteam(4) + revise(4) + r2-redteam(4) + revise(4) +
    // r3-redteam(4) + revise(4) = 24 turns. (No r4 red-team: cap reached at
    // revision_count==3 after the 3rd REJECT.)
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = Vec::new();
        for n in 1..=3 {
            t.extend(redteam_turns(&format!("REJECT (revision-{n})")));
            t.extend(run_turns(1)); // revision smith
        }
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_redteam_loop(&run_dir, "SPEC", hyps, &fixture_agents_dir(), provider, "fake-model", 1, &mut swarm)
        .await
        .unwrap();
    assert!(res.survivors.is_empty());
    assert_eq!(res.killed, vec!["hypothesis-smith-1".to_string()]);
    assert_eq!(res.redteam_dirs.len(), 3); // r1, r2, r3
    assert_eq!(swarm.retry_counts.get("hypothesis-smith-1"), Some(&3));
    assert_eq!(swarm.escalations.len(), 1);
    assert!(swarm.escalations[0].reason.contains("exceeded 3 red-team revisions"));
}

// Shared fixture builder for the loop tests: one hypothesis in a run dir.
fn redteam_loop_fixture(n: usize) -> (SwarmState, Vec<Hypothesis>, PathBuf) {
    let tmp = tempdir().unwrap();
    let run_dir = tmp.path().join("run");
    let finder_dir = run_dir.join("gap-finder-1");
    fs::create_dir_all(&finder_dir).unwrap();
    fs::write(finder_dir.join("output.md"), "FINDER BODY").unwrap();
    let hyps: Vec<Hypothesis> = (1..=n)
        .map(|i| {
            let dir = run_dir.join(format!("hypothesis-smith-{i}"));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("output.md"), format!("HYPOTHESIS {i}")).unwrap();
            Hypothesis {
                name: format!("hypothesis-smith-{i}"),
                dir,
                gap: gap("gap-1", "gap-finder-1", &finder_dir, "The gap."),
            }
        })
        .collect();
    let swarm = SwarmState {
        run_id: "rid".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
        novelty_target: "hypothesis".into(),
        max_parallel: 1,
        phases: vec![],
        escalations: vec![],
        retry_counts: HashMap::new(),
    };
    (swarm, hyps, run_dir)
}
```

The test file already imports `HashMap`? Check: the 4a test file's top `use` block. If not, add `use std::collections::HashMap;` in this appended block (mid-file `use` is fine). The implementer verifies and adds it if missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p megaresearcher-research --test orchestrator build_redteam_prompt redteam_loop 2>&1 | tail -15`
Expected: FAIL — `unresolved import ...::redteam`.

- [ ] **Step 3: Create the fixture + write implementation**

Create the byte-identical fixture:
```bash
cp agents/red-team.md crates/research/tests/fixtures/agents/red-team.md
diff -q agents/red-team.md crates/research/tests/fixtures/agents/red-team.md && echo "byte-identical"
```
(Note: `red-team.md` contains "honest" verbatim from v0 — this is a byte-identical copy, NOT implementer-produced text; the ban does not apply.)

Create `crates/research/src/orchestrator/redteam.rs`:

```rust
//! Phase 4 red-team critique loop. For each hypothesis, dispatch a red-team
//! worker, parse its verdict, and either advance the hypothesis (APPROVE),
//! revise it (REJECT, up to 3 revisions), or kill it (KILL / cap / unparseable).
//! Killed/intractable hypotheses land in `swarm.escalations` for the audit
//! trail; survivors advance to Phase 5.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};
use crate::orchestrator::gate::{verify_wave, GateStatus};
use crate::orchestrator::hypothesis::{redispatch_smith_revision, Hypothesis};
use crate::orchestrator::verdict::{parse_redteam_verdict, RedTeamVerdict};
use crate::orchestrator::OrchestratorError;
use crate::state::swarm_state::SwarmState;

/// Cap on red-team revisions before a hypothesis is escalated.
pub const REVISION_CAP: u32 = 3;

/// The outcome of the Phase 4 loop.
#[derive(Debug, Clone)]
pub struct RedTeamResult {
    pub survivors: Vec<Hypothesis>,
    pub redteam_dirs: Vec<PathBuf>,
    pub killed: Vec<String>,
}

/// Build the red-team worker spec for one critique round.
pub fn build_redteam_prompt(
    spec_text: &str,
    hypothesis_output: &str,
    gap_finder_output: &str,
    output_dir: &Path,
) -> WorkerSpec {
    let prior: [(&str, &str); 2] = [
        ("Hypothesis under critique", hypothesis_output),
        ("Gap-finder output for the targeted gap", gap_finder_output),
    ];
    WorkerSpec {
        name: output_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "red-team".to_string()),
        role: "red-team".to_string(),
        output_dir: output_dir.to_path_buf(),
        shared_dir: output_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        prompt: build_prompt(spec_text, &prior, None, output_dir),
    }
}

/// Run the Phase 4 loop over `hypotheses`. Mutates `swarm.retry_counts` (per-
/// hypothesis revision count) and `swarm.escalations` (killed hypotheses).
pub async fn run_redteam_loop(
    run_dir: &Path,
    spec_text: &str,
    hypotheses: Vec<Hypothesis>,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    swarm: &mut SwarmState,
) -> Result<RedTeamResult, OrchestratorError> {
    let mut survivors = Vec::new();
    let mut killed = Vec::new();
    let mut redteam_dirs = Vec::new();

    for hyp in hypotheses.into_iter() {
        let n = hyp
            .name
            .trim_start_matches("hypothesis-smith-")
            .parse::<u32>()
            .unwrap_or(0);
        let finder_output = fs::read_to_string(hyp.gap.finder_dir.join("output.md"))
            .unwrap_or_else(|_| "(no output.md)".to_string());
        let hypothesis_output =
            fs::read_to_string(hyp.dir.join("output.md")).unwrap_or_else(|_| "(no output.md)".to_string());

        let mut revision_count: u32 = 0;
        loop {
            let round = revision_count + 1;
            let rt_dir = run_dir.join(format!("red-team-{n}-r{round}"));
            fs::create_dir_all(&rt_dir).ok();
            let spec = build_redteam_prompt(spec_text, &hypothesis_output, &finder_output, &rt_dir);
            let outcomes = dispatch_wave(
                vec![spec.clone()],
                agents_dir,
                provider.clone(),
                default_model,
                max_parallel.max(1),
            )
            .await?;
            let gates =
                verify_wave(outcomes, std::slice::from_ref(&spec), agents_dir, provider.clone(), default_model)
                    .await?;
            if gates[0].status == GateStatus::Escalated {
                crate::orchestrator::preflight::add_escalation(
                    swarm,
                    &hyp.name,
                    "red-team missing artifacts after retry",
                    round,
                );
                killed.push(hyp.name.clone());
                redteam_dirs.push(rt_dir);
                break;
            }

            let verdict = parse_redteam_verdict_file(&rt_dir.join("output.md"))?;
            redteam_dirs.push(rt_dir.clone());
            match verdict {
                Some(RedTeamVerdict::Approve) => {
                    survivors.push(hyp);
                    break;
                }
                Some(RedTeamVerdict::Kill) => {
                    crate::orchestrator::preflight::add_escalation(
                        swarm,
                        &hyp.name,
                        "red-team KILL (irrecoverable)",
                        round,
                    );
                    killed.push(hyp.name.clone());
                    break;
                }
                Some(RedTeamVerdict::Reject { revision: _ }) => {
                    if revision_count >= REVISION_CAP {
                        crate::orchestrator::preflight::add_escalation(
                            swarm,
                            &hyp.name,
                            "exceeded 3 red-team revisions",
                            round,
                        );
                        killed.push(hyp.name.clone());
                        break;
                    }
                    revision_count += 1;
                    swarm.retry_counts.insert(hyp.name.clone(), revision_count);
                    let critique = fs::read_to_string(rt_dir.join("output.md"))
                        .unwrap_or_else(|_| "(no output.md)".to_string());
                    redispatch_smith_revision(&hyp, spec_text, &critique, agents_dir, provider.clone(), default_model)
                        .await?;
                    // Loop: next round dispatches red-team again on the revised hypothesis.
                    continue;
                }
                None => {
                    crate::orchestrator::preflight::add_escalation(
                        swarm,
                        &hyp.name,
                        "red-team produced no parseable verdict",
                        round,
                    );
                    killed.push(hyp.name.clone());
                    break;
                }
            }
        }
    }

    Ok(RedTeamResult {
        survivors,
        redteam_dirs,
        killed,
    })
}

fn parse_redteam_verdict_file(path: &Path) -> Result<Option<RedTeamVerdict>, OrchestratorError> {
    let text = fs::read_to_string(path)?;
    Ok(parse_redteam_verdict(&text))
}
```

Wait — `parse_redteam_verdict_file` already exists in `verdict.rs` (Task 2) returning `io::Result<Option<RedTeamVerdict>>`. Reuse it instead of redefining. Replace the local `parse_redteam_verdict_file` call with `crate::orchestrator::verdict::parse_redteam_verdict_file(&rt_dir.join("output.md")).map_err(OrchestratorError::Io)?` and delete the local helper. The implementer MUST do this (DRY — reuse Task 2's function). The corrected loop body uses:

```rust
            let verdict =
                crate::orchestrator::verdict::parse_redteam_verdict_file(&rt_dir.join("output.md"))
                    .map_err(OrchestratorError::Io)?;
            redteam_dirs.push(rt_dir.clone());
```

and the `parse_redteam_verdict_file` local fn + its `use` are removed. Also `parse_redteam_verdict` import becomes unused if only the file variant is used — keep `use ...verdict::{parse_redteam_verdict_file, RedTeamVerdict};` and drop `parse_redteam_verdict`. The implementer adjusts imports accordingly so clippy is clean.

Modify `crates/research/src/orchestrator/mod.rs`: add `pub mod redteam;`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test orchestrator build_redteam_prompt redteam_loop 2>&1 | tail -20`
Expected: PASS — 5 new tests green.

Hygiene:
```
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
cargo test -p megaresearcher-research
```
Expected: clippy exit 0; all green. Fixture byte-identical: `diff -q agents/red-team.md crates/research/tests/fixtures/agents/red-team.md` (no output).

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/redteam.rs crates/research/src/orchestrator/mod.rs crates/research/tests/orchestrator.rs crates/research/tests/fixtures/agents/red-team.md
git commit -m "feat(rs): Phase 4b Task 4 — red-team critique loop (cap 3 revisions)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: evaldesign.rs — Phase 5 eval-designer fan-out

**Files:**
- Create: `crates/research/src/orchestrator/evaldesign.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` (add `pub mod evaldesign;`)
- Create: `crates/research/tests/fixtures/agents/eval-designer.md` (byte-identical copy of repo-root `agents/eval-designer.md`)
- Test: `crates/research/tests/orchestrator.rs` (append evaldesign tests)

**Interfaces:**
- Consumes: `dispatch::{build_prompt, dispatch_wave, WorkerSpec}`, `gate::{verify_wave, GateStatus}`, `hypothesis::Hypothesis`, `state::swarm_state::SwarmState`, `preflight::add_escalation`, `OrchestratorError`, `claurst_api::LlmProvider`, `serde`/`serde_yml`.
- Produces:
  - `pub struct EvalDesignResult { pub eval_dirs: Vec<PathBuf>, pub phase_workers: Vec<(String, String)> }`
  - `pub fn build_evaldesigner_prompt(spec_text: &str, hypothesis_output: &str, output_dir: &Path) -> WorkerSpec`
  - `pub fn parse_intractable(manifest_yaml: &str) -> bool`
  - `pub async fn run_eval_designers(run_dir: &Path, spec_text: &str, survivors: &[Hypothesis], agents_dir: &Path, provider: Arc<dyn LlmProvider>, default_model: &str, max_parallel: u32, swarm: &mut SwarmState) -> Result<EvalDesignResult, OrchestratorError>`

**Behavior:**
- One eval-designer per survivor, `output_dir = run_dir/eval-designer-<N>` (N = the survivor's hypothesis number). Prompt inlines the hypothesis `output.md` as `"Hypothesis to test"`. dispatch_wave → verify_wave. On Escalated → `Err(OrchestratorError::Escalated([name]))` (halts — a missing-artifact eval-designer is a worker failure, like smiths).
- After gate Passed, read `eval-designer-<N>/manifest.yaml`; `parse_intractable` reads `flagged_intractable: true`. If intractable → `add_escalation(swarm, hyp.name, "eval-designer flagged intractable compute", 0)` and phase_worker = `(name, "intractable")`. The eval dir is STILL included in `eval_dirs` (the synthesist reads the intractable eval-designer output for the audit trail). Otherwise phase_worker = `(name, "passed")`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/research/tests/orchestrator.rs`:

```rust
use megaresearcher_research::orchestrator::evaldesign::{
    build_evaldesigner_prompt, parse_intractable, run_eval_designers, EvalDesignResult,
};

// Scripted turns for an eval-designer that writes a manifest with a given
// flagged_intractable value.
fn evaldesign_turns(intractable: bool) -> Vec<Vec<StreamEvent>> {
    let manifest = format!(
        "role: eval-designer\nhypothesis: h\ndatasets_count: 1\nbaselines_count: 3\nfalsification_experiments_count: 2\nflagged_intractable: {}\n",
        intractable
    );
    vec![
        write_turn("output.md", "# Eval plan\n\nThe experiment design."),
        write_turn("manifest.yaml", &manifest),
        write_turn("verification.md", "# Verification\n\nok"),
        final_turn("Done."),
    ]
}

#[test]
fn build_evaldesigner_prompt_inlines_hypothesis() {
    let tmp = tempdir().unwrap();
    let spec = build_evaldesigner_prompt("SPEC TEXT", "HYP BODY", &tmp.path().join("eval-designer-1"));
    assert_eq!(spec.name, "eval-designer-1");
    assert_eq!(spec.role, "eval-designer");
    assert!(spec.prompt.contains("SPEC TEXT"));
    assert!(spec.prompt.contains("HYP BODY"));
}

#[test]
fn parse_intractable_reads_flag() {
    assert!(!parse_intractable("role: eval-designer\nflagged_intractable: false\n"));
    assert!(parse_intractable("role: eval-designer\nflagged_intractable: true\n"));
    // Missing field defaults to false.
    assert!(!parse_intractable("role: eval-designer\n"));
}

#[tokio::test]
async fn eval_designers_one_per_survivor_and_gates() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(2);
    let survivors: Vec<Hypothesis> = hyps;
    // 2 eval-designers x 4 turns = 8 turns; neither intractable.
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = evaldesign_turns(false);
        t.extend(evaldesign_turns(false));
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_eval_designers(&run_dir, "SPEC", &survivors, &fixture_agents_dir(), provider, "fake-model", 1, &mut swarm)
        .await
        .unwrap();
    assert_eq!(res.eval_dirs.len(), 2);
    assert!(res.eval_dirs[0].ends_with("eval-designer-1"));
    assert!(res.eval_dirs[1].ends_with("eval-designer-2"));
    assert_eq!(
        res.phase_workers,
        vec![
            ("hypothesis-smith-1".to_string(), "passed".to_string()),
            ("hypothesis-smith-2".to_string(), "passed".to_string()),
        ]
    );
    assert!(swarm.escalations.is_empty());
}

#[tokio::test]
async fn eval_designer_intractable_is_escalated_but_kept_in_dirs() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    let survivors: Vec<Hypothesis> = hyps;
    let fake = Arc::new(FakeProvider::new("fake", evaldesign_turns(true)));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_eval_designers(&run_dir, "SPEC", &survivors, &fixture_agents_dir(), provider, "fake-model", 1, &mut swarm)
        .await
        .unwrap();
    assert_eq!(res.eval_dirs.len(), 1); // still included for the audit trail
    assert_eq!(res.phase_workers, vec![("hypothesis-smith-1".to_string(), "intractable".to_string())]);
    assert_eq!(swarm.escalations.len(), 1);
    assert_eq!(swarm.escalations[0].worker, "hypothesis-smith-1");
    assert!(swarm.escalations[0].reason.contains("intractable"));
}

#[tokio::test]
async fn eval_designers_halt_on_gate_escalation() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    let survivors: Vec<Hypothesis> = hyps;
    let turns: Vec<Vec<StreamEvent>> = vec![final_turn("nothing written")];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let err = run_eval_designers(&run_dir, "SPEC", &survivors, &fixture_agents_dir(), provider, "fake-model", 1, &mut swarm)
        .expect_err("should escalate");
    match err {
        OrchestratorError::Escalated(names) => assert_eq!(names, vec!["hypothesis-smith-1".to_string()]),
        other => panic!("expected Escalated, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p megaresearcher-research --test orchestrator build_evaldesigner_prompt parse_intractable eval_designers 2>&1 | tail -15`
Expected: FAIL — `unresolved import ...::evaldesign`.

- [ ] **Step 3: Create the fixture + write implementation**

Create the byte-identical fixture:
```bash
cp agents/eval-designer.md crates/research/tests/fixtures/agents/eval-designer.md
diff -q agents/eval-designer.md crates/research/tests/fixtures/agents/eval-designer.md && echo "byte-identical"
```
(`eval-designer.md` contains "Honest." verbatim from v0 — byte-identical copy, ban N/A.)

Create `crates/research/src/orchestrator/evaldesign.rs`:

```rust
//! Phase 5 eval-designer fan-out: one eval-designer per surviving hypothesis.
//! If a designer flags intractable compute, the hypothesis is recorded in
//! `swarm.escalations` (the audit trail) but its eval-designer output is still
//! passed to the synthesist.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;
use serde::Deserialize;

use crate::orchestrator::dispatch::{build_prompt, dispatch_wave, WorkerSpec};
use crate::orchestrator::gate::{verify_wave, GateStatus};
use crate::orchestrator::hypothesis::Hypothesis;
use crate::orchestrator::preflight::add_escalation;
use crate::orchestrator::OrchestratorError;
use crate::state::swarm_state::SwarmState;

/// The Phase 5 outcome: the eval-designer dirs (for synthesis) and the
/// per-hypothesis phase-worker status (`passed` / `intractable`).
#[derive(Debug, Clone)]
pub struct EvalDesignResult {
    pub eval_dirs: Vec<PathBuf>,
    pub phase_workers: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct EvalDesignerManifest {
    #[serde(default)]
    flagged_intractable: bool,
}

/// Build the eval-designer worker spec for one survivor.
pub fn build_evaldesigner_prompt(
    spec_text: &str,
    hypothesis_output: &str,
    output_dir: &Path,
) -> WorkerSpec {
    let prior: [(&str, &str); 1] = [("Hypothesis to test", hypothesis_output)];
    WorkerSpec {
        name: output_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "eval-designer".to_string()),
        role: "eval-designer".to_string(),
        output_dir: output_dir.to_path_buf(),
        shared_dir: output_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        prompt: build_prompt(spec_text, &prior, None, output_dir),
    }
}

/// Read `flagged_intractable` from an eval-designer manifest. Missing field or
/// unparseable manifest -> false (never treat a malformed manifest as intractable).
pub fn parse_intractable(manifest_yaml: &str) -> bool {
    serde_yml::from_str::<EvalDesignerManifest>(manifest_yaml)
        .map(|m| m.flagged_intractable)
        .unwrap_or(false)
}

/// Phase 5: one eval-designer per survivor. Mutates `swarm.escalations` for
/// intractable designs. Returns the eval dirs (all of them, including
/// intractable ones, for the synthesis audit trail) and phase-worker statuses.
pub async fn run_eval_designers(
    run_dir: &Path,
    spec_text: &str,
    survivors: &[Hypothesis],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    swarm: &mut SwarmState,
) -> Result<EvalDesignResult, OrchestratorError> {
    let specs: Vec<WorkerSpec> = survivors
        .iter()
        .map(|hyp| {
            let n = hyp
                .name
                .trim_start_matches("hypothesis-smith-")
                .parse::<u32>()
                .unwrap_or(0);
            let output_dir = run_dir.join(format!("eval-designer-{n}"));
            fs::create_dir_all(&output_dir).ok();
            let hypothesis_output =
                fs::read_to_string(hyp.dir.join("output.md")).unwrap_or_else(|_| "(no output.md)".to_string());
            build_evaldesigner_prompt(spec_text, &hypothesis_output, &output_dir)
        })
        .collect();
    let outcomes = dispatch_wave(
        specs.clone(),
        agents_dir,
        provider.clone(),
        default_model,
        max_parallel,
    )
    .await?;
    let gates = verify_wave(outcomes, &specs, agents_dir, provider, default_model).await?;

    let mut eval_dirs = Vec::with_capacity(specs.len());
    let mut phase_workers = Vec::with_capacity(specs.len());
    for (spec, gate) in specs.into_iter().zip(gates.iter()) {
        if gate.status == GateStatus::Escalated {
            return Err(OrchestratorError::Escalated(vec![spec.name]));
        }
        // The hypothesis name is hypothesis-smith-<N>; the eval-designer is
        // eval-designer-<N>. Recover N from the eval-designer dir name.
        let n = spec
            .name
            .trim_start_matches("eval-designer-")
            .parse::<u32>()
            .unwrap_or(0);
        let hyp_name = format!("hypothesis-smith-{n}");
        let manifest =
            fs::read_to_string(spec.output_dir.join("manifest.yaml")).unwrap_or_default();
        if parse_intractable(&manifest) {
            add_escalation(swarm, &hyp_name, "eval-designer flagged intractable compute", 0);
            phase_workers.push((hyp_name, "intractable".to_string()));
        } else {
            phase_workers.push((hyp_name, "passed".to_string()));
        }
        eval_dirs.push(spec.output_dir);
    }
    Ok(EvalDesignResult { eval_dirs, phase_workers })
}
```

Modify `crates/research/src/orchestrator/mod.rs`: add `pub mod evaldesign;`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test orchestrator build_evaldesigner_prompt parse_intractable eval_designers 2>&1 | tail -15`
Expected: PASS — 5 new tests green.

Hygiene:
```
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
cargo test -p megaresearcher-research
```
Expected: clippy exit 0; all green. Fixture byte-identical: `diff -q agents/eval-designer.md crates/research/tests/fixtures/agents/eval-designer.md` (no output).

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/evaldesign.rs crates/research/src/orchestrator/mod.rs crates/research/tests/orchestrator.rs crates/research/tests/fixtures/agents/eval-designer.md
git commit -m "feat(rs): Phase 4b Task 5 — eval-designer fan-out + intractable flagging

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: wire Phases 3/4/5 into execute() + extend run_synthesist

**Files:**
- Modify: `crates/research/src/orchestrator/mod.rs` (extend `execute()` with Phases 3/4/5; import new modules)
- Modify: `crates/research/src/orchestrator/synthesize.rs` (extend `run_synthesist` signature: +`smith_dirs`, `redteam_dirs`, `eval_dirs`)
- Create: `crates/research/tests/fixtures/specs/hypothesis-spec.md`
- Create: `crates/research/tests/fixtures/plans/hypothesis-plan.md`
- Test: `crates/research/tests/orchestrator.rs` (update the 4a `run_synthesist` direct test to the new signature; add a minimal hypothesis-target execute test)

**Interfaces (new `run_synthesist` signature):**
```rust
#[allow(clippy::too_many_arguments)]
pub async fn run_synthesist(
    run_dir: &Path,
    spec_text: &str,
    plan_text: &str,
    scout_dirs: &[PathBuf],
    gap_dirs: &[PathBuf],
    smith_dirs: &[PathBuf],
    redteam_dirs: &[PathBuf],
    eval_dirs: &[PathBuf],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
) -> Result<(WorkerSpec, WorkerOutcome), OrchestratorError>
```
The `prior` vec appends, after the existing "Scout"/"Gap-finder" sections: `"Hypothesis-smith <n>"` (from each `smith_dir/output.md`), `"Red-team <n>-r<round>"` (from each `redteam_dir/output.md`), `"Eval-designer <n>"` (from each `eval_dir/output.md`). The section labels use each dir's file name so the synthesist sees `hypothesis-smith-1`, `red-team-1-r1`, `eval-designer-1`, etc. Empty slices add no sections (gap-finding runs unchanged).

**execute() modification** — between the existing Phase 2 block (after `consolidate_gaps(&run_dir, &gap_dirs)?; write_swarm(&swarm, &run_dir)?;`) and the existing Phase 6 block, insert:

```rust
        // Phases 3/4/5 — hypothesis-target path (skipped for gap-finding runs).
        let (smith_dirs, redteam_dirs, eval_dirs): (Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>) =
            if !target.skips_critique_phases() {
                // Phase 3 — hypothesis-smith (one per gap).
                let gaps = collect_gaps(&run_dir, &gap_dirs)?;
                set_phase(&mut swarm, "hypothesis-smith", "running", vec![]);
                write_swarm(&swarm, &run_dir)?;
                let hypotheses = dispatch_hypothesis_smiths(
                    &run_dir,
                    &spec_text,
                    &gaps,
                    &self.config.agents_dir,
                    self.provider.clone(),
                    &self.config.default_model,
                    self.config.max_parallel,
                )
                .await?;
                let smith_dirs: Vec<PathBuf> = hypotheses.iter().map(|h| h.dir.clone()).collect();
                let smith_workers: Vec<(String, String)> =
                    hypotheses.iter().map(|h| (h.name.clone(), "passed".to_string())).collect();
                set_phase(&mut swarm, "hypothesis-smith", "complete", smith_workers);
                write_swarm(&swarm, &run_dir)?;

                // Phase 4 — red-team critique loop.
                set_phase(&mut swarm, "red-team", "running", vec![]);
                write_swarm(&swarm, &run_dir)?;
                let rt = run_redteam_loop(
                    &run_dir,
                    &spec_text,
                    hypotheses,
                    &self.config.agents_dir,
                    self.provider.clone(),
                    &self.config.default_model,
                    self.config.max_parallel,
                    &mut swarm,
                )
                .await?;
                let redteam_dirs = rt.redteam_dirs;
                let rt_workers: Vec<(String, String)> = rt
                    .survivors
                    .iter()
                    .map(|h| (h.name.clone(), "approved".to_string()))
                    .chain(rt.killed.iter().map(|n| (n.clone(), "killed".to_string())))
                    .collect();
                set_phase(&mut swarm, "red-team", "complete", rt_workers);
                write_swarm(&swarm, &run_dir)?;
                if rt.survivors.is_empty() {
                    let killed = rt.killed.clone();
                    return Err(OrchestratorError::Escalated(killed));
                }

                // Phase 5 — eval-designer (one per survivor).
                set_phase(&mut swarm, "eval-designer", "running", vec![]);
                write_swarm(&swarm, &run_dir)?;
                let ed = run_eval_designers(
                    &run_dir,
                    &spec_text,
                    &rt.survivors,
                    &self.config.agents_dir,
                    self.provider.clone(),
                    &self.config.default_model,
                    self.config.max_parallel,
                    &mut swarm,
                )
                .await?;
                let eval_dirs = ed.eval_dirs;
                set_phase(&mut swarm, "eval-designer", "complete", ed.phase_workers);
                write_swarm(&swarm, &run_dir)?;
                (smith_dirs, redteam_dirs, eval_dirs)
            } else {
                (vec![], vec![], vec![])
            };
```

Then the existing Phase 6 block changes the `run_synthesist` call to pass the three new dir slices:

```rust
        let (synth_spec, synth_outcome) = run_synthesist(
            &run_dir,
            &spec_text,
            &plan_text,
            &scout_dirs,
            &gap_dirs,
            &smith_dirs,
            &redteam_dirs,
            &eval_dirs,
            &self.config.agents_dir,
            self.provider.clone(),
            &self.config.default_model,
            self.config.max_parallel,
        )
        .await?;
```

And the `mod.rs` `use` block adds:
```rust
use crate::orchestrator::evaldesign::run_eval_designers;
use crate::orchestrator::gaps::collect_gaps;
use crate::orchestrator::hypothesis::dispatch_hypothesis_smiths;
use crate::orchestrator::redteam::run_redteam_loop;
```

- [ ] **Step 1: Write the failing tests**

First, update the existing 4a test `run_synthesist_inlines_all_outputs_and_writes_artifacts` (around line 623) to the new 12-arg signature — insert three empty `&[]` slices after `&[g1]`:

```rust
    let (spec, outcome) = run_synthesist(
        &run_dir,
        "SPEC",
        "PLAN",
        &[s1],
        &[g1],
        &[],
        &[],
        &[],
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
    )
    .await
    .unwrap();
```
(The rest of that test is unchanged — it asserts the prompt contains SCOUT1/GAP1/SPEC/PLAN and the artifacts exist. Empty smith/redteam/eval slices add no sections, so behavior is identical.)

Then append a new minimal hypothesis-target execute test + fixtures.

Create `crates/research/tests/fixtures/specs/hypothesis-spec.md`:
```markdown
# Hypothesis-target spec fixture

A tiny spec body for the hypothesis-target integration test. The novelty target
lives in the PLAN frontmatter, not here; the spec is just text passed to workers.
```

Create `crates/research/tests/fixtures/plans/hypothesis-plan.md`:
```markdown
---
novelty_target: hypothesis
---

## Phase 1 — literature-scout dispatches

### Sub-topic A

Survey prior art on technique X applied to modality A.

## Phase 2 — gap-finder dispatches

### Slice 1

Cross-read the scout bibliographies and identify gaps.
```

Add a `fixture_hypothesis_spec_path` + `fixture_hypothesis_plan_path` helper (mirror the 4a `fixture_spec_path`/`fixture_plan_path`) and append the minimal execute test:

```rust
fn fixture_hypothesis_spec_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/specs/hypothesis-spec.md")
}
fn fixture_hypothesis_plan_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plans/hypothesis-plan.md")
}

// Scripted turns for a gap-finder that writes a manifest with a structured
// gaps: list of `count` gaps.
fn gapfinder_turns_with_gaps(count: usize) -> Vec<Vec<StreamEvent>> {
    let mut gaps_yaml = String::from("gaps:\n");
    for i in 1..=count {
        gaps_yaml.push_str(&format!(
            "  - id: gap-{i}\n    statement: Gap {i} statement.\n    type: contradiction\n"
        ));
    }
    let manifest = format!("role: gap-finder\ngaps_count: {count}\n{gaps_yaml}");
    vec![
        write_turn("output.md", "# Gaps\n\nSome gaps identified."),
        write_turn("manifest.yaml", &manifest),
        write_turn("verification.md", "# Verification\n\nok"),
        final_turn("Done."),
    ]
}

#[tokio::test]
async fn execute_runs_full_hypothesis_path_minimal() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    let spec_path = research_base.join("specs/hypothesis-spec.md");
    fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
    fs::copy(fixture_hypothesis_spec_path(), &spec_path).unwrap();

    // 1 scout (4) + 1 gap-finder with 1 gap (4) + 1 smith (4) + 1 red-team APPROVE (4)
    // + 1 eval-designer not-intractable (4) + 1 synthesist (4) = 24 turns.
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = run_turns(1);            // scout-1
        t.extend(gapfinder_turns_with_gaps(1)); // gap-finder-1 (writes 1 gap)
        t.extend(run_turns(1));             // hypothesis-smith-1
        t.extend(redteam_turns("APPROVE")); // red-team-1-r1
        t.extend(evaldesign_turns(false));  // eval-designer-1
        t.extend(run_turns(1));             // synthesist
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
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
    let out = orch
        .execute(&spec_path, &fixture_hypothesis_plan_path(), "2026-06-27-0400-d1e2f3")
        .await
        .unwrap();
    let run_dir = out.run_dir.clone();
    let by_name: std::collections::HashMap<String, String> =
        out.phase_statuses.into_iter().collect();
    assert_eq!(by_name["literature-scout"], "complete");
    assert_eq!(by_name["gap-finder"], "complete");
    assert_eq!(by_name["hypothesis-smith"], "complete");
    assert_eq!(by_name["red-team"], "complete");
    assert_eq!(by_name["eval-designer"], "complete");
    assert_eq!(by_name["synthesist"], "complete");
    assert!(out.escalations.is_empty());
    // Run-root output.md + symlink resolve.
    assert!(run_dir.join("output.md").exists());
    let link = research_base.join("specs/hypothesis-spec-latest.md");
    assert!(link.exists());
    assert_eq!(
        fs::read_to_string(&link).unwrap(),
        fs::read_to_string(run_dir.join("output.md")).unwrap()
    );
    // The smith + red-team + eval-designer dirs exist.
    assert!(run_dir.join("hypothesis-smith-1").join("output.md").exists());
    assert!(run_dir.join("red-team-1-r1").join("output.md").exists());
    assert!(run_dir.join("eval-designer-1").join("output.md").exists());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p megaresearcher-research --test orchestrator execute_runs_full_hypothesis_path_minimal 2>&1 | tail -25`
Expected: FAIL — `run_synthesist` signature mismatch (4a test + execute both reference the old 9-arg form until synthesize.rs is updated; the minimal test fails because Phases 3/4/5 aren't wired).

- [ ] **Step 3: Apply the implementation edits**

(a) Edit `crates/research/src/orchestrator/synthesize.rs` `run_synthesist` — replace the signature + the `prior`-building body. New signature (shown above). New prior-building body (replaces the existing scout+gap loop tail; keep the Plan + scout + gap blocks, then append smith/redteam/eval):

```rust
    let mut prior: Vec<(String, String)> = Vec::new();
    prior.push(("Plan".to_string(), plan_text.to_string()));
    for d in scout_dirs {
        let name = dir_name(d);
        let body = read_output(d);
        prior.push((format!("Scout {name}"), body));
    }
    for d in gap_dirs {
        let name = dir_name(d);
        let body = read_output(d);
        prior.push((format!("Gap-finder {name}"), body));
    }
    for d in smith_dirs {
        let name = dir_name(d);
        let body = read_output(d);
        prior.push((format!("Hypothesis-smith {name}"), body));
    }
    for d in redteam_dirs {
        let name = dir_name(d);
        let body = read_output(d);
        prior.push((format!("Red-team {name}"), body));
    }
    for d in eval_dirs {
        let name = dir_name(d);
        let body = read_output(d);
        prior.push((format!("Eval-designer {name}"), body));
    }
    let prior_refs: Vec<(&str, &str)> = prior
        .iter()
        .map(|(l, c)| (l.as_str(), c.as_str()))
        .collect();
```

and add two private helpers at the bottom of `synthesize.rs` (replacing the inline `file_name`/`read_to_string` closures used in 4a):

```rust
fn dir_name(d: &Path) -> String {
    d.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn read_output(d: &Path) -> String {
    fs::read_to_string(d.join("output.md")).unwrap_or_else(|_| "(no output.md)".to_string())
}
```

Keep `#[allow(clippy::too_many_arguments)]` on the now-12-arg `run_synthesist`. The rest of `run_synthesist` (output_dir, spec, dispatch_wave, extract single outcome) is unchanged.

(b) Edit `crates/research/src/orchestrator/mod.rs`:
- Add the four new `use` lines (shown above).
- Insert the Phases 3/4/5 block (shown above) between the Phase 2 block and the Phase 6 block.
- Change the `run_synthesist(...)` call to the 12-arg form (shown above).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test orchestrator 2>&1 | tail -15`
Expected: PASS — the updated 4a `run_synthesist` direct test + the new minimal hypothesis execute test + all prior 4a/4b tests green.

Hygiene:
```
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
cargo test -p megaresearcher-research
```
Expected: clippy exit 0; all green. Verify the 4a gap-finding integration test STILL passes (it exercises the `else` branch — empty smith/redteam/eval slices): `cargo test -p megaresearcher-research --test orchestrator full_gap_finding_integration_test`.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/mod.rs crates/research/src/orchestrator/synthesize.rs crates/research/tests/orchestrator.rs crates/research/tests/fixtures/specs/hypothesis-spec.md crates/research/tests/fixtures/plans/hypothesis-plan.md
git commit -m "feat(rs): Phase 4b Task 6 — wire Phases 3/4/5 into execute() + extend synthesist

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: full hypothesis-target integration test (§13.4-equivalent)

**Files:**
- Test: `crates/research/tests/orchestrator.rs` (append the full integration test)

This task adds no production code — it proves the full determinism model holds end-to-end for a hypothesis-target run with a revision loop and a kill.

**Test scenario** (2 gaps → 2 smiths; smith-1 APPROVE first round; smith-2 REJECT then revise then APPROVE; both survivors get eval-designers (not intractable); synthesist finalizes). Then a SEPARATE test exercises the KILL path (smith-1 KILL → run halts with `Err(Escalated)`).

- [ ] **Step 1: Write the test**

Append to `crates/research/tests/orchestrator.rs`:

```rust
#[tokio::test]
async fn full_hypothesis_integration_test_with_revision_loop() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    let spec_path = research_base.join("specs/hypothesis-spec.md");
    fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
    fs::copy(fixture_hypothesis_spec_path(), &spec_path).unwrap();

    // 2 scouts (plan has 2 scout assignments? No — hypothesis-plan has 1 scout.
    // Use the 1-scout plan: 1 scout + 1 gap-finder(2 gaps) + 2 smiths +
    // red-team-1 APPROVE + red-team-2 REJECT + smith-2 revision + red-team-2-r2 APPROVE
    // + 2 eval-designers + 1 synthesist.
    //
    // Turn sequence (max_parallel=1, deterministic call order):
    //  1 scout-1            (4)
    //  2 gap-finder-1       (4) — manifest with 2 gaps
    //  3 hypothesis-smith-1  (4)
    //  4 hypothesis-smith-2 (4)   [Phase 3: both smiths in one wave]
    //  5 red-team-1-r1      (4) — APPROVE
    //  6 red-team-2-r1      (4) — REJECT (revision-1)
    //  7 hypothesis-smith-2 revision (4)
    //  8 red-team-2-r2      (4) — APPROVE
    //  9 eval-designer-1    (4)  [for survivor hypothesis-smith-1]
    // 10 eval-designer-2    (4)  [for survivor hypothesis-smith-2]
    // 11 synthesist         (4)
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = run_turns(1);                  // scout-1
        t.extend(gapfinder_turns_with_gaps(2));    // gap-finder-1 (2 gaps)
        t.extend(run_turns(2));                    // smith-1, smith-2
        t.extend(redteam_turns("APPROVE"));        // red-team-1-r1
        t.extend(redteam_turns("REJECT (revision-1)")); // red-team-2-r1
        t.extend(run_turns(1));                    // smith-2 revision
        t.extend(redteam_turns("APPROVE"));        // red-team-2-r2
        t.extend(evaldesign_turns(false));        // eval-designer-1
        t.extend(evaldesign_turns(false));        // eval-designer-2
        t.extend(run_turns(1));                    // synthesist
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
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
    let out = orch
        .execute(&spec_path, &fixture_hypothesis_plan_path(), "2026-06-27-0430-a2b3c4")
        .await
        .unwrap();
    let run_dir = out.run_dir.clone();
    let by_name: std::collections::HashMap<String, String> =
        out.phase_statuses.into_iter().collect();
    // All six phases complete; no critique phase skipped.
    assert_eq!(by_name["literature-scout"], "complete");
    assert_eq!(by_name["gap-finder"], "complete");
    assert_eq!(by_name["hypothesis-smith"], "complete");
    assert_eq!(by_name["red-team"], "complete");
    assert_eq!(by_name["eval-designer"], "complete");
    assert_eq!(by_name["synthesist"], "complete");
    assert!(out.escalations.is_empty());

    // Phase 4 produced two red-team rounds for hypothesis 2 (r1 reject, r2 approve)
    // and one for hypothesis 1.
    assert!(run_dir.join("red-team-1-r1").join("output.md").exists());
    assert!(run_dir.join("red-team-2-r1").join("output.md").exists());
    assert!(run_dir.join("red-team-2-r2").join("output.md").exists());
    // Two eval-designer dirs (one per survivor).
    assert!(run_dir.join("eval-designer-1").join("output.md").exists());
    assert!(run_dir.join("eval-designer-2").join("output.md").exists());
    // Run-root output.md + symlink resolve to it.
    assert!(run_dir.join("output.md").exists());
    let link = research_base.join("specs/hypothesis-spec-latest.md");
    assert!(link.exists());
    assert_eq!(
        fs::read_to_string(&link).unwrap(),
        fs::read_to_string(run_dir.join("output.md")).unwrap()
    );
    // The synthesist prompt inlined smith + red-team + eval-designer outputs.
    // (Inspect the synthesist dir's prompt via the artifact? The prompt is not
    // persisted; instead assert the synthesist wrote its three artifacts.)
    assert!(run_dir.join("synthesist").join("output.md").exists());
    assert!(run_dir.join("synthesist").join("manifest.yaml").exists());
}

#[tokio::test]
async fn full_hypothesis_integration_test_kill_halts_run() {
    let tmp = tempdir().unwrap();
    let research_base = tmp.path().join("research");
    fs::create_dir_all(&research_base).unwrap();
    let spec_path = research_base.join("specs/hypothesis-spec.md");
    fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
    fs::copy(fixture_hypothesis_spec_path(), &spec_path).unwrap();

    // 1 scout + 1 gap-finder(1 gap) + 1 smith + 1 red-team KILL.
    // The red-team KILL escalates hypothesis-smith-1; all survivors gone ->
    // execute returns Err(Escalated(["hypothesis-smith-1"])).
    let turns: Vec<Vec<StreamEvent>> = {
        let mut t = run_turns(1);                  // scout-1
        t.extend(gapfinder_turns_with_gaps(1));    // gap-finder-1
        t.extend(run_turns(1));                    // hypothesis-smith-1
        t.extend(redteam_turns("KILL (irrecoverable)")); // red-team-1-r1
        t
    };
    let fake = Arc::new(FakeProvider::new("fake", turns));
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
    let err = orch
        .execute(&spec_path, &fixture_hypothesis_plan_path(), "2026-06-27-0500-f4e5d6")
        .await
        .expect_err("kill should escalate and halt");
    match err {
        OrchestratorError::Escalated(names) => {
            assert_eq!(names, vec!["hypothesis-smith-1".to_string()]);
        }
        other => panic!("expected Escalated, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test orchestrator full_hypothesis_integration_test 2>&1 | tail -20`
Expected: PASS — both new tests green. If a turn-count mismatch surfaces (a desync where a worker escalates unexpectedly), the most likely cause is the FakeProvider running out of scripted turns (the `.or_else(last)` clamp then re-emits the last turn, which can cause a worker to write the wrong artifact). The fix is to recount the turns against the dispatch order documented in the test comment — every worker must consume exactly 4 turns in the order listed.

Hygiene:
```
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
cargo test -p megaresearcher-research
```
Expected: clippy exit 0; all green.

- [ ] **Step 3: Commit**

```bash
git add crates/research/tests/orchestrator.rs
git commit -m "test(rs): Phase 4b Task 7 — full hypothesis-target integration tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: green sweep + final whole-branch review

**Files:** none modified (sweep only); the controller runs the checks and dispatches the final review.

- [ ] **Step 1: fmt**

Run: `cargo fmt --all --check`
Expected: clean. If drift, run `cargo fmt --all` and commit as a separate `style(rs): Phase 4b sweep — cargo fmt` commit.

- [ ] **Step 2: clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: exit 0, zero warnings.

- [ ] **Step 3: test**

Run: `cargo test --workspace`
Expected: 0 failures. Record the total (Phase 4a baseline 1444 + the new 4b tests: ~5 gaps + 7 verdict + 5 hypothesis + 5 redteam + 5 evaldesign + 1 updated + 3 integration ≈ 31 new; assert the number grew by roughly that and 0 failures).

- [ ] **Step 4: v0 untouched**

Run: `git diff --stat <4a-tip>..HEAD -- lib/ tests/test_*.py skills/ agents/ .claude-plugin/ commands/ hooks/ mcp/ tools/ml-intern`
Expected: empty (the v0 port-reference is untouched; the only `agents/` reads were byte-identical `cp` into fixtures, which live under `crates/research/tests/fixtures/agents/`, not repo-root `agents/`).

- [ ] **Step 5: structure check**

Confirm:
- `crates/research/src/orchestrator/` now has `consolidate.rs dispatch.rs dispatch_plan.rs evaldesign.rs gaps.rs gate.rs hypothesis.rs mod.rs preflight.rs redteam.rs synthesize.rs verdict.rs`.
- `crates/research/tests/fixtures/agents/` has `literature-scout.md gap-finder.md synthesist.md hypothesis-smith.md red-team.md eval-designer.md`.
- `lib.rs` declares `pub mod orchestrator;` (unchanged from 4a; no new `pub mod` at crate root).

- [ ] **Step 6: ledger**

Append to `.superpowers/sdd/progress.md`:
```
=== Phase 4b: ALL TASKS COMPLETE ===
```
plus per-task lines as each task completes during execution (the controller appends these during SDD, not in this plan step).

- [ ] **Step 7: final whole-branch review**

The controller runs `scripts/review-package <4a-tip> HEAD` and dispatches the final code reviewer on the most capable available model (opus), pointing it at the review package + this plan + the ledger's Phase 4b section. The reviewer verifies: the determinism model holds end-to-end (shared `Arc<dyn LlmProvider>` + global `call_index` under `max_parallel=1` across the Phase 4 revision loop); `swarm.retry_counts`/`swarm.escalations` mutate correctly; the relative-symlink `finalize_run` still resolves; edit scope (only `crates/research/` + fixtures; v0 untouched); no crate-root re-exports; no api-crate changes; banned-phrase scan clean across implementer-produced text (fixtures exempt as byte-identical copies). Triage any deferred Minors.

- [ ] **Step 8: push (after review Ready)**

After the final review returns Ready (fix any Critical/Important, re-review), the controller pushes Phase 4b to `origin/main` and records `=== Phase 4b PUSHED to origin/main (<4a-tip>..<4b-tip>) ===` in the ledger, continuing the per-phase push cadence.

---

## Self-Review

**1. Spec coverage.** Every Phase 4b responsibility maps to a task:
- Gap enumeration (deterministic) → Task 1.
- Red-team verdict parsing → Task 2.
- Phase 3 smith dispatch + revision re-dispatch → Task 3.
- Phase 4 critique loop (APPROVE/REJECT/KILL/cap/None) → Task 4.
- Phase 5 eval-designer fan-out + intractable flagging → Task 5.
- execute() wiring + synthesist extension → Task 6.
- Full integration tests (revision loop + kill) → Task 7.
- Sweep + final review → Task 8.
v0's Phase 3/4/5 behaviors (one smith per gap, cap-3 revision loop, KILL escalation, eval-designer per survivor, intractable surfacing) are all represented. The gap-finding path (4a) is preserved unchanged via the `else { (vec![], vec![], vec![]) }` branch + empty synthesist slices (Task 6's updated 4a test + the 4a integration test re-run confirm this).

**2. Placeholder scan.** Every step has complete code. The two inline "implementer MUST use this corrected form" notes (Task 3: drop `specs_iter_index`; Task 4: reuse `verdict::parse_redteam_verdict_file` not a local redef) give the exact final code — no ambiguity. No "TBD"/"TODO"/"add error handling"/"similar to Task N".

**3. Type consistency.** `Gap`, `Hypothesis`, `RedTeamVerdict`, `RedTeamResult`, `EvalDesignResult` are defined once and referenced consistently. `run_synthesist`'s new 12-arg signature is used identically in the 4a-test update (Task 6) and the execute() call (Task 6). `collect_gaps`/`dispatch_hypothesis_smiths`/`run_redteam_loop`/`run_eval_designers` signatures match their call sites in execute(). `add_escalation` (from 4a preflight) is reused; `swarm.retry_counts: HashMap<String,u32>` (from Phase 2) is the retry-count store. `GateStatus`/`verify_wave`/`dispatch_wave`/`build_prompt` are consumed unchanged from 4a.

**4. Determinism audit.** The FakeProvider turn accounting in Task 7's comment (11 worker-dispatches × 4 turns = 44 for the revision-loop test, plus the documented order) matches the dispatch order forced by `max_parallel=1`. The revision loop's red-team rounds are sequential per hypothesis, so the global `call_index` advances monotonically in the documented order. The kill test (7 worker-dispatches × 4 = 28 turns) halts after the KILL with `Err(Escalated)`, so no further turns are consumed.

**5. Edit-scope audit.** Only `crates/research/` (+ fixtures under `crates/research/tests/fixtures/`) is modified. No `Cargo.toml`/`Cargo.lock` change expected (all deps — `serde`, `serde_yml`, `regex`, `once_cell`, `tokio`, `futures` — are already workspace deps from Phases 2/3). The v0 `agents/*.md` are read-only `cp` sources for byte-identical fixture copies (Tasks 3/4/5). `lib.rs` is NOT touched (the `orchestrator` module was already declared in 4a). No api-crate change.