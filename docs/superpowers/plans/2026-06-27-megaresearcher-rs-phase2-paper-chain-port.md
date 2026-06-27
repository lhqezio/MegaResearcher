# Phase 2 — Paper-Chain + State Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the six pure-logic `lib/paper_chain/*` modules and the run-state module to deterministic Rust inside `crates/research`, as 1:1 behavior ports of the existing Python, with 1:1 Rust translations of the existing `tests/test_*.py`. No LLM is involved.

**Architecture:** Each Python module becomes one Rust file under `crates/research/src/paper_chain/` (flat-file style, matching claurst). `research::state` is new Rust (no Python source) modeled on the orchestrator skill's `swarm-state.yaml` schema + run-id rule. Every filesystem-touching function returns `io::Result<T>` (the Python originals raise `OSError`/`FileNotFoundError`; Rust surfaces those as `Err` so the Phase 4 orchestrator can distinguish "file missing" from "no verdict line"). Integration tests live under `crates/research/tests/`, one file per Python test, mirroring the `tests/test_*.py` layout.

**Tech Stack:** Rust 2021 edition (workspace), `regex`, `once_cell` (static `Regex`es — claurst convention), `serde` + `serde_yml` (swarm-state serde), `chrono` (run-id UTC stamp), `getrandom` (run-id hex), `tempfile` (dev-dep for tests).

## Global Constraints

Copied verbatim from the design spec (`docs/superpowers/specs/2026-06-26-megaresearcher-rs-design.md`) and the project rules. Every task's requirements implicitly include this section.

- **1:1 ports with 1:1 tests.** "Pure-logic unit tests (port the existing `tests/test_*.py`): `verdict`, `regression`, `protocol_parser`, `scaffold`, `finalize`, `preflight`, `swarm-state` serde, run-id generation. Deterministic; 1:1 Rust ports with 1:1 tests." (spec §12) — behavior and assertions match the Python originals; do not "improve" the logic.
- **No LLM in Phase 2.** "Port pure logic + state … No LLM yet." (spec §13.2) — deterministic ports only.
- **Each build phase ends green.** "Each build phase ends with the test suite green so the harness is never broken mid-port." (spec §12) — after every task, `cargo check -p megaresearcher-research` and the task's tests pass.
- **Old plugin files are the port reference and must not be touched.** "The old plugin scaffolding (`.claude-plugin/`, `commands/`, `hooks/`, `mcp/`, `lib/`, `skills/`, `tools/ml-intern`) is kept during the build as the port reference." (spec §2) — Phase 2 edits ONLY `crates/research/` and the workspace `Cargo.toml`. Do NOT modify `lib/`, `tests/`, `skills/`, or any other v0 file.
- **GPL-3.0.** License is already set (Phase 1); Phase 2 changes no license file.
- **No git worktrees.** Hard rule. Work directly on branch `main`. Before dispatching any implementer, confirm `git branch --show-current` returns `main` and put the literal branch name `main` in the dispatch prompt so the subagent never runs `git switch`.
- **Banned phrases/words** (global rule, applies to doc-comments and commit messages too): never use "load-bearing", "this is doing a lot of work", "real" (emphatic), "honest/honestly/to be honest". Genuine technical terms ("real-time", "real number") are fine.
- **Port the parsing functions only, not the `_main` CLIs.** The Python modules each have a `_main` CLI invoked by the v0 skill via `python3 -m lib.paper_chain.X`. In Rust the Phase 4 orchestrator calls the library functions directly; there is no binary in Phase 2. Do NOT port `_main`.

## File Structure

```
crates/research/
├── Cargo.toml                          # Task 1: add chrono/getrandom/once_cell/regex/serde/serde_yml deps + tempfile dev-dep
└── src/
    ├── lib.rs                           # Task 1: pub mod paper_chain; pub mod state; + CRATE_NAME
    ├── paper_chain.rs                   # Task 1: empty doc-comment; Tasks 2–7 each add one `pub mod X;` line
    ├── paper_chain/
    │   ├── verdict.rs                   # Task 2
    │   ├── regression.rs                # Task 3
    │   ├── protocol_parser.rs           # Task 4
    │   ├── scaffold.rs                   # Task 5
    │   ├── finalize.rs                   # Task 6
    │   └── preflight.rs                  # Task 7
    ├── state.rs                         # Task 1: empty doc-comment; Task 8 adds three `pub mod` lines
    └── state/
        ├── run_id.rs                     # Task 8
        ├── swarm_state.rs                # Task 8
        └── run_tree.rs                   # Task 8
crates/research/tests/
├── smoke.rs                             # existing (Task 1) — keep untouched
├── fixtures/protocols/
│   ├── specs_protocol.md                # Task 4 — verbatim copy of tests/fixtures/protocols/specs_protocol.md
│   └── malformed.md                     # Task 4 — verbatim copy of tests/fixtures/protocols/malformed.md
├── verdict.rs                           # Task 2 — 1:1 port of tests/test_verdict_parser.py
├── regression.rs                        # Task 3 — 1:1 port of tests/test_regression.py
├── protocol_parser.rs                  # Task 4 — 1:1 port of tests/test_protocol_parser.py
├── scaffold.rs                          # Task 5 — 1:1 port of tests/test_scaffold.py
├── finalize.rs                          # Task 6 — 1:1 port of tests/test_finalize.py
├── preflight.rs                         # Task 7 — 1:1 port of tests/test_preflight.py
└── state.rs                             # Task 8 — run_id + swarm_state + run_tree tests
Cargo.toml (workspace root)              # Task 1: add `serde_yml = "0.0.13"` to [workspace.dependencies]
```

Module style is flat-file (claurst convention: 196 flat `src/*.rs` vs 5 `mod.rs`). `paper_chain.rs` is the module file; its submodules live in `paper_chain/*.rs`. Same for `state.rs` / `state/*.rs`.

All functions that touch the filesystem return `io::Result<T>`. Tests call `.unwrap()` on them (the Python tests always feed existing files; `.unwrap()` corresponds to Python's implicit "file exists in tests"). `parse_verdict` returns `io::Result<Option<String>>` — `Ok(None)` means "file read, no verdict line"; `Err` means "file unreadable/missing" (the Python original raises `FileNotFoundError`, and spec §11 requires the orchestrator to escalate on a missing review file rather than silently treating it as "no verdict").

---

### Task 1: Scaffold the research module tree + dependencies

**Files:**
- Modify: `Cargo.toml` (workspace root — add `serde_yml` to `[workspace.dependencies]`)
- Modify: `crates/research/Cargo.toml` (add the port deps + `tempfile` dev-dep)
- Modify: `crates/research/src/lib.rs` (declare the two top-level modules)
- Create: `crates/research/src/paper_chain.rs` (empty module file — doc comment only)
- Create: `crates/research/src/state.rs` (empty module file — doc comment only)

**Interfaces:**
- Consumes: the Phase 1 stub (`crates/research` with `claurst-core` dep + `CRATE_NAME` + `smoke.rs`).
- Produces: `megaresearcher_research::paper_chain` and `megaresearcher_research::state` as empty `pub` modules; the workspace `serde_yml` dependency; the research crate's port dependencies. Tasks 2–8 fill the submodules.

- [ ] **Step 1: Add `serde_yml` to the workspace dependencies**

In `/Users/ggix/MegaResearcher/Cargo.toml`, inside the `[workspace.dependencies]` section, add this line (place it next to `serde`/`serde_json`; exact position is not significant to cargo):

```toml
serde_yml = "0.0.13"
```

- [ ] **Step 2: Rewrite `crates/research/Cargo.toml`**

Replace the entire file with:

```toml
[package]
name = "megaresearcher-research"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
claurst-core = { workspace = true }
chrono = { workspace = true }
getrandom = { workspace = true }
once_cell = { workspace = true }
regex = { workspace = true }
serde = { workspace = true }
serde_yml = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }

[lib]
name = "megaresearcher_research"
path = "src/lib.rs"
```

- [ ] **Step 3: Rewrite `crates/research/src/lib.rs`**

Replace the entire file with:

```rust
//! MegaResearcher v1 — the research layer.
//!
//! Phase 2 ports the pure-logic paper-chain modules and the run-state module
//! from the v0 Python plugin (`lib/paper_chain/*` plus the orchestrator skill's
//! `swarm-state.yaml` schema) to deterministic Rust. No LLM is involved yet —
//! these are 1:1 ports with 1:1 tests. Later phases add the orchestrator, the
//! worker primitive, the front-half guided phases, the HTML export, and the
//! doom-loop discipline trait.

pub mod paper_chain;
pub mod state;

pub const CRATE_NAME: &str = "megaresearcher-research";
```

- [ ] **Step 4: Create the two empty module files**

Create `crates/research/src/paper_chain.rs`:

```rust
//! Paper-chain pure-logic modules. 1:1 ports of `lib/paper_chain/*`.
//! Submodules are declared by Tasks 2–7 as they are ported.
```

Create `crates/research/src/state.rs`:

```rust
//! Run-state: `swarm-state.yaml` serde, run-id generation, run-tree management.
//! Submodules are declared by Task 8.
```

- [ ] **Step 5: Verify it compiles and the smoke test still passes**

```bash
cd /Users/ggix/MegaResearcher
cargo check -p megaresearcher-research 2>&1 | tail -5
cargo test -p megaresearcher-research --test smoke 2>&1 | tail -5
```
Expected: `cargo check` finishes clean; the smoke test prints `test research_crate_links ... ok` and `test result: ok. 1 passed`.

- [ ] **Step 6: Commit**

```bash
cd /Users/ggix/MegaResearcher
git add Cargo.toml Cargo.lock crates/research/Cargo.toml crates/research/src/lib.rs crates/research/src/paper_chain.rs crates/research/src/state.rs
git commit -m "feat(rs): scaffold research module tree + paper_chain/state deps"
```

---

### Task 2: Port `paper_chain::verdict`

**Files:**
- Modify: `crates/research/src/paper_chain.rs` (add `pub mod verdict;`)
- Create: `crates/research/src/paper_chain/verdict.rs`
- Test: `crates/research/tests/verdict.rs`

**Interfaces:**
- Consumes: Task 1's empty `paper_chain` module.
- Produces: `pub fn parse_verdict(review_path: &Path) -> io::Result<Option<String>>` — returns `Ok(Some("APPROVE"|"REVISE"|"KILL"))` on the first `^VERDICT: (APPROVE|REVISE|KILL)$` line, `Ok(None)` if no such line, `Err` if the file is unreadable. Also `pub static VALID_VERDICTS: &[&str]`.

- [ ] **Step 1: Write the failing test**

Create `crates/research/tests/verdict.rs` (1:1 port of `tests/test_verdict_parser.py`):

```rust
//! 1:1 port of tests/test_verdict_parser.py.

use std::fs;
use std::path::PathBuf;

use megaresearcher_research::paper_chain::verdict::parse_verdict;

fn write_temp_md(text: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("review.md");
    fs::write(&path, text).unwrap();
    (dir, path)
}

#[test]
fn test_approve() {
    let (_d, p) = write_temp_md("# Review\n\nSummary...\n\nVERDICT: APPROVE\n");
    assert_eq!(parse_verdict(&p).unwrap(), Some("APPROVE".to_string()));
}

#[test]
fn test_revise() {
    let (_d, p) = write_temp_md("# Review\nVERDICT: REVISE\n");
    assert_eq!(parse_verdict(&p).unwrap(), Some("REVISE".to_string()));
}

#[test]
fn test_kill() {
    let (_d, p) = write_temp_md("# Review\nVERDICT: KILL\n");
    assert_eq!(parse_verdict(&p).unwrap(), Some("KILL".to_string()));
}

#[test]
fn test_verdict_must_be_last_nonblank_line() {
    // Verdict line not at end → still parsed (we scan, not strict-last)
    let (_d, p) = write_temp_md("# Review\nVERDICT: APPROVE\n\nSome trailing notes.\n");
    assert_eq!(parse_verdict(&p).unwrap(), Some("APPROVE".to_string()));
}

#[test]
fn test_no_verdict() {
    let (_d, p) = write_temp_md("# Review\n\nNo verdict here.\n");
    assert_eq!(parse_verdict(&p).unwrap(), None);
}

#[test]
fn test_malformed_verdict() {
    let (_d, p) = write_temp_md("# Review\nVERDICT: MAYBE\n");
    assert_eq!(parse_verdict(&p).unwrap(), None);
}

#[test]
fn test_case_sensitivity() {
    let (_d, p) = write_temp_md("# Review\nverdict: approve\n");
    assert_eq!(parse_verdict(&p).unwrap(), None);
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test verdict 2>&1 | tail -15
```
Expected: compile error — `unresolved module verdict` (the `paper_chain` module does not yet declare it).

- [ ] **Step 3: Implement `verdict.rs`**

Add this line to `crates/research/src/paper_chain.rs` (after the doc comment):

```rust
pub mod verdict;
```

Create `crates/research/src/paper_chain/verdict.rs`:

```rust
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test verdict 2>&1 | tail -10
```
Expected: `test result: ok. 7 passed; 0 failed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/ggix/MegaResearcher
git add crates/research/src/paper_chain.rs crates/research/src/paper_chain/verdict.rs crates/research/tests/verdict.rs
git commit -m "feat(rs): port paper_chain::verdict to Rust (1:1 of lib/paper_chain/verdict.py)"
```

---

### Task 3: Port `paper_chain::regression`

**Files:**
- Modify: `crates/research/src/paper_chain.rs` (add `pub mod regression;`)
- Create: `crates/research/src/paper_chain/regression.rs`
- Test: `crates/research/tests/regression.rs`

**Interfaces:**
- Consumes: Task 1's `paper_chain` module.
- Produces:
  - `pub fn extract_weaknesses(review_path: &Path) -> io::Result<Vec<String>>` — returns the `## Weaknesses` section's bullet lines with the leading `- ` dropped.
  - `pub fn detect_regression(v1_path: &Path, v2_path: &Path) -> io::Result<(bool, usize, usize)>` — `(flagged, closed_count, new_count)`; `flagged = new >= closed && new > 0`.

- [ ] **Step 1: Write the failing test**

Create `crates/research/tests/regression.rs` (1:1 port of `tests/test_regression.py`):

```rust
//! 1:1 port of tests/test_regression.py.

use std::fs;
use std::path::PathBuf;

use megaresearcher_research::paper_chain::regression::{detect_regression, extract_weaknesses};

const REVIEW_V1: &str = "\
# Review v1
## Strengths
- Good idea.

## Weaknesses
- W1: Insufficient ablation coverage.
- W2: Citation for claim X does not resolve.
- W3: Method section unclear about step 3.

## Suggested Revisions
...

VERDICT: REVISE
";

const REVIEW_V2_ALL_CLOSED_NEW_PROBLEMS: &str = "\
# Review v2
## Strengths
- Improved.

## Weaknesses
- W4: New citation Y is also unresolved.
- W5: Ablation table has off-by-one error.
- W6: Discussion contradicts results.
- W7: New related-work section misattributes finding.

VERDICT: REVISE
";

const REVIEW_V2_PARTIAL_CLOSE: &str = "\
# Review v2
## Weaknesses
- W2: Citation still unresolved (carried over).
- W4: New small typo in abstract.

VERDICT: REVISE
";

fn write_temp_md(text: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("review.md");
    fs::write(&path, text).unwrap();
    (dir, path)
}

#[test]
fn test_extract_weaknesses_basic() {
    let (_d, p) = write_temp_md(REVIEW_V1);
    let ws = extract_weaknesses(&p).unwrap();
    assert_eq!(ws.len(), 3, "Expected 3 weaknesses, got {:?}", ws);
    assert!(ws[0].starts_with("W1:"), "{}", ws[0]);
}

#[test]
fn test_extract_weaknesses_empty_section() {
    let (_d, p) = write_temp_md("## Weaknesses\n\nVERDICT: APPROVE\n");
    let ws = extract_weaknesses(&p).unwrap();
    assert!(ws.is_empty());
}

#[test]
fn test_regression_fires_when_new_outnumber_closed() {
    let (_d1, v1) = write_temp_md(REVIEW_V1);
    let (_d2, v2) = write_temp_md(REVIEW_V2_ALL_CLOSED_NEW_PROBLEMS);
    let (flagged, closed_count, new_count) = detect_regression(&v1, &v2).unwrap();
    // v1 had 3, v2 has 4 NEW; closed = 3, new = 4 — regression
    assert!(flagged, "Expected regression flag: closed={}, new={}", closed_count, new_count);
    assert_eq!(new_count, 4);
    assert_eq!(closed_count, 3);
}

#[test]
fn test_regression_does_not_fire_on_partial_close() {
    let (_d1, v1) = write_temp_md(REVIEW_V1);
    let (_d2, v2) = write_temp_md(REVIEW_V2_PARTIAL_CLOSE);
    // v1 had 3; v2 has 2 (W2 carried, W4 new). closed=2 (W1, W3), new=1 (W4). No regression.
    let (flagged, closed_count, new_count) = detect_regression(&v1, &v2).unwrap();
    assert!(!flagged);
    assert_eq!(closed_count, 2);
    assert_eq!(new_count, 1);
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test regression 2>&1 | tail -15
```
Expected: compile error — `unresolved module regression`.

- [ ] **Step 3: Implement `regression.rs`**

Add to `crates/research/src/paper_chain.rs`:

```rust
pub mod regression;
```

Create `crates/research/src/paper_chain/regression.rs`:

```rust
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
    // lstrip("- ") strips leading chars in the set {'-', ' '}.
    let body = line.trim_start_matches(|c: char| c == '-' || c == ' ').trim();
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test regression 2>&1 | tail -10
```
Expected: `test result: ok. 4 passed; 0 failed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/ggix/MegaResearcher
git add crates/research/src/paper_chain.rs crates/research/src/paper_chain/regression.rs crates/research/tests/regression.rs
git commit -m "feat(rs): port paper_chain::regression to Rust (1:1 of lib/paper_chain/regression.py)"
```

---

### Task 4: Port `paper_chain::protocol_parser`

**Files:**
- Modify: `crates/research/src/paper_chain.rs` (add `pub mod protocol_parser;`)
- Create: `crates/research/src/paper_chain/protocol_parser.rs`
- Create: `crates/research/tests/fixtures/protocols/specs_protocol.md` (verbatim copy of `tests/fixtures/protocols/specs_protocol.md`)
- Create: `crates/research/tests/fixtures/protocols/malformed.md` (verbatim copy of `tests/fixtures/protocols/malformed.md`)
- Test: `crates/research/tests/protocol_parser.rs`

**Interfaces:**
- Consumes: Task 1's `paper_chain` module.
- Produces:
  - `pub struct Protocol { substrate: Option<String>, sample_size: Option<i64>, seed: Option<i64>, baselines: Vec<String>, metrics: Vec<String>, decision_rules: Vec<DecisionRule> }` with `pub fn empty()` and `pub fn is_empty() -> bool` (true when `substrate.is_none()`, mirroring Python's `result == {}`).
  - `pub struct DecisionRule { raw: String }` (mirrors Python's `{"raw": ...}`).
  - `pub fn parse_protocol(protocol_path: &Path) -> io::Result<Protocol>` — returns `Protocol::empty()` when no `Substrate:` line is found.

- [ ] **Step 1: Copy the fixture files verbatim**

Create `crates/research/tests/fixtures/protocols/specs_protocol.md` with this exact content (byte-identical to `tests/fixtures/protocols/specs_protocol.md`):

```
# Eval design — S1 cross-family writer/reviewer split

## Pre-registered settings

- Substrate: SPECS-Review-Benchmark
- Sample size: 22 perturbations
- Seed: 42
- Baselines: stage-matched same-family 2-stage
- Metric: paired-difference flaw-detection rate, Correctness+Evaluations axes
- Decision rule: F1 lift ≥ 0.05 absolute, single-comparison paired-difference bootstrap, p < 0.13

## Other content the parser must ignore
...
```

Create `crates/research/tests/fixtures/protocols/malformed.md` with this exact content:

```
# Just some random text
No structured fields here.
```

- [ ] **Step 2: Write the failing test**

Create `crates/research/tests/protocol_parser.rs` (1:1 port of `tests/test_protocol_parser.py`):

```rust
//! 1:1 port of tests/test_protocol_parser.py.

use std::path::PathBuf;

use megaresearcher_research::paper_chain::protocol_parser::parse_protocol;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/protocols")
}

#[test]
fn test_parse_specs_protocol() {
    let result = parse_protocol(&fixtures().join("specs_protocol.md")).unwrap();
    assert_eq!(result.substrate.as_deref(), Some("SPECS-Review-Benchmark"));
    assert_eq!(result.sample_size, Some(22));
    assert_eq!(result.seed, Some(42));
}

#[test]
fn test_parse_baselines_list() {
    let result = parse_protocol(&fixtures().join("specs_protocol.md")).unwrap();
    assert!(result.baselines[0].contains("stage-matched same-family 2-stage"));
}

#[test]
fn test_parse_metric() {
    let result = parse_protocol(&fixtures().join("specs_protocol.md")).unwrap();
    assert!(result.metrics.iter().any(|m| m.contains("flaw-detection")), "{:?}", result.metrics);
}

#[test]
fn test_parse_decision_rule() {
    let result = parse_protocol(&fixtures().join("specs_protocol.md")).unwrap();
    assert!(result.decision_rules.len() >= 1);
    assert!(result.decision_rules[0].raw.contains("0.05"));
}

#[test]
fn test_parse_malformed_returns_empty() {
    let result = parse_protocol(&fixtures().join("malformed.md")).unwrap();
    assert!(result.is_empty());
}
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test protocol_parser 2>&1 | tail -15
```
Expected: compile error — `unresolved module protocol_parser`.

- [ ] **Step 4: Implement `protocol_parser.rs`**

Add to `crates/research/src/paper_chain.rs`:

```rust
pub mod protocol_parser;
```

Create `crates/research/src/paper_chain/protocol_parser.rs`:

```rust
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
static SEED_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*[-*]\s*Seed:\s*(\d+)").unwrap());
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
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test protocol_parser 2>&1 | tail -10
```
Expected: `test result: ok. 5 passed; 0 failed`.

- [ ] **Step 6: Commit**

```bash
cd /Users/ggix/MegaResearcher
git add crates/research/src/paper_chain.rs crates/research/src/paper_chain/protocol_parser.rs crates/research/tests/protocol_parser.rs crates/research/tests/fixtures
git commit -m "feat(rs): port paper_chain::protocol_parser to Rust (1:1 of lib/paper_chain/protocol_parser.py)"
```

---

### Task 5: Port `paper_chain::scaffold`

**Files:**
- Modify: `crates/research/src/paper_chain.rs` (add `pub mod scaffold;`)
- Create: `crates/research/src/paper_chain/scaffold.rs`
- Test: `crates/research/tests/scaffold.rs`

**Interfaces:**
- Consumes: Task 1's `paper_chain` module.
- Produces: `pub fn scaffold_paper_dir(run_dir: &Path) -> io::Result<PathBuf>` — creates `<run_dir>/paper/` (idempotent) and an empty `paper/revision-log.jsonl` if absent, returns the `paper/` path.

- [ ] **Step 1: Write the failing test**

Create `crates/research/tests/scaffold.rs` (1:1 port of `tests/test_scaffold.py`):

```rust
//! 1:1 port of tests/test_scaffold.py.

use std::fs;
use std::path::PathBuf;

use megaresearcher_research::paper_chain::scaffold::scaffold_paper_dir;

fn new_run() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

#[test]
fn test_creates_paper_subdir() {
    let run = new_run();
    let paper = scaffold_paper_dir(run.path()).unwrap();
    assert_eq!(paper, run.path().join("paper"));
    assert!(paper.is_dir());
}

#[test]
fn test_creates_revision_log_jsonl() {
    let run = new_run();
    let paper = scaffold_paper_dir(run.path()).unwrap();
    let log = paper.join("revision-log.jsonl");
    assert!(log.exists(), "revision-log.jsonl should exist");
    assert_eq!(fs::read_to_string(&log).unwrap(), "", "revision-log.jsonl should be empty");
}

#[test]
fn test_idempotent() {
    let run = new_run();
    let p1 = scaffold_paper_dir(run.path()).unwrap();
    fs::write(p1.join("draft-v1.md"), "# draft").unwrap();
    let p2 = scaffold_paper_dir(run.path()).unwrap(); // safe to re-run
    assert_eq!(p1, p2);
    assert!(p1.join("draft-v1.md").exists(), "Idempotent scaffold must not destroy existing content");
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test scaffold 2>&1 | tail -15
```
Expected: compile error — `unresolved module scaffold`.

- [ ] **Step 3: Implement `scaffold.rs`**

Add to `crates/research/src/paper_chain.rs`:

```rust
pub mod scaffold;
```

Create `crates/research/src/paper_chain/scaffold.rs`:

```rust
//! Scaffold the `paper/` subdirectory under a swarm run dir.
//! 1:1 port of `lib/paper_chain/scaffold.py`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Create `<run_dir>/paper/` with an empty `revision-log.jsonl`.
///
/// Idempotent: safe to call multiple times; preserves existing files.
/// Returns the `paper/` path. `Err` if the directory cannot be created.
pub fn scaffold_paper_dir(run_dir: &Path) -> io::Result<PathBuf> {
    let paper = run_dir.join("paper");
    fs::create_dir_all(&paper)?;
    let log = paper.join("revision-log.jsonl");
    if !log.exists() {
        fs::write(&log, "")?;
    }
    Ok(paper)
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test scaffold 2>&1 | tail -10
```
Expected: `test result: ok. 3 passed; 0 failed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/ggix/MegaResearcher
git add crates/research/src/paper_chain.rs crates/research/src/paper_chain/scaffold.rs crates/research/tests/scaffold.rs
git commit -m "feat(rs): port paper_chain::scaffold to Rust (1:1 of lib/paper_chain/scaffold.py)"
```

---

### Task 6: Port `paper_chain::finalize`

**Files:**
- Modify: `crates/research/src/paper_chain.rs` (add `pub mod finalize;`)
- Create: `crates/research/src/paper_chain/finalize.rs`
- Test: `crates/research/tests/finalize.rs`

**Interfaces:**
- Consumes: Task 1's `paper_chain` module.
- Produces: `pub fn finalize_paper(paper_dir: &Path, final_verdict: &str) -> io::Result<PathBuf>` — copies the highest-numbered `draft-vN.md` to `paper.md`, writes `paper-history.md` (final verdict + ordered `review-vN.md` contents + `revision-log.jsonl` if non-empty), returns the `paper.md` path. `Err(io::ErrorKind::NotFound)` if no `draft-vN.md` exists.

- [ ] **Step 1: Write the failing test**

Create `crates/research/tests/finalize.rs` (1:1 port of `tests/test_finalize.py`):

```rust
//! 1:1 port of tests/test_finalize.py.

use std::fs;
use std::path::PathBuf;

use megaresearcher_research::paper_chain::finalize::finalize_paper;

/// Build a paper/ dir with draft-v1.md, draft-v2.md (if v2), review-v1.md, log.
fn setup_paper_dir(latest_draft: &str) -> tempfile::TempDir {
    let run = tempfile::tempdir().unwrap();
    let paper = run.path().join("paper");
    fs::create_dir(&paper).unwrap();
    fs::write(paper.join("draft-v1.md"), "# Draft v1\n\nContent.\n").unwrap();
    if latest_draft == "v2" {
        fs::write(paper.join("draft-v2.md"), "# Draft v2\n\nRevised.\n").unwrap();
    }
    fs::write(paper.join("review-v1.md"), "# Review v1\n\nVERDICT: REVISE\n").unwrap();
    fs::write(
        paper.join("revision-log.jsonl"),
        "{\"round\":1,\"review_point_index\":0,\"addressed\":true,\
         \"change_summary\":\"fixed W1\",\"line_range_modified\":[10,15]}\n",
    )
    .unwrap();
    run
}

#[test]
fn test_finalize_with_v1_only() {
    let run = setup_paper_dir("v1");
    let paper = run.path().join("paper");
    let out = finalize_paper(&paper, "APPROVE").unwrap();
    assert_eq!(out, paper.join("paper.md"));
    assert!(out.exists());
    assert!(fs::read_to_string(&out).unwrap().contains("Draft v1"));
    let history = paper.join("paper-history.md");
    assert!(history.exists());
    assert!(fs::read_to_string(&history).unwrap().contains("Review v1"));
}

#[test]
fn test_finalize_with_v2() {
    let run = setup_paper_dir("v2");
    let paper = run.path().join("paper");
    let out = finalize_paper(&paper, "APPROVE").unwrap();
    assert!(fs::read_to_string(&out).unwrap().contains("Draft v2"), "paper.md must point at latest draft");
}

#[test]
fn test_finalize_includes_revision_log_in_history() {
    let run = setup_paper_dir("v2");
    let paper = run.path().join("paper");
    finalize_paper(&paper, "APPROVE").unwrap();
    let history = fs::read_to_string(paper.join("paper-history.md")).unwrap();
    assert!(history.contains("fixed W1"), "revision-log entries must appear in history");
}

#[test]
fn test_finalize_records_final_verdict() {
    let run = setup_paper_dir("v1");
    let paper = run.path().join("paper");
    finalize_paper(&paper, "APPROVE").unwrap();
    let history = fs::read_to_string(paper.join("paper-history.md")).unwrap();
    assert!(history.contains("Final verdict: APPROVE"));
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test finalize 2>&1 | tail -15
```
Expected: compile error — `unresolved module finalize`.

- [ ] **Step 3: Implement `finalize.rs`**

Add to `crates/research/src/paper_chain.rs`:

```rust
pub mod finalize;
```

Create `crates/research/src/paper_chain/finalize.rs`:

```rust
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
    history.push_str(&format!("# Paper history\n\nFinal verdict: {}\n", final_verdict));
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test finalize 2>&1 | tail -10
```
Expected: `test result: ok. 4 passed; 0 failed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/ggix/MegaResearcher
git add crates/research/src/paper_chain.rs crates/research/src/paper_chain/finalize.rs crates/research/tests/finalize.rs
git commit -m "feat(rs): port paper_chain::finalize to Rust (1:1 of lib/paper_chain/finalize.py)"
```

---

### Task 7: Port `paper_chain::preflight`

**Files:**
- Modify: `crates/research/src/paper_chain.rs` (add `pub mod preflight;`)
- Create: `crates/research/src/paper_chain/preflight.rs`
- Test: `crates/research/tests/preflight.rs`

**Interfaces:**
- Consumes: Task 1's `paper_chain` module.
- Produces:
  - `pub fn preflight_check(run_dir: &Path) -> io::Result<(bool, String)>` — `(ok, reason)`; `reason` empty when ok. Checks `output.md` exists, `swarm-state.yaml` exists, `novelty_target` (regex `(?m)^novelty_target:\s*(\S+)\s*$`) equals `hypothesis`, and each `eval-designer-*` subdir has `output.md`. `Err` only if `swarm-state.yaml` exists but is unreadable.
  - `pub fn preflight_check_with_paper(run_dir: &Path, paper_mode: bool) -> io::Result<(bool, String, Vec<String>)>` — adds a non-blocking `VERCEL_TOKEN` warning when `paper_mode` and the env var is unset-or-empty.

- [ ] **Step 1: Write the failing test**

Create `crates/research/tests/preflight.rs` (1:1 port of `tests/test_preflight.py`):

```rust
//! 1:1 port of tests/test_preflight.py.

use std::env;
use std::fs;

use megaresearcher_research::paper_chain::preflight::{preflight_check, preflight_check_with_paper};

/// Create a temporary run dir matching the swarm-state.yaml shape.
fn make_run(novelty_target: Option<&str>, with_output: bool, with_eval_designers: usize) -> tempfile::TempDir {
    let run = tempfile::tempdir().unwrap();
    if with_output {
        fs::write(run.path().join("output.md"), "# Research direction\n").unwrap();
    }
    if let Some(nt) = novelty_target {
        fs::write(
            run.path().join("swarm-state.yaml"),
            format!("novelty_target: {}\n", nt),
        )
        .unwrap();
    }
    for i in 0..with_eval_designers {
        let d = run.path().join(format!("eval-designer-S{}", i + 1));
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("output.md"), format!("# Eval design {}\n", i + 1)).unwrap();
    }
    run
}

#[test]
fn test_happy_path() {
    let run = make_run(Some("hypothesis"), true, 3);
    let (ok, reason) = preflight_check(run.path()).unwrap();
    assert!(ok, "Expected OK, got refusal: {}", reason);
}

#[test]
fn test_missing_output_md() {
    let run = make_run(Some("hypothesis"), false, 3);
    let (ok, reason) = preflight_check(run.path()).unwrap();
    assert!(!ok);
    assert!(reason.contains("output.md"), "Expected reason to name output.md; got: {}", reason);
}

#[test]
fn test_missing_swarm_state() {
    let run = make_run(None, true, 3);
    let (ok, reason) = preflight_check(run.path()).unwrap();
    assert!(!ok);
    assert!(reason.contains("swarm-state"));
}

#[test]
fn test_wrong_novelty_target_gap_finding() {
    let run = make_run(Some("gap-finding"), true, 0);
    let (ok, reason) = preflight_check(run.path()).unwrap();
    assert!(!ok);
    assert!(reason.contains("hypothesis") && reason.contains("gap-finding"));
}

#[test]
fn test_no_eval_designer_outputs() {
    let run = make_run(Some("hypothesis"), true, 0);
    let (ok, reason) = preflight_check(run.path()).unwrap();
    assert!(!ok);
    assert!(reason.contains("eval-designer"));
}

#[test]
fn test_preflight_warns_about_vercel_token_when_paper() {
    // When --paper is set and VERCEL_TOKEN absent, preflight returns ok=true
    // with a non-empty warnings list. The warning does not block.
    let run = make_run(Some("hypothesis"), true, 3);
    let saved = env::var_os("VERCEL_TOKEN");
    env::remove_var("VERCEL_TOKEN");
    let (ok, _reason, warnings) = preflight_check_with_paper(run.path(), true).unwrap();
    assert!(ok);
    assert!(warnings.iter().any(|w| w.contains("VERCEL_TOKEN")));
    if let Some(v) = saved {
        env::set_var("VERCEL_TOKEN", v);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test preflight 2>&1 | tail -15
```
Expected: compile error — `unresolved module preflight`.

- [ ] **Step 3: Implement `preflight.rs`**

Add to `crates/research/src/paper_chain.rs`:

```rust
pub mod preflight;
```

Create `crates/research/src/paper_chain/preflight.rs`:

```rust
//! Pre-flight checks for the paper-drafting chain.
//! 1:1 port of `lib/paper_chain/preflight.py`. The chain runs only when:
//!   1. `output.md` exists at the run root (synthesist produced it)
//!   2. `swarm-state.yaml` exists at the run root
//!   3. the run's `novelty_target` is `hypothesis`
//!   4. each `eval-designer-*` subdir has its own `output.md`

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use regex::Regex;

static NOVELTY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^novelty_target:\s*(\S+)\s*$").unwrap());

/// Return `(ok, reason)`. `reason` is empty when `ok`. `Err` if `swarm-state.yaml`
/// exists but cannot be read.
pub fn preflight_check(run_dir: &Path) -> io::Result<(bool, String)> {
    let output_md = run_dir.join("output.md");
    if !output_md.exists() {
        return Ok((
            false,
            format!(
                "Pre-flight refusal: output.md not found at {}. Re-run /research-execute first to produce the synthesist's output.",
                output_md.display()
            ),
        ));
    }

    let state = run_dir.join("swarm-state.yaml");
    if !state.exists() {
        return Ok((
            false,
            format!("Pre-flight refusal: swarm-state.yaml not found at {}.", state.display()),
        ));
    }

    let text = fs::read_to_string(&state)?;
    let target = match NOVELTY_RE.captures(&text) {
        Some(c) => c.get(1).unwrap().as_str().to_string(),
        None => {
            return Ok((
                false,
                format!("Pre-flight refusal: novelty_target not found in {}.", state.display()),
            ));
        }
    };
    if target != "hypothesis" {
        return Ok((
            false,
            format!(
                "Pre-flight refusal: paper chain only runs on hypothesis-target outputs. This run's novelty_target is {} (expected hypothesis); gap-finding runs lack the eval-designer protocols the paper chain consumes.",
                target
            ),
        ));
    }

    let eval_dirs: Vec<PathBuf> = fs::read_dir(run_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().starts_with("eval-designer-"))
                .unwrap_or(false)
        })
        .collect();
    if eval_dirs.is_empty() {
        return Ok((
            false,
            format!(
                "Pre-flight refusal: no eval-designer-* subdirs in {}. Paper chain requires Phase 5 protocols as input.",
                run_dir.display()
            ),
        ));
    }
    for d in &eval_dirs {
        if !d.join("output.md").exists() {
            return Ok((
                false,
                format!("Pre-flight refusal: eval-designer subdir {} missing output.md.", d.display()),
            ));
        }
    }

    Ok((true, String::new()))
}

/// Extended preflight returning `(ok, reason, warnings)`. When `paper_mode` is
/// true, adds a non-blocking `VERCEL_TOKEN` warning if the env var is unset or
/// empty (mirrors Python's falsy `os.environ.get("VERCEL_TOKEN")`).
pub fn preflight_check_with_paper(
    run_dir: &Path,
    paper_mode: bool,
) -> io::Result<(bool, String, Vec<String>)> {
    let (ok, reason) = preflight_check(run_dir)?;
    let mut warnings: Vec<String> = Vec::new();
    let token_set = env::var_os("VERCEL_TOKEN").map(|v| !v.is_empty()).unwrap_or(false);
    if ok && paper_mode && !token_set {
        warnings.push(
            "VERCEL_TOKEN not set — Phase 6.5 (experimentalist) will fail immediately when it tries to spin up a sandbox. Set the env var before invoking /research-execute --paper if you want experiments."
                .to_string(),
        );
    }
    Ok((ok, reason, warnings))
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test preflight 2>&1 | tail -10
```
Expected: `test result: ok. 6 passed; 0 failed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/ggix/MegaResearcher
git add crates/research/src/paper_chain.rs crates/research/src/paper_chain/preflight.rs crates/research/tests/preflight.rs
git commit -m "feat(rs): port paper_chain::preflight to Rust (1:1 of lib/paper_chain/preflight.py)"
```

---

### Task 8: Port `research::state` (run_id, swarm_state, run_tree)

`research::state` has no Python source — it is new Rust modeled on the orchestrator skill's `swarm-state.yaml` schema and run-id rule. The schema (from `skills/executing-research-plan/SKILL.md`): `run_id`, `spec_path`, `plan_path`, `novelty_target`, `max_parallel`, `phases` (each with `status` and `workers` list with status `pending`), `escalations: []`, `retry_counts: {}`. The run-id rule (same skill): `YYYY-MM-DD-HHMM-<6-char-hex>` (UTC date+time + 6 lowercase hex).

**Files:**
- Modify: `crates/research/src/state.rs` (add the three `pub mod` lines)
- Create: `crates/research/src/state/run_id.rs`
- Create: `crates/research/src/state/swarm_state.rs`
- Create: `crates/research/src/state/run_tree.rs`
- Test: `crates/research/tests/state.rs`

**Interfaces:**
- Consumes: Task 1's `state` module.
- Produces:
  - `pub fn run_id_from_parts(stamp: &str, hex6: &str) -> String` — pure; `format!("{}-{}", stamp, hex6)`.
  - `pub fn generate_run_id() -> io::Result<String>` — `Utc::now().format("%Y-%m-%d-%H%M")` + 3 `getrandom` bytes → 6 lowercase hex.
  - `pub struct SwarmState { run_id, spec_path, plan_path, novelty_target, max_parallel, phases: Vec<Phase>, escalations: Vec<Escalation>, retry_counts: HashMap<String,u32> }` + `Phase`, `Worker`, `Escalation` — all `Serialize, Deserialize, Debug, Clone, PartialEq`. `SwarmState::read(path) -> io::Result<Self>` and `SwarmState::write(&self, path) -> io::Result<()>` via `serde_yml`.
  - `pub fn run_dir(base: &Path, run_id: &str) -> PathBuf` — `base.join("runs").join(run_id)`; pure.
  - `pub fn create_run_tree(base: &Path, run_id: &str) -> io::Result<PathBuf>` — mkdir -p, returns the path.

- [ ] **Step 1: Write the failing test**

Create `crates/research/tests/state.rs`:

```rust
//! run_id, swarm_state serde, and run_tree tests.

use std::collections::HashMap;
use std::path::PathBuf;

use megaresearcher_research::state::run_id::{generate_run_id, run_id_from_parts};
use megaresearcher_research::state::run_tree::{create_run_tree, run_dir};
use megaresearcher_research::state::swarm_state::{Escalation, Phase, SwarmState, Worker};

// ---- run_id -------------------------------------------------------------

#[test]
fn test_run_id_from_parts() {
    assert_eq!(
        run_id_from_parts("2026-06-27-1430", "a1b2c3"),
        "2026-06-27-1430-a1b2c3"
    );
}

#[test]
fn test_generate_run_id_format() {
    let id = generate_run_id().unwrap();
    let parts: Vec<&str> = id.split('-').collect();
    assert_eq!(parts.len(), 5, "run-id must have 5 dash-separated parts: {}", id);
    assert_eq!(parts[0].len(), 4, "year (4 digits): {}", id);
    assert!(parts[0].chars().all(|c| c.is_ascii_digit()), "year digits: {}", id);
    assert_eq!(parts[1].len(), 2, "month: {}", id);
    assert!(parts[1].chars().all(|c| c.is_ascii_digit()), "month digits: {}", id);
    assert_eq!(parts[2].len(), 2, "day: {}", id);
    assert!(parts[2].chars().all(|c| c.is_ascii_digit()), "day digits: {}", id);
    assert_eq!(parts[3].len(), 4, "HHMM: {}", id);
    assert!(parts[3].chars().all(|c| c.is_ascii_digit()), "HHMM digits: {}", id);
    assert_eq!(parts[4].len(), 6, "hex6: {}", id);
    assert!(
        parts[4].chars().all(|c| c.is_ascii_digit() || matches!(c, 'a'..='f')),
        "hex6 must be 6 lowercase hex chars: {}",
        id
    );
}

// ---- swarm_state serde --------------------------------------------------

fn sample_state() -> SwarmState {
    SwarmState {
        run_id: "2026-06-27-1430-a1b2c3".to_string(),
        spec_path: "docs/research/specs/foo.md".to_string(),
        plan_path: "docs/research/plans/foo.md".to_string(),
        novelty_target: "hypothesis".to_string(),
        max_parallel: 4,
        phases: vec![Phase {
            name: "phase_1_literature_scout".to_string(),
            status: "pending".to_string(),
            workers: vec![Worker {
                name: "scout-1".to_string(),
                status: "pending".to_string(),
            }],
        }],
        escalations: vec![Escalation {
            worker: "hypothesis-smith-3".to_string(),
            reason: "red-team KILL after 3 rounds".to_string(),
            retry_count: 3,
        }],
        retry_counts: {
            let mut m = HashMap::new();
            m.insert("hypothesis-smith-3".to_string(), 3u32);
            m
        },
    }
}

#[test]
fn test_swarm_state_file_roundtrip() {
    let state = sample_state();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swarm-state.yaml");
    state.write(&path).unwrap();
    let loaded = SwarmState::read(&path).unwrap();
    assert_eq!(state, loaded);
}

#[test]
fn test_swarm_state_minimal_roundtrip() {
    // Empty phases/escalations/retry_counts must round-trip too.
    let state = SwarmState {
        run_id: "2026-06-27-0900-000000".to_string(),
        spec_path: "s".to_string(),
        plan_path: "p".to_string(),
        novelty_target: "gap-finding".to_string(),
        max_parallel: 4,
        phases: vec![],
        escalations: vec![],
        retry_counts: HashMap::new(),
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swarm-state.yaml");
    state.write(&path).unwrap();
    let loaded = SwarmState::read(&path).unwrap();
    assert_eq!(state, loaded);
}

// ---- run_tree -----------------------------------------------------------

#[test]
fn test_run_dir_path() {
    let base = tempfile::tempdir().unwrap();
    let dir: PathBuf = run_dir(base.path(), "2026-06-27-1430-a1b2c3");
    assert_eq!(dir, base.path().join("runs").join("2026-06-27-1430-a1b2c3"));
    assert!(!dir.exists(), "run_dir must not touch the filesystem");
}

#[test]
fn test_create_run_tree_makes_dir() {
    let base = tempfile::tempdir().unwrap();
    let created = create_run_tree(base.path(), "2026-06-27-1430-a1b2c3").unwrap();
    assert!(created.is_dir());
    assert_eq!(created, base.path().join("runs").join("2026-06-27-1430-a1b2c3"));
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test state 2>&1 | tail -15
```
Expected: compile error — `unresolved module run_id` / `run_tree` / `swarm_state`.

- [ ] **Step 3: Implement the three state submodules**

Replace `crates/research/src/state.rs` with:

```rust
//! Run-state: `swarm-state.yaml` serde, run-id generation, run-tree management.

pub mod run_id;
pub mod run_tree;
pub mod swarm_state;
```

Create `crates/research/src/state/run_id.rs`:

```rust
//! Run-id generation. The orchestrator skill's rule: `YYYY-MM-DD-HHMM-<6hex>`
//! (UTC date+time + 6 lowercase hex chars).

use std::io;

use chrono::Utc;

/// Assemble a run-id from a pre-formatted UTC stamp (`YYYY-MM-DD-HHMM`) and a
/// 6-char lowercase hex string. Pure; deterministic; unit-tested directly.
pub fn run_id_from_parts(stamp: &str, hex6: &str) -> String {
    format!("{}-{}", stamp, hex6)
}

/// Generate a run-id from the current UTC time and 3 random bytes (6 hex chars).
pub fn generate_run_id() -> io::Result<String> {
    let stamp = Utc::now().format("%Y-%m-%d-%H%M").to_string();
    let mut buf = [0u8; 3];
    getrandom::getrandom(&mut buf).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let hex6 = format!("{:02x}{:02x}{:02x}", buf[0], buf[1], buf[2]);
    Ok(run_id_from_parts(&stamp, &hex6))
}
```

Create `crates/research/src/state/swarm_state.rs`:

```rust
//! `swarm-state.yaml` schema + (de)serialization. The orchestrator (Phase 4) is
//! the single writer; the TUI is a read-only view. Schema per the design spec and
//! the orchestrator skill: run_id, spec_path, plan_path, novelty_target,
//! max_parallel, phases (each status + workers), escalations, retry_counts.
//!
//! `#[serde(default)]` is applied to the optional/collection fields so Phase 4
//! can extend the structs without breaking deserialization of Phase-2-written
//! files, and vice versa.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

fn default_max_parallel() -> u32 {
    4
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwarmState {
    pub run_id: String,
    pub spec_path: String,
    pub plan_path: String,
    pub novelty_target: String,
    #[serde(default = "default_max_parallel")]
    pub max_parallel: u32,
    #[serde(default)]
    pub phases: Vec<Phase>,
    #[serde(default)]
    pub escalations: Vec<Escalation>,
    #[serde(default)]
    pub retry_counts: HashMap<String, u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phase {
    pub name: String,
    pub status: String,
    #[serde(default)]
    pub workers: Vec<Worker>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Worker {
    pub name: String,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Escalation {
    pub worker: String,
    pub reason: String,
    #[serde(default)]
    pub retry_count: u32,
}

impl SwarmState {
    /// Read `swarm-state.yaml` from `path`.
    pub fn read(path: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        serde_yml::from_str(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Write `self` to `path` as YAML.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let text =
            serde_yml::to_string(self).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, text)
    }
}
```

Create `crates/research/src/state/run_tree.rs`:

```rust
//! Run-tree path management: `docs/research/runs/<run-id>/`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The run directory path under `base` (typically `docs/research`):
/// `base/runs/<run_id>`. Pure; does not touch the filesystem.
pub fn run_dir(base: &Path, run_id: &str) -> PathBuf {
    base.join("runs").join(run_id)
}

/// Create the run directory (mkdir -p) and return its path.
pub fn create_run_tree(base: &Path, run_id: &str) -> io::Result<PathBuf> {
    let dir = run_dir(base, run_id);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/ggix/MegaResearcher
cargo test -p megaresearcher-research --test state 2>&1 | tail -12
```
Expected: `test result: ok. 5 passed; 0 failed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/ggix/MegaResearcher
git add crates/research/src/state.rs crates/research/src/state/run_id.rs crates/research/src/state/swarm_state.rs crates/research/src/state/run_tree.rs crates/research/tests/state.rs
git commit -m "feat(rs): port research::state (run_id, swarm_state serde, run_tree)"
```

---

### Task 9: Phase 2 green sweep

**Files:**
- No source changes (this is verification). May touch `Cargo.lock` if a `cargo` command regenerates it.

**Interfaces:**
- Consumes: Tasks 1–8.
- Produces: a workspace that is fully green from the repo root — fmt, clippy, and all tests pass — satisfying the spec's "each build phase ends with the test suite green" rule.

- [ ] **Step 1: Full green sweep from repo root**

```bash
cd /Users/ggix/MegaResearcher
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -5
cargo test --workspace 2>&1 | tail -5
```
Expected: fmt check clean; clippy clean (no warnings); all tests pass (the Phase 1 baseline of 1363 tests plus the new research-crate tests: 7 verdict + 4 regression + 5 protocol_parser + 3 scaffold + 4 finalize + 6 preflight + 5 state + 1 smoke = 35 new; total ≈ 1398).

- [ ] **Step 2: If `cargo fmt --all --check` fails**

Run `cargo fmt --all` and commit the formatting separately:

```bash
cd /Users/ggix/MegaResearcher
cargo fmt --all
git add -u
git commit -m "style(rs): fmt research crate (phase 2)"
```

Then re-run the Step 1 block to confirm clean.

- [ ] **Step 3: If `Cargo.lock` changed**

If any `cargo` command regenerated `Cargo.lock`, commit it:

```bash
cd /Users/ggix/MegaResearcher
git add Cargo.lock
git commit -m "chore(rs): refresh Cargo.lock after phase 2 port"
```

- [ ] **Step 4: Confirm the final research-crate state**

```bash
cd /Users/ggix/MegaResearcher
echo "=== paper_chain submodules ===" && ls crates/research/src/paper_chain/
echo "=== state submodules ===" && ls crates/research/src/state/
echo "=== research integration tests ===" && ls crates/research/tests/
echo "=== workspace members ===" && cargo metadata --no-deps --format-version 1 | python3 -c "import sys,json; print('\n'.join(p['name'] for p in json.load(sys.stdin)['packages']))"
```
Expected: `paper_chain/` has `verdict.rs regression.rs protocol_parser.rs scaffold.rs finalize.rs preflight.rs`; `state/` has `run_id.rs swarm_state.rs run_tree.rs`; `tests/` has `smoke.rs verdict.rs regression.rs protocol_parser.rs scaffold.rs finalize.rs preflight.rs state.rs` + `fixtures/`; 10 workspace packages (unchanged from Phase 1).

- [ ] **Step 5: Confirm old v0 plugin files are untouched**

```bash
cd /Users/ggix/MegaResearcher
git diff --stat 3852b87..HEAD -- lib/ tests/test_verdict_parser.py tests/test_regression.py tests/test_protocol_parser.py tests/test_scaffold.py tests/test_finalize.py tests/test_preflight.py skills/
```
Expected: empty (no changes to the v0 port-reference files). `3852b87` is the Phase 1 final commit.

---

## Self-Review

**1. Spec coverage.** Spec §13.2 ("Port pure logic + state") lists: `paper_chain::{verdict, regression, protocol_parser, scaffold, finalize, preflight}` (Tasks 2–7 ✓) and `research::state` (swarm-state serde Task 8 swarm_state ✓, run-id Task 8 run_id ✓, run-tree Task 8 run_tree ✓). Spec §12 lists the pure-logic unit tests: verdict/regression/protocol_parser/scaffold/finalize/preflight (Tasks 2–7 tests ✓) + swarm-state serde + run-id generation (Task 8 tests ✓). The `experiment` and `sandbox` paper-chain modules are explicitly out of Phase 2 scope (deferred to Phase 7 / SP2a) — spec §13.2 does not list them. The consolidations (`bibliography.md`, `gaps.md`) and spec-latest symlink are listed under `research::state` in spec §4 but the Phase 1 plan's handoff note scoped Phase 2 state to "swarm-state serde, run-id generation, run-tree" only; consolidations/symlink are orchestrator-phase concerns (Phase 4). No Phase 2 spec requirement is unaddressed.

**2. Placeholder scan.** Every code step contains complete Rust (no `TODO`/`TBD`/`...` in code blocks). The `...` tokens appear only inside the verbatim `specs_protocol.md` fixture (the fixture's literal last line is `...`, matching the source file byte-for-byte) and in doc comments describing the Python regex — not in executable Rust. Regex patterns, struct fields, test bodies, and commit messages are all concrete. No "add appropriate error handling" / "similar to Task N" / "fill in details" anywhere.

**3. Type consistency.** Signatures are consistent across tasks and match the Produces blocks: `parse_verdict(&Path) -> io::Result<Option<String>>` (Task 2 produces / Task 9 sweep uses); `extract_weaknesses(&Path) -> io::Result<Vec<String>>` and `detect_regression(&Path, &Path) -> io::Result<(bool, usize, usize)>` (Task 3); `parse_protocol(&Path) -> io::Result<Protocol>` with `Protocol { substrate: Option<String>, sample_size: Option<i64>, seed: Option<i64>, baselines: Vec<String>, metrics: Vec<String>, decision_rules: Vec<DecisionRule> }` and `DecisionRule { raw: String }` (Task 4 — test asserts `.substrate.as_deref()`, `.sample_size == Some(22)`, `.baselines[0]`, `.metrics.iter()`, `.decision_rules[0].raw`, all matching the struct fields); `scaffold_paper_dir(&Path) -> io::Result<PathBuf>` (Task 5); `finalize_paper(&Path, &str) -> io::Result<PathBuf>` (Task 6); `preflight_check(&Path) -> io::Result<(bool, String)>` and `preflight_check_with_paper(&Path, bool) -> io::Result<(bool, String, Vec<String>)>` (Task 7); `run_id_from_parts(&str, &str) -> String`, `generate_run_id() -> io::Result<String>`, `SwarmState::{read, write}`, `run_dir(&Path, &str) -> PathBuf`, `create_run_tree(&Path, &str) -> io::Result<PathBuf>` (Task 8). The `io::Result` error convention is applied uniformly to every fs-touching function; the `Phase`/`Worker`/`Escalation` field names (`name`, `status`, `worker`, `reason`, `retry_count`) are used identically in the struct defs and the test's `sample_state()`. No name drift between definition and use.