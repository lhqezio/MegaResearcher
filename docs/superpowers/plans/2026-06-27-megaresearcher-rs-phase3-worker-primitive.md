# MegaResearcher v1 — Phase 3: Worker Primitive + Fake Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the in-process worker primitive (fresh-context query loop against `LlmProvider`), the scoped worker tools (jailed Read/Write), and the prompt-asset loader, all exercised by a deterministic fake provider — so worker-contract tests pass without network.

**Architecture:** A `Worker` holds a system prompt (the agent file body), a set of `Arc<dyn Tool>`, and an `Arc<dyn LlmProvider>`. `Worker::run(prompt)` drives the query loop: build a `ProviderRequest` (struct literal) → `provider.create_message_stream` → accumulate `StreamEvent`s into assistant `ContentBlock`s → if any `ToolUse`, dispatch the tools and append `ToolResult` blocks as a user message, loop; else return the final assistant text. Bounded by `max_turns` (default 50). The fake provider (`tests/common/fake_provider.rs`) returns canned `Vec<StreamEvent>` turn sequences. Scoped `Read`/`Write` tools hard-jail every `file_path` under the worker's output dir (lexical `..`/absolute rejection) before any I/O. No api-crate changes; the existing object-safe `LlmProvider` trait is the seam.

**Tech Stack:** Rust (edition 2021), `tokio` (full), `futures` (`Stream`, `StreamExt`, `stream::iter`), `async-trait`, `serde`/`serde_json`, `serde_yml`, `claurst-api` (`LlmProvider`, `ProviderRequest`, `StreamEvent`, …), `claurst-core` (`Message`, `ContentBlock`, `ToolDefinition`, `UsageInfo`, `ProviderId`), `tempfile` (dev/test).

## Global Constraints

Carried verbatim from the design spec (`docs/superpowers/specs/2026-06-26-megaresearcher-rs-design.md`) and the project `CLAUDE.md`. Every task's requirements implicitly include this section.

- **Worker contract (spec §6):** fresh context `[user: <prompt>]`; system prompt = the agent file body only; result = the final assistant text only; the worker cannot prompt the user; bounded by `max_turns`; in-process query loop. **First cut drops** autocompact, message normalization (orphan tool_use synthesis / trailing-thinking strip), and the abort/cancel token — the fake controls context, matching spec §6 ("the minimal deterministic worker is: normalize → autocompact (threshold-based) → stream → tool dispatch → loop on needsFollowUp"; for Phase 3 we defer the normalize and autocompact stages to when a real provider exists).
- **Provider seam (spec §15, resolved):** use claurst's existing `api` crate multi-provider `LlmProvider` trait (`claurst_api::LlmProvider`), consumed as `Arc<dyn LlmProvider>`. Do NOT thin the api crate to Anthropic-only. `ProviderRequest` is built as a **struct literal** (no builder). `create_message_stream` returns `Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>`, consumed via `futures::StreamExt::next`. No api-crate modifications in Phase 3.
- **max_turns (spec §15, resolved):** `DEFAULT_MAX_TURNS = 50`, configurable per worker via `WorkerConfig::max_turns`.
- **Scoped tools (spec §7):** a hard path check inside the Rust Read/Write tools rejects any `file_path` not under the worker's assigned output dir **before** any I/O runs (treated as bypass-immune, fires regardless of mode). Workers get NO Bash. The tool set is scoped file I/O only — MCP research tools arrive in Phase 5. A worker may read a shared `docs/research/` dir (`Read(docs/research/**)` allowed) but writes are jailed to `<worker-dir>`.
- **tool_result block (spec §7):** the dispatch result is wrapped as a user message containing `ContentBlock::ToolResult { tool_use_id, content: ToolResultContent::Text(..), is_error }`; on deny/throw/invalid input the content uses an `<tool_use_error>...</tool_use_error>` wrapper and `is_error: Some(true)`.
- **Prompt-asset loader (spec §8 / §13.3):** parse the v0 `agents/<name>.md` file (YAML frontmatter + body) into `(name, description, model, body)`; the body is the worker system prompt; the `description` frontmatter field is the orchestrator's routing key. Use `serde_yml` (already a research dep from Phase 2) for frontmatter — it handles `description: |` block scalars natively. Do NOT reuse claurst-core's `parse_skill_file` / `DiscoveredSkill` (it line-scans `description:` and cannot parse block scalars).
- **Artifact gate (spec §11):** `check_artifacts(dir, &["output.md","manifest.yaml","verification.md"])` returns the missing names; a non-empty result is the verification gate the orchestrator uses to flag a deliberately-missing `manifest.yaml`. Phase 3 implements the pure check; the redispatch-once-then-escalate policy is the orchestrator's (Phase 4).
- **Testing (spec §12):** worker primitive tests with a fake provider (canned tool-call sequences) — assert scoped Read/Write only touch the worker's dir, the three artifacts get written, and `check_artifacts` flags a deliberately-missing `manifest.yaml`. Deterministic fakes, no network.
- **Each build phase ends green (spec §13):** `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` all green from repo root. **Per-task hygiene adds `cargo clippy -p megaresearcher-research -- -D warnings`** (process fix from the Phase 2 sweep, where per-task `cargo check` let 5 verbatim-port clippy lints slip through to the workspace sweep).
- **Edit scope:** ONLY `crates/research/` (+ the workspace `Cargo.toml` if a dep must be added — none should be needed; all required deps are already in Phase 2's `[dependencies]`). The old plugin files (`lib/`, repo-root `tests/test_*.py`, `skills/`, repo-root `agents/`) are the **port reference** and MUST NOT be modified. Phase 3 *copies* `agents/literature-scout.md` verbatim into `crates/research/tests/fixtures/agents/`; the repo-root `agents/literature-scout.md` is not touched.
- **No git worktrees; branch `main`.** Confirm `git branch --show-current` before dispatching any implementer and include `main` in every dispatch prompt. Do not `git switch` mid-task.
- **GPL-3.0.** No new licenseable assets: Rust ports of the project's own v0 files plus new test infra. New crates.io deps (if any) are MIT/Apache.
- **Banned phrases/words (global CLAUDE.md):** never use "load-bearing", "this is doing a lot of work" (or close variants), the emphatic adjective "real" (e.g. "real run", "real-world"), or "honest/honestly/to be honest" in any artifact. Genuine technical terms ("real-time", the literal name of a crate) are exempt.
- **io::Result convention:** fs-touching public functions return `std::io::Result` (e.g. `prompt_asset::load`).
- **Commit messages** end with `Co-Authored-By: Claude <noreply@anthropic.com>`.

---

## File Structure

All under `crates/research/` unless noted.

| File | Responsibility | Task |
|---|---|---|
| `src/lib.rs` | Re-exports `prompt_asset`, `worker`, `worker_tools` (public modules) | 1 |
| `src/prompt_asset.rs` | `PromptAsset { name, description, model, body }`, `parse(&str)`, `load(&Path) -> io::Result<PromptAsset>` | 2 |
| `src/worker_tools.rs` | `Tool` trait, `ToolResult`, `ScopedRead`, `ScopedWrite`, `jail_under`, `check_artifacts` | 3 |
| `tests/common/mod.rs` | `pub mod fake_provider;` (shared test-helper module, cargo's `tests/` subdir pattern) | 4 |
| `tests/common/fake_provider.rs` | `FakeProvider` implementing `LlmProvider` (canned `Vec<Vec<StreamEvent>>` turns) | 4 |
| `tests/fake_provider.rs` | Isolation test for the fake itself | 4 |
| `tests/fixtures/agents/literature-scout.md` | Verbatim copy of `agents/literature-scout.md` (fixture for Task 2 test) | 2 |
| `tests/prompt_asset.rs` | `prompt_asset` parse/load tests | 2 |
| `tests/worker_tools.rs` | ScopedRead/ScopedWrite jail + `check_artifacts` tests | 3 |
| `src/worker.rs` | `Worker`, `WorkerConfig`, `WorkerOutcome`, `WorkerStop`, query loop, `accumulate` | 5 |
| `tests/worker.rs` | Worker-contract tests (writes-3-artifacts, max_turns, unknown-tool, EndTurn) | 5 |
| `.superpowers/sdd/progress.md` | Ledger append (existing file) | each task |

Decomposition rationale: each task is a self-contained, independently testable deliverable a reviewer can gate on its own. Task 1 wires empty modules (compiles + smoke still green). Task 2 adds the loader (pure parse, no dep on api). Task 3 adds the tool trait + scoped tools + artifact check (pure, no dep on api). Task 4 adds the fake provider (depends on the api trait only). Task 5 wires the worker (depends on Tasks 3+4: `Tool`, `ToolResult`, `FakeProvider`). Task 6 is the green sweep (fmt/clippy/test from root). No task depends on a later task.

---

### Task 1: Scaffold — module wiring + deps confirmation

**Files:**
- Modify: `crates/research/src/lib.rs`
- Test: `crates/research/tests/smoke.rs` (existing — must still pass unchanged)

**Interfaces:**
- Consumes: Phase 2's `lib.rs` (`pub mod paper_chain; pub mod state; pub const CRATE_NAME;`)
- Produces: `pub mod prompt_asset; pub mod worker; pub mod worker_tools;` (empty module stubs) so the crate compiles and the smoke test still links.

- [ ] **Step 1: Read the current `lib.rs`**

Run: `sed -n '1,60p' crates/research/src/lib.rs`
Confirm it currently declares `pub mod paper_chain;`, `pub mod state;`, `pub const CRATE_NAME: &str = ...;` (Phase 2 surface).

- [ ] **Step 2: Add the three new public module declarations**

Edit `crates/research/src/lib.rs` to add (after the existing `pub mod` lines):

```rust
pub mod prompt_asset;
pub mod worker;
pub mod worker_tools;
```

Create three empty module files so the crate compiles. Each file is just the module doc comment for now:

`crates/research/src/prompt_asset.rs`:
```rust
//! Worker prompt-asset loader: parse v0 `agents/<name>.md` into (name, description, model, body).
```

`crates/research/src/worker_tools.rs`:
```rust
//! Scoped worker tools: jailed Read/Write + the three-artifact presence check.
```

`crates/research/src/worker.rs`:
```rust
//! The in-process worker primitive: a fresh-context query loop against an `LlmProvider`.
```

- [ ] **Step 3: Confirm the workspace deps are already present**

Run: `grep -E 'futures|async-trait|serde_yml|claurst-api|claurst-core|tokio' crates/research/Cargo.toml`
Confirm `[dependencies]` already contains (from Phase 2): `tokio = { version = "1.44", features = ["full"] }`, `futures = "0.3"`, `async-trait = "0.1"`, `serde_json = "1"`, `serde_yml = "..."`, `claurst-api = { path = "crates/api" }`, `claurst-core = { path = "crates/core" }`, and `[dev-dependencies] tempfile = "3"`. If any of `futures`, `async-trait`, or `serde_yml` is missing, add it. (They should all be present — Phase 2 added `serde_yml`; claurst's own deps already pull `futures`/`async-trait` transitively but the research crate must name them explicitly.)

- [ ] **Step 4: Build + smoke still green**

Run:
```bash
cargo build -p megaresearcher-research 2>&1 | tail -5
cargo test -p megaresearcher-research --test smoke 2>&1 | tail -5
```
Expected: build succeeds; smoke test passes (`1 passed`).

- [ ] **Step 5: fmt + clippy hygiene**

Run:
```bash
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research -- -D warnings 2>&1 | tail -5
```
Expected: fmt makes no changes (empty stubs); clippy exit 0.

- [ ] **Step 6: Commit**

```bash
git add crates/research/src/lib.rs crates/research/src/prompt_asset.rs crates/research/src/worker.rs crates/research/src/worker_tools.rs
git commit -m "$(cat <<'EOF'
feat(rs): scaffold research::prompt_asset / worker / worker_tools modules (phase 3)

Wire three empty public modules so the crate compiles and the smoke test
links. Implementations arrive in the following tasks.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: prompt_asset — PromptAsset + load + verbatim fixture

**Files:**
- Modify: `crates/research/src/prompt_asset.rs`
- Create: `crates/research/tests/prompt_asset.rs`
- Create: `crates/research/tests/fixtures/agents/literature-scout.md` (verbatim copy)

**Interfaces:**
- Consumes: `serde_yml` (Deserialize), `std::fs`, `std::io`, `std::path::Path`.
- Produces:
  - `pub struct PromptAsset { pub name: String, pub description: String, pub model: String, pub body: String }` (derives `Debug, Clone, PartialEq`)
  - `pub fn parse(text: &str) -> Result<PromptAsset, String>`
  - `pub fn load(path: &Path) -> std::io::Result<PromptAsset>`

**Fixture facts (verified against `agents/literature-scout.md`):** frontmatter is `name: literature-scout`, `description: |` (YAML block scalar, multi-line, contains the `<example>` block), `model: inherit`. The body (after the closing `---`) starts with `"You are a literature scout for MegaResearcher."` and contains the section headers `## Inputs you receive`, `## Tools you use`, `## What to produce`, `## Discipline rules`. The `description` value contains the substrings `"Survey prior art for a sub-topic"`, `"annotated bibliography"`, and `"<example>"`.

- [ ] **Step 1: Copy the fixture verbatim**

```bash
mkdir -p crates/research/tests/fixtures/agents
cp agents/literature-scout.md crates/research/tests/fixtures/agents/literature-scout.md
diff agents/literature-scout.md crates/research/tests/fixtures/agents/literature-scout.md && echo "BYTE-IDENTICAL"
```
Expected: `BYTE-IDENTICAL` (empty diff). The repo-root `agents/literature-scout.md` is NOT modified.

- [ ] **Step 2: Write the failing test**

`crates/research/tests/prompt_asset.rs`:
```rust
//! prompt_asset parse/load tests.

use std::path::PathBuf;

use megaresearcher_research::prompt_asset::{load, parse};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/agents")
}

#[test]
fn test_parse_literature_scout() {
    let asset = load(&fixtures().join("literature-scout.md")).unwrap();
    assert_eq!(asset.name, "literature-scout");
    assert_eq!(asset.model, "inherit");
    assert!(asset.description.contains("Survey prior art for a sub-topic"));
    assert!(asset.description.contains("annotated bibliography"));
    assert!(asset.description.contains("<example>"));
    assert!(
        asset.body.starts_with("You are a literature scout for MegaResearcher."),
        "body should start with the agent system-prompt opener; got: {:?}",
        asset.body.chars().take(80).collect::<String>()
    );
    assert!(asset.body.contains("## Inputs you receive"));
    assert!(asset.body.contains("## Tools you use"));
    assert!(asset.body.contains("## What to produce"));
    assert!(asset.body.contains("## Discipline rules"));
}

#[test]
fn test_parse_rejects_missing_frontmatter() {
    // No leading "---\n" delimiter.
    let result = parse("just a body, no frontmatter at all");
    assert!(result.is_err(), "must reject a file with no frontmatter");
}

#[test]
fn test_parse_rejects_missing_closing_delimiter() {
    // Opening delimiter but no closing "---".
    let result = parse("---\nname: x\ndescription: y\nmodel: inherit\n\nbody never closes");
    assert!(result.is_err(), "must reject unclosed frontmatter");
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p megaresearcher-research --test prompt_asset 2>&1 | tail -15`
Expected: FAIL — `load` / `parse` not found (module body empty). (The fixture-copy test `test_parse_literature_scout` will fail to compile against missing `load`/`parse`.)

- [ ] **Step 4: Implement prompt_asset**

`crates/research/src/prompt_asset.rs`:
```rust
//! Worker prompt-asset loader: parse v0 `agents/<name>.md` into (name, description, model, body).
//!
//! The v0 worker agent files are YAML-frontmatter + body markdown:
//!
//! ```text
//! ---
//! name: literature-scout
//! description: |
//!   Survey prior art ...
//! model: inherit
//! ---
//!
//! You are a literature scout for MegaResearcher. ...
//! ```
//!
//! The body (after the closing `---`) is the worker system prompt. The
//! `description` frontmatter field is the orchestrator's routing key.

use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize)]
struct FrontMatter {
    name: String,
    description: String,
    model: String,
}

/// A parsed worker agent file.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptAsset {
    /// The agent's short id, e.g. `literature-scout`.
    pub name: String,
    /// The full `description:` block scalar — used by the orchestrator to route.
    pub description: String,
    /// The declared model (v0 uses `inherit`).
    pub model: String,
    /// The body after the closing `---` — the worker system prompt.
    pub body: String,
}

/// Parse the raw text of an agent file into a [`PromptAsset`].
///
/// The text must start with a `---` delimiter line and contain a closing
/// `---` delimiter line; the text between them is YAML frontmatter, and the
/// text after the closing delimiter is the body.
pub fn parse(text: &str) -> Result<PromptAsset, String> {
    let after_open = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
        .ok_or_else(|| "file must start with a frontmatter delimiter '---'".to_string())?;

    // The closing delimiter is the first line after the opener that is exactly "---".
    let close_rel = after_open
        .lines()
        .position(|line| line.trim() == "---")
        .ok_or_else(|| "missing closing frontmatter delimiter '---'".to_string())?;

    let frontmatter_text: String = after_open
        .lines()
        .take(close_rel)
        .collect::<Vec<_>>()
        .join("\n");
    let body: String = after_open
        .lines()
        .skip(close_rel + 1)
        .collect::<Vec<_>>()
        .join("\n");

    let front: FrontMatter = serde_yml::from_str(&frontmatter_text)
        .map_err(|e| format!("invalid frontmatter YAML: {e}"))?;

    Ok(PromptAsset {
        name: front.name,
        description: front.description,
        model: front.model,
        body: body.trim().to_string(),
    })
}

/// Load and parse an agent file from disk.
pub fn load(path: &Path) -> io::Result<PromptAsset> {
    let text = fs::read_to_string(path)?;
    parse(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p megaresearcher-research --test prompt_asset 2>&1 | tail -15`
Expected: PASS — 3 passed.

- [ ] **Step 6: fmt + clippy hygiene**

Run:
```bash
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research -- -D warnings 2>&1 | tail -5
```
Expected: fmt clean; clippy exit 0.

- [ ] **Step 7: Commit**

```bash
git add crates/research/src/prompt_asset.rs crates/research/tests/prompt_asset.rs crates/research/tests/fixtures/agents/literature-scout.md
git commit -m "$(cat <<'EOF'
feat(rs): add research::prompt_asset — parse v0 agent files (phase 3)

PromptAsset { name, description, model, body } with parse() (serde_yml
frontmatter handling the `description: |` block scalar) and load() (io::Result).
Verbatim-copy literature-scout.md into tests/fixtures/agents/ as the fixture.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: worker_tools — Tool trait, ScopedRead/ScopedWrite, check_artifacts

**Files:**
- Modify: `crates/research/src/worker_tools.rs`
- Create: `crates/research/tests/worker_tools.rs`

**Interfaces:**
- Consumes: `async_trait::async_trait`, `serde_json::Value`, `std::fs`, `std::path::{Path, PathBuf}`.
- Produces (used by Task 5's worker):
  - `pub struct ToolResult { pub content: String, pub is_error: bool }` with `ToolResult::ok(impl Into<String>)`, `ToolResult::err(impl Into<String>)`
  - `#[async_trait] pub trait Tool: Send + Sync { fn name(&self) -> &str; fn description(&self) -> &str; fn input_schema(&self) -> serde_json::Value; async fn call(&self, input: serde_json::Value) -> ToolResult; fn is_concurrency_safe(&self) -> bool { false } fn is_read_only(&self) -> bool { false } }`
  - `pub struct ScopedRead { ... }` + `ScopedRead::new(dir)` / `ScopedRead::with_shared(dir, shared)`
  - `pub struct ScopedWrite { ... }` + `ScopedWrite::new(dir)`
  - `pub fn jail_under(root: &Path, file_path: &str) -> Result<PathBuf, String>`
  - `pub fn check_artifacts(dir: &Path, expected: &[&str]) -> Vec<String>`

**Jail semantics (spec §7):** the hard path check rejects (a) any absolute `file_path` and (b) any `..` parent-dir component, before any filesystem I/O. A path that passes both checks is `root.join(file_path)`. This is a deterministic lexical jail (no `canonicalize`, no symlink resolution) — sufficient for Phase 3's trusted worker output dirs and deterministic tests; canonicalize-based symlink hardening is a future hardening, out of Phase 3 scope.

- [ ] **Step 1: Write the failing tests**

`crates/research/tests/worker_tools.rs`:
```rust
//! ScopedRead/ScopedWrite jail + check_artifacts tests.

use megaresearcher_research::worker_tools::{check_artifacts, ScopedRead, ScopedWrite};
use serde_json::json;

#[tokio::test]
async fn test_scoped_write_creates_file_in_dir() {
    let d = tempfile::tempdir().unwrap();
    let w = ScopedWrite::new(d.path());
    let r = w.call(json!({"file_path":"output.md","content":"# hi"})).await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(std::fs::read_to_string(d.path().join("output.md")).unwrap(), "# hi");
}

#[tokio::test]
async fn test_scoped_write_rejects_parent_dir_escape() {
    let d = tempfile::tempdir().unwrap();
    let w = ScopedWrite::new(d.path());
    let r = w.call(json!({"file_path":"../escape.md","content":"x"})).await;
    assert!(r.is_error);
    assert!(r.content.contains("tool_use_error"), "{}", r.content);
}

#[tokio::test]
async fn test_scoped_write_rejects_absolute_path() {
    let d = tempfile::tempdir().unwrap();
    let w = ScopedWrite::new(d.path());
    let r = w.call(json!({"file_path":"/etc/passwd","content":"x"})).await;
    assert!(r.is_error);
    assert!(r.content.contains("tool_use_error"));
}

#[tokio::test]
async fn test_scoped_write_missing_fields_is_error() {
    let d = tempfile::tempdir().unwrap();
    let w = ScopedWrite::new(d.path());
    let r = w.call(json!({"file_path":"x.md"})).await; // missing content
    assert!(r.is_error);
    assert!(r.content.contains("tool_use_error"));
}

#[tokio::test]
async fn test_scoped_read_from_worker_dir() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("a.txt"), "alpha").unwrap();
    let r = ScopedRead::new(d.path()).call(json!({"file_path":"a.txt"})).await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(r.content, "alpha");
}

#[tokio::test]
async fn test_scoped_read_falls_through_to_shared_dir() {
    let d = tempfile::tempdir().unwrap();
    let shared = tempfile::tempdir().unwrap();
    std::fs::write(shared.path().join("shared.md"), "shared content").unwrap();
    let r = ScopedRead::with_shared(d.path(), shared.path())
        .call(json!({"file_path":"shared.md"}))
        .await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(r.content, "shared content");
}

#[tokio::test]
async fn test_scoped_read_rejects_escape() {
    let d = tempfile::tempdir().unwrap();
    let r = ScopedRead::new(d.path()).call(json!({"file_path":"../secret"})).await;
    assert!(r.is_error);
    assert!(r.content.contains("tool_use_error"));
}

#[tokio::test]
async fn test_scoped_read_missing_file_is_error() {
    let d = tempfile::tempdir().unwrap();
    let r = ScopedRead::new(d.path()).call(json!({"file_path":"nope.md"})).await;
    assert!(r.is_error);
    assert!(r.content.contains("tool_use_error"));
}

#[test]
fn test_check_artifacts_flags_missing_manifest() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("output.md"), "").unwrap();
    std::fs::write(d.path().join("verification.md"), "").unwrap();
    let missing = check_artifacts(d.path(), &["output.md", "manifest.yaml", "verification.md"]);
    assert_eq!(missing, vec!["manifest.yaml".to_string()]);
}

#[test]
fn test_check_artifacts_all_present() {
    let d = tempfile::tempdir().unwrap();
    for name in &["output.md", "manifest.yaml", "verification.md"] {
        std::fs::write(d.path().join(name), "").unwrap();
    }
    assert!(check_artifacts(d.path(), &["output.md", "manifest.yaml", "verification.md"]).is_empty());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p megaresearcher-research --test worker_tools 2>&1 | tail -15`
Expected: FAIL — `ScopedRead`, `ScopedWrite`, `check_artifacts` not found.

- [ ] **Step 3: Implement worker_tools**

`crates/research/src/worker_tools.rs`:
```rust
//! Scoped worker tools: jailed Read/Write + the three-artifact presence check.
//!
//! The hard path jail (spec §7) rejects absolute paths and `..` parent-dir
//! components before any I/O. Write is jailed to the worker output dir; Read
//! may also read a shared research dir. Tool errors are wrapped in
//! `<tool_use_error>...</tool_use_error>`.

use std::fs;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::Value;

/// The outcome of a tool call: the text content to place in the `tool_result`
/// block, and whether it is an error.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: false }
    }
    pub fn err(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: true }
    }
}

/// Wrap a message in the spec §7 `<tool_use_error>` envelope.
fn tool_use_error(message: impl Into<String>) -> String {
    format!("<tool_use_error>{}</tool_use_error>", message.into())
}

/// A worker tool. Phase 3 ships the scoped file-I/O tools; Phase 5 adds async
/// MCP tools on the same trait.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;
    async fn call(&self, input: Value) -> ToolResult;
    /// Safe to run concurrently with other tools. Read-only tools return true.
    fn is_concurrency_safe(&self) -> bool {
        false
    }
    fn is_read_only(&self) -> bool {
        false
    }
}

/// Hard path jail: resolve `file_path` under `root`, rejecting absolute paths
/// and any `..` parent-dir component. Returns the joined path on success.
pub fn jail_under(root: &Path, file_path: &str) -> Result<PathBuf, String> {
    let p = Path::new(file_path);
    if p.is_absolute() {
        return Err(format!("absolute paths are not allowed: {file_path}"));
    }
    if p.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return Err(format!("parent-dir (..) components are not allowed: {file_path}"));
    }
    Ok(root.join(file_path))
}

/// Try to read `file_path` under `root`. Returns:
/// - `Some(Ok(text))` if the file exists and reads cleanly,
/// - `Some(Err(msg))` if the jail rejects the path or the read fails,
/// - `None` if the file does not exist under `root` (so the caller can fall
///   through to a shared dir).
fn read_under(root: &Path, file_path: &str) -> Option<Result<String, String>> {
    match jail_under(root, file_path) {
        Ok(p) => {
            if !p.exists() {
                return None;
            }
            Some(fs::read_to_string(&p).map_err(|e| format!("read failed: {e}")))
        }
        Err(msg) => Some(Err(msg)),
    }
}

/// A `Read` tool scoped to the worker output dir, optionally with a shared
/// research dir for read-only access to shared artifacts.
pub struct ScopedRead {
    dir: PathBuf,
    shared: Option<PathBuf>,
}

impl ScopedRead {
    /// Read only from the worker output dir.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into(), shared: None }
    }
    /// Read from the worker dir first, then fall through to `shared`.
    pub fn with_shared(dir: impl Into<PathBuf>, shared: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into(), shared: Some(shared.into()) }
    }
}

#[async_trait]
impl Tool for ScopedRead {
    fn name(&self) -> &str {
        "Read"
    }
    fn description(&self) -> &str {
        "Read a file under the worker output dir (or the shared research dir)."
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string" }
            },
            "required": ["file_path"]
        })
    }
    async fn call(&self, input: Value) -> ToolResult {
        let file_path = match input.get("file_path").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::err(tool_use_error("missing 'file_path' string field")),
        };
        if let Some(result) = read_under(&self.dir, file_path) {
            return match result {
                Ok(text) => ToolResult::ok(text),
                Err(msg) => ToolResult::err(tool_use_error(msg)),
            };
        }
        if let Some(shared) = &self.shared {
            if let Some(result) = read_under(shared, file_path) {
                return match result {
                    Ok(text) => ToolResult::ok(text),
                    Err(msg) => ToolResult::err(tool_use_error(msg)),
                };
            }
        }
        ToolResult::err(tool_use_error(format!("file not found: {file_path}")))
    }
    fn is_concurrency_safe(&self) -> bool {
        true
    }
    fn is_read_only(&self) -> bool {
        true
    }
}

/// A `Write` tool jailed to the worker output dir.
pub struct ScopedWrite {
    dir: PathBuf,
}

impl ScopedWrite {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
}

#[async_trait]
impl Tool for ScopedWrite {
    fn name(&self) -> &str {
        "Write"
    }
    fn description(&self) -> &str {
        "Write a file under the worker output dir."
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "Path relative to the worker output dir." },
                "content": { "type": "string" }
            },
            "required": ["file_path", "content"]
        })
    }
    async fn call(&self, input: Value) -> ToolResult {
        let file_path = match input.get("file_path").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::err(tool_use_error("missing 'file_path' string field")),
        };
        let content = match input.get("content").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::err(tool_use_error("missing 'content' string field")),
        };
        match jail_under(&self.dir, file_path) {
            Ok(p) => {
                let parent = p.parent().unwrap_or(Path::new(""));
                if let Err(e) = fs::create_dir_all(parent).and_then(|_| fs::write(&p, content)) {
                    ToolResult::err(tool_use_error(format!("write failed: {e}")))
                } else {
                    ToolResult::ok(format!("wrote {file_path} ({} bytes)", content.len()))
                }
            }
            Err(msg) => ToolResult::err(tool_use_error(msg)),
        }
    }
    fn is_read_only(&self) -> bool {
        false
    }
}

/// Return the names from `expected` whose files do not exist under `dir`.
/// The orchestrator's verification gate (spec §11): a non-empty result means the
/// worker's three-artifact contract is unmet.
pub fn check_artifacts(dir: &Path, expected: &[&str]) -> Vec<String> {
    expected
        .iter()
        .filter(|name| !dir.join(name).exists())
        .map(|s| s.to_string())
        .collect()
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test worker_tools 2>&1 | tail -15`
Expected: PASS — 10 passed.

- [ ] **Step 5: fmt + clippy hygiene**

Run:
```bash
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research -- -D warnings 2>&1 | tail -5
```
Expected: fmt clean; clippy exit 0.

- [ ] **Step 6: Commit**

```bash
git add crates/research/src/worker_tools.rs crates/research/tests/worker_tools.rs
git commit -m "$(cat <<'EOF'
feat(rs): add research::worker_tools — Tool trait, jailed Read/Write, artifact check (phase 3)

ScopedRead/ScopedWrite hard-jail file_path under the worker output dir
(lexical absolute/.. rejection before any I/O). tool_result errors wrap in
<tool_use_error>. check_artifacts lists missing artifact names for the
verification gate. Async Tool trait grounds the Phase 5 MCP tools.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: FakeProvider — canned StreamEvent turns (test infra)

**Files:**
- Create: `crates/research/tests/common/mod.rs`
- Create: `crates/research/tests/common/fake_provider.rs`
- Create: `crates/research/tests/fake_provider.rs`

**Interfaces:**
- Consumes: `claurst_api::{LlmProvider, ProviderRequest, ProviderResponse, ProviderError, ProviderCapabilities, ProviderStatus, StreamEvent, StopReason, SystemPromptStyle, ModelInfo}`, `claurst_core::provider_id::ProviderId`, `futures::{Stream, stream::iter}`, `async_trait`, `std::sync::atomic::AtomicUsize`.
- Produces:
  - `pub struct FakeProvider { ... }` (in `tests/common/fake_provider.rs`)
  - `FakeProvider::new(id: &str, turns: Vec<Vec<StreamEvent>>) -> Self`
  - `FakeProvider::call_count(&self) -> usize`
  - `impl LlmProvider for FakeProvider` — `create_message_stream` returns the `turns[call_index]` events as `Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>`; when `call_index` exceeds the scripted turns it re-emits the **last** turn (so a worker that keeps looping a tool-use turn terminates on `max_turns` deterministically).

**Placement rationale:** `tests/common/` is cargo's official shared-test-helper pattern — a subdir of `tests/` included via `mod common;` is compiled into each test binary, not as a separate binary. The fake lives in test infra (not shipped, not feature-gated) per spec §12; Phase 4's orchestrator integration test reuses it by `mod common;`.

**Note on import paths:** the types below are all re-exported at the `claurst_api` crate root (verified: `pub use provider::{LlmProvider, ModelInfo};`, `pub use provider_types::*;`, `pub use types::*;`). Use `claurst_api::<Name>`. `ProviderId` comes from `claurst_core::provider_id::ProviderId` (`ProviderId::new(impl Into<String>)`). If a path does not resolve as written, the symbol is also available at its origin module (`claurst_api::provider::LlmProvider`, `claurst_api::provider_types::StreamEvent`, `claurst_api::types::SystemPrompt`); use whichever the compiler accepts — the type names are exact.

- [ ] **Step 1: Create the shared test-helper module**

`crates/research/tests/common/mod.rs`:
```rust
//! Shared test helpers for the research crate's integration tests.
pub mod fake_provider;
```

`crates/research/tests/common/fake_provider.rs`:
```rust
//! A deterministic `LlmProvider` that emits canned `StreamEvent` turn sequences.
//!
//! Used by the worker-contract tests (Task 5) and the Phase 4 orchestrator
//! integration test. Lives in test infra, not the shipped library.

use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::stream::iter;
use futures::Stream;

use claurst_api::{
    LlmProvider, ModelInfo, ProviderCapabilities, ProviderError, ProviderRequest, ProviderResponse,
    ProviderStatus, StopReason, StreamEvent, SystemPromptStyle,
};
use claurst_core::provider_id::ProviderId;

/// A fake provider that returns scripted turn sequences.
///
/// Each call to `create_message_stream` pops the next turn (a `Vec<StreamEvent>`).
/// Once the scripted turns are exhausted, it re-emits the last turn — so a
/// worker that keeps issuing tool-use turns terminates on `max_turns`
/// deterministically rather than panicking on an out-of-range index.
pub struct FakeProvider {
    id: ProviderId,
    turns: Vec<Vec<StreamEvent>>,
    call_index: AtomicUsize,
}

impl FakeProvider {
    pub fn new(id: &str, turns: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            id: ProviderId::new(id),
            turns,
            call_index: AtomicUsize::new(0),
        }
    }

    /// Number of `create_message_stream` calls so far.
    pub fn call_count(&self) -> usize {
        self.call_index.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for FakeProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "fake"
    }

    async fn create_message(&self, _req: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        Err(ProviderError::Other {
            provider: self.id.clone(),
            message: "FakeProvider: use create_message_stream".into(),
            status: None,
            body: None,
        })
    }

    async fn create_message_stream(
        &self,
        _req: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError> {
        let idx = self.call_index.fetch_add(1, Ordering::SeqCst);
        let events = self
            .turns
            .get(idx)
            .or_else(|| self.turns.last())
            .cloned()
            .unwrap_or_default();
        let stream = iter(events.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        Ok(vec![])
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        Ok(ProviderStatus::Healthy)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: false,
            image_input: false,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: false,
            system_prompt_style: SystemPromptStyle::TopLevel,
        }
    }
}
```

- [ ] **Step 2: Write the failing isolation test**

`crates/research/tests/fake_provider.rs`:
```rust
//! Isolation test for the FakeProvider itself (independent of the worker).

mod common;

use std::pin::Pin;

use claurst_api::{ProviderRequest, StreamEvent, StopReason};
use claurst_core::types::{ContentBlock, Message, UsageInfo};
use futures::Stream;
use futures::StreamExt;

use common::fake_provider::FakeProvider;

fn dummy_request() -> ProviderRequest {
    ProviderRequest {
        model: "fake-model".into(),
        messages: vec![Message::user("hi")],
        system_prompt: None,
        tools: vec![],
        max_tokens: 1024,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: None,
        provider_options: serde_json::json!({}),
    }
}

#[tokio::test]
async fn test_fake_provider_emits_scripted_events() {
    let turn = vec![
        StreamEvent::MessageStart {
            id: "m1".into(),
            model: "fake-model".into(),
            usage: UsageInfo::default(),
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text { text: String::new() },
        },
        StreamEvent::TextDelta { index: 0, text: "hello".into() },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::MessageDelta {
            stop_reason: Some(StopReason::EndTurn),
            usage: Some(UsageInfo::default()),
        },
        StreamEvent::MessageStop,
    ];
    let provider = FakeProvider::new("fake", vec![turn]);
    assert_eq!(provider.call_count(), 0);

    let stream: Pin<Box<dyn Stream<Item = _> + Send>> =
        provider.create_message_stream(dummy_request()).await.unwrap();
    futures::pin_mut!(stream);
    let mut collected = Vec::new();
    while let Some(item) = stream.next().await {
        collected.push(item.unwrap());
    }
    assert_eq!(collected.len(), 6);
    assert!(matches!(collected[0], StreamEvent::MessageStart { .. }));
    assert!(matches!(collected[2], StreamEvent::TextDelta { text: _, .. }));
    assert!(matches!(collected[5], StreamEvent::MessageStop));
    assert_eq!(provider.call_count(), 1);
}

#[tokio::test]
async fn test_fake_provider_repeats_last_turn_when_exhausted() {
    let turn = vec![StreamEvent::MessageStop];
    let provider = FakeProvider::new("fake", vec![turn]);
    for _ in 0..3 {
        let mut stream = provider.create_message_stream(dummy_request()).await.unwrap();
        while stream.next().await.is_some() {}
    }
    assert_eq!(provider.call_count(), 3);
}
```

- [ ] **Step 3: Run the test to verify it compiles and passes**

Run: `cargo test -p megaresearcher-research --test fake_provider 2>&1 | tail -20`
Expected: PASS — 2 passed.

**If an import path fails to resolve:** the trait method signatures reference types whose re-export paths are verified at the crate root. If `claurst_api::ModelInfo`, `claurst_api::ProviderResponse`, `claurst_api::ProviderCapabilities`, or `claurst_api::SystemPromptStyle` do not resolve at the root, fall back to their origin modules (`claurst_api::provider::{LlmProvider, ModelInfo}`, `claurst_api::provider_types::{ProviderRequest, ProviderResponse, ProviderCapabilities, ProviderStatus, StreamEvent, StopReason, SystemPromptStyle}`). Do not change the type names or the trait method bodies — only the `use` paths. `ProviderId` is `claurst_core::provider_id::ProviderId` with `ProviderId::new(impl Into<String>)` (confirmed).

- [ ] **Step 4: fmt + clippy hygiene**

Run:
```bash
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research -- -D warnings 2>&1 | tail -5
```
Expected: fmt clean; clippy exit 0. (Clippy covers test targets under `--all-targets` semantics when run with `-p` plus the workspace flag; if clippy reports nothing for the `-p` crate alone, the Task 6 workspace sweep is the net.)

- [ ] **Step 5: Commit**

```bash
git add crates/research/tests/common/mod.rs crates/research/tests/common/fake_provider.rs crates/research/tests/fake_provider.rs
git commit -m "$(cat <<'EOF'
feat(rs): add FakeProvider test helper — canned StreamEvent turns (phase 3)

Deterministic LlmProvider impl emitting scripted Vec<Vec<StreamEvent>>; re-emits
the last turn once exhausted so looping tool-use workers terminate on max_turns.
Lives in tests/common/ (cargo's shared test-helper pattern) for Phase 4 reuse.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: worker — the query loop + worker-contract tests

**Files:**
- Modify: `crates/research/src/worker.rs`
- Create: `crates/research/tests/worker.rs`

**Interfaces:**
- Consumes (from Task 3): `crate::worker_tools::{Tool, ToolResult}`; (from Task 4 / api): `claurst_api::{LlmProvider, ProviderRequest, ProviderError, StreamEvent, StopReason, SystemPrompt}`; `claurst_core::types::{ContentBlock, Message, ToolDefinition, ToolResultContent, UsageInfo}`; `futures::{Stream, StreamExt}`; `std::sync::Arc`, `std::pin::Pin`, `std::collections::BTreeMap`, `std::path::PathBuf`.
- Produces:
  - `pub const DEFAULT_MAX_TURNS: u32 = 50;`
  - `pub struct WorkerConfig { pub max_turns: u32, pub max_tokens: u32, pub model: String }` + `impl Default` (max_turns 50, max_tokens 4096, model "claude-sonnet-4-6")
  - `#[derive(Debug, Clone, PartialEq, Eq)] pub enum WorkerStop { EndTurn, MaxTurns }`
  - `#[derive(Debug, Clone)] pub enum WorkerError { Provider(ProviderError), BadStream(String) }`
  - `#[derive(Debug, Clone)] pub struct WorkerOutcome { pub final_text: String, pub turns: u32, pub stop: WorkerStop, pub usage: UsageInfo }`
  - `pub struct Worker { pub system_prompt: String, pub tools: Vec<Arc<dyn Tool>>, pub provider: Arc<dyn LlmProvider>, pub config: WorkerConfig, pub output_dir: PathBuf }`
  - `Worker::new(system_prompt: impl Into<String>, tools: Vec<Arc<dyn Tool>>, provider: Arc<dyn LlmProvider>, config: WorkerConfig, output_dir: impl Into<PathBuf>) -> Self`
  - `Worker::run(&self, prompt: &str) -> Result<WorkerOutcome, WorkerError>`

**Loop semantics (spec §6, first cut):** for `turn in 0..max_turns`: build a `ProviderRequest` (struct literal: `model`, `messages`, `system_prompt: Some(SystemPrompt::Text(self.system_prompt.clone()))`, `tools: tool_defs`, `max_tokens`, all temperature/top_p/top_k/thinking/stop_sequences empty, `provider_options: json!({})`) → `provider.create_message_stream(req)` → `accumulate` the stream into `(Vec<ContentBlock>, Option<StopReason>, Option<UsageInfo>)` → extract any `ToolUse { id, name, input }` blocks → push an assistant message with all accumulated blocks → if no tool uses, return `WorkerOutcome { final_text: last_text, turns: turn+1, stop: EndTurn }`; else dispatch each tool use (find tool by `name()`, else `ToolResult::err(<tool_use_error>unknown tool: {name}</tool_use_error>)`), push a user message of `ToolResult` blocks, loop. If the loop exhausts `max_turns`, return `stop: MaxTurns, turns: max_turns`.

**`accumulate` semantics:** consume the stream via `StreamExt::next`; track `ContentBlockStart` blocks by index in a `BTreeMap<usize, ContentBlock>` (the seed block), accumulate `TextDelta` text and `InputJsonDelta` partial-json into per-index `String` buffers; on `ContentBlockStop { index }`, merge the accumulated text into a `Text { text }` block or parse the accumulated json into a `ToolUse { input }` block (`serde_json::from_str::<Value>`; on parse failure leave the seed input), collect finalized blocks into a `BTreeMap<usize, ContentBlock>` keyed by index (so `into_values()` yields ascending-index order); capture `MessageDelta { stop_reason, usage }`; break on `MessageStop`; map `StreamEvent::Error { message, .. }` to `WorkerError::BadStream`. Ignore `ThinkingDelta`/`SignatureDelta`/`ReasoningDelta` (deferred — no thinking config in Phase 3).

- [ ] **Step 1: Write the failing worker-contract tests**

`crates/research/tests/worker.rs`:
```rust
//! Worker-contract tests: a deterministic fake provider drives the query loop.

mod common;

use std::sync::Arc;

use claurst_api::{LlmProvider, StopReason, StreamEvent};
use claurst_core::types::{ContentBlock, UsageInfo};
use serde_json::json;

use common::fake_provider::FakeProvider;
use megaresearcher_research::worker::{Worker, WorkerConfig, WorkerStop};
use megaresearcher_research::worker_tools::{ScopedWrite, Tool};

/// A turn that calls Write(file, content) and stops with ToolUse.
fn write_turn(file: &str, content: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart {
            id: "m".into(),
            model: "fake".into(),
            usage: UsageInfo::default(),
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text { text: String::new() },
        },
        StreamEvent::TextDelta {
            index: 0,
            text: format!("writing {file}").into(),
        },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::ContentBlockStart {
            index: 1,
            content_block: ContentBlock::ToolUse {
                id: format!("tu_{file}"),
                name: "Write".into(),
                input: json!({ "file_path": file, "content": content }),
            },
        },
        StreamEvent::ContentBlockStop { index: 1 },
        StreamEvent::MessageDelta {
            stop_reason: Some(StopReason::ToolUse),
            usage: Some(UsageInfo::default()),
        },
        StreamEvent::MessageStop,
    ]
}

/// A turn with only a Text block, stopping with EndTurn.
fn final_turn(text: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart {
            id: "m".into(),
            model: "fake".into(),
            usage: UsageInfo::default(),
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text { text: String::new() },
        },
        StreamEvent::TextDelta { index: 0, text: text.into() },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::MessageDelta {
            stop_reason: Some(StopReason::EndTurn),
            usage: Some(UsageInfo::default()),
        },
        StreamEvent::MessageStop,
    ]
}

#[tokio::test]
async fn test_worker_writes_three_artifacts_and_ends() {
    let out = tempfile::tempdir().unwrap();
    let write = Arc::new(ScopedWrite::new(out.path())) as Arc<dyn Tool>;
    let turns = vec![
        write_turn("output.md", "# Output\n\nThe bibliography."),
        write_turn("manifest.yaml", "role: literature-scout\npapers_count: 8\n"),
        write_turn("verification.md", "# Verification\n\nAll checks passed."),
        final_turn("Done. All three artifacts written."),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let worker = Worker::new(
        "You are a literature scout.",
        vec![write],
        provider,
        WorkerConfig {
            max_turns: 10,
            max_tokens: 1024,
            model: "fake-model".into(),
        },
        out.path(),
    );
    let outcome = worker
        .run("Survey the topic. Write output.md, manifest.yaml, verification.md.")
        .await
        .unwrap();

    assert_eq!(outcome.stop, WorkerStop::EndTurn);
    assert_eq!(outcome.turns, 4);
    assert_eq!(outcome.final_text, "Done. All three artifacts written.");
    assert_eq!(
        std::fs::read_to_string(out.path().join("output.md")).unwrap(),
        "# Output\n\nThe bibliography."
    );
    assert_eq!(
        std::fs::read_to_string(out.path().join("manifest.yaml")).unwrap(),
        "role: literature-scout\npapers_count: 8\n"
    );
    assert_eq!(
        std::fs::read_to_string(out.path().join("verification.md")).unwrap(),
        "# Verification\n\nAll checks passed."
    );
    assert_eq!(fake.call_count(), 4);
}

#[tokio::test]
async fn test_worker_terminates_on_max_turns() {
    let out = tempfile::tempdir().unwrap();
    let write = Arc::new(ScopedWrite::new(out.path())) as Arc<dyn Tool>;
    // A single tool-use turn the fake repeats every call.
    let turn = write_turn("repeat.md", "x");
    let fake = Arc::new(FakeProvider::new("fake", vec![turn]));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let worker = Worker::new(
        "sys",
        vec![write],
        provider,
        WorkerConfig {
            max_turns: 2,
            max_tokens: 1024,
            model: "fake-model".into(),
        },
        out.path(),
    );
    let outcome = worker.run("write repeatedly").await.unwrap();
    assert_eq!(outcome.stop, WorkerStop::MaxTurns);
    assert_eq!(outcome.turns, 2);
    assert_eq!(fake.call_count(), 2);
    assert!(out.path().join("repeat.md").exists());
}

#[tokio::test]
async fn test_worker_unknown_tool_is_error_block_not_panic() {
    let out = tempfile::tempdir().unwrap();
    // No tools wired, but the fake requests a "Write" call.
    let turns = vec![
        vec![
            StreamEvent::MessageStart {
                id: "m".into(),
                model: "fake".into(),
                usage: UsageInfo::default(),
            },
            StreamEvent::ContentBlockStart {
                index: 0,
                content_block: ContentBlock::ToolUse {
                    id: "tu_1".into(),
                    name: "Write".into(),
                    input: json!({"file_path":"x.md","content":"x"}),
                },
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: Some(UsageInfo::default()),
            },
            StreamEvent::MessageStop,
        ],
        final_turn("Recovered after the missing tool."),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let worker = Worker::new(
        "sys",
        vec![],
        provider,
        WorkerConfig {
            max_turns: 10,
            max_tokens: 1024,
            model: "fake-model".into(),
        },
        out.path(),
    );
    let outcome = worker.run("use a tool i do not have").await.unwrap();
    // The worker did not panic; it dispatched an error tool_result and the next
    // turn ended cleanly.
    assert_eq!(outcome.stop, WorkerStop::EndTurn);
    assert_eq!(outcome.final_text, "Recovered after the missing tool.");
    // The unknown tool never wrote anything.
    assert!(!out.path().join("x.md").exists());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p megaresearcher-research --test worker 2>&1 | tail -20`
Expected: FAIL — `Worker`, `WorkerConfig`, `WorkerStop` not found (module body empty).

- [ ] **Step 3: Implement worker**

`crates/research/src/worker.rs`:
```rust
//! The in-process worker primitive: a fresh-context query loop against an `LlmProvider`.
//!
//! First-cut loop (spec §6), dropping autocompact / normalization / abort:
//! build a ProviderRequest -> stream -> accumulate blocks -> dispatch tool_use
//! -> append tool_result -> loop, bounded by max_turns. The result is the
//! final assistant text only.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt};
use serde_json::Value;

use claurst_api::{LlmProvider, ProviderError, ProviderRequest, StreamEvent, StopReason, SystemPrompt};
use claurst_core::types::{ContentBlock, Message, ToolDefinition, ToolResultContent, UsageInfo};

use crate::worker_tools::{Tool, ToolResult};

/// Default turn ceiling for a worker. Tunable per worker via `WorkerConfig`.
pub const DEFAULT_MAX_TURNS: u32 = 50;

/// Per-worker configuration.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub max_turns: u32,
    pub max_tokens: u32,
    pub model: String,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            max_turns: DEFAULT_MAX_TURNS,
            max_tokens: 4096,
            model: "claude-sonnet-4-6".to_string(),
        }
    }
}

/// Why the worker stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStop {
    /// The provider ended the turn with no pending tool calls.
    EndTurn,
    /// The turn ceiling was reached with tool calls still pending.
    MaxTurns,
}

/// Worker-level error.
#[derive(Debug, Clone)]
pub enum WorkerError {
    Provider(ProviderError),
    BadStream(String),
}

/// The outcome of a worker run.
#[derive(Debug, Clone)]
pub struct WorkerOutcome {
    /// The final assistant text (the worker's result, per spec §6).
    pub final_text: String,
    /// Number of provider turns executed.
    pub turns: u32,
    /// Why the loop stopped.
    pub stop: WorkerStop,
    /// Accumulated usage from the final turn.
    pub usage: UsageInfo,
}

/// A worker: a system prompt, a tool set, a provider, and a turn ceiling.
pub struct Worker {
    pub system_prompt: String,
    pub tools: Vec<Arc<dyn Tool>>,
    pub provider: Arc<dyn LlmProvider>,
    pub config: WorkerConfig,
    pub output_dir: PathBuf,
}

impl Worker {
    pub fn new(
        system_prompt: impl Into<String>,
        tools: Vec<Arc<dyn Tool>>,
        provider: Arc<dyn LlmProvider>,
        config: WorkerConfig,
        output_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            tools,
            provider,
            config,
            output_dir: output_dir.into(),
        }
    }

    fn tool_defs(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    /// Run the worker against a single user prompt. Returns the final assistant
    /// text plus stop metadata.
    pub async fn run(&self, prompt: &str) -> Result<WorkerOutcome, WorkerError> {
        let mut messages: Vec<Message> = vec![Message::user(prompt.to_string())];
        let tool_defs = self.tool_defs();
        let mut last_assistant_text = String::new();
        let mut usage = UsageInfo::default();

        for turn in 0..self.config.max_turns {
            let req = ProviderRequest {
                model: self.config.model.clone(),
                messages: messages.clone(),
                system_prompt: Some(SystemPrompt::Text(self.system_prompt.clone())),
                tools: tool_defs.clone(),
                max_tokens: self.config.max_tokens,
                temperature: None,
                top_p: None,
                top_k: None,
                stop_sequences: vec![],
                thinking: None,
                provider_options: Value::Object(serde_json::Map::new()),
            };
            let stream = self
                .provider
                .create_message_stream(req)
                .await
                .map_err(WorkerError::Provider)?;
            let (blocks, _stop_reason, turn_usage) = accumulate(stream).await?;
            if let Some(u) = turn_usage {
                usage = u;
            }

            // Pull text + tool_use out of the finalized blocks, preserving order.
            let mut tool_uses: Vec<(String, String, Value)> = Vec::new();
            for block in &blocks {
                match block {
                    ContentBlock::Text { text } => {
                        if !text.is_empty() {
                            last_assistant_text = text.clone();
                        }
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_uses.push((id.clone(), name.clone(), input.clone()));
                    }
                    _ => {}
                }
            }

            messages.push(Message::assistant_blocks(blocks));

            if tool_uses.is_empty() {
                return Ok(WorkerOutcome {
                    final_text: last_assistant_text,
                    turns: turn + 1,
                    stop: WorkerStop::EndTurn,
                    usage,
                });
            }

            // Dispatch each tool use and collect ToolResult blocks.
            let mut results: Vec<ContentBlock> = Vec::new();
            for (id, name, input) in tool_uses {
                let result = match self.tools.iter().find(|t| t.name() == name) {
                    Some(tool) => tool.call(input).await,
                    None => ToolResult::err(format!(
                        "<tool_use_error>unknown tool: {name}</tool_use_error>"
                    )),
                };
                results.push(ContentBlock::ToolResult {
                    tool_use_id: id,
                    content: ToolResultContent::Text(result.content),
                    is_error: Some(result.is_error),
                });
            }
            messages.push(Message::user_blocks(results));
        }

        Ok(WorkerOutcome {
            final_text: last_assistant_text,
            turns: self.config.max_turns,
            stop: WorkerStop::MaxTurns,
            usage,
        })
    }
}

/// Accumulate a stream of `StreamEvent`s into finalized `ContentBlock`s in
/// ascending block-index order, plus the final stop reason and usage.
async fn accumulate(
    stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
) -> Result<(Vec<ContentBlock>, Option<StopReason>, Option<UsageInfo>), WorkerError> {
    let mut seeds: BTreeMap<usize, ContentBlock> = BTreeMap::new();
    let mut text_bufs: BTreeMap<usize, String> = BTreeMap::new();
    let mut json_bufs: BTreeMap<usize, String> = BTreeMap::new();
    let mut finished: BTreeMap<usize, ContentBlock> = BTreeMap::new();
    let mut stop_reason: Option<StopReason> = None;
    let mut usage: Option<UsageInfo> = None;

    let mut stream = stream;
    while let Some(event) = stream.next().await {
        let event = event.map_err(WorkerError::Provider)?;
        match event {
            StreamEvent::MessageStart { .. } => {}
            StreamEvent::ContentBlockStart { index, content_block } => {
                seeds.insert(index, content_block);
            }
            StreamEvent::TextDelta { index, text } => {
                text_bufs.entry(index).or_default().push_str(&text);
            }
            StreamEvent::InputJsonDelta { index, partial_json } => {
                json_bufs.entry(index).or_default().push_str(&partial_json);
            }
            StreamEvent::ContentBlockStop { index } => {
                if let Some(mut block) = seeds.remove(&index) {
                    match &mut block {
                        ContentBlock::Text { text } => {
                            if let Some(buf) = text_bufs.remove(&index) {
                                *text = buf;
                            }
                        }
                        ContentBlock::ToolUse { input, .. } => {
                            if let Some(buf) = json_bufs.remove(&index) {
                                if let Ok(value) = serde_json::from_str::<Value>(&buf) {
                                    *input = value;
                                }
                            }
                        }
                        _ => {}
                    }
                    finished.insert(index, block);
                }
            }
            StreamEvent::MessageDelta {
                stop_reason: sr,
                usage: u,
            } => {
                if let Some(sr) = sr {
                    stop_reason = Some(sr);
                }
                if let Some(u) = u {
                    usage = Some(u);
                }
            }
            StreamEvent::MessageStop => break,
            StreamEvent::Error { message, .. } => {
                return Err(WorkerError::BadStream(message));
            }
            // ThinkingDelta / SignatureDelta / ReasoningDelta: ignored (no
            // thinking config in Phase 3).
            _ => {}
        }
    }

    let blocks: Vec<ContentBlock> = finished.into_values().collect();
    Ok((blocks, stop_reason, usage))
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test worker 2>&1 | tail -20`
Expected: PASS — 3 passed.

**If `claurst_api::SystemPrompt` does not resolve:** it is re-exported via `pub use types::*;` and `pub use provider_types::*;` at the crate root. If your toolchain still cannot find it there, import from `claurst_api::types::SystemPrompt`. Do not change the variant (`SystemPrompt::Text(body)`) — that is confirmed.

- [ ] **Step 5: fmt + clippy hygiene**

Run:
```bash
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research -- -D warnings 2>&1 | tail -5
```
Expected: fmt clean; clippy exit 0. If clippy flags the `_stop_reason` unused binding, keep the leading underscore (it is intentionally unused — the worker dispatches on tool presence, not stop reason, for the first cut) — clippy accepts `_`-prefixed unused bindings under `-D warnings`.

- [ ] **Step 6: Commit**

```bash
git add crates/research/src/worker.rs crates/research/tests/worker.rs
git commit -m "$(cat <<'EOF'
feat(rs): add research::worker — fresh-context query loop (phase 3)

Worker::run builds a ProviderRequest (struct literal, SystemPrompt::Text =
agent body), streams via LlmProvider, accumulates StreamEvents into assistant
ContentBlocks, dispatches ToolUse to the scoped tools, appends ToolResult
user messages, and loops — bounded by max_turns (default 50). Returns the
final assistant text. Worker-contract tests cover the three-artifact write,
max_turns termination, and unknown-tool error-block recovery.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Phase 3 green sweep from repo root

**Files:**
- Read-only verification; ledger append to `.superpowers/sdd/progress.md`.

**Interfaces:** n/a (verification task).

- [ ] **Step 1: fmt check from root**

Run: `cargo fmt --all --check 2>&1 | tail -5`
Expected: exit 0, no output (FMT CLEAN).

- [ ] **Step 2: clippy from root, all targets**

Run: `cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -5`
Expected: exit 0 (CLIPPY CLEAN). If any lint fires in new Phase 3 code, fix behavior-preserving and re-run; if a lint fires in pre-existing Phase 1/2 or claurst code, that is out of Phase 3 scope — record it in the report but do not fix it here.

- [ ] **Step 3: full test suite from root**

Run: `cargo test --workspace 2>&1 | tail -20`
Expected: all pass. Phase 2 baseline was 1398 passed / 0 failed. Phase 3 adds: prompt_asset (3), worker_tools (10), fake_provider (2), worker (3) = 18 new tests, for an expected 1416 passed / 0 failed (± any claurst-baseline drift, which should be zero — Phase 3 touches no claurst code). Record the actual count.

- [ ] **Step 4: confirm v0 port-reference files untouched**

Run: `git diff --stat <phase2-tip>..HEAD -- lib/ tests/test_*.py skills/ agents/`
where `<phase2-tip>` is the commit before Task 1's scaffold commit (use `git log --oneline -8` to find it). Expected: empty diff (Phase 3 added no files under those paths; the repo-root `agents/literature-scout.md` was only *read* and *copied into* `crates/research/tests/fixtures/agents/`, not modified).

- [ ] **Step 5: confirm research-crate file structure**

Run:
```bash
ls crates/research/src/
ls crates/research/tests/
```
Expected:
- `src/`: includes `prompt_asset.rs`, `worker.rs`, `worker_tools.rs` alongside the Phase 2 `paper_chain/`, `state/`, `lib.rs`.
- `tests/`: includes `prompt_asset.rs`, `worker_tools.rs`, `fake_provider.rs`, `worker.rs`, `common/`, `fixtures/agents/literature-scout.md` alongside the Phase 2 test files and `smoke.rs`.

- [ ] **Step 6: append the Phase 3 ledger**

Append to `.superpowers/sdd/progress.md` (one block):

```markdown
## Phase 3 — Worker primitive + fake provider

- Task 1: complete (commits <base7>..<head7>, scaffold prompt_asset/worker/worker_tools empty modules).
- Task 2: complete (commits <base7>..<head7>, prompt_asset parse/load + verbatim literature-scout.md fixture; 3 passed).
- Task 3: complete (commits <base7>..<head7>, Tool trait + ScopedRead/ScopedWrite hard jail + check_artifacts; 10 passed).
- Task 4: complete (commits <base7>..<head7>, FakeProvider in tests/common/; 2 passed).
- Task 5: complete (commits <base7>..<head7>, worker query loop + worker-contract tests; 3 passed).
- Task 6: complete — fmt/clippy/test green from root; v0 port-reference files untouched; research crate has prompt_asset/worker/worker_tools + 4 new test files + common/ + fixtures/agents/.
=== Phase 3: ALL 6 TASKS COMPLETE ===
```

(Replace the `<base7>..<head7>` tokens with the actual short SHAs from `git log --oneline` for each task's commit range. If Task 6 itself produces no commit — it is a verification task — record the SHA of the last Task 5 commit as the phase tip.)

- [ ] **Step 7: report**

Write `.superpowers/sdd/task-6-report.md` recording: fmt result, clippy result, the actual `cargo test --workspace` passed/failed counts, the v0-untouched confirmation, and the file-structure confirmation. This is the phase exit report.

---

## Self-Review

**1. Spec coverage.** Spec §6 (worker primitive, first cut): Task 5 implements the loop (build req → stream → accumulate → dispatch tool_use → append tool_result → loop on tool presence → return final text), bounded by `max_turns` (Task 5 `DEFAULT_MAX_TURNS=50`, resolving §15). Deferred stages (normalize, autocompact, abort) are explicitly cut and named in Global Constraints — matching spec §6's "first cut". ✓ Spec §7 (tools): Task 3 ships the `Tool` trait + `ScopedRead`/`ScopedWrite` with the hard path jail (absolute + `..` rejection before I/O), no Bash, `<tool_use_error>` envelope, `is_error` on deny/throw/invalid, `tool_result` as a user message of `ToolResult` blocks (Task 5 wires the user-message append). The full permission engine (deny/ask/modes) is deferred — named. ✓ Spec §8/§13.3 (prompt-asset loader): Task 2 parses v0 agent files via `serde_yml` (handles `description: |`), body = system prompt, description = routing key; explicitly does NOT reuse `parse_skill_file`. ✓ Spec §11 (artifact gate): Task 3 `check_artifacts` returns missing names; the redispatch-once-then-escalate policy is the orchestrator's (Phase 4) — named. ✓ Spec §12 (testing): Tasks 2/3/4/5 cover prompt-asset parse, scoped Read/Write jail + `check_artifacts` flagging a missing manifest, fake-provider isolation, and worker-contract (writes-3-artifacts, max_turns, unknown-tool) — all deterministic, no network. ✓ Spec §13.3 ("Worker primitive + fake provider … Worker-contract tests green"): Tasks 1–6 deliver exactly that. ✓ Spec §15 open questions (provider abstraction, max_turns, HTML export, Vercel client): provider abstraction = keep `LlmProvider` (named), max_turns = 50 (named), HTML export = Phase 8 (out of scope, named), Vercel client = Phase 7 (out of scope, named). ✓

**2. Placeholder scan.** No "TBD/TODO/implement later". The only soft spots are the import-path fallback notes in Task 4 Step 3 and Task 5 Step 4 — these name exact alternative module paths, not placeholders; the code is complete. The `<base7>..<head7>` tokens in the Task 6 ledger are explicitly "replace with actual SHAs from `git log`" — a fill-in instruction for a verification task, not a code placeholder. No step says "add error handling" or "write tests for the above" without code.

**3. Type consistency.** `ToolResult { content, is_error }` (Task 3) consumed in Task 5's worker dispatch and the unknown-tool branch — same field names. `Tool::call(&self, Value) -> ToolResult` (Task 3) matches Task 5's `tool.call(input).await`. `ScopedWrite::new(dir)` / `ScopedRead::new(dir)` / `ScopedRead::with_shared(dir, shared)` (Task 3) match the Task 5 test construction. `FakeProvider::new(id, turns)` / `call_count()` (Task 4) match Task 5's `fake.clone() as Arc<dyn LlmProvider>` and `fake.call_count()`. `Worker::new(system_prompt, tools, provider, config, output_dir)` / `WorkerConfig { max_turns, max_tokens, model }` / `WorkerStop::{EndTurn, MaxTurns}` / `WorkerOutcome { final_text, turns, stop, usage }` (Task 5) match the Task 5 tests. `ProviderRequest` field list (Task 4 dummy_request, Task 5 run) matches the verified struct shape. `StreamEvent` variants used (MessageStart, ContentBlockStart, TextDelta, InputJsonDelta, ContentBlockStop, MessageDelta, MessageStop, Error) match the verified enum. `ContentBlock::{Text, ToolUse, ToolResult}` and `ToolResultContent::Text` and `Message::user/assistant_blocks/user_blocks` match the verified claurst-core types. `SystemPrompt::Text(body)`, `ProviderId::new`, `ProviderError::Other { provider, message, status, body }`, `ProviderStatus::Healthy`, `SystemPromptStyle::TopLevel`, `ProviderCapabilities` field set, `ModelInfo` return type — all match the verified api crate. No drift between tasks.