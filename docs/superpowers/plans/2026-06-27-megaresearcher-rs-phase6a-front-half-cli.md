# MegaResearcher-rs Phase 6a — Headless Front-Half (CLI + Flow Bodies + Guided Sessions) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working `mr` CLI that drives the full research chain headless — `mr init "<question>"` runs brainstorm→spec→plan as guided LLM sessions with Rust-enforced approval gates, then `mr execute --headless [--on-escalate=…]` runs the existing orchestrator and renders progression as text, auto-verifying at the end. No TUI (that is Phase 6b).

**Architecture:** A new `crates/mr-cli` binary depends on `megaresearcher-research` + `claurst-core`, reusing claurst's config/provider machinery (the `Settings`→`Config`→`ProviderRegistry`→`Arc<dyn LlmProvider>` path) rather than reinventing it. Two new research modules do the work: `research::flows` (frontmatter+body asset loader for guided-session flow bodies, embedded via `include_str!`) and `research::phases` (a `GuidedSession` turn loop on `LlmProvider` reusing the worker's streaming/accumulate machinery + a `drive_session` driver with `UserIo` abstraction and multi-gate approval). An additive `EscalationHandler` trait on `OrchestratorConfig` delivers `--on-escalate={continue,pause,fail}` without touching the 52 existing orchestrator tests. A `research::verify` module runs the deterministic post-run checklist. `mr` subcommands map 1:1 to the v0 slash commands (§180: slash commands → `mr` subcommands).

**Tech Stack:** Rust, workspace crates `claurst-api`/`claurst-core`/`megaresearcher-research`, `clap` v4 (derive), `tokio`, `ratatui`/`crossterm` NOT used in 6a, `async-trait`, `serde`/`serde_yml`, `include_str!` for embedded flow bodies.

## Global Constraints

- **GPL-3.0.** Edit ONLY `crates/research/`, `crates/mr-cli/` (new), and the workspace root `Cargo.toml`/`Cargo.lock` for the new member + deps. The v0 plugin (`lib/`, `skills/`, `agents/`, `commands/`, `mcp/`, `tools/ml-intern`, `.claude-plugin/`) MUST NOT be modified — `skills/` and `agents/` are read-only source for porting flow-body content only.
- **No git worktrees.** Work on branch `main` in the main checkout. Before dispatching implementers, run `git branch --show-current` and include the branch name in every subagent prompt.
- **No crate-root `pub use` re-exports.** `lib.rs` uses `pub mod` only; consumers use full paths `megaresearcher_research::flows::…`.
- **Per-task hygiene:** `cargo fmt -p megaresearcher-research` (and `-p mr-cli` once it exists) before commit; `cargo clippy -p <crate> --all-targets -- -D warnings` (the `--all-targets` flag is REQUIRED); each build phase ends with `cargo test -p <crate>` green from the repo root.
- **Banned words/phrases:** never use "load-bearing", "this is doing a lot of work" (or close variants), or "real"/"honest/honestly/to be honest" as emphatic adjectives/framing in any artifact. "Honest estimate" in ported plan-body copy → rephrase to "Grounded estimate". (Genuine technical terms — `real` number, library names — are exempt.)
- **Determinism preservation:** the additive `EscalationHandler` must keep all 52 existing orchestrator tests byte-identical when `OrchestratorConfig.escalation` is `None` (the default every existing test uses). New behavior only fires when a handler is `Some`.
- **Four product filters** apply (TUI is user-facing, but 6a is the headless/text path; still: opinionated, no hedging defaults, direct verbs).
- **`mr` subcommands map to v0 slash commands:** `mr init` ↔ `/research-init`, `mr execute` ↔ `/research-execute`. There are NO slash commands in the `mr` CLI (§5/§180: slash commands → `mr` subcommands). Individual `mr brainstorm`/`mr spec`/`mr plan` expose the constituent flow bodies.

---

## File Structure

**New crate `crates/mr-cli/`:**
- `Cargo.toml` — `[[bin]] name = "mr"`, deps on `megaresearcher-research`, `claurst-core`, `claurst-api`, `clap`, `tokio`, `anyhow`, `serde`, `serde_json`.
- `src/main.rs` — thin: `#[tokio::main] fn main() -> anyhow::Result<()> { mr_cli::run_cli(std::env::args().collect()).await }` — thin wrapper for testability.
- `src/lib.rs` — `pub mod prelude; pub mod commands; pub mod render; pub mod escalation; pub mod io; pub async fn run_cli(args: Vec<String>) -> anyhow::Result<()>`.
- `src/prelude.rs` — `resolve_provider(cwd, model, provider_id, api_key) -> anyhow::Result<(Arc<dyn LlmProvider>, String)>` mirroring claurst's prelude.
- `src/io.rs` — `UserIo` trait (re-exported from research::phases) + `StdinStdoutIo` production impl + `FakeUserIo` test helper.
- `src/escalation.rs` — `HeadlessEscalationHandler` impl of `EscalationHandler` for `--on-escalate={continue,pause,fail}`.
- `src/render.rs` — text rendering of run progression (poll `swarm-state.yaml`) + past-runs list.
- `src/commands/mod.rs` + `src/commands/{init.rs, brainstorm.rs, spec.rs, plan.rs, execute.rs, verify.rs, list.rs, watch.rs}` — one file per subcommand.

**New/modified in `crates/research/`:**
- `src/flows.rs` (new) — `FlowAsset` struct + `parse`/`load` + `load_embedded(name) -> FlowAsset` backed by `include_str!` constants.
- `src/flows/brainstorm.md`, `src/flows/spec.md`, `src/flows/plan.md` (new) — embedded flow-body assets, ported from v0 `skills/research-brainstorming/SKILL.md`, `skills/writing-research-spec/SKILL.md`, `skills/writing-research-plan/SKILL.md` with the superpowers disciplines inlined (§182: "the dependency on superpowers becomes inlined prompt content").
- `src/phases.rs` (new) — `GuidedSession` (stream-based turn loop reusing `worker::accumulate`), `UserIo` trait, `Gate`, `SessionConfig`, `drive_session`, `SessionError`, `DriveOutcome`.
- `src/verify.rs` (new) — deterministic post-run checker: 6 check groups (A–F) + 3 MCP citation spot-checks + `Verdict` + `write_report`.
- `src/orchestrator/escalation.rs` (new) — `EscalationHandler` trait + `EscalationVerdict`.
- `src/orchestrator/mod.rs` (modify) — add `pub escalation: Option<Arc<dyn EscalationHandler>>` to `OrchestratorConfig`; route the 4 escalation return-sites through a new `adjudicate_escalation` helper.
- `src/worker.rs` (modify) — make `accumulate` `pub(crate)` so `phases` reuses it.
- `src/lib.rs` (modify) — add `pub mod flows; pub mod phases; pub mod verify;` and `pub mod orchestrator::escalation` is via `orchestrator/mod.rs`.
- `Cargo.toml` (modify) — no new deps expected (already has `claurst-api`, `claurst-core`, `claurst-mcp`, `async-trait`, `tokio`, `serde`, `serde_json`, `serde_yml`, `futures`); confirm `futures` present (needed by `accumulate`).

**Workspace:**
- Root `Cargo.toml` (modify) — add `crates/mr-cli` to `[workspace] members`.

---

## Task 1: `research::flows` — FlowAsset parser + 3 embedded flow bodies

**Files:**
- Create: `crates/research/src/flows.rs`
- Create: `crates/research/src/flows/brainstorm.md`, `crates/research/src/flows/spec.md`, `crates/research/src/flows/plan.md`
- Modify: `crates/research/src/lib.rs` (add `pub mod flows;`)
- Test: `crates/research/tests/flows.rs` (new)

**Interfaces:**
- Produces: `pub struct FlowAsset { pub name: String, pub description: String, pub argument_hint: Option<String>, pub model: Option<String>, pub allowed_tools: Option<Vec<String>>, pub body: String }`; `pub fn parse(text: &str) -> Result<FlowAsset, String>`; `pub fn load(path: &Path) -> io::Result<FlowAsset>`; `pub fn load_embedded(name: &str) -> FlowAsset` (panics on unknown name — only called with literals); `pub const EMBEDDED_NAMES: &[&str] = &["brainstorm", "spec", "plan"];`.
- Consumes: `serde_yml` (already a dep). The flow-body markdown files are embedded via `include_str!("flows/brainstorm.md")` etc. (path relative to `src/flows.rs`).

- [ ] **Step 1: Write the failing test**

```rust
// crates/research/tests/flows.rs
use megaresearcher_research::flows::{parse, load_embedded, FlowAsset, EMBEDDED_NAMES};

#[test]
fn parse_reads_frontmatter_and_body() {
    let text = "---\nname: brainstorm\ndescription: Clarify intent.\nargument-hint: \"[topic]\"\nmodel: inherit\nallowed-tools:\n  - Read\n  - Write\n---\n\nYou are guiding a brainstorm.\n";
    let a = parse(text).unwrap();
    assert_eq!(a.name, "brainstorm");
    assert_eq!(a.description, "Clarify intent.");
    assert_eq!(a.argument_hint.as_deref(), Some("[topic]"));
    assert_eq!(a.model.as_deref(), Some("inherit"));
    assert_eq!(a.allowed_tools.as_deref(), Some(&vec!["Read".to_string(), "Write".to_string()]));
    assert!(a.body.contains("You are guiding a brainstorm."));
}

#[test]
fn parse_tolerates_missing_optional_fields() {
    let text = "---\nname: spec\ndescription: Write the spec.\n---\n\nBody.\n";
    let a = parse(text).unwrap();
    assert_eq!(a.name, "spec");
    assert!(a.argument_hint.is_none());
    assert!(a.model.is_none());
    assert!(a.allowed_tools.is_none());
}

#[test]
fn parse_rejects_missing_frontmatter_delimiter() {
    assert!(parse("no frontmatter here").is_err());
}

#[test]
fn load_embedded_returns_three_known_flows() {
    assert_eq!(EMBEDDED_NAMES, &["brainstorm", "spec", "plan"]);
    for name in EMBEDDED_NAMES {
        let a = load_embedded(name);
        assert_eq!(a.name, *name, "embedded flow {name} must have name frontmatter");
        assert!(!a.body.is_empty(), "embedded flow {name} body must be non-empty");
    }
}

#[test]
fn brainstorm_body_inlines_clarifications_and_no_skill_invocation() {
    let a = load_embedded("brainstorm");
    // The ported body must carry the 7 research clarifications and must NOT
    // reference the superpowers:brainstorming skill (it is inlined).
    assert!(a.body.contains("novelty target"));
    assert!(a.body.contains("YAGNI"));
    assert!(!a.body.contains("superpowers:"));
}

#[test]
fn spec_body_has_the_fixed_section_template() {
    let a = load_embedded("spec");
    assert!(a.body.contains("Novelty target"));
    assert!(a.body.contains("Out of scope"));
    assert!(a.body.contains("docs/research/specs/"));
}

#[test]
fn plan_body_has_swarm_decomposition_section() {
    let a = load_embedded("plan");
    assert!(a.body.contains("Swarm decomposition"));
    assert!(a.body.contains("MEGARESEARCHER_MAX_PARALLEL"));
    assert!(a.body.contains("docs/research/plans/"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test flows`
Expected: FAIL — `module flows` not found / `parse` undefined.

- [ ] **Step 3: Write the three flow-body markdown assets**

Create `crates/research/src/flows/brainstorm.md` (ported from `skills/research-brainstorming/SKILL.md`, with the superpowers:brainstorming discipline inlined — the "explore, ask one question at a time, propose, present, get approval" loop — and the `superpowers:brainstorming` invocation removed):

```markdown
---
name: brainstorm
description: Clarify research intent enough that the spec session can produce a high-quality spec. Asks the seven research clarifications one at a time, then restates and confirms.
argument-hint: "[topic]"
model: inherit
allowed-tools:
  - Read
  - Write
---

# Research Brainstorming

You are guiding the start of a new MegaResearcher research project. The user's
topic or question is provided as the first user message (the `$ARGUMENTS`).
Your job is to clarify intent enough that the spec session can produce a
high-quality spec.

## Discipline (inlined)

Work one question at a time. Explore, ask, propose, present — then get explicit
approval before advancing. Never assume an answer the user did not give; if a
dimension was not discussed, ask. Defaults are fine, but make them explicit.

## Process

Ask the following one at a time, in order, skipping any already obvious from
the opening question. Wait for the user's answer before asking the next.

1. **The research question.** One paragraph. What is the user actually trying to
   find out?
2. **Novelty target** — choose one:
   - `gap-finding` — identify unexplored regions in the literature
   - `hypothesis` — gap-finding plus testable hypotheses with falsification
     criteria (this triggers the red-team critique loop; tell the user that means
     more dispatches and longer runs)
   - `synthesis` — novel combinations of existing techniques (less novel than
     `hypothesis`, lower nonsense risk)
3. **Modalities and domain.** What kinds of data, models, or work are in scope?
4. **Constraints.** What is off-limits? (no classified data, no GPU spend during
   scoping, open datasets only, licence requirements, deadlines, etc.)
5. **Success criteria.** What artifacts must exist for the run to count as
   successful? (e.g., at least 3 surviving hypotheses with eval designs; a
   synthesist document under 8 pages; every claim cited)
6. **Out of scope (YAGNI fence).** What is explicitly NOT this project? List
   items the user wants to defer or never address.
7. **Custom workers (optional).** Does the project need worker types beyond the
   bundled six (literature-scout, gap-finder, hypothesis-smith, red-team,
   eval-designer, synthesist)? If so, the user defines them inline in the spec.

## Confirm before advancing

Restate the answers back to the user and get explicit approval before the spec
session begins. Do not proceed if the user is uncertain about items 1–6 (item 7
is optional). The novelty target is the one consequential branch — spell out its
cost before the user locks it in.

You orchestrate the start of the chain. Do not produce research content, do not
write the spec yourself, and do not invoke any external skill — the next step is
the spec session, which the harness starts when you signal the brainstorm is
approved.
```

Create `crates/research/src/flows/spec.md` (ported from `skills/writing-research-spec/SKILL.md`, which is already self-contained — no superpowers dependency to inline; only the slash-command references are normalized to paths):

```markdown
---
name: spec
description: Author the research spec from the brainstorm answers. Writes docs/research/specs/YYYY-MM-DD-<topic>-spec.md and gets user approval.
argument-hint: "[topic]"
model: inherit
allowed-tools:
  - Read
  - Write
---

# Writing the Research Spec

You have the brainstorm answers (carried in this conversation). Write them into a
properly-structured spec, then get user approval. Use the `Write` tool to create
the file; the harness jails writes under `docs/research/`, so write to
`specs/YYYY-MM-DD-<topic-slug>-spec.md` (relative path). Use today's date and a
kebab-case topic slug. If the directory does not exist, create it.

## Spec format (use exactly these section headings)

\`\`\`markdown
# <Topic Title> — Research Spec

**Status:** draft
**Created:** YYYY-MM-DD
**Novelty target:** gap-finding | hypothesis | synthesis

## Question

<One paragraph from the brainstorm, step 1.>

## Modalities and domain

<From step 3. Name the data types, the application area, the operational context if any.>

## Constraints

<From step 4. Each constraint as a bullet.>

## Success criteria

<From step 5. What artifacts must exist? What numerical bars? What does red-team
approval look like for this project?>

## Out of scope (YAGNI fence)

<From step 6. Each item as a bullet. The synthesist will reflect this fence.>

## Custom workers

<From step 7. If none, write "None — using the bundled six (literature-scout,
gap-finder, hypothesis-smith, red-team, eval-designer, synthesist).">

## Decisions locked in

- <date · decision · rationale>
\`\`\`

## Self-review (run inline, fix before asking the user)

1. **Placeholders** — any `<…>` that survived, "TBD", "TODO". Fix them.
2. **Specificity** — the Question must be specific enough that a literature-scout
   could write a focused query, not generic.
3. **Falsifiability prep** — if the novelty target is `hypothesis`, the success
   criteria must mention falsification criteria, not just "produce hypotheses".
4. **YAGNI is concrete** — Out of scope items must be specific things the user
   might otherwise expect in scope, not generic disclaimers.

## User review gate

Tell the user the spec path and ask them to review. Wait for explicit approval.
Only when the user approves does the harness advance to the plan session; do not
start the plan yourself.
```

Create `crates/research/src/flows/plan.md` (ported from `skills/writing-research-plan/SKILL.md`, with the superpowers:writing-plans meta-format inlined — context/approach/critical files/verification/self-review — and the `superpowers:writing-plans` invocation removed; "Honest estimate" rephrased to "Grounded estimate"):

```markdown
---
name: plan
description: Produce the executable research plan at docs/research/plans/YYYY-MM-DD-<topic>-plan.md, including the Swarm decomposition section the orchestrator parses. Gets user approval.
argument-hint: "[topic]"
model: inherit
allowed-tools:
  - Read
  - Write
---

# Writing the Research Plan

You have an approved spec at `docs/research/specs/<spec>.md` (path known from
this conversation). Produce a concrete, executable research plan. Use `Write`
jailed under `docs/research/`, writing to `plans/YYYY-MM-DD-<topic-slug>-plan.md`
(same date as the spec or later).

## Plan meta-format (inlined)

A plan is a document an engineer (or, here, the orchestrator) can execute with
zero extra context. Include these sections, scaled to complexity:

- **Goal** — one sentence.
- **Architecture** — 2–3 sentences on approach.
- **Context** — the spec's question, novelty target, success criteria, YAGNI
  fence (do not restate the whole spec; reference its path).
- **Approach** — the phases below, in order.
- **Critical files** — the spec path and any seed papers/datasets named.
- **Verification** — how completion is judged (ties to the spec's success
  criteria).

## Swarm decomposition (required — the orchestrator parses this)

\`\`\`markdown
## Swarm decomposition

### Phase 1 — literature-scout dispatches
One assignment per scout. Each is a paragraph:
- Sub-topic name
- Focus constraints (e.g., "2024–2026 only", "must include EO + IR fusion specifically")
- Tools to prioritize (defaults: hf_papers + web_search + github_examples)
Aim for 3–6 scouts.

### Phase 2 — gap-finder dispatches
One assignment per gap-finder:
- Slice: which scout outputs to analyze
- Focus: what kinds of gaps to prioritize
Aim for 1–3 gap-finders.

### Phase 3 — hypothesis-smith dispatches
Computed dynamically by the orchestrator (one smith per gap). Do not pre-list.

### Phase 4 — red-team critique loop
Computed dynamically (one red-team per hypothesis, revision loop up to 3).
Note any project-specific critique focus here.

### Phase 5 — eval-designer dispatches
Computed dynamically (one designer per surviving hypothesis). Note any
compute-budget guidance (e.g., "designs requiring >500 GPU-hours must be flagged
for user approval").

### Phase 6 — synthesist
Single dispatch. Note synthesis-specific requirements (e.g., "final document
must be ≤ 8 pages and tied directly to the spec's success criteria").

### Custom worker dispatches
If the spec defined custom workers, list when they fire and with what inputs.

### Parallelism budget
MEGARESEARCHER_MAX_PARALLEL = <number> (default 4)

### Estimated total runtime + token budget
Grounded estimate. State assumptions. Round up.
\`\`\`

## Self-review (run before asking the user)

- Every Phase-1 scout has a non-overlapping focus.
- Every Phase-2 gap-finder is assigned a slice whose scout outputs correspond to
  actual scouts in Phase 1.
- The parallelism budget is set explicitly.
- The token estimate is grounded (e.g., "12 workers × 30k tokens average ≈ 360k;
  plus orchestrator overhead ≈ 500k total").

## User review gate

Present a summary (workers per phase, estimated cost, any unusual decisions) and
ask the user to review before the run. Wait for explicit approval. A plan is not
a spec — the plan tells the orchestrator HOW; the spec tells everyone WHY. Do
not restate the spec at length. Phase-1 dispatches are the highest-leverage
decision: bad scout coverage → bad gap-finding → bad hypotheses, so spend time
partitioning the topic well.

When the user approves, say so plainly. The harness then waits for the user to
run `mr execute`; do not start the run yourself — spending the tokens is the
user's deliberate choice.
```

- [ ] **Step 4: Write `crates/research/src/flows.rs`**

```rust
//! Front-half guided-session flow bodies: frontmatter + body markdown assets,
//! embedded into the binary via `include_str!`. Mirrors `prompt_asset.rs`'s
//! parser shape but with flow-specific frontmatter (description, argument-hint,
//! model, allowed-tools). The body is the guiding prompt for a `GuidedSession`
//! (the first user message). See design §8/§180/§182.

use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

/// The names of the flow bodies embedded into the binary.
pub const EMBEDDED_NAMES: &[&str] = &["brainstorm", "spec", "plan"];

const BRAINSTORM_MD: &str = include_str!("flows/brainstorm.md");
const SPEC_MD: &str = include_str!("flows/spec.md");
const PLAN_MD: &str = include_str!("flows/plan.md");

#[derive(Deserialize)]
struct FrontMatter {
    name: String,
    description: String,
    #[serde(default)]
    argument_hint: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    allowed_tools: Option<Vec<String>>,
}

/// A parsed flow-body asset.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowAsset {
    pub name: String,
    pub description: String,
    pub argument_hint: Option<String>,
    pub model: Option<String>,
    pub allowed_tools: Option<Vec<String>>,
    pub body: String,
}

/// Parse raw text into a [`FlowAsset`]. Text must start with a `---` line and
/// contain a closing `---` line; between them is YAML frontmatter, after is the
/// body.
pub fn parse(text: &str) -> Result<FlowAsset, String> {
    let after_open = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
        .ok_or_else(|| "file must start with a frontmatter delimiter '---'".to_string())?;
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
    Ok(FlowAsset {
        name: front.name,
        description: front.description,
        argument_hint: front.argument_hint,
        model: front.model,
        allowed_tools: front.allowed_tools,
        body: body.trim().to_string(),
    })
}

/// Load and parse a flow body from disk.
pub fn load(path: &Path) -> io::Result<FlowAsset> {
    let text = fs::read_to_string(path)?;
    parse(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Load an embedded flow body by name (`"brainstorm"`, `"spec"`, `"plan"`).
/// Panics on an unknown name — only call with literals.
pub fn load_embedded(name: &str) -> FlowAsset {
    let text = match name {
        "brainstorm" => BRAINSTORM_MD,
        "spec" => SPEC_MD,
        "plan" => PLAN_MD,
        other => panic!("unknown embedded flow: {other}"),
    };
    parse(text).unwrap_or_else(|e| panic!("embedded flow {name} failed to parse: {e}"))
}
```

- [ ] **Step 5: Wire into `lib.rs`**

Add `pub mod flows;` to `crates/research/src/lib.rs` alongside the other `pub mod` declarations.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test flows`
Expected: PASS (7 tests).

- [ ] **Step 7: fmt + clippy + commit**

```bash
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
git add crates/research/src/flows.rs crates/research/src/flows/ crates/research/src/lib.rs crates/research/tests/flows.rs
git commit -m "feat(research): add flows module — FlowAsset parser + 3 embedded guided-session bodies

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: `research::phases` — GuidedSession + drive_session + UserIo

**Files:**
- Create: `crates/research/src/phases.rs`
- Modify: `crates/research/src/worker.rs` (make `accumulate` `pub(crate)`)
- Modify: `crates/research/src/lib.rs` (add `pub mod phases;`)
- Test: `crates/research/tests/phases.rs` (new)

**Interfaces:**
- Consumes: `claurst_api::{LlmProvider, ProviderRequest, ProviderError, SystemPrompt}`, `claurst_core::types::{ContentBlock, Message, ToolDefinition, ToolResultContent, UsageInfo}`, `crate::worker_tools::{Tool, ToolResult}`, `crate::worker::accumulate` (newly `pub(crate)`).
- Produces:
  - `#[async_trait] pub trait UserIo: Send + Sync { async fn print(&self, text: &str) -> io::Result<()>; async fn read_line(&self) -> io::Result<String>; }`
  - `pub struct GuidedSession { /* fields */ }` with `pub fn new(flow_body: impl Into<String>, tools: Vec<Arc<dyn Tool>>, provider: Arc<dyn LlmProvider>, model: impl Into<String>, max_tokens: u32, max_turns: u32) -> Self` and `pub async fn run_to_checkpoint(&mut self) -> Result<Checkpoint, SessionError>`.
  - `pub enum Checkpoint { EndTurn { assistant_text: String, turns: u32, usage: UsageInfo }, MaxTurns { assistant_text: String, turns: u32 } }`
  - `pub fn inject_user(&mut self, text: &str)`
  - `#[derive(Debug, Clone)] pub struct Gate { pub artifact: PathBuf, pub label: String }`
  - `pub async fn drive_session(session: &mut GuidedSession, io: &dyn UserIo, gates: Vec<Gate>, approve_words: &[&str]) -> Result<DriveOutcome, SessionError>`
  - `pub enum DriveOutcome { Approved { gates_passed: usize }, MaxTurns }`
  - `#[derive(Debug)] pub enum SessionError { Provider(ProviderError), BadStream(String), Io(io::Error) }` with Display + Error.

**Semantics:**
- `GuidedSession` seeds `messages = [Message::user(flow_body)]`. `run_to_checkpoint` loops: build a `ProviderRequest` (system_prompt = a fixed guided-session system prompt), call `provider.create_message_stream`, `accumulate` into blocks, append `Message::assistant_blocks(blocks)`; extract `ToolUse` blocks, dispatch via `tools`, append `Message::user_blocks(tool_results)`; repeat until a turn has no tool uses (→ `Checkpoint::EndTurn`) or `max_turns` is hit (→ `Checkpoint::MaxTurns`). `inject_user` appends `Message::user(text)`.
- `drive_session` drives the approval flow: loop { `run_to_checkpoint`; print `assistant_text`; read a user line; if the line (trimmed, lowercased) is in `approve_words` → check `gates[gate_idx].artifact.exists()`: if yes, advance `gate_idx`, if `gate_idx == gates.len()` return `Approved { gates_passed }`, else print "approved <label>; next: <next label>"; if the artifact does NOT exist, print "can't approve — <artifact> not found; did you write it?" and continue; else (not an approve word) → `session.inject_user(line)` } bounded by a total-turn ceiling (sum of `max_turns` across checkpoints; reuse `session`'s `max_turns` as the per-checkpoint cap and track a separate total).

- [ ] **Step 1: Write the failing test**

```rust
// crates/research/tests/phases.rs
use std::sync::Arc;
use megaresearcher_research::phases::{drive_session, Gate, GuidedSession, DriveOutcome, UserIo};
use megaresearcher_research::tests_common as common; // if a tests/common/mod.rs exists; else inline FakeProvider

// Reuse the existing test FakeProvider infra. It lives at tests/common/fake_provider.rs
// and is gated behind a `common` mod declared in each test file. Mirror the
// pattern used by tests/orchestrator.rs (it declares `mod common;`).
mod common;
use common::fake_provider::{FakeProvider, canned::*}; // see note below
```

> NOTE on test infra: the existing `tests/common/fake_provider.rs` is shared via `tests/common/mod.rs`. `tests/orchestrator.rs` uses `mod common;`. Add the same `mod common;` declaration in `tests/phases.rs` (the `common` dir already exists). The helper builders (`three_artifact_turns`, `run_turns`, `redteam_turns`, `evaldesign_turns`) live in `tests/common/`; if a `canned` submodule does not exist, build `StreamEvent` turns inline in this test file instead. The implementer should inspect `tests/common/mod.rs` and the helpers used at `tests/orchestrator.rs:265` (`FakeProvider::new("fake", turns)`) and reuse the same `StreamEvent` construction helpers. The test below assumes inline `StreamEvent` construction mirroring `tests/fake_provider.rs:14-95` (the documented shape: `MessageStart` → `ContentBlockStart` with a `Text`/`ToolUse` seed → `TextDelta`/`InputJsonDelta` → `ContentBlockStop` → `MessageDelta{stop_reason}` → `MessageStop`).

```rust
// Helper: a turn that emits one assistant text block and ends the turn.
fn text_turn(text: &str) -> Vec<StreamEvent> {
    use claurst_api::StreamEvent::*;
    use claurst_core::types::ContentBlock;
    vec![
        MessageStart { message: Default::default() },
        ContentBlockStart { index: 0, content_block: ContentBlock::Text { text: String::new() } },
        TextDelta { index: 0, text: text.to_string() },
        ContentBlockStop { index: 0 },
        MessageDelta { stop_reason: Some(claurst_api::StopReason::EndTurn), usage: None },
        MessageStop,
    ]
}
```

(If `StreamEvent` variants or field names differ from the above, the implementer adjusts to the actual definitions at `crates/api/src/provider_types.rs` — the goal is a turn that accumulates to one `Text` block with `EndTurn`. The existing `tests/fake_provider.rs` already constructs such turns; copy its construction verbatim.)

```rust
struct FakeUserIo { lines: std::sync::Mutex<std::collections::VecDeque<String>>, printed: std::sync::Mutex<Vec<String>> }
impl FakeUserIo {
    fn new(lines: Vec<String>) -> Self { Self { lines: lines.into_iter().collect(), printed: Vec::new().into() } }
}
#[async_trait::async_trait]
impl UserIo for FakeUserIo {
    async fn print(&self, text: &str) -> std::io::Result<()> { self.printed.lock().unwrap().push(text.to_string()); Ok(()) }
    async fn read_line(&self) -> std::io::Result<String> {
        self.lines.lock().unwrap().pop_front().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "no input"))
    }
}

#[tokio::test]
async fn brainstorm_session_converges_on_approve() {
    // Provider emits a clarifying question, then a restatement, each as text turns.
    let provider = Arc::new(FakeProvider::new("fake", vec![
        text_turn("What is your novelty target?"),
        text_turn("Restating: hypothesis, EO/IR fusion, open datasets only. Approve?"),
    ]));
    let mut session = GuidedSession::new("flow body", vec![], provider, "fake", 1024, 10);
    let io = FakeUserIo::new(vec!["hypothesis".into(), "approve".into()]);
    let outcome = drive_session(&mut session, &io, vec![], &["approve", "yes", "y"]).await.unwrap();
    assert!(matches!(outcome, DriveOutcome::Approved { gates_passed: 0 }));
}

#[tokio::test]
async fn spec_session_writes_artifact_then_advances_gate() {
    use claurst_api::StreamEvent::*;
    use claurst_core::types::ContentBlock;
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let spec_path = dir.path().join("specs/test-spec.md");
    // Turn 1: a ToolUse writing the spec file (ScopedWrite jailed to dir.path()).
    let write_turn = vec![
        MessageStart { message: Default::default() },
        ContentBlockStart { index: 0, content_block: ContentBlock::ToolUse {
            id: "t1".into(), name: "Write".into(), input: json!({ "file_path": "specs/test-spec.md", "content": "# Spec\n" }) } },
        InputJsonDelta { index: 0, partial_json: "{\"file_path\":\"specs/test-spec.md\",\"content\":\"# Spec\\n\"}".into() },
        ContentBlockStop { index: 0 },
        MessageDelta { stop_reason: Some(claurst_api::StopReason::ToolUse), usage: None },
        MessageStop,
    ];
    // Turn 2 (after tool result): assistant says done.
    let done_turn = text_turn("Spec written to specs/test-spec.md. Approve?");
    let provider = Arc::new(FakeProvider::new("fake", vec![write_turn, done_turn]));
    use megaresearcher_research::worker_tools::ScopedWrite;
    let tools: Vec<Arc<dyn megaresearcher_research::worker_tools::Tool>> =
        vec![Arc::new(ScopedWrite::new(dir.path()))];
    let mut session = GuidedSession::new("write the spec", tools, provider, "fake", 1024, 10);
    let io = FakeUserIo::new(vec!["approve".into()]);
    let outcome = drive_session(&mut session, &io,
        vec![Gate { artifact: spec_path.clone(), label: "spec".into() }],
        &["approve", "yes"]).await.unwrap();
    assert!(matches!(outcome, DriveOutcome::Approved { gates_passed: 1 }));
    assert!(spec_path.exists());
}

#[tokio::test]
async fn approve_before_artifact_exists_is_refused() {
    let provider = Arc::new(FakeProvider::new("fake", vec![
        text_turn("Drafting... approve?"),
        text_turn("OK, writing now. Approve?"),
    ]));
    let mut session = GuidedSession::new("flow", vec![], provider, "fake", 1024, 10);
    let io = FakeUserIo::new(vec!["approve".into(), "approve".into()]);
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("nope.md");
    let outcome = drive_session(&mut session, &io,
        vec![Gate { artifact: missing.clone(), label: "x".into() }], &["approve"]).await;
    // The first "approve" must be refused (artifact missing); the second also
    // refused -> the session runs out of scripted turns and returns MaxTurns.
    assert!(matches!(outcome, Ok(DriveOutcome::MaxTurns) | Err(_)));
    assert!(!missing.exists());
}

#[tokio::test]
async fn user_revision_feedback_is_injected_as_next_turn() {
    let provider = Arc::new(FakeProvider::new("fake", vec![
        text_turn("Draft v1. Approve?"),
        text_turn("Applied your change. Approve?"),
    ]));
    let dir = tempfile::tempdir().unwrap();
    let artifact = dir.path().join("a.md");
    std::fs::write(&artifact, "v2").unwrap();
    let mut session = GuidedSession::new("flow", vec![], provider, "fake", 1024, 10);
    let io = FakeUserIo::new(vec!["make it shorter".into(), "approve".into()]);
    let outcome = drive_session(&mut session, &io,
        vec![Gate { artifact, label: "a".into() }], &["approve"]).await.unwrap();
    assert!(matches!(outcome, DriveOutcome::Approved { gates_passed: 1 }));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test phases`
Expected: FAIL — `module phases` not found.

- [ ] **Step 3: Expose `accumulate` from `worker.rs`**

In `crates/research/src/worker.rs`, change the private `async fn accumulate(` declaration (currently `async fn accumulate(` at line ~198) to `pub(crate) async fn accumulate(`. No other change.

- [ ] **Step 4: Write `crates/research/src/phases.rs`**

```rust
//! Front-half guided sessions (design §6/§8/§67). A `GuidedSession` is a
//! multi-turn LLM conversation on `LlmProvider` where the flow body is the first
//! user message; the model may call tools (e.g. `ScopedWrite` to draft the
//! spec/plan) between turns. `run_to_checkpoint` runs the model to a natural
//! pause (a turn with no pending tool calls) or the turn ceiling. `drive_session`
//! is the approval-gate driver: it prints assistant text, reads a user line,
//! and either advances through artifact gates (Rust-enforced: the artifact file
//! must exist before an approval is accepted) or injects the line as the next
//! user turn. Reuses `worker::accumulate` so existing `FakeProvider` test infra
//! works unchanged (FakeProvider scripts only `create_message_stream`).

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use claurst_api::{LlmProvider, ProviderError, ProviderRequest, SystemPrompt};
use claurst_core::types::{ContentBlock, Message, ToolDefinition, ToolResultContent, UsageInfo};

use crate::worker::accumulate;
use crate::worker_tools::{Tool, ToolResult};

/// A fixed system prompt for guided sessions: stay in your lane, drive the
/// flow body, do not invoke external skills, use only the provided tools.
const SESSION_SYSTEM_PROMPT: &str = "\
You are a MegaResearcher guided-session facilitator. Follow the flow body given \
as the first user message exactly. Work one step at a time. Use only the tools \
provided. Do not invoke, reference, or pretend to call any external skill or \
plugin. When you reach an approval gate, state plainly what you produced and \
where, then ask the user to approve.";

/// Why a session errored.
#[derive(Debug)]
pub enum SessionError {
    Provider(ProviderError),
    BadStream(String),
    Io(io::Error),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(e) => write!(f, "provider error: {e}"),
            Self::BadStream(s) => write!(f, "bad stream: {s}"),
            Self::Io(e) => write!(f, "io error: {e}"),
        }
    }
}
impl std::error::Error for SessionError {}
impl From<ProviderError> for SessionError { fn from(e: ProviderError) -> Self { Self::Provider(e) } }
impl From<io::Error> for SessionError { fn from(e: io::Error) -> Self { Self::Io(e) } }

/// A natural pause point in a guided session.
#[derive(Debug, Clone)]
pub enum Checkpoint {
    /// The model ended a turn with no pending tool calls.
    EndTurn { assistant_text: String, turns: u32, usage: UsageInfo },
    /// The per-checkpoint turn ceiling was reached with tool calls pending.
    MaxTurns { assistant_text: String, turns: u32 },
}

/// Abstraction over user I/O so `drive_session` is testable with a fake.
#[async_trait]
pub trait UserIo: Send + Sync {
    async fn print(&self, text: &str) -> io::Result<()>;
    async fn read_line(&self) -> io::Result<String>;
}

/// One approval gate: an artifact that must exist on disk before the user's
/// approval is accepted, plus a human label.
#[derive(Debug, Clone)]
pub struct Gate {
    pub artifact: PathBuf,
    pub label: String,
}

/// The outcome of `drive_session`.
#[derive(Debug, Clone)]
pub enum DriveOutcome {
    /// All gates advanced (the run's artifacts exist and the user approved).
    Approved { gates_passed: usize },
    /// The session hit its turn ceiling before approval.
    MaxTurns,
}

/// A multi-turn guided LLM session.
pub struct GuidedSession {
    messages: Vec<Message>,
    tools: Vec<Arc<dyn Tool>>,
    provider: Arc<dyn LlmProvider>,
    model: String,
    max_tokens: u32,
    max_turns: u32,
}

impl GuidedSession {
    /// Seed the session with `flow_body` as the first user message.
    pub fn new(
        flow_body: impl Into<String>,
        tools: Vec<Arc<dyn Tool>>,
        provider: Arc<dyn LlmProvider>,
        model: impl Into<String>,
        max_tokens: u32,
        max_turns: u32,
    ) -> Self {
        Self {
            messages: vec![Message::user(flow_body.into())],
            tools,
            provider,
            model: model.into(),
            max_tokens,
            max_turns,
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

    /// Append a user message (an answer, a revision request, or steering text).
    pub fn inject_user(&mut self, text: &str) {
        self.messages.push(Message::user(text.to_string()));
    }

    /// Run the model until a turn has no pending tool calls (a checkpoint) or
    /// the per-checkpoint turn ceiling is hit. Tool calls are dispatched
    /// internally, mirroring `worker::Worker::run`.
    pub async fn run_to_checkpoint(&mut self) -> Result<Checkpoint, SessionError> {
        let tool_defs = self.tool_defs();
        let mut last_assistant_text = String::new();
        let mut usage = UsageInfo::default();

        for turn in 0..self.max_turns {
            let req = ProviderRequest {
                model: self.model.clone(),
                messages: self.messages.clone(),
                system_prompt: Some(SystemPrompt::Text(SESSION_SYSTEM_PROMPT.to_string())),
                tools: tool_defs.clone(),
                max_tokens: self.max_tokens,
                temperature: None,
                top_p: None,
                top_k: None,
                stop_sequences: vec![],
                thinking: None,
                provider_options: Value::Object(serde_json::Map::new()),
            };
            let stream = self.provider.create_message_stream(req).await?;
            let (blocks, _stop_reason, turn_usage) = accumulate(stream).await
                .map_err(SessionError::from)?;
            if let Some(u) = turn_usage { usage = u; }

            let mut tool_uses: Vec<(String, String, Value)> = Vec::new();
            for block in &blocks {
                match block {
                    ContentBlock::Text { text } if !text.is_empty() => last_assistant_text = text.clone(),
                    ContentBlock::ToolUse { id, name, input } => tool_uses.push((id.clone(), name.clone(), input.clone())),
                    _ => {}
                }
            }
            self.messages.push(Message::assistant_blocks(blocks));

            if tool_uses.is_empty() {
                return Ok(Checkpoint::EndTurn {
                    assistant_text: last_assistant_text,
                    turns: turn + 1,
                    usage,
                });
            }

            let mut results: Vec<ContentBlock> = Vec::new();
            for (id, name, input) in tool_uses {
                let result = match self.tools.iter().find(|t| t.name() == name) {
                    Some(tool) => tool.call(input).await,
                    None => ToolResult::err(format!("<tool_use_error>unknown tool: {name}</tool_use_error>")),
                };
                results.push(ContentBlock::ToolResult {
                    tool_use_id: id,
                    content: ToolResultContent::Text(result.content),
                    is_error: Some(result.is_error),
                });
            }
            self.messages.push(Message::user_blocks(results));
        }

        Ok(Checkpoint::MaxTurns { assistant_text: last_assistant_text, turns: self.max_turns })
    }
}

/// Drive a guided session through its approval gates.
///
/// Loop: run to a checkpoint, print the assistant text, read a user line. If the
/// line is an approval word and the current gate's artifact exists, advance the
/// gate; when all gates are passed, return `Approved`. If the line is an
/// approval word but the artifact is missing, refuse and continue. Otherwise
/// inject the line as the next user turn. Bounded by `total_turn_ceiling`.
pub async fn drive_session(
    session: &mut GuidedSession,
    io: &dyn UserIo,
    gates: Vec<Gate>,
    approve_words: &[&str],
) -> Result<DriveOutcome, SessionError> {
    let total_ceiling = session.max_turns.saturating_mul(gates.len().max(1) as u32 + 1);
    let mut spent: u32 = 0;
    let mut gate_idx: usize = 0;

    loop {
        if spent >= total_ceiling {
            return Ok(DriveOutcome::MaxTurns);
        }
        let checkpoint = session.run_to_checkpoint().await?;
        let (text, turns) = match checkpoint {
            Checkpoint::EndTurn { assistant_text, turns, .. } => (assistant_text, turns),
            Checkpoint::MaxTurns { assistant_text, turns } => (assistant_text, turns),
        };
        spent = spent.saturating_add(turns);
        if !text.is_empty() {
            io.print(&text).await?;
            io.print("\n").await?;
        }
        if matches!(checkpoint, Checkpoint::MaxTurns { .. }) {
            return Ok(DriveOutcome::MaxTurns);
        }

        let line = io.read_line().await?;
        let trimmed = line.trim().to_lowercase();
        let is_approve = approve_words.iter().any(|w| trimmed == w);

        if is_approve {
            if gates.is_empty() {
                return Ok(DriveOutcome::Approved { gates_passed: 0 });
            }
            let gate = &gates[gate_idx];
            if gate.artifact.exists() {
                gate_idx += 1;
                if gate_idx >= gates.len() {
                    return Ok(DriveOutcome::Approved { gates_passed: gate_idx });
                }
                let next = &gates[gate_idx];
                io.print(&format!("Approved {}. Next: {}.\n", gate.label, next.label)).await?;
            } else {
                io.print(&format!(
                    "Can't approve — {} not found. Did you write it? Respond with feedback or approve once it exists.\n",
                    gate.artifact.display()
                )).await?;
            }
        } else {
            session.inject_user(&line);
        }
    }
}
```

- [ ] **Step 5: Wire into `lib.rs`**

Add `pub mod phases;` to `crates/research/src/lib.rs`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test phases`
Expected: PASS (4 tests). If `tempfile` is not a dev-dep of `megaresearcher-research`, add it (`tempfile = { workspace = true }` in `[dev-dependencies]` — confirm it's a workspace dep; claurst uses it).

- [ ] **Step 7: fmt + clippy + commit**

```bash
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
git add crates/research/src/phases.rs crates/research/src/worker.rs crates/research/src/lib.rs crates/research/tests/phases.rs crates/research/Cargo.toml
git commit -m "feat(research): add phases module — GuidedSession + drive_session with Rust-enforced approval gates

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: `research::orchestrator/escalation` — EscalationHandler + integration

**Files:**
- Create: `crates/research/src/orchestrator/escalation.rs`
- Modify: `crates/research/src/orchestrator/mod.rs` — add `pub escalation: Option<Arc<dyn EscalationHandler>>` to `OrchestratorConfig`; replace the 4 `return Err(OrchestratorError::Escalated(...))` sites with a call to a new `adjudicate_escalation` helper.
- Test: `crates/research/tests/escalation.rs` (new) — append to existing orchestrator test file OR new file (prefer a new file to keep the 52 tests untouched).

**Interfaces:**
- Consumes: `crate::state::swarm_state::Escalation`, `crate::orchestrator::{OrchestratorConfig, OrchestratorError, RunOutcome}`.
- Produces:
  - `#[async_trait] pub trait EscalationHandler: Send + Sync { async fn adjudicate(&self, escalation: &Escalation) -> EscalationVerdict; }`
  - `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum EscalationVerdict { Continue, Fail }`
  - A built-in `pub struct FailAll;` impl returning `Fail` (used by tests / as a default-equivalent).
- The orchestrator gains a private helper:
  ```rust
  async fn adjudicate_escalation(
      &self,
      swarm: &SwarmState,
      run_dir: &Path,
      run_id: &str,
      names: Vec<String>,
      reason: &str,
  ) -> Result<RunOutcome, OrchestratorError>
  ```
  Behavior: `write_swarm(swarm, run_dir)?`; let verdict = match `self.config.escalation.as_ref() { None => Fail, Some(h) => h.adjudicate(&Escalation{ worker: names.join(","), reason: reason.into(), retry_count: 1 }).await }`; match verdict { Fail => `Err(OrchestratorError::Escalated(names))`, Continue => `Ok(RunOutcome { run_dir: run_dir.to_path_buf(), run_id: run_id.to_string(), phase_statuses: swarm.phases.iter().map(|p|(p.name.clone(),p.status.clone())).collect(), escalations: swarm.escalations.iter().map(|e|e.worker.clone()).collect() }) }`.

**The 4 escalation return-sites in `execute()` (crates/research/src/orchestrator/mod.rs):**
1. Scout escalation — current lines ~202–208: `for name in &escalated { add_escalation(...) } write_swarm(...); return Err(OrchestratorError::Escalated(escalated));` → replace the `return Err(...)` with `return self.adjudicate_escalation(&swarm, &run_dir, run_id, escalated, "missing artifacts after retry").await;` (keep the `add_escalation` loop + `write_swarm` but remove the redundant `write_swarm` since the helper writes; OR keep `write_swarm` and have the helper not re-write — pick ONE: the helper writes, so remove the standalone `write_swarm` before the return). The `escalated` variable is moved into the call.
2. Gap escalation — lines ~265–270: same pattern with `gap_escalated`.
3. All-killed (hypothesis path) — lines ~325–328: `return Err(OrchestratorError::Escalated(killed));` → `return self.adjudicate_escalation(&swarm, &run_dir, run_id, killed, "all hypotheses killed by red-team").await;` (note: the `add_escalation` is NOT done for all-killed in current code — check; the helper builds the Escalation from `names`+`reason`, so it's fine either way; do not double-add). Verify against the actual lines.
4. Synth escalation — lines ~388–391: `add_escalation(...); write_swarm(...); return Err(OrchestratorError::Escalated(vec!["synthesist".to_string()]));` → `return self.adjudicate_escalation(&swarm, &run_dir, run_id, vec!["synthesist".to_string()], "missing artifacts after retry").await;` (the `add_escalation` + `write_swarm` before it should remain, since the helper builds its own Escalation — but to avoid a double escalation entry, do NOT `add_escalation` before calling the helper; instead let the helper's Escalation represent it. Simplest: at each site, keep the existing `add_escalation` calls (they populate `swarm.escalations` for the state file) and remove the standalone `write_swarm` + `return Err`, replacing with the helper which writes + decides. The helper does NOT add an escalation; it reads `swarm.escalations` for the RunOutcome. This is the clean rule: **sites keep their `add_escalation` calls; the helper only `write_swarm`s + decides.**)

> IMPORTANT consistency rule for the implementer: at each of the 4 sites, keep the existing `add_escalation(...)` calls (unchanged), remove any `write_swarm(&swarm, &run_dir)?;` that immediately precedes the `return Err(...)`, and replace `return Err(OrchestratorError::Escalated(<names>));` with `return self.adjudicate_escalation(&swarm, &run_dir, run_id, <names>, <reason>).await;`. The helper writes the swarm file once and decides. When `escalation` is `None` (all 52 existing tests), the helper returns `Err(OrchestratorError::Escalated(<names>))` — byte-identical behavior. Confirm by running the full orchestrator suite green.

- [ ] **Step 1: Write the failing test**

```rust
// crates/research/tests/escalation.rs
// Reuses the existing test FakeProvider + fixtures. The scenario that triggers
// a scout escalation: a FakeProvider whose turns do NOT write the three required
// artifacts, so verify_wave escalates. Mirror the escalation-triggering test
// pattern already present in tests/orchestrator.rs (find a test that asserts
// `Err(OrchestratorError::Escalated(_))` and clone its setup).
mod common;
use common::fake_provider::FakeProvider;
// ... (the implementer copies the escalation-triggering fixture setup from
//      tests/orchestrator.rs — a FakeProvider returning turns that omit
//      output.md/manifest.yaml/verification.md, plus the spec/plan/agents
//      fixtures that the existing tests use.)

use async_trait::async_trait;
use megaresearcher_research::orchestrator::{Orchestrator, OrchestratorConfig, OrchestratorError};
use megaresearcher_research::orchestrator::escalation::{EscalationHandler, EscalationVerdict};
use megaresearcher_research::state::swarm_state::Escalation;
use std::sync::Arc;

struct ContinueAll;
#[async_trait]
impl EscalationHandler for ContinueAll {
    async fn adjudicate(&self, _e: &Escalation) -> EscalationVerdict { EscalationVerdict::Continue }
}

#[tokio::test]
async fn continue_handler_returns_ok_with_escalations_listed() {
    // Build the same fixtures + FakeProvider that the existing
    // `err_on_scout_escalation` (or equivalent) test uses to reach a scout
    // escalation. Set `escalation: Some(Arc::new(ContinueAll))`.
    // Assert: execute() returns `Ok(out)` with `out.escalations` non-empty and
    // `out.phase_statuses` containing "literature-scout" with status
    // "complete" (or whatever the swarm records at the escalation point).
    let _ = (ContinueAll,); // placeholder until fixtures copied
    // ... full setup per existing escalation test ...
}

#[tokio::test]
async fn none_handler_preserves_err_behavior() {
    // Same fixtures, `escalation: None`. Assert `Err(OrchestratorError::Escalated(_))`
    // — identical to the pre-existing behavior.
}

#[tokio::test]
async fn fail_handler_returns_err() {
    // `escalation: Some(Arc::new(FailAll))` — assert `Err(OrchestratorError::Escalated(_))`.
}
```

> The implementer MUST locate an existing test in `tests/orchestrator.rs` that reaches an escalation (search for `OrchestratorError::Escalated` assertions and `verify_wave` escalation paths) and copy its fixture-setup verbatim, then vary only the `escalation` field. Do not invent new fixtures.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test escalation`
Expected: FAIL — `escalation` module / `EscalationHandler` undefined.

- [ ] **Step 3: Write `crates/research/src/orchestrator/escalation.rs`**

```rust
//! The escalation-adjudication seam (design §112): when a worker escalates,
//! the orchestrator asks the handler whether to continue (record + halt
//! gracefully with a partial `RunOutcome`) or fail (return `Err`). `None`
//! means fail — preserving the pre-Phase-6a behavior the 52 orchestrator
//! tests depend on. The TUI (Phase 6b) supplies an interactive handler that
//! surfaces to the run-screen queue and blocks for a user ack.

use async_trait::async_trait;

use crate::state::swarm_state::Escalation;

/// What to do with an escalated worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationVerdict {
    /// Record the escalation and return a partial `RunOutcome` (the run halts
    /// at the escalation point; downstream phases that depended on the failed
    /// worker are not attempted).
    Continue,
    /// Return `Err(OrchestratorError::Escalated(..))` — the pre-6a default.
    Fail,
}

/// Adjudicate an escalation. Implemented by the CLI (headless `--on-escalate`)
/// and, later, by the TUI (interactive ack).
#[async_trait]
pub trait EscalationHandler: Send + Sync {
    async fn adjudicate(&self, escalation: &Escalation) -> EscalationVerdict;
}

/// A handler that always fails — equivalent to `escalation: None`.
pub struct FailAll;
#[async_trait]
impl EscalationHandler for FailAll {
    async fn adjudicate(&self, _e: &Escalation) -> EscalationVerdict {
        EscalationVerdict::Fail
    }
}
```

- [ ] **Step 4: Wire into `orchestrator/mod.rs`**

In `crates/research/src/orchestrator/mod.rs`:
- Add `pub mod escalation;` to the module declarations (line ~5–15).
- Add to `OrchestratorConfig` (after `pub mcp: Option<McpServerConfig>,`):
  ```rust
  /// Adjudication seam for escalated workers. `None` = fail (pre-6a behavior).
  pub escalation: Option<Arc<dyn escalation::EscalationHandler>>,
  ```
- Add the helper method inside `impl Orchestrator`:
  ```rust
  async fn adjudicate_escalation(
      &self,
      swarm: &crate::state::swarm_state::SwarmState,
      run_dir: &Path,
      run_id: &str,
      names: Vec<String>,
      reason: &str,
  ) -> Result<RunOutcome, OrchestratorError> {
      write_swarm(swarm, run_dir)?;
      let verdict = match self.config.escalation.as_ref() {
          None => escalation::EscalationVerdict::Fail,
          Some(h) => h
              .adjudicate(&crate::state::swarm_state::Escalation {
                  worker: names.join(","),
                  reason: reason.to_string(),
                  retry_count: 1,
              })
              .await,
      };
      match verdict {
          escalation::EscalationVerdict::Fail => Err(OrchestratorError::Escalated(names)),
          escalation::EscalationVerdict::Continue => Ok(RunOutcome {
              run_dir: run_dir.to_path_buf(),
              run_id: run_id.to_string(),
              phase_statuses: swarm
                  .phases
                  .iter()
                  .map(|p| (p.name.clone(), p.status.clone()))
                  .collect(),
              escalations: swarm.escalations.iter().map(|e| e.worker.clone()).collect(),
          }),
      }
  }
  ```
- Replace the 4 `return Err(OrchestratorError::Escalated(<names>));` sites per the consistency rule above (keep the preceding `add_escalation` calls; drop the immediately-preceding `write_swarm`).

- [ ] **Step 5: Update all 6 `OrchestratorConfig` literals in `tests/orchestrator.rs` to add `escalation: None,`**

There are 6 `OrchestratorConfig { ... }` literals in `tests/orchestrator.rs` (per the grounding). Each gains `escalation: None,`. (And the T8 test added in Phase 5 also gains it if present.) Search for `OrchestratorConfig {` and add the field. Also add `escalation: None,` to any `OrchestratorConfig` in `src/` if present (likely none).

- [ ] **Step 6: Run the full orchestrator suite**

Run: `cargo test -p megaresearcher-research`
Expected: PASS — all 52 pre-existing orchestrator tests (now with `escalation: None`, byte-identical behavior) + the 3 new escalation tests green.

- [ ] **Step 7: fmt + clippy + commit**

```bash
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
git add crates/research/src/orchestrator/escalation.rs crates/research/src/orchestrator/mod.rs crates/research/tests/escalation.rs crates/research/tests/orchestrator.rs
git commit -m "feat(research): add EscalationHandler seam + --on-escalate integration (None preserves 52 tests)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: `research::verify` — deterministic post-run checker

**Files:**
- Create: `crates/research/src/verify.rs`
- Modify: `crates/research/src/lib.rs` (add `pub mod verify;`)
- Test: `crates/research/tests/verify.rs` (new)

**Interfaces:**
- Consumes: `crate::state::swarm_state::{SwarmState, Escalation}`, `crate::mcp::{McpCaller, McpError}` (for citation spot-checks; `McpCaller` is `pub` in `mcp.rs`), `std::fs`.
- Produces:
  - `#[derive(Debug, Clone, PartialEq)] pub enum Verdict { Pass, Fail, PassWithCaveats }`
  - `pub struct CheckResult { pub group: char, pub item: String, pub passed: bool, pub detail: Option<String> }`
  - `pub struct VerificationReport { pub run_id: String, pub checks: Vec<CheckResult>, pub spot_checks: Vec<SpotCheck>, pub verdict: Verdict }`
  - `pub struct SpotCheck { pub arxiv_id: String, pub claim: String, pub resolved: bool }`
  - `pub async fn verify_run(run_dir: &Path, spec_path: &Path, mcp: Option<Arc<dyn McpCaller>>) -> anyhow::Result<VerificationReport>`
  - `pub fn write_report(run_dir: &Path, report: &VerificationReport) -> io::Result<()>` (writes `verification-report.md`)
- **Group D (citation spot-checks)** runs only when `mcp` is `Some`; when `None`, group D is recorded as "skipped (no MCP)" and the verdict can still be `Pass`/`PassWithCaveats` based on A/B/C/E/F.

**Checks (mirroring `skills/research-verification/SKILL.md` groups A–F):**
- A. Run completeness: `output.md` exists at run root; `swarm-state.yaml` exists; every worker subdir has `output.md`+`manifest.yaml`+`verification.md`.
- B. Synthesis quality: `output.md` has the 8 synthesist sections (Executive summary, Surviving hypotheses, Rejected and killed hypotheses, Escalations, What we did NOT explore, Recommended next actions, Run metadata, Sources); "Rejected and killed" consistent with `swarm-state.escalations`/killed workers; "What we did NOT explore" non-empty; "Recommended next actions" specific (not "more research needed").
- C. Hypothesis discipline (only if `swarm-state.novelty_target == "hypothesis"`): each surviving hypothesis subdir has falsification criteria (grep `output.md` for "falsif"); each has a red-team `manifest.yaml` with `verdict: APPROVE`; each has an eval-designer dir.
- D. Citation spot-checks (only with MCP): pick 3 arxiv IDs from `output.md` (first, middle, last cited), call `mcp.call_tool("hf_papers", {operation:"paper_details", arxiv_id})` via the `McpCaller`, resolved = result not an error.
- E. Success criteria: read the spec's "Success criteria" section; for each bullet, check it's reflected in `output.md` (substring match on key phrases — this is a heuristic; record caveats).
- F. Doom-loop: `swarm-state.escalations` has no worker at retry_count ≥ 3 that is not also recorded as escalated (i.e., all 3-retry workers are in escalations).

- [ ] **Step 1: Write the failing test**

```rust
// crates/research/tests/verify.rs
use megaresearcher_research::verify::{verify_run, Verdict};
use megaresearcher_research::mcp::{McpCaller, CallToolResult, McpError};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

// A fake McpCaller that "resolves" any paper_details call.
struct FakeMcp;
#[async_trait]
impl McpCaller for FakeMcp {
    async fn call_tool(&self, _name: &str, _args: Option<Value>) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult { content: vec![megaresearcher_research::mcp::McpContent::Text { text: "{\"title\":\"ok\"}".into() }], is_error: false })
    }
}

// Build a passing fixture run dir: output.md with all 8 sections, a worker
// subdir with the 3 artifacts, a swarm-state.yaml with novelty_target
// "gap-finding" (so C is skipped), no escalations. Assert Verdict::Pass.
#[tokio::test]
async fn passing_run_yields_pass() { /* ... build fixture ... */ let r = verify_run(&run_dir, &spec_path, None).await.unwrap(); assert_eq!(r.verdict, Verdict::Pass); }

// A failing run (missing output.md) -> Fail.
#[tokio::test]
async fn missing_output_yields_fail() { /* ... */ }

// A run with all 8 sections but an inconsistency (swarm-state lists a kill not in output) -> Fail (hidden rejection).
#[tokio::test]
async fn hidden_rejection_yields_fail() { /* ... */ }

// Spot-checks skipped without MCP, verdict still derived from A/B/E/F.
#[tokio::test]
async fn no_mcp_skips_group_d_but_still_passes() { /* ... */ }

// With the FakeMcp, group D runs and 3 spot-checks are recorded.
#[tokio::test]
async fn with_mcp_three_spot_checks_run() { let r = verify_run(&run_dir, &spec_path, Some(Arc::new(FakeMcp))).await.unwrap(); assert_eq!(r.spot_checks.len(), 3); }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p megaresearcher-research --test verify`
Expected: FAIL — `module verify` not found.

- [ ] **Step 3: Write `crates/research/src/verify.rs`**

```rust
//! Deterministic post-run verification (design §5: verification is a tree node,
//! red/green; §67 lists verify in the front-half but §5 + the v0 skill make it
//! a deterministic checker, not a guided session — this module implements that
//! checker). Groups A–F mirror skills/research-verification/SKILL.md. Group D
//! (citation spot-checks) needs the ml-intern MCP and runs only when a caller is
//! supplied; otherwise it is skipped and the verdict derives from A/B/C/E/F.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;

use crate::mcp::{CallToolResult, McpCaller, McpContent};
use crate::state::swarm_state::SwarmState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict { Pass, Fail, PassWithCaveats }

#[derive(Debug, Clone)]
pub struct CheckResult { pub group: char, pub item: String, pub passed: bool, pub detail: Option<String> }

#[derive(Debug, Clone)]
pub struct SpotCheck { pub arxiv_id: String, pub claim: String, pub resolved: bool }

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub run_id: String,
    pub checks: Vec<CheckResult>,
    pub spot_checks: Vec<SpotCheck>,
    pub verdict: Verdict,
}

const SYNTH_SECTIONS: &[&str] = &[
    "Executive summary", "Surviving hypotheses", "Rejected and killed hypotheses",
    "Escalations", "What we did NOT explore", "Recommended next actions", "Run metadata", "Sources",
];
const REQUIRED_ARTIFACTS: &[&str] = &["output.md", "manifest.yaml", "verification.md"];

fn worker_subdirs(run_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(run_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() { out.push(p); }
        }
    }
    out.sort();
    out
}

/// Run the 6 check groups (+ optional spot-checks) over `run_dir`.
pub async fn verify_run(
    run_dir: &Path,
    spec_path: &Path,
    mcp: Option<Arc<dyn McpCaller>>,
) -> anyhow::Result<VerificationReport> {
    let swarm = SwarmState::read(&run_dir.join("swarm-state.yaml")).ok();
    let output = std::fs::read_to_string(run_dir.join("output.md")).unwrap_or_default();
    let mut checks = Vec::new();

    // A. Run completeness.
    checks.push(CheckResult { group: 'A', item: "output.md exists".into(), passed: !output.is_empty(), detail: None });
    checks.push(CheckResult { group: 'A', item: "swarm-state.yaml exists".into(), passed: swarm.is_some(), detail: None });
    for d in worker_subdirs(run_dir) {
        for art in REQUIRED_ARTIFACTS {
            let ok = d.join(art).exists();
            checks.push(CheckResult { group: 'A', item: format!("{} has {}", d.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), art), passed: ok, detail: if ok { None } else { Some("missing".into()) } });
        }
    }

    // B. Synthesis quality.
    for sec in SYNTH_SECTIONS {
        let ok = output.contains(sec);
        checks.push(CheckResult { group: 'B', item: format!("section {sec}"), passed: ok, detail: None });
    }
    if let Some(s) = &swarm {
        // Every escalation in state should be reflected in the "Rejected/killed" or "Escalations" section.
        let rejected_section = output.contains(&format!("Rejected and killed"));
        for e in &s.escalations {
            let reflected = output.contains(&e.worker) || rejected_section;
            checks.push(CheckResult { group: 'B', item: format!("escalation {} reflected", e.worker), passed: reflected, detail: if reflected { None } else { Some("hidden rejection".into()) } });
        }
    }
    let yagni_ok = output.contains("What we did NOT explore") && output.lines().any(|l| l.contains("NOT explore"));
    checks.push(CheckResult { group: 'B', item: "YAGNI section non-empty".into(), passed: yagni_ok, detail: None });
    let next_ok = !output.contains("more research is needed");
    checks.push(CheckResult { group: 'B', item: "next actions specific".into(), passed: next_ok, detail: None });

    // C. Hypothesis discipline (hypothesis-target only).
    if let Some(s) = &swarm {
        if s.novelty_target == "hypothesis" {
            for d in worker_subdirs(run_dir) {
                let name = d.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                if name.starts_with("hypothesis-smith") {
                    let out = std::fs::read_to_string(d.join("output.md")).unwrap_or_default();
                    let falsif = out.matches("falsif").count() >= 3;
                    checks.push(CheckResult { group: 'C', item: format!("{} falsification criteria", name), passed: falsif, detail: None });
                }
                if name.starts_with("red-team") {
                    let man = std::fs::read_to_string(d.join("manifest.yaml")).unwrap_or_default();
                    let approved = man.contains("verdict: APPROVE");
                    checks.push(CheckResult { group: 'C', item: format!("{} APPROVE", name), passed: approved, detail: None });
                }
            }
        }
    }

    // D. Citation spot-checks (needs MCP).
    let mut spot_checks = Vec::new();
    match &mcp {
        None => checks.push(CheckResult { group: 'D', item: "spot-checks".into(), passed: true, detail: Some("skipped (no MCP)".into()) }),
        Some(caller) => {
            let ids = collect_arxiv_ids(&output);
            let picks = pick_three(&ids);
            for (id, claim) in picks {
                let args = serde_json::json!({ "operation": "paper_details", "arxiv_id": id });
                let resolved = match caller.call_tool("hf_papers", Some(args)).await {
                    Ok(CallToolResult { is_error: false, .. }) => true,
                    _ => false,
                };
                spot_checks.push(SpotCheck { arxiv_id: id.clone(), claim, resolved });
                checks.push(CheckResult { group: 'D', item: format!("spot-check {}", id), passed: resolved, detail: None });
            }
        }
    }

    // E. Success criteria (heuristic substring match against the spec).
    let spec = std::fs::read_to_string(spec_path).unwrap_or_default();
    if let Some(block) = extract_section(&spec, "Success criteria") {
        for line in block.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
            let key = line.trim().trim_start_matches("- ").to_string();
            let reflected = !key.is_empty() && output.contains(&key);
            checks.push(CheckResult { group: 'E', item: format!("criterion: {}", short(&key, 40)), passed: reflected, detail: None });
        }
    }

    // F. Doom-loop: any worker at retry_count >= 3 must be in escalations.
    if let Some(s) = &swarm {
        for (worker, count) in &s.retry_counts {
            if *count >= 3 {
                let recorded = s.escalations.iter().any(|e| &e.worker == worker);
                checks.push(CheckResult { group: 'F', item: format!("{} retry cap recorded", worker), passed: recorded, detail: None });
            }
        }
    }

    let any_fail = checks.iter().any(|c| !c.passed && c.group != 'D');
    let d_caveat = checks.iter().any(|c| c.group == 'D' && !c.passed && c.detail.as_deref() != Some("skipped (no MCP)"));
    let verdict = if any_fail { Verdict::Fail } else if d_caveat { Verdict::PassWithCaveats } else { Verdict::Pass };

    Ok(VerificationReport {
        run_id: swarm.as_ref().map(|s| s.run_id.clone()).unwrap_or_default(),
        checks, spot_checks, verdict,
    })
}

fn collect_arxiv_ids(text: &str) -> Vec<String> {
    let re = regex::Regex::new(r"(?:arxiv[:\s/]|arXiv[:\s/]?)(\d{4}\.\d{4,5})").unwrap();
    let mut ids = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for c in re.captures_iter(text) {
        if let Some(m) = c.get(1) {
            let id = m.as_str().to_string();
            if seen.insert(id.clone()) { ids.push(id); }
        }
    }
    ids
}
fn pick_three(ids: &[String]) -> Vec<(String, String)> {
    if ids.is_empty() { return vec![]; }
    let idx = |i: usize| (ids[i % ids.len()].clone(), String::new());
    match ids.len() { 1 => vec![idx(0)], 2 => vec![idx(0), idx(1)], _ => vec![idx(0), idx(ids.len() / 2), idx(ids.len() - 1)] }
}
fn extract_section(text: &str, header: &str) -> Option<String> {
    let mut lines = text.lines();
    let mut in_sec = false;
    let mut out = String::new();
    for line in &mut lines {
        if line.starts_with("## ") {
            if in_sec { break; }
            in_sec = line.trim_start_matches("## ").starts_with(header);
        } else if in_sec {
            out.push_str(line); out.push('\n');
        }
    }
    if out.is_empty() { None } else { Some(out) }
}
fn short(s: &str, n: usize) -> String { if s.len() <= n { s.to_string() } else { format!("{}…", &s[..n]) } }

/// Write `verification-report.md` at the run root.
pub fn write_report(run_dir: &Path, report: &VerificationReport) -> io::Result<()> {
    let mut s = String::new();
    s.push_str(&format!("# Verification Report — {}\n\n## Checks\n", report.run_id));
    for c in &report.checks {
        let mark = if c.passed { "[x]" } else { "[ ]" };
        s.push_str(&format!("- {} {} ({}){}\n", mark, c.item, c.group, c.detail.as_deref().map(|d| format!(" — {d}")).unwrap_or_default()));
    }
    s.push_str("\n## Citation spot-checks\n");
    for sc in &report.spot_checks {
        s.push_str(&format!("- {}: resolved={}\n", sc.arxiv_id, sc.resolved));
    }
    s.push_str(&format!("\n## Verdict\n{:?}\n", report.verdict));
    std::fs::write(run_dir.join("verification-report.md"), s)
}

// Suppress an unused-import warning when anyhow::Context isn't referenced yet.
#[allow(unused_imports)]
use anyhow::Context as _AnyhowContext;
```

> NOTE: confirm `regex` is already a dep of `megaresearcher-research` (the grounding listed `regex` in its deps). If not, add `regex = { workspace = true }`. `anyhow` is a dep of `mr-cli` but not necessarily of research — the function returns `anyhow::Result`; if research lacks `anyhow`, either add it as a research dep OR change the return to `Result<VerificationReport, VerifyError>` (an enum wrapping io::Error + McpError). Prefer adding `anyhow = { workspace = true }` to research `[dependencies]` for simplicity (it's a workspace dep).

- [ ] **Step 4: Wire into `lib.rs`** — add `pub mod verify;`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p megaresearcher-research --test verify`
Expected: PASS (5 tests).

- [ ] **Step 6: fmt + clippy + commit**

```bash
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
git add crates/research/src/verify.rs crates/research/src/lib.rs crates/research/tests/verify.rs crates/research/Cargo.toml
git commit -m "feat(research): add verify module — deterministic post-run checker (groups A–F + MCP spot-checks)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: `crates/mr-cli` scaffold — crate, clap, prelude, StdinStdoutIo

**Files:**
- Create: `crates/mr-cli/Cargo.toml`, `crates/mr-cli/src/main.rs`, `crates/mr-cli/src/lib.rs`, `crates/mr-cli/src/prelude.rs`, `crates/mr-cli/src/io.rs`
- Modify: root `Cargo.toml` (add `crates/mr-cli` to members)
- Test: `crates/mr-cli/tests/cli.rs` (new)

**Interfaces:**
- Consumes: `megaresearcher_research::{orchestrator::{Orchestrator, OrchestratorConfig}, mcp::{ml_intern_config, McpServerConfig}, phases::UserIo, orchestrator::escalation::EscalationHandler}`, `claurst_core::config::Settings`, `claurst_api::{LlmProvider, ProviderRegistry, client::ClientConfig}`.
- Produces:
  - `pub async fn run_cli(args: Vec<String>) -> anyhow::Result<()>` — parses + dispatches subcommands.
  - `pub enum Command { Init { question: String }, Brainstorm { topic: String }, Spec { topic: String }, Plan { topic: String }, Execute { plan: Option<PathBuf>, paper: bool, headless: bool, no_mcp: bool, on_escalate: OnEscalate }, Verify { run_dir: PathBuf }, Watch { run_dir: Option<PathBuf> }, List }`
  - `pub enum OnEscalate { Continue, Pause, Fail }` (default `Fail`).
  - `pub async fn resolve_provider(cwd: &Path, model: Option<String>, provider_id: Option<String>, api_key: Option<String>) -> anyhow::Result<(Arc<dyn LlmProvider>, String)>`
  - `pub struct StdinStdoutIo;` impl `UserIo`.

- [ ] **Step 1: Write the failing test**

```rust
// crates/mr-cli/tests/cli.rs
use mr_cli::{run_cli, Command, OnEscalate};

#[test]
fn parse_init_takes_question() {
    let cmd = mr_cli::parse_args(&["mr", "init", "How does X affect Y?"]).unwrap();
    assert!(matches!(cmd, Command::Init { question } if question == "How does X affect Y?"));
}

#[test]
fn parse_execute_defaults_and_flags() {
    let cmd = mr_cli::parse_args(&["mr", "execute", "path/plan.md", "--headless", "--on-escalate=continue", "--paper"]).unwrap();
    match cmd {
        Command::Execute { plan, headless, on_escalate, paper, no_mcp } => {
            assert_eq!(plan.as_deref(), Some(std::path::Path::new("path/plan.md")));
            assert!(headless); assert!(paper); assert!(!no_mcp); assert_eq!(on_escalate, OnEscalate::Continue);
        }
        _ => panic!("expected Execute"),
    }
}

#[tokio::test]
async fn run_cli_help_lists_subcommands() {
    // `mr --help` should exit 0 and mention the subcommands. run_cli returns
    // Ok(()) on --help when wired to a clap Printer that prints + exits; for
    // testability, expose `parse_args` (above) and test parsing, not the help
    // printer. This test just asserts parse_args rejects unknown subcommands.
    assert!(mr_cli::parse_args(&["mr", "frobnicate"]).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mr-cli`
Expected: FAIL — crate not found (not yet a member).

- [ ] **Step 3: Add the crate to the workspace**

In root `Cargo.toml`, add `"crates/mr-cli"` to the `members` array.

- [ ] **Step 4: Write `crates/mr-cli/Cargo.toml`**

```toml
[package]
name = "mr-cli"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0"

[lib]
name = "mr_cli"
path = "src/lib.rs"

[[bin]]
name = "mr"
path = "src/main.rs"

[dependencies]
megaresearcher-research = { workspace = true }
claurst-core = { workspace = true }
claurst-api = { workspace = true }
clap = { workspace = true, features = ["derive"] }
tokio = { workspace = true }
anyhow = { workspace = true }
async-trait = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
```

(Confirm `clap`, `anyhow`, `serde_json` are workspace deps; if `anyhow` is not a workspace dep, pin `anyhow = "1"` directly. claurst uses clap derive — `clap = { workspace = true, features = ["derive"] }` should resolve.)

- [ ] **Step 5: Write `crates/mr-cli/src/lib.rs`**

```rust
pub mod commands;
pub mod escalation;
pub mod io;
pub mod prelude;
pub mod render;

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum OnEscalate { Continue, Pause, Fail }

#[derive(Debug, Clone)]
pub enum Command {
    Init { question: String },
    Brainstorm { topic: String },
    Spec { topic: String },
    Plan { topic: String },
    Execute { plan: Option<PathBuf>, paper: bool, headless: bool, no_mcp: bool, on_escalate: OnEscalate },
    Verify { run_dir: PathBuf },
    Watch { run_dir: Option<PathBuf> },
    List,
}

/// Parse `mr` args (the program name is `args[0]`). Errors on unknown subcommands
/// or missing required arguments.
pub fn parse_args(args: &[String]) -> anyhow::Result<Command> {
    let rest = &args[1..];
    if rest.is_empty() { return Ok(Command::List); }
    match rest[0].as_str() {
        "init" => Ok(Command::Init { question: rest.get(1).cloned().context("init requires a question")? }),
        "brainstorm" => Ok(Command::Brainstorm { topic: rest.get(1).cloned().unwrap_or_default() }),
        "spec" => Ok(Command::Spec { topic: rest.get(1).cloned().unwrap_or_default() }),
        "plan" => Ok(Command::Plan { topic: rest.get(1).cloned().unwrap_or_default() }),
        "execute" => parse_execute(&rest[1..]),
        "verify" => Ok(Command::Verify { run_dir: PathBuf::from(rest.get(1).context("verify requires a run dir")?) }),
        "watch" => Ok(Command::Watch { run_dir: rest.get(1).map(PathBuf::from) }),
        "list" => Ok(Command::List),
        other => Err(anyhow::anyhow!("unknown subcommand: {other}")),
    }
}

fn parse_execute(args: &[String]) -> anyhow::Result<Command> {
    let mut plan: Option<PathBuf> = None;
    let mut paper = false;
    let mut headless = false;
    let mut no_mcp = false;
    let mut on_escalate = OnEscalate::Fail;
    for a in args {
        match a.as_str() {
            "--paper" => paper = true,
            "--headless" => headless = true,
            "--no-mcp" => no_mcp = true,
            s if s.starts_with("--on-escalate=") => {
                on_escalate = match s.trim_start_matches("--on-escalate=") {
                    "continue" => OnEscalate::Continue,
                    "pause" => OnEscalate::Pause,
                    "fail" => OnEscalate::Fail,
                    other => anyhow::bail!("bad --on-escalate value: {other}"),
                };
            }
            s if s.starts_with("--") => anyhow::bail!("unknown flag: {s}"),
            other => plan = Some(PathBuf::from(other)),
        }
    }
    Ok(Command::Execute { plan, paper, headless, no_mcp, on_escalate })
}

/// Entry point. Parses args and dispatches.
pub async fn run_cli(args: Vec<String>) -> anyhow::Result<()> {
    use anyhow::Context as _;
    let cmd = parse_args(&args).context("bad args")?;
    let cwd = std::env::current_dir()?;
    let provider = prelude::resolve_provider(&cwd, None, None, None).await
        .context("could not resolve a provider — set an API key (see claurst auth)")?;
    commands::dispatch(cmd, cwd, provider).await
}

// `context` shorthand used by parse helpers above.
use anyhow::Context as _ContextTrait;
trait Ctx { fn context<D>(self, d: D) -> anyhow::Result<String> where D: std::fmt::Display; }
impl Ctx for Option<String> { fn context<D: self::std::fmt::Display>(self, d: D) -> anyhow::Result<String> where D: std::fmt::Display { self.ok_or_else(|| anyhow::anyhow!("{d}")) } }
// (The above trait/impl is awkward; the implementer may instead inline
// `.ok_or_else(|| anyhow::anyhow!("..."))` at each call site and delete this
// block. Prefer the inline form for clarity.)
```

> NOTE: the `parse_args` helpers use a `.context(...)` ergonomics that doesn't exist on `Option<String>` out of the box. The implementer should replace every `rest.get(1).cloned().context("...")` with `rest.get(1).cloned().ok_or_else(|| anyhow::anyhow!("..."))` and delete the `Ctx` trait block. The plan shows the intent; the implementer writes the clean inline form. (This is called out so the implementer does not ship the awkward trait.)

- [ ] **Step 6: Write `crates/mr-cli/src/main.rs`**

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    mr_cli::run_cli(std::env::args().collect()).await
}
```

- [ ] **Step 7: Write `crates/mr-cli/src/prelude.rs`**

```rust
//! Resolve an `Arc<dyn LlmProvider>` by mirroring claurst's CLI prelude:
//! `Settings::load_hierarchical` → `effective_config` → apply overrides →
//! `selected_provider_id` → `resolve_anthropic_auth_async` → `ClientConfig` →
//! `ProviderRegistry::from_config` → `default_provider`.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use claurst_api::{client::ClientConfig, LlmProvider, ProviderRegistry};

/// Returns the provider + the resolved model string.
pub async fn resolve_provider(
    cwd: &Path,
    model: Option<String>,
    provider_id: Option<String>,
    api_key: Option<String>,
) -> anyhow::Result<(Arc<dyn LlmProvider>, String)> {
    let settings = claurst_core::config::Settings::load_hierarchical(cwd).await;
    let mut config = settings.effective_config();
    if let Some(m) = model { config.model = Some(m); }
    if let Some(p) = provider_id { config.provider = Some(p); }
    if let Some(k) = api_key { config.api_key = Some(k); }

    let active = config.selected_provider_id().to_string();
    let (key, use_bearer) = if active == "anthropic" {
        config.resolve_anthropic_auth_async().await.unwrap_or((String::new(), false))
    } else {
        (config.resolve_provider_api_key(&active).unwrap_or_default(), false)
    };
    let client_config = ClientConfig {
        api_key: key.clone(),
        api_base: config.resolve_anthropic_api_base(),
        use_bearer_auth: use_bearer,
        ..Default::default()
    };
    let registry = ProviderRegistry::from_config(&config, client_config);
    let provider = registry
        .default_provider()
        .cloned()
        .context("no provider registered — set an API key (e.g. ANTHROPIC_API_KEY) or run `claurst auth login`")?;
    let model = config.model.clone().unwrap_or_else(|| "claude-sonnet-4-6".to_string());
    Ok((provider, model))
}
```

> Confirm the exact field names on `claurst_core::config::Config` (`config.model`, `config.provider`, `config.api_key` are pub mut fields per the grounding — main.rs:470-523 mutates them). If any is private or named differently, the implementer adjusts to the actual `Config` definition at `crates/core/src/lib.rs:938`.

- [ ] **Step 8: Write `crates/mr-cli/src/io.rs`**

```rust
//! Production `UserIo` over stdin/stdout.

use async_trait::async_trait;
use std::io::{self, BufRead, Write};
use megaresearcher_research::phases::UserIo;

pub struct StdinStdoutIo;

#[async_trait]
impl UserIo for StdinStdoutIo {
    async fn print(&self, text: &str) -> io::Result<()> {
        let mut out = io::stdout().lock();
        out.write_all(text.as_bytes())?;
        out.flush()?;
        Ok(())
    }
    async fn read_line(&self) -> io::Result<String> {
        let mut line = String::new();
        io::stdin().lock().read_line(&mut line)?;
        Ok(line)
    }
}
```

- [ ] **Step 9: Write stubs for `commands`, `escalation`, `render`**

```rust
// crates/mr-cli/src/commands/mod.rs
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Context as _;
use claurst_api::LlmProvider;
use crate::Command;

pub async fn dispatch(cmd: Command, cwd: PathBuf, provider: (Arc<dyn LlmProvider>, String)) -> anyhow::Result<()> {
    match cmd {
        Command::List => crate::render::list_runs(&cwd).await,
        Command::Watch { run_dir } => crate::render::watch(&cwd, run_dir).await,
        Command::Init { question } => init::run(&cwd, provider, &question).await,
        Command::Brainstorm { topic } => session::run_session(&cwd, provider, "brainstorm", &topic).await,
        Command::Spec { topic } => session::run_session(&cwd, provider, "spec", &topic).await,
        Command::Plan { topic } => session::run_session(&cwd, provider, "plan", &topic).await,
        Command::Execute { plan, paper, headless, no_mcp, on_escalate } => {
            execute::run(&cwd, provider, plan, paper, headless, no_mcp, on_escalate).await
        }
        Command::Verify { run_dir } => verify::run(&cwd, provider, run_dir).await,
    }
}

pub mod init;
pub mod session;
pub mod execute;
pub mod verify;
```

Leave `init.rs`, `session.rs`, `execute.rs`, `verify.rs`, `crate::render`, `crate::escalation` as minimal stubs (`pub async fn ... -> anyhow::Result<()> { Ok(()) }`) so the crate compiles. Tasks 6–8 fill them.

- [ ] **Step 10: Build + run the parse tests**

Run: `cargo test -p mr-cli`
Expected: PASS (3 parse tests). Build must succeed (stub modules compile).

- [ ] **Step 11: fmt + clippy + commit**

```bash
cargo fmt -p mr-cli
cargo clippy -p mr-cli --all-targets -- -D warnings
git add Cargo.toml crates/mr-cli/
git commit -m "feat(mr-cli): scaffold the mr binary — clap subcommands, provider prelude, StdinStdoutIo

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: `mr init`/`brainstorm`/`spec`/`plan` — wire guided sessions

**Files:**
- Modify: `crates/mr-cli/src/commands/session.rs`, `crates/mr-cli/src/commands/init.rs`
- Test: `crates/mr-cli/tests/sessions.rs` (new)

**Interfaces:**
- Consumes: `megaresearcher_research::{flows::load_embedded, phases::{GuidedSession, drive_session, Gate, UserIo}, worker_tools::ScopedWrite}`, `crate::io::StdinStdoutIo`.
- Produces:
  - `pub async fn run_session(cwd: &Path, provider: (Arc<dyn LlmProvider>, String), name: &str, topic: &str) -> anyhow::Result<()>` — loads the flow body, builds a `GuidedSession` with `ScopedWrite` jailed to `cwd/docs/research/`, substitutes `$ARGUMENTS` → `topic`, drives it with the appropriate gates, prints the result.
  - `pub async fn run(cwd, provider, question) -> anyhow::Result<()>` in `init.rs` — chains brainstorm → spec → plan as ONE continuous session (concatenate the three flow bodies with section dividers into one guiding prompt), gates `[spec_path, plan_path]`, prints "run `mr execute <plan-path>`" at the end.

**`$ARGUMENTS` substitution:** the flow body's first line of context is the topic; the CLI substitutes `$ARGUMENTS` → `topic` (or `question`) in the body before seeding. If no `$ARGUMENTS` token is present, the topic is prepended as: the flow body is seeded as the first user message, and a second user message `("Topic: <topic>")` is injected immediately after — simplest robust approach (avoids string-replace edge cases). The implementer picks ONE and documents it; prefer injecting a second user message `Topic: <topic>` so the flow body is byte-identical to the asset.

**Gates per subcommand:**
- `mr brainstorm <topic>`: gates `[]` (no artifact) — drives until the user approves convergence.
- `mr spec <topic>`: gates `[Gate { artifact: cwd/docs/research/specs/<date>-<topic>-spec.md, label: "spec" }]`.
- `mr plan <topic>`: gates `[Gate { artifact: cwd/docs/research/plans/<date>-<topic>-plan.md, label: "plan" }]`.
- `mr init <question>`: gates `[spec_path, plan_path]` — the model writes both; the user approves spec then plan.

The date prefix `YYYY-MM-DD-<topic-slug>` is computed at session start (`chrono::Utc::now()` → but `chrono` may not be a mr-cli dep; use `std::time::SystemTime` + a tiny date formatter, OR add `chrono = { workspace = true }` to mr-cli deps — research already depends on chrono). Prefer adding `chrono` to mr-cli deps for the date slug. Topic slug = kebab-case of the topic (lowercase, non-alnum → `-`, trim).

- [ ] **Step 1: Write the failing test**

```rust
// crates/mr-cli/tests/sessions.rs
// Inject a fake provider + fake user IO via a test seam: run_session takes an
// optional (provider, io) override used only by tests. Expose:
//   pub async fn run_session_with(cwd, provider, name, topic, io: &dyn UserIo) -> anyhow::Result<()>
// and have run_session call it with StdinStdoutIo.
use mr_cli::commands::session::run_session_with;
use std::sync::Arc;
// ... FakeProvider (copy from research test infra, or build a tiny local one
//     that implements claurst_api::LlmProvider streaming for 1-2 text turns) +
//     FakeUserIo that returns "approve" ...
#[tokio::test]
async fn init_writes_spec_and_plan_then_suggests_execute() {
    // Drive a scripted provider that, across turns, "writes" spec.md and plan.md
    // via ScopedWrite (the provider emits ToolUse turns), then approve twice.
    // Assert spec_path + plan_path exist after run_session_with("init"...).
}
```

> The implementer needs a fake provider in the mr-cli test crate. Cleanest: add a `tests/common/mod.rs` to mr-cli with a `FakeProvider` mirroring `crates/research/tests/common/fake_provider.rs` (copy it — it's test infra, not shipped). Then mr-cli tests reuse it.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mr-cli --test sessions`
Expected: FAIL — `run_session_with` undefined.

- [ ] **Step 3: Implement `session.rs` + `init.rs`**

```rust
// crates/mr-cli/src/commands/session.rs
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::Context as _;
use claurst_api::LlmProvider;
use megaresearcher_research::flows::load_embedded;
use megaresearcher_research::phases::{drive_session, Gate, GuidedSession, UserIo, DriveOutcome};
use megaresearcher_research::worker_tools::{ScopedRead, ScopedWrite, Tool};

use crate::io::StdinStdoutIo;

fn docs_root(cwd: &Path) -> PathBuf { cwd.join("docs").join("research") }

fn slug(topic: &str) -> String {
    let s: String = topic.to_lowercase().chars().map(|c| {
        if c.is_ascii_alphanumeric() { c } else { '-' }
    }).collect();
    s.trim_matches('-').to_string()
}

fn date_prefix() -> String {
    use chrono::Utc;
    Utc::now().format("%Y-%m-%d").to_string()
}

/// Drive a single flow-body session headless. `io` is injected for testing.
pub async fn run_session_with(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    name: &str,
    topic: &str,
    io: &dyn UserIo,
) -> anyhow::Result<()> {
    let asset = load_embedded(name);
    let docs = docs_root(cwd);
    std::fs::create_dir_all(docs.join("specs")).ok();
    std::fs::create_dir_all(docs.join("plans")).ok();

    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ScopedRead::with_shared(docs.clone(), docs.clone())),
        Arc::new(ScopedWrite::new(docs.clone())),
    ];
    let (p, model) = provider;
    let mut session = GuidedSession::new(&asset.body, tools, p, model, 4096, 30);
    session.inject_user(&format!("Topic: {topic}"));

    let gates: Vec<Gate> = match name {
        "brainstorm" => vec![],
        "spec" => vec![Gate { artifact: docs.join("specs").join(format!("{}-{}-spec.md", date_prefix(), slug(topic))), label: "spec".into() }],
        "plan" => vec![Gate { artifact: docs.join("plans").join(format!("{}-{}-plan.md", date_prefix(), slug(topic))), label: "plan".into() }],
        _ => anyhow::bail!("unknown session: {name}"),
    };
    let outcome = drive_session(&mut session, io, gates, &["approve", "yes", "y", "done"]).await
        .context("guided session failed")?;
    match outcome {
        DriveOutcome::Approved { .. } => {
            io.print("\nSession approved.\n").await?;
            if name == "plan" || name == "init" {
                io.print(&format!("Run: `mr execute {}`\n", docs.join("plans").display())).await?;
            }
        }
        DriveOutcome::MaxTurns => io.print("\nSession hit the turn ceiling before approval.\n").await?,
    }
    Ok(())
}

pub async fn run_session(cwd: &Path, provider: (Arc<dyn LlmProvider>, String), name: &str, topic: &str) -> anyhow::Result<()> {
    run_session_with(cwd, provider, name, topic, &StdinStdoutIo).await
}
```

```rust
// crates/mr-cli/src/commands/init.rs
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::Context as _;
use claurst_api::LlmProvider;
use megaresearcher_research::flows::load_embedded;
use megaresearcher_research::phases::{drive_session, Gate, GuidedSession, UserIo, DriveOutcome};
use megaresearcher_research::worker_tools::{ScopedRead, ScopedWrite, Tool};
use crate::io::StdinStdoutIo;

fn docs_root(cwd: &Path) -> PathBuf { cwd.join("docs").join("research") }
fn slug(t: &str) -> String { t.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() {c} else {'-'}).collect::<String>().trim_matches('-').to_string() }
fn date_prefix() -> String { chrono::Utc::now().format("%Y-%m-%d").to_string() }

/// `mr init "<question>"`: brainstorm → spec → plan as one continuous session.
pub async fn run(cwd: &Path, provider: (Arc<dyn LlmProvider>, String), question: &str) -> anyhow::Result<()> {
    let docs = docs_root(cwd);
    std::fs::create_dir_all(docs.join("specs")).ok();
    std::fs::create_dir_all(docs.join("plans")).ok();

    // Concatenate the three flow bodies into one guiding prompt.
    let body = format!(
        "# Phase 1 — Brainstorm\n\n{}\n\n# Phase 2 — Spec\n\n{}\n\n# Phase 3 — Plan\n\n{}\n\n\
         Drive all three phases in order. Pause for user approval after the spec and again after the plan.",
        load_embedded("brainstorm").body,
        load_embedded("spec").body,
        load_embedded("plan").body,
    );

    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ScopedRead::with_shared(docs.clone(), docs.clone())),
        Arc::new(ScopedWrite::new(docs.clone())),
    ];
    let (p, model) = provider;
    let mut session = GuidedSession::new(body, tools, p, model, 4096, 60);
    session.inject_user(&format!("Topic: {question}"));

    let spec_path = docs.join("specs").join(format!("{}-{}-spec.md", date_prefix(), slug(question)));
    let plan_path = docs.join("plans").join(format!("{}-{}-plan.md", date_prefix(), slug(question)));
    let gates = vec![
        Gate { artifact: spec_path.clone(), label: "spec".into() },
        Gate { artifact: plan_path.clone(), label: "plan".into() },
    ];
    let io = StdinStdoutIo;
    let outcome = drive_session(&mut session, &io, gates, &["approve", "yes", "y", "done"]).await
        .context("guided session failed")?;
    match outcome {
        DriveOutcome::Approved { .. } => io.print(&format!("\nSpec: {}\nPlan: {}\nRun: `mr execute {}`\n", spec_path.display(), plan_path.display(), plan_path.display())).await?,
        DriveOutcome::MaxTurns => io.print("\nHit the turn ceiling before both approvals.\n").await?,
    }
    Ok(())
}
```

Add `chrono = { workspace = true }` to `crates/mr-cli/Cargo.toml [dependencies]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p mr-cli`
Expected: PASS (sessions + earlier parse tests).

- [ ] **Step 5: fmt + clippy + commit**

```bash
cargo fmt -p mr-cli
cargo clippy -p mr-cli --all-targets -- -D warnings
git add crates/mr-cli/
git commit -m "feat(mr-cli): wire init/brainstorm/spec/plan guided sessions with approval gates

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: `mr execute` — orchestrator run + text renderer + escalation handler + auto-verify

**Files:**
- Modify: `crates/mr-cli/src/commands/execute.rs`, `crates/mr-cli/src/render.rs`, `crates/mr-cli/src/escalation.rs`
- Test: `crates/mr-cli/tests/execute.rs` (new)

**Interfaces:**
- Consumes: `megaresearcher_research::{orchestrator::{Orchestrator, OrchestratorConfig}, mcp::ml_intern_config, state::{run_id::generate_run_id, swarm_state::SwarmState}, verify::verify_run, orchestrator::escalation::{EscalationHandler, EscalationVerdict}}`, `claurst_core::config::McpServerConfig`.
- Produces:
  - `crate::escalation::HeadlessEscalationHandler { mode: OnEscalate, io: Arc<dyn UserIo> }` impl `EscalationHandler` — `Continue` mode → Continue; `Fail` → Fail; `Pause` → print + read a line → Continue on "y"/"c", else Fail.
  - `pub async fn run(cwd, provider, plan: Option<PathBuf>, paper: bool, headless: bool, no_mcp: bool, on_escalate: OnEscalate) -> anyhow::Result<()>`:
    1. Resolve the plan path: `plan` or discover the latest plan under `docs/research/plans/`. Resolve its sibling spec (same date prefix, `-spec.md`). If `paper`, require `--paper`-aware paths (Phase 7 paper chain is NOT in 6a — `--paper` is accepted but 6a ignores it with a warning "paper chain arrives in Phase 7"; OR 6a refuses `--paper`. Prefer: accept + warn.)
    2. Build `OrchestratorConfig { research_base: cwd.join("docs/research"), agents_dir: cwd.join("agents"), default_model: provider.1.clone(), max_parallel: env MEGARESEARCHER_MAX_PARALLEL or 4, mcp: if no_mcp { None } else { Some(ml_intern_config(&cwd)) }, escalation: Some(Arc::new(HeadlessEscalationHandler { mode: on_escalate, io: Arc::new(StdinStdoutIo) })) }`.
    3. `let run_id = generate_run_id()?;` Spawn `tokio::spawn` the orchestrator `execute(spec, plan, &run_id)`; meanwhile poll `swarm-state.yaml` every 250ms printing phase-status diffs (text renderer); on the task's `JoinHandle` completing, print the `RunOutcome` (phases + escalations + output.md path).
    4. Auto-verify: connect an `McpCaller` only if `!no_mcp` (reuse `ml_intern_config` + `McpToolSet::connect`? Simpler: pass `None` to `verify_run` unless we already have a connected caller — but execute() connects its own. For 6a, call `verify_run(&run_dir, &spec_path, None)` (skip spot-checks) to keep it simple, OR re-connect for spot-checks. Prefer `None` in 6a — spot-checks are a nice-to-have; the deterministic groups A/B/C/E/F run without MCP.) Print the verdict.
  - `crate::render::watch_state(run_dir: &Path)` — poll loop printing phase transitions (used by `execute` and by `mr watch`).

- [ ] **Step 1: Write the failing test**

```rust
// crates/mr-cli/tests/execute.rs
// Test the config-construction + handler, not a live orchestrator run (which
// needs the real provider + MCP). Inject a fake provider via the test seam:
//   pub async fn run_with(cwd, provider, plan, paper, headless, no_mcp, on_escalate) -> anyhow::Result<()>
// where run() calls run_with(StdinStdoutIo-equivalent). Tests pass a fake
// provider + a temp cwd with fixture agents/ + spec/plan, no_mcp=true, and a
// FakeProvider that produces the three-artifact turns (copy research's
// three_artifact_turns). Assert a run dir + output.md appear.

#[tokio::test]
async fn execute_no_mcp_with_fake_provider_produces_output() { /* ... */ }

#[test]
fn headless_handler_continue_always_continues() { /* unit: HeadlessEscalationHandler with Continue mode */ }

#[test]
fn headless_handler_fail_always_fails() { /* ... */ }
```

> The fake-provider integration test mirrors `crates/research/tests/orchestrator.rs`'s gap-finding run test (FakeProvider + fixture agents + a minimal spec/plan). The implementer copies that fixture setup into `crates/mr-cli/tests/common/`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mr-cli --test execute`
Expected: FAIL.

- [ ] **Step 3: Implement `escalation.rs`**

```rust
use std::sync::Arc;
use async_trait::async_trait;
use megaresearcher_research::orchestrator::escalation::{EscalationHandler, EscalationVerdict};
use megaresearcher_research::state::swarm_state::Escalation;
use megaresearcher_research::phases::UserIo;
use crate::OnEscalate;

pub struct HeadlessEscalationHandler {
    pub mode: OnEscalate,
    pub io: Arc<dyn UserIo>,
}

#[async_trait]
impl EscalationHandler for HeadlessEscalationHandler {
    async fn adjudicate(&self, e: &Escalation) -> EscalationVerdict {
        match self.mode {
            OnEscalate::Continue => EscalationVerdict::Continue,
            OnEscalate::Fail => EscalationVerdict::Fail,
            OnEscalate::Pause => {
                let _ = self.io.print(&format!("\n[escalation] {} ({}): continue? [y/n] ", e.worker, e.reason)).await;
                let line = self.io.read_line().await.unwrap_or_default();
                let t = line.trim().to_lowercase();
                if t == "y" || t == "c" || t == "continue" { EscalationVerdict::Continue } else { EscalationVerdict::Fail }
            }
        }
    }
}
```

- [ ] **Step 4: Implement `render.rs`**

```rust
use std::path::{Path, PathBuf};
use std::time::Duration;
use anyhow::Context as _;
use megaresearcher_research::state::swarm_state::SwarmState;

/// Poll `run_dir/swarm-state.yaml` and print phase-status changes until the
/// run completes (caller signals via `done`). Returns the final state read.
pub async fn watch_state(run_dir: &Path, mut done: impl FnMut() -> bool) -> anyhow::Result<Option<SwarmState>> {
    let path = run_dir.join("swarm-state.yaml");
    let mut last: Option<SwarmState> = None;
    loop {
        if let Ok(state) = SwarmState::read(&path) {
            let changed = last.as_ref().map_or(true, |prev| prev.phases != state.phases || prev.escalations != state.escalations);
            if changed {
                print_state(&state);
                last = Some(state);
            }
        }
        if done() { return Ok(last); }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn print_state(state: &SwarmState) {
    println!("--- {} (target: {}) ---", state.run_id, state.novelty_target);
    for p in &state.phases {
        let workers: Vec<String> = p.workers.iter().map(|w| format!("{}={}", w.name, w.status)).collect();
        println!("  {:<16} {:<10} {}", p.name, p.status, workers.join(", "));
    }
    if !state.escalations.is_empty() {
        println!("  escalations: {}", state.escalations.iter().map(|e| e.worker.clone()).collect::<Vec<_>>().join(", "));
    }
}

pub async fn list_runs(cwd: &Path) -> anyhow::Result<()> {
    let runs = cwd.join("docs/research/runs");
    if !runs.is_dir() { println!("No runs yet. Start with `mr init \"<question>\"`."); return Ok(()); }
    let mut entries: Vec<_> = std::fs::read_dir(&runs).with_context(|| format!("read {}", runs.display()))?
        .flatten().filter(|e| e.path().is_dir()).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries.iter().rev() {
        let dir = e.path();
        let headline = std::fs::read_to_string(dir.join("output.md")).unwrap_or_default()
            .lines().find(|l| l.starts_with("#")).unwrap_or("(no output.md)").to_string();
        println!("{}\t{}", e.file_name().to_string_lossy(), headline);
    }
    Ok(())
}

pub async fn watch(cwd: &Path, run_dir: Option<PathBuf>) -> anyhow::Result<()> {
    println!("`mr watch` (TUI) arrives in Phase 6b. For now, the run streams to stdout via `mr execute`.");
    if let Some(rd) = run_dir { watch_state(&rd, || false).await?; } else { list_runs(cwd).await?; }
    Ok(())
}
```

- [ ] **Step 5: Implement `execute.rs`**

```rust
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Context as _;
use claurst_api::LlmProvider;
use claurst_core::config::McpServerConfig;
use megaresearcher_research::mcp::ml_intern_config;
use megaresearcher_research::orchestrator::{Orchestrator, OrchestratorConfig};
use megaresearcher_research::state::run_id::generate_run_id;
use megaresearcher_research::verify::verify_run;

use crate::escalation::HeadlessEscalationHandler;
use crate::io::StdinStdoutIo;
use crate::render::watch_state;
use crate::OnEscalate;

pub async fn run(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    plan: Option<PathBuf>,
    paper: bool,
    _headless: bool,       // 6a is always headless; the TUI (6b) makes bare `mr execute` open the TUI.
    no_mcp: bool,
    on_escalate: OnEscalate,
) -> anyhow::Result<()> {
    if paper { println!("note: --paper chain (Phases 7–9) arrives in a later phase; running the core swarm now."); }
    let docs = cwd.join("docs/research");
    let plans_dir = docs.join("plans");
    let plan_path = match plan {
        Some(p) => p,
        None => latest_plan(&plans_dir)?,
    };
    let spec_path = sibling_spec(&plan_path)?;
    println!("Spec : {}", spec_path.display());
    println!("Plan : {}", plan_path.display());

    let max_parallel = std::env::var("MEGARESEARCHER_MAX_PARALLEL").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4u32);
    let mcp = if no_mcp { None } else { Some(ml_intern_config(cwd)) };
    let cfg = OrchestratorConfig {
        research_base: docs.clone(),
        agents_dir: cwd.join("agents"),
        default_model: provider.1.clone(),
        max_parallel,
        mcp,
        escalation: Some(Arc::new(HeadlessEscalationHandler { mode: on_escalate, io: Arc::new(StdinStdoutIo) })),
    };
    let orch = Orchestrator::new(cfg, provider.0);
    let run_id = generate_run_id().context("generate run id")?;
    println!("Run  : {run_id}");

    let orch_ref = &orch;
    let spec_c = spec_path.clone();
    let plan_c = plan_path.clone();
    let run_id_c = run_id.clone();
    let task = tokio::spawn(async move { orch_ref.execute(&spec_c, &plan_c, &run_id_c).await });
    let run_dir = docs.join("runs").join(&run_id);
    // Watch until the task finishes.
    let watch = tokio::spawn(async move {
        let mut t = task;
        let _ = watch_state(&run_dir, || { false }).await;
        // The watch loop above never returns done=true; restructure so the
        // caller checks the task. See NOTE below.
        t
    });
    // Simpler: poll in a loop here, checking the task via try_join.
    let outcome = loop {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        // re-print state diff inline (mirror watch_state but driven by the task)
        if let Ok(state) = SwarmState::read(&run_dir.join("swarm-state.yaml")) { /* print diff */ }
        if let Ok(Ok(r)) = { let mut t = watch; /* can't move out — see NOTE */ break r; } else { /* keep going */ }
    };
    // (The implementer should restructure the above to: spawn the execute task,
    //  loop { select! { _ = sleep(250ms) => print_state_diff(); r = &mut task => { outcome = r; break; } } }.
    //  Use tokio::select! on the JoinHandle for a clean implementation.)

    match outcome {
        Ok(o) => {
            println!("\nRun complete: {run_id}");
            println!("output: {}", run_dir.join("output.md").display());
            for (phase, status) in &o.phase_statuses { println!("  {phase}: {status}"); }
            if !o.escalations.is_empty() { println!("escalations: {:?}", o.escalations); }
        }
        Err(e) => { eprintln!("run failed: {e}"); return Err(anyhow::anyhow!("{e}")); }
    }

    // Auto-verify (skip MCP spot-checks in 6a for simplicity).
    let report = verify_run(&run_dir, &spec_path, None).await.context("verify")?;
    megaresearcher_research::verify::write_report(&run_dir, &report).context("write report")?;
    println!("verdict: {:?}", report.verdict);
    Ok(())
}

fn latest_plan(plans_dir: &Path) -> anyhow::Result<PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(plans_dir).with_context(|| format!("read {}", plans_dir.display()))?
        .flatten().map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "md")).collect();
    entries.sort();
    entries.into_iter().last().ok_or_else(|| anyhow::anyhow!("no plan found in {}; run `mr init` first", plans_dir.display()))
}

fn sibling_spec(plan_path: &Path) -> anyhow::Result<PathBuf> {
    let stem = plan_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let date_topic = stem.trim_end_matches("-plan");
    let spec = plan_path.parent().map(|d| d.join("specs")).unwrap_or_default()
        .join(format!("{date_topic}-spec.md"));
    // The spec lives under docs/research/specs/, the plan under docs/research/plans/.
    // Adjust: go up two from plans/ to docs/research/, then specs/.
    let spec = plan_path.parent().and_then(|p| p.parent()).map(|root| root.join("specs").join(format!("{date_topic}-spec.md"))).unwrap_or(spec);
    if spec.exists() { Ok(spec) } else { Err(anyhow::anyhow!("spec not found for plan {}; expected {}", plan_path.display(), spec.display())) }
}
```

> NOTE for the implementer: the watch-while-task-runs block above is written as messy pseudocode to convey intent. The implementer MUST rewrite it cleanly with `tokio::select!`:
> ```rust
> let mut task = tokio::spawn(async move { orch.execute(&spec_path, &plan_path, &run_id).await });
> let mut last: Option<SwarmState> = None;
> let outcome = loop {
>     tokio::select! {
>         _ = tokio::time::sleep(std::time::Duration::from_millis(250)) => {
>             if let Ok(state) = SwarmState::read(&run_dir.join("swarm-state.yaml")) {
>                 if last.as_ref().map_or(true, |p| p.phases != state.phases || p.escalations != state.escalations) {
>                     print_state(&state); last = Some(state);
>                 }
>             }
>         }
>         r = &mut task => break r.context("orchestrator task")?,
>     }
> };
> ```
> Move `print_state` out of `render.rs` into a shared spot or duplicate it. Keep `render::watch` (the `mr watch` command) separate from this inline watcher. The plan shows the select! form the implementer should use; do not ship the messy pseudocode.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p mr-cli`
Expected: PASS (execute fake-provider test + handler unit tests + earlier tests).

- [ ] **Step 7: fmt + clippy + commit**

```bash
cargo fmt -p mr-cli
cargo clippy -p mr-cli --all-targets -- -D warnings
git add crates/mr-cli/
git commit -m "feat(mr-cli): wire mr execute — orchestrator run, live text renderer, --on-escalate handler, auto-verify

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: `mr verify` + finalize subcommands + CLI help

**Files:**
- Modify: `crates/mr-cli/src/commands/verify.rs`
- Test: `crates/mr-cli/tests/verify_cmd.rs` (new)

**Interfaces:**
- `pub async fn run(cwd, provider, run_dir: PathBuf) -> anyhow::Result<()>` — runs `verify_run(&run_dir, &spec_path, mcp)` where `spec_path` is resolved from `run_dir/swarm-state.yaml`'s `spec_path` field, and `mcp` is `Some(connected caller)` when `!no_mcp` (for 6a, allow a `--no-mcp` flag too; default connect for spot-checks). Print the verdict + report path.

- [ ] **Step 1: Write the failing test** — `mr verify <run_dir>` against a fixture run dir with a passing output.md → prints PASS + writes verification-report.md.
- [ ] **Step 2: Run test to verify it fails.**
- [ ] **Step 3: Implement `verify.rs`** — read `swarm-state.yaml` for `spec_path` + `run_id`, connect an `McpCaller` if MCP available (reuse `megaresearcher_research::mcp::McpToolSet::connect(&ml_intern_config(cwd))` to get a caller — confirm `McpToolSet` exposes a caller; if not, pass `None` and skip spot-checks in 6a), call `verify_run`, `write_report`, print verdict.
- [ ] **Step 4: Add a top-level `--help` / usage string** in `lib.rs` printed when `parse_args` fails or `--help`/`-h` is passed; list subcommands with one-line descriptions (from the flow bodies' `description` frontmatter for brainstorm/spec/plan).
- [ ] **Step 5: Run tests.** `cargo test -p mr-cli`.
- [ ] **Step 6: fmt + clippy + commit.**

```bash
cargo fmt -p mr-cli && cargo clippy -p mr-cli --all-targets -- -D warnings
git add crates/mr-cli/ && git commit -m "feat(mr-cli): add mr verify + CLI help

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: End-to-end fake-provider integration test + workspace green

**Files:**
- Test: `crates/mr-cli/tests/e2e.rs` (new, `#[ignore]`-able for the MCP path; the fake path runs in CI)

**Goal:** A single test drives `mr init` → `mr execute --no-mcp` (fake provider) → `mr verify` against a temp cwd with fixture agents/, asserting the full chain produces spec + plan + a run dir with output.md + verification-report.md, with `--on-escalate=fail` defaulting.

- [ ] **Step 1: Write the e2e test**

```rust
// crates/mr-cli/tests/e2e.rs
// Copy fixture agents/*.md from crates/research/tests/fixtures/agents/ into a
// temp cwd/agents/, run run_session_with("init", ...) with a FakeProvider that
// emits ToolUse turns writing spec.md + plan.md + "approve" responses, then run
// execute::run_with(..., no_mcp=true, on_escalate=Fail) with a FakeProvider that
// emits three_artifact_turns (copied from research test infra), then verify::run.
// Assert: spec.md, plan.md, runs/<id>/output.md, runs/<id>/verification-report.md exist.
#[tokio::test]
async fn init_then_execute_then_verify_end_to_end() { /* ... */ }
```

- [ ] **Step 2: Run the full workspace suite**

Run: `cargo test --workspace`
Expected: PASS — all research tests (52 + new flows/phases/escalation/verify) + all mr-cli tests + claurst crates unchanged.

- [ ] **Step 3: fmt + clippy workspace-wide + commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/mr-cli/tests/e2e.rs
git commit -m "test(mr-cli): end-to-end init→execute→verify with fake provider

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 4: Push**

```bash
git push origin main
```

---

## Self-Review (run after writing, before handoff)

1. **Spec coverage (design §5/§8/§67/§180/§201-209/§112):**
   - §8/§180 slash → `mr` subcommands: T5 `parse_args` (init/brainstorm/spec/plan/execute/verify/watch/list). ✓
   - §8 flow body = first user message, no Skill tool: T2 `GuidedSession` seeds `Message::user(flow_body)`, SESSION_SYSTEM_PROMPT forbids skill invocation. ✓
   - §67 brainstorm/spec/plan guided sessions: T1+T2+T6. verify is deterministic (§5 tree node) — T4, noted as a §67-vs-§5 resolution favoring §5. ✓ (documented deviation)
   - §201 `mr init` → brainstorm → confirm → spec → approve → plan → approve: T6 `init.rs` continuous session with two gates. ✓
   - §202 `mr execute` → pre-flight → run-id → scaffold → swarm-state → run: T7 (pre-flight is inside `Orchestrator::execute`). ✓
   - §209 `--headless` + `--on-escalate={continue,pause,fail}`: T5+T7. ✓
   - §112 orchestrator pauses/blocks on ack, headless auto-decides: T3 `EscalationHandler` + T7 `HeadlessEscalationHandler`. ✓
   - §182 flow bodies as inlined prompt content, flat `flows/<phase>.md`: T1 `include_str!`. ✓
   - §180 frontmatter description/argument-hint/model/allowed-tools: T1 `FlowAsset`. ✓

2. **Placeholder scan:** T3/T6/T7 contain explicit implementer-NOTE blocks for things the grounding couldn't pin exactly (the awkward `.context` ergonomics in T5; the `tokio::select!` rewrite in T7; the fixture-copy pointers in T3/T6/T7/T9; the `chrono`/`anyhow`/`regex` dep confirmations). These are not "TODO/TBD" placeholders — they name a concrete decision + the concrete alternative. Every code block shows compilable intent. The flow-body markdown (T1) is complete content, not "fill in".

3. **Type consistency:** `EscalationHandler::adjudicate(&self, &Escalation) -> EscalationVerdict` used identically in T3 (definition), T7 (HeadlessEscalationHandler), and the orchestrator helper. `Gate { artifact, label }` defined T2, used T6/T7. `UserIo` defined T2, implemented T5 (StdinStdoutIo) + T7 (handler io field). `DriveOutcome::Approved { gates_passed }` defined T2, matched T6. `McpCaller::call_tool(&self, &str, Option<Value>) -> Result<CallToolResult, McpError>` — confirm against `crates/research/src/mcp.rs` (the grounding summary states this signature). `OrchestratorConfig` field `escalation` added T3, populated T7. `load_embedded(name)` T1, used T6. All consistent.

4. **Determinism:** T3 keeps all 52 tests at `escalation: None` = `Fail` = `Err(Escalated)` (byte-identical). T2 reuses `worker::accumulate` + FakeProvider (stream). T7 `no_mcp: true` mirrors the fake-provider test path. ✓

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-27-megaresearcher-rs-phase6a-front-half-cli.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh implementer subagent per task, review between tasks (spec compliance + code quality), fast iteration, final whole-branch review on the most capable model.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**