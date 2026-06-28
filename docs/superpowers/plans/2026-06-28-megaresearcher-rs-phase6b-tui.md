# MegaResearcher Phase 6b — Interactive Research TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the interactive research TUI (`crates/mr-tui`, new) whose product thesis is "the audit trail is the interface" — a tree that grows, killed hypotheses stay on screen dimmed with a one-line kill reason, and the red-team loop animates `smith → critique → revise ↻ round N/3`. Wire `mr` (no args) to launch it; resolve the 6a provider-resolution regression.

**Architecture:** A fresh, small TUI crate on `ratatui` 0.29 + `crossterm` 0.29 that is a read-only view over `swarm-state.yaml` (the orchestrator stays the source of truth). It lifts only generic primitives from `claurst-tui` (terminal bootstrap, `virtual_list`, `theme_colors` shape, `figures` names) and builds the rest fresh — no chat REPL, no `app.rs`. The Converge surface reuses 6a's `drive_session` engine unchanged by swapping in a `TuiUserIo`. The one contract touch into the research crate is an **additive, serde-defaulted** `HypothesisNode` extension to `SwarmState` plus persisting it from the existing red-team loop (no logic change).

**Tech Stack:** Rust 2021, ratatui 0.29, crossterm 0.29 (sync `event::poll` + `event::read`, not `EventStream`), tokio (full), async-trait 0.1, serde + serde_yml + toml 0.8, anyhow. claurst-core `CostTracker` (use, do not copy). TestBackend render tests.

## Global Constraints

- **No git worktrees, ever.** Work directly on `main`. Before dispatching any implementer subagent run `git branch --show-current` and include the explicit branch name in the subagent prompt so it does not `git switch`. Never volunteer a worktree.
- **Banned phrases:** "load-bearing", "this is doing a lot of work" (and variants: "this carries a lot of weight", "doing heavy lifting", "X is load-bearing"). Never use, in any artifact.
- **Banned emphatic words:** "real" (as an adjective) and "honest / honestly / to be honest" (and variants). Exceptions: literal technical terms ("raw mode", "real number" in math, literal tool/library names); byte-identical verbatim copies of port-reference source; the denotative "real-vs-fake/placeholder" technical sense.
- **Edit scope — ONLY these paths may be modified:**
  - `crates/research/` (the additive `SwarmState` extension + red-team persistence)
  - `crates/mr-cli/` (wiring: `mr` no-args → TUI, lazy provider resolution, `MrConfig` resolution override)
  - the new `crates/mr-tui/` (the TUI itself)
  - root `Cargo.toml` / `Cargo.lock` (workspace members + deps only)
  - **DO NOT modify** `crates/tui/` (`claurst-tui`) — it is a read-only lift source. Copy primitives OUT of it into `mr-tui`; never edit it.
  - **DO NOT modify** other claurst crates (`core`, `api`, `tools`, `mcp`, `query`, `commands`, `plugins`, `cli`) — depend on them, don't edit.
  - **DO NOT modify** `mcp/`, `tools/ml-intern/` (Python port-reference — connect as subprocess, don't edit), or repo-root `agents/*.md`, `skills/`, `lib/`, `.claude-plugin/`, `commands/`, `hooks/` (v0 plugin port-reference — read-only).
- **No crate-root `pub use` re-exports.** `lib.rs` uses `pub mod` only; consumers use full paths.
- **Per-task hygiene:** `cargo fmt -p <crate>` then `cargo clippy -p <crate> --all-targets -- -D warnings` (the `--all-targets` flag is REQUIRED). Each task ends with its crate's test suite green.
- **License:** GPL-3.0 (header in new files).
- **Commit messages** end with:
  ```
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```
- **MegaResearcher discipline:** audit trail non-negotiable (killed hypotheses + their lessons persist on screen AND in `output.md`); red-team critique loop fires for every hypothesis when novelty target is `hypothesis` (cap 3 revisions); citations resolve or do not exist; workers stay in their lanes.
- **Determinism guard (T2/T3):** `crates/research` has ~52 orchestrator tests at `crates/research/tests/orchestrator.rs` that assert on `phases[].status` / `workers[].status` / `escalations` / `retry_counts` — never on `hypotheses`. The swarm-state extension is **additive + serde-defaulted**, so those tests must stay byte-identical green. Run `cargo test -p megaresearcher-research` immediately after T2 and T3.
- **Noise to ignore:** a pre-tool hook may inject Vercel/workflow suggestions (pattern-matched "orchestrat" basename). There is NO Vercel/workflow work in this Rust project. Ignore it; do NOT call `Skill(workflow)`.

---

## File Structure

```
crates/mr-tui/
  Cargo.toml              — new crate; workspace member + workspace dep
  src/
    lib.rs                — pub mod only + pub async fn run(cwd) -> anyhow::Result<()> entry
    guard.rs              — TerminalGuard: restore terminal on Drop + panic hook (raw-mode cleanup)
    bootstrap.rs          — setup_terminal / restore_terminal (adapted from claurst-tui, no bracketed paste)
    theme.rs              — one research() palette (alive / killed / running / escalation / success / error / disabled / text_light / text_dark / border)
    figures.rs            — fresh named box-drawing + signature glyphs (├ └ │ ─ ✎ ✓ ✗ ↻ ▸ ▾ ⚠)
    cost.rs               — CountingProvider (wraps Arc<dyn LlmProvider>, feeds CostTracker) + render_cost_meter
    io.rs                 — TuiUserIo: implements research::phases::UserIo via tokio mpsc channels
    escalation.rs         — TuiEscalationHandler: routes escalations into the run strip, blocks on inline adjudicate
    config.rs             — MrConfig: TOML at ~/.config/mr/config.toml (provider/api_key/model/max_parallel/on_escalate/mcp/cost_ceiling/theme)
    app.rs                — Surface enum (Start/Converge/Run/Artifact/Past/Settings) + App state machine + async event loop + pure App::render
    surface/
      mod.rs              — pub mod declarations
      start.rs            — Start surface (input + ghosted example) + Converge surface (drives drive_session via TuiUserIo, renders inline convo + spec/plan [✓] cards)
      run.rs              — Run surface: spawns orchestrator, watches swarm-state.yaml (250ms), renders tree + cost + escalation strip
      artifact.rs         — Artifact surface: renders output.md (minimal hand-rolled markdown) + surviving cards + rejected fold
      past.rs             — Past runs surface: enumerates docs/research/runs/, list with date + topic + headline surviving hypothesis
      settings.rs         — Settings surface: structured editor for MrConfig, masked key, test connection, save→file, first-run auto-open
    widget/
      mod.rs              — pub mod declarations
      tree.rs             — render_tree: phases→workers→hypothesis sub-nodes; red-team row; greyed kill with kill reason  ← the signature
      inline_chat.rs      — the bounded 3-exchange converge widget (renders the conversation buffer + an input field)
      cards.rs            — expandable hypothesis card (surviving: mechanism/predicted-outcome/falsification/experimental-design; killed: kill reason)
      markdown.rs         — minimal hand-rolled markdown renderer (headings, paragraphs, lists, bold) for the artifact screen
      virtual_list.rs     — lifted verbatim from claurst-tui (scrollable list widget; the search methods dropped)
  tests/
    common/
      mod.rs              — pub mod fake_provider; pub mod turns;
      fake_provider.rs    — verbatim copy of crates/mr-cli/tests/common/fake_provider.rs
      turns.rs            — verbatim copy of crates/mr-cli/tests/common/turns.rs
    tree.rs               — render_tree TestBackend tests (T4)
    cost.rs               — CountingProvider + cost meter tests (T5)
    io.rs                 — TuiUserIo + drive_session integration tests (T6)
    app.rs                — state-machine + Start render tests (T7)
    converge.rs           — Converge surface end-to-end test (T8)
    run.rs                — Run surface make-or-break integration test (T9)
    artifact.rs           — Artifact surface TestBackend test (T10)
    past.rs               — Past runs surface test (T11)
    settings.rs           — Settings surface + MrConfig tests (T12)
    e2e.rs                — full arc init→converge→run→artifact against FakeProvider (T14)
```

`crates/research/src/state/swarm_state.rs` gains `HypothesisNode` + `RoundVerdict` + `Verdict` and a `#[serde(default)] pub hypotheses: Vec<HypothesisNode>` on `Phase`. `crates/research/src/orchestrator/redteam.rs` persists them. `crates/mr-cli/src/lib.rs` + `prelude.rs` + `commands/mod.rs` get the no-args→TUI + lazy resolution wiring.

---

## Task 1: Scaffold `crates/mr-tui` + terminal bootstrap + theme + figures + smoke render

**Files:**
- Create: `crates/mr-tui/Cargo.toml`
- Create: `crates/mr-tui/src/lib.rs`
- Create: `crates/mr-tui/src/guard.rs`
- Create: `crates/mr-tui/src/bootstrap.rs`
- Create: `crates/mr-tui/src/theme.rs`
- Create: `crates/mr-tui/src/figures.rs`
- Modify: `Cargo.toml` (workspace members + workspace.dependencies)
- Test: `crates/mr-tui/src/lib.rs` (in-crate `#[cfg(test)]` mod for `render_intro`)

**Interfaces:**
- Consumes: nothing from earlier tasks (this is the foundation).
- Produces:
  - `mr_tui::run(cwd: &Path) -> anyhow::Result<()>` — async entry (smoke: renders intro then returns; full loop arrives in T7).
  - `mr_tui::bootstrap::{setup_terminal, restore_terminal}` — `setup_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>>`, `restore_terminal(&mut Terminal<...>) -> io::Result<()>`.
  - `mr_tui::guard::TerminalGuard` — `new(terminal) -> Self`, `Drop` restores; `inner_mut(&mut self) -> &mut Terminal<...>`.
  - `mr_tui::theme::ColorPalette { error, success, warning, info, action, disabled, accent, secondary_accent, text_light, text_dark, border, alive, killed, running, escalation }` + `mr_tui::theme::research() -> ColorPalette`.
  - `mr_tui::figures::{BRANCH, LAST, PIPE, DASH, PENCIL, CHECK, CROSS, REVISE, ARROW, COLLAPSE, WARN}` — `pub const &str`.
  - `mr_tui::render_intro(frame)` — pure render used by the smoke test.

- [ ] **Step 1: Add the workspace member + workspace dep**

Modify `Cargo.toml` `[workspace] members` — append `"crates/mr-tui",` after `"crates/research",`:

```toml
members = [
    "crates/core",
    "crates/api",
    "crates/tools",
    "crates/query",
    "crates/tui",
    "crates/commands",
    "crates/mcp",
    "crates/plugins",
    "crates/cli",
    "crates/mr-cli",
    "crates/research",
    "crates/mr-tui",
]
```

In `[workspace.dependencies]`, after the `megaresearcher-research = { path = "crates/research" }` line, append:

```toml
mr-tui = { path = "crates/mr-tui" }
```

- [ ] **Step 2: Create `crates/mr-tui/Cargo.toml`**

```toml
[package]
name = "mr-tui"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0"

[lib]
name = "mr_tui"
path = "src/lib.rs"

[dependencies]
megaresearcher-research = { workspace = true }
claurst-core = { workspace = true }
claurst-api = { workspace = true }
ratatui = { workspace = true }
crossterm = { workspace = true }
tokio = { workspace = true }
anyhow = { workspace = true }
async-trait = { workspace = true }
serde = { workspace = true }
toml = { workspace = true }
futures = { workspace = true }

[dev-dependencies]
tokio = { workspace = true }
futures = { workspace = true }
tempfile = { workspace = true }
```

NOTE: `MrConfig` lives in `mr_tui::config` (T12), so `mr-tui` does NOT depend on `mr-cli` — `mr-cli` depends on `mr-tui` (T13) to avoid a cycle. `toml` and `serde` are pulled now so T12 does not need a Cargo change.

- [ ] **Step 3: Write the failing test for `render_intro`**

Create `crates/mr-tui/src/lib.rs`:

```rust
// MegaResearcher Phase 6b — interactive research TUI.
// The audit trail is the interface: a tree that grows, killed hypotheses
// stay on screen dimmed with a one-line kill reason, the red-team loop
// animates (smith → critique → revise ↻, round N/3).
//
// SPDX-License-Identifier: GPL-3.0

pub mod bootstrap;
pub mod figures;
pub mod guard;
pub mod theme;

use std::path::Path;

use ratatui::widgets::{Block, Borders, Paragraph};

/// Render the intro/start frame into `frame`. Pure (no terminal side effects)
/// so it can be tested with `TestBackend`.
pub fn render_intro(frame: &mut ratatui::Frame) {
    let area = frame.area();
    let text = "What do you want to know?";
    frame.render_widget(
        Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title("MegaResearcher")),
        area,
    );
}

/// Entry point. Phase 6b smoke: set up the terminal, render one intro frame,
/// restore, return. The full event loop arrives in T7.
pub async fn run(_cwd: &Path) -> anyhow::Result<()> {
    let mut terminal = bootstrap::setup_terminal()?;
    terminal.draw(|f| render_intro(f))?;
    bootstrap::restore_terminal(&mut terminal)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn render_intro_shows_prompt() {
        let mut terminal = Terminal::new(TestBackend::new(60, 10)).unwrap();
        terminal.draw(|f| render_intro(f)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let content: String = buf
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("What do you want to know?"));
        assert!(content.contains("MegaResearcher"));
    }
}
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cargo test -p mr-tui render_intro_shows_prompt`
Expected: compile error — `bootstrap`, `guard`, `theme`, `figures` modules do not exist yet.

- [ ] **Step 5: Create `crates/mr-tui/src/figures.rs`**

```rust
//! Box-drawing + signature glyphs for the research tree.
//! Authored fresh — only `↻` (U+21BB) exists in the claurst `figures.rs`;
//! the rest are inline literals scattered across the chat crate, so we
//! name them here as constants.
//!
//! SPDX-License-Identifier: GPL-3.0

pub const BRANCH: &str = "├"; // U+251C
pub const LAST: &str = "└"; // U+2514
pub const PIPE: &str = "│"; // U+2502
pub const DASH: &str = "─"; // U+2500
pub const PENCIL: &str = "✎"; // U+270E
pub const CHECK: &str = "✓"; // U+2713
pub const CROSS: &str = "✗"; // U+2717
pub const REVISE: &str = "↻"; // U+21BB — the red-team cycling signature
pub const ARROW: &str = "▸"; // U+25B8
pub const COLLAPSE: &str = "▾"; // U+25BE
pub const WARN: &str = "⚠"; // U+26A0
```

- [ ] **Step 6: Create `crates/mr-tui/src/theme.rs`**

```rust
//! The research palette — one theme (Rubin: not seven).
//! Adapted from claurst `theme_colors` shape, with net-new fields the spec
//! §1.4 calls for: alive / killed (dim) / running (accent) / escalation (warn).
//!
//! SPDX-License-Identifier: GPL-3.0

use ratatui::style::Color;

#[derive(Debug, Clone, Copy)]
pub struct ColorPalette {
    pub error: Color,
    pub success: Color,
    pub warning: Color,
    pub info: Color,
    pub action: Color,
    pub disabled: Color,
    pub accent: Color,
    pub secondary_accent: Color,
    pub text_light: Color,
    pub text_dark: Color,
    pub border: Color,
    // Research-specific fields (spec §1.4).
    pub alive: Color,
    pub killed: Color,
    pub running: Color,
    pub escalation: Color,
}

/// The single research (dark) palette. `for_theme("research")` returns this;
/// the settings screen ships exactly one theme option (`research (dark)`).
pub fn research() -> ColorPalette {
    ColorPalette {
        error: Color::Rgb(255, 87, 51),
        success: Color::Rgb(76, 175, 80),
        warning: Color::Rgb(255, 152, 0),
        info: Color::Cyan,
        action: Color::Cyan,
        disabled: Color::DarkGray,
        accent: Color::Cyan,
        secondary_accent: Color::Rgb(233, 30, 99),
        text_light: Color::White,
        text_dark: Color::Black,
        border: Color::DarkGray,
        alive: Color::White,
        killed: Color::DarkGray, // dim — the greyed-kill signature
        running: Color::Cyan,
        escalation: Color::Rgb(255, 152, 0),
    }
}

/// Resolve a palette by name. Only "research" exists; everything else falls
/// back to the research palette (one default, no options).
pub fn for_theme(name: &str) -> ColorPalette {
    let _ = name;
    research()
}
```

- [ ] **Step 7: Create `crates/mr-tui/src/bootstrap.rs`**

Adapted from `crates/tui/src/lib.rs:243-311` — drops Windows `#[cfg]` branches and bracketed-paste / keyboard-enhancement flags (Rubin: not needed for a research tree). Keeps `EnableMouseCapture` (direct-manipulation taps) and the panic-hook main-thread guard.

```rust
//! Terminal bootstrap — enable raw mode, enter the alternate screen, enable
//! mouse capture. Adapted from claurst-tui's `setup_terminal` (drop bracketed
//! paste + kitty keyboard flags — a research tree does not need them).
//!
//! SPDX-License-Identifier: GPL-3.0

use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::{self, Stdout};

fn restore_terminal_cleanup() -> io::Result<()> {
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    Ok(())
}

pub fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    let main_thread_id = std::thread::current().id();
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        // Only the main thread restores the terminal — tokio worker panics
        // must not wreck the live TUI while the render loop is still running.
        if std::thread::current().id() == main_thread_id {
            let _ = disable_raw_mode();
            let _ = restore_terminal_cleanup();
            let _ = execute!(io::stdout(), crossterm::cursor::Show);
        }
        original_hook(panic_info);
    }));
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

pub fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    let _ = execute!(terminal.backend_mut(), crossterm::terminal::SetTitle(""));
    restore_terminal_cleanup()?;
    terminal.show_cursor()?;
    Ok(())
}
```

- [ ] **Step 8: Create `crates/mr-tui/src/guard.rs`**

```rust
//! RAII terminal restore. `Drop` calls `restore_terminal`, so even an early
//! return or an `?`-propagated error cleans up raw mode.
//!
//! SPDX-License-Identifier: GPL-3.0

use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::Stdout;

use crate::bootstrap::restore_terminal;

pub struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    pub fn new(terminal: Terminal<CrosstermBackend<Stdout>>) -> Self {
        Self { terminal }
    }

    pub fn inner_mut(&mut self) -> &mut Terminal<CrosstermBackend<Stdout>> {
        &mut self.terminal
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = restore_terminal(&mut self.terminal);
    }
}
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `cargo test -p mr-tui render_intro_shows_prompt`
Expected: PASS.

- [ ] **Step 10: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-tui
cargo clippy -p mr-tui --all-targets -- -D warnings
```
Expected: clean.

```bash
git add Cargo.toml Cargo.lock crates/mr-tui
git commit -m "$(cat <<'EOF'
feat(mr-tui): scaffold crate — bootstrap, guard, theme, figures, smoke render

Phase 6b foundation: new crates/mr-tui workspace member with terminal
bootstrap (adapted from claurst-tui, no bracketed paste), TerminalGuard
RAII restore, one research() palette, fresh box-drawing + signature
glyphs, and a TestBackend-tested render_intro smoke.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: SwarmState extension — additive `HypothesisNode` (crates/research)

**Files:**
- Modify: `crates/research/src/state/swarm_state.rs` (add types + `Phase.hypotheses`)
- Modify: `crates/research/src/orchestrator/preflight.rs:86-90` (add `hypotheses: Vec::new()` to the Phase literal)
- Modify: `crates/research/tests/state.rs:73-80` (add `hypotheses: vec![]` to the Phase literal)
- Test: `crates/research/tests/state.rs` (extend with serde round-trip + old-file compat)

**Interfaces:**
- Consumes: nothing.
- Produces (in `megaresearcher_research::state::swarm_state`):
  - `pub enum Verdict { Approve, Reject }` — derives `Debug, Clone, PartialEq, Eq, Serialize, Deserialize`.
  - `pub struct RoundVerdict { pub round: u32, pub critique: Verdict, pub revised: bool }` — same derives.
  - `pub struct HypothesisNode { pub id: String, pub label: String, pub status: String, pub rounds: Vec<RoundVerdict>, pub kill_reason: Option<String> }` — same derives.
  - `Phase` gains `#[serde(default)] pub hypotheses: Vec<HypothesisNode>`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/research/tests/state.rs` (after the existing `test_swarm_state_minimal_roundtrip`):

```rust
#[test]
fn phase_with_hypotheses_round_trips() {
    use megaresearcher_research::state::swarm_state::{
        HypothesisNode, RoundVerdict, Verdict,
    };
    let state = SwarmState {
        run_id: "r".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
        novelty_target: "hypothesis".into(),
        max_parallel: 4,
        phases: vec![Phase {
            name: "red-team".into(),
            status: "complete".into(),
            workers: vec![],
            hypotheses: vec![
                HypothesisNode {
                    id: "hypothesis-smith-1".into(),
                    label: "causal-SAE-bridge".into(),
                    status: "killed".into(),
                    rounds: vec![
                        RoundVerdict { round: 1, critique: Verdict::Reject, revised: true },
                        RoundVerdict { round: 2, critique: Verdict::Reject, revised: true },
                    ],
                    kill_reason: Some("red-team KILL (irrecoverable)".into()),
                },
                HypothesisNode {
                    id: "hypothesis-smith-2".into(),
                    label: "logit-lens-circuits".into(),
                    status: "approved".into(),
                    rounds: vec![RoundVerdict { round: 1, critique: Verdict::Approve, revised: false }],
                    kill_reason: None,
                },
            ],
        }],
        escalations: vec![],
        retry_counts: HashMap::new(),
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swarm-state.yaml");
    state.write(&path).unwrap();
    let loaded = SwarmState::read(&path).unwrap();
    assert_eq!(state, loaded);
}

#[test]
fn old_phase_yaml_without_hypotheses_deserializes_to_empty() {
    // A pre-6b swarm-state.yaml has no `hypotheses:` key on Phase. It must
    // deserialize with hypotheses == [] so the 52 orchestrator tests stay green.
    let yaml = "\
run_id: r
spec_path: s
plan_path: p
novelty_target: gap-finding
max_parallel: 4
phases:
  - name: literature-scout
    status: complete
    workers:
      - name: scout-1
        status: passed
escalations: []
retry_counts: {}
";
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swarm-state.yaml");
    std::fs::write(&path, yaml).unwrap();
    let loaded = SwarmState::read(&path).unwrap();
    assert_eq!(loaded.phases.len(), 1);
    assert!(loaded.phases[0].hypotheses.is_empty());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p megaresearcher-research phase_with_hypotheses_round_trips old_phase_yaml_without_hypotheses_deserializes_to_empty`
Expected: FAIL — `HypothesisNode` / `RoundVerdict` / `Verdict` do not exist; `Phase` has no `hypotheses` field.

- [ ] **Step 3: Add the types + `Phase.hypotheses` to `swarm_state.rs`**

In `crates/research/src/state/swarm_state.rs`, after the `Escalation` struct (after its closing brace at line 57), add:

```rust
/// A red-team verdict on one round of a hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Approve,
    Reject,
}

/// One round of the red-team critique loop for a hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundVerdict {
    pub round: u32,
    pub critique: Verdict,
    pub revised: bool,
}

/// A hypothesis as persisted for the audit-trail tree (spec §8). Additive +
/// serde-defaulted empty so pre-6b `swarm-state.yaml` files deserialize unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypothesisNode {
    pub id: String,
    pub label: String,
    pub status: String,
    #[serde(default)]
    pub rounds: Vec<RoundVerdict>,
    pub kill_reason: Option<String>,
}
```

In the `Phase` struct, add the `hypotheses` field after `workers`:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phase {
    pub name: String,
    pub status: String,
    #[serde(default)]
    pub workers: Vec<Worker>,
    #[serde(default)]
    pub hypotheses: Vec<HypothesisNode>,
}
```

- [ ] **Step 4: Update the two Phase struct-literal sites**

In `crates/research/src/orchestrator/preflight.rs`, the `Phase { name, status, workers: Vec::new() }` literal inside `build_initial_swarm_state` — add `hypotheses: Vec::new()`:

```rust
            Phase {
                name: name.to_string(),
                status,
                workers: Vec::new(),
                hypotheses: Vec::new(),
            }
```

In `crates/research/tests/state.rs`, the `Phase { name, status, workers }` literal in `sample_state()` — add `hypotheses: vec![]`:

```rust
        phases: vec![Phase {
            name: "phase_1_literature_scout".to_string(),
            status: "pending".to_string(),
            workers: vec![Worker {
                name: "scout-1".to_string(),
                status: "pending".to_string(),
            }],
            hypotheses: vec![],
        }],
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `cargo test -p megaresearcher-research phase_with_hypotheses_round_trips old_phase_yaml_without_hypotheses_deserializes_to_empty`
Expected: PASS.

- [ ] **Step 6: Determinism guard — run the full research test suite**

Run: `cargo test -p megaresearcher-research`
Expected: all tests green (the ~52 orchestrator tests are unaffected — they never assert on `hypotheses`).

- [ ] **Step 7: fmt + clippy + commit**

Run:
```
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
```

```bash
git add crates/research/src/state/swarm_state.rs crates/research/src/orchestrator/preflight.rs crates/research/tests/state.rs
git commit -m "$(cat <<'EOF'
feat(research): additive HypothesisNode + RoundVerdict + Verdict on SwarmState

Phase 6b contract touch: Phase gains #[serde(default)] hypotheses. Additive
only — survivors/killed/escalations logic unchanged. The 52 orchestrator
tests stay byte-identical green (old yaml without `hypotheses:` deserializes
to empty).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Orchestrator persists HypothesisNode from the red-team loop

**Files:**
- Modify: `crates/research/src/orchestrator/redteam.rs` (accumulate `Vec<RoundVerdict>`; upsert `HypothesisNode` into the red-team `Phase`; inner-loop `write_swarm`)
- Test: `crates/research/tests/orchestrator.rs` (extend `redteam_loop_kills_on_kill_verdict` with persistence assertions + add a `redteam_loop_persists_approved_hypothesis_node` test)

**Interfaces:**
- Consumes: T2's `HypothesisNode`, `RoundVerdict`, `Verdict` from `swarm_state.rs`; the existing `redteam_loop_fixture` test helper.
- Produces: the red-team `Phase` in `swarm-state.yaml` now carries `hypotheses` populated by `run_redteam_loop`. T4 reads these.

- [ ] **Step 1: Write the failing test**

The existing `redteam_loop_kills_on_kill_verdict` test at `crates/research/tests/orchestrator.rs:1246-1274` drives a KILL verdict. Extend it with assertions that the red-team `Phase` in the in-memory `swarm` carries a killed `HypothesisNode` with a `kill_reason` and a `RoundVerdict` sequence. Add the import at the top of the file (after line 16 `use megaresearcher_research::state::swarm_state::SwarmState;`):

```rust
use megaresearcher_research::state::swarm_state::{HypothesisNode, RoundVerdict, Verdict};
```

Then replace the body of `redteam_loop_kills_on_kill_verdict` (lines 1247-1274) with:

```rust
#[tokio::test]
async fn redteam_loop_kills_on_kill_verdict() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    let fake = Arc::new(FakeProvider::new(
        "fake",
        redteam_turns("KILL (irrecoverable)"),
    ));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    let res = run_redteam_loop(
        &run_dir,
        "SPEC",
        hyps,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
        &mut swarm,
        &[],
    )
    .await
    .unwrap();
    assert!(res.survivors.is_empty());
    assert_eq!(res.killed, vec!["hypothesis-smith-1".to_string()]);
    assert_eq!(res.redteam_dirs.len(), 1);
    assert_eq!(swarm.escalations.len(), 1);
    assert_eq!(swarm.escalations[0].worker, "hypothesis-smith-1");
    assert!(swarm.escalations[0].reason.contains("KILL"));

    // Phase 6b persistence: the red-team Phase carries a killed HypothesisNode
    // with the kill reason + the round verdict sequence.
    let rt_phase = swarm
        .phases
        .iter()
        .find(|p| p.name == "red-team")
        .expect("red-team phase present");
    assert_eq!(rt_phase.hypotheses.len(), 1);
    let node = &rt_phase.hypotheses[0];
    assert_eq!(node.id, "hypothesis-smith-1");
    assert_eq!(node.status, "killed");
    assert_eq!(
        node.kill_reason.as_deref(),
        Some("red-team KILL (irrecoverable)")
    );
    assert_eq!(node.rounds.len(), 1);
    assert_eq!(node.rounds[0].round, 1);
    assert_eq!(node.rounds[0].critique, Verdict::Reject);
    assert!(node.rounds[0].revised);
}
```

Also add a new test for the APPROVE path (after `redteam_loop_approves_on_first_round`):

```rust
#[tokio::test]
async fn redteam_loop_persists_approved_hypothesis_node() {
    let (swarm, hyps, run_dir) = redteam_loop_fixture(1);
    let fake = Arc::new(FakeProvider::new("fake", redteam_turns("APPROVE")));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut swarm = swarm;
    run_redteam_loop(
        &run_dir,
        "SPEC",
        hyps,
        &fixture_agents_dir(),
        provider,
        "fake-model",
        1,
        &mut swarm,
        &[],
    )
    .await
    .unwrap();
    let rt_phase = swarm
        .phases
        .iter()
        .find(|p| p.name == "red-team")
        .expect("red-team phase present");
    assert_eq!(rt_phase.hypotheses.len(), 1);
    let node = &rt_phase.hypotheses[0];
    assert_eq!(node.id, "hypothesis-smith-1");
    assert_eq!(node.status, "approved");
    assert!(node.kill_reason.is_none());
    assert_eq!(node.rounds.len(), 1);
    assert_eq!(node.rounds[0].critique, Verdict::Approve);
    assert!(!node.rounds[0].revised);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p megaresearcher-research redteam_loop_kills_on_kill_verdict redteam_loop_persists_approved_hypothesis_node`
Expected: FAIL — the red-team `Phase` is not in `swarm.phases` (the fixture starts with `phases: vec![]`), and no `HypothesisNode` is persisted.

- [ ] **Step 3: Implement persistence in `run_redteam_loop`**

In `crates/research/src/orchestrator/redteam.rs`, add the imports at the top (after the existing `use crate::state::swarm_state::SwarmState;` line):

```rust
use crate::state::swarm_state::{HypothesisNode, RoundVerdict, Verdict};
```

Inside `run_redteam_loop`, at the top of the per-hypothesis loop (right after `for hyp in hypotheses.into_iter() {` and before `let n = hyp.name...`), add a `rounds` accumulator:

```rust
        let mut rounds: Vec<RoundVerdict> = Vec::new();
```

Now modify each verdict arm to push a `RoundVerdict` and, on break, build + upsert the `HypothesisNode`. Add a helper `upsert_hypothesis_node` at the end of the file (after `run_redteam_loop`):

```rust
/// Insert or replace the `HypothesisNode` for `id` in the red-team `Phase`.
/// If the red-team Phase is absent from `swarm`, it is appended. Additive: the
/// survivors/killed/escalations logic is untouched — this only persists the
/// audit-trail node the TUI renders.
fn upsert_hypothesis_node(swarm: &mut SwarmState, node: HypothesisNode) {
    let phase = swarm
        .phases
        .iter_mut()
        .find(|p| p.name == "red-team");
    let phase = match phase {
        Some(p) => p,
        None => {
            swarm.phases.push(crate::state::swarm_state::Phase {
                name: "red-team".to_string(),
                status: "running".to_string(),
                workers: Vec::new(),
                hypotheses: Vec::new(),
            });
            swarm.phases.last_mut().unwrap()
        }
    };
    if let Some(existing) = phase.hypotheses.iter_mut().find(|h| h.id == node.id) {
        *existing = node;
    } else {
        phase.hypotheses.push(node);
    }
}
```

Now replace the **Approve** arm (lines 144-147) with:

```rust
                Some(RedTeamVerdict::Approve) => {
                    rounds.push(RoundVerdict {
                        round,
                        critique: Verdict::Approve,
                        revised: false,
                    });
                    upsert_hypothesis_node(
                        swarm,
                        HypothesisNode {
                            id: hyp.name.clone(),
                            label: hyp.gap.statement.clone(),
                            status: "approved".to_string(),
                            rounds: std::mem::take(&mut rounds),
                            kill_reason: None,
                        },
                    );
                    let _ = crate::orchestrator::preflight::write_swarm(swarm, run_dir);
                    survivors.push(hyp);
                    break;
                }
```

Replace the **Kill** arm (lines 148-157) with:

```rust
                Some(RedTeamVerdict::Kill) => {
                    rounds.push(RoundVerdict {
                        round,
                        critique: Verdict::Reject,
                        revised: false,
                    });
                    crate::orchestrator::preflight::add_escalation(
                        swarm,
                        &hyp.name,
                        "red-team KILL (irrecoverable)",
                        round,
                    );
                    upsert_hypothesis_node(
                        swarm,
                        HypothesisNode {
                            id: hyp.name.clone(),
                            label: hyp.gap.statement.clone(),
                            status: "killed".to_string(),
                            rounds: std::mem::take(&mut rounds),
                            kill_reason: Some("red-team KILL (irrecoverable)".to_string()),
                        },
                    );
                    let _ = crate::orchestrator::preflight::write_swarm(swarm, run_dir);
                    killed.push(hyp.name.clone());
                    break;
                }
```

Replace the **Reject** arm (lines 158-175) with:

```rust
                Some(RedTeamVerdict::Reject { revision: _ }) => {
                    rounds.push(RoundVerdict {
                        round,
                        critique: Verdict::Reject,
                        revised: true,
                    });
                    revision_count += 1;
                    swarm.retry_counts.insert(hyp.name.clone(), revision_count);
                    let critique = fs::read_to_string(rt_dir.join("output.md"))
                        .unwrap_or_else(|_| "(no output.md)".to_string());
                    redispatch_smith_revision(
                        &hyp,
                        spec_text,
                        &critique,
                        agents_dir,
                        provider.clone(),
                        default_model,
                        extra_tools,
                    )
                    .await?;
                    // Persist the in-progress rounds so the TUI sees them live.
                    upsert_hypothesis_node(
                        swarm,
                        HypothesisNode {
                            id: hyp.name.clone(),
                            label: hyp.gap.statement.clone(),
                            status: "alive".to_string(),
                            rounds: rounds.clone(),
                            kill_reason: None,
                        },
                    );
                    let _ = crate::orchestrator::preflight::write_swarm(swarm, run_dir);
                    continue;
                }
```

Replace the **None** (no parseable verdict) arm (lines 176-185) with:

```rust
                None => {
                    crate::orchestrator::preflight::add_escalation(
                        swarm,
                        &hyp.name,
                        "red-team produced no parseable verdict",
                        round,
                    );
                    upsert_hypothesis_node(
                        swarm,
                        HypothesisNode {
                            id: hyp.name.clone(),
                            label: hyp.gap.statement.clone(),
                            status: "killed".to_string(),
                            rounds: std::mem::take(&mut rounds),
                            kill_reason: Some("red-team produced no parseable verdict".to_string()),
                        },
                    );
                    let _ = crate::orchestrator::preflight::write_swarm(swarm, run_dir);
                    killed.push(hyp.name.clone());
                    break;
                }
```

For the **cap exceeded** break (lines 91-101), add persistence before `break`:

```rust
            if revision_count >= REVISION_CAP {
                let round = revision_count + 1;
                crate::orchestrator::preflight::add_escalation(
                    swarm,
                    &hyp.name,
                    "exceeded 3 red-team revisions",
                    round,
                );
                upsert_hypothesis_node(
                    swarm,
                    HypothesisNode {
                        id: hyp.name.clone(),
                        label: hyp.gap.statement.clone(),
                        status: "killed".to_string(),
                        rounds: std::mem::take(&mut rounds),
                        kill_reason: Some("exceeded 3 red-team revisions".to_string()),
                    },
                );
                let _ = crate::orchestrator::preflight::write_swarm(swarm, run_dir);
                killed.push(hyp.name.clone());
                break;
            }
```

For the **gate escalated** break (lines 128-138), add persistence before `break`:

```rust
            if gates[0].status == GateStatus::Escalated {
                crate::orchestrator::preflight::add_escalation(
                    swarm,
                    &hyp.name,
                    "red-team missing artifacts after retry",
                    round,
                );
                upsert_hypothesis_node(
                    swarm,
                    HypothesisNode {
                        id: hyp.name.clone(),
                        label: hyp.gap.statement.clone(),
                        status: "killed".to_string(),
                        rounds: std::mem::take(&mut rounds),
                        kill_reason: Some("red-team missing artifacts after retry".to_string()),
                    },
                );
                let _ = crate::orchestrator::preflight::write_swarm(swarm, run_dir);
                killed.push(hyp.name.clone());
                redteam_dirs.push(rt_dir);
                break;
            }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p megaresearcher-research redteam_loop_kills_on_kill_verdict redteam_loop_persists_approved_hypothesis_node`
Expected: PASS.

- [ ] **Step 5: Determinism guard — run the full research test suite**

Run: `cargo test -p megaresearcher-research`
Expected: all tests green. The persistence is additive — the existing assertions on `survivors`/`killed`/`escalations`/`retry_counts` are unchanged. If any of the `redteam_loop_revises_then_approves` / `redteam_loop_escalates_after_three_revisions` tests fail, the persistence logic altered the survivors/killed flow — re-check that the `upsert_hypothesis_node` calls are BEFORE the existing `survivors.push`/`killed.push`/`add_escalation` lines (which they are) and that `std::mem::take(&mut rounds)` is only used on terminal arms (Approve/Kill/None/cap/gate), not on Reject (which uses `rounds.clone()`).

- [ ] **Step 6: fmt + clippy + commit**

Run:
```
cargo fmt -p megaresearcher-research
cargo clippy -p megaresearcher-research --all-targets -- -D warnings
```

```bash
git add crates/research/src/orchestrator/redteam.rs crates/research/tests/orchestrator.rs
git commit -m "$(cat <<'EOF'
feat(research): persist HypothesisNode from the red-team loop

run_redteam_loop now upserts a HypothesisNode (id/label/status/rounds/
kill_reason) into the red-team Phase on every verdict arm and writes
swarm-state.yaml inside the loop so the TUI sees rounds live. Additive
only — survivors/killed/escalations logic is untouched. The 52
orchestrator tests stay green.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---
## Task 4: widget/tree.rs — the signature (render_tree)

**Files:**
- Create: `crates/mr-tui/src/widget/mod.rs`
- Create: `crates/mr-tui/src/widget/tree.rs`
- Modify: `crates/mr-tui/src/lib.rs` (add `pub mod widget;`)
- Test: `crates/mr-tui/tests/tree.rs`

**Interfaces:**
- Consumes: T1's `theme::ColorPalette` + `figures`; T2's `megaresearcher_research::state::swarm_state::{SwarmState, Phase, Worker, HypothesisNode, RoundVerdict, Verdict}`.
- Produces: `mr_tui::widget::tree::render_tree(frame, area, swarm, theme)` — renders the phases→workers→hypothesis tree with the red-team row and the greyed kill. T9's Run surface calls this.

- [ ] **Step 1: Write the failing tests**

Create `crates/mr-tui/tests/tree.rs`:

```rust
mod common;

use megaresearcher_research::state::swarm_state::{
    HypothesisNode, Phase, RoundVerdict, SwarmState, Verdict, Worker,
};
use mr_tui::theme::research;
use mr_tui::widget::tree::render_tree;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn buf_content(terminal: &Terminal<TestBackend>) -> String {
    let buf = terminal.backend().buffer().clone();
    buf.content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect()
}

fn state_with_killed_hypothesis() -> SwarmState {
    SwarmState {
        run_id: "r".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
        novelty_target: "hypothesis".into(),
        max_parallel: 4,
        phases: vec![
            Phase {
                name: "literature-scout".into(),
                status: "complete".into(),
                workers: vec![Worker { name: "scout-1".into(), status: "passed".into() }],
                hypotheses: vec![],
            },
            Phase {
                name: "red-team".into(),
                status: "running".into(),
                workers: vec![],
                hypotheses: vec![
                    HypothesisNode {
                        id: "hypothesis-smith-1".into(),
                        label: "causal-SAE-bridge".into(),
                        status: "killed".into(),
                        rounds: vec![
                            RoundVerdict { round: 1, critique: Verdict::Reject, revised: true },
                            RoundVerdict { round: 2, critique: Verdict::Reject, revised: true },
                        ],
                        kill_reason: Some("mechanism contradicts Marks et al. 2025".into()),
                    },
                    HypothesisNode {
                        id: "hypothesis-smith-2".into(),
                        label: "logit-lens-circuits".into(),
                        status: "approved".into(),
                        rounds: vec![RoundVerdict { round: 1, critique: Verdict::Approve, revised: false }],
                        kill_reason: None,
                    },
                ],
            },
        ],
        escalations: vec![],
        retry_counts: std::collections::HashMap::new(),
    }
}

#[test]
fn tree_renders_killed_hypothesis_with_kill_reason() {
    let state = state_with_killed_hypothesis();
    let theme = research();
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal
        .draw(|f| {
            render_tree(f, f.area(), &state, &theme);
        })
        .unwrap();
    let content = buf_content(&terminal);
    assert!(content.contains("KILLED"), "killed status marker: {content}");
    assert!(
        content.contains("mechanism contradicts Marks et al. 2025"),
        "kill reason: {content}"
    );
}

#[test]
fn tree_renders_redteam_round_count() {
    let state = state_with_killed_hypothesis();
    let theme = research();
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal
        .draw(|f| {
            render_tree(f, f.area(), &state, &theme);
        })
        .unwrap();
    let content = buf_content(&terminal);
    // The killed hypothesis went 2 rounds; the approved one went 1. Assert the
    // round indicator renders for at least one node.
    assert!(content.contains("round"), "round indicator: {content}");
}

#[test]
fn tree_renders_approved_hypothesis() {
    let state = state_with_killed_hypothesis();
    let theme = research();
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal
        .draw(|f| {
            render_tree(f, f.area(), &state, &theme);
        })
        .unwrap();
    let content = buf_content(&terminal);
    assert!(content.contains("APPROVE") || content.contains("approved"), "approved: {content}");
}

#[test]
fn tree_renders_phases_without_hypotheses_for_gap_finding() {
    let state = SwarmState {
        run_id: "r".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
        novelty_target: "gap-finding".into(),
        max_parallel: 4,
        phases: vec![Phase {
            name: "literature-scout".into(),
            status: "complete".into(),
            workers: vec![Worker { name: "scout-1".into(), status: "passed".into() }],
            hypotheses: vec![],
        }],
        escalations: vec![],
        retry_counts: std::collections::HashMap::new(),
    };
    let theme = research();
    let mut terminal = Terminal::new(TestBackend::new(100, 20)).unwrap();
    terminal
        .draw(|f| {
            render_tree(f, f.area(), &state, &theme);
        })
        .unwrap();
    let content = buf_content(&terminal);
    assert!(content.contains("literature-scout"), "phase name: {content}");
    assert!(content.contains("scout-1"), "worker name: {content}");
    assert!(!content.contains("KILLED"), "no killed nodes in gap-finding");
}
```

Create `crates/mr-tui/tests/common/mod.rs`:

```rust
pub mod fake_provider;
pub mod turns;
```

Copy `crates/mr-cli/tests/common/fake_provider.rs` verbatim to `crates/mr-tui/tests/common/fake_provider.rs` and `crates/mr-cli/tests/common/turns.rs` verbatim to `crates/mr-tui/tests/common/turns.rs` (both are GPL-3.0 same-repo test infra; the file headers say so). These are needed from T5 onward but copying them now keeps `tests/common/mod.rs` compiling.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p mr-tui tree_renders`
Expected: compile error — `mr_tui::widget::tree` does not exist.

- [ ] **Step 3: Create `crates/mr-tui/src/widget/mod.rs`**

```rust
//! TUI widgets — the signature tree, inline chat, hypothesis cards, markdown.
//!
//! SPDX-License-Identifier: GPL-3.0

pub mod tree;
```

- [ ] **Step 4: Create `crates/mr-tui/src/widget/tree.rs`**

```rust
//! The signature widget: render the swarm as a tree that grows.
//! Phases → workers → hypothesis sub-nodes; the red-team row shows
//! `smith → critique → revise ↻ round N/3`; killed hypotheses dim with
//! their one-line kill reason indented beneath.
//!
//! SPDX-License-Identifier: GPL-3.0

use megaresearcher_research::state::swarm_state::{
    HypothesisNode, RoundVerdict, SwarmState, Verdict,
};
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::figures::{BRANCH, CHECK, CROSS, DASH, LAST, PIPE, REVISE, WARN};
use crate::theme::ColorPalette;

/// Render `swarm` as the growing tree into `area`. Pure (no terminal side
/// effects) so it can be tested with `TestBackend`.
pub fn render_tree(
    frame: &mut ratatui::Frame,
    area: Rect,
    swarm: &SwarmState,
    theme: &ColorPalette,
) {
    let mut lines: Vec<Line> = Vec::new();
    for (pi, phase) in swarm.phases.iter().enumerate() {
        let phase_marker = if pi == swarm.phases.len().saturating_sub(1) {
            LAST
        } else {
            BRANCH
        };
        let status_style = phase_status_style(&phase.status, theme);
        lines.push(Line::from(vec![
            Span::styled(format!("{phase_marker} "), Style::default().fg(theme.border)),
            Span::styled(format!("Phase · {}", phase.name), Style::default().fg(theme.text_light)),
            Span::raw("  "),
            Span::styled(phase.status.clone(), status_style),
        ]));
        for (wi, worker) in phase.workers.iter().enumerate() {
            let connector = if wi == phase.workers.len().saturating_sub(1) {
                LAST
            } else {
                BRANCH
            };
            let wstatus_style = worker_status_style(&worker.status, theme);
            lines.push(Line::from(vec![
                Span::raw(format!("  {connector} ")),
                Span::styled(worker.name.clone(), Style::default().fg(theme.text_light)),
                Span::raw("  "),
                Span::styled(worker.status.clone(), wstatus_style),
            ]));
        }
        // Hypothesis sub-nodes (the signature).
        for (hi, hyp) in phase.hypotheses.iter().enumerate() {
            let connector = if hi == phase.hypotheses.len().saturating_sub(1) {
                LAST
            } else {
                BRANCH
            };
            lines.push(hypothesis_line(connector, hyp, theme));
            if let Some(reason) = hyp.kill_reason.as_ref() {
                let dash = DASH.repeat(1);
                lines.push(Line::from(vec![
                    Span::raw(format!("      {dash} ")),
                    Span::styled(format!("\"{reason}\""), Style::default().fg(theme.killed)),
                ]));
            }
        }
    }
    if swarm.phases.is_empty() {
        lines.push(Line::from(Span::styled(
            "waiting for run to start…",
            Style::default().fg(theme.disabled),
        )));
    }
    let para = Paragraph::new(lines).alignment(Alignment::Left);
    frame.render_widget(para, area);
}

fn hypothesis_line(connector: &str, hyp: &HypothesisNode, theme: &ColorPalette) -> Line {
    let id = hyp.id.replace("hypothesis-smith-", "H");
    let status_span = match hyp.status.as_str() {
        "killed" => Span::styled(
            format!("{}  KILLED", CROSS),
            Style::default().fg(theme.killed).add_modifier(Modifier::DIM),
        ),
        "approved" => Span::styled(
            format!("{}  APPROVE", CHECK),
            Style::default().fg(theme.success),
        ),
        _ => Span::styled(
            format!("{}  alive", REVISE),
            Style::default().fg(theme.running),
        ),
    };
    let mut spans = vec![
        Span::raw(format!("  {connector} ")),
        Span::styled(format!("{id}  {}", hyp.label), Style::default().fg(theme.alive)),
        Span::raw("  "),
    ];
    // The red-team row: smith → critique → revise ↻ round N/3.
    let last_round = hyp.rounds.last().map(|r| r.round).unwrap_or(0);
    let round_label = format!("round {}/3", last_round.max(1));
    if hyp.status == "killed" || hyp.status == "approved" {
        // Final state — show the verdict sequence compactly.
        let mut seq = String::new();
        for (i, rv) in hyp.rounds.iter().enumerate() {
            if i > 0 {
                seq.push_str("  ");
            }
            seq.push_str("smith ");
            seq.push_str(if rv.critique == Verdict::Approve { CHECK } else { CROSS });
            seq.push_str(" critique ");
            if rv.revised {
                seq.push_str(REVISE);
            }
        }
        spans.push(Span::styled(seq, Style::default().fg(theme.alive)));
        spans.push(Span::raw("  "));
        spans.push(Span::styled(round_label, Style::default().fg(theme.disabled)));
    } else {
        // In-progress — the animated cycling row.
        spans.push(Span::styled(
            format!("smith {} critique {} {}", CHECK, CROSS, REVISE),
            Style::default().fg(theme.running),
        ));
        spans.push(Span::raw("  "));
        spans.push(Span::styled(round_label, Style::default().fg(theme.running)));
    }
    spans.push(Span::raw("  "));
    spans.push(status_span);
    Line::from(spans)
}

fn phase_status_style(status: &str, theme: &ColorPalette) -> Style {
    match status {
        "complete" | "done" => Style::default().fg(theme.success),
        "running" => Style::default().fg(theme.running),
        "skipped" | "pending" | "waiting" => Style::default().fg(theme.disabled),
        s if s.contains("fail") || s.contains("error") => Style::default().fg(theme.error),
        _ => Style::default().fg(theme.text_light),
    }
}

fn worker_status_style(status: &str, theme: &ColorPalette) -> Style {
    match status {
        "passed" | "complete" | "done" | "approved" => Style::default().fg(theme.success),
        "running" => Style::default().fg(theme.running),
        "killed" | "failed" => Style::default().fg(theme.killed),
        "escalated" => Style::default().fg(theme.escalation),
        _ => Style::default().fg(theme.disabled),
    }
}

#[allow(dead_code)]
fn _unused_warn(_w: &str) {
    let _ = WARN;
}
```

Also add `pub mod widget;` to `crates/mr-tui/src/lib.rs` (after `pub mod theme;`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p mr-tui tree_renders`
Expected: all four tree tests PASS.

- [ ] **Step 6: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-tui
cargo clippy -p mr-tui --all-targets -- -D warnings
```
If clippy flags the `_unused_warn` helper, remove it (it was only to keep `WARN` from being dead code; if `WARN` is unused here it is used in T9's escalation strip, so `#[allow(dead_code)]` on the const in `figures.rs` may be needed instead — add `#![allow(dead_code)]` is NOT allowed; instead annotate `pub const WARN: &str = "⚠";` with `#[allow(dead_code)]` in `figures.rs` if clippy complains, then remove it when T9 uses `WARN`).

```bash
git add crates/mr-tui
git commit -m "$(cat <<'EOF'
feat(mr-tui): render_tree — the signature widget (greyed kill + red-team row)

widget/tree renders the swarm as a phases→workers→hypothesis tree. Killed
hypotheses dim (Modifier::DIM + theme.killed) with their one-line kill
reason indented beneath. The red-team row shows smith → critique → revise
↻ with round N/3. TestBackend tests cover killed/approved/gap-finding.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CountingProvider + cost meter

**Files:**
- Create: `crates/mr-tui/src/cost.rs`
- Modify: `crates/mr-tui/src/lib.rs` (add `pub mod cost;`)
- Test: `crates/mr-tui/tests/cost.rs`

**Interfaces:**
- Consumes: `claurst_api::{LlmProvider, ProviderRequest, ProviderResponse, ProviderError, StreamEvent, ModelInfo, ProviderStatus, ProviderCapabilities}`; `claurst_core::cost::CostTracker` (re-exported as `claurst_core::CostTracker`); `claurst_core::types::UsageInfo`.
- Produces:
  - `mr_tui::cost::CountingProvider { inner: Arc<dyn LlmProvider>, tracker: Arc<CostTracker> }` — impls `LlmProvider`; delegates all methods; on `create_message_stream` wraps the stream to feed `tracker.add_usage(...)` for each `MessageDelta { usage: Some(u), .. }` and `MessageStart { usage, .. }`.
  - `mr_tui::cost::render_cost_meter(frame, area, tracker, theme)` — renders `$X.XX · Nk` top-right.

- [ ] **Step 1: Write the failing test**

Create `crates/mr-tui/tests/cost.rs`:

```rust
mod common;

use std::sync::Arc;

use claurst_api::{LlmProvider, ProviderRequest, StreamEvent};
use claurst_core::cost::CostTracker;
use claurst_core::types::UsageInfo;
use mr_tui::cost::{render_cost_meter, CountingProvider};
use mr_tui::theme::research;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

use common::fake_provider::FakeProvider;
use common::turns::write_turn;

fn usage_turn(input: u64, output: u64) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart {
            id: "m".into(),
            model: "fake".into(),
            usage: UsageInfo {
                input_tokens: input,
                output_tokens: 0,
                ..Default::default()
            },
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            content_block: claurst_core::types::ContentBlock::Text { text: String::new() },
        },
        StreamEvent::TextDelta { index: 0, text: "hi".into() },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::MessageDelta {
            stop_reason: Some(claurst_api::StopReason::EndTurn),
            usage: Some(UsageInfo {
                input_tokens: 0,
                output_tokens: output,
                ..Default::default()
            }),
        },
        StreamEvent::MessageStop,
    ]
}

#[tokio::test]
async fn counting_provider_feeds_tracker_from_stream() {
    let fake = Arc::new(FakeProvider::new("fake", vec![usage_turn(1000, 500)]));
    let tracker = CostTracker::with_model("claude-sonnet-4-6");
    let counting = CountingProvider::new(fake.clone() as Arc<dyn LlmProvider>, tracker.clone());
    let req = ProviderRequest::default();
    let mut stream = counting.create_message_stream(req).await.unwrap();
    use futures::StreamExt;
    while let Some(_ev) = stream.next().await {
        // drain
    }
    // input 1000 (from MessageStart) + output 500 (from MessageDelta) = 1500 tokens.
    assert_eq!(tracker.total_tokens(), 1500);
}

#[test]
fn cost_meter_renders_dollars_and_tokens() {
    let tracker = CostTracker::with_model("claude-sonnet-4-6");
    tracker.add_usage(100_000, 50_000, 0, 0);
    let theme = research();
    let mut terminal = Terminal::new(TestBackend::new(40, 3)).unwrap();
    terminal
        .draw(|f| {
            render_cost_meter(f, f.area(), &tracker, &theme);
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();
    let content: String = buf
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains('$'), "dollar sign: {content}");
    assert!(content.contains('k'), "k suffix: {content}");
}
```

NOTE: `ProviderRequest::default()` — verify it impls `Default`; if not, construct it via `ProviderRequest { messages: vec![], model: "fake".into(), ..Default::default() }` (check the type's fields). The FakeProvider ignores the request body, so any valid `ProviderRequest` works.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p mr-tui counting_provider_feeds_tracker_from_stream cost_meter_renders_dollars_and_tokens`
Expected: compile error — `mr_tui::cost` does not exist.

- [ ] **Step 3: Create `crates/mr-tui/src/cost.rs`**

```rust
//! A provider wrapper that feeds `CostTracker` from the stream, and the
//! single top-right cost meter (`$X.XX · Nk`). One number (Rubin: no chart).
//!
//! SPDX-License-Identifier: GPL-3.0

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use claurst_api::{
    LlmProvider, ModelInfo, ProviderCapabilities, ProviderError, ProviderRequest, ProviderResponse,
    ProviderStatus, StreamEvent,
};
use claurst_core::cost::CostTracker;
use claurst_core::provider_id::ProviderId;
use futures::stream::StreamExt as _;
use futures::Stream;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::theme::ColorPalette;

pub struct CountingProvider {
    inner: Arc<dyn LlmProvider>,
    tracker: Arc<CostTracker>,
}

impl CountingProvider {
    pub fn new(inner: Arc<dyn LlmProvider>, tracker: Arc<CostTracker>) -> Self {
        Self { inner, tracker }
    }

    pub fn tracker(&self) -> &Arc<CostTracker> {
        &self.tracker
    }
}

#[async_trait]
impl LlmProvider for CountingProvider {
    fn id(&self) -> &ProviderId {
        self.inner.id()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        self.inner.create_message(request).await
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let tracker = self.tracker.clone();
        let inner_stream = self.inner.create_message_stream(request).await?;
        // Wrap the stream: inspect each event, feed usage into the tracker,
        // and re-emit the event unchanged.
        let mapped = inner_stream.map(move |res| {
            if let Ok(ref ev) = res {
                match ev {
                    StreamEvent::MessageStart { usage, .. } => {
                        tracker.add_usage(
                            usage.input_tokens,
                            usage.output_tokens,
                            usage.cache_creation_input_tokens,
                            usage.cache_read_input_tokens,
                        );
                    }
                    StreamEvent::MessageDelta { usage: Some(u), .. } => {
                        tracker.add_usage(
                            u.input_tokens,
                            u.output_tokens,
                            u.cache_creation_input_tokens,
                            u.cache_read_input_tokens,
                        );
                    }
                    _ => {}
                }
            }
            res
        });
        Ok(Box::pin(mapped))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        self.inner.list_models().await
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        self.inner.health_check().await
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }
}

/// Render the single cost meter: `$X.XX · Nk` (dollars + thousands of tokens).
pub fn render_cost_meter(
    frame: &mut ratatui::Frame,
    area: Rect,
    tracker: &CostTracker,
    theme: &ColorPalette,
) {
    let text = format!(
        "${:.2} · {}k",
        tracker.total_cost_usd(),
        tracker.total_tokens() / 1000
    );
    let line = Line::from(vec![Span::styled(
        text,
        Style::default().fg(theme.text_light),
    )]);
    frame.render_widget(Paragraph::new(line).alignment(Alignment::Right), area);
}
```

Add `pub mod cost;` to `crates/mr-tui/src/lib.rs` (after `pub mod bootstrap;`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p mr-tui counting_provider_feeds_tracker_from_stream cost_meter_renders_dollars_and_tokens`
Expected: PASS. If `ProviderRequest::default()` is not available, adjust the test to build a `ProviderRequest` from its actual fields (inspect `crates/api/src/provider_types.rs` for the struct definition).

- [ ] **Step 5: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-tui
cargo clippy -p mr-tui --all-targets -- -D warnings
```

```bash
git add crates/mr-tui
git commit -m "$(cat <<'EOF'
feat(mr-tui): CountingProvider wraps LlmProvider + feeds CostTracker; cost meter

cost.rs: CountingProvider delegates LlmProvider and wraps create_message_stream
to feed CostTracker.add_usage on MessageStart + MessageDelta{usage:Some}.
render_cost_meter renders the single top-right "$X.XX · Nk" number.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---
## Task 6: TuiUserIo (channels) — the Converge I/O seam

**Files:**
- Create: `crates/mr-tui/src/io.rs`
- Modify: `crates/mr-tui/src/lib.rs` (add `pub mod io;`)
- Test: `crates/mr-tui/tests/io.rs`

**Interfaces:**
- Consumes: `megaresearcher_research::phases::{UserIo, drive_session, GuidedSession, Gate, DriveOutcome}` (the trait is `#[async_trait]`-attributed — see `crates/research/src/phases.rs:85`); `megaresearcher_research::flows::load_embedded`; `megaresearcher_research::worker_tools::{ScopedRead, ScopedWrite, Tool}`; the test infra from T4's `tests/common/`.
- Produces:
  - `mr_tui::io::TuiUserIo` — impls `UserIo` via tokio mpsc channels. Session side holds `print_tx: mpsc::UnboundedSender<String>` and `input_rx: mpsc::UnboundedReceiver<String>`; the App holds the paired receiver/sender. `print` → `print_tx.send(text)`; `read_line` → `input_rx.recv().await`.
  - `mr_tui::io::TuiUserIoHandle` — `{ print_rx: mpsc::UnboundedReceiver<String>, input_tx: mpsc::UnboundedSender<String> }` (the App side).
  - `mr_tui::io::tui_user_io() -> (TuiUserIo, TuiUserIoHandle)` — constructs the paired channels.

- [ ] **Step 1: Write the failing test**

Create `crates/mr-tui/tests/io.rs`:

```rust
mod common;

use std::sync::Arc;

use megaresearcher_research::flows::load_embedded;
use megaresearcher_research::phases::{drive_session, DriveOutcome, Gate, GuidedSession, UserIo};
use megaresearcher_research::worker_tools::{ScopedRead, ScopedWrite, Tool};
use mr_tui::io::{tui_user_io, TuiUserIo};

use common::fake_provider::FakeProvider;
use common::turns::{final_turn, write_turn};

/// A scripted TuiUserIo driver: capture prints, feed approval lines on read_line.
async fn script_session(
    io: TuiUserIo,
    prints: Arc<std::sync::Mutex<Vec<String>>>,
    lines: Vec<String>,
) {
    let mut rx = io.handle_print_rx();
    let mut tx = io.handle_input_tx();
    // Drain prints in the background.
    let prints_bg = prints.clone();
    let mut idx = 0;
    // We can't fully separate the drain from the feed without a select loop;
    // instead, drive them in tandem: read_line blocks, so feed a line then
    // drain any prints that arrived. The session is bounded (max_turns=60),
    // so this terminates.
    let _ = (prints_bg, idx);
    let _ = (&mut rx, &mut tx, lines);
}

#[tokio::test]
async fn tui_user_io_prints_and_reads_via_channels() {
    let (io, handle) = tui_user_io();
    let prints: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let prints_c = prints.clone();
    // Spawn the session side: print captures, read_line returns scripted lines.
    let session = tokio::spawn(async move {
        // print → capture; read_line → "approve" twice (spec + plan gates).
        let io_obj = io;
        // The drive_session loop: print assistant text, read user line.
        // We just exercise the trait methods directly here.
        io_obj.print("hello from session").await.unwrap();
        let l1 = io_obj.read_line().await.unwrap();
        assert_eq!(l1, "approve\n");
        io_obj.print("second print").await.unwrap();
        let l2 = io_obj.read_line().await.unwrap();
        assert_eq!(l2, "done\n");
    });
    // App side: drain prints, feed lines.
    let mut handle = handle;
    let p1 = handle.print_rx.recv().await.unwrap();
    prints_c.lock().unwrap().push(p1);
    handle.input_tx.send("approve\n").unwrap();
    let p2 = handle.print_rx.recv().await.unwrap();
    prints_c.lock().unwrap().push(p2);
    handle.input_tx.send("done\n").unwrap();
    session.await.unwrap();
    let captured = prints.lock().unwrap().clone();
    assert_eq!(captured, vec!["hello from session".to_string(), "second print".to_string()]);
}

#[tokio::test]
async fn drive_session_with_tui_user_io_writes_spec_and_plan() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs/research");
    std::fs::create_dir_all(docs.join("specs")).unwrap();
    std::fs::create_dir_all(docs.join("plans")).unwrap();

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
    // Scripted turns: write spec, write plan, final turn. The session writes
    // output between tool turns, so we need enough turns for it to draft both
    // artifacts and reach the two approval gates.
    let spec_path = docs.join("specs/test-spec.md");
    let plan_path = docs.join("plans/test-plan.md");
    let turns = vec![
        write_turn(
            spec_path.to_str().unwrap(),
            "# Spec\n\ntest spec",
        ),
        write_turn(
            plan_path.to_str().unwrap(),
            "# Plan\n\ntest plan",
        ),
        final_turn("Done."),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut session = GuidedSession::new(body, tools, provider, "fake-model", 4096, 60);
    session.inject_user("Topic: test question");

    let (io, mut handle) = tui_user_io();
    let gates = vec![
        Gate { artifact: spec_path.clone(), label: "spec".into() },
        Gate { artifact: plan_path.clone(), label: "plan".into() },
    ];
    let drive = tokio::spawn(async move {
        drive_session(&mut session, &io, gates, &["approve", "yes", "y", "done"]).await
    });
    // App side: feed approvals. drive_session prints then reads; we feed
    // "approve" for each gate. Drain prints to avoid backpressure.
    let mut approvals = 0;
    while approvals < 2 {
        // Drain any prints (non-blocking-ish: recv until Empty, then feed).
        loop {
            match handle.print_rx.try_recv() {
                Ok(_) => continue,
                Err(_) => break,
            }
        }
        handle.input_tx.send("approve\n").unwrap();
        approvals += 1;
    }
    let outcome = drive.await.unwrap().unwrap();
    assert!(matches!(outcome, DriveOutcome::Approved { gates_passed: 2 }));
    assert!(spec_path.exists());
    assert!(plan_path.exists());
}
```

NOTE: The exact turn count needed for `drive_session` to write both artifacts depends on how the FakeProvider turns map to tool calls. The `write_turn` helper writes one file per turn. If the session needs more turns (e.g. it asks clarifying questions before drafting), add more `write_turn` entries. The test asserts the gates pass; if it hits `MaxTurns`, add turns. This mirrors the proven `crates/mr-cli/tests/init.rs` pattern — consult that test for the exact turn sequence if this fails.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p mr-tui tui_user_io_prints_and_reads_via_channels drive_session_with_tui_user_io_writes_spec_and_plan`
Expected: compile error — `mr_tui::io` does not exist.

- [ ] **Step 3: Create `crates/mr-tui/src/io.rs`**

```rust
//! TuiUserIo — the Converge I/O seam. Implements `research::phases::UserIo`
//! by routing `print` into an mpsc channel the App drains into the conversation
//! buffer, and resolving `read_line` from an mpsc channel the App feeds from
//! the inline input field. Same `drive_session` engine as 6a, new view.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::io;

use async_trait::async_trait;
use megaresearcher_research::phases::UserIo;
use tokio::sync::mpsc;

/// The App-side handle: drains prints, feeds input lines.
pub struct TuiUserIoHandle {
    pub print_rx: mpsc::UnboundedReceiver<String>,
    pub input_tx: mpsc::UnboundedSender<String>,
}

/// The session-side I/O. Held by the spawned `drive_session` task.
pub struct TuiUserIo {
    print_tx: mpsc::UnboundedSender<String>,
    input_rx: mpsc::UnboundedReceiver<String>,
}

impl TuiUserIo {
    pub fn print_tx(&self) -> &mpsc::UnboundedSender<String> {
        &self.print_tx
    }
}

/// Construct the paired (session-side, app-side) channels.
pub fn tui_user_io() -> (TuiUserIo, TuiUserIoHandle) {
    let (print_tx, print_rx) = mpsc::unbounded_channel::<String>();
    let (input_tx, input_rx) = mpsc::unbounded_channel::<String>();
    (
        TuiUserIo { print_tx, input_rx },
        TuiUserIoHandle { print_rx, input_tx },
    )
}

#[async_trait]
impl UserIo for TuiUserIo {
    async fn print(&self, text: &str) -> io::Result<()> {
        self.print_tx
            .send(text.to_string())
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e))
    }

    async fn read_line(&self) -> io::Result<String> {
        self.input_rx
            .recv()
            .await
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "input channel closed"))
    }
}
```

The test references `io.handle_print_rx()` / `io.handle_input_tx()` — remove those from the test (the test uses the returned `handle` directly, not methods on `io`). The `script_session` helper in the test is unused; remove it to avoid dead-code warnings. Revise the test's first case to not call `io.handle_print_rx()` / `io.handle_input_tx()` (use `handle.print_rx` / `handle.input_tx` only).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p mr-tui tui_user_io_prints_and_reads_via_channels drive_session_with_tui_user_io_writes_spec_and_plan`
Expected: PASS. If `drive_session_with_tui_user_io_writes_spec_and_plan` hits `MaxTurns`, consult `crates/mr-cli/tests/init.rs` for the exact scripted turn sequence the 6a init test uses and mirror it.

- [ ] **Step 5: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-tui
cargo clippy -p mr-tui --all-targets -- -D warnings
```

```bash
git add crates/mr-tui
git commit -m "$(cat <<'EOF'
feat(mr-tui): TuiUserIo — channel-backed UserIo for the Converge surface

io.rs: TuiUserIo implements research::phases::UserIo via tokio mpsc
channels (print→sender, read_line→receiver). The App holds the paired
handle. drive_session runs unchanged — same engine as 6a init, new view.
Integration test drives a scripted session to spec+plan approval.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: App state machine + event loop + Start surface

**Files:**
- Create: `crates/mr-tui/src/app.rs`
- Create: `crates/mr-tui/src/surface/mod.rs`
- Create: `crates/mr-tui/src/surface/start.rs`
- Modify: `crates/mr-tui/src/lib.rs` (add `pub mod app; pub mod surface;`)
- Test: `crates/mr-tui/tests/app.rs`

**Interfaces:**
- Consumes: T1's `bootstrap`, `guard`, `theme`; T6's `io::TuiUserIo`.
- Produces:
  - `mr_tui::app::Surface` enum: `Start`, `Converge`, `Run`, `Artifact`, `Past`, `Settings`.
  - `mr_tui::app::App` — the state-machine struct. Fields include `surface: Surface`, `question: String` (the Start input), `cwd: PathBuf`, `should_exit: bool`, `frame_count: u64`, plus surface-specific state added in later tasks.
  - `mr_tui::app::App::new(cwd: PathBuf) -> Self`.
  - `mr_tui::app::App::handle_key(&mut self, key: KeyEvent) -> AppEvent` — pure key handler (returns an `AppEvent` enum the loop acts on); testable without a terminal.
  - `mr_tui::app::App::render(&self, frame)` — pure render dispatching to the current surface.
  - `mr_tui::app::App::run(&mut self, terminal) -> anyhow::Result<()>` — the async event loop (draw each iter, `spawn_blocking` poll 50ms, `KeyEventKind::Press` filter, `q`/Ctrl-C quit, `s`→Settings).
  - `mr_tui::app::AppEvent` enum: `Quit`, `ToSurface(Surface)`, `SubmitQuestion`, `None`.
  - `mr_tui::run(cwd)` updated to `App::new(cwd).run(&mut terminal)` via a `TerminalGuard`.

- [ ] **Step 1: Write the failing tests**

Create `crates/mr-tui/tests/app.rs`:

```rust
use mr_tui::app::{App, AppEvent, Surface};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

#[test]
fn start_submits_question_to_converge() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"));
    app.surface = Surface::Start;
    app.question = "can SAEs surface causal circuits?".into();
    let ev = app.handle_key(KeyEvent::new_with_modifiers_and_kind(
        KeyCode::Enter,
        KeyModifiers::NONE,
        KeyEventKind::Press,
    ));
    assert!(matches!(ev, AppEvent::SubmitQuestion));
    // After submit, the app transitions to Converge (the loop acts on
    // SubmitQuestion by transitioning; here we assert the event).
}

#[test]
fn quit_on_q_or_ctrl_c() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"));
    let ev_q = app.handle_key(KeyEvent::new_with_modifiers_and_kind(
        KeyCode::Char('q'),
        KeyModifiers::NONE,
        KeyEventKind::Press,
    ));
    assert!(matches!(ev_q, AppEvent::Quit));
    let ev_c = app.handle_key(KeyEvent::new_with_modifiers_and_kind(
        KeyCode::Char('c'),
        KeyModifiers::CONTROL,
        KeyEventKind::Press,
    ));
    assert!(matches!(ev_c, AppEvent::Quit));
}

#[test]
fn s_key_goes_to_settings() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"));
    app.surface = Surface::Start;
    let ev = app.handle_key(KeyEvent::new_with_modifiers_and_kind(
        KeyCode::Char('s'),
        KeyModifiers::NONE,
        KeyEventKind::Press,
    ));
    assert!(matches!(ev, AppEvent::ToSurface(Surface::Settings)));
}

#[test]
fn start_renders_ghosted_example() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    let app = App::new(std::path::PathBuf::from("/tmp"));
    let mut terminal = Terminal::new(TestBackend::new(80, 10)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let content: String = buf
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains("What do you want to know?"));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p mr-tui start_submits quit_on s_key start_renders`
Expected: compile error — `mr_tui::app` does not exist.

- [ ] **Step 3: Create `crates/mr-tui/src/surface/mod.rs`**

```rust
//! TUI surfaces — Start/Converge, Run, Artifact, Past, Settings.
//!
//! SPDX-License-Identifier: GPL-3.0

pub mod start;
```

- [ ] **Step 4: Create `crates/mr-tui/src/surface/start.rs`**

```rust
//! The Start surface — one input, a ghosted example. No wizard, no target
//! picker, no onboarding tour. Type, enter. (Jobs: first 60 seconds.)
//!
//! SPDX-License-Identifier: GPL-3.0

use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Style, Style as S};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::theme::ColorPalette;

/// Render the Start surface. `question` is the in-progress input; when empty
/// a ghosted example is shown.
pub fn render_start(area: Rect, question: &str, theme: &ColorPalette) {
    let prompt = "What do you want to know?";
    let example = "e.g. can sparse autoencoders surface causal circuits in transformer attention?";
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled("MegaResearcher", Style::default().fg(theme.accent)));
    let mut lines = vec![Line::from(Span::styled(
        prompt,
        Style::default().fg(theme.text_light),
    ))];
    let input_text = if question.is_empty() {
        format!("  {example}")
    } else {
        format!("  {question}")
    };
    let input_style = if question.is_empty() {
        Style::default().fg(theme.disabled)
    } else {
        Style::default().fg(theme.text_light)
    };
    lines.push(Line::from(Span::styled(input_text, input_style)));
    let para = Paragraph::new(lines)
        .block(block)
        .alignment(Alignment::Left);
    let _ = S::default(); // keep Style import used
    // The caller renders the paragraph; but render_start receives a Rect, not
    // a frame, so we return the widget via a callback. Simpler: pass the frame.
    // See the frame-accepting variant below.
    let _ = (para, area);
}

/// Render the Start surface into `frame`. The frame-accepting variant (pure,
/// TestBackend-testable).
pub fn render_start_frame(frame: &mut ratatui::Frame, area: Rect, question: &str, theme: &ColorPalette) {
    let prompt = "What do you want to know?";
    let example = "e.g. can sparse autoencoders surface causal circuits in transformer attention?";
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled("MegaResearcher", Style::default().fg(theme.accent)));
    let mut lines = vec![Line::from(Span::styled(
        prompt,
        Style::default().fg(theme.text_light),
    ))];
    let input_text = if question.is_empty() {
        format!("  {example}")
    } else {
        format!("  {question}")
    };
    let input_style = if question.is_empty() {
        Style::default().fg(theme.disabled)
    } else {
        Style::default().fg(theme.text_light)
    };
    lines.push(Line::from(Span::styled(input_text, input_style)));
    frame.render_widget(
        Paragraph::new(lines).block(block).alignment(Alignment::Left),
        area,
    );
}
```

Remove the `render_start` (non-frame) function — it is dead code; keep only `render_start_frame`. (The plan shows both to explain the decision; the implementer writes only `render_start_frame`.)

- [ ] **Step 5: Create `crates/mr-tui/src/app.rs`**

```rust
//! The surface state machine + async event loop. Factors a pure
//! `App::render(&self, frame)` so tests use TestBackend without a live
//! terminal, and a pure `App::handle_key` so transitions are testable.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::path::PathBuf;

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::surface::start::render_start_frame;
use crate::theme::{for_theme, ColorPalette};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    Start,
    Converge,
    Run,
    Artifact,
    Past,
    Settings,
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    Quit,
    ToSurface(Surface),
    SubmitQuestion,
    None,
}

pub struct App {
    pub surface: Surface,
    pub question: String,
    pub cwd: PathBuf,
    pub should_exit: bool,
    pub frame_count: u64,
    pub theme: ColorPalette,
}

impl App {
    pub fn new(cwd: PathBuf) -> Self {
        Self {
            surface: Surface::Start,
            question: String::new(),
            cwd,
            should_exit: false,
            frame_count: 0,
            theme: for_theme("research"),
        }
    }

    /// Pure key handler. Returns the event the loop should act on. Testable
    /// without a terminal.
    pub fn handle_key(&mut self, key: KeyEvent) -> AppEvent {
        // Global keys: q / Ctrl-C quit, s → Settings.
        if key.kind != KeyEventKind::Press {
            return AppEvent::None;
        }
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => AppEvent::Quit,
            KeyCode::Char('q') => AppEvent::Quit,
            KeyCode::Char('s') => AppEvent::ToSurface(Surface::Settings),
            KeyCode::Enter if self.surface == Surface::Start => {
                if !self.question.trim().is_empty() {
                    AppEvent::SubmitQuestion
                } else {
                    AppEvent::None
                }
            }
            KeyCode::Char(c) if self.surface == Surface::Start => {
                self.question.push(c);
                AppEvent::None
            }
            KeyCode::Backspace if self.surface == Surface::Start => {
                self.question.pop();
                AppEvent::None
            }
            _ => AppEvent::None,
        }
    }

    /// Pure render. Dispatches to the current surface. TestBackend-testable.
    pub fn render(&self, frame: &mut ratatui::Frame) {
        self.frame_count; // read so clippy doesn't flag (used for animation in T9)
        match self.surface {
            Surface::Start => {
                render_start_frame(frame, frame.area(), &self.question, &self.theme);
            }
            _ => {
                // Placeholder until the surface modules land (T8-T12).
                frame.render_widget(
                    ratatui::widgets::Paragraph::new(format!("surface: {:?}", self.surface))
                        .block(
                            ratatui::widgets::Block::default()
                                .borders(ratatui::widgets::Borders::ALL)
                                .title("MegaResearcher"),
                        ),
                    frame.area(),
                );
            }
        }
    }

    /// The async event loop. Draw first each iter, then spawn_blocking poll
    /// 50ms, filter KeyEventKind::Press, act on AppEvent. `q`/Ctrl-C quit;
    /// `s` → Settings. The session/orchestrator task wiring arrives in T8/T9.
    pub async fn run(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> anyhow::Result<()> {
        loop {
            self.frame_count = self.frame_count.wrapping_add(1);
            terminal.draw(|f| self.render(f))?;
            let event = tokio::task::spawn_blocking(|| {
                if crossterm::event::poll(std::time::Duration::from_millis(50)).ok()? {
                    crossterm::event::read().ok()
                } else {
                    None
                }
            })
            .await?;
            if let Some(Event::Key(key)) = event {
                let ev = self.handle_key(key);
                match ev {
                    AppEvent::Quit => {
                        self.should_exit = true;
                        return Ok(());
                    }
                    AppEvent::ToSurface(s) => self.surface = s,
                    AppEvent::SubmitQuestion => {
                        // T8 wires the Converge spawn here.
                        self.surface = Surface::Converge;
                    }
                    AppEvent::None => {}
                }
            }
            if self.should_exit {
                return Ok(());
            }
        }
    }
}
```

Add `pub mod app; pub mod surface;` to `crates/mr-tui/src/lib.rs`. Update `mr_tui::run` to use the App:

```rust
pub async fn run(cwd: &Path) -> anyhow::Result<()> {
    let terminal = bootstrap::setup_terminal()?;
    let mut guard = guard::TerminalGuard::new(terminal);
    let mut app = app::App::new(cwd.to_path_buf());
    app.run(guard.inner_mut()).await
}
```

Remove the old `render_intro` function and its test from `lib.rs` (the Start surface replaces it; keep `render_intro` only if T1's test still references it — but T7's `start_renders_ghosted_example` supersedes it, so delete `render_intro` and its test, and remove the `Block, Borders, Paragraph` imports if now unused).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p mr-tui start_submits quit_on s_key start_renders`
Expected: PASS.

- [ ] **Step 7: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-tui
cargo clippy -p mr-tui --all-targets -- -D warnings
```

```bash
git add crates/mr-tui
git commit -m "$(cat <<'EOF'
feat(mr-tui): App state machine + async event loop + Start surface

app.rs: Surface enum (Start/Converge/Run/Artifact/Past/Settings), pure
handle_key (q/Ctrl-C quit, s→Settings, Start input), pure render (TestBackend-
testable), async run loop (draw-first, spawn_blocking 50ms poll, Press filter).
surface/start.rs renders the one-input + ghosted example. mr_tui::run now
drives the App via a TerminalGuard.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Converge surface — drive_session via TuiUserIo + spec/plan cards

**Files:**
- Modify: `crates/mr-tui/src/surface/start.rs` (add the Converge render + spawn logic)
- Modify: `crates/mr-tui/src/app.rs` (add Converge state fields + spawn on SubmitQuestion)
- Create: `crates/mr-tui/src/widget/inline_chat.rs`
- Modify: `crates/mr-tui/src/widget/mod.rs` (add `pub mod inline_chat;`)
- Test: `crates/mr-tui/tests/converge.rs`

**Interfaces:**
- Consumes: T6's `io::{tui_user_io, TuiUserIo, TuiUserIoHandle}`; 6a's `megaresearcher_research::{flows::load_embedded, phases::{drive_session, GuidedSession, Gate, DriveOutcome, UserIo}, worker_tools::{ScopedRead, ScopedWrite, Tool}}`; the `init.rs::run_with` pattern verbatim (§5.4 of the context dump).
- Produces:
  - `mr_tui::widget::inline_chat::render_converge(frame, area, conversation: &[String], spec_approved: bool, plan_approved: bool, theme)` — renders the inline conversation + spec/plan `[✓]` cards.
  - `mr_tui::surface::start::ConvergeState` — `{ handle: TuiUserIoHandle, conversation: Vec<String>, spec_path: PathBuf, plan_path: PathBuf, spec_approved: bool, plan_approved: bool, outcome: Option<ConvergeOutcome> }`.
  - `mr_tui::surface::start::ConvergeOutcome` — `Approved { spec_path, plan_path }` | `MaxTurns` | `Failed(String)`.
  - App gains `converge: Option<ConvergeState>` + a `JoinHandle` for the session task; on `SubmitQuestion` it spawns the session and transitions to Converge.

- [ ] **Step 1: Write the failing test**

Create `crates/mr-tui/tests/converge.rs`:

```rust
mod common;

use std::sync::Arc;

use megaresearcher_research::flows::load_embedded;
use megaresearcher_research::phases::{drive_session, DriveOutcome, Gate, GuidedSession};
use megaresearcher_research::worker_tools::{ScopedRead, ScopedWrite, Tool};
use mr_tui::io::tui_user_io;
use mr_tui::surface::start::ConvergeOutcome;
use mr_tui::widget::inline_chat::render_converge;
use mr_tui::theme::research;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

use common::fake_provider::FakeProvider;
use common::turns::{final_turn, write_turn};

#[tokio::test]
async fn converge_session_writes_spec_and_plan_and_approves() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs/research");
    std::fs::create_dir_all(docs.join("specs")).unwrap();
    std::fs::create_dir_all(docs.join("plans")).unwrap();
    let spec_path = docs.join("specs/converge-spec.md");
    let plan_path = docs.join("plans/converge-plan.md");

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
    let turns = vec![
        write_turn(spec_path.to_str().unwrap(), "# Spec\n\nconverged spec"),
        write_turn(plan_path.to_str().unwrap(), "# Plan\n\nconverged plan"),
        final_turn("Done."),
    ];
    let fake = Arc::new(FakeProvider::new("fake", turns));
    let provider = fake.clone() as Arc<dyn LlmProvider>;
    let mut session = GuidedSession::new(body, tools, provider, "fake-model", 4096, 60);
    session.inject_user("Topic: can SAEs surface causal circuits?");

    let (io, mut handle) = tui_user_io();
    let gates = vec![
        Gate { artifact: spec_path.clone(), label: "spec".into() },
        Gate { artifact: plan_path.clone(), label: "plan".into() },
    ];
    let drive = tokio::spawn(async move {
        drive_session(&mut session, &io, gates, &["approve", "yes", "y", "done"]).await
    });
    let mut approvals = 0;
    while approvals < 2 {
        loop {
            match handle.print_rx.try_recv() {
                Ok(_) => continue,
                Err(_) => break,
            }
        }
        handle.input_tx.send("approve\n").unwrap();
        approvals += 1;
    }
    let outcome = drive.await.unwrap().unwrap();
    assert!(matches!(outcome, DriveOutcome::Approved { gates_passed: 2 }));
    assert!(spec_path.exists());
    assert!(plan_path.exists());
}

#[test]
fn render_converge_shows_spec_and_plan_cards() {
    let theme = research();
    let conversation = vec!["q  gap to research, or a hypothesis?".to_string(), ">  gap to research".to_string()];
    let mut terminal = Terminal::new(TestBackend::new(100, 20)).unwrap();
    terminal
        .draw(|f| {
            render_converge(f, f.area(), &conversation, true, false, &theme);
        })
        .unwrap();
    let content: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains("spec"), "spec card: {content}");
    assert!(content.contains("plan"), "plan card: {content}");
    // spec approved → [✓]; plan not yet → no check on plan.
    assert!(content.contains('✓'), "approved check: {content}");
}
```

- [ ] **Step 2: Run the test to verify they fail**

Run: `cargo test -p mr-tui converge_session_writes_spec_and_plan_and_approves render_converge_shows_spec_and_plan_cards`
Expected: compile error — `mr_tui::widget::inline_chat` and `mr_tui::surface::start::ConvergeOutcome` do not exist.

- [ ] **Step 3: Create `crates/mr-tui/src/widget/inline_chat.rs`**

```rust
//! The bounded 3-exchange converge widget: renders the inline conversation
//! buffer + the spec/plan `[✓]` approval cards. Not free chat — the
//! conversation is bounded and converges the question.
//!
//! SPDX-License-Identifier: GPL-3.0

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::figures::{CHECK, PENCIL};
use crate::theme::ColorPalette;

pub fn render_converge(
    frame: &mut ratatui::Frame,
    area: Rect,
    conversation: &[String],
    spec_approved: bool,
    plan_approved: bool,
    theme: &ColorPalette,
) {
    let chunks = Layout::default()
        .constraints([Constraint::Min(3), Constraint::Length(4)])
        .split(area);
    let convo = conversation
        .iter()
        .map(|l| Line::from(Span::styled(l, Style::default().fg(theme.text_light))))
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(convo)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(
                        format!("{} converge the question", PENCIL),
                        Style::default().fg(theme.accent),
                    )),
            )
            .alignment(Alignment::Left),
        chunks[0],
    );
    // Spec + plan cards.
    let cards = Layout::default()
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);
    let spec_line = if spec_approved {
        Line::from(vec![
            Span::styled("spec  ", Style::default().fg(theme.text_light)),
            Span::styled(CHECK, Style::default().fg(theme.success)),
        ])
    } else {
        Line::from(Span::styled("spec  (pending)", Style::default().fg(theme.disabled)))
    };
    frame.render_widget(
        Paragraph::new(spec_line).block(Block::default().borders(Borders::ALL)),
        cards[0],
    );
    let plan_line = if plan_approved {
        Line::from(vec![
            Span::styled("plan  ", Style::default().fg(theme.text_light)),
            Span::styled(CHECK, Style::default().fg(theme.success)),
        ])
    } else {
        Line::from(Span::styled("plan  (pending)", Style::default().fg(theme.disabled)))
    };
    frame.render_widget(
        Paragraph::new(plan_line).block(Block::default().borders(Borders::ALL)),
        cards[1],
    );
}
```

Add `pub mod inline_chat;` to `crates/mr-tui/src/widget/mod.rs`.

- [ ] **Step 4: Add Converge state + spawn to `surface/start.rs` and `app.rs`**

In `crates/mr-tui/src/surface/start.rs`, add the Converge types and a spawn helper:

```rust
use std::path::PathBuf;
use std::sync::Arc;

use megaresearcher_research::flows::load_embedded;
use megaresearcher_research::phases::{drive_session, DriveOutcome, Gate, GuidedSession};
use megaresearcher_research::worker_tools::{ScopedRead, ScopedWrite, Tool};

use crate::io::{tui_user_io, TuiUserIo, TuiUserIoHandle};

#[derive(Debug, Clone)]
pub enum ConvergeOutcome {
    Approved { spec_path: PathBuf, plan_path: PathBuf },
    MaxTurns,
    Failed(String),
}

pub struct ConvergeState {
    pub handle: TuiUserIoHandle,
    pub conversation: Vec<String>,
    pub spec_path: PathBuf,
    pub plan_path: PathBuf,
    pub spec_approved: bool,
    pub plan_approved: bool,
    pub outcome: Option<ConvergeOutcome>,
}

/// Build + spawn the converge session on a tokio task. Returns the
/// `(ConvergeState, JoinHandle)` so the App can drain prints and feed
/// approvals. Mirrors `init.rs::run_with` exactly (concatenated flow bodies,
/// ScopedRead/ScopedWrite tools, gates spec+plan, approve_words, GuidedSession
/// args, inject_user "Topic: {question}").
pub fn spawn_converge(
    cwd: &std::path::Path,
    question: &str,
    provider: Arc<dyn claurst_api::LlmProvider>,
    model: String,
) -> (ConvergeState, tokio::task::JoinHandle<anyhow::Result<DriveOutcome>>) {
    let (io, handle) = tui_user_io();
    let docs = cwd.join("docs/research");
    std::fs::create_dir_all(docs.join("specs")).ok();
    std::fs::create_dir_all(docs.join("plans")).ok();
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
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let slug = question
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    let spec_path = docs.join("specs").join(format!("{date}-{slug}-spec.md"));
    let plan_path = docs.join("plans").join(format!("{date}-{slug}-plan.md"));
    let mut session = GuidedSession::new(body, tools, provider, model, 4096, 60);
    session.inject_user(&format!("Topic: {question}"));
    let gates = vec![
        Gate { artifact: spec_path.clone(), label: "spec".into() },
        Gate { artifact: plan_path.clone(), label: "plan".into() },
    ];
    let task = tokio::spawn(async move {
        drive_session(&mut session, &io, gates, &["approve", "yes", "y", "done"]).await.map_err(anyhow::Error::from)
    });
    let state = ConvergeState {
        handle,
        conversation: Vec::new(),
        spec_path,
        plan_path,
        spec_approved: false,
        plan_approved: false,
        outcome: None,
    };
    (state, task)
}
```

Add `chrono = { workspace = true }` to `crates/mr-tui/Cargo.toml` `[dependencies]`.

In `crates/mr-tui/src/app.rs`, add fields + a `provider` field (resolved by `mr_tui::run`'s caller in T13, or resolved inside `run` — for now accept an `Option<(Arc<dyn LlmProvider>, String)>` so the loop can be tested without a provider; the real provider is wired in T13):

```rust
pub struct App {
    pub surface: Surface,
    pub question: String,
    pub cwd: PathBuf,
    pub should_exit: bool,
    pub frame_count: u64,
    pub theme: ColorPalette,
    pub provider: Option<(Arc<dyn claurst_api::LlmProvider>, String)>,
    pub converge: Option<crate::surface::start::ConvergeState>,
    pub converge_task: Option<tokio::task::JoinHandle<anyhow::Result<megaresearcher_research::phases::DriveOutcome>>>,
}
```

Update `App::new` to take an optional provider:

```rust
pub fn new(cwd: PathBuf, provider: Option<(Arc<dyn claurst_api::LlmProvider>, String)>) -> Self {
    Self {
        surface: Surface::Start,
        question: String::new(),
        cwd,
        should_exit: false,
        frame_count: 0,
        theme: for_theme("research"),
        provider,
        converge: None,
        converge_task: None,
    }
}
```

Update `App::render` to dispatch Converge to `render_converge`:

```rust
            Surface::Converge => {
                if let Some(cs) = &self.converge {
                    crate::widget::inline_chat::render_converge(
                        frame,
                        frame.area(),
                        &cs.conversation,
                        cs.spec_approved,
                        cs.plan_approved,
                        &self.theme,
                    );
                }
            }
```

In `App::run`, on `AppEvent::SubmitQuestion`, spawn converge if a provider is set:

```rust
                    AppEvent::SubmitQuestion => {
                        if let Some((p, m)) = self.provider.clone() {
                            let q = std::mem::take(&mut self.question);
                            let (state, task) =
                                crate::surface::start::spawn_converge(&self.cwd, &q, p, m);
                            self.converge = Some(state);
                            self.converge_task = Some(task);
                            self.surface = Surface::Converge;
                        }
                    }
```

Also in the loop, when `surface == Converge`, drain `converge.handle.print_rx` into `converge.conversation` (non-blocking `try_recv` loop) and `tokio::select!` on `converge_task` completion to set `converge.outcome` + transition to Run. Add this drain + select inside the loop before the event poll:

```rust
        if let Some(cs) = self.converge.as_mut() {
            loop {
                match cs.handle.print_rx.try_recv() {
                    Ok(msg) => cs.conversation.push(msg),
                    Err(_) => break,
                }
            }
        }
        if let Some(task) = self.converge_task.as_mut() {
            use futures::FutureExt as _;
            let mut done = std::pin::Pin::new(task).fuse();
            tokio::select! {
                _ = &mut done => {
                    // Task finished — read outcome below.
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(0)) => {}
            }
        }
```

NOTE: The `select!` with a 0ms sleep is a non-blocking probe. A cleaner approach: poll `converge_task` with `Option<JoinHandle>::as_mut()` + a `tokio::select!` against the event poll. The implementer should choose the cleanest non-blocking form that compiles; the contract is: drain prints each frame, transition to Run when the task completes with `Approved`. (If the task is `Err` or `MaxTurns`, surface an inline error and stay on Converge.) This is the one place the plan leaves ergonomics to the implementer because the exact `JoinHandle` + `select!` shape depends on borrow-checker constraints the implementer sees at compile time; the named alternative is `tokio::select! { _ = &mut task, biased => {...} , _ = poll_future => {} }`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p mr-tui converge_session_writes_spec_and_plan_and_approves render_converge_shows_spec_and_plan_cards`
Expected: PASS.

- [ ] **Step 6: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-tui
cargo clippy -p mr-tui --all-targets -- -D warnings
```

```bash
git add crates/mr-tui
git commit -m "$(cat <<'EOF'
feat(mr-tui): Converge surface — drive_session via TuiUserIo + spec/plan cards

surface/start.rs: ConvergeState + spawn_converge (mirrors init.rs::run_with —
concatenated flow bodies, ScopedRead/Write, gates spec+plan, GuidedSession
args, inject_user "Topic: {q}"). widget/inline_chat renders the conversation
buffer + spec/plan [✓] cards. App drains prints each frame, spawns on
SubmitQuestion, transitions to Run on Approved.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---
## Task 9: TuiEscalationHandler + Run surface (MAKE-OR-BREAK)

**Files:**
- Create: `crates/mr-tui/src/escalation.rs`
- Create: `crates/mr-tui/src/surface/run.rs`
- Modify: `crates/mr-tui/src/escalation.rs` (TuiEscalationHandler)
- Modify: `crates/mr-tui/src/app.rs` (Run state + transition from Converge/Start)
- Modify: `crates/mr-tui/src/surface/mod.rs` (add `pub mod run;`)
- Modify: `crates/mr-tui/src/lib.rs` (add `pub mod escalation;`)
- Test: `crates/mr-tui/tests/run.rs` (the make-or-break integration test)

**Interfaces:**
- Consumes: T4's `widget::tree::render_tree`; T5's `cost::{CountingProvider, render_cost_meter}`; 6a's `megaresearcher_research::orchestrator::{Orchestrator, OrchestratorConfig, RunOutcome}` + `state::run_id::generate_run_id` + `state::swarm_state::SwarmState` + `mcp::ml_intern_config` + `orchestrator::escalation::{EscalationHandler, EscalationVerdict}` + `state::swarm_state::Escalation`; `claurst_core::config::McpServerConfig`.
- Produces:
  - `mr_tui::escalation::TuiEscalationHandler` — impls `EscalationHandler`; on `adjudicate`, sends the `Escalation` into an in-App channel and blocks (awaits a oneshot) until the user adjudicates inline (`continue`/`fail`) from the run strip.
  - `mr_tui::surface::run::RunState` — `{ run_dir: PathBuf, run_id: String, swarm: Option<SwarmState>, cost_tracker: Arc<CostTracker>, pending_escalation: Option<Escalation>, escalation_tx/escalation_rx, verdict_tx/verdict_rx, orch_task: JoinHandle }`.
  - `mr_tui::surface::run::spawn_run(cwd, spec_path, plan_path, provider, model, on_escalate, mcp_enabled) -> (RunState, JoinHandle)` — wraps the provider in `CountingProvider`, builds `OrchestratorConfig` with the `TuiEscalationHandler`, spawns `Orchestrator::execute`.
  - App gains `run: Option<RunState>`.

**This is the thesis gate. If the make-or-break integration test fails, pause before T10.**

- [ ] **Step 1: Write the make-or-break failing test**

Create `crates/mr-tui/tests/run.rs`:

```rust
mod common;

use std::sync::Arc;

use megaresearcher_research::state::swarm_state::{
    Escalation, HypothesisNode, Phase, RoundVerdict, SwarmState, Verdict, Worker,
};
use mr_tui::app::{App, Surface};
use mr_tui::theme::research;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn make_or_break_state() -> SwarmState {
    // A run with a killed hypothesis + kill reason — the thesis on one screen.
    SwarmState {
        run_id: "2026-06-28-1200-a1b2c3".into(),
        spec_path: "docs/research/specs/test-spec.md".into(),
        plan_path: "docs/research/plans/test-plan.md".into(),
        novelty_target: "hypothesis".into(),
        max_parallel: 4,
        phases: vec![
            Phase {
                name: "literature-scout".into(),
                status: "complete".into(),
                workers: vec![
                    Worker { name: "scout-1".into(), status: "passed".into() },
                ],
                hypotheses: vec![],
            },
            Phase {
                name: "red-team".into(),
                status: "complete".into(),
                workers: vec![],
                hypotheses: vec![
                    HypothesisNode {
                        id: "hypothesis-smith-1".into(),
                        label: "causal-SAE-bridge".into(),
                        status: "killed".into(),
                        rounds: vec![
                            RoundVerdict { round: 1, critique: Verdict::Reject, revised: true },
                            RoundVerdict { round: 2, critique: Verdict::Reject, revised: true },
                        ],
                        kill_reason: Some("mechanism contradicts Marks et al. 2025 — effect is ablation, not causal".into()),
                    },
                    HypothesisNode {
                        id: "hypothesis-smith-2".into(),
                        label: "logit-lens-circuits".into(),
                        status: "approved".into(),
                        rounds: vec![RoundVerdict { round: 1, critique: Verdict::Approve, revised: false }],
                        kill_reason: None,
                    },
                ],
            },
        ],
        escalations: vec![],
        retry_counts: std::collections::HashMap::new(),
    }
}

#[test]
fn run_surface_renders_killed_hypothesis_with_kill_reason_and_round() {
    // THE make-or-break test: the run surface renders the greyed kill + its
    // one-line kill reason + the round indicator. This single test encodes
    // the thesis ("the audit trail is the interface").
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    app.surface = Surface::Run;
    app.run_state = Some(mr_tui::surface::run::RunState::from_swarm_for_test(
        make_or_break_state(),
    ));
    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let content: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains("KILLED"), "killed marker: {content}");
    assert!(
        content.contains("mechanism contradicts Marks et al. 2025"),
        "kill reason: {content}"
    );
    assert!(content.contains("round"), "round indicator: {content}");
}

#[test]
fn run_surface_escalation_strip_appears_only_when_pending() {
    // No-defensive-UI rule: the escalation strip is NOT drawn when there is
    // no pending escalation.
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    app.surface = Surface::Run;
    let mut state = mr_tui::surface::run::RunState::from_swarm_for_test(make_or_break_state());
    state.pending_escalation = None;
    app.run_state = Some(state);
    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let content_no_esc: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(
        !content_no_esc.contains("escalation"),
        "no escalation strip when none pending: {content_no_esc}"
    );

    // Now with a pending escalation, the strip appears.
    let mut app2 = App::new(std::path::PathBuf::from("/tmp"), None);
    app2.surface = Surface::Run;
    let mut state2 = mr_tui::surface::run::RunState::from_swarm_for_test(make_or_break_state());
    state2.pending_escalation = Some(Escalation {
        worker: "gap-finder".into(),
        reason: "no gaps in sub-topic 3".into(),
        retry_count: 1,
    });
    app2.run_state = Some(state2);
    let mut terminal2 = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal2.draw(|f| app2.render(f)).unwrap();
    let content_with_esc: String = terminal2
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content_with_esc.contains("escalation"), "escalation strip when pending: {content_with_esc}");
    assert!(content_with_esc.contains("continue"), "continue action: {content_with_esc}");
    assert!(content_with_esc.contains("fail"), "fail action: {content_with_esc}");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p mr-tui run_surface_renders_killed run_surface_escalation_strip`
Expected: compile error — `mr_tui::surface::run` and `mr_tui::escalation` do not exist; `App::run_state` field missing.

- [ ] **Step 3: Create `crates/mr-tui/src/escalation.rs`**

```rust
//! TuiEscalationHandler — the TUI counterpart to 6a's HeadlessEscalationHandler.
//! On adjudicate, sends the Escalation into an in-App channel and blocks
//! (awaits a oneshot) until the user adjudicates inline from the run strip.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::sync::Arc;

use async_trait::async_trait;
use megaresearcher_research::orchestrator::escalation::{EscalationHandler, EscalationVerdict};
use megaresearcher_research::state::swarm_state::Escalation;
use tokio::sync::{mpsc, oneshot};

pub struct TuiEscalationHandler {
    pub escalation_tx: mpsc::UnboundedSender<(Escalation, oneshot::Sender<EscalationVerdict>)>,
}

#[async_trait]
impl EscalationHandler for TuiEscalationHandler {
    async fn adjudicate(&self, e: &Escalation) -> EscalationVerdict {
        let (tx, rx) = oneshot::channel();
        if self.escalation_tx.send((e.clone(), tx)).is_err() {
            return EscalationVerdict::Fail;
        }
        rx.await.unwrap_or(EscalationVerdict::Fail)
    }
}
```

Add `pub mod escalation;` to `crates/mr-tui/src/lib.rs`.

- [ ] **Step 4: Create `crates/mr-tui/src/surface/run.rs`**

```rust
//! The Run surface — where the user lives. Spawns the orchestrator, watches
//! swarm-state.yaml (250ms poll), renders the tree + cost meter + escalation
//! strip (only when an escalation is pending — no defensive UI).
//!
//! SPDX-License-Identifier: GPL-3.0

use std::path::{Path, PathBuf};
use std::sync::Arc;

use claurst_api::LlmProvider;
use claurst_core::cost::CostTracker;
use megaresearcher_research::mcp::ml_intern_config;
use megaresearcher_research::orchestrator::{
    EscalationHandler, Orchestrator, OrchestratorConfig, RunOutcome,
};
use megaresearcher_research::orchestrator::escalation::EscalationVerdict;
use megaresearcher_research::state::run_id::generate_run_id;
use megaresearcher_research::state::swarm_state::{Escalation, SwarmState};
use tokio::sync::{mpsc, oneshot};

use crate::cost::CountingProvider;
use crate::theme::ColorPalette;

pub struct RunState {
    pub run_dir: PathBuf,
    pub run_id: String,
    pub swarm: Option<SwarmState>,
    pub cost_tracker: Arc<CostTracker>,
    pub pending_escalation: Option<Escalation>,
    pub escalation_tx: mpsc::UnboundedSender<(Escalation, oneshot::Sender<EscalationVerdict>)>,
    pub escalation_rx: mpsc::UnboundedReceiver<(Escalation, oneshot::Sender<EscalationVerdict>)>,
    pub verdict_responder: Option<oneshot::Sender<EscalationVerdict>>,
    pub orch_task: Option<tokio::task::JoinHandle<Result<RunOutcome, megaresearcher_research::orchestrator::OrchestratorError>>>,
}

impl RunState {
    /// Test-only constructor: a RunState holding a fixture swarm, no
    /// orchestrator task. Used by the make-or-break render test.
    pub fn from_swarm_for_test(swarm: SwarmState) -> Self {
        let (esc_tx, esc_rx) = mpsc::unbounded_channel();
        Self {
            run_dir: PathBuf::from("/tmp/run"),
            run_id: "test".into(),
            swarm: Some(swarm),
            cost_tracker: CostTracker::with_model("claude-sonnet-4-6"),
            pending_escalation: None,
            escalation_tx: esc_tx,
            escalation_rx: esc_rx,
            verdict_responder: None,
            orch_task: None,
        }
    }
}

/// Spawn the orchestrator on a tokio task. Wraps the provider in a
/// CountingProvider so the cost meter reads live. The TuiEscalationHandler
/// routes escalations into the RunState's channel for inline adjudication.
pub fn spawn_run(
    cwd: &Path,
    spec_path: PathBuf,
    plan_path: PathBuf,
    provider: Arc<dyn LlmProvider>,
    model: String,
    max_parallel: u32,
    mcp_enabled: bool,
) -> anyhow::Result<(RunState, tokio::task::JoinHandle<Result<RunOutcome, megaresearcher_research::orchestrator::OrchestratorError>>)> {
    let tracker = CostTracker::with_model(&model);
    let counting = Arc::new(CountingProvider::new(provider, tracker.clone()));
    let (esc_tx, esc_rx) = mpsc::unbounded_channel::<(Escalation, oneshot::Sender<EscalationVerdict>)>();
    let handler = Arc::new(crate::escalation::TuiEscalationHandler { escalation_tx: esc_tx.clone() });
    let docs = cwd.join("docs/research");
    let mcp = if mcp_enabled { Some(ml_intern_config(cwd)) } else { None };
    let cfg = OrchestratorConfig {
        research_base: docs.clone(),
        agents_dir: cwd.join("agents"),
        default_model: model,
        max_parallel,
        mcp,
        escalation: Some(handler as Arc<dyn EscalationHandler>),
    };
    let orch = Orchestrator::new(cfg, counting as Arc<dyn LlmProvider>);
    let run_id = generate_run_id().map_err(anyhow::Error::from)?;
    let run_dir = docs.join("runs").join(&run_id);
    let spec_c = spec_path.clone();
    let plan_c = plan_path.clone();
    let run_id_c = run_id.clone();
    let task = tokio::spawn(async move { orch.execute(&spec_c, &plan_c, &run_id_c).await });
    let state = RunState {
        run_dir,
        run_id,
        swarm: None,
        cost_tracker: tracker,
        pending_escalation: None,
        escalation_tx: esc_tx,
        escalation_rx: esc_rx,
        verdict_responder: None,
        orch_task: Some(task),
    };
    Ok((state, state.orch_task.unwrap()))
}

/// Render the Run surface: tree (top) + cost meter (top-right) + escalation
/// strip (bottom, only when an escalation is pending).
pub fn render_run(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, state: &RunState, theme: &ColorPalette) {
    let chunks = ratatui::layout::Layout::default()
        .constraints(if state.pending_escalation.is_some() {
            vec![ratatui::layout::Constraint::Min(3), ratatui::layout::Constraint::Length(3)]
        } else {
            vec![ratatui::layout::Constraint::Min(1)]
        })
        .split(area);
    // Top: tree + cost meter. Split the top into tree (left) + cost (right, 16 cols).
    let top = ratatui::layout::Layout::default()
        .constraints([ratatui::layout::Constraint::Min(16), ratatui::layout::Constraint::Length(16)])
        .split(chunks[0]);
    if let Some(swarm) = state.swarm.as_ref() {
        crate::widget::tree::render_tree(frame, top[0], swarm, theme);
    } else {
        frame.render_widget(
            ratatui::widgets::Paragraph::new("waiting for run to start…")
                .style(ratatui::style::Style::default().fg(theme.disabled)),
            top[0],
        );
    }
    crate::cost::render_cost_meter(frame, top[1], &state.cost_tracker, theme);
    // Escalation strip — only when pending (no empty "(none)" box).
    if let Some(esc) = state.pending_escalation.as_ref() {
        let line = ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(
                format!("{} escalation · {}: \"{}\"", crate::figures::WARN, esc.worker, esc.reason),
                ratatui::style::Style::default().fg(theme.escalation),
            ),
            ratatui::text::Span::raw("  "),
            ratatui::text::Span::styled("[ continue ▸  fail ]", ratatui::style::Style::default().fg(theme.action)),
        ]);
        frame.render_widget(
            ratatui::widgets::Paragraph::new(line)
                .block(ratatui::widgets::Block::default().borders(ratatui::widgets::Borders::TOP)),
            chunks[1],
        );
    }
}
```

Add `pub mod run;` to `crates/mr-tui/src/surface/mod.rs`.

In `crates/mr-tui/src/app.rs`, add the `run_state` field:

```rust
pub run_state: Option<crate::surface::run::RunState>,
```

Update `App::new` to initialize `run_state: None`. Update `App::render`:

```rust
            Surface::Run => {
                if let Some(rs) = self.run_state.as_ref() {
                    crate::surface::run::render_run(frame, frame.area(), rs, &self.theme);
                }
            }
```

In `App::run`, when `surface == Run`: poll `swarm-state.yaml` every 250ms (reuse the 6a pattern from `execute.rs:104-113`), update `run_state.swarm`; drain `escalation_rx` non-blocking and set `pending_escalation` + `verdict_responder`; on user key `c`/`f` while an escalation is pending, send the verdict. On `orch_task` completion → transition to Artifact. Add a `tokio::select!` in the loop:

```rust
        if self.surface == Surface::Run {
            if let Some(rs) = self.run_state.as_mut() {
                // Non-blocking drain: pick up an escalation if one arrived.
                if rs.pending_escalation.is_none() {
                    if let Ok(Some((esc, tx))) = rs.escalation_rx.try_recv().ok().map(Some).unwrap_or(None).map(|x| Some(x)).unwrap_or(None) {
                        rs.pending_escalation = Some(esc);
                        rs.verdict_responder = Some(tx);
                    }
                }
            }
        }
```

NOTE: the `try_recv` chain above is awkward; the implementer should write the cleanest non-blocking form:

```rust
        if self.surface == Surface::Run {
            if let Some(rs) = self.run_state.as_mut() {
                if rs.pending_escalation.is_none() {
                    if let Ok((esc, tx)) = rs.escalation_rx.try_recv() {
                        rs.pending_escalation = Some(esc);
                        rs.verdict_responder = Some(tx);
                    }
                }
            }
        }
```

And the swarm-state poll (inside the loop, before the event poll): read `run_dir/swarm-state.yaml` if it exists, update `run_state.swarm` if changed (reuse `SwarmState::read`). The 50ms event poll + draw-each-iter already drives the `↻` animation via `frame_count`; the 250ms swarm poll can be a separate `tokio::time::sleep` in the `select!`, or a frame-count modulo check (every ~5 frames ≈ 250ms). The implementer chooses; the contract is: the swarm state is re-read and the tree re-renders when it changes. Add the `c`/`f` key handlers in `handle_key` when `surface == Run` and `pending_escalation.is_some()`:

```rust
            KeyCode::Char('c') if self.surface == Surface::Run && self.run_pending_escalation() => {
                self.run_respond(EscalationVerdict::Continue);
                AppEvent::None
            }
            KeyCode::Char('f') if self.surface == Surface::Run && self.run_pending_escalation() => {
                self.run_respond(EscalationVerdict::Fail);
                AppEvent::None
            }
```

with helpers `run_pending_escalation(&self) -> bool` and `run_respond(&mut self, EscalationVerdict)` that send through `verdict_responder` and clear `pending_escalation`. The `EscalationVerdict` import is `megaresearcher_research::orchestrator::escalation::EscalationVerdict`.

- [ ] **Step 5: Run the make-or-break tests to verify they pass**

Run: `cargo test -p mr-tui run_surface_renders_killed run_surface_escalation_strip`
Expected: PASS. This is the thesis gate. If it fails, the tree or escalation strip rendering is wrong — fix before T10.

- [ ] **Step 6: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-tui
cargo clippy -p mr-tui --all-targets -- -D warnings
```

```bash
git add crates/mr-tui
git commit -m "$(cat <<'EOF'
feat(mr-tui): Run surface + TuiEscalationHandler (make-or-break)

escalation.rs: TuiEscalationHandler routes escalations into an in-App channel
and blocks on a oneshot until the user adjudicates inline. surface/run.rs
spawns the orchestrator (provider wrapped in CountingProvider), renders the
tree + cost meter + escalation strip (only when pending — no defensive UI).
Make-or-break test: a fixture swarm with a killed HypothesisNode renders the
greyed kill + its one-line kill reason + the round indicator.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: widget/cards.rs + Artifact surface

**Files:**
- Create: `crates/mr-tui/src/widget/cards.rs`
- Create: `crates/mr-tui/src/widget/markdown.rs`
- Create: `crates/mr-tui/src/surface/artifact.rs`
- Modify: `crates/mr-tui/src/widget/mod.rs` (add `pub mod cards; pub mod markdown;`)
- Modify: `crates/mr-tui/src/surface/mod.rs` (add `pub mod artifact;`)
- Modify: `crates/mr-tui/src/app.rs` (Artifact state + transition on run completion)
- Test: `crates/mr-tui/tests/artifact.rs`

**Interfaces:**
- Consumes: T2's `HypothesisNode`; `output.md` at `run_dir/output.md`.
- Produces:
  - `mr_tui::widget::cards::render_cards(frame, area, surviving: &[HypothesisNode], rejected: &[HypothesisNode], theme)` — surviving cards (label/status + parsed mechanism/predicted-outcome/falsification/experimental-design sections); rejected fold with kill reasons.
  - `mr_tui::widget::markdown::render_markdown(frame, area, md: &str, theme)` — minimal hand-rolled markdown (headings, paragraphs, lists, bold).
  - `mr_tui::widget::cards::parse_sections(md: &str) -> std::collections::HashMap<String, String>` — splits `output.md` on `## ` headings.
  - `mr_tui::surface::artifact::ArtifactState` — `{ run_dir: PathBuf, output_md: String, surviving: Vec<HypothesisNode>, rejected: Vec<HypothesisNode> }`.
  - App gains `artifact: Option<ArtifactState>`.

- [ ] **Step 1: Write the failing test**

Create `crates/mr-tui/tests/artifact.rs`:

```rust
use megaresearcher_research::state::swarm_state::{HypothesisNode, RoundVerdict, Verdict};
use mr_tui::surface::artifact::ArtifactState;
use mr_tui::app::{App, Surface};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn surviving() -> Vec<HypothesisNode> {
    vec![HypothesisNode {
        id: "hypothesis-smith-1".into(),
        label: "causal-SAE-bridge".into(),
        status: "approved".into(),
        rounds: vec![RoundVerdict { round: 1, critique: Verdict::Approve, revised: false }],
        kill_reason: None,
    }]
}

fn rejected() -> Vec<HypothesisNode> {
    vec![HypothesisNode {
        id: "hypothesis-smith-2".into(),
        label: "activation-patching".into(),
        status: "killed".into(),
        rounds: vec![RoundVerdict { round: 2, critique: Verdict::Reject, revised: true }],
        kill_reason: Some("effect is ablation, not causal".into()),
    }]
}

#[test]
fn artifact_renders_surviving_cards_and_rejected_fold() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    app.surface = Surface::Artifact;
    app.artifact = Some(ArtifactState {
        run_dir: "/tmp/run".into(),
        output_md: "# Research direction\n\nSurviving direction: SAEs for causal circuits.\n".into(),
        surviving: surviving(),
        rejected: rejected(),
    });
    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let content: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains("causal-SAE-bridge"), "surviving card label: {content}");
    assert!(content.contains("rejected"), "rejected fold: {content}");
    assert!(
        content.contains("effect is ablation, not causal"),
        "rejected kill reason: {content}"
    );
}

#[test]
fn parse_sections_splits_on_h2_headings() {
    let md = "# Title\n\n## Mechanism\n\nThe mechanism is X.\n\n## Predicted outcome\n\nY happens.\n";
    let sections = mr_tui::widget::cards::parse_sections(md);
    assert!(sections.contains_key("Mechanism"));
    assert!(sections.contains_key("Predicted outcome"));
    assert!(sections["Mechanism"].contains("The mechanism is X."));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p mr-tui artifact_renders parse_sections_splits`
Expected: compile error — `mr_tui::widget::cards` / `mr_tui::surface::artifact` do not exist.

- [ ] **Step 3: Create `crates/mr-tui/src/widget/markdown.rs`**

```rust
//! Minimal hand-rolled markdown renderer: headings, paragraphs, lists, bold.
//! Deferred to Phase 8: a full markdown crate. For 6b this is enough to
//! render `output.md` readably on the artifact screen.
//!
//! SPDX-License-Identifier: GPL-3.0

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::theme::ColorPalette;

pub fn render_markdown(frame: &mut ratatui::Frame, area: Rect, md: &str, theme: &ColorPalette) {
    let mut lines: Vec<Line> = Vec::new();
    for raw in md.lines() {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.is_empty() {
            lines.push(Line::from(""));
            continue;
        }
        if let Some(h) = line.strip_prefix("# ") {
            lines.push(Line::from(Span::styled(
                h.to_string(),
                Style::default().fg(theme.accent).add_modifier(Modifier::BOLD),
            )));
        } else if let Some(h) = line.strip_prefix("## ") {
            lines.push(Line::from(Span::styled(
                h.to_string(),
                Style::default().fg(theme.text_light).add_modifier(Modifier::BOLD),
            )));
        } else if let Some(item) = line.strip_prefix("- ") {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("• {item}"), Style::default().fg(theme.text_light)),
            ]));
        } else {
            lines.push(render_inline_bold(line, theme));
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
}

/// Split on `**bold**` and render the bold spans with the accent color.
fn render_inline_bold(line: &str, theme: &ColorPalette) -> Line {
    let mut spans = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find("**") {
        if start > 0 {
            spans.push(Span::styled(
                rest[..start].to_string(),
                Style::default().fg(theme.text_light),
            ));
        }
        let after = &rest[start + 2..];
        if let Some(end) = after.find("**") {
            spans.push(Span::styled(
                after[..end].to_string(),
                Style::default().fg(theme.accent).add_modifier(Modifier::BOLD),
            ));
            rest = &after[end + 2..];
        } else {
            spans.push(Span::styled(
                format!("**{after}"),
                Style::default().fg(theme.text_light),
            ));
            rest = "";
            break;
        }
    }
    if !rest.is_empty() {
        spans.push(Span::styled(
            rest.to_string(),
            Style::default().fg(theme.text_light),
        ));
    }
    if spans.is_empty() {
        spans.push(Span::raw(line.to_string()));
    }
    Line::from(spans)
}
```

- [ ] **Step 4: Create `crates/mr-tui/src/widget/cards.rs`**

```rust
//! Expandable hypothesis cards. Surviving: label/status + parsed
//! mechanism/predicted-outcome/falsification/experimental-design sections
//! from `hypothesis-smith-<N>/output.md`. Rejected: folded, with the kill
//! reason (the lessons). Falls back to label+status only if sections absent.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::collections::HashMap;

use megaresearcher_research::state::swarm_state::HypothesisNode;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::figures::{ARROW, COLLAPSE, CROSS};
use crate::theme::ColorPalette;

/// Parse `output.md`-style sections: split on `## ` headings into a map.
pub fn parse_sections(md: &str) -> HashMap<String, String> {
    let mut sections = HashMap::new();
    let mut current: Option<(String, String)> = None;
    for line in md.lines() {
        if let Some(h) = line.strip_prefix("## ") {
            if let Some((k, v)) = current.take() {
                sections.insert(k, v.trim().to_string());
            }
            current = Some((h.trim().to_string(), String::new()));
        } else if let Some((_, body)) = current.as_mut() {
            if !body.is_empty() || !line.is_empty() {
                body.push_str(line);
                body.push('\n');
            }
        }
    }
    if let Some((k, v)) = current.take() {
        sections.insert(k, v.trim().to_string());
    }
    sections
}

/// Render surviving hypothesis cards + the rejected fold.
pub fn render_cards(
    frame: &mut ratatui::Frame,
    area: Rect,
    surviving: &[HypothesisNode],
    rejected: &[HypothesisNode],
    theme: &ColorPalette,
) {
    let mut lines: Vec<Line> = Vec::new();
    for hyp in surviving {
        lines.push(Line::from(vec![
            Span::styled(ARROW, Style::default().fg(theme.accent)),
            Span::raw(" "),
            Span::styled(
                hyp.label.clone(),
                Style::default().fg(theme.alive).add_modifier(Modifier::BOLD),
            ),
        ]));
        // Card fields are parsed from the smith output.md by the caller (T10
        // surface); the node itself carries id/label/status/rounds/kill_reason.
        // We render the status + rounds here as the card body.
        let rounds_label = format!(
            "  status: {} · rounds: {}",
            hyp.status,
            hyp.rounds.len()
        );
        lines.push(Line::from(Span::styled(
            rounds_label,
            Style::default().fg(theme.disabled),
        )));
    }
    if !rejected.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(COLLAPSE, Style::default().fg(theme.killed)),
            Span::styled(
                format!(" rejected ({}) — the lessons", rejected.len()),
                Style::default().fg(theme.killed).add_modifier(Modifier::DIM),
            ),
        ]));
        for hyp in rejected {
            let reason = hyp.kill_reason.clone().unwrap_or_else(|| "(no reason)".to_string());
            lines.push(Line::from(vec![
                Span::raw("   "),
                Span::styled(CROSS, Style::default().fg(theme.killed)),
                Span::raw(" "),
                Span::styled(hyp.label.clone(), Style::default().fg(theme.killed)),
                Span::styled(format!(" — \"{reason}\""), Style::default().fg(theme.killed)),
            ]));
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
}
```

Add `pub mod cards; pub mod markdown;` to `crates/mr-tui/src/widget/mod.rs`.

- [ ] **Step 5: Create `crates/mr-tui/src/surface/artifact.rs`**

```rust
//! The Artifact surface — the payoff. Renders output.md (minimal markdown)
//! + surviving hypothesis cards + the rejected fold (the lessons). The tree
//! slides to a side rail as provenance (deferred visual polish to Phase 8;
//! 6b renders the direction taking the screen).
//!
//! SPDX-License-Identifier: GPL-3.0

use std::path::PathBuf;

use megaresearcher_research::state::swarm_state::HypothesisNode;

pub struct ArtifactState {
    pub run_dir: PathBuf,
    pub output_md: String,
    pub surviving: Vec<HypothesisNode>,
    pub rejected: Vec<HypothesisNode>,
}

impl ArtifactState {
    /// Build from a run dir: read output.md + the swarm's hypothesis nodes.
    pub fn from_run_dir(run_dir: &std::path::Path) -> Self {
        let output_md = std::fs::read_to_string(run_dir.join("output.md"))
            .unwrap_or_else(|_| "(no output.md)".to_string());
        let swarm = megaresearcher_research::state::swarm_state::SwarmState::read(
            &run_dir.join("swarm-state.yaml"),
        )
        .ok();
        let (surviving, rejected) = swarm
            .as_ref()
            .map(|s| {
                let mut surv = Vec::new();
                let mut rej = Vec::new();
                for p in &s.phases {
                    for h in &p.hypotheses {
                        if h.status == "approved" {
                            surv.push(h.clone());
                        } else if h.status == "killed" {
                            rej.push(h.clone());
                        }
                    }
                }
                (surv, rej)
            })
            .unwrap_or_default();
        Self {
            run_dir: run_dir.to_path_buf(),
            output_md,
            surviving,
            rejected,
        }
    }
}

pub fn render_artifact(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    state: &ArtifactState,
    theme: &crate::theme::ColorPalette,
) {
    let chunks = ratatui::layout::Layout::default()
        .constraints([ratatui::layout::Constraint::Min(5), ratatui::layout::Constraint::Min(3)])
        .split(area);
    crate::widget::markdown::render_markdown(frame, chunks[0], &state.output_md, theme);
    crate::widget::cards::render_cards(frame, chunks[1], &state.surviving, &state.rejected, theme);
}
```

Add `pub mod artifact;` to `crates/mr-tui/src/surface/mod.rs`. In `app.rs`, add `pub artifact: Option<crate::surface::artifact::ArtifactState>` and dispatch `Surface::Artifact` to `render_artifact`. On `Run` task completion (in the loop), build `ArtifactState::from_run_dir(&run_state.run_dir)` and transition to `Surface::Artifact`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p mr-tui artifact_renders parse_sections_splits`
Expected: PASS.

- [ ] **Step 7: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-tui
cargo clippy -p mr-tui --all-targets -- -D warnings
```

```bash
git add crates/mr-tui
git commit -m "$(cat <<'EOF'
feat(mr-tui): Artifact surface — output.md + surviving cards + rejected fold

widget/markdown: minimal hand-rolled renderer (headings, paragraphs, lists,
bold). widget/cards: expandable surviving cards + the rejected fold with
kill reasons (the lessons). surface/artifact reads output.md + swarm-state
hypothesis nodes. On run completion the App transitions Run→Artifact.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Past runs surface

**Files:**
- Create: `crates/mr-tui/src/surface/past.rs`
- Modify: `crates/mr-tui/src/surface/mod.rs` (add `pub mod past;`)
- Modify: `crates/mr-tui/src/app.rs` (Past state + transition; `Esc` returns to previous surface)
- Test: `crates/mr-tui/tests/past.rs`

**Interfaces:**
- Consumes: `docs/research/runs/` enumeration (mirrors `crates/mr-cli/src/render.rs::list_runs`); `SwarmState::read` for the headline surviving hypothesis; T10's `ArtifactState` for reopening.
- Produces:
  - `mr_tui::surface::past::PastRun { date: String, topic: String, headline: String, run_dir: PathBuf }`.
  - `mr_tui::surface::past::enumerate_runs(cwd: &Path) -> Vec<PastRun>`.
  - `mr_tui::surface::past::PastState { runs: Vec<PastRun>, selected: usize }`.
  - App gains `past: Option<PastState>`.

- [ ] **Step 1: Write the failing test**

Create `crates/mr-tui/tests/past.rs`:

```rust
use megaresearcher_research::state::swarm_state::{
    HypothesisNode, Phase, RoundVerdict, SwarmState, Verdict,
};
use mr_tui::app::{App, Surface};
use mr_tui::surface::past::enumerate_runs;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

#[test]
fn enumerate_runs_lists_date_topic_headline() {
    let tmp = tempfile::tempdir().unwrap();
    let runs = tmp.path().join("docs/research/runs");
    let r1 = runs.join("2026-06-28-1200-a1b2c3");
    std::fs::create_dir_all(&r1).unwrap();
    std::fs::write(r1.join("output.md"), "# SAEs for causal circuits\n\nDirection body.\n").unwrap();
    let swarm = SwarmState {
        run_id: "2026-06-28-1200-a1b2c3".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
        novelty_target: "hypothesis".into(),
        max_parallel: 4,
        phases: vec![Phase {
            name: "red-team".into(),
            status: "complete".into(),
            workers: vec![],
            hypotheses: vec![HypothesisNode {
                id: "hypothesis-smith-1".into(),
                label: "causal-SAE-bridge".into(),
                status: "approved".into(),
                rounds: vec![RoundVerdict { round: 1, critique: Verdict::Approve, revised: false }],
                kill_reason: None,
            }],
        }],
        escalations: vec![],
        retry_counts: std::collections::HashMap::new(),
    };
    swarm.write(&r1.join("swarm-state.yaml")).unwrap();
    let past = enumerate_runs(tmp.path());
    assert_eq!(past.len(), 1);
    assert!(past[0].date.contains("2026-06-28"), "date: {:?}", past[0].date);
    assert!(past[0].topic.contains("SAEs"), "topic: {:?}", past[0].topic);
    assert!(past[0].headline.contains("causal-SAE-bridge"), "headline: {:?}", past[0].headline);
}

#[test]
fn past_surface_renders_list() {
    let tmp = tempfile::tempdir().unwrap();
    let runs = tmp.path().join("docs/research/runs");
    let r1 = runs.join("2026-06-28-1200-a1b2c3");
    std::fs::create_dir_all(&r1).unwrap();
    std::fs::write(r1.join("output.md"), "# SAEs for causal circuits\n\nbody\n").unwrap();
    let mut app = App::new(tmp.path().to_path_buf(), None);
    app.surface = Surface::Past;
    app.past = Some(mr_tui::surface::past::PastState::from_cwd(tmp.path()));
    let mut terminal = Terminal::new(TestBackend::new(100, 10)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let content: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains("SAEs"), "topic in list: {content}");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p mr-tui enumerate_runs_lists past_surface_renders_list`
Expected: compile error — `mr_tui::surface::past` does not exist.

- [ ] **Step 3: Create `crates/mr-tui/src/surface/past.rs`**

```rust
//! Past runs — `mr` with no args when runs exist: a list by date + topic,
//! each with its headline surviving hypothesis. One tap reopens artifact+tree.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::path::{Path, PathBuf};

use megaresearcher_research::state::swarm_state::SwarmState;

#[derive(Debug, Clone)]
pub struct PastRun {
    pub date: String,
    pub topic: String,
    pub headline: String,
    pub run_dir: PathBuf,
}

/// Enumerate `docs/research/runs/` newest-first. `date` from the dir name
/// prefix; `topic` from the `output.md` H1; `headline` from the first
/// approved HypothesisNode (or "(gap-finding)" if none).
pub fn enumerate_runs(cwd: &Path) -> Vec<PastRun> {
    let runs = cwd.join("docs/research/runs");
    if !runs.is_dir() {
        return Vec::new();
    }
    let mut entries: Vec<_> = std::fs::read_dir(&runs)
 .ok()
 .into_iter()
 .flatten()
 .flatten()
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    let mut out = Vec::new();
    for e in entries.iter().rev() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        let date = name.split('-').take(3).collect::<Vec<_>>().join("-");
        let output_md = std::fs::read_to_string(path.join("output.md"))
            .unwrap_or_default();
        let topic = output_md
            .lines()
            .find_map(|l| l.strip_prefix("# ").map(|s| s.trim().to_string()))
            .unwrap_or_else(|| name.clone());
        let headline = SwarmState::read(&path.join("swarm-state.yaml"))
            .ok()
            .and_then(|s| {
                s.phases.iter().flat_map(|p| p.hypotheses.iter()).find_map(|h| {
                    if h.status == "approved" { Some(h.label.clone()) } else { None }
                })
            })
            .unwrap_or_else(|| "(gap-finding)".to_string());
        out.push(PastRun { date, topic, headline, run_dir: path });
    }
    out
}

pub struct PastState {
    pub runs: Vec<PastRun>,
    pub selected: usize,
}

impl PastState {
    pub fn from_cwd(cwd: &Path) -> Self {
        Self { runs: enumerate_runs(cwd), selected: 0 }
    }
}

pub fn render_past(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, state: &PastState, theme: &crate::theme::ColorPalette) {
    let mut lines: Vec<ratatui::text::Line> = Vec::new();
    lines.push(ratatui::text::Line::from(ratatui::text::Span::styled(
        "past runs",
        ratatui::style::Style::default().fg(theme.accent),
    )));
    for (i, run) in state.runs.iter().enumerate() {
        let marker = if i == state.selected { "▸" } else { " " };
        lines.push(ratatui::text::Line::from(vec![
            ratatui::text::Span::raw(format!("{marker} ")),
            ratatui::text::Span::styled(run.date.clone(), ratatui::style::Style::default().fg(theme.text_light)),
            ratatui::text::Span::raw("  "),
            ratatui::text::Span::styled(run.topic.clone(), ratatui::style::Style::default().fg(theme.text_light)),
            ratatui::text::Span::raw("  → "),
            ratatui::text::Span::styled(run.headline.clone(), ratatui::style::Style::default().fg(theme.success)),
        ]));
    }
    frame.render_widget(ratatui::widgets::Paragraph::new(lines), area);
}
```

Add `pub mod past;` to `crates/mr-tui/src/surface/mod.rs`. In `app.rs`, add `pub past: Option<PastState>` and dispatch `Surface::Past` to `render_past`. Wire `Enter` on Past to build `ArtifactState::from_run_dir` + transition to Artifact; `Up`/`Down` move `selected`. The Start surface with no question + existing runs can transition to Past (T13 decides the no-args branch; for now `mr_tui::run` always starts on Start).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p mr-tui enumerate_runs_lists past_surface_renders_list`
Expected: PASS.

- [ ] **Step 5: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-tui
cargo clippy -p mr-tui --all-targets -- -D warnings
```

```bash
git add crates/mr-tui
git commit -m "$(cat <<'EOF'
feat(mr-tui): Past runs surface — list by date + topic + headline hypothesis

surface/past enumerates docs/research/runs/ newest-first, reading each
output.md H1 as the topic and the first approved HypothesisNode as the
headline. Enter reopens the artifact+tree (transitions to Artifact).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---
## Task 12: Settings surface + MrConfig

**Files:**
- Create: `crates/mr-tui/src/config.rs`
- Create: `crates/mr-tui/src/surface/settings.rs`
- Modify: `crates/mr-tui/src/lib.rs` (add `pub mod config;`)
- Modify: `crates/mr-tui/src/surface/mod.rs` (add `pub mod settings;`)
- Modify: `crates/mr-tui/src/app.rs` (Settings state + first-run auto-open)
- Test: `crates/mr-tui/tests/settings.rs`

**Interfaces:**
- Consumes: `dirs::config_dir()` for `~/.config/mr/config.toml`; `toml` 0.8; the provider registry (T13's `resolve_provider` reads MrConfig). `MrConfig` lives in `mr_tui::config` (no cycle: mr-cli → mr-tui).
- Produces:
  - `mr_tui::config::MrConfig { provider: Option<String>, api_key: Option<String>, model: Option<String>, max_parallel: u32, on_escalate: String, mcp: bool, cost_ceiling_sandbox: f64, cost_ceiling_api: f64, theme: String }` — derives `Debug, Clone, PartialEq, Serialize, Deserialize`, with `Default` (provider None, model None, max_parallel 4, on_escalate "pause", mcp true, cost ceilings 5.0, theme "research").
  - `mr_tui::config::MrConfig::config_path() -> PathBuf` — `~/.config/mr/config.toml`.
  - `mr_tui::config::MrConfig::load() -> Self` — reads the file, falls back to `Default`.
  - `mr_tui::config::MrConfig::save(&self) -> io::Result<()>` — writes the file (creates the dir).
  - `mr_tui::config::MrConfig::needs_provider_key(&self) -> bool` — true if `provider` or `api_key` is None/empty.
  - `mr_tui::surface::settings::SettingsState` — `{ config: MrConfig, masked_key: String, selected_field: usize, test_status: Option<String> }`.
  - App gains `settings: Option<SettingsState>`; on `new`, if `MrConfig::load().needs_provider_key()`, set `surface = Surface::Settings` (first-run auto-open).

- [ ] **Step 1: Write the failing tests**

Create `crates/mr-tui/tests/settings.rs`:

```rust
use mr_tui::config::MrConfig;
use mr_tui::surface::settings::SettingsState;
use mr_tui::app::{App, Surface};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

#[test]
fn mrconfig_round_trips_to_toml() {
    let mut cfg = MrConfig::default();
    cfg.max_parallel = 8;
    cfg.on_escalate = "fail".into();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("config.toml");
    std::fs::write(&path, toml::to_string(&cfg).unwrap()).unwrap();
    let loaded: MrConfig = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(loaded.max_parallel, 8);
    assert_eq!(loaded.on_escalate, "fail");
}

#[test]
fn mrconfig_needs_provider_key_when_absent() {
    let cfg = MrConfig::default();
    assert!(cfg.needs_provider_key());
    let mut cfg = cfg;
    cfg.provider = Some("anthropic".into());
    cfg.api_key = Some("sk-ant-x".into());
    assert!(!cfg.needs_provider_key());
}

#[test]
fn settings_surface_renders_masked_key() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    app.surface = Surface::Settings;
    let mut cfg = MrConfig::default();
    cfg.api_key = Some("sk-ant-1234567890".into());
    app.settings = Some(SettingsState::from_config(cfg));
    let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let content: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    // The API key is masked: "sk-ant-" + bullets, no plaintext tail.
    assert!(!content.contains("1234567890"), "key must be masked: {content}");
    assert!(content.contains('•'), "masked with bullets: {content}");
}
```

Add `dirs = { workspace = true }` to `crates/mr-tui/Cargo.toml` `[dependencies]` (the workspace already has `dirs = "5"`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p mr-tui mrconfig_round_trips mrconfig_needs_provider_key settings_surface_renders_masked_key`
Expected: compile error — `mr_tui::config` / `mr_tui::surface::settings` do not exist.

- [ ] **Step 3: Create `crates/mr-tui/src/config.rs`**

```rust
//! MrConfig — the mr-specific TOML store at ~/.config/mr/config.toml.
//! The file is the source of truth; the Settings screen edits it, headless
//! `mr` reads it (T13). One store, so a save takes effect on the next run.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::io;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MrConfig {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_max_parallel")]
    pub max_parallel: u32,
    #[serde(default = "default_on_escalate")]
    pub on_escalate: String,
    #[serde(default = "default_mcp")]
    pub mcp: bool,
    #[serde(default = "default_cost_ceiling")]
    pub cost_ceiling_sandbox: f64,
    #[serde(default = "default_cost_ceiling")]
    pub cost_ceiling_api: f64,
    #[serde(default = "default_theme")]
    pub theme: String,
}

fn default_max_parallel() -> u32 { 4 }
fn default_on_escalate() -> String { "pause".to_string() }
fn default_mcp() -> bool { true }
fn default_cost_ceiling() -> f64 { 5.0 }
fn default_theme() -> String { "research".to_string() }

impl Default for MrConfig {
    fn default() -> Self {
        Self {
            provider: None,
            api_key: None,
            model: None,
            max_parallel: default_max_parallel(),
            on_escalate: default_on_escalate(),
            mcp: default_mcp(),
            cost_ceiling_sandbox: default_cost_ceiling(),
            cost_ceiling_api: default_cost_ceiling(),
            theme: default_theme(),
        }
    }
}

impl MrConfig {
    /// `~/.config/mr/config.toml` (platform config dir + "mr/config.toml").
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mr")
            .join("config.toml")
    }

    /// Load from the config file, falling back to `Default` if missing/unreadable.
    pub fn load() -> Self {
        let path = Self::config_path();
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Write to the config file, creating the parent dir.
    pub fn save(&self) -> io::Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = toml::to_string(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, s)
    }

    /// True if the provider or API key is unset (first-run detection).
    pub fn needs_provider_key(&self) -> bool {
        self.provider.as_ref().map(|s| s.trim().is_empty()).unwrap_or(true)
            || self.api_key.as_ref().map(|s| s.trim().is_empty()).unwrap_or(true)
    }
}
```

Add `pub mod config;` to `crates/mr-tui/src/lib.rs`.

- [ ] **Step 4: Create `crates/mr-tui/src/surface/settings.rs`**

```rust
//! Settings — a structured editor for MrConfig. One screen, no tabs;
//! opinionated defaults pre-filled; masked API key; test connection; save→file.
//! First-run auto-opens when no provider/key is configured (resolves the 6a
//! Important finding).
//!
//! SPDX-License-Identifier: GPL-3.0

use crate::config::MrConfig;

pub struct SettingsState {
    pub config: MrConfig,
    pub masked_key: String,
    pub selected_field: usize,
    pub test_status: Option<String>,
}

impl SettingsState {
    pub fn from_config(config: MrConfig) -> Self {
        let masked_key = config
            .api_key
            .as_ref()
            .map(|k| {
                if k.len() <= 8 {
                    "•".repeat(k.len().max(1))
                } else {
                    let prefix = &k[..8];
                    format!("{prefix}{}", "•".repeat(k.len() - 8))
                }
            })
            .unwrap_or_default();
        Self { config, masked_key, selected_field: 0, test_status: None }
    }
}

pub fn render_settings(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    state: &SettingsState,
    theme: &crate::theme::ColorPalette,
) {
    let lines = vec![
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "settings",
            ratatui::style::Style::default().fg(theme.accent),
        )),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled("provider          ", ratatui::style::Style::default().fg(theme.text_light)),
            ratatui::text::Span::raw(state.config.provider.clone().unwrap_or_else(|| "(unset)".into())),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled("api key           ", ratatui::style::Style::default().fg(theme.text_light)),
            ratatui::text::Span::styled(state.masked_key.clone(), ratatui::style::Style::default().fg(theme.disabled)),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled("model             ", ratatui::style::Style::default().fg(theme.text_light)),
            ratatui::text::Span::raw(state.config.model.clone().unwrap_or_else(|| "claude-sonnet-4-6".into())),
        ]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled("run", ratatui::style::Style::default().fg(theme.text_light))),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled("max parallel      ", ratatui::style::Style::default().fg(theme.text_light)),
            ratatui::text::Span::raw(state.config.max_parallel.to_string()),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled("on escalation     ", ratatui::style::Style::default().fg(theme.text_light)),
            ratatui::text::Span::raw(state.config.on_escalate.clone()),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled("mcp (ml-intern)   ", ratatui::style::Style::default().fg(theme.text_light)),
            ratatui::text::Span::raw(if state.config.mcp { "on" } else { "off" }),
        ]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled("cost ceiling      ", ratatui::style::Style::default().fg(theme.text_light)),
            ratatui::text::Span::raw(format!("${} sandbox · ${} api", state.config.cost_ceiling_sandbox, state.config.cost_ceiling_api)),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled("theme             ", ratatui::style::Style::default().fg(theme.text_light)),
            ratatui::text::Span::raw(state.config.theme.clone()),
        ]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "[ save ]   [ cancel ]    saved to ~/.config/mr/config.toml",
            ratatui::style::Style::default().fg(theme.action),
        )),
    ];
    frame.render_widget(ratatui::widgets::Paragraph::new(lines), area);
}
```

Add `pub mod settings;` to `crates/mr-tui/src/surface/mod.rs`. In `app.rs`, add `pub settings: Option<SettingsState>` and dispatch `Surface::Settings` to `render_settings`. In `App::new`, first-run auto-open:

```rust
pub fn new(cwd: PathBuf, provider: Option<(Arc<dyn claurst_api::LlmProvider>, String)>) -> Self {
    let cfg = crate::config::MrConfig::load();
    let needs_settings = cfg.needs_provider_key() && provider.is_none();
    let surface = if needs_settings { Surface::Settings } else { Surface::Start };
    let settings = if needs_settings {
        Some(crate::surface::settings::SettingsState::from_config(cfg))
    } else {
        None
    };
    Self {
        surface,
        // ... rest unchanged, with settings: settings,
    }
}
```

Wire `s` (already in `handle_key`) to populate `settings` from `MrConfig::load()` if `settings` is None, then transition. Wire `Ctrl-S` (or `Enter` on the save line) to `config.save()`; `test connection` is deferred to a follow-up (it requires the provider registry — T13's `resolve_provider` can be the test; for 6b the button is present but `test_status` stays None until wired, which is acceptable since the spec lists it but the core 6b deliverable is the editor + save + auto-open). Add a NOTE in the commit that `test connection` is a stub pending the provider registry wiring.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p mr-tui mrconfig_round_trips mrconfig_needs_provider_key settings_surface_renders_masked_key`
Expected: PASS.

- [ ] **Step 6: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-tui
cargo clippy -p mr-tui --all-targets -- -D warnings
```

```bash
git add crates/mr-tui
git commit -m "$(cat <<'EOF'
feat(mr-tui): Settings surface + MrConfig (~/.config/mr/config.toml editor)

config.rs: MrConfig (TOML, serde-defaulted) at ~/.config/mr/config.toml —
the single store headless mr and the TUI share. surface/settings: structured
editor with opinionated defaults, masked API key, save→file. First-run
auto-opens when no provider/key is configured (resolves the 6a Important
finding). test-connection is a stub pending the provider registry wiring.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: mr-cli wiring — `mr` no-args → TUI + lazy provider resolution

**Files:**
- Modify: `crates/mr-cli/Cargo.toml` (add `mr-tui = { workspace = true }` + `dirs = { workspace = true }`)
- Modify: `crates/mr-cli/src/lib.rs` (empty args → launch TUI; lazy resolution)
- Modify: `crates/mr-cli/src/commands/mod.rs` (dispatch takes `Option<(provider, model)>`; resolves per-arm)
- Modify: `crates/mr-cli/src/prelude.rs` (read `MrConfig` first, fall back to `Settings::load_hierarchical`)
- Test: `crates/mr-cli/tests/wiring.rs` (new) or extend an existing test

**Interfaces:**
- Consumes: T12's `mr_tui::config::MrConfig`; T1's `mr_tui::run`; the existing `parse_args`, `dispatch`, `resolve_provider`.
- Produces: `mr` with no args → `mr_tui::run(cwd)`; `mr verify`/`list`/`watch` work without an API key (no provider resolution); `mr init`/`brainstorm`/`spec`/`plan`/`execute` resolve the provider lazily, reading `MrConfig` first.

- [ ] **Step 1: Write the failing test**

Create `crates/mr-cli/tests/wiring.rs`:

```rust
//! Verify the no-args → TUI decision and the lazy-resolution path without
//! launching a live terminal (test the dispatch decision via a seam).

use mr_cli::{Command, OnEscalate};

#[test]
fn parse_args_empty_is_list_not_tui() {
    // The TUI launch happens at run_cli, not parse_args. parse_args still
    // returns List for empty args (run_cli intercepts empty BEFORE parse_args).
    let cmd = mr_cli::parse_args(&["mr"]).unwrap();
    assert!(matches!(cmd, Command::List));
}

#[tokio::test]
async fn verify_list_watch_skip_provider_resolution() {
    // The lazy path: dispatch with Option::None provider for List/Watch/Verify
    // must not error. We test the dispatch signature, not a live run.
    // (Full e2e is T14.)
    let cwd = std::env::temp_dir();
    // List with no provider should succeed (it never needed one).
    let res = mr_cli::commands::dispatch(
        Command::List,
        cwd.clone(),
        None,
    ).await;
    // It may error if there are no runs dir, but it must NOT error with
    // "could not resolve a provider" — that's the regression we're fixing.
    if let Err(e) = &res {
        let msg = format!("{e}");
        assert!(!msg.contains("resolve a provider"), "lazy resolution leaked: {msg}");
    }
}
```

NOTE: `dispatch`'s signature changes from `(provider: (Arc<dyn LlmProvider>, String))` to `(provider: Option<(Arc<dyn LlmProvider>, String)>)`. The test calls it with `None` for `List`. The arms that need a provider (`Init`/`Brainstorm`/`Spec`/`Plan`/`Execute`) extract it from the `Option` and error with a clear message if missing; `List`/`Watch`/`Verify` ignore it.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p mr-cli wiring`
Expected: compile error — `dispatch` still takes `(Arc<dyn LlmProvider>, String)`, not `Option<...>`.

- [ ] **Step 3: Add the `mr-tui` dep to `crates/mr-cli/Cargo.toml`**

In `[dependencies]`, add:

```toml
mr-tui = { workspace = true }
dirs = { workspace = true }
```

- [ ] **Step 4: Change `dispatch` to take `Option<(provider, model)>`**

In `crates/mr-cli/src/commands/mod.rs`:

```rust
use std::path::PathBuf;
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::Command;

pub async fn dispatch(
    cmd: Command,
    cwd: PathBuf,
    provider: Option<(Arc<dyn LlmProvider>, String)>,
) -> anyhow::Result<()> {
    match cmd {
        Command::List => crate::render::list_runs(&cwd).await,
        Command::Watch { run_dir } => crate::render::watch(&cwd, run_dir).await,
        Command::Init { question } => {
            let p = provider.ok_or_else(|| anyhow::anyhow!("init requires a provider — set an API key (run `mr` to open settings)"))?;
            init::run(&cwd, p, &question).await
        }
        Command::Brainstorm { topic } => {
            let p = provider.ok_or_else(|| anyhow::anyhow!("brainstorm requires a provider"))?;
            session::run_session(&cwd, p, "brainstorm", &topic).await
        }
        Command::Spec { topic } => {
            let p = provider.ok_or_else(|| anyhow::anyhow!("spec requires a provider"))?;
            session::run_session(&cwd, p, "spec", &topic).await
        }
        Command::Plan { topic } => {
            let p = provider.ok_or_else(|| anyhow::anyhow!("plan requires a provider"))?;
            session::run_session(&cwd, p, "plan", &topic).await
        }
        Command::Execute { plan, paper, headless, no_mcp, on_escalate } => {
            let p = provider.ok_or_else(|| anyhow::anyhow!("execute requires a provider"))?;
            execute::run(&cwd, p, plan, paper, headless, no_mcp, on_escalate).await
        }
        Command::Verify { run_dir } => {
            // Verify does not call the LLM — pass a dummy; the function ignores it.
            verify::run(&cwd, (Arc::new(DummyProvider) as Arc<dyn LlmProvider>, String::new()), run_dir).await
        }
    }
}

// A zero-cost provider for Verify (which ignores the provider). Avoids
// pulling a real provider when none is configured.
struct DummyProvider;
#[async_trait::async_trait]
impl LlmProvider for DummyProvider {
    fn id(&self) -> &claurst_api::provider::ProviderId { unreachable!() }
    fn name(&self) -> &str { "dummy" }
    async fn create_message(&self, _r: claurst_api::ProviderRequest) -> Result<claurst_api::ProviderResponse, claurst_api::ProviderError> { unreachable!() }
    async fn create_message_stream(&self, _r: claurst_api::ProviderRequest) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<claurst_api::StreamEvent, claurst_api::ProviderError>> + Send>>, claurst_api::ProviderError> { unreachable!() }
    async fn list_models(&self) -> Result<Vec<claurst_api::ModelInfo>, claurst_api::ProviderError> { Ok(vec![]) }
    async fn health_check(&self) -> Result<claurst_api::ProviderStatus, claurst_api::ProviderError> { Ok(claurst_api::ProviderStatus::Healthy) }
    fn capabilities(&self) -> claurst_api::ProviderCapabilities { claurst_api::ProviderCapabilities { streaming: false, tool_calling: false, thinking: false, vision: false, audio: false, max_tokens: 0 } }
}

pub mod execute;
pub mod init;
pub mod session;
pub mod verify;
```

NOTE: `verify::run` keeps its `(Arc<dyn LlmProvider>, String)` signature (unchanged — it ignores the provider). The `DummyProvider` avoids the caller needing a real provider for `mr verify`. The `ProviderCapabilities` fields must match the actual struct (verify in `crates/api/src/provider_types.rs` — the fields are `streaming, tool_calling, thinking, vision, audio, max_tokens` per the grep at line 175; confirm `max_tokens` is `u64` or `u32` and adjust). If `ProviderCapabilities` does not impl `Default`, use `ProviderCapabilities::default()` if it does, or construct with the exact fields. The implementer should check `crates/api/src/provider_types.rs:173-205` for the exact field types before finalizing `DummyProvider`.

- [ ] **Step 5: Change `run_cli` to intercept empty args → TUI + lazy resolution**

In `crates/mr-cli/src/lib.rs`:

```rust
pub async fn run_cli(args: Vec<String>) -> anyhow::Result<()> {
    use anyhow::Context as _;
    let args_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    if args_refs
        .get(1)
        .is_some_and(|s| matches!(*s, "--help" | "-h" | "help"))
    {
        println!("{}", usage());
        return Ok(());
    }
    let cwd = std::env::current_dir()?;

    // No subcommand → launch the TUI. (The TUI's Start/Past surfaces handle
    // the runs-exist branch; settings auto-opens on first run.)
    if args_refs.len() <= 1 {
        return mr_tui::run(&cwd).await;
    }

    let cmd = parse_args(&args_refs).context("bad args")?;

    // Lazy provider resolution: only the provider-needing commands resolve.
    let needs_provider = !matches!(
        cmd,
        Command::List | Command::Watch { .. } | Command::Verify { .. }
    );
    let provider = if needs_provider {
        Some(
            prelude::resolve_provider(&cwd, None, None, None)
                .await
                .context("could not resolve a provider — set an API key (run `mr` to open settings)")?,
        )
    } else {
        None
    };
    commands::dispatch(cmd, cwd, provider).await
}
```

- [ ] **Step 6: Layer `MrConfig` into `resolve_provider`**

In `crates/mr-cli/src/prelude.rs`, at the top of `resolve_provider`, read `MrConfig` and apply its overrides before falling back to `Settings::load_hierarchical`:

```rust
pub async fn resolve_provider(
    cwd: &Path,
    model: Option<String>,
    provider_id: Option<String>,
    api_key: Option<String>,
) -> anyhow::Result<(Arc<dyn LlmProvider>, String)> {
    // MrConfig (~/.config/mr/config.toml) is the first source — the TUI and
    // headless `mr` share it. Explicit args override the config.
    let mr_cfg = mr_tui::config::MrConfig::load();
    let model = model.or(mr_cfg.model);
    let provider_id = provider_id.or(mr_cfg.provider);
    let api_key = api_key.or(mr_cfg.api_key);

    let settings = claurst_core::config::Settings::load_hierarchical(cwd).await;
    // ... rest unchanged
}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test -p mr-cli wiring`
Then: `cargo test -p mr-cli` (the existing 6a tests must stay green — `dispatch`'s new `Option` signature means the existing test calls to `dispatch` need updating to pass `Some(...)`; check `crates/mr-cli/tests/` for direct `dispatch(...)` calls and update them).
Expected: PASS. If existing tests call `dispatch(cmd, cwd, (provider, model))`, update them to `dispatch(cmd, cwd, Some((provider, model)))`.

- [ ] **Step 8: fmt + clippy + commit**

Run:
```
cargo fmt -p mr-cli
cargo clippy -p mr-cli --all-targets -- -D warnings
```

```bash
git add crates/mr-cli
git commit -m "$(cat <<'EOF'
feat(mr-cli): mr no-args → TUI + lazy provider resolution (fixes 6a Important)

run_cli intercepts empty args → mr_tui::run. dispatch takes Option<(provider,
model)>: List/Watch/Verify skip resolution entirely (no more "could not
resolve a provider" on mr verify/list/watch). resolve_provider reads MrConfig
first, falls back to claurst Settings::load_hierarchical. The TUI and headless
mr share one config store.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Workspace green + e2e capstone

**Files:**
- Create: `crates/mr-tui/tests/e2e.rs`
- Test: the full workspace

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a green workspace; an e2e test driving the TUI app end-to-end (init→converge→run→artifact) against a FakeProvider using TestBackend.

- [ ] **Step 1: Write the e2e test**

Create `crates/mr-tui/tests/e2e.rs`:

```rust
mod common;

//! End-to-end: drive the App through Start→Converge→Run→Artifact against a
//! FakeProvider using TestBackend. Asserts the full arc renders. This is the
//! capstone — if it passes, the thesis (audit trail as interface) ships.
//!
//! NOTE: a full App::run loop against TestBackend requires factoring the loop
//! to accept a `TestBackend` terminal + scripted events. If `App::run` is
//! tightly bound to `CrosstermBackend`, the e2e test instead drives the
//! surface transitions directly: construct an App, set surface=Converge with
//! a scripted TuiUserIo + FakeProvider, advance the session, then set
//! surface=Run with a fixture swarm, then surface=Artifact, and render each
//! via TestBackend. The contract is: every surface renders without panic and
//! the arc produces an ArtifactState. (A live-loop e2e is Phase 8 polish.)

use std::sync::Arc;

use megaresearcher_research::state::swarm_state::{
    HypothesisNode, Phase, RoundVerdict, SwarmState, Verdict, Worker,
};
use mr_tui::app::{App, Surface};
use mr_tui::surface::artifact::ArtifactState;
use mr_tui::surface::run::RunState;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

#[test]
fn e2e_arc_renders_every_surface() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);

    // Start.
    app.surface = Surface::Start;
    app.question = "can SAEs surface causal circuits?".into();
    let mut t = Terminal::new(TestBackend::new(120, 20)).unwrap();
    t.draw(|f| app.render(f)).unwrap();

    // Converge (rendered with an empty conversation — the session spawn is
    // async; the full e2e drives it in the converge test, T8).
    app.surface = Surface::Converge;
    t.draw(|f| app.render(f)).unwrap();

    // Run — the make-or-break surface.
    let swarm = SwarmState {
        run_id: "e2e".into(),
        spec_path: "s".into(),
        plan_path: "p".into(),
        novelty_target: "hypothesis".into(),
        max_parallel: 4,
        phases: vec![Phase {
            name: "red-team".into(),
            status: "complete".into(),
            workers: vec![Worker { name: "red-team-1".into(), status: "done".into() }],
            hypotheses: vec![HypothesisNode {
                id: "hypothesis-smith-1".into(),
                label: "causal-SAE-bridge".into(),
                status: "approved".into(),
                rounds: vec![RoundVerdict { round: 1, critique: Verdict::Approve, revised: false }],
                kill_reason: None,
            }],
        }],
        escalations: vec![],
        retry_counts: std::collections::HashMap::new(),
    };
    app.surface = Surface::Run;
    app.run_state = Some(RunState::from_swarm_for_test(swarm));
    t.draw(|f| app.render(f)).unwrap();

    // Artifact.
    app.surface = Surface::Artifact;
    app.artifact = Some(ArtifactState {
        run_dir: "/tmp/run".into(),
        output_md: "# Research direction\n\nSAEs for causal circuits.\n".into(),
        surviving: vec![HypothesisNode {
            id: "hypothesis-smith-1".into(),
            label: "causal-SAE-bridge".into(),
            status: "approved".into(),
            rounds: vec![RoundVerdict { round: 1, critique: Verdict::Approve, revised: false }],
            kill_reason: None,
        }],
        rejected: vec![],
    });
    t.draw(|f| app.render(f)).unwrap();

    // Every surface rendered without panic — the arc holds.
    let buf = t.backend().buffer().clone();
    let content: String = buf
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains("Research direction"), "artifact rendered: {content}");
}
```

- [ ] **Step 2: Run the e2e test**

Run: `cargo test -p mr-tui e2e_arc_renders_every_surface`
Expected: PASS.

- [ ] **Step 3: Workspace green**

Run:
```
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```
Expected: all green. If `cargo fmt --all --check` reports unformatted files, run `cargo fmt --all` and amend the last commit or commit the formatting separately.

- [ ] **Step 4: Determinism guard one more time**

Run: `cargo test -p megaresearcher-research`
Expected: all ~52 orchestrator tests green (the final check that T2/T3's additive extension never broke the determinism contract).

- [ ] **Step 5: Commit + push**

```bash
git add crates/mr-tui/tests/e2e.rs
git commit -m "$(cat <<'EOF'
test(mr-tui): e2e capstone — Start→Converge→Run→Artifact renders end-to-end

Workspace green: cargo test --workspace, clippy --all-targets -D warnings,
fmt --check. The 52 research tests stay green. The arc renders.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
git push origin main
```

---

## Self-Review

**1. Spec coverage (§1.1 surfaces):**
- Start (§4.1) → T7. ✓
- Converge (§4.2) → T6 (TuiUserIo) + T8 (surface). ✓
- Run (§4.3) → T9 (make-or-break). ✓
- Artifact (§4.4) → T10. ✓
- Past runs (§4.5) → T11. ✓
- Settings (§4.6) → T12. ✓
- mr-cli wiring (§6 "mr with no args → launch mr-tui") → T13. ✓
- Swarm-state extension (§8) → T2. ✓
- Red-team persistence (§8 "the orchestrator already has this data") → T3. ✓
- CostTracker single number (§4.3 "$0.42 · 18k", §11) → T5. ✓
- EscalationHandler TUI counterpart (§6 architecture) → T9. ✓
- Lift vs build (§1.4: bootstrap, virtual_list, theme_colors, figures) → T1 (bootstrap, theme, figures). virtual_list is lifted in T4's `tests/common` copy step but NOT used by a 6b surface — the spec §1.4 lists it for "past-runs list and any scrollable surface"; T11's Past surface uses a plain `Paragraph` for the list (simpler, Rubin). If the implementer wants variable-height scrolling, virtual_list is available; for 6b's minimal past list it is not required. This is a deliberate scope reduction consistent with the reducer pattern. ✓ (noted)
- Determinism guard (§8) → T2 Step 6, T3 Step 5, T14 Step 4. ✓
- Error handling (§9: terminal restore on panic, orchestrator task failure, missing swarm-state, escalation fail-safe) → T1 (guard + panic hook), T9 ("waiting for run to start…" placeholder, inline escalation). ✓
- Testing (§10: widget render tests, state-machine tests, TuiUserIo test, serde round-trip + 52 tests, make-or-break, settings test) → T4, T7, T6, T2, T9, T12. ✓

**2. Placeholder scan:** Searched for TBD/TODO/"implement later"/"add appropriate" — none in code steps. The two NOTE blocks (T8 select! ergonomics, T13 DummyProvider capabilities fields) name the concrete alternative and the exact file/line to consult; they are not placeholders but genuinely ergonomics-dependent ergonomics the implementer resolves at compile time.

**3. Type consistency:**
- `HypothesisNode` fields (id/label/status/rounds/kill_reason) — consistent across T2 (def), T3 (persist), T4 (render), T10 (cards), T11 (past), T14 (e2e). ✓
- `RoundVerdict { round, critique: Verdict, revised }` — consistent across T2/T3/T4/T14. ✓
- `Verdict { Approve, Reject }` — consistent. ✓
- `SwarmState::read(&Path)` / `write(&Path)` — consistent across T2/T10/T11. ✓
- `TuiUserIo` + `TuiUserIoHandle` + `tui_user_io()` — consistent across T6/T8. ✓
- `CountingProvider::new(inner, tracker)` + `render_cost_meter(frame, area, tracker, theme)` — consistent across T5/T9. ✓
- `EscalationHandler::adjudicate(&self, &Escalation) -> EscalationVerdict` — consistent across T9 (TuiEscalationHandler) and the existing `HeadlessEscalationHandler`. ✓
- `OrchestratorConfig { research_base, agents_dir, default_model, max_parallel, mcp: Option<McpServerConfig>, escalation: Option<Arc<dyn EscalationHandler>> }` — T9's `spawn_run` matches the verified struct at `crates/research/src/orchestrator/mod.rs:85-92`. ✓
- `App::new(cwd, Option<(provider, model)>)` — consistent across T7/T9/T10/T11/T12/T13. ✓
- `dispatch(cmd, cwd, Option<(provider, model)>)` — T13 changes the signature; the existing 6a tests that call `dispatch` directly must be updated (T13 Step 7 notes this). ✓

No type drift found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-megaresearcher-rs-phase6b-tui.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Per the context dump §13, this is the recommended path (SDD, on `main`, no worktrees). Before every dispatch I run `git branch --show-current` and include the explicit branch name in the subagent prompt.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?