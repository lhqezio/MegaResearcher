# MegaResearcher v1 — Phase 6b TUI Design

**Status:** design, awaiting implementation plan
**Date:** 2026-06-28
**Parent design:** `docs/superpowers/specs/2026-06-26-megaresearcher-rs-design.md` (§5 UX, §13 phasing)
**Amends the parent:** §1/§3/§5 say "lift claurst's `tui` crate, research-themed." Inspection of the actual workspace shows `claurst-tui` is a 45,579-LOC chat REPL (`app.rs` 7,127 LOC) built around chat semantics — the opposite product shape from this design's "audit trail is the interface" tree. This phase **builds a fresh, small research TUI and lifts only generic primitives** (terminal bootstrap, `virtual_list`, `theme_colors`) from `claurst-tui`. The parent's "lift the tui crate" line is overridden; see §1.4.

## 0. Purpose

Phase 6a shipped the headless `mr` CLI front-half (guided sessions, orchestrator execute, verify). Phase 6b ships the **interactive research TUI** — the surface a human stares at. This is where the product thesis either lands or doesn't: **the audit trail is the interface.** Most research tools show only what survived; this tool shows the evolution of ideas — hypotheses born, attacked, revised, killed or surviving — as a tree that grows in front of the user, with killed hypotheses staying on screen, dimmed, each with its one-line kill reason. The rejected ideas and the reasoning that killed them are the point.

The verb, in plain words: **watch a room of researchers tear your question apart and hand you what survived** — not "runs a six-phase swarm with adversarial red-team critique." The research-direction document is the payoff; the run that produced it is the proof.

## 1. Scope

### 1.1 In scope for 6b (all four surfaces)

1. **Start** — one input, a ghosted example. No wizard, no target picker, no onboarding tour.
2. **Converge** — brainstorm → spec → plan as one flowing surface: the model asks 2–3 sharp questions, the user answers, it converges; spec and plan cards appear with in-place `[✓]` approval. Reuses 6a's `drive_session` engine.
3. **Run** — the tree that grows: phases appear as the previous completes; each worker a node with status (running/done/escalated/killed); the red-team loop animates on hypothesis nodes (`smith → critique → revise ↻`, `round N/3`); killed hypotheses dim with their one-line kill reason; one cost number top-right; an escalation strip that appears only when an escalation exists and is adjudicated inline.
4. **Artifact** — on synthesis completion, the tree slides to a side rail as provenance and the research direction takes the screen: rendered `output.md`, surviving hypotheses as expandable cards (mechanism / predicted outcome / falsification / experimental design), rejected hypotheses folded at the bottom with their lessons.
5. **Past runs** — `mr` with no args when runs exist: a list by date + topic, each with its headline surviving hypothesis; one tap reopens artifact + tree.

### 1.2 Deferred to Phase 8

- HTML export (the one thing that leaves the terminal — "here's the reasoning, not the answer"). In 6b, shareability lives in the screenshot-worthy run screen itself.
- The `verify` phase as a distinct surface (verification is a final tree node, red/green — expand if you care).
- Expandable-cards polish beyond the 6b minimal-card rendering (6b ships working expandable cards; visual refinement is Phase 8).

### 1.3 Non-goals (explicit, to prevent scope creep)

- No settings screen. Config lives in a file; `max_parallel`/provider = defaults + advanced flags.
- No token-usage dashboard with charts. One live number on the run screen.
- No modal celebration when a phase completes. The tree growing is the feedback.
- No onboarding tour. The first screen is the input.
- No defensive UI for states the design should not permit (no empty escalation box, no empty-state artwork beyond the start screen and the past-runs list itself).
- No chat REPL. The converge conversation is bounded (3 exchanges) and converges the question; it is not free chat.
- No lifting of `claurst-tui`'s `app.rs`, `prompt_input`, `render`, or any chat-coupled module.

### 1.4 Spec amendment: lift vs. build

The parent design (§1, §3, §5) says "lift claurst's `tui` crate, research-themed." The actual `claurst-tui` crate is a 45,579-LOC chat REPL: `app.rs` (7,127 LOC) holds `DisplayMessage`, `ToolUseBlock`, `TurnMetadata`, slash-command intercepts, agent modes, speech mode, onboarding, plugin views — a chat mental model. Theming a chat REPL into a tree is a gut job that drags the wrong mental model along. Under Rubin (subtract) and Jobs (one default, no onboarding tour), Phase 6b **builds a fresh, small research TUI** on `ratatui` 0.29 + `crossterm` 0.29 and **lifts only generic primitives** from `claurst-tui`:

| Lifted from `claurst-tui` | LOC | Why it earns its place |
|---|---|---|
| Terminal bootstrap (`enable_raw_mode` / `EnterAlternateScreen` / `CrosstermBackend`) | ~30 | Generic, not chat-coupled; rewriting it buys nothing. |
| `virtual_list` (scrollable list widget) | 462 | Reusable for the past-runs list and any scrollable surface. |
| `theme_colors` (palette) | 212 | Adapted to the research palette (alive / dim-killed / running / escalation-warn). |
| `figures` (box-drawing glyphs) | 33 | Tree rendering. |

Everything else in `claurst-tui` is **not lifted**. The parent's "lift the tui crate" line is superseded by this section.

## 2. The user, and the moment

The user is a researcher (ML/AI, or a research-minded engineer/analyst/PM) with a question they can't answer alone. They want a swarm to survey prior art, find gaps, forge hypotheses, try to kill them, design the falsification, and hand back a direction. They have been burned by LLM tools that return a confident answer with no reasoning — plausible-but-wrong is their specific fear. That fear is the entire product. The audit trail exists because trust is the thing being sold.

Three jobs-to-be-done:

1. *"Find me something worth researching in ⟨area⟩."* — gap-finding run.
2. *"Pressure-test this hypothesis before I sink months into it."* — hypothesis-target run (the red-team loop is the product here).
3. *"Give me a defensible direction with the reasoning visible — not just the answer."*

## 3. The signature (the 3%)

Take a familiar shape — a terminal tree/dashboard — and shift it 3%: **killed hypotheses stay on screen, dimmed, each with its one-line kill reason.** Everything else is borrowed shape (tree, list, scrollable doc, side rail). The greyed kill is the brand moment — the thing that makes the run screenshot-worthy and the direction trustworthy. The red-team cycling row (`smith → critique → revise ↻`, `round N/3`) is the second signature: the animation *is* the documentation of the critique discipline.

## 4. The four surfaces — what's in view

### 4.1 Start

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      What do you want to know?                               │
│              ┌──────────────────────────────────────────────────────────┐    │
│              │ e.g. can sparse autoencoders surface causal circuits in  │    │
│              │     transformer attention?                                │    │
│              └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

One input, a ghosted example. No wizard, no target picker, no onboarding tour. Type, enter. (Jobs: first 60 seconds; one default, no options.)

### 4.2 Converge — one flowing surface, not a wizard

The conversation unfolds inline beneath the input. The model asks 2–3 sharp questions (novelty target, modalities, date range), the user answers, it converges — three exchanges, max. Then the spec card and plan card appear; the user approves each in place. This is the only place a chat exchange is justified — it is bounded and it converges the question, not free chat. (Rubin: not a 5-screen wizard; one flowing surface.)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ✎ can SAEs surface causal circuits in transformer attention?                 │
│   ┌ converge the question ──────────────────────────────────────────────┐    │
│   │ q  gap to research, or a hypothesis to pressure-test?                │    │
│   │ >  gap to research                                                   │    │
│   │ q  modalities / domain?                                              │    │
│   │ >  interpretability, SAEs, circuits                                  │    │
│   │ q  date range + constraints?                                         │    │
│   │ >  2024–2026, transformer attention only                            │    │
│   │ → converged.                                                         │    │
│   └──────────────────────────────────────────────────────────────────────┘    │
│   spec  ▸  gap-finding · SAEs + circuits + interpretability · 2024–26  [✓]   │
│   plan  ▸  3 scouts → gap-finder → synthesist                          [✓]   │
│                       [ start the run ▸ ]                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

Reuses 6a's `GuidedSession` + approval gates, rendered in the TUI instead of stdin. Same `drive_session` engine, new view (see §6).

### 4.3 Run — where the user lives (the make-or-break)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ✎ "can SAEs surface causal circuits in transformer attention?"  $0.42 · 18k  │
│   spec: gap-finding · 3 modalities · 2024–26              [ expand ▸ ]       │
├──────────────────────────────────────────────────────────────────────────────┤
│  Phase 1 · scouts                                              done          │
│  ├ literature-scout · transformers+SAE          ✓                            │
│  ├ literature-scout · circuits                 ✓                             │
│  └ literature-scout · interpretability         ✓                             │
│  Phase 2 · gaps                                                done          │
│  └ gap-finder                                   ✓                            │
│  Phase 3 · hypotheses                                          done          │
│  ├ H1  causal-SAE-bridge                        ✓                            │
│  ├ H2  activation-patching                     ✓                             │
│  └ H3  logit-lens-circuits                     ✓                             │
│  Phase 4 · red-team                              round 2/3                   │
│  ├ H1  smith ✓  critique ✗  revise ↻                                       │
│  ├ H2  smith ✓  critique ✗  KILLED                                         │
│  │      ╴ "mechanism contradicts Marks et al. 2025 — effect is abl, not causal"│
│  └ H3  smith ✓  critique ✓  APPROVE                                        │
│  Phase 5 · designs                              waiting                       │
│  Phase 6 · synthesis                            waiting                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ ⚠ escalation · gap-finder: "no gaps in sub-topic 3"      [ continue ▸  fail ]│
└──────────────────────────────────────────────────────────────────────────────┘
```

Element by element, each justifying itself:

- **Top line — the question, always.** One line, the spec collapsible beneath it. The run is long; the question anchors it.
- **The tree — the whole canvas.** Phases appear as the previous completes; the tree *grows*. The growth is the feedback that a phase finished — no celebration, no toast. (Jobs: animation as explanation. Rubin: no separate progress bar.)
- **The red-team row — the signature.** `smith ✓ → critique ✗ → revise ↻` with `round N/3`. The cycling indicator is the documentation of the critique discipline — you watch it try, fail, revise, try again.
- **The greyed kill — the thesis on a single line.** H2 dims, marked KILLED, with the one-line kill reason indented beneath. The greying teaches "rejected ideas are preserved, not erased." This is the thing no other research tool shows.
- **Cost — one number, top-right.** `$0.42 · 18k`. Not a chart, not a dashboard. (Rubin: one live number.)
- **Escalation strip — bottom, only when it exists.** When something needs the user, the run pauses and the strip appears; the user adjudicates in place (`continue` / `fail`), never a full-screen modal. When there is no escalation, the strip is not drawn — no empty "(none)" box. (Jobs: modal interruption is bad UI; no defensive UI for states that should not persist.)

Not in view: no settings, no token chart, no phase-complete modal, no separate verification view (verification is a final tree node, red/green — expand if you care).

### 4.4 Artifact — the payoff

When synthesis completes, the tree slides to a side rail as provenance and the direction takes the screen. Surviving hypotheses as expandable cards; the rejected section folds at the bottom with its lessons — the killed ideas still here, still teaching.

```
┌─ ✎ SAEs for causal circuits ── provenance ▸ ─────────────────────────────────┐
│  # Research direction                                                        │
│  Surviving direction: …                                                      │
│  ▸ H1  causal-SAE-bridge                                                     │
│       mechanism · predicted outcome · falsification · experimental design    │
│  ▸ H3  logit-lens-circuits                                                   │
│       mechanism · predicted outcome · falsification · experimental design    │
│  ▾ rejected (2) — the lessons                                                │
│     · H2 activation-patching — "effect is ablation, not causal"              │
│     · …                                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Past runs

`mr` with no args when runs exist → a list by date + topic, each with its headline surviving hypothesis. One tap reopens artifact + tree.

```
┌─ past runs ──────────────────────────────────────────────────────────────────┐
│  2026-06-28  SAEs for causal circuits       → H1 causal-SAE-bridge    open ▸  │
│  2026-06-25  retrieval-augmented agents     → H2 sparse-memory-rag    open ▸  │
│  2026-06-21  eval contamination in benches  → (gap-finding)          open ▸  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 5. Interaction model (Jobs)

- **One default, no options** on the start screen. The novelty target is conversed, not picked from a dropdown.
- **Animation as explanation**: the tree growing teaches the phase order; the red-team `↻` teaches the critique loop; the greying teaches "rejected ideas are preserved."
- **No modal takeovers.** Escalations adjudicate inline. Phase completion = the tree grows, nothing else.
- **Direct manipulation.** Tap a node → expand its output. Tap a surviving hypothesis → expand its cards. Tap a kill reason → read the red-team critique that produced it.
- **It just works.** No defensive UI for states the design should not permit.

## 6. Architecture — `crates/mr-tui` (new, small)

A view over `swarm-state.yaml` (orchestrator stays the source of truth; the TUI never owns run state). ~2–4k LOC.

```
crates/mr-tui/src/
  app.rs            — surface state machine (Start/Converge/Run/Artifact/Past) + event loop + terminal bootstrap
  io.rs             — TuiUserIo: implements research::phases::UserIo by routing print→conversation buffer, read_line→inline field
  escalation.rs     — TuiEscalationHandler: surfaces escalations into the run strip, blocks on inline adjudicate (TUI counterpart to 6a's HeadlessEscalationHandler)
  surface/
    start.rs        — Start + converge (drives drive_session via TuiUserIo; renders inline convo + spec/plan [✓] cards)
    run.rs          — Run: renders SwarmState as the growing tree + cost + escalation strip; watches swarm-state.yaml
    artifact.rs     — Artifact: renders output.md + surviving expandable cards + rejected fold
    past.rs         — Past runs: lists docs/research/runs/
  widget/
    tree.rs         — phases→workers→hypothesis sub-nodes (red-team verdict sequence + greyed kill)  ← the signature
    inline_chat.rs  — the bounded 3-exchange converge widget
    cards.rs        — expandable hypothesis cards
  theme.rs          — adapted from claurst theme_colors (alive / dim-killed / running / escalation-warn)
  guard.rs          — terminal-restore on drop + panic hook (raw-mode cleanup)
```

`mr-cli` gains: `mr` with no args → launch `mr-tui`; subcommands (`execute`, `verify`, `list`, `watch`) → headless as today. Mirrors claurst's `cli`/`tui` split without inheriting `claurst-tui`'s code.

### 6.1 The DRY seam (why 6a pays off here)

The converge surface reuses 6a's `drive_session(&mut session, io, gates, approve_words)` unchanged — only the `io` changes. `TuiUserIo` implements `UserIo` by writing `print` into the conversation buffer and resolving `read_line` from an inline input field. The approval gates become the on-screen `[✓]` affordances. Same engine, new view — no second guided-session implementation.

## 7. Data flow

- **Converge**: `drive_session` + `TuiUserIo`, gates `[spec, plan]`, flow bodies via `load_embedded`.
- **Run**: spawn the orchestrator on a tokio task (as 6a's `execute::run_with` does); the run surface watches `swarm-state.yaml` (250ms poll, matching 6a — no new dep) and renders diffs. Escalations route through `TuiEscalationHandler` → the inline strip, blocking on `continue`/`fail`.
- **Artifact**: on synthesis done, read `output.md` + the hypothesis nodes from state → cards + rejected fold.
- **Past runs**: enumerate `runs/`, read each `output.md` headline + surviving hypothesis.

## 8. The swarm-state extension (the one contract touch — additive)

`swarm-state.yaml` today carries `phases[name, status, workers[name, status]]`, `escalations`, `retry_counts`. The signature view needs data this does not yet carry:

```rust
// new, optional, serde-defaulted empty → existing 52 orchestrator tests unaffected
pub struct HypothesisNode {
    pub id: String,                  // "H1"
    pub label: String,               // "causal-SAE-bridge"
    pub status: String,              // alive | killed | approved
    pub rounds: Vec<RoundVerdict>,   // red-team sub-iterations (the ↻ animation data)
    pub kill_reason: Option<String>, // the greyed one-liner
}
pub struct RoundVerdict {
    pub round: u32,
    pub critique: Verdict,           // approve | reject
    pub revised: bool,
}
```

`Phase` gains `pub hypotheses: Vec<HypothesisNode>` (default empty; only Phase 3/4 populate it). The orchestrator already has this data at runtime — the red-team loop parses APPROVE/REJECT/KILL and knows the round + reason; it just does not persist it. The work is persisting it.

**Determinism guard:** the 52 orchestrator tests assert on `phases[].status` and `workers[].status`. Additive optional fields (serde-defaulted empty) must leave them green. The implementation plan must run the 52-test suite immediately after the `SwarmState` extension and assert zero changes before proceeding.

## 9. Error handling

- **Terminal restore on panic** — `guard.rs` restores raw mode on drop + a panic hook. A TUI that wrecks the terminal on a panic is unusable.
- **Orchestrator task failure** → the run node goes red with the error one-liner; the TUI stays up.
- **Missing swarm-state** → "waiting for run to start…" placeholder, not a crash.
- **Escalation-handler error** → fail-safe (fail the worker), shown in the strip.

## 10. Testing

- **Widget render tests** with ratatui `TestBackend`: the tree renders a killed hypothesis dim with its kill reason; the red-team row shows `round N/3`; the escalation strip appears only when an escalation exists (asserts the no-defensive-UI rule).
- **State-machine tests**: surface transitions Start→Converge→Run→Artifact; gate approval advances.
- **`TuiUserIo` test**: `drive_session` with 6a's FakeProvider + a scripted conversation buffer → spec/plan cards appear, gates pass.
- **SwarmState serde round-trip** + the 52 orchestrator tests still green (the determinism guard).
- **The make-or-break integration test**: a fake-provider run (6a's `run_turns`, including a killed hypothesis with a kill reason) → drive the TUI app against the produced `swarm-state.yaml` → assert the run surface renders the tree with the greyed kill. This single test encodes the thesis.

## 11. Open questions deferred to the implementation plan

- File-watch mechanism: 250ms poll (matches 6a, no new dep) vs. the `notify` crate (snappier growth, new dep). Recommendation: poll in 6b; revisit if the growth feels sluggish.
- Markdown rendering for the artifact screen: hand-rolled vs. a crate (e.g. `tui-markdown`/`ratatui-markdown` or a lightweight parser). Recommendation: minimal hand-rolled rendering for 6b (headings, paragraphs, lists, bold); defer a full markdown crate to Phase 8 polish.
- Exact hypothesis-card field extraction: parse `output.md` sections vs. read structured hypothesis nodes from `swarm-state.yaml`. Recommendation: read structured nodes from state (the §8 extension makes this clean); fall back to `output.md` section parsing for fields the state does not carry.
- Cost-number source: aggregate from per-worker usage in `swarm-state.yaml` (if persisted) vs. a separate cost tracker. Recommendation: a `CostTracker` shared between the orchestrator and the TUI (mirrors claurst's `CostTracker`), surfaced as the single top-right number.