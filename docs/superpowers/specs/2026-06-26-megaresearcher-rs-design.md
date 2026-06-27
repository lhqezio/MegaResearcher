# MegaResearcher v1 — Rust Harness Design

**Status:** design, awaiting implementation plan
**Date:** 2026-06-26
**Supersedes:** the Claude Code plugin (current `main`), retained during the build as the port reference and deleted at feature parity.
**License of the new harness:** GPL-3.0 (derived from lifted claurst code). The prompt assets (former `agents/*.md` + skill bodies) remain Apache-2.0; see §2.

## 0. Purpose and one-paragraph summary

MegaResearcher today is a Claude Code plugin: a research swarm whose orchestrator is a *skill prompt* that drives six leaf-worker subagents via Claude Code's `Task` tool. v1 rebuilds this as a **standalone Rust research agent** — a new CLI/TUI tool, hyper-tailored for research — in which the orchestrator becomes **deterministic Rust control flow** that calls the LLM provider directly per worker. The same six workers, the same three-artifact contract, the same `swarm-state.yaml`, the same red-team loop and escalation discipline are preserved; only the runtime changes from "Claude Code executing a skill" to "Rust executing the orchestrator."

The harness is built by **forking and trimming claurst** (an open-source Rust reimplementation of Claude Code, GPL-3.0) and studying the reconstructed Claude Code TypeScript source (`yasasbanukaofficial/claude-code`) to replicate the agent loop, subagent dispatch, tool-calling, and MCP-client mechanics faithfully.

## 1. Decisions (locked during brainstorm)

| Decision | Choice |
|---|---|
| License / how to use claurst | Fork-and-trim claurst; harness is GPL-3.0. Existing Apache plugin stays untouched and separate during the build, removed at parity. |
| Orchestrator | Deterministic Rust swarm runner; workers are leaf LLM calls. |
| Discipline layer (superpowers) | Inlined into worker prompts. No runtime skills concept. |
| Paper chain + runners | Ported to Rust (single binary; Vercel Sandbox as a Rust HTTP client). |
| Interface | Interactive research TUI (lift claurst's `tui` crate), with a headless CLI fallback. |
| Where it lives | Replaces the current repo as MegaResearcher v1. |
| ml-intern MCP server | Stays a Python stdio subprocess the Rust MCP client drives (porting ml-intern itself is out of scope). |

Four judgment calls confirmed from the Claude Code source study (§8):

1. Workers get **no Bash** — scoped Read/Write + MCP research tools only (stricter than Claude Code).
2. Path-scoping is a **hard pre-permission check inside the file tools**, not a deny rule or sandbox.
3. **Plugins skipped entirely**; a flat `flows/` dir resolved against `$MR_ROOT` covers the small slice we need.
4. **uv venv pre-warm** before first MCP connect — the one place we are smarter than Claude Code.

## 2. Repo and license

- The repo becomes the Rust Cargo workspace. Root `LICENSE` = GPL-3.0 (inherited from the claurst lift). A `NOTICE` records that `assets/prompts/**` (former `agents/*.md`, skill bodies, and inlined discipline) remain Apache-2.0. Apache code flowing into a GPL derivative is permitted; the exception is documented so prompt provenance is traceable.
- The old plugin scaffolding (`.claude-plugin/`, `commands/`, `hooks/`, `mcp/`, `lib/`, `skills/`, `tools/ml-intern`) is **kept during the build as the port reference**, then **deleted at feature parity**. Git history preserves it.
- `ml-intern` stays a Python stdio subprocess (moved to `python/`). Porting ml-intern (litellm, vendored snapshot, full agent runtime) to Rust is out of scope and not how this works today. Only the harness-side MCP *client* is Rust.

## 3. Workspace layout

```
megaresearcher/                  Cargo workspace, GPL-3.0
├── Cargo.toml
├── LICENSE / NOTICE
├── crates/
│   ├── core/        ← lifted from claurst: agent loop, message types, streaming
│   ├── tools/       ← lifted: tool-call dispatch + schema
│   ├── mcp/         ← lifted: MCP stdio client (drives ml-intern subprocess)
│   ├── api/         ← lifted: provider abstraction (Anthropic default; others optional)
│   ├── tui/         ← lifted + trimmed: ratatui UI, research-themed
│   ├── cli/         ← lifted: arg parsing + headless mode
│   └── research/    ← NEW: the MegaResearcher layer
├── assets/prompts/ ← Apache-2.0: former agents/*.md + skill bodies + inlined discipline + flow bodies
├── python/         ← transient: ml-intern MCP server (driven as subprocess)
└── docs/research/  ← run outputs (identical layout to today)
```

Trimmed from claurst: `buddy`, `acp` (editor protocol), `bridge`, voice/mic, `/share` gist, Free Mode routing beyond the providers we keep. Upstream claurst fixes stay pullable via a `claurst` git remote.

## 4. Components

| Module | Responsibility | Origin |
|---|---|---|
| `research::orchestrator` | The deterministic swarm runner. Pre-flight, run-id + scaffold, `swarm-state.yaml` read/write, wave dispatch with a `max_parallel` semaphore, the red-team critique loop (parse verdict → APPROVE/REJECT/KILL branch → retry≤3 → escalate), eval-designer fan-out, synthesist single dispatch, per-worker verification gate, escalation surfacing. This is `executing-research-plan` turned into code. | new |
| `research::worker` | The leaf-LLM call primitive — see §6. | new (mechanics grounded in CC source) |
| `research::worker_tools` | Scoped Read/Write tools given to each worker, limited to its own output dir — see §7. | new |
| `research::phases` | Front-half guided sessions: `brainstorm`, `spec`, `plan`, `verify`. LLM-driven conversational sessions in the TUI; the flow body is the guiding prompt; Rust enforces approval gates and writes artifacts. | new (flow bodies → assets) |
| `research::paper_chain` | Port of `lib/paper_chain`: `protocol_parser`, `verdict`, `regression`, `scaffold`, `finalize`, `preflight`, `experiment`, `sandbox` (Vercel Sandbox HTTP client), + runner registry + 5 skeleton runners. Pure Rust. | port |
| `research::state` | `swarm-state.yaml` schema, run-tree management (`docs/research/runs/<run-id>/`), consolidations (`bibliography.md`, `gaps.md`), spec-latest symlink. | port |
| `research::export` | HTML renderer: a run + its kill-reason audit trail → single self-contained HTML. | new |
| `research::discipline` | The doom-loop detector and the correction-injection trait — see §9. | new (hardcoded; CC hook runtime skipped) |
| `core / tools / mcp / api / tui / cli` | Lifted from claurst, trimmed. | lifted |

## 5. UX — the interface

The signature idea: **the audit trail is the interface.** Most research tools show only what survived. This tool shows the evolution of ideas — hypotheses born, attacked, revised, killed or surviving — as a tree that grows in front of the user. Rejected hypotheses stay on screen, greyed, with their one-line kill reason, because the rejected ideas and the reasoning that killed them are the point. Everything else is a borrowed shape (a terminal agent like claurst); this is the thing that is ours.

The verb, in plain words: "watch a room of researchers tear your question apart and hand you what survived" — not "runs a six-phase swarm with adversarial red-team critique." The research-direction document is the payoff; the run that produced it is the proof.

### Four screens

1. **Start (first 60 seconds).** One input — *What do you want to know?* — with a ghosted example. No wizard, no target picker, no onboarding tour. Type a question, hit enter. Brainstorm runs inline beneath the input as a short conversation: the model asks 2–3 sharp questions (novelty target, modalities, constraints), the user answers, it converges. Three exchanges, max. That is the whole onboarding.

2. **The run — where the user lives.** The home screen once execution starts:
   - One line at top: the question + the converged spec (collapsible).
   - The run tree: Phase 1 scouts → Phase 2 gaps → Phase 3 hypotheses → Phase 4 red-team → Phase 5 designs → Phase 6 synthesis. Each worker is a node with a status (running / done / escalated / killed). Phases appear as the previous one completes — the tree grows.
   - The red-team loop animates on the hypothesis node: smith → critique → revise → re-critique, visible as a cycling animation. The animation is the documentation of the discipline. When a hypothesis is killed, the node greys out with the kill reason — the greying teaches "rejected ideas are preserved, not erased."
   - A persistent cost/token meter (one number, not a chart) and a live escalations queue on the side. When something needs the user, the run pauses and pings there; the user adjudicates in place, never in a full-screen modal.
   - Killed hypotheses stay in the tree. The audit trail is this screen, not a separate doc.

3. **The artifact.** When synthesis completes, the tree slides left and the research direction takes the whole screen — rendered, scrollable, surviving hypotheses as expandable cards (mechanism / predicted outcome / falsification / experimental design). The rejected-hypotheses section folds at the bottom with its lessons. The thing the user came for becomes the screen; the tree sits beside it as provenance. `--paper` continues the same view: experiments appear as nodes (cost + pass/fail), the draft→review→revise loop animates like the red-team loop, the final `paper.md` lands center-stage the same way.

4. **Past runs.** `mr` with no args lists runs by date + topic, each with its headline surviving hypothesis; one tap reopens artifact + tree. The spec-latest symlink becomes "the most recent answer to this question" in a list.

### One thing that leaves the terminal

A tool is not viral, but a research artifact is shareable. One share format, done well: every run exports to a single self-contained **HTML** — the artifact plus the collapsed kill-reason audit trail — that a collaborator opens without installing anything. "Here's the reasoning, not just the answer." Not a gist, not a dashboard link. This replaces claurst's `/share` gist with something that matters for research.

### Cuts (Rubin)

- No settings screen. Config lives in a file. `max_parallel`, provider — fixed defaults, advanced flags only.
- No token-usage dashboard with charts. One live number on the run screen.
- No modal celebration when a phase completes. The tree growing is the feedback.
- No separate verification-report view. Verification is a final tree node that goes red/green; expand it if you care.
- No onboarding tour. The first screen is the input.
- No defensive UI for states the design should not permit.

### How the UX lands on the components

- `research::tui` (the lifted `tui` crate, research-themed) owns the four screens.
- `research::state` writes `swarm-state.yaml` per phase; the TUI renders that file as it changes (file-watch). The orchestrator is the source of truth; the TUI is a view. This keeps the run resumable and scriptable — `mr execute --headless` writes the same state file; `mr watch` renders it.
- The escalations queue reads `swarm-state.escalations`; the orchestrator pauses (blocks on an ack) when it surfaces one. Headless mode auto-decides per `--on-escalate={continue,pause,fail}`.
- `research::export` renders the run tree + `output.md` to HTML.

## 6. The worker primitive (source-grounded)

A worker is the unit the orchestrator dispatches. Its contract mirrors Claude Code's subagent dispatch, verified against the reconstructed source:

- **Fresh context window.** The worker's message list is built as exactly `[user: <prompt>]`. The orchestrator's history is never spliced in. (CC: `AgentTool.tsx` builds `promptMessages = [createUserMessage({ content: prompt })]`; `coordinatorMode.ts` states "Workers can't see your conversation. Every prompt must be self-contained.")
- **System prompt = agent file body only.** The agent definition's frontmatter `description` is the orchestrator's routing key (which worker to fire); the markdown body is the worker's instructions. (CC: `loadAgentsDir.ts` `parseAgentFromMarkdown` — `whenToUse = frontmatter['description']` for routing, `getSystemPrompt` = closure over `content.trim()` for the worker prompt.) We separate routing metadata from worker instructions accordingly.
- **Tool set = global denylist → per-worker denylist → per-worker allowlist** (or wildcard = all remaining). The spawn/Task tool is in the global denylist by construction, so workers are leaves by architecture — no runtime nested-dispatch guard is needed. (CC: `ALL_AGENT_DISALLOWED_TOOLS` includes the Agent tool; `resolveAgentTools` never adds it even if explicitly listed.)
- **Result = the worker's final assistant text only.** Its internal tool trajectory is hidden from the orchestrator (recorded to a sidechain transcript). Usage/tokens travel as metadata, not in the result string. (CC: `finalizeAgentTool` + `mapToolResultToToolResultBlockParam`.)
- **Worker cannot prompt the user.** `shouldAvoidPermissionPrompts: true`; AskUserQuestion in the denylist. (CC: `forkedAgent.ts` `createSubagentContext`.)
- **Isolated filesystem state.** Clone file-state/memoization per worker so one worker's reads don't pollute another's cache. Fresh agentId; fresh queryTracking depth = parent+1. (CC: `forkedAgent.ts`.)
- **Own abort signal, linked to the orchestrator's** so one worker or all can be cancelled. (CC: `createChildAbortController`.)
- **Bounded `max_turns`** so a stuck worker terminates. (CC: `runAgent.ts` `maxTurns`.)
- **In-process nested query loop**, not a subprocess. (CC: `runAgent` calls `query()` recursively.) Subprocess/remote isolation is an opt-in feature we do not use.
- **Parallelism = N worker calls per orchestrator turn.** Dispatch independent workers concurrently; serialize workers that touch overlapping files. (CC: `coordinatorMode.ts` + the "single message with multiple tool uses" instruction.)

### The query loop a worker runs (mirrors `queryLoop`)

```
loop:
  normalize_messages()        # drop pure-system/progress, merge attachments,
                              # ensure tool_result pairing (synthesize placeholders
                              # for orphaned tool_use), strip trailing thinking
  compact_if_needed()         # snip / microcompact / autocompact slots
  stream = api.call(system, messages, tools)
  collect tool_use blocks from stream
  if no tool_use: terminate(reason)
  else:
    partition tool_use by is_concurrency_safe
    run safe batches with bounded concurrency (~10), serial otherwise,
      each through: validate → permission → call() → map result → discipline.after_step()
    append tool_result user messages
    (if after_step returned a correction, inject it as a separate
     system-reminder user message before the next loop iteration)
    continue
```

- **Stop decision = `needsFollowUp`** (any `tool_use` present), not a raw `stop_reason` switch. (CC: `query.ts`.) Recovery paths: max_tokens escalation, prompt_too_long reactive compact, `max_turns` cap, abort.
- **Compaction.** Autocompact fires near the context limit (effective window − a buffer). It makes a separate small-model summarization call with system prompt `"You are a helpful AI assistant tasked with summarizing conversations."`, only `FileRead` available, thinking disabled, and replaces history with `[boundary, summary, attachments, hookResults]`. Circuit-break after 3 consecutive failures. (CC: `compactConversation`, `autoCompact.ts`.)
- **Message block types** the Rust enum must cover: `text`, `tool_use` (`id`,`name`,`input`), `tool_result` (`tool_use_id`,`content`,`is_error`), `thinking`, `redacted_thinking`, `image`, `document`. Roles: `user`, `assistant`. `system` is a separate top-level field, not a message.
- **System prompt is a layered `Vec<SystemBlock>`** with a static cacheable prefix, an explicit boundary marker, and a dynamic suffix; per-block `cache_control`. Built once per worker and frozen (render once, reuse the bytes) for cache-key stability. (CC: `getSystemPrompt`, `splitSysPromptPrefix`, `renderedSystemPrompt`.)
- **Streaming.** Parse `message_start`, `content_block_start`, `content_block_delta` (`text_delta`/`input_json_delta`/`thinking_delta`/`signature_delta`), `content_block_stop`, `message_delta` (usage+stop_reason), `message_stop`. Each content block becomes its own assistant message chunk. (CC: `claude.ts` stream switch.)

For the first cut, drop the feature-gated paths (history-snip, context-collapse, reactive-compact, token-budget, coordinator-mode, templates). The minimal deterministic worker is: normalize → autocompact (threshold-based) → stream → tool dispatch → loop on `needsFollowUp`. Add recovery paths once the base loop is stable.

## 7. Tool calling, permissions, and scoped worker tools

Wire shape (sent to the provider): `{ name, description, input_schema }` where `input_schema` is a JSON Schema with `type: 'object'` + `properties`. The Rust `Tool` trait exposes `name()`, `description()`, `input_schema() -> serde_json::Value`, `validate_input()`, `check_permissions()`, `call()`, `map_result_to_block()`, plus `is_concurrency_safe()` and `is_read_only()`.

- **Per tool_use execution order:** validate → permission decision → `call()` (try/catch) → map result → `discipline.after_step()` → user message. (CC's order is validate → PreToolUse hooks → permission → `call()` → PostToolUse hooks → map result; we collapse both hook points into the single `research::discipline` trait fired after the call — see §8/§9. There is no general hook runtime.)
- **tool_result block** is a user message wrapping `ContentBlockParam[]` with the `tool_result` first. Content is either a string or an array of `{type:'text'|'image', ...}`. Set `is_error: true` on deny / ask-rejected / throw / abort / unknown-tool / invalid-input, using the `<tool_use_error>...</tool_use_error>` wrapper convention. (CC: `StreamingToolExecutor.ts`.)
- **Permission engine:** deny-rules → ask-rules → tool-specific `check_permissions` (content rules) → mode bypass → whole-tool allow → passthrough→ask. Bypass-immune safety checks (`.git/`, settings files) fire inside `check_permissions` regardless of mode. (CC: `permissions.ts`.) Modes to replicate: `default`, `acceptEdits`, `bypassPermissions`, `plan`, `dontAsk`, and an `auto`-classifier mode.
- **Tool pool assembly:** sort built-ins by name, sort MCP tools by name, concat with built-ins first, dedupe by name (built-ins win). This keeps the tool list byte-stable across turns, preserving the prompt cache. (CC: `assembleToolPool`.)
- **Size-gate large results:** if a Read/MCP result exceeds a threshold, persist to a sidecar file and inline a stub with read instructions. Otherwise worker transcripts blow the context budget. (CC: `maxResultSizeChars`.)

### Scoped worker tools — built by us (confirmed gap)

Claude Code has **no** path-scoped Read/Write; it relies on deny rules + the Bash sandbox's `denyWrite` list. We go stricter for untrusted leaf workers:

- A **hard path check inside the Rust `Read`/`Write` tools** rejects any `file_path` not under the worker's assigned output dir *before* the permission engine runs. Treated like Claude Code's bypass-immune safety checks — fires regardless of mode.
- Pair it with a deny rule `Read(/**)` + allow `Read(<worker-dir>/**)` so the rule system and the tool both enforce the same boundary. A worker may also be allowed to read from a shared `docs/research/` dir for synthesis inputs — model that as allow `Read(docs/research/**)` plus the hard write-jail to `<worker-dir>`, never as a broad `additionalDirectories` entry that silently widens writes.
- **Workers get no Bash.** Their tool set is scoped file I/O + the MCP research tools only. Bash is reserved for the experimentalist runner's sandboxed execution (§10), not for leaf workers.

## 8. Skills, slash commands, hooks, plugins (source-grounded)

- **Skills → front-half flow bodies.** A skill body is injected by reading the markdown body, substituting `$ARGUMENTS`/`${SKILL_DIR}`, and injecting as a **meta user message** (not system, not tool_result). (CC: `getPromptForCommand` + `SkillTool.ts` returning `createUserMessage({ content, isMeta: true })`.) For our front-half phases, the flow body becomes the first user message of the LLM session. **No Skill tool, no model-discovery listing** — the orchestrator picks phases explicitly, so the 1%-of-context skill listing Claude Code builds is dead weight we drop.
- **Slash commands → `mr` subcommands.** `/name` is `findCommand(name) → getPromptForCommand(args) → user message`. Our subcommands mirror this with a CLI arg match instead of a command registry. `mr brainstorm <topic>` loads `flows/brainstorm.md`, substitutes `$ARGUMENTS`, starts the session. Mirror only these frontmatter fields as subcommand metadata: `description` (help), `argument-hint` (usage), `model` (per-flow override), `allowed-tools` (per-flow allowlist). Skip `disable-model-invocation`, `user-invocable`, `paths`, `version`.
- **Hooks → hardcoded doom-loop detector.** Claude Code's hook runtime (subprocess spawn, stdin JSON, matchers, parallel execution, multi-source merging) exists for user-configurable, untrusted, cross-plugin shell hooks. We have one trusted orchestrator. The only thing we need is the output contract: a post-step function returns `Option<Correction>` and the loop injects it as a system-reminder-tagged user message before the next model call. Implement as a direct Rust trait `fn after_step(&self, step: &Step) -> Option<String>` on the orchestrator — no subprocess, no JSON parsing, no matchers. **One thing worth copying verbatim:** the correction is its own message, never mutated into the tool_result. Keeping that separation lets the model treat the correction as new information and lets it survive compaction cleanly. (CC: `hook_additional_context` attachment → `<system-reminder>` user message.)
- **Plugins — skipped entirely.** No marketplaces, no untrusted code, no auto-updating ecosystem, no runtime superpowers dependency (inlined). MegaResearcher's dependency on superpowers becomes inlined prompt content, so there is no runtime dependency to resolve. Use a flat `flows/<phase>.md` dir resolved against a single `$MR_ROOT` env var — that covers the small slice of plugin functionality we need (path substitution) without the rest.

## 9. MCP client (source-grounded)

The Rust MCP client drives the Python ml-intern subprocess, replicating Claude Code's client contract plus one improvement:

- **Config:** `{ mcpServers: { name: { command, args, env, type?: "stdio" } } }`; `type` optional, defaults to stdio.
- **Env interpolation:** `${VAR:-default}` resolved from process env via regex; missing vars = warning, not fatal (keep the literal token). Apply to `command`, each `arg`, each `env` value. Do not special-case `CLAUDE_PLUGIN_ROOT` — it is just another env var. (CC: `envExpansion.ts`.)
- **Spawn:** parent env as base, overridden by the server's `env` map; pass the merged map as the child environment. Pipe stderr to a capped capture buffer (e.g. 64 KiB) for diagnostics; do not let it hit the UI. (CC: `client.ts` `StdioClientTransport` + `subprocessEnv`.)
- **Handshake:** use the official Rust `mcp` crate (e.g. `rmcp`); let it run `initialize` and negotiate the protocol version — do not pin one manually. Declare the `roots` capability (omit `elicitation` — we do not implement it). (CC: `Client` with `capabilities: { roots: {} }`.)
- **Tool discovery:** after connect, check `capabilities.tools`; if absent, expose no tools. Call `tools/list`, namespace each as `mcp__<normalize(server)>__<normalize(tool)>` where normalize replaces `[^a-zA-Z0-9_-]` with `_`. Store `mcpInfo { serverName, toolName }` for routing. Cap descriptions (2048 chars) before sending to the model. (CC: `buildMcpToolName`, `normalization.ts`.)
- **Routing:** parse `mcp__server__tool`, look up the live client by server name, call `tools/call` with `{name, arguments, _meta}`. (CC: `mcpInfoFromString`, `callMCPTool`.)
- **Result normalization:** handle three shapes — `toolResult` (string), `structuredContent` (JSON-stringify), `content[]` (map each item: text→text block; image→base64 image block; audio/non-image-blob→persist to file, return path text; resource text→prefixed text block; resource_link→prefixed text). If `isError: true`, treat the first text block as the error and surface as a tool_result error. Implement a size cap with truncation (simple truncation above N tokens is fine for research tools; avoid the image-compression path). (CC: `transformMCPResult`.)
- **Timeouts:** connection 30s (configurable via `MCP_TIMEOUT`); per-tool 5–10 min (Claude Code's 27.8h default is unreasonable). Both configurable via env.
- **Failure surfacing:** on connection failure, record the server as failed with its error string and **expose zero tools** from it (the model never sees them — no noisy error). On mid-call crash, return the error as the tool_result content. Missing env = warning, not a hard stop, but warn loudly so the user sets `HF_TOKEN`.
- **uv venv pre-warm (our improvement):** Claude Code does nothing smart here — `uv run`'s first-run venv creation can exceed the 30s handshake timeout and just fail. We pre-warm the ml-intern venv via `uv sync --project <path>` before the first `client.connect`, and cache the resulting venv so subsequent runs are fast.

## 10. Data flow (a gap-finding run, end to end)

1. **`mr init "<question>"`** → TUI opens at the Start screen. Brainstorm runs inline (LLM session, flow body as guiding prompt) → user confirms novelty target/modalities → `research::phases::spec` writes `docs/research/specs/<date>-<topic>-spec.md` → TUI shows it, user approves → `research::phases::plan` writes `docs/research/plans/<date>-<topic>-plan.md` → user approves. Gates are Rust-enforced; artifact content is LLM-generated.
2. **`mr execute`** (or "continue" in TUI) → orchestrator **pre-flight**: prompt assets present, ml-intern subprocess reachable (pre-warmed), provider key set, `docs/research/runs/` exists. (No superpowers check — that concept is gone; the discipline is in the prompts.) Generate run-id `YYYY-MM-DD-HHMM-<6hex>`, scaffold the run dir, write initial `swarm-state.yaml`. TUI switches to the Run screen and begins watching the state file.
3. **Phase 1** — orchestrator reads the plan's scout dispatches, loads the `literature-scout` prompt asset per scout, builds each worker call (spec + assignment + output path inlined into the single `prompt`), dispatches in waves of `max_parallel` through `api`. Each worker runs its query loop (hf_papers via the `mcp` client → ml-intern subprocess; scoped Read/Write for its own dir), writes its three artifacts. Orchestrator runs the **verification gate** per worker (three artifacts exist; one retry on miss; spot-check verification.md claims; escalate on second miss). Consolidate `bibliography.md`. Mark phase done in `swarm-state.yaml` → TUI tree grows.
4. **Phases 2–6** analogous. Phase 4 is the explicit **Rust red-team loop**: dispatch red-team worker → parse the verdict line from its `output.md` via `paper_chain::verdict` → branch on APPROVE / REJECT / KILL → REJECT re-dispatches hypothesis-smith with the red-team output inlined, re-runs red-team, increments retry, caps at 3 → on cap or KILL, append to `swarm-state.escalations`, pause the run, ping the TUI escalations queue.
5. **Phase 6.5 / 7–9** (only `--paper`) — `research::paper_chain` runs in Rust: experimentalist dispatches each runner in a Vercel Sandbox VM (Rust HTTP client), captures `results.json`/`repro.yaml`; drafter → reviewer → reviser loop with `verdict`/`regression` Rust modules gating each round; `finalize` writes `paper.md` + `paper-history.md`. Failed experiments surface in `paper-history.md` exactly as today.
6. **`research::phases::verify`** — a final LLM verification call + Rust spot-checks (run produced `output.md`, spec-latest symlink valid, every escalation resolved) → `verification-report.md` → final tree node goes green/red.
7. TUI slides to the Artifact screen with `output.md` rendered. One command → HTML export.

The orchestrator is the single writer of `swarm-state.yaml`; the TUI is a read-only file-watch view of it. `mr execute --headless` runs the same orchestrator with no TUI and auto-acks escalations per `--on-escalate={continue,pause,fail}`. `mr watch` reattaches the TUI to any in-progress run.

## 11. Error handling

- **Per-worker:** missing artifact → one redispatch with the specific missing path called out → escalate (same contract as today). Spot-check failure (verification.md claims PASS but a required check is obviously unmet) → one redispatch → escalate.
- **Verdict parse failures** (red-team, peer-review): `verdict` returns `NONE` → treat as failure #1, redispatch the reviewer once with feedback → escalate. Never silently advance on an unparseable verdict.
- **Experiment budget:** hard-stop at $5 sandbox + $5 API per experiment; `failed_budget` escalates with the cost figure. Transient (`failed_timeout`/`failed_exception`) → one retry. `failed_runner_not_implemented` (skeleton runners) → continue, drafter falls back, no retry.
- **Provider / MCP transport failures:** propagate as worker errors → same one-retry-then-escalate policy. ml-intern subprocess crash → orchestrator detects via the MCP client, surfaces a clean "research tools unavailable" escalation, does not retry silently.
- **Escalation discipline:** every escalation surfaces worker/hypothesis, what failed, retry count, and a concrete next-step — and **pauses** for user adjudication. Headless `--on-escalate=pause` writes the escalation to state and exits non-zero so CI/outer loops can decide. The orchestrator never makes an adjudication call on the user's behalf (carried over verbatim from today's discipline rules).

## 12. Testing

- **Pure-logic unit tests** (port the existing `tests/test_*.py`): `verdict`, `regression`, `protocol_parser`, `scaffold`, `finalize`, `preflight`, `swarm-state` serde, run-id generation. Deterministic; 1:1 Rust ports with 1:1 tests.
- **Worker primitive tests** with a **fake provider** (canned tool-call sequences): assert the scoped Read/Write tools only touch the worker's dir, the three artifacts get written, and the verification gate flags a deliberately-missing `manifest.yaml`. Replaces the manual `tests/manual_dispatch.py` approach with deterministic fakes.
- **Orchestrator integration test:** fake provider + fake Vercel Sandbox backend → run a gap-finding swarm against a tiny fixture spec/plan → assert the run tree, `swarm-state.yaml` phase statuses, `output.md`, a killed-hypothesis audit trail, and the spec-latest symlink. The fake sandbox mirrors `FakeSandboxBackend` from the current `sandbox.py`.
- **TUI:** a snapshot/smoke test that the four screens render from a fixture `swarm-state.yaml`, and that approval gates block progression until acked. No full E2E TUI harness — the orchestrator tests cover behavior; the TUI is a thin view.
- **ml-intern subprocess contract:** a smoke test that the Rust MCP client can list and call the 9 tools against the subprocess (the `test_mcp_imports.py` spirit, ported to a Rust MCP-client test). Includes the uv pre-warm path.

Each build phase ends with the test suite green so the harness is never broken mid-port.

## 13. Build phasing (what the implementation plan will expand)

Each phase is a mergeable, testable increment:

1. **Fork & trim.** Bring claurst into the repo as the workspace base; get it building; delete `buddy`/`acp`/`bridge`/voice/`/share`; add the `research` crate stub. Set root LICENSE=GPL-3.0 + NOTICE.
2. **Port pure logic + state.** `paper_chain::{verdict, regression, protocol_parser, scaffold, finalize, preflight}`, `research::state` (swarm-state serde, run-id, run-tree), with ported unit tests. No LLM yet.
3. **Worker primitive + fake provider.** `research::worker`, scoped `worker_tools`, prompt-asset loader, query loop against `api` with a fake. Worker-contract tests green.
4. **Orchestrator phases 1–6.** Pre-flight, scaffold, wave dispatch, verification gate, red-team loop, consolidations, synthesist. Fake-provider integration test for a gap-finding run.
5. **MCP wiring.** Rust `mcp` client drives the ml-intern subprocess; uv pre-warm; 9-tool smoke test. Workers can now actually call hf_papers.
6. **Front-half phases in TUI.** Start screen, brainstorm/spec/plan guided sessions, approval gates, artifact writing. Run screen rendering `swarm-state.yaml` live.
7. **Paper chain + Vercel Sandbox.** `paper_chain::{experiment, sandbox}` as Rust HTTP client; experimentalist dispatch; drafter/reviewer/reviser loop; finalize. Fake-sandbox test, then Vercel with `VERCEL_TOKEN`.
8. **Verification + HTML export + past-runs.** `verify` phase, `research::export` HTML renderer, runs list.
9. **Cutover.** Delete the old plugin files (`.claude-plugin/`, `commands/`, `hooks/`, `mcp/`, `lib/`, `skills/`, `tools/ml-intern`→`python/`); repo is the Rust harness. Update README.

## 14. Non-goals (explicit, to prevent scope creep)

- Porting ml-intern to Rust (stays a Python subprocess).
- A plugin / marketplace / untrusted-hook ecosystem (skipped — §8).
- A runtime superpowers dependency (inlined into prompts — §8).
- Voice, ACP/editor protocol, the buddy companion, gist sharing (trimmed from claurst).
- Multi-provider routing beyond Anthropic-default + optional extras (claurst's Free Mode is dropped).
- A token-usage dashboard, settings screen, or onboarding tour (cut — §5).
- Bash access for leaf workers (no — §7).

## 15. Open questions deferred to the implementation plan

- Exact provider abstraction: which crates for the Anthropic streaming client, and whether to keep claurst's `api` crate's multi-provider scaffolding or thin it to Anthropic-only.
- `max_turns` default per worker role (CC uses 200 for forks; research workers may want less).
- Whether the HTML export is generated on-demand or also written to the run dir.
- Vercel Sandbox Rust HTTP client: hand-rolled vs. an existing SDK crate.

