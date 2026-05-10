---
name: executing-research-plan
description: Use after a research plan has been written and approved by the user, to run the research swarm. The skill IS the orchestrator — it reads the plan and dispatches worker subagents in waves directly from the main session. (Avoids the harness restriction that subagents cannot dispatch other subagents.) Verifies the superpowers plugin is reachable. The /research-execute slash command is the user-facing entry point.
---

# Executing the Research Plan

You are about to spend significant tokens (typically 200k–500k for a `gap-finding` run, more for `hypothesis`). You ARE the orchestrator — there is no separate orchestrator subagent. Worker subagents are leaf nodes; this skill drives them from the main session.

## Why this skill, not a subagent, runs the swarm

Claude Code's harness forbids nested agent dispatch: a subagent cannot use the Task tool to spawn other subagents. The orchestrator must run in a context that *can* dispatch — that's the main session, where this skill executes. Worker subagents (literature-scout, gap-finder, hypothesis-smith, red-team, eval-designer, synthesist) are leaves and dispatch nothing further, so they remain subagents.

## Pre-flight checks (all must pass; halt if any fail)

**1. Superpowers is installed.** The workers and this skill invoke superpowers skills (`subagent-driven-development`, `dispatching-parallel-agents`, `verification-before-completion`, `receiving-code-review`, `test-driven-development`, `requesting-code-review`, `using-git-worktrees`, `systematic-debugging`).

Verify by checking that `~/.claude/plugins/cache/claude-plugins-official/superpowers/*/skills/dispatching-parallel-agents/SKILL.md` exists. If not, stop and tell the user:

> MegaResearcher requires the `superpowers` plugin. Install it and try again. (Marketplace: claude-plugins-official; plugin name: superpowers.)

**2. The MCP server is configured.** Confirm `mcp__ml-intern__hf_papers` is available. If not, MCP isn't loaded — tell the user to verify `HF_TOKEN` is set and that they've restarted Claude Code in this project.

**3. The plan and spec exist and are approved.** The user must have explicitly approved both. If unsure, ask before proceeding.

**4. The consuming project has `docs/research/runs/`.** Create if missing.

## Generate the run-id and scaffold

Run id: `YYYY-MM-DD-HHMM-<6-char-hex>` (today's UTC date+time + a short random hash).

Create `docs/research/runs/<run-id>/` and one subdir per worker named in the plan's *Swarm decomposition* section:

- One subdir per Phase-1 scout (e.g., `scout-1/`, `scout-2/`, …)
- One subdir per Phase-2 gap-finder
- (Phase-3 hypothesis-smith subdirs are created dynamically per gap as Phase 3 fires)
- (Phase-4 red-team subdirs are created dynamically per critique invocation)
- (Phase-5 eval-designer subdirs are created dynamically per surviving hypothesis)
- One `synthesist/` for Phase 6

Write `docs/research/runs/<run-id>/swarm-state.yaml` recording: run_id, spec path, plan path, novelty_target, max_parallel, phases (each with status, workers list with status `pending`), escalations: [], retry_counts: {}.

## Phase execution — invoke superpowers:dispatching-parallel-agents

For every phase that dispatches workers in parallel, invoke `superpowers:dispatching-parallel-agents` and follow its discipline (single message with N tool calls, etc.).

### Phase 1 — literature-scout

Read the plan's *Phase 1 — literature-scout dispatches* section. Group scouts into waves of size `MEGARESEARCHER_MAX_PARALLEL` (default 4). For each wave:

1. Send a single message containing one Agent tool call per scout in the wave. Each call has:
   - `subagent_type`: `literature-scout`
   - `description`: 3–5 word summary of the sub-topic
   - `prompt`: the full spec content + the scout's specific assignment (paste the relevant paragraph from the plan verbatim) + the explicit output path `docs/research/runs/<run-id>/scout-<N>/` + a reminder that the worker must produce all three artifacts (output.md, manifest.yaml, verification.md)

2. Wait for the wave to complete.

3. Run the *Per-worker verification gate* (below) on each scout's output.

After all scouts pass: write `docs/research/runs/<run-id>/bibliography.md` consolidating every scout's `output.md` (just paths + sub-topic labels — the full content lives in each scout's dir). Update `swarm-state.yaml` to mark Phase 1 complete.

### Phase 2 — gap-finder

Read the plan's *Phase 2 — gap-finder dispatches* section. One wave (typically 2–3 gap-finders). Each gap-finder gets:

- The full spec
- Its specific assignment (which scout outputs to read)
- Output path `docs/research/runs/<run-id>/gap-finder-<N>/`
- The full content of the relevant scout `output.md` files **inlined into the prompt** (don't expect the gap-finder to find them by path; subagents have weaker filesystem context than the main session)

Run the verification gate. After all pass: write `docs/research/runs/<run-id>/gaps.md` consolidating outputs. Mark Phase 2 complete.

### Phase 3 — hypothesis-smith (only if novelty target ≠ `gap-finding`)

Skip this phase entirely if the spec's novelty target is `gap-finding`.

Otherwise: parse the consolidated `gaps.md`. Dispatch one hypothesis-smith per gap (in waves of `max_parallel`). Each smith gets:

- Full spec
- Its assigned gap (one paragraph from `gaps.md`)
- Output path `docs/research/runs/<run-id>/hypothesis-smith-<N>/`

Run the verification gate.

### Phase 4 — red-team critique loop (only if Phase 3 ran)

For each hypothesis from Phase 3:

1. Dispatch a `red-team` subagent. Prompt: full spec + the hypothesis's `output.md` content + the gap-finder's `output.md` for the targeted gap + path `docs/research/runs/<run-id>/red-team-<N>/`.
2. Read the red-team's `output.md`. Parse the verdict line (`APPROVE | REJECT (revision-N) | KILL (irrecoverable)`).
3. If APPROVE: hypothesis advances. Mark in swarm-state.
4. If REJECT: increment retry count for that hypothesis in swarm-state. If retry < 3, dispatch hypothesis-smith again (revision invocation: include the red-team's full output as context). Then re-dispatch red-team on the revised hypothesis. Continue the loop.
5. If retry hits 3 OR red-team returns KILL: add to swarm-state.escalations. Surface to user as: "Hypothesis-N escalated after N rounds because [reasons]. Continue with the surviving hypotheses, or pause for adjudication?"

Cap is 3 revisions per hypothesis. Do not silently exceed.

### Phase 5 — eval-designer (only if Phase 4 produced survivors)

One eval-designer per surviving hypothesis. Each gets the spec + the approved hypothesis + path. Run verification gate. If a designer flags `flagged_intractable: true` in its manifest, surface to user before continuing.

### Phase 6 — synthesist

Single dispatch. Prompt:

- Full spec
- Full plan
- All Phase-1 scout outputs (inline)
- All Phase-2 gap-finder outputs (inline)
- All Phase-3 hypothesis-smith outputs (inline, including all revisions)
- All Phase-4 red-team verdicts (inline)
- All Phase-5 eval-designer outputs (inline)
- The full `swarm-state.yaml`
- Output path: `docs/research/runs/<run-id>/synthesist/` AND the run-root `output.md`

After return: update the symlink `docs/research/specs/<spec-basename>-latest.md` → the new run's `output.md`.

## Per-worker verification gate

After every worker completes, before treating its output as usable:

1. Confirm all three artifacts exist at the worker's output path: `output.md`, `manifest.yaml`, `verification.md`. If any are missing, **redispatch once with explicit feedback** ("You did not write `manifest.yaml` to /full/path/. The worker contract requires all three. Produce it now."). Increment that worker's retry count in `swarm-state.yaml`.

2. If after one retry any artifact is still missing: add the worker to `swarm-state.escalations` and surface to user. Do NOT silently advance.

3. Spot-check: read the worker's `verification.md`. If it claims "PASS" but the worker contract's required checks are obviously not met (e.g., literature-scout claiming PASS but `output.md` cites no arxiv IDs), redispatch once with the specific failed check called out.

## Escalation discipline

When you escalate to the user, surface:
- Which worker / hypothesis
- What failed (specific objection or missing artifact)
- The retry count
- A concrete next-step recommendation (continue without it, retry with different prompt, abandon the run)

Do NOT make the decision on the user's behalf.

## After the swarm completes

1. Verify the run produced `docs/research/runs/<run-id>/output.md`.
2. Verify the symlink `docs/research/specs/<spec-basename>-latest.md` exists and points to the new output.
3. Invoke the `research-verification` skill on the run.
4. Report to the user: path to `output.md`, total worker invocations, escalations (if any), verification verdict.

## Discipline rules

- **Pre-flight is non-negotiable.** Mid-run dep failures waste tokens and produce confused output.
- **Inline content into worker prompts.** Subagents are not reliable at reading files by path; embed the relevant content. This is also why we keep workers as leaves — every dispatch is self-contained.
- **One retry per worker, then escalate.** Multiple silent retries hide systemic problems and burn tokens.
- **Escalations go to the user.** The skill never makes adjudication calls on the user's behalf.
- **No phase parallelism.** Phases are strictly sequential. Within a phase, workers are parallel up to `MEGARESEARCHER_MAX_PARALLEL`.
- **Idle phases are skipped, not dispatched.** If novelty target is `gap-finding`, do not dispatch hypothesis-smith / red-team / eval-designer at all.
