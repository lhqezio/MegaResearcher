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

## Optional: `--paper` flag

If the `/research-execute` invocation includes `--paper`, the orchestrator runs three additional phases (7, 8, 9) after Phase 6 to produce a paper draft. The flag is consumed in the main session — detect it from the invocation arguments and gate Phases 7–9 on its presence.

If `--paper` is set, run an additional pre-flight check before starting Phase 7:

```
python3 lib/paper_chain/preflight.py docs/research/runs/<run-id>/
```

If this exits non-zero, surface the stderr message to the user and refuse to start Phase 7. Do NOT start Phases 7–9 if pre-flight fails.

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

### Phase 7 — manuscript-drafter (only if `--paper`)

Skip this phase entirely if `--paper` is not set.

Otherwise:

1. Run `python3 lib/paper_chain/scaffold.py docs/research/runs/<run-id>/` to create the `paper/` subdirectory.
2. Dispatch ONE `megaresearcher:manuscript-drafter` subagent with the prompt containing:
   - Full content of `docs/research/runs/<run-id>/output.md` (research-direction)
   - Full content of every `docs/research/runs/<run-id>/eval-designer-*/output.md`
   - Output path: `docs/research/runs/<run-id>/paper/`
   - Reminder of the three required artifacts: `draft-v1.md`, `drafter-manifest.yaml`, `drafter-verification.md`
3. Wait for completion. Run the per-worker verification gate.
4. **Citation-integrity gate (failure #2):** read the draft. Every arXiv ID appearing in `draft-v1.md` must also appear in the research-direction's Sources section. Use:
   ```
   grep -oE "arXiv:[0-9]{4}\.[0-9]{4,5}" docs/research/runs/<run-id>/paper/draft-v1.md | sort -u
   grep -oE "arXiv:[0-9]{4}\.[0-9]{4,5}" docs/research/runs/<run-id>/output.md | sort -u
   ```
   The first set must be a subset of the second. If not, re-dispatch the drafter once with the offending arXiv IDs called out. After one retry, escalate.
5. Update `swarm-state.yaml`:
   ```yaml
   phase_7_manuscript_drafter:
     status: completed
     output: paper/draft-v1.md
   ```

### Phase 8 — peer-reviewer + reviser loop (only if `--paper`)

Skip if `--paper` not set. Otherwise: loop with N starting at 1, capped at 2.

**Round N:**

1. Dispatch ONE `megaresearcher:peer-reviewer` subagent with:
   - Full content of `docs/research/runs/<run-id>/paper/draft-v<N>.md`
   - Full content of `docs/research/runs/<run-id>/output.md` (research-direction, for context)
   - Output path: `docs/research/runs/<run-id>/paper/`
   - Required artifact filenames: `review-v<N>.md`, `reviewer-manifest-v<N>.yaml`, `reviewer-verification-v<N>.md`
2. Wait for completion. Run the per-worker verification gate.
3. Parse the verdict:
   ```
   python3 lib/paper_chain/verdict.py docs/research/runs/<run-id>/paper/review-v<N>.md
   ```
   Verdict is one of `APPROVE`, `REVISE`, `KILL`, or `NONE` (parse failure).
4. **If `APPROVE`:** record in `swarm-state.yaml`, exit Phase 8 loop, proceed to Phase 9.
5. **If `KILL`:** record in `swarm-state.yaml`, append to `swarm-state.escalations` with the reviewer's reasoning, SKIP Phase 9, surface to user. Run still produces the last `draft-v<N>.md` for inspection but no `paper.md`.
6. **If `NONE` (parse failure):** treat as failure #1 (missing artifact). Re-dispatch reviewer once with explicit feedback. After one retry, escalate.
7. **If `REVISE` and N < 2:**
   - Dispatch ONE `megaresearcher:reviser` subagent with:
     - Full content of `docs/research/runs/<run-id>/paper/draft-v<N>.md`
     - Full content of `docs/research/runs/<run-id>/paper/review-v<N>.md`
     - Output path: `docs/research/runs/<run-id>/paper/`
     - Required artifact filenames: `draft-v<N+1>.md`, `reviser-manifest-v<N>.yaml`, `reviser-verification-v<N>.md`, and APPEND to `revision-log.jsonl`
   - Wait for completion. Run the per-worker verification gate.
   - Run the same citation-integrity gate as Phase 7 step 4 on `draft-v<N+1>.md`.
   - Increment N. Loop back to step 1.

After Round 2's review (review-v2) is produced (step 2 of the second iteration), run **regression detection** before parsing the verdict:
```
python3 lib/paper_chain/regression.py \
  docs/research/runs/<run-id>/paper/review-v1.md \
  docs/research/runs/<run-id>/paper/review-v2.md
```
If REGRESSION (exit 1), append to escalations with note "runaway revision detected" and surface to user for adjudication BEFORE deciding whether to proceed (regardless of verdict). Do NOT auto-advance on regression flag.

8. **If `REVISE` and N == 2:** cap reached. Record in `swarm-state.yaml`, append to escalations with verdict and reviewer reasoning. Surface to user: "2 review rounds completed, final verdict still REVISE. Continue manually, accept the last draft as Phase 9 input, or abandon?" Do NOT auto-advance to Phase 9.

After loop exit (APPROVE or escalation):
```yaml
phase_8_review_loop:
  status: completed|escalated
  rounds_completed: 1|2
  final_verdict: APPROVE|REVISE|KILL
  rounds:
    - round: 1
      review: paper/review-v1.md
      revision: paper/draft-v2.md  # null if APPROVE on this round
    # - round: 2 ... only if first round was REVISE
```

### Phase 9 — finalize (only if `--paper` AND Phase 8 ended in `APPROVE` OR user accepted last draft)

Skip if `--paper` not set, or if Phase 8 exited via `KILL`, or if the user declined to accept the last draft after a cap-2 REVISE.

Otherwise:

1. Run finalize:
   ```
   python3 lib/paper_chain/finalize.py docs/research/runs/<run-id>/paper/ <final-verdict>
   ```
   This:
   - Writes `paper.md` (copy of the latest `draft-v<N>.md`)
   - Concatenates all `review-v<N>.md` files + `revision-log.jsonl` + final verdict marker into `paper-history.md`
2. Update `swarm-state.yaml`:
   ```yaml
   phase_9_finalize:
     status: completed
     paper: paper/paper.md
   ```
3. The run's paper deliverable is now at `docs/research/runs/<run-id>/paper/paper.md`. Surface this path to the user.

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
