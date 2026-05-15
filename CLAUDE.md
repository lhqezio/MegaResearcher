# CLAUDE.md — MegaResearcher

This file is loaded into Claude's context when MegaResearcher is the active plugin context.

## Hard dependency: superpowers

MegaResearcher's skills and subagents invoke superpowers skills throughout. Without the `superpowers` plugin installed, the swarm degrades severely: no `verification-before-completion` discipline, no `dispatching-parallel-agents` primitives, no `receiving-code-review` for the red-team worker. The `executing-research-plan` skill checks for superpowers and refuses to run if it's missing.

If you (Claude) are about to dispatch the swarm and superpowers is not present, stop and instruct the user to install it.

## Skill invocation map (when MegaResearcher fires which superpowers skill)

| MegaResearcher entry | Invokes superpowers skill |
|---|---|
| `research-brainstorming` skill | `brainstorming` |
| `writing-research-plan` skill | `writing-plans` |
| `executing-research-plan` skill | `dispatching-parallel-agents` |
| `executing-research-plan` skill (the orchestrator runs in the main session, not as a subagent — see Architectural note below) | `subagent-driven-development` |
| `red-team` worker | `receiving-code-review` (adapted for hypotheses) |
| `eval-designer` + worker code | `test-driven-development` |
| Any worker that writes code | `requesting-code-review` |
| `research-verification` skill | `verification-before-completion` |
| Parallel baseline experiments | `using-git-worktrees` |
| Worker hits a bug | `systematic-debugging` |

## Where research artifacts live

Always under `docs/research/` in the **consuming project**, never in the MegaResearcher plugin directory:

- `docs/research/specs/<spec>.md` — research specs (per-project)
- `docs/research/plans/<plan>.md` — research plans
- `docs/research/runs/<run-id>/` — swarm run outputs (one subdir per worker, plus the run's `output.md`, `swarm-state.yaml`, `verification-report.md`)
- `docs/research/specs/<spec>-latest.md` — symlink to the most recent run's output.md for that spec

## Discipline rules baked into MegaResearcher

These are NOT optional. Violating them defeats the plugin's purpose:

1. **Audit trail is non-negotiable.** Every rejected/killed hypothesis appears in the synthesist's final document with the lesson it contributes. No silent rejections.
2. **The red-team critique loop fires for every hypothesis** when the spec's novelty target is `hypothesis`. Cap is 3 revisions; further rejections escalate to the user.
3. **Pre-registration of decision rules** in eval-designer outputs. Post-hoc thresholds are how plausible-but-wrong findings survive.
4. **Citations resolve or do not exist.** If `hf_papers paper_details` doesn't return a paper, the paper does not exist for purposes of any MegaResearcher output.
5. **Workers stay in their lanes.** Literature-scouts produce bibliographies, not hypotheses. Hypothesis-smiths forge hypotheses, not designs. Eval-designers design experiments, not run them. Synthesist synthesizes existing outputs, does not produce new claims.

## Architectural note: the orchestrator is a skill, not a subagent

Claude Code's harness forbids nested agent dispatch — a subagent cannot use the Task tool to spawn other subagents. Therefore the swarm orchestrator (the thing that dispatches workers in waves and runs the critique loop) lives in the `executing-research-plan` **skill**, which executes in the main session and CAN dispatch agents. Worker subagents (literature-scout, gap-finder, hypothesis-smith, red-team, eval-designer, synthesist) are leaves and dispatch nothing further.

There is intentionally no `agents/research-swarm.md`. If you (Claude) see references to a "research-swarm orchestrator subagent" in older docs or commits, ignore them — that pattern was tried and reverted because it broke against the nested-dispatch restriction.

## Optional paper-drafting chain (SP1+SP2a)

`/research-execute --paper` extends the existing chain with 3 additional phases:

- **Phase 7** — `manuscript-drafter` produces `paper/draft-v1.md` from `output.md` + eval-designer protocols
- **Phase 8** — `peer-reviewer` + `reviser` loop, cap 2 rounds, early-exit on APPROVE
- **Phase 9** — finalize → `paper/paper.md` + `paper/paper-history.md`

Requires the underlying run's novelty target to be `hypothesis` (paper chain consumes Phase 5 eval-designer outputs). Pre-flight refuses on `gap-finding`-target outputs.

The paper chain produces NO fabricated experimental results — the Experimental Plan section embeds the eval-designer protocols as "we will measure X via Y" (no numbers). SP2 will add an experimentalist worker that replaces the plan with results.

Architecture: same single-session orchestrator + leaf-worker pattern. New agents are leaves; new Python helpers in `lib/paper_chain/` handle verdict parsing, regression detection, pre-flight, scaffold, and finalize.

**SP2a additions:**
- **Phase 6.5** — `experimentalist` runs each surviving hypothesis's protocol via a Vercel Sandbox VM. Inserts BETWEEN Phase 6 (synthesist) and Phase 7 (drafter). Requires `VERCEL_TOKEN` env var.
- **5 skeleton runners** in `lib/runners/{specs,abgen,citeme,limitgen,paperwrite_bench}/` satisfy the Runner schema contract. Real benchmark integrations are deferred to per-runner follow-up plans (SP2a.1 SPECS, SP2a.2 AbGen, SP2a.3 CiteME, SP2a.4 LimitGen, SP2a.5 PaperWrite-Bench).
- **Drafter modification** — Section 6 of `draft-v1.md` becomes "Experiments & Results" with real numbers when `paper/experiments/<hyp-id>/results.json` has `status: completed`; falls back to SP1's option-γ "Experimental Plan" stub when results are absent or `status: failed` (with a `[Experimental data unavailable: <code>]` marker for failed cases).
- **Cost ceiling: $5 sandbox + $5 API per experiment.** Hard-stop on exceed. 1 retry on transient failures (timeout / exception). Failed experiments visible in `paper-history.md`.

## Common failure modes and what to do

- **Worker returns without all three required artifacts** (output.md, manifest.yaml, verification.md): the skill re-dispatches once with explicit instructions about the missing artifact. After one retry, escalate to the user.
- **All hypotheses killed by red-team**: the gap-finding was likely off, not the hypotheses. Pause and ask the user.
- **Eval-designer flags an intractable compute budget**: surface to the user; do not silently include intractable experiments in the synthesis.
- **Pre-flight finds superpowers missing**: refuse to run, point user at install instructions.
