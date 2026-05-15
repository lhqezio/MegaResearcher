---
description: Execute an approved MegaResearcher research plan. Invokes the `executing-research-plan` skill, which runs the orchestrator in the main session and dispatches worker subagents in waves. Will spend significant tokens — only use after the plan is reviewed and approved.
argument-hint: "<path-to-plan> [--paper]"
---

The user invoked `/research-execute $ARGUMENTS`.

`$ARGUMENTS` is the path to an approved research plan at `docs/research/plans/<plan>.md`.

Invoke the `executing-research-plan` skill with that plan path. The skill will:
1. Run pre-flight checks (superpowers installed, MCP server reachable, plan exists, plan approved by user)
2. Generate a run-id
3. Run the six-phase swarm from the main session, dispatching worker subagents in waves
4. Surface escalations as they arise
5. Run `research-verification` after the swarm completes

Do not bypass the pre-flight checks. Do not auto-approve escalations.

## Optional `--paper` flag

If the invocation ends with `--paper`, the orchestrator runs three additional phases (7, 8, 9) after Phase 6 synthesist to produce a paper draft. Requires the underlying research plan's novelty target to be `hypothesis` (paper chain refuses to run on `gap-finding`-target plans because it consumes Phase 5 eval-designer outputs).

Output of the paper chain lands at `docs/research/runs/<run-id>/paper/paper.md`. The original research-direction at `docs/research/runs/<run-id>/output.md` is unchanged.

**Phase 6.5 (SP2a):** when `--paper` is set, the orchestrator inserts a new Phase 6.5 between the synthesist and the drafter — the `experimentalist` worker runs each surviving hypothesis's protocol in a Vercel Sandbox VM. Requires `VERCEL_TOKEN` env var; without it, all experiments fail with `runner_not_implemented` (skeleton mode) and the drafter falls back to SP1's option-γ stub. Per-experiment budget ceiling: $5 sandbox + $5 API.
