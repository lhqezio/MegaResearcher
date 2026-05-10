---
name: executing-research-plan
description: Use after a research plan has been written and approved by the user, to dispatch the research-swarm orchestrator subagent. Verifies the superpowers plugin is reachable (hard dependency); if not, refuses to run with a clear error. Invokes superpowers:dispatching-parallel-agents discipline. The /research-execute slash command is the user-facing entry point that triggers this skill.
---

# Executing the Research Plan

You are about to spend significant tokens. Before dispatching the swarm, verify everything is in order.

## Pre-flight checks (all must pass; halt and fix any that fail)

**1. Superpowers is installed and reachable.** The orchestrator and several worker subagents invoke superpowers skills (`subagent-driven-development`, `dispatching-parallel-agents`, `verification-before-completion`, `receiving-code-review`, `test-driven-development`, `requesting-code-review`, `using-git-worktrees`, `systematic-debugging`).

Verify by attempting to read `~/.claude/plugins/cache/claude-plugins-official/superpowers/*/skills/dispatching-parallel-agents/SKILL.md` (the path may vary slightly by version; if you can find the superpowers plugin directory anywhere under `~/.claude/plugins/`, that's sufficient).

If superpowers is NOT installed: stop. Tell the user:

> "MegaResearcher requires the `superpowers` plugin. Install it and try again. (Marketplace: claude-plugins-official; plugin name: superpowers.)"

**2. The MCP server is configured.** Confirm that `mcp__ml-intern__hf_papers` is available as a tool. If not, the MCP server is not running — tell the user to verify their `.env` is set up (HF_TOKEN minimum) and that they've restarted Claude Code in this project.

**3. The plan exists and is approved.** The user must have explicitly approved the plan before running this. If you have any doubt, ask before proceeding.

**4. The consuming project has a `docs/research/runs/` directory.** Create it if missing.

## Dispatch

Once pre-flight passes, invoke `superpowers:dispatching-parallel-agents` for the discipline, then dispatch the `research-swarm` orchestrator subagent with this prompt structure:

```
You are the research-swarm orchestrator. Execute this research plan:

PLAN: <full content of docs/research/plans/<plan>.md>

SPEC: <full content of docs/research/specs/<spec>.md>

RUN ID: <YYYY-MM-DD-HHMM-<short-hash>>

OUTPUT BASE: docs/research/runs/<run-id>/

Follow your system prompt. Run all six phases. Honor the critique loop discipline strictly. Report back when complete with the path to docs/research/runs/<run-id>/output.md.
```

Generate the run-id. Use today's date + UTC time (HHMM) + a short random hash.

## Monitoring during execution

The orchestrator dispatches workers and may pause on escalations (Phase 4 if a hypothesis hits the 3-revision cap, or Phase 5 if a worker flags an intractable compute budget). When the orchestrator returns control to you for an escalation, surface the specific question to the user clearly. Don't make an executive decision on the user's behalf for an escalation — that's why it's an escalation.

## After the swarm completes

1. Verify the run produced `docs/research/runs/<run-id>/output.md`
2. Verify the symlink `docs/research/specs/<spec-basename>-latest.md` points at the new run's output
3. Invoke `research-verification` skill on the run before reporting "done" to the user

## Discipline rules

- **Pre-flight is non-negotiable.** Failed deps mid-run waste tokens and produce confused output.
- **Escalations go to the user, not to your judgment.** The orchestrator escalates because the situation needs human adjudication.
- **No silent retries.** If the orchestrator fails, surface the failure and ask the user. Do not auto-restart.
