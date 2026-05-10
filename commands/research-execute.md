---
description: Execute an approved MegaResearcher research plan. Dispatches the research-swarm orchestrator. Will spend significant tokens — only use after the plan is reviewed and approved.
argument-hint: "<path-to-plan>"
---

The user invoked `/research-execute $ARGUMENTS`.

`$ARGUMENTS` is the path to an approved research plan at `docs/research/plans/<plan>.md`.

Invoke the `executing-research-plan` skill with that plan path. The skill will:
1. Run pre-flight checks (superpowers installed, MCP server reachable, plan exists, plan approved by user)
2. Generate a run-id
3. Dispatch the `research-swarm` orchestrator subagent
4. Surface escalations as they arise
5. Run `research-verification` after the swarm completes

Do not bypass the pre-flight checks. Do not auto-approve escalations.
