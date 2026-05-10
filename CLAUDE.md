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
| `research-swarm` orchestrator | `subagent-driven-development` |
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

## Common failure modes and what to do

- **Worker returns without all three required artifacts** (output.md, manifest.yaml, verification.md): orchestrator re-dispatches with explicit instructions about the missing artifact. After 3 retries, escalate.
- **All hypotheses killed by red-team**: the gap-finding was likely off, not the hypotheses. Pause and ask the user.
- **Eval-designer flags an intractable compute budget**: surface to the user; do not silently include intractable experiments in the synthesis.
- **Pre-flight finds superpowers missing**: refuse to run, point user at install instructions.
