---
name: writing-research-plan
description: Use after a research spec has been written and approved by the user. Produces a research plan at docs/research/plans/YYYY-MM-DD-<topic>-plan.md that includes the standard superpowers plan structure plus a swarm-decomposition section telling the orchestrator which workers fire when. Invokes superpowers:writing-plans for the meta-format. Terminal state is the /research-execute slash command.
---

# Writing the Research Plan

You have an approved research spec at `docs/research/specs/<spec>.md`. Your job is to produce a concrete, executable research plan.

## Process

**Step 1 — Invoke `superpowers:writing-plans`.** Use it for the standard plan-document structure (context, approach, critical files, verification, etc.). The research plan IS a plan in superpowers' sense; we just add a swarm-specific section.

**Step 2 — Add the *Swarm decomposition* section.** This is what makes a research plan executable by the `executing-research-plan` skill (the orchestrator that runs in the main session). It must specify:

```markdown
## Swarm decomposition

### Phase 1 — literature-scout dispatches
List one assignment per scout. Each assignment is a paragraph:
- Sub-topic name
- Focus constraints (e.g., "2024–2026 only", "must include EO + IR fusion specifically")
- Any tools the scout should prioritize (defaults to hf_papers + web_search + github_examples)

Aim for 3–6 scouts. More scouts = more parallel work but more tokens.

### Phase 2 — gap-finder dispatches
List one assignment per gap-finder. Each is:
- Slice: which scout outputs to analyze (e.g., "literature-scout-1 + literature-scout-3 outputs")
- Focus: what kinds of gaps to prioritize (e.g., "unexplored intersections between EO and SIGINT fusion")

Aim for 1–3 gap-finders.

### Phase 3 — hypothesis-smith dispatches
Computed dynamically by the orchestrator from gap-finder outputs (one smith per gap). You do not pre-list these.

### Phase 4 — red-team critique loop
Computed dynamically (one red-team per hypothesis, with revision loop up to 3 attempts). Note any project-specific critique focus here (e.g., "red-team should pay particular attention to data-leakage risks").

### Phase 5 — eval-designer dispatches
Computed dynamically (one designer per surviving hypothesis). Note any compute-budget guidance (e.g., "designs that require >500 GPU-hours must be flagged for user approval").

### Phase 6 — synthesist
Single dispatch. Note any synthesis-specific requirements (e.g., "final document must be ≤ 8 pages and tied directly to the spec's success criteria").

### Custom worker dispatches
If the spec defined custom workers, list when they fire and with what inputs.

### Parallelism budget
MEGARESEARCHER_MAX_PARALLEL = <number> (default 4)

### Estimated total runtime + token budget
Honest estimate. State assumptions.
```

**Step 3 — Self-review.** Use `superpowers:writing-plans`' self-review pass plus check:
- Every Phase-1 scout has a non-overlapping focus
- Every Phase-2 gap-finder is assigned a real slice (the scout outputs it cites must correspond to actual scouts in Phase 1)
- The parallelism budget is set explicitly
- Token budget estimate is grounded (e.g., "12 workers × 30k tokens average = ~360k tokens; plus orchestrator overhead = ~500k tokens total")

## Where the plan lives

`docs/research/plans/YYYY-MM-DD-<topic-slug>-plan.md` in the consuming project. Same date as the spec or later.

## User review gate

After writing the plan, present a summary (count of workers per phase, estimated cost, any unusual decisions) and ask:

> "Plan written and committed to `<path>`. Review it before we run `/research-execute`?"

Wait for explicit approval before pointing the user at `/research-execute`.

## Terminal state

Tell the user to run `/research-execute <plan-path>` when ready. Do not invoke `executing-research-plan` yourself unless the user explicitly asks — they should explicitly choose to spend the tokens.

## Discipline rules

- **A plan is not a spec.** The plan tells the orchestrator HOW; the spec tells everyone WHY. Don't restate the spec at length in the plan.
- **Phase-1 dispatches are the highest-leverage decision.** Bad scout coverage → bad gap-finding → bad hypotheses. Spend time partitioning the topic well.
- **Honest token estimates.** Underestimating leads to mid-run budget exhaustion. Round up.
