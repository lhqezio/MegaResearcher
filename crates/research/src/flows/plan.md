---
name: plan
description: Produce the executable research plan at docs/research/plans/YYYY-MM-DD-<topic>-plan.md, including the Swarm decomposition section the orchestrator parses. Gets user approval.
argument-hint: "[topic]"
model: inherit
allowed-tools:
  - Read
  - Write
---

# Writing the Research Plan

You have an approved spec at `docs/research/specs/<spec>.md` (path known from
this conversation). Produce a concrete, executable research plan. Use `Write`
jailed under `docs/research/`, writing to `plans/YYYY-MM-DD-<topic-slug>-plan.md`
(relative path, resolving to
`docs/research/plans/YYYY-MM-DD-<topic-slug>-plan.md`; same date as the spec or
later).

## Plan meta-format (inlined)

A plan is a document an engineer (or, here, the orchestrator) can execute with
zero extra context. Include these sections, scaled to complexity:

- **Goal** — one sentence.
- **Architecture** — 2–3 sentences on approach.
- **Context** — the spec's question, novelty target, success criteria, YAGNI
  fence (do not restate the whole spec; reference its path).
- **Approach** — the phases below, in order.
- **Critical files** — the spec path and any seed papers/datasets named.
- **Verification** — how completion is judged (ties to the spec's success
  criteria).

## Swarm decomposition (required — the orchestrator parses this)

```markdown
## Swarm decomposition

### Phase 1 — literature-scout dispatches
One assignment per scout. Each is a paragraph:
- Sub-topic name
- Focus constraints (e.g., "2024–2026 only", "must include EO + IR fusion specifically")
- Tools to prioritize (defaults: hf_papers + web_search + github_examples)
Aim for 3–6 scouts.

### Phase 2 — gap-finder dispatches
One assignment per gap-finder:
- Slice: which scout outputs to analyze
- Focus: what kinds of gaps to prioritize
Aim for 1–3 gap-finders.

### Phase 3 — hypothesis-smith dispatches
Computed dynamically by the orchestrator (one smith per gap). Do not pre-list.

### Phase 4 — red-team critique loop
Computed dynamically (one red-team per hypothesis, revision loop up to 3).
Note any project-specific critique focus here.

### Phase 5 — eval-designer dispatches
Computed dynamically (one designer per surviving hypothesis). Note any
compute-budget guidance (e.g., "designs requiring >500 GPU-hours must be flagged
for user approval").

### Phase 6 — synthesist
Single dispatch. Note synthesis-specific requirements (e.g., "final document
must be ≤ 8 pages and tied directly to the spec's success criteria").

### Custom worker dispatches
If the spec defined custom workers, list when they fire and with what inputs.

### Parallelism budget
MEGARESEARCHER_MAX_PARALLEL = <number> (default 4)

### Estimated total runtime + token budget
Grounded estimate. State assumptions. Round up.
```

## Self-review (run before asking the user)

- Every Phase-1 scout has a non-overlapping focus.
- Every Phase-2 gap-finder is assigned a slice whose scout outputs correspond to
  actual scouts in Phase 1.
- The parallelism budget is set explicitly.
- The token estimate is grounded (e.g., "12 workers × 30k tokens average ≈ 360k;
  plus orchestrator overhead ≈ 500k total").

## User review gate

Present a summary (workers per phase, estimated cost, any unusual decisions) and
ask the user to review before the run. Wait for explicit approval. A plan is not
a spec — the plan tells the orchestrator HOW; the spec tells everyone WHY. Do
not restate the spec at length. Phase-1 dispatches are the highest-leverage
decision: bad scout coverage → bad gap-finding → bad hypotheses, so spend time
partitioning the topic well.

When the user approves, say so plainly. The harness then waits for the user to
run `mr execute`; do not start the run yourself — spending the tokens is the
user's deliberate choice.