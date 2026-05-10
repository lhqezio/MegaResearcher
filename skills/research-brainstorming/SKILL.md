---
name: research-brainstorming
description: Use when starting a new research project under MegaResearcher, before writing a research spec. Wraps superpowers:brainstorming with research-specific clarifying questions (novelty bar, modalities, constraints). The terminal state is invoking the writing-research-spec skill.
---

# Research Brainstorming

You are about to start a new research project. Your job is to clarify the research intent enough that the next skill (`writing-research-spec`) can produce a high-quality spec.

## Process

**Step 1 — Invoke `superpowers:brainstorming` first.** That skill handles the general "explore, ask, propose, present" loop. Use it as your scaffold. The user agreed to this composition when they installed MegaResearcher.

**Step 2 — Layer in research-specific clarifications.** Before invoking `writing-research-spec`, you MUST have clear answers to all of these. Ask whichever are not already obvious from context:

1. **The research question.** One paragraph. What is the user actually trying to find out?
2. **Novelty target** — choose one:
   - `gap-finding` — identify unexplored regions in the literature
   - `hypothesis` — gap-finding + propose testable hypotheses with falsification criteria
   - `synthesis` — novel combinations of existing techniques (less novel than `hypothesis`, lower nonsense risk)
3. **Modalities and domain.** What kinds of data, models, or work are in scope? (e.g., "multi-modal sensor fusion for ISR; EO/IR + RF + telemetry"; or "legal precedent search in tax law")
4. **Constraints.** What's off-limits? (No classified data, no GPU spend during scoping, must use open datasets, licence requirements, deadline pressure, etc.)
5. **Success criteria.** What artifacts must exist for the user to consider the run successful? (e.g., "at least 3 surviving hypotheses with eval designs; a synthesist document under 8 pages; every claim cited")
6. **Out of scope (YAGNI fence).** What is explicitly NOT this project? List items the user wants to defer or never address. The synthesist will reflect this fence in the final document.
7. **Custom workers (optional).** Does the project need worker types beyond the bundled six (literature-scout, gap-finder, hypothesis-smith, red-team, eval-designer, synthesist)? If so, the user defines them inline in the spec.

**Step 3 — Confirm before moving on.** Before invoking `writing-research-spec`, restate the answers back to the user and get explicit approval. Do not proceed if the user is uncertain about any of items 1–6 (item 7 is optional).

## Terminal state

Invoke `writing-research-spec`. Do not invoke any other skill. Do not start producing research content yourself — you orchestrate the start of the SDD chain, not the work.

## Discipline rules

- **No assumed answers.** If item 4 (constraints) wasn't discussed, ask. Defaults are fine ("no GPU spend during scoping unless approved") but make them explicit.
- **One question at a time** when iterating with the user (per `superpowers:brainstorming`'s convention).
- **The novelty target is consequential.** It determines which workers fire and which discipline rules apply most heavily. If the user picks `hypothesis`, the red-team critique loop is mandatory; spell that out so they understand the cost (more agent dispatches, longer runs).
