---
name: writing-research-spec
description: Use after research-brainstorming has clarified intent, to author the research spec. Produces a structured markdown document at docs/research/specs/YYYY-MM-DD-<topic>-spec.md in the consuming project. The next skill in the chain is writing-research-plan. Required gate before writing-research-plan can fire.
---

# Writing the Research Spec

You have completed `research-brainstorming` and have answers to the seven clarification items. Your job is to write those answers into a properly-structured research spec, then get user approval.

## Where the spec lives

`docs/research/specs/YYYY-MM-DD-<topic-slug>-spec.md` in the **consuming project** (NOT in the MegaResearcher plugin directory). Use today's date. Topic slug is kebab-case.

If the directory doesn't exist, create it.

## Spec format (use exactly these section headings)

```markdown
# <Topic Title> — Research Spec

**Status:** draft
**Created:** YYYY-MM-DD
**Novelty target:** gap-finding | hypothesis | synthesis

## Question

<One paragraph from research-brainstorming step 1.>

## Modalities and domain

<From step 3. Be specific. Name the data types, the application area, the operational context if any.>

## Constraints

<From step 4. Each constraint as a bullet. Examples:
- No classified data; open / synthetic datasets only
- No GPU spend during scoping
- All datasets must be CC-BY or more permissive>

## Success criteria

<From step 5. What artifacts must exist? What numerical bars (if any)? What does red-team approval look like for this project?>

## Out of scope (YAGNI fence)

<From step 6. Each item as a bullet. The synthesist will reflect this fence in the final document.>

## Custom workers

<From step 7. If the user did not define custom workers, write "None — using the bundled six (literature-scout, gap-finder, hypothesis-smith, red-team, eval-designer, synthesist).">

<If custom workers were defined, list each as:
### <worker-name>
- Remit: <what this worker does>
- Inputs: <what it consumes>
- Outputs: <what it produces beyond the standard output.md/manifest.yaml/verification.md>
- Tools: <which MCP tools or other resources it needs>>

## Decisions locked in

- <date · decision · rationale>
```

## Self-review (run inline, fix if found)

After writing, scan for:

1. **Placeholders** — any `<...>` that didn't get filled, "TBD", "TODO". Fix.
2. **Specificity** — the Question section should not be generic ("study fusion methods"). It should be specific enough that a literature-scout could write a focused query.
3. **Falsifiability prep** — if the novelty target is `hypothesis`, the success criteria should mention falsification criteria, not just "produce hypotheses."
4. **YAGNI is concrete** — Out of scope items should be specific things the user might otherwise expect to be in scope, not generic disclaimers.

## User review gate

After writing and self-reviewing the spec, ask the user:

> "Spec written and committed to `<path>`. Please review it and let me know if you want to make any changes before we start writing the research plan."

Wait for the user's response. Only proceed when they approve.

## Terminal state

Once approved, invoke `writing-research-plan`. Do not invoke any other skill. Do not start dispatching workers — that's the executing-research-plan skill's job, gated by an approved plan.
