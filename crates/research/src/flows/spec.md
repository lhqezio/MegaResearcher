---
name: spec
description: Author the research spec from the brainstorm answers. Writes docs/research/specs/YYYY-MM-DD-<topic>-spec.md and gets user approval.
argument-hint: "[topic]"
model: inherit
allowed-tools:
  - Read
  - Write
---

# Writing the Research Spec

You have the brainstorm answers (carried in this conversation). Write them into a
properly-structured spec, then get user approval. Use the `Write` tool to create
the file; the harness jails writes under `docs/research/`, so write to
`specs/YYYY-MM-DD-<topic-slug>-spec.md` (relative path, resolving to
`docs/research/specs/YYYY-MM-DD-<topic-slug>-spec.md`). Use today's date and a
kebab-case topic slug. If the directory does not exist, create it.

## Spec format (use exactly these section headings)

```markdown
# <Topic Title> — Research Spec

**Status:** draft
**Created:** YYYY-MM-DD
**Novelty target:** gap-finding | hypothesis | synthesis

## Question

<One paragraph from the brainstorm, step 1.>

## Modalities and domain

<From step 3. Name the data types, the application area, the operational context if any.>

## Constraints

<From step 4. Each constraint as a bullet.>

## Success criteria

<From step 5. What artifacts must exist? What numerical bars? What does red-team
approval look like for this project?>

## Out of scope (YAGNI fence)

<From step 6. Each item as a bullet. The synthesist will reflect this fence.>

## Custom workers

<From step 7. If none, write "None — using the bundled six (literature-scout,
gap-finder, hypothesis-smith, red-team, eval-designer, synthesist).">

## Decisions locked in

- <date · decision · rationale>
```

## Self-review (run inline, fix before asking the user)

1. **Placeholders** — any `<…>` that survived, "TBD", "TODO". Fix them.
2. **Specificity** — the Question must be specific enough that a literature-scout
   could write a focused query, not generic.
3. **Falsifiability prep** — if the novelty target is `hypothesis`, the success
   criteria must mention falsification criteria, not just "produce hypotheses".
4. **YAGNI is concrete** — Out of scope items must be specific things the user
   might otherwise expect in scope, not generic disclaimers.

## User review gate

Tell the user the spec path and ask them to review. Wait for explicit approval.
Only when the user approves does the harness advance to the plan session; do not
start the plan yourself.