---
description: Start a new MegaResearcher project. Kicks off the spec-driven research chain (brainstorm → spec → plan → execute → verify).
argument-hint: "[topic]"
---

The user invoked `/research-init $ARGUMENTS`.

Start a new research project. Invoke the `research-brainstorming` skill from MegaResearcher. If `$ARGUMENTS` contains an initial topic, use it as the seed for the brainstorm; otherwise begin with an open clarifying conversation.

After brainstorming, the chain continues: `writing-research-spec` → `writing-research-plan` → `/research-execute`. Walk the user through gates as they arise; do not skip any.
