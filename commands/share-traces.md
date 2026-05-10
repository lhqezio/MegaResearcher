---
description: View or change the visibility of the HF dataset that stores ml-intern session traces (mirrors ml-intern's /share-traces).
argument-hint: "[status|public|private]"
---

The user invoked `/share-traces $ARGUMENTS`.

Behavior:
- `status` (or no arg): print the current value of `ML_INTERN_TRACES_REPO`, whether `HF_TOKEN` is present, and — if both are set — call the HF API to report the dataset's current `private` flag.
- `public`: flip the dataset to public. Confirm with the user before doing this; published session traces may contain prompts, paths, and tool output.
- `private`: flip the dataset to private.

Implementation: use the `huggingface_hub` Python API via `uv run --project tools/ml-intern python -c '...'`. Call `HfApi().update_repo_visibility(repo_id=..., repo_type="dataset", private=<bool>)`. Read `ML_INTERN_TRACES_REPO` and `HF_TOKEN` from the shell environment.

If `ML_INTERN_TRACES_REPO` is unset, tell the user how to set it (export the env var; see `.claude/hooks/upload_traces.py` docstring).
