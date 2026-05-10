# ml-intern-cc

A Claude Code integration that exposes Hugging Face's [`ml-intern`](https://github.com/huggingface/ml-intern) research tools and behaviors **inside** Claude Code, instead of running ml-intern as a parallel CLI.

## What this is

Three pieces, wired into the project's `.claude/` and root config:

1. **MCP server** (`ml_intern_cc/server.py`) — wraps ml-intern's tool modules and exposes them as MCP tools to CC. We `import` from `tools/ml-intern/` rather than vendoring, so `git pull` upstream gets you tool fixes for free.
2. **PostToolUse hook** (`.claude/hooks/doom_loop.py`) — port of ml-intern's `agent/core/doom_loop.py`. Reads CC's transcript JSONL after every tool call, detects identical-consecutive or A→B→A→B repetition, and injects a corrective system message via `additionalContext`.
3. **SessionEnd hook** (`.claude/hooks/upload_traces.py`) — calls ml-intern's own `agent/core/session_uploader.py` with `--format claude_code` to upload the CC transcript to a private HF dataset.

Plus a `/share-traces` slash command for parity with ml-intern's interactive UX.

## What we *don't* port

- The agent loop itself — CC has one
- Auto-compaction — CC handles it
- Approval gates — CC's permission system handles it
- The HF Spaces sandbox tool — CC's Bash is the local sandbox
- The internal `research_tool` — CC subagents replace it

The result: ml-intern's *tools* and the *doom-loop / trace-upload behaviors* are available natively in CC, with one agent loop instead of two.

## Tools exposed (v1)

| Tool                          | Source module                          |
|-------------------------------|----------------------------------------|
| `mcp__ml-intern__hf_papers`           | `agent/tools/papers_tool.py`     |
| `mcp__ml-intern__hf_inspect_dataset`  | `agent/tools/dataset_tools.py`   |
| `mcp__ml-intern__hf_docs_explore`     | `agent/tools/docs_tools.py`      |
| `mcp__ml-intern__hf_docs_fetch`       | `agent/tools/docs_tools.py`      |
| `mcp__ml-intern__hf_repo_files`       | `agent/tools/hf_repo_files_tool.py` |
| `mcp__ml-intern__github_examples`     | `agent/tools/github_find_examples.py` |
| `mcp__ml-intern__github_list_repos`   | `agent/tools/github_list_repos.py` |
| `mcp__ml-intern__github_read_file`    | `agent/tools/github_read_file.py` |
| `mcp__ml-intern__web_search`          | `agent/tools/web_search_tool.py` |

## One-time setup

Prerequisites: `uv`, Python 3.11+, `git`. The ml-intern clone lives at `../tools/ml-intern/` (already cloned by the project bootstrap).

```bash
# Install MCP server deps (creates ml_intern_cc/.venv)
cd /Users/ggix/ND-Challenge/ml_intern_cc
uv sync

# Configure secrets
cp .env.example .env
# …edit .env, fill in HF_TOKEN. GITHUB_TOKEN is optional —
# only needed if you want github_examples / github_list_repos / github_read_file.

# Sanity-check the server starts
uv run python server.py < /dev/null
# (it'll wait for MCP stdio; ctrl-C to exit. No error == ready.)
```

CC will pick up `.mcp.json` automatically the next time you open the project. The first tool invocation will prompt you to approve the MCP server.

## Trace upload (optional)

Set `ML_INTERN_TRACES_REPO=<your-hf-username>/ml-intern-sessions` in your shell env (not `.env` — the SessionEnd hook reads from the process env CC was launched with). Create the dataset repo on HF first; the uploader expects it to exist. Default visibility: private.

## Tested against

- ml-intern: tracking `main`, last verified at clone time (see `tools/ml-intern/.git/`)
- CC plugin/hook schema: `hooks.json` array-of-{matcher, hooks} form, `${CLAUDE_PROJECT_DIR}` in commands, `transcript_path` in hook stdin, `additionalContext` in `hookSpecificOutput`

## Known gaps

- `/share-traces` is implemented as an instruction-style slash command (the model runs the HF API call). A standalone script would be more deterministic; deferred to v2.
- No tests yet for the doom-loop hook against synthetic CC transcripts. Add `tests/` with a few JSONL fixtures.
- The MCP server has no health check beyond import-time errors. If `agent.tools.*` upstream renames a handler, the server fails on startup with an `ImportError` — log and pin a known-good ml-intern SHA.
