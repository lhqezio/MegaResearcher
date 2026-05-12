# ml-intern-cc

The MCP server that exposes Hugging Face's [`ml-intern`](https://github.com/huggingface/ml-intern) research tools to Claude Code, plus the hooks that port ml-intern's doom-loop detection and trace upload into the CC runtime.

For the design rationale (vendoring strategy, `sys.path` import, hook layout, testing approach), see [`docs/architecture.md`](../docs/architecture.md).

## Tools exposed

| Tool                                  | Source module                          |
|---------------------------------------|----------------------------------------|
| `mcp__ml-intern__hf_papers`           | `agent/tools/papers_tool.py`           |
| `mcp__ml-intern__hf_inspect_dataset`  | `agent/tools/dataset_tools.py`         |
| `mcp__ml-intern__hf_docs_explore`     | `agent/tools/docs_tools.py`            |
| `mcp__ml-intern__hf_docs_fetch`       | `agent/tools/docs_tools.py`            |
| `mcp__ml-intern__hf_repo_files`       | `agent/tools/hf_repo_files_tool.py`    |
| `mcp__ml-intern__github_examples`     | `agent/tools/github_find_examples.py`  |
| `mcp__ml-intern__github_list_repos`   | `agent/tools/github_list_repos.py`     |
| `mcp__ml-intern__github_read_file`    | `agent/tools/github_read_file.py`      |
| `mcp__ml-intern__web_search`          | `agent/tools/web_search_tool.py`       |

## Setup

Requirements: `uv` and Python 3.11+. The vendored ml-intern lives at `../tools/ml-intern/`.

```bash
cd "$CLAUDE_PLUGIN_ROOT/mcp"   # or whatever path you cloned MegaResearcher to, then /mcp
uv sync

cp .env.example .env
# HF_TOKEN required. GITHUB_TOKEN optional (only needed for the three GitHub tools).

# Sanity-check the server starts
uv run python server.py < /dev/null
# Waits for MCP stdio; ctrl-C to exit. No error == ready.
```

Claude Code picks up `.mcp.json` from the plugin root automatically the next time you open a project with MegaResearcher enabled. The first tool call will prompt you to approve the MCP server.

## Trace upload (optional)

Set `ML_INTERN_TRACES_REPO=<your-hf-username>/ml-intern-sessions` in the shell env Claude Code is launched with (not `.env` — the SessionEnd hook reads from the process env). Create the dataset on Hugging Face first; the uploader expects it to exist. Default visibility: private.
