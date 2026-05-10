# MegaResearcher Architecture (for contributors)

This document is for contributors. End-user documentation is in [README.md](../README.md).

## The three layers

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 3: Skills + Slash commands                                 │
│   research-brainstorming → writing-research-spec →               │
│   writing-research-plan → executing-research-plan →              │
│   research-verification                                          │
│   /research-init  /research-execute  /share-traces               │
└──────────────────────────────────────────────────────────────────┘
                              ↓ invokes
┌──────────────────────────────────────────────────────────────────┐
│ Layer 2: Subagents                                               │
│   research-swarm (orchestrator)                                  │
│     ├─ literature-scout (parallel, Phase 1)                      │
│     ├─ gap-finder       (parallel, Phase 2)                      │
│     ├─ hypothesis-smith (parallel, Phase 3 + revisions)          │
│     ├─ red-team         (sequential per hypothesis, Phase 4)     │
│     ├─ eval-designer    (parallel, Phase 5)                      │
│     └─ synthesist       (single, Phase 6)                        │
└──────────────────────────────────────────────────────────────────┘
                              ↓ uses tools from
┌──────────────────────────────────────────────────────────────────┐
│ Layer 1: MCP server + hooks (vendored from ml-intern-cc)         │
│   9 tools: hf_papers, hf_inspect_dataset, hf_docs_explore,       │
│            hf_docs_fetch, hf_repo_files, github_examples,        │
│            github_list_repos, github_read_file, web_search       │
│   2 hooks: PostToolUse (doom_loop), SessionEnd (upload_traces)   │
└──────────────────────────────────────────────────────────────────┘
                              ↓ wraps
                    huggingface/ml-intern
                  (vendored at tools/ml-intern/)
```

## Why ml-intern is vendored, not a submodule

Submodules cause UX pain (`git submodule update --init` is easy to forget). A
vendored snapshot at a pinned SHA is reproducible, works offline, and bumping
ml-intern becomes a deliberate maintainer action: pull upstream → run smoke
tests → snapshot → release.

The pinned SHA is recorded at [`tools/ml-intern.sha`](../tools/ml-intern.sha).
To bump ml-intern:

```bash
cd /tmp && rm -rf ml-intern && git clone https://github.com/huggingface/ml-intern.git
NEW_SHA=$(cd ml-intern && git rev-parse HEAD)
rm -rf <plugin>/tools/ml-intern && cp -R ml-intern <plugin>/tools/ml-intern
rm -rf <plugin>/tools/ml-intern/.git
echo "vendored ml-intern @ $NEW_SHA" > <plugin>/tools/ml-intern.sha
cd <plugin> && python3 tests/test_mcp_imports.py  # MUST pass
```

## Why we import ml-intern via `sys.path`, not as a path dep

The `mcp/pyproject.toml` declares ml-intern as a path dependency, which gives
us its full transitive deps in the venv. The server then imports `agent.tools.*`
modules by adding `tools/ml-intern/` to `sys.path` at startup. This works because
ml-intern's `agent/__init__.py` requires the full agent runtime (litellm, etc.)
to be importable — we can't pick-and-choose tool modules.

## Why the doom-loop hook is project-local discipline

The PostToolUse doom-loop hook fires for every tool call in any session running
inside a project that has MegaResearcher installed. It's pure stdlib so it's
cheap to invoke. It detects two patterns:

1. 3+ identical consecutive calls → injects a "STOP repeating" correction
2. Repeating sequences A→B→A→B (length 2–5, 2+ reps) → injects a "STOP cycle" correction

Corrections are injected as `additionalContext` in the hook's stdout JSON, which
CC merges into the assistant's next turn. This is non-blocking — the hook never
prevents a tool call, only nudges the model.

## Why upload_traces uses the mcp uv project, not ml-intern's

The `agent.core.session_uploader` module has `--format claude_code` support and
runs as a script. The hook calls it via `uv run --project mcp` (which has
ml-intern installed editable, so `python -m agent.core.session_uploader` works
in that venv). Avoids creating a separate venv for `tools/ml-intern/` itself.

## Where agent system prompts live

In `agents/<name>.md`. Each is a single markdown file with frontmatter (name,
description, model: inherit) and a body that's the agent's system prompt.

Worker subagents share a contract: they receive a spec + an assignment + an
output path; they produce three artifacts (`output.md`, `manifest.yaml`,
`verification.md`); they invoke `superpowers:verification-before-completion`
before claiming completion. Each worker's `.md` file spells out role-specific
required checks.

## Where skill bodies live

In `skills/<name>/SKILL.md`. Same convention as superpowers' skills. Each skill
has frontmatter (name + description controlling auto-fire) and a body with
instructions for Claude when the skill is invoked.

## Hook configuration

`hooks/hooks.json` registers PostToolUse (matching all tools) and SessionEnd.
Both use `${CLAUDE_PLUGIN_ROOT}` for plugin-relative paths so the plugin works
regardless of where it's installed.

## MCP configuration

`.mcp.json` registers the ml-intern MCP server as stdio, launched via
`uv run --project ${CLAUDE_PLUGIN_ROOT}/mcp python ${CLAUDE_PLUGIN_ROOT}/mcp/server.py`.
The `env` block forwards HF_TOKEN, GITHUB_TOKEN, and ML_INTERN_PATH (so the
server can find the vendored ml-intern even if its sys.path tweak fails).

## Testing strategy

Two smoke tests in [`tests/`](../tests/):

- [`test_mcp_imports.py`](../tests/test_mcp_imports.py) — every wrapped handler imports, server registers exactly 9 tools
- [`test_doom_loop.py`](../tests/test_doom_loop.py) — synthetic transcripts trigger the right corrections

Run before any release:

```bash
python3 tests/test_doom_loop.py
python3 tests/test_mcp_imports.py
```

Integration testing (end-to-end swarm run) is currently manual: run
`/research-init` against a small topic and verify the run produces all expected
artifacts. Automating this is on the roadmap but requires careful budget control
(it dispatches many agents).

## Release checklist

1. `tests/test_doom_loop.py` passes
2. `tests/test_mcp_imports.py` passes (requires `mcp/` venv synced)
3. Bump version in `.claude-plugin/plugin.json`
4. Update `tools/ml-intern.sha` if ml-intern was bumped
5. Tag release in git
