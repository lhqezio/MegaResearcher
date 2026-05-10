# MegaResearcher

A Claude Code plugin for **research-team-swarm** workflows under spec-driven development discipline. Capable of synthesising **novel research directions**: gap-finding, hypothesis generation with falsification criteria, and adversarial red-team critique.

Built on top of [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern) (HF/arxiv/web/GitHub research tools) and integrating deeply with the [`superpowers`](https://github.com/obra/superpowers) skill library (TDD, planning, verification, parallel agent dispatch).

## What it does

You start with a research question. MegaResearcher walks you through a five-step spec-driven chain:

```
/research-init  →  research-brainstorming  →  writing-research-spec  →  writing-research-plan  →  /research-execute  →  research-verification
```

The execution step dispatches a six-phase swarm of specialised subagents:

```
Phase 1: literature-scout      (parallel, one per sub-topic)
Phase 2: gap-finder            (parallel over consolidated bibliography)
Phase 3: hypothesis-smith      (parallel, one per identified gap)
Phase 4: red-team CRITIQUE LOOP    ← adversarial critique with revision cap
Phase 5: eval-designer         (parallel, one per surviving hypothesis)
Phase 6: synthesist            (single, composes final research-direction doc)
```

The result is a `docs/research/runs/<run-id>/output.md` containing surviving hypotheses (with mechanism, falsification criteria, and experimental design), a transparent audit trail of rejected hypotheses, and an explicit reflection of what the research deliberately did NOT explore.

## Why the red-team loop matters

Plausible-sounding nonsense is the default failure mode of LLM-driven research tools. The red-team worker exists specifically to attack hypotheses with technical rigor — independently re-verifying gap claims, spot-checking citations, attacking mechanisms, and rejecting unfalsifiable hand-waving. Hypotheses must survive the critique loop (or get killed and recorded in the audit trail) before reaching the eval-designer.

This is non-negotiable. The plugin will not let you skip it.

## Requirements

- Claude Code with plugin support
- The [`superpowers`](https://github.com/obra/superpowers) plugin (hard dependency)
- `uv` (Python package manager)
- A Hugging Face token (HF_TOKEN) — required for the HF/arxiv tools
- A GitHub token (GITHUB_TOKEN) — optional; only needed if you want the GitHub code-search tools

## Installation

> Marketplace install instructions will go here once the plugin is published. For now, install as a local directory plugin:

```bash
git clone https://github.com/lhqezio/MegaResearcher.git ~/MegaResearcher
cd ~/MegaResearcher/mcp
cp .env.example .env
# Edit .env to add HF_TOKEN (and optionally GITHUB_TOKEN)
uv sync
```

Then add MegaResearcher to your Claude Code plugins (path-source install — exact mechanism depends on your CC version).

## What you install

A single plugin that bundles:

- **9 MCP tools** (research instruments wrapping ml-intern): `hf_papers`, `hf_inspect_dataset`, `hf_docs_explore`, `hf_docs_fetch`, `hf_repo_files`, `github_examples`, `github_list_repos`, `github_read_file`, `web_search`
- **7 subagents** (1 orchestrator + 6 workers): `research-swarm`, `literature-scout`, `gap-finder`, `hypothesis-smith`, `red-team`, `eval-designer`, `synthesist`
- **5 skills** (the SDD chain): `research-brainstorming`, `writing-research-spec`, `writing-research-plan`, `executing-research-plan`, `research-verification`
- **3 slash commands**: `/research-init`, `/research-execute`, `/share-traces`
- **2 hooks**: PostToolUse doom-loop detector, SessionEnd transcript uploader
- **A vendored snapshot of ml-intern** (`tools/ml-intern/`) pinned to a known-good SHA

## What you don't get

- Your API tokens (configure via `mcp/.env`)
- Your research outputs (those go to your *consuming project's* `docs/research/`, not into the plugin)
- The superpowers plugin (install separately — required peer dependency)

## Configuration

Environment variables (set in `mcp/.env` for tools; in your shell for hooks):

| Variable | Required? | Purpose |
|---|---|---|
| `HF_TOKEN` | yes | HF API access for paper/dataset/docs/repo tools |
| `GITHUB_TOKEN` | no | GitHub API access; without it, three GitHub tools error gracefully |
| `ML_INTERN_TRACES_REPO` | no | If set (`<your-hf-username>/ml-intern-sessions`), session transcripts upload to this private HF dataset on session end |
| `ML_INTERN_TRACES_PRIVATE` | no | `true` (default) or `false` for trace dataset visibility |
| `MEGARESEARCHER_MAX_PARALLEL` | no | Max parallel workers per phase (default 4) |

## Repository layout

```
MegaResearcher/
├── .claude-plugin/plugin.json      # plugin manifest
├── .mcp.json                       # MCP server registration
├── agents/                         # 7 subagent definitions
├── skills/                         # 5 skill definitions
├── commands/                       # 3 slash commands
├── hooks/                          # PostToolUse + SessionEnd hooks
├── mcp/                            # the FastMCP server wrapping ml-intern
├── tools/ml-intern/                # vendored ml-intern snapshot
├── docs/                           # plugin documentation
└── tests/                          # smoke tests
```

## Status

v0.1 — initial release. Tested against the DND IDEaS multi-modal AI scoping use case as the first concrete consumer.

## License

Apache-2.0. The vendored `tools/ml-intern/` is also Apache-2.0 (upstream licence preserved).
