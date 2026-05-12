# MegaResearcher

A Claude Code plugin that runs a small group of research subagents against a research question and produces a research direction document: hypotheses, falsification criteria, experimental designs, and a record of the ideas that were rejected along the way.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-orange)](https://github.com/lhqezio/MegaResearcher)
[![Built on ml-intern](https://img.shields.io/badge/built%20on-ml--intern-yellow)](https://github.com/huggingface/ml-intern)
[![Powered by superpowers](https://img.shields.io/badge/powered%20by-superpowers-purple)](https://github.com/obra/superpowers)

## Example runs

Two real runs are committed in this repo, including spec, plan, every per-worker subdirectory, swarm state, and verification report. Nothing is post-processed for the demo.

| Topic | Pipeline | What's in the output |
|---|---|---|
| **[Recursive reasoning on subquadratic-attention backbones](docs/research/examples/recursive-subquadratic-fusion/run-2026-05-10-0729-766039/output.md)** | Full hypothesis pipeline — 5 scouts + 2 gap-finders + 6 hypothesis-smiths × 2 rounds + 6 red-team × 2 rounds + 5 eval-designers + synthesist (**38 worker invocations**) | 5 surviving hypotheses with mechanism, falsification criteria, and experimental designs; 1 hypothesis killed by red-team and preserved with the reasoning that killed it |
| **[Multi-modal ISR fusion landscape](docs/research/examples/multimodal-fusion-gap-finding/run-2026-05-10-0615-0ece4e/output.md)** | Gap-finding only — 6 scouts + 3 gap-finders + synthesist (**10 worker invocations**) | 4 priority gaps converged on by independent gap-finders; 8-candidate shortlist with explicit licence-driven discards; 3 top picks spanning 3 deployment contexts |

[Browse the examples folder](docs/research/examples/) for the full audit trail.

## What it does

You write a one-paragraph research question. The plugin walks you through a brainstorm, drafts a spec, drafts a plan, and (once you approve both) runs a six-phase swarm:

1. `literature-scout` — one per sub-topic, builds an annotated bibliography
2. `gap-finder` — partitions the bibliography, looks for unexplored intersections
3. `hypothesis-smith` — one per gap, forges a testable hypothesis
4. `red-team` — adversarial critique loop, up to 3 revisions per hypothesis
5. `eval-designer` — one per surviving hypothesis, designs the experiment
6. `synthesist` — composes the final document

The orchestrator that runs these waves is a skill (`executing-research-plan`) that runs in the main session, not a subagent. Claude Code doesn't allow nested agent dispatch, so the orchestrator has to live where it can call `Task`.

## Output

Each run writes to `docs/research/` in the project you're working in:

```
docs/research/
├── specs/
│   └── 2026-05-10-multimodal-fusion-spec.md
├── plans/
│   └── 2026-05-10-multimodal-fusion-plan.md
└── runs/
    └── 2026-05-10-1430-a3f9b2/
        ├── output.md                  # the deliverable
        ├── swarm-state.yaml           # what ran, when, and what it produced
        ├── verification-report.md     # spot-checks
        ├── bibliography.md            # consolidated Phase 1
        ├── gaps.md                    # consolidated Phase 2
        ├── literature-scout-1/
        ├── gap-finder-1/
        ├── hypothesis-smith-1/        # includes revision history
        ├── red-team-1/                # verdict + objections + spot-checks
        ├── eval-designer-1/
        └── ...
```

`output.md` is self-contained. It includes an executive summary, the surviving hypotheses (mechanism, predicted outcome, falsification criteria, experimental design), the rejected hypotheses with the reasoning behind each rejection, a section listing what the spec's YAGNI fence intentionally left out, and a recommended next action.

## The red-team loop

Every hypothesis goes through critique from an independent agent that:

- Re-runs the literature query and rejects the gap claim if it finds prior work the gap-finder missed
- Spot-checks at least three citations against the actual papers (`hf_papers paper_details`)
- Attacks the mechanism step by step, demanding citations for causal claims
- Steelmans the strongest counter-argument
- Tests whether the falsification criteria can actually be operationalized
- Tags each objection `Critical | Important | Suggestion`

If the hypothesis-smith can't satisfy the red-team within 3 rounds, it escalates to you. Rejected hypotheses are recorded in `output.md` along with the reasoning that rejected them.

## Workflow

```
/research-init
   ↓
research-brainstorming        clarifies novelty target, modalities, constraints
   ↓
writing-research-spec         writes docs/research/specs/<date>-<topic>-spec.md
   ↓                          you review + approve
writing-research-plan         writes docs/research/plans/<date>-<topic>-plan.md
   ↓                          you review + approve
/research-execute
   ↓
executing-research-plan       runs the six phases
   ↓
research-verification         evidence-based completion gate
   ↓
output.md
```

There are three approval gates before execution starts.

## Built on superpowers

MegaResearcher depends on the [`superpowers`](https://github.com/obra/superpowers) plugin and calls its skills directly. If superpowers isn't installed, `executing-research-plan` will refuse to run.

| MegaResearcher entry             | superpowers skill it invokes        |
|----------------------------------|-------------------------------------|
| `research-brainstorming`         | `brainstorming`                     |
| `writing-research-plan`          | `writing-plans`                     |
| `executing-research-plan`        | `dispatching-parallel-agents`, `subagent-driven-development` |
| `red-team` worker                | `receiving-code-review` (adapted)   |
| `eval-designer` + worker code    | `test-driven-development`           |
| Any worker that writes code      | `requesting-code-review`            |
| `research-verification`          | `verification-before-completion`    |
| Parallel baseline experiments    | `using-git-worktrees`               |
| Worker hits a bug                | `systematic-debugging`              |

## What's in the box

| | Count | Names |
|---|---:|---|
| MCP tools     | 9 | `hf_papers`, `hf_inspect_dataset`, `hf_docs_explore`, `hf_docs_fetch`, `hf_repo_files`, `github_examples`, `github_list_repos`, `github_read_file`, `web_search` |
| Subagents     | 6 | `literature-scout`, `gap-finder`, `hypothesis-smith`, `red-team`, `eval-designer`, `synthesist` |
| Skills        | 5 | `research-brainstorming`, `writing-research-spec`, `writing-research-plan`, `executing-research-plan`, `research-verification` |
| Slash commands| 3 | `/research-init`, `/research-execute`, `/share-traces` |
| Hooks         | 2 | PostToolUse doom-loop detector, SessionEnd transcript uploader |
| Vendored      | — | [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern), pinned in `tools/ml-intern.sha` |

## Install

Requirements: Claude Code, `uv`, and a Hugging Face token. The [`superpowers`](https://github.com/obra/superpowers) plugin will be auto-installed as a dependency.

Inside Claude Code:

```
/plugin marketplace add lhqezio/MegaResearcher
/plugin install megaresearcher@megaresearcher
```

That's it. Set `HF_TOKEN` in your shell before launching Claude Code (get one at https://huggingface.co/settings/tokens):

```bash
export HF_TOKEN=hf_...
# Optional, only needed for the three GitHub tools:
export GITHUB_TOKEN=ghp_...   # or: $(gh auth token)
```

Then from any project:

```
/research-init multi-modal fusion architectures for ISR
```

The MCP server syncs its Python deps on first invocation via `uv run` — no manual `uv sync` needed.

## Configuration

| Variable | Required | Purpose |
|---|:---:|---|
| `HF_TOKEN`                    | yes | HF API access (papers, datasets, docs, repo files) |
| `GITHUB_TOKEN`                | no  | GitHub API access; without it the three GitHub tools surface a clean error |
| `ML_INTERN_TRACES_REPO`       | no  | `<your-hf-username>/ml-intern-sessions` to enable trace upload to a private HF dataset |
| `ML_INTERN_TRACES_PRIVATE`    | no  | `true` (default) or `false` for that dataset's visibility |
| `MEGARESEARCHER_MAX_PARALLEL` | no  | Max parallel workers per phase, default 4 |

## Repository layout

```
MegaResearcher/
├── .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── .mcp.json
├── agents/             # 6 subagent definitions
├── skills/             # 5 skill definitions
├── commands/           # 3 slash commands
├── hooks/              # doom_loop.py + upload_traces.py + hooks.json
├── mcp/                # FastMCP server wrapping ml-intern
│   ├── server.py
│   ├── pyproject.toml
│   └── .env.example
├── tools/ml-intern/    # vendored snapshot, SHA in tools/ml-intern.sha
├── docs/architecture.md
└── tests/              # smoke tests
```

## Rules the workers enforce

- Every rejected hypothesis is recorded in `output.md` along with the reasoning that rejected it.
- Cited arxiv IDs are validated via `hf_papers paper_details`. Citations that don't resolve are dropped.
- Hypotheses without a finite experiment that could disprove them are not advanced.
- Eval-designers pre-register what result counts as support and what counts as falsification before the experiment is described.
- Workers don't cross roles: scouts produce bibliographies, smiths produce hypotheses, designers produce experiments, the synthesist composes.

## Built on

- [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern) — the research tools (HF Papers, arxiv, datasets, docs, GitHub code search, web search), the doom-loop detector, and the trace upload pipeline. Vendored as a pinned snapshot.
- [`superpowers`](https://github.com/obra/superpowers) — the discipline layer: spec-driven planning, parallel agent dispatch, verification, code review patterns. Hard dependency.
- [Claude Code](https://claude.com/claude-code) — the runtime.

## License

Apache-2.0. The vendored `tools/ml-intern/` keeps its own Apache-2.0 license; see [`tools/ml-intern/LICENSE`](tools/ml-intern/LICENSE) and [`tools/ml-intern.sha`](tools/ml-intern.sha) for the pinned upstream commit.
