# MegaResearcher

A Claude Code plugin that runs a small swarm of research subagents against a research question and produces a defended research direction: hypotheses with falsification criteria, experimental designs, and an audit trail of what got killed and why.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-orange)](https://github.com/lhqezio/MegaResearcher)
[![Built on ml-intern](https://img.shields.io/badge/built%20on-ml--intern-yellow)](https://github.com/huggingface/ml-intern)
[![Powered by superpowers](https://img.shields.io/badge/powered%20by-superpowers-purple)](https://github.com/obra/superpowers)

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

`output.md` is self-contained. It contains an executive summary, the surviving hypotheses (mechanism, predicted outcome, falsification criteria, experimental design), the killed hypotheses with the reasoning that killed them, an explicit "what we did NOT explore" section reflecting the spec's YAGNI fence, and a recommended next action that names a specific hypothesis.

## The red-team loop

This is the part that matters. Every hypothesis has to survive critique from an independent agent that:

- Re-runs the literature query and rejects the gap claim if it finds prior work the gap-finder missed
- Spot-checks at least three citations against the actual papers (`hf_papers paper_details`)
- Attacks the mechanism step by step, demanding citations for causal claims
- Steelmans the strongest counter-argument
- Tests whether the falsification criteria can actually be operationalized
- Tags each objection `Critical | Important | Suggestion`

If the hypothesis-smith can't satisfy the red-team within 3 rounds, it escalates to you. Killed hypotheses don't disappear — they show up in `output.md` with the lesson they taught.

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

Three approval gates between brainstorm and execute, so a swarm run doesn't kick off without you signing off.

## Built on superpowers

MegaResearcher hard-depends on the [`superpowers`](https://github.com/obra/superpowers) plugin and calls its skills as runtime primitives. If superpowers isn't installed, `executing-research-plan` refuses to run.

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

Requirements: Claude Code, `uv`, the [`superpowers`](https://github.com/obra/superpowers) plugin, and a Hugging Face token.

```bash
git clone https://github.com/lhqezio/MegaResearcher.git ~/MegaResearcher
cd ~/MegaResearcher/mcp && uv sync

cp .env.example .env && $EDITOR .env
# HF_TOKEN required; GITHUB_TOKEN optional
```

Wire it into Claude Code in `~/.claude/settings.json`:

```jsonc
{
  "enabledPlugins": {
    "superpowers@claude-plugins-official": true,
    "megaresearcher@megaresearcher": true
  },
  "extraKnownMarketplaces": {
    "megaresearcher": {
      "source": { "source": "directory", "path": "/Users/you/MegaResearcher" }
    }
  }
}
```

Restart Claude Code, then from any project:

```
/research-init multi-modal fusion architectures for ISR
```

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

## Discipline rules

These are baked into the workers and the verification skill, not aspirational:

- **Audit trail.** Every killed hypothesis ends up in `output.md` with the reasoning that killed it. No silent rejections.
- **Citations resolve.** Every cited arxiv ID gets validated via `hf_papers paper_details`. If it doesn't resolve, it doesn't get cited.
- **Falsification required.** A hypothesis without a finite experiment that could disprove it doesn't get advanced.
- **Pre-registered decision rules.** Eval-designers state up front what counts as support and what counts as falsification.
- **Workers stay in their lanes.** Scouts produce bibliographies, smiths forge hypotheses, designers design experiments, the synthesist composes.

## Built on

- [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern) — the research tools (HF Papers, arxiv, datasets, docs, GitHub code search, web search), the doom-loop detector, and the trace upload pipeline. Vendored as a pinned snapshot.
- [`superpowers`](https://github.com/obra/superpowers) — the discipline layer: spec-driven planning, parallel agent dispatch, verification, code review patterns. Hard dependency.
- [Claude Code](https://claude.com/claude-code) — the runtime.

## License

Apache-2.0. The vendored `tools/ml-intern/` keeps its own Apache-2.0 license; see [`tools/ml-intern/LICENSE`](tools/ml-intern/LICENSE) and [`tools/ml-intern.sha`](tools/ml-intern.sha) for the pinned upstream commit.
