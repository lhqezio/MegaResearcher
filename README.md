# MegaResearcher

> A research team in your terminal. Spec-driven. Adversarially critiqued. Hypotheses with falsification criteria. Citations that actually exist.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-orange)](https://github.com/lhqezio/MegaResearcher)
[![Built on ml-intern](https://img.shields.io/badge/built%20on-ml--intern-yellow)](https://github.com/huggingface/ml-intern)
[![Powered by superpowers](https://img.shields.io/badge/powered%20by-superpowers-purple)](https://github.com/obra/superpowers)

---

Turn a one-paragraph research question into a **defended research direction**.

MegaResearcher dispatches a swarm of seven specialist subagents — literature scouts, gap-finders, hypothesis-smiths, an adversarial red-team, eval-designers, and a synthesist — and returns a research-direction document containing surviving hypotheses, full experimental designs, falsification criteria, and a transparent audit trail of every idea that was killed and why.

```
                       ┌──────────────────────────────────────────┐
       Your spec  ───▶ │   research-swarm orchestrator            │ ───▶  output.md
                       └──────────────────────────────────────────┘            │
                                          │                                    │
              ┌───────────────────────────┼───────────────────────────┐        │
              ▼                           ▼                           ▼        │
        literature-scout            gap-finder                 hypothesis-smith │
              │                           │                           │        │
              └───────────┬───────────────┴───────────┬───────────────┘        │
                          ▼                           ▼                        │
                       red-team  ◀──── critique loop ─────┐                    │
                          │                               │                    │
                          │      revise (cap: 3 rounds)   │                    │
                          └────▶ hypothesis-smith ────────┘                    │
                          │                                                    │
                          ▼ APPROVE                                            │
                    eval-designer                                              │
                          │                                                    │
                          ▼                                                    │
                     synthesist  ─────────────────────────────────────────────▶┘
```

---

## Why this is different

|                                          | Just chat with Claude | ml-intern alone | MegaResearcher |
|------------------------------------------|:--------------------:|:--------------:|:--------------:|
| Read papers, datasets, code              |          ✓           |       ✓        |       ✓        |
| Spec-driven workflow with review gates   |          —           |       —        |       ✓        |
| Parallel specialist agents               |          —           |       —        |       ✓        |
| Adversarial red-team critique loop       |          —           |       —        |       ✓        |
| Falsification criteria enforced          |          —           |       —        |       ✓        |
| Audit trail of rejected ideas            |          —           |       —        |       ✓        |
| Citations independently re-verified      |          —           |    partial     |       ✓        |
| Doom-loop detection across the swarm     |          —           |       ✓        |       ✓        |

This isn't "Claude with extra tools." It's a research lab on a chip.

---

## What you actually get

After running `/research-execute` against an approved plan, your project ends up with:

```
docs/research/
├── specs/
│   └── 2026-05-10-multimodal-fusion-spec.md     ← your one-paragraph spec
├── plans/
│   └── 2026-05-10-multimodal-fusion-plan.md     ← swarm decomposition
└── runs/
    └── 2026-05-10-1430-a3f9b2/
        ├── output.md                             ← THE deliverable
        ├── swarm-state.yaml                      ← what happened, when
        ├── verification-report.md                ← spot-checks + verdict
        ├── bibliography.md                       ← consolidated Phase 1 output
        ├── gaps.md                               ← consolidated Phase 2 output
        ├── literature-scout-1/
        │   ├── output.md, manifest.yaml, verification.md
        ├── literature-scout-2/  ...
        ├── gap-finder-1/  ...
        ├── hypothesis-smith-1/  (with revision history if red-team kicked back)
        ├── red-team-1/    (verdict + objections + spot-checks)
        ├── eval-designer-1/
        └── ...
```

The headline `output.md` is self-contained and proposal-ready. It includes:

- **Executive summary** — question, novelty target, headline findings, bottom-line recommendation
- **Surviving hypotheses** — each with mechanism, predicted outcome with magnitude, falsification criteria, and a full experimental design (datasets, baselines, metrics, decision rule, compute budget)
- **Rejected and killed hypotheses** — with the reasoning that killed each one (mandatory transparency — hidden rejections destroy the swarm's epistemic value)
- **What we did NOT explore** — explicit reflection of the spec's YAGNI fence
- **Recommended next actions** — names a specific hypothesis, not "more research is needed"

---

## The six phases

```
Phase 1   literature-scout      parallel, one per sub-topic in the plan
Phase 2   gap-finder            parallel, partitions the consolidated bibliography
Phase 3   hypothesis-smith      parallel, one per identified gap
Phase 4   red-team CRITIQUE     sequential per hypothesis, 3-revision cap
                                (THIS is where novelty quality gets enforced)
Phase 5   eval-designer         parallel, one per surviving hypothesis
Phase 6   synthesist            single, composes the final research-direction doc
```

**Phase 4 is the load-bearing one.** Every hypothesis must survive an adversarial critique by an independent agent that:

- Re-runs the literature query and rejects the gap claim if it finds prior work the gap-finder missed
- Spot-checks at least three citations against the actual papers (`hf_papers paper_details`)
- Attacks the mechanism step-by-step, demanding citations for every causal claim
- Steelmans the strongest counter-argument
- Tests whether the falsification criteria can actually be operationalized into a finite experiment, or whether they're unfalsifiable hand-waving
- Tags every objection `Critical | Important | Suggestion`

If the hypothesis-smith can't satisfy the red-team in 3 rounds, the hypothesis is escalated to you. If it gets killed entirely, it appears in the audit trail with the lesson it taught — never silently dropped.

---

## The five-step spec-driven chain

```
/research-init
   ↓
research-brainstorming   →   clarifies novelty target, modalities, constraints
   ↓                          (wraps superpowers:brainstorming)
writing-research-spec    →   produces docs/research/specs/<date>-<topic>-spec.md
   ↓                          USER REVIEWS + APPROVES
writing-research-plan    →   produces docs/research/plans/<date>-<topic>-plan.md
   ↓                          (wraps superpowers:writing-plans + swarm decomposition)
                              USER REVIEWS + APPROVES
/research-execute
   ↓
[research-swarm runs the six phases]
   ↓
research-verification    →   evidence-based completion gate
   ↓                          (wraps superpowers:verification-before-completion)
output.md ready
```

Three explicit user-approval gates between brainstorm and execute. No silent token-burn.

---

## Deep integration with superpowers

MegaResearcher hard-depends on the [`superpowers`](https://github.com/obra/superpowers) skill library and invokes its skills as runtime primitives:

| MegaResearcher entry             | Invokes superpowers skill                     |
|----------------------------------|-----------------------------------------------|
| `research-brainstorming`         | `brainstorming`                               |
| `writing-research-plan`          | `writing-plans`                               |
| `executing-research-plan`        | `dispatching-parallel-agents`                 |
| `research-swarm` orchestrator    | `subagent-driven-development`                 |
| `red-team` worker                | `receiving-code-review` (adapted)             |
| `eval-designer` + worker code    | `test-driven-development`                     |
| Any worker that writes code      | `requesting-code-review`                      |
| `research-verification`          | `verification-before-completion`              |
| Parallel baseline experiments    | `using-git-worktrees`                         |
| Worker hits a bug                | `systematic-debugging`                        |

The result: rigor borrowed from a battle-tested skill library, applied automatically to every research artifact. You don't pick TDD vs. no-TDD per worker; the workers know.

---

## What's bundled

| | Count | Names |
|---|---:|---|
| **MCP tools** | 9 | `hf_papers`, `hf_inspect_dataset`, `hf_docs_explore`, `hf_docs_fetch`, `hf_repo_files`, `github_examples`, `github_list_repos`, `github_read_file`, `web_search` |
| **Subagents** | 7 | `research-swarm` + `literature-scout`, `gap-finder`, `hypothesis-smith`, `red-team`, `eval-designer`, `synthesist` |
| **Skills** | 5 | `research-brainstorming`, `writing-research-spec`, `writing-research-plan`, `executing-research-plan`, `research-verification` |
| **Slash commands** | 3 | `/research-init`, `/research-execute`, `/share-traces` |
| **Hooks** | 2 | PostToolUse doom-loop detector, SessionEnd transcript uploader |
| **Vendored upstream** | — | [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern), pinned to a known-good SHA |

---

## Quick start

**Requirements:** Claude Code · `uv` · the [`superpowers`](https://github.com/obra/superpowers) plugin · a Hugging Face token

```bash
# 1. Clone
git clone https://github.com/lhqezio/MegaResearcher.git ~/MegaResearcher

# 2. Install MCP server deps
cd ~/MegaResearcher/mcp && uv sync

# 3. Configure tokens (HF_TOKEN required, GITHUB_TOKEN optional)
cp .env.example .env && $EDITOR .env
# Or set them in your shell (~/.zshrc):
#   export HF_TOKEN="hf_..."
#   command -v gh >/dev/null && export GITHUB_TOKEN="$(gh auth token)"

# 4. Wire into Claude Code (add to ~/.claude/settings.json)
```

```jsonc
// ~/.claude/settings.json
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

Restart Claude Code. Then in any project:

```
/research-init multi-modal fusion architectures for ISR
```

The brainstorming skill takes over from there.

---

## Configuration

| Variable | Required | Purpose |
|---|:---:|---|
| `HF_TOKEN`                    | ✓ | HF API access — papers, datasets, docs, repo files |
| `GITHUB_TOKEN`                | — | GitHub API access; without it, three GitHub tools surface a clean error |
| `ML_INTERN_TRACES_REPO`       | — | Set to `<your-hf-username>/ml-intern-sessions` to enable session-trace upload to a private HF dataset |
| `ML_INTERN_TRACES_PRIVATE`    | — | `true` (default) or `false` for trace dataset visibility |
| `MEGARESEARCHER_MAX_PARALLEL` | — | Max parallel workers per phase (default 4); higher = faster + more token spend |

---

## Repository layout

```
MegaResearcher/
├── .claude-plugin/
│   ├── plugin.json                  # plugin manifest
│   └── marketplace.json             # directory-source marketplace metadata
├── .mcp.json                        # MCP server registration
├── agents/                          # 7 subagent definitions (.md each)
├── skills/                          # 5 skill definitions (one dir each)
├── commands/                        # 3 slash command definitions
├── hooks/                           # doom_loop.py + upload_traces.py + hooks.json
├── mcp/                             # FastMCP server wrapping ml-intern
│   ├── server.py
│   ├── pyproject.toml
│   └── .env.example
├── tools/ml-intern/                 # vendored snapshot, pinned SHA in tools/ml-intern.sha
├── docs/architecture.md             # contributor docs
└── tests/                           # smoke tests for hooks + MCP server
```

---

## Design philosophy

**Audit trail is non-negotiable.** Every rejected hypothesis appears in the synthesist's final document with the lesson it taught. Hidden rejections destroy the swarm's epistemic value.

**Citations resolve or do not exist.** Every cited arxiv ID gets validated via `hf_papers paper_details`. No invented citations. Ever.

**Falsification criteria are required.** A hypothesis without a finite experiment that could disprove it is not a hypothesis — it's vibes. The plugin will refuse to advance it.

**Pre-registration of decision rules.** Eval-designers must state in advance what result would constitute support and what would constitute falsification. Post-hoc thresholds are how plausible-but-wrong findings survive.

**Workers stay in their lanes.** Scouts produce bibliographies, smiths forge hypotheses, designers design experiments, the synthesist composes — no role poaches another's job.

---

## Built on

- [**huggingface/ml-intern**](https://github.com/huggingface/ml-intern) — provides the underlying research tools (HF Papers, arxiv, datasets, docs, GitHub code search, web search), the doom-loop detector, and the session-trace upload pipeline. Vendored as a pinned snapshot.
- [**superpowers**](https://github.com/obra/superpowers) — provides the discipline layer: spec-driven planning, parallel agent dispatch, verification-before-completion, code review patterns. Hard dependency.
- [**Claude Code**](https://claude.com/claude-code) — the runtime.

---

## License

Apache-2.0. The vendored `tools/ml-intern/` retains its own Apache-2.0 license — see [`tools/ml-intern/LICENSE`](tools/ml-intern/LICENSE) and [`tools/ml-intern.sha`](tools/ml-intern.sha) for the pinned upstream commit.
