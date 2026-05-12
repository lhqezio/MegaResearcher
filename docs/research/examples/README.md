# Example runs

Two real runs, committed verbatim. Each directory has the spec the user wrote, the plan that was generated from it, and the full run output — every per-worker subdirectory, the `swarm-state.yaml`, and the verification report.

## `recursive-subquadratic-fusion/`

**Question.** Can TRM-style architectural recursion ride on subquadratic-attention backbones (NSA / MoBA / DSA / SeerAttention-R) to deliver depth-of-reasoning *and* sub-quadratic context scaling jointly? Where does the formal regime of Gupta et al. (arXiv:2505.14840) say this fusion can and cannot work?

**Novelty target.** `hypothesis` (full pipeline)

**Workers fired.** 5 literature-scouts + 2 gap-finders + 6 hypothesis-smiths × 2 rounds + 6 red-team × 2 rounds + 5 eval-designers + 1 synthesist = **38 invocations**

**What survived.** 5 hypotheses (H1, H3, H4, H5, H6) with mechanism, predicted-outcome magnitudes, falsification criteria, and full experimental designs. 1 hypothesis killed (H2) and preserved in section 4 with the red-team reasoning that killed it.

→ **[Read the synthesist output](recursive-subquadratic-fusion/run-2026-05-10-0729-766039/output.md)**

## `multimodal-fusion-gap-finding/`

**Question.** Where in the 2024–2026 multi-modal fusion literature for ISR/C2 are the gaps a small team can credibly close to TRL 4–5 in twelve months using only open or synthetic data?

**Novelty target.** `gap-finding` (literature + gap-finding + synthesis only — no hypothesis / red-team / eval-designer phases)

**Workers fired.** 6 literature-scouts + 3 gap-finders + 1 synthesist = **10 invocations**

**What survived.** 4 priority gaps identified by independent convergence between two gap-finders, plus a third gap-finder that filtered the consolidated literature for TRL-4–5 buildability and produced a ranked eight-candidate shortlist with explicit licence-driven discards. Headline: three top picks spanning three deployment contexts.

→ **[Read the synthesist output](multimodal-fusion-gap-finding/run-2026-05-10-0615-0ece4e/output.md)**

---

## How to read a run directory

```
<topic>/
├── spec.md                                  # the one-paragraph question, structured
├── plan.md                                  # the swarm decomposition
└── run-<id>/
    ├── output.md                            # synthesist's deliverable — start here
    ├── swarm-state.yaml                     # which workers fired, when, with what verdict
    ├── verification-report.md               # spot-checks against the spec's success criteria
    ├── bibliography.md                      # consolidated Phase 1 output
    ├── gaps.md                              # consolidated Phase 2 output
    ├── scout-1/, scout-2/, ...              # per-scout bibliographies + verification.md
    ├── gap-finder-1/, ...                   # per-gap-finder gap lists
    ├── hypothesis-smith-N/                  # forged hypothesis + revision-1/ if red-team kicked back
    ├── red-team-N/                          # critique + objections + spot-checks + verdict
    ├── eval-designer-N/                     # experimental protocol, baselines, pre-registered decision rule
    └── synthesist/                          # final compose
```

Everything is plain markdown and YAML. Nothing is post-processed for the demo — what you see is what the swarm wrote.
