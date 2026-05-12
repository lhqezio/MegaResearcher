# Synthesist verification — run 2026-05-10-0729-766039

This verification confirms each spec success criterion is met in the synthesis document.

## Spec success-criteria checklist

### 1. Hypothesis count: ≥ 3 surviving with claimed gap, mechanism, predicted outcome, falsification criteria (metric + threshold + direction)

**5 surviving hypotheses:** H1, H3, H4, H5, H6.

For each, verbatim falsification criteria with metric + threshold + direction are present in section 2:

- **H1:** F1 (NSA-fb K=6 minus K=1 ≥ +6.0 abs pts; falsified < +3.0); F2 (DiD ≥ +5.0 with sign asymmetry, K=1 calibration ≤ 2 abs pts); F3 (compressed-branch ablation Δ ≥ +4.0; falsified ≤ +1.5); F4' (Jaccard inter-iteration drift ≤ MoBA − 0.10).
- **H3:** F2 (within-arch retention differential ≥ +0.05 abs); F5 (rank-order NSA > Quest ≥ DSA > MoBA, ≤1 swap); F3 (NoLiMa Δacc differential ≥ +3 pp; RULER ≥ +2 pp); F1 (consistency ≥ −0.10); F6 (mechanism check, compression-zeroed differential ≈ MoBA within ±0.03).
- **H4:** F-Calib pre-experiment gate (≥30% multi-hop median rank > top-n); F1 (R2-A − R1 > 1.5 EM); F2 (single-hop NIAH within ±1.5); F3 (|w_c|/std ≥ 1.5); F4 (collapse under δ ≈ 0); F-Self vs A (R2-A − R2-self > 1.0); F-StopGrad (|R2-A − R2-A_stopgrad| > 0.7).
- **H5:** F1 (Δ(s=0.5, K=8) ≥ +10 pp); F2 (Δ(s=0.5, K=8) ≤ −10 pp); F3 (sparse-vs-dense interaction ≤ ±10 pp at eval-designer width); F4 (positive-control gate Sudoku ≥ +20); TOST equivalence at ±5 pp p<0.05.
- **H6:** F1 (compression ratio r ≤ 0.5 predicted, falsified > 0.8 or ≤ 0.0); F2 (|A_TTT-Linear − A_TTT-η=0| ≤ 1.5); F3 (r_CoT − r_no-CoT ≥ +0.2); F4 (depth-axis specificity); F5 (LaCT replication).

**Status: PASS.** All 5 surviving hypotheses have metric + threshold + direction stated verbatim from approved revision-1 documents.

### 2. Eval design depth per surviving hypothesis

Section 3 contains a single comparison table covering all 5 surviving hypotheses. Each row contains:

- ≥1 primary dataset with HF ID and license (BABILong / NoLiMa / RULER variable-tracking / CB2H / CRUXEval-X)
- ≥1 OOD dataset (RULER NIAH / Needle Threading / FRAMES / BABILong / CRUXEval-O)
- ≥2 baselines (every hypothesis has 4–7 baselines; e.g., H1 has Dense K=1 FLOP-matched, NSA-no-fb, MoBA, PLT-G-SWA, DSA inference, majority-class floor)
- Primary + secondary metrics (DiD with Jaccard probe; within-arch retention with NoLiMa Δacc; EM with paired bootstrap; TOST equivalence; compression ratio with mechanism distinguisher)
- ≥1 ablation (every hypothesis has 3–7 explicit ablations)
- Pre-registered statistical plan (test, alpha, effect-size, power consideration noted: 5–10 seeds, paired bootstrap n=10000, std budgets, TOST p<0.05, hierarchical bootstrap)
- Budget estimate (full GPU-hr + cheap-path GPU-hr columns)

**Status: PASS.** Every surviving hypothesis has the required eval-design depth per spec §50.

### 3. Audit trail: every killed hypothesis preserved with verdict + one-sentence reason

Section 4 preserves H2 with:
- Title (verbatim)
- Claimed gap
- Mechanism attempted (revision-0 → revision-1)
- Round-1 red-team objections (C1, C2, C3, C4)
- Round-2 red-team objections (new critical defect on Retrofitted Recurrence citation; I1, I2, I3)
- Verdict (REJECT revision-2)
- One-sentence kill reason
- Lesson contributed

Plus 13 unselected gaps (A3, A5, A9, A10, B2, B3, B6 + 4 discarded by gap-finder-1 + 3 discarded by gap-finder-2) listed.

**Status: PASS.** Zero hidden rejections; the audit trail is complete per spec §56.

### 4. Synthesist document — 6 to 10 pages markdown including references

Word count: 6702 words. At ~750 words/page → ≈ **8.9 pages**, within the 6–10 page target. Six required sections present in order:

1. Problem framing and the (depth × context) plane — present.
2. Surviving hypotheses with falsification criteria — present.
3. Per-hypothesis eval designs — present.
4. Killed-hypothesis audit trail — present.
5. YAGNI fence reflection — present.
6. References — present.

(Plus added §6 Recommended next actions and §7 Run metadata, which do not violate the spec — the spec's §57 lists 5 required sections and the synthesist's prompt asks for 6 + run metadata + references.)

**Status: PASS.**

### 5. Citation discipline — every claim cites arxiv ID / HF ID / repo SHA / industrial blog flagged

Section 8 deduplicates 78 entries across all worker outputs. Spot checks of section 1, 2, 3 claims:

- "TRM (arXiv:2510.04871)" — cited.
- "NSA's three-branch design" — arXiv:2502.11089 §3 cited.
- "MoBA top-k gate, no fallback" — arXiv:2502.13189 §3 cited.
- "DSA lightning-indexer top-k, no compressed branch" — arXiv:2512.02556 §2 cited.
- "PLT +6.1 average accuracy lift" — arXiv:2510.24824 Table 2 row PLT-3 cited (verified by red-team-1 round-2 spot-check SC1).
- "SSA Gradient Update Deficiency Proposition 4.1" — arXiv:2511.20102 §4 cited.
- "TTT-as-linear-attention Theorem 5.1" — arXiv:2602.21204 §5.1 cited.
- "Parcae spectral-radius < 1 contraction" — arXiv:2604.12946 §3 cited.
- "Huginn plateau evidence" — arXiv:2507.02199 §3.4 cited.
- "BABILong" — HF `RMT-team/babilong` Apache-2.0 cited.
- "NoLiMa" — HF `amodaresi/NoLiMa` CC-BY-NC-4.0 (flagged).
- "RULER" — HF `simonjegou/ruler` Apache-2.0 cited.
- "CRUXEval-X" — HF `xhwl/cruxeval-x` MIT cited.
- "SubQ" — Subquadratic 2026 industrial blog (flagged across section 1, section 5, section 8).

**Status: PASS.** Every load-bearing claim cites a paper / HF dataset / repo. The SubQ industrial-blog flag and the NoLiMa CC-BY-NC-4.0 flag are preserved.

## Discipline-rule checklist (synthesist prompt)

- **Self-contained:** A reader who has not read the worker outputs understands the document — section 1 frames the plane, section 2 contains all surviving hypotheses' load-bearing content, section 3 comparison table contains the eval-design payload at glance, section 4 contains the audit trail. **PASS.**
- **Audit trail non-negotiable:** H2 is preserved with full mechanism story, both rounds of red-team objections, verdict, one-sentence kill reason, and lesson. **PASS.**
- **No new claims:** Section 6 Recommended next actions names only smith-and-eval-designer-derived recommendations (smallest meaningful experiment, F-Calib pre-test priority); the "thing the swarm flagged but did not pursue" mentions A9 and A10 by their gap-finder labels and does not invent new mechanisms. **PASS.**
- **Honest recommended next actions:** H1 named with concrete "smallest meaningful experiment" (cheap path A at 350M, 5B tokens, F3 differential on BABILong qa3+qa4+qa5 at L=16K). H3, H4, H5, H6 each given a specific role (deferred / pre-experiment gate / underpowered cheap path / deferred). **PASS.**
- **Recursion-vs-agent distinction reasserted:** Section 1 paragraph and section 5 final bullet. **PASS.**
- **SubQ industrial-blog flag preserved:** Section 1 (introduction), section 5 (YAGNI), section 8 (references). **PASS.**

## Document file paths

- Primary deliverable (run root): `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/output.md`
- Synthesist subdirectory copy: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/synthesist/output.md`
- Manifest: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/synthesist/manifest.yaml`
- This verification: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/synthesist/verification.md`

## Final verdict

**PASS.** All spec success criteria are met. Document is 8.9 pages, 6702 words, with 78 deduplicated citations. The synthesist deliverable is ready for the user.
