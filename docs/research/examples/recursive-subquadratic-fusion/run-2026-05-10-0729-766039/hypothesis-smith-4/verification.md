# Verification — H4 hypothesis-smith

## Required-checks table

| Check | Status | Evidence |
|---|---|---|
| Hypothesis is in if/then form | PASS | output.md §2 opens with "If we train ... then on a multi-hop long-context reasoning benchmark ..." |
| ≥ 3 falsification criteria | PASS | F1–F5 in §5; each is a metric + threshold + direction |
| Each falsification has metric + threshold + direction | PASS | F1: MuSiQue EM, ≤ 0.5, less. F2: NIAH-single EM, ≥ 1.0, greater. F3: |w_c|/std, < 1.0, less. F4: EM gap under dense top-n, ≥ 1.0, greater. F5: MuSiQue EM (R3 vs R2), ≥ 0.0, greater-or-equal. |
| Every mechanism claim has a citation | PASS | M1→2502.11089 §3,§6.1; M2→2107.05407, 1603.08983; M3→2502.11089 §3, 2507.10524; M4→2405.16039, 2310.07096; M5→2404.15574, 2412.10319; M6→2509.04475. Speculative element (functional form of c_{t,k}) is explicitly flagged in §3 last paragraph as not predicted by prior art. |
| All cited arxiv IDs resolved via paper_details | PASS | Verified during construction: 2510.04871, 2502.11089, 2107.05407, 1603.08983, 2507.10524, 2502.13189, 1807.03819, 2310.07096, 2509.04475, 2405.16039, 2602.11451, 2512.02556, 2601.10679, 2404.15574 — all returned valid metadata. 2412.10319 and 2504.17768 are cited transitively from gap-finder-1 (verified by gap-finder); flagged as inherited verification. |
| Risks section non-empty | PASS | Risks A–E in §7, each with "what the hypothesis still contributes" if it materializes |
| Halting-vs-sparse-routing distinction explicit | PASS | §1 restates the gap as "depth halting" (ACT/PonderNet/UT/MoR/LoopFormer) vs "attention sparse routing" (NSA/MoBA/DSA); §2 names four distinct coupling regimes (R0–R3); §3 M1 grounds why the two are coupled at the residual stream |
| Non-additive prediction (routers must INTERACT) | PASS | §2 "non-additive: R2's improvement is not 'halting helps + sparsity helps' but specifically 'halting decisions made aware of which blocks were dropped.'" F1 directly tests non-additivity (R2 must beat R1 by ≥ 0.5 EM, where R1 = both routers running independently). F4 tests that the gap collapses when there is nothing to drop. |
| One sparse pattern chosen | PASS | NSA (2502.11089) — selection branch §3 cited as the operational target for c_{t,k} |
| Cheaper-falsification path included | PASS | §8: full path (frozen NSA + recursive head fine-tune, ~1 GPU-week) and smoke test (inference-only correlation, < 1 GPU-day) |
| Magnitude predictions specific | PASS | §4 table: R2 = R0 + 1.5–3.5 EM on MuSiQue, R0 + 1.5–3.0 EM on FRAMES, R2 ≤ R0 ± 0.5 EM on single-hop NIAH; R3 = R2 − 1.0 to −2.5 EM. Reasoning grounded in SCBench / Sparse Frontier 2-6 EM swing range and MoR 1-3 EM gain range. |

## Self-critical notes

1. **Speculative element** — the precise functional form of the coverage signal c_{t,k} (raw mass vs log-mass vs KL) is not predicted by prior art. This is flagged in §3 and addressed by the ablation in §6 and Risk E. Three concrete forms must be tried; if all three fail, M3 is falsified, not the broader hypothesis. This is consistent with the discipline rule "if you cannot ground the mechanism in cited work, say so explicitly."

2. **The 0.5 EM threshold in F1 is tight.** A noise floor of 0.5 EM on MuSiQue at this scale is realistic per Sparse Frontier 2504.17768, but a single seed could fail F1 by chance. The required-experiments §6 implicitly requires ≥ 3 seeds; the verification check on F3 already specifies 3 seeds. We assume eval-designer will lock in ≥ 3 seeds for F1 too.

3. **The R3 (shared-router) prediction (§4 table, R3 < R2 by 1.0–2.5 EM) is the weakest part of the hypothesis.** The mechanism (M4) leans on analogy to MoEUT/SUT (sparse-over-experts), not on direct prior art for sparse-over-token-positions with halting. Risk E covers this only loosely. If F5 fires, it is informative about the *direction* of coupling, not just an isolated null.

4. **Token-position vs token-time distinction.** NSA selects key *blocks* (groups of token positions). The hypothesis treats "block dropped" and "key block missing" as equivalent for the halting calculation, which is correct because halting is computed after attention output is summed over selected blocks. This is consistent with NSA §3.2 (output is Attn(q_t, K̃_t, Ṽ_t) where K̃ is restricted to selected blocks).

## Final status

PASS — submit for red-team review.

Caveats forwarded to red-team:
- The functional-form question for c_{t,k} is speculative and acknowledged.
- F1's threshold of 0.5 EM may need to be revisited if MuSiQue noise floor at the chosen scale is empirically larger.
- Risk D (scale-dependence per Sparse Frontier) is the most likely route by which all results become null at affordable scales; the cheaper-falsification path partially mitigates by using a production-scale frozen backbone (DSA / DeepSeek-V3.2 2512.02556).
