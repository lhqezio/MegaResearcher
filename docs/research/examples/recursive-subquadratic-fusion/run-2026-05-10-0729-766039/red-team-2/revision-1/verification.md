# Verification — Red-Team Round 2 (Revision 1)

Per receiving-code-review discipline: technical rigor and verification, not performative agreement.

## Independent literature queries (≥3 required; ran 4)

| # | Query | Result | Outcome |
|---|---|---|---|
| 1 | `depth recurrent Mamba SSM continued pretraining retrofit` | LBMamba, Motion Mamba, MambaMIM, MambaTrack, Hidden Attention of Mamba — none retrofit depth-recurrence onto Mamba for reasoning | Gap survives. |
| 2 | `looped transformer Mamba SSM math reasoning depth iteration` | Computational Limits SSM (2412.06148), Reasoning with Latent Thoughts (2502.17416), Adaptive Loops and Memory in Transformers (2603.08391), M1 (2504.10449). M1 is closest but is output-CoT not depth-recurrent retrofit. | Gap survives. |
| 3 | `TRM HRM SSM linear RNN long chain reasoning Lean` | TRM (2510.04871), HRM (2506.21734), Resurrecting Recurrent Neural Networks, M1, Longhorn — no SSM-on-Lean paper | Gap survives. |
| 4 | `state space model formal theorem proving Lean miniF2F PutnamBench` | Hilbert, PutnamBench, Lean-STaR, APOLLO, Seed-Prover 1.5 — all Transformer/LLM, no SSM | Gap survives. |
| 5 | `TRM tiny recursive model two state z_L z_H deep supervision Mamba retrofitting` | TRM, Mamba-3, Mamba time series — no retrofit of TRM onto SSM | Gap survives. |

**Conclusion on gap:** Narrow gap (TRM-style and FPR-style on Mamba evaluated on Lean) survives. The smith correctly notes novelty is reduced by Retrofitted Recurrence and Think-at-Hard, but neither covers Mamba+TRM-style+Lean.

## Citation spot-checks (≥3 required; ran 3 + verified retraction of 2)

### Spot-check 1: arXiv:2511.07384 (Retrofitted Recurrence) — recipe applies to which architecture?

Read §3 (Model Definition, Model Surgery) and §4 (Training Recurrent Language Models §4.1, §4.2, §4.3.1) verbatim.

**Claim by smith:** "this is no longer a novel training recipe — it is the published recipe re-applied to a Mamba substrate."

**Reality:** The paper uses the Geiping/Huginn architecture: Prelude P (transformer layers including embeddings) → Recurrent Block R (set of unique transformer blocks; iterated as s_i = R(e, s_{i−1}) for i ∈ {1, ..., r}) → Coda C (transformer layers including unembeddings). It is applied only to Transformer substrates: TinyLlama, OLMo, Llama-3.2. The recipe details (Poisson-Lognormal r-sampling, Muon optimizer, curriculum from k → 32 mean recurrences, truncated 8-step backprop) are designed for this single-state iteration on a Transformer.

**Verdict:** The smith's claim is a citation misuse. The recipe is being extended to (i) a different recursion modality (TRM-style, FPR-style) AND (ii) a different substrate (Mamba). This is the same shape of error as the round-1 C1 (claiming a paper supports something it doesn't). Flagged as new C1 (revision-1) in the critique.

### Spot-check 2: arXiv:2510.04871 (TRM) — what does the deep supervision protocol require?

The smith claims "TRM-style block-level recursion" can be retrofitted under Retrofitted Recurrence's recipe. TRM §4 specifies a two-state z_L (low-level) / z_H (high-level "answer estimate") iteration with f_L applied n times, then f_H once, with weights shared (per §4.3 "Single network"). TRM §2.4 specifies deep supervision (cross-entropy at each recursion step). Retrofitted Recurrence has neither two-state semantics nor deep supervision — it has single-state s_i = R(e, s_{i−1}) and final-step cross-entropy.

**Verdict:** Confirms that the smith is conflating two different recipes. Either the smith intends to keep TRM's deep supervision (which contradicts "the recipe") or drop it (which contradicts "TRM-style"). Unspecified in the revision.

### Spot-check 3: arXiv:2503.10799 (FPR) — Stage 1 prediction grounding at L=200?

Read §4 (Fixed-Point Mamba), Appendix D (Evaluation, §D.2 State Tracking), Appendix E (E.2 Long-Range State-Tracking).

The main A_5/S_5 setup (FPR Fig 4): train length 16, eval lengths 2 through 50.
The long-range setup (FPR §E.2 Fig 9): train length 128, eval lengths [2, 512].

The smith's prediction table (§4 of revision-1) uses train length = 16 and eval lengths {5, 10, 50, 200}. This puts L=200 outside the published range of the train-length-16 setup. The smith claims "FPR-Mamba ≥ 70% at L=50 per FPR Fig 4" which is plausible but not directly verified in numbers; the L=200 prediction is uncalibrated.

**Verdict:** The smith correctly marks L=200 as TBD for FPR-Mamba, but then asserts ≥10 absolute points difference at L=200 as the gating prediction. Threshold is arbitrary. Flagged as I1 (revision-1) in the critique.

### Spot-check 4 (round-1 retraction verification): FPR contraction-vs-function-class — properly retracted?

The §3 of revision-1 now states: "expressivity-class claims about which can or cannot represent permutation composition are NOT made: FPR-Mamba demonstrably solves S_5 (FPR §4.4 Fig 4)". This is a clean retraction of the round-1 C1 misreading.

**Verdict:** Round-1 C1 is properly addressed.

### Spot-check 5 (round-1 retraction verification): TC⁰-escape — properly retracted?

§3 now states: "K passes of a TC⁰ block at fixed K stays in TC⁰. Both variants gain expressivity at constant K only via the *new parameterization* (FPR's dense Q⁻¹Λ; TRM's iterated z_H), not via depth-buys-expressivity arguments à la CoT-Solves-Serial." This is a clean retraction of the round-1 C2 misreading.

**Verdict:** Round-1 C2 is properly addressed.

## Verdict-vs-severity consistency check

- Verdict: REJECT (revision-2)
- Critical count: 1 (new defect: arXiv:2511.07384 recipe misapplied to TRM-style/FPR-style/Mamba)
- Important count: 4 (uncalibrated 10-point threshold; ratio fragility under saturation/floor; F6 fallback under-specified; §2 should lead with interaction prediction)
- Suggestion count: 3
- Gap claim: survives

A REJECT with 1 Critical and 4 Important is consistent. An APPROVE would be invalid given the unaddressed C-grade defect on the new training-recipe claim.

## Discipline checks

- [x] Steelman constructed (§5: "compute-scaling claim dressed up as architecture comparison"; counter-counter via interaction prediction).
- [x] Spot-checked ≥3 citations (5 here).
- [x] Ran ≥3 independent literature queries (5 here).
- [x] Verdict severity matches: 1 Critical → REJECT (not APPROVE).
- [x] KILL not invoked: gap survives, falsification suite operational, synthetic probe gate provides cheap kill — these are repairable defects, not irrecoverable ones.
- [x] No performative agreement (acknowledged real progress on round-1 objections); no performative skepticism (specific, citationally-grounded objections).
