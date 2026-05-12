# Verification — Hypothesis H3 (retrieval-head re-formation under sparse attention × recursion)

Per `superpowers:verification-before-completion` discipline. Each check has the verbatim claim being tested, the evidence, and PASS/FAIL.

## C1. Hypothesis statement is in if/then form
**Check.** §2 opens with "**If** a TRM-style architectural recursion … is layered on a transformer backbone whose attention has been replaced with NSA …, **then** retrieval-head retention score … will be ≥ 0.85× the dense-attention retention baseline …, **and** the recovery will be monotonic in K …" The contrast clause for MoBA is also if/then-structured.
**Result.** PASS.

## C2. At least 3 falsification criteria, each genuinely falsifiable, each with metric+threshold+direction
**Check.**
- F1: metric = retention(K=4) − retention(K=1) on NSA; threshold = +0.05 absolute; direction = increase required. Falsifiable by direct experiment.
- F2: metric = (NSA Δretention K1→K4) − (MoBA Δretention K1→K4); threshold = +0.05 absolute; direction = NSA recovers more. Falsifiable.
- F3: metric = NoLiMa accuracy delta and EverMemBench-S evidence-access metric; threshold = ≥ 3-point NSA improvement K1→K4 OR < 3-point MoBA improvement; direction specified for each. Falsifiable.
- F4 (optional): metric = top-k coverage of needle block at pass 2 vs pass 1; threshold = ≥ 5% relative; direction = increase. Falsifiable (mechanism check, supplementary).
Three required; four supplied. Each has an experiment that could return a number that triggers falsification.
**Result.** PASS.

## C3. Every mechanism claim has a citation
**Check.** §3 has three explicit mechanism claims:
- M1 (retrieval-head copy-paste circuit, argmax metric, universality across model families) — cited to arXiv:2404.15574 §2, §3.1, Table 1.
- M2 (NSA compression branch covers all blocks; MoBA has no fallback) — cited to arXiv:2502.11089 §2.2 + §2.3 + Figure 2; arXiv:2502.13189 §2.2 Eq. 5.
- M3 (TRM K-pass recursion refines latent state, fresh query at each pass) — cited to arXiv:2510.04871 §3, §4.
- Composition (M1+M2+M3 → predicted outcome): explicitly labeled as composition, no new uncited mechanism claim introduced.
- Subsidiary G&A claim — cited to arXiv:2602.11374.
No mechanism statement lacks a citation. The "speculative" subsidiary mechanism (compression-branch granularity may be too coarse — R4) is explicitly flagged in the Risks section, not hidden in the Mechanism section.
**Result.** PASS.

## C4. All cited arxiv IDs resolve via hf_papers paper_details
**Check.** All 13 arxiv IDs in §8 were verified live during this session via `mcp__plugin_megaresearcher_ml-intern__hf_papers` `paper_details`:
- 2404.15574 — Retrieval Head — verified.
- 2502.11089 — NSA — verified.
- 2502.13189 — MoBA — verified.
- 2510.04871 — TRM — verified.
- 2602.11374 — Retrieval-Aware Distillation for Transformer-SSM Hybrids — verified.
- 2407.15891 — RazorAttention — verified.
- 2502.05167 — NoLiMa — verified (license non-commercial — flagged in manifest).
- 2410.04422 — Hyper-multi-step — verified.
- 2504.17768 — Sparse Frontier — verified.
- 2506.08889 — SeerAttention-R — verified.
- 2406.10774 — Quest — verified.
- 2512.02556 — DeepSeek-V3.2 / DSA — verified.
- 2601.20276 — Beyond the Needle's Illusion — verified.
**Result.** PASS.

## C5. The "Risks to the hypothesis" section is non-empty
**Check.** §7 contains five risks (R1 native vs post-hoc sparsity, R2 K=1 ceiling, R3 recursion destroys retrieval heads, R4 compression too coarse, R5 NoLiMa contamination). Each names what the hypothesis still contributes if the risk materializes.
**Result.** PASS.

## C6. Architectural-coherence respected (information path explicit)
**Check.** Spec mandate: "a hypothesis that requires recursion to attend to tokens the sparse pattern fully drops is incoherent. The mechanism must specify the path by which information flows back to the recursion." §3 explicitly addresses this: "the mechanism does not propose that recursion attends to fully dropped tokens. It specifies the path explicitly — the NSA compression branch is the load-bearing fallback channel; MoBA has none." The information path is named (compression branch → residual stream → latent state z → refined query at next pass) and grounded in arXiv:2502.11089 Figure 2.
**Result.** PASS.

## C7. Non-additive prediction
**Check.** Spec mandate: "non-additive prediction." §2 specifies an *interaction* — the recursion × sparsity-pattern joint effect — as the load-bearing claim, not the main effect of either. F2 is the explicit differential test (NSA Δ minus MoBA Δ). The prediction would *not* be derivable from independent measurements of "sparsity matters" plus "recursion matters" — it requires the conjunction.
**Result.** PASS.

## C8. Cheaper-falsification-path subsection present
**Check.** §5a "Cheaper falsification path" specifies a single training-free post-hoc experiment (one model, NSA-vs-MoBA mask, K=1 vs K=4, 50 NoLiMa samples) with a single-number kill condition (NSA Δretention < 0.03 absolute → don't fund full eval). Limitations of post-hoc swap are acknowledged.
**Result.** PASS.

## C9. Spec-discipline checks
- **Architectural recursion only (not CoT/agent):** TRM K-pass forward-pass recursion. PASS.
- **YAGNI (no training, no kernel, no AGI claims):** Recursion is zero-shot or minimal-adapter; no novel kernels; no claim about general intelligence. The cheap falsification path is explicitly training-free. PASS.
- **Citation discipline:** Every mechanism claim cited; license flag on NoLiMa carried into manifest. PASS.
- **Architectural coherence:** Information path through compression branch is explicit. PASS.
- **Scale plausibility:** Test runs on 1–3B parameter backbones at 32K–64K context — within published NSA/MoBA regimes. Cheap falsification on smaller. PASS.

## C10. On revisions: every red-team objection has explicit response
**Not applicable.** This is revision 0 (initial submission). On future revisions, this section will list each red-team objection with response.
**Result.** N/A.

---

## Overall verdict
**PASS** — hypothesis is falsifiable with three independent metric+threshold+direction criteria, mechanism is grounded in cited prior art, architectural-coherence rule respected via explicit information path, prediction is non-additive (interaction term), risks are enumerated, cheaper falsification path supplied, all citations resolve.

Submit.
