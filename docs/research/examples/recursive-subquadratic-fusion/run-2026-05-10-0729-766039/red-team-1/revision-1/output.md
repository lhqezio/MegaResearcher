# Red-team-1 round-2 critique of H1-FB rev-1 ("Compressed-Summary Fallback as a *Sufficient* Recursion Substrate")

**Worker:** red-team-1
**Critiquing:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-1/revision-1/output.md`
**Round:** 2 (against revision-1; reviewing whether round-1 critique was addressed)

---

## Round-1 objection-by-objection status

### CRITICAL objections

**RT-1 (CRITICAL: Citation error on NSA single-pass deficit). STATUS: ADDRESSED.**

The revision (§"RT-1") quotes the corrected NSA Tab 1 numbers (0.456 vs 0.443, +1.3 pts) and Tab 2 (0.469 vs 0.437, +3.2 pts) and rebuilds §4 magnitude argument. I independently re-read NSA arXiv:2502.11089 §4 Tables 1-2 (`hf_papers read_paper 2502.11089 4`) and confirmed: NSA average 0.456 vs Full Attn 0.443; LongBench NSA 0.469 vs Full Attn 0.437. Multi-hop subsets (HPQ +0.087, 2Wiki +0.051) — the revision's reframing of NSA-K=1 as "parity-or-better with multi-hop advantage" matches the paper. The new prediction table now lists "A_n ≈ A_d (parity)" for NSA K=1, consistent with the data. Citation discipline is fixed.

**RT-2 (CRITICAL: Non-additivity isolation). STATUS: ADDRESSED.**

The revision reformulates F2 (§5) as a difference-in-differences with sign-asymmetry constraint AND K=1-calibration constraint:
- DiD: `[(NSA-fb K=6) − (NSA-fb K=1)] − [(MoBA K=6) − (MoBA K=1)] >= +5.0 abs points`
- Sign asymmetry: `(NSA-fb K=6 − K=1) > 0` AND `(MoBA K=6 − K=1) <= 0`
- K=1 calibration: `|NSA-fb K=1 − MoBA K=1| <= 2 abs points` enforced via matched-sparsity training; if violated, test is **inconclusive** (not falsified) and re-run.

The sign-asymmetry clause specifically rules out the "scaled additive" failure mode (both gain similarly from recursion). The K=1-calibration constraint forecloses the "MoBA is just worse at K=1" steelman that round-1 raised. This is exactly the reformulation round-1 prescribed in (R-2). It is now genuinely a non-additive interaction prediction.

**RT-3 (CRITICAL: F4 inoperable per-iteration logit-probe). STATUS: ADDRESSED.**

F4 (logit-probe) is deleted; F4' (attention-pattern Jaccard probe between iteration-1 and iteration-K=6 selected blocks) replaces it. The metric (`Jaccard(NSA-fb, I_1, I_6) <= Jaccard(MoBA, I_1, I_6) − 0.10`) is operationally sound: NSA's selected-block IDs are recoverable per-iteration without LM-head probing. The probe doesn't require interpretable intermediate logits, sidestepping the Huginn arXiv:2507.02199 "Coda Lens inconsistencies" concern. The attention-pattern observation is feasible at the proposed scale.

**RT-4 (CRITICAL: DSA != NSA cheap-path conflation). STATUS: ADDRESSED.**

Cheap path is restructured into two distinct tests:
- **Cheap path A** (~250 GPU-hours): NSA-only, training-aware F3 ablation at 350M params on 5B tokens — explicitly trains both A1 (NSA-with-fallback) and A2 (NSA-no-fallback) from scratch. No DSA conflation.
- **Cheap path B** (~50 GPU-hours): DSA as a *separate, weaker* prediction on DeepSeek-V3.2 open weights. A K=6 inference-recursion lift > +5 points on DSA would be counter-evidence; lift ≤ +4 points is consistent with mechanism. Explicitly framed as "bonus prediction, not stand-in."

I verified DSA's structure (`hf_papers read_paper 2512.02556 2`): DSA is lightning indexer + token-level top-k of selected MLA latents, with no compressed-summary branch. The "DSA has equivalent compressed-fallback structure" claim from rev-0 is gone. Architecturally clean.

**RT-5 (CRITICAL: Compute-match confound). STATUS: PARTIALLY ADDRESSED — citation accuracy issue.**

The revision commits to FLOP-matching: K=6 recursion adds 5 extra block-runs, K=1 baseline adds 5 *non-tied* extra layers of the same width to match total FLOPs. The recipe is methodologically defensible (this is the standard looped-vs-deeper ablation, see e.g. Iso-Depth Scaling Laws arXiv:2604.21106).

**However, the citation is mis-specified.** The revision repeatedly cites "Ouro arXiv:2510.25741 §4 protocol" and "Ouro Table 5 protocol" for FLOP-matching. I read Ouro §4 (`hf_papers read_paper 2510.25741 4`) and §5 (`hf_papers read_paper 2510.25741 5`):
- Ouro §4 is "Training Looped Language Models" (training pipeline / data composition).
- Ouro Table 5 is "Data composition for Stage 2 (CT Annealing)" — NOT a FLOP-match protocol.
- Ouro §5 (Tables 7-8) compares Ouro to dense baselines at *trained-token-matched* (and roughly parameter-similar) regime — e.g., "1.4B Ouro with 4 recurrent steps vs 4B Qwen3-Base." This is *parameter-budget* matching, not FLOP-matching of a non-shared deeper variant against a weight-tied recursion model.

The recipe the revision specifies (non-shared deeper K=1 vs weight-tied K=6 at matched FLOPs) is in fact *not from Ouro*. It is the "iso-depth" comparison from arXiv:2604.21106 ("Iso-Depth Scaling Laws for Looped Language Models," Sept 2026 — within knowledge cutoff) or a generic ablation in the looped-LM literature. **The methodology is sound but the citation is wrong.** This is a remediable citation-discipline issue, not a structural one. **Important, not Critical.**

### SERIOUS objections

**RT-6 (PLT counter-evidence). STATUS: ADDRESSED.**

The revision weakens the central claim from "compressed branch is *necessary*" to "compressed-summary OR sliding-window-style fallback is *sufficient*; no-fallback (MoBA, DSA) is the negative case." MoBA is now the load-bearing contrast. PLT's +6.1 (G-SWA = sliding-window fallback) is reframed as supporting the broader fallback claim. I verified PLT-3 in `hf_papers read_paper 2510.24824 3`: row (6) PLT-3 = 40.8 vs vanilla 34.7 = +6.1 average accuracy. The revision's interpretation of PLT is faithful to the paper.

**Sub-concern (architectural-coherence on PLT/G-SWA framing):** The revision frames G-SWA as "a partial fallback (it preserves a local-window channel that can be re-attended on iteration t+1)." But re-reading PLT §3.1, G-SWA is a *per-loop* sliding window — each loop has its own dedicated window (loop-specific KV) — and the +3.5 G-SWA gain (rows 4→5) is attributed to "per-loop specificity" rather than to a fallback channel. So the mechanism by which G-SWA supports recursion (per-loop dedicated KV) is *not* the mechanism by which NSA's compressed branch is alleged to support recursion (cross-iteration global summary). The revision's "fallback channel" generalization is plausible but architecturally heterogeneous — both architectures support recursion, but possibly via different mechanisms. This is **Important, not Critical**: the central F2 falsifier (NSA vs MoBA DiD) doesn't depend on PLT's mechanism being identical to NSA's; PLT is now consistency evidence, not a load-bearing claim.

**RT-7 (Tunnel Vision misapplied). STATUS: ADDRESSED.**

The revision drops Tunnel Vision framing from M2 entirely. MoBA prediction softened from "≤ +2, possibly negative" to "≤ +3, plausibly flat" — preserving the sign-asymmetry contrast against NSA's predicted +6 to +10 without overcommitting to negative values. ParaThinker is no longer in the citation list (verified §10 — confirmed absent).

**RT-8 (Magnitude transfer puzzle→text). STATUS: ADDRESSED.**

The TRM puzzle anchor is dropped; the new magnitude prior is anchored on PLT's +6.1 text-MoE lift at 680M activated params on 150B tokens. The predicted NSA-fb K=6 lift (+6 to +10) is bracketed at ~PLT and slightly above (justified by NSA's compressed branch being >= G-SWA's window for multi-hop). This is anchored on a published text+sparse+loop data point, addressing the round-1 objection.

**RT-9 (350M feasibility). STATUS: ADDRESSED.**

Full experiment is moved to 1B params, citing BABILong arXiv:2406.10149 §3.1 for above-floor performance at this scale. Cheap path A retained at 350M only at L=16K (smaller, where 350M is above floor). I have not independently verified BABILong's exact 1B-class noise floor, but the revision's framing — full kill-test at 1B, cheap path A as "go/no-go signal not kill-test" at 350M — is methodologically appropriate.

**RT-11 (M3 uncited at hinge). STATUS: ADDRESSED.**

M3 is demoted to auxiliary speculation, explicitly flagged as not load-bearing. The §3 text says: "M3 demoted to auxiliary speculation — explicitly NOT load-bearing... If retrieval heads do not form there, M2 still suffices — the hypothesis loses interpretive depth but not falsifiable substance." This addresses the round-1 concern that retrieval-heads-in-compressed-branch was an uncited hinge.

**RT-13 (Sparse Frontier mismatch). STATUS: ADDRESSED.**

The "Vertical-Slash for retrieval, Block-Sparse for reasoning" appeal is removed from M2; Sparse Frontier is now cited only for Jaccard / pattern-drift methodology in F4'. This is appropriate — Sparse Frontier studies training-free sparse attention and the methodological transfer (Jaccard) is legitimate.

### MINOR objections

**RT-10 (Scale plausibility — 3800 GPU-hours over fence). STATUS: ADDRESSED.**

Compute is moved to ~1100 GPU-hours full + ~300 GPU-hours cheap = ~1400 total, all under the 2000 GPU-hour fence. The arithmetic is plausible (8 pretraining runs × 50B tokens × 6e9 FLOPs/token / 700 TFLOPs effective ≈ 950 hours; +150 for evaluation/probing). The cheap path budgets are similarly reasonable.

**RT-12 (R3 oversells null fallback). STATUS: ADDRESSED.**

§7 R3 now reads: "this is no longer oversold; it is a real-but-modest contribution." The R3 framing is no longer claimed to satisfy the spec's full novelty target on a null result.

**RT-14 (HRM critique 2601.10679 softened). STATUS: ADDRESSED.**

HRM critique citation 2601.10679 is dropped from §10 (verified absent). No need to engage with its specific claim if it's not invoked.

---

## Summary table of round-1 objection status

| Round-1 objection | Severity | Status |
|---|---|---|
| RT-1 NSA citation error | CRITICAL | ADDRESSED |
| RT-2 Non-additivity isolation | CRITICAL | ADDRESSED |
| RT-3 F4 inoperable | CRITICAL | ADDRESSED |
| RT-4 DSA != NSA cheap path | CRITICAL | ADDRESSED |
| RT-5 Compute-match confound | CRITICAL | PARTIALLY ADDRESSED (methodology fixed; Ouro citation mis-specified) |
| RT-6 PLT counter-evidence | SERIOUS | ADDRESSED |
| RT-7 Tunnel Vision misapplied | SERIOUS | ADDRESSED |
| RT-8 Magnitude transfer | SERIOUS | ADDRESSED |
| RT-9 350M feasibility | SERIOUS | ADDRESSED |
| RT-11 M3 uncited | SERIOUS | ADDRESSED |
| RT-13 Sparse Frontier mismatch | MINOR | ADDRESSED |
| RT-10 Compute over fence | MINOR | ADDRESSED |
| RT-12 R3 oversells | MINOR | ADDRESSED |

All 5 critical objections from round-1 are substantively addressed. RT-5 has a remediable citation error.

---

## New objections introduced by the revision

**N1 (Important — citation accuracy). Ouro §4 / Table 5 cited for FLOP-matching protocol that is not in §4 / Table 5.** The revision cites "Ouro arXiv:2510.25741 §4 protocol" and "Ouro Table 5 protocol" three times in §"RT-5", §2, §6 for the non-shared deeper FLOP-matched K=1 baseline. I verified Ouro §4 is the training pipeline section (data stages, recurrent step schedule) and Table 5 is "Data composition for Stage 2 (CT Annealing)." The methodology the revision describes (non-tied deeper K=1 baseline at matched inference FLOPs) is *correct* and standard in the looped-LM literature (see Iso-Depth Scaling Laws arXiv:2604.21106 or Relaxed Recursive Transformers arXiv:2410.20672), but it is **not in the cited Ouro section/table.** The fix is one line — change the citation to the correct paper or to the specific Ouro figure that does present recurrent-vs-non-recurrent trade-offs (Tables 10/11 in Ouro §5.3 show recurrent-depth ablation, but at fixed parameter budget, not FLOP-matched against a non-shared deeper baseline). This is a citation-discipline issue analogous in kind (though smaller in magnitude) to the rev-0 NSA-Tab-1 misreading. **Important, fixable in a one-line edit.**

**N2 (Suggestion — PLT mechanism is heterogeneous from NSA's).** PLT-3's +6.1 lift uses **G-SWA with per-loop-specific KV cache** (rows 4→5 in PLT Tab 2: KV-share alone = 36.2; + G-SWA = 39.7 due to "per-loop specificity restored"). The mechanism is per-loop dedicated short-window KV, which is structurally different from "compressed summary persists across iterations and is re-attended at higher weight." The revision conflates these as both being "fallback channels." If the spec demands strict mechanism-citation discipline, the revision should either (a) acknowledge the mechanism heterogeneity and weaken the PLT-as-supporting-evidence claim, or (b) propose to add a PLT-style backbone to the comparison. Not Critical: the central F2 prediction (NSA vs MoBA DiD with sign asymmetry) is independent of how PLT is interpreted.

**N3 (Suggestion — F4' threshold of 0.10 is uncalibrated).** F4' specifies `Jaccard(NSA-fb) <= Jaccard(MoBA) − 0.10`. The 0.10 magnitude is not anchored to any prior measurement of inter-iteration selection drift. Sparse Frontier reports cross-pattern Jaccards in §4 but not cross-iteration drift on natively-trained models (no such data exists). The threshold could be tightened or loosened by an order of magnitude with no published prior. The hypothesis should either anchor 0.10 to a measurement (e.g., a pilot on the cheap-path-A model) or treat F4' as a *directional* prediction (NSA Jaccard < MoBA Jaccard) rather than a magnitude one. Not Critical: F4' is a *partial* falsifier of M2 that doesn't kill the F2 outcome, so a misanchored threshold is a calibration issue, not a coherence one.

**N4 (Suggestion — central claim drift evaluation).** Round-1 critique correctly identified that the original "NSA wins on multi-hop because compressed branch is the pivot" claim was vulnerable. The revision's reformulation is *both* more defensible AND a slightly weaker contribution: the new claim is "MoBA fails, NSA + sliding-window fallback both succeed." Is this still a non-additive interaction prediction worth a Phase-5 dispatch?

The non-additivity test is now: NSA-fb K=6 lift > 0, MoBA K=6 lift <= 0, with DiD >= +5 controlled by K=1 calibration. **This IS a non-additive interaction**: it's not "NSA always wins" (which would be an additive marginal — NSA preserves more info), it's "the same recursion operator gives opposite-sign deltas on two architectures matched at K=1." The sign-asymmetry clause is specifically what distinguishes this from additive scaling. So the contribution is preserved at a meaningful level: **gap-cell fill + non-additive interaction prediction with novel falsifier**. The contribution magnitude is slightly reduced (the original claim was sharper) but it's still a publishable hypothesis.

---

## Gap re-verification

I ran two independent queries this round:
- `hf_papers search "native sparse attention TRM recursion BABILong long context multi-hop"` (limit 10) — returned NSA, NSA-Latent variant (2511.00819), HiP, Flash Sparse Attention, MTraining, DHSA, AsyncTLS, MSA. **None pair architectural recursion with NSA on BABILong.** Closest is "Optimizing Native Sparse Attention with Latent Attention" (2511.00819), which adds latent attention to NSA but does not add recursion. Gap survives.
- `hf_papers search "MoBA mixture block attention recursion looped iterative refinement"` (limit 10) — returned MoBA, FlashMoBA (an optimization, not a recursion variant), RecursiveVLM (multimodal, not text-multi-hop), MoSA, VMoBA (video), MoR. **No published architectural-recursion + MoBA fusion on BABILong.** RecursiveVLM (2602.09080) is multimodal/VLM and does not test text-multi-hop reasoning.
- `hf_papers search "looped transformer weight tied recursion non-shared deeper baseline FLOP matched"` (limit 8) — returned Parcae (2604.12946), SpiralFormer (2602.11698), MoD, Sliced Recursive Transformer, Iso-Depth Scaling Laws (2604.21106), Relaxed Recursive Transformers, PLT, Hyperloop. The Iso-Depth paper is exactly the FLOP-matched comparison the revision invokes — *this* is the paper the revision should cite for the FLOP-match protocol, not Ouro §4. None of these test on NSA/MoBA backbones.

**Gap survives.** The (architectural recursion × natively-trainable sparse attention) cell on BABILong remains empty.

---

## Citation spot-checks (this round)

**(SC1) PLT +6.1 lift.** Verified directly via `read_paper 2510.24824 3`. Table 2 row (6) PLT-3 = 40.8 average; row (1) Seed-MoE 680M/13B = 34.7 average. 40.8 − 34.7 = 6.1. **Confirmed.**

**(SC2) NSA Table 1/2 corrected numbers.** Verified directly via `read_paper 2502.11089 4`. Table 1: NSA 0.456 vs Full Attn 0.443 (+0.013); Table 2: NSA 0.469 vs Full Attn 0.437 (+0.032). HPQ +0.087 multi-hop; 2Wiki +0.051 multi-hop. **Confirmed.** The revision's "A_n ≈ A_d (parity)" framing is correct.

**(SC3) DSA architecture (no compressed branch).** Verified directly via `read_paper 2512.02556 2`. DSA = lightning indexer (Eq 1) + top-k selection (Eq 2) — there is no compressed branch anywhere in §2.1. **Confirmed.** The revision's "DSA has no fallback channel" claim is architecturally accurate.

**(SC4) Ouro §4 / Table 5 cited for FLOP-matching protocol.** **NOT confirmed.** Ouro §4 is "Training Looped Language Models" (training stages, data composition) per `read_paper 2510.25741 4`. Ouro Table 5 is "Data composition for Stage 2 (CT Annealing)" — explicitly visible in the §4 read. Ouro's recurrent-depth-vs-baseline comparison is in §5 (Tables 7-11), and even there the comparison is parameter-budget-matched (1.4B Ouro vs 4B Qwen3) at trained-token-matched, not FLOP-matched at non-shared-deeper-baseline. **Citation N1 is a real issue.** Methodology is sound; citation is wrong.

---

## Verdict-determining analysis

The revision substantively addresses 5/5 critical and 6/6 serious round-1 objections. One new Important issue (N1: Ouro citation mis-specified) is introduced but is a one-line citation fix, not a mechanistic or coherence issue. The remaining new objections (N2, N3, N4) are Suggestion-level.

The central claim has shifted from "compressed branch is the necessary pivot" to "compressed-summary OR sliding-window fallback is sufficient; no-fallback is the negative case" — the new claim is still a non-additive interaction prediction (sign asymmetry on matched K=1, controlled DiD), is falsifiable at a budget under fence, and addresses the gap-cell. The contribution magnitude is slightly reduced from the rev-0 framing but remains publishable and meaningful.

The hypothesis is now defensible. I would defend it publicly with the one-line citation correction in N1.

---

## Severity-tagged objections (this round)

**Critical (must fix):** None.

**Important (should fix):**
1. **N1 (Ouro citation mis-specification for FLOP-matching protocol).** Replace "Ouro arXiv:2510.25741 §4 protocol" / "Ouro Table 5 protocol" with the correct citation. Recommended: arXiv:2604.21106 ("Iso-Depth Scaling Laws for Looped Language Models") for the iso-depth FLOP-matched comparison, OR arXiv:2410.20672 ("Relaxed Recursive Transformers") which discusses parameter-tied vs non-tied trade-offs. If the smith does not have time to verify either, the language can simply be "non-shared deeper FLOP-matched baseline (standard looped-LM ablation)" without paper citation — the methodology stands without it.

**Suggestion (nice to have):**
2. **N2 (PLT mechanism heterogeneity).** Acknowledge in §3 M2 that PLT's G-SWA fallback (per-loop dedicated KV) is mechanistically different from NSA's compressed branch (cross-iteration global summary). One sentence suffices: "PLT's G-SWA is a fallback of a different mechanistic flavor (per-loop short-window KV) than NSA's compressed branch (cross-iteration global summary); both are consistent with the sufficient-fallback claim, but the F2 falsifier is specifically about NSA-vs-MoBA, not about NSA-vs-PLT."
3. **N3 (F4' Jaccard threshold uncalibrated).** Either anchor the 0.10 threshold to a pilot measurement on cheap-path-A, or weaken to a directional prediction. Treat as guidance for eval-designer rather than a hypothesis-blocking issue.
4. **N4 (central-claim contribution-magnitude drift).** Already implicitly addressed by R3 framing in §7. No additional fix needed.

---

## Recommendation to hypothesis-smith

**The revision is approvable.** The round-1 critical objections are substantively addressed, the central claim is now genuinely non-additive, and the falsification criteria are operationalizable. The one Important issue (N1) is a citation correction that does not affect the science.

If the smith chooses to do a third revision, fix N1 (one-line citation change) and optionally N2 (one-sentence mechanism heterogeneity acknowledgment). If the orchestrator approves, this hypothesis is ready for Phase-5 eval-designer dispatch as-is, with the reviewer note that N1 should be fixed in synthesis.

---

## Verdict

APPROVE
