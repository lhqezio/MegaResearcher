# Verification — red-team-1 round 2

## Independent literature queries (>= 3 required)

1. `hf_papers search "native sparse attention TRM recursion BABILong long context multi-hop"` (limit 10) — 10 results, none pair architectural recursion with NSA on BABILong; closest is "Optimizing NSA with Latent Attention" (arXiv:2511.00819) which adds latent attention but not recursion. **Gap survives.**

2. `hf_papers search "MoBA mixture block attention recursion looped iterative refinement"` (limit 10) — 10 results, no published architectural-recursion + MoBA fusion on BABILong. RecursiveVLM (arXiv:2602.09080) is multimodal/VLM and not relevant. **Gap survives.**

3. `hf_papers search "looped transformer weight tied recursion non-shared deeper baseline FLOP matched"` (limit 8) — 8 results. The Iso-Depth Scaling Laws paper (arXiv:2604.21106) is exactly the FLOP-matched comparison the revision invokes; this is the paper the revision should cite for the FLOP-match protocol, not Ouro §4. None test on NSA/MoBA backbones. **Gap survives + identified citation correction for N1.**

4. `hf_papers search "recursion sparse attention BABILong multi-hop reasoning long context fallback"` (limit 10) — 10 results, none pair architectural recursion with NSA/MoBA on BABILong. **Gap survives.**

## Citation spot-checks (>= 3 required)

**SC1: PLT +6.1 lift claim** — `hf_papers read_paper 2510.24824 3` confirms Table 2 row (6) PLT-3 = 40.8 average accuracy; row (1) Seed-MoE 680M/13B = 34.7 average. Difference = +6.1. **Verified.**

**SC2: NSA Tables 1/2 (revised parity claim)** — `hf_papers read_paper 2502.11089 4` confirms Table 1: NSA 0.456 vs Full Attn 0.443 (+0.013); Table 2: NSA 0.469 vs Full Attn 0.437 (+0.032); HPQ +0.087 multi-hop, 2Wiki +0.051 multi-hop. **Verified.** The rev-1 magnitude argument is consistent with the paper.

**SC3: DSA architecture (no compressed branch)** — `hf_papers read_paper 2512.02556 2` confirms DSA = lightning indexer (Eq 1) + top-k selection (Eq 2), with no compressed branch. **Verified.** The rev-1 "DSA has no fallback" claim is correct.

**SC4: Ouro §4 / Table 5 cited for FLOP-matching protocol** — `hf_papers read_paper 2510.25741 4` and `5` show Ouro §4 is "Training Looped Language Models" (training stages, data composition, recurrent step schedule), and Table 5 is "Data composition for Stage 2 (CT Annealing)." Ouro's recurrent-depth ablation in §5 (Tables 7-11) is parameter-budget-matched (e.g., 1.4B Ouro vs 4B Qwen3) at trained-token-matched, NOT FLOP-matched at non-shared-deeper-baseline. **Citation does NOT support the claim.** This is the new Important objection N1. The methodology described in the revision is sound (and matches arXiv:2604.21106 Iso-Depth Scaling Laws), but the citation is wrong.

## Verdict-severity match check

- Verdict: APPROVE
- Critical count: 0
- Important count: 1 (N1, a one-line citation fix)
- Suggestion count: 3 (N2, N3, N4)

An APPROVE with 0 Criticals is consistent with the discipline rule. The Important issue (N1) is a citation accuracy issue that does not affect the science of the hypothesis (the methodology is correctly described and standard in the looped-LM literature). The remaining Suggestion-level issues are guidance for eval-designer and synthesist, not blockers.

This is the kind of issue the synthesist can fix in one line during synthesis without requiring a third revision; the hypothesis-smith has already used 1 of 3 revisions, and forcing a third revision for a single one-line citation correction is a waste of swarm budget. APPROVE is appropriate.

## Discipline rules satisfied

- [x] Steelmanned the position that the hypothesis is wrong: round-1 already constructed the steelman (additive marginals), and rev-1 specifically forecloses it via the K=1 calibration constraint and sign-asymmetry clause in F2.
- [x] Verified each round-1 objection is addressed by reading the relevant section of the revision and quoting specific changes.
- [x] Did not perform performative agreement: identified one new Important issue (N1) plus three Suggestions, and partially-addressed status on RT-5.
- [x] Would defend the hypothesis publicly: yes, with the N1 citation correction. The central F2 falsifier (DiD with sign asymmetry under K=1 calibration) is a genuine non-additive interaction prediction that fills the gap-cell.
