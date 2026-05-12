# H5 (revision-1) — Latent Plateau, Not Lock-In: Architectural Recursion Escapes Tunnel Vision Under Sparse Attention but Fails to Scale on Long-Context Out-of-Distribution Reasoning

**Role:** hypothesis-smith-5
**Targeting gap:** `gap-finder-2/output.md` Gap #7 (B7) — "the (architectural recursion x Tunnel-Vision under SubQ) intersection is empty"
**Revision:** 1
**Polarity (FLIPPED from revision-0):** Architectural recursion ESCAPES the discrete-commitment lock-in mechanism that defines Tunnel Vision (arXiv:2509.04475 sec. 2.2), but under subquadratic attention on long-context out-of-distribution reasoning, K-pass recursion produces a *distinct* failure mode — **latent plateau**: post-K accuracy is statistically indistinguishable from K=1 accuracy (no useful work performed by extra passes), unlike on training-distribution puzzles where K materially helps (arXiv:2510.04871 sec. 5).

---

## Changes from revision-0 (response to red-team round 1 critique)

The red-team raised three Critical objections (C1, C2, C3) plus six Important issues. Each is addressed below. **The most consequential change is the polarity flip**, motivated by the red-team's correct identification that revision-0's mechanism story was wrong about what TRM's deep supervision does (C2) and that the HRM-attractor mechanism does not transfer to TRM (C3). Revision-1 rebuilds the mechanism from scratch around what is *positively known* about deep-supervision-trained recursive operators and weight-tied iterated attention, citing Huginn (arXiv:2507.02199) and Parcae (arXiv:2604.12946) for grounded mechanism claims that revision-0 missed.

**Critical objections addressed:**

- **C1 — Lock-in metric L was broken.** Resolved by *abandoning* the lock-in metric entirely as the primary outcome. The new primary outcome is the **per-instance K-vs-1 utility delta** Δ(s, K) = accuracy(s, K) − accuracy(s, K=1), measured under random-position needle controls. This metric is well-defined regardless of whether K=1 is correct and does not depend on identifying "the correct answer the model would have given at K=1." Lock-in is no longer claimed; the prediction is now a null effect of K under sparse + long-context (a *positive* statement of plateau).

- **C2 — Deep supervision misread.** Acknowledged: revision-0's claim that deep supervision is an "implicit commitment device reinforcing wrong direction" misread arXiv:2510.04871 sec. 4.1. The TRM training objective pushes z_H toward the correct y at every supervision step (cross-entropy against ground truth applied at every full-recursion step), which *dissolves* wrong directions during training. Revision-1 drops the commitment-device framing and rebuilds the mechanism around (a) what the trained operator's iterates actually do at inference time, (b) Parcae's spectral-norm argument for stable looped models, and (c) Huginn's empirical evidence that depth-recurrent transformers plateau rapidly and do not match CoT.

- **C3 — HRM-attractor mechanism does not transfer to TRM.** Acknowledged: HRM's spurious attractors arise *because* of the fixed-point assumption + 1-step IFT gradient (arXiv:2601.10679 sec. 4.4). TRM (sec. 4.1) explicitly removes both. Revision-1 takes path (c) the red-team explicitly offered: **flip polarity** to "architectural recursion ESCAPES Tunnel-Vision-style lock-in" — but predicts a different, empirically distinguishable failure (latent plateau). The new mechanism is positively grounded in arXiv:2507.02199 sec. 3.4 (Huginn plateau evidence) and arXiv:2604.12946 sec. 3 (spectral-radius < 1 contraction).

**Important objections addressed:**

- **I1 — F1 noise-floor calibration.** Section 5 now includes an explicit power calculation. With 200 instances per (s, K) cell and binary correctness, per-cell SE ≈ √(0.25/200) ≈ 0.035 (3.5 pp). Interaction-term SE ≈ 0.07 (7 pp). The new falsification thresholds are calibrated to ≥ 2 SE.
- **I2 — Citation gaps (Parcae, Superlinear Multi-Step Attention).** Both are now load-bearing. Parcae (arXiv:2604.12946) provides the positive mechanism for *why* TRM-style recursion plateaus rather than locks-in: with bounded ρ(A̅) < 1, the recurrent residual contracts toward a fixed point. Superlinear Multi-Step Attention (arXiv:2601.18401) is engaged in section 1: it *builds* a working multi-step subquadratic architecture, demonstrating that the K-vs-sparse composition is feasible — but does not test out-of-distribution long-context where my plateau prediction applies, so the gap remains.
- **I3 — Theorem 4.2 → recursion extension was asserted, not argued.** Section 3 now explicitly argues: under weight-tying, the iterated A^t is the t-th power of a single stochastic matrix, which by Perron-Frobenius converges to its dominant left eigenvector regardless of t-time-variance. The qualitative direction (first-token sink under causal/sliding-window mask) follows. The *quantitative* prediction is now framed as a hypothesis-derived corollary, not a theorem-derived fact, per I3 + I5.
- **I4 — "Asymmetric value" hedge.** Removed entirely. The hypothesis now commits to falsification as falsification: if the predicted plateau does not manifest, the hypothesis is wrong, full stop.
- **I5 — α-exponent prediction not citation-grounded.** The α-exponent prediction is dropped from the primary falsification set. It moves to secondary diagnostics with a hypothesis-derived (not citation-grounded) tag.
- **I6 — Risk 2 (TRM-doesn't-transfer) is interpretability-killing.** Section 5 now includes a pre-registered K=1 stability gate: K=1 long-context accuracy must exceed 30% before K-scaling measurements are meaningful. If the gate fails, the result is "TRM-style recursion does not transfer to long context" (a different finding, not a falsification of H5).

**Suggestions adopted:**
- **S1 — naming.** Renamed from "Latent Tunnel Vision" to "Latent Plateau, Not Lock-In" — the failure mode is plateau, not Tunnel-Vision-style lock-in.
- **S4 — K* prediction.** Dropped (the K* threshold was over-specific given M2's looseness under weight-tying).

---

## 1. Targeted gap

The fusion thesis pins architectural recursion (TRM-style, K passes within a single forward pass; arXiv:2510.04871 sec. 4.1) onto subquadratic-attention backbones for long-context reasoning. Gap-finder-2 #7 establishes that no prior work tests how this composition behaves under long-context out-of-distribution reasoning where the operator was trained on short-context puzzles. Three near-adjacent works define the gap's boundary:

- **Tunnel Vision in CoT** (arXiv:2509.04475 sec. 2.2) characterizes the discrete-token-commitment failure: an early flawed token poisons the trajectory irreversibly. This phenomenon is by construction inapplicable to architectural recursion, which has no committed token between passes.
- **HRM mechanistic critique** (arXiv:2601.10679 sec. 4.4) documents spurious-fixed-point attractors *for HRM specifically*, where the failure mode arises from HRM's fixed-point assumption and the 1-step IFT gradient that justifies stopping. TRM (arXiv:2510.04871 sec. 4.1) explicitly drops both. Revision-0 incorrectly inherited this attractor framing for TRM; revision-1 rejects it.
- **Huginn / depth-recurrent CoT decoding** (arXiv:2507.02199 sec. 3.4) is the most directly evidential prior: scaling Huginn's recurrent steps from 4 to 32 yields only modest gains (3.11 → 4.93 GSM8K), with performance plateauing thereafter, while explicit CoT reaches 24.87/38.13. This is concrete empirical evidence that depth-recurrent latent reasoning **plateaus** — does not lock in to wrong answers, but does not scale either.
- **Parcae** (arXiv:2604.12946 sec. 3) shows that stable looped LMs require ρ(A̅) < 1 (spectral radius bounded below 1); under this constraint, the recurrent residual *contracts* toward a fixed point rather than diverging or oscillating. Successful looped LMs *avoid* attractor instability by design.
- **Superlinear Multi-Step Attention** (arXiv:2601.18401) builds a working multi-step subquadratic architecture (O(L^(1+1/N))) — empirically demonstrates that K-pass attention under subquadratic complexity can be made to function. Tests on NIAH but not on the position-controlled long-context out-of-distribution stimulus the present hypothesis targets.

The empty cell is: how does a deep-supervision-trained TRM-style operator (architecturally recursive, no fixed-point assumption, no IFT) behave when transferred from training-distribution puzzles to long-context retrieval-conditioned reasoning under a subquadratic-attention backbone? Revision-0 predicted lock-in (the HRM analogue). Revision-1 predicts plateau (the Huginn analogue) — a defensible, falsifiable, distinct prediction that engages with what the red-team correctly identified as the actual mechanics of deep supervision.

---

## 2. Hypothesis statement (if/then form)

**If** a TRM-style architecturally recursive operator (K passes of latent refinement on a fixed deep-supervision target; arXiv:2510.04871 sec. 4.1) is composed with a subquadratic attention backbone (sparsity ratio s ∈ {0.25, 0.5}; e.g., NSA arXiv:2502.11089 / Sparse-Frontier Vertical-Slash arXiv:2504.17768 patterns) and run on a position-controlled long-context reasoning stimulus that requires re-attending to mid-context evidence (license-clean reconstruction of NoLiMa-style 2-hop fact-link templates following arXiv:2502.05167 with biographical-template construction matching arXiv:2407.11963's NeedleBench Ancestral Trace Challenge paradigm; see section 6 for stimulus precision), **then** the K-vs-1 accuracy gain Δ(s, K) = accuracy(s, K=8) − accuracy(s, K=1) under sparse attention will be **statistically indistinguishable from zero** (|Δ(s, K=8)| ≤ 3 pp with 95% CI containing zero) — i.e., extra recursion passes perform no useful work. By contrast, on the training-distribution puzzle stimulus (Sudoku-Extreme, per arXiv:2510.04871 sec. 5) the same operator yields Δ ≥ 20 pp from K=1 to K=8.

The non-additive prediction: the *interaction* between recursion (K) and sparse-vs-dense backbone is **negative for plateau-symmetry** — under dense attention, K-vs-1 may yield a small positive Δ on long-context (recursion partially compensates for distribution shift); under sparse attention, K-vs-1 collapses Δ to zero regardless. The interaction is the load-bearing claim: sparsity does not produce *worse* lock-in, it produces *flat* plateau.

---

## 3. Mechanism

The mechanism rests on three positively-grounded claims:

**M1 — Deep-supervision-trained recursive operators contract toward a fixed point under bounded spectral radius; they do not produce instance-specific spurious attractors.** Parcae (arXiv:2604.12946 sec. 3) recasts looped LM dynamics as a non-linear time-variant dynamical system h_t = A̅·h_{t−1} + B̅·e + R̅(h_{t−1}, e); stability requires ρ(A̅) < 1, under which the recurrent state contracts toward a fixed point. TRM-style recursion, while not architecturally identical (TRM has explicit y, z separation per arXiv:2510.04871 sec. 4.2), is a *successful instance* of deep-supervision-trained recursion that empirically converges (arXiv:2510.04871 Table 1 — 87.4% on Sudoku-Extreme). Empirically converging recursion is incompatible with HRM-style spurious attractors. **The implication: TRM does not lock-in.** The prior failure-mode prediction (HRM-style attractors) is therefore wrong for TRM. This is consistent with Huginn (arXiv:2507.02199 sec. 3.2, 3.4): rank trajectories show non-smooth oscillations across recurrent blocks (sec. 3.2), but performance does not catastrophically degrade with more recurrence — it plateaus (sec. 3.4: 4 → 32 steps yields 3.11 → 4.93, plateau).

**M2 — Under weight-tied recursion, the iterated stochastic-attention matrix A^t converges to its dominant eigenvector — an input-independent direction.** arXiv:2502.01951 Theorems 4.1–4.2 are stated for time-varying W^(t); the underlying graph-theoretic and stochastic-matrix arguments still apply under weight-tying as a special case. With weight-tying, A is fixed; A^t for t → ∞ converges to v·e^T where v is the dominant left eigenvector. Under causal mask, v concentrates on the first token (arXiv:2502.01951 Theorem 4.1); under sliding-window mask of width w, v concentrates on the first token at rate ϵ^⌈(N−1)/(w−1)⌉ (Theorem 4.2). **The qualitative implication: under sparse weight-tied attention, K-pass iteration accelerates each token's effective context contraction toward content-independent first-token-sink positions.** This is *NOT* a per-instance attractor (which would predict different wrong answers for different instances, i.e., lock-in); it is a *uniform* content-blind contraction (which predicts no instance-specific signal — i.e., plateau where K passes give the same content-poor representation to the output head). The *quantitative* prediction (rate of contraction in K) is hypothesis-derived, not theorem-derived (per red-team I3): the theorem authorizes the qualitative direction only.

**M3 — Long-context out-of-distribution stimulus is the regime where M1 + M2 manifest as plateau rather than as either lock-in or improvement.** Score Dilution at Test Time (arXiv:2512.13898) and Long Context, Less Focus (arXiv:2602.15028) establish that long-context attention scores dilute, making single-pass retrieval lossy. On a training-distribution puzzle (Sudoku-Extreme), TRM's deep-supervision objective has aligned the operator's iterates with the task structure, and recursion materially improves accuracy (arXiv:2510.04871 sec. 5: 87.4%). On an out-of-distribution long-context stimulus, the iterates have no training-distribution-aligned target to converge toward, and per M1 the operator's iterates contract toward whatever fixed point the (sparse + long-context) input produces — which by M2 is the content-blind first-token-sink direction. Each additional K does not introduce *wrongness* (no lock-in), but also does not introduce new mid-context information (no improvement). Net: K-vs-1 plateau.

**Why this is qualitatively distinct from Tunnel Vision (engaging the spec).** Tunnel Vision (arXiv:2509.04475 sec. 2.2) operates in discrete output space: the model emits tokens; each token is sampled from softmax; the next forward pass conditions on the committed token. Architectural recursion has no commit-and-condition step. Revision-0 incorrectly tried to map this to "deep-supervision-as-commitment-device"; the red-team correctly rejected this. The *correct* qualitative claim is that architectural recursion **escapes** Tunnel Vision (no token commit → no irreversible poisoning), but suffers a different, milder failure mode (latent plateau) that has been empirically observed for related architectures (Huginn) and is mechanistically grounded in Parcae's spectral-radius dynamics.

If M1–M3 are correct, the predicted plateau is mechanistically grounded. If empirical results show K-vs-1 *improves* under sparse + long-context (Δ ≫ 0), then either M1 is wrong (the deep-supervision objective transfers to OOD long-context after all — a finding that would substantially strengthen the fusion thesis), or M2 is wrong (the weight-tied attention center-node bias is overcome by recursion — also a substantive finding). If results show K-vs-1 *degrades* (Δ ≪ 0), then revision-0's lock-in prediction was correct after all and revision-1 over-corrected. Either off-axis result is itself diagnostic.

---

## 4. Predicted outcome with magnitude

Let:
- **Δ(s, K)** = K-vs-1 accuracy gain on the long-context out-of-distribution stimulus.
- **Δ_train(K)** = K-vs-1 accuracy gain on the training-distribution puzzle stimulus (Sudoku-Extreme).
- **U(s, K)** = U-curve depth: accuracy at end positions minus accuracy at mid positions.

**Predictions, each with magnitude and direction:**

1. **Δ(s=0.5, K=8) is null.** |Δ(0.5, 8)| ≤ 3 pp, with 95% CI containing zero. Sample size 200 instances per cell (per-cell SE ≈ 3.5 pp; difference SE ≈ 5 pp; null detection threshold |Δ| ≤ 5 pp gives power > 0.8 at α = 0.05). The plateau is statistical, not just descriptive.

2. **Δ_train(K=8) ≥ 20 pp.** This is a positive control: TRM on its training distribution should still benefit from K. arXiv:2510.04871 Table 1 reports T=3, n=6 → 87.4% on Sudoku-Extreme; the K=1 ablation (T=2, n=2 → 73.7%) gives a 13.7 pp baseline gap; with our K=8 vs K=1 framing under matched parameters the expected delta should be at least 20 pp on the matched-distribution control. If this fails, the operator is broken and the long-context measurement is uninterpretable (positive-control gate).

3. **Sparse-vs-dense interaction in long-context: Δ(s=0.5, 8) − Δ(s=1.0, 8) ≤ −5 pp.** Under dense attention, K may help slightly on long-context (recursion partially compensates for context-length distribution shift); under sparse, the help collapses. The interaction is at least 5 pp in the direction "sparse blocks recursion utility."

4. **U(s=0.5, K=8) ≈ U(s=0.5, K=1).** Under sparse, recursion neither flattens nor deepens the U-curve. |ΔU| ≤ 3 pp. (Revision-0 predicted deepening; revision-1 predicts flatness — the plateau extends to the position-controlled stimulus.)

**Conditions under which the hypothesis should hold:**
- Stimulus requires re-attending to mid-context evidence (positions 30–70% of context length) that is not lexically matchable to the query — license-clean NoLiMa-style 2-hop biographical template (arXiv:2502.05167; reconstruct, do not redistribute) with NeedleBench Ancestral Trace Challenge-style position-grid construction (arXiv:2407.11963).
- Context length ≥ 8K tokens.
- Architectural recursion uses deep supervision per arXiv:2510.04871 sec. 4.1.
- K=1 baseline accuracy on the long-context stimulus is ≥ 30% (positive-control gate).

**Conditions under which the hypothesis should NOT hold:**
- Training-distribution puzzles (Sudoku-Extreme, ARC-AGI) — there Δ ≥ 20 pp is predicted (positive-control gate).
- K=1 baseline below 30% — interpretability gate fails; experiment yields a different finding ("TRM does not transfer to long context") rather than testing H5.
- Stimuli where the answer is in first 10% / last 10% of context — center-node bias accidentally aligns with truth.

---

## 5. Falsification criteria

Each criterion individually falsifies the hypothesis if observed. All thresholds are calibrated to the planned sample size (200 instances per (s, K) cell), giving per-cell binary-outcome SE ≈ 3.5 pp and difference SE ≈ 5 pp (per red-team I1).

**F1 — Recursion materially helps under sparse + long-context.** Metric: Δ(s=0.5, K=8). Threshold: if Δ ≥ +10 pp (well above 2 difference SEs), the plateau prediction is falsified. Direction: prediction is |Δ| ≤ 3 pp (null); falsification is Δ ≥ +10 pp.

**F2 — Recursion materially hurts under sparse + long-context.** Metric: Δ(s=0.5, K=8). Threshold: if Δ ≤ −10 pp, the plateau prediction is falsified in the opposite direction (revision-0's lock-in framing was right after all). Direction: prediction is |Δ| ≤ 3 pp (null); falsification is Δ ≤ −10 pp.

**F3 — No sparse-vs-dense interaction in long-context.** Metric: Δ(s=0.5, 8) − Δ(s=1.0, 8). Threshold: if this difference is in the interval [−5 pp, +5 pp], the load-bearing interaction claim (M2's center-node sink amplification) is falsified. Direction: prediction is ≤ −5 pp; falsification is anything outside that.

**F4 — Positive-control gate fails.** Metric: Δ_train(K=8). Threshold: if Δ_train ≤ +10 pp on Sudoku-Extreme matched-parameter K=1-vs-K=8 comparison, the operator is not behaving as TRM should and the experiment cannot test H5. Outcome: not a falsification of H5, but a kill of the experimental setup. Pre-registered.

**Calibrated commitment.** F1 and F2 are *symmetric falsifications in opposite directions*. The hypothesis is the narrow null prediction between them: Δ in (−3, +3). Either F1 or F2 alone falsifies the polarity. This is a strong-falsification commitment — there is no rhetorical escape direction in which "the result is informative either way." If F1 fires, revision-1 was too pessimistic about recursion's OOD transfer (and the fusion thesis is in better shape than this hypothesis predicted). If F2 fires, revision-0's lock-in framing was right and revision-1's flip was wrong. Either is a clean refutation.

---

## 6. Required experiments (sketch — eval-designer details)

**Stimulus construction (license-clean, scope-bounded per red-team S2 + assignment "NoLiMa license-clean reconstruction is unbounded scope").**

- *Template family:* synthetic biographical paragraphs constructed from a fixed schema: each persona has 5 attributes (birth_city, mother_birth_city, profession, hobby, residence_city) drawn from disjoint name pools. The query is a 2-hop chain: "What city was the mother of [persona] born in?" The needle paragraph contains "[persona]'s mother was born in [city]." No lexical overlap between query and needle (the surface form "mother" does not appear; the needle uses "her mother" or paraphrase). Reconstructs the NoLiMa principle (arXiv:2502.05167) without redistributing NoLiMa text.
- *Distractor density:* fill remaining context with persona paragraphs from non-target personae sharing one attribute slot with target (e.g., same birth_city different mother), creating attribute-confusable distractors.
- *Position grid:* needle placed at 8 positions ∈ {5%, 15%, 30%, 45%, 55%, 70%, 85%, 95%} of context length.
- *Context lengths:* {8K, 32K} tokens (drop 128K vs revision-0 to bound scope; 8K and 32K are sufficient for sparse-attention center-node bias per arXiv:2502.01951 Theorem 4.2 rate dependence on N/w).
- *Stimulus validation step (per S2):* verify that K=1 dense-attention baseline accuracy is ≥ 50% on at least one position; if not, the stimulus is too hard and lacks a measurable signal range. Drop attribute-confusion until this holds. This is a stimulus-validation gate, separate from the K=1 model-stability gate.

**Models & ablations (factorial design).**

- *Backbone × Sparsity:* dense baseline (s = 1.0); NSA-trained sparse (arXiv:2502.11089) at s = 0.5; Sparse-Frontier Vertical-Slash pattern (arXiv:2504.17768) at s = 0.5 as cross-pattern check. Independently-pretrained sparse-attention checkpoints — *not* post-hoc sparsified — to avoid distribution-shift confound.
- *Recursion × Depth:* TRM-style architectural recursion at K ∈ {1, 2, 4, 8}, with deep supervision per arXiv:2510.04871 sec. 4.1. Reduce the K-grid from revision-0's {1,2,4,8,16} given the plateau prediction (more K does not need testing past plateau onset).
- *Critical control (matched-FLOPs CoT baseline).* For K=8, the matched-FLOPs CoT control runs sequential CoT generation under the same backbone with the same total compute. This separates "K plateau" from "matched compute helps in any modality."
- *Critical control (training-distribution positive control).* Sudoku-Extreme run at K=1 vs K=8 to verify Δ_train ≥ 20 pp gate.

**Operationalization of Δ(s, K) (per red-team C1).** Δ(s, K) = accuracy(s, K) − accuracy(s, K=1) on the long-context stimulus, averaged over 200 instances per (s, K, position) cell. Accuracy is exact-match correctness against the city-name target. **No lock-in metric, no per-instance K=1-baseline comparison.** This is a population-level statistic, robust to per-instance K=1 wrongness.

**Secondary diagnostics (not load-bearing for falsification, per red-team I5):**
- *Attractor-basin probe.* For 50 randomly selected instances at K=4, perturb the latent z with random ε-noise and re-run K=4→K=8. If the K=8 answer is *identical* across perturbed and unperturbed runs, the iterates are converging to a fixed point (consistent with M1). If perturbations cause divergence, M1 is in trouble.
- *α-exponent in needle count.* Multi-needle variant N ∈ {1, 2, 4, 8} fit accuracy ∝ N^(−α). Report α(s, K=1) vs α(s, K=8). This is hypothesis-derived, not citation-grounded — moved out of falsification set.

**Pre-registration:**
- All cell sample sizes (200), thresholds, and gates are committed before data collection.
- The K=1 stability gate (≥ 30% long-context K=1 accuracy) and stimulus-validation gate (≥ 50% K=1 dense-attention at some position) are non-negotiable: if either fails, experiment yields a different finding, not a falsification of H5.

---

## 7. Risks to the hypothesis

**Risk 1 — TRM-style operator does not transfer to long context at all.** TRM was demonstrated on fixed-context puzzles (arXiv:2510.04871 sec. 5; sec. 4.5 uses attention-free architecture for fixed small contexts). At 8K+ tokens, the operator may produce no stable answer at K=1. The K=1 stability gate (≥ 30% accuracy) catches this; if it fails, the experiment yields the finding "TRM does not transfer to long context" (a different research output, not a falsification). Without the gate this risk would be interpretability-killing per red-team I6.

**Risk 2 — Deep supervision generalizes farther than predicted.** If the deep-supervision objective generalizes well to OOD long-context inputs, then the operator's iterates *do* improve on each pass even on long-context stimuli. F1 fires, and the plateau prediction is wrong. This would be a *positive* result for the fusion thesis but a falsification of revision-1. Honest disclosure: this is plausible — TRM achieved 7.8% on ARC-AGI-2 (arXiv:2510.04871 sec. 5), an OOD test of sorts, suggesting some OOD transfer.

**Risk 3 — Plateau is a property of low K=1 baseline, not of recursion.** If K=1 long-context accuracy is exactly 30% (the gate threshold) and K=8 is exactly 30%, the plateau may be a floor effect (model is at noise; nothing can improve it) rather than a substantive plateau. *Mitigation:* the stimulus-validation gate ensures dense K=1 reaches ≥ 50% at some position, so a non-trivial measurable signal range exists.

**Risk 4 — Sparsity-pattern dependence (held over from revision-0).** NSA, Vertical-Slash, Block-Sparse have different center-node structures (arXiv:2502.01951 Theorems 4.2–4.3). The plateau prediction might hold for some patterns and not others. *Mitigation:* the design includes two independent sparse-attention patterns (NSA + Vertical-Slash); the prediction is plateau on *both*. If only one shows plateau, the hypothesis is partially confirmed and which pattern is which becomes a substantive secondary finding.

**Risk 5 — Confound between "plateau" and "no useful work at any K."** The hypothesis predicts plateau between K=1 and K=8 specifically because of long-context OOD; but if K=1 already fails to use mid-context information at all, the plateau is just the floor. *Mitigation:* the dense-attention K=1 baseline at long-context (s=1.0, K=1) is the reference: if dense K=1 succeeds at mid-context retrieval but sparse K=1 does not, and *neither* benefits from K=8 (in the long-context OOD regime), the plateau is real and sparsity-conditional.

---

## 8. Sources

| Citation | arxiv ID | Role in this hypothesis |
|---|---|---|
| Less is More: Recursive Reasoning with Tiny Networks (TRM) | 2510.04871 | Defines architectural recursion with deep supervision (sec. 4.1); the operator under test; positive-control benchmark (sec. 5) |
| Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer (Huginn) | 2507.02199 | Empirical evidence of plateau in depth-recurrent latent reasoning (sec. 3.4: 4 → 32 steps yields 3.11 → 4.93, plateaus, vs. CoT 24.87/38.13); discontinuities in interpretability across recurrent blocks (sec. 3.2) — load-bearing for M1 |
| Parcae: Scaling Laws For Stable Looped Language Models | 2604.12946 | Spectral-radius < 1 contraction argument (sec. 3); positive mechanism for *why* deep-supervision-trained recursion plateaus rather than locks in — load-bearing for M1 |
| Mechanistic Analysis of HRM | 2601.10679 | Documents spurious attractors *in HRM specifically* (sec. 4.4), arising from fixed-point assumption + 1-step IFT. **Revision-1 explicitly distinguishes from TRM, which removes both.** |
| ParaThinker / Tunnel Vision | 2509.04475 | Defines Tunnel Vision in CoT (sec. 2.2); the failure mode this hypothesis claims architectural recursion *escapes* |
| Position-Bias Emergence | 2502.01951 | Theorems 4.1–4.2 (graph-theoretic position-bias under causal/sliding-window masks); load-bearing for M2; theorem statements verified by red-team |
| Lost in the Middle | 2307.03172 | U-curve metric (M3, F3) |
| Hyper-multi-step | 2410.04422 | Cited only for stimulus design; α-exponent prediction moved to secondary diagnostic per red-team I5 |
| Score Dilution at Test Time | 2512.13898 | Long-context attention dilution (M3) |
| Long Context, Less Focus | 2602.15028 | Attention dilution mechanism (M3) |
| NoLiMa | 2502.05167 | Position-controlled stimulus design (license-flagged: reconstruct biographical-template paradigm, do not redistribute) |
| NeedleBench (Ancestral Trace Challenge) | 2407.11963 | Reference paradigm for license-clean multi-hop biographical-template construction; Ancestral Trace Challenge is precisely the 2-hop biographical reasoning chain (per red-team scope-bounding S2) |
| Native Sparse Attention (NSA) | 2502.11089 | Subquadratic-attention backbone candidate |
| The Sparse Frontier | 2504.17768 | Sparsity patterns; Vertical-Slash for cross-pattern check |
| Superlinear Multi-Step Attention | 2601.18401 | Engaged in section 1: builds a working multi-step subquadratic architecture; demonstrates feasibility of K-pass × subquadratic; does not test out-of-distribution long-context plateau, so the gap survives |
| Parallel-R1 | 2509.07980 | Parallel-thinking remediations operate at prompt layer, not architectural |
| Thought Rollback | 2412.19707 | Same — parallel/backtracking at prompt layer only |
| To Backtrack or Not to Backtrack | 2504.07052 | Same — limits of sequential search at prompt layer |
