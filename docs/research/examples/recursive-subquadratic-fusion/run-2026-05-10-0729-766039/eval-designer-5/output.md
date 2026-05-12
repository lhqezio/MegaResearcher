# Eval design — H5 (revision-1): "Latent Plateau, Not Lock-In"

**Role:** eval-designer-5
**Hypothesis input:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-5/revision-1/output.md`
**Red-team approval:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-5/revision-1/output.md` (APPROVE)
**Design philosophy.** This is a *null-effect* hypothesis. Standard NHST is structurally unable to confirm a null. The design therefore rests on **TOST equivalence testing** for the plateau prediction (Lakens 2017; Schuirmann 1987), with one-sided tests for the directional falsification arms (F1 recursion-helps, F2 recursion-hurts). Every design choice below targets falsifiability: the experiment must be able to *fail* the hypothesis, and three independent ways to fail are pre-registered.

---

## 1. Hypothesis being tested (self-contained restatement)

**Claim (M1+M2+M3 composed).** A TRM-style architecturally recursive operator (deep-supervision-trained, K passes of latent refinement; arXiv:2510.04871 sec. 4.1), composed with a subquadratic-attention backbone (sparsity ratio s ≈ 0.5 via NSA arXiv:2502.11089 or MoBA arXiv:2502.13189), evaluated on a position-controlled long-context out-of-distribution stimulus (≥ 8K context, mid-context 2-hop biographical needle), exhibits **K-vs-1 accuracy gain Δ(s, K=8) statistically equivalent to zero in the equivalence band Δ ∈ (−3, +3) pp** — extra recursion passes perform no useful work. Mechanism: deep-supervision-trained recursion contracts toward a fixed point under bounded ρ(A̅) < 1 (Parcae sec. 3, arXiv:2604.12946); under weight-tied recursion + sparse mask, A^t converges by Perron-Frobenius to the dominant eigenvector (a content-blind first-token-sink direction; arXiv:2502.01951 Theorems 4.1–4.2), so each pass produces the same content-poor representation rather than introducing instance-specific wrongness.

**Falsification criteria (pre-registered, calibrated to per-cell SE ≈ 3.5 pp and difference SE ≈ 5 pp at N = 200 per cell):**

- **F1 (recursion HELPS).** Δ(s=0.5, K=8) ≥ +10 pp falsifies plateau (in the recursion-helps direction; Risk 2 of hypothesis).
- **F2 (recursion HURTS).** Δ(s=0.5, K=8) ≤ −10 pp falsifies plateau (in the lock-in direction; revision-0's original prediction).
- **F3 (no sparse-vs-dense interaction).** Per red-team I-B, the original 5 pp threshold was below the interaction SE; widened in this design to ±10 pp.
- **F4 (positive-control gate).** Δ_train(K=8) ≤ +10 pp on Sudoku-Extreme — operator broken; experiment uninterpretable.

The plateau prediction itself commits to **|Δ(s=0.5, K=8)| ≤ 3 pp with TOST p < 0.05 against ±5 pp equivalence margin**. Without the TOST commitment, "the 95% CI contains zero" is a confirmation pattern (absence of evidence ≠ evidence of absence). With TOST, the prediction is a positive statistical claim that can also fail.

---

## 2. Datasets

The hypothesis predicts a regime-specific (long-context OOD) effect. The design therefore needs three stimulus families: (a) a license-clean, position-controlled long-context stimulus where M1+M2+M3 should manifest plateau; (b) a multi-hop reasoning stress-test where the steelman (recursion buys real capacity) would predict Δ ≫ 0; (c) a training-distribution positive control where Δ_train ≥ 20 pp is the gate.

### 2.1 Primary stimulus — Constructed Biographical 2-Hop (CB2H)

**Status:** constructed in-house, license CC-BY-4.0 (clean).
**Why constructed (not redistributed NoLiMa).** NoLiMa (arXiv:2502.05167) is non-commercial; the spec mandates license-clean. We reconstruct the 2-hop-no-lexical-overlap design rather than redistribute, using the NeedleBench Ancestral Trace Challenge paradigm (arXiv:2407.11963; HF dataset `opencompass/NeedleBench` config `atc_needles`, Apache-2.0) as a methodological reference for biographical chain construction.

**Construction protocol** (deterministic seed; release as separate dataset):
- Persona schema: 5 attributes (birth_city, mother_birth_city, profession, hobby, residence_city) drawn from disjoint name pools of 1000 cities and 5000 first/last name pairs.
- Query is a **2-hop chain**: "What city was the mother of [persona] born in?" Needle paragraph contains "[persona]'s mother was born in [city]" or paraphrase ("her mother", "their mother") — **no surface form "mother" appears identically between query and needle in the lexical-control split**.
- Distractor density: fill remaining context with 50–500 persona paragraphs from non-target personae sharing one slot with target (same birth_city, different mother), creating attribute-confusable distractors.
- Position grid: needle at 8 positions ∈ {5%, 15%, 30%, 45%, 55%, 70%, 85%, 95%} of context length.
- Context lengths: **L ∈ {8K, 32K, 128K}** tokens (extending revision-1's drop-of-128K — the design budget allows the third tier; see compute section).
- Sample sizes per (L, position, K, s) cell: **200** (per-cell binary-outcome SE ≈ √(0.25/200) = 3.54 pp; difference SE ≈ 5.00 pp; interaction SE ≈ 7.07 pp). Total CB2H instances: 3 L × 8 pos × 4 K × 4 backbone × 200 = 76,800 instances. Generated once, deterministically, scored offline.

**Stimulus-validation gate** (red-team S2 from round 1 retained): before model evaluation, verify dense-attention K=1 baseline accuracy ≥ 50% at *some* position. If not, distractor density is too aggressive — reduce until this holds. This gate is committed *before* model runs and recorded in `manifest.yaml`.

### 2.2 Secondary stimulus — BABILong (k-step reasoning OOD)

- **Dataset:** `RMT-team/babilong` (HF; Apache-2.0). Inspected: 255 config/split rows covering qa1–qa20 across context lengths {0K, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K}.
- **Why included:** BABILong's qa3 (three-arg-relations), qa5 (three-arg coreference), qa15 (basic-deduction), qa18 (size-reasoning) are k-step reasoning subtypes the recursion-helps steelman (arXiv:2604.21106 — recurrence-equivalence exponent 0.46) most plausibly applies to. If F1 fires anywhere, it most plausibly fires here.
- **Splits used:** qa3, qa5, qa15, qa18 at L ∈ {8K, 32K, 128K}. 200 instances per (qa, L, K, s) cell.
- **License.** Apache-2.0 (verified via `hf_inspect_dataset`).

### 2.3 Positive-control — Sudoku-Extreme

- **Dataset:** `sapientinc/sudoku-extreme` (HF; license: open per dataset card — verify on retrieval; the TRM repo arXiv:2510.04871 publishes this as the eval set). Inspected: 405 MB train, 44.7 MB test; default config; columns {source, question, answer, rating}.
- **Why included:** This is TRM's reference benchmark. Verifies the operator under test still produces TRM-paper-consistent K-vs-1 deltas, so any null finding on CB2H/BABILong is attributable to OOD stimulus rather than a broken operator. F4 (positive-control gate) hangs on this.
- **Splits.** 1000 test instances at K ∈ {1, 2, 4, 8}. No long context — Sudoku is fixed-context.

### 2.4 Training data (re-train sparse-attention TRM operators)

- **Pretraining corpus for sparse-attention backbone:** `HuggingFaceFW/fineweb-edu` 10B sample (Apache-2.0, attribution required), restricted to sequence length up to 128K via packing. We use *independently-pretrained* sparse-attention checkpoints (red-team I3 of round 1: avoid post-hoc sparsification). For NSA, public checkpoint `deepseek-ai/NSA-base` if available (verify license at retrieval time), otherwise re-pretrain a 350M-param NSA-trained model on fineweb-edu under Apache-2.0.
- **TRM training data:** TRM's published Sudoku-Extreme training split (`sapientinc/sudoku-extreme`/train) for the K-pass operator's deep-supervision objective. Per arXiv:2510.04871 sec. 4.1.

### 2.5 Licence summary

| Dataset | HF ID | License | Verified | Use |
|---|---|---|---|---|
| CB2H | (constructed; release CC-BY-4.0) | CC-BY-4.0 | self-issued | Primary stimulus |
| BABILong | `RMT-team/babilong` | Apache-2.0 | inspected | Secondary OOD stress |
| Sudoku-Extreme | `sapientinc/sudoku-extreme` | per dataset card | inspected | Positive-control |
| NeedleBench (reference only) | `opencompass/NeedleBench` | Apache-2.0 | inspected | Methodological reference for CB2H construction |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | ODC-BY 1.0 | per dataset card | Pretraining (if needed) |

NoLiMa is *not* used as a stimulus dataset (license-incompatible per spec); CB2H is the license-clean replacement.

---

## 3. Models, backbones, and ablations

The hypothesis is mechanistically about the *interaction* between (a) deep-supervision-trained architectural recursion and (b) sparse-attention backbones at long context. The factorial design crosses these systematically.

### 3.1 Backbone factor (4 levels)

| Backbone | Sparsity ratio s | Source | Role |
|---|---|---|---|
| Dense (FlashAttention-2) | 1.0 | Standard transformer baseline | Reference; **flash-attention dense control** per assignment |
| NSA | ~0.5 | arXiv:2502.11089, repo `deepseek-ai/Native-Sparse-Attention` (or re-pretrained) | Primary sparse |
| MoBA | ~0.5 | arXiv:2502.13189, repo `moonshotai/moba` (2113 stars) | Secondary sparse — cross-pattern check |
| Sparse-Frontier Vertical-Slash | ~0.5 | arXiv:2504.17768 | Tertiary sparse — different geometry |

Independently pretrained sparse-attention checkpoints (NOT post-hoc sparsified). If pretraining cost is prohibitive, fall back to two backbones (dense + NSA only) — see "cheaper path" in section 8.

### 3.2 Recursion-depth factor (4 levels)

K ∈ {1, 2, 4, 8}. Reduced from revision-0's {1, 2, 4, 8, 16}: K=16 unnecessary for plateau-detection given Huginn's 4→32 plateau curve (arXiv:2507.02199 sec. 3.4).

### 3.3 Recursion-implementation factor (3 conditions)

| Condition | Description | Role |
|---|---|---|
| TRM-DS | TRM-style depthwise recursion with deep supervision per arXiv:2510.04871 sec. 4.1 | Primary operator under test |
| TRM-frozen-η=0 | TRM with the test-time-training gradient step disabled (η=0) | **Frozen TTT control** per assignment — isolates whether plateau is mechanism vs. residual TTT effect |
| Matched-FLOPs CoT | Standard CoT decoding under same backbone, total compute equal to K=8 forward passes | **Matched-FLOPs CoT-only control** per assignment — separates "K plateau" from "matched compute helps in any modality" |

### 3.4 Full factorial design

| Factor | Levels | Count |
|---|---|---|
| Backbone | dense, NSA, MoBA, Vertical-Slash | 4 |
| K | 1, 2, 4, 8 | 4 |
| Recursion-implementation | TRM-DS, TRM-η=0, CoT-matched | 3 |
| Context length L | 8K, 32K, 128K | 3 |
| Stimulus | CB2H, BABILong (qa3/5/15/18), Sudoku-Extreme | 3 |
| Position (CB2H only) | 8 positions | 8 |

**Pre-registered design.** Not all 4 × 4 × 3 × 3 × 3 × 8 cells are populated — Sudoku-Extreme has no L or position dimension; CoT-matched only at K=8; full grid is committed in supplementary `cells.csv`. The load-bearing cells are:
- Dense × K∈{1,8} × {CB2H, BABILong, Sudoku} × all L × all positions
- NSA × K∈{1,8} × {CB2H, BABILong, Sudoku} × all L × all positions
- MoBA × K∈{1,8} × {CB2H} × {32K} × all positions (cross-pattern check)
- All TRM-η=0 ablations only at K∈{1,8}, dense + NSA only

### 3.5 Model size

350M-parameter base model (matched across backbones). Choice rationale: TRM (arXiv:2510.04871) demonstrates the operator in 7M parameters, but 350M is the smallest scale where independently-pretrained NSA/MoBA checkpoints exist and the sparse-attention center-node bias predictions of Theorem 4.2 are operative on realistic sequences. Model size is a *secondary* factor flagged in the risk section.

---

## 4. Metrics

### 4.1 Primary metric — Δ(s, K)

Per the red-team C1 resolution: the primary metric is the **population-level K-vs-1 utility delta**:

Δ(s, K) = accuracy(s, K) − accuracy(s, K=1)

computed over 200 instances per (s, K, position, L) cell. **Population-level statistic, NOT per-instance.** Accuracy is exact-match correctness against the city-name target (CB2H), the answer string (BABILong), or the full Sudoku grid (Sudoku-Extreme).

Standard error: Δ has SE ≈ √(p(1−p)/n + p'(1−p')/n) ≈ 5 pp at p, p' ≈ 0.5 and n=200. CIs computed via bootstrap (10,000 resamples per cell).

### 4.2 Secondary metrics

- **U-curve depth U(s, K) = accuracy(end positions) − accuracy(mid positions).** End = {5%, 95%}; mid = {45%, 55%}. From arXiv:2307.03172 (Lost in the Middle). Per F4 / red-team I3.
- **Position-stratified accuracy A(s, K, pos).** Reports accuracy at each of 8 positions. Detects whether plateau (Δ ≈ 0) is uniform across positions or only at mid-context positions (which would be a more nuanced finding consistent with M2's center-node-sink prediction).
- **Attractor-basin perturbation Δ_pert(K=4).** For 50 random instances at K=4, perturb latent z with random ε-noise (ε ∈ {0.01, 0.05, 0.1} of z's L2-norm), re-run K=4→K=8. Report fraction of (perturbed, unperturbed) pairs producing identical K=8 answers. M1 prediction: high identity rate (contraction toward a fixed point); steelman alternative: low identity rate.
- **α-exponent in needle count.** Multi-needle CB2H variant N ∈ {1, 2, 4, 8}; fit accuracy ∝ N^(−α); report α(s, K=1) vs α(s, K=8). Tagged hypothesis-derived per red-team I5.

### 4.3 What the secondary metrics catch

- **U-curve flatness** would catch the case where overall Δ ≈ 0 but mid-context accuracy actually improved (uniform-shift lock-in artifact).
- **Position-stratified A** would catch the case where K=8 helps at position 5% but hurts at 70%, averaging to Δ ≈ 0 (would falsify the "uniform plateau" reading even though Δ passes TOST).
- **Δ_pert** catches the case where K=8 = K=4 by coincidence rather than by contraction (M1 in trouble).

---

## 5. Statistical analysis plan (pre-registered)

This is the load-bearing section. The hypothesis predicts a **null effect within an equivalence band**. Standard NHST is structurally unable to confirm a null. We use **TOST (Schuirmann 1987; Lakens 2017)** for the plateau equivalence claim and one-sided tests for the directional falsification arms.

### 5.1 Plateau test — TOST equivalence (primary outcome)

**Claim:** Δ(s=0.5, K=8) on CB2H at L=32K, mid-positions, is in equivalence band [−5, +5] pp.

**Test.** Two one-sided tests at α = 0.05:
- T_lower: H0_lower: Δ ≤ −5 vs H1_lower: Δ > −5 → reject if (Δ̂ + 5) / SE > z_{0.95}.
- T_upper: H0_upper: Δ ≥ +5 vs H1_upper: Δ < +5 → reject if (Δ̂ − 5) / SE < −z_{0.95}.
- **Plateau supported** iff both H0_lower and H0_upper rejected (Δ in [−5, +5] with 90% CI on Δ̂ ∈ [Δ̂−1.645·SE, Δ̂+1.645·SE] entirely inside [−5, +5]).

**Equivalence margin choice.** ±5 pp = 1× difference SE at N=200, slightly looser than the hypothesis's textual |Δ| ≤ 3 pp prediction. Rationale: the textual ±3 pp band cannot be distinguished from ±5 pp band at the planned sample size (per-cell SE = 3.54 pp; the 90% CI for an observed Δ̂ = 0 is [−5.8, +5.8], wider than ±3 pp). Pre-registering ±3 pp would set up a test that can never succeed even if the plateau is exact. **Pre-registration commits to ±5 pp as the operationalized equivalence margin.**

**Power.** With Δ_true = 0, SE = 5 pp, ε = 5 pp, α = 0.05: power = Pr(|Δ̂| < 5 − 1.645·SE) = Pr(|Δ̂| < −3.225) = 0 — the test as stated has zero power. **Resolution:** increase N to 400 per cell (SE drops to √2 × 3.54 / √2 = 3.54 pp; ε − 1.645·SE = 5 − 5.82 = −0.82, still problematic). Increase N to 800 per cell (per-cell SE = 1.77, difference SE = 2.50, ε − 1.645·SE = 5 − 4.11 = 0.89, power ≈ 0.64). **Final pre-registered N = 800 per (s, K) cell for the load-bearing CB2H L=32K mid-position cells**, total 800 × 4 × 4 × 4 = 51,200 instances at the core, plus the wider grid at N=200. Power ≈ 0.65; honest disclosure that this is below the conventional 0.80 — see "Risks to the experiment" section.

### 5.2 F1 / F2 — directional falsification (one-sided tests)

- **F1:** H0: Δ < +10 pp vs H1: Δ ≥ +10 pp. Reject H0 (i.e., F1 fires) if (Δ̂ − 10) / SE > z_{0.95}, i.e., Δ̂ ≥ 10 + 1.645·SE_diff = 14.1 pp at N=800.
- **F2:** H0: Δ > −10 pp vs H1: Δ ≤ −10 pp. Reject H0 if (Δ̂ + 10) / SE < −z_{0.95}, i.e., Δ̂ ≤ −14.1 pp.

### 5.3 F3 — interaction test

Δ(s=0.5, K=8) − Δ(s=1.0, K=8) on CB2H L=32K. Pre-registered threshold widened to ±10 pp per red-team I-B (originally 5 pp, below interaction SE of 7 pp). Interaction SE at N=800 = √(2 × 2.5²) ≈ 3.54 pp. F3 fires if interaction is in [−10, +10] (no interaction). Hypothesis predicts interaction ≤ −5 pp; demoted from primary falsification to secondary metric per red-team I-B.

### 5.4 F4 — positive-control gate

Sudoku-Extreme: Δ_train(K=8) on 1000 instances, SE ≈ √(0.25/1000) = 1.58 pp, difference SE ≈ 2.24 pp. F4 fires if Δ_train ≤ +10 pp = (Δ̂ − 10) / 2.24 < z_{0.95}, i.e., Δ̂ ≤ 13.7 pp. **Pre-flight check (red-team round-2 S-B):** verify TRM matched-(T,n) K=1-vs-K=8 delta on Sudoku-Extreme is ≥ 20 pp BEFORE locking F4 threshold. If our re-implementation gives K=1=70% and K=8=72%, the +20 pp prediction is wrong about the K=1-vs-K=8 framing and F4 needs recalibration. The pre-flight check happens in week 1; F4 threshold can be revised down to +5 pp if needed (with the revised threshold pre-registered before remaining experiments run).

### 5.5 Multiple-comparison correction

Primary outcomes: 1 TOST + 2 one-sided (F1, F2) + 1 gate (F4) = 4 tests. **Bonferroni at α = 0.05 / 4 = 0.0125** for the primary family. Secondary metrics (U-curve, position-stratified, Δ_pert, α-exponent, F3) reported with Benjamini-Hochberg FDR control at q = 0.10. Justified because the primary tests are *not* a screening family — they are a small pre-specified set of falsifiers.

### 5.6 Pre-registration commitment

Before any model is trained or evaluated, the following are committed in `pre-registration.json` and hashed (SHA-256) into `manifest.yaml`:

1. Sample sizes per cell (N=800 core, N=200 expanded grid).
2. Equivalence margin ε = 5 pp.
3. F1 threshold +14.1 pp; F2 threshold −14.1 pp; F3 threshold ±10 pp; F4 threshold +13.7 pp (subject to pre-flight revision).
4. Stimulus-validation gate (dense K=1 ≥ 50% at some position).
5. K=1 stability gate (long-context K=1 ≥ 30%).
6. Metric definitions (Δ, U, A, Δ_pert, α).
7. Bonferroni for primary family; BH-FDR for secondary.

### 5.7 Decision rule (concise)

**Hypothesis SUPPORTED** ⇔ all of:
- F4 does not fire (operator works on training distribution).
- Stimulus-validation gate passes.
- K=1 stability gate passes (K=1 long-context ≥ 30%).
- TOST equivalence at α = 0.0125 against ε = ±5 pp on CB2H L=32K mid-position. Both T_lower AND T_upper reject.
- F1 does NOT fire (Δ̂ < +14.1 pp).
- F2 does NOT fire (Δ̂ > −14.1 pp).

**Hypothesis FALSIFIED** ⇔ any of:
- F1 fires (Δ̂ ≥ +14.1 pp): plateau wrong, recursion helps.
- F2 fires (Δ̂ ≤ −14.1 pp): plateau wrong, recursion hurts (revision-0 was right).
- TOST equivalence rejected on at least one side at α = 0.0125 (Δ outside [−5, +5]).

**Inconclusive** ⇔ neither equivalence rejected NOR F1/F2 fire (Δ̂ in [+5, +14.1] or [−14.1, −5]). Honest disclosure of "directionally suggestive but inconclusive" per red-team feedback.

---

## 6. Falsification experiments (one per criterion)

### 6.1 Experiment F1-test — recursion HELPS on long-context OOD

**Design.** TRM-DS K∈{1, 8} × dense+NSA × CB2H L∈{8K, 32K, 128K} × 8 positions × N=800 (mid-position cells) or 200 (off-cells). Compute Δ(NSA, K=8) at L=32K averaged over mid-positions. Test F1.
**What constitutes falsification.** Δ̂ ≥ +14.1 pp: recursion under sparse + long-context produces a measurable improvement, contradicting M3. Plateau is wrong; the steelman of arXiv:2604.21106 (recurrence-equivalence exponent 0.46) is operative even on OOD long context. **State this in advance: if F1 fires, the hypothesis is refuted, the fusion thesis is in better shape than predicted, and revision-1 was too pessimistic.** No rhetorical escape.

### 6.2 Experiment F2-test — recursion HURTS on long-context OOD

**Same design as F1-test.** Test F2.
**What constitutes falsification.** Δ̂ ≤ −14.1 pp: revision-0's lock-in framing was right after all; revision-1's polarity flip was a mistake. Plateau is wrong. **State in advance: if F2 fires, return to revision-0-style mechanism.**

### 6.3 Experiment F3-test — sparse-vs-dense interaction

**Design.** Same instances and positions; compute Δ(NSA, K=8) − Δ(dense, K=8) at L=32K mid-positions.
**What constitutes falsification.** Interaction in [−10, +10] pp: M2's amplification claim (sparse-attention center-node sink × weight-tied recursion) is not load-bearing; the plateau (if observed) is *not* sparsity-conditional. Mechanism M2 is in trouble even if the plateau holds. Demoted to secondary per red-team I-B.

### 6.4 Experiment F4-test — positive-control gate

**Design.** TRM-DS K∈{1, 8} × dense × Sudoku-Extreme × N=1000.
**What constitutes invalidation (not falsification of H5).** Δ_train ≤ +13.7 pp. The operator does not behave as TRM should. The CB2H/BABILong measurements are uninterpretable. **NOT** a falsification of H5 — a kill of the experimental setup. Pre-registered as a gate.

### 6.5 Experiment K=1-stability-test (gate)

**Design.** TRM-DS K=1 × dense × CB2H L=32K × N=400.
**Gate.** If K=1 accuracy < 30% on long-context, TRM does not transfer to long context at all. Result is the finding "TRM does not transfer to long context" — different research output, NOT a falsification of H5. Pre-registered.

### 6.6 Experiment stimulus-validation-test (gate)

**Design.** Dense K=1 × CB2H L=32K × all 8 positions × N=200.
**Gate.** If max-over-position dense-K=1 accuracy < 50%, distractor density too aggressive. Reduce until gate passes. Pre-registered before model evaluation.

---

## 7. Ablations

The ablations isolate where (if anywhere) the plateau effect comes from. Each is a *targeted* disabling of one component of the proposed mechanism.

### 7.1 Frozen TTT control (η=0)

TRM-DS with the inner-loop TTT gradient step disabled. K∈{1, 8} × NSA × CB2H L=32K × N=400.
**Diagnostic role.** If plateau holds for TRM-η=0 (Δ ≈ 0) the same way as for TRM-DS, the plateau is a property of the *recursive operator structure*, not the TTT update. If TRM-η=0 shows Δ ≪ 0 (recursion hurts) and TRM-DS shows Δ ≈ 0, the deep supervision objective is genuinely contracting the iterates (M1 supported). If TRM-η=0 shows Δ ≈ 0 AND TRM-DS shows Δ > 0 (recursion helps), the TTT update is doing work that the structure alone is not — interesting but contrary to M1.

### 7.2 Matched-FLOPs CoT control

CoT decoding under same dense / NSA backbone with total FLOPs equal to K=8 forward passes. Matches the *amount of compute* across recursion vs explicit CoT.
**Diagnostic role.** If matched-FLOPs CoT shows Δ_CoT ≫ 0 on long-context OOD while TRM-DS K=8 shows Δ_TRM ≈ 0, the plateau is *modality-specific* (latent recursion plateaus, explicit CoT does not) — supporting M3. If both show Δ ≈ 0, the long-context OOD stimulus is unsolvable at this scale by either modality and the H5 finding is a regime-edge artifact.

### 7.3 FlashAttention dense control

Standard FlashAttention-2 dense backbone. Already in the primary backbone factor (s=1.0). Serves as the *dense* arm of F3.
**Diagnostic role.** If dense backbone shows Δ_dense > +5 pp on long-context OOD while NSA shows Δ ≈ 0, the sparsity is the active ingredient (M2 supported). If both show Δ ≈ 0, M2's center-node-sink prediction is not load-bearing.

### 7.4 K-sweep ablation

K ∈ {1, 2, 4, 8} per condition. Tests whether plateau onset is gradual or step-function.
**Diagnostic role.** Plateau prediction: Δ(s=0.5, K=2) ≈ Δ(s=0.5, K=4) ≈ Δ(s=0.5, K=8) ≈ 0. Steelman prediction: Δ ∝ K^0.46 (per arXiv:2604.21106), giving Δ(K=2) = +X·1.37, Δ(K=4) = +X·1.87, Δ(K=8) = +X·2.55 for some X>0.

### 7.5 Cross-pattern check (NSA vs MoBA vs Vertical-Slash)

Tests whether plateau is NSA-specific or generic to sparse attention. If only NSA shows plateau and MoBA/VS show Δ > 0, the finding is sparsity-pattern-specific (Risk 4 of hypothesis).

### 7.6 Context-length sweep

L ∈ {8K, 32K, 128K}. Tests at-what-context the plateau emerges. M2 prediction: plateau strengthens with L (per Theorem 4.2 rate ε^⌈(N−1)/(w−1)⌉, larger N → faster contraction).

---

## 8. Compute budget

Honest, non-TBD estimates. Calibrated against TRM (arXiv:2510.04871: 7M-param model, ~24 H100-hours per Sudoku run) and NSA (arXiv:2502.11089: 27B model, FLOPs profile public).

### 8.1 Pretraining of sparse-attention checkpoints (if needed)

If public NSA-base / MoBA-base 350M checkpoints exist, **skip** (cost = 0). If not:
- 350M-param NSA pretrain on FineWeb-Edu 50B tokens: ~2,000 H100-hours (estimated from NSA paper FLOPs scaling).
- Same for MoBA: ~2,000 H100-hours.
- Vertical-Slash: ~2,000 H100-hours.
- **Total worst-case pretraining: 6,000 H100-hours.**

### 8.2 TRM-style operator training

- Train TRM-DS K-pass operator on top of each backbone via deep supervision on Sudoku-Extreme (per arXiv:2510.04871 sec. 4.1). 4 backbones × 1 operator each. ~50 H100-hours per backbone (TRM is small; bottleneck is 4-pass deep-supervision unrolling).
- 4 × 50 = **200 H100-hours.**
- TRM-η=0 ablation: same training, additional 200 H100-hours.

### 8.3 Inference-only evaluation

For each cell, K passes × backbone forward × N instances. At L=32K and 350M params:
- Single forward at L=32K, dense: ~0.5 sec / instance on H100.
- Single forward at L=32K, NSA s=0.5: ~0.3 sec / instance.
- K=8 → 8× per instance.

Total inference cells:
- CB2H: 4 backbones × 4 K × 3 L × 8 positions × 800 instances × 1 op (TRM-DS) = 307,200 forward passes × avg 0.4 sec × K_avg 4 = ~5.5M GPU-seconds = ~1,500 H100-hours for CB2H.
- BABILong: 4 backbones × 4 K × 3 L × 4 qa × 200 = 38,400 forward passes; ~200 H100-hours.
- Sudoku-Extreme: 4 backbones × 4 K × 1000 = 16,000 forward passes (small context, fast); ~20 H100-hours.
- TRM-η=0 ablation: ~500 H100-hours.
- Matched-FLOPs CoT control: ~500 H100-hours.

**Total inference: ~2,720 H100-hours.**

### 8.4 Aggregate budget

| Phase | H100-hours | Notes |
|---|---|---|
| Backbone pretraining (worst case) | 6,000 | Skip if public checkpoints exist (likely halves) |
| TRM-DS + η=0 operator training | 400 | Across 4 backbones |
| CB2H inference | 1,500 | Largest stimulus |
| BABILong inference | 200 | |
| Sudoku-Extreme inference | 20 | |
| Ablation runs (η=0, CoT-matched) | 1,000 | |
| Buffer (re-runs, debugging) | 800 | 10% buffer |
| **Total** | **~9,920 H100-hours** | |

**Cost.** At $3/hr H100 spot: ~$30,000. At $4.50/hr H100 reserved: ~$45,000.

**Seeds.** All training runs at 3 seeds; all inference at 1 seed (deterministic given trained model). Reported intervals are bootstrap CIs over instances, NOT seeds, since per-cell N is large and instance-level variance dominates.

**Calendar time.** ~6 weeks on a 64-H100 cluster (10K hours / 64 GPUs ≈ 6.5 weeks).

### 8.5 Cheaper path (de-scoped variant)

If full budget is infeasible, the *minimum viable* design preserves the load-bearing TOST + F1/F2 commitments at reduced power:

1. **Drop two backbones.** Keep dense + NSA only. Saves 4,000 H100-hours pretraining if needed and ~1,400 inference. (Keeps F1/F2/F4. Loses F3-cross-pattern check.)
2. **Drop L=128K.** Keep 8K + 32K only. Saves ~30% inference. (M2's rate-prediction is partially testable at L=32K.)
3. **Drop BABILong.** CB2H + Sudoku only. Saves ~200 hours. (Loses k-step OOD generalization probe.)
4. **Reduce CB2H N to 400 per core cell.** TOST power drops to ~0.45. Honest disclosure: confirmation becomes harder; falsification (F1/F2) is still viable since they are one-sided and have lower N requirement.
5. **Skip Vertical-Slash + MoBA, skip CoT-matched.** Leaves dense + NSA + TRM-DS + TRM-η=0 only.

**Cheapest viable budget:** ~2,500 H100-hours, ~$8,000. This is the path I recommend if the user must triage.

### 8.6 Intractable-flag

Flagged as **NOT intractable** at full budget (~10K H100-hours, ~$30K, 6 weeks on a 64-GPU cluster — well within typical academic-lab budgets). Flagged as **tractable on cheap path** (~2.5K hours, ~$8K). The hypothesis IS testable.

---

## 9. Risks to the experiment

These are risks that the *experiment*, even if executed correctly, would yield a misleading result.

### 9.1 Baseline-tuning asymmetry

**Risk.** TRM training on the dense backbone vs sparse backbone may converge to different operating points; sparse-backbone TRM may be undertrained, making any sparse plateau a training-effort artifact.
**Mitigation.** Match training compute exactly across backbones (same FLOPs budget for the deep-supervision objective); report training-loss curves for all 4 backbones; require all to reach within 5% of best train-loss before evaluation.

### 9.2 Data leakage

**Risk.** CB2H is constructed from a fixed-vocabulary city/name pool. If the pretraining corpus (FineWeb-Edu) heavily features these cities/names, the model may have memorized priors that interact with K-pass refinement.
**Mitigation.** Use synthetic city names (rare-token combinations, e.g., "Velmorth", "Karzhan-on-the-Lake") with no FineWeb-Edu hits. Verify via document retrieval against pretrained corpus before stimulus finalization.

### 9.3 Evaluation-suite drift

**Risk.** The TRM operator is trained on Sudoku-Extreme; CB2H and BABILong are entirely OOD. If the OOD-ness is so extreme that K=1 is at ceiling (or floor), Δ is uninterpretable.
**Mitigation.** K=1 stability gate (≥ 30%) and stimulus-validation gate (≥ 50% dense-K=1 at some position). Both pre-registered as kill-criteria.

### 9.4 Scale generalization

**Risk.** 350M is small. The mechanism may behave differently at 7B+. The plateau finding may not transfer.
**Disclosure.** This is a known limitation. The hypothesis is not making cross-scale claims; it is making a specific scale-and-architecture claim at the size where independently-pretrained sparse-attention checkpoints are tractable to obtain. Cross-scale follow-up flagged as future work.

### 9.5 TOST under-power

**Risk.** Per section 5.1, even at N=800 the TOST power is ~0.65, below the conventional 0.80. A "plateau supported" result is a weaker claim than ideal.
**Mitigation.** Honest pre-registration; report the achieved CI width. If a future replication wants to strengthen the claim, increase N to 2000 (power ~0.85). The current design optimizes for *falsification* (F1/F2 power ~0.80 at N=800), not equivalence-confirmation power, since the hypothesis asks "can it be falsified?" not "is the null exactly true?"

### 9.6 Stimulus inversion

**Risk.** CB2H's 2-hop chain is not as adversarial as NoLiMa, since we are constructing it from scratch. The plateau may be inadequate-stimulus-difficulty rather than mechanism.
**Mitigation.** Run the secondary BABILong stimulus (qa3/15 are 3-hop). If plateau holds on CB2H but Δ ≫ 0 on BABILong, the hypothesis fails on BABILong (a falsification path); if plateau holds on both, the finding is robust.

### 9.7 Steelman engagement (red-team I-A)

**Risk.** arXiv:2604.21106 (recurrence-equivalence exponent 0.46) is contradictory in spirit. If on-distribution recursion is genuinely worth ~r^0.46 unique blocks, the baseline expectation is Δ ∝ K^0.46 even on OOD.
**Engagement.** The K-sweep ablation (section 7.4) tests this directly: under the steelman, Δ(K=2) = X·1.37, Δ(K=4) = X·1.87, Δ(K=8) = X·2.55. Under plateau, all four are ~0. The K-sweep is the discriminating test between the two predictions. This is a *direct* engagement with the steelman that the red-team round 2 flagged as missing.

### 9.8 "Plateau is floor effect" (Risk 3 of hypothesis)

**Risk.** If K=1 long-context accuracy is exactly 30% (gate threshold) and K=8 is exactly 30%, the plateau is a floor.
**Mitigation.** The stimulus-validation gate (dense K=1 ≥ 50% at some position) ensures non-trivial measurable signal range. If ALL positions are at floor, the gate fails and the experiment is re-stimuluated.

---

## 10. Sources

| Citation | arXiv ID | Role |
|---|---|---|
| TRM (Tiny Recursive Models) | 2510.04871 | Defines the operator under test (sec. 4.1); positive-control benchmark |
| Huginn (depth-recurrent latent reasoning) | 2507.02199 | Plateau evidence (sec. 3.4); load-bearing for M1 |
| Parcae (looped LM stability) | 2604.12946 | ρ(A̅) < 1 contraction; load-bearing for M1 |
| HRM mechanistic critique | 2601.10679 | Spurious attractors *for HRM specifically*; distinguished from TRM |
| Tunnel Vision in CoT (ParaThinker) | 2509.04475 | Discrete-token-commitment failure mode that recursion *escapes* |
| Position-Bias Emergence | 2502.01951 | Theorems 4.1–4.2; load-bearing for M2 |
| Lost in the Middle | 2307.03172 | U-curve metric |
| Hyper-multi-step | 2410.04422 | α-exponent (secondary diagnostic) |
| Score Dilution at Test Time | 2512.13898 | Long-context attention dilution (M3) |
| Long Context, Less Focus | 2602.15028 | Attention dilution mechanism (M3) |
| NoLiMa | 2502.05167 | Reference for 2-hop-no-lexical-overlap stimulus paradigm (license-incompatible; reconstructed in CB2H) |
| NeedleBench (Ancestral Trace) | 2407.11963 | Reference paradigm for biographical chains; HF dataset `opencompass/NeedleBench` |
| Native Sparse Attention (NSA) | 2502.11089 | Primary sparse backbone |
| MoBA | 2502.13189 | Secondary sparse backbone (cross-pattern) |
| The Sparse Frontier (Vertical-Slash) | 2504.17768 | Tertiary sparse backbone |
| Superlinear Multi-Step Attention | 2601.18401 | Closest published "multi-step subquadratic"; positions the gap |
| Iso-Depth Scaling Laws (recurrence equivalence) | 2604.21106 | Steelman: r^0.46 recurrence value; engaged in K-sweep ablation per red-team I-A |
| Recursive Latent Space Reasoning (OOD) | 2510.14095 | Adjacent: recursion + OOD on a different task; positioning |
| BABILong | 2406.10149 | Long-context k-step reasoning OOD; HF dataset `RMT-team/babilong`, Apache-2.0 |
| Sudoku-Extreme | (TRM repo) | Positive-control; HF dataset `sapientinc/sudoku-extreme` |
| FineWeb-Edu | (HF) | Pretraining corpus `HuggingFaceFW/fineweb-edu`, ODC-BY 1.0 |
| Lakens (TOST tutorial) | (10.1177/1948550617697177) | Statistical methodology for equivalence testing |
| Schuirmann (TOST original) | (10.1007/BF01068419) | Two-one-sided-tests procedure |

**Repositories referenced (verified via `hf_papers`/`github_examples`):**
- `SamsungSAILMontreal/TinyRecursiveModels` (6,496 stars) — TRM reference impl
- `moonshotai/moba` (2,113 stars) — MoBA reference impl
- `Relaxed-System-Lab/Flash-Sparse-Attention` (615 stars) — NSA-compatible kernel impl

**HF datasets verified via `hf_inspect_dataset`:**
- `RMT-team/babilong` ✓ (Apache-2.0; 255 config/split rows, qa1–qa20 across 0K–128K)
- `sapientinc/sudoku-extreme` ✓ (default config; 405 MB train, 44.7 MB test)
- `opencompass/NeedleBench` ✓ (5 configs including atc_needles)
