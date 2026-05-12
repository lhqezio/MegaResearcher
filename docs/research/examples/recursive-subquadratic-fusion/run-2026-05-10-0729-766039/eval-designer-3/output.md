# Eval Design — H3 (Architectural recursion × sparse-attention substrate × retrieval-head retention)

**Hypothesis input:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-3/revision-1/output.md`
**Approving red-team round:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-3/revision-1/output.md`
**Output path:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/eval-designer-3/`

---

## 1. Hypothesis being tested (restatement, self-contained)

**Primary, load-bearing prediction (F2 differential).** When a TRM-mapped output-conditioned re-feed (the model's pass-k draft answer is appended as `"\nAnswer-so-far: <draft>\n"` to the prompt and the unmodified backbone is rerun) is layered over four sparse-attention substrates at matched sparsity ratio (NSA, MoBA, Quest, DSA) and a dense control, the *within-architecture retrieval-head retention* differential satisfies:

```
[ retention_NSA(K=4) − retention_NSA(K=1) ]  −  [ retention_MoBA(K=4) − retention_MoBA(K=1) ]   ≥   +0.05 absolute
```

where for architecture A, `H_A := {heads with retrieval score ≥ 0.1 at K=1 on A}` is identified independently per arXiv:2404.15574 §2 (≈ 600 instances), and `retention_A(K) := |{h ∈ H_A : score(h,K) ≥ 0.1}| / |H_A|` (so `retention_A(1) = 1` by construction).

**Secondary predictions.**
- F5 (ordering): rank-order of `retention_A(K=4) − retention_A(K=1)` across A ∈ {NSA, Quest, DSA, MoBA} matches **NSA > Quest ≥ DSA > MoBA**, with at most one swap permitted.
- F3 (task-level transfer on NoLiMa at 32K): `(NSA Δacc K=4 vs K=1) − (MoBA Δacc K=4 vs K=1) ≥ +3 percentage points`.
- F1 (consistency check, demoted): `retention_NSA(K=4) − retention_NSA(K=1) ≥ −0.10` (absolute K-effect on NSA must not be catastrophically negative).
- F6 (mechanism check, load-bearing for M2a): with NSA's compression branch zeroed (`g_cmp=0`), `retention_NSA-noCmp(K=4) − retention_NSA-noCmp(K=1)` should approximately equal MoBA's differential within ±0.03 absolute. If it remains as positive as full NSA's, M2a was not the load-bearing mechanism — the ML2-routing-quality alternative explanation (red-team round-2 steelman) takes over.

**Falsification gates.**
- F2 fires (differential < +0.05) → hypothesis **fully falsified**.
- F5 fires (>1 swap) → mechanism decomposition (M2a + M2b) wrong.
- F3 fires (< +3 pp on NoLiMa) → head-level effect doesn't propagate to task-level.
- F1 fires (Δ < −0.10) → absolute claim wrong (partial falsification).
- F6 fires → M2a not load-bearing → routing-quality alternative explanation favored.

**Sharpness diagnostic (auxiliary, not falsification on its own).** Track entropy `H_A,K := mean over h ∈ H_A and over needles of H(p^h_{q,K}(·))` of the head's per-key attribution distribution. If retention drops *and* entropy rises with K, smearing (R3) is the cause; if retention drops with entropy flat, the cause is selection failure (M2a/M2b not delivering signal).

---

## 2. Datasets

The hypothesis predicts about retrieval-head behavior on **needle-style retrieval at long context (32–64 K tokens)** under sparse attention × output-conditioned re-feed. We use one primary task-level benchmark, one OOD primary, and one synthetic NIAH protocol that anchors the retrieval-head identification step (head-level metric), plus two complementary stress sets.

### 2.1 Primary task-level benchmark — NoLiMa

- **HF ID:** `amodaresi/NoLiMa` (verified via `hf_inspect_dataset` — schema `text: string`, single `train` split, 5.4 MB, 1 parquet file)
- **arXiv:** 2502.05167 (Modarressi et al., ICML 2025)
- **License:** CC-BY-NC-4.0 (non-commercial; per the Adobe Research GitHub repo `adobe-research/NoLiMa`). **Flagged.** Use is permitted for research/non-commercial evaluation, which fits this academic-style protocol; if downstream productization is in scope, this dataset must be replaced.
- **Why appropriate:** NoLiMa is constructed to require non-literal-match retrieval (the question and the needle don't share surface tokens), which is exactly the regime where retrieval heads — and therefore M2a/M2b — are stressed. NoLiMa reports 5–10 point gaps between configurations at 32 K context per arXiv:2502.05167, making the +3 pp F3 threshold plausibly above noise.
- **Splits and sample sizes for this experiment:**
  - The HF dataset hosts only the haystack texts; needles are templated separately per the project repo `github.com/adobe-research/NoLiMa`. We will instantiate **500 NoLiMa items at 32 K context** and **200 items at 64 K context** per architecture × K combination, using the standard needle-set from the project's release.
  - Per-architecture × per-K, the same 500 fixed-seed items are used (paired design — the same context/question is reused across K values to remove item variance from the differential).
  - 5 re-feed sampling seeds (output temperature ≠ 0 for the draft answer) per (arch, K) cell.
- **Statistical power.** Power to detect a +3 pp differential at α=0.05 with item-paired design and 500 items: assuming per-item NoLiMa variance σ² ≈ 0.20 (Bernoulli at p≈0.5), per-arch SE ≈ √(0.5·0.5/500) ≈ 0.022, differential SE ≈ √2·0.022 ≈ 0.032. A 3 pp effect is therefore ≈ 0.94σ — **underpowered for a single-seed run**. With 5 seeds (paired across seeds within an item), effective N grows to ≈ 1500 effective comparisons, SE drops to ≈ 0.018, and the 3 pp effect becomes 1.66σ. Still marginal; we therefore report 95% CI on the differential and require the lower bound to exceed 0 as the F3 trigger (rather than only point estimate).

### 2.2 OOD primary — RULER (NIAH-13 + variable-length stress)

- **HF ID:** `simonjegou/ruler` (verified — configs at 4096 / 8192 / 16384 context; underlying NVIDIA RULER suite scales to 32K/64K/128K)
- **arXiv:** 2404.06654 (Hsieh et al., COLM 2024)
- **License:** Apache-2.0 (per `github.com/NVIDIA/RULER` and the HF mirror)
- **Why appropriate:** RULER provides 13 NIAH variants (single, multi-key, multi-value, multi-query, NIAHmv, common-words extraction, frequency-words extraction, QA, variable-tracking) — covering multi-step retrieval / aggregation cases that NoLiMa's single-needle mostly doesn't. It's the standard OOD task-level signal for long-context retrieval and would catch a NoLiMa-specific overfit. Frees us from NoLiMa's non-commercial license for the supplementary OOD result.
- **Splits and sample sizes:** Use RULER's `niah_single_1`, `niah_multikey_1`, `niah_multivalue`, `niah_multiquery`, and `vt` (variable-tracking) tasks at 16384 context (HF mirror) and the regenerated 32768 / 65536 contexts via NVIDIA's protocol. **300 items per task × 5 tasks = 1500 items per (arch, K) cell**.
- **Use:** Secondary task-level falsification path. The same F3-style differential (`(NSA Δacc K=4 vs K=1) − (MoBA Δacc K=4 vs K=1)`) must hold at ≥ +2 pp on the RULER aggregate to corroborate F3. (Threshold lowered to +2 pp because RULER variance is lower per item than NoLiMa due to discrete answers; pre-registered.)

### 2.3 Synthetic NIAH protocol (head-level metric anchor)

- **Source:** arXiv:2404.15574 §2 (Wu et al., Retrieval Head); reproduce the 600-instance protocol in-house (10 needle depths × 20 context lengths × 3 needle/value pairs). No HF dataset — generated procedurally from the paper's recipe.
- **License:** Procedurally generated; no license issue.
- **Why appropriate:** **Mandatory** anchor for the within-architecture retrieval-head identification step (defines `H_A` per architecture). The within-architecture retention metric is unintelligible without this protocol. NoLiMa items are not used to identify `H_A` — they are used only at the task-level (F3).
- **Sample sizes:** 600 instances per architecture for `H_A` identification at K=1; same 600 instances reused at K∈{2,4,8} for retention measurement. 5 re-feed sampling seeds for K>1.

### 2.4 Adversarial / stress OOD — Needle Threading

- **HF ID:** `jonathan-roberts1/needle-threading` (verified — 5 configs: Single_Needle, Multiple_Needles, Conditional_Needles, Single_Threads, Multi_Threads; up to 1 GB)
- **arXiv:** 2411.05000 (Roberts et al., 2024)
- **License:** Per HF dataset card — confirm before use; tentatively MIT/Apache-class but flagged for verification.
- **Why appropriate:** Conditional and threaded needle tasks stress the *multi-step* retrieval regime that single-pass top-k may particularly miss; output-conditioned re-feed should *help most* on these. Directional sub-prediction (auxiliary, not falsification): NSA + K=4 advantage over MoBA + K=4 should be **larger on Multi_Threads than on Single_Needle**, at the same context budget. If absent, the M2a-as-anchoring-residue-stream story is weakened.
- **Sample sizes:** 200 items per (Single_Needle, Multi_Threads) × per (arch, K) cell.

### 2.5 Beyond-the-Needle's-Illusion / EverMemBench-S — flagged but not committed

- **arXiv:** 2601.20276 (Lin et al., 2026)
- **HF status:** The hypothesis cites this as an OOD candidate, but the dataset is **not yet publicly verified on HF** (web_search returned only paper-abstract sites; no HF mirror surfaced). The paper describes a 326 M-token MemoryBank constructed for adversarial NIAH.
- **Treatment:** **Flagged as desirable-but-conditional.** If the authors have released the dataset by run-time, include it as an additional OOD set (300 items per arch × K, evidence-access metric per their §4). Otherwise: **drop in favor of RULER+Needle Threading**, both of which are verified-public. Do not block experiment kickoff on EverMemBench availability.

### 2.6 Datasets summary table

| Role | Dataset | HF ID | License | Splits used | Items per (arch, K) |
|---|---|---|---|---|---|
| Head-level metric anchor (M1) | Synthetic NIAH per arXiv:2404.15574 §2 | (procedural) | n/a | full 600-instance protocol | 600 |
| Primary task-level (F3) | NoLiMa | `amodaresi/NoLiMa` | CC-BY-NC-4.0 (flagged) | templated needles over haystacks | 500 @ 32K + 200 @ 64K |
| OOD primary task-level (F3 corroboration) | RULER (NIAH-13 subset) | `simonjegou/ruler` | Apache-2.0 | 5 tasks @ 16K/32K/64K | 1500 |
| OOD adversarial (auxiliary directional) | Needle Threading | `jonathan-roberts1/needle-threading` | MIT/Apache-class (verify) | Single_Needle + Multi_Threads | 400 |
| Conditional OOD (if released) | EverMemBench-S | (unverified) | TBD | TBD | 300 |

---

## 3. Backbones (architectures under test)

The hypothesis is predicated on substrates with **native** training; post-hoc emulation is the cheap fallback (§7 below). At full scale, we train five 1.3 B-parameter decoder-only transformers from a shared dense initialization, each then continued-pretrained with one of the substrate variants (or, for Quest, applied post-hoc since Quest is inference-only by design).

| ID | Substrate | Native vs post-hoc | Reference | Notes |
|---|---|---|---|---|
| BB-Dense | Full causal attention | native (control) | (standard) | Provides ceiling for retention; expected `retention(K=4) ≈ 1.0` (no signal lost). |
| BB-NSA | Native Sparse Attention (compressed + selected + sliding, gated) | native | arXiv:2502.11089 §3 | Block size 32 (per paper config), top-k blocks per query. Compression-branch zeroing supported via inference-time gate clamp. |
| BB-MoBA | Mixture of Block Attention (top-k block gate, no compression) | native | arXiv:2502.13189 §2.2 | Block size matched to NSA. |
| BB-Quest | Query-aware top-k page selection | post-hoc on BB-Dense | arXiv:2406.10774 §3 | Quest is inference-only by design; applied as a KV-cache page selector during forward pass. |
| BB-DSA | Lightning-indexer top-k routing | native | arXiv:2512.02556 | Scaled-down implementation of DeepSeek-V3.2's lightning-indexer. |

**Sparsity ratio.** Held constant at 0.10 across BB-NSA, BB-MoBA, BB-Quest, BB-DSA (top-k blocks/pages select 10% of the KV state). For BB-NSA, this is the selected-attention branch's top-k; the compression and sliding branches remain at full coverage but at reduced resolution / radius respectively, per the paper.

**Parameter scale.** 1.3 B is a deliberate compromise — large enough that arXiv:2404.15574's retrieval-head detection works robustly (their smallest model in Table 1 is Llama-2-7B; we target the smallest where the protocol still gives a non-degenerate `H_A` of size ≥ 6, which empirically holds at ≥ 1 B per `hf_papers` confirmations of the Retrieval Head literature on 1B-class models). Larger scale (7B) would strengthen the result but exceeds the 2000 GPU-hour fence (see §8).

**Training recipe.** Shared 100 B-token dense pretrain on a SlimPajama-class mix at 8K context, then 30 B-token continued pretrain at 32K with each substrate's native gating loss, then 10 B-token continued pretrain at 64K (ABF rope rescale). Total ≈ 140 B tokens per substrate variant.

**Pretrained-checkpoint fallback.** If MoonshotAI's `moonshotai/Kimi-K2.5` (MoBA-class) and a NSA-class open checkpoint become available at the 1–3B scale by run-time, they can substitute the from-scratch training for those substrates. As of this writing, no 1-3B-class native NSA checkpoint is publicly released. The cheap path (§7) handles this risk.

---

## 4. Baselines

We require ≥ 3 baselines per the protocol; we have 5, organized into three classes per the standard rubric.

### 4.1 Strongest prior-art baselines

- **B1 — G&A retrieval-head distillation (arXiv:2602.11374).** Bick et al.'s Gather-and-Aggregate-head preservation method, applied to the same 1.3 B backbone with each substrate. G&A is the *competing approach* (red-team I4) — it preserves retrieval-head function via supervision rather than via recursion. Predicted: G&A achieves higher absolute retention than recursion alone; whether the *NSA-vs-MoBA differential* persists under G&A supervision is an interesting open question. **Used as a competing-approach baseline, not as a co-evidence baseline.**
- **B2 — SSA: Sparse-to-Standard Alignment (arXiv:2511.20102).** Output-feature alignment training that explicitly addresses NSA/MoBA fragility on retrieval-head behavior, without recursion. **Direct competing baseline** (red-team I10). If SSA closes the gap between NSA and MoBA without recursion, the M2a story is undermined.
- **B3 — Two-pass CoT prompting (arXiv:2503.09819 Attrieval, structurally identical at one iteration).** The red-team I9 caveat: under the revision-1 mapping, output-conditioned re-feed *is* essentially two-pass CoT with answer-prepended input. The honest baseline is therefore the same K-pass protocol applied to **dense attention** (BB-Dense + K=4), which establishes the baseline gain attributable to two-pass CoT alone, independent of any sparse-attention mechanism. The hypothesis's M2a story specifically predicts that the sparse-attention substrate × K interaction exceeds dense × K — this is a non-trivial claim because dense should saturate quickly.

### 4.2 Ablations of the proposed technique

- **B4 — Compression-branch ablation (F6, mandatory).** BB-NSA with `g_cmp = 0` clamped during inference (or, more conservatively, retrained with `g_cmp = 0` for the final 5 B tokens of continued pretrain to allow the network to redistribute mass to selected + sliding). Predicted: `retention_NSA-noCmp(K=4) − retention_NSA-noCmp(K=1)` collapses to MoBA-class, within ±0.03 of MoBA's differential.
- **B5 — Naive output-replay ablation (M3 falsification).** Re-feed the model's pass-k *final hidden states at the answer position* (without textual draft), no input modification. Per the red-team round-1 critique, this does **not** change the queries at any layer at pass k+1. Predicted: no recovery for any architecture (zero recursion benefit). Confirms M2b is indeed the operative mechanism for the textual re-feed protocol.

### 4.3 Trivial / sanity baselines

- **B6 — Random-K-selection control.** At pass k+1, replace `concat(x, "Answer-so-far: ", y_k)` with `concat(x, "Answer-so-far: ", random_text_of_length(L_y))`. Predicted: no recovery (random text doesn't refine queries toward the needle). Distinguishes "appending any text shifts queries to recover" from "appending the model's own draft shifts queries informatively."
- **B7 — K=0 / no-recursion floor.** All architectures at K=1 only; the within-architecture metric returns 1.0 by construction, but the task-level NoLiMa accuracy at K=1 is a non-trivial floor and is the dominant comparison axis for F3.

---

## 5. Metrics

### 5.1 Primary metric (head-level, F2)

`Δretention_A := retention_A(K=4) − retention_A(K=1)`, with `retention_A(K) := |{h ∈ H_A : score(h, K) ≥ 0.1}| / |H_A|` and `H_A` identified per arXiv:2404.15574 §2 on architecture A's own forward pass at K=1.

**Argmax over the gated NSA mixture (resolves I1).** For NSA's three-branch attention, position `p`'s contribution to head output is `Σ_b g_b · attn_b(q, p)` summed across branches `b ∈ {compressed, selected, sliding}`; the head's attended position at query `q` is `argmax_p Σ_b g_b · attn_b(q, p)`. For Quest/DSA/MoBA (single-branch sparse), it's the standard top-1 over the post-gating attention distribution. For BB-Dense, it's the standard argmax over softmax(QK/√d). Per-architecture computation is symmetric.

**Score definition (verbatim from arXiv:2404.15574 §2).** A head's retrieval score at one instance is `|g_h ∩ k| / |k|` where `g_h` is the set of needle tokens for which the head's argmax-attention position lands inside the needle index range *and* the attended token equals the currently generated token; `k` is the needle. Average over 600 instances per arch.

### 5.2 Primary differential (F2)

`Δ_F2 := Δretention_NSA − Δretention_MoBA`. Pre-registered threshold: `Δ_F2 ≥ +0.05` and 95% CI lower bound > 0 over 5 seeds.

### 5.3 Ordering metric (F5)

Rank-order of `Δretention_A` across A ∈ {NSA, Quest, DSA, MoBA}. Predicted: NSA > Quest ≥ DSA > MoBA. Falsified if more than one swap occurs.

### 5.4 Task-level metric (F3)

NoLiMa exact-match accuracy at 32K context. Differential `Δ_F3 := (acc_NSA(K=4) − acc_NSA(K=1)) − (acc_MoBA(K=4) − acc_MoBA(K=1))`. Pre-registered threshold: `Δ_F3 ≥ +0.03` (3 pp) with 95% CI lower bound > 0.

RULER aggregate accuracy serves the same role with threshold `Δ_F3-RULER ≥ +0.02` (2 pp).

### 5.5 Mechanism check (F6)

`Δ_F6 := |Δretention_NSA-noCmp − Δretention_MoBA|` averaged over seeds. Pre-registered threshold: `Δ_F6 ≤ 0.03 absolute`. If `Δ_F6 > 0.03` (i.e., NSA-noCmp's recovery is more than 3 pp from MoBA's), then either M2a was *not* load-bearing (NSA-noCmp ≈ NSA, the routing-quality alternative explanation wins) or the F6 implementation introduced a confound — diagnosed by the sharpness probe and by the input-attention-mass rebalance check.

### 5.6 Auxiliary diagnostics (catch failure modes)

- **Sharpness probe.** `H_A,K := mean_{h ∈ H_A, needles n} entropy(p^h_{q_n,K}(·))`. Reported per (arch, K). If retention drops with K and entropy rises, R3 (smearing) is firing; if retention drops and entropy is flat, M2a/M2b is failing.
- **Per-K top-k overlap with needle block.** Fraction of needle blocks that are in selected-top-k at pass K. Auxiliary mechanism check: predicts NSA's selected-branch top-k at K=4 should include the needle block more often than at K=1, and the gain should exceed MoBA's at K=4.
- **NoLiMa-vs-RULER consistency.** `Δ_F3` and `Δ_F3-RULER` should agree in sign; if they disagree by more than 2 pp, dataset-specific overfitting is suspected.
- **Routing-quality probe (red-team round-2 steelman).** Post-hoc evaluate MoBA's gate signal-to-noise ratio (per FlashMoBA arXiv:2511.11571's diagnostic): if MoBA's gate SNR is < 1.5 at pass 1, the F2 differential may be driven by MoBA's poor routing rather than by NSA's compression channel. Documented but not falsifying — the F6 ablation is the disambiguator.

---

## 6. Statistical analysis plan (pre-registered)

### 6.1 Decision rule for primary outcome (F2)

We pre-register the F2 decision rule **before any data is observed**:

> **Accept H3** if and only if, across **five independent training seeds** (or, for cheap path, five inference seeds with temperature 0.7 on the draft-answer generation):
>
> 1. The point estimate `Δ_F2 ≥ +0.05 absolute`,
> 2. The 95% bias-corrected and accelerated (BCa) bootstrap CI on `Δ_F2` excludes 0,
> 3. F6 fires *as predicted* (i.e., `Δretention_NSA-noCmp ≈ Δretention_MoBA` within ±0.03 absolute),
> 4. F1 does not fire (`Δretention_NSA(K=4) − Δretention_NSA(K=1) ≥ −0.10`).

**Reject H3** if `Δ_F2 < +0.05` with the 95% CI excluding +0.05.

**Inconclusive** if the 95% CI on `Δ_F2` straddles +0.05 (cheap-test scale) — flag for full-eval scale-up before final decision.

### 6.2 Multiple-comparisons / FDR strategy

We test five primary falsification criteria (F1–F3, F5, F6) plus the auxiliary RULER consistency. Family-wise α = 0.05 controlled via **Benjamini-Hochberg FDR** on the family of `{F1, F2, F3, F3-RULER, F5, F6}` comparisons. The single primary outcome (F2) is reported at α=0.05 unadjusted because it is pre-registered as the load-bearing prediction; the others are reported with BH-adjusted q-values to control for the 6-comparison family.

### 6.3 Seed strategy

- **Full eval:** 5 independent training seeds × 5 re-feed sampling seeds = 25 seed combinations per (arch, K) cell. The pre-registered analysis collapses re-feed seeds into within-training-seed averages, then bootstraps over training seeds (the higher-variance level).
- **Cheap path:** 1 training seed (the post-hoc swap on a single dense backbone) × 5 re-feed sampling seeds. Bootstrap over re-feed seeds + items.

### 6.4 Effect-size estimator

Paired-item bootstrap (10 000 resamples) on `Δ_F2`. Report mean, 95% BCa CI, and Cohen's `d_z` for the differential. Pre-register `d_z ≥ 0.5` as a corroborating effect-size threshold.

### 6.5 Stopping rule

No early stopping. The full grid (5 substrates × 4 K-values × 5 datasets × 5 training seeds) is run to completion before any falsification call. This avoids garden-of-forking-paths pathologies that adaptive stopping introduces.

### 6.6 Pre-registration artifacts

The decision rule (§6.1), comparison family (§6.2), seed strategy (§6.3), and stopping rule (§6.5) are committed in this document at run start. Any deviation from the plan must be explicitly flagged in the synthesist's report as a post-hoc analysis.

---

## 7. Falsification experiments (one per criterion)

| Falsification # | Criterion | Experiment | Result that falsifies |
|---|---|---|---|
| F2 | Differential `Δ_F2 ≥ +0.05` | Full 5-substrate × 4-K grid on synthetic NIAH (head-level) at 32K context | `Δ_F2 < +0.05` with 95% CI lower bound below +0.05 |
| F5 | Ordering NSA > Quest ≥ DSA > MoBA | Same grid; rank-order `Δretention_A` | More than one swap from predicted order |
| F3 | Task-level differential `Δ_F3 ≥ +3 pp` on NoLiMa | Same grid evaluated on NoLiMa @ 32K | `Δ_F3 < +0.03` with 95% CI lower bound below +0.03 |
| F3-RULER | Corroborating differential on RULER | Same grid evaluated on RULER NIAH-13 subset @ 16K/32K/64K | `Δ_F3-RULER < +0.02` |
| F1 | NSA absolute retention loss | Read off `Δretention_NSA` from F2 grid | `Δretention_NSA < −0.10` (NSA catastrophically loses retrieval circuitry) |
| F6 | Compression-branch is load-bearing | Run BB-NSA with `g_cmp=0` clamp; compute `Δretention_NSA-noCmp` | `Δretention_NSA-noCmp − Δretention_MoBA > +0.03` (NSA-noCmp still recovers like full NSA, so M2a wasn't the cause) |
| Sharpness diag | Distinguish smearing from selection failure | Compute `H_A,K` entropy probe at all K | (Diagnostic only — does not falsify; informs interpretation) |
| Aux directional (Needle Threading) | NSA advantage larger on Multi_Threads than Single_Needle | Compute `Δacc_NSA-vs-MoBA` per task | If MultiThreads advantage ≤ SingleNeedle advantage, M2a-as-anchoring-residue story is weakened (not falsifying) |
| Aux mechanism (B5 naive replay) | Textual re-feed is load-bearing for M2b | B5: re-feed hidden states without textual draft | If B5 produces same recovery as textual re-feed, the "textual draft modifies queries" story is wrong; effect is hidden-state-side |
| Aux sanity (B6 random text) | Informativeness of draft is load-bearing | B6: append random text instead of draft | If B6 produces same recovery as textual re-feed, query-distribution-shift alone (not informative content) drives the effect |

**One falsification experiment per criterion is satisfied** (F1, F2, F3, F3-RULER, F5, F6 each have a dedicated experiment). The auxiliary experiments enrich the interpretation but are not load-bearing for the falsification decision.

---

## 8. Ablations

| Ablation | What's disabled | What it isolates | Predicted outcome |
|---|---|---|---|
| F6: NSA-noCmp | Compression branch (`g_cmp=0`) | M2a (compression-signal channel) | `Δretention` collapses to MoBA-class within ±0.03 |
| B5: hidden-state replay | Textual draft (replay hidden states only) | M2b vs hidden-state side-channel | No recovery (M2b operates via input modification, not residual) |
| B6: random-text re-feed | Informativeness of draft | Draft-content informativeness vs query-distribution-shift alone | No recovery (random text doesn't refine queries toward needle) |
| Sliding-only NSA (NSA-noSlide-AND-noCmp) | All branches except selected | Selected-branch only ≈ MoBA case | Should match MoBA's `Δretention` exactly (sanity check) |
| K-sweep | K ∈ {1, 2, 4, 8, 16} | Saturation of recursion benefit | NSA saturates at K∈{4,8}; MoBA may not saturate (linear, weak) |
| Sparsity-ratio sweep | top-k ratio ∈ {0.05, 0.10, 0.20, 0.50} on both NSA and MoBA at K∈{1,4} | Whether M2a-vs-no-M2a gap depends on sparsity strength | Differential largest at sparsity=0.05; vanishes at 0.50 |
| Block-size sweep on NSA | NSA compression block size ∈ {8, 16, 32, 64} | M2a's signal-to-noise dependence on pooling granularity | Smaller block size → larger `Δretention_NSA` (per-needle SNR ≈ 1/block-size) |

---

## 9. Compute budget

### 9.1 Full-scale path (with from-scratch substrate training)

| Component | Compute | Rationale |
|---|---|---|
| Dense backbone pretrain (shared) | 1 × 100 B tokens × 1.3 B params × 6 FLOP/token-param = 7.8e20 FLOP | Standard recipe |
| Continued pretrain per substrate (4 substrates × 30 B + 10 B tokens) | 4 × 4.0e20 = 1.6e21 FLOP | NSA, MoBA, DSA + control variants |
| Eval forward passes (5 substrates × 4 K × 5 seeds × 600 NIAH + 500 NoLiMa + 1500 RULER + 400 NeedleThread = ~3000 items per cell × 60 cells = 180K forward passes at 64K context) | ≈ 180K × 64K × 1.3 B × 2 = 3.0e19 FLOP | Cheap relative to training |
| Ablation runs (F6 noCmp, K-sweep, sparsity sweep, block-size sweep) | ≈ 0.5e21 FLOP | All inference-only or short retrains |
| **Total** | **≈ 2.9e21 FLOP** | |
| **Translated to GPU-hours on H100 (≈ 1 PFLOPS sustained for dense, 0.5 for sparse)** | **≈ 1.6K — 3.2K H100-hours** | |
| **5 seeds for stochastic stability** | **≈ 8K — 16K H100-hours** | **Exceeds the 2000 GPU-hour fence.** |

**Flag: full-scale path is intractable at the protocol-required 5-seed level on the YAGNI fence.**

We therefore split the eval into two scales:

### 9.2 Cheap path (recommended, fits the fence)

Per the hypothesis §5a — single dense-pretrained 1.3 B backbone, **post-hoc** apply each sparse-attention mask (NSA-emulation with hand-implemented compressed branch via mean-pooled K/V over each block; MoBA-emulation with top-k blocks; Quest with per-page top-k; DSA-emulation with a frozen 2-layer MLP indexer trained for 1 B tokens to mimic top-k from a known-good signal). Output-conditioned re-feed at K∈{1, 2, 4, 8} on:

- **200 NoLiMa items @ 32K** (cheap test in hypothesis §5a — increased from 50 per I2)
- **300 RULER items @ 16K** (NVIDIA mirror, no regeneration needed)
- **600 synthetic NIAH** for `H_A` identification per arch

| Component | GPU-hours |
|---|---|
| Dense backbone pretrain | (assumed available — Llama-3.2-1B or Qwen-2.5-1.5B as base) |
| Indexer training for DSA-emulation | ≈ 200 H100-hours |
| Forward-pass eval (5 substrate-masks × 4 K × 5 seeds × ~3000 items × ~32K avg context × 1.3B) | ≈ 600 H100-hours |
| Ablations (F6, B5, B6, sweeps at limited scope) | ≈ 400 H100-hours |
| **Total** | **≈ 1200 H100-hours** ✓ within fence |

This **cheap path** is the recommended primary execution. The full-scale path (§9.1) is a follow-up if the cheap path corroborates F2 strongly.

### 9.3 Acknowledged caveats

The cheap path has the R1 caveat (post-hoc swaps may exhibit pathologies that contaminate the K-axis effect). The hypothesis §5a explicitly acknowledges this: a *strong negative* result on the cheap path is informative (kills the full eval); a *marginal* result is "inconclusive at cheap scale" and triggers the full eval. The pre-registered decision rule (§6.1) handles both cases.

---

## 10. Risks to the experiment (could make a real result misleading)

| Risk | What goes wrong | Mitigation |
|---|---|---|
| **D1. Data leakage on NoLiMa** | The base-dense model saw NoLiMa-class needle patterns in pretraining; retention measured on NoLiMa is inflated. | Use RULER + Needle Threading as OOD corroboration. F3 must hold on at least *both* NoLiMa and RULER for the corroboration to count. |
| **D2. Baseline-tuning asymmetry** | NSA tuned more carefully than MoBA, biasing F2 in NSA's favor. | Use shared-base post-hoc masks (cheap path) — no per-substrate tuning. For full path, use the **same hyperparameters** for all substrate continued pretrains (learning rate, batch size, schedule); pre-register this. |
| **D3. Routing-quality confound (red-team round-2 steelman)** | F2 is real but driven by MoBA's bad routing, not by NSA's compression channel. | F6 (compression-branch ablation) is the disambiguator. We additionally report MoBA's gate SNR per arXiv:2511.11571's diagnostic. |
| **D4. Two-pass CoT overlap (red-team I9)** | Output-conditioned re-feed is operationally identical to two-pass CoT; the "architectural recursion" framing is mostly relabelling. | B3 (dense + K=4) controls for the dense-attention-only K-effect; the M2a story specifically predicts the differential persists *beyond* the dense-K floor. The paper's framing should be reframed (per red-team I9) as "asymmetric robustness of NSA's compression-channel under iterative query refinement" — synthesist task. |
| **D5. K=1 retention at ceiling (R2)** | All architectures preserve their own retrieval heads at K=4 at 32K; differential is too small to measure. | Stress-test at 64K and at adversarial NoLiMa distractor configurations. Pre-register a 64K-context-only fallback if 32K shows ceiling. |
| **D6. Recursion-induced smearing (R3)** | Retention drops because the K-iteration smears query mass, not because selection fails. | Sharpness probe (entropy of head's attention distribution) diagnoses smearing. F2 differential survives symmetric smearing; F1 (consistency) catches asymmetric harm. |
| **D7. EverMemBench unavailability** | Cited OOD set has no public release at run-time. | RULER + Needle Threading suffice as OOD; EverMemBench is desirable-but-conditional. |
| **D8. NSA / MoBA from-scratch training costs** | Full path exceeds the GPU-hour fence. | Cheap path (§9.2) is the recommended primary; full path is conditional follow-up. |
| **D9. Within-architecture metric: H_A may be tiny** | At 1.3B scale, an architecture's `H_A` may be < 6 heads, making retention quantization coarse. | Pre-test on each substrate at K=1 to confirm `|H_A| ≥ 8`. If smaller, lower the retrieval-score threshold from 0.1 → 0.05 (per arXiv:2404.15574 §3 sensitivity discussion) to enlarge `H_A`. |
| **D10. Eval-suite drift across K** | The exact items used for `H_A` identification at K=1 differ from items at K>1 (would compromise within-architecture comparison). | Use *the same 600 instances* for `H_A` identification and for retention measurement at K>1 (paired design). Pre-register this. |

---

## 11. Sources

- **arXiv:2404.15574** — Wu et al., "Retrieval Head Mechanistically Explains Long-Context Factuality." (Head-level metric anchor; protocol; threshold = 0.1.)
- **arXiv:2502.11089** — Yuan et al., "Native Sparse Attention." (BB-NSA substrate; compressed + selected + sliding three-branch design; block size 32; F6 ablation target.)
- **arXiv:2502.13189** — Lu et al., "MoBA: Mixture of Block Attention." (BB-MoBA substrate; top-k gate, no compression channel.)
- **arXiv:2406.10774** — Tang et al., "Quest." (BB-Quest; query-aware page-level top-k; inference-only.)
- **arXiv:2512.02556** — DeepSeek-AI, "DeepSeek-V3.2." (BB-DSA; lightning-indexer top-k routing.)
- **arXiv:2510.04871** — Jolicoeur-Martineau, "Tiny Recursive Model." (Recursion mapping; output-conditioned re-feed.)
- **arXiv:2502.05167** — Modarressi et al., "NoLiMa." (Primary task-level dataset. HF: `amodaresi/NoLiMa`. License: CC-BY-NC-4.0 — flagged.)
- **arXiv:2404.06654** — Hsieh et al., "RULER." (OOD primary task-level dataset. HF: `simonjegou/ruler`. License: Apache-2.0.)
- **arXiv:2411.05000** — Roberts et al., "Needle Threading." (Auxiliary stress OOD. HF: `jonathan-roberts1/needle-threading`. License: verify per dataset card before use.)
- **arXiv:2406.10149** — Kuratov et al., "BABILong." (Considered as additional OOD; HF: `RMT-team/babilong`. License: Apache-2.0. Not committed in primary plan; can be folded in if EverMemBench is unavailable.)
- **arXiv:2601.20276** — Lin et al., "Beyond the Needle's Illusion / EverMemBench-S." (Conditional OOD; flagged as desirable-but-not-required pending HF availability.)
- **arXiv:2602.11374** — Bick et al., "Retrieval-Aware Distillation for Transformer-SSM Hybrids." (B1 competing baseline — G&A.)
- **arXiv:2511.20102** — SSA paper. (B2 competing baseline — sparse-to-standard alignment.)
- **arXiv:2503.09819** — Attrieval. (B3 reference for the two-pass CoT structural equivalence; red-team I9.)
- **arXiv:2511.11571** — FlashMoBA / Optimizing Mixture of Block Attention. (Red-team round-2 steelman for routing-quality confound; D3 mitigation.)
- **arXiv:2502.05171** — Geiping et al., "Huginn / Recurrent-Depth." (R3 framing — task-dependent effects of depthwise recursion. Per red-team I6, framing softened from "underperforms on needle-style" to "depthwise recursion has task-dependent effects.")
- **arXiv:2503.10799** — Movahedi et al., "Fixed-Point RNNs." (R3 framing as analogy — recursion as fixed-point iteration. Per red-team S6, marked as analogy not direct support.)
- **arXiv:2510.20787** / **arXiv:2602.11698** / **arXiv:2601.21582** — SpiralFormer / Dreamer (depth-recurrent architectures with sparse attention; cited as adjacent prior art for non-overlap per red-team I8).

**Dataset HF IDs (verified via `hf_inspect_dataset`):**
- `amodaresi/NoLiMa` — verified, single train split, 5.4 MB.
- `simonjegou/ruler` — verified, configs at 4096/8192/16384.
- `RMT-team/babilong` — verified, 255 config/split combos covering 0K–10M context.
- `jonathan-roberts1/needle-threading` — verified, 5 configs.
- EverMemBench-S — **not verified on HF**, flagged as conditional.
