# Eval design for H6-SUB rev-1 — Sub-additive Redundancy Between TRM-style Depthwise Recursion and TTT-style Sequence-time Recursion

**Worker:** eval-designer-6
**For hypothesis:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-6/revision-1/output.md` (H6-SUB rev-1, red-team APPROVE verdict at round 2)
**Pre-registration date:** 2026-05-10. The decision rules in §5 are committed before any run executes.

---

## 0. Hypothesis being tested (restatement, with falsification criteria)

**Central claim.** Wrap a TRM-style weight-tied K_arch ∈ {1, 2, 4, 8} depthwise-recursion operator (arXiv:2510.04871 §2.4 deep supervision) around five ~125M-parameter backbones — **TTT-Linear** (arXiv:2407.04620), **LaCT** (arXiv:2505.23884), **Mamba** (arXiv:2407.14207 SSM lineage), **dense softmax attention** (TRM-faithful, arXiv:2510.04871 §4.5), and **frozen-η=0 TTT-Linear** (the F2 distinguisher; see §2.5 for the corrected interpretation per red-team I2) — and evaluate on CRUXEval-X (arXiv:2408.13001, HF `xhwl/cruxeval-x`) at AST-extracted program-recursion-depth d ∈ {1, 2, 3, 4, ≥5}, no-CoT primary condition.

**Predictions.**

- **F1 (primary, compression ratio).** `r = ΔA_TTT-Linear(K=4) / max(ΔA_Mamba(K=4), ΔA_dense(K=4), ΔA_η=0(K=4))` on the d ≥ 3 cohort, no-CoT. Predicted r ≈ 0.33. **Falsified** if r ≥ 0.8 (TTT shows ≥ 80% of best non-TTT gain → redundancy claim wrong) **OR** r ≤ 0.0 with mean direction (TTT actively hurt — different mechanism). r ∈ (0.0, 0.5] is the predicted band; r ∈ (0.5, 0.8) is an ambiguous zone triggering the seed-escalation pre-registered in §5.4.
- **F2 (mechanism distinguisher, the surviving steelman test).** `|A_TTT-Linear(K=4) − A_TTT-η=0(K=4)| ≤ 1.5` absolute points on the d ≥ 3 cohort. **Falsified** if |Δ| > 1.5. (Predicted ≈ 1.0.) This is the prediction the "TTT just has less headroom" steelman cannot easily explain.
- **F3 (CoT confound).** `r_CoT − r_no-CoT ≥ +0.2`. **Falsified** if r_CoT − r_no-CoT < +0.2 or sign-flip. (Predicted shift ≈ +0.37.)
- **F4 (depth-axis specificity).** Compression must grow with d: r_{d≥3} ≤ r_{d=1}; AND r_{d=1} ≥ 0.7 (compression should not appear at d=1). **Falsified** if either bound violated.
- **F5 (LaCT replication).** `ΔA_LaCT(K=4) − ΔA_TTT-Linear(K=4) ≤ +2.0`. **Falsified** if ΔA_LaCT exceeds ΔA_TTT-Linear by more than 2.0 abs points.

**Residual red-team round-2 issues to handle in this design (carried forward):**
- **I1 (Important, statistics).** CRUXEval-X 19-language expansion is not 19× independent — translations of the same ~800 base problems. Effective N for d ≥ 3 is ~80–150 base problems. Mitigated in §5 by hierarchical (problem-clustered) bootstrap + 10-seed pre-registration in the ambiguous r-zone.
- **I2 (Important, η=0 mis-characterised).** η=0 collapses to a position-wise MLP, NOT static linear attention. Mitigated in §2.5 by adding a true "fixed linear attention" backbone (FLA-stationary) alongside η=0 so F2 has the correct distinguisher.
- **I3 (Important, M2 over-generalises TRM §4.5).** Attention-removal in TRM helps only on Sudoku (L ≤ D). Mitigated in §2 by *not* relying on attention-removal at the construction level; we reframe M2 to depend only on "TRM re-runs the backbone-induced operator," supported by arXiv:2604.21106's recurrence-equivalence result.
- **I4 (Important, K_arch axis ambiguity vs TRM T/n/N_sup).** Mitigated in §2.6 by an explicit gradient-flow specification with an ablation (full-backprop-through-K_arch vs TRM-faithful "gradient through last process only").
- **I5 (Important, AST depth-extraction unspecified).** Mitigated in §1.3 with a concrete Python `ast` extraction script, plus a dry-run d-distribution table from CRUXEval-O.

---

## 1. Datasets

Every dataset is a real Hugging Face dataset; license verified via `hf_inspect_dataset`.

### 1.1 Primary: CRUXEval-X (15K items × 19 languages)

- **Name / HF ID:** `xhwl/cruxeval-x` (verified Status=Valid; configs `default` × splits {Java, Cpp, Go, CS, D, Julia, JavaScript, PHP, Perl, Python, Lua, R, Racket, Ruby, Rust, Scala, Shell, Swift, TypeScript} = **19 languages**, ~800 base problems each → 15,200 total items).
- **License:** MIT (verified via the upstream GitHub repo `CRUXEVAL-X/cruxeval-x` LICENSE; the HF redistribution mirrors the upstream license).
- **Verified via:** `hf_inspect_dataset xhwl/cruxeval-x` — schema `{id: int16, code: string, input_reasoning: string, output_reasoning: string}`. Sample rows confirm 19-language structure with shared `id` indexing the same base problem across languages.
- **Why this dataset.** The hypothesis predicts about per-instance program-recursion-depth d. CRUXEval-X is the only public benchmark giving (a) controllable per-instance computational structure (short Python/Java/etc functions with deterministic input → output traces), (b) a 19-language axis enabling cross-language averaging that reduces per-base-problem noise, and (c) compatibility with AST static analysis for d-extraction. CRUXEval-O is the predecessor used in §1.2 as a single-language cross-check.
- **Splits used.**
  - **Primary test (no-CoT):** All 19 languages × all base problems, partitioned by per-base-problem AST depth d ∈ {1, 2, 3, 4, ≥5}. The d ≥ 3 cohort (~80–150 base problems × 19 languages = 1,520–2,850 items, **but effective N is ~80–150 — see §5.2 power analysis**) is the F1, F2, F5 primary cohort.
  - **CoT-confound condition (F3):** Same item set, different prompt (chain-of-thought enabled per arXiv:2401.03065 §5 prompt template).
  - **d=1 negative-control (F4):** Same item set restricted to d=1 cohort.
  - **Held-out d-stratified leakage check:** All 19-language items where the model's pretraining corpus contains the exact problem string. Pre-registered as 0% leakage check via SHA-1 hashing of code blocks against pretraining corpus shards (see §1.5).

### 1.2 Secondary cross-check: CRUXEval-O (800 items, Python only)

- **Name / HF ID:** `cruxeval-org/cruxeval` (verified Status=Valid; default/test split, 800 examples; schema `{code, input, output, id}`).
- **License:** MIT (per upstream `facebookresearch/cruxeval` repo; HF card inherits).
- **Why.** Single-language Python cross-check that the cross-language averaging in CRUXEval-X is not driving the F1 ratio. At 800 items × per-d cohort the per-d count is small (estimated d ≥ 3: ~40-80 items per AST dry-run in §1.4) and statistical power is weak; CRUXEval-O is reported as a *consistency* signal, not a primary falsifier. Both `input_prediction` (predict input from code+output) and `output_prediction` (predict output from code+input) sub-tasks evaluated; primary metric is `output_prediction` (matches CRUXEval-X structure cleanly).

### 1.3 AST depth-extraction (resolves red-team I5)

The pre-registered AST extraction script. This is the *operational definition* of d, committed before any run.

```python
# d_extract.py — pre-registered AST depth-extraction for CRUXEval-X (Python AST).
# For non-Python languages, transpile back to Python via the CRUXEval-X 'id' field
# (which indexes the shared CRUXEval-O Python source) and run on Python AST.
# Rationale: CRUXEval-X language translations are mechanically generated from CRUXEval-O Python;
# d is therefore an attribute of the underlying Python program, shared across all 19 languages
# of the same id. This avoids per-language AST tooling and ensures d is well-defined per base problem.

import ast

def compute_d(code: str) -> int:
    """Returns max nesting depth of control-flow / function nodes inside `def f(...)`.

    Counts (each adds +1 to the running depth at its body):
      - ast.For, ast.While, ast.AsyncFor, ast.AsyncWhile  (loop nodes)
      - ast.If, ast.IfExp                                  (conditional nodes)
      - ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda  (nested def / lambda — self-reference proxy)
      - ast.comprehension (within ListComp/SetComp/DictComp/GeneratorExp)

    Does NOT count: assignments, ast.Call (function calls — call depth is a different axis),
    ast.With, ast.Try (resource/exception nodes that do not iterate or branch).

    Returns 0 for an empty body, 1 for straight-line code, 2 for one level of nesting, etc.
    For multi-function programs, returns max over functions. For programs with no control flow, d=1.
    """
    tree = ast.parse(code)
    counted_node_types = (
        ast.For, ast.While, ast.AsyncFor,
        ast.If, ast.IfExp,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
        ast.comprehension,
    )

    def depth(node, current=0):
        is_counted = isinstance(node, counted_node_types)
        new_d = current + (1 if is_counted else 0)
        child_max = current
        for child in ast.iter_child_nodes(node):
            child_max = max(child_max, depth(child, new_d))
        return max(new_d, child_max)

    return max(1, depth(tree, 0))


def bucket_d(d: int) -> str:
    if d == 1: return "d=1"
    if d == 2: return "d=2"
    if d == 3: return "d=3"
    if d == 4: return "d=4"
    return "d>=5"
```

**The hypothesis tests "iteration-of-reasoning depth," and the CRUXEval programs do not contain user-defined recursion** (per red-team I5 acknowledgment: most CRUXEval functions are non-recursive utility code; "deep" cases come from nested control flow). Pre-registered: d is **AST nesting depth of control/function nodes**, not call-graph recursion. This is the operational definition committed up front.

### 1.4 Dry-run d-distribution (committed numbers)

A dry run on CRUXEval-O (800 items) using the script above is **pre-registered for execution before any model training** and reported as part of the eval-design output. The expected (pre-registered upper-bound) per-d cohort sizes are:

| d-bucket | Estimated CRUXEval-O cohort | Estimated CRUXEval-X cohort (× 19 langs / shared base) | Effective base problems (pre-cluster) |
|---|---|---|---|
| d = 1 | ~40-80 | ~760-1520 | ~40-80 |
| d = 2 | ~250-400 | ~4750-7600 | ~250-400 |
| d = 3 | ~200-300 | ~3800-5700 | ~200-300 |
| d = 4 | ~80-150 | ~1520-2850 | ~80-150 |
| d ≥ 5 | ~30-80 | ~570-1520 | ~30-80 |

**The d ≥ 3 union cohort (the F1 primary surface)** is therefore ~310-530 base problems, which language-clustered to effective N ≈ 310-530 (NOT 5,890-10,070). This is the load-bearing number for the §5.2 power analysis. **If the dry run yields d ≥ 3 base-problem count below 200**, the eval-designer escalates to 15 seeds (per §5.4) before any main run — pre-registered.

### 1.5 Pretraining corpus

- **HF ID:** `HuggingFaceFW/fineweb-edu` (default/train) **+** `codeparrot/github-code-clean` Python/Java/Cpp/JavaScript/Go/Rust/Ruby/PHP/TypeScript subsets, mixed at 60% text / 40% code by token count. Verified Status=Valid for both. Licenses: ODC-BY-1.0 (FineWeb-Edu), permissive subset of GitHub-Code-Clean (Apache-2.0/MIT/BSD-2/BSD-3 sub-configs only, dropping copyleft to avoid license leakage).
- **Use.** All 5 backbones × 4 K_arch cells share identical 50B-token corpus, identical shuffle seed, identical tokenizer (Llama-3 32K BPE per arXiv:2407.04620 §3.2 for the 125M anchor). Identical pre-training corpus is **load-bearing** because per-arm differences must be attributable to (backbone, K_arch) — not to corpus drift.
- **Leakage check.** Before main run, all CRUXEval-O / CRUXEval-X code blocks are SHA-1 hashed and matched against the pre-training corpus shards. Pre-registered acceptance: **<0.5% exact-match rate**; if exceeded, the matched items are removed from the eval cohort and the d-distribution recomputed.

---

## 2. Backbones (treatment arms + baselines)

All five backbones share: tokenizer (Llama-3 32K BPE), optimizer (AdamW, β1=0.9, β2=0.95, wd=0.1, peak LR=3e-4 with cosine decay to 3e-5), training data (50B tokens of fineweb-edu + github-code-clean, fixed shuffle seed), context length (8K — adequate for CRUXEval-X function lengths well under 1K tokens), batch size (~2M tokens), gradient accumulation, init scheme. Only the sequence-time backbone module and the TRM outer-loop wrapper differ.

Parameter counts are matched to ~125M ± 5% per arm. Parameter-matching: TTT-Linear and LaCT include their inner-loop fast-weight buffers in the count (per arXiv:2407.04620 §3.2); dense and Mamba match by adjusting hidden width / depth.

### 2.1 Backbone B1 — TTT-Linear (PRIMARY treatment arm)

- **Architecture.** TTT-Linear backbone per arXiv:2407.04620 §2.1, 12 layers × 768 hidden × 12 heads, inner-loop fast weight `W_t ∈ ℝ^{d_k × d_v}`, inner-loop optimizer SGD with η = 1.0, mini-batch size 16, MLP wrapper around the inner update per the paper. Reference impl: `test-time-training/ttt-lm-jax` (cited in hypothesis §6).
- **TRM wrapper.** K_arch ∈ {1, 2, 4, 8} per §2.6.
- **Inner-loop reset (PRIMARY, TRM-faithful):** When outer iteration k+1 begins, TTT inner-loop fast weights `W_t` reset to the trained initialization `W_0` per the hypothesis's primary commitment (TRM §4.1 reads `f_L` from a fresh `(z_L+z_H+x)` each iteration).

### 2.2 Backbone B2 — LaCT (REPLICATION treatment arm)

- **Architecture.** LaCT backbone per arXiv:2505.23884 §3, 12 layers × 768 hidden × 12 heads, chunked TTT update with chunk size 256 per the paper. Reference impl: `a1600012888/LaCT` GitHub (`lact_llm/` subdirectory). Inner-loop fast weights persist within a chunk and reset at chunk boundaries per the LaCT paper.
- **TRM wrapper.** K_arch ∈ {1, 2, 4, 8} with reset at outer-iteration boundaries (matching B1 protocol).
- **Role.** F5 replication: LaCT is a TTT variant per arXiv:2602.21204 §5.3's linear-attention rewrite. If LaCT shows compression comparable to TTT-Linear, the M1 mechanism generalizes to chunked TTT. If LaCT shows *no* compression, the M1 mechanism is too narrow.

### 2.3 Backbone B3 — Mamba SSM (PRIMARY non-TTT control)

- **Architecture.** Mamba backbone per arXiv:2407.14207 (Longhorn) baseline lineage, matched-parameter 12 layers × 768 hidden × selective-state-update SSM. Reference impl: `state-spaces/mamba` GitHub. Architecture chosen because Longhorn frames Mamba as amortized online learning — i.e., it shares "sequence-time online update" with TTT but lacks the *per-token operator induction* of TTT (per M3).
- **TRM wrapper.** Same K_arch ∈ {1, 2, 4, 8} schedule.
- **Role.** Controls for "any-sequence-time-online-learner shows compression" vs "operator-induction-specific shows compression." If Mamba *also* shows compression, the redundancy mechanism reduces from "TTT-specific operator-iteration" to "any-online-learner" — still publishable but a weaker claim (smith R4).

### 2.4 Backbone B4 — Dense softmax attention (PRIMARY non-TTT control)

- **Architecture.** Standard dense Transformer, 12 layers × 768 hidden × 12 heads, RoPE positional encoding, matched parameter count. TRM-faithful per arXiv:2510.04871 §4.5.
- **TRM wrapper.** Same K_arch schedule.
- **Role.** The "stationary attention operator" baseline. Predicted to show the largest ΔA(K=4 − K=1) — its operator does not iterate at the sequence axis at all, so TRM's depth-axis is genuinely-novel iteration.

### 2.5 Backbones B5a + B5b — η=0 frozen TTT AND fixed-linear-attention (CO-PRIMARY F2 distinguishers)

This is the **eval-design-level fix for red-team I2** (the cited corollary "η=0 ⇒ static linear attention" was wrong; η=0 collapses to a position-wise MLP). Two backbones are run instead of one:

- **B5a — η=0 frozen TTT.** Same architecture as B1 (TTT-Linear) with inner-loop η = 0 set throughout *training*. Per the corrected reading of arXiv:2602.21204 §5.1 Theorem 5.1: this is `o_t = ϕ(q_t; Θ_0) · W_0` — a fixed position-wise MLP with no sequence mixing.
  - Role: Tests whether the *trained* inner-loop iteration property is what makes TTT redundant with TRM.
- **B5b — fixed linear attention (FLA-stationary).** A true linear-attention layer with `o_t = ϕ(q_t)(S_0 + Σ_{i ≤ t} ϕ(k_i)^⊤ v_i)` and **trainable but stationary** `W_q`, `W_k`, `W_v`, `S_0` (no inner-loop update at all; the full `S` is filled by the cumulative sum). This is the actual "static learned linear attention operator" the smith intended in §3 M4.
  - Role: Disambiguates F2. If A_TTT-Linear(K=4) converges to A_FLA-stationary(K=4) at K=4 (within 1.5 abs pts), the operator-induction property of trained TTT has been replaced by TRM's outer iteration. If A_TTT-Linear(K=4) converges to A_η=0(K=4) but NOT A_FLA-stationary(K=4), the convergence is to "no sequence mixing" rather than to "fixed linear attention," which is a *different* mechanism (cross-token information flow via TRM's (y, z) updates filling the gap).

The F2 falsification criterion is **revised to operate over both B5a and B5b**:
- F2-η0: |A_TTT-Linear(K=4) − A_η=0(K=4)| ≤ 2.0 (loosened from 1.5 per red-team's "marginally powered" finding; see §5.2 power calc).
- F2-FLA: |A_TTT-Linear(K=4) − A_FLA-stationary(K=4)| ≤ 2.0.

**Mechanism interpretation table for F2 outcomes:**

| F2-η0 holds? | F2-FLA holds? | Interpretation |
|---|---|---|
| YES | YES | TRM's outer iteration provides BOTH the trained inner-loop iteration value AND the sequence-mixing value. Strong support for M3. |
| YES | NO | TRM compensates for the inner-loop iteration but not for sequence mixing. Mixed: trained-η benefit replaced by TRM, but FLA's stationary attention still adds value. Weakly supports M3 with caveat. |
| NO | YES | TRM compensates for sequence-mixing of trained TTT (i.e., trained TTT collapses toward FLA at K=4) but the trained-η benefit remains. Inconsistent with M3 as stated; would suggest TRM provides sequence-mixing iteration rather than operator-induction iteration. |
| NO | NO | Neither convergence holds. F2 falsifies; M3 is wrong as stated. |

### 2.6 TRM wrapper construction (resolves red-team I4)

TRM has three axes (TRM §2-§4): `n` (inner f_L iterations per full recursion), `T` (full recursion processes per supervision step, T-1 without gradient), `N_sup` (supervision steps reusing latents). The hypothesis's K_arch is **operationalised as T**, with `n` fixed at 4 (TRM Table 1 best on Sudoku-Extreme used n=6 at 5M; we use n=4 at 125M as a conservative compromise; sensitivity ablation in §7.4) and `N_sup = 1` (single supervision step, no latent reuse beyond a single batch).

**Gradient-flow specification (PRIMARY):** Match TRM's "gradient through last process only" recipe: K_arch − 1 outer recursion processes run *without* gradient (detached), then the K_arch-th process runs *with* full backprop through all `n=4` evaluations of `f_L` and 1 of `f_H`.
**Gradient-flow ablation (SECONDARY, §7.3):** Full backprop through all K_arch outer iterations. Different from TRM but more directly tests the redundancy mechanism (deep supervision plays a smaller role; all gradient paths are exercised). Run only at K=4 across 5 backbones × 5 seeds (single ablation cell).

This commits the construction at the gradient-flow level, addressing red-team I4.

### 2.7 Trivial sanity baselines (B6, B7)

- **B6 — Random-token baseline.** Generates a random token from the vocabulary as the predicted output. Computed analytically (no compute spent). Establishes accuracy floor (~1% on output_prediction with single-token answers).
- **B7 — Nearest-neighbor on token-level.** For each test item, retrieve the nearest pretraining-corpus example by Jaccard token-overlap and copy its output. ~10 GPU-hours one-shot evaluation. Establishes the "memorize-and-retrieve" floor.
- **B8 — Strongest prior-art reference.** Published CRUXEval-O numbers for similarly-sized open models (Pythia-160M ≈ 24%, GPT-Neo-125M ≈ 27%, per CRUXEval paper arXiv:2401.03065 Table 2). No re-running needed; cited as anchor.

### 2.8 Backbone summary table

| Backbone | Sequence-time mechanism | Role | K_arch sweep | Seeds (primary) | Cells |
|---|---|---|---|---|---|
| B1 TTT-Linear | Inner-loop SGD on fast weights | F1, F2, F3, F4 primary treatment | {1, 2, 4, 8} | 5 (10 in ambig zone) | 4 |
| B2 LaCT | Chunked TTT-update, MLP-wrapper | F5 replication | {1, 2, 4, 8} | 5 | 4 |
| B3 Mamba | Selective SSM, no operator-induction | F1 control: any-online-learner test | {1, 2, 4, 8} | 5 | 4 |
| B4 Dense softmax | Static stationary attention | F1 control: pure-stationary test | {1, 2, 4, 8} | 5 | 4 |
| B5a TTT-η=0 | Position-wise MLP, no sequence mixing | F2-η0 distinguisher | {1, 2, 4, 8} | 5 | 4 |
| B5b FLA-stationary | True static linear-attention | F2-FLA distinguisher (red-team I2 fix) | {1, 2, 4, 8} | 5 | 4 |
| B6 Random | — | Sanity floor | — | — | analytic |
| B7 NN-retrieval | — | Memorize floor | — | — | 1 |
| B8 Prior-art ref | Pythia-160M / GPT-Neo-125M | Anchor | — | — | cited |

Total trained cells: **6 backbones × 4 K_arch × 5 seeds = 120 model-runs** for primary; **+ 5 backbones × 1 cell × 5 seeds = 25 runs** for the gradient-flow ablation (§7.3); **+ 5 backbones × 1 cell × 5 seeds = 25 runs** for the persist-vs-reset ablation (§7.1) — see §8 budget.

---

## 3. Metrics

All metrics are **pre-registered before any run executes**. Implementation: a single Python evaluation script frozen as `eval_h6sub_v1.py` and committed to a public artifact before any backbone training.

### 3.1 Primary metric M_primary — compression ratio r

**Formal definition.**
For each (backbone B, K_arch ∈ {1, 4}, seed s, cohort c) cell, compute mean exact-match accuracy `A(B, K, s, c)` over the items in cohort c. Define ΔA(B, c) = mean over s of [A(B, K=4, s, c) − A(B, K=1, s, c)]. The compression ratio is

```
r(c) = ΔA(B1=TTT-Linear, c) / max(ΔA(B3=Mamba, c), ΔA(B4=Dense, c), ΔA(B5a=η=0, c), ΔA(B5b=FLA, c))
```

Primary cohort c = d ≥ 3, no-CoT, all 19 languages averaged at the *base-problem* level (each base problem contributes a single mean accuracy averaged across its 19 language translations; the bootstrap in §5 resamples base problems, not language-instances — this is the I1 fix).

**Decision rule (F1).** Bootstrap 95% CI on r is computed by hierarchical (problem-clustered, then seed-clustered) bootstrap with 10,000 resamples. **Hypothesis is supported** if the upper 95% bound on r ≤ 0.5. **Hypothesis is falsified** if the lower 95% bound on r ≥ 0.8. **Ambiguous zone** (lower bound < 0.8 and upper bound > 0.5) triggers seed-escalation to 10 per cell (see §5.4).

### 3.2 Secondary metric M_sec1 — F2 distinguisher gaps

`Δ_η0(K=4) = A_TTT-Linear(K=4, d≥3) − A_η=0(K=4, d≥3)` and `Δ_FLA(K=4) = A_TTT-Linear(K=4, d≥3) − A_FLA-stationary(K=4, d≥3)`. Reported as means with bootstrap 95% CIs.

**Decision rule (F2).** Both |Δ_η0| ≤ 2.0 and |Δ_FLA| ≤ 2.0 ⇒ F2 supported. Either > 2.0 ⇒ F2 falsified (with the §2.5 interpretation table). Threshold loosened from hypothesis's 1.5 to 2.0 to address red-team's "marginally powered" power-analysis finding; see §5.2.

### 3.3 Secondary metric M_sec2 — CoT-confound shift

`r_CoT − r_no-CoT` on the same d ≥ 3 cohort. **Decision rule (F3).** ≥ +0.2 ⇒ F3 supported. < +0.2 or sign-flip ⇒ F3 falsified.

### 3.4 Secondary metric M_sec3 — depth-axis specificity

r computed on d=1 and on d≥3 cohorts. **Decision rule (F4).** r_{d=1} ≥ 0.7 AND r_{d=1} > r_{d≥3} ⇒ F4 supported. Either bound violated ⇒ F4 falsified.

### 3.5 Secondary metric M_sec4 — LaCT replication

`ΔA_LaCT(K=4) − ΔA_TTT-Linear(K=4)` on d ≥ 3 cohort. **Decision rule (F5).** ≤ +2.0 ⇒ F5 supported. > +2.0 ⇒ F5 falsified (LaCT diverges from TTT-Linear in a way that breaks M1's claim of LaCT-as-linear-attention from arXiv:2602.21204 §5.3).

### 3.6 Diagnostic metric M_diag1 — per-K curve shape

A(B, K, s, d) for K ∈ {1, 2, 4, 8} reported as line plots per backbone × per d-bucket. Captures non-monotone shapes that a single K=1-vs-K=4 contrast misses. Pre-registered Hartigan dip test on the per-cell seed distribution (per red-team round-1 S3) to detect bimodality.

### 3.7 Diagnostic metric M_diag2 — recurrence-equivalence-exponent calibration

Per arXiv:2604.21106's φ recurrence-equivalence exponent: fit `log A(K) ≈ φ · log K + c` per backbone. The hypothesis predicts `φ_TTT-Linear < φ_Mamba ≈ φ_dense` (saturation faster on TTT). Reported as a *consistency* diagnostic — not a primary falsifier — to anchor F1 against the looped-LM scaling-law literature (red-team round-2 S1).

### 3.8 Diagnostic metric M_diag3 — η sensitivity ablation

For B1 TTT-Linear, sweep inner-loop η ∈ {0.0, 0.1, 0.5, 1.0, 2.0} at K=4 only, single seed × d ≥ 3. Tests whether F2 convergence is *specific to* η=0 or shows a smooth η-dependence (the latter would be a softer mechanism story than a sharp η=0 collapse).

### 3.9 Trivial-baseline floor (sanity)

Random-baseline accuracy is reported analytically. Any backbone whose K=1 accuracy is within 2 abs pts of B6 random is flagged: that backbone failed to train. If TTT-Linear K=1 falls in this regime, the entire experiment is aborted (a kill-switch on the construction itself, per hypothesis R6).

---

## 4. Falsification experiments (one per criterion)

Each F-criterion has a pre-registered experiment with explicit kill thresholds.

| F-crit | Experiment | Arms | Cohort | Kill threshold (FALSIFIED if) | Support threshold |
|---|---|---|---|---|---|
| F1 | Compression ratio | B1, B3, B4, B5a, B5b × K∈{1,4} × 5 seeds | d ≥ 3, no-CoT, base-problem-clustered | 95% lower-CI on r ≥ 0.8 | 95% upper-CI on r ≤ 0.5 |
| F2-η0 | TTT-Linear vs η=0 at K=4 | B1, B5a × K=4 × 5 seeds | d ≥ 3, no-CoT | 95% CI on Δ_η0 excludes [−2.0, +2.0] | 95% CI ⊆ [−2.0, +2.0] |
| F2-FLA | TTT-Linear vs FLA-stationary at K=4 | B1, B5b × K=4 × 5 seeds | d ≥ 3, no-CoT | 95% CI on Δ_FLA excludes [−2.0, +2.0] | 95% CI ⊆ [−2.0, +2.0] |
| F3 | CoT-confound | B1 vs argmax-non-TTT × K∈{1,4} × {CoT, no-CoT} × 5 seeds | d ≥ 3 | r_CoT − r_no-CoT < +0.2, or sign-flip | r_CoT − r_no-CoT ≥ +0.2 |
| F4 | Depth-axis specificity | All backbones × K∈{1,4} × 5 seeds | d=1 AND d≥3 | r_{d=1} < 0.7 OR r_{d=1} ≤ r_{d≥3} | r_{d=1} ≥ 0.7 AND r_{d=1} > r_{d≥3} |
| F5 | LaCT replication | B2, B1 × K∈{1,4} × 5 seeds | d ≥ 3, no-CoT | ΔA_LaCT − ΔA_TTT-Linear > +2.0 | ΔA_LaCT − ΔA_TTT-Linear ≤ +2.0 |

**Each F-criterion can produce a result that disconfirms the hypothesis.** Pre-registered direction is committed in §0.

---

## 5. Statistical analysis plan (pre-registered)

### 5.1 Primary statistical test (F1)

**Hierarchical bootstrap with problem-level clustering, then seed-level resampling** (resolves red-team I1). Algorithm:

```
For B = 10000 bootstrap iterations:
  Step 1: Resample base problems with replacement from the d ≥ 3 cohort.
          Each sampled base problem brings ALL 19 of its language instances together.
  Step 2: For each backbone × K × seed, compute A on the resampled problem set.
  Step 3: Resample seeds with replacement (5 per cell -> 5 sampled with replacement).
  Step 4: Compute r* on the resampled set.
Report:
  median(r*), 2.5%-quantile, 97.5%-quantile.
```

This bootstrap correctly accounts for (a) base-problem-level variance (the dominant source per red-team I1) and (b) seed-level variance. It is the "clustered bootstrap that resamples problems-with-all-19-languages-as-units" the red-team requested.

### 5.2 Power analysis (revised per red-team I1)

Conservative effective N: at d ≥ 3 with ~310 base problems (mid-range of dry-run estimate §1.4), within-seed binomial std at 30% accuracy is `√(0.30·0.70/310) ≈ 2.6 abs pts`. SEM-on-mean across 5 seeds: `2.6/√5 ≈ 1.16 abs pts`. SEM-on-ΔA (subtracting K=1 from K=4 for one backbone): `1.16 × √2 ≈ 1.64 abs pts`. SEM-on-ratio r at predicted ΔA_TTT=2.0, ΔA_dense=6.0: `√((1.64/2.0)² + (1.64/6.0)²) ≈ 0.86`.

→ At 5 seeds, the bootstrap 95% CI on r is approximately ±1.7 (2× SEM) wide. With predicted r ≈ 0.33 and falsification threshold 0.8, we have 0.47 separation vs ±1.7 noise — **insufficient power at 5 seeds**.

→ At 10 seeds: SEM scales by `1/√2`; CI width drops to ±1.2. Still tight, but center at 0.33 ⇒ upper CI ≈ 1.5; still not cleanly excluding 0.8.

→ At 15 seeds: CI width ≈ ±1.0; upper CI on r ≈ 1.3.

**Pre-registered seed escalation rule:**
- **Phase A (initial):** 5 seeds per cell across all 6 backbones × 4 K_arch = 120 model runs.
- **Phase B (conditional escalation to 10 seeds):** Triggered if the Phase A bootstrap CI on r contains both 0.5 and 0.8 (the ambiguous zone). Adds 5 seeds × 6 backbones × 2 K_arch (K=1 and K=4 only) = 60 additional runs.
- **Phase C (conditional escalation to 15 seeds):** Triggered if the Phase B 95% CI on r still spans the falsification threshold 0.8. Adds another 5 seeds × the same 12 cells = 60 more runs.

This staged escalation is **load-bearing** for honest power and is reflected in the §8 compute budget.

**For F2 specifically (the marginally-powered criterion per red-team):** Pre-registered 10 seeds for B1 vs B5a and B1 vs B5b at K=4 from Phase A onwards. SEM on Δ_η0 at 10 seeds: `1.16 / √2 × √2 ≈ 1.16` abs pts; CI width ≈ ±2.3. The threshold 2.0 sits within the noise floor — this is the red-team's "marginally powered" concern. **Mitigation:** If F2's Phase A 10-seed CI contains both −2.0 and +2.0 (ambiguous), escalate F2 cells to 15 seeds (same trigger as F1 Phase C).

### 5.3 Multiple-comparison correction

Five pre-registered F-criteria → Bonferroni-corrected α = 0.05 / 5 = 0.01 per-test, or equivalently bootstrap 99% CIs. Pre-registered ordering of importance: F1 (primary), F2 (mechanism distinguisher), F3 (CoT confound), F5 (LaCT replication), F4 (depth specificity). Pre-registered: **F1 and F2 are co-primary**; F3-F5 are confirmatory secondaries reported with Holm–Bonferroni step-down at α = 0.01.

### 5.4 Pre-registered escalation rules

```
Decision tree (pre-registered, frozen 2026-05-10):
  A. Run Phase A: 5 seeds × 6 backbones × 4 K_arch.
  B. Compute F1 95% CI on r and F2-η0 / F2-FLA 95% CIs.
  C. If F1 95% CI ⊂ [0, 0.5]: REPORT "F1 supported"
     If F1 95% CI ⊂ [0.8, ∞]:  REPORT "F1 falsified"
     If F1 95% CI spans 0.5 OR 0.8:  ESCALATE to Phase B (10 seeds total).
  D. After Phase B: same decision; if still ambiguous, ESCALATE to Phase C (15 seeds total).
  E. After Phase C: report with current CI; do not run beyond Phase C (compute fence).
  F. Same staged rule applies independently to F2-η0 and F2-FLA.
```

The pre-registration is committed before Phase A begins; eval-design freezes the decision tree.

### 5.5 Inconclusive-result handling

If Phase C still leaves F1 in the ambiguous zone, the result is reported as **inconclusive**, NOT as "supported" or "falsified." This is the honest outcome and is pre-registered. (We do not allow post-hoc threshold relaxation.)

### 5.6 Pre-registration record

A pre-registration record file at `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/eval-designer-6/preregistration.json` is created **before any model training begins**, containing: F1-F5 thresholds, primary cohort (d ≥ 3), bootstrap protocol (problem-clustered hierarchical), seed escalation rules, multiple-comparison correction, AST extraction script SHA-1 hash, eval script SHA-1 hash. This file is read-only after Phase A starts.

---

## 6. Ablations

Five pre-registered ablations isolate where the predicted effect comes from.

### 6.1 A1 — Inner-loop fast-weight reset vs persist (PRIMARY architecture ablation)

- **Reset (primary):** `W_t` resets to `W_0` between TRM outer iterations (TRM-faithful).
- **Persist (ablation):** `W_t` from outer iteration k becomes the initial `W_0^{(k+1)}` for iteration k+1. Tests whether the hypothesis's "operator-induction sub-additivity" survives the persistent-state regime, which replicates 2602.21204 §4.1's "more inner-loop steps at training time" (the inverse of inference-time degradation).
- **Cost.** B1, B2 (TTT-Linear, LaCT) × K=4 only × 5 seeds = 10 runs. (Persistence undefined for B3-B5b; not run.)
- **Decision.** Predicted: persist variant shows even smaller ΔA(K=4), confirming saturation. If persist *increases* ΔA, the mechanism story is wrong (inner-loop iteration is not saturating).

### 6.2 A2 — Gradient-flow recipe (resolves red-team I4)

- **TRM-faithful (primary):** Gradient through last process only (K_arch − 1 detached, 1 with backprop).
- **Full-backprop ablation:** Backprop through all K_arch outer iterations.
- **Cost.** All 5 trained backbones × K=4 × 5 seeds = 25 runs.
- **Decision.** Tests whether F1 outcome depends on TRM's specific gradient-flow choice. Predicted: full-backprop shows stronger ΔA on dense/Mamba (more gradient signal) but the **ratio r is approximately preserved** because the gradient-flow asymmetry affects all backbones similarly.

### 6.3 A3 — η-sensitivity sweep

- η ∈ {0.0, 0.1, 0.5, 1.0, 2.0} on B1 (TTT-Linear) at K=4, single seed × d ≥ 3.
- **Cost.** 5 runs.
- **Decision.** Tests whether F2-η0 is a sharp transition or smooth. Smooth dependence weakly supports a graded-saturation mechanism.

### 6.4 A4 — d-bucket sweep (depth axis, F4 mechanism)

- All 5 backbones × K=4 × 5 seeds × {d=1, d=2, d=3, d=4, d≥5}. No new training; just per-d sub-cohort evaluation.
- **Cost.** 0 additional GPU-hours.
- **Decision.** Predicted r monotone increasing with d (compression grows with reasoning depth). Falsifies F4 if non-monotone.

### 6.5 A5 — Persist-with-detach gradient ablation

- Persist `W_t` across outer iterations BUT detach gradient at outer-iteration boundaries (only the last iteration's gradient flows through `W_t`).
- **Cost.** B1 × K=4 × 3 seeds = 3 runs.
- **Decision.** Disentangles "persistent state" from "gradient through persistent state." Diagnostic only.

---

## 7. Cheaper falsification path

This addresses the §6a path in the hypothesis and the spec's "cheaper-than-2000-hour kill-test" requirement.

### 7.1 Cheap-Path-A: open-weights TTT/Mamba/dense + light TRM fine-tune

1. Take open-weights TTT-Linear or LaCT pre-trained checkpoint at the closest available scale (`a1600012888/LaCT` GitHub release; `test-time-training/ttt-lm-jax` release).
2. Take open-weights Mamba and dense-Transformer 125M-class checkpoints (`state-spaces/mamba-130m` from HF; `EleutherAI/pythia-160m` for dense).
3. For each, fine-tune (not just test-time wrap) a TRM-style K_arch ∈ {1, 4} weight-tied wrapper on the final 4 transformer blocks for 2B tokens with deep supervision.
4. 3 seeds per (backbone, K_arch) cell × 4 backbones × 2 K_arch = 24 runs.
5. Evaluate CRUXEval-X at d ≥ 3.

**Expected cost.** ~150 GPU-hours (4 backbones × 2 K × 3 seeds × 2B tokens at 125M, on 8×H100; per-run ~3 hours; 24 × 3 = 72 hours of training + ~80 hours of evaluation overhead and logging).

**Kill-switch (cheap-path level).** If `r = ΔA_TTT-Linear / max(ΔA_dense, ΔA_Mamba) ≥ 0.7` on the cheap path, the §8 full-pre-training run is **not warranted** — the redundancy claim is dead. If `r ≤ 0.5`, the §8 run is funded with high confidence.

**Caveat.** Cheap-Path-A tests fine-tune-time wrapping on pretrained-without-recursion models, which is a *weaker probe* than full pre-training (per Ouro arXiv:2510.25741: iteration must be baked in at pre-training to fully manifest). A null cheap-path result is suggestive but does not strictly falsify the full prediction; a positive cheap-path result is strong evidence to fund §8.

### 7.2 Cheap-Path-B: η-sensitivity (no TRM training)

A cheaper-still bound: just the η-sweep on existing TTT-Linear checkpoint (A3 ablation alone). 5 GPU-hours. Not a full kill-test, but a cheap consistency check.

---

## 8. Compute budget

All numbers use the standard FLOP arithmetic for 125M Transformer/Mamba/TTT models, calibrated to the TTT paper's published 125M/50B-token costs (arXiv:2407.04620 §3.2).

### 8.1 Phase A (primary, 5 seeds)

- 6 backbones × 4 K_arch × 5 seeds = **120 runs**.
- Per-run compute: 125M params × 50B tokens × 6 FLOPs/param/token = 3.75e19 FLOPs/run.
- At 700 effective TFLOPs on H100 SXM: 3.75e19 / 7e14 = 5.4e4 sec/run = 15 hours/run.
- TRM K-wrapping multiplier: K_arch=1: ×1.0, K=2: ×2.0, K=4: ×4.0, K=8: ×8.0 (forward+backward through K outer iterations; gradient-through-last-process cuts to ×(1 + (K-1)·0.3) ≈ ×1.0/×1.3/×1.9/×3.1 with gradient-flow recipe — using the TRM recipe, *not* full backprop).
- Average multiplier across K ∈ {1,2,4,8}: ~1.83. Per-K average run cost: 15 × 1.83 = ~27 hours.
- **Phase A total: 120 runs × ~27 hours = ~3,240 GPU-hours** (TRM-faithful gradient-flow recipe, average across K).

### 8.2 Ablations (A1-A5)

- A1 (reset vs persist): 10 runs × ~15 hours = 150 GPU-hours.
- A2 (full-backprop): 25 runs × ~50 hours (full-backprop ×8 multiplier at K=4) = 1,250 GPU-hours.
- A3 (η-sweep): 5 runs × 15 hours = 75 GPU-hours.
- A4 (d-bucket sweep): 0 (post-hoc analysis).
- A5 (persist-with-detach): 3 runs × 27 hours = 81 GPU-hours.
- **Ablations total: ~1,556 GPU-hours.**

### 8.3 F3 (CoT condition) re-evaluation

CoT is an inference-time prompt change; no retraining. Inference-only on existing Phase A models at K∈{1,4} × CoT-on × 6 backbones × 5 seeds = 60 inference runs × ~5 GPU-hours each = 300 GPU-hours.

### 8.4 Phase B / Phase C escalation budget

If F1 enters the ambiguous zone (per §5.4):
- Phase B (5 → 10 seeds, 12 cells): 60 additional runs × ~27 hours = ~1,620 GPU-hours.
- Phase C (10 → 15 seeds, 12 cells): 60 more runs × ~27 hours = ~1,620 GPU-hours.

### 8.5 Total budget (honest)

| Phase | Cumulative cost | Budget posture |
|---|---|---|
| Cheap-Path-A only (kill-test) | ~150 GPU-hr | Under fence; runs first as gate |
| Phase A primary (no escalation) + ablations + F3 | ~3,240 + ~1,556 + ~300 = **~5,096 GPU-hr** | **OVER 2,000-hr fence** |
| Phase A + ablations + F3 + Phase B | ~6,716 GPU-hr | Over fence |
| Phase A + ablations + F3 + Phase B + Phase C | ~8,336 GPU-hr | Over fence |
| Hardware: 8× H100 SXM, ~700 effective TFLOPs |  |  |

**`flagged_intractable: true` for the full Phase A + ablations.** The minimum experiment that genuinely falsifies F1 (the primary criterion) is Phase A at ~3,240 GPU-hours, plus F3 condition (300 hours), totaling ~3,540 GPU-hours — above the 2,000-hour fence. The cheaper Path-A (§7.1, ~150 GPU-hours) is a viable kill-test for the trivial-null case (r ≈ 1.0) but does not falsify the full pre-training prediction (per the Ouro caveat).

### 8.6 Compute-reduction options (if Phase A budget is unavailable)

Pre-registered fallback rescoping rules:
- **R1 (drops dense):** 5 backbones instead of 6 — saves ~17%. F1's `max ΔA_other` over 3 controls instead of 4. Acceptable if the cheap path already established dense as the ΔA-largest control.
- **R2 (drops K=8):** K_arch ∈ {1, 2, 4} only. Saves ~25%. F1 unchanged (only uses K=1 and K=4). F4 weakened (no K=8 saturation point).
- **R3 (drops 4 of 19 languages):** Use 5 best-resourced languages (Python, Java, Cpp, Go, JavaScript). Saves ~50% of *eval* time (small; ~150 hr) but does NOT save training time. Effective N drops by less than 1.5× because language-correlation is high.
- **R4 (drops B5b FLA-stationary):** F2 reduces to F2-η0 only. Saves 17%. Loses the disambiguation of red-team I2.
- **Combined R1+R2+R4:** ~5,096 → ~2,800 GPU-hr — closer to fence but still over.

**Recommendation to user:** flag as intractable; the cheap-Path-A is the realistic in-fence kill-test, and the full Phase A run requires explicit budget approval or rescoping per R1+R2.

---

## 9. Risks to the experiment

Risks distinct from the hypothesis-level R1-R6 (which are about whether the prediction *holds*); these are about whether the *experiment is sound* even if the prediction is correct.

### 9.1 Data leakage (CRUXEval in pretraining)

CRUXEval-O was released January 2024; CRUXEval-X August 2024. Both pre-date plausible pre-training cutoffs. **Mitigation (§1.5):** SHA-1 exact-match scan of all CRUXEval items against pretraining-corpus shards; pre-registered acceptance <0.5%.

### 9.2 Baseline-tuning asymmetry

If TTT-Linear's hyperparameters are inherited from arXiv:2407.04620's reference (well-tuned) but Mamba's are quickly-set defaults, dense's are quickly-set defaults, etc., compression ratio r is confounded with tuning effort. **Mitigation:** All five backbones use a fixed shared hyperparameter recipe (Llama-3 BPE, AdamW with the values in §2 prelude), and per-backbone hyperparameters (TTT inner-η, Mamba dt_rank) are set to published reference values without per-K tuning. Seed-cluster bootstrap reports per-seed variance for each cell; if any backbone's K=1 seed variance is ≥3× another's at K=1, that flags a possible tuning asymmetry — pre-registered diagnostic.

### 9.3 TRM construction breaks at 125M

Per hypothesis R6: TRM is a 5M-parameter design; the K-wrapping may not work at 125M. **Mitigation:** Pre-registered kill-switch (§3.9): TRM K=4 dense-softmax must beat the same 125M dense-Transformer without TRM-wrapping. If this fails, Phase A is aborted; the cheap-Path-A on open-weights checkpoints becomes the only viable path.

### 9.4 Bimodal seed distributions

Smith R5 / red-team round-1 S3. **Mitigation:** Pre-registered Hartigan dip test (§3.6) on the per-(B, K) seed accuracy distribution. If bimodality detected at p < 0.01 in the K=4 cells, the bootstrap mean is misleading; report seed histograms alongside means and treat the affected cell as "split-regime" (not a clean falsification).

### 9.5 Cross-language correlation overstating effective N

Red-team I1's load-bearing concern. **Mitigation already designed in:** the §5.1 hierarchical bootstrap resamples at the base-problem level (each problem with all 19 languages as one unit). Power analysis in §5.2 uses ~310 base problems as effective N, not 5,890 item-instances.

### 9.6 Hyperparameter-recipe mismatch between TTT-Linear and η=0

The η=0 backbone (B5a) is trained from scratch with η=0; this is *not* the same as taking trained TTT-Linear and zeroing η at inference. Mitigation: this is the correct experimental control per smith §4 / §6 — it tests "what if we never had inner-loop iteration?" at training time. Reported alongside an inference-time-η=0 ablation on B1 (existing model with η=0 forced at inference) as a diagnostic — the gap between B5a and inference-time-zero shows how much of the trained backbone's parameters depend on η>0.

### 9.7 Eval-suite drift

CRUXEval-X HF dataset version may change. **Mitigation:** Pin the dataset commit SHA in the pre-registration record; download once before Phase A and store locally; do not re-pull during runs.

### 9.8 No CoT-helpful sub-split for CRUXEval-X

The CoT-helpful / CoT-harmful split is documented for CRUXEval-O (arXiv:2401.03065 §5) but not pre-extracted for CRUXEval-X. **Mitigation:** F3 CoT-confound condition uses a uniform CoT prompt template (the standard "Let's think step by step" prefix) on the full d ≥ 3 cohort, comparing r_CoT vs r_no-CoT. The "CoT-helpful sub-split" is reported as a secondary diagnostic on CRUXEval-O alone.

---

## 10. Sources

### 10.1 Hypothesis-cited papers (carried from hypothesis-smith-6)

- arXiv:2510.04871 — TRM (Less is More: Recursive Reasoning with Tiny Networks). §2.4 deep supervision, §4.1 full-recursion gradient, §4.5 attention-on ablation (NOT load-bearing in this design — only used for §2.6 axis specification).
- arXiv:2407.04620 — TTT (Learning to Learn at Test Time). §2.1 TTT updating hidden state, §3.2 125M-parameter setup.
- arXiv:2505.23884 — LaCT (Test-Time Training Done Right). §3 chunked TTT update.
- arXiv:2602.21204 — TTT as Linear Attention. §5.1 Theorem 5.1 (TTT ≡ learned linear attention), §5.2 inner-loop trajectory mechanism, §5.3 LaCT linear-attention rewrite.
- arXiv:2407.14207 — Longhorn (SSM as amortized online learning). Mamba control framing.
- arXiv:2510.25741 — Ouro. "Iteration baked in at pre-training" caveat for §7.1.
- arXiv:2502.05171 — Huginn. Depth-recursion at scale, scale-precedent for §2.6.
- arXiv:2401.03065 — CRUXEval. Test surface secondary; §5 CoT subsplits.
- arXiv:2408.13001 — CRUXEval-X. Primary test surface (15K items × 19 languages).

### 10.2 Added by eval-designer (per red-team round-2 suggestions)

- arXiv:2604.21106 — How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped LLMs. φ ≈ 0.46 recurrence-equivalence exponent. Used in M_diag2 (§3.7) as the saturation-of-recurrence consistency anchor and in the steelman defense (§9 risks).

### 10.3 Datasets (HF IDs, all licensed permissive)

- `xhwl/cruxeval-x` — MIT (verified). Primary test surface.
- `cruxeval-org/cruxeval` — MIT (verified). Secondary cross-check.
- `HuggingFaceFW/fineweb-edu` — ODC-BY-1.0 (verified). Pretraining corpus (text).
- `codeparrot/github-code-clean` — Permissive sub-configs only (Apache-2.0/MIT/BSD). Pretraining corpus (code).

### 10.4 Reference implementations (GitHub)

- `test-time-training/ttt-lm-jax` (cited via hypothesis §6) — TTT-Linear reference impl.
- `a1600012888/LaCT` (verified via web search) — LaCT reference impl.
- `state-spaces/mamba` — Mamba reference impl.
- `CRUXEVAL-X/cruxeval-x` (GitHub) — eval scripts upstream (MIT).

### 10.5 Eval scripts (this design's commits)

- `eval_h6sub_v1.py` — main evaluation script (to be committed before Phase A).
- `d_extract.py` — AST depth-extraction (committed in §1.3).
- `preregistration.json` — pre-registration record (committed in §5.6).
