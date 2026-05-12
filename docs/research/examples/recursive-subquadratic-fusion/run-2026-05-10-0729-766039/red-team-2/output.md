# Red-Team Critique of H2 — TC⁰ escape via external TRM-style recursion on diagonal SSMs

## 1. Verdict

The hypothesis surfaces a real, unstudied gap (no SSM evaluated on PutnamBench/miniF2F via depth-time recursion) and the falsification suite is unusually well-structured. But the *load-bearing* mechanism — that Fixed-Point RNNs are "contraction-bounded and CANNOT represent permutation composition" while TRM is unconstrained — is **factually wrong against the cited paper**, and the depth-buys-expressivity import from CoT-Solves-Serial is a non-trivial extrapolation the source paper does not endorse. Several magnitude predictions (2% L_long baseline, ≥+5 absolute delta on ≤640+244 problems) are not credibly calibrated, and the "frozen Mamba + TRM wrapper" architecture is unprecedented and contradicts both TRM's and HRM's published training recipe. These are repairable but require non-cosmetic surgery.

**Verdict line at end of file.**

## 2. Gap re-verification

Independent literature queries I ran via `hf_papers search` (all 2026-05-10):

| Query | Top hits relevant to gap | Verdict |
|---|---|---|
| `TRM tiny recursive model SSM Mamba` | TRM (2510.04871) is on Sudoku/Maze/ARC-AGI puzzles only. No SSM hit. | Gap survives. |
| `Mamba PutnamBench Lean formal proof` | No Mamba+PutnamBench paper. M1 (2504.10449) is closest — Mamba CoT reasoning on math, not formal proof, not architectural recursion. | Gap survives. |
| `SSM linear attention theorem proving Lean miniF2F` | All hits are Transformer/LLM-based provers; no SSM/linear-attention prover papers. | Gap survives. |
| `Mamba RWKV linear RNN PutnamBench miniF2F CoqGym formal` | Same — no SSM theorem-proving evaluation. | Gap survives. |
| `recursive depth iteration state-space model SSM Mamba theorem proving` | No depth-recurrent SSM on theorem proving. | Gap survives. |
| `depth recurrent looped pretrained Mamba SSM retrofit` | **arXiv:2511.07384 ("Retrofitted Recurrence", McLeish et al, 19 upvotes, Nov 2025)** retrofits depth-recurrence onto pretrained Transformers (TinyLlama, OLMo, Llama-3.2). Does NOT do SSM. Does NOT freeze the backbone (it does continued pretraining). | **Adjacent prior art** — gap-finder did not surface this. The gap (TRM-on-Mamba-on-Lean) survives, but the architectural novelty is reduced: depth-recurrent retrofit on pretrained models is now an established research direction. |
| `looped transformer iteration constant depth TC0 latent reasoning` | Multiple looped-transformer papers exist (2502.17416 "Reasoning with Latent Thoughts", 2602.11451 "LoopFormer", 2511.08577 "Think-at-Hard"). | Looped-transformer recursion is a well-studied design pattern. TRM is one instantiation. The "recursion-around-frozen-substrate" angle is much closer to existing work than the hypothesis suggests. |

**Conclusion on gap:** The narrow gap (TRM-style recursion specifically wrapped on a Mamba backbone, evaluated on PutnamBench + miniF2F + CoqGym, compared against FPR-Mamba) does survive — no paper has done this exact comparison. But the gap is more modest than presented: depth-recurrent retrofitting (arXiv:2511.07384, Nov 2025) is concurrent published work the smith should have engaged with, and the architectural precedent of "looped layers around a pretrained block" is well-established for Transformers. **Gap claim survives in the narrow form but the framing oversells novelty.**

## 3. Citation spot-checks

### 3.1 Fixed-Point RNNs (arXiv:2503.10799) — citation MISREPRESENTED [CRITICAL]

**Hypothesis claims (§3, point 2):**
> "Fixed-Point RNNs *require* a contraction map to converge (arXiv:2503.10799 abstract / formal claim). TRM does not — the iterated map can implement non-contractive compositions, including the non-commutative permutation products that long tactic chains demand. Permutation composition is *not* a contraction map; it has eigenvalues on the unit circle. **A contraction-constrained iteration cannot represent permutation composition exactly**; an unconstrained iteration can."

**What the paper actually says (Section 3 + Theorem 3.1):**
- The contraction property is a CONSTRAINT ON THE DEPTH-ITERATION SOLVER (Lipschitz < 1 in **h**, Banach-Caccioppoli convergence guarantee), not on the function the RNN implements at the fixed point.
- The CONVERGED state at the fixed point parameterizes a **dense linear RNN** with non-diagonal transition Q⁻¹Λ, which DOES escape the diagonal-SSM TC⁰ bound. This is the entire point of the paper, stated in the abstract: "...investigate parameterizations of a large class of dense linear RNNs as fixed-points of parallelizable diagonal linear RNNs."
- **Figure 4 of the paper explicitly demonstrates FP-Mamba solving the A_5 and S_5 state-tracking tasks** — i.e., permutation composition. The hypothesis's claim that FPR "CANNOT represent permutation composition exactly" is directly falsified by the cited paper's own headline experiment.

**Severity: CRITICAL.** The structural argument that distinguishes TRM from FPR — the load-bearing claim of the hypothesis — is based on a misreading. The contraction is on the depth-solver dynamics, not on the function class. FPR is *designed* to express dense linear RNNs (which can represent permutation composition), and the paper shows it does so.

If FPR can solve S_5, the predicted gap "[Mamba+TRM(K=6) − Mamba+TRM(K=1)] − [FPR-Mamba(K=6) − FPR-Mamba(K=1)] ≥ +5" loses its mechanistic foundation.

### 3.2 TRM (arXiv:2510.04871) — citation PARTIALLY MISREPRESENTED [IMPORTANT]

**Hypothesis claims (§3):**
> "TRM (arXiv:2510.04871) iterates a learned block (attention + FFN + residual + a small latent answer state) K_arch times within the forward pass under deep supervision, with **no contraction-map constraint** — the iterated map is free to be expansive, periodic, or attractor-seeking."

**What the paper actually says (Section 4.1 "No fixed-point theorem required"):**
> "HRM assumes that the recursions converge to a fixed-point... To bypass this theoretical requirement, we define a full recursion process containing n evaluations of f_L and 1 evaluation of f_H. Then, we simply back-propagate through the full recursion process. **Through deep supervision, the model learns to take any (z_L, z_H) and improve it through a full recursion process, hopefully making z_H closer to the solution.** This means that by the design of the deep supervision goal, **running a few full recursion processes (even without gradients) is expected to bring us closer to the solution.**"

So TRM does *not* require the **theoretical** contraction guarantee (no Banach Theorem invocation, no IFT 1-step gradient approximation), but the deep-supervision objective **trains the model to approximate a contraction toward the answer**. The functional behavior at training equilibrium is "iterated map approaches solution" — which is empirically attractor-seeking. The hypothesis's framing — "iterated map is free to be expansive, periodic, or attractor-seeking" — is technically true at initialization but is misleading about the trained behavior. **TRM is not "unconstrained" in any sense that matters for the structural argument.**

**Severity: IMPORTANT.** The "TRM unconstrained vs FPR constrained" distinction the hypothesis pivots on is far weaker than presented. Both methods aim for attractor-seeking iteration on the trained model.

### 3.3 CoT-Solves-Serial (arXiv:2402.12875) — citation OVER-EXTRAPOLATED [CRITICAL]

**Hypothesis claims (§3):**
> "CoT Solves Inherently Serial Problems (arXiv:2402.12875) proves that constant-depth Transformers with O(T) iterations of *sequential generation* leave TC⁰ and reach P/poly. **The proof generalizes to any constant-depth substrate: K iterations of a learned block compose to depth K × d_block.**"

**What the paper actually says (Section 1):**
> "we prove that a constant-precision **transformer** with T intermediate steps and embedding dimension logarithmic in the sequence length can express any functions computable by a circuit of size T."

T is the count of *autoregressively emitted tokens fed back as inputs* — not depth iterations within a forward pass without token emission. The depth-buys-expressivity result depends on the model emitting symbols and reading them back, which expands the *prompt length* the next forward pass attends to. This is qualitatively different from depth-iterating a block over a fixed-size hidden state without re-reading the output.

The hypothesis's claim "K iterations of a learned block compose to depth K × d_block" is the **trivial circuit-composition observation** — but K passes of a TC⁰ circuit at constant K is still TC⁰. The TC⁰ bound is preserved under composition for **any constant K**. To escape TC⁰ via depth iteration, one of two things must happen: (a) K must scale with input length (which destroys the "constant depth" property — at which point the model is no longer in the TC⁰ regime in the first place), or (b) a non-TC⁰ primitive (e.g., output emission expanding effective input, or input-dependent transition matrices per Illusion-of-State §5.2) must be added. The hypothesis assumes (a) implicitly but states K ∈ {1, 2, 4, 6, 8} — a fixed small constant. **K=6 forward iterations of a TC⁰ Mamba block is still in TC⁰**, so the structural argument fails.

The Huginn paper (arXiv:2502.05171) — listed as cited in the hypothesis sources — was actually shown to have *limited* interpretable latent CoT despite up to 32 recurrence depth (arXiv:2507.02199 "Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer"); marginal gains from increased recurrence depth in Huginn. This is direct empirical pushback on the depth-iterates-buys-expressivity claim.

**Severity: CRITICAL.** The TC⁰-escape mechanism the hypothesis attributes to TRM-style recursion does not actually escape TC⁰ at constant K. The cited theory paper does not support what the hypothesis claims it supports.

### 3.4 Illusion of State (arXiv:2404.08819) — citation FAITHFUL on the bound, but the escape mechanisms it endorses (§5) are non-linearity and input-dependent transitions, not external recursion. The hypothesis's claim that external recursion is a TC⁰ escape is *not* endorsed by this paper.

### 3.5 RWKV-7 (arXiv:2503.14456) — citation FAITHFUL. RWKV-7 does claim TC⁰ escape via vector-valued in-context-learning rates and state tracking. F3 is a real risk that the hypothesis correctly identifies.

## 4. Mechanism critique (section by section)

### 4.1 "Why diagonal-state SSMs fail at long tactic chains" — partially supported
The TC⁰ argument for diagonal SSMs is correct (Illusion of State + Computational Limits + Mamba COPY bound). The leap from "diagonal SSM in TC⁰" to "long tactic chains require non-TC⁰ computation because they involve non-commutative substitution composition" is plausible but not rigorously justified. Lean tactic chains are not literally S_5 word problems — they include heavy retrieval, premise selection, and term unification. The state-tracking framing may not be the dominant binding constraint; many published Lean failures are retrieval/premise-selection failures, not state-composition failures.

### 4.2 "Why external TRM-style recursion plausibly escapes TC⁰" — load-bearing argument is broken
- K passes of a TC⁰ block at constant K is still TC⁰. (Composition closure of TC⁰.)
- CoT-Solves-Serial result requires symbol emission with input-length-scaling T, not constant-K depth iteration.
- The depth K × d_block argument is the trivial observation that depth composes, but it does not buy expressivity escape.
**Verdict: the structural argument that this mechanism escapes TC⁰ is not established.** The hypothesis is reduced to: "more compute helps long tactic chains," which is an empirical conjecture with no specific TC⁰-escape story.

### 4.3 "Why Fixed-Point-RNN is different and weaker on long chains" — directly false per cited paper
- (1) "Object refined" distinction is a real architectural difference, conceded.
- (2) The contraction-vs-no-contraction claim is **factually wrong**, see §3.1. FPR can express dense linear RNNs at fixed points and the cited paper shows FPR-Mamba solves A_5 and S_5.
- (3) "Iteration scope" claim is partially right — FPR is a re-parameterization of a single dense linear RNN, while TRM iterates more state — but this is a count-of-parameters argument, not an expressivity argument.

### 4.4 "Non-additive interaction" — restatement of the prediction without independent support
The non-additivity prediction is the prediction itself, not a reason. The argument reduces to: "TRM and FPR both help on long chains but TRM helps more (because of contraction)." Since the contraction argument is broken, the rest of the chain has no support.

### 4.5 "Distinction from CoT and from agent scaffolding" — a strength of the hypothesis
This is a genuine and useful disambiguation. The hypothesis correctly distinguishes architectural recursion from output-CoT and from MCTS scaffolding. **The methodology of holding test-time-sampling constant across arms is sound.** F2 (short-chain leakage) is a clean way to detect parameter-count confounds.

## 5. Falsifiability assessment

| Criterion | Operationalizable? | Concerns |
|---|---|---|
| F1 (equal-mechanism) | Yes — concrete +1 vs +5 absolute-point threshold with 95% CI | But: at predicted L_long baseline of 2%, a 5-point delta is enormous. With ~30 problems per bin and pass@1 rates ≤ 14%, the standard error on the difference of two differences is ~5–6 points. F1 is **probably underpowered** to discriminate +5 vs +1 reliably. The smith should add a power calculation. |
| F2 (short-chain leakage) | Yes — clean | OK |
| F3 (RWKV-7 dominance) | Yes — clean | This is a real possibility (R3 already concedes it). The hypothesis says F3 firing falsifies the **comparative** claim against internal fixes. But §8 R3 explicitly retreats to "head-to-head comparison is itself the contribution" — this is a **familiar dodge**: F3 should be a real falsification, not a fallback narrative. |
| F4 (coherence) | Yes — non-monotonicity threshold is concrete | OK |
| F5 (retrieval-dependence) | Yes — clean | But this is an attribution test, not a primary falsification. |

**The falsification suite is the strongest part of the hypothesis.** The criteria are concrete, quantitative, and directional. The main weakness is F1's statistical power at the predicted baseline rates.

## 6. Strongest counter-argument (steelman)

The strongest argument that the hypothesis is wrong:

**"H2 is a re-skinned compute-scaling claim with a confused mechanism story."**

Specifically: (1) any K-pass scheme is just adding compute at inference; (2) Retrofitted-Recurrence (arXiv:2511.07384) already showed that depth-recurrent retrofit improves math reasoning on Llama-3.2 — without any TC⁰ argument, just compute scaling; (3) FPR's iteration is similarly compute scaling; (4) the difference between TRM and FPR is parameterization, not expressivity class — both express dense linear RNN extensions of Mamba. Therefore the predicted gap on L_long is most plausibly driven by parameter count and optimization dynamics, NOT by a TC⁰-escape vs contraction-bounded distinction.

Under this steelman: the experiment is still useful (it *would* establish whether TRM-style or FPR-style retrofit better suits long-chain Lean), but the hypothesis is *over-mechanized*. The mechanism story should be stripped down: "TRM's parameterization of the depth iteration is more flexible than FPR's, and on long chains this matters." The contraction-vs-no-contraction structural argument should be dropped.

## 7. Severity-tagged objections

### Critical (must fix before approval)

**C1.** The contraction-vs-no-contraction structural argument is factually wrong against the cited paper. FPR is designed to parameterize *dense* linear RNNs at the fixed point, and the cited paper shows FPR-Mamba solving permutation composition (A_5, S_5). The hypothesis's load-bearing distinction does not survive citation check. (See §3.1.)

**C2.** The CoT-Solves-Serial result does not generalize to constant-K depth iteration without token emission. K passes of a TC⁰ Mamba block at fixed constant K remains in TC⁰. The "depth-buys-expressivity" mechanism does not apply to the regime the hypothesis is in (K ∈ {1,2,4,6,8}). (See §3.3.)

**C3.** "Wrap TRM around a frozen Mamba" is unprecedented. TRM (arXiv:2510.04871) trains the recursive block end-to-end. HRM (arXiv:2506.21734) trains the entire model end-to-end. Retrofitted Recurrence (arXiv:2511.07384) does continued pretraining, NOT freezing. The hypothesis pushes a training recipe onto the eval-designer without a precedent or a justification for why a frozen backbone wouldn't simply degrade the wrapper to noise. The closest precedent is *fine-tuning of inserted recurrence around a frozen backbone* (akin to LoRA semantics), but this changes the experiment substantially. The smith must specify which params get gradients, and produce evidence (or at minimum a plausibility argument) that the frozen-backbone variant is competently trained.

**C4.** The L_long baseline (Mamba(K=1) ≈ 2% on PutnamBench-hard tier + miniF2F-IMO) is anchored to *synthetic-task* papers (SD-SSM regular-language ablations, Negative Eigenvalues parity curves). Lean tactic prediction is not a regular language task, and these calibrations are extrapolations across multiple distribution shifts (synthetic→formal-math, regular-language→tactic-chain-completion). The 2% may be a wild overestimate (Mamba with no Lean pretraining could be near-zero on PutnamBench-hard at any K) or underestimate. The +5 absolute-point gap prediction depends on this anchor. **Without independent calibration on at least a small-scale Lean-Mamba pilot**, the magnitude predictions are unfounded.

### Important (should fix)

**I1.** The frozen-Mamba checkpoint for the proposed eval has not been pretrained on Lean/Mathlib. ReProver baselines establish ~25% on miniF2F low-tier with a *Lean-pretrained* T5 retrieval/generation pipeline. Off-the-shelf Mamba 1.4B/2.8B has zero formal-proof exposure — the entire stratification may collapse to noise across ALL arms, not just K=1. This needs a pilot.

**I2.** F1 is statistically underpowered at the predicted rates. With predicted Mamba+TRM(K=6) at 11% and FPR-Mamba(K=6) at 5% on ~30-problem L_long bins, the standard error on the difference-of-differences is ~5–6 points. The 95% CI for the difference will probably overlap both +1 (F1 fires) and +5 (hypothesis confirmed). The bin sizes need to be re-specified (probably need ≥ 100 L_long problems per arm) or the threshold relaxed.

**I3.** F3 (RWKV-7 dominance) is a real risk; §8 R3 retreats to "head-to-head comparison is itself the contribution" if F3 fires. This is exactly the unfalsifiable-fallback failure mode flagged by the spec. The smith should commit to: if F3 fires, the hypothesis's comparative claim is dead, and the contribution is reframed as a negative result on TRM-style recursion vs internal fixes. Restate F3 unambiguously.

**I4.** "TRM's deep supervision is also attractor-seeking" — the §3 distinction between TRM (unconstrained) and FPR (contraction-constrained) is not crisp once you read TRM §4.1 carefully. Both methods train the iteration to approach a target. The hypothesis should restate the distinction as *one of training procedure / parameterization*, not function-class expressivity.

**I5.** The "non-additive interaction" framing is operationally confounded with parameter count. R2 in §8 acknowledges this but only commits to "report the FLOP-matched, param-matched, K-matched triple grid as a deliverable." The smith should add a falsification criterion that *kills* the hypothesis if param count alone explains the gap (e.g., "if a parameter-matched **dense Transformer with TRM wrapper** shows the same K-gap on L_long, the SSM-specific mechanism story is falsified").

### Suggestion (nice to have)

**S1.** Add Retrofitted Recurrence (arXiv:2511.07384) and Looped Transformer / Think-at-Hard (arXiv:2511.08577) as comparison baselines or at least as discussion of architectural precedent.

**S2.** Cite the HRM mechanistic analysis (arXiv:2601.10679) which argues HRM's gains come from "guessing behavior" and fixed-point violations — directly relevant since TRM is an HRM variant.

**S3.** The cheaper falsification path (S5 permutation composition probe, §7) is **clean and well-thought-out**. Strongly recommend running this BEFORE any Lean evaluation. If TRM and FPR are indistinguishable on synthetic permutation composition, the hypothesis's structural argument is dead and Lean compute should not be spent.

**S4.** The §8 R1/R2/R3 risk discussion is unusually disciplined for a hypothesis at this stage. Keep this section.

## 8. Recommendation to hypothesis-smith

This hypothesis has a strong falsification skeleton wrapped around a broken mechanism story. The fix is non-cosmetic but tractable. Specifically:

1. **Fix C1** by reading FPR §3.2 + §4 + Figure 4 carefully. The contraction is on the depth-solver, not on the function class. FPR-Mamba *can* solve S_5 / permutation composition. Drop the "FPR cannot represent permutation composition" claim entirely. Replace with: "FPR and TRM differ in *how* they iterate — FPR iterates a state-transition matrix solver; TRM iterates a block-level latent. Whether this matters empirically on long Lean chains is an open question."
2. **Fix C2** by either (a) acknowledging that the TC⁰-escape argument requires K to scale with input length (and re-specifying the experiment with K_arch ∈ {O(1), O(log L), O(L)} bins), or (b) reframing the hypothesis as a compute-scaling claim without invoking TC⁰ escape. Option (b) is cleaner.
3. **Fix C3** by specifying the wrapper-training protocol: which params get gradients, what's the analogue of LoRA-around-Mamba, what's the cited precedent. Acknowledge that "frozen Mamba + trained TRM wrapper" is a novel training recipe with unknown properties; add a §4 ablation that confirms wrapper-training works at all on a small pilot before claiming the K-sweep is interpretable.
4. **Fix C4** by either running a small-scale Lean-Mamba pilot to calibrate the L_long baseline, or by switching the headline magnitude prediction to a *relative* claim (e.g., "the K-vs-1 gap on L_long is at least 2x the K-vs-1 gap on L_short for TRM") that is robust to the base rate.
5. **Fix I2** by recomputing required bin sizes from the predicted rates and running a power calculation. Move the L_long ≥ 50 stratum to ≥ 100 problems if available.
6. **Fix I3** by removing the §8 R3 fallback and restating F3 cleanly.
7. **Strengthen with S3** — the synthetic permutation-composition probe at variable L is the highest-leverage cheap falsification. It should be required *before* any Lean experiments are dispatched. If Smith adopts this gating, much of the C1/C2 mechanism critique is moot because the probe would empirically settle the structural distinction.

This is a worthwhile experiment that needs revision-1 surgery before approval.

---

REJECT (revision-1)
