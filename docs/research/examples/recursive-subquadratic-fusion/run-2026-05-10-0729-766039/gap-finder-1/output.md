# Gap-finder A — Architecture-side gaps in the (recursion × subquadratic-backbone) fusion thesis

## Slice scope

- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/scout-1/output.md` (architectural recursion / iterative-depth networks)
- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/scout-2/output.md` (subquadratic / sparse / formal-regime attention)
- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/scout-3/output.md` (state-space and linear-attention backbones)

This gap-finder restricts itself to the *architecture lineage* of the spec: combinations of the recursive operator with substrate primitives, and structural compatibility/incompatibility claims. Training-objective, data, and evaluation-side gaps are out-of-scope and assigned to gap-finder-2.

## Gaps

### Gap 1 — TRM-style depthwise recursion has never been instantiated on any *natively trainable sparse-attention* backbone (NSA/MoBA/DSA/HISA/HSA)

**Type:** Unexplored intersection.

**Statement.** Every published architectural-recursion paper (TRM 2510.04871, HRM 2506.21734, Universal Transformer 1807.03819, Huginn 2502.05171, Ouro 2510.25741, MoR 2507.10524, Relaxed Recursive Transformers 2410.20672, Mixture-of-LoRAs Recursive 2512.12880, LoopFormer 2602.11451, Thinking Deeper Not Longer 2603.21676) wraps weight-tied iteration around a *dense* softmax-attention block; every published natively trainable sparse-attention paper (NSA 2502.11089, MoBA 2502.13189, DSA 2512.02556, HISA 2603.28458, HSA 2511.23319, SeerAttention-R 2506.08889) deploys the sparse operator inside a *single-pass* decoder. PLT 2510.24824 is the lone exception — it couples a loop with Gated Sliding-Window Attention — but a sliding window is the simplest fixed-pattern sparsity and is strictly weaker than block-routed (NSA) or token-routed (DSA) selection, and PLT does not analyze whether tokens dropped in early loops survive into late loops. The spec's fusion target — a recursive operator stacked on a *learned* sparsity pattern that re-decides per loop which blocks/tokens matter — has therefore zero published instantiations.

**Evidence.** Recursion-side: 2510.04871, 2506.21734, 1807.03819, 2502.05171, 2510.25741, 2507.10524, 2410.20672, 2512.12880, 2602.11451, 2603.21676. Sparse-attention-side: 2502.11089, 2502.13189, 2512.02556, 2603.28458, 2511.23319, 2506.08889. Closest existing fusion: PLT 2510.24824 (sliding-window only, no learned routing).

**Verification query.** `hf_papers search query="looped transformer native sparse attention NSA recursion"` → 10 results; only 2510.24824 (PLT) actually fuses loops with subquadratic attention, and that is sliding-window not learned-sparse. No result combines a learned-sparse backbone (NSA, MoBA, DSA) with weight-tied iteration. Also `hf_papers search query="recursive transformer Mamba state space model architectural recursion"` → 10 results, none combine architectural-recursion-style depth with NSA/MoBA/DSA.

**Why it matters for the spec.** The spec's headline question is whether parameter-efficient depth-of-reasoning composes with sub-quadratic context scaling; the entire (recursion × native-sparse-attention) cell of the design matrix is empty, so any positive instantiation is on uncharted ground.

---

### Gap 2 — TRM-style depthwise recursion has never been instantiated on any SSM / linear-RNN backbone (Mamba, RWKV, RetNet, GLA, DeltaNet, Hyena), and the conceptual relationship to Fixed-Point RNNs (2503.10799) is unspecified

**Type:** Unexplored intersection plus untested conceptual claim.

**Statement.** No paper in the corpus stacks architectural recursion in the TRM (2510.04871) / HRM (2506.21734) / Huginn (2502.05171) sense around an SSM or linear-attention block (Mamba 2312.00752, Mamba-2 2405.21060, Mamba-3 2603.15569, RWKV 2305.13048 / Eagle-Finch 2404.05892 / RWKV-7 2503.14456, RetNet 2307.08621, GLA 2312.06635, DeltaNet 2406.06484, Longhorn 2407.14207, Hyena 2302.10866). Scout-3 explicitly confirms this absence ("Confirmed: no paper combines TRM-style architectural recursion with any SSM backbone"). The closest candidate is Fixed-Point RNNs 2503.10799, which parameterizes a dense linear RNN as the K-iteration fixed point of a diagonal linear RNN — but the iterations there refine the *state-transition operator* of a *single* linear-RNN layer toward an attractor, whereas TRM-style recursion re-runs an *entire* block (attention + FFN + residual) over inputs *and* state with deep supervision, and is not constrained to a contraction map. No prior art establishes whether one is a strict superset, an instance, or genuinely orthogonal.

**Evidence.** SSM/linear-attention backbones: 2312.00752, 2405.21060, 2603.15569, 2305.13048, 2404.05892, 2503.14456, 2307.08621, 2312.06635, 2406.06484, 2407.14207, 2302.10866. Recursion side: 2510.04871, 2506.21734, 2502.05171, 2510.25741, 2507.10524. Adjacent: 2503.10799 (Fixed-Point RNNs).

**Verification query.** `hf_papers search query="looped transformer Mamba state-space recursion depth"` → 8 results; closest is TransMamba 2503.24067 (parameter-shared Transformer↔Mamba switching, not iterated depth). Also `hf_papers search query="recursive transformer Mamba state space model architectural recursion"` → 10 results, none combine TRM-style depthwise recursion with an SSM or linear-attention backbone.

**Why it matters for the spec.** The spec specifically calls out "TRM-style recursion on Mamba" as a target combination, and asks whether Fixed-Point RNNs already subsumes the move; both questions are unaddressed in the existing literature.

---

### Gap 3 — DEQ-style implicit fixed-point depth has never been combined with any subquadratic operator inside f, even though the implicit-gradient trick is operator-agnostic

**Type:** Unexplored intersection.

**Statement.** DEQ 1909.01377 supplies a constant-memory training trick for "infinite-depth" models defined by `z = f(z, x)`, and the trick composes with any operator inside f, yet every DEQ instantiation in the scout corpus uses a dense-attention or convolutional f. HRM 2506.21734 inherits the implicit fixed-point flavor (1-step approximate gradient through the equilibrium) but again uses dense attention. None of the SSM/linear-attention backbones (2312.00752, 2405.21060, 2305.13048, 2503.14456, 2407.04620), and none of the trainable sparse-attention backbones (2502.11089, 2502.13189, 2512.02556), have been wrapped in a DEQ. Stability of the equilibrium under sparse f is also unknown — Reformer's LSH (2001.04451) and MoBA's (2502.13189) routing are *discrete* selections that may make the fixed-point map non-contractive in standard norms, which has not been characterized.

**Evidence.** DEQ: 1909.01377; implicit-fixed-point recursion in HRM: 2506.21734 (with mechanistic critique 2601.10679 finding fragile fixed points); subquadratic candidates not yet placed in DEQ form: 2312.00752, 2405.21060, 2305.13048, 2503.14456, 2502.11089, 2502.13189, 2512.02556, 2001.04451.

**Verification query.** `hf_papers search query="DEQ deep equilibrium fixed point Mamba SSM linear attention"` → 8 results; none place an SSM, linear-attention, or sparse-attention operator inside a DEQ-style fixed-point. Also `hf_papers search query="DEQ deep equilibrium subquadratic linear attention RWKV"` → 10 results, all are RWKV variants or surveys, none use the implicit-gradient construction.

**Why it matters for the spec.** Implicit-depth via fixed-point is one of the two architectural recursion lineages (the other being explicit weight-tied iteration), and its compatibility with the substrates the spec depends on is untested.

---

### Gap 4 — Per-token adaptive halting (ACT/PonderNet/UT/MoR/LoopFormer) has never been jointly trained with per-token sparse routing (NSA/MoBA/DSA), even though both produce per-token routing decisions

**Type:** Unexplored intersection.

**Statement.** ACT 1603.08983, PonderNet 2107.05407, Universal Transformer 1807.03819, MoR 2507.10524 (token-level adaptive recursion), and LoopFormer 2602.11451 produce per-token *depth* routing; NSA 2502.11089, MoBA 2502.13189, and DSA 2512.02556 produce per-token *attention* routing. Both classes of router operate on the same hidden state per layer, and the joint compute budget is the product of expected loop count and expected attended block count. No published architecture trains both simultaneously. Worse, when sparse attention drops a key block from the attended set, the per-token halting signal for *other* tokens that needed cross-attention to that block becomes ill-defined — the halting logit is computed on a representation that is missing the information needed to decide whether to halt. Sparse Universal Transformer 2310.07096 and MoEUT 2405.16039 use *sparse-over-experts* with halting, but expert sparsity is not token-position sparsity; they do not address the joint problem.

**Evidence.** Halting: 1603.08983, 2107.05407, 1807.03819, 2507.10524, 2602.11451, 2310.07096, 2405.16039. Sparse routing: 2502.11089, 2502.13189, 2512.02556.

**Verification query.** `hf_papers search query="adaptive halting sparse attention per-token computation"` → 10 results, all on token-level attention sparsification or KV pruning (Token Sparse Attention 2602.03216, SpAtten 2012.09852, FASA 2602.03152); none combine these with per-token *depth* halting. Also `hf_papers search query="PonderNet ACT halting linear attention RWKV Mamba subquadratic"` → 8 results, none combine ACT/PonderNet halting with subquadratic attention.

**Why it matters for the spec.** The spec explicitly asks whether per-token halting and per-token sparsification must be coupled; the joint regime is unmeasured, and the well-definedness of the halting signal under sparse routing is an unstated structural assumption.

---

### Gap 5 — Looped Transformers as Programmable Computers (2301.13196) is structurally incompatible with token-dropping sparse attention, but no paper has measured at what sparsity ratio the construction breaks

**Type:** Untested assumption (across the looped-transformer-as-computation literature).

**Statement.** The Looped Transformers as Programmable Computers construction (2301.13196) relies on full attention precisely retrieving program-counter cells; any token-dropping sparse-attention pattern that does not guarantee attendability of arbitrary positions would falsify the Turing-completeness argument. The follow-up Looped Transformers Better at Learning Algorithms 2311.12424 and Expressive Power of Looped Transformers 2410.01405 inherit the dense-attention assumption. BigBird 2007.14062 proves universal-approximation-plus-Turing-completeness for *random + window + global* sparse attention with a single forward pass, and Sparse Transformer 1904.10509 argues that "depth recovers what sparsity drops" via the strided pattern, but neither result re-derives the looped-program construction under sparsity, nor measures the empirical sparsity ratio at which the construction degrades. NSA 2502.11089, MoBA 2502.13189, and DSA 2512.02556 routing decisions are *content-dependent* — a token's eligibility to be attended depends on the current loop's representation, so the construction's program-counter cell may become inaccessible in some loops and re-accessible in others, an interaction unstudied anywhere.

**Evidence.** Theoretical recursion: 2301.13196, 2311.12424, 2410.01405. Theoretical sparse-attention expressivity: 2007.14062, 1904.10509. Empirical learned-sparse: 2502.11089, 2502.13189, 2512.02556.

**Verification query.** `hf_papers search query="looped transformer native sparse attention NSA recursion"` → 10 results; the closest, PLT 2510.24824, uses sliding-window not learned-sparse and does not test the program-counter retrieval property. No theoretical paper re-derives 2301.13196's construction under content-dependent sparsity.

**Why it matters for the spec.** If the spec aims to keep the "depth-as-computation" expressivity guarantee while moving to a sub-quadratic substrate, the regime where this guarantee survives sparsity is currently unknown.

---

### Gap 6 — TRM-style depthwise recursion vs TTT-layer per-token inner-loop recursion: no paper compares, composes, or distinguishes them

**Type:** Unexplored intersection plus contradiction-of-interpretation.

**Statement.** TTT layers 2407.04620 already perform an inner optimization step *per token along the sequence*, and follow-ups (Test-Time Training Done Right 2505.23884, TTT-as-linear-attention 2602.21204, One-Minute Video 2504.05298, End-to-End TTT for Long Context 2512.23675) extend this picture; TRM 2510.04871 / HRM 2506.21734 / Huginn 2502.05171 / Ouro 2510.25741 perform iteration *along depth, every token in parallel*. The two recursion axes operate orthogonally — sequence-time inner-loop vs depth-time outer-loop — but no published work instantiates both jointly, ablates one against the other, or characterizes whether stacking them is redundant, complementary, or actively destructive (e.g., the inner TTT update may converge faster than the outer recursion can propagate signal, leaving the outer loop iterating over a stale fast-weight state). Longhorn 2407.14207 (online-learning SSM) and TTT itself sit in a similar conceptual space without being compared to depthwise recursion.

**Evidence.** Sequence-time inner-loop: 2407.04620, 2505.23884, 2602.21204, 2504.05298, 2512.23675, 2407.14207. Depth-time outer-loop: 2510.04871, 2506.21734, 2502.05171, 2510.25741, 2507.10524, 2602.11451.

**Verification query.** `hf_papers search query="TTT test-time training depth recursion looped"` → 10 results; all explore sequence-time TTT or its extensions. None compose TTT-style per-token inner loops with depthwise weight-tied outer loops.

**Why it matters for the spec.** The spec explicitly asks whether TRM-style depthwise recursion stacked on TTT is redundant or complementary; the literature is silent.

---

### Gap 7 — Diagonal-state SSMs (Mamba, RWKV-pre-7, GLA, mLSTM) live in TC⁰ and cannot do parity / state-tracking in one forward pass; whether *external* depthwise recursion lifts a diagonal-state SSM out of TC⁰ has never been tested, and there is a contradiction-of-interpretation with two competing internal fixes (negative eigenvalues; non-diagonal selective transitions)

**Type:** Contradiction (between proposed-fixes) plus unexplored intersection (recursion-as-third-fix).

**Statement.** Illusion of State 2404.08819 and Computational Limits via Circuit Complexity 2412.06148 prove diagonal selective SSMs live in TC⁰ and cannot solve permutation composition, parity, or arithmetic-formula evaluation in a single pass. Two backbone-internal fixes have been published: Negative Eigenvalues 2411.12537 (allow negative-eigenvalue transitions) and Selective SSMs on Regular Languages 2412.19350 / SD-SSM (allow dense / non-diagonal selected transitions). RWKV-7 Goose 2503.14456 *claims* its in-context-learning-rate construction escapes the TC⁰ ceiling. CoT Solves Inherently Serial Problems 2402.12875 establishes that depth-of-iteration buys real expressivity. Yet no paper tests whether *external* TRM-style depthwise recursion applied K times around a diagonal-state SSM also lifts it out of TC⁰, nor whether it does so at a different K-cost than the internal fixes. The four candidate fixes (negative eigenvalues, dense transitions, RWKV-7's ICL-rate construction, external recursion) sit in unrelated papers with no head-to-head; if RWKV-7's claim holds, depthwise recursion may be redundant; if it doesn't, depthwise recursion may be the cheapest path.

**Evidence.** TC⁰ floor: 2404.08819, 2412.06148. Internal fixes: 2411.12537, 2412.19350, 2503.14456. Theoretical depth-buys-expressivity: 2402.12875. Recursion candidates: 2510.04871, 2502.05171, 2503.10799.

**Verification query.** `hf_papers search query="recursive depth iteration RWKV-7 state-tracking expressivity"` → 8 results; all extend RWKV-7 internally (Meta-State 2504.08247, State Tuning 2504.05097, WuNeng 2504.19191) or extend the internal-fix line (DeltaProduct 2502.10297). None test external depthwise recursion as a TC⁰-escape mechanism.

**Why it matters for the spec.** Whether the spec's recursion can substitute for, complement, or be made redundant by RWKV-7-style internal fixes is the central architecture-side decision and is currently undetermined.

---

### Gap 8 — Retrieval-head behavior under sparse attention is uncharacterized, and recursion may either rescue or destroy it; no paper measures retrieval-head survival across loops

**Type:** Untested assumption (load-bearing for the fusion thesis).

**Statement.** Retrieval Head 2404.15574 establishes that a small set of attention heads is mechanistically responsible for arbitrary-position fact retrieval in long-context dense attention; disabling them degrades reasoning and increases hallucination. NoLiMa 2502.05167 and Hyper-multi-step 2410.04422 extend the picture: long-context tasks decompose into multi-step retrieval. SCBench 2412.10319 finds sparse-attention KV compression more robust than Mamba/hybrid linear-RNNs on multi-turn, but does not isolate the retrieval-head subset. Sparse Frontier 2504.17768 shows Vertical-Slash patterns help retrieval and Block-Sparse helps reasoning, but does not measure whether the same heads remain "retrieval heads" once sparsity is imposed during training. SeerAttention-R 2506.08889 self-distills sparse gating for *reasoning* models but does not test whether retrieval-head behavior is preserved across loop iterations of a recursive operator. Whether depthwise recursion rebuilds retrieval-head functionality in *non-retrieval* heads at later loops, or amplifies the loss when the original retrieval heads are sparsified away, is unmeasured.

**Evidence.** Retrieval-head mechanism: 2404.15574, 2502.05167, 2410.04422. Sparse-attention empirics: 2412.10319, 2504.17768, 2506.08889. Recursion candidates that haven't probed retrieval heads: 2510.04871, 2506.21734, 2502.05171, 2510.25741, 2507.10524, 2510.24824.

**Verification query.** `hf_papers search query="retrieval head sparse attention long context reasoning"` → 10 results; closest are Adaptive Long-Context Head Identification 2502.09647 and Focus Directions 2503.23306, both of which characterize retrieval heads under dense attention; none track retrieval-head behavior across recursive loops or under natively trainable sparse attention.

**Why it matters for the spec.** Retrieval-head behavior is the most precise mechanistic statement of long-context fusion-thesis failure mode; recursion's effect on it (positive, negative, or null) directly determines whether the spec's combined system retains the retrieval competence each component has individually.

---

### Gap 9 — DSA's lightning indexer is itself O(L²); no published recursive-operator design considers per-loop amortization of the hidden quadratic step in production "sparse" attention

**Type:** Untested assumption / missing baseline.

**Statement.** DSA / DeepSeek-V3.2 2512.02556 routes attention via a "lightning indexer" that scores every token to pick top-k — but the indexer itself is O(L²). HISA 2603.28458 introduces a hierarchical block-then-token index to avoid this, at the cost of two layers of dropped tokens. The Gupta et al. 2505.14840 dense-subquadratic regime achieves Õ(n^(2-1/d)·polylog B) only for bounded constant d; SETH hardness pushes it back toward quadratic for d=poly(n). Alman-Song's bounded-entries threshold 2302.13214 (B = Θ(√log n)) shifts under softmax temperature, and recursive operators that sharpen distributions across loops may push the system across the hardness boundary at different loops. *No published recursive architecture amortizes the indexer cost across loops* (e.g., reusing the index from loop k at loop k+1 with a delta update, analogous to MoR's KV reuse 2507.10524 but for the routing computation). MoR explicitly notes the dense-attention quadratic cost dominates at long context but treats KV reuse, not indexer reuse, as the mitigation.

**Evidence.** Hidden-quadratic in sparse-attention pipelines: 2512.02556, 2603.28458, 2505.14840, 2302.13214, 2402.04497. KV-reuse-only recursion: 2507.10524. Recursion that does not address indexer cost: 2510.04871, 2506.21734, 2502.05171, 2510.25741, 2602.11451, 2510.24824, 2603.21676.

**Verification query.** `hf_papers search query="adaptive halting sparse attention per-token computation"` → 10 results, none discuss per-loop reuse or amortization of attention-routing computation in a recursive setting. `hf_papers search query="looped transformer native sparse attention NSA recursion"` similarly returns no paper amortizing indexer state across loop iterations.

**Why it matters for the spec.** A "sub-quadratic" claim that contains a hidden O(L²) routing step undoes the headline efficiency promise; whether recursion enables amortizing or worsens that hidden cost is a structural design question with no published answer.

---

### Gap 10 — No paper distinguishes "recursion refines an SSM hidden state" from "recursion refines latent answer tokens" from "recursion refines input embeddings" as competing fusion targets

**Type:** Untested conceptual distinction (architecture-side) creating an unexplored intersection grid.

**Statement.** TRM 2510.04871 refines a small latent answer state alongside inputs; HRM 2506.21734 has two coupled recurrent modules refining each other; Huginn 2502.05171 / Ouro 2510.25741 / Thinking Deeper Not Longer 2603.21676 iterate over a generic latent block; MoR 2507.10524 routes tokens through variable-depth recursion of the *same* block; Adaptive Loops and Memory 2603.08391 separates "more iterations" from "more storage" via a gated memory bank. When the substrate becomes a Mamba/RWKV/GLA backbone, the SSM hidden state is *itself* a learned compressed memory — so "what does recursion refine" multiplies into at least four candidates: (a) the SSM hidden state, (b) a separate latent answer object, (c) a memory-token bank (Hymba 2411.13676 introduces meta-tokens; Zamba 2405.16712 has a single shared attention module), (d) input embeddings. Hybrid Architectures Systematic Analysis 2510.04800 maps inter- vs intra-layer fusion of attention + SSM but does not consider where in the stack a recursive operator refines a state object. Scout-3 explicitly flags this as open; no published architecture has even posed the four-way design grid, let alone ablated it.

**Evidence.** Recursion targets vary across: 2510.04871, 2506.21734, 2502.05171, 2510.25741, 2603.21676, 2507.10524, 2603.08391. Memory-token / shared-module variants in subquadratic backbones: 2411.13676, 2405.16712. Hybrid-architecture map without recursion: 2510.04800.

**Verification query.** `hf_papers search query="recursive transformer hybrid SSM Jamba Mamba attention reasoning"` → 10 results; none disambiguate "what state object is refined by recursion" when the substrate is sub-quadratic. `hf_papers search query="looped transformer Mamba state-space recursion depth"` → 8 results, similarly silent on the four-way distinction.

**Why it matters for the spec.** Once the substrate is sub-quadratic, the choice of refinement target ceases to be a stylistic detail and becomes a load-bearing design decision (an SSM's compressed state may not retain enough information to *be* refined; a memory-token bank may be the only viable target); the spec must commit to one and the literature offers no comparative guidance.

---

## Discarded candidate gaps

These were considered but rejected because verification surfaced existing prior art.

### Discarded 1 — "Nobody has built a hybrid attention + sparse + linear model"

Rejected. RWKV-X 2504.21463 already combines RWKV with sparse attention for linear-complexity hybrid; ARWKV 2501.15570 distills Qwen 2.5 into RWKV-7 attention; BASED 2402.18668 combines linear + sliding-window attention; Apriel-H1 2511.02651 hybridizes attention + SSM for reasoning; Samba 2406.07522, Jamba 2403.19887, Hymba 2411.13676, Griffin/Hawk 2402.19427 all instantiate inter-layer hybrids. The hybrid-substrate cell is well-populated; only the (recursion × hybrid-substrate) cell is empty, which is captured in Gaps 1, 2, and 10.

### Discarded 2 — "Linear attention has not been combined with looped/iterative training"

Rejected. Test-Time Training 2407.04620 *is* an inner-loop iterative training within a linear-attention-style layer; Test-Time Training with KV Binding Is Secretly Linear Attention 2602.21204 makes the equivalence formal. Longhorn 2407.14207 is an SSM as amortized online learner. The "iterative training inside a sub-quadratic layer" cell is occupied by the TTT lineage; the genuine gap is the *depthwise outer-loop* lineage, which is captured in Gap 6 as the TRM-vs-TTT distinction.

### Discarded 3 — "Sparse attention has never been characterized for reasoning"

Rejected. SeerAttention-R 2506.08889 explicitly targets sparse-attention adaptation for long-chain reasoning models; Sparse Frontier 2504.17768 ablates sparse-attention patterns on retrieval vs reasoning workloads; SCBench 2412.10319 measures sparse-attention robustness on multi-turn; Hyper-multi-step 2410.04422 decomposes long-context difficulty into retrieval vs logic-based retrieval. The "sparse attention for reasoning" cell is studied; what remains uncharacterized is sparse attention *under recursion* (Gap 8) and recursive fusion of halting + sparsity (Gap 4).

### Discarded 4 — "Universal Transformer has never been combined with sparse attention"

Rejected (partially). Sparse Universal Transformer 2310.07096 combines UT with sparse-mixture-of-experts; MoEUT 2405.16039 closes the parameter-compute gap with MoE. These cover *expert-sparsity*, not *token-position sparsity*. The gap as stated is too broad — narrowed to "UT + per-token attention sparsity" the gap is genuine and is folded into Gap 4 (halting × sparse routing).

---

## Summary table

| # | Gap (one line) | Load-bearing prior art (arxiv IDs) |
|---|---|---|
| 1 | TRM-style recursion never instantiated on natively-trainable sparse-attention (NSA/MoBA/DSA/HISA/HSA) | 2510.04871 / 2502.11089 / 2502.13189 / 2512.02556 / 2603.28458 / 2511.23319 / 2510.24824 |
| 2 | TRM-style recursion never instantiated on any SSM/linear-RNN backbone; relation to Fixed-Point RNNs unspecified | 2510.04871 / 2312.00752 / 2503.14456 / 2503.10799 |
| 3 | DEQ-style implicit-fixed-point depth never combined with subquadratic operator inside f | 1909.01377 / 2506.21734 / 2601.10679 / 2312.00752 / 2502.11089 |
| 4 | Per-token halting jointly with per-token sparse routing has zero published instantiations; halting signal ill-defined under sparsity | 1603.08983 / 2107.05407 / 2507.10524 / 2502.11089 / 2502.13189 / 2512.02556 |
| 5 | Looped-transformer-as-computer construction relies on dense attention; sparsity-ratio break point unmeasured | 2301.13196 / 2311.12424 / 2410.01405 / 2007.14062 / 2502.11089 |
| 6 | TRM-style depthwise recursion vs TTT per-token inner-loop never compared, composed, or ablated | 2510.04871 / 2407.04620 / 2602.21204 / 2407.14207 |
| 7 | TC⁰ ceiling for diagonal-state SSMs has three internal fixes; external recursion as fourth fix never tested; possibly contradictory with RWKV-7 expressivity claim | 2404.08819 / 2412.06148 / 2411.12537 / 2412.19350 / 2503.14456 / 2402.12875 |
| 8 | Retrieval-head behavior under (sparse attention × recursion) joint regime never measured | 2404.15574 / 2502.05167 / 2412.10319 / 2504.17768 / 2506.08889 |
| 9 | DSA-style indexer is O(L²); no recursive design amortizes routing across loops (vs MoR's KV-only reuse) | 2512.02556 / 2603.28458 / 2505.14840 / 2302.13214 / 2507.10524 |
| 10 | Four-way distinction (recursion refines: SSM state / latent answer / memory-token bank / input embeddings) never posed for sub-quadratic substrates | 2510.04871 / 2506.21734 / 2510.25741 / 2411.13676 / 2405.16712 / 2510.04800 |
