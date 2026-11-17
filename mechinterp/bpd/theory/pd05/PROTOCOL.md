# BPD-PD-05 | Count supervision and straight-through gradient diagnosis

**Preregistered on November 17, 2026, before PD-05 training.** All seeds, thresholds, estimators, the source and planned evidence are frozen in [CONFIG.json](CONFIG.json). This study extends [PD-04's registered failure](../pd04/RESULTS_20261117.md) (learned adaptive top-k 0/24, calibrated independent 6/24, count oracle 22/24). The 24 model seeds below have not been used by PD-01 through PD-04. Every new claim is limited to the two-dimensional shared-weight toy.

## Falsifiable question

PD-04's learned cardinality branch failed on mixed single/double support despite recovering its read axes in most seeds and a 22/24 count-oracle positive control. This experiment separates failure to infer cardinality from failure of the straight-through derivative through a binary count decision.

With a shared rank-two \`BPDModel/TiedComponentLinear\` matrix (\`W=I_2\`) invoked at four passes, the executed mask is

\[
h(x)=\operatorname{onehot}(\operatorname{argmax}\ell(x))+
\mathbf1[c(x)\ge0]\left(1-\operatorname{onehot}(\operatorname{argmax}\ell(x))\right).
\]

Here \(\ell(x)\in\mathbb R^2\) and \(c(x)\in\mathbb R\) are separate affine heads on \(|x|\), shared over passes. A one-hot mask executes one factor. A two-hot mask executes both. The forward masks are *exact binary values* using the actual BPD wrapper, with residual \(\Delta W\) masked off, zero-ablation semantics, and sphere rescaling disabled.

### Distinct gradient hypotheses

The original count STE uses \(p(1-p)\) multiplying the count-logit gradient, where \(p=\sigma(c)\). Two prospective unsupervised replacements preserve the identical hard execution:

- **Identity-logit STE:** multiply by 1. This retains count gradients even for saturated logits.
- **Temperature-2 STE:** multiply by \(\frac12\sigma(c/2)[1-\sigma(c/2)]\), a broader derivative window.
- **Original sigmoid-logit STE:** multiply by \(\sigma(c)[1-\sigma(c)]\), the PD-04 control.

All use the same softmax Jacobian for the assignment head on one-component passes. The count-head intervention changes only its backward estimator; the forward rule, sources, target, and sparsity weight remain fixed.

**Count-supervised training** supplies a separate training-only binary label \(y_k(x)=\mathbf1[\|x\|_0=2]\) to the count head through BCE-with-logits, weight 1.0. It gives the count label at every training pass, while leaving the assignment head and both read/write factors trainable. This is a supervised positive control. The zero coordinates in this toy make the count label linearly separable from \(|x|\), so success would demonstrate recoverable gating under direct supervision, rather than unsupervised discovery of a hidden mechanism.

## Eight prespecified arms

| Arm | Learned count | Count gradient | Training-only count label | Train V,U? |
|---|---|---|---|---|
| unsupervised_sigmoid | Yes | sigmoid-logit | No | Yes |
| unsupervised_identity | Yes | identity-logit | No | Yes |
| unsupervised_tempered2 | Yes | temperature-2 | No | Yes |
| supervised_sigmoid (primary) | Yes | sigmoid-logit | **Yes** | Yes |
| supervised_identity | Yes | identity-logit | Yes | Yes |
| supervised_frozen | Yes | sigmoid-logit | Yes | **No**, SVD planted |
| oracle_count | Source-provided | assignment-softmax | Exact count directly | Yes |
| calibrated_independent | Two binary gates | sigmoid-logit | No | Yes |

For all trainable arms, the shared factors \((V,U)\) receive gradients and Adam updates for **every one of 240 training steps**. Every arm within a seed starts with identical initialized parameters, except the declared SVD-frozen arm. The model observes only \(|x|\); no other source-family or phase label is passed. The count-supervised arms alone receive the cardinality label through a direct BCE term. The oracle arm receives true cardinality as an explicitly labeled upper bound.

## Source, seeds and budgets

The source is the **unchanged balanced PD-04 family generator**. Each training batch contains eight disjoint, eight overlap, and eight nested four-pass sequences. Disjoint supports are coord 0, coord 0, coord 1, coord 1; overlap requires both coords in all passes; nested requires coord 0 in the first two passes and both in the final two. Active amplitudes are independently uniform in [0.75,1.25] with independent balanced signs; inactive entries are exactly zero. Calibrate and evaluate separately on **128 fresh examples per family**. Split RNG offsets remain 100000+seed (train), 300000+seed (calibration), and 200000+seed (held-out).

Exactly 24 new model seeds: \`571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719\`. Engineering-only seed: 1009, excluded from science.

Optimization: 240 Adam steps, LR 0.04, batch 24, weight decay zero, CPU single thread. All arms minimize reconstruction MSE + 0.30 physical matrix reconstruction MSE + 0.05 actual binary gate density. The two supervised arms and supervised-frozen arm additionally add 1.0 times BCE-with-logits(count head, true count). The supervised-label term is included during training only. A held-out count label is used solely by offline scoring. No phase label is used as a training signal.

The independent-gate comparator trains at 0.5 and selects one global threshold from \(\{0.30,0.35,0.40,0.45,0.50,0.55,0.60\}\) using only the balanced calibration set, minimizing calibration binary MSE + 0.05 mean binary gate density; ties choose the highest threshold. Other arms use count threshold 0.5 with no calibration tuning.

## Frozen success criterion

Each seed succeeds for an arm iff all of:

- Read-axis cosine alignment \(\ge0.90\) under a **single** globally aligned component permutation/sign convention.
- Overall and each family-specific binary inference MSE \(\le0.02\).
- At least 90% of the 128 held-out four-pass trajectories per family execute **exactly** the planted component mask, under the same component mapping. Report disjoint, overlap, nested separately.
- The executed gates are exactly 0/1; both factor parameters receive nonzero first-step gradient and cumulative update for trainable arms.

Primary registered claim: **supervised_sigmoid beats unsupervised_sigmoid**. Pass only if supervised succeeds in at least **18/24** seeds, earns at least **six net seed wins**, and has exact two-sided McNemar \(p\le0.05\). The seed is the unit of statistical replication. The same success definition applies to secondary comparisons of unsupervised STE alternatives, supervised identity, and oracle. Report every candidate's 24-seed fraction and Wilson 95% confidence interval with exact counts. A failed primary gate stays FAILED regardless of secondary improvements or a very small MSE.

## Diagnostics fixed before unsealing

For every arm/seed/family, archive mean binary MSE, correct-four-pass support fraction, false-off/false-on gate count, average executed cardinality, read-axis cosine, count prediction accuracy, signed count-logit margin \((2y_k-1)c\), and fraction of \(|c|\ge4\) indicating saturation. Also log first-step count-gradient norm and factor-update norm. Missing gradients for an unused head in an oracle/control arm are expected; every supervised trainable arm must show count-head gradients. The training supervision label must never appear in an unsupervised arm's objective or its mask forward.

### Honest evidence boundary

The expected-\(L_0\) mathematical theorem predicts the planted coordinate basis as the strict optimum on this exact source law, with mean 1.5 active components/pass. STE-based training remains surrogate optimization, and direct count supervision exposes information that an unsupervised mechanistic interpreter would need to discover. PD-04 and all previous registered failures, including their original receipts, remain unchanged.

The source-only smoke must verify exact binary forward masks, count/assignment and U,V gradients in the applicable modes, deterministic source-family support, matching initialization, and calibration/heldout separation **before** committing the science execution lock. The first full run is executed once by a source-pinned GitHub Actions workflow, and its raw JSON and scientific PASS or FAIL will be published unchanged.
