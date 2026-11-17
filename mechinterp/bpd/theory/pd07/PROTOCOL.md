# BPD-PD-07 | Factor-read conditioned sparse gates on rotated 4D supports

**Preregistered on 17 November 2026, before any PD-07 model training.** [CONFIG.json](CONFIG.json) fixes the 24 previously unused seeds, comparison, gate architecture, objective, held-out families and acceptance thresholds. The PD-06 original [registered failure](../pd06/RESULTS_20261117.md) remains unchanged.

## Question

PD-06 recovered the planted rotated read axes in 32/32 count-supervised models, yet complete four-pass support recovery averaged 31.18% on withheld pairs and 56.20% on the withheld triple. The same source produced 100% unseen-support recovery when the full support was supplied as an oracle. PD-07 asks whether attaching each gate to **its own learned physical read coefficient** permits compositional transfer to unseen support combinations without access to source labels.

For one shared \`W=I_4\` physical linear layer and four shared rank-one components, with trainable read matrix V∈R4×4, define

\[
z_j(x)=\frac{x^\top V_j}{\max(\|V_j\|_2,10^{-6})},\quad
\tau_j = 0.25+0.5\sigma(\theta_j),\quad
p_j(x)=\sigma\left(10(|z_j(x)|-\tau_j)\right).
\]

Each component executes if \`p_j>=0.5\`; the actual forward mask is an exact binary 0 or 1, with a straight-through gradient through the sigmoid probabilities. Each \(\theta_j\) starts at zero, so the initial per-component threshold is 0.5. The column normalization prevents trivial gate changes from simply rescaling V. This architecture exposes **no source-family, latent-support, true Q or count labels** to its primary gate.

The \`read_coupled\` arm differentiates through z into V, while a \`read_detached\` ablation stops that *gate-feature* gradient. In both, the BPD physical component factor pair V,U receives training gradients from reconstruction and matrix faithfulness on every Adam update. This separates representation-gate coupling from the capacity to train a physical dictionary.

## Exact source match and combinatorial holdouts

Re-use the frozen PD-06 4D generator and QR sign rule *without changing its source law*. New seed–rotation pairs are:

\`947,953,967,971,977,983,991,997,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103\`.

Each seed fixes an independently sampled orthogonal Q, with \`x=aQ.T\` and latent a sparse in a fixed four-dimensional coordinate basis. The model sees the dense signed observed vector x. Active latent amplitudes are independently signed with magnitude uniform on [0.75,1.25]. Training samples four singleton supports; cycle pairs (0,1),(1,2),(2,3),(0,3); and triples (0,1,2),(0,1,3),(1,2,3), with k sampled uniformly in {1,2,3}. Evaluation has 192 boards each of six frozen strata: seen singletons, seen pairs, withheld diagonal pairs (0,2)/(1,3), seen triples, withheld triple (0,2,3), and a four-pass transition [singleton, withheld pair, withheld triple, withheld pair]. Each generated pass is independent, and all candidate components must use **one global alignment permutation** per fitted seed.

Training RNG seed offset is 100000+seed. Held-out evaluation uses 200000+seed plus 10000×family index, exactly as in PD-06. No held-out board or unseen support enters the train objective. The independent model seed plus rotation is the inferential unit.

## Six locked arms

| Arm | Gate input | Count/support labels? | V,U updated? |
|---|---|---|---|
| **read_coupled (primary)** | Componentwise normalized learned read coefficient x@V | None | Yes, including gate-feature path |
| **raw_independent (primary control)** | Raw signed x, shared MLP→4 independent logits | None | Yes |
| read_detached | Same x@V feature with stop-gradient through V on gate input | None | Yes, via physical write and faithfulness |
| raw_count_supervised | Raw x, original PD-06 count-supervised top-k | Count label *only during training* | Yes |
| read_planted_frozen | Componentwise read threshold with V=Q, U=Q.T | None | **No**, frozen positive control |
| oracle_support | Exact source support at train/test | Full support privileged | Yes |

Every trainable arm starts from exactly the same random initialized V, U and raw MLP parameters within a seed. The planted-frozen arm is explicitly different. Oracle support is privileged and must remain labelled in all reports. The physical shared model is \`BPDModel/TiedComponentLinear\` with zero ablation, no sphere rescaling, and residual \`delta_mask=0\`. Each of the four passes calls the same physical weight module.

Each arm receives **240 Adam updates**, batch 32, LR 0.025, weight decay zero, one CPU thread. Loss is mean squared output reconstruction + 0.30 mean squared physical weight residual + 0.05 mean actual binary gate fraction. Only \`raw_count_supervised\` receives a count-label cross-entropy term of weight 1.0. Gate probabilities, threshold parameters and the read dictionary are trained jointly in the primary arm. No post-training threshold selection or extra holdout calibration occurs.

## Scientific gate and evidence

A seed succeeds only if its global aligned read basis has mean cosine≥0.90, full held-out mean MSE≤0.03, **each seen-family** binary MSE≤0.04 with four-pass exact support≥0.80, and **each withheld-family** binary MSE≤0.06 with exact support≥0.70. Trainable arms must have nonzero first-step V,U gradients and final V,U update norms. Every executed mask must be exactly binary on every family.

The predeclared **primary** comparison is read_coupled versus raw_independent. Pass requires all three: at least 18/24 independent read_coupled seed recoveries; ≥6 net read_coupled-only versus raw-independent-only recoveries; exact two-sided paired McNemar p≤0.05. Preserve every paired seed result and a Wilson 95% interval for each arm, including scientific failures.

Report per-family false-offs and false-ons, exact four-pass component masks, learned read-axis alignment, independent seed-level statistics, and the unmodified full JSON first-run artifact. A partial reconstruction success is explicitly insufficient. The comparison is exploratory beyond this toy and does not establish recovery of real transformer mechanisms.

## What the controls can establish

When V is correctly aligned with Q, \(|x^\top V_j|\) is either zero or at least 0.75 for active components. A threshold initialized at 0.5 therefore separates active from inactive components with a margin for this planted source. That is a **strong representation-dependent inductive bias**, rather than a general latent discovery method. \`read_planted_frozen\` tests its capacity explicitly.

The raw MLP comparison tests whether this read-dependent bias improves *transfer of learned factors* across the forbidden combinations. \`read_detached\` tests whether gate-to-read gradients contribute. \`raw_count_supervised\` repeats the label-access distinction from PD-06; oracle support tests the achievable factorization capacity. Read threshold gradients, factor gradients, mask exactness, leak resistance and source split correctness must pass excluded-seed engineering tests before the science release.

A first full run is executed by a one-time source-locked workflow after all engineering tests pass. All scores and failures remain public without retrospective changes. Exact global expected-L0 uniqueness of the rotated coordinate reads, and finite Lean support-mass calculations, remain in PD-06; this optimizer experiment is scientifically separate.
