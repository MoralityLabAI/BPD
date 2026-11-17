# BPD-PD-02: hard-gated full-dictionary phase routing

**Prospective registration:** November 17, 2026. This registration and machine-readable [CONFIG.json](CONFIG.json) were committed **before any BPD-PD-02 training**. Source parent is BPD PR #2 merged into main at \`ca46f5827947770dbb6218a0b58f85826c27bc68\`. Preserve the BPD-PD-01 failure and frozen historical receipts. Any future scientific changes need an additional dated amendment before their associated data.

## Question and falsifier

Will exact binary execution gates make a fully trainable, shared rank-two BPD dictionary recover the planted phase-disjoint mechanism more reliably than an otherwise matched differentiable sigmoid-gate training run?

BPD-PD-01 achieved near-zero reconstruction loss with dense gates and mixed read directions at seed 71, while the trained seed 89 recovered the axes. Its registered trained-dictionary positive gate failed (1/2). A low reconstruction loss will remain insufficient to count as recovery in BPD-PD-02.

## Matched comparison

All arms use the **actual** \`BPDModel\` and \`TiedComponentLinear\`, one physical weight matrix \`W=I_2\` called at four passes, two shared rank-one factors \`(V,U)\`, a shared affine gate network fed \`|x|\`, and residual \`delta_mask=0\`. Mean-write replacement and sphere rescaling are disabled. The learned read/write factors both receive gradients and optimizer steps throughout each trainable stage-one run.

Disjoint source: passes 0 and 1 receive \`(±a,0)\`; passes 2 and 3 receive \`(0,±a)\`. Independent amplitudes are uniform on [0.75,1.25], signs equiprobable. Mixed negative source: independent nonzero signed amplitudes in both coordinates at every pass, with identical marginal amplitude law.

The four arms for every model seed are:
1. **hard_disjoint**, the primary arm: binarize gate probabilities \`p=sigmoid(logit)\` with \`h=1[p>=0.5]\`, use the straight-through value \`h+p-stop_gradient(p)\` in both the actual forward and the gate-count objective. The forward is exactly 0/1 at every component invocation. Gradients use the sigmoid surrogate.
2. **soft_disjoint**, matched training baseline: use \`p\` directly with the same weights, optimizer budget, and training input stream. For fairness, score its **binary-thresholded** inference behavior with \`h\` as well as its native soft-output MSE.
3. **hard_mixed**, negative control: same hard training on a source with both coordinates populated in each phase.
4. **hard_frozen**, positive control: SVD-initialize and freeze \`(V,U)\` at the planted identity factorization, train gates for the same budget.

The three trainable arms use the identical random factor and gate initialization within a seed, produced by resetting \`torch.manual_seed(seed)\` before constructing each arm. The soft and hard disjoint arms draw identical stage-one batches, and their direct comparison conditions on a common initialization and input stream.

## Seed cohort, budgets, and selection

The **24 fresh model seeds** are exactly:

\`137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263\`.

The original 71 and 89 seeds remain historical only. Use 200 stage-one Adam steps, LR 0.04, batch 16, one process and one CPU thread. Train on fresh batches from generator \`100000 + seed\`. Calibration uses 128 examples from \`300000+seed\`; held-out uses another 128 from \`200000+seed\`. The \`torch\` version and source commit must be written into receipts.

Stage-one loss in all arms is

\[
L_\mathrm{recon} + 0.30 L_\mathrm{faith} + 0.05 L_\mathrm{gate},
\]

where \`L_recon\` is mean squared output error across all token, pass and output coordinates, \`L_faith\` is mean squared matrix reconstruction error, and \`L_gate\` is **the actual mean number of active binary gates divided by two** for hard arms. The gradient through binary gate count is straight-through. For the soft arm, \`L_gate\` is the mean probability. Use \`delta_mask=0\` so physical residuals cannot fill missed writes.

Report *pre-switch, stage-one* results as the primary scientific comparison. After this frozen checkpoint, the hard-disjoint arm alone also receives a separate gate-only switch-penalty comparison: freeze V and U bitwise; independently start from the stage-one gate for each candidate coefficient in \`{0.01,0.05,0.20}\`, train 25 Adam steps at LR .01, and compare calibration against coefficient 0. Each candidate must retain reconstruction and actual binary gate fraction within +0.005 absolute of the stage-one calibration reference. Choose minimum calibration switch score among feasible candidates, allowing coefficient 0. Publish calibration and eventual held-out secondary values; the stage-two selected result never replaces stage-one primary success.

## Predeclared success rules

For a **single seed's hard-disjoint stage-one** success require all of:
- held-out exact **binary** forward MSE <= 0.020;
- normalized read-axis matching cosine >= 0.90, allowing read-component permutations/signs;
- at least 95% of held-out boards exhibit both phase-0 masks equal and one-hot, both phase-1 masks equal and one-hot, and opposite winning components between phases, after choosing a *single* global component assignment;
- nonzero trainable V,U parameter update and nonzero first-step factor gradient; hard masks must be exactly binary during forward.

Compute the soft-disjoint comparator's success with the same held-out **binary-thresholded** inference gate criterion, after training using soft probabilities. Also report its continuous-mask native MSE, so the cost of binarizing a learned soft dictionary is visible.

The **cohort primary gate** requires all of:
- >=18/24 independent hard-disjoint successes,
- hard-disjoint net success wins minus soft-disjoint wins >=6,
- exact two-sided paired McNemar sign-test p <= 0.05 over discordant seed outcomes.

These outcomes may fail. Report all 24 per-seed paired rows and exact discordant counts, Wilson 95% intervals for single-arm success proportions, and per-seed MSE/axis/sparsity measures. A cohort of 24 measures sensitivity to initialization on this toy, rather than general neural mechanism identification.

## Audits and computational scope

Separate source-only unit tests from registered results. Unit tests may exercise small 4-step engineering smokes using seed 1009, which is excluded from the registered cohort; they cannot select objective coefficients, cohort thresholds, seeds or schedules. They must establish: disjoint source support, same hard/soft initialization and training batches, exactly binary forward masks, gradients reaching V,U and gate parameters, no residual shortcut, no train/held-out sample contamination, correctly frozen V,U during stage two and valid exact paired statistics.

Only when those tests pass may the 96 full-budget arms execute. Preserve the first run even if it fails. Build one deterministic JSON receipt and SHA256 source manifest for the exact code that produced it, and upload it as CI artifact. Do not access, merge into, or reparameterize the original BPD-PD-01 receipt during this run.

The experiment is two-dimensional and uses a continuous straight-through surrogate for optimizing a discrete-forward L0 count. It does **not** prove convergence to an exact expected-L0 global minimizer. The pinned VPD parity test, phase-disjoint original failure, B1–B4 protocol, and the trained neural BPD identifiability hypothesis remain separate evidence.
