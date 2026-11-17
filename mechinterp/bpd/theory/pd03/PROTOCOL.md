# BPD-PD-03: top-1 routing versus calibrated binary gates

**Prospective registration, 17 November 2026.** This document and CONFIG.json were committed before any PD-03 model training or seed-level evaluation. PD-02 remains a permanently recorded failure (hard gates 6/24; matched soft gates 10/24, paired p=0.34375). New seeds, a new routing rule, and a new threshold-calibration rule are fixed here. All changes to this experiment require a dated pre-data amendment.

## Mechanistic question

In PD-02, hard-gated trainable dictionaries recovered the coordinate read axes in 24/24 seeds while 498/12,288 required gate invocations were false-offs. The resulting full-board routing failures motivate two prospective approaches.

**Top-1** enforces exactly one component per input position and pass. Let the shared gate logits be ell(x), probabilities p=sigmoid(ell). Its forward mask is the one-hot vector at argmax ell, with lowest component index breaking ties. Its backward derivative is taken through softmax(ell / 1). Use the actual BPD masked execution path. The average gate-count contribution is exactly 1/2.

**Margin-calibrated binary** keeps two independently executable components. During training, the exact forward mask is 1[p>=0.5] with straight-through sigmoid derivatives. Add a 0.02-weight penalty on mean relu(0.10-abs(p-0.5)) to discourage near-threshold logits, without phase labels. After training, choose one shared threshold from {0.30,0.35,0.40,0.45,0.50,0.55} using *only independent calibration examples*, minimizing binary reconstruction MSE + 0.05*binary gate fraction. Choose the largest threshold in a numerical tie. Freeze the threshold before held-out inference. This may trade false-offs against false-ons. Both types must be counted.

There is also an independent hard-STE baseline at threshold 0.5 with the same 200-update budget, which replicates PD-02's training rule on fresh seeds. Every trainable arm optimizes and updates **both read/write factors (V,U) throughout all 200 training steps**, alongside its shared gate network.

## Original source and matched controls

Run the existing BPDModel with one physical bias-free W=I2 shared across four passes and two rank-one factor components. Route through the real TiedComponentLinear wrapper with zero-ablation and sphere_rescale=false, and set residual delta_mask to zero. A source sample is a length-four trajectory of input vectors. For the phase-disjoint source, passes 0 and 1 use (signed amplitude,0), and passes 2 and 3 use (0,signed amplitude). The amplitudes are independent uniform samples in [0.75,1.25] with independent balanced signs. In the mixed-source negative control, both coordinates are always active.

The source and model generator are those of BPD-PD-01/02. The gate receives abs(x) via a shared affine map. No clock, phase ID, planted-support label, or direct gate target is supplied. The target output remains x for each pass.

Every seed has seven arms: top1_disjoint, margin_disjoint, hard_disjoint, top1_mixed, margin_mixed, top1_frozen, margin_frozen. All trainable arms use identical initial factors, gate parameters, and training inputs within each seed by resetting torch.manual_seed(seed). SVD-frozen controls train the gates for the identical optimization budget and expose residual gating errors with exact planted read axes. The mixed controls test whether sparsity constraints destroy a mechanism requiring both components.

## Locked independent seeds and compute

Fresh model seeds are 269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409. The excluded engineering seed is 1009. PD-01 and PD-02 model seeds must stay outside the registered cohort.

Use deterministic CPU with one Torch thread and one training process. Each arm gets 200 Adam updates, learning rate 0.04, weight decay 0, batch 16, and source training generator offset 100000+seed. Use independent calibration (128 inputs, seed offset 300000) and held-out (128, seed offset 200000) populations. Evaluate held-out only once after training and calibration selection. Training does not use the prior result's model weights, only the existing architecture. No 7M TRM checkpoint is opened.

The shared base loss is MSE(output,x) + 0.30*MSE(W-W_hat,0) + 0.05*mean component gate fraction. Top-1 uses the hard one-hot mask in this expression. Margin uses its actual hard binary mask and the additional margin penalty. The independent hard baseline uses the same base loss without margin. All gate derivatives use the declared straight-through estimator, while factor gradients pass through the real shared masked component operator.

## Frozen primary gate

At the **seed** level, call an arm successful iff its held-out exact-binary execution has MSE <= 0.020, mean axis read-cosine >= 0.90 after a global permutation/sign alignment, and at least 95% of held-out input trajectories satisfy four exact phase-appropriate masks under **one globally consistent component assignment**. Both read and write factors must have nonzero training updates and nonzero first-step gradients. The one-hot, factor, and mask checks are audited on excluded seed 1009 before cohort release.

The primary comparison is top1_disjoint versus margin_disjoint (with its calibration-chosen threshold). It passes only if top-1 succeeds in >=18/24 registered seeds, has at least six more paired seed successes than margin, and yields two-sided exact McNemar p <= 0.05. Report both arms' Wilson 95% intervals, per-seed MSE, axis match, board-level route fraction, gate density, false-off/false-on count, and held-out loss. The hard_disjoint arm is an additional historical-rule baseline on the same fresh seeds.

The calibration threshold's selection may fail to improve the margin arm. Report the entire calibration objective curve in that case. For every arm, report all 24 models including failures. Binary mask execution and full factor training must be verified; otherwise mark the run **engineering-invalid** rather than making a scientific inference.

## Scientific boundaries and execution lock

A deterministic one-hot gate may perfectly route a rank-two planted factorization while failing on mixed sources. Gate margin and threshold selection can change execution density while preserving near-zero reconstruction. The experiment tests factor recovery and execution reliability only in this fixed 2D toy.

Neither the idealized global expected-L0 theorem nor learned causal mechanisms in 7M TRM follow from a positive result. A negative result remains published without retraining or retroactive thresholds. Preserve PD-02's full original receipt, the older VPD parity, and B1–B4 gates.

The source, config, tests and registered workflow are committed before the single science-release lock. A dedicated CI workflow first verifies the source parent SHA, original hashes, all 24 seeds, and seven arms, then runs excluded-seed engineering checks. The complete 168-arm first-run receipt is uploaded as a GitHub Actions artifact; its scientific pass/fail is printed without changing CI's engineering verdict.
