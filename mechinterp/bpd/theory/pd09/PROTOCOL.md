# BPD-PD-09 | Unsupervised hyperplane-mass basis discovery after conditional PD-08 success

**Preregistered 17 November 2026 before any PD-09 model training.** [CONFIG.json](CONFIG.json) was committed first at c0b7e5a467374248d4f07172a572a6a7d91b5f76. Original PD-01 to PD-08 scientific results remain unchanged. In particular the **canonical merged PD-08** found 23/24 compositional successes for read-conditioned gates on *count-supervised* frozen factors, versus 0/24 for independent raw-input gates. Its 144-arm original run is [44114011511](https://github.com/MoralityLabAI/BPD/actions/runs/44114011511).

## Mechanistic question

Can a dictionary learn the rotated sparse source axes **using only observed dense inputs**, so the simple fixed read threshold that succeeded conditionally in PD-08 can generalize to support combinations absent during dictionary fitting?

The idealized expected-L0 objective minimizes

\[
J(V)=\mathbb{E}\,\|xV\|_0
=4-\sum_{j=1}^4\Pr[xV_j=0].
\]

For the frozen PD-06 support law, its unique global optimum is the latent source basis, up to nonzero scaling and permutation. The exact support masses in latent coordinates are 19/36,15/36,19/36,19/36. The expected active-component count is two per pass.

PD-09 optimizes a differentiable **unlabelled approximation** to this mass objective, avoiding count labels. For normalized read columns u_j and a smoothing scale sigma define

\[
J_{\sigma}(V)
=\frac14\sum_{j=1}^4\mathbb{E}_x
  \left[1-\exp\left(-\frac{\langle x,u_j\rangle^2}
                              {2\sigma^2}\right)\right].
\]

As sigma→0 and source outcomes remain bounded, this approaches the expected L0 fraction for a fixed basis. Its finite-sigma local optimizer dynamics are unknown. A continuous population optimum does not establish that Adam recovers it.

## Frozen source, rotations, and holdouts

Use the **unchanged PD-06 source generator**. Sparse latent row vectors a have 1,2,3 active coordinates drawn among all four singleton supports, four adjacent pairs and three training triples. Independent nonzero signed amplitudes have absolute magnitude uniformly distributed over [0.75,1.25]. Observed x=aQᵀ is dense under a deterministic random orthogonal Q sampled independently for each seed. Gate/dictionary training receives **x only**; source support masks and cardinality are returned by the generator solely for the named positive control and final scoring.

The withheld diagonal pairs (0,2) and (1,3), triple (0,2,3), and novel four-pass transition never appear in training. The six full evaluation families are the PD-06 singleton, seen pair, unseen pair, seen triple, unseen triple, and unseen-support transition. Each has 192 fresh four-pass trajectories. Per-family heldout RNG is 200000+seed+10000×family index. Training RNG is 100000+seed.

The 24 **fresh** rotation/model seed IDs are 1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471. These differ from all earlier PD registrations. Excluded source-only smoke seed is 2009.

## Five arms and fair training budgets

All arms are trained/evaluated on the actual same shared \`BPDModel/TiedComponentLinear\` physical W=I4 identity teacher with four rank-one components, zero ablation, no sphere rescaling and \`delta_mask=0\`.

1. **orthogonal_annealed (primary):** train a raw 4×4 matrix A with QR retraction to a differentiable orthogonal read basis V=QR(A), with U=Vᵀ. The physical factorization is therefore exactly W=I4 when all components execute. Optimize J_sigma using **only x** with sigma=0.6 for updates 0–119, 0.3 for 120–239, and 0.15 for 240–359. No labels, gating network, or hidden Q enter the training loss. Training variable A starts from the same randomly initialized PD-06 component V.
2. **orthogonal_fixed:** identical architecture, initialization and 360-update budget, with sigma=0.6 for every update. This isolates the prospective annealing schedule.
3. **untied_sparse:** update actual independent physical BPD V and U parameters for 360 steps using the same annealed smooth-L0 term on normalized V, plus matrix-faithfulness reconstruction (weight 0.30), normalized read-column Gram off-diagonal penalty (weight 0.50), and read-column norm-to-one penalty (weight 0.15). This tests whether a less constrained physical parameterization can achieve comparable basis discovery.
4. **random_orthogonal:** retain the QR-orthogonalized random initial dictionary without training, to measure architectural capacity under no learned basis. This is an explicitly frozen negative control.
5. **count_supervised_positive:** train independent BPD V,U and its PD-06 GateMLP for 360 updates with the original privileged true-count CE and physical reconstruction/sparsity objective. This is the upper comparator and is not counted as an unsupervised success.

All trainable arms use Adam, LR .025, weight decay zero, batch 32, one CPU thread, and the **same source batch sequence within a seed**. No gate is trained after the dictionary. At evaluation, every arm executes the same fixed normalized read threshold \(|xV_j|/\|V_j\|\ge0.5\), returning **exact 0/1 masks** through the genuine tied BPD component wrapper.

The orthogonal QR arms tie the decoder to the orthogonal read factor. This changes the model family compared with the untied original BPD factorization, and that restriction is an explicit inductive bias. The experiment must not call their training a generic unconstrained BPD gradient update. The orthogonal learned physical V,U tensors will be installed into the actual BPD component wrapper for every heldout intervention. All trainable parameter gradients and nonzero parameter updates must be audited.

## Registered success criteria and statistical test

A seed-level mechanism success requires all of:
- global read-basis permutation/sign alignment with planted Q, mean absolute cosine ≥0.90;
- full held-out MSE ≤0.03, per-family seen MSE ≤0.04 and per-family unseen MSE ≤0.06;
- exact complete four-pass component-support masks on ≥80% of heldout trajectories in **every seen family** and ≥70% in **every unseen family**;
- actual binary component writes with zero residual path; nonzero stage-one optimizer gradient and update for the three trainable **unsupervised** arms and the supervised positive.

The primary hypothesis **orthogonal_annealed beats orthogonal_fixed** is a PASS only if at least **18/24** annealed seed recoveries, at least **six net paired seed wins** over fixed-sigma, and exact two-sided McNemar p≤0.05. Preserve all 24 outcomes even if any threshold fails. The seed + its independent QR rotation is the statistical unit; all pass-level or support observations are nested diagnostics.

Report per-arm independent-seed success counts and Wilson 95% intervals, exact McNemar, all four matched read cosines per seed, false-off and false-on counts by held-out family, total sample complexity, and the learned physical factor reconstruction error. The positive count-supervised arm's performance must remain explicitly marked privileged.

## Engineering evidence and falsifier

Before any registered science weights are generated, run source-only tests on seed 2009 to verify: deterministic QR rotation, gradients through differentiable QR to A, orthogonality and exact physical rank-one sum, nonzero factor gradients for untied V/U, independence of **every unsupervised arm's objective** from support/count label tensors, no forbidden pair/triple in the source training distribution, exact 0/1 forward masks, source split independence and the same initial factor parameters across trainable arms.

The first data run uses a single frozen commit/SHA and a one-time GitHub execution lock. Archive the original full JSON, compute seed-level paired outcomes once, and preserve scientific FAIL as FAIL. No parameter, threshold, seed, source split, annealing schedule or success criterion can be changed after looking at the registered outcomes.

The exact idealized-L0 basis theorem is a separate analytic statement; this study tests the **finite-sample trainability of a particular smoothed surrogate**. Even a positive result stays inside the four-dimensional orthogonally mixed synthetic identity-teacher class and gives no automatic theorem for nonlinear recurrent 7M Sudoku transformer components.
