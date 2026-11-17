# BPD-PD-06: rotated four-dimensional compositional sparse support

**Prospective registration: 17 November 2026.** [CONFIG.json](CONFIG.json) was committed first, at 0c0b320d2ce6d6a0a44497166f8e60d9d998b3c2, before any PD-06 training, checkpoint generation, or outcome measurement. This document commits the scientific design before source implementation. Any substantive change after the first data requires a separate study ID and fresh seeds.

## Scientific question

PD-05 succeeded with direct count supervision (21/24 versus 0/24 unsupervised) in a two-dimensional source whose cardinality could be read from exact zeros in the observed input. PD-06 moves the zeros to a *latent* basis. It tests whether cardinality supervision still helps when every observed coordinate is almost surely nonzero, and whether the learned dictionary and gates compose on **previously unseen support combinations**.

An independent seeded orthogonal rotation Q acts on sparse latent row vectors a in R^4:

\[
x = a Q^\top,\qquad a = xQ,\qquad W_{\mathrm{teacher}}=I_4.
\]

Q comes from deterministic Gaussian QR using generator seed 900000 + model_seed and deterministic diagonal sign convention. Q is shared by all matched arms of that seed and is fixed across that seed's train/calibration/heldout splits. The gate sees the dense signed x through a shared nonlinear MLP, with neither Q nor latent coordinates passed explicitly to its input.

The physical target is one shared, bias-free nn.Linear(4,4), called at four passes. Every trained arm uses the repo's actual BPDModel/TiedComponentLinear wrapper with C=4 rank-one factors, zero ablation, sphere_rescale=false and a **zeroed residual delta mask**. Both V and U train on every optimization step for all non-frozen arms; the planted-frozen positive control explicitly initializes V=Q, U=Q.T and freezes these factors.

## Train support versus genuinely unseen combinations

Each active latent amplitude is a continuously sampled signed value with absolute magnitude uniform in [0.75,1.25]. Inactive latent coordinates are exactly zero. Observed x is dense after the orthogonal rotation. The generator samples a support independently for each of the four passes.

For training, support cardinality k is sampled with equal probability from 1, 2, and 3. Within k, supports are uniformly chosen from:

- k=1: {0}, {1}, {2}, {3}
- k=2, **seen pairs**: {0,1}, {1,2}, {2,3}, {0,3}
- k=3, **seen triples**: {0,1,2}, {0,1,3}, {1,2,3}

Hold out the two diagonal pairs {0,2} and {1,3}, and triple {0,2,3}. Those support sets never occur in training or calibration for parameter fitting. Calibration includes labels for selecting/reporting any permitted controls but is independent from heldout. The six frozen evaluation strata, 192 four-pass trajectories each, are:

1. k1_seen: four singleton supports;
2. k2_seen: four training pairs;
3. k2_unseen: two held-out diagonal pairs;
4. k3_seen: three training triples;
5. k3_unseen: the held-out triple {0,2,3};
6. transition_unseen: [singleton, unseen pair, unseen triple, unseen pair] across four passes.

All six strata are held out by random seed (offset 200000+seed). The unseen-support strata are also held out **combinatorially**. Training uses generator offset 100000+seed, calibration uses 300000+seed. Stratified evaluation respects the seed as the statistical unit rather than treating correlated passes or trajectories as independent replicates.

## Model and inference

A shared per-pass MLP takes the signed observed x, with Linear(4,32), GELU and separate assignment and count heads. The assignment head produces four logits s_i; the count head produces three logits for k∈{1,2,3}. A deterministic top-k selector executes exactly k physical component factors. The default hard decision is k = 1 + argmax(count_logits); component assignment is the k largest assignment logits. Deterministic descending-rank tie breaking uses the lower component index.

Training uses **exact binary forward masks**, with a continuous rank/count surrogate in the backward pass. The count logits receive gradients through their softmax mixture of smooth top-k approximations, and assignment logits receive gradients from the same surrogate. The gradient estimator and temperature 0.5 are part of the registered experiment. No inference soft mask is allowed in the scientific score.

The planted basis Q defines the true decoder/read directions. For every learned V, align normalized columns to Q's columns by maximum absolute cosine over all 4!=24 permutations; use one global permutation for every heldout stratum and pass. Report mean cosine and minimum matched cosine, full mask support accuracy, false-off and false-on counts, MSE, and gating cardinality. This avoids assuming the learned component numbering agrees with the source.

## Seven prespecified arms

- **unsupervised_topk:** reconstruction, matrix faithfulness and binary gate-usage penalties only; no count or support labels enter training or routing.
- **count_supervised:** add cross-entropy against true support size k on training examples, coefficient 1.0. This is the registered primary arm, with V,U fully trainable.
- **support_supervised:** add the same count loss and an extra BCE-with-logits against the true component-assignment mask in *the source's declared coordinate order*, weight 1.0. This supplies much more information than count and breaks latent permutation gauge; treat it as a deliberately privileged positive control.
- **oracle_count:** give true cardinality to the *routing algorithm* in training and evaluation, and learn assignment and V,U. It is a labelled upper bound, not learned count.
- **oracle_support:** give the true four-component latent support mask during train and evaluation. The read/write factors still train from random initialization. It measures whether correct gating allows the factorization to learn the rotated physical basis.
- **count_supervised_planted_frozen:** initialize and freeze V,U at Q and Q.T; train the same MLP gate with count cross-entropy. This isolates gate difficulty from factor learning.
- **dense_all:** always execute all four factors. Its task MSE may be excellent while exact support sparsity is wrong, directly auditing reconstruction-only claims.

Every non-frozen arm is initialized with exactly the same factor and gate parameters within a seed. The one exception is the explicitly planted-frozen positive control. Eight families or more cannot be added after the first run.

## Frozen compute and success gates

32 independent model seeds: 727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941. Engineering smoke uses excluded seed 1009 only.

One CPU thread and one process, 360 Adam updates per arm at LR .025, training batch 32. Loss: MSE(reconstructed x,x) + 0.30 MSE(W_hat,I4) + 0.05 binary gate-density. Labelled count/control terms are added only in the explicitly named arms. Every heldout stratum is evaluated after training completes and without any parameter or threshold updates. The source/factor/rotation initialization hash is recorded per arm and checked identical within matched trainable arms.

A trainable arm succeeds for a seed only if it meets *all* conditions:
- globally matched read-axis mean absolute cosine >=0.90;
- overall heldout reconstruction MSE <=0.03;
- **each seen** evaluation stratum MSE <=0.04 and exact four-pass planted support >=0.80;
- **each unseen** evaluation stratum MSE <=0.06 and exact four-pass planted support >=0.70;
- nonzero V,U first-step gradient and nonzero factor update, and exact binary forward masks.

The primary claim **count_supervised beats unsupervised_topk** requires >=24/32 successful seeds in count_supervised, net >=8 seed-level paired successes and exact two-sided McNemar p<=0.05. Report Wilson 95% intervals, exact pair discordance, and outcomes for all 32 seeds and seven arms, preserving all failures. The success metric is a strict component-identifiability and compositional transfer metric, not merely accurate identity reconstruction.

## Idealized mathematical benchmark and limitation

For the **training support law**, each k is equally likely and supports within k are uniform. In latent coordinates, coordinate-read zero masses are

\[
z(e_0)=z(e_2)=z(e_3)=19/36,\qquad z(e_1)=5/12.
\]

Their sum is 2, giving exact idealized J_min=4−2=2 active components/pass. The exact expected-L0 optimizer is the four rotated coordinate reads Qe_j up to permutation and scaling: independent continuously distributed nonzero amplitudes make arbitrary mixed-direction hyperplane cancellations probability zero. Every singleton support has positive probability, making each non-coordinate direction strictly worse than the appropriate coordinate basis exchange.

This paper-level argument describes the **global expected-L0 idealization**. The trained BPD minimizes a binary-forward straight-through surrogate with finite updates. Its optimizer may fail to reach the identifiable readout, and even a recovered readout can suffer incorrect state-dependent masks. The mathematical theorem is not claimed to be a Lean proof of rotated neural recovery.

A positive result remains a toy result: data lie on a union of known-dimensional rotated sparse subspaces and the target is the identity map. A negative result will be archived unchanged. Later work can address unknown nonlinear encoders, state-dependent recurrent transitions and actual 7M Sudoku checkpoints under a new prospective protocol.

## Execution integrity

Commit CONFIG and PROTOCOL before any PD-06 training. Implement and check source-only logic on excluded smoke seed 1009, including dense observation, strict holdout support disjointness, QR orthogonality, factor gradients, exact masks and true-label isolation. Run one locked 32×7 first scientific cohort only after tests pass. Archive unmodified full JSON receipt with the source hash. The original PD-01 through PD-05 results, including PD-05's count-supervision PASS, remain intact.
