# BPD-PD-04 | Adaptive component cardinality with mixed-support recurrent input

**Registered November 17, 2026, before generating any PD-04 weights or measurements.** The machine-readable [CONFIG](CONFIG.json) fixes every seed, distribution, optimizer, and success threshold. This study follows [BPD-PD-03](../pd03/RESULTS_20261117.md), which recovered disjoint single-component routing in 24/24 top-1 seeds while failing on inputs requiring two independent simultaneous factor writes. The previous prospective PD-03 paired primary gate also failed (five top-1-only wins; McNemar p=0.0625). All old results stay frozen.

## Scientific question and mechanism

Can a learned gate dynamically choose one or two active rank-one factors within a shared BPD physical weight, and does this recover the true activity structure more reliably than independent binary gates when both cardinalities are required in the **same training distribution**?

The teacher is one shared, bias-free 2×2 identity physical matrix called on four passes. The student uses the repository's genuine \`BPDModel\` and \`TiedComponentLinear\`, with two rank-one factors \`(V,U)\`, pass-indexed masks and \`delta_mask=0\` to prohibit residual-write shortcuts. V and U receive gradients throughout all 240 updates in each trainable arm, including primary adaptive and paired comparator. The gate observes \`|x|\` through a shared affine logits map and has no phase clock, target-support label, or training-time oracle input.

### Independent source families

Every training batch contains eight trajectories of each family and is shuffled. Every calibration and held-out set has 128 fresh examples of each family, created from distinct per-seed generators. At each pass, active coordinates have random independent signs and amplitudes drawn uniformly from [0.75,1.25]. The inactive coordinates are exactly zero.

| Source family | Pass 0 | Pass 1 | Pass 2 | Pass 3 | Mechanistic role |
|---|---|---|---|---|---|
| **Disjoint** | coord 0 | coord 0 | coord 1 | coord 1 | One factor needed at a time |
| **Overlap** | both | both | both | both | Two simultaneous factors needed |
| **Nested** | coord 0 | coord 0 | both | both | Cardinality transition within one trajectory |

The source family label is used to stratify *analysis* and balance generation. Only the input \`x\` is presented to the primary, comparator, and other non-oracle arms. The oracle-cardinality arm explicitly derives cardinality from \`x != 0\` and is a labeled upper-bound control.

### Gate parameterization and seven arms

The shared gate has a two-logit assignment head and a separate one-logit count head, each affine on \`|x|\`. All trainable arms start from the same initialized parameter tensors within each seed, even when a head is inactive.

**Adaptive top-k primary:** Select the highest assignment logit with a deterministic tie-break to component 0. Set \`k = 1 + 1[sigmoid(count_logit) >= 0.5]\`. Output a one-hot mask if k=1 or (1,1) if k=2. Backprop uses a softmax Jacobian for the top-1 selection at k=1 and a sigmoid straight-through gradient for the count decision. The forward mask is exactly 0 or 1.

**Calibrated independent comparator:** Threshold each sigmoid assignment logit independently. Train at threshold 0.5 with a coefficient 0.02 margin penalty \`relu(0.10 - |p-0.5|)\`; after stage-one training, choose **one global threshold** from \`{0.30,0.35,0.40,0.45,0.50,0.55,0.60}\` using only the balanced calibration population. Minimize calibration binary MSE + 0.05 times mean binary gate activity, breaking exact score ties in favor of the largest threshold. The chosen threshold remains fixed on all held-out families.

**Other arms:** independent hard at threshold 0.5; fixed top-1; fixed all-two (dense) routing; oracle input-support count with learned top-1 selection; and the adaptive architecture with SVD-frozen planted factors and a trained gate. The last is an engineering/optimization positive control.

The frozen all-two arm can approximate \`W=I\` while failing sparsity identification on disjoint passes. Fixed top-1 cannot execute both independent rank-one writes at the same pass, so overlap and nested families measure its capacity limitation. Oracle-cardinality tests whether learned component assignment works if the count is supplied perfectly, without serving as evidence of learned adaptive count.

### Budget and objective

24 new model seeds: \`419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569\`. These exclude all PD-01/02/03 registered seeds. Excluded engineering seed 1009.

Each arm receives 240 Adam updates at LR 0.04 and batch size 24, all on CPU with one thread. The shared loss is

\[
\operatorname{MSE}(\widehat W_g x,x)
+0.30\,\operatorname{MSE}(\widehat W-W,0)
+0.05\,\mathbb E[\text{active gates}/2].
\]

The calibrated independent arm adds \`0.02*mean(relu(0.10-|p-0.5|))\`. Binary masks, including their gate-count penalty, use straight-through derivatives. Both V and U are learned throughout stage one in every non-frozen arm. This is a differentiable surrogate whose binary forward executions approximate the idealized L0 objective; optimization is not asserted to reach its global optimum.

The training generator uses offset 100000+seed. Calibration and held-out generators use offsets 300000+seed and 200000+seed respectively. Each source family has 128 distinct population draws per evaluation set. The calibration stage never reads held-out inputs or outcomes, and evaluation does not update parameters.

### Predetermined scoring, tests, and failure gates

A seed is a **success** for an arm only when all these conditions hold on its frozen held-out set:

1. Assignment-invariant read-axis cosine >=0.90 using one global row-to-coordinate match per seed.
2. Overall exact-binary reconstruction MSE <=0.02 **and** MSE <=0.02 in each source family.
3. At least 90% of held-out *four-pass trajectories in each family* have all four exact masks equal to the planted support pattern under the same global component match.
4. For trainable arms, first-step V/U gradient norm and full-training V/U update norm are nonzero; every executed gate is exactly binary.

Primary hypothesis is **adaptive cardinality against calibrated independent binary gating**. Require all of: >=18/24 adaptive successes, >=6 net paired seed successes above the comparator, and exact two-sided McNemar p<=0.05 on discordant seeds. A separate exact Wilson 95% interval is reported per arm. Report all 24 seeds and seven arms, per-family gate false-offs/false-ons, reconstruction, support agreement, gate density, count logits, and factor alignment. A failed condition is a scientific FAIL even if an arm achieves high overall task MSE or the difference narrowly misses p<=0.05.

The scientific unit is the independently initialized model seed. Passes and trajectories are correlated observations nested within a seed and cannot inflate the statistical sample size.

### Compute and evidence safeguards

Run an excluded-seed 1009 source-only smoke before the first registered cohort. Test source support, the **exact** one-hot/dual-component forward masks, nonzero count-head and factor gradients, absence of residual bypass, matched initialization across arms, calibration isolation, and correct stratum-specific support masks. The first full science run is triggered by an immutable \`EXECUTE_LOCK.json\` with a fixed source commit SHA and the staged source integrity guard.

No trained or reference neural Sudoku TRM checkpoints are used. The original PD-01/02/03 negative registered results remain independent. A successful toy may motivate a larger sparse-support experiment only under fresh registration. The source and calibration settings cannot change in response to the first held-out results.
