# BPD-PD-15: norm-weighted Gaussian sparsity for clustered 16D mechanisms

**Prospective registration: November 18, 2026.** [CONFIG.json](CONFIG.json) was first committed at \`348b3dd6441d0a3b9708708cac04dc4ca19db392\`, before any PD-15 scientific model training. This protocol fixes the original strong-margin clustered source, independent seeds, objective, arms, physical masks, exact candidate budgets, held-out thresholds and scientific decision. Preserve the first registered result whether it passes or fails.

## Registered question and motivation

[PD-14's frozen first cohort](../pd14/RESULTS_20261118.md) demonstrated a split: correcting the **specific Gaussian-mass objective inversion** in all 16 strong-margin independent pools left narrow_single at 3/16 complete mechanism recoveries. The wide σ=.15 search recovered 0/16 strong seeds, while at σ=.05 a nine-sweep search recovered 16/16 weak-margin seeds. Three independent three-sweep starts failed in every arm. The original PD-14 scientific gate was FAIL.

For the strongly correlated source, there is a second rotation-invariant means of emphasizing independently informative data. The squared input norm is invariant under the hidden orthogonal source rotation: if x=aQᵀ, then ||x||²=||a||². Weighting **observed dense** vectors by exp(-λ||x||²) decreases the influence of high-cardinality, coactive clustered examples without using a support label or Q.

The single predeclared primary test asks whether **λ=1, σ=.15** improves complete causal component recovery over **λ=0, σ=.15**, when each uses the same nine-sweep full-angle search on the same 2048 trajectories and same QR initial dictionary. Separately, the fixed-remix population/objective diagnostic tests whether weighting changes the ordering of Q and a specified 45° mixed basis on independently sampled sources.

## Frozen 32 strong-margin sources

Use the **exact PD-12** source generator and physical BPD evaluator, imported unchanged. Dimension is **16**; source supports have k=1,2,3,4 independently per pass with equal k probability. At k≥2, with probability .85 choose an allowed anchored contiguous k-window, otherwise sample uniformly from all allowed training supports. The original antipodal pairs, stride-two triples and stride-two quartets are withheld, never used for training. Nonzero magnitudes are uniform on [.75,1.25] with independent signs. The hidden orthogonal Q is drawn using the frozen PD-12 seeded QR.

CONFIG registers 32 **previously unused model/rotation seeds** in a single strong-margin cell. They are independent statistical units. Engineering seed 4001 is excluded from the registered set. Generate **one 2048-board four-pass training pool per seed** with RNG 100000+seed. Every learned arm sees exactly the same 8192 dense observed pass vectors. Initial read basis is the original PD-12 deterministic-sign random QR seeded 610000+seed. The trained BPD factor has V orthogonal and U=Vᵀ, with physical W=I16.

The eight original PD-12 evaluation strata (k1/k2/k3/k4 seen, k2/k3/k4 unseen and transition_unseen) each contain 96 four-pass trajectories from RNG 200000+seed+10000×family_index. Source Q and latent supports are available only to the generator, planted oracle and post-fit evaluator.

## Norm weighting and exact candidate accounting

For each observed **physical 16D vector x** from the training pool, compute

\[
w_\lambda(x)=\exp(-\lambda \sum_{j=1}^{16} x_j^2).
\]

For fixed λ and σ, optimize the **weighted mean read zero mass**:

\[
J_{\lambda,\sigma}(V)=\frac{\sum_{n=1}^{N} w_\lambda(x_n)
             \sum_{j=1}^{16} \exp[-(x_n^\top V_j)^2/(2\sigma^2)]}
            {16\sum_{n=1}^{N}w_\lambda(x_n)}.
\]

The source-observable weighting has no support-count estimator, no learned mask, no latent Q input, no clipping and no fitted parameter. The positive w values are independent of the candidate basis V, making each pairwise weighted Givens objective monotone when zero angle is a candidate. For λ=0, w=1 and the optimization is exactly PD-12/PD-14's original Gaussian mass rule.

The four learned arms form one paired 2×2 factorial with identical **nine full sweeps × 120 lexicographic pairs × 37 angles = 39,960 candidate-angle scores per seed**. Each score reads all 8192 training pass vectors for 327,352,320 candidate-by-vector projections per learned arm:

| Arm | σ | λ | Unlabelled optimizer | Scores |
|---|---:|---:|---|---:|
| **uniform_wide (primary control)** | .15 | 0 | Nine full-angle Givens sweeps | 39,960 |
| **weighted_wide (primary)** | .15 | **1.0** | Same sweeps, same initial QR, fixed norm weights | 39,960 |
| uniform_narrow | .05 | 0 | Same sweeps | 39,960 |
| weighted_narrow | .05 | 1.0 | Same sweeps | 39,960 |
| planted_Q_oracle | none | none | Privileged true Q; physical positive control | 0 |

Use the exact PD-12 global37 candidate angle grid in [−90°,90°] inclusive, same 1e-12 tie rule, same lexicographic order and no early stopping or annealing. Score λ=0 through the original source function for direct parity. Implement λ=1 as the weighted pair projection objective. Both update a float64 orthogonal V and its cached projections. Log the exact candidate count, weights' effective sample size \((\sum w)^2/\sum w^2\), train mass path, orthogonality, nonzero updates, and walltime. The extra weighted reductions are reported independently from candidate-angle count. No source Q, support masks or test outcomes influence the weights, optimization or stopping.

## Frozen physical mechanism endpoint

Use the original PD-12 [BPDModel/TiedComponentLinear](../../components.py) physical wrapper with four tied passes, 16 rank-one components, zero residual delta mask, zero ablation and no sphere rescaling. Threshold read activations using the original fixed **0.375** strong-margin threshold, with exact 0/1 masks. Match learned read directions to true Q using the same deterministic Hungarian assignment, shared across all eight family evaluations.

A seed counts as a **complete mechanism recovery** only if mean absolute matched read cosine≥.90, overall held-out MSE≤.035, every held-out family MSE≤.05, full four-pass exact component-support fraction≥.80 on **every** seen family and ≥.70 on **every** unseen family, all masks exactly binary, a nonzero optimizer update, and read-factor orthogonality error<1e-4. The oracle is exempt only from optimizer-update requirement.

## Single primary scientific claim, frozen success gates

The sole primary statistical unit is one of 32 distinct model/rotation seeds, comparing **weighted_wide** and **uniform_wide** on the same pool, same initial QR, equal nine-sweep budget. Scientific mechanism PASS requires all:

- **At least 18/32** weighted_wide complete mechanism recoveries.
- **At least 10 net weighted-only paired seed wins** over uniform_wide.
- Exact **two-sided McNemar p≤0.05** from 32 seed-level paired success booleans.

The λ/narrow cell comparisons and any cost or read-axis differences are secondary and descriptive, with no second retroactive significance claim.

## Independent source-objective diagnostic and sample-efficiency check

Generate a distinct **1024-board** diagnostic pool for each source using RNG 310000+seed and the *unchanged* strong-margin source law. This diagnostic is evaluated after training and excluded from the optimizer. In the known latent Q coordinate frame, compare Q with the fixed orthogonal four-pair 45° remix rotating pairs (0,1), (4,5), (8,9), (12,13) at σ=.15. Evaluate the same Gaussian zero mass under λ=0 and λ=1, with weights computed from observed physical x only.

The frozen objective-consistency gate requires **≥30/32** diagnostic pools with both expected signs: at λ=0 the remix scores *higher* than Q, and at λ=1 the planted Q scores *higher* than the remix. This is a separate descriptive consistency gate, not another newly significant mechanism claim. The study-level scientific PASS requires both this source-objective gate and the registered paired mechanism gate, with their outcomes reported separately.

A population analysis of the source law before choosing fresh PD-15 seeds gives fixed four-pair remix-minus-Q differences **+0.01194736 at λ=0** and **−0.00951834 at λ=1** (σ=.15). The predicted crossing is near λ=.53253. The weight law with λ=1 has population effective sample fraction **0.484** and raises the normalized k=1 mixture share from 25% to approximately **63.7%**. These are pre-data source-law quadrature predictions, **not** scored outcomes from the registered 32 seeds. The diagnostic must independently verify them.

## Engineering admission and provenance

On excluded seed 4001 only, verify exact PD-12 generator parity and nested training pool hashes, deterministic weight calculations from observed x, invariance of weight under any orthogonal initial read dictionary, effective sample size, weighted/unweighted angle-score equality, monotone sweep mass, exact nine-sweep score counts, orthogonality, full physical BPD binary masks and oracle success. Do not query or train any registered seed during preflight.

Commit this protocol before implementation. Once the excluded-seed CI and Lean candidate-budget proof pass, freeze one complete source commit in EXECUTE_LOCK.json. The registered GitHub Actions job must verify source lock ancestry and unmodified protected files, run all 32×5=160 original arms, independently audit every seed and stored diagnostic, and archive the full original JSON for 90 days regardless of scientific PASS/FAIL. A failed engineering check must remain an engineering failure; all PD-01–PD-14 source and scientific records remain unchanged.

Scope: this study addresses a fully specified orthogonally mixed sparse-linear identity teacher. Any transfer to learned compact TRM/recurrent-transformer components requires a separate intervention study.
