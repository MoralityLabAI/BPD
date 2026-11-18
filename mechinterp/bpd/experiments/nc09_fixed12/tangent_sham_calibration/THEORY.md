# FC02 math: source-gradient projection matching and the proper root-mean null

## 1. Geometry of physical retained writes

The physical 20D tied layer has a BPD full-rank factorization W=VU in a fixed orthogonal basis. For a clean pass-t BPD read z_t, retained rank-r projector P_t executes a physical write z_t P_t U. The source-only clean physical write RMS is

W(P)^2 = (1/80) sum_{t=0}^3 E_n ||a_{n,t} P_t U||_2^2.

This matches the RMS actually used to screen local physically matched shams in NC09 Fixed12/FC01. A sample from the same-tuple local Grassmann neighborhood of P_t is obtained by drawing Q_t orthogonal to the selected eigenspace basis E_t and setting B_t(theta)=cos(theta) E_t + sin(theta) Q_t, with H_t=B_t B_t^T. If E_t and Q_t have orthonormal columns and E_t^T Q_t=0, H_t is an exact rank-r orthogonal projector with principal angles theta to P_t. A nonzero minimum theta ensures a nonidentical intervention.

The NC09 Fixed12 source-DP P is selected from the model's own final-logit score gradients via a passwise sum of first-order omitted-score proxy losses under a 12-slot total rank constraint. It is frozen before the local sham family is constructed. Every proposed sham retains its *exact same passwise ranks* and the same total budget of 12. Orthogonal physical factor gauge is verified by transporting the selected physical intervention, rather than rotating the trained model weights.

## 2. Source-only first-order score effect

Let f(x) denote the model's own final logit; let g_{n,t}=∇_{z_{n,t}} f(x_n), with the model's detached output sign if used in the frozen parent derivative. Under the clean reference trajectory, a joint read truncation z_t → z_t P_t changes the final score to first order by

d_n(P)=sum_{t=0}^{3} g_{n,t}^T (I-P_t) z_{n,t},

up to sign convention. This is a local additive approximation; actual BPD interventions change intermediate hidden states and can produce nonlinear cross-pass Jacobian terms.

FC02 matches the **signed mean m(P)** and **RMS v(P)** of d_n(P) across all 1024 source-only discovery cases, in addition to W(P). Matching m preserves the average signed score direction; matching v limits a second-moment score dose. The absolute thresholds 0.005 logits and 25% relative tolerance permit finite-source numerical small-signal cases while remaining source-frozen. They do not ensure per-case derivative alignment, a universal task-label causal null, exact future-state matching or identical terminal true-label loss.

This conservative null is scientifically motivated by the SF03 signed-gradient-matched control, where a source-matched first-order predictor accounted for much observed task transfer. FC02 applies the idea to actual learned four-pass NC09 physical projector interventions.

## 3. An exact sham-only root pseudo-effect

Take one root's eight source-only selected matched sham terminal losses L_0,...,L_7, and let their mean be M=(1/8)ΣL_j. If sham J is selected uniformly as a pseudo-target independently of terminal labels, define its causal-looking benefit

Z_J=(1/7) Σ_{k≠J} L_k - L_J.

Then Z_j=(8/7)(M-L_j), giving

(1/8) Σ_{j=0}^7 Z_j=(8/7)(M - (1/8)ΣL_j)=0.

This conditional expectation is an **exact finite identity** for any eight real sham outcomes, requiring no normality or exchangeability of root training. A single deterministic uniform pseudo-target index J_i is selected independently per root and recorded. The eight possible alternative indices are correlated hypothetical controls, never eight independent null model roots.

The substantive DP12 effect uses the corresponding eight-sham mean M_i minus independently scored DP12 loss. This is an expected sham-versus-DP12 contrast at **fixed matched rank, source physical write dose and first-order model-score dose**, and is the relevant aggregate estimand for future heldout effects.

## 4. The model-root mean null, distinguished from FC01's maximum

For n independent trained model roots and one independent sham pseudo-selection per root, the mean pseudo-effect is

Zbar_n=(1/n) Σ_i Z_{i,J_i}.

Conditional on a fixed collection of eight-sham outcomes per root, E[Zbar_n]=0 exactly. Its conditional variance is

Var(Zbar_n | L)=(1/n²) Σ_i [(8/7)^2 (1/8) Σ_j (L_{i,j}-M_i)^2],

when J_i are independent uniform indices across roots. This variance decreases at order 1/n for stable root-level variance; it measures the **mean of independent pseudo-selected root contrasts**.

FC01 instead fixed T_i=max_j |L_ij - median L_i| for each root and took max_{59} T_i. Under iid model-root T_i, its sample maximum covers the 95th population percentile of **one root's worst orientation effect** with probability 1-.95^59≈95.15%. This is a distribution-free *rootwise* tolerance theorem. It is **not** a 95% tail threshold for a mean of 32 independent pseudo-target contrasts. Its numerical value 0.294179 NLL nats may be large even when the null mean has much smaller standard error. The original FC01 calibrated hold is preserved, as it should be.

## 5. Frozen nested-root planning bootstrap, with limits

FC02 observes 59 fresh rootwise eight-sham NLL vectors. It draws 256 outer root-cluster bootstrap samples of 59 complete root sham vectors, and for each draws 4096 hypothetical means of 32 replacement-sampled roots and independently pseudo-selected shams. For each outer sample it stores the empirical 95th percentile of these 32-root means, and its aggregate planning statistic is the empirical 95th percentile of the 256 outer tail thresholds.

This is a **nested model-root bootstrap estimate** of a positive future 32-root sham-only mean null threshold, including approximate variation in its 59-root empirical source distribution. There is no distribution-free 95/95 finite-sample guarantee for that nested quantile, and no exact scientific p-value follows from it. Conditional on model-root arrays, the pseudo-sham choice itself has a known finite distribution and exact zero expected value. Sampling uncertainty about future model-root geometry requires additional assumptions or a separate prospective calibration validation.

A future scientifically confirmatory design must pre-register a validated root-level randomization or predictive null procedure, address matched control exchangeability and familywise error, and only then release its independent heldout model roots. The present FC02 planning statistic alone cannot supersede EvalMath's original FC01 hold.

## 6. Practical causal interpretation

A positive FC02 matched-tangent development result would say that the source-only DP12 pass-specific causal projector preserves an incremental externally defined future-task advantage beyond local physical shams matched on three source moments. If it fails, the previous NC09 equal-rank developmental benefit remains as a separate outcome, while the stronger first-order sham may account for it.

All comparisons depend on the synthetic trained DenseLooper source and native four-pass BPDModel/TiedComponentLinear intervention. They do not prove individual neuron identity, globally minimal task quotient, semantic transfer across independently trained activation bases, or GL(20) gauge invariance.
