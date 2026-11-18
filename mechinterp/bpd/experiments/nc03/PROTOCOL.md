# BPD-NC-03: gauge-covariant recurrent causal subspaces

**Prospective protocol.** Source-only registration on 2026-11-18; commit before implementing NC-03 or generating its training, discovery and evaluation data. This new experiment follows NC-02, whose native paths were exactly recovered in 8/8 engineering roots while orthogonally rotating the factor basis removed *all* qualifying coordinate communities. NC-02 raw results: https://github.com/MoralityLabAI/BPD/pull/67.

## Question and predicted equivariance

Does a rank-2 *causal-gradient subspace* recover the same physically active recurrent computation in different orthonormal coordinates even when individual neuron communities disappear? This is a **new subspace estimator**, not Goodfire's graph algorithm or a learned BPD decomposition.

At pass t, let z_t = a_t V be the pre-write factor read and g_t = ∂s/∂z_t for the label-aligned signed final logit s. Estimate C_t = E[g_tᵀg_t] on discovery examples and P_t = projector onto the leading TWO eigenvectors of C_t (fixed rank 2, no adaptive rank). Under factor rotation V′=VQ, U′=QᵀU, the predicted covariance transforms as C′_t=QᵀC_tQ, so P′_t=QᵀP_tQ when the two-dimensional eigenspace is identifiable. The physically acting map V′P′_tU′ should equal VP_tU. For native V=I, U=Wᵀ. For rotated V=Q, U=QᵀWᵀ. All operations are on one truly shared 20×20 physical weight across four nonlinear passes.

Use NC-02's *fixed* trained stage-shift model, context-label oracle and four-lane/five-stage architecture, but **fresh engineering model roots 300–307**, disjoint from NC-02. Each root trains on exactly 2048 examples for 128 updates, then discovers on 1024 examples and evaluates on a disjoint 1024 examples. Training, discovery and evaluation use NC-02 split indices 1/2/3 with the new root seeds. Every unit is one independently trained model root; data within the root are repeated measurements.

## Discovery, oracle scoring and intervention

1. Compute raw per-sample per-pass z-gradient of the label-aligned signed final logit with the native `BPDModel/TiedComponentLinear` wrapper, collecting the full 20-dimensional gradient vector. Fix the gradient covariance estimator to the uncentered second moment in float64. Reject eigenspace when λ₂ <=1e-12 or λ₂/λ₃ < 100 (taking λ₃=0 when numerically absent).
2. Select the top two eigenvectors per pass. Report projector idempotence and symmetry, ground-truth overlap `tr(P_true P_hat)/2` and largest principal angle. The true per-pass subspace in **physical pre-weight input coordinates** is span of stage-t lanes 0 and 1. Ground truth enters only this retrospective score.
3. Apply complete closed-loop pass-indexed projectors through `BPDModel.intervene("mix", all 20 components, mode="override_read", read_override=[V @ P_t], scale=0)`. This keeps the **complete** selected subspace without changing target W or introducing a residual channel. Necessity is the complementary projector `I-P_t`, using the same native counterfactual pathway.
4. Report every intermediate readout KL(clean||altered), final KL in nats, external-label NLL change, external-label accuracy, clean-context physical write RMS and realized closed-loop write delta. Full-ones exact projector parity must pass target fullmodel at each readout (max logit difference ≤5e-5, max float64 KL ≤1e-7, relative W residual ≤1e-4).
5. Native-versus-rotated primary invariance checks on exactly matched held-out examples: each root's maximal physical operator discrepancy `max_t ||VP_tU - V′P′_tU′||_F / ||W||_F` ≤1e-4, and max passwise difference in the retained model's logits ≤5e-5. Report regardless of mechanism score. Include direct numerical eigenspace covariance equivariance checks before causal interpretation.

## Model-blind shams and negative controls

For each pass, take the leading four eigenvectors of the **factor activation covariance**, computed from discovery examples. This identifies the active input subspace without reading the planted lane identities. Generate 2048 candidate random orthonormal rank-2 projectors *inside these four dimensions*, with a root- and candidate-index seeded PRNG. Candidate shams are **physical operators chosen in the native basis**, then transported to the rotated factor basis by the recorded Q, so both arms evaluate the *same physical null intervention*. Match each candidate to the discovered projector's **retained clean-context physical write RMS** within 15%; choose the eight closest eligible candidates, without evaluating any of them on held-out outcomes. Report proposals, unique candidate count, rejected-dose count and eligible count. A root with fewer than eight eligible shams is **blocked**, with complete diagnostic receipt. Shams are repeated comparisons within a model root; they never count toward independent sample size.

Known-site zero-readout control: replace both learned readout scales with zero for the discovery pass and check that the gradient covariance is exactly zero and the estimator returns **NO_IDENTIFIABLE_SUBSPACE**. The control must never use the label oracle to choose the projector or the shams.

The **primary causal contrast** on each independent root is the median of the eight sham keep-only final KL outcomes minus the learned projector keep-only final KL. Report necessity as a second outcome. The projected write RMS is a dose diagnostic, and the level of behavior preservation (absolute KL) is a separate endpoint. Predeclare minimum meaningful contrast ≥0.02 nats and an absolute kept-model final KL ≤0.05 nats for a positive engineering mechanism result.

## Scope, power and gates

* Engineering population: roots 300–307. A one-root CI smoke with discovery/evaluation sizes 256 each is allowed and must be labeled exploratory.
* Primary comparison family: one algorithm, two coordinate bases, shared eight model-blind physical shams; comparison between bases is paired within each root. Any subsequent sweep or basis variant is an additional comparison requiring amendment.
* Engineering checks: 8/8 task roots ≥95% accurate in each context, physical reconstruction closes, rank-2 eigengap passes, oracle projector overlap ≥0.99 in both bases, physical operator/held-out function equivariance as stated, zero-readout null, dose-qualified shams. The primary root-mean causal contrast must be reported with a *clustered* root bootstrap across eight roots. `ENGINEERING_POSITIVE_CONTROL` requires primary contrast CI lower bound >0.02 nats for native and rotated, and absolute kept-model KL ≤0.05. If criteria fail, retain `NOT_RESOLVED_OR_FAILED`; these are scientific outcomes, **not CI/test failures**.
* The narrow root family is a constrained positive control. Its bootstrap interval does not establish out-of-family generalization, learned feature discovery, or a theorem for arbitrary invertible gauge transformations. Orthogonal Q is the only registered symmetry. A repeated singular eigenvalue at rank 2 blocks the claimed projector uniqueness.
* Confirmatory cohort remains **scientifically sealed** and is not specified or sampled in NC-03 source. Any later confirmation requires a distinct 64+-root population, frozen comparison family and exact EvalMath `mechinterp-power` preflight. A nonparametric 95th-quantile/95%-confidence sham bound requires ≥59 independent sham roots, beyond these eight.
* Follow repository AGENTS.md and preserve ARG-03-N / AMM-03 seals. Keep raw case-level output aggregates, all eight sham scores and numerical parity/gradient diagnostics with source/protocol SHA-256.

## Evidence classes

This experiment can establish **synthetic instrument validity under orthogonal reparameterization** if the gates pass. It does not demonstrate semantic neuron communities in natural language models. The next experiment must test unconstrained learned recurrence and independent checkpoints to address that question.
