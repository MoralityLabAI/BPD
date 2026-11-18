# BPD-EFD-02: future-dark dimension versus curvature-level-set feasibility

**18 November 2026. Source-only geometric pilot, observed FAIL (28/64 admitted)**, downstream of independently audited LGA-09's 65-dimensional first-order future-dark physical pair subspace. This note reports the separate preregistered [LOOPSCOPE-13 / LGA-10 feasibility run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44319953707), archived in [PR #202](https://github.com/MoralityLabAI/LoopedTransformers/pull/202). The experiment did **not** open reserved source 31064, original trained TRM checkpoints or protected confirmation cohorts.

## Geometric issue

LGA-09 established that on the 96D predecessor/goal hidden-state pair, the current-head/energy tangent subspace K has 75D, with ten independent future-goal first derivatives. Its first-order future-silent subspace N=K∩ker(J_future) consequently has 65D at all 1,536 original held-out model-reached states.

A natural source-donor candidate z can be decomposed as `z=z0+q0`, where z0 belongs to row(J_future|K), q0∈N, and `r=||q0||`. Every full-J-matched sham with the same physical norm has the form `z0+r q` for a unit q∈N. To control the next Taylor term, freeze the signed donor-minus-original future margin Hessian H_m. Equal second-order response means

`(z0+r q)ᵀH_m(z0+r q) = zᵀH_m z`.

The null dimension 65 guarantees a large sphere of **first-order** equivalent directions. It does not guarantee that the *intersection of that sphere with the donor's scalar quadratic curvature level set* contains physically distant directions. A fixed source donor can be close to an extremal point of this restricted quadratic form.

LGA-10 preregistered a deterministic search along 2D planes containing the original q0. It required eight independent controls, each with `|cos θ|≤0.85` relative to q0, and at most 32 random planes with a 64-angle bracketing grid per control. The tests verified exact curvature calculations and native HVP parity. On the **excluded untrained source** seed 61064 with eight IID bases × eight legal donors, only **28/64 (43.75%)** of the source-goal cases admitted all eight controls; **36/64** failed `INELIGIBLE_HESSIAN_BRACKET`. The preregistered native admission threshold was 95%. Successful controls matched curvature within **1.39×10⁻¹⁶** analytic absolute error. [Original `feasibility.json` SHA](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44319953707/artifacts/14434876017): `ee6dfc30920b429f8edd8ff0b42d773763dc186b19abf460ab93d010bb3563f5`.

This result is **source-only engineering failure**. Random-plane failure need not imply mathematical nonexistence of distant same-curvature shams. The untrained model's quadratic observer and future-margin Hessian are also insufficient to infer curvature selection in any trained recurrent model.

## Useful mathematical diagnostic

At fixed cosine `c=q·q0/r` (with `|c|≤0.85`), write

`q = c u + sqrt(1−c²) v`, where `u=q0/r`, `v∈N∩u⊥`, `||v||=1`.

Then, with `a_c=z0+r c u`, the curvature becomes

`Q_c(v)=a_cᵀH_m a_c + 2r sqrt(1−c²) (H_m a_c)·v + r²(1−c²)vᵀH_m v`.

This is a **quadratic optimization on a 63-sphere in the 64D N∩u⊥ space**. Compute its attainable minimum and maximum numerically for a source-only fixed grid of c values using the restricted Hessian spectrum and deterministic trust-region solvers. If the donor target curvature lies in that attained interval, a bracketed interpolation between distinct feasible extrema gives a genuinely separated matched sham with exact first/second-order constraints. If the target lies outside all feasible intervals, report a **geometric nonexistence under the frozen separation threshold** rather than continuing random-plane attempts.

For a prospective LGA-10B redesign, test whether such spectral constrained matching raises source-only admission toward ≥95%, while recording the fraction of cases with provably or numerically infeasible distant curvature level sets. A new protocol, separate 31065 source roster and EvalMath preflight must precede any trained-model empirical inference. A relaxed angular threshold or new sampler is a **new study**, leaving the first LGA-10 source-only FAIL untouched.

BPD generalization: a high-dimensional physical dark projector may be a viable carrier of delayed information while a stricter second-order matched **control ensemble** has near-zero diversity. Separate (a) null dimension, (b) attainable curvature-level-set volume, (c) distance to source-donor direction and (d) native task response. Orthogonal factor transport must preserve all physical derivatives and the metric before attributing those effects to BPD components. Learned sparse BPD factors, Nanbeige and Sudoku have not been tested in LGA-10.
