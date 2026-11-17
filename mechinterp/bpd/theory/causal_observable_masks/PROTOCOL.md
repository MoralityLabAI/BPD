# BPD-CQ-01: delayed-observable sparse-mask quotient

**Source-only exact rational audit, 17 November 2026.** This new isolated study extends the BPD expected-L0 optimal-basis/mask algebra to finite-horizon observable quotients. The existing B1–B4, PD-01/02/03, FSI-02, SCM-20–23 and sealed ARG-03-N protocols remain immutable.

## Question
For two equivalent **global expected-L0-optimal** full-rank read bases related by `M = R M_*`, which component masks are indistinguishable through a finite sequence of linear recurrent readouts? When may their observable-equivalence set fail to be a coordinate Boolean algebra? How does the minimum number of simultaneously active components depend on a delayed readout?

Use an analytically continuous full-dimensional Gaussian activation law for which `Pr[m.a=0]=0` for every nonzero read direction, so every invertible basis is an optimum for the *idealized* unregularized expected-L0 objective. This supplies the optimal-basis premise; a seeded rational example is a mathematical witness, not a trained dictionary or Gaussian sample.

## Frozen mathematical model
Read basis anchor `M_*`, equivalent read basis `R M_*`, write `U`, linear recurrence `T`, readout `C`, reached-input subspace matrix `S`. For a binary component mask `D`, form

`E_D = (R^-1 D R - D) M_* S`.

Observed deviation after `j` further recurrences is `C T^j U E_D`. A mask is k-horizon observationally invariant iff all deviations vanish for `j = 0..k`. Its Gramian is `G_k = sum_{j=0}^k (C T^j U)^T(C T^j U)`, and the exact positivity identity gives equivalence to `E_D^T G_k E_D=0`.

The current system is linear with fixed `T` and `C` only. In a nonlinear model this is a tangent-space **diagnostic**, requiring finite-patch and curvature controls to establish actual causal invariance. Masks related by one optimal mixing are checked relative to the declared anchor, with exact data-span restrictions. This does not identify canonical neural basis components.

## Frozen reference validation (subsequent to engineering tests)
The replay uses its own seeded rational worlds rather than preexisting test seeds: 384 worlds with `seed=2026111801`. World indices `0..127` have 2D shear `R=[[1,t],[0,1]]`, dark-at-zero readout `C=[0,1]` and swap recurrence, full reached input space. Worlds `128..255` have the 4D shear with `S=(0,-1,-1,1)` and delayed readout, testing nonclosure of the binary invariant-mask set. Worlds `256..383` are general 4D random rational unimodular shears, random coordinate readouts and coordinate-permutation recurrences with `S=I`, exercising the Gramian criterion under a nontrivial input span. Each twist is chosen from `{-7,-5,-3,-2,-1,1,2,3,5,7}`. All arithmetic uses Python Fraction with exact equality.

Every world must satisfy Gramian/direct-response equivalence, horizon-monotone mask sets, and stabilization after dimension-minus-one transitions (Cayley-Hamilton). In the first two cohorts the exact analytic anchors require 4→2 and 16→6 invariant masks between horizons zero and one. For the 4D cohort, the two invariant masks `(1,1,0,0)` and `(1,0,1,0)` have a noninvariant intersection `(1,0,0,0)`. The phase-disjoint one-coordinate source admits a top-1 preserving gate; the mixed two-coordinate source requires both gates to preserve present and next readouts. The random cohort has no fixed positive numerical success threshold: report its complete count distribution and all invariant failures.

Only source integrity and mathematical identities determine engineering validity. This is a deterministic, source-only stress suite, **not 384 independent neural-model observations**, and no inferential p-value is computed. The extended seed is distinct from the engineering test seed `20261117`.

## Comparison with SCM-21 and BPD-PD-03
Use SCM-21's **public** final `summary.json` from `Compactification-Theory` as a retrospective cross-check. Its rank-four native-write intervention was much smaller than readout-aligned rank-four differences, so orientation and scale remain confounded in those receipts. Preserve original clean and mediation denominators and report source SHA. Never use SCM-21 test outputs to select BPD dictionaries, neural masks or a prospective threshold.

Next neural work: freeze trained BPD component axes using **calibration only**; compute current/future-readout Jacobians through the chosen native pass; fit candidate mask blocks by delayed-observable quotient; run matched-norm native parameter edits versus random/head-visible/temporal transport controls. Register independent generator-world counts and an effect-floor power check in EvalMath. A reconstruction-only success is inadequate; use actual native causal task outcomes.

### Explicit barriers
- ARG-03-N's sealed 168-case `test.jsonl` is excluded. Never import, open, or evaluate it.
- No 7M TRM, Huginn or SCM-21 private checkpoints are available in this sandbox.
- Preserve the source-level difference between full BPD optimum-mask commutators and finite-horizon observational equivalence. The latter can fail Boolean closure.
