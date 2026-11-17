# SAU-BPD01: the linear expected-L0 boundary for curved features

**Research note, 2026-11-17.** Motivated by [SauersML/Manifold-SAE](https://github.com/SauersML/Manifold-SAE) and [SauersML/weight-lens](https://github.com/SauersML/weight-lens). The former uses smooth chart-valued features with a learned coordinate; its published joint multi-atom recovery gate still has blocked results. The latter identifies vocabulary-aligned directions of LoRA weight updates, with limited results for non-vocabulary-shaped behavior. Neither is evidence of a native recurrent BPD mechanism.

This document extends the conceptual *boundary* of [OPTIMAL_BASIS_AND_MASKS.md](OPTIMAL_BASIS_AND_MASKS.md). It does not replace its finite matroid argument, alter any B1-B4 measured results or run the strictly sealed ARG-03-N test.

## Exact circle counterexample for linear sparse-basis identification

Let theta be uniform on [0,2pi), a(theta)=(cos(theta),sin(theta)) in R2. Every nonzero row m has hyperplane mass

```
z(m) = Pr[m . a(theta) = 0] = 0.
```

Indeed, for nonzero m the trigonometric equation has precisely two roots modulo 2pi, a zero-measure set. For every invertible 2x2 linear read basis M, the expected-L0 objective is

```
J(M) = 2 - z(m_1) - z(m_2) = 2.
```

The optimum set contains every invertible basis. This is the *free-remixing* case of the existing filtration theorem, although the data law lives on a one-dimensional nonlinear manifold and is singular relative to two-dimensional Lebesgue measure. The geometric intrinsic dimension (one) must not be substituted for the expected number of nonzero coordinates in a linear basis (two).

The covariance is I/2. For any rank-one orthogonal projection P, the mean residual squared norm is

```
E ||a - Pa||^2 = trace((I-P) Cov[a]) = 1/2.
```

A decoder g(theta)=(cos theta,sin theta), given the true 1D phase coordinate, reconstructs exactly. This oracle capacity comparison does **not** match a trained TopK SAE's decoder parameter count, compute, coordinate discovery or L0 budget. It is a mathematical witness for the gap between a rank-one *linear subspace* and a one-dimensional *curved chart*. An independent CPU witness is in [Compactification-Theory SAU-CURVE01](https://github.com/MoralityLabAI/Compactification-Theory/tree/research/sauersml-curved-chart-20261117/research/curved_sae_gauge_v1).

## Fixed intervention masks cannot isolate the whole curved tangent

The tangent t(theta)=(-sin(theta),cos(theta)) varies across theta. At theta=0 and theta=pi/2, the tangents are (0,1) and (-1,0), which span R2. If a linear rank-one projector P were identity on the tangent line for every theta, its range would contain both vectors and would have rank at least two: contradiction.

The local tangent projector is **state conditional**, P_theta=t(theta)t(theta)^T, and is rank one separately at each phase. Under a physical frame rotation R, the correct transformation is

```
a'(theta) = R a(theta);
P'_theta = R P_theta R^T.
```

This is *gauge covariance of a projector field*. It is distinct from the existing static common-centralizer criterion DR=RD for fixed BPD diagonal masks under changes among full-rank linear optimal bases.

For recurrent dynamics h_{k+1}=F(h_k), a chart edit should be evaluated through the actual downstream derivative J_out(h_T) multiplied by the propagated physical tangent perturbation at its chosen pass. A projector-field identity on calibration data, an SAE feature match and a behavioral effect are separate tests.

## Work package BPD-CHART01 (prospective)

1. **Synthetic baseline:** measure linear fixed-rank and dictionary-matched TopK reconstruction on equal train/test phases, recording L0, decoder width, nonlinear parameter count and training budget. Include full-rank expected-L0 flatness as an analytic reference.
2. **Two-atom ambiguity:** vary coactivation and tangent-plane coherence. Report minimum singular value of the stacked tangent frame, cross-atom effect leakage and false-alignment rate. Following the published Manifold-SAE program, treat a diverged or unavailable joint solver as BLOCKED rather than a positive result.
3. **Checkpoint gauge:** learn alignment and direction orientation solely from new calibration states. Check held-out native causal responses under matched physical tangent doses against state-shuffled and random-subspace controls, with puzzle-cluster inference.
4. **Observation audit:** before activating any SAE hooks, run the [LoopedTransformers SAU-OBS01](https://github.com/MoralityLabAI/LoopedTransformers/tree/research/sauersml-observer-invariance-20261117/mechinterp/observer_footprint_cpu_v1) parity gate under the exact GPU/runtime/precision. A changed compilation may be a causal intervention in its own right.
5. **Weight readout adjunct:** treat a token-lens SVD of parameter edits as a *proposal generator*. Match each proposed write direction to rank- and norm-matched random updates, then intervene at the native recurrent pass and evaluate a frozen task margin. Token lists are not evidence that an inferred behavior is localized in that direction.

**Acceptance:** report new source-locked model/evaluation data, a calibration-frozen effect above both matched controls with cluster-level uncertainty, an observer parity audit, and enough independent seeds to bound selection. Keep the original linear BPD and ARG-03-N protocols intact. If these gates fail, publish the failure.

## Source and licensing boundary

The note independently derives the unit-circle statements. Manifold-SAE is AGPL-3.0; the inspected Watching repository root contains no license. No upstream implementation was copied. The proposed BPD experiment requires independently authored code or explicit compatible licensing.
