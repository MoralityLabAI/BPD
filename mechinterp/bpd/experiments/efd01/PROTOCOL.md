# BPD-EFD-01: Energy-fiber geometry and physically covariant delayed observability

**Prospective registration committed before implementation, 18 November 2026.** Complementary source-only mathematical instrument for MoralityLabAI/LoopedTransformers LOOPSCOPE-10 / LGA-07. It references the **existing** LOOPSCOPE-08 POSITIVE on the trained four-pass horizon and LOOPSCOPE-09 **FAIL** beyond four recurrent passes. No pretrained checkpoint or protected dataset will be opened in BPD-EFD-01 source-only CI.

## Mathematical object and exact claims

For a finite-dimensional real physical state subspace U (dimension d), current observable Jacobian A:U -> R^m, and an external differentiable energy E with restricted gradient row g, define
  K = ker A intersect ker g.
If C is the row stacking [A;g], then **dim K=d-rank(C)>=d-rank(A)-1** by rank-nullity. K is defined in the physical state, independent of any chosen component dictionary. Finite current-readout and energy invariance do **not** follow from this differential statement.

Given a future task-readout Jacobian B:U -> R^p, define delayed differential capacity r_delayed=rank(B|K) with the same SVD threshold as for K. This quantifies locally future-observable directions currently invisible to A and first-order energy-tangent. It does not quantify a complete nonlinear transfer or task correctness.

For a nonzero gradient g and native displacement v, define alpha= -<g,v>/<g,g> and r=v+alpha*g. Then <g,r>=0. A corresponding antisymmetric matrix Q=(r g^T-g r^T)/||g||² satisfies Qg=r. **This identity is algebraic and non-identifying:** arbitrary Q can represent any orthogonal residual. Claiming a learned low-rank rotational mechanism would require an independently frozen, restricted Q estimated on old source states and tested on new ones.

For any orthogonal physical change of coordinates R, transport g'=R^Tg, v'=R^Tv, A'=AR, B'=BR, and K'=R^T K; alpha, dim K and r_delayed must remain invariant. More general nonorthogonal factor transformations require transport of the physical metric and are *not* claimed by Euclidean projector tests. The production TiedComponentLinear uses W_hat=(VU)^T; its singleton factor coordinates do not define intrinsic mechanisms.

## Initial source-only implementation (engineering phase)

A NumPy-only exact-math instrument will implement robust SVD row-null basis, orthogonal projector, energy-descent/tangent decomposition, skew representer and future restricted-rank under a registered common tolerance **rtol=1e-9** and absolute zero threshold **1e-12**. Reject nonfinite arrays and inconsistent shape contracts. Near-degenerate singular spectra must be reported and tested against a known-rank oracle. Engineering controls:
- a manufactured future-only dark channel with zero present observability and nonzero future rank;
- a zero-future row oracle with r_delayed=0;
- orthogonal coordinate change preserving physical results, dimensions and projectors;
- no observable gradient / zero energy gradient case;
- finite-step curvature causing energy increase despite locally negative g dot v;
- exact K constraint residual and projector idempotence;
- an unregularized skew representer test explicitly labeled non-identifying.

A CI workflow will run only excluded synthetic shapes with Python 3.11 and NumPy; it must not pull model weights, train a model or generate any scientific source roster. All subsequent native experimentation requires separate, prospective source/model/sham manifests and the EvalMath instrument-resolution preflight. No claim of BPD physical causal mediation, sparse feature semantics, or rank compression follows from algebraic tests.

## Planned BPD-native extension (separate protocol)

Apply these constraints to actual BPDModel/TiedComponentLinear physical writes and orthogonally transported gauge-covariant projector pairs. Match clean and **realized closed-loop physical L2 dose** in shams, check full-mask head parity, and compute true correct-target recovery on naturally reachable donor states. Preserve source admissibility failures instead of replacing cases. A derivative-equal sham and a finite-head/energy preservation test are both required; the existing SF05 tangent-projection result shows why first-order silence alone fails as a finite guarantee. Never repurpose NC-03/06/07 evaluation roots or CQ-08 confirmation roots for new confirmatory significance.

Independent trained model roots form the inference units. Within-root tokens, loops, donors, K-basis columns and random-sham draws are repeated measurements. An independent 59-root calibration and a sealed confirmation population, with a fixed practical effect and power checked by EvalMath, must precede any confirmatory result. Protected ARG-03-N and AMM-03/04/05 test populations remain sealed.
