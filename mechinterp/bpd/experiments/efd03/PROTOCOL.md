# BPD-EFD-03: complete future-observability quotient of a physical dark fiber

**Prospective source-only mathematical instrument registration, 18 November 2026, before writing EFD-03 code or generating any new scored model/source data.** Companion to the independent LoopedTransformers LOOPSCOPE-12/LGA-09 protocol (registered first at 5719f8d), which tests the entire ten-logit first-order future response on a 96D physically supported predecessor/goal pair.

## Mathematical object and rank certificate

Let U be a real Euclidean physical state support of dimension n. The current readout derivative is A:U→R^m, frozen external scalar derivative is g:U→R and full future logits differential is B:U→R^c. Define K=ker(A)∩ker(g), and H=K∩ker(B). Then rank-nullity yields

  dim K = n − rank([A;g]) ≥ n − rank(A) − 1,
  dim H = dim K − rank(B|K) ≥ n − rank(A) − 1 − rank(B).

For the original tiny recursive cyclic-copy model, pair U=R^96, current head has two rank-ten blocks so rank A=20, the energy row adds one independent dimension, and ten future logit outputs imply rank B≤10. Generic dim K=75 and dim H≥65. These are exact linear-algebraic statements when ranks are ordinary mathematical ranks. Any SVD threshold produces an explicitly marked **numerical** rank with its own conditioning and residual.

For z∈K, the minimum-norm solution z0 of Bz0=Bz inside K is the projection of z onto rowspace(B|K). Therefore w=z−z0∈H, w⊥z0, and ||z||²−||z0||²=||w||²≥0. For any unit q∈H, the family z'=z0+||w||q satisfies z'∈K, Bz'=Bz and ||z'||=||z||. Such shams are exactly matched in the full ten-logit first-order prediction and physical Euclidean dose, while finite nonlinear source-conditioned task behavior remains an independently evaluated endpoint.

The family degenerates when H is zero-dimensional or ||w||≈0. Report degeneracy, basis dimension, spectral gap and residual; do not claim a diverse sham where the source edit is already entirely in the future-rowspace. Under an orthogonal physical reparameterization Q, transport A'=AQ, g'=Qᵀg, B'=BQ, z'=Qᵀz; all ranks and physical family relations must be preserved. Arbitrary nonorthogonal BPD factor transformations require transporting the metric and do not preserve ordinary Euclidean random-sham measures.

## Stage 0 source-only test contract

Use NumPy FP64, deterministic excluded test seed 61063 only, no trained model weights and no scientific source seed 31063. Implement (i) right-null bases of stacked/current/energy physical operators under rtol=1e−9/atol=1e−12 with explicit spectral and constraint telemetry; (ii) restricted future-logit rowspace and H nullspace; (iii) eight independent same-future-J, same-K, equal-L2 shams with independent fixed hash-derived seeds; (iv) numerical tests against exact known-rank 96D/75D/65D witnesses, zero-row and near-degenerate spectra; (v) orthogonal coordinate/gauge transport and counterexamples where matching one signed margin leaves other future logits free; (vi) finite quadratic response divergence despite exact first-order constraints.

Stage 0 CI runs only Python stdlib + NumPy 1.26.4 CPU tests and cannot fetch pretrained weights or access protected ARG-03-N, AMM or CQ cohorts. Engineering tests establish numerical instrument correctness only.

## Stage 1 / future source-natives

Native LOOPSCOPE-12's preregistered same-K and full-ten-Jacobian controls are a separate experiment with original exposed frozen 44K teachers (development only). BPD-EFD-03's independent numerical algorithm may audit its Jacobian/spectra and gauge-stable physical projections, but those checks alone do not prove internal optimization, learned semantic features or sparse BPD component mediation.

A later BPDModel/TiedComponentLinear scientific experiment must freeze fresh independently trained model roots, real factor reconstruction and full-mask parity, actual physical retained component write subspaces, orthogonally transported subspace projectors, source-reachable donor behavior, and a root-level EvalMath mechanistic-variation preflight. If invoking a 95th-quantile/95%-confidence distribution-free maximum sham tolerance, calibrate at least 59 independent trained roots and protect an independent sealed confirmation population. The three old source-exposed teacher roots are not a formal confirmation sample.

Preserve first original LGA-05/06/07B/07C/08 scientific receipts and all failures. All `3106x` source seeds already opened in earlier experiments remain exposed; source-only tests use excluded seeds only.
