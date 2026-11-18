# SUB-05: clock-indexed quotient algebra and causal cocycles

## State-space quotient with an observable clock

Let H be a finite-dimensional Euclidean space, X an input set, and F_x:H→H a differentiable shared transition. The recurrent computation is h_(t+1)=F_x(h_t) with the same physical F at every pass. For each pass define an orthogonal rank-r projection P_t:H→H and the quotient q_t(h)=P_t h. The clocked state space is the disjoint union of the images of P_t, labeled by t.

There is a well-defined deterministic quotient map G_(t,x):im(P_t)→im(P_(t+1)) obeying q_(t+1)∘F_x=G_(t,x)∘q_t on a domain D_t if and only if P_(t+1)F_x is constant on every P_t fiber of D_t. This is an elementary fiber-factorization criterion. It does not require a stationary physical projection.

If D_t is invariant/compatible with the next reachable domain, the quotient process composes:

  q_(t+k) ∘ F_x^k =
    G_(t+k-1,x) ∘ ... ∘ G_(t,x) ∘ q_t.

The composition follows by induction over k. The final readout R(h4) additionally factors through q4 only if R is constant on its fibers, an independent condition.

## Differentiable sufficient condition and finite-dose defect

For δ∈ker(P_t) and an entire segment h+sδ in a convex admissible fiber, the fundamental theorem of calculus gives

  P_(t+1)[F_x(h+δ)-F_x(h)] =
    ∫_(s=0)^1 P_(t+1) D_h F_x(h+sδ)(I-P_t) δ ds.

Therefore a uniform operator-norm upper bound ε_t on P_(t+1)D_hF_x(z)(I-P_t) over the segment implies

  ||P_(t+1)[F_x(h+δ)-F_x(h)]||₂ ≤ ε_t ||δ||₂.

If the Jacobian cross-block vanishes along every fiber segment in D_t, quotient closure is exact there. Vanishing *only on clean h_t* is insufficient to guarantee exact closure for finite δ.

For F_x(h)=tanh(Wh+Ux+b),

  D_h F_x(h)=diag(1-tanh²(Wh+Ux+b))W.

A global upper bound on every cross-block is ||W||op. The experiment's A term approximates a much sharper source-distribution Frobenius moment on clean states. It is a local surrogate rather than a uniform certificate.

## Why the clock matters even with tied weights

The empirical distributions of h1,h2,h3,h4 differ because the same nonlinear W is repeatedly applied to changing states. The most useful physical read directions can therefore vary by pass. A synthetic exact example uses 12 coordinates partitioned into three four-dimensional blocks. Let W cyclically permute these blocks and choose any rank-eight P1 retaining two blocks. Set P_(t+1)=W P_t W^T. Then

  P_(t+1) W (I-P_t)=0

for every pass, while P1 W(I-P1) is nonzero. For a block-separable elementwise nonlinear transition, this moving-fiber closure can remain exact for finite off-fiber perturbations. A time permutation disrupting the alignment typically restores cross-block coupling even though it uses exactly the same four rank-eight projectors.

The structural source-only test in SUB-05 verifies this finite example separately from trained-model measurements. It is an exact algebraic positive control, not a claim about a trained model.

## Source-only learning objective

For the learned model define D_t(x)=D_hF_x(h_t(x)) and output-observability C_t=E[g_t g_t^T], g_t=∂R(h4)/∂h_t. The registered bundle objective is

  A=E_(x,t)||P_(t+1)D_t(x)(I-P_t)||²_F,
  B=(1/4) Σ_t tr((I-P_t)C_t),
  S=(1/(3⋅2r)) Σ_t||P_(t+1)-P_t||²_F,
  L=A/a0+0.35 B/b0+0.50 S.

The normalizers a0 and b0 come from the frozen stationary SUB-04 source-only projector on the same discovery examples, and every P_t starts in that same stationary subspace. A targets next-state quotient closure; B guards eventual output relevance at each pass; S penalizes gratuitous rotation. The final QR orthonormal factors have 4×8 basis dimensions, while the source model still reuses one matrix W.

QR/Adam optimization and observation of a decreasing source loss do not certify global optimality. The finite-dose heldout test evaluates whether any benefit extends beyond the linearized discovery objective.

## Strongest null and limits

Seven fixed nonidentity temporal permutations score the exact same four learned subspaces. The one-root statistic

  Δ=min(C_static, min_j C_timeperm,j)-C_aligned

demands improvement over the strongest (lowest closure-error) fixed control. This family is prespecified, so selecting its minimum does not create an unreported comparison; its conservatism is intentional. Other angular shams probe stability of the learned time-indexed subspaces under physical rotation without modifying the shared W.

Each scored perturbation has equal Euclidean dose α, with a matched raw Gaussian before projection onto a different candidate null fiber. This is an operational latent-state intervention, not a measured energy or physical device equivalence.

The sampled C(bundle) is an RMS of retained **next-state** changes. Terminal source-logit intervention RMS and the autonomous clocked-student's true-label behavior are separate outcomes. An approximate time-dependent quotient on the sampled domain does not establish full mechanistic identity or phenomenal continuity.

## Statistical evidential scope

Eight independently trained sources form eight independent analysis units. Multiple time steps, input states, projector search iterations and rotated controls within each root do not increase independent n. A prospective strong effect and root-bootstrap interval at n=8 remain engineering only until a distinct ≥59-root independent sham-tail calibration, strict EvalMath power/variation review and separately sealed confirmation cohort.
