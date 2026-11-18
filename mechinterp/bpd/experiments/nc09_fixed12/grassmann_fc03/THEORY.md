# BPD-NC09-FC03: a constrained local Grassmann chart for causal physical shams

## Rank-preserving geometry with a nontrivial displacement

Fix a trained recurrent model, its original source-selected DP12 passwise rank tuple r=(r_1,r_2,r_3,r_4), each r_t in {2,...,6} and sum r_t=12, and a 20×r_t orthonormal basis E_t of the retained BPD read subspace. For a seeded complementary orthonormal frame Q_t with E_t^T Q_t=0 and Q_t^T Q_t=I, construct

B_t(theta)=cos(theta) E_t + sin(theta) Q_t,
H_t(theta)=B_t(theta) B_t(theta)^T.

The orthogonal-frame identities give B_t^T B_t=I, so H_t is a symmetric idempotent rank-r_t projector. Furthermore

E_t^T B_t = cos(theta) I

and all r_t principal angles between the original subspace and its sham are exactly theta (0<=theta<=pi/2). For the Frobenius chordal distance,

||H_t-P_t||_F^2 = 2 r_t sin^2(theta)

where P_t=E_t E_t^T. Dividing the sum over the four passes by 2 sum_t r_t=24 gives the source-normalized distance exactly sin(theta). FC03 freezes theta=.25 radians, so every optimized sham remains separated from DP12 by about **0.2474** independent of its optimized complement direction. The lower bound of .20 therefore tests a real edit, rather than accepting identity masquerading as a null.

The solver exposes only eight free parameters y_{t,1},y_{t,2}, two per recurrent pass, by combining three independent complement frames and reorthonormalizing. Even when y changes, the QR complement condition ensures the constant-angle property in exact arithmetic. It is a local numerical chart on a product of Grassmann manifolds, not an exhaustive search over every rank-r projector. Float64 orthogonal projector traces and idempotence errors are independently checked from archived byte receipts.

## Source matching as a feasible-region problem

Let x_n be a clean model-only derivative discovery example. The original source law supplies factor reads z_{n,t}, prewrite activations a_{n,t}, and the detached-sign gradient of its own final score g_{n,t} with respect to the read. For any tuple of retained projectors H, define the clean retained physical write norm W(H), the additive first-order lost model score d_n(H), its signed mean m(H) and its RMS v(H):

W(H)^2 = (1/(4*20)) sum_t E_n || a_{n,t} H_t U ||^2,

d_n(H) = sum_t g_{n,t}^T (I-H_t) z_{n,t},
m(H) = E_n d_n(H),
v(H) = sqrt(E_n d_n(H)^2).

The original FC02 null sampler tried four fixed angles and random complementary directions, accepting only if all three source inequalities passed:
- |W(H)/W(P)-1|<=0.15.
- |m(H)-m(P)|<=max(0.005,0.25|m(P)|).
- |v(H)-v(P)|<=max(0.005,0.25v(P)).

For some model roots these inequalities occupy a small subset of randomly proposed complementary directions; FC02 observed 19/59 roots with too few qualifying random proposals. FC03 preserves exactly those three inequalities, changing the algorithm to constrained optimization within a fixed-angle rank-preserving Grassmann chart.

The objective is a sum of positive-part squared normalized **constraint excesses**. If c_i(H) is the absolute normalized residual divided by its frozen acceptance tolerance (or a pairwise separation shortfall normalized to 0.18), the penalty is sum_i [max(0,c_i-1)]^2 + 0.001 ||y||_2^2/8. This loss is a numerical feasibility search rather than a statistical task predictor. Its gradient is computed through the QR chart using clean model-score source information, with full source constraints independently checked after every deterministic Adam step. Once a chart reaches the feasible set, its *first* eligible projector is frozen; optimization never searches heldout task labels. Thirty-two new charts and at most 64 steps per chart are fixed for each root.

## Separating the eight nominal sham interventions

All shams share the same original passwise ranks and fixed distance from DP12. They must also be at least 0.18 normalized chordal units apart from every previously accepted sham. This disallows eight clones of one favorable first-order tangent direction. The first eligible projector in chart order is accepted, and the next chart's penalty includes pairwise separation from all previously accepted shams. The code archives each selected projector as 4×20×20 float64 little-endian bytes with SHA256, enabling independent reconstruction of trace, idempotence, physical source metrics and pairwise distances without changing model code.

Because the actual matching constraints use root-specific source moments, some models may have no feasible candidate in the chosen eight-dimensional chart at theta=.25. FC03 has an explicit source FAIL for that outcome. The inverse probability of sampling a feasible chart and local optimizer convergence are part of its measured engineering performance, and the sixteen roots cannot be replenished.

## Relation to FC01 and FC02 uncertainty

FC01 used eight source-write-RMS-matched local physical shams nested under each of 59 independent trained models. Its sample maximum of rootwise absolute median-centered sham deviations gave a conservative distribution-free upper 95/95 tolerance bound for *single-root worst orientation*. The resulting EvalMath preflight held confirmation. FC02 used stronger signed model-score matching and attempted a nested bootstrap null for a future *32-root mean*, with an exact conditional zero pseudo-selected sham expectation, but its rootwise source admission failed on 19/59.

FC03 tests *source-only feasibility* of matching and separation with an optimization method. Its root NLL outcomes are measured under the original true four-pass closed-loop BPD physical edits, with external labels withheld from the solver. A positive feasibility gate would support a distinct future 59-root calibration on independent sources. It does not alter the exact FC01 95/95 conclusion, certify the FC02 incomplete aggregate null or authorize an independent 32-root scientific confirmation run.

The source gradients g_t are local clean-state derivatives of a self-score, and the sum of per-pass first-order effects approximates a nonlinear recurrent trajectory. Matching m and v is much weaker than matching every per-example causal response. Physical task NLL may differ across shams even when W,m,v and principal-angle geometry are matched. Retain independent labeled heldout outcomes to measure that difference and state the source-family boundary.
