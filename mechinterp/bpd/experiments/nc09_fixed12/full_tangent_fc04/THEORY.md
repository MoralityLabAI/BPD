# FC04: full complementary tangent Jacobians and constrained projector feasibility

## Rank-preserving fixed-angle Grassmann chart

For one recurrent pass let E ∈ R^(20×r) be an orthonormal DP12 read basis and C ∈ R^(20×(20-r)) its orthogonal complement. For any full-column-rank Y ∈ R^((20-r)×r), form

Q(Y)=C qf(Y), B(Y)=cos θ E + sin θ Q(Y), H(Y)=B(Y)B(Y)^T,

where θ=0.25 radians and qf denotes reduced QR with positive diagonal signs. Because EᵀQ=0 and QᵀQ=I, the columns of B are orthonormal. H is a symmetric idempotent rank-r projector, and its r principal angles from P=EEᵀ are exactly θ. Summing the passwise chordal distances and dividing by twice the fixed total rank 12 gives

d_chord(H,P)=sqrt(Σ_t ||H_t-P_t||_F² / 24)=sin θ≈0.247404.

This gives a certified nonzero intervention above the original .20 lower bound, while preserving NC09's original per-pass rank tuple and 12-slot total.

The full Y chart uses Σ_t (20-r_t)r_t coordinates, generally around 200 in this model family. QR contains a right-factor gauge: different Y matrices can represent the same Q or H. Full-rank neighborhoods of Y form a continuous local chart. Differentiation through its QR factor is valid away from column-rank degeneracies and sign changes of the diagonal convention.

By comparison FC03 searched only eight predetermined complementary-frame mixture coordinates. Enlarging the chart can make previously inaccessible source orientations reachable. It also increases numerical cost and exposes singular directions, so FC04 records the actual constraint Jacobian singular values and source failures instead of claiming a guaranteed solution.

## Why precisely three source constraints matter

On each independent model root's 1024 clean derivative-discovery cases, define FC02's physically and causally motivated observables:

W(H) = sqrt(E_{n,t} ||a_{n,t} H_t U||² / 20),
d_n(H)=Σ_t g_{n,t}ᵀ(I-H_t) z_{n,t},
m(H)=E_n d_n(H), v(H)=sqrt(E_n d_n(H)²).

The model-final-score derivative g and factor read z are taken at clean recurrent states. No externally labeled terminal task outcome enters these observables. The three source matching gates are W ratio deviation ≤.15, signed mean deviation ≤max(.005,.25|m(P)|), and RMS deviation ≤max(.005,.25v(P)).

These three real-valued observables define a smooth map f from a local product Grassmann chart to R³ where differentiable. If a point y₀ satisfies f(y₀)=0 and the differential has rank three, the implicit-function theorem gives a local solution manifold of codimension three near y₀. This says nothing about existence of such a y₀ at the preregistered nonzero angle, and the prescribed numerical chart may not find it on all trained sources.

## Damped minimum-norm correction

The FC04 normalized signed residual f(Y) is

((W(H)/W(P)-1)/.15,
 (m(H)-m(P))/tol_m,
 (v(H)-v(P))/tol_v).

Each original source gate passes when the absolute value of its residual is ≤1. Let J=∂f/∂Y flattened, a 3×d Jacobian. The registered correction direction is

Δ = -Jᵀ(JJᵀ + 10^-3 I₃)^-1 f.

This is the damped least-squares minimum-norm correction in the large parameter chart, not a guess that the native recurrent update is conservative or gradient descent. Its norm is clipped to ≤.5. Deterministic backtracking in {1,.5,.25,.125} accepts the first direction whose positive squared normalized source constraint excess decreases. Every chart has at most 64 corrections, with 32 independent fixed-seed starts.

The row-space correction adjusts physical write dose and clean first-order signed/RMS score differences while leaving the large Jacobian nullspace unutilized for most of the update. Random initial full complementary frames vary those nullspace degrees of freedom across starts. The original independent source audit still enforces pairwise projector separation ≥.18 and original DP12 displacement ≥.20.

A rank-deficient J can create an impossible linear correction in some source direction. The damped inverse exists, and the chart is allowed to fail. This is important for weak-gradient models, whose absolute source-score thresholds created FC02/FC03's admission failures.

## Causal and statistical boundaries

The score d_n(H) sums clean-state derivatives across four recurrent passes. True physical BPD override-read edits change subsequent hidden states, and nonlinear interaction terms can dominate higher-order differences. Matching W,m,v does not prove identical externally labeled future-task NLL for source DP12 versus its sham, nor does it imply neuronal identity or a unique minimal task quotient.

The first FC04 development cohort uses sixteen fresh independently trained models, and all full same-tuple controls are source-selected before final task labels are read. The registered success criterion is a source-feasibility comparison versus the unchanged FC03 solver, with original FC02 rejection as an additional descriptive control. The independent unit is a trained model, never its chart starts or local sham projectors.

Even a positive FC04 development gate would only justify a new prospective 59-root sham calibration and independent EvalMath variation/power review before the separately sealed scientific confirmation population could be accessed.
