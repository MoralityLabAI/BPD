# NC-12 mathematical notes: low-query source-assisted projector transfer

The source-only registered protocol is [PROTOCOL.md](PROTOCOL.md), first commit `6a4b5ebc16fae730f17b0e5dac5564eaf419c071`. These notes explain the frozen method and make no claims from NC-12 held-out results.

## Why a target gradient budget smaller than hidden width needs a prior

Let the model A and B physical pre-`mix` reads at recurrent pass `t` be `a_A,t` and `a_B,t`, each in `R^20`. The local output-score Jacobians are rows

```
g_A,t(x) = ∂s_A(x)/∂a_A,t
g_B,t(x) = ∂s_B(x)/∂a_B,t.
```

On paired inputs `x_1,...,x_q`, stack unit-normalized Jacobians into `G_A,t, G_B,t in R^(q×20)`. Their empirical cross moment

`C_t=G_A,t^T G_B,t / q`

has `rank(C_t) ≤ min(q,20)`. For q=4,8,16, the usual orthogonal polar factor is underdetermined on at least `20-q` input directions. Since these directions may carry most of the *source projector* `P_A,t`, an arbitrary SVD null-space completion can change the transported physical write intervention while leaving the fitted observed Jacobian correspondence unchanged.

NC-12 therefore fits a single orthogonal **activation prior** `R_act,t` on an independent 128-row paired, unlabeled calibration sample. The registered low-query Jacobian map is

```text
R_(q,t) = polar(G_A(q,t)^T G_B(q,t)/q + 0.05 R_act,t)
```


For a generic nonsingular matrix `M=UΣV^T`, `polar(M)=UV^T`, and

`R_(q,t)^T R_(q,t)=I`.

Multiplication by the activation prior's inverse reveals the regularizer's geometry:

```text
(C_t + 0.05 R_act,t) R_act,t^T = C_t R_act,t^T + 0.05 I
```


The ridge stabilizes directions invisible to q Jacobian rows. Its numerical singular-value diagnostics are recorded, and its fixed coefficient is **not optimized from held-out task behavior**. This gives a deterministic chart rule without asserting that the chart is a true isomorphism of independent learned model circuits.

Transport uses `P_A→B,q,t=R_q,t^T P_A,t R_q,t`. Each `P` is a symmetric rank-four projector, so this preserves rank and orthogonal geometry. The intervention's physical operator inside the B model is `P_A→B,q,t W_B^T`. The original B model's nonlinear recurrence means that a small geometric correspondence error may propagate through all subsequent reused physical writes.

## Why gradient fit is a proxy for causal transfer

For a clean trajectory, let `z_t` denote its physical prewrite read, `U_B=W_B^T`, and `g_t` the target logit's local derivative with respect to the write. The first-order effect of retaining a projected write at one pass is approximated by

```text
Delta_s ~= sum_t [(alpha * z_t P_(q,t) U_B - z_t U_B) dot g_(w,t)]
```


where `z_t` is a physical prewrite **row vector**, `U_B=W_B^T` and `g_(w,t)` is its post-write **column** score gradient. Higher-order recurrent effects are omitted from this first-order approximation. Higher-order terms include changes to later recurrent hidden states, their nonlinear Jacobians and feedback into subsequent writes. Consequently, maximizing paired-Jacobian alignment is a **source-only observational objective**. Evaluating native BPD closed-loop interventions on *untouched* external task labels is required to determine whether the transported rank-four write retains predictive behavior.

The direct B-only comparator is especially consequential: with q target gradient rows, it may extract a rank-four B subspace more efficiently than a source-assisted orthogonal map, especially if A and B's internal nonlinear transforms differ substantially even on the shared input branch.

## Matched information and physical-dosage controls

Each q cell uses a nested prefix of the same 32 B Jacobian rows. Therefore the **unique** primary B-gradient query count is 32 over the entire sweep, while the per-cell effective budget is q. All methods have access to the same 128 paired B activation rows. The A-assisted method additionally uses A's pretrained projector and A's 1024-case discovery gradients; source gradient access is disclosed separately from target access.

Dose matching is performed on `split=6`, disjoint from gradient calibration and held-out task testing. For rank-four projectors `P`, the retained clean write energy is proportional to

```text
D_B(P)^2 = (1/(4*20)) sum_t tr(M_t P_t (U_B U_B^T) P_t)
M_t = E[a_(B,t)^T a_(B,t)]  (with a_(B,t) a row vector)
```


This matches the physical `mix` write RMS by gain calibration within fixed bounds [0.5,1.5]. The physical nulls use independent local angular projector draws, accepting the first eight within ±15% of the transported projector's discovery-dose RMS. Physical-write RMS matching controls energy, although it does not equate downstream causal Jacobians or clean-to-intervention state-distribution shifts.

## Statistical unit, multiplicity and evidence class

Every independent training root generates **two dependent models** and multiple nested q results. The root-pair is the resampling unit, so the 8 roots produce 8 independent paired observations per contrast. NC-12 predeclares 4 q values and 4 directional test endpoints. Bonferroni familywise alpha=.05 across 16 comparisons produces `1−.05/16=99.6875%` two-sided per-test bootstrap intervals, with 4,096 whole-root draws.

An eight-root bootstrap is a developmental uncertainty estimate. It cannot calibrate a nonparametric 95th-percentile sham variation floor with 95% confidence. That tail needs at least 59 independent sham worlds even under the sample-maximum bound. Fresh sham calibration and a separate EvalMath power preflight must precede any confirmatory claim.
