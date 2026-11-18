# BPD-NC-09 Fixed12: first-order allocation, physical dose and task semantics

## Finite source-only rank allocation

For each clean four-pass recurrent model, let z_t(x) be its full 20D physical BPD factor read before shared weight multiplication, and g_t(x) the derivative of the teacher's own final logit with respect to that read. Under an orthogonal factor basis, define C_t = E[g_t g_t^T] on a frozen derivative-discovery sample, and P_t(r) the orthogonal top-r spectral projector. For each pass and rank, the original NC-08 proxy is

L_t(r) = E[(g_t^T (I-P_t(r)) z_t)^2].

NC-09 chooses one tuple (r_0,...,r_3) of per-pass ranks from {2,3,4,5,6} with total sum exactly 12 minimizing Σ_t L_t(r_t). It enumerates all 5^4 tuples and fixes lexicographic tiebreaks. This is an exact finite multiple-choice knapsack under source-model data. Its global optimum is for the *registered sum of local proxy scores*, rather than a claim of global optimum causal retained write or downstream task NLL.

A first-order derivation suggests L_t: if the read-to-physical-write truncation is δz_t=-(I-P_t)z_t, the local model-final-score perturbation at a clean trajectory has leading term g_t^T δz_t. Squaring and averaging gives L_t. Summing over passes discards cross-time effects, including g_t^T δz_t g_s^T δz_s, and assumes linear behavior at clean states. A recurrent nonlinear model may redistribute task information under intervention. NC-09 tests whether this approximation still improves the *actual* closed-loop terminal task.

## Equal total rank and physical retained write dose

A fixed rank-3 projector at all four passes has total rank 12, exactly equal to NC-09's passwise allocation. Equal rank budget alone does not imply equal clean-source retained physical write RMS. For clean activation a_t and physical decoder U, let

D(P)^2 = E_{t,x}[||a_t P_t U||^2] / 20.

For any scalar gain α>0, the clean retained write dose satisfies D(αP)=α D(P). Consequently, on the same independent derivative-discovery set, the fixed source-only gain

α_* = D(P_DP12) / D(P_flat3)

provides exact retained **physical write RMS** matching between the equal-total-rank methods. Its allowed source admission interval [0.5,1.5] is frozen before any new model training. The matched-dose control is independent of true task labels, heldout model outputs and post-edit future hidden states.

This pair of comparisons isolates different things. DP12 versus flat3 at α=1 estimates the benefit of how write directions are distributed across passes at fixed dimension. DP12 versus α_* flat3 tests that benefit when the clean retained write dose is also equalized. Passing both, with independently defined terminal task behavior and physical local shams, is a stronger causal attribution than observing a single equal-rank NLL advantage. The boosted/suppressed flat3 gain means its projector is scaled; this controls physical dose but changes the exact idempotent projector operator, and therefore requires separate reporting.

The original NC-08 DP12-minus-flat3 observation was a **secondary result on exposed eight-root development data**. NC-09's new independent roots and exact source preregistration establish a genuinely fresh developmental test of that hypothesis; they do not turn either dataset into a calibrated scientific confirmation.

## Native closed-loop gauge transport

Use a full BPDModel/TiedComponentLinear representation with exact orthogonal factors W=VU. The gain-weighted physical retained write is α V P_t U, applied independently at each of four tied nonlinear recurrence steps through override-read. With an orthogonal factor coordinate change R, V'=VR, U'=R^T U, and P'_t=R^T P_t R, giving

α V' P'_t U' = α V P_t U

as a physical operator identity. Equivalent source inputs should yield equivalent full four-pass logits, apart from numerical tolerance. This is an orthogonal gauge statement. It does not imply invariance under arbitrary GL(20) factor rescalings without a corresponding physical metric.

The final true-label task is separately defined from the model's own self-score used to discover the projectors, and no external heldout labels may enter the allocation, gain, sham admission or gauge checks. Four recurrent steps use newly updated hidden states after each physical edit. This creates a behavioral causal endpoint beyond isolated factor/covariance observational alignment.

## Source legal shams and conditional nature of dose matching

A model-blind, source-only local sham rotates each pass's retained rank-r_t eigenbasis toward an independent orthogonal complement through one frozen principal angle from {0.45,0.65,0.85,1.05}. The first eight out of 4096 proposals whose retained clean-source RMS is within 15% of DP12's RMS are used. The sham matches total rank, passwise rank tuple and approximate physical write dose, while the true task NLL supplies the final independent behavioral criterion.

Matching clean-source RMS does not match the future edited-state distribution or task-directional Jacobian response. NC-09 reports actual heldout write displacement and clean/retained KL by each remaining pass. No trajectory can be reweighted or shimmed based on heldout outcomes to repair a failed control.

## Root-level inference and claims

Sixteen independently trained 20D recurrence roots are the statistical units. The five registered paired root metrics measure DP12 improvement versus same-total-rank flat3 and gain-matched flat3, absolute true-label fidelity, extra cost versus flat rank4, and causal specificity versus eight local physically dose-matched shams. A deterministic 4096-draw whole-root percentile bootstrap with seed 909009 forms joint engineering admission criteria. Each required contrast must pass. The conjunction is an intersection-union **development gate**, not an independently calibrated scientific confirmation.

If all sixteen roots are source-admitted, the fixed DP12 projector beats both 12-slot controls by practical mean margins with positive root intervals, preserves absolute task fidelity and remains within the stronger rank-four reference limit while outperforming local rank/dose-matched shams, the experiment qualifies as developmental evidence that pass-specific rank allocation is causally useful in this trained synthetic task. A failure preserves all original original outcomes.

A later scientifically confirmatory program must independently sample at least 59 sham calibration model roots, run the EvalMath independent-root power/variation preflight and preregister at least 32 separate confirmation roots. The original NC-05 through NC-08, PD-15 through PD-17 and other protected confirmation cohorts remain unmodified.
