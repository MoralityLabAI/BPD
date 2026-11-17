# SF05-A2: post-Stage-A finite public-level-set correction

**Post-result engineering extension, November 18, 2026.** This document was drafted *after* inspecting [Stage A's failed finite public preservation](RESULTS_STAGE_A.md). It does not alter [Stage A's frozen code/thresholds](LOCK.json), and therefore **cannot be treated as Stage A preregistration or as fresh science**. A2 may use only the same previously excluded development roots 901 and 902. All intervention outcomes are explicitly exploratory.

## Why a new construction is necessary

Stage A identified a geometric failure: the Euclidean projection of a full physical donor write onto \`ker Dq∩ker D(q∘F_U)\` preserves first-order public silence but destroys much of the nonlinearly coordinated donor state. On root 901, the full physical write had public-output deviation 0.011 and 100% authorized recovery; tangent projection increased finite public deviation to 0.354 and reduced authorized recovery to 26.9%.

A2 solves the **finite** public level-set equation for the exact, *role-U future readout* \`G_U(h)=q(F(h,x_U))\` at each fixed recipient prequery state \`h\`. Starting from \`v_0\in\ker Dq\`, compute residual \`r_j=G_U(h+v_j)-G_U(h)\` and \`g_j=P_0 ∇G_U(h+v_j)\`. Form a constrained Newton step

\[
s_j=-\frac{r_j}{\langle g_j,g_j\rangle}g_j
\]

when \`||g_j||>10^{-9}\`. Restrict each step to both \`0.25 ||v_0||\` and absolute norm \`0.5\`. Among backtracking step sizes \`{1,1/2,1/4,1/8}\` accept only changes that reduce the **actual finite public-output residual**. Freeze 8 maximum iterations and target per-sample \`|r|≤10^{-4}\`. Preserve coordinate zero exactly on every step; log nonconverged and zero-gradient cases.

This uses an explicitly privileged public-output oracle and its gradient, evaluated on the intervened model state. It is a local nonlinear numerical correction, not a deployable privacy filter or a BPD-specific discovery.

## Exactly matched comparison

Replay the same SF04 models and 512 diagnostic pairs per excluded root. Compare:

- Selected 14/20 original singleton BPD factors (gauge-dependent);
- Full BPD physical write \`(φ_d-φ_r)VU\`, and gauge-stable BPD output-SVD rank 16;
- Frozen teacher physical \`C\` and exact dark donor-state difference as independent positive controls;
- Raw dark candidate, Stage-A tangent intersection candidate, and A2 finite-corrected **raw** candidate as separately reported arms.

Apply A2 correction to the *same-sample, same-initial-dose* random current-dark and signed-authorized-gradient-matched shams, with 16 fixed random streams per excluded root. The signed null may use the candidate's *raw dark* authorized tangent as a privileged predictor but must match dose and maintain current readout. Record the additional correction norm/dose and converged percentage alongside authorized donor-target recovery, finite public residual before/after, and public-output error on the authorized-eligible stratum.

The solver's construction forces finite public-output agreement *when converged*. Therefore **convergence by itself is an engineering result**, while preserved authorized recovery, comparable correction dose and robustness across independent task families are the scientific questions. Relative advantage to a same-algorithm null is descriptive on these already used roots.

## Engineering acceptance

The source-only analytic controls should pass 4+ unit tests, including finite linear and quadratic counterexamples. All finite corrections must preserve the actual current scalar readout exactly. Any converged example must satisfy the frozen \`10^{-4}\` public output tolerance, with finite correction norms and no concealed failure case. A2 run executes **only** roots 901 and 902; preserve the original raw Stage A receipt unmodified. No fixed positive recovery rate or within-root confidence claim is registered.

## Separate future experiment

Only after A2, write a fresh protocol that can independently test finite public-level-set correction on disjoint source-generation families and teacher roots, with a *nonprivileged* deployable mechanism arm separated from the oracle correction. The 59-root null calibration and 32-root sealed confirmation suggested in issue #42 remain future work. All ARG-03-N cases stay sealed.
