# SUB-06 theory: wrong-clock source fitting, quotient cocycles and angular margins

## 1. Time-indexed quotient criterion

For a shared differentiable physical transition \`F_x:H→H\` and orthogonal rank-r projectors \`P_t\`, a deterministic clock-indexed quotient \`G_(t,x)\` exists on a fiber-compatible domain precisely when \`P_(t+1)F_x\` is constant along every \`P_t\` fiber. The local cross-block Jacobian \`P_(t+1)D_h F_x(h_t)(I-P_t)\` measures a first-order necessary condition. For a finite null-fiber perturbation \`δ∈ker(P_t)\`, the exact identity is

\`P_(t+1)[F_x(h+δ)-F_x(h)] = ∫₀¹ P_(t+1)D_hF_x(h+sδ)(I-P_t)δ ds\`.

Exact zero coupling over all admissible line segments is sufficient for exact fiber constancy. The original source model is the same shared \`F_x\` at all passes; the observable clock enters through P_t and G_t.

## 2. Matched source-fitting null and inference

Let \`D_t(x)=D_hF_x(h_t(x))\`, t=1,2,3 and \`C_t=E[g_tg_t^T]\`, t=1..4, with \`g_t\` the derivative of the terminal logit with respect to h_t. A bundle P minimizes the source-only registered objective

\`L_(D,C)(P) = A_(D)(P)/a0 + .35 B_(C)(P)/b0 + .50 S(P)\`,

where A sums \`||P_(t+1)D_t(I-P_t)||_F²\`, B is passwise output loss, S is the adjacent-projector smoothness regularizer. SUB-05 generated controls by permuting *after* minimizing the correctly ordered objective. SUB-06 independently minimizes source objectives \`L_(D∘τ_j,C∘π_j)\` for seven fixed wrong schedules. Each arm uses the same discovered matrices, stationary initial basis, 120 Adam/QR steps, optimizer seed/configuration and fixed a0/b0, with no finite-dose source selection based on evaluation responses.

The wrong schedules preserve every observed D_t(x) and C_t exactly once within their respective arrays. Their mislabelling intentionally destroys the correct time assignment, and D/C permutations are not required to form a physically realizable sequence. This is a **matched optimization budget control** rather than an exchangeable randomization test. It does not prove that source fits would behave identically under the true clock.

For new independently trained source i, let C_i(P) be the original-source finite-dose next-state RMS. The primary predeclared comparison is

\`d_i = min(C_i(static), C_i(wrong_0),…,C_i(wrong_6)) − C_i(aligned)\`.

All eight candidates are source-fitted before heldout inspection. Taking this within-root minimum imposes the strongest null after all candidates are scored, a fixed conservative model-selection penalty. Each trained source model is one independent statistical unit, including all seven fitted alternatives and every control.

## 3. Quantifying approximate angular robustness

For two rank-r projector bundles P and Q and any physical Jacobian D, the cross-blocks satisfy

\`||P_(t+1)D(I−P_t) − Q_(t+1)D(I−Q_t)||_F ≤ ||D||_F (||P_(t+1)−Q_(t+1)||_op + ||P_t−Q_t||_op)\`.

Proof: add and subtract \`Q_(t+1)D(I−P_t)\` and use operator/Frobenius submultiplicativity. Thus a small principal-angle change controls the *linearized* cross-block. For orthogonal projections of equal rank, \`||P−Q||_op\` equals the sine of the largest principal angle between their ranges.

Finite-dose coupling also changes the isotropic impulse law. For a fixed raw Gaussian z, write v_P=(I−P)z and δ_P=α v_P/||v_P||. If both projected norms are bounded below by γ>0 for this z,

\`||δ_P−δ_Q||₂ ≤ 2α ||P−Q||_op ||z||₂ / γ\`.

For the tied tanh transition \`F_x(h)=tanh(Wh+Ux+b)\`, global Lipschitz L≤||W||_op. Applying the preceding bound to the finite-dose next-state terms yields

\`||P_(t+1)ΔF(δ_P) − Q_(t+1)ΔF(δ_Q)||₂ ≤ αL ||P_(t+1)−Q_(t+1)||_op + (2α L ||z||₂/γ)||P_t−Q_t||_op\`.

The γ condition matters: normalizing a nearly vanished projection can amplify tiny angular differences. The bound is finite-sample, state-specific and preserves an explicit dependence on ||W||op and the normalized impulse geometry. It is not a global uniqueness theorem for learned causal subspaces.

## 4. Exact algebraic reference and source-fitted interpretations

In a synthetic linear 12D recurrence with three four-coordinate blocks cyclically permuted by W, a moving rank-eight projection P_t=W^(t−1) P1 W^T(t−1) obeys \`P_(t+1)W(I−P_t)=0\`. Stationary P1 can have nonzero coupling, and wrong-time assignments can change the block overlap. This is an exact finite-dimensional witness for time-dependent quotient closure; it does not establish that a nonlinear trained model identifies a unique moving bundle.

A positive SUB-06 finite-dose root advantage, after source-fitted wrong-clock controls, would support the relevance of **correct time assignment** to synthetic model causal abstraction under this frozen objective and intervention law. It would still require independent 59+ root sham calibration, EvalMath power/variation preflight and a sealed confirmation cohort for scientific generalization. Matching the optimizer budget removes the obvious training-exposure asymmetry of SUB-05, while alternative objectives, tied-model gauge freedom and source design choices remain scientific limitations.
