# SUB-04 theory: finite-dose closure, future observability and joint invariant subspaces

## The quotient criterion

For state transition F_x:H→H and a map q:H→Z, a deterministic quotient G_x exists on a specified invariant state domain D precisely when q∘F_x is constant over every q-fiber intersecting D, for each allowed input x. For an orthogonal rank-r projector P, the fiber has the form h+ker(P). A nonzero measured \`P[F_x(h+δ)-F_x(h)]\` with Pδ=0 is a concrete counterexample to exact closure on any domain containing both states.

SUB-04 measures this finite-distance difference *inside the original trained model*, after perturbations at passes 1–3. Its primary C(P) is pooled Euclidean state RMS, and a smaller C(P) is evidence of better **sampled finite-dose quotient closure** only.

## Linearized causal subspace objective

For one recurrent step \`F_x(h)=tanh(W h+Ux+b)\`, the exact Jacobian is

  \`D_hF_x(h)=diag(1−tanh²(W h+Ux+b)) W\`.

If δ is uniformly distributed on the radius-α sphere in ker(P), then

  \`E[δδ^T|x]=α²/(d-r)(I-P)\`

and a first-order approximation gives

  \`E[||P D_hF_x(h) δ||² | x]=α²/(d-r)||P D_hF_x(h)(I-P)||_F²\`.

Averaging the three clean recurrent states yields source quantity A(P). It is a **two-sided coupling criterion**: changes in the discarded coordinates cross into the retained coordinates after one step. Minimizing \`tr((I-P) C_out)\` alone addresses terminal readout sensitivity and does not minimize this two-sided term.

The registered joint objective \`A(P)/A(P_obs)+0.5 tr((I-P)C_out)/B(P_obs)\` explicitly trades output preservation against cross-block step coupling. Because \`P D (I-P)\` is nonlinear in P, its global minimum is not generally the leading eigenspace of a fixed Gramian. The 80-step QR/Adam construction is a source-only candidate search, not an algebraic certificate of global optimality. Rotating P through the Grassmann manifold changes both which impulses are considered null and what output coordinates are retained, so all empirical comparisons are conditioned on the registered equal-dose isotropic impulse distribution.

## Finite perturbation guarantee and its boundary

For an everywhere differentiable step F_x and δ∈ker(P):

  \`P[F_x(h+δ)-F_x(h)] = ∫₀¹ P D_hF_x(h+sδ)(I-P) δ ds\`.

Consequently,

  \`||P[F_x(h+δ)-F_x(h)]||₂ ≤ ||δ||₂ sup_(0≤s≤1)||P D_hF_x(h+sδ)(I-P)||op\`.

For the tied tanh transition, \`||D_hF_x(h)||op ≤ ||W||op\` globally, hence \`||PΔF|| ≤ α||W||op\`. SUB-04 reports this conservative global operator-norm envelope alongside measured finite-difference closure; a small sampled C(P) cannot by itself turn the local source objective into a uniform guarantee.

For a proposed projected surrogate \`G_x(z)=P F_x(z)\` acting in range(P), suppose along a specified reachable and counterfactual domain we have representative defect \`||P F_x(h)-G_x(P h)||≤ε\`, and G_x is L-Lipschitz there. Recurrence of the projected error yields \`e_k ≤ ε Σ_(j=0)^(k-1) L^j\` from matched initial projections. In general, an empirical finite-dose impulse test at one α does not establish this uniform ε because the source state may have discarded coordinates of a different magnitude. The clean surrogate KL and task NLL directly expose this additional gap.

## Per-pass output behavior and independent task outcomes

The after-pass-t intervention changes h_t and reruns the **source** for 4−t subsequent steps. Its terminal-logit RMS is evidence about causal readout sensitivity from the perturbed state. The clean projected surrogate \`h'_(t+1)=P F_x(h'_t)\` uses its own recurrent hidden state, and its Bernoulli KL and true-label NLL measure clean-task fidelity. These readouts need not agree: a projected state can preserve logits under off-manifold impulses while failing to reconstruct the source's autonomous clean trajectory.

The developmental external task labels train the source and evaluate heldout true correctness. They are excluded from projector discovery and optimization. Per-pass corrupted-correct and repaired-wrong counts are recorded to prevent treating generic logit disruption as reasoning improvement.

## Angular controls and preservation claims

SUB-03's nearest 0.2-radian rotated Gramian was a strong endpoint-only control. SUB-04 uses the same fixed angle family, applied to a joint projector ordered internally by future-output importance. Every local and global comparator is an orthogonal conjugate of the rank-eight P_joint; therefore its projector spectrum contains exactly eight ones and four zeros. A synthetic symmetric operator with C_out's eigenvalues and P_joint's eigenframe is conjugated as a separate numerical isospectral witness. These constructions do not preserve the orientation of the original source C_out. The direct near-control \`C(P_θ=.2)−C(P_joint)\` is descriptive and kept separate from the primary joint-vs-obs engineering decision.

## Statistical scope

The original eight independent trained roots are the only valid independent units for the registered development gates. Source prompts, loop steps, eight angle controls, loss iterations and two readouts are within-root repeated measurements. The minimum +0.020 hidden-state RMS improvement is fixed in the protocol. An engineering PASS still requires separate independent-sham-tail calibration, prospective EvalMath resolution, and new untouched model roots before scientific confirmation.

This experiment studies synthetic computational mechanisms. Preserving a projected recurrent transition on a sampled finite domain does not imply preservation of subjective experience or physical substrate independence in a biological replacement.
