# SUB-01 theory: what it means to preserve a recurrent mechanism

## 1. Exact dynamical quotient

Let \`F_x:H→H\` be a time-homogeneous recurrent transition and let \`q:H→Z\` be a proposed coarse-graining. There exists a well-defined deterministic quotient transition \`G_x:Z→Z\` with

\`q∘F_x = G_x∘q\`

on a chosen invariant domain D **if and only if** \`qF_x\` is constant on every nonempty fiber \`q^{-1}(z)∩D\`, for every allowed input x. Necessity follows because equal q states must have equal G values. For sufficiency define \`G_x(z)=qF_x(h)\` for any h in the fiber; constancy makes this single-valued. Invariance of D under the transition is required for repeated application.

For a rank-r orthogonal projection \`P=UU^T\` and quotient coordinate \`q(h)=U^Th\`, a practical restricted test samples states and paired perturbations \`delta∈ker(P)\`. The one-step causal fiber defect is

\`d_r(h,x,delta)=||P[F_x(h+delta)-F_x(h)]||_2\`.

A nonzero defect is a constructive obstruction to an exact quotient on a domain containing this pair. A zero empirical mean or finite-sample maximum supplies only an observation about sampled pairs; it is not a global mathematical proof of exact quotient closure.

The specific surrogate in SUB-01, \`G_r(z,x)=U^TF_x(Uz)\`, takes the zero-fiber representative \`Uz\`. Therefore even if clean predictions match, the true projected dynamics can depend on discarded state coordinates. The complement's hidden influence is the object to measure.

## 2. End-to-end behavior and intervention preservation

Let T=4, h0=0, terminal task-logit R(hT). There are several logically distinct quantities.

- **Clean-task behavior:** external-label accuracy/NLL for R(hT) and R(UzT).
- **Behavioral distillation:** KL between the source and surrogate's terminal Bernoulli distributions, which can be small because the output has low sensitivity.
- **Counterfactual transition equivalence:** interventions \`h→h+delta\` with \`q(delta)=0\` should leave subsequent q states unchanged if the quotient is exact on the intervention domain.
- **Hidden-to-output effect:** source logit shift \`R(F_x^3(h1+delta))-R(F_x^3(h1))\` for equal-q h1 states, after injecting delta post-first-pass. A nonzero value shows that the proposed quotient discards an input-to-output causal pathway on that domain.

The null-fiber directions are calibrated to exactly equal physical Euclidean L2 impulse norm per example. Eight random subspaces of the same dimension serve as dose-matched shams. Since the reference state trajectories are created by a trained synthetic model, this is an operational state-space dose; it makes no claim about energy expenditure or actual physical hardware perturbation.

A causal compactification frontier can be defined relative to a declared intervention distribution I and state domain D:

\`C_(eps_b,eps_c)(F;D,I)=min_r{r: KL_clean(r)<=eps_b and Effect_fiber(r)<=eps_c}\`.

SUB-01 takes the minimum over ranks 2,4,6,8,10 according to its preregistered *selection-only* gates. Since the source intervention distribution changes with rank through ker(P_r), this frontier is protocol-relative. Monotonicity in r is not guaranteed for observed terminal causal effect, despite the nested algebraic projectors.

## 3. Exact invertible transport

For an orthogonal Q, set \`z=Qh\`. Define \`F'_x(z)=QF_x(Q^Tz)\` and \`R'(z)=R(Q^Tz)\`. For any source impulse I_delta(h)=h+delta, define I'_delta(z)=z+Qdelta. Then

\`QF_x(I_delta(h))=F'_x(I'_delta(Qh))\`.

Induction over recurrent passes gives identical complete trajectories in source coordinates and equal final outputs (up to floating-point rounding). The test compares clean parity, intervention parity and every intermediate state. Misapplying the *untransported* vector delta in z coordinates is a designated negative control: the clean outputs remain identical while the causal intervention correspondence fails.

This is a coordinate reparameterization of one computational mechanism. It provides an exact algebraic positive control for the measurement, not evidence that distinct physical substrates realize the same consciousness.

## 4. Approximate closure and error bounds

If a candidate quotient G is L-Lipschitz and the per-step defect obeys

\`||qF_x(h)-G_x(qh)|| ≤ eps\`

uniformly on all reachable and counterfactual states used by both rollouts, then with matched initial q states the projected trajectory discrepancy after k passes is at most

\`eps (1+L+...+L^(k-1))\`.

For a state readout \`Rbar\` that is K-Lipschitz and a source readout-factorization error at most \`rho\`,

\`|R(hk)-Rbar(zk)| ≤ rho + K eps sum_{i=0}^{k-1} L^i\`.

Both assumptions matter. An output may depend directly on a discarded coordinate even where the transition quotient closes. Conversely the output can be insensitive despite a large state commutator. SUB-01 measures terminal behavior and next-step projected-state defect separately for this reason.

## 5. Connection to physical substrate independence

Burger's distinction between computational and physical substrate independence motivates asking which physical constraints must be retained. SUB-01 formalizes a narrower, tractable problem: invariance of a learned digital recurrence under coordinate transport and quotient maps. Exact gauge transport is an algebraic guarantee; imperfect quotient transfer is a model-specific empirical result. Material replacement, preservation of biological dynamics, subjective consciousness and personal continuity would require additional operational observables and experiments beyond this harness.

## 6. Evaluation and failure boundaries

The eight model roots 1200..1207 comprise engineering development. Every rank and sham shares its root; 1024 heldout input samples do not create 1024 independent replicates of the model mechanism. Complete uncertainty requires independent root resampling. Formal source-only tests establish mathematical properties of the implementation and planted toy counterexamples. Any claim of robust mechanism equivalence would further require a preregistered equivalence margin and independent-root confidence bounds contained within that margin.

The original 59-root independent sham-tail calibration requirement and strict EvalMath preflight are inherited from the BPD research agent instructions. Protected test partitions remain sealed.
