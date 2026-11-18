# BPD-SUB-04: registered multipass transition-quotient causal closure

**Protocol-first registration: 2026-11-18, prior to any SUB-04 source implementation, model training or heldout evaluation.** Separate [SUB-03 negative scientific preflight](../sub03/RESULTS_20261118.md) remains intact. This experiment is an eight-model *engineering development* study only. The 59-root sham calibration, strict confirmation release and 64-model confirmation cohort are separate future decisions.

## Research question and mathematical endpoint

SUB-02 reduced final-logit response to null-fiber interventions while leaving a projected next-step state commutator near 0.108 hidden-state RMS. SUB-03 found that 0.20-radian nearby Gramian rotations kept nearly all of the terminal-output advantage. Can a joint rank-eight subspace improve **finite-dose projected state-transition closure** across post-pass-one, post-pass-two and post-pass-three interventions while keeping the terminal task readout useful?

Let F_x(h)=tanh(Wh+Ux+b), with tied W∈R^(12×12). Let P be a rank-eight orthogonal projector in the original physical hidden-state coordinates. On clean h_t(x), draw isotropic physical impulse δ_{t,x}∈ker(P) with ||δ||_2=α. Define

  C_t(P) = sqrt(E_x ||P [F_x(h_t+δ_{t,x}) − F_x(h_t)]||_2²),

and the preregistered pooled *state-closure* endpoint

  C(P) = sqrt((C_1(P)²+C_2(P)²+C_3(P)²)/3).

The **primary signed paired model-root effect** in units of hidden-state Euclidean RMS is

  Δ_i = C_i(P_obs) − C_i(P_joint).

Positive Δ means that the joint subspace allows a smaller observable next-state change from interventions in its discarded directions, using equal physical impulse L2 dose and the same fixed raw Gaussian directions for every candidate. Output-logit response from continuing the *unaltered source model* after each impulse, clean projected-surrogate KL and independent external task-label NLL are separately recorded. Pooled state closure tests the finite counterfactual domain sampled here. A global exact quotient would require uniform fiber constancy and is outside the evidence.

## Independent roots, partitions and source

- **Exactly eight independently trained engineering roots 1500..1507**, excluded smoke root 1499. Reject other roots. All seeds are disjoint from SUB-01 1200..1207, SUB-02 1300..1307, SUB-03 1400..1463 and protected ARG-03-N / AMM / CQ-08 data.
- Keep the SUB-01 Looper and SUB-02 training/data law, already source-locked: width 12, input width 4, four tied tanh passes, iid standard-Gaussian x, independent binary task y=[.85x0+.6x1+.35x2x3≥0]. Adam lr .01, 256 minibatch-128 updates on 2048 train examples. The source implementation, original training batch seed law and original model weight initialization are unchanged.
- Per root, split 2: 512 **discovery-only** examples for PCA, Gramian and optimization; split 3: 512 **source-only selection/physical-dose checking** examples, with no intervention scoring or rank selection; split 4: 1024 **heldout development** inputs, scored once after freezing each root's candidate projectors and all eight sham orientations. Split-4 labels never train a projector.
- Model trained once per root, with source/experiment SHA-256, seed, data sizes, Python/Torch/Numpy versions, projectors, source-only diagnostics, per-pass physical dose, complete sham family and all heldout responses in retained JSON receipts. Future confirmation sets are ungenerated.

## Source-only candidate projectors

**Rank fixed to r=8** for all candidates; no adaptive rank choice. Keep source model weights frozen when calculating projectors.

1. \`P_pca\`: unchanged SUB-01 top-eight eigenspace of the float64 **uncentered pooled clean-state second moment** over all four passes from discovery x.
2. \`P_obs\`: unchanged SUB-02 top-eight eigenspace of \`C_out=mean_x j_x j_x^T\`, where \`j_x=dR(F_x^3(h1))/dh1\` and the model is frozen; eigendecompose in float64.
3. \`P_joint\`: differentiable source-only rank-eight Grassmann optimization initialized at the **top-eight eigenvectors of C_out**, ordered descending by eigenvalue, with a fixed objective to balance *linearized next-state closure* and terminal output.

For each discovery x and t=1,2,3, define the physical-state step Jacobian \`D_t(x)=diag(1-h_(t+1)(x)^2)W\`. Let

  A(P) = (1/(3N)) Σ_(x,t) ||P D_t(x) (I-P)||_F²,
  B(P) = tr((I-P) C_out).

Fix \`a0=A(P_obs)\` and \`b0=B(P_obs)\` from **discovery only**, each clamped below at 1e-12 for normalization. Optimize

  L(P) = A(P)/a0 + 0.5 B(P)/b0

over orthogonal rank-eight P=QQᵀ. Parameterize Q=qr(M).Q where M starts at the original float64 Gramian eigenvector basis. Run **exactly 80 Adam steps at lr=0.02, float64 on CPU**, no early stopping, no search over steps/learning rates/loss weights, no outcome-based fallback. Set \`P_joint\` to the final QR state. Report initial/final A, B, L, all-step source trace, projector idempotence/symmetry/trace, and the principal-angle distance to P_obs. Numeric instability or nonfinite gradients makes the root ineligible, with failure preserved.

This joint optimization minimizes a *linearized discovery objective*, not the finite-dose heldout C(P). The terminal-output part is a regularizer; optimizing A(P) alone can make the projected readout useless. The objective does not guarantee a uniform quotient or a complete physical mechanism.

## Heldout interventions and controls

- Amplitude is fixed **α=0.25 sqrt(mean_discovery ||h1||²)**, a constant physical Euclidean hidden-state L2 impulse norm across all tested passes and candidates. Independently seed raw \`z_(x,t)~N(0,I_12)\` for each pass, root and split; use **the same z** for P_joint, P_obs, P_pca and all eight sham projectors. Define \`δ=α (I-P)z/||(I-P)z||\`. Source step uses this δ only once after clean h_t at t=1,2,3.
- Score the next-state projected difference \`P(F_x(h_t+δ)-F_x(h_t))\` at every pass. Continue source rollouts from altered h_t+δ to terminal h4 and record the final-logit response on that pass. Record the clean and perturbed next-state norms, actual δ physical L2 norms and projection residuals; block if max | ||δ||−α | >5e-6 or max ||Pδ|| >5e-6.
- For task fidelity, execute the **autonomous projected student** of SUB-01 at P_joint, P_obs and P_pca; report source-versus-projector clean Bernoulli KL, actual external-label task NLL and correct-count/accuracy, with source clean true-label NLL. These remain separate from direct source-model intervention effects.
- Four local, **exact rank and physical dose matched** rotations of P_joint at theta \`[0.20,0.40,0.60,0.80]\` rad, preserving the four most output-observable directions *within* P_joint and mixing the other four with a seeded Haar basis of its orthogonal complement. Four additional globally Haar-rotated rank-eight P_joint shams. Construct each by \`P'=QP_joint Q^T\` with Q orthogonal; construction preserves projector eigenvalues exactly (eight ones, four zeros). Assign the spectrum of C_out to an **artificial orthogonal-frame matrix** whose top-eight eigenspace is P_joint and conjugate it by Q to check exact source-eigenvalue law; this is a numerical control and is **not** the unchanged source Gramian. Each sham gets the same raw z and amplitude for all three passes. Shams are fixed from discovery geometry and predetermined seeds and cannot be selected by outcome.
- Exact identity/full-rank positive control \`P=I_12\` gives algebraically zero null-space Jacobian \`P D (I-P)=0\`. Orthogonal gauge positive control: \`F_Q(z)=QF(Q^Tz)\`, \`R_Q(z)=R(Q^Tz)\`; clean and correctly transported impulse state/logit differences max ≤5e-5. Intentionally **untransported** impulse under Q must yield response RMS >1e-5. Source task competence ≥.80 in each x3-sign context; eigenspace/symmetry/idempotence ≤1e-5 and trace error ≤1e-4; all eight shams rank/dose admitted.

## Frozen engineering decision

Primary comparison family **one**: fixed rank-eight \`C(P_obs)-C(P_joint)\`, pooled over three predeclared passes. Uncertainties and sign counts use **eight independently trained model roots**, not 8×1024 examples, 3 passes or the sham arms.

Engineering signal only if all eight roots meet competence, numerical, sham and dose/gauge admissions, **mean Δ ≥ +0.020 Euclidean hidden-state RMS**, at least **7/8 roots Δ>0**, and a 2000-draw whole-root percentile bootstrap (seed 150404) has 95% lower endpoint >0. Additionally preserve output: mean source-vs-joint clean Bernoulli KL ≤ 0.040 nats; mean joint minus obs clean KL ≤ +0.015 nats; mean joint minus obs true-label NLL ≤ +0.030 nats; mean terminal-logit impulse RMS joint ≤ obs +0.040 over the three passes. All four mean fidelity gates are part of the registered engineering decision. Even if passed, the verdict is \`ENGINEERING_SIGNAL_ONLY\`. Otherwise, \`ENGINEERING_NULL_OR_INELIGIBLE\`.

Secondary descriptive outcomes: per-pass C_t and terminal-logit RMS, task correct/corruption, same-dose sham C, the **direct** near-rotation \`C(P_θ=.20)-C(P_joint)\` and 0.40/0.60/0.80 angular responses; source-only A(P), B(P), spectral projection distances, and global Lipschitz upper bound \`||W||op\`. No secondary arm rescues the primary. Any inferential extension across the secondary comparisons requires a new multiplicity declaration and heldout units.

## EvalMath and release boundary

This 8-root engineering study has **no independent 59-root sham-tail calibration**. Once original results are archived, run the pinned EvalMath v1 \`preflight.py\` from \`MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a\` on the eight root-level primary effects and same-root nulls in normalized bounded units \`b(s)=s/(1+|s|)\` with minimum \`b(.020)\`, planning target \`b(.050)\`, \`m=1\`, planned future confirmation capacity 64 and flags for independent development/untouched heldout confirmation. The **only allowed scientific phase verdict is pilot-only**, because 8 roots cannot calibrate a nonparametric 95/95 sham bound. Preserve strict CLI exit 2 and every reason; run its source-only structural checks before any development model.

No SUB-04 confirmation or 59-root calibration population is generated by this PR. Further studies require separately registered independent cohorts, with sealed final readouts. This is a synthetic recurrent computational experiment only; biological substrate identity, consciousness and the physics of synthetic mind replacement are outside its evidence.
