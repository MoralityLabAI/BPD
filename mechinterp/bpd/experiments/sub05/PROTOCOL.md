# BPD-SUB-05: prospective clock-indexed causal quotient bundles

**Registration first: November 18, 2026.** This protocol is committed before SUB-05 implementation, source-only tests, training, or evaluation of any new SUB-05 model. **Phase:** 8-root synthetic engineering development, with independent sham calibration and prospective confirmation SEALED. Predecessor SUB-04 remains an immutable registered engineering FAIL (+0.018058 state RMS versus +0.020 minimum); SUB-03 strict science preflight remains negative.

## Scientific question

A weight-tied recurrent transition F_x can encounter different reachable-state distributions at each loop. Does allowing a *pass-specific* causal quotient q_t(h)=P_t h, rank(P_t)=8, improve finite-dose **next-state** causal closure over the frozen SUB-04 stationary joint projector, while preserving clean task behavior? The stronger claim that temporal *alignment* matters must survive the **best of seven fixed time-permuted versions of precisely the same four learned projectors**.

For t=1,2,3, define on the original trained source:

  C_t(P_t,P_(t+1))² = E_x ||P_(t+1)[F_x(h_t+δ_(t,P_t))-F_x(h_t)]||²,

where δ_t is uniform on the physical radius-α sphere in ker(P_t), drawn through projection of the same preseeded raw Gaussian vector for all candidates. Define C(bundle)=sqrt((sum_(t=1)^3 C_t²)/3), units **hidden-state Euclidean RMS**. For a fixed *stationary* projector P, compare P1=P2=P3=P4=P. This is a sampled causal quotient defect, not an exact global statement about all states.

**Single primary preregistered contrast per independent trained root:**

  Δ_i = min{C_i(P_static^4), C_i(π_1 bundle), ..., C_i(π_7 bundle)}
        − C_i(aligned bundle).

The fixed min is a **strongest-negative-control** statistic and a direct test of temporal alignment beyond parameter count. A positive root must improve over the stationary SUB-04 fitted benchmark **and every registered temporal permutation**. Retain C_static−C_aligned, median C_perm−C_aligned and individual perm values as descriptive secondary outcomes; none can rescue primary failure. No selection of permutation after outcomes other than the frozen min.

## Models, source, independence and splits

- **Eight new independent engineering model roots 1600..1607**, and **excluded smoke seed 1599**. No other roots can be trained with this runner. Disjoint from SUB-01/02/03/04 roots (1200..1507), existing BPD protected tests, and any future SUB-05 calibration or confirmation cohort.
- Reuse the original SUB-01 \`Looper\` (12D, 4 shared-tanh passes, four Gaussian input features) and the same deterministic source model training contract as SUB-04: external y=[.85 x0 + .6 x1 + .35 x2 x3 >= 0], 2048 training examples, Adam lr 0.01, minibatch 128, exactly 256 updates, deterministic seeds for parameters/minibatches, one source model per root. Keep original imported source files byte-for-byte unchanged.
- Per root, split 2 = **512 discovery states**, used solely to fit four projectors and the stationary baseline, their local controls and α. Split 3 = 512 **physical-dose and geometric calibration** states with no finite-dose outcome queries and no rank/arm selection. Split 4 = 1024 heldout engineering states with independently defined task labels; scored exactly once after the projector family is frozen.
- Independent unit is the trained model root. Three passes, 7 temporal permutations, eight angular controls, source inputs, repeated gates and multiple readouts are repeated *within* each root.
- A model root is eligible only if separately >=0.80 clean external-label accuracy on x3>=0 and x3<0; full-rank model parity, rank/dose/gauge checks and complete precommitted control roster pass. All eight attempted roots, including errors, remain in the denominator; no re-training or seed replacement.

## Fixed 4-projector source-only optimizer

First fit the **unchanged stationary SUB-04 joint projector** P_static on the new root's split-2 discovery inputs, using its fixed **80 Adam steps at lr 0.02**, source loss A(P)/A(P_obs)+0.5 B(P)/B(P_obs), exactly as frozen. This baseline is not fitted on SUB-04 historical roots and receives the same source inputs, hidden width and physical impulse law.

For each t=1,2,3,4 define C_t^out=E_x[g_t g_tᵀ] on the discovery split, where g_t=∂(source terminal logit)/∂h_t of the original trained source, computed with frozen source weights. With D_t(x)=diag(1−h_(t+1)(x)²) W from the original physical transition, let for the bundle P1..P4:

  A(P_1..P_4)=1/(3N) Σ_(x,t=1..3) ||P_(t+1)D_t(x)(I−P_t)||²_F,
  B(P_1..P_4)=1/4 Σ_(t=1..4) tr((I−P_t) C_t^out),
  S(P_1..P_4)=1/(3⋅2r) Σ_(t=1..3) ||P_(t+1)−P_t||²_F.

Fix a0=max(A(P_static^4),1e−12) and b0=max(B(P_static^4),1e−12) from discovery only. Optimize:

  L(P1..P4) = A/a0 + 0.35 B/b0 + 0.50 S

on four rank-eight Grassmann factors. Parameterize each P_t=Q_t Q_tᵀ where Q_t=qr(M_t).Q, initialized **identically** from the frozen P_static float64 orthonormal basis. This isolates any learned depth variation from the stationary solution and handles rank-deficient terminal-readout Gramian eigenvectors without arbitrary initialization. Execute **exactly 120 Adam updates at lr=0.02, float64 CPU**, no early stopping, no tuning/fallback, no finite-dose heldout outcome during optimization. Freeze P_t afterward. Record full 120-step source objective trace, final A, B, S, per-pass projector idempotence/symmetry/trace and distances/angles from stationary and adjacent passes. Any nonfinite objective/gradient invalidates the root.

The pass-specific future-logit covariance is a *source-only constraint* on the bundle and differs from the terminal-only B of SUB-04. This experiment tests whether that extra temporal freedom helps; the source model weights W remain physically tied.

## Frozen controls, physical interventions, task fidelity

- Physical impulse norm α = **0.25 sqrt(mean_discovery ||h1||²)**. Draw independent N(0,I) vectors for each root, split and pass t with the original SUB-04 fixed raw-direction generator. Reuse each actual raw z_(x,t) for **all** projectors, including stationary, aligned and every sham; δ=α(I−P_t)z/||(I−P_t)z||. Max norm error and retained-projection norm <=5e−6 per example per pass. Record pre/post next-state norm and actual terminal physical hidden displacements.
- Seven predetermined **same-parameter time permutations**, using zero-based order tuples: (1,2,3,0), (2,3,0,1), (3,0,1,2), (1,0,3,2), (2,1,0,3), (0,2,1,3), (3,2,1,0). Every sham uses the exact same four trained projectors and optimization cost. All seven are scored and archived even if one unexpectedly performs better than the registered bundle. The primary null is the *best* (lowest defect) permutation and the stationary benchmark.
- Eight independently seeded **rank/dose matched angle controls**: four local bundle-wise rotations θ={.20,.40,.60,.80}, mixing four weaker future-output directions inside P_t into ker(P_t) at each t, and four global Haar conjugacies Q P_t Qᵀ using the same Q for all four passes. The observed source W and physical coordinate basis stay fixed. Score every control on all three interventions, preserving its source-only construction checks.
- Full rank P_t=I_12 gives exact zero null-fiber cross-block Jacobian as a structural **algebraic** reference, with no attempt to normalize an empty null space. Orthogonal physical-coordinate gauge Q must preserve all three source-causal projected next-state responses and terminal logits under correctly transported P_t' and δ_t'; require max state/logit error <=5e−5. Wrong untransported interventions must change logits with RMS>1e−5. Test on fixed first 64 heldout inputs, source-only and before aggregate admission.
- For each candidate and control, actually rerun F_x from h_t+δ through remaining passes (do not reuse clean downstream states). Record pooled/per-pass projected next-state RMS, terminal-logit RMS, correct-to-incorrect corruptions, incorrect-to-correct repairs, actual impulse doses, state radii and terminal state displacement.
- **Clean autonomous clocked student:** start h0=0, then after each transition apply the corresponding P_t in physical space (at t=1..4); final source head reads projected h4. Report Bernoulli KL vs clean teacher, external-label NLL, true correctness, source clean NLL, and full-rank P_t=I parity. A separate unchanged stationary student uses P_static on all passes.

## Frozen engineering gates and statistical contract

All eight roots must meet source competence, 4×rank8 geometry, all seven permutations, all eight angular controls, physical dose/gauge closure and source optimizer numerical fidelity.

The *single primary* signed root effect Δ_i must satisfy:
- 8/8 independent models admitted, >=7/8 roots Δ_i>0;
- mean Δ>=**+0.015 Euclidean hidden-state RMS**;
- 2000-draw 95% whole-model-root percentile bootstrap, seed **160505**, lower endpoint >0.

Additionally require all predeclared clean/recurrent fidelity gates on root means: aligned clocked-student mean clean KL <= **0.040 nats**, mean aligned-minus-stationary clean KL <= **+0.015 nats**, mean aligned-minus-stationary true-label NLL <= **+0.030 nats**, and mean aligned-minus-stationary pooled terminal-logit impulse RMS <= **+0.040 logits RMS**. Preserve full-head parity <=5e−6. The observational source-only objective itself is not a heldout selection gate. Return \`ENGINEERING_SIGNAL_ONLY\` only if *every* gate passes; otherwise \`ENGINEERING_NULL_OR_INELIGIBLE\`.

Secondary descriptions are individual C_t, C(P_static), all time-permutation C values, direct near-angle C, loss trajectories, causal fiber distance and clean task repair/corruption. No post-hoc winning competitor or alternative effect threshold rescues the original verdict.

## EvalMath and confirmation seals

After freezing original receipts, run strict source-pinned \`MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a\` preflight on root effects and fixed time-permutation sham effects, one row/model root, using bounded b(s)=s/(1+|s|). Set alpha=.05, \`scientific_min_effect="0.014778325123152709"\` (b(.015)), \`target_effect="0.038461538461538464"\` (b(.04)), measurement_resolution=.002, comparisons=1, planned confirmation capacity 64, pilot development-only/independent/heldout-confirmation sealed. With eight development roots, the independent 95/95 sham tail cannot be calibrated; retain \`REDESIGN_OR_PILOT_ONLY\` and exact reasons with strict exit code 2. **No SUB-05 scientific calibration or confirmation models are generated in this PR**, even if the engineering pilot passes. A future 59+ independent-root sham calibration and a fresh, separately registered powered confirmation set are prerequisites.

## Interpretive boundary

Exact time-dependent quotient closure on a convex fiber domain would require \`P_(t+1)D F_x(h+sδ)(I-P_t)=0\` along all fiber segments; numerical finite-dose development samples can only provide evidence at the chosen state/impulse distribution. A clock-indexed quotient is mathematically a nonautonomous representation of a tied recurrent system, and preserving its synthetic causal transition says nothing on its own about physical substrate independence, biological replacement, phenomenal consciousness or continuity of personal identity.
