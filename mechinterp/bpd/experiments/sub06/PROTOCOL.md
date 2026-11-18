# BPD-SUB-06: source-fitted wrong-clock controls for time-indexed causal quotients

**Prospective registration, 2026-11-18.** This file is the first SUB-06 research commit before SUB-06 code or new training data. It responds to [issue #116](https://github.com/MoralityLabAI/BPD/issues/116). The original SUB-05 result (+0.015300 hidden-state RMS over best post-fit permutation and stationary; eight-root engineering PASS, science preflight NOT READY) remains untouched.

## Question and primary endpoint

Does correctly clock-labelled **source fitting** yield a causally better four-projector recurrent quotient than *independently source-fitted wrong-clock bundles* that receive the same architecture, initial subspace, optimizer steps and discovery states?

For each trained 12D four-pass tied-tanh model root, train a correctly indexed bundle \`P_1..P_4\` and **seven** wrong-clock source-fitted bundles. Score each in its *fitted native order* against the unchanged physical source transition sequence on heldout inputs. For any bundle K, define at t=1,2,3

\`C_t(K)=sqrt(E_x ||P_(t+1)[F_x(h_t+δ_t)-F_x(h_t)]||_2²)\`, where \`δ_t∈ker(P_t)\`, \`||δ_t||_2=α\`.

\`C(K)=sqrt((C_1²+C_2²+C_3²)/3)\`.

The single preregistered root effect in Euclidean state RMS is:

\`d_i=min(C_static, C_wrong[0],…,C_wrong[6])−C_correct\`.

The minimum is taken **within each independent root**. Positive means the correctly source-fitted clock bundle beats both the source-fitted wrong-clock family and the matched stationary SUB-04 joint projector. No evaluation of alternative clock candidates, comparison family or rank outside the frozen roster may rescue the result. Exact finite-source structural proofs have an algebraic evidence class separate from this learned-model engineering result.

## Frozen population, data and optimizer

- **Fresh 8-root engineering cohort:** independent source-model seeds 1700..1707. Excluded **smoke root 1699**, optionally trained for 16 updates only. Reject all other roots. These are disjoint from SUB-01..05, AMM, ARG-03-N, CQ-08 and prospective confirmation cohorts.
- Preserve **exact** SUB-01 source \`Looper\`: four shared tanh passes, physical hidden width 12, 4 Gaussian input features, binary external label \`y=1[0.85x0+0.6x1+0.35x2x3>=0]\`; Adam lr .01, 256 updates, minibatch 128, 2048 train inputs with frozen independent generator and weight-init schedule.
- Root splits: 512 source-only discovery (split 2), 512 **dose and numerical geometry checking only** (split 3), 1024 *development heldout* inputs and true labels (split 4). Read no heldout input or label until candidate subspaces, full 7-control family and shams are frozen. Only root/model is the independent unit.
- Fixed rank **8** for each of four pass-indexed projectors. Fit a fresh stationary SUB-04 joint P_static on each new root's discovery set with **the unchanged** 80 Adam/QR steps at lr .02. It supplies the common initial orthonormal basis and source-only normalization. Import prior SUB-01, SUB-04, SUB-05 source unchanged, verified by Git history.
- Discovery-only arrays: \`D_t(x)=diag(1-h_(t+1)(x)^2)W\` for t=1..3, and \`C_t=E_x[g_tg_tᵀ]\` for t=1..4 with \`g_t=∂(source terminal logit)/∂h_t\`. Use float64 and original model weights frozen.
- **Correct-clock arm:** use D in the original order [D1,D2,D3] and C in [C1..C4]. **Seven wrong-clock arms** permute the clock *for source fitting* using exactly the SUB-05 seven zero-based permutations:
  \`(1,2,3,0), (2,3,0,1), (3,0,1,2), (1,0,3,2), (2,1,0,3), (0,2,1,3), (3,2,1,0)\`.
  For permutation π, fit **fresh** \`Q_1..Q_4\` using \`C'_t=C_{π(t)}\` and \`D'_t=D_{κ_π(t)}\` where \`κ_π\` is the relative transition-map index for the first three transition positions. To avoid a pathologically undefined fourth transition and to preserve all physical Jacobian arrays *exactly once*, set \`κ_π(t) = π(t) mod 3\` for t=0..2 only if all three indices are distinct; for other π instead use a separately frozen **three-transition permutation** \`τ_π\` listed here:
  \`(1,2,0), (2,0,1), (0,1,2), (1,0,2), (2,1,0), (0,2,1), (2,1,0)\`.
  **Operational rule: always use τ_π as written**, even when the first-three π entries could produce another ordering. Every control receives exactly the same **multiset of three D arrays** and four C matrices as the aligned arm. D and C are different data types; their clock permutations need not describe a physically realizable transition chain. This is deliberately a falsifiable **mislabelled source schedule null**.
- Each bundle, aligned and wrong-clock, has four independent Q_t=QR(M_t) factors, initialized to the identical *stationary basis*, and receives exactly **120 Adam steps at lr=.02 on CPU float64**. Objective and normalization are exactly the SUB-05 design:
  \`A=mean_(x,t)||P_(t+1)D_t(x)(I-P_t)||_F²\`,
  \`B=(1/4)Σ_t tr((I-P_t)C_t)\`,
  \`S=(1/(3*2r))Σ_t||P_(t+1)−P_t||²_F\`,
  \`L=A/a0+0.35B/b0+0.50S\`.
  **a0 and b0 shared across all arms**, computed from static bundle and *correct-clock* D,C; A_static and B_static are invariant to the input order. No hyperparameter search, re-initialization, early stopping, or choosing source training trajectories by outcome. Failed gradients, bad QR, failed geometry or incompetence remain recorded.
- The wrong-clock model is evaluated with \`Q_t\` in its **native fitted order**, no restoration of the source permutation. It is scored on true source F_x and true t=1..3. The frozen comparator family is stationary plus all seven wrong-clock fit models. Compute all eight sets of model outputs independent of their performance.

## Equal-dose interventions, controls and admission

- Physical intervention dose \`α=.25 sqrt(E_discovery||h1||²)\`. For root, heldout split and pass t, use the **same** seeded Gaussian 12-vector \`z_(x,t)\` for every arm. Set \`δ=α(I−P_t)z/||(I−P_t)z||\`. Inject once after clean h_t, re-run the actual source on the changed state, compare *projected* next step with clean and archive terminal output response, correct-to-incorrect counts, incorrect-to-correct counts and physical state radius. Dose and null-projection errors <=5e−6.
- Fixed additional **descriptive controls**: the post-fit seven temporal permutations of the **correctly source-fitted bundle**, one rank-8 PCA or source-output-Gramian static reference if available without selection, plus rank-12 exact fiber-zero algebraic reference and orthogonal physically transported-gauge parity for first 64 heldout inputs. These descriptive arms do not enter the primary family, and if CPU cost limits apply the seven fitted wrong-clock and stationary primary arms retain priority.
- Admission: both heldout x3-sign context accuracies >=.80; full-rank clean-logit parity <=5e−6; all subspace rank trace errors <=1e−4, idempotence/symmetry <=1e−5; all eight fitted families completed and all predeclared controls scored; maximum transported next-step and logit gauge error <=5e−5, mis-transported intervention logit RMS >1e−5; physical dose tolerances pass. Actual heldout masks and all failed roots are serialized.
- **Clean clocked-student fidelity** using P_t after each step on source inputs: mean aligned clean-vs-source Bernoulli KL <=.040 nats; aligned minus static KL <= +.015 nats; aligned minus static *external true-label* NLL <= +.030 nats; aligned minus static source terminal-logit intervention RMS <= +.040. Full-rank student logit parity <=5e−6. Preserve all output accuracies and intervention repair/corruption.
- A matched source-selection advantage requires all **8/8** independent model roots eligible, **at least 7/8** positive primary d_i, mean **d_i >= +0.010 Euclidean hidden-state RMS**, and a 2000-draw percentile bootstrap over the 8 root effects (seed 170606) with lower 95% endpoint >0. Engineering verdict is \`ENGINEERING_SIGNAL_ONLY\` iff every admission, primary and fidelity gate passes. Otherwise \`ENGINEERING_NULL_OR_INELIGIBLE\`. This is a stronger mechanism-specific comparison and a fresh minimum set before new training.
- Secondary descriptive measures: matched-fit null root-wise minimum, all seven wrong-clock fit scores and post-fit wrong-order controls, source-fitting loss traces, retained subspace principal angles, source A/B/S per arm under *true* clock, KL/NLL/correctness, and per-pass state closure. No adjustment of primary or exploration-specific inference after observing data.

## EvalMath, sham-tail calibration and protected populations

- Engineering n=8 cannot certify a nonparametric 95th-percentile/95%-confidence sham-tail bound, which needs >=59 **independent calibrated roots**. Run pinned EvalMath v1 script at \`MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a\` using one root-level bounded \`b(d)=d/(1+|d|)\` pilot effect, seven within-root wrong-clock false improvements \`b(max(0,C_static−C_wrong_j))\`, fixed raw scientific minimum .010 => bounded \`0.009900990099009901\`, target raw .025 => bounded \`0.024390243902439025\`, measurement resolution .002, alpha=.05, comparisons=1, hypothetical future confirm capacity 64, and true independence/seal flags.
- Expect \`REDESIGN_OR_PILOT_ONLY\` with \`sham_quantile_uncalibrated\` on 8 roots. Save exact strict exit code 2 and full reasons. Even a positive engineering result cannot generate or release scientific confirmation. A future 59+ root *independently preregistered* strong-null calibration must measure the root-level largest wrong-clock fitted false gain, with all prior model seeds excluded, before power and release can be revisited.
- No SUB-06 calibration/confirmation, ARG-03-N test, AMM-03..09, CQ-08 confirmation or other protected splits are generated, opened, or re-used. Original SUB-01..05 results, positive and negative, remain immutable.

## Interpretation and provenance

The null is *mislabelled source fitting* under equal model class, source discovery data and optimization budget, followed by genuine finite-dose evaluation in the original physical clock. The nulls preserve each discovered C_t and D_t multiset but intentionally break the alignment of those arrays to the recurrent timeline. Passing would support some source-specific **temporal assignment information** in a small trained recurrent synthetic model. This remains a digital, output/task-conditional causal quotient experiment and makes no claim about biological substrate replacement, consciousness or personal identity.
