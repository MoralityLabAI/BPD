# BPD-NC09-FC04: full-tangent constraint-Jacobian sham solver

**Prospective development registration, November 18, 2026.** This protocol is the first FC04 scientific/source commit. Any FC04 code and newly trained model roots follow it. The FC01 REDESIGN_OR_PILOT_ONLY, FC02 CALIBRATION_INCOMPLETE and FC03 ENGINEERING_NULL_OR_INELIGIBLE original results remain unchanged. **Scientific confirmation stays sealed.**

## Precise question

FC03 used eight scalar complement-frame coordinates to optimize fixed-angle, 12-slot physical BPD projectors against source-only retained-write RMS W and final-model-score first-order signed mean m and RMS v. It admitted eight distinct shams on 13/16 independent new models, versus 11/16 using the original FC02 random rejection sampler. The frozen 16/16 feasibility gate failed, including root 14001 with zero admitted shams. The failure concentrated in models with small original signed and RMS source-score signals and tight absolute tolerances.

FC04 tests whether a **full complementary Stiefel tangent chart with a three-observable Jacobian correction** can recover valid, distinct physical sham projectors from **fresh independent model roots**, while preserving FC02's original source constraints and FC03's nontrivial geometric separation.

This is a new source-admission **engineering experiment**, and it must not be presented as proof that matching source gradients certifies behavioral equivalence.

## Fresh model roots and protected parent mechanism

- **Sixteen independent 20D trained recurrent model roots 15000–15015** for the original development cohort, with an **excluded engineering smoke root 14999**. No replacement. Disjoint from NC04–NC12, NC09 Fixed12 900–915, FC01 12000–12058, FC02 13000–13058, FC03 14000–14015 and all protected scientific calibration/confirmation roots.
- Train exact NC09 Fixed12 DenseLooper for 384 Adam steps at lr .005/batch 128, using 2048 externally labeled train examples. One 20D dense shared physical-mix matrix is reused across four true nonlinear recurrent passes; learned encoder/preweight/head and original task label law unchanged.
- New independent source splits per root: 1024 clean derivative-discovery examples (split2), 512 model-only validation diagnostics (split4), and 1024 externally labeled heldout evaluation cases (split3). Only model training and terminal evaluation use external task labels.
- Reuse original NC09 source-only passwise DP12, which selects a four-pass rank tuple r_t∈{2,...,6} with sum=12 by minimizing the source-only omitted final-score linearization. Keep flat3, source-RMS-dose-matched flat3, flat4 and reversed rank controls, original native full-rank BPDModel/TiedComponentLinear four-pass override-read, orthogonal factor-gauge checks, and full-mask closure.
- Original NC09 Fixed12 source SHA256 **8d1b88eea8518a83669f03f9126136ca3d56ac5bac5fe041e2e03fb12922d380** and PROTOCOL SHA256 **dc9423f1fd9e308221c41ed2af83e7398a59dd0eb0beefa8394091e373156e02** are immutable. The original FC03 solver Git blob SHA **0fc8c6989e16d6deeb8e642029c63414458a43bf** is the frozen comparator, and original FC02 tangent_shams.py Git blob SHA **cb9e722b474b1dd52b5303f05332a9d3e5f88dc0** is the frozen random-sampler comparator.
- Each model root must retain clean accuracy≥.85 for each source context, original mix density≥.90, all-pass full-mask parity max logit error≤5×10^-5 and Bernoulli KL≤10^-7, eigengap source admission, original physical dose gain in [.5,1.5], and native-vs-orthogonal-rotated physical projector operator relative error≤10^-4 and full-loop logit difference≤5×10^-5. Failures stay in the denominator.

## Full tangent parameterization and exact geometry

For each pass t, let E_t∈R^(20×r_t) be the selected orthonormal DP12 read basis and let C_t∈R^(20×(20-r_t)) be its deterministic orthonormal complement obtained through complete QR of E_t. A full-rank free chart Y_t∈R^((20-r_t)×r_t) defines

Q_t(Y_t)=C_t qf(Y_t),

B_t(Y_t)=cos(θ) E_t + sin(θ) Q_t(Y_t),

H_t(Y_t)=B_t(Y_t)B_t(Y_t)^T,

where qf is reduced QR with positive diagonal sign convention and **θ=.25 radians fixed across all source roots, passes and iterations**. Parameter dimension is the full sum_t (20-r_t)r_t, about two hundred coordinates, rather than FC03's eight preselected complementary-frame mixing coordinates. Redundant right-rotational directions are allowed and tracked.

In exact arithmetic, H_t is symmetric, idempotent, rank-r_t, and all its principal angles from the frozen DP12 subspace P_t=E_tE_t^T are θ. Its normalized chordal separation from DP12 is therefore exactly sin(.25)=.247403959..., above the frozen minimum **.20**. Every accepted H must also be at least **.18 normalized chordal distance from every earlier accepted sham**. Source-rank traces, symmetry/idempotence, original-projector separation and pairwise source separation are independently audited from archived float64 full 4×20×20 projectors.

## Exactly original three source-matching constraints

On the **same 1024 clean discovery cases** for all proposed projectors H, use the unchanged FC02/03 observables:

W(H)=sqrt(mean_(n,t) ||a_(n,t) H_t U||²/20)

d_n(H)=sum_t g_(n,t)^T (I-H_t) z_(n,t)

m(H)=mean_n d_n(H), v(H)=sqrt(mean_n d_n(H)^2),

where a is the physical prewrite activation, z the full factor read, U the physical decoder, and g the detached-sign **model's own** final-score gradient at clean recurrent reads. True task labels and test logits are inaccessible to construction.

Against the frozen DP12 P, require every retained sham simultaneously:
1. |W(H)/W(P)-1|≤**0.15**.
2. |m(H)-m(P)|≤**max(.005,.25|m(P)|)** *model logit units*.
3. |v(H)-v(P)|≤**max(.005,.25v(P))** *model logit units*.
4. Normalized chordal distance from DP12≥**.20**; pairwise distance to each previously accepted H≥**.18**.

No normalizing by heldout task effects, relaxing small-score source tolerances, reducing angle, moving rank or replacing source samples after seeing model-root outcomes.

## Source-only full Jacobian correction

For each fresh root, draw exactly **32 independent initial Y charts**. One chart means four full random Gaussian complementary coordinates Y_t, seeded sequentially by NumPy generator **15_040_000+root**, with entries scaled by (20-r_t)^(-1/2) so the QR starts are full-rank. Every root has at most **64 projected correction updates per chart**, with **at most four fixed backtracking trial scales (1, 1/2, 1/4, 1/8)**. Never use task labels or validation outcomes in the correction.

For a proposed Y, compute the signed normalized three-vector

f(Y)=((W(H)/W(P)-1)/.15, (m(H)-m(P))/tol_m, (v(H)-v(P))/tol_v).

Its three Jacobian rows are obtained by exact PyTorch float64 automatic differentiation of the full source moment function with respect to **all** Y coordinates, treating all clean discovery tensors and E,C as frozen. Let J∈R^(3×d) and λ=10^-3. Fix the damped minimum-norm Gauss-Newton direction

Δ=-J^T (JJ^T + λ I_3)^(-1) f(Y).

Clip the entire concatenated Δ to Euclidean norm≤**0.5**. The solver tries Y+αΔ for α in **(1,.5,.25,.125)**, accepting the first candidate whose squared sum of positive normalized absolute *source constraint excesses* strictly decreases, or whose source constraints pass and whose pairwise distance to earlier shams passes. When all trial scales fail, retain Y and proceed to the next update; at most 64 updates per chart.

Pairwise-source-distance failures at a source-feasible chart are treated as chart failures and move on to the next seeded independent chart. The source distance to DP12 is independently verified after each candidate. Zero-difference or singular Jacobian paths are logged as failures without replacing the source, chart seed, step size or optimizer.

For each chart, check the full exact source criteria initially and after every update, accepting at most its **first eligible sham**. Accept at most **eight sham projectors** in chart order among the fixed 32 starts. Once eight are accepted, the remaining chart slots are marked skipped, with no new roots/charts sampled. Preserve all iterations attempted, rejection reasons, normalization, Jacobian singular values/rank (threshold singular value 10^-8), condition numbers, source metric errors, pairwise separation and projector-byte SHA256. Each root is one inference unit regardless of number of starts.

This Jacobian correction is a numerical feasibility procedure. The implicit function theorem would yield a local codimension-three equality submanifold where J has full row rank and a feasible equality solution already exists. FC04 does not claim that a global feasible set exists at θ=.25 for every trained source or that this chart exhausts it.

## Fixed algorithmic controls and actual task endpoint

Run **both exact unchanged original FC03 eight-coordinate 32-chart/64-step Adam solver** and FC02 **4096-proposal random sham sampler** on each new root. Each uses the identical original W/m/v source tolerances, exact passwise rank and source examples. They have separate compute budgets and cannot be reported as equal-FLOP. All source-only samplers run before accessing the root's externally labeled evaluation cases. Record qualified and admitted shams for all three algorithms; if any sampler fails eight, preserve its partial list and any available task outcomes.

Every accepted FC04 sham is evaluated by **actual four-step closed-loop BPDModel/TiedComponentLinear override-read** on independent final-label cases. Report original DP12, flat3, physical-source-dose-matched flat3, flat4 task NLL, all relevant KL and true heldout physical writes. Numerically check gauge and source geometry independent of the new optimizer.

## Frozen first sixteen-root engineering decision

**FULL_TANGENT_ENGINEERING_POSITIVE** requires all:
1. **16/16** genuinely independent fresh model roots admitted under original clean task/native/full-mask/gauge conditions, with **eight FC04 full-tangent shams each** meeting all three frozen source constraints and both distance gates.
2. The original FC03 eight-coordinate solver has full eight-sham admission on **at most 14/16** new roots. Thus FC04 improves complete-source admission by **at least two independent trained models** under the same match thresholds. FC02 random full-admission results are also reported.
3. Every accepted FC04 projector's SHA256, rank, trace, symmetry/idempotence≤10^-6, DP12 separation≥.20, and pairwise separation≥.18 independently reproduce from archived float64 matrices; no external-label access before projector selection.
4. All 16 full native heldout task and sham NLL outcomes are archived. Record the descriptive 4096-repetition whole-root percentile bootstrap (seed **15041504**) of FC04 mean-of-eight-shams minus DP12 task loss when 16/16 admitted.

Any failure is **ENGINEERING_NULL_OR_INELIGIBLE**, with complete original source-receipt archive. An engineering code exception is separate. No rescue through additional starts, larger damping grids, label-conditioned sham selection or excluded-root replacement.

## Scientific hold

FC04 develops only 16 new model roots. It does not launch independent 59-root sham-recalibration or 32-root scientific confirmation. The FC01 REDESIGN_OR_PILOT_ONLY and FC02 CALIBRATION_INCOMPLETE original scientific holds remain binding. A positive feasibility result would motivate a **new** independent 59-root recalibration with validated root-mean sham variation and exact EvalMath resolution/power review, followed by a separately locked independent confirmation protocol.

The experiment is about source-only physical causal-sham matching on a synthetic 20D tied nonlinear recurrence. Fixed-angle geometry and linear algebra are exact in real arithmetic; sample source-match and heldout behavioral claims are empirical. Matching three first-order source moments does not imply equivalent future task-label behavior.
