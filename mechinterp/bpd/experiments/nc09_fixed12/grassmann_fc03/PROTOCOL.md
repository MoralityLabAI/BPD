# BPD-NC09-FC03: constrained Grassmann sham solver

**Prospective registration, November 18 2026.** Protocol FIRST before source implementation and before training the fresh cohort. This is a new engineering-development experiment, not an update or rerun of FC01/FC02 and not scientific confirmation.

## Motivation, source exposure and scientific question

Original NC09 Fixed12 showed source-derived passwise 12-slot physical BPD write allocation improved independently labeled task NLL over same-budget flat rank three. FC01's independent 59-root matched-sham calibration measured a rootwise 95/95 maximum orientation floor of +0.294179 NLL nats and received pinned EvalMath REDESIGN_OR_PILOT_ONLY. FC02 preregistered source physical write RMS, signed model-self final-score mean, and score RMS sham matching. Its first 59-root run was CALIBRATION_INCOMPLETE: only 40/59 independent trained models found eight matches in 4096 local random orientations, with 19 nonrandom source-admission failures. The 40 eligible FC02 models showed a descriptive mean stronger-sham-minus-DP12 true-label NLL effect of only +0.010241. Those previous registered outcomes and independent confirmation hold remain intact.

FC03 asks whether a **continuously optimized local Grassmann chart**, keeping exactly FC02's physical/tangent matching tolerances and a *nontrivial source projector separation*, produces eight feasible source-only shams more reliably than FC02's rejection sampler on newly trained models.

## Independent model roots, fixed teacher and source splits

- First engineering-development cohort: **sixteen fresh independently trained recurrent model roots 14000–14015**. Source engineering excluded smoke root **13999**. No replacement, retraining, cherry-picking, reuse of historical models (including FC01 roots 12000–12058 and FC02 13000–13058), or protected scientific confirmation roots.
- Same 20D dense fully learned four-pass shared physical-mix nonlinear DenseLooper, exactly 384 Adam steps on 2048 labeled train examples, batch128, learning rate .005, minibatch generator seed = model_root + 1737701. Independent 1024 **model-only gradient discovery** examples (split2), 512 **model-only diagnostics** (split4), and 1024 **externally labeled final evaluation** examples (split3). Task labels may train the intact teacher and score final heldout outcomes only.
- Unmodified parent NC09 Fixed12 physical source must have experiment SHA256 **8d1b88eea8518a83669f03f9126136ca3d56ac5bac5fe041e2e03fb12922d380** and PROTOCOL SHA256 **dc9423f1fd9e308221c41ed2af83e7398a59dd0eb0beefa8394091e373156e02**. Reuse its exact source-only DP12 rank12 allocation, flat3, source-RMS dose-matched flat3, flat4, and native four-pass BPDModel/TiedComponentLinear full-rank override-read interventions. Copy the exact model training contract with a new root guard.
- Each root must pass clean accuracy >=.85 in both task contexts; mix density >=.90; full BPD teacher mask closure max logit <=5e-5, all-pass KL <=1e-7; DP12/flat3/flat4 spectral eligibility; dose gain [0.5,1.5]; all four primary physical operator native/orthogonal gauge relative error <=1e-4, complete-loop logit error <=5e-5. Preserve each failure.

## Frozen Grassmann chart, matching constraints and optimization

For the original DP12 orthonormal basis E_t (20 × r_t) at each of four passes, with r_t in {2,3,4,5,6} and total rank 12, construct three independent seeded orthonormal complementary frames Q_{t,0}, Q_{t,1}, Q_{t,2} by projecting independent Gaussian matrices into the orthogonal complement of E_t and taking reduced QR, with deterministic diagonal sign convention. For eight real variables y_{t,k} in [-1,1], set

Q_t(y) = qf[(I-E_t E_t^T) (Q_{t,0}+y_{t,0}Q_{t,1}+y_{t,1}Q_{t,2})],
B_t(y)=cos(theta) E_t + sin(theta) Q_t(y), H_t(y)=B_t(y)B_t(y)^T,

where qf uses reduced QR with diagonal-positive sign convention and **theta=0.25 radians for every pass, every chart, every update**. In exact arithmetic B_t is an orthonormal basis and H_t is a rank-r_t orthogonal projector. Its normalized chordal distance from P_t=E_tE_t^T equals sin(theta)=0.2474, guaranteeing an edit away from the identity projector.

For clean discovery examples n, let z[n,t] be BPD factor read, g[n,t] the detached-sign model's own final-score gradient, a[n,t] the physical prewrite activation and U the factor decoder. For arbitrary H_t define the **same FC02 source observables**:

W(H)=sqrt(mean_{n,t} ||a[n,t] H_t U||^2 / 20),
d_n(H)=sum_t g[n,t]^T (I-H_t) z[n,t],
m(H)=mean_n d_n(H), v(H)=sqrt(mean_n d_n(H)^2).

Freeze exactly FC02's original accepted source tolerances: relative physical RMS error **<=0.15**; absolute signed first-order mean difference **<=max(.005, .25|m(P)|)** model-logit units; absolute first-order RMS difference **<=max(.005,.25v(P))** model-logit units. No change to these tolerances after source work.

Also require aggregate normalized chordal distance to the DP12 P >=**0.20**, and distance to every previously accepted sham >=**0.18**. Normalize distance between pass-indexed projectors H,G as sqrt(sum_t ||H_t-G_t||_F^2 / (2*12)). For all accepted shams report source exact errors, physical write RMS, score target and selected separation.

### Deterministic optimizer and source-only budget

Use exactly **32 independent chart starts**, initial y=0, NumPy/Torch RNG seeded from **14,030,000+model_root**. For each chart independently generate its three complement frames for all four passes, and audit the starting candidate against all five constraints. Accept the FIRST passing iterate and record its chart index and step, then continue to the next chart until all 32 have been attempted. If initial candidate fails, perform at most **64 float64 PyTorch Adam steps**, fixed lr **0.08**, projecting the eight y coordinates to [-1,1] after each step. Do not change the chart, rank, angle or source examples during optimization.

Frozen loss: sum over W, m, v of squared positive normalized excess beyond the respective original tolerances, plus squared positive pairwise-distance shortfall divided by .18, plus **0.001*mean(y^2)** regularization. A first feasible candidate is accepted regardless of residual optimizer loss. Source observables and separation are **checked independently against the full 1024 clean discovery observations** after each step. The optimizer only accesses clean model prewrite activations, model-score gradients, frozen factor matrices and previously accepted source-only shams. Zero external task labels, heldout outcomes or task NLL enter matching.

Each successful chart yields at most one accepted sham projector. Retain the first eight passing candidates across chart order. If fewer than eight succeed among the 32 starts, preserve **SHAM_MATCH_INELIGIBLE** without increasing budget. Every root has exactly 32 starts, with unused starts after eight are recorded as skipped, not as new independent roots. Report score/energy residuals and minimum physical Grassmann separation. Projector symmetry/idempotence max error must be <=1e-6.

The separately frozen excluded-13999 full-budget engineering pilot trains for all 384 steps and runs all 32 charts and max 64 updates, using the full 1024 discovery/1024 final source sizes. An unsuccessful excluded smoke is preserved, and new engineering source can only be redesigned under a distinct new study ID and independent root cohort.

## Matched original FC02 rejection baseline and independent future task

On each NEW root, also run the **original unmodified FC02 local random sampler** on the same source: 4096 independent local rank-tuple proposals, original angles (0.20,0.30,0.45,0.65), original seed 13,130,000+root, FC02's same W/m/v source tolerance gates. This source-only sampler is a fixed, fully costed baseline. Report its exact qualified count and first eight outcomes if available. It does not get adapted to the new solver's successful source geometry.

For all admitted shams, evaluate actual closed-loop four-pass physical BPDModel/TiedComponentLinear task NLL, accuracy and KL on the 1024 independent final labeled inputs, plus frozen DP12, flat3, dose-flat3 and flat4. Even when fewer than eight shams match, retain their individual true-label task outcomes as *descriptive incomplete-set observations*. No cherry-picking, pseudo-independence or imputation.

## Frozen sixteen-root engineering decision

**CONSTRAINED_GRASSMANN_ENGINEERING_POSITIVE** requires simultaneously:
1. **16/16** newly trained independent roots pass original task/native/orthogonal physical source gates and supply eight **solver-matched** source-only shams each, satisfying W/m/v tolerances, distance from P>=.20, pairwise distance>=.18, exact pass ranks and finite outcomes.
2. The original FC02 rejection sampler fully admits eight shams on at most **14/16 roots**. Thus solver admission must exceed matched original-rejection admission by **at least two independent trained roots** on the identical source family.
3. Every selected solver projector has max symmetry/idempotence error <=1e-6; code confirms no task-label access during source selection and reports all 32 proposal starts, iteration counts, source metrics and original full model-root receipts.
4. All 16 full native four-pass heldout task evaluations are complete, including DP12 and the frozen equal-rank controls; all eight solver shams have externally scored NLL, with their mean/median differences and within-sham variation published without a new causal significance claim.

If any gate fails, preserve **ENGINEERING_NULL_OR_INELIGIBLE** and the full first-run evidence. A technical error is recorded separately. The root-level 16-sample descriptive paired bootstrap uses **4096** whole-root resamples, seed **14031403**, for solver sham-average-minus-DP12 and matched-original-FC02 sham effects only when 16/16 eligible.

FC03 does not perform any new independent 59-root nonparametric sham calibration or scientific confirmation. Its 16 roots are engineering development. FC01's REDESIGN_OR_PILOT_ONLY and FC02's CALIBRATION_INCOMPLETE remain the original scientific holds. A future 59-root recalibration and independently validated 32-root confirmation contract would require a fresh prospective study, new independent sources and an EvalMath resolution/power audit.

## Mathematical evidence class

Exact finite algebra in orthogonal charts ensures constant principal angle, retained rank and normalized chordal separation from P. Feasibility of W/m/v is numerical and empirically verified, with an explicit failing-root case. A first-order model-score match on clean states is not equivalence of future nonlinear true-label task behavior. Model roots, rather than optimizer starts, shams, cases or recurrent steps, are independent statistical units.
