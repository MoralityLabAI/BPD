# BPD-NC09-FC02: task-derivative-matched physical shams and aggregate-null calibration

**Prospective registration: November 18, 2026.** Commit this protocol BEFORE source implementation or training calibration roots. Original NC09 Fixed12 and FC01 first-run source/results remain untouched. This is independent calibration/development, not scientific confirmation.

## Prior evidence and question

NC09 Fixed12 registered a developmental PASS (16 roots, model-score-selected DP12 true-label NLL +0.027179 over flat3 and +0.028017 over discovery-dose-matched flat3). FC01 trained 59 independent roots and recovered +0.032142 allocation benefit but gave a rootwise 95/95 **worst-sham-orientation** floor +0.294179 nats, dwarfing the +0.027 planning target. Exact pinned EvalMath returned REDESIGN_OR_PILOT_ONLY and held confirmation sealed. The distinct earlier SF03 tangent-matched dark-fiber experiment found that first-order sensitivity matching can remove apparent mechanism benefits in a different synthetic task.

FC02 tests a stronger source-legal null that matches retained physical write energy AND first-order effects on the model's own final score. It calibrates the null variability of the **mean effect across 32 independently trained model roots**, rather than substituting a worst single-root orientation as the variation of that mean. It also preserves the original distribution-free rootwise 95/95 floor as a separate diagnostic.

## Sources, model and original frozen causal arms

- **59 entirely new independent 20D DenseLooper roots 13000–13058**; excluded engineering smoke seed **12999**. No root replacement. Disjoint from NC04–NC11, NC09 900–915, FC01 12000–12058 and all protected experiments.
- Exactly the original NC09 Fixed12 source: four tied nonlinear passes, fully trainable shared physical mix, 384 Adam steps on 2048 labeled train cases (batch128, lr .005, minibatch seed root+1737701). Separate 1024 clean score-gradient discovery examples (split2), 512 model-only diagnostics (split4), 1024 independent true-label final cases (split3). No task-label information enters source projector, sham, gain or physical gauge admission.
- Reuse unchanged original NC09 experiment.py with SHA256 **8d1b88eea8518a83669f03f9126136ca3d56ac5bac5fe041e2e03fb12922d380** and original protocol SHA256 **dc9423f1fd9e308221c41ed2af83e7398a59dd0eb0beefa8394091e373156e02**. Copy only its exact training wrapper with a new root roster guard.
- Preserve the registered source-only DP12 per-pass rank tuple, exactly 12 retained physical write directions, the fixed flat3 rank12, flat3 source-RMS-dose matched rank12 (gain 0.5–1.5), stronger flat4 rank16, and reversed-rank descriptive control. Actual physical BPDModel/TiedComponentLinear override-read runs four genuine closed-loop recurrent updates.
- Source/root technical gate: >=.85 clean accuracy on each task context, physical matrix density >=.90, full-mask native teacher closure logit <=5e-5 and KL <=1e-7, selected eigen/gauge/gain eligibility, eight admitted physical shams, and orthogonal basis physical-map relative error <=1e-4 / all-pass logit discrepancy <=5e-5. Original eligibility failures remain in the 59-root denominator.

## New source-only task-derivative-matched sham

On **clean discovery** trajectories, let z[n,t] in R^20 be the full factor read, g[n,t] its model-self final-logit score gradient with detached model prediction sign, a[n,t] the physical prewrite activation and U the full decoder. For any pass-rank-matched projector H_t define

W(H)=sqrt(mean_{n,t} ||a[n,t] H_t U||² / 20),

d_n(H)=sum_{t=0}^{3} g[n,t]^T (I-H_t) z[n,t],

m(H)=mean_n d_n(H), v(H)=sqrt(mean_n d_n(H)²).

The scalar d_n approximates the *signed first-order final model-score change* from simultaneously removing the complementary read directions on the clean trajectory. It is NOT the actual nonlinear true-label NLL effect. The source-only gradients and weights are disjoint from final task labels.

For each independent root, draw exactly **4096** seeded local same-rank projector proposals using NumPy RNG seed **13130000+root**. Choose one angle per proposal from **{0.20,0.30,0.45,0.65} radians** and use it at all four passes, with independent reduced-QR random complement directions. Every proposal is an orthogonal rank-r_t projector with the identical 12-slot tuple. The minimum 0.20 principal angle precludes an unchanged projector.

Admit the **first eight** proposals simultaneously matching all of:
1. source retained physical write RMS ratio |W(H)/W(P_DP12)-1| <= **0.15**;
2. signed source-score mean |m(H)-m(P_DP12)| <= **max(0.005,0.25|m(P_DP12)|)** logit units;
3. source-score RMS |v(H)-v(P_DP12)| <= **max(0.005,0.25v(P_DP12))** logit units.

Compute on ALL original 1024 discovery inputs; no final labels/heldout source outputs can be used. Preserve all 4096 candidate counts, selected angles and three source-constraint differences. If fewer than eight qualified, preserve SHAM_MATCH_INELIGIBLE. No first-run proposal/angle/threshold modification.

## Null matched to the 32-root mean estimand

For each root i let L_ij be the heldout true-label task NLL of the eight preselected derivative/dose-matched physical shams. The primary root effect is **S_i = mean_j L_ij - NLL(DP12_i)**. Independently designate a uniformly random sham pseudo-target index J_i from RNG **18180000+i** (i is the actual registered seed). The sham-only pseudo-effect for any index j is

Z_ij = mean_{k != j} L_ik - L_ij.

Exactly **E_J[Z_iJ | L_i0,...,L_i7]=0**; all eight choices remain correlated within one independently trained model. One model root, not eight shams or 1024 examples, is the inferential unit.

With 59 full distinct calibration roots, the frozen **aggregate 32-root null planning statistic** uses NumPy RNG **18180018** and an exact nested-bootstrap algorithm:
- For each of **256 outer** replicates, sample 59 whole observed model-root sham vectors with replacement.
- For each outer replicate, simulate **4096 inner** null draws of a 32-independent-root mean: sample 32 roots with replacement from the outer 59 and independently choose one of 8 pseudo-target sham indices for each selected root; calculate the 32-root average Z.
- Take the empirical 95th percentile across 4096 simulated mean null effects for that outer replicate. Set B_mean32 to the empirical 95th percentile across the 256 outer 95th-percentile values. Preserve all 256 outer estimates and seeds.

B_mean32 is an **approximate nested-cluster-bootstrap planning threshold** for the positive tail of the root-mean pseudo-selected sham null, not an exact distribution-free 95/95 theorem, not a confirmatory p-value, and not an already validated multiple-comparison threshold. Treat its nominal 95/95 description as bootstrap quantiles only.

Also compute the original FC01 rootwise **95/95 tolerance diagnostic** T_max=max_i max_j |L_ij-median_k L_ik|, with exact coverage confidence 1-.95^59, solely for the 59 independent rootwise nulls. Never silently substitute T_max for a root-mean null variance and never discard it.

## Frozen engineering-calibration gates and scientific hold

On 59 independent root receipts compute signed final-NLL effects A_i=flat3-DP12, D_i=discovery-dose-matched-flat3-DP12, S_i=mean eight new shams-DP12, C_i=DP12-clean, F_i=DP12-flat4. Compute five simultaneous paired whole-root percentile bootstrap intervals with **4096** resamples and seed **13131313** (descriptive engineering intervals; individual arms already fixed).

An **AGGREGATE_TANGENT_NULL_DEVELOPMENT_POSITIVE** result requires ALL:
1. **59/59** technically eligible trained roots, each with eight shams simultaneously matching physical energy, signed first-order mean and RMS, finite outcomes, original source hashes/gauge/closure verified.
2. Both A mean >=+.008 and D mean >=+.006 nats, with each root-bootstrap 95% lower endpoint >0.
3. C mean/upper <=+.050, F mean/upper <=+.020, and S mean >=+.020 with bootstrap lower >0, all in true-label NLL nats.
4. S's root-bootstrap LOWER 95% endpoint is strictly **greater than B_mean32 + 0.020** nats (meaningful extra causal specificity above the aggregate pseudo-target null).
5. A normal-approximate 80%-power, one-sided Bonferroni-three-comparison alpha .05 check for a future n=32 independent model-root mean with signal margin (mean S-B_mean32-.020), using 59-root sample SD of S, requires **<=32 roots**; also enforce the exact all-positive sign-test minimum model roots (at least six with three comparisons).

If any root is ineligible, preserve **CALIBRATION_INCOMPLETE** and do not produce a completed null threshold. If full 59-root admission passes but any engineering/power gate fails, preserve **AGGREGATE_TANGENT_NULL_PILOT_ONLY** and all original signed results, simulated nulls and reasons. The positive verdict describes **development/calibration only**. It never unlocks scientific confirmation on its own: a new prospective confirmation protocol, a validated aggregate-null coverage procedure and a separately approved EvalMath power/variation contract on new calibration roots must precede release of a sealed 32-root scientific cohort.

Original FC01's REDESIGN_OR_PILOT_ONLY is unchanged and remains binding to that first protocol. No new 32-root scientific confirmation models or protected ARG/AMM/CQ sources are generated, trained or evaluated in FC02.
