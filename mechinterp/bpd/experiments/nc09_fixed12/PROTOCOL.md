# BPD-NC-09: fixed twelve-slot passwise causal allocation in trained recurrences

**Prospective development registration: November 18, 2026.** Commit this complete protocol before implementing NC-09 or training any of its new model roots. The first original result is retained whether its scientific-development decision passes or fails. Do not recycle the unused NC-09 cross-task-transfer branch or any prior NC-05–08 model/data seeds.

## Motivation and registered question

NC-08 prospectively allocated a variable number of physical BPD read/write directions to each of four nonlinear recurrent steps. It reduced the average total retained rank to 10.75, admitting all eight physical sham controls, and passed absolute task fidelity, but its source-only smallest-budget selector missed the frozen true-label NLL-cost upper limit against fixed rank four (observed +0.024799 versus allowed +0.020). Independently, its **predeclared secondary, descriptive** fixed-budget comparison suggested source-derived passwise allocation at budget 12 had lower heldout task NLL than flat rank three (same total 12 physical directions): 0.117212 versus 0.130279, difference 0.013068 nats. NC-08's registered FAIL and that source-exposed comparison remain unchanged.

NC-09 tests that allocation advantage on **sixteen new independently trained model roots** with the total physical write rank fixed to exactly 12, and separates allocation from changes in retained *physical write dose*. It does not select a budget from true-label data, change source-only cutoffs or tune rank after seeing heldout outcomes.

## Source model and entirely fresh cohorts

- Engineering development models: roots **900 through 915 inclusive**, 16 independent newly initialized 20D DenseLooper models. Excluded smoke root **899**. No historical NC-04 through NC-08 training roots, prior evaluation cases, protected BPD-CQ-08/ARG-03-N/AMM-03 populations, calibration roots or confirmation roots are used.
- Same architecture as NC-08: 20D fully learned dense recurrent shared physical mix applied at four tied nonlinear passes, trainable encoder/preweight/head, task-defined context-dependent binary sign label. Exactly **384 Adam steps**, learning rate .005, minibatch 128 on **2048** train examples sampled from seed/split 1.
- Every root has independent **1024 derivative-discovery** cases (split 2), **512 clean-model validation diagnostic** cases (split 4), and **1024 final task-evaluation** cases (split 3). True labels are available for supervised model training and for final heldout evaluation only. No validation external task labels are accessed.
- Admit a trained root only when clean heldout accuracy >=.85 in each context, mix weight density >=.90 (absolute weight >=1e-4), and full-target native BPDModel/TiedComponentLinear output closes at all four passes: max absolute logit difference <=5e-5, KL(clean||full) <=1e-7 nats. One trained model root, not inputs/shams/passes, is one statistical unit.

## Locked source-only projectors, fixed rank and exact allocation

On clean derivative-discovery inputs, compute full-factor model-score gradient g[n,t] of detached-sign model final logit, and clean factor activation z[n,t], all at each of four shared BPD reads. External labels cannot enter g. Source-only pass covariance C_t=mean(g_t g_t^T). For each pass t and r in **{2,3,4,5,6}**, require lambda_r>1e-12 and eigengap lambda_r/max(lambda_(r+1),1e-12)>=1.001. The passwise projector P_t(r) is the top-r orthogonal C_t eigenspace, computed in float64.

Exactly preserve NC-08's source-only omitted-score proxy

L_t(r) = mean_n [(g[n,t]^T (I-P_t(r)) z[n,t])^2].

Enumerate all 5^4=625 rank tuples and choose a single rank tuple r* minimizing sum_{t=0}^3 L_t(r_t) **subject to sum_t r_t=12**, all eigengaps and eigenvalue floors. Ties within 1e-12 use lexicographic tuple order. This is the only selected budget and the only optimized tuple per model. A missing eligible tuple means the entire root is ineligible; never repair by choosing a higher budget after heldout results.

Fixed arms on the same new teacher and final evaluation examples:
1. **DP12 primary:** pass-specific rank tuple r*, full write gain 1.0.
2. **flat3 primary control:** ranks (3,3,3,3), total 12, gain 1.0.
3. **flat3-dose-matched control:** same (3,3,3,3), physical read gain alpha chosen on derivative-discovery examples as source clean retained-write RMS(DP12) / clean retained-write RMS(flat3). The ratio must lie in **[0.5,1.5]**; otherwise mark that root ineligible. On the same discovery source this exactly matches the retained write RMS while preserving 12 projector directions; report achieved absolute/relative mismatch.
4. **flat4 fidelity reference:** ranks (4,4,4,4), total 16, gain 1.0.
5. **reverse-allocation descriptive placebo:** ranks (r*_3,r*_2,r*_1,r*_0), total 12, each pass using *its own* top-r eigenvectors, gain 1.0. This predeclared permutation checks whether allocation **to a particular pass** matters beyond the multiset of four ranks. If r* is palindromic the reverse may coincide; report this explicitly. No source/test result changes which reverse is used.

Flat3 and flat4 sources require their fixed eigengaps. If any of the first four arms' eigenspace is mathematically ineligible, record the root as engineering-ineligible. The reverse is descriptive and its eigengap is checked separately. The fixed-12 DP projectors are chosen without using the clean-model validation source, model external true labels, later-source readouts or any heldout task measure.

## Independent model-only and physical validation

The **512 clean-model validation inputs** measure actual four-pass teacher-versus-retained Bernoulli KL by pass, mean final KL, 95th percentile example KL, sign-flip fraction and retained physical write dose for each frozen arm. These statistics are **diagnostics**, never used to reject or select among DP12/flat3/flat4 arms or train a new projector. The regression claim rests on the separately labeled final task outcomes, and the model's own predictive KL may disagree with external-label NLL as in NC-07/08.

All interventions use the genuine physical BPDModel/TiedComponentLinear shared mix and override-read with full coordinates and a pass-indexed projector P_t, so the edit changes each subsequent nonlinear hidden state. The new flat3 dose-matched control uses the same native physical BPD override-read path with gain alpha, without modifying encoder, head or pretrained model weights; its source dose (not downstream edited-state dose) is matched by construction. Record actual post-edit write displacement on final task inputs. No residual bypass, sphere rescaling or per-example outcome-conditioned gain.

Orthogonal factor rotation of the exact same full physical matrix must transport P as R^T P R while preserving the native V P U operator, including the per-arm scalar gain. On at least 64 heldout-free validation inputs, maximum relative operator discrepancy <=1e-4 and max absolute complete four-pass retained-logit difference <=5e-5 for DP12, flat3, flat3-dose and flat4. Nonorthogonal gauge invariance is outside this study.

## Local physical sham controls

For each root, draw a **fixed 4096-proposal** local projector sham pool for the **DP12 tuple**, independent source RNG seed 9,090,000 + root. A sham retains the same ranks at all four passes and rotates each source eigenspace towards a new independent orthogonal complement using one angle sampled from {0.45,0.65,0.85,1.05} radians per proposal (shared over four passes). Admit the **first eight** candidates with clean derivative-discovery retained physical write RMS within +/-15% of DP12's source write RMS. The sham pool is frozen before the final true-label evaluation split is accessed, and is never refilled or filtered by terminal task outcome. If fewer than eight shams qualify, mark root as ineligible; all original receipts remain.

The source matched-dose null controls the *pass-specific rank tuple* and retained write energy. An exact rank-matched source-dose sham need not match the future recurrent Jacobian or task-directional dose; report those limitations. No test-side dose matching or outcome-based angle choice is allowed.

## Single prospective development decision

On **16 completely fresh independent trained model roots**, compute true-label final NLL, accuracy, clean-vs-kept KL, native closed-loop writes and the eight matched local physical sham outcomes on the 1024-example final evaluation split. Define root-level paired true-label NLL advantages:

A_i=NLL(flat3_i)-NLL(DP12_i),
D_i=NLL(dose-matched-flat3_i)-NLL(DP12_i),
F_i=NLL(DP12_i)-NLL(flat4_i),
C_i=NLL(DP12_i)-NLL(clean_i),
S_i=median_{8 shams}NLL(sham_i)-NLL(DP12_i).

A frozen **DEVELOPMENT_POSITIVE** requires **all**:
- **16/16 roots fully admitted**: task/full-mask/gauge/eigengaps, DP12 exists, valid source dose ratio and eight qualified sham controls. All failures remain in the intention-to-test population.
- A positive same-budget allocation improvement **mean A >=+0.008 nats** with whole-root two-sided bootstrap 95% lower endpoint **>0**.
- An incremental improvement over the dose-matched same-rank control **mean D >=+0.006 nats** with root bootstrap lower **>0**. Both allocation contrasts are necessary, so there is no posthoc comparator selection.
- Mean absolute task cost **C <=+0.050 nats**, with upper bootstrap 95% **<=+0.050**.
- Mean extra task cost versus the 16-dimension flat rank-four reference **F <=+0.020 nats**, with upper bootstrap 95% **<=+0.020**.
- Mean matched same-tuple sham advantage **S >=+0.020 nats**, with bootstrap lower **>0**.

Compute all confidence intervals with a deterministic **4096-resample root-cluster percentile bootstrap** using NumPy random generator seed **909009**. Each draw resamples all five metrics as a unit for 16 independent model roots; prompts, four passes, candidate rank tuples and physical shams never count as additional roots. Because every listed threshold and comparison is jointly required, treat this as an intersection-union developmental gate and report each marginal paired interval without claiming a confirmatory familywise p value. Root power remains limited; a failure is **DEVELOPMENT_NULL_OR_INELIGIBLE**, preserving all 16 receipts and entire first-run summary. Every comparator and acceptance threshold stays fixed.

Secondary, descriptive outputs include pass-specific rank histogram, reverse-allocation task cost and matched-ESS/physical-dose differences, root-level KL miscalibration and model-error tail, pass-specific source proxy loss and bootstrap omission of any ineligible root. A result on the source-exposed NC-08 eight roots cannot rescue this cohort.

## Future scientific hold

The separately named original NC-09 cross-task-transfer branch is untouched. This fixed-budget development is scoped to 16 new models. It runs **zero** independent 59-root nonparametric sham-calibration models and **zero** 32-root scientific confirmation models, and no protected ARG-03-N, AMM-03 or BPD-CQ-08 cohorts. A later experiment needs a new prospective confirmation protocol, an EvalMath independent-model-root power/variation preflight and full independent sham calibration. All prior NC-04–08 and PD-15–17 results remain immutable.

The scientific object is a learned nonlinear four-pass physical model's task-conditioned retained write projector, under orthogonal factor gauge. The experiment does not identify a globally minimal sufficient circuit, a general pretrained TRM causal quotient or an optimum under arbitrary GL(20) transformations.
