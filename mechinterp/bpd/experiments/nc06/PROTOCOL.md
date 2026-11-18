# BPD-NC-06: norm-conditioned rank-three causal projectors in a trained nonlinear recurrence

**Prospective registration, November 18 2026.** This document is the first research commit before NC-06 source implementation, model training or scoring. All eight NC-06 development roots and future scientific calibration/confirmation sets are distinct from historical NC-04, NC-05 and PD-15–17 sources.

## Why this experiment

The first registered PD-15 and PD-16 experiments recovered 32/32 clustered 16D source-specific identity-teacher factors with observed norm weighting, versus 0/32 under unweighted or exact-ESS shuffled controls. PD-17 prospectively found 32/32 norm-weighted recovery under the original source, but 0/32 under a radial-camouflaged source with no norm/cardinality relation, while an observed-only angular method recovered 32/32 under both laws. The source-specific weight-selection signal is therefore established and its generality restricted.

The closest *actual learned recurrent* test is NC-05: eight independent freely trained dense 20D four-step nonlinear recurrences, where a rank-four full-gain gradient-covariance physical BPD projector met a strict source-registered eight-cell development gate. Mean keep-only task NLL cost was +0.021219 nats, with family-adjusted upper bound +0.045398, and matched-sham NLL advantage +0.052182 nats. NC-04 fixed rank two, which missed the 0.05-nat absolute fidelity requirement. Both are synthetic engineering, with scientific confirmation sealed.

NC-06 tests whether source-observable norm conditioning can recover a smaller, **rank-three** task-relevant causal projector on actual trained nonlinear recurrent roots, by increasing useful gradient contrast before choosing physical directions. This is a new developmental hypothesis; success or failure will be preserved.

## Frozen teacher, roots and independence

Use eight new independent model roots **600–607**, and engineering smoke seed **599** only. Reimplement the exact NC-05 DenseLooper model-training contract: 20D unconstrained full-density trainable shared physical mix, nonlinear preweight map, four tied passes, encoder and scalar head; 2048 independent train inputs, 384 Adam updates, minibatch 128, learning rate .005, independent external task labels as in NC-05. Per root: 1024 fresh discovery inputs, 1024 independent heldout evaluation inputs, each generated from the original sample_x(seed, split, N).

Admit a trained model only if both context-specific heldout accuracies >=0.85, weight density >=.90, native full-ones BPDModel/TiedComponentLinear four-pass logit parity <=5e-5, and all-pass KL(clean||full) <=1e-7 nats. Discovery never accesses external labels; final evaluation uses independently defined true task labels solely after all source-only projectors, shams and physical-dose matches are fixed. One independently trained model root is the unit of inference.

## Source-only covariance and fixed four-arm comparison

At each recurrent pass t, record clean prephysical-write activation a_{n,t} in physical input coordinates, model-score gradient g_{n,t}=d[sgn(model final logit)×final logit]/dz_{n,t} in the 20D native BPD factor read coordinates. Sign is detached; outer products g g^T are score-sign invariant. For each pass set the discovery-only mean energy s_t = mean_n ||a_{n,t}||^2 and define

w_{n,t}=exp[-||a_{n,t}||²/s_t].

The observed-only weights are fixed before evaluating any external labels, and no support/cardinality target is postulated. Compare the rank-three orthogonal top-eigenspace of:

- **norm_r3**, C_t = sum_n w_{n,t} g_{n,t}g_{n,t}^T / sum_n w_{n,t}.
- **uniform_r3**, exactly the unweighted NC-05 passwise covariance, same gradients.
- **shuffle_r3**, the exact norm-weight multiset independently permuted across discovery examples at each pass with seed 9160000 + 100*root + pass; ESS and histogram exactly matched to norm_r3.
- **uniform_r4_reference**, the unweighted NC-05 rank-four projector, full gain, an additional descriptive positive reference.

All share the exact same trained recurrent teacher, discovery examples, BPD full-rank factors and independent evaluation split per root. Every covariance is float64. Eigengap floor: lambda_r>1e-12 and lambda_r/max(lambda_(r+1),1e-12)>=1.001 at every pass. A failed arm is marked unidentifiable with no backfilled direction or post-hoc rank. Record all eigenvalues, passwise weight ESS, correlation of observed norm weight with squared model-score gradient norm and selection stability. The correlation is a diagnostic, not a gate.

The projectors are compared under native V=I, U=W^T and one independently seeded orthogonally rotated full-rank BPD factor basis; P_rot=Q^T P_native Q. Frozen relative physical write operator error <=1e-4 and maximum retained clean-vs-rotated complete-rollout logit discrepancy <=5e-5. Only orthogonal gauge covariance is claimed.

## Actual closed-loop physical BPD and dose controls

At all four tied recurrence passes, retain the selected rank-r physical write through BPDModel.intervene(mode=override_read), using all 20 factor coordinates and zero residual bypass. The perturbation changes the *next* hidden state; future passes must execute on their newly generated states. Interventions never modify the teacher's encoder, preweight module or readout head. Fixed retained-write gain alpha=1 in all four arms.

For each rank-three projector, generate 1024 discovery-only local physical rank-three projector proposals around the selected subspace at the NC-05 angle grid theta in {0.45,0.65,0.85,1.05} radians. Keep eight projectors whose retained clean physical write RMS lies within +/-15% of the candidate's source-only discovery dose. All eight independent roots must have eight accepted shams for every ranked rank-three arm; failed sham matching is retained as a blocked root. Under test, report exact actual physical write displacements as an additional unmatched-dose limitation.

For all heldout examples record full clean and keep-only final true-label NLL, accuracy, KL by pass, retained physical write RMS, and median NLL of the eight admitted rank-matched local shams. Report model-root paired differences between uniform_r3 and norm_r3, and between shuffle_r3 and norm_r3. Same-rank shams are selected without outcome access.

## Frozen eight-root engineering decision

A developmental positive signal requires all of:

1. Eight task-competent dense trained roots, native full-mask closure, rank-three eigengap, rotated-basis parity and eight matched physical shams per learned rank-three arm.
2. Norm-rank3 mean true-label NLL cost relative to the intact model <=0.05 nats and upper 95% root-bootstrap bound <=0.05.
3. Mean NLL(uniform_r3)-NLL(norm_r3) >=0.01 nats with 2000-repetition root-bootstrap 95% lower endpoint >0.
4. Mean NLL(shuffle_r3)-NLL(norm_r3) >=0.01 nats with the same positive lower endpoint.
5. Mean median-local-sham NLL minus norm-rank3 NLL >=0.02 nats with positive root-bootstrap lower endpoint.

Bootstrap resamples eight entire independent trained model roots, seed 606006. The fourth unweighted rank-four reference must be reported regardless of its outcome, but is not a rescue arm for failed rank-three criteria. Report every failure as ENGINEERING_NULL_OR_INELIGIBLE, preserve full raw receipts and refrain from claiming statistical confirmation from eight highly related roots.

An eventual NC-07 confirmation requires a new prospective protocol, at least 59 independent sham-calibration roots for a nonparametric 95th-percentile/95%-confidence floor, an EvalMath power/variation preflight, and 32 completely fresh confirmation model roots. Those populations, along with protected ARG-03-N and AMM-03 confirmatory material, stay sealed. Their roots are not generated, evaluated or reserved by any development runner.

## Scientific scope

This is a true four-pass tied-weight closed-loop learned nonlinear model, with an independently defined external task. A norm-weighted rank-three engineering signal could improve the causal subspace rank/fidelity frontier under this synthetic teacher. It would still need source/architecture transfer, learned-mechanism labeling, task-legal donor checks and independent confirmation. PD-15–17's identity-teacher source results cannot themselves certify native recurrent task mediation.
