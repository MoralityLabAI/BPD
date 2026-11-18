# BPD-NC-08: passwise causal rank allocation under a fixed physical write budget

**Prospective source-only registration, November 18, 2026.** Commit this protocol BEFORE implementing NC-08 or generating any NC-08 trained model root. Its source, eight independent development roots, passwise rank policy, physical sham budget and frozen engineering gates must remain unchanged after the first run. NC-04–07 and PD-15–17 outcomes are historical development context.

## Question and prior evidence

NC-06's original eight independently trained 20D four-loop dense recurrences passed absolute task fidelity at rank 3 but norm-conditioned gradient covariance added only 0.00035 nats versus uniform and exact-ESS shuffled controls. NC-07 prospectively selected a single rank r shared over all four recurrent passes from {2,3,4,6}, using a model-only clean-to-retained KL selector. It selected (3,4,3,4,6,3,6,3), giving mean **four physical write coordinates per pass**, and only 7/8 roots met the eight matched-sham requirement because a rank-six model had seven eligible candidates among 1,024 source proposals. The original engineering compression gate failed, while selected task NLL was promising as a post-hoc diagnostic.

NC-08 asks whether allocating the retained physical write **rank differently at each recurrence pass** yields a source-only, smaller aggregate rank budget while retaining task-fidelity and sham specificity. A gradient-based proxy builds a small fixed, equal-budget candidate set; the actual nonlinear four-loop teacher self-KL chooses among those budgets on a separate source split.

This is a fresh **engineering development** study. It is not a confirmation, and all independent sham-calibration and scientific confirmation populations remain sealed.

## Models, roots and splits

Train exactly eight **fresh independent model roots 800–807**; excluded smoke root **799** is used only for engineering. Use the unchanged NC-07 DenseLooper training contract: 20D dense shared physical BPD mix applied in four nonlinear tied passes, trainable encoder/preweight/head, 2,048 training samples, exactly 384 Adam updates, batch 128, learning rate 0.005. The external task is the same context-dependent NC-05/06/07 sign classification, with labels used for supervised training only.

Each root has separate seeded data: 1,024 **model-score derivative discovery** examples (split 2), 512 **rank-selection** examples (split 4), and 1,024 **external-label terminal evaluation** examples (split 3). Selection and shams must be fully fixed before final-label access. Admit only roots with >90% dense physical mix, >=85% clean heldout accuracy in each of two task contexts, and full-target BPD closure at every readout: maximum clean-vs-full logits <=5e-5, mean Bernoulli KL <=1e-7 nats. One independently initialized/trainable model is the unit of inference.

## Frozen source-only factor-rank allocation policy

In the original native full-rank BPD factor coordinates, for each of four passes t and each discovery source sample n, collect clean factor read z[n,t,:] and self-final-logit score derivative g[n,t,:]. The gradient sign is the detached model-prediction sign; outer products remain sign-invariant and **no external label enters projector discovery**.

At each t compute C_t=mean_n g[n,t]g[n,t]^T in float64, with eigvectors ordered descending by eigenvalue. The allowed per-pass ranks are **r_t in {2,3,4,5,6}**, with a fixed **total rank budget B=sum_t r_t in {8,10,12,14}**. For each pass/rank define P_t(r) as the top-r orthogonal eigenspace projector. An eligible rank must have eigenvalue lambda_r>1e-12 and eigengap lambda_r/max(lambda_(r+1),1e-12)>=1.001. Retain all per-pass spectral diagnostics.

The immutable source-only local proxy for pass t and rank r is

L_t(r)=mean_n [ g[n,t]^T (I-P_t(r)) z[n,t] ]^2.

It estimates the squared first-order effect on the model's self-score of discarding the complementary **read** coordinates at that pass. The proxy is computed on the derivative discovery source only and makes no assertion of exact nonlinear recurrence fidelity.

For each fixed total B, compute exactly one passwise rank tuple r(B) minimizing sum_t L_t(r_t) among eligible tuples with sum_t r_t=B. Enumerate all finite 5^4 possibilities deterministically and break equal-score ties (absolute difference <=1e-12) by lexicographically smallest 4-tuple. Missing budgets are marked ineligible; no relaxation is permitted. This is an exact finite multiple-choice knapsack, not a learned function or an external-label rank fit.

For each eligible r(B), build P_{t,r_t}, retain the projector on **all four tied BPD passes**, and evaluate its clean-vs-retained self-prediction on the **independent 512-example selection source**. Choose the first passing B in ascending order 8,10,12,14. For a candidate to pass, require (unchanged NC-07 selection gates):
- Mean final KL(clean||kept) <=0.027 nats.
- Mean KL at each of the four passes <=0.050.
- 95th percentile of per-example final KL <=0.250.
- Fraction of model final-logit sign flips <=0.040.
- Discovery clean retained write RMS / full physical write RMS >=0.350.
- Original eigengap/closure eligibility.

Any root with no passing budget abstains and fails engineering admission. The policy cannot upgrade to budget 16 or choose an unregistered tuple after examining task labels. Report every B candidate's tuple, proxy, source-only KL and failure reason.

Two fixed control projectors are evaluated on the **same final 1,024 evaluation inputs**: flat rank-three (3,3,3,3), total rank 12, and flat rank-four (4,4,4,4), total rank 16. Also evaluate the frozen discovery-proxy-optimal budget-12 tuple r(12) on heldout as a **same-total-budget allocation comparator** to flat rank3 (secondary descriptive only); if it is ineligible, retain that null. The all-on teacher is the clean oracle. This source predeclaration prevents test-label selection of successful budgets.

## Sham population, gauge and actual physical recurrence

Freeze **2,048** local same-pass-rank sham proposals per root at principal-angle grid {0.45,0.65,0.85,1.05} radians. For each candidate proposal, draw independent complement QR directions for each pass with that pass's selected rank. Use the same chosen angle across four passes. Seed a fixed NumPy source generator with 8,080,000+root; choose the FIRST eight proposals with discovery retained physical write RMS within +/-15% of the selected candidate's discovery physical dose. There is no test-label or score-based sham resampling, calibration, or candidate enlargement after outcome inspection.

The selected intervention runs the actual native BPDModel/TiedComponentLinear override-read physical write with pass-indexed P_t and gain 1.0, over four **closed-loop nonlinear hidden-state updates**. Zero residual bypass, no source label or test outcome in its construction. Check full-target native BPD closure and orthogonal physical gauge transport P'_t=R^T P_t R against the same native physical operator V P_t U. Relative native/rotated physical-map error <=1e-4 and max four-pass final logit error <=5e-5 on 64 selection-only inputs.

The sham gate requires exactly **eight source-qualified physical shams** for each selected tuple on all eight roots. A failed source match is an ineligible root; never expand 2048 proposals in the original experiment. Report every candidate count and actual heldout write displacement.

## Frozen independent-root engineering decision

On separately drawn 1,024 heldout inputs per root, compute independently labeled true task final NLL, accuracy, per-pass KL(clean||kept), and physical write displacement for the selected variable-rank method, flat rank3, flat rank4, budget-12 DP comparator and all eight selected-rank shams. Record all original outcomes and absence of selected ranks.

A registered **ENGINEERING_POSITIVE** requires all of:

1. **8/8** independent models meet clean task, full-mask closure, source-only selected budget, selected eigengaps, eight sham qualifications and physical native/rotated parity.
2. Mean selected total rank budget **<=12.5** and **at least 5/8** roots select budget B<=12. This is a meaningful >=3.5-dimensional saving versus four passes of rank four (total 16).
3. Mean selected minus clean external-label NLL <=0.05 nats AND 2,000-draw model-root percentile bootstrap 95% UPPER bound <=0.05.
4. Mean median same-rank-sham minus selected external task NLL >=+0.020 nats AND root-bootstrap LOWER bound >0.
5. Model-root paired bootstrap upper 95% on selected minus flat rank-four true-label NLL <=+0.020 nats.

Bootstrap RNG seed **808008**, sampling eight whole independently trained model roots, NOT individual prompts, passes, sham proposals or source snippets. Root-level eligibility is fail-closed. If any root fails, preserve the original summary verdict **ENGINEERING_NULL_OR_INELIGIBLE** and mark bootstrapped effect gates unscored; any posthoc all-root calculation must be separately labeled descriptive. No retrospective threshold movement, root exclusion, candidate budget rescue or alternative shams.

The source-only r(12) versus flat-rank3 comparison of externally labeled test NLL is a fixed **secondary contrast**, with no new significance claim; it directly audits whether the first-order gradient proxy allocates rank more usefully at equal total write dimension.

## Future confirmation hold and scope

The independent **59-root** matched-physical-sham calibration and **32-root** scientific confirmation populations stay completely ungenerated/unevaluated. They require a new source-registered confirmation protocol, a root-level EvalMath variation/power audit, and a fixed selected mechanism before access. NC-08 runs eight development models and engineering smoke root only. Protected ARG-03-N, AMM-03 and BPD-CQ-08 independent confirmation sets are untouched.

This experiment tests pass-specific *physical write rank allocation*, source-only model-self predictive KL, and future external-label task behavior in one trained nonlinear synthetic recurrent model family. It does not prove globally minimum-dimensional causal quotient, pretrained TRM proof reasoning, per-neuron mechanisms or general GL(20) gauge invariance.
