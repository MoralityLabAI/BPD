# AMM-10: exploratory dissection of exposed AMM09R1 causal memory

**Analysis plan committed 2026-11-18 before computing any AMM-10 ablation metrics.** Evidence class: exposed training-root cross-fitting and retrospective development diagnostics. This is a **post-selection analysis** of an already-published study, not a preregistered independent causal confirmation. All AMM09R1 and historical confirmation populations remain strictly sealed. The original AMM09R1 model generator, source, global memory, 64 development roots and results must remain unchanged; no model-root fitting, no reproduction of development features and no alteration to any acceptance seal.

## Basis and immutable original receipts

Source of record: merged [AMM09R1 PR #117](https://github.com/MoralityLabAI/BPD/pull/117) at commit \`09b8739fe960e20e48ebcfe731995ffed9380843\`.

Two original GitHub Actions source-locked artifacts:
- Training: [run 44294944939](https://github.com/MoralityLabAI/BPD/actions/runs/44294944939), artifact \`amm09r1-training-ubuntu-22.04\`, member \`training.json\`, 16,143,086 bytes, SHA256 \`015f8c78181514a0702e56ad707444699e48b377d9bf29da8ec60a42a0d48d30\`.
- Development: [run 44295607653](https://github.com/MoralityLabAI/BPD/actions/runs/44295607653), artifact \`amm09r1-development-source-locked\`, member \`development.json\`, 1,868,507 bytes, SHA256 \`682473974fa15db6005dab588d66d73c2a58bba1d5c8f1f4a91a93899ea88e5a\`.

The auditor must require these original full-file hashes, the frozen training and development policy/bit locks, all 64 proper root names, exactly six admissible masks, and the exact original per-case action/outcome accounting. Training features are recorded per mask. Development features have **deliberately been omitted**; only their SHA256 commitments remain. Development may be used for already-observed-policy and best-fixed-mask diagnostics, **never rescored with an ablated policy**. No attempt to recreate or refit development models is permitted. Reject any input whose source root key contains \`confirmation\`.

## Mechanistic algebra (static, exact ranking identity)

For a candidate mask \`S\`, freeze the original 18-feature linear score \`s(S)=x(S)·beta\`; it is clipped to [0,1] before selecting the largest score. The original features at zero-index positions are:

\`0\` intercept; \`1\` |S|/3; \`2\` current recipient q1; \`3\` donor current q1 change; \`4\` physical dose; \`5\` Δq2; \`6\` Δq3; \`7,8\` absolute shifts; \`9,10\` squared shifts; \`11\` shift product; \`12,13\` raw patched q2/q3; \`14\` patch/full-donor cosine; \`15\` normalized SVD gap; \`16\` nuisance-dose ratio; \`17\` family bit.

Because raw readouts satisfy \`q2(S)=q2(unpatched)+Δq2(S)\` and similarly for q3, the exact **pre-clipping rank score differences** are determined by \`|S|/3, Δq2, Δq3, |Δq2|, |Δq3|, Δq2², Δq3², Δq2·Δq3, cosine\`, with coefficients \`[beta1,beta5+beta12,beta6+beta13,beta7,beta8,beta9,beta10,beta11,beta14]\`. Every other feature contributes a single per-case constant common to all six candidates. Monotone clipping may produce tied output scores. Check this identity on all **exposed training features** and log max absolute discrepancy and selected-action disagreements; never assume a numerical identity absent verified receipt evidence.

Distinguish removing a candidate-invariant feature from the **already fitted policy** (rank score algebra) versus refitting a ridge on fewer inputs (which can change learned coefficients). Family-bit removal from frozen score cannot itself change an unclipped pairwise ordering, although family removal during training can affect fitted parameters.

## Root-held-out training-data feature ablations

Use original **64 exposed TRAINING model roots**, each with 128 donor cases and six binary candidate outcomes. Assign fixed folds \`fold = root_index % 8\`. For each of eight folds, fit each variant to the **other 56 training roots** using the original two-action scoring data with *all six masks*, ridge λ=10 on slopes and free intercept, then score the eight heldout training roots. This yields **64 distinct out-of-fit roots** and 8,192 heldout training cases per policy. Every root is held out of the ridge that scores it; the entire 64-root corpus is already researcher-exposed and cannot support an independent confirmation claim.

Frozen refit feature selections, zero-based:

- \`full18\`: all 0..17.
- \`no_family\`: all except 17.
- \`no_alignment\`: all except 14.
- \`no_delayed_probes\`: 0,1,2,3,4,14,15,16,17.
- \`no_geometry\`: 0,1,2,3,5,6,7,8,9,10,11,12,13,17.
- \`linear_probe\`: 0,1,5,6,12,13,14.
- \`compact_causal\`: 0,1,5,6,7,8,9,10,11,14.
- \`cardinality_only\`: 0,1.
- \`alignment_only\`: 0,14.

No feature normalization, hyperparameter adjustment or model selection using heldout folds. Fit float64 ridge using the registered solver \`(XᵀX + diag(penalty))^{-1}Xᵀy\`; if the solver is unstable, stop and report failure. Do not quantize fold-fit coefficients: they are new exploratory estimators, not the frozen AMM09R1 policy. Match original prediction-time probability clipping and \`argmax\` tie-to-lowest-mask behavior.

Preselected analytic control rules using **only candidate-safe inputs**, with the same six masks and no fitted parameters:
- \`max_cosine\`: largest x14, ties lowest.
- \`max_cardinality\`: largest x1, ties lowest.
- \`max_balanced_shift\`: largest \`min(x7,x8)\`, ties lowest.
- \`max_total_shift\`: largest \`x7+x8\`, ties lowest.
- \`max_signed_shift_product\`: largest x11, ties lowest.
- \`max_shift_norm\`: largest \`x9+x10\`, ties lowest.
- \`pooled_fold\`: training-selected globally best fixed admissible mask on 56 in-fold roots, then score the eight heldout roots.
- \`family_pooled_fold\`: best fixed admissible mask per family among 56 in-fold training roots, then score heldout roots of each family.
- \`root_fixed_hindsight\`: best fixed mask **using that root's heldout answers**, clearly labeled audit-only oracle and never a decision policy.
- \`casewise_oracle\`: success on the best of six candidates for each heldout case, also audit-only.

Retain every root, mask, case bit, model family and training-fold assignment. For each arm output aggregate hits, percent, per-root scores, independently descriptive paired-root bootstrap 2,000 CI vs pooled_fold, and root-sign directions. The full comparison family is exploratory, with **no posthoc significance or multiplicity-adjusted confirmatory success claim**.

## Exposed development diagnostics

Using the immutable DEVELOPMENT outcome/action receipt only:

1. Reconstruct and verify original \`memory_only\`, \`pooled\`, \`memory_plus_local\`, \`local_ridge\`, \`local_count\`, and \`random_mask\` selected bits, and verify the baseline totals 12,502 vs 8,038.
2. Compute the best fixed admissible mask for **each individual development model root** using all its own 512 heldout outcomes. Report this as **hindsight structural ceiling** and the number of roots where original memory-only beats that ceiling's fixed choice (possible because memory switches per case).
3. Compute a **family-specific pooled policy** selected solely from the original 64 training roots' candidate-hit counts, then evaluate its two fixed choices on the already exposed 64 development roots. It receives the same amortized training corpus as frozen memory and uses only the public family bit, no test labels.
4. Report memory-only's per-mask action histogram by family, the fraction of roots with multiple actions, how often it changes from pooled and how much of its advantage survives versus family_pooled.
5. Report hindsight best **single fixed mask chosen from development** across all roots as a selection-exposed diagnostic and retain each mask's development hit count.

Development is source-locked and has **no full feature matrix**, so **no new feature-ablated development policy evaluation** is authorized or possible from retained receipts. Infer no nonlinear/TRM/generalization results from these diagnostics.

## Statistical/evidence boundary

Independent unit: one AMM09R1 trained model root, n=64. Nested repeated cases: 128/root for training CV, 512/root for development diagnostics. Physical dose: the original registered equal-L2 current-dark mask. Scientific minimum: AMM09R1's +8pp; original prospective pooled-repeatability matched sham and corrected EvalMath preflight remain the **only** registered scientific admission evidence. AMM10 reuses already-seen roots and has no independent prospective sham/cohort, therefore **no new confirmation or EvalMath READY claim may be derived from AMM10 exploratory effects**. Preserve historical EvalMath receipt and label new intervals descriptive. Disclose every variant, including negative outcomes.

A future prospective AMM11 test must preregister any chosen simplified policy or geometric heuristic, freeze its source, give the same global/local budgets to controls, and draw entirely new root populations. All AMM09R1 and earlier 128-root protected confirmation populations remain sealed.
