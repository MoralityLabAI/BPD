# NC-13: post-result review of the next research step

**18 November 2026. Post-result engineering interpretation.** This review reads the original sixteen receipts and their archived calibration queries. It adds no model evaluations. The registered protocol, source lock and original outcomes retain their original status.

The next research task should establish **joint delivered-dose and control feasibility on exposed or excluded engineering material**, with a separately frozen method. A fresh behavioral cohort would currently face the same unresolved admission problem.

## What the original study establishes

The complete sixteen-pair context-E comparison gives mixed-minus-shared NLL **−0.02644280 nats**, with the registered nominal 99.1667% interval **[−0.05702703, +0.00034636]**. It fails the required lower bound above +0.01. Shared-context source discovery therefore supplies no admitted improvement under the E policy in this study. The interval includes zero.

Every D endpoint is ineligible. Shared D matched in 8/16 pairs; target-native D matched in 15/16. Only 40/128 fixed shams matched, and every pair lacked at least one of its eight required shams. The original verdict remains `ENGINEERING_NO_RECOVERY_OR_INELIGIBLE`.

Source: `NC13_ORIGINAL_SUMMARY.json`, SHA-256 `99b89e34b2d9ecda9606d5c20471b049a704f1b027b04a2cd42228b50fe45eca`; original `root_2601300.json` through `root_2601315.json`. The summary records each root receipt's hash. The [protocol](PROTOCOL.md) defines the endpoints and admission rules.

## What the failed searches reveal

Across 176 D calibrations, 68 matched and **108 stopped with `NO_REGISTERED_BRACKET`**. All failed searches completed 17 finite grid queries. Every search that entered the 24-step refinement matched; one sham matched directly on the grid. Total search cost was 4,600 dose-pool rollouts, with a maximum of 403 per pair against the registered 451 limit.

| Failed arm family | Failures | Position of every sampled displacement relative to its anchor | Smallest sampled excess, min / median / max |
|---|---:|---|---|
| Shared source | 8 | Above anchor | 0.734% / 4.207% / 10.580% |
| Fixed shams | 88 | Above anchor | 0.191% / 7.791% / 29.083% |
| Permuted alignment | 11 | Above anchor | 2.166% / 11.090% / 44.476% |
| Target-native | 1 | Below anchor | Largest sampled displacement remains 2.962% below anchor |

All failed curves have an interior minimum and decrease then increase across the sampled grid. The target-native exception occurs at root 2601310. Extra refinement steps would leave these recorded failures unchanged because their searches never obtained a sign-changing bracket. These samples leave behavior between queried gains, and beyond the registered bounds, unresolved. They provide no global impossibility certificate. [Frozen search implementation](dose.py)

A post-result calculation intersects the sampled displacement ranges for shared, target-native and all eight fixed shams. Fifteen pairs have an overlap above their original anchor. Root 2601310 has none: its largest sampled target-native displacement is **0.559284**, while one sham's smallest is **0.582090**. The mixed projector's gain was fixed at 1, so its alternative displacement range was never measured. This calculation motivates a future common-dose feasibility rule; it supplies no replacement NC-13 score.

The seven pairs admitting both shared D and target D also expose why absolute losses matter. Their mean shared NLL rises by **0.00147073** from E to D, while target-native NLL rises by **0.01701151**. Greater comparator degradation accounts for the descriptive **+0.01554078** gap reduction. These selected seven pairs support diagnosis only. Their admission-based selection changes the population from the registered sixteen-pair cohort.

## A concrete next task

Register an exposed-material CPU feasibility study whose endpoint is complete control admission under a fixed access budget. Use the archived search traces to specify its candidate method before any new model query. Treat the common dose and the sham generator as a joint design choice. A changed dose anchor requires calibrating the mixed arm too; the resulting estimand and additional target access belong in the new protocol.

Candidate construction should seek eight controls that satisfy the declared delivered-dose tolerance and a separation requirement. Record each proposal and every rejected search. Establish the complete control family's admission rate across the frozen engineering roster before choosing a fresh behavioral population. An optimizer's exhausted search remains a bounded failure unless an independent certificate establishes infeasibility. The original NC-13 gain bounds and failures stay in its record.

Preserve the direct target-native comparator with the same access and continuation horizon. Any later derivative-matched control consumes an additional, explicitly budgeted kind of target information. A future causal endpoint should measure the complete native response at the registered intervention radius, with competence and identity replay checked first.

## Connection to the LoopedTransformers studies

[LGA-09's original report](https://github.com/MoralityLabAI/LoopedTransformers/blob/6b4ecbbcced768b60596a8138b210250aa81d33e/mechinterp/loopscope12/RESULTS_FIRST_NATIVE_20261118.md) supplies an executed full-response control. Physical predecessor/goal edits in a cyclic-copy TRM beat eight shams matched on all ten future-logit first derivatives by **17.68, 21.88 and 23.83 percentage points** across three historically exposed teachers. All 1,536 source-goal cases admitted their controls. The independent checkpoint population remains those three teachers; the source-goal cases cluster within base strings.

NC-13 transfers a rank-four read projector between separately trained scalar-output models and measures task NLL. LGA-09 edits a physical state inside a teacher, then scores a legal donor target through native h2-to-h4 continuation. Their effect sizes and populations retain separate interpretations. LGA-09 motivates studying finite causal response after a matched local derivative, while leaving curvature and higher-order dynamics unresolved.

[LGA-10's excluded feasibility report](https://github.com/MoralityLabAI/LoopedTransformers/blob/6b4ecbbcced768b60596a8138b210250aa81d33e/mechinterp/loopscope13/SOURCE_FEASIBILITY_RESULT_20261118.md) added a scalar margin-curvature constraint and a nontrivial angle requirement. Its untrained model with synthetic energy admitted complete eight-sham families in **28/64 cases**, below its 95% gate. Thirty-six cases exhausted the fixed random-plane search. The shared lesson concerns establishing a usable control before behavioral inference; each study's observed failure belongs to its own constraints and search procedure.

Keep LGA-10 trained source 31064 held. Reserved LGA source 31059, ARG-03-N's 168 cases and the AMM, CQ and SUB protected populations retain their seals. This review opens no fresh cohort or pretrained compute.
