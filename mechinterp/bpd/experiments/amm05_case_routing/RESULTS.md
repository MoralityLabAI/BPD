# AMM-05: case-conditioned causal-response memory under mixed donor edits

**2026-11-18. Registered development result: directional effect, BELOW independent-root mechanistic-resolution threshold.** This is a trained synthetic linear BPD intervention experiment. The 128 prospectively reserved AMM-05 confirmation model roots remain **strictly sealed**. The AMM-03, AMM-04 and ARG-03-N protected populations also remain untouched.

## Protocol and source-before-data audit

- Pre-source protocol: [PROTOCOL.md](PROTOCOL.md) at commit `375a9510d63dbda09de6feecf1a441183f4c539c`.
- Pre-source three-unlabeled-readout accounting amendment: [AMENDMENT_PRE_SOURCE.md](AMENDMENT_PRE_SOURCE.md) at `7bee4fb35d16c8d5022c5102d6db3ea9b69b356d`.
- Source/tests/CI frozen at `07aa23e0c1616882da95556b55edcacfaaf10032` before any new training/development worlds were scored.
- First 64-training-model result and frozen policy SHA256 committed at `75f8a2d96276b2627d75281e3b99485940de0051` **before** the first development-score run.
- [GitHub Actions independent training/development run](https://github.com/MoralityLabAI/BPD/actions/runs/44238309747): 13 engineering tests including real native BPDModel/TiedComponentLinear PassMaskSchedule execution **PASS**, 64-root training digest **PASS**, development case/route output recorded, and pinned corrected EvalMath preflight **REDESIGN_OR_PILOT_ONLY**.
- Exact code SHA256: `models.py=32a5296e7b488e67b4350011748e3db8f3e507eea552e44249b24ca0dd6471ff`, `case.py=91c1230188f5f0551dc38fa1382796985f3c668ebe710e12700059fdc215b1b3`, `experiment.py=6f3df2911a64beaba325a3b8c001864300a72a9a3d18243d36fb030bc9db76cf`.
- Frozen cross-root memory `sha256=e9cfdbccf68d9209004ad99ac9527d127860ae06e171e4c68bdaff346eaa2fa5`. Complete independent development raw receipt canonical SHA256 `0192fa2fb4bcd84f714bab850929e7a8a5235dfebd0ea74ded59798f23c2c556`. Bit-level candidate/choice, noisy calibration and repeatability digest `7db132e014e0deaa66bdbde5b387326febd7c82d826f8e373f8c41eecfdf0850`.

## What changed from AMM-04

AMM-04 chose one BPD component for an entire trained model. AMM-05 changes both the target-relevant and nuisance input coordinates independently on each donor pair, which produces a *case-dependent physical direction* while keeping an independent external donor-target value fixed by the task source. Each new trained model has an independent rank-two physical teacher and separate BPD factors with a gauge-canonical SVD basis. The recipient receives one projected current-dark candidate BPD component write, matched to the full donor intermediate-state write's L2 dose, followed by another *learned weight-tied* pass.

An eleven-dimensional candidate causal-response signature uses only first-pass states and **three shared unlabeled future-readout probes per donor pair** (one unpatched baseline, two matched-dose candidate continuations). Both competing decision rules see the same features. **The donor's future continuation and externally known task label are withheld from both policies** and used solely to score and audit their already frozen decisions.

A global ridge policy (penalty 10) trains from 64 independent training model roots, 128 donor pairs each, with both component outcomes observed. Each development root receives *four new local calibration pairs*, two component readouts/pair, and **eight identical p=.35 noisy binary feedback labels** for the memory and local policies. The proposed rule adds a λ=20 local ridge correction to its frozen cross-root memory; the fully local comparator fits a new λ=2 eleven-feature ridge using exactly those same eight corrupted labels. Both choose **one component per heldout case**; the memory corpus is an explicitly disclosed additional amortized cost. No confirmation-root training, labeling or scoring has occurred.

## Frozen training model roots

Independent 64-root candidate results, 128 pairs/root (8,192 cases per candidate):

| Canonical BPD action | Correct training pairs | Hit rate |
|---|---:|---:|
| Component 0 | **3,153 / 8,192** | 38.49% |
| Component 1 | 2,060 / 8,192 | 25.15% |

The globally pooled baseline selects component 0. The frozen cross-root ridge learns the interaction between patch-specific delayed response, first-pass state contrast, physical dose, candidate/full-donor-write alignment and gauge-canonical weight geometry. The precise eleven ridge coefficients and all case-level training rows are in the archived raw training receipt; the source-locked digest is in `results/TRAINING_LOCK.json`.

## Development cohort: independent models, full ITT

**64 separately trained development roots × 128 disjoint heldout donor-recipient pairs/root = 8,192 scored cases per policy.** Each root also contains four separate noisy calibration pairs and 128 further independent *repeatability* donor pairs for the same fitted local policy. Every generated pair, including failures and negative effects, remains in the denominator.

| Frozen strategy | Correct / 8,192 | Accuracy |
|---|---:|---:|
| **Training-only case-conditioned memory** | **3,796** | **46.34%** |
| **Case-conditioned memory + noisy local ridge (registered primary)** | **3,641** | **44.45%** |
| **Equal-budget local ridge (registered primary comparator)** | **3,160** | **38.57%** |
| Four-pair noisy local component-count rule | 3,070 | 37.48% |
| Global training-pooled component | 3,286 | 40.11% |
| Per-case absolute delayed causal-response geometry rule | 2,041 | 24.91% |
| Per-case root-keyed random component | 2,608 | 31.84% |
| Equal-dose isotropic random dark A | 1,563 | 19.08% |
| Equal-dose isotropic random dark B | 1,705 | 20.81% |
| Unpatched sham | 0 | 0.00% |
| Full donor intermediate state, *larger dose* | 8,188 | 99.95% |

The primary independent-root effect is **+481/8,192 = +0.0587158** (**+5.87 percentage points**) for memory+local minus the equal-budget local ridge. Its 64 root directions are **38 positive / 12 negative / 14 tied**; the two-sided cluster-bootstrap inference was not used as the confirmatory p statistic. The registered one-sided exact root-sign p is **0.000152932**, Holm-adjusted for the three-comparison family to **0.000305864**. The paired root-bootstrap 95% percentile interval from the frozen 2,000 resamples is **[+0.019165, +0.099487]**. This interval is for independent model roots, not 8,192 pseudo-independent observations.

| Registered positive contrast | Root-mean gain | Root directions +/−/tie | Holm-adjusted sign p | Root-bootstrap 95% CI |
|---|---:|---|---:|---|
| Memory+local vs local ridge (primary) | **+0.058716** | **38/12/14** | **0.000306** | **[+0.019165,+0.099487]** |
| Memory+local vs noisy local-count | +0.069702 | 33/18/13 | 0.024437 | [+0.025513,+0.119873] |
| Memory+local vs case geometry rule | +0.195313 | 46/14/4 | 0.000063 | [+0.125732,+0.268677] |

Memory+local and local ridge selected different components on **2,317/8,192** test cases (28.28%); memory routed between both components within **58/64** model roots, and local ridge did so within **51/64**. An independent explicit audit compared the model-predicted donor full continuation to the *casewise test-label oracle*. The donor-continuation proxy achieved the test-label oracle's ceiling in **all 64 development roots**, scoring **4,532/8,192 (55.32%)**. This explains why accessing a donor full continuation would be a shortcut; it is **prohibited as a policy input** and counted only as an audit. The nondeployable casewise oracle sits **1,372/8,192 (16.75pp)** above the local ridge, so the AMM-04 test ceiling has been relieved.

Noisy local feedback reduced the frozen memory-only policy's accuracy on this cohort: 3,796 hits versus 3,641 with the additional noisy calibration update, a difference of 155 hits. That is a development comparison and is *not* one of the three prospectively registered inferential contrasts. The observed judge-noise flip count is **164 / 512** local labels across roots; all original clean calibration labels are stored separately as audit-only data.

## Integrity and causal limits

- Independent physical teacher behavioral task MSE maximum `6.00518e-8`; independent BPD factor task MSE maximum `0.090557`. All roots, including higher factor-error roots, are retained.
- Minimum rank-two SVD gap `0.079763`, above the registered separated-singular-value diagnostic threshold.
- Max active same-case L2 dose error `2.38419e-7`; immediate current readout displacement exactly `0`.
- On excluded engineering roots, full-product GL2 shears and re-canonicalized rank-one patch invariants pass. Actual repository `BPDModel/TiedComponentLinear` with `PassMaskSchedule` matches independent masked forward within 1e-5 in hosted CI.
- Separate SHA256 root namespaces and independent model/factor initializations ensure that **64**, rather than 8,192, is the scientific sample size.
- A full donor intermediate-state replacement is reported with its different physical dose and cannot serve as an equal-dose causal baseline.

## Corrected EvalMath preflight: confirmation blocked

The frozen [EvalMath mechinterp-resolution preflight](https://github.com/MoralityLabAI/EvalMath/blob/6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a/skills/evalmath-mechinterp-power/scripts/preflight.py) (merged PR #51) was rerun in hosted CI on all **64 independent development-root paired effects**, with no repeated-case aggregation masquerading as separate roots. The sham variation is the difference between the *same selected local ridge policy* on two independently drawn 128-pair sets in each root, at the same physical dose. Sixty-four independent sham roots give the sample maximum an upper 95/95 tolerance interpretation.

- Registered minimum meaningful scientific effect: **+0.08** (8pp); target: +0.20; measurement resolution: 1/128.
- **Nonparametric 95th-quantile/95%-confidence sham floor = +0.140625 (18/128, 14.06pp)**, which dominates the scientific and measurement floors.
- Observed independent-root mean effect: **+0.0587158**, below this calibrated variation threshold.
- Corrected planner's normal-power estimate: 79 independent confirmation worlds for the **hypothetical +0.20 target**, not for the observed +0.0587 signal; the registered independent confirmation population has 128 roots. Exact one-sided sign-test family requires at least six independent roots even in the ideal all-positive case.
- **Decision: `REDESIGN_OR_PILOT_ONLY`**, because `pilot_effect_not_above_mechanistic_variation_floor`. This **blocks any confirmation run** and any claim of an established transferable BPD mechanism. The 128 AMM-05 confirmation model roots remain untouched. The pre-data protocol/source is preserved; all results are retained.

The significant directional root-sign result shows that memory can change intervention choices in these trained synthetic models. The registered evidence standard additionally requires a physically meaningful effect outside root-calibrated uncertainty. AMM-05 did not meet that stricter criterion. A new study should reduce repeatability variability by improving the *causal instrument or independent model-root calibration*, and should investigate why unreliable local feedback sometimes reduces memory-only performance. Raising the repeated-case count alone would not increase the number of independent model roots.

## Reproduction and scope

`results/TRAINING_LOCK.json`, `results/DEVELOPMENT_LOCK.json` and `results/PREFLIGHT.json` pin source blobs, training policy, complete scored-case/selected-action hashes, independent cohorts and the corrected EvalMath gate. GitHub Actions regenerates the full 64-training and 64-development receipts in one CPU workflow, comparing locked digests byte-for-byte. The complete `amm05_training.json`, `amm05_development.json`, `amm05_pilot.json` and `amm05_preflight.json` are archived in the linked hosted workflow artifact.

This is a **synthetic trained linear recurrent BPD strategy-memory experiment**. It does not establish neuron-level identifiability, nonlinear transformer causal structure, actual LLM agent-to-agent learning, general mechanistic transfer across architectures or a deployed Buzz/Nostr memory protocol. AMM-03/04 and ARG-03-N sealed test cohorts were never accessed.
