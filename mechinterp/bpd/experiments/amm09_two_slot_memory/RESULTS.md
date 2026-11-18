# AMM-09: bounded two-slot causal routing, prospective development and independent audit

**2026-11-18. Evidence class: positive prospective DEVELOPMENT effect in independently trained synthetic rank-three recurrent BPD, with cross-host numerical reproducibility unresolved.** All **128 AMM09 confirmation roots remain sealed**. Earlier AMM03–08 and ARG-03-N protected populations remain untouched. **A human-approved release manifest has not been issued.**

## Registration and prospective sequence

- [PROTOCOL.md](PROTOCOL.md) was registered before any AMM09 source or models at `f6a96069417265388c4e7cef4828a16bf2f8d4d3`. It discloses the AMM08 selection exposure and restricts masks to six nonempty subsets of at most two rank-three SVD components.
- Experimental `models.py`, `case.py`, `experiment.py`, the 16 engineering tests and staged CI were frozen at `742402734493d42951d82dd3f0a7becd1fa8462a` before any AMM09 training. The first hosted source-only run [GitHub Actions #44281274200](https://github.com/MoralityLabAI/BPD/actions/runs/44281274200) passed all 16 tests, including actual three-pass `BPDModel/TiedComponentLinear PassMaskSchedule` parity.
- The global policy's 64-root TRAIN result and candidate-bit digest were **committed at `fde1d40cf248738205ec1069ad61a593d10b387f` before any DEVELOPMENT model was fitted**.
- The original independent, once-scored DEVELOPMENT run [GitHub Actions #44281548742](https://github.com/MoralityLabAI/BPD/actions/runs/44281548742) reproduced the first-run training lock byte-for-byte, scored **64 disjoint development models**, computed the correct primary pilot input and received corrected EvalMath `READY_FOR_CONFIRMATION`.
- **The original `experiment.py.analysis` mislabels one comparison's numerator**. Its `memory_only_vs_pooled` key used `memory_plus_local` outcomes, while the preregistered primary and `preflight_input` correctly use `memory_only - pooled`. This is a post-scoring reporting bug. The experimental source and original `results/DEVELOPMENT_LOCK.json` remain unchanged. The independently written [analysis amendment](AMENDMENT_POST_DEV_ANALYSIS.md) and `audit_development.py` rebuild every comparison from original named treatment and control arms and verify every candidate-choice bit. The source-preserving [original-receipt audit workflow](https://github.com/MoralityLabAI/BPD/actions/runs/44283653555) downloads the exact original hosted evidence and verifies its raw file hashes before independent analysis.

### Frozen source, data and result provenance

| Artifact | SHA-256 |
|---|---|
| `models.py` | `2fec8382c067147d60d9f2d4fe8d171ce3d5e15f405afffbbab5389023bc1a03` |
| `case.py` | `f72788b058d7eff58218ed4e0c729437820a8cfdd2518609409f2fc7fec9132a` |
| `experiment.py` | `57af128ca48ef2dcd527abfcd443373705ecd046a460d53bafe2a0222df56897` |
| Frozen global training-policy SHA | `c536d7bda5c49507138708e6fd9e0187bbb057990248116aa73cfd40777e95ef` |
| Training candidate-bits SHA | `c46dc9aebe2834559ec18e0f936d2b2cbfff962553ed10905123199de555192d` |
| Complete development selected-action/candidate semantic SHA | `fe6f801c9788c9f5805dfb762a350357fef4513ba53d964bb1c418d7fbc21f55` |
| Original DEVELOPMENT raw JSON SHA | `0423c8302c9cbb99b9413ac7e22cfcc7d0f10d921eff06c0f4a17cda33ff3925` |
| Original hosted artifact ZIP SHA | `4affd7357053e31c8c43a62a2e5ea3aa751f647af9c779e66a35ff09829a37a6` |

All SHA values above are source/first-run data locks. The original Actions archive includes `amm09_training.json`, `amm09_development.json`, `amm09_pilot.json` and `amm09_preflight.json`. The post-scoring independent audit output `results/ANALYSIS_CORRECTED.json` is separate from the unmodified original raw development receipt.

## Registered intervention and matched baselines

Every independent model root trains a new genuine **6D rank-three, three-pass weight-tied physical recurrence** and a separate BPD V/U decomposition from three external pass-specific task labels. Fresh AMM09 SHA256 streams are disjoint from AMM01–08. Donor edits change the nuisance and two target source coordinates; a scored intervention has to repair **both** independently defined future donor readouts. The first-pass current-readout is kept invariant. Every admissible mask is projected current-dark and normalized to the same per-case physical L2 donor-write norm, followed by two learned tied recurrent continuations.

There are **exactly six admissible masks**: `(0),(1),(2),(0,1),(0,2),(1,2)`, with the triple mask entirely absent from training, test, local calibration, policy inference and the oracle. Each policy receives identical SVD-canonical 18-feature signatures/action and **14 unlabeled continuation readout values per test case**. Every new model root gets 16 local calibration pairs and six noisy candidate labels per pair, a fixed **96-label budget** with independent known p=.20 flips. Global strategy memory additionally pays the disclosed 64-training-root corpus cost. The strongest fixed *admissible* pooled mask is selected on the same training corpus as memory. The local-ridge and memory-plus-local arms receive identical 96 calibration labels.

## Frozen TRAINING policy

64 independent trained root models ×128 donor cases/root = **8,192 training cases per mask**. Global 18-dimensional ridge λ=10 with source-locked four-decimal coefficients; pooled mask chosen by training hit count is index **4**, `(0,2)`.

| Admissible mask | Training correct / 8,192 | Accuracy |
|---|---:|---:|
| (0) | 999 | 12.19% |
| (1) | 599 | 7.31% |
| (2) | 522 | 6.37% |
| (0,1) | 1,786 | 21.80% |
| **(0,2), pooled fixed baseline** | **1,801** | **21.98%** |
| (1,2) | 1,425 | 17.40% |

## Original prospective DEVELOPMENT outcomes

**64 independent newly trained model roots ×512 heldout donor cases/root = 32,768 scored cases per policy**, with 512 independent repeatability pairs/root for the same frozen pooled policy. Every generated root and test case is included. The registered **primary** is `memory_only - pooled` and compares the same training-world access at exactly the same intervention-slot and unlabeled probe budget.

| Registered / descriptive policy | Correct / 32,768 | Accuracy |
|---|---:|---:|
| **Cross-model memory_only, primary** | **11,278** | **34.417%** |
| Memory plus 96 noisy local feedback labels | 9,702 | 29.608% |
| From-scratch local ridge using same 96 labels | 7,533 | 22.989% |
| Strongest fixed permitted pooled mask (0,2), primary control | 6,436 | 19.641% |
| Local per-mask noisy hit count | 6,064 | 18.506% |
| Root-hashed per-case random allowed mask | 4,542 | 13.861% |
| Equal-dose isotropic random-dark A / B | 1,522 / 1,560 | 4.645% / 4.761% |
| Unpatched sham | 1 | 0.003% |
| Full donor state, explicitly **different dose** | 32,503 | 99.191% |

### Corrected four-comparison independent-root inference

The original source-named `analysis` key for the primary was wrong; all figures below are from the **separate independent raw-bit auditor**, which uses treatment/control definitions from the pre-data protocol. Every confidence interval resamples **64 independently trained model roots**, with 2,000 fixed-root bootstrap replicates.

| Registered contrast | Effect (percentage points) | Positive / negative / tied roots | Holm-adjusted one-sided root sign p | Paired-root bootstrap 95% interval |
|---|---:|---|---:|---|
| **memory_only − pooled (PRIMARY)** | **+14.777** | **55 / 8 / 1** | **1.9523 × 10⁻⁹** | **[+11.154,+18.661]** |
| memory_plus_local − pooled | +9.967 | 47 / 16 / 1 | .0001171 | [+6.299,+13.751] |
| memory_plus_local − local_ridge | +6.619 | 47 / 15 / 2 | .00008668 | [+4.315,+9.140] |
| memory_plus_local − memory_only | **−4.810** | 14 / 47 / 3 | .999996 | [−6.802,−2.930] |

The primary is significantly positive after all **four** registered comparisons are adjusted. Its root-bootstrap **lower limit +11.154pp exceeds the preregistered scientific +8pp minimum**, and its test-answer-independent EvalMath input already used the correct +14.777pp root-mean effect.

The effect is present in both source families: **wren +13.65pp**, **basalt +15.91pp** (descriptive subdivisions; both groups have 32 independently trained models). The memory-only strategy selects different actions from the pooled fixed mask on **21,459/32,768 cases (65.49%)**, choosing multiple admissible masks on **all 64 models**. Its chosen-mask counts by index are `{0:57,1:2,2:0,3:15007,4:11309,5:6393}`. This demonstrates case-conditioned physical selection, unlike AMM08's unrestricted full-mask collapse.

The *retrospectively chosen best fixed mask on DEVELOPMENT* is (0,1), with **6,801/32,768** successes; it also scores below memory-only's 11,278. This comparison is diagnostic because the fixed mask was chosen using test answers. The six-mask **audit-only casewise oracle** reaches **15,539/32,768 (47.42%)**, leaving **9,103 cases (27.78pp)** above preregistered pooled. No oracle or donor future can enter the agent policy.

Local feedback is a separate observed negative: memory-plus-local loses **1,576 cases (4.81pp)** relative to training-only memory. Its corrected registered secondary interval is entirely negative. This strengthens the case for an independent confidence-calibrated mechanism before permitting local updates.

## Physical integrity, fidelity and corrected EvalMath

- **63/64** development model roots pass the prespecified heldout teacher/factor fitting and singular-gap criteria (minimum 56); flagged root `development/0026` has factor 3-pass task MSE .06827 and relative weight error .34853. It remains in every intention-to-treat score. The 63 good-fit models show a descriptive +14.94pp primary advantage.
- Maximum equal-dose mismatch across candidate and random-dark arms: **4.7684e−7**; maximum immediate readout displacement **0**. Registered required bound 1e−5. Excluded engineering-root GL(3) shear, canonical rank-one reconstruction and native three-pass BPD execution parity passed all 16 tests.
- Calibration feedback flipped **1,221 / 6,144** local labels (**19.87%**), consistent with registered 20% noise.
- Matched repeatability on the *same fixed pooled policy* over 512 fresh donor pairs per independent root gave a nonparametric **95th-population-quantile/95%-confidence upper tolerance of 33/512 = 0.064453125** (6.45pp). With scientific +8pp minimum and 1/512 resolution, the mechanism floor is **+8pp**.
- The **corrected pinned EvalMath** `MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a` returned **`READY_FOR_CONFIRMATION`** from all 64 *correct* primary root effects, target +16pp, four comparisons and planned 128 new independent confirmation models. Nominal normal planning projected 44 roots for the *hypothetical +16pp* effect. This is an **admission decision**, not a scientific confirmation.

## Independent cross-host numerical replay limitation: scientific hold

After the initial development run, a later hosted CPU worker reran all **64 TRAIN model roots** with **identical source SHA256 values and SHA-derived root streams**. Its fitted teacher and BPD floating-point weights changed at machine precision on all 64 roots, affecting **27 / 49,152 individual training candidate success bits (0.0549%) across eight model roots**. The mask-specific bit flips were `mask_0:6, mask_1:2, mask_2:1, mask_3:6, mask_4:8, mask_5:4`. The global 18 ridge coefficients differed by as much as **0.0036** after the registered four-place coefficient rounding. The pooled best mask remained (0,2). This is **not an exact cross-host model-root replay**.

The registered original training-policy digest and candidate-bits lock correctly rejected that re-run **before scoring any new development root**. Both original and alternative training raw receipts are archived. This numeric portability failure is a substantive engineering limitation; it must not be silently remedied by loosening success thresholds, swapping in a different fitted policy, redefining independent roots or claiming bit-exact cross-host reproducibility.

The separate **original-source-of-record audit** downloads and SHA-verifies the original GitHub Actions raw development file and recomputes all four preregistered comparisons without importing or modifying any model-policy code. This independently verifies the original significant **development** result and its stated case choices, but it cannot by itself establish deterministic training across differing host CPU implementations.

**Confirmation remains sealed despite EvalMath READY.** The pre-data protocol additionally requires a *separately committed explicit human-approved source/policy/preflight-bound release manifest*. Such permission has not been given. A fresh deterministic-backend engineering test and a prospective independent-root validation will also be appropriate before considering a confirmatory inference. At present the current study provides **promising developmental causal evidence with an unresolved cross-host reproducibility issue**.

## Reproduction and evidence boundary

The source-frozen `experiment.py` and original `results/DEVELOPMENT_LOCK.json` intentionally retain their historical analysis numerator bug for byte-for-byte provenance; use `audit_development.py` and `results/ANALYSIS_CORRECTED.json` for preregistered inference. The CI original-receipt audit verifies the exact raw hashes, six-mask action boundaries and original named-case success bits; the separate strict training CI documents the cross-host numerical divergence.

All prospective and historical `confirmation` namespaces remain sealed. The result is restricted to **synthetically trained linear recurrent 6D rank-three BPD interventions under a rank-at-most-two physical slot budget**, without demonstrating transferable nonlinear transformer mechanisms, pretrained LLM agent cognition, live cooperating agents or deployed Buzz/Nostr memory.
