# AMM-07: replicated-label identification and casewise causal routing

**2026-11-18. Source-locked, prospective DEVELOPMENT result.** Evidence class: independently trained synthetic rank-two recurrent BPD with a shared noisy judge. The causal-routing benefit of independent reliability information was **below the preregistered mechanistic-resolution floor**. The 128 prospective AMM-07 confirmation models remain **sealed**. The AMM-03/04/05/06 and ARG-03-N protected populations are unchanged.

## Registration, execution and numeric source audit

- Protocol registered first at `ae41cbf635c759283e90e568954f20338acd5439`, before any AMM-07 source or model training. It fixes the two duplicate-label channels, 16 calibration pairs, 1024 scored cases/root, root-level effects, 8-point scientific floor and all vetoes.
- Initial source + excluded engineering tests + CI froze at `5d50271bc6063275c283eed71d3b80fdcee4d463`. The first [training-only hosted run](https://github.com/MoralityLabAI/BPD/actions/runs/44273845237) passed 15 source tests and trained exactly 64 new independent model roots.
- A second hosted training replay found matching *binary candidate outcomes and root identifiers* but slightly different float ridge coefficients (maximum difference about 1.5e-7) from numerical model-feature variation across CPU workers. It failed the original whole-JSON memory SHA. **No development models were scored.**
- [AMENDMENT_TRAINING_NUMERIC.md](AMENDMENT_TRAINING_NUMERIC.md) was committed at `82e712d2`, before any development experiment. It prescribes four-decimal canonical global ridge coefficients and an immutable memory ID derived from policy metadata, source and exact training candidate bits. Original pre-repair lock remains preserved as `results/TRAINING_PRE_NUMERIC_REPAIR.json`.
- Repaired frozen source and excluded-root tests passed at [Actions run 44274880670](https://github.com/MoralityLabAI/BPD/actions/runs/44274880670). **All 16 tests passed**, including actual native `BPDModel/TiedComponentLinear` `PassMaskSchedule` parity. The replacement, reproducible training policy was committed at `c80b24b2ebc2b777c62b11c95369a2cc8e530842` **before the first development scoring**.
- First independent [development + EvalMath run 44275141774](https://github.com/MoralityLabAI/BPD/actions/runs/44275141774) trained 64 new development roots and saved every scored case and all 64 root-level quality classifications. No confirmation generation occurred.

Exact AMM-07 frozen model/experiment source SHA256:
- `models.py` `b28915f0c00bf39299c5ddfe26c444b510150588700b3566d5c58d2adf983608`
- `case.py` `a2fd168b3e609a759d5721a10b9906e820b254cc9026135683fd8ab14fd47817`
- `noise.py` `34117a86e03ea19f5c279662c97e5a68a12fefdb79f12711ff6e9ee468794f43`
- `experiment.py` `735f9c8a698176bf88e43c564005577e4a110de99f77f22f35b33fca4c0658a9`
- Frozen training policy SHA256 `1ead62295b7201378e53318d6e6501ce850cf0ffb4ae8b45c33a59217e4118bb`.
- Full development scored-case/selected-action/duplicate-feedback semantic SHA256 `35dd3d5e78c2f206f51ad1220063b10166903f16abb05dbb46759021ef5ca58a`. The individual first-run raw `amm07_dev.json` file SHA256 is `8657d358dfba07b0747ebe42afe8bb8b9e06c3d3d99cff98d9716085fae154e5`; its container ZIP SHA256 is `d911f75f5227ebfa1f4f105576c965b8eefec2133bc7aba71d35520e75b699d0`. Cross-worker exact replay is judged on behavior, policy and source identities rather than on nonportable least-significant floats.

## Independent duplicated feedback and the causal intervention

Each root independently fits a 4D rank-two weight-tied physical teacher and a separately trained BPD V/U decomposition, using external two-pass task labels. An intervention chooses one current-dark, equal L2-dose SVD-canonical component patch after pass one and scores the donor's external second-pass target after the learned shared transition. All pair-level failures remain in the intention-to-treat denominator. The full donor intermediate-state replacement is a larger-dose ceiling.

Each root receives **16 independent calibration donor pairs**, two candidate component interventions per pair, with **two independently corrupted binary reports per candidate**: 64 label reports, identical for every strategy. Root quality q is hidden and is either 10% or 45% error probability, balanced with family. The first eight calibration pairs provide 16 duplicate-disagreement indicators for estimating q; the other eight support source-disjoint within-root policy updates.

From independent bit flips, `P(report1 != report2 | q)=2q(1-q)`: this is exactly `0.18` for q=.10 and `0.495` for q=.45. With equal prior q weights and 16 duplicate pairs, Bayesian MAP classification labels LOW noise when disagreement count `D<=5`. Under the declared sampling law its exact finite-binomial expected quality-classification accuracy is approximately **91.73%** (94.73% conditional on low noise, 88.74% conditional on high noise). This is a mathematical synthetic-channel calculation, not a neural result.

The **quality-aware** arm updates its frozen global ridge memory from the last eight calibration pairs using a soft Bayesian posterior for each unknown clean success label, marginalized with the quality posterior from duplicate disagreement. The **quality-blind** primary control uses the *same eight fit pairs and all the same observed feedback* but fixes q weights at (0.5,0.5). The strong **blind-all** comparator uses *all sixteen* calibration pairs for its fixed-mixture update. Additional arms include memory only, source-free local ridge using all sixteen pairs, fixed pooled component, random-dark, sham and larger-dose full-donor ceiling. All agents receive the same three unlabeled candidate readout probes per test case; no policy accesses hidden q, clean calibration truth, donor future or heldout task success.

## Frozen training memory

64 independent TRAIN model roots ×128 donor pairs/root, both candidate outcomes scored. There are **8,192** distinct training donor cases per candidate.

| SVD-canonical component | Correct TRAIN cases | Training rate |
|---|---:|---:|
| Component 0 | 3,088/8,192 | 37.70% |
| Component 1 | 1,787/8,192 | 21.81% |

The global pooled policy selects component 0; the frozen eleven-dimensional ridge weights use `lambda=10` with four-decimal canonical precision, preserving the same trained behavioral labels across independent hosted replays.

## Registered DEVELOPMENT results: 64 independent model roots

Each model root receives 1024 fresh heldout test pairs and 1024 separately drawn repeatability pairs. Thus **65,536 heldout test cases per policy**, with all feedback, candidate action choices and outcomes retained. The scientific independent-unit count is **64**.

| Frozen policy | Correct / 65,536 | Accuracy |
|---|---:|---:|
| Fixed-mixture **blind-all** (strong comparator, all 16 fit pairs) | **30,618** | **46.72%** |
| **Quality-aware** (registered primary) | **30,367** | **46.34%** |
| Fixed-mixture **quality-blind** (matched last eight fit pairs) | **30,167** | **46.03%** |
| Training-only **memory** | 30,051 | 45.85% |
| Source-free **local ridge** | 28,533 | 43.54% |
| Training-pooled fixed component | 26,505 | 40.44% |
| Random component | 20,442 | 31.19% |
| Equal-dose random-dark A / B | 13,153 / 13,163 | 20.07% / 20.09% |
| No-patch sham | 0 | 0% |
| Full donor state (different dose) | 65,536 | 100% |

The preregistered primary effect, **quality-aware minus matched-fit quality-blind**, is **+200 / 65,536 = +0.00305176** (+0.305 percentage points).

| Registered one-sided independent-root comparison | Mean effect | Positive / negative / ties | Exact root sign p | Holm-adjusted p | Root-bootstrap 95% |
|---|---:|---|---:|---:|---:|
| Quality-aware vs quality-blind (primary) | **+0.003052** | **17 / 17 / 30** | **0.567917** | **1.000** | **[−0.000061,+0.007034]** |
| Quality-aware vs stronger blind-all | −0.003830 | 15 / 19 / 30 | 0.804236 | 1.000 | [−0.008163,−0.000336] |
| Quality-aware vs memory only | +0.004822 | 20 / 13 / 31 | 0.148103 | 0.444310 | [−0.001221,+0.013977] |

The numeric accuracy gain of the primary arm comes from **17 benefiting roots and 17 harmed roots** with 30 ties. Repeated donor cases cannot be used as additional independent roots; the registered sign test gives no causal routing advantage. The stronger blind-all training variant obtained 251 more correct donor cases than quality-aware on the same test set. A descriptive root-bootstrap interval for that **negative** comparison excludes zero, though it does not satisfy the registered *positive-direction* hypothesis.

The policy's component choice differed from quality-blind on **1,123 of 65,536 test cases** and at least once on **52 of 64 model roots**. Actual correct/incorrect outcome bits differed on 512 paired cases.

### Judge reliability and factor-fidelity audit

The theoretically identifiable noisy channel was recovered:

| Hidden root channel (audit only) | Correct reliability classifications |
|---|---:|
| Low noise, q=.10 | **32/32** (100%) |
| High noise, q=.45 | **27/32** (84.38%) |
| **Total** | **59/64** (92.19%) |

This exceeds the preregistered reliability prerequisite of **52/64**. The recorded first-half duplicate-disagreement means were **2.844** under low noise and **7.688** under high noise, compared with theoretical expectations 2.88 and 7.92.

Per-channel outcome rates for the principal arms are similar in direction. On low-noise roots, quality-aware scored 49.32% vs quality-blind 49.01%; on high-noise roots, 43.35% vs 43.05%. This is descriptive stratification after the registered overall comparison.

The separate factor-fidelity prerequisite also passed **64/64** roots. Maximum teacher task MSE was `3.03e-8`; maximum factor task MSE `0.0001049`; maximum relative reconstructed-weight error `0.01632`; minimum canonical SVD singular gap `0.02597` (above .01). Maximum active physical dose mismatch was `2.384e-7`, and immediate current-readout displacement was exactly zero. Engineering native BPDModel execution and GL(2) canonical gauge checks passed.

## Corrected EvalMath preflight and exact stop

The preregistered independent-root 95/95 matched sham is the difference between the *same frozen quality-blind strategy* applied to 1024 primary and 1024 independently generated repeatability cases within each development model root. The 64 independent-world maximum is the nonparametric upper tolerance bound for the population 95th quantile at >=95% confidence.

- **Sham variation floor:** `52/1024 = 0.05078125` (5.08 percentage points).
- **Scientific minimum:** `0.08` (+8pp), dominating sham and measurement resolution `1/1024`.
- **Registered target:** `0.15` (+15pp).
- **Observed independent-root mean effect:** `0.00305176` (+0.305pp), root effect standard deviation `0.014734`.
- **Corrected pinned EvalMath** `MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a` returns **`REDESIGN_OR_PILOT_ONLY`**, with `pilot_effect_not_above_mechanistic_variation_floor`.
- Two additional audit prerequisites passed: quality classifier 59/64 >=52 and fidelity 64/64 >=56.
- **Third veto fails:** audit-only casewise oracle headroom above quality-blind is **5,161/65,536 = 7.875%**, just below the prespecified **8%** minimum. That oracle is based on hidden heldout labels and cannot be used as a policy.
- The 128 AMM-07 CONFIRMATION roots remain **unscored and sealed**. The preflight and oracle-headroom failure prohibit constructing a release manifest or describing the result as confirmation.

The improvement in identifying q **does not translate into the registered meaningful behavior effect**. This supplies a clean separation between *calibrating an external feedback channel* and *learning a better model-specific causal intervention policy*. In this synthetic two-component setting, a broad fixed-mixture training update that uses all calibration cases was more useful than allocating eight pairs solely to root-quality classification.

## Reproduction and next experiment

The `results/TRAINING_LOCK.json`, `DEVELOPMENT_LOCK.json`, and `PREFLIGHT.json` files lock the exact semantic model-policy, root and action-bit records. A dedicated GitHub Actions workflow replays the source, the 64 independently trained training models, and the 64 development models using the same corrected EvalMath planner. A raw 12.6 MB hosted receipt archive contains `amm07_train.json`, `amm07_dev.json`, `amm07_pilot.json` and `amm07_pf.json`, with the separate full file hashes above.

The next candidate should test **per-case confidence in causal-response features**, retaining a fully label-budget-matched strong blind-all comparator, or seek a richer model family where learned BPD intervention choices have more than the present ~7.9pp remaining oracle headroom. New root populations and a separately registered target are essential. The AMM-07 negative study and historical training numeric-replay failure remain in Git for audit transparency.

The result is restricted to synthetic trained 4D linear recurrent BPD interventions. It cannot establish neuron-level identifiability in pretrained language models, nonlinear/TRM causal transfer, actual collaborating agents, or a live Buzz/Nostr system.
