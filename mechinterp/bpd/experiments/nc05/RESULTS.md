# NC-05: rank–dose frontier in new dense trained recurrent models

**Date:** 2026-11-18 UTC. **Scientific evidence:** source-registered, synthetic **engineering-only** rank–dose frontier. **Prospective protocol-only commit:** `f7f1d2346f9d3f871049a3d03416ef7980fe90f8`, prior to NC-05 implementation/data.

**Hosted CI:** [GitHub Actions 44238771681](https://github.com/MoralityLabAI/BPD/actions/runs/44238771681) passed all **10 source tests**, one-root reduced smoke and the full eight-root registered CPU development job. **Raw artifact:** [14398507457](https://github.com/MoralityLabAI/BPD/actions/runs/44238771681/artifacts/14398507457) contains `NC05_RANK_DOSE_SUMMARY.json` plus **all eight complete root receipts** `nc05_root_500.json` through `nc05_root_507.json`.

- Experiment source SHA-256: `84e84c4d9728d19b065434b9f90de34aea5d59544fbebe12c48b129848903c44`.
- Frozen protocol file SHA-256: `36ad03b7e6e79ea4821facefb5aaf4247551ed920dfbe40b6e72125993cfc070`.
- Downloaded complete ZIP SHA-256: `4c1f7d5e54faa2d0f28133e4e36c6be7f9f1c470f31503979271ea36841faa09`.

## Registered population

**Eight independently trained full-density, freely learned nonlinear recurrent models**, seeds **500–507**, with separate streams of **2,048 train, 1,024 discovery and 1,024 held-out evaluation examples per root**, exactly 384 Adam updates. Four shared physical `mix` uses per model. External labels enter train and held-out outcome scoring only; the projector is discovered from the model's final logit and uncentered gradient second moment.

All **8/8** roots passed context-specific task accuracy ≥85%, model density ≥90%, target/full-mask BPD parity and the zero-output-head gradient null. The minimum observed `mix` density was **99.75%**. The mean clean-model accuracy increased with recurrent depth:

| Readout pass | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Held-out accuracy | 77.03% | 84.51% | 92.57% | **95.56%** |

## Frozen eight-cell comparison and outcome

The primary endpoint is the **true-label NLL increase of rank-r, gain-α retained model relative to the intact model**. The sham contrast is the median external-label NLL of eight physical-RMS-matched *local rotated* rank-r projectors minus that of the discovered rank-r projector. Both are in nats. Intervals below bootstrap independent model roots, 4,096 resamples; the **99.375% two-sided cell intervals** are Bonferroni adjusted for all eight declared rank × gain comparisons.

| Rank | Gain | Eligible roots | Mean NLL cost vs clean | Family-adjusted interval for cost | Mean NLL advantage vs shams | Family-adjusted interval for advantage | Joint registered gate |
|---|---:|---:|---:|---|---:|---|---|
| 2 | 0.75 | 8/8 | +0.114416 | [0.032828, 0.210814] | +0.049971 | [0.031988, 0.070401] | FAIL |
| 2 | 1.00 | 8/8 | +0.168964 | [0.060706, 0.291329] | +0.063834 | [0.038458, 0.095379] | FAIL |
| 4 | 0.75 | 8/8 | +0.027042 | [0.001783, 0.064816] | +0.058544 | [0.046696, 0.069530] | FAIL |
| **4** | **1.00** | **8/8** | **+0.021219** | **[0.001898, 0.045398]** | **+0.052182** | **[0.031546, 0.072910]** | **PASS** |
| 6 | 0.75 | 8/8 | +0.011054 | [-0.005686, 0.026609] | +0.068435 | [0.051063, 0.087459] | PASS |
| 6 | 1.00 | 8/8 | -0.001259 | [-0.002795, 0.000921] | +0.061038 | [0.038857, 0.079591] | PASS |
| 8 | 0.75 | 6/8 | +0.002788 | [-0.012197, 0.016167] | +0.048891 | [0.033295, 0.063182] | **INELIGIBLE** |
| 8 | 1.00 | 6/8 | -0.000925 | [-0.001398, -0.000589] | +0.035462 | [0.009522, 0.052990] | **INELIGIBLE** |

**Prospectively registered engineering verdict: `ENGINEERING_RANK_DOSE_FRONTIER` with smallest qualifying full-gain rank = 4.** The rank-4/full-gain upper family-adjusted fidelity bound **0.045398 ≤0.05** and lower sham-contrast bound **0.031546 >0.02**, with eight eligible roots. Rank 6 passes both full-gain and 0.75-gain gates. Rank 4 at 0.75 gain fails its upper fidelity interval despite a low mean fidelity cost.

### Matched physical-sham admission remains a substantive limitation

The 1,024-candidate **local angular null**, with a fixed rank-specific 15% discovery write RMS tolerance, admitted:

| Rank | Qualifying candidates out of 1,024, by root 500–507 |
|---|---|
| 2 | 111, 145, 114, 111, 88, 61, 116, 130 |
| 4 | 79, 66, 23, 99, 58, 44, 57, 36 |
| 6 | 36, 18, 53, 92, 61, 24, 43, 14 |
| 8 | 37, **1**, 21, 129, 27, 14, 28, **6** |

The rank-8 failed physical-dose control in **roots 501 and 507** (one and six eligible candidate shams). Those roots were **not scored as eligible** for either rank-8 gain. The six-root rank-8 means/intervals are reported as conditional diagnostics only; they do not satisfy the eight-root gate, irrespective of their clean-model proximity. No sham dose tolerance, candidate count, null family, or rank gate was changed in response.

## Primary rank-4 full-gain effects by independent root

| Root | Kept-minus-clean true-label NLL, nats | Median sham-minus-kept true-label NLL, nats |
|---|---:|---:|
| 500 | +0.02840 | +0.05867 |
| 501 | +0.06790 | +0.07334 |
| 502 | +0.03166 | +0.08825 |
| 503 | -0.00317 | +0.04426 |
| 504 | +0.00221 | +0.01437 |
| 505 | -0.00111 | +0.05107 |
| 506 | +0.00586 | +0.02710 |
| 507 | +0.03802 | +0.06040 |

Rank-4 full-gain mean held-out accuracy is **94.46%**; rank-6 full gain is **95.58%**, compared with the **95.56%** intact model mean. Rank-2 full gain retains **91.47%**. A slightly negative kept-minus-clean external NLL can occur because projecting a trained model acts as regularization on this external task; it is not the same quantity as KL(clean||projected), which remains nonnegative.

## Physical gauge controls and interpretation

All statistically eligible cells were scored in **both native and orthogonally rotated exact BPD factor bases**, with the **same physical shams**, held-out examples, shared training weights and rank/gain settings. Maximum projector physical-map relative discrepancy in raw rank diagnostics across the eight roots and all four ranks was **3.31×10^-7**, well below the frozen 1e-4 requirement. The source records rotated-vs-native complete-rollout output differences separately.

The NC-05 primary sham family is **local angular perturbations around the discovered rank-r subspace**, with θ ∈ {0.45, 0.65, 0.85, 1.05} radians and discovery-write RMS matching. NC-04 used an activation-PCA Haar null. Consequently, **their numerical matched-sham effect sizes cannot be treated as a controlled longitudinal comparison**. NC-05 directly measures the rank and gain frontier on fresh roots under one fixed local-sham protocol.

The rank-4 result supports a compact, causally useful **synthetic BPD read subspace** at a single physical shared linear write site. It does not establish a four-dimensional whole-model circuit, learned SAE component identifiability, Goodfire's native-neuron community recovery, LLM semantic structure, or out-of-family model generalization. The encoder, nonlinear preweight map and head remain intact throughout interventions.

A Bonferroni-corrected root-bootstrap interval on **eight closely related model roots** is a small-sample engineering diagnostic. It is not calibrated for general populations, and eight independent roots cannot provide a nonparametric 95th-sham-percentile 95%-confidence bound (≥59 independent roots needed). The NC-05 confirmation population remains **unregistered and scientifically sealed**. Before any confirmatory claim, preregister fresh roots, run EvalMath's mechanistic-power preflight with independent matched-sham variance, fix the full rank/gain comparison family, and retain the protected ARG-03-N and AMM-03 seals.
