# BPD-SF03: independent-root gradient-matched confirmation

**Prospective hosted result (2026-11-17), source-frozen before calibration.** The 59 calibration and 32 confirmation roots were executed in separate [calibration workflow](https://github.com/MoralityLabAI/BPD/actions/runs/44120290966) and [confirmation workflow](https://github.com/MoralityLabAI/BPD/actions/runs/44121089342). GitHub Actions verified the original source hashes and preconfirmation threshold. All 32 confirmation roots and 59 calibration roots are archived in [calibration raw JSON](results/calibration_59_hosted.json) and [confirmation raw JSON](results/confirmation_32_hosted.json). The [preconfirmation calibration lock](CALIBRATION_LOCK.json) is preserved.

## Frozen verdicts

- **Strong incremental effect: FAIL, 0/32 roots** exceed the preregistered calibrated tolerance plus 5 percentage points with acceptable factor fit; required ≥24/32.
- **Equivalence: FAIL, 22/32 roots** simultaneously have reconstruction MSE ≤1e-4 and |BPD-minus-tangent-sham|≤5 pp; required ≥28/32 and the full root-mean bootstrap CI inside ±5 pp. The mean CI satisfies its part of the condition, while the combined root count fails.
- **Factor-fit engineering: 23/32 pass**, 9/32 fail: `5104`, `5106`, `5110`, `5112`, `5118`, `5121`, `5122`, `5125`, `5126`. Preserve all failures. No rerun/replacement, no revised cutoff.
- The calibration max pseudo-sham root gap is **T = 0.060361117**, fixed **before confirmation**. Strong-effect threshold is **T + 0.05 = 0.110361117**. The 59-root maximum is a one-sided 95th-percentile upper tolerance bound with ≈95.15% confidence under iid same-family root assumptions, not a universal model or security claim.

## Independent confirmation summary

| Measure | Result |
|---|---:|
| Root count | 32 distinct freshly trained teachers |
| Eligible authorized matched pairs | 6977 / 12288 generated |
| Selected six-factor BPD recovery, unweighted root mean | 90.51% |
| Signed-gradient-matched shams, root mean | 89.45% |
| Unsigned-gradient-matched shams, root mean | 44.65% |
| Mean BPD advantage vs signed matched shams | **+1.05 pp** |
| Registered 10,000-resample bootstrap 95% interval for root mean | **+0.36 to +1.77 pp** ([0.003638583, 0.017694429] unrounded) |
| Maximum registered root gap | 6.01 pp |
| Strong effect above calibrated bound | **0/32** |
| Joint equivalence root gate | **22/32** |
| First-pass |∂readout/∂authorized secret| root mean | 0.0102 |
| Second-pass |∂readout/∂authorized secret| root mean | 0.9756 |
| Second-pass cross-user sensitivity | 0.0284 |
| Second-pass unauthorized private sensitivity | 0.0231 |
| Maximum write norm-match error | 3.576e-7 |
| Maximum signed local Jacobian-match error | 5.364e-7 |

The 32 roots share one synthetic scalar-routing task family. Those 64 signed sham directions within each root are controls, not separate independently sampled roots. The small positive observed mean gap is inconsistent with a claim of a strong incremental BPD mechanism at this preregistered margin. Failure of the equivalence gate leaves formal equivalence unestablished.

## Exact per-root records

| Root | Eligible | BPD recovery | Signed sham mean | Gap | Factor reconstruction MSE | Engineering |
|---:|---:|---:|---:|---:|---:|:---:|
| 5101 | 211 | 91.47% | 91.32% | 0.15 pp | 1.006e-15 | pass |
| 5102 | 221 | 69.68% | 68.98% | 0.71 pp | 1.152e-14 | pass |
| 5103 | 231 | 99.13% | 98.60% | 0.53 pp | 2.716e-5 | pass |
| 5104 | 213 | 59.15% | 59.63% | -0.48 pp | 1.873e-4 | FAIL |
| 5105 | 222 | 100.00% | 100.00% | 0.00 pp | 4.917e-16 | pass |
| 5106 | 242 | 85.95% | 85.18% | 0.77 pp | 1.110e-4 | FAIL |
| 5107 | 219 | 95.43% | 94.44% | 0.99 pp | 2.459e-15 | pass |
| 5108 | 202 | 81.19% | 82.80% | -1.62 pp | 9.968e-5 | pass |
| 5109 | 203 | 99.51% | 98.42% | 1.09 pp | 1.195e-12 | pass |
| 5110 | 217 | 70.05% | 68.44% | 1.61 pp | 3.528e-4 | FAIL |
| 5111 | 209 | 100.00% | 99.72% | 0.28 pp | 2.712e-6 | pass |
| 5112 | 210 | 81.43% | 81.47% | -0.04 pp | 8.627e-4 | FAIL |
| 5113 | 219 | 100.00% | 100.00% | 0.00 pp | 1.259e-9 | pass |
| 5114 | 215 | 87.91% | 83.95% | 3.96 pp | 1.791e-6 | pass |
| 5115 | 225 | 100.00% | 100.00% | 0.00 pp | 3.897e-7 | pass |
| 5116 | 203 | 100.00% | 100.00% | 0.00 pp | 4.118e-11 | pass |
| 5117 | 215 | 100.00% | 99.34% | 0.66 pp | 7.344e-5 | pass |
| 5118 | 208 | 100.00% | 98.98% | 1.02 pp | 4.427e-4 | FAIL |
| 5119 | 204 | 100.00% | 100.00% | 0.00 pp | 7.788e-16 | pass |
| 5120 | 230 | 99.57% | 99.07% | 0.50 pp | 2.672e-5 | pass |
| 5121 | 207 | 93.24% | 90.01% | 3.23 pp | 7.408e-4 | FAIL |
| 5122 | 209 | 43.06% | 47.02% | -3.96 pp | 1.018e-4 | FAIL |
| 5123 | 213 | 100.00% | 99.46% | 0.54 pp | 2.378e-11 | pass |
| 5124 | 218 | 100.00% | 100.00% | 0.00 pp | 2.751e-8 | pass |
| 5125 | 246 | 86.59% | 80.57% | 6.01 pp | 3.221e-4 | FAIL |
| 5126 | 230 | 95.65% | 92.36% | 3.29 pp | 3.080e-4 | FAIL |
| 5127 | 214 | 98.60% | 97.48% | 1.12 pp | 2.658e-7 | pass |
| 5128 | 237 | 90.72% | 86.15% | 4.57 pp | 8.175e-11 | pass |
| 5129 | 228 | 96.49% | 90.89% | 5.61 pp | 1.368e-7 | pass |
| 5130 | 223 | 100.00% | 99.96% | 0.04 pp | 1.404e-9 | pass |
| 5131 | 209 | 100.00% | 96.50% | 3.50 pp | 1.495e-5 | pass |
| 5132 | 224 | 71.43% | 71.79% | -0.36 pp | 2.389e-5 | pass |

## Interpretation and next experiment

Early secret sensitivity is small and second-pass authorized sensitivity is near unity, so the trained teacher learns delayed role-dependent decoding. The high recovery of the BPD-selected write is largely explained by its privileged signed Jacobian effect; a controlled numerical difference of ≈1 pp remains on the particular synthetic source. Factor-fit failures are substantial, and a representative source-family-disjoint nonlinear/text setting is still untested.

A follow-up should target provenance/source-binding under interleaved users and distinguish true identity-sensitive routing from reusing semantic content. Preregister independent user/secret/collision counterfactuals, actual current-vs-future readout, gauge-stable *physical subspaces*, and output-channel leakage. Do not import these SF03 roots into a new confirmatory test or revisit ARG-03-N's sealed test.
