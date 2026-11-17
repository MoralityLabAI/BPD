# BPD-CQ-04: prospective independent CPU pilot results

Protocol committed: \`808050f9\`. Experiment source frozen: \`d1200e46\`. Prospective seed roster: 701, 709, 719, 727, 733, 739, 743, 751. Full per-root source-locked counts: [results/LOCKED_COUNTS.json](results/LOCKED_COUNTS.json). Independent CPU: Python 3.13.5, PyTorch 2.10.0+cpu. Hosted native results are tracked separately in the PR and GitHub Actions artifacts.

## Readout accuracy, 8 independent learned recurrences × 256 inputs

| Seed | Adaptive | Calibration-optimal static mix | Matched random slots | Full | Adaptive slots / 4 | Donor eligible / 128 |
|---|---:|---:|---:|---:|---:|---:|
| 701 | 249 | 191 | 151 | 256 | 2.574 | 7 |
| 709 | 254 | 168 | 96 | 256 | 2.328 | 1 |
| 719 | 250 | 171 | 151 | 256 | 2.453 | 0 |
| 727 | 252 | 169 | 96 | 256 | 2.008 | 0 |
| 733 | 246 | 164 | 63 | 256 | 1.742 | 10 |
| 739 | 248 | 165 | 101 | 256 | 1.734 | 11 |
| 743 | 248 | 163 | 81 | 256 | 1.965 | 17 |
| 751 | 255 | 179 | 73 | 256 | 1.875 | 21 |
| **Total** | **2002** | **1370** | **812** | **2048** | **2.085** | **67** |

The adaptive controller solves **97.75%** of the two-target cases with 2.085 of 4 possible component activations on average, i.e. 47.88% fewer slots than full execution. Its calibrated input-blind static mixture solves 66.89% with an actual mean cost of 2.177 slots, while exact-per-case-cardinality randomized positions solve 39.65%. The primary +632/2048 difference is **30.86 percentage points**. Every seed exceeds the preregistered exploratory +15 pp improvement, 90% accuracy and <=3 mean-slot thresholds. Direction-only 8/8 positive independent roots gives two-sided exact sign p=0.0078125; a conservative Holm correction over the two registered comparisons gives upper bound 0.015625 for this direction check. This is a model-seed-family pilot, and the EvalMath preflight blocks confirmation.

All three source strata are preserved. Dual-source cases contribute **752/788** adaptive hits versus **466/788** static hits, with both original a,b nonzero (barring zero-probability exact draws) and independently sampled. Source-A-only and source-B-only are easier supports, and the controller observes their zero coordinates directly. Its four-bit plan uses initial-state features (a,b,|a|,|b|), with no later hidden-state feedback. An exact solver and mask-selection oracle know the task structure; this result shows trained resource allocation, not latent-variable discovery.

## Donor eligibility failure and geometry

Fresh recipient/donor pairs share a and change b. Eligibility requires both complete masked trajectories accurate, identical four-bit controller plans and a nontrivial native component patch; **67/1024** pass (6.54%), with zero eligible in seeds 719 and 727. Across this severely selected subset, independent CPU native write formula 67/67, current-dark norm-matched native 67/67, current-dark random 10/67, sham 0/67, larger-budget full intermediate donor replacement 67/67. The conditional result has severe selection bias and cannot answer broad source-swap robustness. The experiment retains every failed pair count and does not replace seeds. The primary 2-pass outcome uses all 2,048 examples.

Under a fixed nontrivial \`GL(2)\` shear, the full physical BPD operator remains invariant while retaining the original hard execution mask bits cuts correct two-readout cases from **2002/2048 to 1220/2048** (59.57%). This gives a concrete basis-dependence obstruction. A reoptimized or block-invariant policy after shear is a separate prospective experiment.

## EvalMath power/evidence

Replayed the [EvalMath mechanistic-resolution preflight](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) with four excluded engineering roots 997, 1009, 1013, 1021. Estimated pilot mean root effect 0.3252; development root effect SD 0.02695. The observed 0.03516 maximum sham was solely *calibrated static-mixture RNG variation*, not an independent mechanism-null percentile guarantee. The strict tool exited 2: \`REDESIGN_OR_PILOT_ONLY\`, reasons \`heldout_unsealed\`, \`sham_quantile_uncalibrated\`, \`too_few_pilot_worlds\`. The \`approx_worlds_for_80pct_power=2\` normal approximation does not override those hard stops. Evidence class remains **synthetic pilot**.

The next causal-interpretability gym needs an eligible donor cohort where the pass-two controller is held fixed by construction, plus separate adaptive live-feedback routing, basis-invariant block masks and 59+ independent sham calibration roots before confirmation.
