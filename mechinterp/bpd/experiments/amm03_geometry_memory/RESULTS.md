# AMM-03: geometry-conditioned memory transfer, registered development result

**2026-11-18. Evidence: trained synthetic tied-linear BPD developmental null.** The independent confirmation cohort of 128 separately trained model roots remains **sealed**. [PROTOCOL.md](PROTOCOL.md) was frozen before source and new results at commit `cae5a676`. Source and CI were committed through `905b7172` before training. Training-derived policy was locked at `12f8953c` before the development cohort.

## Causal experiment

Each root trains its own 4D, rank-two, weight-tied physical recurrence from external (a,b) task targets and a new rank-two BPD V/U factorization from teacher trajectories. The pairwise donor target is defined independently of BPD by the donor's second-pass source label. Per root, the test agents receive eight *separate calibration donor pairs*, with both component interventions measured, for sixteen labeled local interventions. They then select **one action per root** for 128 fresh test pairs.

The proposed strategy learns seven-dimensional SVD-canonical geometric signatures across 64 independent **training roots** using fixed ridge regression. The `memory_calibrated` and `local_only` arms receive identical 16 per-root calibration intervention results. Memory adds a disclosed amortized training-world corpus; each active patch is projected current-dark and physically L2-dose-matched, followed by a learned tied recurrent pass. Model and label source roots are phase-separated. All test failures remain in the denominator.

## Frozen training memory

- Training: 64 independently trained root models, 96 donor pairs per root with both candidate actions observed.
- Component 0: **4,176 / 6,144** training-pair successes.
- Component 1: **4,403 / 6,144**; the training-pooled decision is component 1.
- A seven-coefficient ridge model was frozen with regularization .01. Exact policy SHA256 `e22ce9bd2dfbbbcf90ba0502a847477835f275b2099f8d79816c35534a7d0e56`.
- Model source SHA256: `models.py=69b5a0c9688efa048ff9e5fb7308fbc2c8bfe15d0b274a9cf61b00b28af8f967`; scorer source SHA256 `experiment.py=4310c6612986f432f702957e7c6d5387f39e38d32c21bd65b05a833179804500`.

## Development results: 64 independent roots × 128 test cases

| Frozen arm | Correct / 8,192 | Accuracy |
|---|---:|---:|
| **Memory + eight-pair local calibration** | **7,774** | **94.90%** |
| **Eight-pair local calibration only** | **7,774** | **94.90%** |
| Memory without local labels | 7,404 | 90.38% |
| Analytic signed-delayed-gain geometry rule | 7,042 | 85.96% |
| Training-global pooled component | 5,905 | 72.08% |
| Root-hashed random component | 6,462 | 78.88% |
| Equal-dose random dark A | 2,198 | 26.83% |
| Equal-dose random dark B | 2,299 | 28.06% |
| Zero-patch sham | 69 | 0.84% |
| Full donor state (different dose) | 8,032 | 98.05% |

**Primary paired root effect: exactly zero on all 64 independent development roots.** Both memory and local-only policies choose the same component on 54 roots; on the remaining 10 roots their selected components have exactly identical held-out success counts. Root-sign p=1; bootstrap interval of the primary effect is [0,0] for these independent roots.

Family-stratified memory/local hits: wren **3,880/4,096**, basalt **3,894/4,096**. Among 64 development roots, 30 have tied local calibration (both 0/8, both 8/8 or other tied counts). A posthoc test-answer oracle that picks the better component separately on *each root's 128 scored cases* obtains **7,784/8,192 (95.02%)**, just **10 additional test outcomes (0.122 percentage points)** beyond local-only. This is an in-sample optimistic diagnostic, unusable for agent action selection, showing that the current assay is close to saturated for the registered primary comparison.

Thirty-seven roots have different raw candidate test hit counts, so the causal intervention instrument distinguishes components on some models. The eight-pair local calibration generally captures that difference already. Geometry-only memory beating the pooled arm is an exploratory description; the prospectively registered primary claim concerns **additional benefit beyond equal-cost local calibration**, which did not appear.

## Engineering integrity

- Every one of the 8,192 held-out pairs retained.
- Maximum difference between active patch L2 norm and the registered per-pair dose **2.3842e-7**; maximal immediate dark-readout displacement **0**.
- Independent physical teacher behavioral MSE maximum **2.67e-8**; trained BPD factor behavioral MSE maximum **0.11991**, with the higher-error roots retained.
- Canonical singular-value gaps across development roots **at least 0.0701**; none is below the registered .01 degeneracy threshold.
- Twelve excluded-root engineering unit tests pass locally (native BPD parity executes in repository/hosted CI rather than a standalone copy). Full-product GL(2) shear and SVD-recanonicalized rank-one factor invariants are checked.
- Complete training/development raw receipts regenerate from source and SHA256-derived root streams; locked candidate/policy case-bits digest `6b6cae4bec764283d7a98b7a8eae1fbfd2dc1e60f8aa7e812f6c97ad037fbc55`.

## EvalMath strict preflight anomaly and scientific hold

Pinned source: `MoralityLabAI/EvalMath@2fdd5fe3f499070b63c474a2455f0e2c00fc55a8`.

The registered 95th-quantile/95%-confidence tolerance bound from two independent random-dark arms over 64 development roots is **0.1875 (24/128)**. The scientific effect minimum is +.10 and target +.20, making the planner's combined variation floor +.1875. The planner returned `READY_FOR_CONFIRMATION` with an estimated required root count of 2 because the observed independent-root primary effects are all zero and the pilot sample standard deviation is also zero. This is a **power-planning pathology**, because the target-vs-null comparison assumes a 20-point effect even though development supplied zero evidence for it. The output must not be interpreted as scientific evidence for memory transfer.

We are exercising an explicit **scientific hold on confirmation** after inspecting development. The preregistered protocol remains unedited; this post-development choice is separately declared, preserving the planning anomaly. All 128 prospective confirmation roots remain unscored. The primary observed effect lies below the 10-point scientific minimum and the test-answer oracle supplies only a 0.12-point in-sample headroom. Spending additional sealed roots on the same saturated assay would be uninformative. This is a developmental null, with confirmation still available for a separately registered compelling hypothesis if warranted.

## What would change this result

A new prospective gym should sample harder causal discriminations using training/calibration-only criteria such as component near-ties, model diagnostic uncertainty or dose-response slope disagreement. It must reserve fresh model roots and separately seeded donor outcomes, retain a strong local-calibration comparator, and require genuine development-world variation beyond the matched-sham floor before confirmation. EvalMath should flag zero-variance/zero-effect pilots that otherwise produce spurious two-world power projections. The result concerns synthetic trained linear models; it makes no claim about natural-language agents, nonlinear transformer causal mechanisms, deployed Buzz or authenticated cross-agent memory transfer.

## Reproduction

```bash
python -m unittest mechinterp.bpd.experiments.amm03_geometry_memory.test_experiment -v
python -m mechinterp.bpd.experiments.amm03_geometry_memory.experiment --phase training --out /tmp/amm03_training.json
python -m mechinterp.bpd.experiments.amm03_geometry_memory.experiment --phase development --memory /tmp/amm03_training.json --out /tmp/amm03_dev.json
```

The locked training and development receipts are under `results/`. Regenerate raw per-case bits and compare to `DEVELOPMENT_LOCK.json` or let the dedicated hosted CI replay them. The 128-root confirmation generation path must remain unused.
