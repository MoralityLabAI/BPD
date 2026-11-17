# BPD-OUT05: A causal-effect sign advantage can coexist with worse absolute prediction error

**Source audit and exact mathematics, 17 November 2026.** Motivated by [SMC-25 PR #226](https://github.com/MoralityLabAI/Compactification-Theory/pull/226), merged into Compactification-Theory after four source-locked GitHub Actions stages passed. Its [SMC25_RESULTS.md](https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/real_recurrence_sae_v1/SMC25_RESULTS.md) preserves the original **FAIL (0/4 SAE pairs)** on two fresh 64-puzzle unique noncanonical Sudoku cohorts, for the registered MAE gate. This BPD addendum analyzes the secondary sign result without modifying that verdict or accessing any sealed ARG-03-N cases.

## Why the distinction matters

SMC-25 computed a native directional tangent for every possible correct Sudoku digit before opening test labels. A calibrated T=2 digit-probability-weighted median of the nine effects was then frozen as the prediction for the finite correct-digit feature-kill-minus-sham margin. It was scored against an equal-forward-compute uniform-nine median, a model-top-digit candidate and a cyclic probability-shift median.

Across **2,009 admitted teacher/SAE-goal effects** observed on 128 distinct new Sudoku puzzles, calibrated median **effect sign** agreement exceeded the uniform median in **8/8** teacher/SAE/family cells. Its **absolute-error gain** relative to uniform had a positive *point estimate* in only **4/8**, and **0/8** Bonferroni-adjusted 99.375% lower confidence bounds exceeded zero. The original scientific gate required corrected MAE improvement against all controls plus sign advantage. Hence **0/4** paired-model criteria passed.

This pattern motivates a separable decision-theoretic question: when is selecting a native intervention's **direction of effect** useful despite poor prediction of its magnitude? The sign of the native effect can control a binary choice to apply or withhold a candidate state edit; the exact margin magnitude matters for dose, utility and safety.

## Exact counterexample: sign classification improves while MAE worsens

Consider nine possible task-labelled causal effects

  d = (-10, -10, -1, -1, -1, +1, +1, +10, +10).

The uniform-nine weighted median is **−1**. Suppose a miscalibrated class distribution places all mass on the first +10 candidate, yielding calibrated weighted median **+10**. Let the actual conditional target distribution put **7/10** mass on one +1 candidate and **3/10** on one −10 candidate.

- Calibrated median +10 has correct sign on **7/10** trials, while uniform median −1 has correct sign on **3/10**.
- Calibrated median has exact conditional MAE **123/10 = 12.3**; uniform has **41/10 = 4.1**.

These are exact rational statements, implemented with Python fractions. They show that sign agreement alone does not order absolute prediction error between two candidate estimators when one is miscalibrated in magnitude. This finite construction is **not** claimed to fit the source SMC-25 teacher's actual label distribution.

## The conditional sign-stability inequality

Fix a hidden state and its candidate nonzero effects d_i. Let the true conditional correct-target distribution be p_i and an estimated distribution be q_i. Define signed probability masses

  S_p = Σ_i p_i sign(d_i),   S_q = Σ_i q_i sign(d_i).

The Bayes-optimal equal-cost binary effect-sign decision follows sign(S_p), with a tie when S_p=0. Because the sign function is bounded in [−1,+1], the elementary total-variation inequality gives

  |S_p−S_q| ≤ 2 TV(p,q).

A strictly positive margin |S_p|>2TV(p,q) is therefore sufficient for the proxy sign(S_q) to agree with the Bayes sign decision. When candidate effects never vanish, a weighted median of the d_i has the corresponding probability-majority sign, away from exact mass ties. A label distribution can thus preserve a binary sign decision while remaining insufficient to reproduce a good numerical effect estimate.

The inequality is conditional on a *real* distribution p over correct target labels after observing the predictor's input. A model's calibrated softmax is a candidate q, not evidence that p has been identified. With zero-effect candidate mass, abstention and sign-zero utilities must be specified separately. These source-independent statements are compatible with BPD-OUT04's labelwise Bayes MAE and squared-risk floors.

## Post-result whole-puzzle sign audit of the SMC-25 receipt

[smc25_sign_vs_mae.py](smc25_sign_vs_mae.py) independently replays the recorded per-goal sign decisions from the original final SMC-25 native scored.json, whose byte SHA-256 is **6e58461acd9c012f3ff9324be1420dc693cb3c318419c9a9bebb2376af2c878c**, available in [Actions artifact #14341883837](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44113889861/artifacts/14341883837). It requires that exact hash by default and checks all 2,009 original eligible goal records and predeclared source sign counts. The audit then computes an **exploratory** per-puzzle mean calibrated-minus-uniform sign-accuracy contrast and resamples all 64 source puzzle IDs 4,000 times with RNG seed 20261117. The 95% intervals below are **posthoc, unadjusted and unregistered**, and all teacher/SAE fits share the same two trained teachers and 128 puzzle sources.

| Teacher/SAE | New family | Correct signs, calibrated vs uniform | Mean per-puzzle sign advantage | Exploratory 95% puzzle bootstrap |
|---|---|---:|---:|---|
| B501 SAE71 | 30 holes | 132 vs 120, n=209 | +0.0462 | [−0.0477, +0.1424] |
| B501 SAE89 | 30 holes | 139 vs 121, n=215 | +0.0783 | [−0.0295, +0.1828] |
| B701 SAE71 | 30 holes | 184 vs 172, n=315 | +0.0409 | [−0.0432, +0.1214] |
| B701 SAE89 | 30 holes | 159 vs 150, n=285 | +0.0200 | [−0.0463, +0.0869] |
| B501 SAE71 | 34 holes | 131 vs 103, n=201 | +0.1451 | [+0.0412, +0.2464] |
| B501 SAE89 | 34 holes | 137 vs 130, n=227 | +0.0042 | [−0.0914, +0.1005] |
| B701 SAE71 | 34 holes | 177 vs 168, n=277 | +0.0478 | [−0.0355, +0.1352] |
| B701 SAE89 | 34 holes | 177 vs 155, n=280 | +0.0814 | [+0.0134, +0.1534] |

The descriptive sign gain is positive in all eight point estimates. Only **two of eight unadjusted** 95% puzzle-bootstrap intervals have a strictly positive lower bound, and both are from the 34-hole family. The 30-hole family has **zero of four** unadjusted positive-lower-bound intervals. This evidence supplies no preregistered teacher-pair success, no corrected eight-cell positive claim and no across-teacher population inference.

The sign results reuse the original scored-test outcomes. Their interpretation is explicitly exploratory; confidence bounds answer a new post-result question whose selection depended on the observed SMC-25 secondary endpoint. A new valid inferential test needs preregistration and fresh source-disjoint Sudoku puzzles.

## Reproduce and future test

The exact mathematical witnesses and the source-locked exploratory audit are in [smc25_sign_vs_mae.py](smc25_sign_vs_mae.py). [Seven unit tests](tests/test_smc25_sign_vs_mae.py) check exact rational signs/MAEs, the TV stability guarantee and its possible failure, probability admission, correct clustering and ineligible goal preservation. Run from the BPD root:

    python -m pytest -q mechinterp/dark_fiber_gyms/tests/test_smc25_sign_vs_mae.py

To reproduce the source exploratory audit, download and extract scored.json from SMC-25 Actions artifact #14341883837, then run:

    python -m mechinterp.dark_fiber_gyms.smc25_sign_vs_mae --scored path/to/scored.json --out sign_audit.json

For a prospective follow-on **SMC-26**, freeze an outcome-oriented **binary effect-sign decision** before test. Preselect the same calibrated-median and uniform-median sign policies from SMC-25 on new training-only calibration examples. Evaluate on two *new* puzzle families generated from source seeds disjoint from SMC-25 and all earlier campaigns, with at least two independently trained capability-matched teachers. Use a fixed sign decision benefit endpoint and cluster-bootstrap test corrected across eight cells. Include a zero/abstention policy, native edit application costs and uniform/rotated probability controls. Register the program gate before generating any test effects. The previous 0/4 MAE failure is preserved as a separate result.
