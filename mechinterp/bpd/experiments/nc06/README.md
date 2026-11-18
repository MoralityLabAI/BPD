# BPD-NC-06: norm-conditioned rank-three recurrent causal projectors

**Prospectively registered November 18, 2026.** [PROTOCOL.md](PROTOCOL.md) was committed before the new model implementation or new training roots.

The eight independent development roots 600–607 train the same full-density 20D nonlinear shared-weight four-pass DenseLooper as NC-05. The projector is discovered from a model's own final-score derivatives and clean intermediate prewrite activations, without external labels. The true task labels enter the original 2048-example model training and a separately sampled 1024-example final evaluation only.

## Fixed comparison

| Physical retained write arm | Rank | Discovery weight |
|---|---:|---|
| **norm_r3** | 3 | exp(-||a_t||² / mean_source||a_t||²) |
| uniform_r3 | 3 | Equal weight |
| shuffle_r3 | 3 | Same norm weights, independently permuted per pass |
| uniform_r4_reference | 4 | Equal weight, NC-05 rank-four anchor |

Each projector is discovered using a distinct pass-indexed uncentered gradient covariance, and executed via actual BPDModel/TiedComponentLinear override-read interventions on every one of four nonlinear recurrent steps. Zero output-head gradients, full-mask parity, orthogonal factor gauge transport, independent true-label task NLL, physical write dose and local same-rank matched shams are tracked.

The new mathematical question is whether a **norm-conditioned rank-three** subspace improves clean-task fidelity over the equally ranked uniform and shuffled controls, moving NC-05's successful rank-four frontier to a smaller physical causal subspace. Since PD-17 established source-specific norm weighting, this is a deliberately falsifiable transfer test. The [NC-05 registered development report](../nc05/RESULTS.md) already showed rank-four/full-gain mean clean-to-kept NLL +.021219 with family-adjusted upper +.045398 in eight independent trained models; NC-06 uses new roots and must not count NC-05's eight as independent replication.

The fixed engineering gate requires eight trained/admitted roots and matched shams, norm-rank-three absolute fidelity mean/upper-bootstrap <=.05 nats, improvement >=.01 nats over each same-rank control with positive 2000-draw root-bootstrap lower endpoint, and matched-sham advantage >=.02 with positive lower endpoint. Every failure remains in the published development receipt, and no scientific confirmation is claimed from eight synthetic roots.

The [experiment](experiment.py), [aggregator](aggregate.py), and [excluded seed 599 tests](test_experiment.py) provide the runnable local/hosted CPU gym. A source lock is required before releasing the original eight-root development. Future >=59 sham-calibration and >=32 confirmation model roots remain ungenerated and sealed, as do unrelated ARG-03-N and AMM-03 protected datasets.


## Original eight-root development: norm-conditioning incremental null

[Original source-locked GitHub Actions run #44255130086](https://github.com/MoralityLabAI/BPD/actions/runs/44255130086) completed all eight new trained dense 20D recurrent roots, their native four-pass physical interventions, matched local sham controls and first evidence archive. The frozen developmental verdict is **ENGINEERING_NULL_OR_INELIGIBLE**. All eight source/physical eligibility controls passed.

Norm-rank3 kept-minus-clean mean true-label NLL was +0.02116 nats, upper 95% root bootstrap +0.03371, so **absolute fidelity passed**. Its median matched-sham advantage was +0.07453 nats, lower 95% +0.05571, also passed. Norm weighting provided **+0.000357 nats** versus unweighted rank3 and **+0.000336** versus exact-ESS shuffle, both far below the registered +0.01 improvement threshold with intervals spanning zero. The frozen developmental hypothesis therefore failed.

The original unweighted rank-four reference gave clean-to-kept NLL +0.00538 nats. Normalized activation-energy weights had an approximately **91% effective sample fraction at each recurrent step**, far weaker conditioning than PD-15's ~48% sparse-source effective fraction. Native orthogonal gauge physical-operator errors stayed below 1.30e-7.

[Detailed source-locked results](RESULTS_20261118.md), [original eight-root table](PER_ROOT_20261118.csv), [independent original SHA and outcome audit](test_published_results.py), and [unmodified full original artifact #14405931779](https://github.com/MoralityLabAI/BPD/actions/runs/44255130086/artifacts/14405931779) preserve this negative incremental result. The independent 59-root sham-calibration and 32-root confirmation phases remain sealed.
