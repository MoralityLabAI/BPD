# BPD-NC-07: source-only KL chooses a causal recurrent write rank

**Prospectively registered November 18, 2026.** [PROTOCOL.md](PROTOCOL.md) was the first research commit (c798a5b), predating all code and all 700–707 trained model roots.

NC-06's eight new nonlinear 20D recurrent models showed that rank-three physical BPD subspaces satisfy a 0.05-nat absolute task-fidelity cap while norm-weighting adds no practical benefit over uniform/shuffled controls. NC-05's rank-four subspace has better fidelity. NC-07 tests whether model-only clean-versus-retained predictive KL can select the **minimum useful physical write rank per trained root**.

The fixed ranks are 2/3/4/6. For each independently trained model, an unweighted source-only gradient covariance produces nested passwise projectors from 1024 derivative examples. A disjoint 512-example rank-selection source computes intact-vs-retained output KL, sign-flip frequency, per-case tail KL and retained physical write RMS. The smallest rank meeting all frozen source-only thresholds is chosen. No external task labels enter this choice.

The selected projector then intervenes through actual full four-pass tied BPDModel/TiedComponentLinear writes. On an independent 1024-example labeled evaluation set, measure external true-label NLL and compare with fixed rank three, fixed rank four and eight discovery-only equal-rank +/-15%-dose-matched local projector shams. An orthogonal factor-coordinate transport checks that the physical write and resulting closed-loop logits are invariant.

The eight-root **engineering-development** gate requires all source/task/sham/gauge admissions, mean selected rank <=3.5 with at least four roots selecting rank <=3, clean-to-retained final NLL mean and upper root-bootstrap <=0.05 nats, positive >=0.02 median-sham advantage, and selected-vs-fixed-rank-four true-label NLL upper root-bootstrap <=0.02 nats. Decisions use whole independently trained roots (2000 draws, seed 707007).

The first run retains ENGINEERING_POSITIVE or ENGINEERING_NULL_OR_INELIGIBLE unchanged. No independent 59-root sham-calibration or 32-root confirmation samples are generated.

See [THEORY.md](THEORY.md) for the Pinsker probability-approximation bound, its limit for true-label loss, source-only spectral variational geometry and orthogonal physical gauge transport. The source-first [experiment](experiment.py), [aggregator](aggregate.py), [excluded-root tests](test_experiment.py) and [synthetic decision tests](test_aggregate.py) implement the frozen experimental contract.


## Frozen first-run development verdict: ENGINEERING_NULL_OR_INELIGIBLE

The [original source-locked eight-root GitHub Actions run #44260710669](https://github.com/MoralityLabAI/BPD/actions/runs/44260710669) completed successfully with green engineering/source/evidence checks and immutable results. The registered all-root development gate **failed**.

The rank selector chose **rank 3 on four roots**, **rank 4 on two** and **rank 6 on two**. Mean selected rank across the eight original roots was **4.0**, exceeding the preregistered 3.5 compression ceiling. **Seven of eight roots were fully admitted**; root 706 selected rank 6 and found **7 physically dose-qualified shams among 1,024 proposals**, versus the required eight, so the original source-admission gate failed. All eight source/teacher full-mask controls completed.

Because admission was 7/8, the original aggregate correctly left its whole-root confidence intervals unscored. Post-hoc descriptive calculations using *all eight original roots including the ineligible one* gave mean held-out selected-minus-clean task NLL +0.01092, mean local matched-sham advantage +0.04224, and mean selected-minus-fixed-rank4 task NLL -0.00373. Those are diagnostics, **not retrospective registered passes**.

[Original first-run report](RESULTS_20261118.md), [eight-root CSV](PER_ROOT_20261118.csv), [independent source-hash and original ZIP auditor](test_published_results.py), and [immutable complete nine-file GitHub artifact #14408012541](https://github.com/MoralityLabAI/BPD/actions/runs/44260710669/artifacts/14408012541) preserve the negative evidence. The archived ZIP SHA256 is 55f831d8c7f37b2db8e7e0e6e79824b8f99a26d189f452a2dd642e3383d4dbce. Neither the original source nor the 59-root calibration/32-root confirmation seals have been modified.
