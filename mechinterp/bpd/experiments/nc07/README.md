# BPD-NC-07: source-only KL chooses a causal recurrent write rank

**Prospectively registered November 18, 2026.** [PROTOCOL.md](PROTOCOL.md) was the first research commit (c798a5b), predating all code and all 700–707 trained model roots.

NC-06's eight new nonlinear 20D recurrent models showed that rank-three physical BPD subspaces satisfy a 0.05-nat absolute task-fidelity cap while norm-weighting adds no practical benefit over uniform/shuffled controls. NC-05's rank-four subspace has better fidelity. NC-07 tests whether model-only clean-versus-retained predictive KL can select the **minimum useful physical write rank per trained root**.

The fixed ranks are 2/3/4/6. For each independently trained model, an unweighted source-only gradient covariance produces nested passwise projectors from 1024 derivative examples. A disjoint 512-example rank-selection source computes intact-vs-retained output KL, sign-flip frequency, per-case tail KL and retained physical write RMS. The smallest rank meeting all frozen source-only thresholds is chosen. No external task labels enter this choice.

The selected projector then intervenes through actual full four-pass tied BPDModel/TiedComponentLinear writes. On an independent 1024-example labeled evaluation set, measure external true-label NLL and compare with fixed rank three, fixed rank four and eight discovery-only equal-rank +/-15%-dose-matched local projector shams. An orthogonal factor-coordinate transport checks that the physical write and resulting closed-loop logits are invariant.

The eight-root **engineering-development** gate requires all source/task/sham/gauge admissions, mean selected rank <=3.5 with at least four roots selecting rank <=3, clean-to-retained final NLL mean and upper root-bootstrap <=0.05 nats, positive >=0.02 median-sham advantage, and selected-vs-fixed-rank-four true-label NLL upper root-bootstrap <=0.02 nats. Decisions use whole independently trained roots (2000 draws, seed 707007).

The first run retains ENGINEERING_POSITIVE or ENGINEERING_NULL_OR_INELIGIBLE unchanged. No independent 59-root sham-calibration or 32-root confirmation samples are generated.

See [THEORY.md](THEORY.md) for the Pinsker probability-approximation bound, its limit for true-label loss, source-only spectral variational geometry and orthogonal physical gauge transport. The source-first [experiment](experiment.py), [aggregator](aggregate.py), [excluded-root tests](test_experiment.py) and [synthetic decision tests](test_aggregate.py) implement the frozen experimental contract.
