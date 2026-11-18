# BPD-ECA-03: A task-grounded causal test of the ECA recurrent sufficient statistic

**Research report, 18 November 2026. Evidence class: development-only trained-GRU causal intervention with independent host-synthetic task truth.**

Companion to the [cross-program BPD-ECA synthesis](BPD_ECA_CROSS_PROGRAM_SYNTHESIS_20261118.md), [BPD-ECA-01/02 causal mechanism report](BPD_ECA01_02_CAUSAL_MECHANISM_REPORT_20261118.md), and [Control-Harness BPD-ECA-03 PR #58](https://github.com/MoralityLabAI/Control-Harness/pull/58).

## Scientific goal and registered design

The earlier BPD-ECA studies established exact rank-one sufficiency of gold-label/report match counts in the original three audit channels. Their source-trained physical projections could alter native GRU behavior, yet improvements toward a **model-self-generated donor-prefix target** failed the root-calibrated sham resolution test. BPD-ECA-03 therefore changed the causal endpoint: did a training-derived physical subspace reduce **true authorization errors** on independent host-labeled cases?

The unchanged ECA-04 16-hidden-unit GRU supplied the recurrent hidden state. The source-locked intervention occurred after six gold-labeled sentinel events, and the remaining six GRU steps and classifier ran natively. The **match-count** rank-one direction was fitted on 512 new independent training roots. The **task-gradient** rank-one direction came from the leading eigenspace of the uncentered gradient covariance of 256 independent training-root cost-weighted authorization log-loss gradients through the GRU suffix. The task-gradient direction was the preregistered primary candidate.

Both candidate directions had globally frozen signs chosen from the same training-only mean task gradient. Eight rank-one random controls received that same orientation rule. Every edit had physical L2 norm **0.25**, and each issuer-root provided **32 new synthetic authorization cases**, with private true permission labels and noisy reports used by the scorer. The complete unedited GRU and the privileged true-channel Bayesian decision rule were controls. Every policy saw the same report, leaving audit acquisition costs and adaptive query selection outside this narrower experiment.

A separate **64-root sham-calibration cohort** established a 95th-percentile/95%-confidence null envelope for rootwise matched-random benefit. **96 independent development roots** were used once for the frozen primary and secondary comparisons. EvalMath's preflight declared **two comparisons**, alpha 0.05, minimum meaningful normalized task-risk improvement **0.025**, hypothetical planning target 0.08, independent gradient-subspace stability minimum 0.98 and an independent-calibration **headroom requirement 0.02**. No protected confirmation roots were generated.

## Source-frozen result: task-correctness headroom was too small

The original [registered CI readout](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44261228404) passed on Python 3.11 and 3.13, using source SHA **b23c8485504e32f1a9e04eff56c9a504b9c22804** and separately registered request **fd914f9b4c7e959232ad3c89d2da1f2f3bc387da**. The dataset contains 3,072 host-labeled task cases per arm, nested within the 96 independent issuer roots.

| Frozen policy | False approvals | Missed permitted actions | Cost 2 FP + FN |
|---|---:|---:|---:|
| Privileged exact true-channel Bayes | 110 | 323 | **543** |
| Eight equal-dose random rank-one directions | 129–135 | 315–331 | **578–593** |
| Training-derived match-count direction | 129 | 330 | **588** |
| Original GRU with zero state intervention | 131 | 328 | **590** |
| Training-derived task-gradient direction | 140 | 330 | **610** |

The registered **task-gradient minus median same-dose sham** effect, expressed as mean normalized **true-cost reduction**, was **−0.00406901** per independent root. The match-count direction's corresponding effect was **−0.000488281**. Both signs are negative, so this experiment found no distinctive positive task-grounded causal benefit from either global hidden-state direction. The primary task-gradient intervention incurred 20 additional weighted-loss units compared with the untouched model.

The task-gradient direction was numerically identifiable on training roots. Its top gradient-covariance eigenvalue was **0.156712** against a second eigenvalue of **0.007164**, with ratio **21.875**; the two independent training halves yielded squared projector overlap **0.986489**, above the frozen **0.98** stability gate. The match-count direction's half-fit overlap was **0.923144**, below its own stability minimum. Both interventions met physical-dose and native full-output parity checks.

The separate **95/95 sham variation floor was 0.140625**, considerably above both actual causal effects and the hypothetical 0.08 power target. EvalMath returned **REDESIGN_OR_PILOT_ONLY** for both candidates, with target effect below calibrated variation, observed effect below the floor, and insufficient resolution for a planned 128-root confirmation.

A more fundamental limit appeared in the independent **task headroom calibration**: the true-channel Bayes oracle improved over the unedited model by only **0.00683594 normalized cost units per case**, below the predeclared **0.02** minimum. The 96-root pilot's oracle headroom was **0.00764974**. Under the registered data-generating process, the unedited trained GRU was already near the synthetic exact-channel ceiling. A context-blind fixed-magnitude state edit had little correctable task error to exploit.

## Replay and scientific integrity

The [independent hosted audit](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44261980479) passed with status **PASS_TASK_CORRECT_CAUSAL_REPLAY**. It checked **10 frozen source files and 11 original artifact output hashes**, reconstructed the task-private labels, public priors, noisy reports, and every policy's true false-grant/missed-grant counts. The archived physical rank-one matrices and source-trained directions reproduced all 160 calibration/development root evaluations, 320 candidate interventions and 1,280 matched random shams. It observed **zero changed task decisions and zero normalized effect drift**. Refit projectors on the identical training roots had squared overlap **1.0000** for both candidate directions.

This isolates the remaining failure as **behavioral and mechanistic specificity**, not numerical reproducibility of the registered intervention. The trained match-count decoder itself had low error, but its physical global direction did not produce a measurable task-grounded advantage against the strongest matched controls.

The experiment preserved BPD's protected ARG-03-N 168-case test and AMM-03 confirmation cohorts. It did not reuse any historical ECA-04 or ECA-05 heldout roots. The only historical artifact used was the original SHA-locked ECA-04 neural checkpoint.

## What to test next

A new experiment requires a behaviorally unsaturated task family and a source-frozen hypothesis. Two grounded ways forward are:

**State-conditional evidence correction.** Train a small controller that uses the public prefix posterior to determine *when and how* to steer a physical direction. Compare with a state-dependent random control that sees precisely the same public information and receives the same physical-dose budget. A null would localize failure to the GRU's hidden causal geometry or the task interface rather than to the arbitrary constant sign.

**Noncomplementary audit channel.** The original ECA-04 channels satisfy f = 1 − t, giving one sufficient match-count scalar at fixed prefix length. Deliberately train a **new** recurrent model on a family where sensitivity and false-positive rate vary independently. Its exact log likelihood can require two independent sufficient event counts. Test whether the learned recurrent representation and intervention architecture undergo the predicted **rank-one to rank-two transition**. This would more directly serve Compactification-Theory and BPD theory.

A registered discovery-stage null-calibration gate should be run before large GPU time. In both designs, independently generated roots, host-true task loss, equal-dose random controls, model-closure checks, and protected confirmation seals remain required. The original ECA-03 negative result and its source-pinned raw artifact should never be altered to search for a favorable analysis.

## Evidence links

[Registered task-grounded run](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44261228404) · [Complete original artifact](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44261228404/artifacts/14408028265) · [Independent physical and task replay](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44261980479) · [Full Control-Harness research report](https://github.com/MoralityLabAI/Control-Harness/blob/main/reports/bpd_eca03_task_grounded_development.md)

All tested issuers, gold references and authorization labels here are synthetic. The study is an internally valid development benchmark on one small GRU and does not establish generalized real-world authorization steering or larger-model mechanism transfer.
