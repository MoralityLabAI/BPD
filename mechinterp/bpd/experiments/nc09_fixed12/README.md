# BPD-NC-09 Fixed12: causal passwise allocation at equal physical rank budget

## Current calibration status, November 18, 2026

The [FC05 independent full-tangent calibration](full_tangent_calibration_fc05/RESULTS_20261118.md) completed all 59 fresh roots with eight matched shams per model. Its registered result is **FULL_TANGENT_CALIBRATION_PILOT_ONLY**. Mean sham-minus-DP12 task NLL was 0.008083953 nats, with marginal root-bootstrap 95% interval [0.005644242, 0.010486967]. The 0.020 scientific margin remains unmet. The independently calibrated rootwise 95/95 floor is 0.058183573 nats, and strict EvalMath returned **REDESIGN_OR_PILOT_ONLY**.

All scientific confirmation populations remain sealed. FC05 evaluated zero confirmation models. The original registration and first-development account below retain their historical results; subsequent FC01–FC05 calibration and engineering studies have their own prospective records.

## Original registration

**Prospectively registered November 18, 2026.** [PROTOCOL.md](PROTOCOL.md) was committed first at e14d54cd08bf99306b31e491ade4e94d72f823f0, before implementation or training.

NC-08's first development result passed its source/sham controls and compressed physical retained write rank by 32.8%, but failed its frozen extra task-cost bound versus flat rank four. Its source-locked **secondary descriptive** budget-12 comparison favored passwise source-derived rank allocation over flat rank three by 0.013068 nats mean true-label NLL on the exposed eight roots. NC-09 tests this same-rank allocation question on **16 fresh independently trained model roots, 900–915**, with excluded engineering seed 899.

Every model is the same unconstrained 20D four-pass tied nonlinear DenseLooper, trained on 2048 cases for 384 Adam steps. Full native BPD interventions use four actual recurrent updates. Source-only gradient covariance and the NC-08 squared omitted-model-score proxy choose exactly one 12-slot pass-specific rank tuple from ranks 2–6. External true labels never enter that choice.

The four predeclared principal controls are the source-DP 12-slot projector, flat rank three (also 12 slots), **flat rank three with source-discovery physical retained-write RMS matched by a frozen scalar gain**, and flat rank four (16 slots, fidelity reference). A reversed pass-rank tuple tests allocation ordering descriptively. The same-tuple sham null uses eight physically matched, seeded local subspaces from exactly 4096 discovery-only proposals.

Independent 1024-example true-label final evaluation, four-pass KL, source-dose and rotated orthogonal-factor parity are recorded for every root. **DEVELOPMENT_POSITIVE** requires all sixteen roots eligible, source allocation advantages of at least +0.008 nats over unscaled flat3 and +0.006 over equal-dose flat3 with positive 95% root-bootstrap lower bounds, mean and upper absolute NLL cost at most +0.05 nats, extra cost upper versus flat rank four at most +0.020, and positive mean >=0.020 matched-sham NLL advantage.

The fixed 4096-draw whole-model-root bootstrap uses RNG 909009. A failed criterion stays **DEVELOPMENT_NULL_OR_INELIGIBLE**. Original raw model-root artifacts and source hashes remain frozen after first launch; prior NC-05–08 results are unchanged. Independent sham calibration (>=59 roots), EvalMath power preflight and 32-root scientific confirmation remain sealed and ungenerated.

See [THEORY.md](THEORY.md) for the physical source-dose gain, finite multiple-choice rank knapsack and limits of first-order score-loss proxies.


## Original prospective result: DEVELOPMENT_POSITIVE

The [first source-locked 16-model development run #44268539237](https://github.com/MoralityLabAI/BPD/actions/runs/44268539237) completed the original cohort with engineering PASS and **all six registered developmental gates PASS**. All 16 roots admitted their eight physical shams. The allocated passwise projectors remained at exactly 12 total dimensions, against the 16-direction flat-rank-four reference.

| Paired true-label final NLL (nats) | Mean | 95% independent-root bootstrap |
| --- | ---: | --- |
| Flat rank3 minus source-DP12 | **+0.02718** | [+0.01170,+0.04646] |
| Source-dose-matched flat rank3 minus source-DP12 | **+0.02802** | [+0.01185,+0.04827] |
| DP12 minus clean teacher | +0.01493 | [+0.00774,+0.02380] |
| DP12 minus flat rank4 | +0.00701 | [+0.00267,+0.01243] |
| Matched same-tuple physical sham minus DP12 | **+0.05341** | [+0.03898,+0.06769] |

All sixteen models individually favored DP12 over **both** 12-dimensional flat-rank-three controls. The source-RMS-dose-matched gain averaged **0.999956** (range .972897–1.041753). The selected passwise rank allocations were (4,3,3,2) on nine models, (4,4,2,2) on four, and (5,3,2,2) on three, with more retained physical directions early in the recurrence.

The predeclared descriptive reversed-rank-order placebo was **+0.10827 nats** worse than DP12 on average with the same 12-slot budget. One individual model root, 903, had higher absolute task cost and a negative local sham margin; the original registered *mean/root-bootstrap* gates passed and the raw counterexample is retained.

[Full frozen first-run report](RESULTS_20261118.md) · [all sixteen root outcomes](PER_ROOT_20261118.csv) · [independent SHA and raw 17-member artifact audit](test_published_results.py) · [original GitHub Actions artifact #14410589675](https://github.com/MoralityLabAI/BPD/actions/runs/44268539237/artifacts/14410589675). The original ZIP SHA256 is **585bec0fea7fd43fa21c2e7d3e32a0794ef760692a644ec52c8a6faf9e10fc89**.

This is **successful development evidence**, with a separate independent 59-root sham calibration, EvalMath power preflight and new 32-root scientific confirmation still sealed.
