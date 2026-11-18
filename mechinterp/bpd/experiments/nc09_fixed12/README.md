# BPD-NC-09 Fixed12: causal passwise allocation at equal physical rank budget

**Prospectively registered November 18, 2026.** [PROTOCOL.md](PROTOCOL.md) was committed first at e14d54cd08bf99306b31e491ade4e94d72f823f0, before implementation or training.

NC-08's first development result passed its source/sham controls and compressed physical retained write rank by 32.8%, but failed its frozen extra task-cost bound versus flat rank four. Its source-locked **secondary descriptive** budget-12 comparison favored passwise source-derived rank allocation over flat rank three by 0.013068 nats mean true-label NLL on the exposed eight roots. NC-09 tests this same-rank allocation question on **16 fresh independently trained model roots, 900–915**, with excluded engineering seed 899.

Every model is the same unconstrained 20D four-pass tied nonlinear DenseLooper, trained on 2048 cases for 384 Adam steps. Full native BPD interventions use four actual recurrent updates. Source-only gradient covariance and the NC-08 squared omitted-model-score proxy choose exactly one 12-slot pass-specific rank tuple from ranks 2–6. External true labels never enter that choice.

The four predeclared principal controls are the source-DP 12-slot projector, flat rank three (also 12 slots), **flat rank three with source-discovery physical retained-write RMS matched by a frozen scalar gain**, and flat rank four (16 slots, fidelity reference). A reversed pass-rank tuple tests allocation ordering descriptively. The same-tuple sham null uses eight physically matched, seeded local subspaces from exactly 4096 discovery-only proposals.

Independent 1024-example true-label final evaluation, four-pass KL, source-dose and rotated orthogonal-factor parity are recorded for every root. **DEVELOPMENT_POSITIVE** requires all sixteen roots eligible, source allocation advantages of at least +0.008 nats over unscaled flat3 and +0.006 over equal-dose flat3 with positive 95% root-bootstrap lower bounds, mean and upper absolute NLL cost at most +0.05 nats, extra cost upper versus flat rank four at most +0.020, and positive mean >=0.020 matched-sham NLL advantage.

The fixed 4096-draw whole-model-root bootstrap uses RNG 909009. A failed criterion stays **DEVELOPMENT_NULL_OR_INELIGIBLE**. Original raw model-root artifacts and source hashes remain frozen after first launch; prior NC-05–08 results are unchanged. Independent sham calibration (>=59 roots), EvalMath power preflight and 32-root scientific confirmation remain sealed and ungenerated.

See [THEORY.md](THEORY.md) for the physical source-dose gain, finite multiple-choice rank knapsack and limits of first-order score-loss proxies.
