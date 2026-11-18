# CQ-08: separately released 32-root synthetic confirmation

The original BPD-CQ-08 physical-model source, read/write decomposition source, component calibration selection, source generator, donor matching controls, and the four frozen source hashes remain byte-for-byte unchanged from the eight-root engineering study.

The original protocol and `REGISTRATION.json` preregistered all **32** independent confirmation model/world roots (odd seeds 11001–11063) and 128 donor pairs per root. The **59 independent sham-calibration roots** and their original receipt are separately locked, with a nonparametric upper 95th-percentile/95%-confidence random-vs-random root floor of **13.28125 percentage points**. The registered minimum practically meaningful effect is **15 percentage points**, the larger of those two numbers.

The independently completed native frozen-checkpoint audit on [GitHub Actions #44239141740](https://github.com/MoralityLabAI/BPD/actions/runs/44239141740) reproduces the originally frozen exposed development receipt, while the exact source-locked sham preflight returns `READY_FOR_CONFIRMATION`. The release is committed in `CONFIRM_UNLOCK.json` before touching any new confirmation seed. This file authorizes the fixed roster only; it cannot authorize source edits or re-running and replacing bad roots.

## Precommitted evaluation

- Primary comparison: causal **correct-cell damage** when calibrated selected two BPD components are ablated from the otherwise full second visit, minus damage caused by the first independent uniform random two-component-off mask. Each root's effect uses **all 128** donor pairs; unmatched and full-inaccurate pairs count as zero in both arms.
- Secondary: the same selected block versus an independently sampled random 2-off component mask. Source bank, component selection, and test targets remain those of the first protocol.
- Root effect floor **strictly greater than 0.15** for a root counted as a sign success. Null is a 0.5 probability of a root clearing that effect floor. Exact **one-sided** Binomial(32, .5) tests with **Holm** correction across the two registered comparisons at family alpha 0.05. Report both effect means, root signs, losing roots and a 20,000-resample 95% whole-root bootstrap interval with seed 20261118.
- Primary gate: mean root effect at least +0.15 and primary Holm-adjusted one-sided sign p below .05; independent engineering gate requires all 32 roots meet 80% reachable nuisance matching, no state-range violations, exact first-state provenance and full two-visit task accuracy. Preserve failures without replacement.
- A raw write-once JSON receipt holds all full root and pair-level outcomes, including physical ablation write norms and gauge diagnostics.

## Essential interpretation boundary

**Selected and random ablations have the same six-active-components execution budget, not the same physical preactivation dose.** On the eight exposed development worlds, selected write norms averaged 20.55 versus 9.04 and 8.68 for the random controls. A positive confirmation establishes the predetermined **equal-cardinality intervention contrast within the synthetic block-separated family**; it cannot certify a physical-dose-matched causal mediation claim. The first state supplied to the masked second step is exactly reachable from an actual legal first cue, but masked second-step states are artificial computational interventions.

The next research design should calibrate physical ablation dose and remove the hand-structured core/nuisance separation in the physical operator.
