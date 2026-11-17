# Idealized BPD expected-L0 identification (2026-11-17)

This directory contains a source-only theoretical addendum. The [note](OPTIMAL_BASIS_AND_MASKS.md) characterizes optimal read bases through hyperplane mass and the threshold-rank filtration, and distinguishes that filtration from masks commuting with all aligned relative optimal transformations.

- `OptimalBasis.lean`: Lean 4.19 / `Init` only. Five-direction exact atomic examples, abstract mask invariance iff commutation, injective/surjective observation bridge, shear counterexample, natural-number recurrence.
- `finite_optima.py`: exact-Fraction 2D enumeration for free masses, nested masses, three-basis cancellation and the rotated unique optimum.
- `test_finite_optima.py`: CPU-only reproducibility/regression tests.

Run from repository root:

```bash
python -m unittest discover -s mechinterp/bpd/theory -p 'test_finite_optima.py' -v
lean mechinterp/bpd/theory/OptimalBasis.lean
```

The dedicated workflow uses Lean 4.19.0 and captures axiom output. The finite Lean results, the general real-matroid theorem, the reduced-map Lipschitz bound, and neural BPD causal claims have separate evidence status.


### 2026-11-17 continuation

- [FiniteFiltration.lean](FiniteFiltration.lean) proves the finite rank-profile score-optimality iff full threshold-saturation theorem, conditional on a greedy witness. [GreedyRank.lean](GreedyRank.lean) proves that a greedy scan constructs the prefix-rank-saturating witness under explicit matroid-style rank laws. Formalizing the laws for concrete real matrix ranks and the sorted-level embedding remains outstanding.
- [finite_matroid_audit.py](finite_matroid_audit.py) and [test_finite_matroid.py](test_finite_matroid.py) independently audit the greedy witness and the optimizer/filtration equivalence for 96 exact-rational finite vector-matroid instances in dimensions 2–4.
- [PHASE_DISJOINT_PROTOCOL.md](PHASE_DISJOINT_PROTOCOL.md) freezes a BPDModel-based two-phase trained-dictionary control and mixed-source negative arm. [RESULTS_PD01_20261117.md](RESULTS_PD01_20261117.md) records its **failed** two-seed recovery gate, with a complete GitHub Actions artifact.
- [phase_disjoint_control.py](phase_disjoint_control.py) uses the actual shared V,U component wrapper and trains both factors throughout stage 1. A frozen-factor gate-only refit runs under calibration-set tolerances, with held-out evaluation only after selection.

Original B1–B4 gates and receipts, VPD parity controls, and phase-disjoint dictionary-training protocol remain unchanged.

### BPD-PD-02 follow-up (2026-11-17)

The independently registered [24-seed hard-gate cohort](pd02/PROTOCOL.md) is complete: its [source-locked FAIL report](pd02/RESULTS_20261117.md) and [24 paired-seed CSV](pd02/PER_SEED_20261117.csv) are committed, with an [immutable full JSON CI artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44067120071/artifacts/14321006811). The hard-gated dictionary achieved 24/24 axis recoveries but only 6/24 complete phase-routing recoveries, against 10/24 for matched soft gating (exact paired McNemar p=0.34375).

A [posthoc omission audit](pd02/POSTHOC_OMISSION_20261117.md) counts 498 required-gate false-offs in 12,288 required invocations and zero observed wrong-component activations. Its four-pass product prediction matches the measured correct-board fraction across seeds. The original preregistration and failed primary gate remain unchanged.

### BPD-PD-03 continuation (2026-11-17)

The [registered 24-seed top-1 vs margin-calibration cohort](pd03/PROTOCOL.md) ran 168 full-budget arms and produced a [source-locked scientific FAIL](pd03/RESULTS_20261117.md) under the predeclared paired threshold. Top-1 recovered 24/24 seed-level planted mechanisms, versus 19/24 calibrated-margin and 3/24 fixed-threshold hard-STE. Five top-1-only outcomes gave exact paired McNemar p=0.0625. Top-1 made zero held-out false-offs or false-ons on the phase-disjoint source; the margin arm made 2,048 extra activations in five failed seeds. Both SVD-frozen controls recovered all 24 seeds. Top-1's mixed-source negative control showed its strict one-hot capacity limit (mean MSE 0.488). All original outcomes and the complete JSON remain archived as [GitHub Actions artifact 14321589616](https://github.com/MoralityLabAI/BPD/actions/runs/44070377352/artifacts/14321589616).


### BPD-PD-04 continuation (2026-11-17)

The [prospective adaptive-cardinality study](pd04/PROTOCOL.md) mixed independent one-component passes with full overlap and within-trajectory nested support across 24 fresh model seeds. Its [first-run scientific report](pd04/RESULTS_20261117.md) records a preregistered **FAIL**: adaptive top-1/top-2 succeeded 0/24, calibrated independent binary gates 6/24, and an oracle count control 22/24. All 168 arms completed; exact paired McNemar p=0.03125 in the comparator's direction. The original [full JSON receipt](https://github.com/MoralityLabAI/BPD/actions/runs/44084196892/artifacts/14327829889) is preserved with [paired-seed CSV](pd04/PER_SEED_20261117.csv) and independent test replay.

An exact [Lean 4.19 planted-capacity witness](pd04/PlantedCapacity.lean) establishes that coordinate one-hot writes cannot reproduce an overlap requiring two nonzero coordinates, and that extra gates on zero coordinates do not change reconstruction. The source-law's idealized expected-L0 optimum remains uniquely on the coordinate read axes, with 1.5 factors active per pass; the failed trained gate is a separate optimizer-level observation.
