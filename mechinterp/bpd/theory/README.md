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
