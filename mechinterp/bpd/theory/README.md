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

Original B1–B4 gates and receipts, VPD parity controls, and phase-disjoint dictionary-training protocol remain unchanged.
