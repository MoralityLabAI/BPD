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


### BPD-PD-05: supervised count versus unsupervised STE gradients (2026-11-17)

The [preregistered 24-seed, eight-arm study](pd05/PROTOCOL.md) tested whether direct cardinality supervision or an alternative STE derivative repairs the adaptive-count failure of PD-04. The [first-run source-locked scientific report](pd05/RESULTS_20261117.md) records a **PASS** for its registered supervised-versus-unsupervised comparison: supervised sigmoid STE succeeded in **21/24** seeds and unsupervised sigmoid in **0/24**, with 21 paired wins and exact two-sided McNemar p=9.5367e-7. Unsupervised identity and temperature-2 derivatives each succeeded in 0/24; supervised identity reached 14/24, the supervised frozen-factor control 23/24, and oracle count 24/24.

[Original complete 192-arm GitHub Actions JSON](https://github.com/MoralityLabAI/BPD/actions/runs/44096809925/artifacts/14335015466), [paired 24-seed CSV](pd05/PER_SEED_20261117.csv), and [exact replay tests](pd05/test_published_results.py) preserve the evidence. [CountSeparation.lean](pd05/CountSeparation.lean) proves a restricted exact quantized source-count threshold witness. The count label is directly computable from the planted 2D input, so the supervised success establishes an optimizer-level positive control rather than unsupervised recovery of hidden mechanisms. All PD-01 through PD-04 results remain intact.


### BPD-PD-06: rotated 4D latent support transfer (2026-11-17)

The [32-seed rotated-source preregistration](pd06/PROTOCOL.md) evaluates latent support cardinalities 1,2,3 under independent unknown orthogonal rotations, holding out two diagonal pairs and a triple throughout all training. The [first-run scientific report](pd06/RESULTS_20261117.md) records **FAIL**: count-supervised top-k recovered 1/32 fully compositional mechanisms versus 0/32 unsupervised (exact paired McNemar p=1.0), while supervised support labels yielded 17/32 and the complete-support oracle 32/32.

The count-supervised arm learned the correct rotated read axes in **32/32 seeds** under its cosine criterion, but only 31.18% of withheld-pair trajectories had exactly correct gate masks. [THEORY.md](pd06/THEORY.md) gives the corresponding idealized population-L0 basis argument; [RotatedSupportMass.lean](pd06/RotatedSupportMass.lean) checks exact finite weights 19/36,15/36,19/36,19/36 and the combinatorial holdout split. The complete [224-arm original GitHub Actions artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44103217529/artifacts/14338021049), [per-seed CSV](pd06/PER_SEED_20261117.csv), and [independent replay](pd06/test_published_results.py) preserve all receipts. No cross-rotation zero-shot transfer or 7M neural TRM result is claimed.


### BPD-PD-07: factor-read gates and compositional transfer (2026-11-17)

The [prospectively registered 24-seed PD07 study](pd07/PROTOCOL.md) tested whether a componentwise binary gate conditioned on its own normalized learned read coefficient could transfer to withheld rotated-four-dimensional support combinations, without count or support labels. Its [source-locked first-run report](pd07/RESULTS_20261117.md) records scientific **FAIL**: jointly trained read-conditioned gate 1/24 versus independent raw gate 0/24, exact paired McNemar p=1.0. Read-detached 1/24, raw count-supervised 0/24, planted-frozen read gate 24/24, full-support oracle 24/24. The primary learnable read factorization had mean read cosine 0.805 and only 4/24 seeds reached cosine ≥0.9.

[Unmodified full 144-arm GitHub Actions artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44109446626/artifacts/14340351331), [24-seed paired CSV](pd07/PER_SEED_20261117.csv) and [independent replay tests](pd07/test_published_results.py) preserve the negative result. [ReadGateSeparation.lean](pd07/ReadGateSeparation.lean) proves that a planted exact read basis supports a fixed gate threshold on all combinations, with a real-norm sufficient margin derived in [READGATE_THEORY.md](pd07/READGATE_THEORY.md). The frozen-factor control passed on 24/24 seeds; learning that basis and routing jointly under the unlabelled surrogate remains unsolved.


### BPD-PD-08: frozen learned dictionary and compositional gate transfer (2026-11-17)

[PD-08](pd08/PROTOCOL.md) first trained a single physical BPD factorization per new seed–rotation pair using count-supervised PD-06 stage-one training, then froze the learned read/write factors bitwise and trained independent label-free gates on matched new source batches. The [source-locked result](pd08/RESULTS_20261117.md) **passed the preregistered primary gate**: a componentwise read-threshold gate recovered 23/24 full six-family mechanisms versus 0/24 for a raw-input independent MLP on identical frozen factors, net +23 and exact two-sided McNemar p=2.384e-7.

Twenty-three dictionaries attained per-component read cosine above the conservative 0.99333 threshold margin. With those factors, a *fixed untrained read threshold 0.5* routed every seen and withheld support correctly. Seed 1237 failed to learn its physical factors, and even a ground-truth support oracle on those factors did not reconstruct the teacher. The learned thresholds approached the upper feasible amplitude limit and slightly harmed exact gate execution. [Original 120-arm JSON](https://github.com/MoralityLabAI/BPD/actions/runs/44114726057/artifacts/14342451949), [paired 24-seed CSV](pd08/PER_SEED_20261117.csv), and [independent replay](pd08/test_published_results.py) preserve the full experiment.

This is conditional on **count supervision in factor discovery**; stage-two read and raw gates received no count/support labels. Fully unsupervised dictionary discovery and real neural mechanism identification remain open.
