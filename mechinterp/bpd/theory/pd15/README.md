# BPD-PD-15: norm-weighted source-observable Gaussian sparsity

**Prospectively registered November 18, 2026.** The fixed seed roster, objective, strict physical endpoint, and decision criterion were committed to [CONFIG.json](CONFIG.json) and [PROTOCOL.md](PROTOCOL.md) before implementing any PD-15 scientific training. The first scientific run is released only after engineering checks and one source lock.

## Purpose

[PD-14](../pd14/RESULTS_20261118.md) showed a strict mechanism scientific FAIL despite the 16/16 source-objective diagnostic inversion being corrected with narrower sigma. The original weak-margin sigma=.05 recovered 16/16 at nine Givens sweeps. Strong clustered sources remained difficult: narrow sigma=.05 recovered 3/16 and original sigma=.15 recovered 0/16.

PD-15 tests a different way to break correlated latent coactivations. Under orthogonal Q, the input norm is latent-rotation invariant, allowing an entirely observed-data weight **w(x)=exp(-||x||²)**. This downweights dense/coactive training trajectories without a cardinality classifier or a true-Q hint. It emphasizes the independently informative low-cardinality part of the original source.

A pre-data support-law quadrature calculation gives an explicit direction: the original four-pair 45° remix exceeds Q in Gaussian mass by +0.011947 at sigma=.15, while λ=1 norm weighting makes the remix score **0.009518 below Q**. The weight-crossing is near λ=.53253, with estimated effective sample fraction 0.484 for λ=1.

## Locked factorial

**32 new clustered strong-margin 16D source rotations** (one independent statistical unit each) × five arms:

| Arm | σ | λ | Candidate angles |
|---|---:|---:|---:|
| uniform_wide, primary control | .15 | 0 | 39,960 |
| **weighted_wide, primary** | .15 | **1** | 39,960 |
| uniform_narrow, secondary | .05 | 0 | 39,960 |
| weighted_narrow, secondary | .05 | 1 | 39,960 |
| planted_Q_oracle, privileged | none | none | 0 |

All learned arms use the same 2048 four-pass training trajectories, same deterministic random QR, original lexicographic Givens order and **nine complete sweeps** over 37 global-angle candidates per pair. Their physical read factor V is 16×16 orthogonal, decoder U=Vᵀ, and zero-ablation BPDModel/TiedComponentLinear evaluates eight original PD-12 seen/withheld families with exact 0/1 threshold masks.

The frozen *mechanism* criterion requires weighted_wide ≥18/32 complete recoveries, ≥10 net paired weighted-only wins over uniform_wide, and exact two-sided McNemar p≤.05. The independent *objective* diagnostic requires the predicted Q/remix ranking reversal on ≥30/32 separately drawn 1024-board strong-margin sources. Scientific PASS requires both.

The other two learned arms report the bandwidth/weight interaction as descriptive secondary results only. All sources and physical gates from PD-01 to PD-14 remain unchanged.

## Reproducibility

- [experiment.py](experiment.py): source-only norm weights, identical full-angle score counts and original physical interventions
- [test_experiment.py](test_experiment.py): excluded seed 4001; no scientific seeds fitted in preflight
- [NormWeightBudget.lean](NormWeightBudget.lean): exact matched 39,960 candidate angles and 327,352,320 observed-vector comparisons
- \`bpd-pd15-preflight.yml\`: frozen roster and excluded-seed CI
- \`bpd-pd15-registered.yml\`: one-time source SHA lock and 160-arm raw JSON archive

The objective witness uses true Q **only** to score a known off-axis transformation in the diagnostic after fitting. Weight construction and all searches accept only observed x.
