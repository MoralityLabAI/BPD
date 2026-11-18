# BPD-PD-16: shuffled-weight placebo for norm-weighted Gaussian basis recovery

**Prospective registration, November 18, 2026.** The original 32 fresh source seeds and decision rules were committed in [CONFIG.json](CONFIG.json) and [PROTOCOL.md](PROTOCOL.md) before PD-16 scientific models were fitted.

[PD-15](../pd15/RESULTS_20261118.md) recovered 32/32 complete mechanisms with the fixed source-observable norm weight exp(-||x||²) at σ=.15, compared with 0/32 for an unweighted matched optimizer. The weight also reduced effective sample size to about 48.5%. PD-16 distinguishes the **assignment of weights to observations** from the effect of the weight distribution and ESS alone.

## Four frozen arms

| Arm | Training weights | Nine-sweep angle scores |
|---|---|---:|
| **norm_aligned (primary)** | exp(-||x_n||²) assigned to the original observed vector x_n | 39,960 |
| **norm_shuffled (primary placebo)** | Exactly the same weights in a fixed seeded random permutation | 39,960 |
| uniform_control | All weights one, original Gaussian optimizer | 39,960 |
| planted_Q_oracle | Privileged source dictionary Q | 0 |

All three learned arms use the same 2048 four-pass observed source trajectories, source-independent QR start, σ=.15, original 37-angle full Givens grid, nine sweeps, physical BPDModel/TiedComponentLinear and eight full-four-pass heldout support families. Every learned arm scores 327,352,320 candidate-by-observed-vector comparisons. The permutation is deterministic and independent of any source label, hidden Q or outcome. The aligned and shuffled arms share their entire weight multiset, sum, sum of squares and effective sample size.

This experiment retains the same 16D clustered strong-margin source generator, probability .85 of anchored support motifs, signed latent magnitudes in [.75,1.25], fixed 0.375 read gate and exact binary masks as PD-12 through PD-15.

## Registered gates

The **primary paired mechanism comparison** uses 32 new independent seeded 16D rotations, norm_aligned versus norm_shuffled. Scientific mechanism PASS requires **≥18/32** aligned complete mechanism recoveries, **≥10 net paired aligned-only wins**, and exact two-sided McNemar p≤.05.

After fitting, a separate diagnostic uses the generator's latent support masks to check that norm weights remain correlated with source cardinality while the permutation has broken that association. It requires **≥30/32** seeds with aligned Pearson r≤−.70, shuffled |r|≤.05 and exact aligned/shuffled ESS difference≤1e-10. This diagnostic never feeds support labels into any training arm. The original scientific decision requires both the primary mechanism and diagnostic gates.

The original [source protocol](PROTOCOL.md), [optimizer](experiment.py), [engineering tests](test_experiment.py) and [Lean cost certificate](ShuffleBudget.lean) are frozen before source-locking and scientific execution. A dedicated registered GitHub Actions workflow runs exactly 128 arms and preserves its complete first-run scientific result whether PASS or FAIL.

This is a mechanism-recovery experiment on a synthetic orthogonally mixed sparse-linear identity teacher. A positive result would support the norm-cardinality-proxy interpretation in this family; nonlinear trained TRM transfer remains an independent follow-up.
