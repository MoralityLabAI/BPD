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


## Frozen original first-run outcome: scientific PASS

The first [source-locked GitHub Actions run #44237142845](https://github.com/MoralityLabAI/BPD/actions/runs/44237142845) completed all 128 arms, excluded-root engineering checks, strict first-run evidence audit, and archive upload. **Norm-aligned recovered 32/32** complete physical clustered-16D mechanisms; its exact weight-multiset/ESS-matched shuffled placebo recovered **0/32**, as did the unweighted control; planted-Q oracle 32/32. Net +32 paired wins, exact two-sided McNemar p=4.656612873077393e-10. The independent cardinality diagnostic passed 32/32. **Both preregistered science gates passed.**

Mean observed aligned weight–cardinality Pearson r was −0.84935, while the shuffled placebo gave −0.00303. Both had identical mean ESS fraction 0.482630. Norm-aligned had zero false-on/off and **100% exact four-pass support across all eight heldout families**, while shuffled and uniform each had 0% exact withheld-quartet routing.

The [full scientific report](RESULTS_20261118.md), [per-seed first-run CSV](PER_SEED_20261118.csv), [independent source/receipt auditor](test_published_results.py) and [original immutable 747,159-byte JSON artifact #14399136163](https://github.com/MoralityLabAI/BPD/actions/runs/44237142845/artifacts/14399136163) preserve the first result. Original JSON SHA256 fc8d6ccdc12dad07fe05723e5de1ee23da7c5576fef8fa23a4e10021763839e4. The source-locked original experiment and earlier PD01–PD15 scientific outcomes remain unchanged.

[PD-17 PR #79](https://github.com/MoralityLabAI/BPD/pull/79) is the independent prospective radial-camouflage stress test of the source norm/cardinality assumption.
