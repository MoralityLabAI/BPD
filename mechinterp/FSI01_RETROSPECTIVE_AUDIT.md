# FSI-01 retrospective audit: BPD and LOOPSCOPE-02 / ARG-03-N

**Date:** 2026-11-17. **Scope:** public, previously evaluated receipts on `main`. The 168-case `ARG-03-N/test.jsonl` remains sealed. This audit reads only the 96-case development grade and LOOPSCOPE-02's already-public 256-case Sudoku results. It trains no models.

## Raw paired results and units

`mechinterp/fsi01_bpd_audit.py` rebuilds paired records from the immutable `benchmark_bundle.json`, `trm_results.json`, and `qwen4b_results.json`. The Sudoku roster comprises exactly 256 unique source-puzzle `origin` identifiers, stratified as four strata of 64. `origin` is used as the cluster key; the premise that puzzle origins represent independent draws is a scientific assumption requiring the benchmark generator's provenance, especially within the difficulty strata.

| Retrospective paired endpoint | Candidate-only | Control-only | Difference | Existing exact McNemar p |
|---|---:|---:|---:|---:|
| TRM target at step 16 vs step 1 | 23 | 4 | +19/256 = +7.42 pp | 0.00031075 |
| TRM target step 16 vs donor-zH control | 32 | 4 | +28/256 = +10.94 pp | 0.00000194 |
| TRM exact board step 16 vs fibre erased | 209 | 0 | +209/256 = +81.64 pp | 2.43e-63 |
| TRM target step 16 vs Qwen 4B | 189 | 3 | +186/256 = +72.66 pp | 3.76e-52 |

All p-values are conditional paired-binomial calculations **under independent board-level paired observations**. The test suite recomputes the exact rational tails from the original binary outcome arrays. FSI-01 also reports outward Clopper–Pearson intervals, root-level resampling, Bonferroni for the four *displayed* contrasts, and the old result hashes. Because these endpoints were selected retrospectively from prior research and the readout is publicly reused, the gate forcibly sets `hypothesis_preregistered=false` and `sealed_readout=false`. It cannot emit `confirmed`.

By difficulty, step 16 vs step 1 changes no target predictions on S1 or S2. The S3 cohort gains 9 with no losses, while S4 gains 14 with 4 losses. This keeps the mechanistic comparison local to the harder generator strata. The original LOOPSCOPE-02 P1–P5 preregistered verdicts remain unchanged, including earlier failed gates.

## ARG-03 development multiple comparisons

The public `grade_dev96.json` contains 96 development rows, 40 named arms and **103 stored paired contrasts**. Recomputing the exact two-sided binomial tails from all `gain`/`loss` counts matches the stored probabilities. Of these comparisons, 19 have unadjusted p below .05. Treating **all 103 reported contrasts as one conservative exploratory family**, a Holm step-down correction leaves **zero** below .05, with smallest adjusted p **0.1559131145477295** (development-calibrated loop-1 versus baseline-1, 22 candidate-only wins vs 5 losses).

This broad family contains overlapping tests and aliases, so it is a retrospective sensitivity audit, **not** a substitute for the prospectively frozen ARG-03 hypotheses. The repeated development evaluations and calibration make the 96-case set unsuitable for independent confirmation. A narrower prospectively specified family may support a different inference, but selecting that family after inspecting this receipt would itself require a fresh confirmation set.

## BPD theory receives a separate evidence label

BPD PR #2's exact expected-L0 maximum-hyperplane-mass formulation, finite rational examples and Lean elementary matrix-mask certificates are **algebraic/finite proofs**, not Bernoulli trials. The optimum-set and invariant-mask scope needs theorem-fidelity review, including the general vector-matroid statement and alignment/gauge caveats. Their validity is assessed with Lean, exact arithmetic and a proof audit, rather than 64-sample confidence intervals.

## Reproduction

With a local checkout of EvalMath at merged commit `5004cab9b366a412290a83945cda76987bb3cd6c` available on `PYTHONPATH`:

```bash
PYTHONPATH=../EvalMath/src:. python -m unittest mechinterp.test_fsi01_bpd_audit -v
PYTHONPATH=../EvalMath/src:. python -m mechinterp.fsi01_bpd_audit --output /tmp/bpd-fsi01-retrospective.json
```

The scoped GitHub Actions workflow pins that exact EvalMath commit. The report includes SHA-256 hashes of every original public input, root counts, exact discordances, BPD nominal and Holm-adjusted p-values, FSI guard reasons and sampling assumptions. No network model inference, GPU run or sealed-data access occurs.
