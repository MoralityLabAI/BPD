# BPD-FSI03: actual 7M TRM causal variation versus matched sham

**Status:** retrospective source-locked audit of the published 2026-11-15 native LOOPSCOPE-04 representative Sudoku campaign. The two actual trained checkpoints are `gaoxin492/step_21700` (`932b5e0...`) and `yagizdevre/model.pt` (`d677233...`). This follow-up accesses the existing public 120-board Gramian receipts. It runs **no new inference**, creates **no weights**, and never opens the ARG-03 168-case held-out test.

## What constitutes a mechanism effect?

The registered Gramian experiment compares a readout-sensitive high-eigenvector to an equal-angle, matched-norm random dark direction. Continuous future-logit changes quantify delayed observability; the behavioral causal endpoint is the downstream Sudoku board result. A larger readout norm does not imply better problem solving. BPD's useful mechanism variation threshold must be specified in **task outcome units**, independently of a conveniently responsive feature or fitted observer.

For this retrospective sensitivity audit we inspect the four historically measured physical doses (angle 0.04, 0.15, 0.35, 0.75) and two binary outcomes: exact solve at horizon 12 and whether any board cells differ from clean. Each model sees the same 120 root boards. The two model checkpoints, four doses, and two endpoints form 16 correlated comparisons, not 1,920 independent cases; the audit applies Holm to all 16. The +10-percentage-point task threshold is a *post-hoc diagnostic of practical variation*, not a preregistered winner criterion.

## Task-level results of the already published data

| Checkpoint | Angle | Exact solve difference high − random | Exact paired McNemar p | Any-board-flip difference | Exact flip-test p |
|---|---:|---:|---:|---:|---:|
| gaoxin492 | 0.04 | 0.00 pp (3 wins, 3 losses) | 1.0000 | 0.00 pp | 1.0000 |
| gaoxin492 | 0.15 | +1.67 pp (4, 2) | 0.6875 | +1.67 pp | 0.5000 |
| gaoxin492 | 0.35 | −0.83 pp (1, 2) | 1.0000 | −0.83 pp | 1.0000 |
| gaoxin492 | 0.75 | +1.67 pp (5, 3) | 0.7266 | −0.83 pp | 1.0000 |
| yagizdevre | 0.04 | −2.50 pp (10, 13) | 0.6776 | +3.33 pp | 0.3438 |
| yagizdevre | 0.15 | −1.67 pp (10, 12) | 0.8318 | +0.83 pp | 1.0000 |
| yagizdevre | 0.35 | −1.67 pp (10, 12) | 0.8318 | +3.33 pp | 0.3877 |
| yagizdevre | 0.75 | −4.17 pp (9, 14) | 0.4049 | +5.83 pp | **0.0923** |

Among 16 exact paired binary tests, **none has unadjusted p<0.05**. Holm therefore confirms none. The maximum positive exact-solve advantage is +1.67 percentage points, far below the diagnostic 10-point floor. The largest any-board-flip advantage is +5.83 points and is not statistically established even before correction. A board flip is not necessarily a correction: always inspect correct-to-wrong corruption and recovery.

## Delayed readout can differ while task outcomes do not

At angle 0.15, the gaoxin high-eigenvector's average lag-12 readout L2 exceeds random by **27.16**, but exact solving improves only +1.67 percentage points with paired p=0.6875. At angle 0.35, its average lag-12 L2 actually trails matched random by **13.86**. For the second checkpoint at angle 0.04, high-minus-random lag-12 L2 averages **+15.28**, while exact solving falls −2.50 percentage points. These continuous response values are reported as descriptive, since direction selection and dose exploration occurred on these same public boards.

## What to do next, before spending GPU

Use the merged [EvalMath mechanistic-power skill](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) with **fresh development roots**. Pre-register one physically matched directional intervention at one dose, with a meaningful corrected-board effect above the empirically calibrated sham variation. Calculate the number of independent source families required before collecting a sealed confirmation. If the pilot yields only ~1–3 pp task differences, choose a stronger intervention or a more discriminating task before scaling to another 64-board evaluation.

The new native follow-up should preserve all original BPD B1–B4 controls: frozen checkpoint source hashes; phase-disjoint training controls; actual matched-norm random and self swaps; invariant full-head pre-perturbation checks; and corrected-versus-corrupted board outcomes at a fixed delayed horizon. Use `sudoku_roster` to generate a nonisomorphic, clue-/difficulty-stratified new board roster and run the existing `loopscope04_trm` at local pinned 7M TRM weights. Treat the source **board** as the unit. Fold both checkpoint responses per board for a cross-checkpoint generalization question, keeping checkpoint identity visible.

The existing native runner explicitly requires local pretrained weights and does not download models. The 168-case ARG-03-N held-out set remains sealed. A genuine new native trial requires an available local gaoxin492/yagizdevre checkpoint and screened fresh puzzle CSV.

## Reproduce without GPU

With merged EvalMath FSI-01 on `EVALMATH_FSI_FILE`:

```bash
EVALMATH_FSI_FILE=../EvalMath/src/spiral_ln/evaluation_finite_sample.py \
  python mechinterp/dark_fiber_gyms/fsi03_native_audit.py \
  --out runs/fsi03_native_public_audit.json
```

This code reads only the three committed public files under `supplements/loopscope04_arg03_receipts/loopscope04/`, validates the aligned 120 root IDs, recomputes every binomial tail with EvalMath and emits input SHA-256 hashes. A dedicated source-only CI job checks the result without PyTorch or GPU access.
