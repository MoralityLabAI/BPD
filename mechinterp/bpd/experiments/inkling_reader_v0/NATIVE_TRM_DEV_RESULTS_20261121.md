# Native pretrained 7M Attention-TRM: fresh development root campaign

Date: 2026-11-21. The [successful GitHub Actions run](https://github.com/MoralityLabAI/BPD/actions/runs/44902293753) includes the complete JSON evidence in artifact [14706930611](https://github.com/MoralityLabAI/BPD/actions/runs/44902293753/artifacts/14706930611).

Pretrained checkpoint: gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att, step_21700. The checkpoint's expected and checked SHA-256 is
932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3.
The recorded code SHA-256 begins 2544738749cc757d. The test uses genuine pretrained recurrent attention/MLP blocks and a raw z_H hidden-state patch after an H-cycle commit. It does NOT exercise Inkling's Attention Residual cross-loop block reader.

## Frozen development surface

24 Sudoku roots were deterministically generated using seeds 20261121 + 101*i for i=0..23. Each problem has 40 blanks and was checked as having exactly one legal solution before model inference. A second independent backtracking enumerator confirmed 24/24 uniquely solvable puzzles after the run. This is a development cohort, with fresh generated examples but no prospective confirmation claim.

The source model has three H cycles and six L cycles per outer step. A patch is applied at the FIRST blank cell, before observing that model's answer, after H-cycle 1 or 2 on the first outer step. The remaining learned recurrence runs to completion. The head-direction vector is the fixed difference between two digit-head rows; the comparison is a seeded random direction, normalized to the same physical L2 dose. Zero-patch and clean runs are additional controls.

The per-sample physical dose is 0.5 × pre-patch state L2 norm, approximately 11.3137. The clean hand-unrolled recurrence reproduces the model's act_step forward exactly (maximum difference 0.0).

## Original results

| Scenario | Clean boards | Head direction | Random direction |
| --- | ---: | ---: | ---: |
| 40 blanks, outer 1, patch H1 | 23/24 | 23/24 | 23/24 |
| 40 blanks, outer 1, patch H2 | 23/24 | 23/24 | 23/24 |
| 40 blanks, outer 2, patch H1 | 24/24 | 24/24 | 24/24 |

| Scenario | Mean absolute digit-logit change, head | Random sham |
| --- | ---: | ---: |
| Outer 1, patch H1 | 0.08677565 | 0.12443353 |
| Outer 1, patch H2 | 0.06420392 | 0.06798894 |
| Outer 2, patch H1 | 0.01284373 | 0.01633112 |

Each mean is across the same 24 source roots and their 81×9 output digit logits. One-step clean failure occurs at root source-sudoku-20263141, cell 35: predicted 5 instead of 2. The H1 intervention changes a different predicted digit at cell 33, while the selected patch cell was index 0. This is evidence of nonlocal downstream dependence on this development example. Two outer steps recovered the correct complete board on every tested root and intervention arm.

The comparison at the same patch phase across outer steps indicates roughly 85.2% reduction in mean absolute head-direction logit change and roughly 86.9% for the random-direction arm. A similar reduction for random noise implies recurrent attenuation rather than a uniquely identified head-direction mechanism. The exact-board endpoint is near ceiling with only one susceptible root.

## Evidence and reproducibility

- Hardware: GitHub hosted CPU, bounded run; no local RTX 5080 allocation.
- Measured execution durations: 21.12 s for H1 one-step, 21.41 s for H2 one-step and 42.34 s for H1 two-step, excluding environment setup and weights download.
- 24 independent source roots, 96 rows per panel, 288 harder-panel observations; four additional one-root smoke rows.
- Complete per-root model identity, input puzzle, exact target, patch site, nominal and realized L2, all digit predictions, selected-cell logits and other output receipts are in the archived JSON files.
- The earlier 22-blank smoke with relative dose 0.2 produced 81/81 correct for all arms and head/random mean-logit changes 0.0602/0.0395. Preserve separately.
- All zero-patch outputs exactly match clean outputs, and no sealed BPD or EvalMath cohorts were opened.

Scientific classification: native pretrained DEVELOPMENT evidence showing reproducible state-dependent output sensitivity, a single nonlocal cell effect, and attenuation across recurrent computation. The experiment did not identify a reusable sparse mechanistic direction above random controls, and did not test the Inkling attention-residual reader. A new preregistered source population with adequate sham power and a memory-suitable native Inkling checkpoint is the appropriate successor.
