# RLT-EXIT-02: logits-only learned recurrent stopping on pretrained 7M TRM

Date 2026-11-21. Evidence: **development-only real pretrained inference, with independent receipt replay**.
The original hosted CPU campaign is [Actions #44911864342](https://github.com/MoralityLabAI/BPD/actions/runs/44911864342),
artifact [bpd-rlt-learned-exit-first-dev #14710246746](https://github.com/MoralityLabAI/BPD/actions/runs/44911864342/artifacts/14710246746).
The separate stdlib [cross-audit and DataGoblin export](https://github.com/MoralityLabAI/BPD/actions/runs/44912385166)
passed and retained a second artifact with audit results and TRAIN-only rollout examples.

This is a **real trained STOP controller** over learned recurrent Sudoku-model outputs.
The underlying 7M Attention-TRM is pretrained upstream; the stop controller
itself is a small deterministic six-feature regularized logistic model.
Neither network implements Inkling's cross-loop Attention Residual Q/K/V/MoE readers.

## Data and model identity

- Upstream model: gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att at pinned
  model checksum 932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3.
- Fresh source-root seeds: TRAIN 20261211, CALIBRATION 20261212, DEVELOPMENT 20261213;
  each consecutive root advances the corresponding seed by 101.
- Train: 96 unique Sudoku questions with 50 blanks. First-pass exact valid 50,
  second-pass 95, with 45 newly repaired after one additional outer step.
- Calibration: 48 independent roots, 26 valid at first pass and 47 at second.
  The threshold was frozen at 0.5656608144205763 to maximize calibration
  first-step exits with zero false exits. It stopped 22/48.
- Development: 48 further independent roots with no reused questions.
  First-step exact valid 24, second-pass 47. Neither source nor outcomes
  were used to fit weights or set the threshold.
- Checkpoint, root IDs, generation seeds, solution verification, step logits,
  timing, logistic coefficients and original source/data SHA hashes are retained
  in the original artifact.

Only six numbers summarizing first-step digit logits reach the STOP controller:
mean, minimum, tenth percentile and median blank-cell top-two logit margins;
mean blank-cell categorical entropy; and mean top-token probability.
At inference it receives no Sudoku validity bit, planted target or
future-step result. The exact Sudoku checker supplies labels during
TRAIN and threshold CALIBRATION and serves as the separate oracle-quality
baseline.

## Independent original development result

| Stopping policy | Solved/48 | Outer model steps | Native model CPU sec | Bad early exits |
| --- | ---: | ---: | ---: | ---: |
| Fixed depth 2 | 47 | 96 | 22.288943 | 0 |
| Exact Sudoku verifier | 47 | 72 | 16.693023 | 0 |
| Learned logits-only threshold | **47** | **75** | **17.284104** | **0** |
| Random gate (CAL rate 22/48) | 37 | 77 | 17.713731 | 11 |

The learned gate stopped 21/48 after the first recurrent step.
It saved 21 of 96 recurrent steps (21.875%) and 22.45% of measured
native model CPU time versus the fixed-two-step policy, retaining
the same 47/48 complete-solution count on this independent
DEVELOPMENT cohort. The exact verifier saved 24 steps (25.0%).
The random gate was seeded independently per root at the frozen
CAL stop probability of 0.4583333333333333 and made 11 wrong
first-step exits, reducing final solves to 37.

The regularized logistic fit reached 89.58% first-step training
accuracy at probability 0.5. TRAIN class counts were 50 valid,
46 invalid. The CAL threshold was chosen by the declared
zero-false-stop rule and was NOT adjusted on the 48 development
outcomes. All full-model recurrent hidden states genuinely
continued between first and second outer steps. The entire
run, including collection, took 165.457 seconds on a hosted CPU
runner after dependencies were installed.

## Cross-audited provenance

The stdlib audit independently rechecked all TRAIN/CAL/DEV questions,
fresh root identity and disjointness, Sudoku validity for first and
second responses, supplied feature arrays, logistic score replay,
calibration threshold, learned and seeded random stop decisions,
and model-step/time sums. It revalidated the report against
input SHA-256 dda473da08434484f4d67e8f0e6fa77b4296d9fa1489a810f529c699dc94a7e1.
It did not reexecute pretrained model weights; the original
native execution was completed in the first Actions campaign.

The DataGoblin corpus export independently screened its input
for train-only lineage custody and verifier agreement. It generated
96 verified stop/continue SFT rows, 96 future-outcome rows,
and 45 verified first-to-second-pass repair preferences.
Zero CAL or DEV examples entered those training exports.
All source and output file hashes are in its manifest.
These are supervised STOP and repair examples, **not** fabricated
hidden reasoning chains or actual Inkling teacher samples.

## Scientific scope and next controls

This is one exposed development study on one fixed Sudoku-trained model.
No uncertainty interval or predeclared minimum meaningful difference
supports a general efficiency claim beyond the measured source roots.
The zero observed wrong early exits is a small-sample observation,
not proof that the future false-stop rate is zero. A separately powered,
fresh confirmation root cohort and paired model-accuracy/cost inference
would be needed for that claim.

A general RLT controller should train on multiple independently checked
task families and learn a conditional value of further computation.
For exact proof/Sudoku tasks, the environment checker can be a reliable
stopping source. For other gyms, compare learned stop decisions against
verified correctness, matched-cost random and constant-depth baselines,
with explicit error penalties and no reuse of held-out labels.
The native Inkling cross-loop routing and Goodfire coordinate-2027
hypotheses remain distinct, untested research questions.
