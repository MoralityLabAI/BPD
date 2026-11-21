# INK-BPD-02: native synthetic recurrent routing, first development outcome

**21 November 2026.** Source-selected, developmental PyTorch CPU intervention. **Primary task-level contrast: 0.000** across twelve task-generator roots. A physically valid routing-gate perturbation changed intermediate gate values, yet every final synthetic task was solved under all five interventions.

## Evidence and reproducibility

- [Registered native protocol](INK_BPD02_NATIVE_PROTOCOL.md), committed before the hosted development evaluation.
- [Native source implementation](native_model.py) and [runner](run_native_dev.py).
- [GitHub Actions native run 44911689235](https://github.com/MoralityLabAI/BPD/actions/runs/44911689235), which trained the source model and scored the original synthetic cohort after the source-only control selection.
- Immutable original [artifact 14709892261](https://github.com/MoralityLabAI/BPD/actions/runs/44911689235/artifacts/14709892261), archive `ink-bpd02-native-dev`, 90-day retention.
- ZIP SHA-256 `9f1f98cb83d7c5859b0dc2e7c81ed6f602a73b48c1c47124e8b606644d95bc5e`.
- Contained `inkbpd02_native_development.json` SHA-256 `e71850d4620341d1bfc6cfddbff6d275c1ec9c3bcd0c536ea34aa80591ff7d87`.
- Model CPU state hash `30500085e2fea51c45840450ca1bef2e9c214a938068b85c2cc9061511b09c44`, from sorted physical tensor contents.
- Pinned PyTorch `2.5.1+cpu` on Ubuntu-24.04; 260 source training steps; 946 trainable parameters; hidden width 16, three shared passes, two CPU threads.
- All four PyTorch native hook/dose/trajectory tests passed. Fresh re-training/replay in the same job was byte-identical. The follow-up [run 44911802984](https://github.com/MoralityLabAI/BPD/actions/runs/44911802984) also passed; the source-only BPD and Fable checks are green.

## Admission and paired results

The source-only ranking at 96 samples admitted 113 candidate coordinate pairs. It selected `[5,8]` as maximum observed gate mover and the disjoint `[1,11]` as minimum observed gate mover. Their source gate shifts were `0.0105865` and `0.0013144`. These pairs were selected **before** the twelve development roots were evaluated, using gate movement only, without task labels.

| Outcome | Routing patch | Disjoint matched sham |
|---|---:|---:|
| Assigned / jointly admitted cases | 384/384 | 384/384 |
| Final correct | 384/384 | 384/384 |
| Mean absolute pass-1 `beta3` shift | 0.0104272 | 0.0015893 |
| Mean full-continuation final-logit L2 shift | 0.000414249 | 0.000101262 |
| Physical per-case L2 dose | 0.16 | 0.16 |
| Source and post-patch writer norm | Equal within native replay tolerance | Equal within native replay tolerance |
| Corrections / corruptions of clean answers | 0 / 0 | 0 / 0 |

**Primary root-macro task accuracy difference = 0.000.** Every one of twelve development roots had all 32 controls admitted and `target_accuracy=sham_accuracy=clean_accuracy=selector_only_accuracy=content_only_accuracy=1.0`. The native model's independent source-validation accuracy was also 1.0.

The target gate moved approximately 6.56 times as much as the sham, and its full-continuation logits moved approximately 4.09 times as much. The reported ratio is descriptive and partly expected because the source protocol explicitly selected the maximum and minimum gate-moving coordinate pairs. It does not establish that the selected coordinates uniquely control routing.

### Per-root intermediate measurements

| Task-generator root | Admitted | Mean abs gate shift, target | Mean abs gate shift, sham | Final task effect |
|---|---:|---:|---:|---:|
| 202701 | 32/32 | .010414 | .001496 | 0 |
| 202702 | 32/32 | .010250 | .001590 | 0 |
| 202703 | 32/32 | .010538 | .001870 | 0 |
| 202704 | 32/32 | .009457 | .000874 | 0 |
| 202705 | 32/32 | .010333 | .001516 | 0 |
| 202706 | 32/32 | .008710 | .001187 | 0 |
| 202707 | 32/32 | .010347 | .001139 | 0 |
| 202708 | 32/32 | .009867 | .001857 | 0 |
| 202709 | 32/32 | .011072 | .001509 | 0 |
| 202710 | 32/32 | .011049 | .001756 | 0 |
| 202711 | 32/32 | .011058 | .001758 | 0 |
| 202712 | 32/32 | .012031 | .002521 | 0 |

## Mechanistic reading

This experiment demonstrates **native hook execution, physical dose matching and propagation into real recurrent model logits**. On this easy task and at the registered dose, the complete recurrent continuation preserves all final decisions despite those internal shifts. The observed perturbations produce a task-success ceiling, so task accuracy cannot distinguish a behaviorally consequential router from a harmless perturbation under the selected 0.16 dose.

This is a *source-selected engineering null* for the exact synthetic checkpoint and intervention. It is not evidence that Goodfire's Inkling coordinator is causally irrelevant, and it is not a BPD decomposition result on pretrained shared weights. Twelve task roots share one trained model; no inter-checkpoint generalization or sham quantile calibration was attempted.

## Registered next instrument redesign

Use new development roots and source-only calibration before any further scoring. Add a physically explicit **gate-inversion positive control** with native downstream continuation to verify that the synthetic task can measure routing disruption at all. Then compare an independently selected source-coordinate perturbation at several *source-calibrated* physical doses that match the sham and preserve native clean competence. Declare only one candidate dose and one complete sham family before any fresh model-task readout.

Separately, test whether a selective route change can be made at the same native write displacement while holding content fixed, and how its downstream logit and semantic effects vary by pass. The original NC-13 admission constraint remains: if any required dose-matched arm fails, mark the corresponding endpoint ineligible. Follow the existing EvalMath preflight for a future source-locked confirmation study and open new independent model checkpoints if claiming architectural generality.

Prospective model and task cohorts in the BPD and LoopTransformers repositories remain sealed.
