# RP03 admission selects delay 31 and records a failed transport gate

The [fixed admission rule](ADMISSION_PLAN.md) selects **31 distractor steps**
and **hidden-state L2 dose 0.75** from the two exposed engineering roots at
6,000 task-only updates. The delay passes its headroom gate. Every candidate
dose fails the transport gate, so 0.75 comes from the stated fallback rule and
carries **`TRANSPORT_GATE_FAILED`**. The machine-readable decision appears in
[selection.json](admission_results/selection.json).

## Delay grid

The shortest delay with both ordinary accuracies between 55% and 90% is 31.
The 63-step candidate also qualifies. Each binding cohort contains 64
all-distinct TRAIN memories queried at all four addresses, yielding 256 cases
and an exact 25% modal-value baseline.

| Root | Delay | Ordinary correct | Ordinary accuracy | Binding correct | Binding accuracy |
|---|---:|---:|---:|---:|---:|
| 27000 | 31 | 388/512 | 75.78% | 178/256 | 69.53% |
| 27001 | 31 | 424/512 | 82.81% | 179/256 | 69.92% |
| 27000 | 63 | 298/512 | 58.20% | 124/256 | 48.44% |
| 27001 | 63 | 352/512 | 68.75% | 118/256 | 46.09% |
| 27000 | 127 | 274/512 | 53.52% | 93/256 | 36.33% |
| 27001 | 127 | 304/512 | 59.38% | 90/256 | 35.16% |

These ordinary runs disable steering. The saved RP02 class and the new state
intervention class produce exactly equal ordinary logits on the prescribed
31-step cohort for both roots.

## Dose grid at delay 31

Each root and dose uses 512 separate TRAIN pairs at delay 2 to choose among
eight candidate projectors, with minimum hybrid cross-entropy by edited slot.
A fresh 256-pair causal cohort per root is shared across doses. Its pooled
targeted count is 256; 202 of the 256 unaffected queries originally have the
correct answer. Corruption rates below use those 202 eligible cases.

| Dose | Fixed targeted correct | Fixed targeted accuracy | Fixed unaffected correct | Fixed corruptions | Rotated targeted correct | Rotated corruptions |
|---|---:|---:|---:|---:|---:|---:|
| 0.75 | 46/256 | 17.97% | 169/256 | 49/202, 24.26% | 16/256 | 31/202, 15.35% |
| 1.5 | 66/256 | 25.78% | 136/256 | 81/202, 40.10% | 20/256 | 40/202, 19.80% |
| 3.0 | 105/256 | 41.02% | 86/256 | 128/202, 63.37% | 33/256 | 67/202, 33.17% |

The gate requires at least 25% targeted accuracy and at most 20% corruption.
Higher doses reach more counterfactual answers while changing more originally
correct unaffected answers. The fallback scores, targeted accuracy minus
corruption rate, are -6.29, -14.32 and -22.35 percentage points for the three
doses. This rule chooses 0.75. Every endpoint remains in the receipt.

Full replacement by the true hybrid prefix state yields **205/256 targeted
answers, 80.08%**, and **196/256 unaffected answers, 76.56%**, with 33/202
original-answer corruptions and 27 unaffected recoveries. Its continuation
matches the native hybrid trajectory's logits exactly. The control has the
full hybrid input and uses its natural physical displacement, which ranges
from 0.886 to 2.869. The mean recipient-state radius is 2.148 for root 27000
and 2.378 for root 27001.

## Checks and scope

The [manifest](admission_results/run_manifest.json) pins the saved checkpoints,
plan, executed source and planning record. The bounded grid completed in 1.62
seconds on one CPU thread. Both checkpoint states remain unchanged, every
projected edit is valid, and matched rotations preserve physical dose. Prior
states remain equal after each intervention. Per-case state radii and actual
incoming displacements appear beside the predictions.

The [replay audit](admission_results/replay_audit.json) checks every delay and
dose record from the saved checkpoints, including 10,752 prediction bits and
192 candidate cross-entropy cells. It also recomputes the group choices,
pooled counts, gate decisions and fallback selection, and checks the saved
projector geometry and source hashes.

All data belong to the TRAIN partition. Candidate recipes are selected on
the same two exposed roots used for the learning curve. The selected dose
therefore carries a failed engineering gate into subsequent pilot reporting.
The repository planning verdict remains `REDESIGN_OR_PILOT_ONLY`.
