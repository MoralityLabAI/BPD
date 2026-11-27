# BPD-RP-03: eight-root development results

**Fixed causal steering changed ordinary delay-31 accuracy by +0.1221 percentage points over task training.** B answered 3,248 of 4,096 cases correctly; A answered 3,243. The registered scientific floor was +5 percentage points. Every Holm-adjusted primary comparison has p = 1. The evidence class is **DEVELOPMENT_ONLY**, with EvalMath verdict **REDESIGN_OR_PILOT_ONLY**.

All eight assigned model roots, 27101 through 27108, completed the frozen recipe. Source commit [`c0b884a4d7949dd5e24c48c1bcdc6d276f2e50f9`](https://github.com/MoralityLabAI/BPD/commit/c0b884a4d7949dd5e24c48c1bcdc6d276f2e50f9) preceded their execution. The run used 6,000 common warmup updates and 240 paired updates per arm. The hidden-state edit had L2 dose 0.75, and its auxiliary gradient had 0.25 times the task-gradient norm before clipping and AdamW.

## Ordinary task performance

Steering is disabled during these evaluations. Each endpoint contains 512 examples per root and arm, giving 4,096 repeated observations across **eight independent roots**.

| Training arm | Delay-31 correct | Delay-31 accuracy | Delay-2 correct | Delay-2 accuracy |
|---|---:|---:|---:|---:|
| A: ordinary task training | 3,243 / 4,096 | 79.1748% | 4,014 / 4,096 | 97.9980% |
| B: fixed causal projector | 3,248 / 4,096 | 79.2969% | 4,039 / 4,096 | 98.6084% |
| C: rotated control | 3,263 / 4,096 | 79.6631% | 4,028 / 4,096 | 98.3398% |
| D: projector tracked through current factors | 3,243 / 4,096 | 79.1748% | 4,038 / 4,096 | 98.5840% |

The [original summary](results/development/summary.json) reports the complete registered comparison family:

| Ordinary delay-31 contrast | Effect, pp | Descriptive root-bootstrap 95% interval, pp | Exact sign-flip p | Holm p | Recovered / corrupted answers |
|---|---:|---:|---:|---:|---:|
| B minus A | +0.1221 | [-1.7822, +1.8799] | 0.90625 | 1.00000 | 239 / 234 |
| B minus C | -0.3662 | [-1.9043, +1.1230] | 0.68750 | 1.00000 | 213 / 228 |
| B minus D | +0.1221 | [-0.3174, +0.5371] | 0.67969 | 1.00000 | 51 / 46 |

A recovery means that B answers a case correctly when the comparison arm answers it incorrectly. Corruption is the reverse. B's net change relative to A is five answers after 239 recoveries and 234 corruptions.

The bootstrap uses 10,000 root resamples with the frozen seed. Its intervals are descriptive, and coverage at this root count remains unestablished. The exact sign-flip calculation uses signed exchangeability across independent roots. Holm adjustment covers the full registered family. An equivalence claim would require a prospective margin and a corresponding analysis.

### Every assigned root

| Root | A correct | B correct | C correct | D correct | B minus A, pp |
|---|---:|---:|---:|---:|---:|
| 27101 | 404 | 423 | 419 | 419 | +3.7109 |
| 27102 | 394 | 372 | 387 | 377 | -4.2969 |
| 27103 | 407 | 418 | 403 | 413 | +2.1484 |
| 27104 | 390 | 390 | 394 | 389 | 0.0000 |
| 27105 | 396 | 397 | 390 | 399 | +0.1953 |
| 27106 | 440 | 454 | 467 | 451 | +2.7344 |
| 27107 | 437 | 437 | 429 | 439 | 0.0000 |
| 27108 | 375 | 357 | 374 | 356 | -3.5156 |

Each cell has denominator 512. The run retains every root. Both protocol completeness fields are true, and [completion.json](results/development/completion.json) binds the eight original result-file hashes.

## Targeted transport and unaffected answers

The following readout applies the same fixed warmup projector to each arm's final model. Targeted cases query the slot whose value should change; unaffected cases query a different slot. Each subgroup has 1,024 cases across the eight roots. The corruption denominator counts answers that the same model got right before the edit.

| Training arm | Requested value after edit | Unaffected answers correct after edit | Unaffected correct answers corrupted | Unaffected wrong answers recovered |
|---|---:|---:|---:|---:|
| A | 179 / 1,024 (17.48%) | 692 / 1,024 | 167 / 799 (20.90%) | 60 / 225 |
| B | 202 / 1,024 (19.73%) | 674 / 1,024 | 192 / 810 (23.70%) | 56 / 214 |
| C | 174 / 1,024 (16.99%) | 674 / 1,024 | 204 / 817 (24.97%) | 61 / 207 |
| D | 200 / 1,024 (19.53%) | 673 / 1,024 | 202 / 815 (24.79%) | 60 / 209 |

B gains 23 targeted answers relative to A while its unaffected correct count falls by 18. These remain descriptive outcomes on the same model roots. The [diagnostic aggregation](results/development/diagnostic_summary.json) retains the current-projector and rotated readouts as well as the fixed-projector results.

At B's final checkpoint, replacing the entire state with the true single-slot hybrid state produces **824 / 1,024 targeted answers (80.47%)**. The replacement matches native hybrid-continuation logits exactly. This control uses privileged hybrid inputs and an unmatched natural displacement, averaging L2 1.4628 for B. It supplies a continuation ceiling within this task.

The [engineering admission](engineering/admission_results/selection.json) retains **TRANSPORT_GATE_FAILED**. Its original selected sparse edit achieved 46 / 256 targeted answers and corrupted 49 / 202 initially correct unaffected answers. RP03 tested the effects of training with that intervention after the competence and delay-headroom checks had passed.

## Component geometry during training

B holds the physical projector fixed. D recomputes the projector from the same selected factor indices at each update. Both detach projector geometry and donor state from the auxiliary gradient. An explicit factor-alignment penalty remains untested by these arms.

At the final evaluation readout:

| Training arm | Mean normalized projector drift | Mean cosine between fixed and current edits | Mean distance between those edits |
|---|---:|---:|---:|
| A | 0.05575 | 0.99684 | 0.05527 |
| B | 0.06083 | 0.99609 | 0.06160 |
| C | 0.06465 | 0.99585 | 0.06441 |
| D | 0.06092 | 0.99615 | 0.06145 |

Normalized drift is `||P_current - P_warmup||_F / sqrt(2 * 4)`. The cosine compares realized edits on the same cases. These measurements show closely aligned physical edits over the 240 paired updates. Establishing a stable memory role also requires selective control of the requested value. Hidden-state meaning can change while projector distance remains small.

The fixed anchors at steps 0, 80, 160 and 240 use **training-partition cases**. They supply repeated diagnostics within each model root. Final readouts use separate evaluation tuples after training. Intermediate anchor measurements were saved as receipts, so their causal readouts receive arithmetic checks; the available warmup and final models receive inference replay.

Five roots select four distinct groups for the four memory slots. Roots **27106, 27107 and 27108** each assign the same group to slots 0 and 2. Candidate groups contain four contiguous component indices. The complete assignments, anchor trajectories and per-root diagnostics appear in [diagnostic_summary.json](results/development/diagnostic_summary.json).

The all-distinct address-binding diagnostic scores A/B/C/D at **1,511 / 1,547 / 1,592 / 1,537 correct out of 2,048**. A query-blind predictor that outputs one value for every address of a memory has exactly 25% accuracy on this balanced diagnostic. The four queries for a memory remain repeated observations.

## Resolution and evidence scope

[EvalMath's saved preflight](results/development/preflight_result.json) returns `REDESIGN_OR_PILOT_ONLY`. Running the external EvalMath skill's strict preflight against the archived input reproduced this result exactly.

| Quantity | Value |
|---|---:|
| Independent units | 8 model roots with four paired descendants |
| Registered primary endpoint | Ordinary delay-31 accuracy, steering disabled |
| Planned / actual inferential comparisons | 3 / 3 |
| Physical edit dose | L2 0.75 |
| Applied auxiliary-gradient ratio | 0.25 of each arm's task-gradient norm before clipping |
| Observed B minus A | +0.1221 pp |
| Scientific variation floor | 5 pp |
| Largest observed absolute C minus A sham effect | 5.2734 pp |
| Population 95th-percentile sham bound at 95% confidence | Uncalibrated |
| Approximate effect resolvable at eight roots | 8.2530 pp |
| Planning target | 10 pp |
| Selection exposure | Engineering roots and smoke selected the recipe; eight development roots now consumed |
| Evidence class | DEVELOPMENT_ONLY |

The planning calculation is conditional on its normal approximation. Its hypothetical ten-point target supplies a design reference. The observed treatment difference is 0.1221 points. Eight roots also leave the requested population sham-tail bound uncalibrated; the sample-maximum order-statistic rule requires 59 independent sham roots for 95/95 coverage. The consumed development readout has its own `heldout_unsealed` flag, and the protocol allocates no confirmation cohort.

The causal controls include a physically matched signed-permutation edit and the tracked projector. C constructs its fixed-projector edit from C's current receiver and donor states, then applies the root-seeded signed permutation. Its edit direction can therefore diverge from B's as training proceeds. A receives the same ordinary receiver, donor and hybrid labels as the other arms. Every arm executes the same diagnostic forward and gradient evaluations. The full-hybrid-state control has its separately stated privileged scope.

## Verification and compute receipts

All **34 focused tests pass**. The independent recurrence audit passes **1,856,350 checks** across all eight roots, with exact prediction bits and aggregate statistics. All **26 frozen source hashes** match the published source and local files. Maximum scalar comparison error is **9.5367431640625e-7** against tolerance 1e-5.

The recovered audit was reproduced byte for byte. Its SHA-256 is `9b6ccfe3acf6715c12c3d4bc0a05ae069e0fed4e7f68c65ce1d335255d5ff4f6`.

The audit replays saved warmup/final recurrence and intervention outputs, including 64 endpoint anchor records. It reconstructs 48,000 warmup batches and 1,920 paired-stage input batches, and checks 7,680 training rows. Another 64 intermediate anchor records receive receipt checks. Optimizer trajectories remain outside this replay's scope. See the complete [audit receipt](results/development/independent_audit.json).

The recorded CPU run used one Torch thread and completed in **389.169 seconds wall time**. Summed incremental process CPU time was **387.290 seconds**. Peak process RSS was **353,540 KiB, or 345.254 MiB**; this is the maximum process high-water reading across sequential roots. The runtime was Python 3.12.14 with PyTorch 2.9.1+cpu. These measurements describe this small 5,380-parameter gym on the recorded host.

`summarize_diagnostics.py` was added after development to aggregate existing receipts. It records its own source hash and every input hash, validates the saved prediction arithmetic and recovery/corruption identities, and checks overlapping counts against the frozen primary summary. It adds zero inferential tests. Original protocol, trainer, primary statistics and result receipts retain their frozen bytes.

## Follow-up design: isolate the intended memory change

The next bounded diagnostic should examine whether the donor's other slots contaminate the sparse edit. On exposed frozen checkpoints, use a separately declared development batch to compare the projected donor displacement with the projected displacement to the true single-slot hybrid, at the same L2 dose:

```text
d_donor  = normalize(P @ (h_donor  - h_receiver)) * 0.75
d_hybrid = normalize(P @ (h_hybrid - h_receiver)) * 0.75
```

The projected hybrid direction uses privileged information and serves as a diagnostic. Record the fraction of the true hybrid displacement outside the selected span, along with the cosine between these projected directions. Report requested-value accuracy and unaffected-answer corruption by edited slot, including the slots that share a component group. Keep matched rotations and full hybrid-state continuation in that readout.

Improvement under the projected hybrid direction would motivate cleaner donor construction or a learned slot-specific displacement. A large projection residual would direct attention to group selection or how training represents the memory. A later alignment-loss experiment can then target a demonstrated form of role drift. Any recipe chosen from these diagnostics needs fresh model roots for its next development comparison.

## Reproduce the published checks

From the repository root, with the pinned requirements installed:

```bash
python -m unittest discover \
  -s mechinterp/bpd/experiments/training_steering_stability \
  -t . -p 'test_*.py' -v

python -m mechinterp.bpd.experiments.training_steering_stability.audit \
  mechinterp/bpd/experiments/training_steering_stability/results/development \
  --out /tmp/rp03_development_audit.json

python mechinterp/bpd/experiments/training_steering_stability/summarize_diagnostics.py --check
```

The repository workflow runs the focused tests and independent replay for both the engineering and development receipts. The [source lock](SOURCE_FREEZE.json), [run manifest](results/development/run_manifest.json) and [original per-root artifacts](results/development/) provide the inputs for these checks.
