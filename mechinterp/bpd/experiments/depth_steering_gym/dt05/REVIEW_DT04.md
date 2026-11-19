# DT04 review and the DT05 changes it motivates

Reviewed against BPD commit
[`1082496445cb99e85cf346597aa87d4cb28f7059`](https://github.com/MoralityLabAI/BPD/tree/1082496445cb99e85cf346597aa87d4cb28f7059/mechinterp/bpd/experiments/depth_steering_gym/dt04)
and the attached **BPD-DT04 Overnight Campaign Report**, dated 19 November 2026.
The source and archived JSON establish the values below. DT05 adds engineering
methods for subsequent experiments. Its implementation gives no new neural
accuracy result.

## What the completed study supports

The original registered verdict remains `INCOMPLETE_OR_INELIGIBLE`: all 59 roots
completed, 56 passed the receipt audit, and 16 passed every admission gate. Three
competent roots exceeded the frozen float32 gauge tolerance. The aggregate
therefore uses 56 audit-clean roots for its ordinary descriptive means. The
subsequent analysis of all 59 roots and the competent subset has the archived
label `POST_LOCK_DESCRIPTIVE_DISSECTION_ONLY`.
See [original summary](../dt04/results/DT04_ORIGINAL_SUMMARY.json),
[original audit](../dt04/results/DT04_ORIGINAL_AUDIT.json), and
[postlock analysis](../dt04/results/POSTLOCK_59ROOT_AND_COMPETENT_DISSECTION.json).

| Observation | Scope | Consequence for the next gym |
|---|---|---|
| Mean task-Gramian advantage over eight signed-Jacobian shams is `+0.0002648305` exact-board fraction at depth 8 across all 59 roots. | Postlock development analysis; about `+0.0265` percentage points. | Keep the matched signed-derivative sham when testing a new intervention. |
| The selected rank-8 state basis has mean one-step closure defect `0.9904`; mean principal-angle drift reaches `51.77` degrees at depth 8 and `53.65` degrees at depth 12. | Geometry measured on DT04's selected basis and calibration states. | Investigate operators that act at each recurrence, or bases transported with the current state. |
| The clean competent subset scores `95.81%` exact at depths 8 and 12. The corresponding means across the 56 audit-clean roots are `41.99%` and `42.05%`. | Nineteen roots selected by depth-6 evaluation competence. | Record competence across every attempted root and retain failed admissions. |
| The sparse BPD rollout scores `94.90%` and `95.39%` exact on those 19 roots at depths 8 and 12. | Factors fit at depths 2, 4, 6; masks derived from a clean teacher trajectory through the evaluation horizon. | Test online routing from the compressed recurrence's own current inputs. |
| The `yagizdevre` checkpoint scores `100%` at step 4, `56.25%` at step 24, and `87.5%` at step 32 on the 16-board bridge cohort. | Descriptive audit of one checkpoint on a small exposed cohort. | Preserve the first publicly valid Sudoku solution instead of discarding it at a later fixed horizon. |

Values come from [RESULTS_20261119.md](../dt04/RESULTS_20261119.md),
[postlock analysis](../dt04/results/POSTLOCK_59ROOT_AND_COMPETENT_DISSECTION.json),
and [hardware/7M archive](../dt04/results/DT04_HARDWARE_AND_7M_AUDIT.json).

### Static state steering needs a different causal contrast

The 59-root postlock analysis reports a `0.046875` upper sham-variation bound
at 95th-percentile/95%-confidence coverage. Its scientific floor is `0.10`
exact-board fraction. The observed mean falls far below that floor. The
competent subset's depth-8 mean advantage is exactly zero in the archive.
Both analyses retain `REDESIGN_OR_PILOT_ONLY`.

The geometry helps explain why a fixed rank-8 perturbation loses alignment
during recurrence. It measures a particular basis, dose, state distribution,
and continuation map. It leaves higher-rank methods and state-dependent
interventions open to a prospective test. A null superiority result also leaves
equivalence unestablished; an equivalence claim requires its own margin and
interval test.

The full dark donor is a positive closure control with mean delivered L2 dose
about `47.2`. The rank-8 comparison arms have mean dose about `2.39`, with
casewise matching for the shams. Those doses differ by roughly a factor of 20.
Use the full donor to assess available recovery headroom. Mechanistic comparisons
must retain the same delivered dose and state-radius accounting as their
controls. See [protocol](../dt04/PROTOCOL_DT04.md) and
[steering implementation](../dt04/causal.py).

### The sparse result includes teacher-derived routing

In `dt04/causal.py`, `_build_pass_schedules()` first runs `target_copy` under
`bpd.target_capture()` through the requested horizon. It collects the clean
teacher's inputs and chooses component masks using
`abs(x_teacher @ V) * norm(U)`. Evaluation requests the full depth-12 trajectory
before running the sparse recurrence. Thus the fit depths constrain the learned
factors, while evaluation routing receives clean teacher state information at
the later depths.

An independently deployable sparse path must choose each mask from its own
current module input. DT05's online-routing experiment should compute the same
score from that input during each invocation, record component usage, and
compare against the archived teacher-scheduled baseline as a separate arm.
Teacher-conditioned fidelity and online fidelity answer different questions.

The term `12/24 components` also describes active components rather than a
measured reduction in process memory or elapsed time. The existing
[`TiedComponentLinear.forward`](../../../components.py) computes the dense
residual branch before multiplying by `delta_mask`, even when that mask is zero.
It also evaluates the component activation matrix before masking. A dedicated
zero-residual online module can remove that discarded residual computation.
Any runtime or memory benefit still needs a workload-matched measurement.

## Preserve solutions as the recurrence advances

The 7M audit provides a direct reason to add first-valid stopping. A Sudoku
candidate can be checked from public inputs: every digit must lie in the
allowed range, all row/column/box constraints must hold, and every given must
remain unchanged. On a solver-certified unique puzzle, a completed candidate
passing those checks is the solution.

Record the first passing candidate and its step for each case. A later iterate
can then be studied without replacing the retained solution. Keep fixed-depth
outcomes in the same receipt so the effect of the policy is visible. Stable
predictions can still violate the puzzle; stability alone must never be treated
as success. The stopping policy must receive no gold answer.

DT04's 7M archive contains aggregate rates rather than enough per-case trajectory
data to replay this policy. New first-valid accuracy and compute savings require
new authorized engineering evaluation. The currently reported `100%` at step 4
describes the exposed 16-board audit. It provides a motivation for the policy,
without establishing performance on fresh puzzles.

Both the SVD kick and its random protected control unstick the remaining
`gaoxin492` S4 board earlier. That observation supports testing a generic
perturbation arm with a fixed dose. It supplies no measured advantage for the
SVD direction. See [hardware audit source](../dt04/hardware_7m_audit.py) and
[archived checkpoint results](../dt04/results/DT04_HARDWARE_AND_7M_AUDIT.json).

## RAM use and throughput

The source records a genuine `6,828,034`-parameter full configuration. At batch
16, its no-grad BF16 inference reaches `61.23 MiB` of peak tensor allocation and
`82.0 MiB` of peak allocator reservation. FP32 reaches `113.05 MiB` and
`144.0 MiB`. These measurements cover forward inference; training and retained
derivative graphs require their own budgets.

Every one of the 18 rows in the hardware archive has
`measured_host_rss_mib = 0.0`. In
[`dt04/memory_guard.py`](../dt04/memory_guard.py), native collection failure falls
through to a zero-valued result. The Windows calls omit explicit ctypes argument
and return signatures for handles. The POSIX fallback uses `ru_maxrss` for both
current RSS and peak RSS. These defects make the archived zero values unusable
as host-memory measurements. The report's `~2.51 GiB` Windows `WorkingSet64`
observation is a separately reported process observation.

Resource admission should therefore distinguish current RSS, process peak RSS,
and available system memory. Unavailable readings need an explicit status. A
budget check that requires an unavailable reading must stop admission. Record
the metric source and use correctly declared Win32 signatures. GPU reports
should retain allocated and reserved values alongside device-wide free memory.

The earlier runner processes roots sequentially. Its causal section also copies
the model to CPU because the historical Jacobian and sham helpers allocate
CPU-default tensors and loop elementwise. These choices limit parallel work
while leaving considerable memory available. Raising a byte ceiling alone does
not change that execution pattern. See
[`evaluate_root_causal_steering`](../dt04/causal.py) and
[`dt04.runner`](../dt04/runner.py).

Useful capacity experiments include a bounded inference batch ladder and
admission-controlled concurrent workers. Immutable input caches can avoid
rebuilding the canonical catalog. A fresh development training comparison can
test a larger physical batch, while preserving the effective optimizer batch
when the purpose is throughput measurement. Record any change to optimization
as an experimental factor.

The old benchmark uses one timing sample per configuration, with no warmup or
repeat distribution. It synchronizes CUDA after the workload. For BF16 full7M,
batch 1 takes `426.87 ms` and batch 16 takes `410.08 ms` in those samples. This
motivates a throughput sweep; it does not give a stable speedup estimate. A new
benchmark should warm up, synchronize around timed regions, repeat measurements,
and report median batch latency plus cases per second. The chosen batch should
be the fastest measured candidate that fits the declared memory headroom.

## Task validity and exposure

The 19 competent Sudoku roots were selected by depth-6 outcomes on the same
64-case evaluation sets later scored at depths 8 and 12. Their means describe a
selected population. Freeze a competence rule on a separate calibration split
before using a later readout for a new generalization claim. Every attempted
root belongs in the run ledger.

The strict synthetic ARC two-rule task remains at `0%` on its heldout rules.
The `99.92%` result applies to a separate task with 192 training and 64 heldout
pointwise four-color lookup functions. Its broader color support and mapping
variety help explain that result. Spatial program transfer and official ARC
performance require their own tests. Storyworld's perfect result measures
terminal completion using a public-state policy. Its implementation expressly
limits the claim to that control setting.
See [task source](../dt04/tasks.py).

Several report details also need to remain explicit:

- The PDF places the `410 ms` and `461 ms` inference values in its step-32
  column. The source runs eight outer steps and stores `depth8_infer_ms`.
- The prose gives `19` high-competence, `16` partial, and `23` low-competence
  roots. The 56-row CSV plus the three recorded gauge outliers yields
  `19`, `16`, and `24` under the stated cutoffs. This correction describes
  archived values and leaves the source report untouched.
- The protocol reserves 256 confirmation roots. The archived preflight input
  uses `confirm_units_planned = 128`. The reproduction preserves that input
  exactly and records the discrepancy.
- The 36-pass gauge failures remain failures under the original `5e-5`
  tolerance. Their physical matrix error is around `3.8e-8` and the report
  records no prediction flips. A revised numerical criterion would need a new
  prospective protocol.

Sources: [CSV](../dt04/results/PER_ROOT_20261119.csv),
[postlock gauge records](../dt04/results/POSTLOCK_59ROOT_AND_COMPETENT_DISSECTION.json),
[protocol](../dt04/PROTOCOL_DT04.md), and
[archived planner input](../dt04/results/POST_DEVELOPMENT_PREFLIGHT.json).

## Reproduced EvalMath gate

The exact archived 56-root `input_spec` was passed to
`skills/evalmath-mechinterp-power/scripts/preflight.py --strict` from EvalMath
commit `bed334f032d31d777bad0e5152e732f57252092b`. The script SHA-256 is
`c2621a7900930f81ed53799ce439ee1c8f2d7006918a04ef57b5381630d15494`.
It exited with status `2` and returned `REDESIGN_OR_PILOT_ONLY`. Its output
matches the archived preflight semantically, field for field.

| Required field | Preserved value or scope |
|---|---|
| Independent unit | Independently trained `PureSudokuTRM` model root. |
| Units | 59 registered; 56 audit-clean roots in the reproduced input; 19 in the posthoc competent subset. |
| Cases | 16 causal-evaluation boards per root, giving 896 for the 56-root input; eight sham arms are repeated measurements on each root. |
| Primary contrast | Rank-8 task-Gramian exact-board fraction minus mean of eight signed-Jacobian shams at depth 8. |
| Physical dose | Casewise L2-matched sham dose; report mean about `2.39`; full dark donor is a larger-dose closure control. |
| Scientific floor | `0.10` exact-board fraction for the primary endpoint. |
| Measurement resolution | Archived input uses `1/16`. |
| Sham floor | Worst observed `0.046875`; 56 input roots cannot calibrate the nonparametric 95/95 bound. The separate postlock 59-root output reports `0.046875` at rank 59. |
| Comparisons | Six registered directional endpoints; additional task, geometry, and selected-subset summaries remain descriptive. |
| Resolution/power | Planner's hypothetical-target resolution is `0.1037216644` for its 128 planned roots. Observed effect `0.0002790179` fails the `0.10` floor. |
| Selection exposure | DT04 development results and the 19-root subset have been inspected; future design choices use this exposure. |
| Causal controls | No edit, full dark donor, eight signed-Jacobian shams, dose-matched BPD/SVD/random protected arms. |
| Verdict | Original study: `INCOMPLETE_OR_INELIGIBLE`. Reproduced planner: `REDESIGN_OR_PILOT_ONLY`. |

The postlock JSON provides the all-59 aggregate planner output, while its
per-root input specification is absent. The artifact manifest lists hashes for
59 original root receipts; those receipt bytes are absent from the committed
tree. Reproducing that second calculation requires those original bytes. The
DT05 evidence record marks it `INPUT_LEDGER_UNAVAILABLE_IN_COMMITTED_TREE`.
No missing sham observations were synthesized from aggregate values.

Reproduction files:

- [Exact archived input](evidence/DT04_PREFLIGHT_INPUT.json)
- [Reproduced planner output](evidence/DT04_PREFLIGHT_REPRODUCED.json)
- [Source hashes, command, exit code, and archive comparison](evidence/REPRODUCTION.json)

## Conditions for a subsequent scientific run

DT04 confirmation roots `2604100..2604355` remain sealed. The current work adds
code and engineering checks. A new scientific run needs a prospectively frozen
development roster with its paired controls, physical dose, minimum effect, and
comparison family. Register the online-routing and stopping policies before
opening its evaluation outcomes. Repeated steps, candidate masks, and puzzle
cells stay attached to their original model root in the analysis.

For a causal mechanism claim, the new independent-root effect must exceed the
scientific floor and a calibrated matched-sham envelope. A larger hypothetical
target cannot supply that evidence. A separate explicit release controls any
confirmation cohort after the development gate.
