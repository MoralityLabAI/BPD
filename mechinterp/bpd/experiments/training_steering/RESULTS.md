# BPD-RP-02: training steering changes one net ordinary answer

The eight-root development experiment completed. Causal steering B answered
**2,070/4,096** ordinary long-delay cases correctly; ordinary training A answered
**2,069/4,096**. B recovered 134 answers and corrupted 133, giving a net gain of
**0.0244 percentage points**. The registered five-point scientific reference
floor remains unmet. Evidence class: **DEVELOPMENT_ONLY**.

The [summary](results/development/summary.json) retains every root and the full
two-contrast family. [Completion and result hashes](results/development/completion.json)
record an 87.91-second CPU run. The two engineering roots, 26000 and 26001, remain
separate in [the smoke summary](results/smoke/summary.json).

## What was trained

Each root supplied a model initialization, input encoding and training stream.
After 200 common warmup updates, a training-only causal intervention scan selected
component groups. A, B and C then started from byte-identical warmup weights and
received 240 paired updates.

The 5,380-parameter GRU-style recurrence trains a native BPD factor map directly:
one V/U pair is shared across sequence steps, with all-open component gates and
residual delta zero. The input map and output head also train. RP-01's post-hoc
BPDTrainer and component-importance network are outside this recipe.

All arms receive receiver, donor and counterfactual-hybrid inputs with ordinary
labels. B adds the loss
from transporting selected donor component reads into the receiver's recurrent
write. C uses a fixed signed permutation of that write. The physical intervention
has L2 norm **0.75** in the GRU preactivation. Each applied auxiliary gradient has
**0.25** times its arm's task-gradient norm before clipping and AdamW; actual
optimizer updates can differ as the models diverge.

The task presents four writes, then distractors and a late query. Each slot holds
one of four values. Training delays range from one to three distractors. Ordinary
evaluation uses delay 2 or delay 7, with **steering disabled**. Training and
evaluation memory tuples occupy disjoint partitions. See [PROTOCOL.md](PROTOCOL.md)
for the data barrier and fixed source recipe.

## Ordinary answer accuracy

There are **eight independent model roots**. Each contributes 512 repeated cases
per ordinary endpoint, yielding 4,096 cases per arm.

| Endpoint | A: task | B: causal steering | C: rotated steering |
|---|---:|---:|---:|
| In-range delay 2 | 2,071/4,096; 50.5615% | 2,077/4,096; 50.7080% | 2,079/4,096; 50.7568% |
| Primary delay 7 | 2,069/4,096; 50.5127% | 2,070/4,096; 50.5371% | 2,068/4,096; 50.4883% |

Counts use 512 ordinary delay-7 cases per arm.

| Root | A correct | B correct | C correct | B−A, pp | B recovery | B corruption |
|---|---:|---:|---:|---:|---:|---:|
| 26101 | 276 | 273 | 275 | −0.5859 | 17 | 20 |
| 26102 | 262 | 264 | 266 | +0.3906 | 16 | 14 |
| 26103 | 269 | 272 | 271 | +0.5859 | 20 | 17 |
| 26104 | 255 | 254 | 252 | −0.1953 | 9 | 10 |
| 26105 | 245 | 249 | 248 | +0.7812 | 20 | 16 |
| 26106 | 266 | 253 | 254 | −2.5391 | 8 | 21 |
| 26107 | 249 | 256 | 255 | +1.3672 | 23 | 16 |
| 26108 | 247 | 249 | 247 | +0.3906 | 21 | 19 |
| Total / mean effect | 2,069 | 2,070 | 2,068 | +0.0244 | 134 | 133 |

Five roots favor B over A; three favor A. The largest loss, root 26106, remains
in the aggregate.

| Delay-7 contrast | Root mean, pp | Exact sign-flip p | Holm p | Descriptive root-bootstrap 95% interval, pp | Recovery / corruption |
|---|---:|---:|---:|---:|---:|
| B−A, primary | +0.0244 | 1 | 1 | [−0.8301, +0.6836] | 134 / 133 |
| B−C, secondary | +0.0488 | 53/64 = 0.828125 | 1 | [−0.1709, +0.2441] | 32 / 30 |

The exact calculations enumerate all 256 root-sign assignments under the stated
independence and sign-symmetry null. Holm covers both contrasts. Bootstrap
intervals use 10,000 seeded resamples of roots and have descriptive scope;
small-root coverage is unestablished. Cases within one model supply repeated
measurements. Both positive-development flags are false.

## Did the trained intervention transport the requested value?

The separate causal evaluation applies the selected-group intervention at delay
7 and scores the known hybrid answer. Each root contributes 256 pairs, split
equally between queries to the edited slot and unaffected slots. These rows use
the same intervention definition on every trained arm.

| Causal endpoint | A | B | C |
|---|---:|---:|---:|
| Targeted slot | 153/1,024; 14.9414% | 155/1,024; 15.1367% | 155/1,024; 15.1367% |
| Unaffected slot | 587/1,024; 57.3242% | 582/1,024; 56.8359% | 581/1,024; 56.7383% |
| All causal pairs | 740/2,048; 36.1328% | 737/2,048; 35.9863% | 736/2,048; 35.9375% |

These are descriptive intervention metrics. B gains two targeted answers over A
and loses five unaffected answers. The ordinary endpoint evaluates the learned
model without the intervention; the causal endpoint evaluates the trained
transport objective. Both show small differences under this recipe.

## Evidence and next design

The [engineering preflight](results/smoke/preflight_result.json) and
[post-development preflight](results/development/preflight_result.json) both return
`REDESIGN_OR_PILOT_ONLY`. The measured effect falls below the scientific floor,
and the independent sham-tail bound remains uncalibrated. Their hypothetical
ten-point planning target supplies no measured gain.

Source freeze [SOURCE_FREEZE.json](SOURCE_FREEZE.json) identifies commit
`a832c9119685d691ffc942b3e9cb29a00005f1a3`, published before these eight roots ran.
The [independent replay](results/development/independent_audit.json) passed
181,773 checks across all eight roots, reproducing prediction bits and aggregate
statistics. It replayed 24,576 ordinary case-conditions and 12,288 causal
case-conditions; maximum scalar error was 2.053e-7. All 15 frozen source hashes
matched. The audit checks saved-checkpoint recurrence and training receipts;
optimizer trajectories remain outside its scope.

A useful next design should first establish competent targeted transport on
separate development data. The present targeted score is about 15%, while
ordinary accuracy stays near 50% at both delays. Component groups were selected
before the fork; their indices remain fixed as V/U train, allowing their semantic
role to evolve. A new version can test whether stabilizing that correspondence
improves targeted hybrid answers while preserving unaffected answers. Its source
and controls would need a fresh freeze before another assigned cohort.
