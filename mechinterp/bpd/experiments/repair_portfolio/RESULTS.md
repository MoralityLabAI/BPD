# BPD-RP-01 results: the quantized fixture has no task-error headroom

BPD-RP-01 completed its four assigned roots and passed the independent replay
audit. The unchanged 3-bit deployment answered all 1,024 terminal cases correctly.
Greedy, multistart and beam each preserved that score, with zero recovered errors
and zero corruptions. Every root carries `INSTRUMENT_HEADROOM_ABSENT`. The present
fixture cannot distinguish allocation policies through correct-answer repair.
These outcomes remain in the complete results and motivate a separately designed
instrument with measurable damage. The evidence class is
`PILOT_ONLY_ENGINEERING`.

The source was frozen remotely at commit
`1f2a87fd16b70ba78098c5060988a3124232e613` before training the four pilot models.
The [execution receipt](pilot_execution_receipt.json) records a run from
2026-11-25 15:20:14.551493 UTC to 15:20:56.750259 UTC, exit 0. Source and search settings
stayed fixed throughout. [The aggregate receipt](pilot_results/aggregate.json)
retains all roots 17001 through 17004; the earlier root 17000 smoke is separate.

## Terminal task outcomes

Each root contributes 256 fresh terminal examples generated after every search
policy was selected. The four trained source models are the independent units.
Their terminal cases reuse the task's eight semantic bit/query patterns with
fresh noise and instance draws.

| Root | Frozen clean | Full-precision BPD | Unchanged 3-bit | Greedy | Multistart | Beam | CI permutation control |
|---|---:|---:|---:|---:|---:|---:|---:|
|17001|256/256|256/256|256/256|256/256|256/256|256/256|256/256|
|17002|256/256|256/256|256/256|256/256|256/256|256/256|233/256|
|17003|256/256|256/256|256/256|256/256|256/256|256/256|256/256|
|17004|256/256|256/256|256/256|256/256|256/256|256/256|188/256|
|All assigned roots|1,024/1,024|1,024/1,024|1,024/1,024|1,024/1,024|1,024/1,024|1,024/1,024|933/1,024|

The baseline for recovery and corruption is the unchanged quantized online BPD
deployment. Each repair arm has recovery 0, corruption 0, and net repair 0. The
unmatched CI permutation control corrupts 91 cases, concentrated in two roots,
for 91.11328125% accuracy. Its physical displacement differs substantially from
the repair interventions, as quantified below.

The descriptive beam-minus-multistart and beam-minus-greedy accuracy contrasts
are both zero on every root. The prespecified scientific reference floor is
0.05 absolute accuracy. Four roots and zero error headroom support an engineering
instrument finding. Equivalence, power, and population effects remain outside
this result's scope.

## What the implementation executed

Every root trained a width 16, three-pass `TinyCrossLoopRouter` for 260 source
steps. The shared `mix` module then received 16 learned BPD components through the
unchanged repository `BPDModel` and `BPDTrainer`. Eighty joint BPD updates used
stochastic reconstruction coefficient 0.5 and adversarial coefficient 0.5, with
max temporal minimality. Persistent PGD performed 240 updates per root using
its shared-batch-and-pass source.

Parameter-change receipts confirm that the learned factors and CI router moved.
Across roots, V's Euclidean change ranged from 0.00614 to 0.00750; U changed by 0.00635
to 0.00910. CI's aggregate Euclidean change ranged from 0.7966 to 1.0623. Positive
adversarial losses and the complete per-step metrics are archived in each
root's `training.json`.

The original BPD engine trains CI from clean-reference inputs. The deployment
hook evaluates CI afresh at each reached pre-mix input, producing genuine
state-dependent routing during native continuation. Full invocation receipts
show one physical V/U pair across all three passes. Reached-message differences
from the clean-reference trajectory ranged from 0.03281 to 0.04413 at the maximum
coordinate level across these roots, so the recorded online inputs differ from
the clean path.

V and U were separately rounded to signed 3-bit codes[-3,3], with frozen scales.
All evaluated repair deployments use zero ablation, sphere rescaling disabled,
and residual delta 0. The frozen clean target matrix remains available only for
the declared reference and identity checks. Every scored repair rollout routes
through the component decomposition.

## Search allocations and chosen codebooks

The search uses a supervised 64-case calibration batch per root. A repair code
contains four integers in[-2,2]. Each integer offsets one contiguous group of
four component gates by 0.25 times that integer. Gates are clipped to[0,1] and
the same codebook is used at every pass.

| Allocation | Shared initial/bootstrap calls | Own candidate calls | Logical total per arm |
|---|---:|---:|---:|
|Greedy,18 serial expansions|10|162|172|
|Multistart,3 paths with 6 expansions|10|162|172|
|Beam,width 3 for 6 rounds|10|162|172|

The common 10 calls execute once, so physical search cost is 496 rollouts per
root and 1,984 across the pilot. All padded noops consume their assigned slots.
Another 44 terminal rollouts cover the 11 declared policy, control and identity
conditions per root. Training rollouts are separate from those search budgets.

| Root | Greedy code | Multistart code | Beam code |
|---|---|---|---|
|17001|[2,-2,2,2]|[2,-2,2,1]|[2,-2,2,1]|
|17002|[0,-2,2,2]|[0,-2,2,2]|[0,-2,2,2]|
|17003|[2,2,0,-1]|[2,2,0,-1]|[2,2,-2,-1]|
|17004|[0,1,-2,2]|[0,1,-2,2]|[0,1,-2,2]|

The differing choices on 17001 and 17003 show that the allocation paths executed
distinct searches. Root 17001 gives greedy extra serial reach. On 17003, beam
finds a calibration CE advantage of approximately 3.7e-9 nats over the selected
greedy and multistart code; its terminal CE is approximately 3.9e-8 nats higher.
These tiny score differences remain descriptive. All selected policies preserve
every terminal answer.

Mean terminal cross-entropy gives the scale of the confidence changes:

| Condition | Mean terminal CE, nats per example |
|---|---:|
|Frozen clean|0.00013102265|
|Full-precision online BPD|0.00013017937|
|Unchanged quantized online BPD|0.00013200188|
|Greedy|0.00013123316|
|Multistart|0.00013123922|
|Beam|0.00013124888|
|CI permutation, unmatched|0.15156637365|

Greedy's CE reduction from unchanged quantized deployment averages 7.687e-7
nats; beam's reduction averages 7.530e-7. The correct-answer endpoint stays at
its ceiling. Every calibration candidate, retention pool, parent link and
selected ID is preserved in `search.json`; full calibration traces preserve
the actual inputs and masks for all 496 calls.

## Fidelity and physical controls

The quantization damage is visible in parameter space. Its operator Frobenius
change ranges from 0.9868 to 1.3645 across roots, approximately 29.2% to 35.3% of the
clean target matrix's Frobenius norm. The full-precision factor residual ranges
from 6.03e-5 to 7.50e-5. A sizable matrix perturbation can coexist with the saturated
task accuracy observed here.

All four source validation scores are 100%. Residual-zero, full-precision online
BPD agrees with clean on every terminal class, passing the frozen 0.95 task
fidelity flag. Its largest all-head absolute logit errors across passes range
from 0.0611 to 0.1527. That records the effect of learned gate masks while the
final predicted classes stay fixed.

The all-open delta 1 engineering identity recovers all clean heads with maximum
error at most 9.54e-7, below its 2e-5 tolerance. Simultaneously permuting V columns,
U rows and the corresponding CI gates also passes, with maximum all-head error
at most 9.54e-7. These checks validate the represented physical operator and its
coordinate bookkeeping.

Permuting CI gates alone changes their correspondence with the fixed factor
coordinates. This control creates the 91 corruptions reported above. Its mean
local write displacement from unbiased gates spans 1.597 to 4.328 across root/pass
cells, while beam spans 0.173 to 0.428. Both quantities use each condition's own
reached input. The comparison has an explicit unmatched-dose label because
permuting a gate multiset does not fix the displacement of the delivered write.

Per-pass receipts also include state norms, state RMS, local displacement from
the clean operator on the same reached input, and trajectory distance from the
clean state. Beam state RMS spans 0.310 to 0.990 across terminal examples and
passes. The state-radius measurements remain diagnostics; this protocol imposes
no matched-radius intervention contract.

## Independent replay and provenance

The [independent audit](pilot_independent_audit.json) reports `PASS` for every
root and for the aggregate. It imports no experiment implementation and
generates no new roots. It verifies 28 artifact hashes, rebuilds 176 retention
decisions, replays 1,984 calibration rollouts, and checks 44 terminal trajectories
from the saved inputs, weights and gates. Calibration reached-input replay error
is exactly zero. The largest terminal replay error is 9.54e-7, arising in the
algebraic delta 1 check. Repair-policy replay errors are zero.

The auditor reconstructs gates from archived raw CI outputs and checks that the
recorded CI inputs match the reached messages in the independent recurrence.
CI-transformer execution is covered by source review and the deployment tests;
independently rebuilding that transformer would be a separate check. The
auditor SHA-256 is
`b49de11fb60678f00579b8587c7460318c79ba5896624b7bb78ce19f47a4159e`.

Before the pilot, four focused tests passed for budget/lineage integrity,
boundary padding and stable ties, the label barrier and quantization, and actual
core PGD plus online/gauge execution. The complete smoke root 17000 and its
source receipt are retained. Source hashes are fixed by
[source_lock.json](source_lock.json); original BPD core files match upstream
commit `6d3ca23a817095f38789751004594777f54cf7d8` byte for byte.

The official strict planner returned `REDESIGN_OR_PILOT_ONLY` for both registered
allocation contrasts before these four roots ran. Its one-root development
input and explicit identity-control/support-bound scope are archived under
`evidence/`. A matched physical sham distribution and a useful error-repair
instrument are still required for a new scientific design. The multidimensional
codebook search supplies no transfer of LS16's scalar robustness certificate.

## Reuse

The runnable gym and [frozen protocol](PROTOCOL.md) provide a concrete starting
point for BPD-assisted deployment calibration. A new repair study should first
register a task and damage recipe with errors to recover, then validate matched
physical controls on separate development roots. Any changed bit width, source
task, training budget or gate support needs a new version and fresh assigned
roots. The four results here remain the archived result of RP-01.
