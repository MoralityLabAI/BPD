# BPD-RP-01: Repair portfolios for quantized BPD deployment

## Status and question

This is a prospective engineering pilot on fresh, public synthetic models. Its
independent unit is a separately trained TinyCrossLoopRouter seed. Four roots
17001 through 17004 are assigned and retained. The pre-freeze smoke uses 17000.
All earlier protected BPD cohorts remain sealed. The evidence class is
`PILOT_ONLY_ENGINEERING`; population inference and scientific confirmation
require a separate design with calibrated physical controls and adequate roots.

The question is whether retaining several alternate global gate-bias codebooks
improves calibration-directed repair of a quantized learned BPD deployment under
the same number of candidate evaluations. Prespecified descriptive allocation
contrasts are beam minus multistart and beam minus greedy in held-out accuracy.
The scientific reference floor is 0.05 absolute accuracy. No matched-sham floor
or resolution estimate is available for this new fixture, so it grants no
readiness verdict. The parent's `evidence/` records the official preflight.

## Frozen model and training recipe

The original `inkling_routing/native_model.py` creates a width16, three-pass
two-slot pointer model. Each root receives 260 source optimizer steps, batch192,
with its existing AdamW recipe. Its eight semantic bit/query patterns recur in
all splits; fresh noise and batch seeds give fresh instances. This measures a
small engineering fixture. The task contains no unseen semantic family.

Only the shared physical `mix` linear is wrapped by the repository's real
`BPDModel`. One learned V[16,16] and U[16,16] pair executes at all three passes.
The fixture adds a singleton position axis before `mix` and removes it after,
preserving the native recurrent forward. Factors start from the core's SVD and
receive 80 joint BPD optimizer steps at initial learning rate0.001, batch32.
Matrix faithfulness coefficient is100000; stochastic reconstruction and
adversarial reconstruction each use0.5. CI uses state_only, d_model16, one
transformer block, two heads, hidden32. Minimality pools with max across passes.
Persistent PGD shares its [1,1,1,17] source across batch and passes and runs two
inner updates plus one outer update per step. Receipts require240 updates.

The unchanged BPD trainer learns CI masks from clean reference trajectories.
Its masked reconstruction branches reuse these reference-conditioned masks.
Deployment uses a separate prehook that computes CI on the current reached
pre-mix message at each pass. It records those full inputs, masks, invocation
indices and physical factor identities. Clean and reached trajectories differ
through recurrent feedback. Training and deployment therefore have an explicit
conditioning difference, whose fidelity is measured.

The ablation config is the core's supported zero mode with sphere_rescale=False.
Uniform pass tolerances are used. Deployment sets every delta mask to zero and
routes every sample through components, so the frozen target residual and its
pass mean contribute exactly zero. State radius and physical displacement are
reported; equal radius is never assumed.

## Fixed damage and repair action

V and U are quantized separately with symmetric3-bit integer codes[-3,3] and
their own maxabs/3 scale. Both scales and code arrays are saved. Every policy
uses the same quantized factors, frozen CI, and frozen scales. Source, CI and
factors receive no optimizer updates during search. Full-precision factors are
saved for the separate unquantized BPD control.

A repair is a four-integer codebook with each entry in[-2,2]. Contiguous groups
of four components share one offset. For component c in group j, the deployed
mask is clamp(g_online[c]+0.25*code[j],0,1). The same codebook applies to every
pass and example, while the underlying g_online depends on each reached input.
The complete finite code space has625 possible global codebooks. The experiment
searches neighborhoods and provides no exhaustive or robustness certificate.
LS16's scalar certificate does not transfer to this multidimensional fixture.

## Search and label exposure

A dedicated64-example calibration batch supplies labeled cross-entropy. These
labels are permitted for supervised policy selection. The proposal generator
accepts only a code tuple. Model rollouts receive a Features object whose labels
property raises. The256-example terminal batch is created and its labels are
accessed only after all three final codebooks have been selected. This order is
recorded in events.json and encoded in execute_root.

Each expansion has nine scored slots: noop, then plus and minus one for each
coordinate. An out-of-range step becomes a padded noop and still consumes a
call. Scores use the actual complete native continuation and calibration CE.
Candidates are sorted by raw floating score, with stable input-order ties.
Parents precede children and earlier slots precede later slots. Retention
deduplicates complete code tuples. Receipts include every scored candidate,
parent link, pool, selected ID, score and padding flag. No score memoization
reduces any assigned call count.

One initial evaluation and one nine-slot bootstrap are physically shared.
Their best three distinct codebooks seed all arms. Greedy expands the best
starter18 times. Multistart expands each of the three starters six times and
returns its best terminal path. Beam expands three frozen parents per round for
six rounds and retains the best three distinct codes after each round.

Each arm is charged10 shared plus162 own calls, totaling172. Physical work per
root is10+3*162=496 calibration rollouts. All full calibration pre-input and gate
traces are saved. Eleven terminal diagnostic rollouts are separate from these
budgets; each selected policy receives exactly one terminal batch rollout.

## Controls, physical metrics, and outcomes

The unchanged quantized online BPD deployment anchors recovery/corruption.
Full-precision online BPD measures decomposition and routing fidelity before
quantization. Frozen native clean output supplies the reference trajectory.
The descriptive task-fidelity flag requires source validation accuracy at least
0.95 and full-precision residual-zero online BPD final class agreement with
clean at least0.95. A failing root is labeled LOW_TASK_FIDELITY_RETAINED and stays
in every aggregate. This flag is separate from numerical full-head identities.
All-open gates with delta1 must recover every head's logits at every pass within
2e-5. Residual-disabled full-precision and quantized all-open runs measure factor
fidelity separately.

A simultaneous permutation of V columns, U rows and CI outputs tests gauge
identity against unchanged quantized deployment. CI-output permutation alone
keeps the factors fixed and serves as an unmatched routing control. Its gate
multiset is preserved, but its delivered physical dose can differ. Per-pass
metrics report full-head error and class agreement, state norms/RMS, trajectory
distance from clean, and local delivered write displacement on each control's
own reached input. Local displacement compares the active gates with unbiased
CI gates and with the frozen clean operator at that same reached input. These
metrics distinguish local write changes from later trajectory separation.

Raw correctness bits are archived for every arm/control and terminal case.
Recovery counts initially wrong cases made correct; corruption counts initially
correct cases made wrong. Net repair equals recovery minus corruption over all
assigned cases. Root-level counts and aggregate descriptive means are reported.
Repeated cases and loop passes do not increase the count of independent roots.

Engineering failures, unsuccessful repair, and roots with no quantized error
headroom remain in the output. A root with zero baseline error carries
INSTRUMENT_HEADROOM_ABSENT. The dose, training steps, code support, scoring rule,
root set and gate groups stay fixed after results. Post-smoke implementation
fixes must be documented before the four-root source freeze. No p-value,
confidence interval, equivalence, power or causal-mechanism claim is emitted.

## Reproduction and provenance

From the BPD repository root:

```bash
python -m unittest mechinterp.bpd.experiments.repair_portfolio.test_experiment
python -m mechinterp.bpd.experiments.repair_portfolio.experiment --phase smoke --output smoke_17000
python -m mechinterp.bpd.experiments.repair_portfolio.experiment --phase hashes
python -m mechinterp.bpd.experiments.repair_portfolio.experiment --phase pilot --source-lock source_lock.json --output pilot_results
```

The pilot requires a parent-supplied remote source lock containing a nonempty
remote_commit, exact source_hashes from the hashes command, and evidence_class
PILOT_ONLY_ENGINEERING. The lock is a provenance gate for this new pilot.
The original core package remains byte-identical to upstream BPD commit
6d3ca23a817095f38789751004594777f54cf7d8. Root receipts include source hashes,
training logs, factor/CI checkpoint, full search lineage, terminal arrays and
per-file hashes. Torch uses one CPU thread and deterministic algorithms.
