# BPD-RP-03: fixed and live steering subspaces during training

**All eight development roots are complete.** Fixed causal steering scores
3,248/4,096 ordinary delay-31 answers (79.30%); task training scores 3,243/4,096
(79.17%), the rotated control 3,263/4,096 (79.66%), and tracked geometry
3,243/4,096 (79.17%). B minus A is +0.1221 percentage points, with 239 recoveries
and 234 corruptions. All three Holm-adjusted p-values are 1. The evidence remains
**DEVELOPMENT_ONLY**, and EvalMath returns **REDESIGN_OR_PILOT_ONLY**.

The [complete report](RESULTS.md) includes every root and the targeted-transport
readouts. B's fixed edit gives 202/1,024 requested values while corrupting
192/810 previously correct unaffected answers. Its final fixed/current edit
cosine averages 0.99609. The next design question concerns slot selectivity and
contamination from the donor's other memory slots.

All 34 focused tests pass. Independent replay passes 1,856,350 checks across
26 frozen source hashes, reproducing the recovered audit byte for byte.
The four paired arms share 6,000 warmup updates, task examples and training
budget. The selected long delay is 31 distractor steps and the state-edit
L2 dose is 0.75.

**Engineering transport admission failed.** The fixed dose rule selected 0.75
as its fallback, with 17.97% targeted transport and 24.26% corruption among
originally correct unaffected answers on the two exposed roots. The study
tests how training changes this weak initial intervention. The complete grid
and `TRANSPORT_GATE_FAILED` flag remain in the
[admission receipt](engineering/admission_results/selection.json).

| Arm | Applied training treatment |
|---|---|
| A_task | Ordinary task gradient |
| B_causal_fixed | Task gradient plus causal steering through the warmup projector |
| C_rotated_fixed | Task gradient plus a signed-permutation control of the fixed-projector edit |
| D_causal_live | Task gradient plus causal steering through the current projector |

The main endpoint is ordinary delay-31 answer accuracy with steering disabled.
B−A measures the training treatment's efficacy. B−C tests its physical direction,
and B−D compares a fixed intervention basis with one recomputed from the model's
current factors. All three contrasts belong to one Holm-adjusted family.
The evidence class remains **DEVELOPMENT_ONLY**.

## How the intervention works

The model reads four addressed writes, processes distractors and answers a late
query. Its GRU-style recurrence contains 5,380 trainable parameters and executes
one native BPD V/U pair across all steps, with residual delta zero.

After common warmup, a training-only scan chooses four component indices for
each semantic memory slot. QR decomposition turns their V-column span into an
orthogonal projector. B retains that projector for the paired training stage.
D recomputes the projector from the same component indices as V trains. Both
projectors are detached from the auxiliary gradient calculation.

A training pair supplies receiver and donor states after all writes. Projecting
their difference and normalizing it produces a hidden-state edit of L2 norm
0.75. The edit enters before the first distractor, so it affects the GRU carry
and recurrent operator together. Its continuation is trained toward the known
counterfactual-hybrid answer. The donor state is detached. C applies a fixed,
root-seeded signed permutation to its own fixed-projector edit.

Every arm receives receiver, donor and hybrid as ordinary supervised examples.
All also execute the same diagnostic forward and gradient evaluations. Applied
auxiliary gradients have 0.25 times the arm's task-gradient norm before clipping
and AdamW. Training logs retain the realized doses and optimizer updates.

## What stability means here

A fixed projector preserves a physical subspace. The meaning carried by hidden
states can evolve during training. Fixed training-anchor measurements at steps
0, 80, 160 and 240 therefore record geometric drift alongside targeted transport
and errors introduced on unaffected queries. Those monitors supply diagnostics;
the training recipe receives no adaptive changes from them.

The [RP02 transport audit](diagnostics/rp02_transport_receipt.json) reuses exposed
checkpoints to motivate the hidden-state intervention. The
[engineering learning curve](engineering/RESULTS_2000.md) documents the initial
warmup investigation. The [extension receipt](engineering/extension_results/selection.json)
selects 6,000 updates after the prescribed competence check. Their selection
exposure remains explicit in
[PROTOCOL.md](PROTOCOL.md). Changes in warmup and intervention site occur together
between RP02 and RP03, so the causal comparisons use paired arms within RP03.

## Run and replay

From the BPD repository root:

```bash
python -m pip install -r mechinterp/bpd/experiments/training_steering_stability/requirements.txt
python -m unittest discover -s mechinterp/bpd/experiments/training_steering_stability -t . -p 'test_*.py' -v
```

Smoke roots 27010 and 27011 use a fresh output directory:

```bash
python -m mechinterp.bpd.experiments.training_steering_stability.experiment \
  --phase smoke --out /tmp/rp03_smoke
```

The eight development roots are 27101 through 27108. Their run requires the
published source freeze:

```bash
python -m mechinterp.bpd.experiments.training_steering_stability.experiment \
  --phase development \
  --source-lock mechinterp/bpd/experiments/training_steering_stability/SOURCE_FREEZE.json \
  --out /tmp/rp03_development
```

Saved models and prediction receipts support an independent recurrence replay:

```bash
python -m mechinterp.bpd.experiments.training_steering_stability.audit \
  mechinterp/bpd/experiments/training_steering_stability/results/development \
  --out /tmp/rp03_replay.json
```

The report must retain every assigned root. Ordinary IID and long-delay cases
remain clustered within those models. Causal transport, the address-binding
diagnostic and the full-hybrid-state positive control receive their own
descriptive readouts.

The additive descriptive aggregation has its own source and input hashes and can
be checked against the original receipts with:

```bash
python mechinterp/bpd/experiments/training_steering_stability/summarize_diagnostics.py --check
```
