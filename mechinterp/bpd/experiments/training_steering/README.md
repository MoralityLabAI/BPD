# BPD-RP-02: mechanistic steering during training

RP-02 provides a paired training A/B on a delayed-memory task. A learns from
ordinary task examples. B adds a loss on causal component interventions, while
C applies the same supervision to a rotated physical intervention.

Every arm receives the receiver, donor and counterfactual hybrid as ordinary
supervised examples. The main readout measures ordinary answers with steering
disabled. This keeps counterfactual label availability equal across treatments.

## Measured development result

Across eight paired model roots, ordinary long-delay accuracy is 2,069/4,096
for A, 2,070/4,096 for B and 2,068/4,096 for C. The B-A change is +0.0244
percentage points, with 134 recoveries and 133 corruptions. Both registered
contrasts have Holm-adjusted p = 1. The five-point scientific reference floor
remains unmet, and the official preflight returns REDESIGN_OR_PILOT_ONLY.

[RESULTS.md](RESULTS.md) contains every root and the causal diagnostics. All
saved predictions pass the independent recurrence replay. The source freeze
is [a832c911](https://github.com/MoralityLabAI/BPD/commit/a832c9119685d691ffc942b3e9cb29a00005f1a3),
published before any of the eight development roots executed.

## How the signal enters training

The model reads four memory writes, processes distractors, then answers a late
query. Its hidden-to-gate map uses one trainable BPD V/U pair across all steps.
A training-only intervention scan selects a component group for each memory
slot after a common warmup. Those groups remain fixed during the paired stage.

At the first distractor, B swaps selected donor component reads into the
recipient's recurrent write. The physical write is normalized to L2 0.75. Its
continuation is trained toward the known hybrid answer. C rotates the write
through a frozen signed permutation, preserving that physical norm. Each
applied auxiliary gradient has 0.25 times its arm's task-gradient norm before
AdamW and clipping.

All arms execute the same diagnostic forwards and gradient evaluations. They
start from byte-identical warmup weights and use the same training batches.
The model contains 5,380 trainable parameters. RP-02 trains the native component
parameterization directly; RP-01's post-hoc importance router is separate.

[PROTOCOL.md](PROTOCOL.md) records the exact task partitions, paired treatments,
roster and inference scope. Results remain **DEVELOPMENT_ONLY**. The earlier
[RP-01 result](../repair_portfolio/RESULTS.md) stays unchanged.

## Run the gym

From the BPD repository root, install the pinned CPU dependencies and run the
focused checks:

```bash
python -m pip install -r mechinterp/bpd/experiments/training_steering/requirements.txt
python -m unittest discover -s mechinterp/bpd/experiments/training_steering -t . -p 'test_*.py' -v
```

The two engineering roots use an unused output directory:

```bash
python -m mechinterp.bpd.experiments.training_steering.experiment \
  --phase smoke --out /tmp/rp02_smoke
```

The eight-root development run requires the published source freeze:

```bash
python -m mechinterp.bpd.experiments.training_steering.experiment \
  --phase development \
  --source-lock mechinterp/bpd/experiments/training_steering/SOURCE_FREEZE.json \
  --out /tmp/rp02_development
```

Root subsets can be selected with `--roots` for resuming bounded work in a new
directory. Every assigned root must appear in the aggregate research report.
Training follows the fixed schedule and retains all checkpoints and outcomes.
No task labels enter the model's forward interface.

## Replay

The independent auditor reconstructs the recurrence from checkpoint tensors,
rebuilds generated inputs and checks the recorded predictions. It also audits
training-batch parity, physical dose and the declared auxiliary-gradient ratio.

```bash
python -m mechinterp.bpd.experiments.training_steering.audit \
  mechinterp/bpd/experiments/training_steering/results/development \
  --out /tmp/rp02_replay.json
```

Raw results contain complete per-case predictions and training traces. The
statistics module treats model roots as independent units and uses a fixed
two-comparison family. Its bootstrap intervals are descriptive.
