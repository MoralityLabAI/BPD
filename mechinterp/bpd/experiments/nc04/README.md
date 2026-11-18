# BPD-NC-04: unconstrained recurrence, self-scored discovery

**Frozen prospective protocol:** [PROTOCOL.md](PROTOCOL.md), source-only commit `0c84c56f8f747737833f0c51201d369dcb24a790`.

NC-03's eigenspace projector recovered planted four-pass stage-shift trajectories and was equivariant under orthogonal BPD factor rotations. NC-04 tests whether that instrument remains causally useful when a 20D shared physical `mix` weight is **densely initialized and freely trained** alongside the rest of a four-pass recurrent network. No stage supports, sign-selected route weights, or intervention label oracle are installed.

Training uses independent external labels for the fixed context-conditioned sign task. **Discovery sees only frozen model logits, activations and gradients.** As a mathematical control, the uncentered gradient second moment ignores any ±1 score sign, so the change from externally aligned to self-aligned score sign is meaningful primarily for the additional positive co-attribution community diagnostic. The primary new empirical stressor is the unconstrained dense learned recurrence.

The primary endpoint is a **held-out external-label NLL improvement** over eight rank-2 Haar subspace shams matched on retained physical write energy, with independently trained models as the bootstrap units. The secondary endpoints are KL to the clean model, readout accuracy, passwise effects, physical intervention displacement, Goodfire-style community graph counts and native/rotated operator parity.

From repo root (Python 3.11+, torch, numpy, pytest):

```bash
python -m pytest -q mechinterp/bpd/experiments/nc04/test_experiment.py
python -m mechinterp.bpd.experiments.nc04.experiment --seed 400 --steps 384 --discovery-size 1024 --evaluation-size 1024 --out /tmp/nc04_root_400.json
```

Repeat for **seeds 400–407** only, then:

```bash
python -m mechinterp.bpd.experiments.nc04.aggregate --directory /tmp --out /tmp/nc04_development.json
```

Hosted CI runs source tests, a one-root reduced engineering smoke and the eight independent development roots. A failed eigengap, task threshold or dose-matching requirement must be retained as a scientific negative/block, with `NOT_RESOLVED_OR_FAILED` in the aggregate.

**Confirmation is unregistered and remains sealed.** The separate ARG-03-N and AMM-03 protected cohorts are also sealed.
