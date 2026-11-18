# BPD-NC-10: cross-task projector transfer between separately trained recurrences

**Prospective, source-only protocol:** [PROTOCOL.md](PROTOCOL.md), committed FIRST as `4354c624103fcc1f7f10125d29b5356583eb3b3d`.

NC-05 found rank-four sufficiency in one dense recurrent task family, NC-06 found a negative weighting contrast at rank three, and NC-07/08 explored model-only rank allocation with preserved failures. NC-10 tests a separate question: whether a passwise causal projector discovered from one trained model's own scores transfers to a second, independently trained model after source-only representation alignment.

Training pairs A and B use **separate independently initialized 20D DenseLoopers**, each with four tied nonlinear physical uses of `mix`. Both tasks share the input `x0` when context `x4>0`; they depend on different inputs when context `x4<=0`. Models share neither weights nor training samples.

For each newly trained pair, source-only discovery estimates **rank-four gradient projectors**. A fixed per-pass orthogonal Procrustes map is fit using **paired, unlabeled positive-context activations** on a separate alignment sample, then transports A's projector into B's native prewrite representation. Final externally labeled B task behavior is measured after native four-pass BPD interventions. Targets and controls include B's own projector, raw unaligned A projector, permuted-pair source alignment, eight **B physical write RMS-matched** locally rotated sham projectors, and an exact within-B orthogonal factor-gauge check.

Run from the BPD repo root:

```bash
python -m pytest -q mechinterp/bpd/experiments/nc10/test_experiment.py
python -m mechinterp.bpd.experiments.nc10.experiment --root 999 --steps 384 --discovery-size 256 --alignment-size 1024 --evaluation-size 256 --proposals 256 --out /tmp/nc10_smoke.json
```

Engineering development: run roots **1000–1007**, each with both A and B trained independently for 384 steps on separate 2048-example training streams, 1024 model-gradient discovery examples, 1024 paired unlabeled alignment inputs, 1024 separate externally labeled final examples and 2048 source-only physical sham proposals. Aggregate with:

```bash
python -m mechinterp.bpd.experiments.nc10.aggregate --directory /tmp/nc10_development --out /tmp/nc10_development/NC10_SUMMARY.json
```

Each pair counts as **one independent root** for bootstrap. There are four registered directional transfer gates and a Bonferroni family-adjusted 98.75% interval per gate. Source checks, scientific nulls, ineligible shams and independent evaluation receipts are preserved, with `ENGINEERING_NULL_OR_INELIGIBLE` as a valid CI result.

**Confirmation remains sealed**. No natural language model mechanism, universal cross-model gauge alignment, or learned SAE causal component is claimed from this synthetic experiment.
