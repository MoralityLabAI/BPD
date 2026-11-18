# AMM-07: replicated feedback and root-level reliability inference

The [prospective protocol](PROTOCOL.md) was committed at `ae41cbf6` before AMM-07 source or data. Each root independently trains a rank-two 4D weight-tied physical transition and a separately initialized BPD V/U decomposition. The externally defined donor target and learned causal intervention instrument follow AMM-06, with completely new SHA256 model roots.

Each new model receives **16 independent local calibration pairs**. Both candidate interventions are judged twice, producing **64 noisy binary reports** from a root-shared hidden 10% or 45% error channel. The first eight calibration pairs provide 16 *truth-free duplicate disagreements* for Bayesian reliability classification. The last eight pairs train a fixed ridge residual using both reports. Both primary policies use the same 16 fit targets and the identical unlabeled causal-response probes. `quality_aware` uses the inferred root noise mixture, while `quality_blind` uses equal fixed mixture weights. A `blind_all` strong control fits on all 16 pairs without quality inference. Training-only memory and equally budgeted local ridge are also reported.

For each independent model root there are **1024 fresh test donor pairs** and **1024 separate repeatability pairs**. Its quality classification and BPD factor fidelity are measured from independent sources and reported for auditing, without discarding failed models. The registered primary scientific minimum is +8 percentage points, with three Holm-adjusted root-level contrasts and the corrected EvalMath nonparametric 95/95 sham calibration. There are 64 independent development roots and 128 **sealed** confirmation roots.

## Source-locked CPU workflow

Run from the BPD repository root on Python 3.12 with CPU PyTorch 2.10:

```bash
python -m unittest mechinterp.bpd.experiments.amm07_replicated_feedback.test_experiment -v

python -m mechinterp.bpd.experiments.amm07_replicated_feedback.experiment --phase training --out /tmp/amm07_training.json
# Commit the reproducible training lock before any new development training.
python -m mechinterp.bpd.experiments.amm07_replicated_feedback.experiment --phase development --memory /tmp/amm07_training.json --out /tmp/amm07_development.json
python -c 'import json;x=json.load(open("/tmp/amm07_development.json"));json.dump(x["preflight_input"],open("/tmp/amm07_pilot.json","w"))'
python /path/to/pinned/EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py --pilot /tmp/amm07_pilot.json --strict --out /tmp/amm07_preflight.json
```

Pinned corrected EvalMath revision: `MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a`. The staged CI tests excluded engineering roots before training; development is blocked until the training lock appears. It creates and archives complete raw training, development and planner JSON and checks source-hash and per-case bit locks.

A non-READY preflight or failure of the 52/64 reliability-classification, 56/64 fidelity or 8-point oracle-headroom **veto** leaves all 128 confirmation models untouched. Confirmation also requires a separately authorized source-locked manifest. AMM-03/04/05/06 and ARG-03-N protected splits remain sealed.

This experiment concerns **synthetic trained linear recurrence**, with an independent noisy-label feedback model. The result cannot establish a nonlinear transformer mechanism or live multi-agent Buzz memory transfer.
