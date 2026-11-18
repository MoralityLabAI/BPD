# NC-09: transfer of recurrent causal subspaces between related tasks

**Prospective registration:** [PROTOCOL.md](PROTOCOL.md), source-only commit `22e00fc22cdd5e3403d936294232bcfc051aa2e2`, committed before implementation or training.

This study takes the rank-four result from NC-05 into an independently initialized **joint multitask model**, with two contexts sharing the `x0` branch and using different private input lanes on the other branch. Each model has one physically shared 32×32 tied recurrent matrix and four learned nonlinear passes. There is **no planted sparse hidden-state support** and no external label is passed into projector discovery.

An A-projector discovered from the model's A logit is retained while completing task B. A separate B-projector provides the target-self comparator. Readout NLL is scored on **new externally defined B labels**, stratified into shared and private branches. The preregistered independent-root effect is the difference in A-versus-B retention costs across these branches. Eight local A-projector physical-dose-matched shams test whether any shared-branch transfer distinguishes recovered directions from random nearby ones. Target-B discovery states determine a bounded gain that equalizes source A and target B physical write RMS.

**Development roots:** eight independently trained models 900–907, 640 updates per root; 2,048 base training, 1,024 paired-base discovery and 1,024 paired-base evaluation examples/model. A separate excluded seed **899** is used for CPU smoke and algebraic tests. All protected confirmation cohorts remain sealed.

## Reproduction

From repo root (Python 3.11+, torch, numpy, pytest):

```bash
python -m pytest -q mechinterp/bpd/experiments/nc09/test_experiment.py
python -m mechinterp.bpd.experiments.nc09.experiment --seed 899 --steps 640 --discovery-size 256 --evaluation-size 256 --proposals 256 --out /tmp/nc09_excluded_smoke.json
```

The development job requires an explicit `EXECUTE_LOCK.json` with Git blob hashes of the model, scoring, aggregation and preflight sources. All eight registered model roots are evaluated exactly once under that lock, then scored by `aggregate.py` with 98.333%-coverage family-adjusted three-endpoint whole-root intervals.

The strict EvalMath power preflight is evaluated separately from the scientific decision. Eight model roots cannot provide the required 59-root independent high-quantile sham calibration, so a ready-for-confirmation label is blocked regardless of an engineering signal.

One model's task-marked input coordinates supply a **shared hidden physical chart**. Cross-task transfer within this model does not imply unaligned projector transfer between independently trained model weights.
