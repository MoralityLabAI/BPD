# NC-02: planted recurrent communities

**Prospective protocol:** [PROTOCOL.md](PROTOCOL.md) at source-only commit `0a8b0c1d676b24dbc657c9709792efb483f768f2`. This positive-control design follows the NC-01 engineering pilot in [BPD PR #66](https://github.com/MoralityLabAI/BPD/pull/66) and Goodfire's [Neuron Communities](https://www.goodfire.com/research/neuron-communities) paper.

The synthetic model repeatedly applies the **same trained 20×20 physical weight** for four nonlinear passes. Each context selects one of two independent causal lanes; two other lanes have equal-gain physical writes but are omitted from the output. Native ground-truth nodes, for each `lane in {0,1}`, are `(pass t, stage t read coordinate 4t+lane)`, `t=0..3`. The oracle path is consulted only after community discovery.

The apparatus compares a native neuron basis, exact SVD-initialized BPD factors, and an orthogonally rotated basis. Discovery uses pass-indexed gradient attribution and the fixed NC-01 phi-graph. Evaluation runs native closed-loop BPD masked factors and discovery-RMS-matched random shams. An additional oracle-tagged decoy control and a zero-readout null check that the purported attribution is causal rather than mere activation correlation.

## Execution

From the repository root, with Python 3.11+ and PyTorch/Numpy/Pytest installed:

```bash
python -m pytest -q mechinterp/bpd/experiments/nc02/test_experiment.py
python -m mechinterp.bpd.experiments.nc02.experiment --seed 200 --steps 128 --discovery-size 1024 --evaluation-size 1024 --out /tmp/nc02_root_200.json
```

The development CI executes **independent roots 200–207** with the same trained-model and sample budgets, retaining all eight JSON receipts and a root-bootstrap summary. For local development, run the registered command with `--seed 201` through `--seed 207`, then:

```bash
python -m mechinterp.bpd.experiments.nc02.aggregate --directory /tmp --out /tmp/nc02_dev_summary.json
```

The CI source-audit job also runs a small, explicitly **engineering-only** smoke with one root and smaller samples. Neither that smoke nor the eight-root analysis unseals a confirmation sample. ARG-03-N and AMM-03 protected datasets remain sealed.

## Inference boundary

Exact oracle recovery is a *planted positive control*. The stage-shift support is imposed by construction, and the trainable weights are tied across lanes. The SVD arm is a full-rank factorization rather than a learned BPD dictionary. A true trained nonlinear transformer, or a learned sparse basis with residual accounting, is the next transfer target after this experiment.
