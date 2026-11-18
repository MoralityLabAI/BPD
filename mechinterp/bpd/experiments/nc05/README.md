# NC-05: frozen rank–dose frontier on a dense learned recurrence

The prospectively registered protocol is [PROTOCOL.md](PROTOCOL.md), source-only commit `f7f1d2346f9d3f871049a3d03416ef7980fe90f8`. This follows the fixed rank-two **fidelity failure** in [NC-04](../nc04/RESULTS.md).

NC-05 trains eight **new** 20D dense recurrent models at roots 500–507 using the same dataset and optimization budget as NC-04. It takes no external labels into projector discovery. Fixed rank grid **2,4,6,8** and physical-write gains **0.75,1.0** define an eight-cell comparison family. Keep-only interventions run four closed-loop passes through the physical shared BPD `override_read` interface.

Every rank receives eight model-blind physical-dose-matched *local* shams from a predeclared 1,024-proposal angular mixture. The sham family differs from NC-04's Haar activation-subspace null, and NC-05 scores are therefore a new evaluation population. Both native and orthogonally rotated bases use identical physical shams, holding model, task and heldout examples fixed.

The primary output is external-label NLL increase relative to the intact model, with sham-adjusted NLL improvement as the causal discrimination statistic. Root-level resampling (8 independent models) gives **99.375% per-cell family-adjusted intervals** across all eight cells. The frozen rank choice is the smallest *full-gain* rank whose upper fidelity CI is ≤0.05 nats and whose lower sham-advantage CI exceeds +0.02 nats. A scientific `NOT_RESOLVED_OR_FAILED` still produces a green successful source/receipt workflow.

From the BPD repo root:

```bash
python -m pytest -q mechinterp/bpd/experiments/nc05/test_experiment.py
python -m mechinterp.bpd.experiments.nc05.experiment --seed 500 --steps 384 --discovery-size 1024 --evaluation-size 1024 --out /tmp/nc05_root_500.json
```

Run remaining roots 501–507 with their own separate `--out` files and aggregate:

```bash
python -m mechinterp.bpd.experiments.nc05.aggregate --directory /tmp --out /tmp/nc05_summary.json
```

Hosted GitHub Actions executes a smaller one-root engineering smoke and the complete eight-root development study with raw JSON receipts. **No confirmation data is generated.** ARG-03-N and AMM-03 protected cohorts remain sealed.
