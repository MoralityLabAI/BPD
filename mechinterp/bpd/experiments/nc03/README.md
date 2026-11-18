# NC-03: gauge-covariant recurrent causal subspaces

This prospective experiment follows [NC-02](../nc02/RESULTS.md), whose Goodfire-style coordinate-community algorithm exactly recovered planted recurrent paths in native coordinates but returned no communities after a function-preserving orthogonal factor rotation.

NC-03 instead uses the leading rank-two eigenspace of the pass-indexed **label-aligned output-gradient covariance**. Its projector is transported between orthogonal factor bases and intervened upon through the native `BPDModel.intervene(... mode="override_read")` interface. It compares physical operator maps, causal retention of the full recurrent rollout and eight discovery-dose-matched Haar subspace shams chosen without consulting ground-truth path identities.

**Source registration:** `a48871f2612f9032ddbf612747f972f701585536`, committed before implementation and all NC-03 model/data sampling. The reference architecture is deliberately constrained. Evidence supports instrumentation under this family only.

From repo root:

```bash
python -m pytest -q mechinterp/bpd/experiments/nc03/test_experiment.py
python -m mechinterp.bpd.experiments.nc03.experiment --seed 300 --steps 128 --discovery-size 1024 --evaluation-size 1024 --out /tmp/nc03_root_300.json
```

To aggregate all **eight** independent engineering roots 300–307 with 2,048 training, 1,024 discovery and 1,024 evaluation cases per root:

```bash
python -m mechinterp.bpd.experiments.nc03.aggregate --directory /tmp --out /tmp/nc03_dev_summary.json
```

Any registered failure is retained as an outcome. Confirmation remains separately sealed pending fresh roots and an EvalMath preflight. Protected ARG-03-N and AMM-03 data remain untouched.
