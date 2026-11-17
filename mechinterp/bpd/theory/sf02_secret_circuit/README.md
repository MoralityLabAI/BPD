# BPD-SF02 — Trained secret routing in a nonlinear weight-tied model

Protocol: [PROTOCOL.md](PROTOCOL.md) | Parent: [SF01 / PR #28](https://github.com/MoralityLabAI/BPD/pull/28) | Source: [Rodríguez / Spectral Faithfulness](https://github.com/idolum-ai/spectral-faithfulness)

The teacher learns to output public context at pass 1, then either one of two private scalar secrets for its authorized role or the public context for an unauthorized role at pass 2. Shared recurrent parameters are trained from independent synthetic samples.

A separately trained full-rank, 12-component BPD `V,U` pair reconstructs one physical matrix. Six factor write directions are selected by development-only counterfactuals; heldout source-matched donor interventions alter the pass-1 hidden state in the exact current-readout kernel and are scored at pass 2 against 255 equal-L2 random-dark sham interventions. Whole-block and full-dark donor controls distinguish successful source transport from off-manifold drift.

The experiment tests learned factor **geometry**, not learned sparse masks, unique factors, natural-language secrecy, or a deployed security property. Singletons are gauge dependent. The source family, task, and readout are purpose-built synthetic.

Excluded engineering seed 901 for local tests:

```bash
python -m unittest discover -s mechinterp/bpd/theory/sf02_secret_circuit/tests -v
python -m mechinterp.bpd.theory.sf02_secret_circuit.native_parity
```

Prospectively registered fresh exploratory roots:

```bash
python -m mechinterp.bpd.theory.sf02_secret_circuit.experiment --seeds 1013 1019 1021 1031 1033 1039 1049 1051 --out /tmp/sf02_results.json
```

The planned inferential calibration needs >=59 additional independent source roots, and a separately sealed source-family-disjoint confirmation. See `LOCK.json` after source freeze and `RESULTS.md` after run.
