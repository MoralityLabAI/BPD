# BPD-SF01: secret transport, provenance, and delayed visibility

**Prospective, source-only.** See [PROTOCOL.md](PROTOCOL.md) for the hypotheses, layerwise causal intervention plan and sealed-test protections. Inspired by [Spectral Faithfulness](https://github.com/idolum-ai/spectral-faithfulness), with a distinct BPD mechanism question.

From repository root:

```bash
python -m pytest -q mechinterp/bpd/experiments/sf01/test_information.py
```

The nine mathematical sanity tests use no trained model, private data or API keys. The information utilities return **bits**, while BPD commonly uses KL in **nats**. Empirical leakage claims will require independent-root uncertainty, correctly specified observer context, and prospective matched-fiber interventions.
