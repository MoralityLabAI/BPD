# BPD-SF05: public-safe source-binding geometry

[Issue #42](https://github.com/MoralityLabAI/BPD/issues/42) · [Prospective Stage A protocol](PROTOCOL.md) · [Mathematical note](THEORY.md) · [SF04 results](../sf04_source_binding/RESULTS.md)

This first stage evaluates the intersection between the current readout's null space and the one-step **public-query** gradient null space. It uses a true SF04 three-pass trained nonlinear model, along with BPD-reconstructed physical output-state SVD subspaces whose effects are unchanged under invertible reparameterization of the factor coordinates.

For every source-label-only recipient/donor pair, it measures authorized retrieval after a prequery hidden-state edit and the finite change to a separate public query on that same state. Partial singleton BPD edits remain an explicitly gauge-dependent comparator. Equal-dose random nulls inside the public intersection and signed-authorized-gradient-matched nulls test whether the projected edits add causal information beyond local sensitivity.

**Stage A is strictly development-only:** seeds 901 and 902 were already used to design SF04 and cannot be counted as fresh evidence.

Run at repository root with PyTorch installed:

```bash
python -m unittest discover -s mechinterp/bpd/theory/sf05_public_safe/tests -v
python -m mechinterp.bpd.theory.sf05_public_safe.stage_a --seeds 901 902 --shams 16 --out /tmp/sf05_stage_a.json
```

This is synthetic scalar readout engineering. No trained natural-language secrecy, source-family-disjoint result, secure communication property, or mechanism advantage beyond privileged gradient controls has been demonstrated. ARG-03-N's sealed heldout data remain untouched.
