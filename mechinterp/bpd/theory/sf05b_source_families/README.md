# BPD-SF05B: structurally disjoint multi-user provenance gym

[Stage 0 registration](PROTOCOL_STAGE0.md) · [Parent issue #48](https://github.com/MoralityLabAI/BPD/issues/48) · [SF05 Stage A/A2 results](../sf05_public_safe/RESULTS_STAGE_A2.md)

A fresh source generator tests whether a recurrent model can associate scalar values with the correct user across interleaved writes and overwrites. Training histories have 2–4 writes with independently sampled source IDs. Structurally heldout histories have 5–6 writes with abstract overwrite templates and randomized label permutations.

The returned batch contains the full event tensor, valid-write mask, public query, independent exact last-write target, metadata, and a mathematically derived source-blind conditional-mean estimator. \`change_last_source\` constructs matched donor/recipient inputs by changing one source tag while retaining every value and the query.

Source-only CPU checks:

```bash
python -m unittest discover -s mechinterp/bpd/theory/sf05b_source_families/tests -v
```

This is **Stage 0 only**. It does not train a new recurrent teacher, fit BPD factors, learn a nonprivileged correction policy, score the reserved calibration/confirmation roots, or inspect any protected ARG-03-N heldout cases. Exact source-blind conditional means are Bayes-optimal for squared error, not for absolute-error losses.
