# BPD-SF04: interleaved user-source binding

[Prospective protocol](PROTOCOL.md) | [BPD-SF03 signed-tangent study](../sf03_tangent_null/RESULTS.md) | [Rodríguez's Spectral Faithfulness](https://github.com/idolum-ai/spectral-faithfulness)

Train a recurrent network from scratch to process two user-tagged secret writes and a later query. Its target is the most recent value written by the queried user; public queries return a separate public scalar. Two user records may overwrite the same key.

The main causal pair changes **only the second record's source label**, preserving both scalar payloads, chronological order, first event and final query. The current pre-query readout is held exactly fixed while a BPD component coalition or dark-state control is transplanted, then one clean shared transition determines the next query output.

The protocol freezes 8 exploratory teacher roots, 14-of-20 development-greedy physical BPD factors, and signed future-Jacobian sham controls. Full factor-block and exact donor-state transfers provide positive controls. Source-blind Bayes expected-value prediction provides a negative task baseline.

The first two roots 901/902 were used for design and are excluded from new data:

```bash
python -m unittest discover -s mechinterp/bpd/theory/sf04_source_binding/tests -v
python -m mechinterp.bpd.theory.sf04_source_binding.native_parity
python -m mechinterp.bpd.theory.sf04_source_binding.experiment --phase excluded
```

Fresh exploratory cohort:

```bash
python -m mechinterp.bpd.theory.sf04_source_binding.experiment --phase fresh --out /tmp/sf04_fresh.json
```

Native model execution is through `BPDModel/TiedComponentLinear`. The learned factorization is full-rank and GL(20) gauge-dependent. This is a synthetic scalar gym, with no LLM text prompts, human data, or secret-exfiltration claim. It does not alter ARG-03-N's seal.
