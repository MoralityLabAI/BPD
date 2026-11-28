# SG-02 source-bound optional penalty
See [SG-02 pilot and derivations](https://github.com/MoralityLabAI/Compactification-Theory/pull/315).

Use source_bound_penalty(schedule.comp_mask, fixed_sites, assigned_bits) when a label is produced at a common fixed site before all visits. For a hard consumer schedule each label costs its last consumer's distance from the producer. This avoids the unanchored singleton degeneracy. Soft-gate expectation assumes independent Bernoulli consumer gates; for correlated gates use joint hard-mask samples. Bit prices and sites are fixed to prevent degenerate collapse.

Existing training objectives are unchanged. The CPU development pilot showed lower assigned source cost with mixed accuracy outcomes, including a negative rotating-task result and a shuffled control that beat anchored persistent-task accuracy. No native causal claim, physical traffic bound, or confirmation authorization. Local helper tests: five passed, including exact enumeration, gradient parity, shape, singleton and fixed-coordinate/price checks; repository test import adapted. Run python -m pytest mechinterp/bpd/test_source_bound.py.
