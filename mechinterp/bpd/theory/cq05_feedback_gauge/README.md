# BPD-CQ-05: recurrent feedback and GL(2)-canonical component masks

[Prospective protocol](PROTOCOL.md) | [Theory](THEORY.md) | [Pilot evidence](RESULTS.md) when scored.

The actual BPD \`TiedComponentLinear\` transition is learned from independent behavioral labels. A second learned BPD factorization is frozen, and its full physical product is canonicalized by float64 SVD before training two hard-gate controllers. First pass chooses mask from original h0. Second pass receives *only reached h1*, so swapping its state can change the execution route. Both passes use the same physical read/write weight pair.

The primary endpoint scores **all generated donor-recipient pairs** against the independently generated donor b, with an explicitly equal-dose random current-dark arm. It counts zero-dose and mask-switch pairs instead of conditioning away failures. Support-aware and calibrated input-blind plans are controlled baselines for task accuracy. Raw vs SVD-canonical sheared basis comparisons separate gauge instability from mask execution.

The protocol records the eight independent pilot seeds 1201…1249, the separated training/calibration/test RNG streams, two registered donor contrasts and the EvalMath independent-sham limit. The model is a synthetic linear two-step task, and the second-stage network is behavior-supervised during training. The experiment remains a **pilot** until a fresh independent 59+ root sham calibration and sealed confirmation cohort.

To run engineering (excluded roots only):

~~~sh
python -m unittest discover -s mechinterp/bpd/theory/cq05_feedback_gauge/tests -v
python -m mechinterp.bpd.theory.cq05_feedback_gauge.experiment --engineering
~~~

After a write-once source-locked pilot receipt appears in \`results/LOCKED_COUNTS.json\`, the workflow additionally runs:

~~~sh
python -m mechinterp.bpd.theory.cq05_feedback_gauge.reference --backend independent --check --out /tmp/cq05_independent.json
python -m mechinterp.bpd.theory.cq05_feedback_gauge.reference --backend native --check --out /tmp/cq05_native.json
~~~

Native and independent receipts are archived separately. Source changes after the registered scoring require a new experiment/version rather than rewriting the lock. Nothing reads the sealed ARG-03-N.
