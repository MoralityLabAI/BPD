# BPD-SF05B Stage 2-B: current-readout and public-query source invariance

[Prospective excluded-root protocol](PROTOCOL.md) · [Stage 2-A donor results](../sf05b_stage2_admission/RESULTS.md) · [Issue #62](https://github.com/MoralityLabAI/BPD/issues/62)

This is a development-only four-arm study on the same two excluded roots 901/902, testing whether the trained *teacher's* shared recurrence can carry source-identity-sensitive authorized answers while keeping its current scalar readout and future public role-U readout insensitive to last-source-tag flips.

Four frozen objectives: exact original Stage 1 baseline, additional current-readout co-closure penalty, additional future public source-invariance penalty, and both. Every arm uses the original 32D hidden/64D tanh feature transition and identical F_train 2–4-write generator and supervised target loss. Donor-source closures are calculated only on the training-family paired source histories.

Every teacher is followed by the exact Stage 1 full-rank physical BPD factor fit and bounded nonprivileged correction policy, both trained only on F_train. Test-time models are evaluated on the **same** Stage 2-A 5–6-write source-aware unconditional, high-contrast and independent public U cohorts, with complete clean/current/future eligibility counts and 16 same-policy corrected matched control arms. Future public output and its gradient may be used at *training time* to compute an explicit regularizer or oracle label; the policy's inference interface excludes them.

Only **901 and 902** are scored. Reserved calibration roots 6101..6159 and confirmation roots 7101..7132 are unrun. The study may identify a better engineering training objective, but cannot establish a root-level statistical mechanism or real language-model secret security.

Run in BPD root with PyTorch 2.5.1:

\`\`\`bash
python -m unittest discover -s mechinterp/bpd/theory/sf05b_stage2b_closure/tests -v
python -m mechinterp.bpd.theory.sf05b_stage2b_closure.experiment \
  --roots 901 902 --samples 512 --shams 16 --out /tmp/sf05b_stage2b.json
\`\`\`
