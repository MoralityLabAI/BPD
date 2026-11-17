# SF05B Stage 2: independent source-aware donor cohorts

[Source-only protocol](PROTOCOL_SOURCE_ONLY.md) · [Exact conditioned Bayes means](THEORY.md) · [Issue #58](https://github.com/MoralityLabAI/BPD/issues/58) · [Stage 1 engineering results](../sf05b_stage1/RESULTS_STAGE1.md)

This source-only generator keeps the structurally heldout 5–6-write source histories from Stage 0. It changes only the source ID on the last valid write, then chooses the **same** authorized query role for recipient and donor from the old or new source tag. It returns independent recipient/donor exact targets, all attempted/qualified denominators and source-blind squared-error conditional means that correctly account for the role-conditioned sampling.

The \`authorized_high_contrast\` stratum uses four deterministic candidate records per retained record and selects only cases with a ≥0.30 **exact oracle-label** difference. The \`authorized_unconditional\` stratum retains all generated records without thresholding by model output. \`public_U\` is a distinct independent cohort whose source labels still change, while both output labels equal the unchanged public scalar.

Source-only verification:

\`\`\`bash
python -m unittest discover -s mechinterp/bpd/theory/sf05b_stage2_donors/tests -v
\`\`\`

No trained model, factor, correction policy, heldout causal score or new 59/32 scientific root is executed here. A Stage 2 teacher/policy protocol and independent-root calibration must be frozen before any future confirmatory model evaluations. The 168 sealed ARG-03-N examples are excluded.
