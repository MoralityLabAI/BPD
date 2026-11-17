# BPD-PD-02: hard 0/1 phase gates with jointly trained factors

- [CONFIG.json](CONFIG.json) and [PROTOCOL.md](PROTOCOL.md) are the dated prospective registration, pinned **before** any registered run.
- [experiment.py](experiment.py) uses actual shared BPD \`TiedComponentLinear\` wrappers with exact 0/1 forward masks and a custom straight-through gradient, comparing 24 new seeds in four source/optimizer arms.
- [test_experiment.py](test_experiment.py) runs source-only assertions and an excluded-seed smoke. It does not sample or fit any registered seed.
- [registered GitHub Actions workflow](../../../../../.github/workflows/bpd-pd02-registered.yml) executes only when \`EXECUTE_LOCK.json\` is first committed. After the source tests pass, that commit fixes its parent SHA so the training source cannot be swapped silently.
- Scientific pass/fail depends on exact binary inference at held-out states. The original BPD-PD-01 failure and idealized-L0 Lean mathematics are preserved independently.

For a source-only engineering check:

\`\`\`bash
python -m pytest -q mechinterp/bpd/theory/pd02/test_experiment.py
\`\`\`

The full registered cohort is launched only by the one-time GitHub Actions lock. The registered script writes per-seed evidence and paired exact McNemar statistics to a JSON artifact. The CI job remains green when the scientific criterion fails, provided all tests and evidence checks pass.

This 2D model is a controlled factor-identifiability experiment, with no inference of neural TRM mechanism recovery.
