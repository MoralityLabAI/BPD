# DT02-A: CPU training on hosted CI

[Registered protocol](PROTOCOL_CI.md) · [Stage 0 gym](README.md) · [BPD-DT02 issue](https://github.com/MoralityLabAI/BPD/issues/98)

This Stage A job trains a **small but actual native Attention-TRM physical recurrence** on three separately graded tasks: 4×4 Sudoku, synthetic ARC-style 4×4 transformations with disjoint program families, and a storyworld expert action policy with independently hosted hidden terminal endings. Its input and output channels use the real TRM token embedding, tied attention and MLP layers, and digit LM head. Supervised depths 2/4/6, unseen 8/12.

A readout-preserving donor-state steering diagnostic uses the whole eleven-token LM head over all sixteen Sudoku board positions, and compares full protected-dark state edits against an attention weight-SVD rank-eight physical subspace at depth 4, 8 and 12. It does not claim trained BPD decomposition or full signed-Jacobian control.

On a CPU machine with PyTorch installed:

\`\`\`bash
python -m unittest mechinterp.bpd.experiments.depth_steering_gym.test_ci_pilot -v
python -m mechinterp.bpd.experiments.depth_steering_gym.ci_pilot \
  --seeds 901 902 --updates 64 --samples 64 --out /tmp/bpd_dt02a.json
\`\`\`

Only previously excluded engineering roots 901/902 are admitted. Gold labels are derived from exact host task generators; synthetic ARC-style transformations are explicitly **not official ARC-1**. No pretrained 7M Sudoku checkpoint, 59/32 science roots or sealed ARG-03-N cases are loaded.
