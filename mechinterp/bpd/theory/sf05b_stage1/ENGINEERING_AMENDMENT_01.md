# SF05B Stage 1 amendment 01: policy clipping autograd repair

**November 18, 2026.** Original Stage 1 source-lock Git blob: `1f85f5006ce32d3eba18158c3d03dcaf1809f526`.

The first hosted run [#44143648755](https://github.com/MoralityLabAI/BPD/actions/runs/44143648755) passed its source hash gate and eight of nine engineering unit tests. The ninth test failed on the *excluded-root train-family-only policy smoke* with a PyTorch autograd version error. The in-place `corr *= scale` assignment changed a tensor needed by `corr.norm()`'s backward graph.

The source fix is exactly `corr = corr * scale`. The original policy Git blob `6d0099b27224e11f9dd01c2b9a1940d676a0e1ee` is superseded for subsequent Stage 1 execution by `d02f1bc491188f6e54ba4282708bd9adcbbc59ec`. This change leaves the network architecture, initialization, training schedule, data splits, policy inference whitelist, correction budget, evaluation controls, and all decision thresholds untouched.

**Timing:** the failed run stopped in engineering unit tests. No seven-pass native parity, excluded-root OOD training, fresh calibration, or heldout model scoring was performed by that run. The original failure and the modified `LOCK_STAGE1.json` remain inspectable. If another engineering correction becomes necessary, record it separately and preserve the prior attempt; any post-data fix must be marked posthoc.

No scientific result or policy generalization claim is established by this repair.
