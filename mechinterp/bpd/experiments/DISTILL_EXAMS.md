# BPD Distill adapters v0.1

Canonical compiler: [Compactification-Theory Distill v0.1.0](https://github.com/MoralityLabAI/Compactification-Theory/tree/ac6ec9243d16f73117a986307851cd031bca7fa8/Distill), pinned at ac6ec9243d16f73117a986307851cd031bca7fa8. These source-locked sidecars add independent design examinations, without changing gym execution, source or evidence. Use reviewed public development material only.

- BPD-DT01: `mechinterp/bpd/experiments/depth_steering_gym/distill_exam.json`
- SCCG-04: `mechinterp/bpd/experiments/sccg01/distill_support_exam.json`

From the pinned Compactification-Theory checkout, run `python -m Distill compile --adapter /absolute/path/to/adapter.json --out /tmp/exam.json`, then `python -m Distill prompt --packet /tmp/exam.json`. Exact task and intervention verification remains with the original gym. Narrative explanations require separate review. No paid model calls, training export or native checkpoint access.
