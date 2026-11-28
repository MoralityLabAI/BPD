# INK-BPD-01: Inkling cross-loop routing design

Read [research transfer and experiment protocol](INKLING_BPD_IMPLICATIONS_20261121.md). This is a **source-only** deterministic math/check module for BPD's proposed routing/content distinction. No model checkpoints, GPU jobs or statistical claims are run by its tests.

```sh
python -m unittest discover -s mechinterp/bpd/experiments/inkling_routing -p 'test_*.py' -v
```

The audit contains two-block routing gap/coordinate contributions, signed projection shares, norm-and-L2-matched orthogonal shams, and three-term route/content/interactivity decomposition. It does not hook real Inkling modules, implement the Goodfire full attention-residual gates, or read full model trajectories. A future adapter will need exact block output and gate semantics.

## Native PyTorch engineering pilot

INK-BPD-02 runs a small learned, shared-weight recurrent router with a **real writer-tensor hook** and native full continuation. [The frozen development protocol](INK_BPD02_NATIVE_PROTOCOL.md) specifies source-only coordinate selection, an orthogonal equal-dose disjoint sham and 12 synthetic task roots. [The native runner](run_native_dev.py) records all assigned cases, joint sham admission, final task outcomes, selector-only/content-only interventions and the checkpoint tensor digest. It is separate from pretrained Inkling evidence and leaves previous cohorts intact.

## Executed INK-BPD-02/03 native development results

- [Matched physical writer-patch results](RESULTS_INKBPD02_DEV_20261121.md): 384/384 joint controls admitted; selected and sham both 384/384 final correct; original CPU model SHA and artifact hashes.
- [Direct gate-inversion results](RESULTS_INKBPD03_GATE_DEV_20261121.md): on 384 new development inputs, the single-pass control creates 92 wrong intermediate answers that are recovered by the final recurrent pass; all-pass inversion leaves 197 final errors. Source-trained checkpoint and full CI-artifact provenance are recorded.

Both tests are backed by actual PyTorch native forward calls; they retain distinct physical-dose and direct-gate treatment definitions. Neither is a pretrained Inkling experiment or confirmation evidence.

## INK-BPD-04: Goodfire reader-tangent geometry

[The INK-BPD-04 mathematical protocol](INK_BPD04_TANGENT_PROTOCOL.md) and [reader-tangent instrument](reader_tangent.py) distinguish raw writer-reader cosine/coordinate score attribution from the physical pre-RMSNorm Jacobian. A writer parallel to the raw block state can align strongly with the reader while producing zero first-order routing movement. Pure-Python finite-difference and radial-null tests run in source CI; a native PyTorch parity test compares the exact recomputed reader gap with a true synthetic writer patch at pass index 1. These are source-only and synthetic interface tests, with no new task roots or pretrained Inkling outcomes.

## INK-BPD-05: local pretrained Inkling 2 Mini read-edge experiment

[The INK-BPD-05 registered engineering protocol](INK_BPD05_PRETRAINED_PROTOCOL.md) ties an actual Hugging Face `inkling2_loop` input hook to the exact upstream November 21 model source. The [hook](pretrained_message_hook.py) edits the *previous pass's block-3 representation* before its twelve next-loop message readers, at a physically matched 8% raw-state L2/RMS-preserving dose. The [offline runner](run_pretrained_dev.py) validates local checkpoint hashes, full-head zero-hook parity, all four Q/K/V/MoE readouts, complete matched sham admission and bounded development-only task outcomes. It requires an already provisioned local full pretrained checkpoint; CI [tests](test_pretrained_message_hook.py) the reader interface against a lightweight synthetic module and never downloads weights. Eight fixed shams and the externally selected coordinate 2027 remain explicitly developmental until an independent EvalMath sham calibration and a fresh sealed confirmation cohort are registered.
