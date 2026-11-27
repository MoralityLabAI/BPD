# BPD-FUSE-02: native recurrent CPU evaluation

**Disposition: HOLD promotion.** The CPU float32 opt-in implementation is committed (5b9dece736741f98ce8935cb79f9896fd5ef39bd); default remains reference. Ordinary-fixture parity passes, but large-factor cancellation violates numerical equivalence.

## Provenance

Protocol FUSE02_PROTOCOL.md at 61e6c4db7ca3bbacc60d6aeb5d48cb8ec1d713a5. Validator validate_fuse02.py at b31a4320377163b372b012563ce16ada1f671f21; Git blob 50ac78b8ef98c3a2527b244072ac249d00256dd1 (local executable has identical content except a blank line normalized after the run). Full timing, comparison and loss receipt: receipt_fuse02_cpu.json at 8cbb29e0f2a386203b4899abbba23079d2da08eb. Container lacked direct GitHub DNS. Executed exact extracted TiedComponentLinear initialization/calibration/component-forward methods and PureSudokuTRM classes/build_mini_trm, verified against trimmed UTF-8 FNV-1a fingerprints recorded in receipt. Full package import integration, native intervention runtime coverage and hosted CI are not established. Target intervention method text is unchanged from base. No changes to defaults, unrolled wrapper, calibration algorithm or parameter/state-dict keys.

## Results

CPU PyTorch 2.10.0+cpu, one thread. All 2,800 ordinary tensor comparisons passed (256 wrapper outputs, 1,152 wrapper gradients, 768 calibration buffers, 162 recurrent tensors, 27 losses, 27 input gradients, 216 factor gradients and 192 updated parameter tensors). Wrapper grid covers zero/pass mean, sphere, soft/binary/zero/one masks, delta, bias, routing, calibration updates and pass offset 2. All 256 fused wrapper calls used candidate path. Explicit near-threshold and CPU autocast cases fell back. All greedy outputs in the ordinary recurrent comparisons agreed. 

| Width | Reference fwd ms | Fused fwd ms | Speedup | Reference fwd+bwd ms | Fused fwd+bwd ms | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
|64|7.484|8.464|0.884|16.812|16.733|1.005|
|128|11.202|10.840|1.033|26.568|23.595|1.126|
|256|29.386|21.590|1.361|72.384|47.699|1.518|

Mini-TRM: B=2, seq=16, puzzle=4, two layers, H1/L2, two outer steps, four wrapped o_proj/down_proj sites, C=min(64,width/4); calibrated pass mean/sphere and soft component/residual masks. Median 11 alternating timed repetitions after 3 warmups. Timing includes rollout setup and gradient clearing, excludes optimizer. Eight paired AdamW steps per width (lr 1e-4), identical repeated synthetic batch and squared-logit objective; no accuracy claim, no full BPD trainer objective. 888 fused invocations per width without fallback. Max recurrent state/logit difference 4.0531e-6, factor-gradient difference 5.9139e-7, updated-parameter difference 9.7714e-6. CPU results only.

## Failed stress test and next gate

Set W_target to the materialized reconstructed weight and use zero component mask, implicit residual mask one. Baseline residual is exactly zero. Reassociation exposes cancellation between XW^T and (XV)U. Absolute max differences at factor scale 1,100,10000 were 9.5367e-7,0.009765625,80 respectively. Last two fail the predeclared tolerance. All three used fusion; the current RMS proximity guard cannot detect this failure (sphere disabled). This is a counterexample to unconditional floating-point equivalence, not to the real-arithmetic identity. Keep opt-in experimental; do not enable in production or merge as a promoted fast path.

Next: compare a low-overhead scale/cancellation screen against a numerically conservative forward-error enclosure, with an exact fallback to original evaluation. Include extreme magnitude, near-zero residuals, nonfinite inputs, near-epsilon sphere cases, matched backward errors, and guard cost; forward certification alone cannot certify gradients. A heuristic screen must be described as such. GPU/autocast fusion stays unsupported pending separate validation. Full package validator invocation from repository root: python -m mechinterp.bpd.experiments.fused_residual.validate_fuse02. No GPU, pretrained checkpoint accuracy, full trainer throughput, peak RSS or LLM tokens/sec measured.
