# FUSE-13: Precision-aware retained residuals

Status: prospective, diagnostic gym only. Base: f0d042b25306e4efcbb1d35b266d2ab85fb7bb49 (FUSE-12 PR #186). Production flags remain default-off. Do not alter FUSE-12 evidence or silently relax tolerances.

## 0. Reconcile source evidence

Run python -m mechinterp.bpd.experiments.fused_residual.fuse13_precision_gym --output runs/fuse13/audit.json.
Expect 50 distinct records, 46 benchmark successes, four recorded native autocast errors. BF16 width-512 trainer failure counts C/F/G = 50/52/52, total 154; FP16 = 42/42/42, total 126. The stress counter nonfinite_parity_tensors counts matched nonfinite outcomes, not finite successes; FP16 has 384 such outcomes. Inspect signs of infinity, NaN masks and finite entries separately. Correct the generated report in a separate evidence-derived patch. Freeze the audit before evaluating candidates.

## A. Native reference contract

Reproduce observe_clean_pass pass_mean error in the native trainer; distinguish cast-before-reduction vs cast-after-reduction and FP32 EMA. Proposed minimal reference repair casts the *completed* statistics to the FP32 buffer dtype before lerp_, preserving native autocast reduction. Verify initial copy and subsequent EMA, capacity expansion, calibration cadence 1/50, clean-window and closed-loop, repeated pass IDs and FP32/TF32 unchanged. Record explicit dtype traces. Freeze repaired native Arm A before testing candidate.

## B/C. Shared compute-dtype storage and independent adjoints

The included fuse13_precision_gym.py implements a standalone microprobe: native repeated reconstruction versus shared detached cast residual with independent per-use autograd nodes, under both per-use and shared factor-cast policies. This is a falsification probe, not a production implementation. It compares outputs, input gradients, factor gradients, finite masks and byte equality. It counts unique square storage retained by autograd and measures CUDA peak allocated bytes. It does not establish full trainer correctness or total memory complexity.

CPU smoke (no GPU required):
    python -m mechinterp.bpd.experiments.fused_residual.fuse13_precision_gym --plan
    python -m mechinterp.bpd.experiments.fused_residual.fuse13_precision_gym --probe --device cpu --mode bf16 --width 64 --passes 6 --output runs/fuse13/cpu_bf16.json

GPU microprobe only through VRAMorama:
    vramorama run --vram 12G --headroom 2048 --strict-fifo --owner bpd-fuse13-probe --guard-power 60 --pause-exit 75 --restart 6 -- python -m mechinterp.bpd.experiments.fused_residual.fuse13_precision_gym --probe --device cuda --allow-cuda --mode bf16 --width 128 --passes 12 --output runs/fuse13/cuda_bf16.json

Run both cast policies. Inspect native autocast cast-node sharing, accumulation order, gradient strides, dtype transitions, and the residual subtraction/cast boundary. Try explicit low-dtype operands with autocast disabled within custom backward; compare against native graph replay. Preserve independent first-order adjoints; aggregation prior to rounding is a different numerical algorithm and requires separate evidence. Cache identity must include site, parameter and residual version, device, compute dtype, autocast state, TF32 flags, stream, shape and stride. Reject unsupported higher-order gradients, checkpoint reentry, cross-stream calls, mutation, consumed graph, or a byte cap violation. Account for full payload bytes at actual element size and all cast representations. Never admit a nonfinite match as a finite correctness success.

Frozen microprobe roots 4013001..4013004, widths 64/128/256, T=1/6/12/24, scale 1/10/100/10000, ordinary/exact/near targets. Expand the small probe to the existing native trainer stress suite (96 cases / 1536 tensors per mode), preserve original roots as regression and use new roots as holdout. ATOL=5e-5, RTOL=5e-4. Require finite bitwise parity for admission; report tolerance-only separately. Overflow stress: identical NaN masks and signed infinity masks, matching finite values and skip/update behavior under dynamic loss scaling, never count overflow as a pass for finite parity. If native FP16 itself overflows, classify separately; keep mode on hold until safe training regime and overflow contract validated.

## D. Frontier / checkpoint transfer

Re-run mandatory CPU regression, 30 lifecycle checks, zero allocator leak, stream ownership, FP32/TF32 stress, full trainer parity with all FUSE-12 state categories. Matched A/G and ablations on widths 128/512/1024/2048, T=6/12/24, calibration off/1/50, both budget regimes. Add frontier parity w4096/T24; candidate-only exploratory frontiers w2048/T48, w4096/T48, w6144/T24, w8192/T12. Preflight each arm separately; never allocate a predicted-over-budget native reference under the 12 GiB lease. If a full native frontier cannot fit, record blocked and use independent smaller full parity plus isolated CPU/reference tensor checks; do not label it full parity. Compare a real pretrained checkpoint only if present, record its source and hash. Use synchronized timings, multiple paired roots, randomized arm order, warmup, CIs, allocated/reserved/NVML/shared-memory telemetry and energy with method noted. WDDM spill comparisons are a separate opt-in campaign with explicit authorization. Never force spill.

## E. Admission learning

Keep exact_saved_mac and density as strong baselines. Learn a correction to a physical memory and execution-cost prior, including device/cast dtype, live residual bytes, actual lease headroom, optimizer/activation overhead, spill risk, reconstruction count and uncertainty. Split whole width, depth, device and checkpoint families; group arms from the same cell together, avoid target leakage. Score oracle step savings, worst-case regret, unsafe admission, classification calibration and false rejection. For unsafe or unvalidated configurations, abstain/fallback. Exact rules own correctness; learned policies remain experimental.

## Execution and receipts

Use strict FIFO VRAMorama 12 GiB lease, 2048 MiB headroom, >=60 W guard, exit 75 for pause, resume only missing cells. Do not kill other leases or disable watchdogs. CPU work can proceed before GPU admission. Run one cell at a time, preserve raw outputs and explicit error/blocked/unsupported status. Record commit, environment, roots, precision/flags, arm, baseline, shape, budget, timing, storage and status. No silent retries or replacement of failures. Gate A before B/C integration; gate B/C before D; E may train from frozen evidence in parallel. Report all denominators and exclusions. No new API or cloud spend. Production defaults stay unchanged.

Deliver: native fix patch, precision candidate behind experimental flag, audit and microprobe receipts, trainer stress and frontier receipts, checkpoint transfer report or blocker, DataGoblin dataset/predictor report, FUSE13_RESULTS.md generated from actual receipts, draft stacked PR with CPU CI. The present gym supplies diagnostic code and plan; native repair, integration, frontier execution and learned model remain work packages.
