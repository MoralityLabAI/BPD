# FUSE-14: Whole-block recurrent cast compactification

Status: CPU synthetic probe completed; full native integration, CUDA, checkpoint transfer and router evaluation pending. Parent FUSE-13: 7a0872632e345501bce81733db30800b2d7079ea. Production defaults unchanged.

## Scope and invariant

The pinned TRMBlock has bias-free qkv_proj and gate_up_proj linears. In the FUSE-13 synthetic BPD trainer these projections are frozen; the wrapped o_proj/down_proj factor adjoints already use the residual pool. For a frozen FP32 weight W, BF16 autocast and fixed device/stream, each use needs the same cast C(W). Input adjoints still execute individually. A shared detached BF16 primal may replace repeated copies of this frozen weight, provided kernel, input cast and layout remain identical. Trainable weights require a separate cast-node/adjoint investigation and are excluded. A mathematical equality is not sufficient evidence for finite-precision bitwise admission.

For R backward-retaining rollouts, T visits and physical projection sites S, saved cast bytes in the idealized native case are R*T*sum(2*numel(W_s)); shared case sum(2*numel(W_s)). Measure actual storage dedup and lifetimes; this is not a total VRAM estimate. For two layers, qkv dimensions 3w*w and gate_up 2i*w with i rounded per native TRM, the shared payload is 4*(3w*w+2i*w) bytes. The other activations, gradients, optimizer buffers, residuals and allocator overhead still consume memory.

## Campaign gates

- A: audit actual saved-tensor aliases, cast counts, dtype, stream, version, stride, shapes and graph lifetimes by site/rollout. Run standalone CPU probe. CUDA tiny probe only after acquiring a valid lease.
- B: isolate qkv only, then gate_up only, then both. Compare native FUSE-13 G, native autocast with cache enabled/disabled, explicit per-call casts, and shared frozen casts. Include clean no-grad, mask-only discarded graphs, stochastic and final PGD rollouts, calibration, full backward and optimizer.
- C: integrate a bounded, explicit step-local scope into the experimental trainer with opt-in disabled by default. Reject trainable, biased, subclassed, parametrized, noncontiguous, mutated or replaced weights, incompatible precision and stream changes. Do not promise safety for .data writes, external concurrent use, compiler capture, offload, FSDP, checkpoint recomputation or unsupported layouts. Fail closed; rerun native only from an untouched state after failure.
- D: frozen matrix widths 128/512/1024/2048 and visits 2/6/12/24, multiple fixed roots, clean_window/closed_loop, calibration off/1/50, binding/full residual budget. Exact finite bytes for forward, losses, input/factor/CI gradients, masks, calibration buffers, parameter and optimizer state over multiple steps. Report overflow masks separately; FP16 remains HOLD. Use uninstrumented timing and independent diagnostic runs.
- E: evaluate checkpointing separately; scopes for recompute must be explicit and parity-tested. No reuse across model mutation or optimizer boundaries.
- F: repeat the five blocked BF16 FUSE-13 frontier cells: (2048,48), (4096,24), (4096,48), (6144,24), (8192,12). Add (8192,24) only after preflight supports it. Refit conservative preflight with measured allocator and activation high-water marks; do not subtract modeled cast bytes from an unrelated peak and claim a measured peak. Use VRAMorama strict FIFO, 12 GiB lease, 2048 MiB free headroom, power guard, exit 75 pause; no forced spill or preemption. Report GPU alloc/reserved, NVML, host RSS, p50/p95 timings, tokens/s and energy. Native arm blocked is a censored comparison, not a speedup.
- G: transfer to an actual local pretrained TRM or DataGoblin checkpoint with verified hash; no automatic download. If unavailable, record blocked_no_local_checkpoint. Pretraining with trainable projections is a separate research target.
- H: export measured per-site rows with checkpoint family, shape, precision, depth, hardware, policy, numerical gate, latency, storage, peak and blocked reason. Keep DataGoblin correctness_path=False. Physical prior is cost ranking only: FUSE-13 precision holdout unsafe admission was 0.25; hard evidence gates must precede ranking. Train a residual predictor only if it beats the same safe physical baseline on locked width/depth/precision groups; report tail regret and false rejections.

## Stop rules

Stop on any finite byte mismatch, unsafe admission, mutation leak, unexplained cast alias, spill, or lease/power violation. FP32 and TF32 remain incumbent unchanged. BF16 experimental only until full parity. No change to PR 187 or existing trainer defaults. CPU synthetic parity is evidence only for the probe, not the native CUDA trainer.

## Initial probe

Run from repo root:

    python -m mechinterp.bpd.experiments.fused_residual.fuse14_projection_probe --device cpu

A CUDA run of the probe must be supervised under a valid lease with independent guards; the standalone probe itself does not acquire a lease or check NVML. Its Block is a proxy (no native attention/rope or BPD factor wrappers). Its shape-based saved storage counter is specific to the fixture, not general alias attribution. The monkeypatch scope is an experimental harness, not production code. It relies on a caller-enforced immutability and no concurrent execution contract. One optimization step is tested; full multistep and independent trainer tests are required.

CPU execution used torch 2.10.0+cpu, widths 16/32/64, visits 2/6/12, roots 14001/14002, two retaining rollouts, AdamW. See receipt_fuse14_cpu_probe_summary.json. CUDA not available in the execution environment.

