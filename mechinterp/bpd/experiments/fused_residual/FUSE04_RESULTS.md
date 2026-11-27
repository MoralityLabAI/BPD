# BPD-FUSE-04 | Cache the original residual reconstruction
2026-11-27. **HOLD training and production promotion.** The reference-arithmetic cache gives bitwise forward parity on all tested stress cases. Sharing its differentiable reconstruction changes factor-gradient accumulation enough to fail 231 of 1152 stress fixtures. Evaluation-only use remains an experimental candidate; no consistent forward speed improvement established.

## Provenance and reproducibility
Prospective protocol 2acb6c5a3026db9f3da6e8f644beea60b2187eb0. Tested cache commit 241894ad4b1a6b1b3f5d8bd450fc65bd11a89505, blob 8f4f113f161ff4e12d19a701418bb0d63fc63350. Validator commit 7963127f18974ed6d04f8ccb968819111018abd1, blob b287e2a18745887f01036cd16999772c777513fc. Both blob hashes agree with the locally executed files. Aggregate receipt commit 9e4f0077107847d6cc5a156fce2510e22f5de85a; raw local receipt SHA256 ce61bbc398cee92d76e878ca132531c1fde4b29757535db435231edef294e9a7. Raw receipt and tested source/extracts delivered in reproduction archive. PyTorch 2.10.0+cpu, Linux x86_64, one thread, highest float32 matmul precision. DNS blocked direct repository retrieval in container. Used exact extracted native baseline (ab93e4da986beef87897f8d1244c7fe7564b5bde), matching recorded FNV-1a fingerprints of constructor/calibration, forward, config, TRM and builder text. This is a deviation from executing the full current FUSE02 module at the protocol's head: its disabled-fusion component arithmetic is the original baseline, but full-package integration and current-branch imports remain untested. Target intervention method omitted from local extract and not exercised. No production wrapper changes in FUSE04. Full native invocation from repository root:
    python -m mechinterp.bpd.experiments.fused_residual.validate_fuse04 --output receipt.json
Excerpt reproduction:
    python validate_fuse04.py --excerpt-dir . --output receipt.json
No pretrained weights downloaded.

## Mechanism and numerical limitation
Within one fixed-weight forward rollout, the residual D=W-(VU)^T is constant. Store its original floating-point tensor once per wrapper and use the unchanged native linear and masking operations for each pass. For T invocations this removes (T-1)D_in*C*D_out reconstruction MACs. Evaluation uses no_grad. Training shares the same autograd node, preserving the exact-real derivative but reassociating factor-gradient accumulation. With per-invocation residual cotangent G_t, the reference accumulates contributions -(G_t^T) U^T and -V^T (G_t^T) separately. Shared reconstruction first aggregates G_t, then multiplies by the factors. Distributivity holds over reals; the two floating-point accumulation orders can differ, particularly when the direct component gradient nearly cancels the residual gradient. This explains why exact forward equality cannot establish backward parity.

Experimental instance-scoped context temporarily replaces weight_delta, restores it on exit, refuses existing fusion flag, nested scopes and duplicate wrappers. CPU float32/highest precision only, no autocast or inference mode, mismatched grad mode bypasses. Eval tracks identity/version/shape/dtype/device/requires_grad of W,V,U; detected change invalidates. Train raises on detected mutation after a cache read; an early residual-backward hook marks scope consumed and prevents subsequent access. Caller must terminate the scope on mutation error, use one sequential fixed-weight non-checkpointed forward graph, backward after exiting, and never mutate returned residual tensors, use unsafe .data writes, reuse graphs, use concurrent/reentrant execution or rely on this diagnostic to detect all autograd boundaries. In particular the implementation does not track mutation of the returned cached residual itself; it is a controlled native-call experiment, not a general public caching API. Numeric residual payload cap defaults 16 MiB; no cap on allocator/autograd saved tensors or total RSS. No production defaults changed.

## Fresh stress grid
8 roots 3600001..3600008, 6 factor scales, 3 target classes (ordinary, exact reconstructed weight, near reconstructed weight), 4 mask kinds, zero vs calibrated pass mean/sphere, three calls sharing factors, independent inputs and masks. All 1152 cases finite. Compare no-grad reference/cache and single shared-gradient reference/cache with common cotangents. Tolerance abs 5e-5 rel 5e-4; forward bitwise equality required.

| Scale | Cases | Eval forward failures | Train forward failures | Train gradient failures |
|---:|---:|---:|---:|---:|
|.1|192|0|0|0|
|1|192|0|0|0|
|3|192|0|0|0|
|10|192|0|0|24|
|100|192|0|0|85|
|10000|192|0|0|122|
|Total|1152|0|0|231|

6912 forward tensor comparisons, 5184 gradient tensor comparisons; V gradient fails169, U164, x/m/d zero. Maximum absolute differences V and U .1875, x/m/d and output zero. First failure seed3600001, scale10, ordinary target, all-one component mask, uncalibrated, max gradient error .00011444091796875. 921/1152 cases pass all gradient tolerances; no safe scale threshold inferred from this stress-enriched finite grid. Cache removes 4608 reconstructions over the combined eval/train cases (2304 per mode), all expected hit/miss counts verified. Do not compare failure counts directly against FUSE03 since the three-pass fixtures and seeds differ.

All19 lifecycle assertions pass: eval hit identity, version and identity invalidation, scope restoration, no-grad bypass and subsequent grad reuse, backward after exit, early-backward detection, train mutation and replacement errors, nesting rejection, exception restoration, aggregate cap, autocast/inference/precision bypass, fresh scope after optimizer step, clean target mode without residual, eval calibration reuse and exact mean buffers.

## Native mini-TRM extracted-source CPU benchmark
Fresh synthetic seeds 3601001+width, B2 seq16 puzzle4 two layers H1/L2 two outer steps, four wrapped o_proj/down_proj sites, C=min(64,width/4), calibrated pass means/sphere, soft masks and residual masks. Single-thread CPU, three warmups/11 alternating timing samples, includes rollout setup and grad clearing but excludes optimizer. No controlled process isolation or repeat-day confidence interval. Each cache rollout: 24 residual requests, 4 reconstructions,20 hits, no bypass. Logical residual reconstruction MACs at256 201326592 ->33554432 (83.33% reduction in this operation only); numeric resident peak128KiB/512KiB/2MiB respectively. Graph and other memory excluded.

| Width | Ref fwd ms | Cache fwd ms | Ratio | Ref fwd+bwd ms | Cache fwd+bwd ms | Ratio |
|---:|---:|---:|---:|---:|---:|---:|
|64|13.2524|17.3019|.766|42.5615|40.0296|1.063|
|128|27.7740|24.7137|1.124|51.0756|45.4110|1.125|
|256|33.9944|34.5929|.983|90.5752|68.8465|1.316|

624 recurrent tensor/loss/gradient/parameter comparisons across initial and paired training rollouts within tolerance, all54 greedy readout comparisons equal. 24 paired AdamW steps (8 per width, lr1e-4, same synthetic squared-logit loss), max state drift4.7684e-6, factor-gradient drift8.0466e-7, parameter drift8.0466e-6. Initial and first pre-update forward states bitwise equal; after parameter drift this need not hold. Forward timing gains are inconsistent, while training timing improves in this run but cannot justify promotion given stress failures. No GPU, mixed precision, pretrained accuracy, actual BPD trainer loss/throughput, total memory or LLM tokens/s measured.

## FUSE-05 candidate
Keep the cached primal for eval. For training, test a custom per-invocation autograd node that returns the same cached residual value but computes each invocation's residual adjoint separately, matching the native transpose/matmul operation order and examining leaf accumulation order. This can retain forward reconstruction savings while sacrificing the shared-backward GEMM savings. A custom VJP alone does not establish bitwise gradient equality; compare against native independent reconstructions on new severe cancellation roots, mask gradients, exact-zero/full-mask controls and optimizer trajectories. Include backward recomputation as a correctness-first control. Measure forward plus backward and memory, and maintain original path for unsupported cases. Separate eval calibration cache from training path. No promotion until the numerical failures are addressed and full native integration is tested.
