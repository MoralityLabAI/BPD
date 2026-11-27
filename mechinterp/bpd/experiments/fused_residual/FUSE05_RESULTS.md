# BPD-FUSE-05 | Cached primal with independent first-order adjoints
2026-11-27. Disposition: finite CPU stress parity achieved; HOLD production promotion. Prospective protocol 048c6b69815617d32b80ee49f5d95872fb6b1179. Executed adjoint_cache.py blob b44b1702bc235790adf7d20f4abf022da049ef67 (commit 1a3ff01533c08c7d71741cc99716fc8d67543e28) and validate_fuse05.py blob 7164e54aa21b57125e9b4bdf53437ca2d0a55fed (commit 8eecc40eba1651bd8662e69ce1262235948f1484) read back with exact local Git blob hashes. The first attempted compressed validator transfer failed integrity checking before any file creation; direct source transfer verified. FUSE04 dependencies unchanged. Four raw receipts and exact source extracts supplied in BPD-FUSE-05-reproduction.zip; repository summary efe8f33701325b4c48eb5b2d66a15d94741631f3 records raw SHA256 digests. Archive SHA256 aee2c91b2c484906c7f4855f825aeefbf48fed21eddd449bb6794a0139b71dd8.

## Mathematical mechanism
Native residual D = W-(VU)^T, with fixed frozen W and tied trainable V,U. FUSE04 differentiable cache shares one reconstruction autograd node; the loss cotangents G_t aggregate before multiplying by U^T and V^T. Over reals this equals separate per-invocation gradients. Floating point accumulation and cancellation caused FUSE04 and the new shared control to fail.

ResidualVJP caches D in no_grad at first request, returns the cached value through a fresh custom autograd Function per invocation, saves original V,U tensors, and computes g_V = (-G_t)^T U^T and g_U = V^T(-G_t)^T independently in backward. ProductVJP caches VU instead, exposes an independent matmul custom node per invocation, and performs the original transpose and W subtraction normally. ProductReplay uses the same cached product, then reconstructs a local native matmul graph and asks autograd for its VJP during backward. All preserve original native masking/linear/sphere arithmetic. Native input and direct component/mask gradients remain connected. Independent custom adjoints do not share the residual cotangent summation. Analytic adjoints eliminate backward replay; actual engine scheduling, layouts and numerical equality still require testing. Once-differentiable contract: higher-order derivatives unsupported.

## Scope and lifecycle
Experimental instance context restores original weight_delta methods in finally, rejects nested/duplicate scopes and reassociated fusion; tracks W,V,U identity, version, shape, stride, dtype/device and requires_grad, cached primal version. Reject detectable mutation and early backward consuming a cached node while scope is active. Fixed CPU float32 contiguous factors/weights, highest matmul precision; unsupported dtype/layout/backend, autocast, inference/no_grad in training, and trainable W bypass. Separate no_grad evaluation mode uses frozen FUSE04 cache. Only sequential fixed-weight first-order non-checkpointed forward with backward after context exit. External writes through .data, external mutation of returned values, concurrent/reentrant calls, graph reuse and higher-order derivatives are outside supported contract; hook is diagnostic, not comprehensive lifecycle proof. Numeric primal cap 16 MiB; excludes saved activations, per-call product-arm residuals, metadata and allocator storage. No default-path changes.

## Registered finite stress grid
Fresh 8 seeds 3700001..3700008, six factor scales .1,1,3,10,100,10000, ordinary/exact/near W, four mask kinds, calibrated mean/sphere+residual mask vs zero ablation. Three separate inputs, same V/U and independent cotangents, summed backward. 1152 cases per arm, all finite; abs5e-5/rel5e-4. Exact tensor equality uses torch.equal, not a raw byte comparison (signed zero not distinguished).

| Arm | Forward failures | Gradient fixture failures | Equal gradient tensors | V/U tensor failures |
|---|---:|---:|---:|---:|
| Shared | 0 | 253 | 2880/5184 |188/170|
| ResidualVJP |0|0|5184/5184|0/0|
| ProductVJP |0|0|5184/5184|0/0|
| ProductReplay |0|0|5184/5184|0/0|

Shared gradient failures by scale: 0,0,0,24,90,139; max abs V .21875 and U .15625; all other tensor max errors zero. Independent variants all max differences zero. 13824 training forward tensor checks plus3456 eval checks, all equal. 20736 gradient tensor comparisons across four arms. Each training arm 2304 cache hits,1152 forward reconstructions instead of3456; independent arms3456 custom backward nodes, replay3456 backward matmul reconstructions, analytic variants zero replay. All 64 lifecycle checks (19 frozen shared,15 each independent) passed. Separate scopes after updates, no-grad bypass, layout/precision fallback, early backward, mutation, cap, frozen factor and restoration tested; not a general floating point proof.

## Recurrent CPU test
Verified original FUSE04 native extracts (constructor/calibration, forward, config, TRM, builder fingerprints checked by load_native). No full-package import integration or target intervention execution. Synthetic mini-TRM widths64/128/256, new seeds3701001+width, B2 seq16 puzzle4, two layers H1/L2 two outer steps, four o/down wrapped sites, C=min(64,width/4), calibrated sphere, identical random soft component/residual masks. Fresh clone per arm, fixed squared-logit objective and 8 paired AdamW steps per width per arm. All2496 recurrent tensor comparisons within tolerance, 216 greedy comparisons equal; all1872 independent-arm recurrent tensors exactly equal including updated parameters; shared56/624 exactly equal. Total96 paired AdamW steps. Each rollout24 requests,4 forward reconstructions,20 hits; independent24 backward nodes, replay24 extra reconstructions. Peak numeric cache payload 128 KiB,512 KiB,2 MiB at respective widths (shape-based). At width256 saved167772160 forward reconstruction MACs vs original; analytic independent VJPs still perform separate backward matmuls. Replay returns the saved reconstruction work in backward.

PyTorch 2.10.0+cpu, Linux x86_64, one thread, highest float32 precision, 3 warmups and11 alternating measured samples per width/arm. Forward-only measurement uses the common FUSE04 eval cache in every arm; variations reflect independent timing sessions, not different forward-only methods. Median forward+backward ms and ratios:
| Arm | 64 ref/cache/ratio | 128 ref/cache/ratio | 256 ref/cache/ratio |
|---|---|---|---|
| Shared |17.763/17.207/1.032|26.883/25.102/1.071|66.585/51.290/1.298|
| ResidualVJP |16.462/18.084/.910|25.384/26.474/.959|66.296/59.199/1.120|
| ProductVJP |16.389/18.760/.874|25.834/27.981/.923|66.268/65.815/1.007|
| ProductReplay |17.331/22.023/.787|28.523/34.005/.839|73.743/77.821/.948|

ResidualVJP is the strongest measured correctness-passing training candidate at width256, 1.12x (10.7% reduction in paired median duration). At smaller widths overhead dominates. Shared speed is disqualified by stress. Replay control validates separate adjoints but is slower. Single run, no timing confidence interval or hardware crossover claim. Timed rollout setup/grad clearing included, optimizer excluded. No full BPD trainer, actual model quality, GPU/mixed precision, total RSS or LLM token throughput result.

## FUSE-06 recommendation
Freeze ResidualVJP as experimental numerical-parity candidate; compare full native imports and real BPDTrainer mask/adversary/optimizer lifecycle before production. Test device and dtype-specific gradient parity against independent reference, with higher-order/activation checkpoint unsupported and explicit fallback. Measure wider layers and longer recurrence with memory high-water, native task loss and intervention ranking controls. Reduce Python scope/lookup overhead only after isolating correctness. A C++/Triton implementation or cached native matmul primal must independently preserve the per-invocation adjoints. Do not promote based on this finite CPU grid alone.

Reproduce from repository root:
    python -m mechinterp.bpd.experiments.fused_residual.validate_fuse05 --arm residual --output receipt.json
This native import route is provided but untested. Tested excerpt invocation in archive:
    python validate_fuse05.py --excerpt-dir . --arm residual --output new_receipt.json
Repeat for shared, product, replay. Raw receipts and manifest in archive.