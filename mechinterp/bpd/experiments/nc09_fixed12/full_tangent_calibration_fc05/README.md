# NC09-FC05: full-tangent sham calibration

FC05 measures source feasibility and sham variation across 59 new independent trained model roots 16000–16058. Its [protocol](PROTOCOL.md) was committed first at `8eb34f3533591513b46f14b902721d7f78bc0d43`. Engineering smoke uses 15999. Scientific confirmation remains sealed.

The sampler preserves the executed FC04 choice sequence, including its 1e-12 descent guard and early exit after failed line search. The new trace records current coordinates, singular values and every attempted scale. Source NPZ files support an independent NumPy recomputation of W/m/v and projector geometry. Task NPZ files retain per-case logits and labels for independent NLL checks. Failed roots stay in the registered 59-root denominator.

The pre-run [historical-target EvalMath review](prereview/FC04_HISTORICAL_TARGET_PREFLIGHT.json) reports REDESIGN_OR_PILOT_ONLY. Original FC04 development specificity was 0.009626 nats against a 0.020 scientific minimum. Even the historical 0.027 target requires 36 independent roots under the normal planning calculation before a calibrated sham floor is supplied. The separate [observed-target review](prereview/FC04_DEVELOPMENT_PREFLIGHT.json) is preserved.

The aggregate keeps the exact 59-root rootwise tolerance bound distinct from FC02's nested-bootstrap planning distribution for a 32-root mean. Marginal 95% bootstrap intervals describe development outcomes. The latter bootstrap leaves confirmation coverage and exchangeability between DP12 and constructed shams unresolved.

## Execution order

1. Freeze source manifest and verify manufactured-input tests.
2. Commit the full-smoke lock and execute 15999 once; compare chosen projector bytes with the frozen FC04 sampler on identical source tensors.
3. After full-smoke admission and independent audits, commit the one-time 59-root execution lock.
4. Audit all original receipts and report the frozen calibration verdict with exact EvalMath output.

The workflows restrict execution to this research branch and first run attempt. They archive all first-run outcomes. Confirmation generation is outside this module's allowed root guard.
