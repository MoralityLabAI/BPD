# NC09-FC05: full-tangent sham calibration

**Original result: FULL_TANGENT_CALIBRATION_PILOT_ONLY.** The [first 59-root run](https://github.com/MoralityLabAI/BPD/actions/runs/44319021117) admitted all 59 models and 472 matched shams. Mean sham-minus-DP12 task NLL was 0.008083953 nats, with marginal root-bootstrap 95% interval [0.005644242, 0.010486967], below the registered 0.020 mean requirement. The independent rootwise 95/95 sham floor was 0.058183573 nats. Strict EvalMath returned **REDESIGN_OR_PILOT_ONLY**. Scientific confirmation remains sealed, with zero confirmation models evaluated.

Read the [complete result and audit report](RESULTS_20261118.md), [original per-root table](results/FC05_PER_ROOT.csv) and [evidence receipt](results/PUBLISHED_EVIDENCE.json). The first excluded smoke retains its failed workflow conclusion. A [separate source replay](results/SMOKE_REPLAY_AUDIT.json) reproduced its original projectors and parameter states before the calibration release, using the saved evidence.

FC05 measures source feasibility and sham variation across 59 new independent trained model roots 16000–16058. Its [protocol](PROTOCOL.md) was committed first at `8eb34f3533591513b46f14b902721d7f78bc0d43`. Engineering smoke uses 15999. Scientific confirmation remains sealed.

The sampler preserves the executed FC04 choice sequence, including its 1e-12 descent guard and early exit after failed line search. The new trace records current coordinates, singular values and every attempted scale. Source NPZ files support an independent NumPy recomputation of W/m/v and projector geometry. Task NPZ files retain per-case logits and labels for independent NLL checks. Failed roots stay in the registered 59-root denominator.

The pre-run [historical-target EvalMath review](prereview/FC04_HISTORICAL_TARGET_PREFLIGHT.json) reports REDESIGN_OR_PILOT_ONLY. Original FC04 development specificity was 0.009626 nats against a 0.020 scientific minimum. Even the historical 0.027 target requires 36 independent roots under the normal planning calculation before a calibrated sham floor is supplied. The separate [observed-target review](prereview/FC04_DEVELOPMENT_PREFLIGHT.json) is preserved.

The aggregate keeps the exact 59-root rootwise tolerance bound distinct from FC02's nested-bootstrap planning distribution for a 32-root mean. Marginal 95% bootstrap intervals describe development outcomes. The latter bootstrap leaves confirmation coverage and exchangeability between DP12 and constructed shams unresolved.

## Completed execution

The 55-file source manifest froze the implementation after the disclosed smoke replay repair. All 22 manufactured-input, source-corruption and execution-guard tests passed. The [first smoke](https://github.com/MoralityLabAI/BPD/actions/runs/44317319540) trained root 15999 once. Its [hosted replay audit](https://github.com/MoralityLabAI/BPD/actions/runs/44318700523) passed with zero additional training, after which the execution lock released roots 16000–16058 once.

Post-run audits checked all 177 original root/source/task hashes, independently recomputed source constraints and final NLLs, and replayed all 2,200 saved solver states with all 1,728 Jacobians. The independent statistical review reproduced the registered effect intervals, all 256 outer null estimates and the strict EvalMath decision. The original 59-root denominator and every failed search trial remain in the evidence.

The workflows restrict execution to this research branch and first run attempt. They archive all first-run outcomes. Confirmation generation is outside this module's allowed root guard.
