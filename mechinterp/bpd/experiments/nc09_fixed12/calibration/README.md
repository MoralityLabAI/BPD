# NC09-FC01: 59 independent model-root sham calibration

**Prospective 2026-11-18 registration.** The immutable [PROTOCOL.md](PROTOCOL.md) was committed before any calibration models were trained.

The original [NC09 Fixed12](../RESULTS_20261118.md) 16-root development was positive on all registered physical causal rank/dose comparisons. Its eight local matched shams per root are nested observations; they provide an estimate of each root's local physical control variation. FC01 supplies the missing **59 genuinely new independent trained-model roots** needed for a distribution-free **95th percentile at 95% confidence** upper bound.

Fresh calibration roots **12000–12058**, excluded-only engineering root **11999**, and the unchanged frozen 20D four-pass DenseLooper training/evaluation conditions apply. Each source root receives the same 12-slot passwise DP source projector, equal 12-slot flat3, source-RMS-matched flat3, 16-slot flat4, reversed-rank descriptive control and eight same-tuple dose-matched source shams from 4096 fixed-seed proposals.

For root i define T_i = max_j |NLL(sham_ij) - median_k NLL(sham_ik)|. The **sample maximum across 59 independent roots** is an upper nonparametric tolerance statistic for the 95th population percentile with 95.15% coverage confidence (under independent exchangeable root sampling). It describes the *rootwise matched-sham null*, rather than an exact significance threshold for a 32-root mean effect.

The [unmodified pinned EvalMath mechinterp preflight](evalmath_preflight.py) comes from [EvalMath commit 6d9c28d](https://github.com/MoralityLabAI/EvalMath/blob/6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a/skills/evalmath-mechinterp-power/scripts/preflight.py), exact upstream blob SHA 0ff062f76f5534d9f18425f986ae3e750397f50c. Its root-level pilot uses the genuinely fresh 59-root flat3-minus-DP12 effect and all eight centered same-root sham NLLs, with minimum meaningful +0.008 nats, nominal target +0.027, resolution +0.001, planned comparison count 3, alpha .05 and planned future 32 roots.

The first-run calibration must archive **all 59 raw model-root JSONs, the exact EvalMath input and output, source SHA and unmodified decision**, including REDESIGN_OR_PILOT_ONLY. Even READY_FOR_CONFIRMATION requires a separate registered 32-root confirmation protocol. This FC01 workflow never generates, trains or scores a confirmation root.

[Original source-only runner](experiment.py) · [59-root aggregation and EvalMath hold](aggregate.py) · [smoke root 11999](test_experiment.py) · [synthetic decision tests](test_aggregate.py) · [95/95 arithmetic certificate](CalibrationCoverage.lean).


## Original 59-root result: calibration hold

The [first source-locked full calibration #44276266076](https://github.com/MoralityLabAI/BPD/actions/runs/44276266076) completed all 59 independently trained 20D four-pass recurrent roots, each with eight fully admitted local same-tuple physical shams. Engineering and original evidence CI passed.

The **frozen calibration verdict is CALIBRATION_REDESIGN_OR_PILOT_ONLY**. The exact pinned EvalMath script returned **REDESIGN_OR_PILOT_ONLY** because the observed causal mean and the +.027 planned target both fell below the 95/95 rootwise null-variation floor. The frozen 32-root confirmation remains sealed.

| Independent 59-root calibration quantity, NLL nats | Value |
| --- | ---: |
| Mean DP12 benefit over equal 12-slot flat3 | **+0.032142** |
| Mean DP12 benefit over source-RMS-dose-matched 12-slot flat3 | **+0.031963** |
| Mean DP12 matched-sham advantage | +0.062433 |
| Mean DP12 absolute intact-model task cost | +0.014535 |
| Mean DP12 extra cost versus stronger flat4 | +0.002349 |
| **95/95 independent root-level sham variation tolerance** | **0.294179** |
| Approximate resolvable effect for 32 roots under pinned EvalMath | **0.313917** |
| Pinned EvalMath scientific-planning decision | **REDESIGN_OR_PILOT_ONLY** |

The original 59-root order-statistic is the maximum root-level absolute deviation of any of eight shams from its root's sham median, with nonparametric 95th-percentile coverage confidence 95.15%. Its highest-root instance was root 12047. This unusually conservative rootwise null statistic is fundamentally different from a mean-over-32-roots effect. Consequently the calibration hold is a statement about **the frozen preflight design's resolution**, while the observed positive mean DP12 benefit remains development evidence rather than confirmation.

[Full first-run report](RESULTS_20261118.md), [original 59-root CSV](PER_ROOT_20261118.csv), [all original root SHA256 hashes](ORIGINAL_59_ROOT_SHA256.csv), [post-result independent EvalMath/ZIP audit](test_published_results.py) and [complete immutable 62-file original artifact #14415331598](https://github.com/MoralityLabAI/BPD/actions/runs/44276266076/artifacts/14415331598) preserve the result. The original ZIP SHA256 is 2c70ad34874ed5761199ab8f9795741377d7898f2dae4f5e960d1997f99926e6.

**0 scientific-confirmation roots were generated or scored.** A future alternative must be independently preregistered with new model roots, a null matched to its actual aggregate estimand, and a fresh EvalMath resolution audit.
