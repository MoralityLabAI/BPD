# AMM-01R corrective prospective amendment (2026-11-18)

**Registered before generating the fresh confirmation cohort.**

## Audit finding

The initial source test `test_confirmation_must_have_preflight` called `bundle("confirmation")`, and `bundle` constructed/scored the entire 512-root confirmation cohort *before* raising the expected missing-preflight error. This happened in an engineering test before the final source lock. Although the outcomes were discarded by the exception, the cohort was executed early, breaching the intended freeze order. The published original development/confirmation receipts and results remain intact for transparency but the old `confirmation` arm is reclassified as **engineering-only / unsealed**. Its initially reported prospective inference must not be used as independent confirmation. Keep the older source commit (`2adc180d`) and digest (`5317c7ff...`) as historical evidence.

## Corrective design freeze

All prior generator parameters, observed features, physical dose, outcome, training and development roots, policies, controls, significance family, minimum +0.15 effect, EvalMath preflight, sign tests and 2000-seed root bootstrap are retained.

**Only the confirmation source population changes:** use a newly named phase `confirm_fresh` with 512 independent roots indexed `0..511`, keyed by `AMM01/world/confirm_fresh/{index}`. Separate noisy case streams remain keyed by the full fresh root and index, 256 cases per root. The old `confirmation` phase is forbidden as a new confirmatory readout. These fresh confirmation roots are disjoint from train, development and the already executed `confirmation` source population.

A source-order fix moves preflight validation ahead of calling `learn_memory()` or `cohort()`, so missing/invalid preflight performs **zero generator queries**. Unit tests must assert this by patching `cohort` and `learn_memory` to error if touched and must not query `confirm_fresh` before its frozen source is committed. The fresh cohort is scored only after (1) committing this amendment, (2) committing the updated source and tests, and (3) rerunning the strict EvalMath preflight on independent development roots. The registered gate and three comparison family remain unchanged.

## Results labeling

The original `CONFIRMATION_LOCK.json` is retained in Git history and marked engineering-only in the report. Place the new locked cohort in `results/CONFIRM_FRESH_LOCK.json`. CI regenerates full raw case traces from the updated source. The old lock is reproducible on the pinned historical source commit, not through the updated module's source digest. No ARG-03-N heldout data are involved.
