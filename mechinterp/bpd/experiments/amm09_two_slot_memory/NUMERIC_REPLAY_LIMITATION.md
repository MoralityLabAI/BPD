# AMM-09: cross-host numerical portability incident

**Post-development engineering disclosure, 2026-11-18. No protected confirmation model was accessed.** This note is separate from the preregistered [scientific protocol](PROTOCOL.md) and [primary-effect correction](AMENDMENT_POST_DEV_ANALYSIS.md).

Two GitHub Actions CPU runners executed the exact same frozen 6D rank-three teacher/factor training code with source SHA256:

- `models.py`: `2fec8382c067147d60d9f2d4fe8d171ce3d5e15f405afffbbab5389023bc1a03`
- `case.py`: `f72788b058d7eff58218ed4e0c729437820a8cfdd2518609409f2fc7fec9132a`
- `experiment.py`: `57af128ca48ef2dcd527abfcd443373705ecd046a460d53bafe2a0222df56897`

The first two hosted runs matched the source-locked memory policy and training candidate bit digest, and the second scored the first independent development cohort. A later host in [Actions #44282758367](https://github.com/MoralityLabAI/BPD/actions/runs/44282758367) fit the same 64 seeded TRAIN roots. Its **trained physical and BPD tensor bytes differed on all 64 roots**, with small but amplified float deviations. The late-run initial-conditions and seed identities did not change.

| Quantity | Original first-run | Independent later-host replay |
|---|---|---|
| Independent training model roots | 64 | 64 |
| Full source SHA256 triplet | Same | Same |
| Best pooled bounded mask | index 4, (0,2) | index 4, (0,2) |
| Training mask-0 successes | 999 | 997 |
| Training mask-1 successes | 599 | 599 |
| Training mask-2 successes | 522 | 523 |
| Training mask-3 successes | 1786 | 1786 |
| Training mask-4 successes | 1801 | 1799 |
| Training mask-5 successes | 1425 | 1427 |
| Memory-policy canonical SHA256 | `c536d7bd…e95ef` | `c1899152…2d4d` |
| Training candidate bits SHA256 | `c46dc9ae…192d` | `2041e25f…15138` |

Across `64 model roots × 128 training cases × 6 candidate masks = 49,152` scored labels, **27 candidate success bits** differed (0.0549%), all within eight model roots. The mask-specific Hamming distances were `{mask0:6, mask1:2, mask2:1, mask3:6, mask4:8, mask5:4}`. The largest absolute difference among the 18 *four-decimal-quantized* global ridge coefficients was **0.0036**. The pooled mask was unchanged. Consequently the source-locked strict SHA256 check failed **before** evaluating any further development models. This was an appropriate fail-closed result, rather than a model behavioral counterexample.

The actual training implementation sets `torch.set_num_threads(1)` and SHA256-based per-root tensor generators; those controls alone do not establish portable bit-for-bit numerical trajectories across differing CPUs and PyTorch/BLAS execution details. Different floating-point accumulation/rounding through 400 teacher and 240 factor optimization steps can accumulate enough to change a handful of borderline success outcomes. The raw archive and this observation do not isolate a single underlying CPU instruction or library setting, so a precise hardware cause remains unverified.

## Reproducibility and evidence boundaries

The original once-scored raw DEVELOPMENT file `amm09_development.json`, SHA256 `0423c8302c9cbb99b9413ac7e22cfcc7d0f10d921eff06c0f4a17cda33ff3925`, and its original training policy were retained without edits, with [original hosted artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44281548742). An independent **source-of-record audit** downloads those exact artifacts, validates their SHA256 and all six-choice candidate-action bits, and recomputes the preregistered root statistics. This supports the **original prospective developmental effect**, while explicitly leaving cross-host exact-model reproduction unestablished.

For future CI, distinguish:
- **Original-receipt integrity:** a hard gate on archive hashes, source identifiers, six-mask eligibility, named-arm bit accounting, all four corrected root tests, and the pinned original EvalMath receipt. Failure blocks publishing or merge.
- **Cross-host model training portability:** an **explicit, independently logged warning** with both policy hashes, pooled action and candidate hit deltas; the subsequent development replay is skipped when training differs. Never call this warning an exact replay pass. A strict source re-training comparison can still be run manually and will report the mismatch.
- **Scientific confirmation:** permanently blocked until a separately source-locked, human-approved release manifest is issued; the current cross-host portability issue is a further reason to run fresh **AMM09R** deterministic-backend validation before any protected confirmation.

Any prospective AMM09R change to numerical backend, precision, loss, optimizer, seed namespace or success threshold must be registered **before its own fresh independent model roots**. It must not reuse AMM09 development roots to tune a new effect or reclassify the current development result as confirmed. A new deterministic policy can be compared in a separate, power-calibrated prospective study.

The current evidence remains a **positive, separately audited developmental result on synthetic trained linear recurrent models, accompanied by an unresolved cross-host training portability issue**. No full-rank-three intervention mask was permitted in its primary or control actions.
