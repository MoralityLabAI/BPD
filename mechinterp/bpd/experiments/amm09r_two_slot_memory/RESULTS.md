# AMM09R1: prospective development under the accepted CPU recipe

The 64-root prospective development study passed every registered admission
gate. Memory-only routing recovered 12,502 of 32,768 held-out cases; the
training-selected fixed mask recovered 8,038. The primary gain was
**13.6230 percentage points**, with a paired-root bootstrap 95% interval of
**[9.8328, 17.5140] points** and Holm-adjusted exact root-sign p
**1.49194e-6**. Its lower interval bound exceeds the registered 8-point floor.

All 128 AMM09R1 confirmation roots remain sealed. The source has no confirmation
runner. Every protected historical cohort remains sealed as well.

## Prospective sequence and numerical evidence

The [protocol](PROTOCOL.md) was committed after the excluded engineering gate
passed. All executable study source, its independent auditor, the workflow,
and the runtime manifest were then frozen before any fresh training root was
generated. The fresh namespace is `AMM09R1/`; the earlier AMM09 development
result informed this follow-up design and remains a separate exposed study.

| Stage | Immutable reference |
| --- | --- |
| Engineering registration | `ffc7143e3b597d840c5000a94a45b45d964931cf` |
| Passing engineering run | [44291185512](https://github.com/MoralityLabAI/BPD/actions/runs/44291185512) |
| Fresh protocol committed | `e4b16494a7d5e70357ede56fe804dd1fef6ab364` |
| Fresh source frozen | `f9bac29a8e386c8285b94818f1087e3681e1cdcd` |
| Fresh training and exact replay | [44294944939](https://github.com/MoralityLabAI/BPD/actions/runs/44294944939) |
| Training policy locked | `bbcbae41c3e8cd5264c297e5d457ca726995a735` |
| Single development execution | [44295607653](https://github.com/MoralityLabAI/BPD/actions/runs/44295607653), workflow run 2, attempt 1 |

The accepted runtime is Python 3.12.12, torch 2.10.0+cpu, and numpy 2.2.6,
with one intra-op and inter-op thread, deterministic algorithms, generic CPU
dispatch, and the registered oneMKL compatibility environment. The runtime
guard verifies the exact PyTorch source revision and binary hashes in
[RUNTIME.json](RUNTIME.json).

Both fresh training workers produced identical complete 16,143,086-byte
training files, including retained feature arrays, fitted model hashes,
candidate bits, and the learned policy. Their separate hosts both reported
AMD EPYC 7763 processors and used Ubuntu 22.04 and 24.04 respectively. The
development worker reported AMD EPYC 9V74. The earlier excluded engineering
fixture established four exact pinned32 process repeats across those two
processor models; its [report](../amm09r_reproducibility/RESULTS.md) gives the
observed portability scope and preserves the unresolved historical incident.

The 64 training roots supplied 128 cases each. The pooled action was index 4,
mask `(0,2)`. The policy digest is
`b34b9a322ff5aeaa517eb5edd4bf0aced9ccf70f0bf97383d8e6cca79de2f0e9`.
Development admission rebuilt its ridge from the retained training features
and required the committed policy lock before drawing development inputs.
Twenty synthetic accounting and protected-root tests passed on each training
worker before fitting.

## Held-out outcomes

One independent unit is a separately trained 6D rank-three recurrent model
and its fitted BPD decomposition. Each of the 64 development roots supplied
512 held-out cases. Every root remains in the reported denominator, including
the six roots that missed at least one source-fidelity criterion.

| Policy or diagnostic | Successes / 32,768 | Rate |
| --- | ---: | ---: |
| Memory only | 12,502 | 38.1531% |
| Memory plus local residual update | 11,340 | 34.6069% |
| Local ridge | 8,956 | 27.3315% |
| Training-selected pooled mask | 8,038 | 24.5300% |
| Local count | 7,988 | 24.3774% |
| Random permitted mask | 5,368 | 16.3818% |
| Random dark direction | 1,648 | 5.0293% |
| Independent random dark direction | 1,612 | 4.9194% |
| Sham | 0 | 0% |
| Six-mask casewise oracle, audit only | 18,141 | 55.3613% |
| Full donor, visible control | 32,538 | 99.2981% |

The full donor uses the same L2 displacement and can change the immediate
readout. The admitted one- and two-component writes preserve that readout.
The full donor's unrestricted geometry gives it a separate control role.
The six-mask oracle uses held-out labels solely to measure available headroom.

The fixed global policy selected two-component masks on every held-out case:
15,228 used `(0,1)`, 11,381 used `(0,2)`, and 6,159 used `(1,2)`. The fixed pooled
comparator used `(0,2)` throughout. All casewise action histories remain in the
raw receipt and the [independent analysis](results/ANALYSIS.json).

Memory-only choices varied within every one of the 64 model roots. Sixty-two
roots used all three pairs, and the remaining two used two pairs each. Its
choice differed from the pooled mask on 21,387 cases, or 65.2679%. A descriptive
check of all six fixed masks on the held-out data also ranks `(0,2)` highest;
the registered training selection was already the strongest fixed comparator
on this cohort. [ACTION_USAGE.json](results/ACTION_USAGE.json) preserves these
post-development descriptions without adding a hypothesis test.

## Registered inference and the adverse local update

The four contrasts use exact one-sided root sign tests, Holm correction across
the declared family, and the registered 2,000 paired-root bootstrap resamples.
Positive, negative, and tied counts refer to independent model-root effects.

| Treatment minus control | Mean gain, pp | Root-bootstrap 95% interval, pp | Positive / negative / tied roots | Holm p |
| --- | ---: | --- | --- | ---: |
| Memory only minus pooled, primary | +13.6230 | [+9.8328, +17.5140] | 51 / 12 / 1 | 1.49194e-6 |
| Memory plus local minus pooled | +10.0769 | [+6.1096, +14.2059] | 49 / 15 / 0 | 2.43646e-5 |
| Memory plus local minus local ridge | +7.2754 | [+4.8004, +9.8480] | 50 / 12 / 2 | 1.82157e-6 |
| Memory plus local minus memory only | -3.5461 | [-5.1971, -2.0020] | 18 / 44 / 2 | 0.999751 |

The local update lost 1,162 successes relative to memory alone. Its registered
positive-direction test fails to support an improvement. The interval lies
below zero, and this adverse secondary outcome remains part of the record.

Every local comparator received the same 16 calibration cases per root and
96 noisy labels. The complete local budget was 6,144 labels, of which 1,303
were flipped by the predeclared 20% noise process. The global prior additionally
uses the disclosed 64-root training corpus. Its result depends on that prior
experience and the registered feature representation.

## Admission gates and fidelity

| Gate | Observed value | Registered requirement |
| --- | --- | --- |
| Primary lower 95% bound | +9.8328 pp | Greater than calibrated floor |
| Calibrated floor | 8 pp | Maximum of physical minimum, resolution, and sham envelope |
| Primary Holm p | 1.49194e-6 | Less than .05 |
| Source fidelity | 58 of 64 roots | At least 56 |
| Allowed-oracle headroom over pooled | 30.8319 pp | At least 8 pp |
| Maximum dose error across all assays | 4.76837158203125e-7 | At most 1e-5 |
| Immediate readout displacement | 0 | Exactly zero |
| Full donor dose error | 0 | At most 1e-5 |
| Pinned EvalMath decision | READY_FOR_CONFIRMATION | READY with no listed reasons |

The six fidelity failures are `development/0008`, `0019`, `0029`, `0051`,
`0052`, and `0061`. All six exceed the relative operator-error limit. Roots
0008 and 0051 also miss the factor task-MSE limit; roots 0029 and 0061 miss the
teacher-MSE limit. Their outcomes remain in every primary and secondary test.
[FIDELITY_STRATA.json](results/FIDELITY_STRATA.json) retains each root's exact
diagnostics and descriptive subgroup totals. The primary mean within the
58 passing roots is +14.1467 pp; the six failing roots have a +8.5612 pp mean.
These post-development descriptive strata add no registered hypothesis tests.

The matched sham compares the fixed pooled policy across two independent
512-case sets within each root. The maximum absolute difference across 64 roots
is 5.46875 pp, giving the registered 95th-quantile tolerance bound at at least
95% confidence. The 8-point physical minimum therefore determines the floor.
Repeated cases within a root do not increase the independent-root count.

## Independent audit and claim scope

The frozen standard-library auditor reconstructs selected outcome bits from
admitted actions and candidate outcomes, verifies every per-root count, and
recomputes all four comparisons. It rebuilds the pilot and executes EvalMath
at commit `6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a` after verifying the planner's
SHA256. A separate local rerun from the downloaded raw receipts reproduced the
hosted analysis, pilot, and preflight files byte for byte, with no model fitting.
The auditor explicitly records that it does not itself refit the training ridge
or replay the noisy-label RNG; the admitted development execution validates the
training ridge from its retained features.

EvalMath reports READY for planning a reserved 128-root confirmation study.
Its approximate power calculation remains a planning estimate. Access to those
roots requires separate explicit human authorization and a newly committed
release protocol. The present evidence is prospective development on this
synthetic linear system. Extension to nonlinear transformers or deployed agents
requires additional experiments.

The exact [training lock](results/TRAINING_LOCK.json),
[development lock](results/DEVELOPMENT_LOCK.json),
[independent analysis](results/ANALYSIS.json), and
[planner output](results/PREFLIGHT.json) preserve this result. The
[training manifest](results/TRAINING_RECEIPTS.json) and
[development manifest](results/DEVELOPMENT_RECEIPTS.json) bind every raw file
and original hosted artifact by SHA256. With the development lock present,
ordinary workflow runs audit those immutable artifacts without generating
another model cohort.
