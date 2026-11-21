# RLT-EXIT-03: frozen learned controller under Sudoku difficulty shift

2026-11-21. **Development-only native pretrained evaluation.**
Original CI [44913713826](https://github.com/MoralityLabAI/BPD/actions/runs/44913713826) succeeded,
including unit tests, checkpoint custody checks and two actual native model sweeps.
Its artifact [bpd-rlt-frozen-difficulty-transfer](https://github.com/MoralityLabAI/BPD/actions/runs/44913713826)
contains all 48 independently generated source worlds and per-policy receipts.

## Frozen policy and separate roots

The controller was fitted and calibrated in merged BPD PR #140 exclusively on
50-blank Sudoku tasks. This study loads the exact prior report whose SHA-256 is
dda473da08434484f4d67e8f0e6fa77b4296d9fa1489a810f529c699dc94a7e1,
including logistic weights, feature mean/scales, and stop threshold
0.5656608144205763. The pretrained gaoxin492/TRM checkpoint SHA-256 is
932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3.

No training, threshold fitting or posterior selection took place on the new
cohorts. Roots were generated with disjoint source seed families:
40 blanks starts 20261311; 52 blanks starts 20261312. Both advance by 101
per independently generated Sudoku. Each puzzle has a unique verified
solution and source hashes are disjoint from all original TRAIN, CAL and
DEVELOPMENT roots. No sealed confirmation roots were used.

## Results

| Difficulty | Policy | Exact solves | Outer steps | Native CPU seconds | Incorrect early exits |
| --- | --- | ---: | ---: | ---: | ---: |
| 40 blanks | Fixed 1 | 24/24 | 24 | N/A (offline first-visit diagnostic) | 0 |
| 40 blanks | Fixed 2 | 24/24 | 48 | 11.540951 | 0 |
| 40 blanks | Exact verifier exit | 24/24 | 24 | 5.813416 | 0 |
| 40 blanks | Learned, original threshold | 24/24 | 24 | 5.790907 | 0 |
| 40 blanks | Matched random (CAL stop rate) | 24/24 | 39 | 9.449739 | 0 |
| 52 blanks | Fixed 1 | 8/24 | 24 | N/A (offline first-visit diagnostic) | 16 |
| 52 blanks | Fixed 2 | 22/24 | 48 | 11.637567 | 0 |
| 52 blanks | Exact verifier exit | 22/24 | 40 | 9.647101 | 0 |
| 52 blanks | Learned, original threshold | 21/24 | 41 | 9.866530 | **1** |
| 52 blanks | Matched random (CAL stop rate) | 17/24 | 39 | 9.428214 | 6 |

At 40 blanks all 24 puzzles were already solved at the first recurrent
step, so the model's learned threshold stopped perfectly and used half
of the fixed two-step budget. This easy shift has a performance ceiling.

At 52 blanks, a single incorrect learned first-step exit cost an
otherwise recovered second-step solution. The remaining cases yielded
21/24 valid solutions, seven steps fewer than fixed-depth, and
about 15.2% less measured native forward time.
A threshold chosen to make zero false stops on 48 CAL roots at the
50-blank source distribution did NOT provide the same property after
difficulty shift. This is a negative stress finding that should
remain visible in later comparisons and controller training.

The random gate's CAL stop rate is 22/48. It uses a deterministic
per-root pseudo-random decision and may consume fewer steps than
the learned gate in a particular cohort. The control is
rate-matched in expectation, not guaranteed to spend the exact same
number of recurrent visits.

The fixed-one diagnostic comes from the first visit of an independent
full-depth model rollout. It does not provide a separate timing estimate.
CPU wall-time variation across separate model invocations limits
precise relative runtime comparisons; visit counts are exact.

## Next registered experiment

An applicability guard may restrict learned stopping to tasks whose
public difficulty metadata falls within a previously designated
safe support set, allowing an otherwise fixed-depth fallback.
One natural monotone guard is to enable learned stopping for at most
50 Sudoku blanks while requiring two steps for harder boards.
This is a **posthoc idea prompted by the observed failure**.
Its evaluation must use fresh, previously untouched development
seeds, with the rule frozen before the run. The already opened
52-blank cohort is excluded from rule assessment and cannot be
called a confirmation sample.

Lean proof and MCP tool calls require distinct independent
verifiers and observable state features; transfer across those task
families remains untested. The observed Sudoku result validates
a bounded native-model difficulty stress test only.
