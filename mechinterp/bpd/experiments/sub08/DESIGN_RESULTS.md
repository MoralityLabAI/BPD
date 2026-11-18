# SUB-08 excluded design pilot result

The registered design screen returned **DESIGN_PASS**. The fixed cost ordering
selected **GRU48 with 2400 updates**. All four candidate configurations passed
all three excluded seeds. This phase used validation for selection; heldout
model predictions were unexposed.

## Complete candidate measurements

| Seed | Width | Updates | Train accuracy | Validation accuracy | Lowest class recall | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1890 | 48 | 2400 | 100.0000% | 100.0000% | 100.0000% | PASS |
| 1890 | 48 | 3600 | 100.0000% | 100.0000% | 100.0000% | PASS |
| 1891 | 48 | 2400 | 100.0000% | 100.0000% | 100.0000% | PASS |
| 1891 | 48 | 3600 | 100.0000% | 100.0000% | 100.0000% | PASS |
| 1892 | 48 | 2400 | 100.0000% | 100.0000% | 100.0000% | PASS |
| 1892 | 48 | 3600 | 100.0000% | 100.0000% | 100.0000% | PASS |
| 1890 | 64 | 2400 | 100.0000% | 99.8047% | 99.0566% | PASS |
| 1890 | 64 | 3600 | 100.0000% | 99.8047% | 99.0566% | PASS |
| 1891 | 64 | 2400 | 100.0000% | 100.0000% | 100.0000% | PASS |
| 1891 | 64 | 3600 | 100.0000% | 100.0000% | 100.0000% | PASS |
| 1892 | 64 | 2400 | 100.0000% | 100.0000% | 100.0000% | PASS |
| 1892 | 64 | 3600 | 100.0000% | 100.0000% | 100.0000% | PASS |

The 2400/3600 checkpoints share a training trajectory within each width and seed.
There were six trajectories, three seed IDs and twelve scored checkpoints.
All scores are retained. The independently written auditor replayed every
saved checkpoint, verified raw targets through literal state transitions, and
reconstructed the fixed selection decision. Maximum replay logit error was zero.

## Prospective handoff

The eight-root source study may now use the selected width48/update2400 setup,
with its own committed COMPETENCE_LOCK before roots1900..1907 are trained.
The engineering gate requires at least seven passing roots out of all eight,
with heldout exact-state accuracy >=90% and each class recall >=85%.

The inherited SUB-07 strict EvalMath result remains REDESIGN_OR_PILOT_ONLY.
SUB-08 has zero causal measurements; future clock discrimination requires a
separate registration and fresh trained roots.

## Sources and validation

The [protocol](DESIGN_PROTOCOL.md) was committed at8bcbd722 before source code
and training. [Source commit b6c391c1](https://github.com/MoralityLabAI/BPD/commit/b6c391c1d05cf72df85ede28135c9955f46c5c9c)
precedes all six trajectories. [Source-only CI](https://github.com/MoralityLabAI/BPD/actions/runs/44316820005)
passed; the local suite passed20 tests. [DESIGN_RECEIPT.json](DESIGN_RECEIPT.json)
contains hashes for every original checkpoint and raw record, and
[DESIGN_AUDIT.json](DESIGN_AUDIT.json) contains independently reconstructed metrics.
