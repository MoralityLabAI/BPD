# Starting certificates at a search budget boundary

The recovered HAD-05 archive runners checked the remaining search budget before
they looked for a complete starting trajectory. Initial archive work could spend
that budget. A run with zero rounds also skipped the completed-state check.
Both paths could report `best: 0` with `solved: false` and an empty certificate.

The corrected runners inspect the full population before collecting archive
rows. Each loop also checks for a completed state before deciding whether it can
afford another flip. A complete starting trajectory now returns its certificate
with zero search and archive work. `near.py` records its native generation as zero.

The focused regression suite is `test_budget_edges.py`. It covers every archive
arm at zero budget, the reproducible seed 3 / length 3 / population 1 / chunk 1 /
rounds 1 failure, and a completed trajectory in a later population slot.

## Source provenance

The recovered sources came from commit
`3b7ee1775b1e7a709c536641105ee4fc3195a5a8` on
`research/had05-ldt-population-20261126` in `MoralityLabAI/BPD`.
Their exact bytes were preserved before applying this patch.

| Historical source | Git blob identifier |
| --- | --- |
| archive.py | `563cdb30ffcc4f0facd341c72bc413d36baf8632` |
| near.py | `b74568f2a18f4cdcc13c3e3bb167d46842156d27` |

Historical result claims belong to those historical source versions. This patch
changes the behavior of runs that already contain a completed starting state.
The source identifiers above continue to identify the recovered versions.
