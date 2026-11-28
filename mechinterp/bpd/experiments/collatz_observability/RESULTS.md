# BPD-COL-01 exact finite results

The [replayable receipt](results/exact_receipt.json) enumerates all 172
admissible parameters `d<=1025`, including 44,204 odd states and 924 eventual
cycles. These parameters were already public in the upstream Collatz work.
The new observer and donor analyses are exhaustive development results.

## Common basin target

Each row below uses the same target: the eventual basin. Scores pool states
across parameters, while each observation also identifies its parameter.
Larger cores receive more weight in this pooled measure. Per-parameter
counts are preserved in the receipt.

| Available observation | Current classes | Current basin ceiling | Stable future classes | Complete-future basin ceiling | Roots with residual basin ambiguity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arithmetic quotient `q` | 653 | 37,317 / 44,204 (84.4200%) | 653 | 37,317 / 44,204 (84.4200%) | 85 |
| Quotient and branch `(q,c)` | 1,134 | 37,404 / 44,204 (84.6168%) | 43,191 | 43,409 / 44,204 (98.2015%) | 35 |
| Quotient, branch and valuation `(q,c,a)` | 4,965 | 37,950 / 44,204 (85.8520%) | 44,204 | 44,204 / 44,204 (100%) | 0 |

The conserved `q` observer has zero delayed-visible pairs. Under `(q,c)`,
1,261,132 unordered distinct pairs match now and differ at the next readout;
2,502,472 pairs that match now differ at some later horizon. All comparisons
stay within one parameter.

Even the complete `(q,c)` future leaves 1,462 unordered pairs with different
basins in the same observation class. Its optimal decoder incurs 795 errors
over the uniform pooled state population. This establishes a finite
target-sufficiency limit for that particular observer.

Branch refinement reaches its last strict refinement by horizon 20 at
`d=557`. The richer branch-and-valuation observer reaches singleton classes
by horizon four for every parameter, which means five emitted observations
including the current one. A separate modular decoder also recovers every
seed from five ordered `(c,a)` emissions.

The receipt's `current_to_next_correct` fields predict each observer's own
next output. Their target alphabets differ across modes. The basin table
above uses a common target for comparisons.

## Protected donors

At `d=13`, replacing one-hot state `e_1` by `e_5` preserves the complete
current `(q,c)` readout. Squared readout differences over horizons `0..4`
are `[0,2,0,2,0]`. A terminal-only measurement at horizon two would therefore
miss the earlier effect. Donor `e_9` instead gives the first difference at
horizon two.

At `d=73`, donors `9` and `37` both have squared physical displacement two
from recipient `1`. Every state is legal, current observations agree and all
belong to basin `3`. The donor `9` changes the next branch. The donor `37`
preserves the entire observable future, certified by the stable partition
and an independent cycle replay.

At `d=29`, states `5` and `9` occupy different cycles while producing the
same `(q,c)` word forever. This fixture prevents a stable observer quotient
from being silently treated as a complete basin identifier.

## Distinguishability and rank

| Horizon | `d=13` distinguishable legal-state classes | Rank of stacked one-hot observation operator |
| --- | ---: | ---: |
| 0 | 3 | 3 |
| 1 | 5 | 4 |
| 2 | 7 | 5 |
| 3 | 7 | 6 |
| 4 through 6 | 7 | 6 |

The nonzero signed vector `e_1-e_3+e_7-e_9` is killed by both the current
readout and the transition, which certifies an ambient unobservable direction
at every horizon. It is outside the allowed set of differences between two
legal one-hot states. The rank and finite-state class count have separate
interpretations.

## Verification

Eight focused tests check the original unaccelerated transition on all
44,204 states, compare refinement with independently replayed finite words,
and exercise exact donor effects and rational-rank witnesses. They also check
the public observation contract and decode every five-emission prefix.
The full serialized receipt replays byte for byte in a standard-library-only
Python process.

An independent review recomputed the branch-observer census with its own
scalar transition, disjoint-set arithmetic quotient, basin traversal and
future-word refinement. It agreed on the parameter/state totals, initial and
stable class counts, basin ceilings and maximum refinement horizon.

The study identifies exact finite behavior under the declared observer and
one-hot representation. Native BPD factor recovery and neural generalization
require the separately described [adapter experiment](BPD_APPLICATION.md).
