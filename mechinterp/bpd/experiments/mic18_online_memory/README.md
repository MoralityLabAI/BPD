# MIC-18: exact delayed-query calibration for BPD

This fixture checks observer aliases, reachable donors and coordinate masks
on an explicitly constructed recurrent ReLU register. It enumerates 2,808
donor cases with assigned target and nuisance roles, covering 15,648
delayed-query cases over J=2..6 binary facts. Those role cases contain
1,404 unique ordered physical recipient/donor pairs: exchanging the two
coordinate roles preserves the donor while changing the registered query.
Every arm uses the same physical update and query matrices. Optimizer
updates, trained models and independently fitted BPD decompositions equal
zero.

## Source and query timing

The model receives J facts one at a time and commits a J-bit state. The
query index arrives afterward. Before the query, the registered observer
returns constant zero. This observer was deliberately made blind to the
retained facts. The delayed query head reads one coordinate through an
integer ReLU network.

For each word, the generator chooses every ordered pair of different
coordinates, called target and nuisance. The donor flips both bits. The
recipient and donor each come from a complete legal observation stream,
at the same final phase J. Candidate selection uses the fixture's known
coordinates and supplies an exact calibration case.

| Arm | Copied coordinates | Squared displacement | Meaning |
| --- | ---: | ---: | --- |
| Clean | 0 | 0 | Original recipient state |
| Whole donor | J | 2 | Full donor effect ceiling |
| Target patch | 1 | 1 | Donor bit at the registered target |
| Nuisance patch | 1 | 1 | Same donor's other changed bit |
| Self sham | 1 | 0 | Target mask using the recipient as source |

Target and nuisance patches have equal cardinality and equal L2 dose. Each
reads the same natural donor. The self sham checks that applying a mask with
an unchanged source preserves the result. Its dose is zero and is recorded
as such. All intervention states remain binary and have explicit legal-word
witnesses at the matched phase.

## Endpoints and cross-query census

Recipient task correctness uses the recipient's independent true bit.
Donor agreement uses the donor's actual query output. Donor-effect recovery
is scored only when the clean recipient and donor outputs differ. Queries
where both sources agree remain in the task and agreement denominators;
their donor-effect field is null.

At the registered target query, the target patch transfers the donor answer
and changes the recipient's correct answer. The nuisance patch preserves
recipient accuracy. Querying the nuisance coordinate exchanges their roles.
Every other query agrees for both sources and all arms. The runner enumerates
this entire cross-query pattern and retains all rows.

Across all widths, each arm has 15,648 task queries and 5,616 queries with a
nonzero donor effect. The exact count expected from the coordinate contract is:

| Arm | Correct recipient answers | Donor effects recovered | Target-query donor effects recovered |
| --- | ---: | ---: | ---: |
| Clean | 15,648 / 15,648 | 0 / 5,616 | 0 / 2,808 |
| Whole donor | 10,032 / 15,648 | 5,616 / 5,616 | 2,808 / 2,808 |
| Target patch | 12,840 / 15,648 | 2,808 / 5,616 | 2,808 / 2,808 |
| Nuisance patch | 12,840 / 15,648 | 2,808 / 5,616 | 0 / 2,808 |
| Self sham | 15,648 / 15,648 | 0 / 5,616 | 0 / 2,808 |

The lower recipient accuracy under donor patches is part of the recorded
result. All clean recipient answers are correct, so this fixture measures
causal corruption and donor transfer. A repair claim would require a task
with an erroneous recipient and a separately defined correction target.

## Observer aliases and memory

The constant prequery observer has one output class. The complete future
query signature has 2^J values. Every pair of different stored words has at
least one query that demands different answers. A deterministic controller
with zero delayed-query errors needs 2^J final states under this interface.
The receipt enumerates all word pairs and records a concrete action-conflict
witness for each width.

The memory cell uses J binary state coordinates and 2J update ReLUs. Its
weights are in {-1,0,1}. It executes J observation updates and then J readout
ReLUs for a single query. Public phase, the one-hot write gate, static
coefficient slots and query work have separate resource fields. Each matched
singleton patch copies one coordinate; the query cost is identical across
arms. Verification tables and raw records belong to the harness, while the
physical cell uses zero codebook lookups.

## Relation to existing BPD contracts

[COL01's application contract](https://github.com/MoralityLabAI/BPD/blob/0b4d66822978590f1150ae2a152ea54f882ec238/mechinterp/bpd/experiments/collatz_observability/BPD_APPLICATION.md)
requires explicit current observers, legal donor sources, physical dose and
separate task and donor endpoints. This fixture supplies finite cases for
those checks. It also records a current-observer alias with a later task
distinction for every different pair of final register states.

[CQ08's protocol](https://github.com/MoralityLabAI/BPD/blob/6d3ca23a817095f38789751004594777f54cf7d8/mechinterp/bpd/theory/cq08_nuisance_mediation/PROTOCOL.md)
holds the natural donor fixed while later component interventions change.
Its component-cardinality controls and physical dose records are separate.
Here the coordinate patch gives exact matches for both quantities. Every
generated pair enters the receipt, including any admission failure; a failed
check stops calibration. The registered finite domain has 2,808 candidates.

These masks are named coordinates in a constructed cell. The port calls its
integer matrices directly. Native `BPDModel` execution, fitted factor
selection, gauge changes and decomposition refits remain outside this
receipt. A native extension can first reproduce these cases, then freeze
its own checkpoint, selection rule, task floor and matched-sham population.
Existing sealed cohorts retain their original status.

## Replay and provenance

The protocol was written before the main enumeration. `SOURCE_LOCK.json`
records source hashes and pinned repository references. `RESULTS.json`
contains every aggregate and declared cost. The complete raw pair census,
with all query outputs, masks, doses, task scores and reachability witnesses,
is stored in `cases.jsonl.gz.b64`. Replay compares its decoded bytes and
checks the source lock.

Run from the BPD repository root with Python 3.10 or newer:

```sh
python -m unittest discover -s mechinterp/bpd/experiments/mic18_online_memory/tests -v
python mechinterp/bpd/experiments/mic18_online_memory/calibration.py --check
```

CI uses Python 3.11 and the standard library. The shared MIC-18 finite-state
evaluator is copied byte for byte, with EvalMath's Apache-2.0 license in
`CORE_LICENSE.txt`. `register.py` is the same original source used in the
companion LoopedTransformers calibration.

The inherited mechanistic-resolution guidance assigns exact finite evidence
to the declared census. There are zero independent trained development
roots, so the sampled-world preflight has no admissible pilot input. Its
power estimate and population sham floor remain unset. The receipt records
`NO_TRAINED_ROOTS_OR_SHAM_CALIBRATION` for that population gate.
