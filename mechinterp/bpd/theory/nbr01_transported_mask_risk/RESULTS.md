# BPD-NBR-01 exact arithmetic results

Registered protocol commit 8bdafbf07e0dc682dce9451051e6174894634579.
Successful exact arithmetic workflow head 2229decc4525234113fbb25ee382cd05cf829b63,
Actions 45525306350, job 138907047265, artifact 14981349054.
worlds.json SHA256 ad0b773e292411f4d9a745383b3d9c670bdae7a7b00ef6c01bc1ba8b78164bd5.

All 128 generated systems, two bases per system and 28 masks per basis
(7,168 enumerated masks) passed exact full-write / horizon-output parity,
direct / Gramian-risk parity and oracle dominance. Exact witness and
orthogonality tests and deterministic byte replay passed. An independent
local replay using integer-scaled writes (denominator four for rotated
coordinates) also passed these parity checks for all systems and masks
and reproduced all reported bounded physical/observable means to six
decimal places. This independent replay was not an independent training
or neural-model replication.

Two pre-scoring engineering fixes are retained: 6f2e8d84 corrects list
versus tuple equality in the orthogonality assertion; 2229decc invokes
the standalone Python script rather than importing the Torch-dependent
BPD package. Neither changes generated worlds, mask risk, comparison
arms, seed or cardinality. Original protocol unchanged.

## Mean bounded reconstruction loss, fixed k=2 of 8

| Policy | Native physical | Native observable | Rotated physical | Rotated observable |
|---|---:|---:|---:|---:|
| Physical diagonal | .128297 | .146208 | .238368 | .239157 |
| Observable diagonal | .137531 | .144603 | .260175 | .221685 |
| Observable greedy | .193147 | .143397 | .284729 | .174819 |
| Observable exact oracle | .171009 | .130303 | .279533 | .167554 |
| Uniform mask | .415751 | .420702 | .421000 | .424485 |

The exact physical oracle matches the physical diagonal on every root
because both dictionaries are orthogonal in this construction.
Observable greedy versus physical-diagonal observable raw risk:
native 27 wins, 92 ties, 9 losses; rotated 68 wins, 59 ties, 1 loss.
Observable diagonal versus physical diagonal: native 5/122/1,
rotated 31/79/18. Oracle improves over greedy in 10 native and 13
rotated roots. The oracle is an exhaustive lower bound, not a learned
or cost-matched policy. On the rotated basis, greedy lowers mean
bounded observable loss from .239157 to .174819 (~26.9% relative).
On the native basis it lowers .146208 to .143397 (~1.9% relative).
Physical loss increases in both cases; the objective tradeoff is real.

## Interpretation

The artificial source explicitly favors sparse native activations and
uses orthogonal writes, signed-permutation dynamics, small readouts and
integer coefficients. It cannot establish a learned BPD advantage or
predict the sign of the NBR neural observable effect. It establishes
the exact risk identity, constructive failure of magnitude ranking
and a controlled demonstration that cancellation-aware selection can
improve observable fidelity at fixed mask cardinality. The next stage
must use real BPD reached states, matched derivative access and
nonlinear finite interventions on fresh roots. Existing sealed
datasets and prior scientific verdicts are unchanged.
