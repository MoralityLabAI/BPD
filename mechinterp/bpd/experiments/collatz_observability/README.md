# BPD-COL-01: exact Collatz observer calibration

**Status: exhaustive finite arithmetic study, 25 November 2026.** This package
connects the [MathCan split-Collatz results](https://github.com/patrickdugan/MathCan/tree/b3eb33ed10485264fe7a593e3b0ace418d9f9a6d/research/collatz_resonance)
to BPD's current-observer, delayed-response and reachable-donor contracts.
Every state in the declared finite population is enumerated. The results are
public development fixtures; neural training and native factor interventions
are separate experiments.

## Why this belongs in BPD

BPD needs examples where the information available to its observer is known
exactly. The split map supplies both a conserved observer with zero closure
defect and a richer observer with exact delayed effects. Legal donor states
and their complete continuations can be checked with integer arithmetic.

The package keeps four questions explicit:

| Question | Exact object |
| --- | --- |
| Can the observer predict its own next output? | Closure of the observer partition under the transition |
| Can the observer identify the eventual basin? | Basin labels within each observation class |
| Does a protected donor change a later output? | Full continuation from two legal states with equal current readout |
| How many linear directions are observable? | Rank of the stacked operators in a specified one-hot embedding |

See [THEORY.md](THEORY.md) for the proofs and
[BPD_APPLICATION.md](BPD_APPLICATION.md) for the native experiment handoff.

## Reproduce

Run from the repository root with Python 3.11 or newer. The package uses the
standard library.

```bash
python -S -m unittest discover -s mechinterp/bpd/experiments/collatz_observability/tests -v
python -S mechinterp/bpd/experiments/collatz_observability/audit.py --check mechinterp/bpd/experiments/collatz_observability/results/exact_receipt.json
```

The receipt records the unchanged canonical oracle's SHA-256 and the source
commit. It also records every parameter separately so aggregate scores can be
recomputed. The canonical Python oracle is vendored to keep replay independent
of network access.

Direct script execution avoids the parent BPD package's neural imports. The
`-S` checks also verify that replay works with only the standard library.

## Information and state contract

Fix a positive parameter `d = 1 mod 4` with `gcd(d,6) = 1`. The legal state set
is the odd core `S_d = {1,3,...,d}`. For an odd state `n`, define

\[
c(n)=\begin{cases}3&n\equiv1\pmod4,\\1&n\equiv3\pmod4,\end{cases}
\quad a(n)=v_2(c(n)n+d),\qquad
F_d(n)=\frac{c(n)n+d}{2^{a(n)}}.
\]

Let `q(n)` label the multiplicative `<2,3>` orbit of `n mod d`. The principal
current observer is `r(n) = (q(n), c(n))`. A public packet exposes `d`, `q`
and `c`. The seed and basin remain host-owned. The sampling model allows every
legal core state as an initial state with `d` fixed.

Giving an unrestricted predictor the seed together with `d` makes the entire
future computable. A task that supplies the seed therefore measures bounded
computation. The information obstruction studied here concerns the explicitly
restricted observer and hidden initial state.

An additional instrument reports `(q,c,a)`. Its valuation `a` requires
observing the transition. This extra information has its own cost and is
reported separately from the branch observer.

## Exact diagnostic examples

| Parameter and states | Current observation | Future effect | What it checks |
| --- | --- | --- | --- |
| `d=13`, recipient `1`, donor `5` | Same `(q,c)=(1,3)` | `1 -> 1`, `5 -> 7`; next branches are `3` and `1` | A current-invisible state change has a delayed visible effect |
| `d=29`, states `5` and `9` | Same `(q,c)=(1,3)` | Cycles `5 <-> 11` and `9 <-> 7` have identical branch words forever | A closed predictive observer can still lose basin identity |
| `d=73`, recipient `1`, effect donor `9`, sham donor `37` | All share `(q,c)=(1,3)` and basin `3` | The effect changes the next branch; the sham preserves the entire branch trace | Equal-dose reachable donor control within one basin |

In the one-hot state representation, each distinct donor has displacement
norm `sqrt(2)` and remains a legal state of norm one. The `d=73` sham preserves
the observable trace at every horizon by the stable-partition certificate.
Its hidden state differs from the recipient's state.

For `d=13`, the number of distinguishable pure-state classes grows from `3`
at the current observation to `5` with one future output and `7` with two.
The stacked linear observability ranks at horizons `0,1,2,3` are
`3,4,5,6`; later horizons retain rank `6`. These are different measured
objects. The receipt checks both with exact arithmetic.

## Reading the audit

The default domain is all 172 admissible parameters through `d=1025`, containing
44,204 odd states. Each observer is refined until its partition is stable.
Horizon `h` includes the current output and `h` future outputs. The full future
partition is an offline oracle used to label experimental cases.

The [result report](RESULTS.md) gives the exact aggregate counts, including
the 35 parameters whose complete branch-observer histories still merge
different basins.

Basin accuracy is the exact Bayes ceiling under a uniform distribution on the
odd states of each parameter. Pooled state counts weight larger cores more
heavily; parameter-level results remain in the receipt. These scores measure
finite information content. They are independent of a learned model's
generalization or compute cost.

The ordered branch-and-valuation decoder is checked separately at length
five. Its inputs contain more information than the branch observer. The
[LoopedTransformers observer study](https://github.com/MoralityLabAI/LoopedTransformers/tree/b14a51015996f0dee81ef033e2e198d9f84951b2/research/collatz_observer_gym)
already records the capacity confound from shuffled finite partitions and
defines a prospective neural discrimination experiment.

## Existing BPD consumers

- [Causal quotient note](../../../../papers/shared_weights_dark_fibers/CAUSAL_QUOTIENT_ARG03.md): exact closure, matched fibers and current-versus-trace information.
- [EFD-01](../efd01/README.md): current-readout kernels and delayed linear observability.
- [CQ07 reachability](../../theory/cq07_reachability/THEORY.md) and [CQ08](../../theory/cq08_nuisance_mediation/README.md): legal donors and later component mediation.
- [DT03](../depth_steering_gym/DT03_REVIEW.md): complete output Jacobians, whole-case recovery and native state continuation.

Ordinary refinement certifies the autonomous transition `F_d`. Closure under
a family of BPD masks or weight edits requires checking every declared edited
transition. The exact finite suite provides known answers for that later
integration.
