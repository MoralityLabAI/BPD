"""Exact observer refinements and intervention fixtures for BPD-COL-01.

Every reported domain is exhaustively enumerated. Refinement at horizon h
uses observations at times 0 through h, with future observations supplied by
the evaluator. These partitions are offline oracles, not online predictors.
The conserved arithmetic quotient is closed under the dynamics; a basin
target is a separate observable. No neural model or held-out data is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from math import gcd
from pathlib import Path
from typing import Hashable

if __package__:
    from .collatz_core import Core, analyze, modular_labels, odd_step, validate_parameter
else:
    # Direct-script execution avoids the enclosing BPD package's eager neural
    # imports, so the exact finite audit needs only the Python standard library.
    from collatz_core import Core, analyze, modular_labels, odd_step, validate_parameter


ROOT = Path(__file__).resolve().parent
CORE_SHA256 = "24ff52171aedf890c33439896ab5e71de65d9b05d557ba61d162f3623aa5dbca"
SOURCE_COMMIT = "b3eb33ed10485264fe7a593e3b0ace418d9f9a6d"
MODES = ("quotient", "quotient_branch", "quotient_branch_halvings")
DEFAULT_RECEIPT = ROOT / "results" / "exact_receipt.json"
MAX_PARAMETER = 8193
AUDIT_MAX_PARAMETER = 1025


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def validate_bound(max_parameter: int) -> None:
    if type(max_parameter) is not int or not 1 <= max_parameter <= MAX_PARAMETER:
        raise ValueError(f"max_parameter must be an integer in [1,{MAX_PARAMETER}]")


def validate_state(d: int, n: int) -> None:
    validate_parameter(d, quotient=True)
    validate_bound(d)
    if type(n) is not int or n < 1 or n > d or n % 2 != 1:
        raise ValueError("state must be a positive odd integer at most d")


def public_observation(d: int, n: int) -> dict[str, int]:
    """Host projects its hidden state onto the declared public current readout.

    Callers must give a model only this return value. The host arguments are
    privileged. Giving n to the model would remove the information bottleneck.
    """
    validate_state(d, n)
    return {"d": d, "quotient": modular_labels(d)[n % d],
            "branch_multiplier": 3 if n % 4 == 1 else 1}


def observer_views(core: Core) -> dict[str, dict[int, Hashable]]:
    labels = modular_labels(core.d)
    q = {n: labels[n % core.d] for n in core.states}
    edges = {n: odd_step(n, core.d) for n in core.states}
    return {
        "quotient": q,
        "quotient_branch": {n: (q[n], 3 if edges[n][2] else 1)
                            for n in core.states},
        "quotient_branch_halvings": {
            n: (q[n], 3 if edges[n][2] else 1, edges[n][1])
            for n in core.states},
    }


def canonical_labels(states: tuple[int, ...], values: dict[int, Hashable]) -> dict[int, int]:
    """Name each cell by its least state, independent of input dict order."""
    representatives: dict[Hashable, int] = {}
    labels = {}
    for n in sorted(states):
        labels[n] = representatives.setdefault(values[n], n)
    return labels


@dataclass(frozen=True)
class Refinement:
    # partitions[h] encodes observations at times 0,...,h.
    partitions: tuple[dict[int, int], ...]
    transition: dict[int, int]

    @property
    def stable(self) -> dict[int, int]:
        return self.partitions[-1]


def refine_observer(core: Core, observation: dict[int, Hashable]) -> Refinement:
    """Coarsest forward-closed refinement of a finite deterministic readout."""
    if set(observation) != set(core.states):
        raise ValueError("observer must cover exactly the finite core")
    labels = canonical_labels(core.states, observation)
    partitions = [labels]
    for _ in range(len(core.states)):
        refined = canonical_labels(core.states, {
            n: (labels[n], labels[core.successors[n]]) for n in core.states})
        if refined == labels:
            transition = {}
            for n in core.states:
                source, target = labels[n], labels[core.successors[n]]
                if transition.setdefault(source, target) != target:
                    raise AssertionError("stable partition failed closure")
            return Refinement(tuple(partitions), transition)
        if len(set(refined.values())) <= len(set(labels.values())):
            raise AssertionError("strict refinement did not increase cell count")
        labels = refined
        partitions.append(labels)
    raise AssertionError("finite observer refinement exceeded its state bound")


def matching_pairs(labels: dict[int, Hashable]) -> int:
    """Unordered distinct pairs in the same cell."""
    return sum(size * (size - 1) // 2 for size in Counter(labels.values()).values())


def ceiling(labels: dict[int, Hashable], targets: dict[int, Hashable]) -> int:
    """Correct count of a Bayes decoder under uniform finite-core sampling."""
    groups: dict[Hashable, Counter] = defaultdict(Counter)
    for n, cell in labels.items():
        groups[cell][targets[n]] += 1
    return sum(max(counts.values()) for counts in groups.values())


def split_pairs(labels: dict[int, Hashable], targets: dict[int, Hashable]) -> int:
    return matching_pairs(labels) - matching_pairs({
        n: (labels[n], targets[n]) for n in labels})


def ordered_prefix(n: int, d: int, length: int) -> tuple[tuple[int, int], ...]:
    validate_state(d, n)
    if type(length) is not int or not 0 <= length <= 64:
        raise ValueError("length must be an integer in [0,64]")
    result = []
    for _ in range(length):
        n, a, triple = odd_step(n, d)
        result.append((3 if triple else 1, a))
    return tuple(result)


def decode_prefix(d: int, prefix: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """Return the seed residue and modulus carried by an observed (c,a) word.

    This is an additional branch-and-halving channel. It cannot reconstruct a
    seed from the conserved quotient's constant trace.
    """
    validate_parameter(d, quotient=True)
    validate_bound(d)
    if len(prefix) > 64:
        raise ValueError("prefix exceeds the 64-step audit budget")
    A = u = B = 0
    for c, a in prefix:
        if type(c) is not int or c not in (1, 3):
            raise ValueError("branch multiplier must be integer 1 or 3")
        if type(a) is not int or not 2 <= a <= 64:
            raise ValueError("halving count must be an integer in [2,64]")
        B = c * B + (1 << A)
        A += a
        u += c == 3
    modulus = 1 << (A + 1)
    return (((1 << A) - d * B) * pow(3 ** u, -1, modulus)) % modulus, modulus


def audit_root(d: int) -> dict:
    validate_bound(d)
    if d > AUDIT_MAX_PARAMETER:
        raise ValueError("five-step decoding audit is scoped to d <= 1025")
    core = analyze(d)
    views = observer_views(core)
    result = {"d": d, "states": len(core.states), "cycles": len(core.cycles),
              "observers": {}}
    for mode, observation in views.items():
        refinement = refine_observer(core, observation)
        next_observation = {n: observation[core.successors[n]] for n in core.states}
        initial = refinement.partitions[0]
        stable = refinement.stable
        record = {
            "initial_classes": len(set(initial.values())),
            "initial_matching_pairs": matching_pairs(initial),
            "next_visible_split_pairs": split_pairs(initial, next_observation),
            "ever_visible_split_pairs": matching_pairs(initial) - matching_pairs(stable),
            "current_to_next_correct": ceiling(initial, next_observation),
            "initial_basin_correct": ceiling(initial, core.basin),
            "stable_classes": len(refinement.transition),
            "stable_horizon": len(refinement.partitions) - 1,
            "stable_matching_pairs": matching_pairs(stable),
            "stable_basin_correct": ceiling(stable, core.basin),
            "stable_distinct_basin_pairs": split_pairs(stable, core.basin),
            "stable_partition_sha256": digest(json_bytes(stable)),
            "stable_transition_sha256": digest(json_bytes(refinement.transition)),
            "refinement_curve": [
                {"horizon": h, "classes": len(set(partition.values())),
                 "basin_correct": ceiling(partition, core.basin)}
                for h, partition in enumerate(refinement.partitions)],
        }
        if mode == "quotient" and (
                record["stable_horizon"] != 0 or record["next_visible_split_pairs"] != 0):
            raise AssertionError("conserved quotient failed its exact closure")
        result["observers"][mode] = record
    decoded = 0
    for n in core.states:
        residue, modulus = decode_prefix(d, ordered_prefix(n, d, 5))
        if modulus <= d - 1 or n % modulus != residue or residue != n:
            raise AssertionError("five observed branch/halving steps failed seed decoding")
        decoded += 1
    result["five_step_extra_channel_decoded"] = decoded
    return result


def exact_rank(rows: list[list[int]]) -> int:
    """Small-fixture rational row rank, with no floating-point cutoff."""
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("ragged matrix")
    matrix = [[Fraction(value) for value in row] for row in rows]
    rank = 0
    for column in range(width):
        pivot = next((i for i in range(rank, len(matrix)) if matrix[i][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        value = matrix[rank][column]
        matrix[rank] = [entry / value for entry in matrix[rank]]
        for i in range(rank + 1, len(matrix)):
            value = matrix[i][column]
            if value:
                matrix[i] = [a - value * b for a, b in zip(matrix[i], matrix[rank])]
        rank += 1
        if rank == min(len(matrix), width):
            break
    return rank


def apply_transition(core: Core, vector: dict[int, int]) -> dict[int, int]:
    """Matrix-free W e_n=e_F(n) in the declared one-hot embedding."""
    output: dict[int, int] = defaultdict(int)
    for n, value in vector.items():
        output[core.successors[n]] += value
    return {n: value for n, value in output.items() if value}


def apply_readout(observation: dict[int, Hashable], vector: dict[int, int]) -> dict:
    """Matrix-free categorical head O e_n=e_r(n)."""
    output: dict[Hashable, int] = defaultdict(int)
    for n, value in vector.items():
        output[observation[n]] += value
    return {label: value for label, value in output.items() if value}


def donor_fixture(core: Core, observation: dict[int, Hashable], left: int, right: int,
                  horizon: int = 4) -> dict:
    vector = Counter({right: 1})
    vector[left] -= 1
    delta = {n: value for n, value in vector.items() if value}
    future_squared = []
    for _ in range(horizon + 1):
        readout = apply_readout(observation, delta)
        future_squared.append(sum(value * value for value in readout.values()))
        delta = apply_transition(core, delta)
    return {
        "recipient": left, "donor": right,
        "recipient_state_squared_norm": 1, "donor_state_squared_norm": 1,
        "physical_squared_dose": 0 if left == right else 2,
        "readout_squared_difference_by_horizon": future_squared,
    }


def d13_fixture() -> dict:
    core = analyze(13)
    views = observer_views(core)
    observer = views["quotient_branch"]
    refinement = refine_observer(core, observer)
    head_labels = sorted(set(observer.values()))
    states_at_horizon = dict(zip(core.states, core.states))
    rows = []
    ranks = []
    for _ in range(len(core.states)):
        rows.extend([[int(observer[states_at_horizon[n]] == label) for n in core.states]
                     for label in head_labels])
        ranks.append(exact_rank(rows))
        states_at_horizon = {n: core.successors[s] for n, s in states_at_horizon.items()}
    kernel = {1: 1, 3: -1, 7: 1, 9: -1}
    if apply_readout(observer, kernel) or apply_transition(core, kernel):
        raise AssertionError("declared unobservable signed mixture failed")
    return {
        "d": 13, "states": list(core.states),
        "successors": core.successors,
        "public_pair": [public_observation(13, 1), public_observation(13, 5)],
        "basin_pair": [core.basin[1], core.basin[5]],
        "positive_donor": donor_fixture(core, observer, 1, 5),
        "delayed_donor": donor_fixture(core, observer, 1, 9),
        "identity_control": donor_fixture(core, observer, 1, 1),
        "quotient_only_null": donor_fixture(core, views["quotient"], 1, 5),
        "partition_cells_by_horizon": [
            [[n for n in core.states if labels[n] == cell]
             for cell in sorted(set(labels.values()))] for labels in refinement.partitions],
        "one_hot_stacked_observer_ranks": ranks,
        "one_hot_ambient_dimension": len(core.states),
        "signed_unobservable_kernel_vector": kernel,
        "kernel_vector_is_an_admissible_one_hot_state_difference": False,
    }


def control_fixtures() -> dict:
    core = analyze(73)
    observer = observer_views(core)["quotient_branch"]
    stable = refine_observer(core, observer).stable
    if not (observer[1] == observer[9] == observer[37]
            and stable[1] == stable[37] and stable[1] != stable[9]
            and core.basin[1] == core.basin[9] == core.basin[37]):
        raise AssertionError("d73 equal-dose matched control contract failed")
    d73 = {
        "d": 73,
        "public_observations": {n: public_observation(73, n) for n in (1, 9, 37)},
        "basins": {n: core.basin[n] for n in (1, 9, 37)},
        "effect": donor_fixture(core, observer, 1, 9),
        "same_future_sham": donor_fixture(core, observer, 1, 37),
        "same_future_sham_certified_by_stable_partition": stable[1] == stable[37],
    }
    core = analyze(29)
    observer = observer_views(core)["quotient_branch"]
    stable = refine_observer(core, observer).stable
    if not (stable[5] == stable[9] and core.basin[5] != core.basin[9]):
        raise AssertionError("d29 observer-equivalent distinct basins fixture failed")
    d29 = {
        "d": 29, "states": [5, 9], "basins": [core.basin[5], core.basin[9]],
        "same_infinite_observer_trace": stable[5] == stable[9],
        "quotient_branch_cells": [list(observer[5]), list(observer[9])],
        "cycles": [list(cycle["nodes"]) for cycle in core.cycles
                   if cycle["minimum"] in (core.basin[5], core.basin[9])],
    }
    return {"d73_matched_effect_and_sham": d73,
            "d29_same_future_observation_distinct_basins": d29}


def generate_receipt(max_parameter: int = 1025) -> dict:
    validate_bound(max_parameter)
    if max_parameter > AUDIT_MAX_PARAMETER:
        raise ValueError("five-step decoding audit is scoped to max_parameter <= 1025")
    core_bytes = (ROOT / "collatz_core.py").read_bytes()
    if digest(core_bytes) != CORE_SHA256:
        raise ValueError("canonical MathCan core hash mismatch")
    parameters = [d for d in range(1, max_parameter + 1, 4) if gcd(d, 6) == 1]
    roots = [audit_root(d) for d in parameters]
    aggregate = {"roots": len(roots), "states": sum(r["states"] for r in roots),
                 "cycles": sum(r["cycles"] for r in roots), "observers": {}}
    for mode in MODES:
        records = [r["observers"][mode] for r in roots]
        keys = ("initial_classes", "initial_matching_pairs", "next_visible_split_pairs",
                "ever_visible_split_pairs", "current_to_next_correct", "initial_basin_correct",
                "stable_classes", "stable_matching_pairs", "stable_basin_correct",
                "stable_distinct_basin_pairs")
        aggregate["observers"][mode] = {
            **{key: sum(record[key] for record in records) for key in keys},
            "roots_with_next_visible_split": sum(r["next_visible_split_pairs"] > 0 for r in records),
            "roots_with_ever_visible_split": sum(r["ever_visible_split_pairs"] > 0 for r in records),
            "roots_with_stable_basin_ambiguity": sum(r["stable_distinct_basin_pairs"] > 0 for r in records),
            "maximum_stable_horizon": max(r["stable_horizon"] for r in records),
        }
    aggregate["five_step_extra_channel_decoded"] = sum(
        r["five_step_extra_channel_decoded"] for r in roots)
    return {
        "schema": "BPD-COL-01/exact-finite-observability/v1",
        "evidence": "exhaustive finite calibration; no trained-model or population claim",
        "source": {
            "repository": "https://github.com/patrickdugan/MathCan",
            "commit": SOURCE_COMMIT,
            "path": "research/collatz_resonance/collatz_core.py",
            "core_sha256": CORE_SHA256,
            "audit_sha256": digest(Path(__file__).read_bytes()),
        },
        "specification": {
            "max_parameter": max_parameter,
            "parameters": "positive d <= max_parameter with d = 1 mod 4 and gcd(d,6) = 1",
            "state_domain": "all odd n from 1 through d, inclusive",
            "pair_measure": "unordered distinct state pairs within each parameter",
            "accuracy_measure": "uniform odd-core states within each root; aggregate pools states",
            "horizon": "h includes current observation and h future observations",
            "observer_refinement": "offline exact future-trace oracle",
            "model_view": "d, quotient, branch_multiplier; hidden n and basin omitted",
            "extra_channel": "five ordered (branch multiplier, halving count) observations",
        },
        "aggregate": aggregate,
        "d13_fixture": d13_fixture(),
        "control_fixtures": control_fixtures(),
        "roots": roots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-parameter", type=int, default=1025)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, default=None)
    destination.add_argument("--check", type=Path)
    args = parser.parse_args()
    receipt = generate_receipt(args.max_parameter)
    serialized = json_bytes(receipt)
    if args.check:
        if args.check.read_bytes() != serialized:
            raise SystemExit("receipt replay mismatch")
        print("Exact receipt replay passed.")
    else:
        output = args.output or DEFAULT_RECEIPT
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(serialized)
        print(f"Wrote {output}")
    print(json.dumps(receipt["aggregate"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
