"""Independently replay RMP-INK-PC01 learned-gate rollout and state-patch receipts.

Source-only audit of an EXPLICIT permutation operator plus one trained gate.
This does not attest provider identity, native Inkling routing or external sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "rmp.inkling_positive_control.v1"
ARCH = "task_supplied_permutation_with_learned_gate_v1"
ARMS = {"clean", "zero", "active_coordinate", "random_direction", "random_coordinate"}
VERIFIER = "rmp.ink_pc01.exact_cycle.v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("oversized development stream")
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("nonobject row")
                rows.append(item)
    if not rows or len(rows) > 20000:
        raise ValueError("empty/oversized stream")
    return rows


def _source_world(row: dict[str, Any]) -> tuple[tuple[int, ...], int, int, int]:
    world = row.get("task_world")
    if not isinstance(world, dict):
        raise ValueError("task world is missing")
    permutation = world.get("permutation")
    if (
        not isinstance(permutation, list) or len(permutation) != 17
        or any(type(x) is not int for x in permutation)
        or sorted(permutation) != list(range(17))
    ):
        raise ValueError("invalid permutation")
    start, hops = world.get("start"), world.get("hops")
    if type(start) is not int or not 0 <= start < 17:
        raise ValueError("invalid start node")
    if type(hops) is not int or not 1 <= hops < 17 or hops != row.get("hops"):
        raise ValueError("invalid hop count")
    root_id, case_id = row.get("root_id"), row.get("case_id")
    if not isinstance(root_id, str) or not isinstance(case_id, str):
        raise ValueError("missing lineage IDs")
    if row.get("task_digest") != _digest({"root_id": root_id, "world": world}):
        raise ValueError("world binding digest mismatch")
    target = start
    for _ in range(hops):
        target = permutation[target]
    if row.get("target") != target:
        raise ValueError("target differs from independently replayed graph")
    return tuple(permutation), start, hops, target


def _next(state: list[float], permutation: tuple[int, ...], alpha: float) -> list[float]:
    message = [0.0] * len(state)
    for i, dest in enumerate(permutation):
        message[dest] = state[i]
    return [(1 - alpha) * x + alpha * y for x, y in zip(state, message, strict=True)]


def _walk_state(
    permutation: tuple[int, ...], start: int, steps: int, alpha: float,
    *, state: list[float] | None = None,
) -> list[float]:
    value = list(state) if state is not None else [float(i == start) for i in range(17)]
    for _ in range(steps):
        value = _next(value, permutation, alpha)
    return value


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def _vector(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 17:
        raise ValueError("raw state vector missing")
    if any(type(x) not in {float, int} or not math.isfinite(x) for x in value):
        raise ValueError("nonfinite raw state")
    return [float(x) for x in value]


def _near(a: float, b: float, *, eps: float = 1e-9) -> bool:
    return math.isclose(a, b, abs_tol=eps, rel_tol=eps)


def _same_state(a: list[float], b: list[float]) -> bool:
    return all(_near(x, y) for x, y in zip(a, b, strict=True))


def _shared(row: dict[str, Any], checkpoint_sha: str) -> None:
    if (
        row.get("schema_version") is not None or row.get("model_sha") != checkpoint_sha
        or row.get("split") != "development" or row.get("verifier_id") != VERIFIER
        or row.get("family") != "explicit_single_cycle_pointer_chase"
        or type(row.get("verified")) is not bool
        or type(row.get("predicted")) is not int
    ):
        raise ValueError("wrong model/split/verifier receipt")


def audit(directory: Path) -> dict[str, Any]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    if summary.get("schema_version") != SCHEMA:
        raise ValueError("wrong experiment schema")
    if summary.get("evidence_class") != "synthetic_trained_gate_development_positive_control":
        raise ValueError("unsupported evidence class")
    for name in ("checkpoint.json", "depth_matrix.jsonl", "interventions.jsonl"):
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if summary["files_sha256"].get(name) != actual:
            raise ValueError(f"source file digest mismatch: {name}")
    theta = checkpoint.get("theta")
    if type(theta) not in {int, float} or not math.isfinite(theta):
        raise ValueError("invalid checkpoint theta")
    alpha = 1 / (1 + math.exp(-theta))
    checkpoint_sha = _digest({"theta": theta, "arch": ARCH})
    if (
        checkpoint.get("sha256") != checkpoint_sha
        or summary.get("checkpoint_sha256") != checkpoint_sha
        or not _near(checkpoint.get("alpha"), alpha)
    ):
        raise ValueError("checkpoint identity mismatch")

    depths = [int(value) for value in summary["depth_accuracy"]]
    if len(set(depths)) != len(depths):
        raise ValueError("duplicate depths")
    depth_rows = _rows(directory / "depth_matrix.jsonl")
    causal_rows = _rows(directory / "interventions.jsonl")
    cases: dict[str, dict[str, Any]] = {}
    roots: dict[str, set[str]] = defaultdict(set)
    matrix: set[tuple[str, int]] = set()
    depth_correct: Counter[int] = Counter()
    for row in depth_rows:
        _shared(row, checkpoint_sha)
        perm, start, hops, target = _source_world(row)
        case, root, depth = row["case_id"], row["root_id"], row["depth"]
        if type(depth) is not int or depth not in depths:
            raise ValueError("unexpected depth")
        if (case, depth) in matrix:
            raise ValueError("duplicate case/depth")
        matrix.add((case, depth))
        old = cases.get(case)
        if old is not None and old["task_digest"] != row["task_digest"]:
            raise ValueError("case world changed across depths")
        cases[case] = row
        roots[root].add(case)
        result = _walk_state(perm, start, min(hops, depth), alpha)
        predicted = _argmax(result)
        expected = predicted == target
        if row["predicted"] != predicted or row["verified"] is not expected:
            raise ValueError("depth prediction fails independent recurrence replay")
        depth_correct[depth] += int(expected)
    if len(matrix) != len(cases) * len(depths):
        raise ValueError("incomplete depth matrix")
    if len(roots) != summary["source_roots"] or len(cases) != summary["cases"]:
        raise ValueError("root/case count mismatch")
    for depth in depths:
        receipt = summary["depth_accuracy"][str(depth)]
        if receipt != {"correct": depth_correct[depth], "total": len(cases)}:
            raise ValueError("depth summary disagrees with replayed cases")

    max_depth = max(depths)
    causal_seen: set[tuple[str, str]] = set()
    arm_correct: Counter[str] = Counter()
    eligible_cases = {case for case, row in cases.items() if row["hops"] >= 4}
    for row in causal_rows:
        _shared(row, checkpoint_sha)
        perm, start, hops, target = _source_world(row)
        case, arm = row["case_id"], row["arm"]
        if case not in eligible_cases or arm not in ARMS or row["depth"] != max_depth:
            raise ValueError("invalid intervention case/arm/depth")
        if row["task_digest"] != cases[case]["task_digest"]:
            raise ValueError("intervention changed task world")
        if (case, arm) in causal_seen:
            raise ValueError("duplicate case/arm")
        causal_seen.add((case, arm))
        if row.get("patch_visit") != 2:
            raise ValueError("unsupported patch visit")
        if arm == "clean":
            if (row.get("pre_patch_raw_state") is not None
                    or row.get("post_patch_raw_state") is not None):
                raise ValueError("clean arm has a patch")
            resulting = _walk_state(perm, start, min(hops, max_depth), alpha)
            displacement = 0.0
        else:
            before = _vector(row.get("pre_patch_raw_state"))
            after = _vector(row.get("post_patch_raw_state"))
            replay_before = _walk_state(perm, start, 2, alpha)
            if not _same_state(replay_before, before):
                raise ValueError("raw pre-patch state does not replay")
            if row.get("pre_patch_argmax") != _argmax(before):
                raise ValueError("raw patch argmax mismatch")
            delta = [y - x for x, y in zip(before, after, strict=True)]
            displacement = math.sqrt(sum(d * d for d in delta))
            if arm == "zero" and displacement > 1e-10:
                raise ValueError("zero patch changed the state")
            if arm == "active_coordinate":
                selected = _argmax(before)
                expected = [0.0] * 17
                expected[selected] -= summary["nominal_dose"] / math.sqrt(2)
                expected[(selected + 1) % 17] += summary["nominal_dose"] / math.sqrt(2)
                if not _same_state(delta, expected):
                    raise ValueError("active-coordinate intervention formula changed")
            resulting = _walk_state(
                perm, start, min(hops, max_depth) - 2, alpha, state=after,
            )
        expected_dose = 0.0 if arm in {"zero", "clean"} else summary["nominal_dose"]
        if (not _near(displacement, expected_dose)
                or not _near(row["realized_raw_l2_displacement"], displacement)
                or not _near(row["nominal_l2_dose"], expected_dose)):
            raise ValueError("realized/nominal intervention dose mismatch")
        predicted = _argmax(resulting)
        passed = predicted == target
        if row["predicted"] != predicted or row["verified"] is not passed:
            raise ValueError("intervened prediction fails independent recurrence replay")
        arm_correct[arm] += int(passed)
    if len(causal_seen) != len(eligible_cases) * len(ARMS):
        raise ValueError("incomplete intervention arm matrix")
    for arm in ARMS:
        if summary["intervention_accuracy"][arm] != {
            "correct": arm_correct[arm], "total": len(eligible_cases)
        }:
            raise ValueError("intervention summary disagrees with replayed cases")
    return {
        "schema_version": "bpd.inkling_control_audit.v1",
        "evidence_class": "independent_source_only_synthetic_replay",
        "checkpoint_sha256": checkpoint_sha,
        "roots": len(roots), "cases": len(cases),
        "depths": sorted(depths), "per_depth_correct": dict(sorted(depth_correct.items())),
        "per_arm_correct": dict(sorted(arm_correct.items())),
        "intervention_cases": len(eligible_cases),
        "receipt_replay_passed": True,
        "claim_boundary": (
            "Explicit graph operator and one trained mixing gate; full state replay "
            "only for this synthetic model. No native Inkling activation evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists")
    report = audit(args.input)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"roots": report["roots"], "replay": report["receipt_replay_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
