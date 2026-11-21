"""Independent stdlib audit for SHA-pinned pretrained TRM root/arm JSON receipts.

Verifies task outcomes, root pairing, physical dose and zero-patch parity.
It is a receipt audit, NOT a native model re-execution or mechanism confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ARMS = {"clean", "zero", "head_direction", "random_direction"}
SHA = "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"


def valid_sudoku(question: str, candidate: str) -> bool:
    if len(question) != 81 or len(candidate) != 81:
        return False
    if any(ch not in "123456789." for ch in question):
        return False
    if any(ch not in "123456789" for ch in candidate):
        return False
    if any(q != "." and candidate[i] != q for i, q in enumerate(question)):
        return False
    groups = [[r * 9 + c for c in range(9)] for r in range(9)]
    groups += [[r * 9 + c for r in range(9)] for c in range(9)]
    groups += [
        [(r + dr) * 9 + c + dc for dr in range(3) for dc in range(3)]
        for r in (0, 3, 6) for c in (0, 3, 6)
    ]
    return all({candidate[i] for i in group} == set("123456789") for group in groups)


def _read(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 10_000_000:
        raise ValueError("receipt exceeds ten MB")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


def audit(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one matrix required")
    summaries = []
    cohort: dict[str, str] | None = None
    for path in paths:
        report = _read(path)
        if (
            report.get("schema_version") != "bpd.native_trm_boundary_pilot.v1"
            or report.get("evidence_class") != "native_pretrained_development"
            or report.get("model") != "gaoxin492"
            or report.get("model_sha256") != SHA
            or report.get("max_clean_parity_difference") != 0
            or set(report.get("arms", [])) != ARMS
        ):
            raise ValueError("wrong checkpoint, schema or model-parity receipt")
        if report.get("outer_steps") not in {1, 2, 3}:
            raise ValueError("invalid outer depth")
        groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in report["rows"]:
            arm, root = row.get("arm"), row.get("root_id")
            if arm not in ARMS or not isinstance(root, str) or arm in groups[root]:
                raise ValueError("unknown or repeated arm/root")
            if (
                row.get("model_sha256") != SHA or row.get("model_id") != "gaoxin492"
                or row.get("experiment_split") != "development"
            ):
                raise ValueError("model/split mismatch")
            q, target, prediction = row["question"], row["clean_exact_solution"], row["predicted"]
            if not valid_sudoku(q, target):
                raise ValueError("invalid planted puzzle")
            if hashlib.sha256(target.encode()).hexdigest() != row.get("target_sha256"):
                raise ValueError("target hash incorrect")
            if row["selected_cell"] != q.index("."):
                raise ValueError("intervention cell differs from first blank")
            if (
                row["cell_correct"] != sum(a == b for a, b in zip(prediction, target))
                or row["board_exact"] != (prediction == target)
                or row["board_valid"] != valid_sudoku(q, prediction)
                or row["selected_cell_correct"]
                != (prediction[row["selected_cell"]] == target[row["selected_cell"]])
            ):
                raise ValueError("case correctness does not replay")
            for field in ("nominal_raw_l2", "actual_raw_l2", "pre_state_norm_l2"):
                if not isinstance(row[field], (int, float)) or not math.isfinite(row[field]):
                    raise ValueError("invalid physical measurement")
            expected = 0 if arm in {"clean", "zero"} else (
                row["pre_state_norm_l2"] * report["relative_dose"]
            )
            if (
                abs(row["nominal_raw_l2"] - expected) > 0.002
                or abs(row["actual_raw_l2"] - expected) > 0.002
            ):
                raise ValueError("physical intervention dose mismatch")
            groups[root][arm] = row
        if len(groups) != report["roots"] or any(set(g) != ARMS for g in groups.values()):
            raise ValueError("incomplete root/arm matrix")
        current = {}
        for root, rows in groups.items():
            clean, zero = rows["clean"], rows["zero"]
            if (
                clean["predicted"] != zero["predicted"]
                or clean["final_selected_cell_logits"] != zero["final_selected_cell_logits"]
                or zero["mean_abs_logit_change"] != 0
                or len({v["question"] for v in rows.values()}) != 1
            ):
                raise ValueError("zero/control parity or root pairing failed")
            current[root] = clean["question"]
        if report["roots"] > 1:
            if cohort is None:
                cohort = current
            elif cohort != current:
                raise ValueError("cohort changed across pass/patch conditions")
        metrics = {}
        for arm in sorted(ARMS):
            selected = [rows[arm] for rows in groups.values()]
            metrics[arm] = {
                "board_exact": sum(r["board_exact"] for r in selected),
                "mean_cell_correct": sum(r["cell_correct"] for r in selected) / len(selected),
                "mean_abs_logit_change": sum(r["mean_abs_logit_change"] for r in selected) / len(selected),
            }
        summaries.append({
            "file": path.name, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "roots": report["roots"], "holes": report.get("holes"),
            "outer_steps": report["outer_steps"],
            "patch_h_cycle": report.get("patch_h_cycle", 1),
            "relative_dose": report["relative_dose"], "metrics": metrics,
        })
    return {
        "schema_version": "bpd.native_trm_receipt_audit.v1",
        "evidence_class": "descriptive_native_development_receipt_replay",
        "checkpoint_sha256": SHA, "matrices": summaries, "verified": True,
        "limitations": (
            "Independently checks output correctness, dose and pairing. "
            "Does not reexecute pretrained model from weights or establish Inkling routing."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    report = audit(args.paths)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "matrices": len(report["matrices"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
