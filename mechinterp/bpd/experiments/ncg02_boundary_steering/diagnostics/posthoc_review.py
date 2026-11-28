"""Replay the exposed-selection-pair NCG-02 geometry diagnostic.

Example:
  python ncg_bpd_posthoc_review.py --root ./ncg02 --output ./review.json

The saved models are read without changing their weights. The 96 source pairs
per slot were already exposed during projector selection. All measurements are
post hoc and descriptive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(root: Path) -> dict:
    root = root.resolve()
    sys.path.insert(0, str(root))
    import pilot

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = root / "results" / "development"
    manifest = json.loads((results / "manifest_before_run.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if sha256(root / name) != expected:
            raise RuntimeError(f"Pinned training source differs: {name}")

    rows = []
    files = {}
    for path in sorted(results.glob("root_*.json")):
        receipt = json.loads(path.read_text())
        seed = receipt["seed"]
        groups = torch.tensor(receipt["source_selection"]["selected_groups"], dtype=torch.long)
        saved = torch.load(results / f"root_{seed}_A_task.pt", weights_only=True)
        frozen = saved["projectors"]
        q0 = [torch.linalg.eigh(p).eigenvectors[:, -4:] for p in frozen]
        row = {
            "seed": seed,
            "selected_group_indices": receipt["source_selection"]["selected_indices"],
            "selected_components": groups.tolist(),
            "shared_component_count": len(set(groups[0].tolist()) & set(groups[1].tolist())),
            "frozen_cross_slot_overlap": float(torch.trace(frozen[0] @ frozen[1]) / 4),
            "phases": {},
        }
        for phase in ["warmup", *pilot.ARMS]:
            checkpoint_path = results / f"root_{seed}_{phase}.pt"
            obj = torch.load(checkpoint_path, weights_only=True)
            state = obj if phase == "warmup" else obj["state_dict"]
            files[str(checkpoint_path.relative_to(root))] = sha256(checkpoint_path)
            model = pilot.ComponentStateMemory(width=16, input_dim=8, values=6, components=16)
            model.load_state_dict(state)
            model.eval()
            current = pilot.projectors(model, groups)
            values = []
            for j in range(2):
                eig = torch.linalg.eigvalsh(q0[j].T @ current[j] @ q0[j]).clamp(0, 1)
                values.append({
                    "mean_squared_principal_cosine": float(torch.trace(frozen[j] @ current[j]) / 4),
                    "minimum_squared_principal_cosine": float(eig.min()),
                    "squared_principal_cosines": eig.tolist(),
                })
            detail = {
                "frozen_to_current_spans": values,
                "current_cross_slot_overlap": float(torch.trace(current[0] @ current[1]) / 4),
                "source_pairs_by_slot": [],
            }
            with torch.no_grad():
                for slot in range(2):
                    batch = pilot.batch(seed, 30000 + slot, 96, 2, slot)
                    recipient = model.rollout(batch["x"])
                    donor = model.rollout(batch["donor_x"])
                    recipient_state = recipient["states"][:, 1]
                    donor_state = donor["states"][:, 1]
                    difference = donor_state - recipient_state
                    denominator = difference.square().sum(1)
                    if bool((denominator <= 0).any()):
                        raise RuntimeError("A donor pair has zero hidden-state difference")
                    slot_result = {
                        "slot": slot,
                        "minimum_donor_difference_norm": float(denominator.sqrt().min()),
                        "by_projector": {},
                    }
                    projectors = [
                        ("frozen_own", frozen[slot]),
                        ("frozen_other", frozen[1 - slot]),
                        ("current_own", current[slot]),
                        ("current_other", current[1 - slot]),
                    ]
                    for label, projector in projectors:
                        captured = (difference @ projector).square().sum(1) / denominator
                        delta, _ = pilot.state_patch(
                            recipient_state, donor_state, projector[None],
                            torch.zeros(96, dtype=torch.long), .75,
                        )
                        logits = model.rollout(batch["x"], patch=delta, patch_step=2)["logits"]
                        predicted = logits >= 0
                        desired = batch["donor_y"].bool()
                        edited = slice(slot * 3, (slot + 1) * 3)
                        other = slice((1 - slot) * 3, (2 - slot) * 3)
                        slot_result["by_projector"][label] = {
                            "mean_captured_difference_energy": float(captured.mean()),
                            "donor_target_BCE": float(F.binary_cross_entropy_with_logits(logits, batch["donor_y"])),
                            "donor_exact_accuracy": float((predicted == desired).all(1).float().mean()),
                            "edited_bit_accuracy": float((predicted[:, edited] == desired[:, edited]).float().mean()),
                            "unaffected_bit_accuracy": float((predicted[:, other] == desired[:, other]).float().mean()),
                        }
                    detail["source_pairs_by_slot"].append(slot_result)
            row["phases"][phase] = detail
        rows.append(row)
    if len(rows) != 8:
        raise RuntimeError(f"Expected the eight completed development roots, found {len(rows)}")

    summary = {}
    for phase in ["warmup", *pilot.ARMS]:
        details = [r["phases"][phase] for r in rows]
        overlaps = [v["mean_squared_principal_cosine"]
                    for d in details for v in d["frozen_to_current_spans"]]
        minima = [v["minimum_squared_principal_cosine"]
                  for d in details for v in d["frozen_to_current_spans"]]
        slots = [s for d in details for s in d["source_pairs_by_slot"]]

        def difference(kind: str, metric: str, own_first: bool = True) -> float:
            total = sum(s["by_projector"][kind + "_own"][metric]
                        - s["by_projector"][kind + "_other"][metric] for s in slots)
            return (1 if own_first else -1) * total / len(slots)

        summary[phase] = {
            "frozen_to_current_overlap_min_mean_max": [min(overlaps), sum(overlaps) / len(overlaps), max(overlaps)],
            "minimum_squared_principal_cosine_over_all_slots": min(minima),
            "mean_current_cross_slot_overlap": sum(d["current_cross_slot_overlap"] for d in details) / len(details),
            "frozen_own_minus_other_capture": difference("frozen", "mean_captured_difference_energy"),
            "current_own_minus_other_capture": difference("current", "mean_captured_difference_energy"),
            "frozen_own_BCE_advantage_over_other": difference("frozen", "donor_target_BCE", False),
            "current_own_BCE_advantage_over_other": difference("current", "donor_target_BCE", False),
            "frozen_own_donor_exact": sum(s["by_projector"]["frozen_own"]["donor_exact_accuracy"] for s in slots) / len(slots),
            "frozen_other_donor_exact": sum(s["by_projector"]["frozen_other"]["donor_exact_accuracy"] for s in slots) / len(slots),
        }
    return {
        "scope": "POST_HOC_DIAGNOSTIC",
        "analysis_selection_pairs_exposed": True,
        "purpose": "Review frozen-projector geometry and cycle-bit slot selectivity without training or checkpoint selection",
        "source_cases": "The 96 training-stream pairs per slot used by discovery: streams 30000 and 30001, delay 2. Selection data reused; these are descriptive measurements, not independent validation.",
        "selection_note": "At warmup, selecting the minimum donor-target BCE among candidates guarantees a nonnegative own-projector BCE advantage over another candidate on these same pairs.",
        "metric_definitions": {
            "span_overlap": "trace(P_frozen P_current)/4: mean squared principal cosine",
            "cross_slot_overlap": "trace(P_slot0 P_slot1)/4",
            "captured_difference_energy": "mean ||(h_donor-h_recipient)P||^2 / ||h_donor-h_recipient||^2",
            "BCE_advantage": "loss with other-slot projector minus loss with own-slot projector; positive favors own",
        },
        "group_collision_roots": [r["seed"] for r in rows if r["shared_component_count"]],
        "frozen_cross_slot_overlap_min_mean_max": [
            min(r["frozen_cross_slot_overlap"] for r in rows),
            sum(r["frozen_cross_slot_overlap"] for r in rows) / len(rows),
            max(r["frozen_cross_slot_overlap"] for r in rows),
        ],
        "summary": summary,
        "roots": rows,
        "checkpoint_sha256": files,
        "training_source_sha256": manifest["source_sha256"],
        "diagnostic_source_sha256": sha256(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged NCG-02 directory")
    parser.add_argument("--output", type=Path, required=True, help="Separate JSON output path")
    args = parser.parse_args()
    result = review(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "scope": result["scope"],
                      "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
