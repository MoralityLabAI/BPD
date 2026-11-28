"""Aggregate exposed RP03 receipts without fitting models or selecting treatments.

This post-development helper leaves the frozen experiment and its primary
statistics unchanged. It checks bit arithmetic before reporting descriptive
geometry, transport, slot-group overlap and resource measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOTS = list(range(27101, 27109))
ARMS = ("A_task", "B_causal_fixed", "C_rotated_fixed", "D_causal_live")
CONDITIONS = ("causal", "causal_live", "causal_sham", "hybrid_state_oracle")
STEPS = (0, 80, 160, 240)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def accuracy(record):
    labels, predictions, correct = (record[k] for k in ("labels", "predictions", "correct"))
    require(len(labels) == len(predictions) == len(correct), "Bit vector lengths differ")
    require(all(x in (0, 1) for x in correct), "Correctness vector must be binary")
    require(correct == [int(p == y) for p, y in zip(predictions, labels)], "Prediction bits differ")
    require(record["accuracy"] == ratio(sum(correct), len(correct)), "Saved accuracy differs")
    return {"correct": sum(correct), "n": len(correct)}


def readout(record):
    """Use each arm's unedited receiver as the recovery/corruption baseline."""
    ordinary = record["ordinary_on_causal"]
    accuracy(ordinary)
    target = record["causal"]["targeted"]
    n = len(target)
    require(n == len(ordinary["correct"]), "Causal baseline length differs")
    target_n = sum(target)
    unaffected_n = n - target_n
    original_correct = sum(c for c, t in zip(ordinary["correct"], target) if not t)
    conditions = {}
    for name in CONDITIONS:
        row = record[name]
        accuracy(row)
        require(len(row["correct"]) == n, "Intervention case count differs")
        require(row["targeted"] == target, "Intervention target masks differ")
        require(row["labels"] == record["causal"]["labels"], "Hybrid labels differ")
        require(all((h != r) == t for h, r, t in zip(row["labels"], ordinary["labels"], target)),
                "Target mask differs from changed semantic labels")
        recovered = sum(not t and not b and c for t, b, c in
                        zip(target, ordinary["correct"], row["correct"]))
        corrupted = sum(not t and b and not c for t, b, c in
                        zip(target, ordinary["correct"], row["correct"]))
        unaffected_correct = sum(c for c, t in zip(row["correct"], target) if not t)
        require(unaffected_correct == original_correct + recovered - corrupted,
                "Recovery/corruption identity failed")
        conditions[name] = {
            "targeted_correct": sum(c for c, t in zip(row["correct"], target) if t),
            "targeted_n": target_n,
            "untargeted_correct": unaffected_correct,
            "untargeted_n": unaffected_n,
            "untargeted_original_correct": original_correct,
            "untargeted_original_wrong": unaffected_n - original_correct,
            "untargeted_recoveries": recovered,
            "untargeted_corruptions": corrupted,
        }
        if name in record["physical_audit"]:
            physical = record["physical_audit"][name]
            require(physical["untargeted_original_recoveries"] == recovered,
                    "Saved recovery receipt differs")
            require(physical["untargeted_original_corruptions"] == corrupted,
                    "Saved corruption receipt differs")
    geometry = record["geometry"]
    result = {
        "ordinary_on_causal": accuracy(ordinary),
        "conditions": conditions,
        "geometry": {
            "projector_chordal_by_slot": geometry["projector_chordal_by_slot"],
            "fixed_live_delta_cosine_mean": geometry["fixed_live_delta_cosine_mean"],
            "fixed_live_delta_distance_mean": geometry["fixed_live_delta_distance_mean"],
            "fixed_invalid": geometry["fixed_invalid"],
            "live_invalid": geometry["live_invalid"],
        },
        "oracle": {
            "native_hybrid_logits_max_error": record["hybrid_state_oracle"]["native_hybrid_logits_max_error"],
            "physical_l2_mean": record["hybrid_state_oracle"]["physical_l2_mean"],
        },
    }
    if "binding" in record:
        result["binding"] = accuracy(record["binding"])
        result["ordinary"] = {name: accuracy(record["eval"][name]) for name in ("iid", "long")}
    return result


def pooled_count(rows):
    correct, n = sum(r["correct"] for r in rows), sum(r["n"] for r in rows)
    return {"correct": correct, "n": n, "accuracy": ratio(correct, n),
            "macro_accuracy": mean(r["correct"] / r["n"] for r in rows)}


def pool(rows):
    result = {"n_roots": len(rows), "conditions": {},
              "ordinary_on_causal": pooled_count([r["ordinary_on_causal"] for r in rows])}
    for name in CONDITIONS:
        conditions = [r["conditions"][name] for r in rows]
        summed = {key: sum(c[key] for c in conditions) for key in conditions[0]}
        summed.update({
            "targeted_accuracy": ratio(summed["targeted_correct"], summed["targeted_n"]),
            "untargeted_accuracy": ratio(summed["untargeted_correct"], summed["untargeted_n"]),
            "untargeted_corruption_fraction_of_original_correct": ratio(
                summed["untargeted_corruptions"], summed["untargeted_original_correct"]),
            "untargeted_recovery_fraction_of_original_wrong": ratio(
                summed["untargeted_recoveries"], summed["untargeted_original_wrong"]),
        })
        result["conditions"][name] = summed
    drifts = [d for r in rows for d in r["geometry"]["projector_chordal_by_slot"]]
    cosines = [r["geometry"]["fixed_live_delta_cosine_mean"] for r in rows]
    result["geometry"] = {
        "projector_chordal_mean_across_roots_and_slots": mean(drifts),
        "projector_chordal_max": max(drifts),
        "projector_chordal_min": min(drifts),
        "fixed_live_delta_cosine_macro_mean": mean(cosines),
        "fixed_live_delta_cosine_root_mean_min": min(cosines),
        "fixed_live_delta_cosine_root_mean_max": max(cosines),
        "fixed_live_delta_distance_macro_mean": mean(
            r["geometry"]["fixed_live_delta_distance_mean"] for r in rows),
        "fixed_invalid": sum(r["geometry"]["fixed_invalid"] for r in rows),
        "live_invalid": sum(r["geometry"]["live_invalid"] for r in rows),
    }
    result["oracle"] = {
        "native_hybrid_logits_max_error": max(r["oracle"]["native_hybrid_logits_max_error"] for r in rows),
        "physical_l2_macro_mean": mean(r["oracle"]["physical_l2_mean"] for r in rows),
    }
    if "binding" in rows[0]:
        result["binding"] = pooled_count([r["binding"] for r in rows])
        result["ordinary"] = {name: pooled_count([r["ordinary"][name] for r in rows])
                              for name in ("iid", "long")}
    return result


def summarize(folder):
    paths = [folder / f"root_{root}" / "result.json" for root in ROOTS]
    completion = json.loads((folder / "completion.json").read_text())
    manifest = json.loads((folder / "run_manifest.json").read_text())
    primary = json.loads((folder / "summary.json").read_text())
    require(completion["completed"] and completion["roots"] == ROOTS, "Incomplete root roster")
    require(manifest["roots"] == primary["roots"] == ROOTS, "Manifest and summary rosters differ")
    require(primary["protocol_root_count_complete"] and primary["protocol_root_roster_complete"],
            "Primary summary does not cover the frozen roster")
    inputs = paths + [folder / name for name in ("completion.json", "run_manifest.json", "summary.json")]
    input_hashes = {str(p.relative_to(REPO)): sha256(p) for p in inputs}
    per_root = []
    for root, path in zip(ROOTS, paths):
        require(sha256(path) == completion["result_sha256"][str(root)], "Completion hash differs")
        source = json.loads(path.read_text())
        require(source["root"] == root, "Root identifier differs")
        groups = source["attribution"]["groups"]
        group_counts = Counter(tuple(g) for g in groups)
        repeated_slot_pairs = [[a, b] for a in range(len(groups)) for b in range(a+1, len(groups))
                               if groups[a] == groups[b]]
        row = {"root": root, "selection": {
            "selected_candidate_by_slot": source["attribution"]["selected_candidate"],
            "groups_by_slot": groups,
            "unique_groups": len(group_counts),
            "duplicate_assignments_beyond_first": len(groups) - len(group_counts),
            "slots_in_repeated_groups": sum(n for n in group_counts.values() if n > 1),
            "same_group_slot_pairs": repeated_slot_pairs,
        }, "final": {}, "anchors": {}, "resources": {
            key: source[key] for key in ("elapsed_seconds", "process_cpu_seconds", "process_peak_rss_kib")}}
        for arm in ARMS:
            treatment = source["arms"][arm]
            require([a["step"] for a in treatment["anchors"]] == list(STEPS), "Anchor roster differs")
            row["final"][arm] = readout(treatment)
            row["anchors"][arm] = {str(a["step"]): readout(a) for a in treatment["anchors"]}
        per_root.append(row)
    final = {arm: pool([r["final"][arm] for r in per_root]) for arm in ARMS}
    anchors = {str(step): {arm: pool([r["anchors"][arm][str(step)] for r in per_root])
                          for arm in ARMS} for step in STEPS}
    for arm in ARMS:
        for endpoint in ("iid", "long"):
            require(final[arm]["ordinary"][endpoint]["correct"] == primary["arms"][arm][endpoint]["correct"],
                    "Primary ordinary numerator differs")
            require(final[arm]["ordinary"][endpoint]["n"] == primary["arms"][arm][endpoint]["n_cases"],
                    "Primary ordinary denominator differs")
        fixed = final[arm]["conditions"]["causal"]
        for prefix in ("targeted", "untargeted"):
            require(fixed[prefix+"_correct"] == primary["arms"][arm]["causal"][prefix]["correct"],
                    "Primary causal numerator differs")
            require(fixed[prefix+"_n"] == primary["arms"][arm]["causal"][prefix]["n_cases"],
                    "Primary causal denominator differs")
    peaks = [r["resources"]["process_peak_rss_kib"] for r in per_root]
    return {
        "schema_version": 1,
        "experiment": "BPD-RP-03",
        "evidence_class": "DEVELOPMENT_ONLY",
        "analysis_scope": "post-development descriptive aggregation of the original saved receipts",
        "independent_unit": "independently initialized model root with four paired training arms",
        "roots": ROOTS,
        "n_roots": len(ROOTS),
        "new_inferential_tests": 0,
        "source_sha256": {str(Path(__file__).resolve().relative_to(REPO)): sha256(Path(__file__))},
        "input_sha256": input_hashes,
        "definitions": {
            "final": "evaluation-partition inputs generated after training",
            "anchors": "fixed training-partition inputs at steps 0,80,160,240; repeated measurements within roots",
            "causal": "fixed warmup projector at every arm's current model state",
            "causal_live": "current V-column projector using the frozen selected indices",
            "causal_sham": "root-seeded signed permutation of the fixed-projector edit",
            "hybrid_state_oracle": "full hybrid-state replacement; privileged input and unmatched natural dose",
            "projector_chordal": "||P_current-P_warmup||_F/sqrt(2*group_size); group_size=4",
            "cosine": "cosine between realized fixed and live edits, averaged over each root's cases",
            "rates": "pooled counts; roots have equal case counts, while conditional corruption denominators differ",
            "group_overlap": "equal selected index sets; candidate groups are disjoint contiguous sets of four",
            "resources": "root elapsed and CPU times are incremental; Linux ru_maxrss is process lifetime high-water KiB",
        },
        "verification": {"completion_hashes_match": True, "saved_prediction_arithmetic_matches": True,
                         "physical_recovery_corruption_receipts_match": True,
                         "primary_summary_counts_match": True, "all_assigned_roots_retained": True},
        "selection": {
            "roots_with_repeated_groups": sum(r["selection"]["unique_groups"] < 4 for r in per_root),
            "duplicate_assignments_beyond_first": sum(r["selection"]["duplicate_assignments_beyond_first"] for r in per_root),
            "same_group_slot_pairs": sum(len(r["selection"]["same_group_slot_pairs"]) for r in per_root),
        },
        "resources": {
            "run_elapsed_seconds": completion["elapsed_seconds"],
            "sum_root_elapsed_seconds": sum(r["resources"]["elapsed_seconds"] for r in per_root),
            "sum_root_cpu_seconds": sum(r["resources"]["process_cpu_seconds"] for r in per_root),
            "process_peak_rss_kib": max(peaks),
            "process_peak_rss_mib": max(peaks)/1024,
            "process_peak_rss_bytes": max(peaks)*1024,
            "rss_aggregation": "maximum process lifetime high-water mark; peaks are not added across sequential roots",
        },
        "final": final,
        "anchors": anchors,
        "per_root": per_root,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=HERE / "results/development")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Compare with the saved aggregation without writing")
    args = parser.parse_args()
    folder = args.results.resolve()
    output = args.out or folder / "diagnostic_summary.json"
    result = summarize(folder)
    if args.check:
        require(json.loads(output.read_text()) == result, "Saved diagnostic aggregation differs")
    else:
        output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"out": str(output), "n_roots": result["n_roots"],
                      "checked_existing": args.check,
                      "verification": result["verification"], "selection": result["selection"],
                      "resources": result["resources"]}, indent=2))


if __name__ == "__main__":
    main()
