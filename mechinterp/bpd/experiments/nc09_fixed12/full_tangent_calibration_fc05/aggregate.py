"""FC05: audit original bytes before summarizing 59 independent model roots.

The FC02 bootstrap functions are imported unchanged. Their output is planning
evidence. The independent-root EvalMath review remains a separate decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from mechinterp.bpd.experiments.nc09_fixed12.calibration import evalmath_preflight
from mechinterp.bpd.experiments.nc09_fixed12.tangent_sham_calibration.aggregate import (
    METRICS, aggregate_root_mean_null, approximate_power, root_mean_bootstrap,
)
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_calibration_fc05.audit import (
    source_audit, task_audit,
)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
ROOTS = tuple(range(16000, 16059))
STUDY = "BPD-NC09-FC05"
PHASE = "FULL_TANGENT_CALIBRATION_DEVELOPMENT"
MANDATORY_ARMS = ("dp12", "flat3", "flat3_dose", "flat4")
PARENT_SOURCE_SHA = "8d1b88eea8518a83669f03f9126136ca3d56ac5bac5fe041e2e03fb12922d380"
PARENT_PROTOCOL_SHA = "dc9423f1fd9e308221c41ed2af83e7398a59dd0eb0beefa8394091e373156e02"
EVALMATH_COMMIT = "6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a"
EVALMATH_BLOB = "0ff062f76f5534d9f18425f986ae3e750397f50c"
FC02_AGGREGATE_BLOB = "5aab79dabe4548361a6545cf575f4c7ab8047f1f"
EXACT_ROOTWISE_CONFIDENCE = 1 - .95 ** 59


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_manifest() -> dict:
    """Recheck every imported/source file against the committed manifest."""
    path = ROOT / "SOURCE_MANIFEST.json"
    errors = []
    digest = None
    checked = 0
    try:
        digest = sha256(path)
        manifest = json.loads(path.read_text())
        files = manifest["files"]
        if not isinstance(files, dict) or not files:
            raise ValueError("Source manifest has no frozen files")
        for name, expected in files.items():
            candidate = (REPO / name).resolve()
            if not candidate.is_relative_to(REPO) or not isinstance(expected, str):
                errors.append("invalid_manifest_path_or_digest:" + str(name))
            elif not candidate.is_file() or sha256(candidate) != expected:
                errors.append("source_file_mismatch:" + name)
            checked += 1
        required = {
            str((ROOT / filename).relative_to(REPO))
            for filename in ("PROTOCOL.md", "experiment.py", "solver.py", "audit.py", "aggregate.py")
        }
        required.update({
            "mechinterp/bpd/experiments/nc09_fixed12/tangent_sham_calibration/aggregate.py",
            "mechinterp/bpd/experiments/nc09_fixed12/calibration/evalmath_preflight.py",
            str((ROOT / "prereview/FC04_HISTORICAL_TARGET_PREFLIGHT_INPUT.json").relative_to(REPO)),
        })
        missing = sorted(required - set(files))
        if missing:
            errors.append("required_source_files_not_frozen:" + ",".join(missing))
        frozen = REPO / "mechinterp/bpd/experiments/nc09_fixed12/tangent_sham_calibration/aggregate.py"
        data = frozen.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != FC02_AGGREGATE_BLOB:
            errors.append("original_FC02_aggregate_blob_mismatch")
    except Exception as error:
        errors.append(type(error).__name__ + ": " + str(error))
    return {"passed": not errors, "manifest_sha256": digest,
            "checked_files": checked, "errors": errors}


def _finite(value) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _bounded(value, lower: float, upper: float) -> bool:
    return _finite(value) and lower <= value <= upper


def _gauge_passed(gauge) -> bool:
    return isinstance(gauge, dict) and (
        _bounded(gauge.get("physical_map_relative_error"), 0., 1e-4)
        and _bounded(gauge.get("max_complete_rollout_logit_error"), 0., 5e-5)
    )


def _metric_dictionary_passed(metric) -> bool:
    if not isinstance(metric, dict):
        return False
    kl = metric.get("kl_by_pass_nats")
    return bool(
        _finite(metric.get("nll_nats")) and metric["nll_nats"] >= 0
        and _bounded(metric.get("accuracy"), 0., 1.)
        and type(metric.get("correct_count")) is int
        and 0 <= metric["correct_count"] <= 1024
        and _finite(metric.get("nll_delta_from_clean_nats"))
        and _bounded(metric.get("final_kl_nats"), 0., float("inf"))
        and _bounded(metric.get("closed_loop_write_displacement_rms"), 0., float("inf"))
        and isinstance(kl, list) and len(kl) == 4
        and all(_bounded(v, 0., float("inf")) for v in kl)
    )


def technical_admission_errors(record: dict) -> list[str]:
    """Check numeric gates even when a root's recorded eligible flag is true."""
    errors = []
    training = record.get("training") or {}
    if (training.get("root") != record.get("root") or training.get("steps") != 384
            or training.get("train_size") != 2048):
        errors.append("training_recipe_or_root")
    if not _bounded(training.get("mix_density"), .90, 1.):
        errors.append("physical_mix_density")
    split = record.get("splits") or {}
    if split != {"train": 2048, "derivative_discovery": 1024,
                 "model_validation": 512, "labeled_calibration": 1024}:
        errors.append("registered_split_sizes")
    contexts = record.get("clean_task_context") or {}
    if any(not _bounded(contexts.get(k), .85, 1.)
           for k in ("context_nonpositive", "context_positive")):
        errors.append("clean_context_accuracy")
    closure = record.get("full_mask_closure") or {}
    if (not _bounded(closure.get("max_abs_logit"), 0., 5e-5)
            or not _bounded(closure.get("max_kl_nats"), 0., 1e-7)):
        errors.append("native_full_mask_closure")
    optimum = record.get("source_rank12_optimum") or {}
    ranks = optimum.get("ranks")
    if (not isinstance(ranks, list) or len(ranks) != 4
            or any(type(r) is not int or r not in range(2, 7) for r in ranks)
            or sum(ranks) != 12 or optimum.get("total_rank") != 12):
        errors.append("source_rank12_optimum_missing_or_invalid")
        ranks = None
    if not _bounded(record.get("physical_source_gain"), .5, 1.5):
        errors.append("physical_source_gain")
    source_arms = record.get("source_arms") or {}
    task_arms = record.get("task_nll_by_arm") or {}
    gauges = record.get("native_rotated_gauge") or {}
    expected_ranks = {"dp12": ranks, "flat3": [3] * 4,
                      "flat3_dose": [3] * 4, "flat4": [4] * 4}
    for name in MANDATORY_ARMS:
        arm = source_arms.get(name) or {}
        if (arm.get("eligible") is not True or arm.get("ranks") != expected_ranks[name]
                or arm.get("total_rank") != (16 if name == "flat4" else 12)):
            errors.append("source_arm_admission:" + name)
        if not _metric_dictionary_passed(task_arms.get(name)):
            errors.append("mandatory_task_metrics:" + name)
        if not _gauge_passed(gauges.get(name)):
            errors.append("fixed_arm_gauge:" + name)
    adm = record.get("sham_admission") or {}
    expected = {
        "root": record.get("root"), "root_seed": 15040000 + record.get("root", 0),
        "charts_configured": 32, "max_corrections_per_chart": 64,
        "source_observation_count": 1024, "fixed_angle_radians": .25,
        "minimum_DP12_chordal": .20, "minimum_pairwise_chordal": .18,
        "source_write_relative_tolerance": .15, "jacobian_damping": .001,
        "minimum_norm_step_limit": .5, "line_search_scales": [1., .5, .25, .125],
        "uses_external_task_labels": False,
    }
    if any(adm.get(key) != value for key, value in expected.items()):
        errors.append("frozen_sampler_configuration")
    if len(adm.get("charts") or []) != 32:
        errors.append("source_chart_census_incomplete")
    if (adm.get("selected") != 8 or len(adm.get("projectors") or []) != 8
            or len(record.get("sham_nlls") or []) != 8):
        errors.append("eight_sham_admission")
    if record.get("selected_sham_score_source_only") is not True:
        errors.append("source_only_sham_selection")
    sham_metrics = record.get("sham_task_metrics") or []
    if len(sham_metrics) != 8 or not all(_metric_dictionary_passed(m) for m in sham_metrics):
        errors.append("sham_full_task_metrics")
    sham_gauges = record.get("native_rotated_sham_gauge") or []
    if len(sham_gauges) != 8 or not all(_gauge_passed(g) for g in sham_gauges):
        errors.append("sham_native_rotated_gauge")
    if record.get("eligible") is not True:
        errors.append("original_root_ineligible")
    return errors


def recompute_effects(record: dict) -> dict:
    """Use audited NLL metrics; preserve one effect and one null vector per root."""
    arms = record["task_nll_by_arm"]
    dp12 = float(arms["dp12"]["nll_nats"])
    shams = np.asarray(record["sham_nlls"], dtype=np.float64)
    if shams.shape != (8,) or not np.isfinite(shams).all():
        raise ValueError("Eight finite sham NLLs are required")
    fixed = [float(arms[name]["nll_nats"]) for name in MANDATORY_ARMS]
    clean = float(record["clean_nll"])
    if not np.isfinite(fixed + [clean]).all():
        raise ValueError("Finite fixed-arm and clean NLLs are required")
    median = float(np.median(shams))
    pseudo = np.asarray([(float(shams.sum()) - value) / 7 - value for value in shams])
    if abs(float(pseudo.mean())) > 1e-10:
        raise AssertionError("Uniform pseudo-target null is not centered")
    variance = float(np.mean(pseudo ** 2))
    analytic_variance = float(64 / 49 * np.mean((shams - shams.mean()) ** 2))
    if not math.isclose(variance, analytic_variance, rel_tol=1e-12, abs_tol=1e-15):
        raise AssertionError("Pseudo-target variance identity failed")
    index = int(np.random.default_rng(18180000 + record["root"]).integers(0, 8))
    return {
        "A_flat3_minus_dp12": fixed[1] - dp12,
        "D_dose_flat3_minus_dp12": fixed[2] - dp12,
        "S_mean_sham_minus_dp12": float(shams.mean()) - dp12,
        "C_dp12_minus_clean": dp12 - clean,
        "F_dp12_minus_flat4": dp12 - fixed[3],
        "S_median_sham_minus_dp12": median - dp12,
        "T_null_worst_abs": float(np.max(np.abs(shams - median))),
        "null_sham_signed_effects": (shams - median).tolist(),
        "Z_pseudo_null_by_sham": pseudo.tolist(),
        "pseudo_selected_sham_index": index,
        "Z_preselected_pseudo_sham": float(pseudo[index]),
        "conditional_pseudo_target_variance": variance,
    }


def available_fixed_effects(record: dict) -> dict:
    """Retain scored fixed-arm contrasts when a root lacks eight valid shams."""
    arms = record.get("task_nll_by_arm") or {}
    dp12 = (arms.get("dp12") or {}).get("nll_nats")
    if not _finite(dp12):
        return {}
    result = {}
    for name, arm_name in (("A_flat3_minus_dp12", "flat3"),
                           ("D_dose_flat3_minus_dp12", "flat3_dose"),
                           ("F_dp12_minus_flat4", "flat4")):
        value = (arms.get(arm_name) or {}).get("nll_nats")
        if _finite(value):
            result[name] = float(dp12 - value if arm_name == "flat4" else value - dp12)
    clean = record.get("clean_nll")
    if _finite(clean):
        result["C_dp12_minus_clean"] = float(dp12 - clean)
    return result


def development_gates(boot: dict, null: dict, power: dict) -> dict:
    """Exactly the FC02 development margins, separate from EvalMath readiness."""
    bmean = null["upper95_across_outer_95pct_estimates"]
    return {
        "all_59_source_eligible_and_three_way_matched": True,
        "DP12_benefit_over_flat3_and_source_dose_match": bool(
            boot["A_flat3_minus_dp12"]["mean"] >= .008
            and boot["A_flat3_minus_dp12"]["lower95"] > 0
            and boot["D_dose_flat3_minus_dp12"]["mean"] >= .006
            and boot["D_dose_flat3_minus_dp12"]["lower95"] > 0),
        "absolute_and_rank4_task_fidelity_and_local_shams": bool(
            boot["C_dp12_minus_clean"]["mean"] <= .05
            and boot["C_dp12_minus_clean"]["upper95"] <= .05
            and boot["F_dp12_minus_flat4"]["mean"] <= .020
            and boot["F_dp12_minus_flat4"]["upper95"] <= .020
            and boot["S_mean_sham_minus_dp12"]["mean"] >= .020
            and boot["S_mean_sham_minus_dp12"]["lower95"] > 0),
        "aggregate_null_exceeded_with_incremental_margin": bool(
            math.isfinite(bmean) and boot["S_mean_sham_minus_dp12"]["lower95"] > bmean + .020),
        "future_32_root_normal_power_planning": power["engineering_power_gate_passed"],
    }


def evalmath_review(external_floor: float, archive_source: str) -> dict:
    """Retain original 16 development effects and add the independent59 floor."""
    result = {"decision": "EVALMATH_NOT_RUN", "input": None, "output": None,
              "strict_exit_code": None, "script_git_blob_sha": None,
              "upstream_commit": EVALMATH_COMMIT, "errors": []}
    try:
        path = Path(evalmath_preflight.__file__)
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        result["script_git_blob_sha"] = blob
        if blob != EVALMATH_BLOB:
            raise AssertionError("Pinned EvalMath source mismatch")
        spec = json.loads((ROOT / "prereview/FC04_HISTORICAL_TARGET_PREFLIGHT_INPUT.json").read_text())
        expected = {"schema_version": 1, "alpha": "1/20", "scientific_min_effect": "1/50",
                    "target_effect": "27/1000", "measurement_resolution": "1/1000",
                    "sham_quantile": "19/20", "sham_confidence": "19/20",
                    "planned_comparisons": 3, "confirm_units_planned": 32,
                    "pilot_from_development_only": True, "heldout_sealed": True,
                    "units_independent": True}
        if any(spec.get(k) != v for k, v in expected.items()):
            raise AssertionError("Original historical-target preflight parameters changed")
        if (len(spec.get("pilot", [])) != 16 or
                {r.get("unit_id") for r in spec["pilot"]} != {f"FC04-root-{r}" for r in range(15000, 15016)}):
            raise AssertionError("Original FC04 development roots changed")
        if not isinstance(archive_source, str) or not archive_source.strip():
            raise ValueError("Exact original FC05 archive source is required")
        spec["experiment_id"] = "BPD-NC09-FC05-INDEPENDENT-CALIBRATION-REVIEW"
        spec["external_sham_floor"] = str(external_floor)
        spec["external_sham_source"] = archive_source
        result["input"] = spec
        # This is the exact routine used by the pinned CLI's --strict switch.
        output = evalmath_preflight.preflight(spec)
        result.update(output=output, decision=output["decision"],
                      strict_exit_code=0 if output["decision"] == "READY_FOR_CONFIRMATION" else 2)
    except ValueError as error:
        result["decision"] = ("EVALMATH_UNSUPPORTED_INPUT_HOLD" if "outside range" in str(error)
                              else "EVALMATH_NOT_RUN")
        result["errors"].append(type(error).__name__ + ": " + str(error))
    except Exception as error:
        result["errors"].append(type(error).__name__ + ": " + str(error))
    return result


def _archive_path(root_dir: Path, record: dict, key: str) -> Path:
    name = (record.get(key) or {}).get("file")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("Expected an archive basename: " + key)
    return root_dir / name


def aggregate(records: list[dict], root_dir: Path, *, archive_source: str,
              input_errors: list[str] | None = None) -> dict:
    root_dir = Path(root_dir)
    integrity = verify_source_manifest()
    counts = Counter(r.get("root") for r in records)
    evidence_errors = list(input_errors or [])
    for identity, count in counts.items():
        if identity not in ROOTS:
            evidence_errors.append(f"unregistered_root:{identity}")
        if count > 1:
            evidence_errors.append(f"duplicate_root:{identity}")
    by = {r.get("root"): r for r in records if counts[r.get("root")] == 1}
    protocol_hash = sha256(ROOT / "PROTOCOL.md")
    experiment_hash = sha256(ROOT / "experiment.py") if (ROOT / "experiment.py").is_file() else None
    per = []
    for seed in ROOTS:
        record = by.get(seed)
        row = {"root": seed, "eligible": False, "errors": [], "rank_tuple": None,
               "selected_shams": 0, "effects": None, "source_audit": None,
               "task_audit": None, "original_json_sha256": None}
        if record is None:
            row["errors"].append("duplicate_root" if counts[seed] > 1 else "missing_root")
            per.append(row)
            continue
        row["original_failure"] = record.get("failure") or record.get("error")
        row["rank_tuple"] = (record.get("source_rank12_optimum") or {}).get("ranks")
        row["selected_shams"] = len(record.get("sham_nlls") or [])
        original_path = root_dir / f"nc09_fc05_root_{seed}.json"
        if original_path.is_file():
            row["original_json_sha256"] = sha256(original_path)
            if json.loads(original_path.read_text()) != record:
                row["errors"].append("original_json_record_disagrees")
        else:
            row["errors"].append("original_json_file_missing")
        identity_errors = []
        for key, expected in (("study", STUDY), ("phase", PHASE),
                              ("protocol_sha256", protocol_hash),
                              ("source_sha256", experiment_hash),
                              ("source_manifest_sha256", integrity["manifest_sha256"])):
            if record.get(key) != expected or expected is None:
                identity_errors.append("source_identity:" + key)
        parent = record.get("parent_integrity") or {}
        if (parent.get("nc09_original_experiment_sha256") != PARENT_SOURCE_SHA
                or parent.get("nc09_original_protocol_sha256") != PARENT_PROTOCOL_SHA):
            identity_errors.append("source_identity:parent")
        row["errors"].extend(identity_errors)
        evidence_errors.extend(f"root_{seed}:{e}" for e in identity_errors)
        if not integrity["passed"]:
            row["errors"].append("runtime_source_manifest_failed")
        original_integrity = record.get("source_integrity") or {}
        if (original_integrity.get("passed") is not True or
                original_integrity.get("manifest_sha256") != integrity["manifest_sha256"]):
            row["errors"].append("original_source_manifest_check_failed")
        row["errors"].extend(technical_admission_errors(record))
        for key, function in (("source_archive", source_audit), ("task_archive", task_audit)):
            label = "source_audit" if key == "source_archive" else "task_audit"
            try:
                audit = function(record, _archive_path(root_dir, record, key))
            except Exception as error:
                audit = {"passed": False, "complete": False,
                         "errors": [type(error).__name__ + ": " + str(error)]}
            row[label] = audit
            if audit.get("passed") is not True or audit.get("complete") is not True:
                row["errors"].append(label + "_failed_or_incomplete")
        try:
            effect = recompute_effects(record)
            recorded = record.get("effects") or {}
            for key, value in effect.items():
                if key == "conditional_pseudo_target_variance":
                    continue
                actual = recorded.get(key)
                if actual is None or not np.allclose(value, actual, rtol=0., atol=1e-10):
                    row["errors"].append("derived_effect_mismatch:" + key)
            row["effects"] = effect
        except Exception as error:
            row["errors"].append("effect_incomplete:" + type(error).__name__ + ": " + str(error))
            row["effects"] = available_fixed_effects(record) or None
        row["eligible"] = not row["errors"]
        per.append(row)
    summary = {
        "study": STUDY, "phase": PHASE, "evidence_class": "CALIBRATION_DEVELOPMENT_ONLY",
        "protocol_sha256": protocol_hash, "source_manifest_audit": integrity,
        "expected_independent_model_roots": 59, "received_records": len(records),
        "eligible_roots": sum(r["eligible"] for r in per),
        "evidence_errors": evidence_errors, "per_root": per,
        "registered_development_verdict": "CALIBRATION_INCOMPLETE",
        "rootwise_95_95_confidence": EXACT_ROOTWISE_CONFIDENCE,
        "rootwise_sham_max_95_95_diagnostic": None, "root_mean_bootstrap": None,
        "aggregate_32_root_pseudo_null": None, "approximate_power": None,
        "gates": None, "post_calibration_evalmath": None,
        "calibration_scientific_confirmation_ready": False,
        "scientific_confirmation_roots_evaluated": 0,
        "other_protected_model_roots_evaluated": 0,
        "original_FC01_verdict_unchanged": "CALIBRATION_REDESIGN_OR_PILOT_ONLY",
        "original_FC02_verdict_unchanged": "CALIBRATION_INCOMPLETE",
        "original_FC03_verdict_unchanged": "ENGINEERING_NULL_OR_INELIGIBLE",
    }
    if summary["eligible_roots"] != 59 or evidence_errors or not integrity["passed"]:
        return summary
    matrix = np.array([[r["effects"][name] for name in METRICS] for r in per])
    pseudo = np.array([r["effects"]["Z_pseudo_null_by_sham"] for r in per])
    boot = root_mean_bootstrap(matrix)
    null = aggregate_root_mean_null(pseudo)
    rootmax = max(r["effects"]["T_null_worst_abs"] for r in per)
    power = approximate_power(boot["S_mean_sham_minus_dp12"]["mean"],
                              boot["S_mean_sham_minus_dp12"]["root_sd"],
                              null["upper95_across_outer_95pct_estimates"])
    gates = development_gates(boot, null, power)
    summary.update(
        rootwise_sham_max_95_95_diagnostic=rootmax, root_mean_bootstrap=boot,
        aggregate_32_root_pseudo_null=null, approximate_power=power, gates=gates,
        registered_development_verdict=("FULL_TANGENT_CALIBRATION_DEVELOPMENT_POSITIVE"
                                        if all(gates.values()) else "FULL_TANGENT_CALIBRATION_PILOT_ONLY"),
        post_calibration_evalmath=evalmath_review(rootmax, archive_source),
    )
    return summary


def write_exports(summary: dict, records: list[dict], root_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["root", "eligible", "rank_tuple", "selected_shams", *METRICS,
              "T_null_worst_abs", "original_json_sha256", "errors"]
    with (out_dir / "FC05_PER_ROOT.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["per_root"]:
            exported = {k: row.get(k) for k in fields if k not in METRICS}
            exported.update({k: (row.get("effects") or {}).get(k) for k in METRICS})
            exported["T_null_worst_abs"] = (row.get("effects") or {}).get("T_null_worst_abs")
            exported["rank_tuple"] = json.dumps(row.get("rank_tuple"))
            exported["errors"] = ";".join(row["errors"])
            writer.writerow(exported)
    with (out_dir / "FC05_ORIGINAL_SHA256.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["root", "kind", "file", "size_bytes", "sha256"])
        writer.writeheader()
        for record in sorted(records, key=lambda r: str(r.get("root"))):
            seed = record.get("root")
            if type(seed) is not int:
                continue
            candidates = [("root", root_dir / f"nc09_fc05_root_{seed}.json")]
            for key in ("source_archive", "task_archive"):
                try:
                    candidates.append((key, _archive_path(root_dir, record, key)))
                except ValueError:
                    pass
            for kind, path in candidates:
                if path.is_file():
                    writer.writerow({"root": seed, "kind": kind, "file": path.name,
                                     "size_bytes": path.stat().st_size, "sha256": sha256(path)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--archive-source", required=True)
    args = parser.parse_args()
    records = []
    load_errors = []
    for path in sorted(args.root_dir.glob("nc09_fc05_root_*.json")):
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict) or type(record.get("root")) is not int:
                raise ValueError("Root receipt needs a dictionary and integer root ID")
            records.append(record)
        except Exception as error:
            load_errors.append({"file": path.name, "error": type(error).__name__ + ": " + str(error)})
    summary = aggregate(records, args.root_dir, archive_source=args.archive_source,
                        input_errors=["root_json_load_error:" + e["file"] for e in load_errors])
    summary["original_root_load_errors"] = load_errors
    write_exports(summary, records, args.root_dir, args.out_dir)
    (args.out_dir / "FC05_ORIGINAL_59_ROOT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    review = summary["post_calibration_evalmath"]
    if review is not None:
        for key, filename in (("input", "FC05_EVALMATH_INPUT.json"),
                              ("output", "FC05_EVALMATH_PREFLIGHT.json")):
            if review.get(key) is not None:
                (args.out_dir / filename).write_text(json.dumps(review[key], indent=2, sort_keys=True) + "\n")
        receipt = {k: v for k, v in review.items() if k not in ("input", "output")}
        (args.out_dir / "FC05_EVALMATH_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"registered_development_verdict": summary["registered_development_verdict"],
                      "eligible_roots": summary["eligible_roots"],
                      "evalmath_decision": review["decision"] if review else None,
                      "scientific_confirmation_roots_evaluated": 0}, indent=2))


if __name__ == "__main__":
    main()
