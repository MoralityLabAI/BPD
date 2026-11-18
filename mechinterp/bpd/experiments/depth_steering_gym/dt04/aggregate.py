"""BPD-DT04 independent receipt audit, whole-root bootstrap, and EvalMath preflight.

Audits all 59 development/calibration root receipts (`2604000..2604058`),
re-verifies archived `.pt` checkpoint SHA-256 digests, checks canonical split
disjointness, protected 11-vocab readout conservation, BPD GL(C) gauge error,
and all 8 x 59 = 472 signed-Jacobian sham certificates. Computes 16,384
whole-root percentile bootstrap intervals (Bonferroni-adjusted across the
6-endpoint family) and executes pinned EvalMath power/sham-tolerance preflight.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mechinterp.bpd.experiments.nc09_fixed12.calibration.evalmath_preflight import preflight

from . import DEV_ROOTS, SMOKE_ROOT, STUDY_ID
from .verify_source import sha256_file, verify_source

N_BOOTSTRAP = 16_384
FAMILY_SIZE = 6
BOOTSTRAP_BASE_SEED = 26040019


def _finite(val) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def audit_single_receipt(record: dict, out_dir: Path) -> list[str]:
    """Return list of integrity or contract errors for one root receipt."""
    errs: list[str] = []
    if record.get("schema") != "BPD-DT04/root-receipt/v1":
        errs.append("invalid_receipt_schema")
        return errs
    if record.get("status") != "COMPLETED":
        errs.append(f"root_not_completed:{record.get('status')}")
        return errs

    fam = record.get("families", {})
    sud = fam.get("sudoku4", {})
    arc = fam.get("arc_style", {})
    sto = fam.get("storyworld", {})
    caus = record.get("causal_steering", {})

    # 1. Verify saved Sudoku4 checkpoint file SHA-256
    ckpt_rel = sud.get("checkpoint_file")
    ckpt_sha = sud.get("checkpoint_sha256")
    if not isinstance(ckpt_rel, str) or not isinstance(ckpt_sha, str):
        errs.append("missing_checkpoint_metadata")
    else:
        ckpt_path = out_dir / ckpt_rel
        if not ckpt_path.is_file():
            errs.append("missing_checkpoint_bytes")
        elif sha256_file(ckpt_path) != ckpt_sha:
            errs.append("checkpoint_sha256_mismatch")

    # 2. Verify canonical Sudoku split admissibility (0 cross-split isomorphic items)
    sp = sud.get("split_audit", {})
    if sp.get("cross_split_isomorphic_items") != 0 or sp.get("admissible") is not True:
        errs.append("sudoku_canonical_isomorphism_leak")

    # 3. Verify same-case depth (6, 8, 12) predictions exist
    for d_key in ("6", "8", "12"):
        d_rec = sud.get("eval_by_depth", {}).get(d_key, {})
        if not isinstance(d_rec.get("predictions"), list) or len(d_rec["predictions"]) != 64:
            errs.append(f"missing_sudoku_same_case_predictions_d{d_key}")

    # 4. Verify ARC official_arc1_evaluated is False and color support audits exist
    if arc.get("official_arc1_evaluated") is not False:
        errs.append("invalid_official_arc1_flag")
    c_aud_leg = arc.get("arc_analogue_strict_2rule", {}).get("color_support_audit", {})
    c_aud_lut = arc.get("arc_lut_192_to_64_induction", {}).get("color_support_audit", {})
    if 3 not in c_aud_leg.get("unsupported_heldout_colors", []):
        errs.append("legacy_arc_color3_deficit_not_detected")
    if c_aud_lut.get("full_4color_support_in_train") is not True:
        errs.append("lut_arc_4color_support_missing")

    # 5. Verify protected 11-vocab readout error <= 1e-6
    prot_err = caus.get("protected_all_vocab_max_error")
    if not _finite(prot_err) or prot_err > 1e-6:
        errs.append(f"protected_readout_violation:{prot_err}")

    # 6. Verify BPD GL(C) factor-gauge invariance
    gauge = caus.get("bpd_physical_factor_audit", {}).get("gauge_invariance", {})
    if (
        gauge.get("passed") is not True
        or not _finite(gauge.get("physical_operator_rel_error"))
        or gauge["physical_operator_rel_error"] > 1e-4
        or not _finite(gauge.get("max_rollout_logit_abs_error"))
        or gauge["max_rollout_logit_abs_error"] > 5e-5
    ):
        errs.append("bpd_gauge_invariance_failed")

    # 7. Verify all 8 privileged signed-Jacobian shams
    shams = caus.get("sham_validation_ledger", [])
    if not isinstance(shams, list) or len(shams) != 8:
        errs.append("incomplete_sham_ledger")
    else:
        for s_item in shams:
            if (
                not _finite(s_item.get("span_error"))
                or s_item["span_error"] > 1e-4
                or not _finite(s_item.get("delivered_l2_error"))
                or s_item["delivered_l2_error"] > 1e-4
                or not _finite(s_item.get("full_signed_jacobian_error"))
                or s_item["full_signed_jacobian_error"] > 1e-4
                or s_item.get("distinct_contrast_available") is not True
            ):
                errs.append(f"sham_certificate_failed:{s_item.get('sham_index')}")

    return errs


def _whole_root_bootstrap(values: list[float], endpoint_idx: int) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_BASE_SEED + 101 * endpoint_idx))
    draws = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    means = arr[draws].mean(axis=1)
    alpha_tail = 0.05 / (2.0 * FAMILY_SIZE)
    lo_bonf, hi_bonf = np.quantile(means, [alpha_tail, 1.0 - alpha_tail])
    lo_95, hi_95 = np.quantile(means, [0.025, 0.975])
    pos = int((arr > 0).sum())
    neg = int((arr < 0).sum())
    ties = int((arr == 0).sum())
    return {
        "n_roots": n,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(arr)),
        "ci95_marginal": [float(lo_95), float(hi_95)],
        "ci_familywise_bonferroni": [float(lo_bonf), float(hi_bonf)],
        "positive_roots": pos,
        "negative_roots": neg,
        "tied_roots": ties,
    }


def _fmt_frac_str(val: float) -> str:
    clipped = max(-1.0, min(1.0, float(val)))
    return f"{clipped:.12f}"


def aggregate_and_audit(out_dir: Path, *, allow_partial_dev: bool = False) -> dict:
    src_ver = verify_source(require_runtime=False)
    roots_dir = out_dir / "roots"
    expected_roots = list(DEV_ROOTS)

    loaded: list[dict] = []
    missing_roots: list[int] = []
    audit_by_root: dict[str, dict] = {}
    total_sham_certs = 0

    for r in expected_roots:
        p = roots_dir / f"root_{r}.json"
        if not p.is_file():
            missing_roots.append(r)
            continue
        rec = json.loads(p.read_bytes())
        errs = audit_single_receipt(rec, out_dir)
        audit_by_root[str(r)] = {
            "passed": len(errs) == 0,
            "errors": errs,
            "receipt_sha256": sha256_file(p),
        }
        if not errs:
            total_sham_certs += len(rec["causal_steering"]["sham_validation_ledger"])
        loaded.append(rec)

    if missing_roots and not allow_partial_dev:
        raise RuntimeError(f"Missing {len(missing_roots)} registered DT04 roots: {missing_roots[:5]}...")

    all_audit_passed = (
        len(missing_roots) == 0
        and len(loaded) == len(expected_roots)
        and all(v["passed"] for v in audit_by_root.values())
    )
    admitted_recs = [
        r for r in loaded
        if audit_by_root[str(r["root"])]["passed"] and r["admission"]["all_gates_admitted"]
    ]
    cohort_eligible = all_audit_passed and len(admitted_recs) == len(expected_roots)

    # Extract per-root series across all loaded & audit-clean roots
    clean_recs = [r for r in loaded if audit_by_root[str(r["root"])]["passed"]]

    def _series(fn) -> list[float]:
        return [float(fn(r)) for r in clean_recs]

    # Competence & depth-transport descriptive summaries
    competence_summary = {}
    for d_k in ("2", "4", "6", "8", "12"):
        competence_summary[f"sudoku4_exact_d{d_k}"] = float(
            np.mean(_series(lambda r, dk=d_k: r["families"]["sudoku4"]["eval_by_depth"][dk]["whole_board_exact"]))
        )
        competence_summary[f"sudoku4_blank_acc_d{d_k}"] = float(
            np.mean(_series(lambda r, dk=d_k: r["families"]["sudoku4"]["eval_by_depth"][dk]["blank_accuracy"]))
        )
    for d_k in ("6", "8", "12"):
        competence_summary[f"arc_legacy2rule_indist_exact_d{d_k}"] = float(
            np.mean(_series(lambda r, dk=d_k: r["families"]["arc_style"]["arc_analogue_strict_2rule"]["eval"]["in_dist"][dk]["exact_grid_accuracy"]))
        )
        competence_summary[f"arc_legacy2rule_heldout_exact_d{d_k}"] = float(
            np.mean(_series(lambda r, dk=d_k: r["families"]["arc_style"]["arc_analogue_strict_2rule"]["eval"]["heldout"][dk]["exact_grid_accuracy"]))
        )
        competence_summary[f"arc_lut192_indist_exact_d{d_k}"] = float(
            np.mean(_series(lambda r, dk=d_k: r["families"]["arc_style"]["arc_lut_192_to_64_induction"]["eval"]["in_dist"][dk]["exact_grid_accuracy"]))
        )
        competence_summary[f"arc_lut192_heldout_cell_d{d_k}"] = float(
            np.mean(_series(lambda r, dk=d_k: r["families"]["arc_style"]["arc_lut_192_to_64_induction"]["eval"]["heldout"][dk]["cell_accuracy"]))
        )
        competence_summary[f"arc_lut192_heldout_exact_d{d_k}"] = float(
            np.mean(_series(lambda r, dk=d_k: r["families"]["arc_style"]["arc_lut_192_to_64_induction"]["eval"]["heldout"][dk]["exact_grid_accuracy"]))
        )
        competence_summary[f"storyworld_terminal_success_d{d_k}"] = float(
            np.mean(_series(lambda r, dk=d_k: r["families"]["storyworld"]["eval_by_depth"][dk]["host_ending_success_rate"]))
        )

    # Steering absolute means across arms at depths (6, 8, 12)
    arm_names = (
        "no_edit",
        "full_dark_donor",
        "task_gramian_r8",
        "bpd_vu_r8_dose",
        "weight_svd_left_r8_dose",
        "random_protected_r8_dose",
    )
    steering_table = {}
    for d_k in ("6", "8", "12"):
        row_d = {}
        for a_name in arm_names:
            row_d[a_name] = {
                "gold_whole_board_exact": float(np.mean(_series(lambda r, dk=d_k, an=a_name: r["causal_steering"]["same_board_corrupt_repair_by_depth"][dk][an]["gold_whole_board_exact"]))),
                "gold_blank_accuracy": float(np.mean(_series(lambda r, dk=d_k, an=a_name: r["causal_steering"]["same_board_corrupt_repair_by_depth"][dk][an]["gold_blank_accuracy"]))),
                "whole_case_donor_recovery": float(np.mean(_series(lambda r, dk=d_k, an=a_name: r["causal_steering"]["same_board_corrupt_repair_by_depth"][dk][an]["whole_case_recovery"]))),
                "net_blank_cell_gain": float(np.mean(_series(lambda r, dk=d_k, an=a_name: r["causal_steering"]["same_board_corrupt_repair_by_depth"][dk][an]["net_blank_cell_gain"]))),
                "delivered_l2_dose": float(np.mean(_series(lambda r, dk=d_k, an=a_name: r["causal_steering"]["same_board_corrupt_repair_by_depth"][dk][an]["delivered_l2_dose_mean"]))),
            }
        # Mean across 8 signed-Jacobian shams
        row_d["signed_jac_sham_mean8"] = {
            "gold_whole_board_exact": float(np.mean([
                np.mean([r["causal_steering"]["same_board_corrupt_repair_by_depth"][d_k][f"signed_jac_sham_{i}"]["gold_whole_board_exact"] for i in range(8)])
                for r in clean_recs
            ])),
            "gold_blank_accuracy": float(np.mean([
                np.mean([r["causal_steering"]["same_board_corrupt_repair_by_depth"][d_k][f"signed_jac_sham_{i}"]["gold_blank_accuracy"] for i in range(8)])
                for r in clean_recs
            ])),
            "whole_case_donor_recovery": float(np.mean([
                np.mean([r["causal_steering"]["same_board_corrupt_repair_by_depth"][d_k][f"signed_jac_sham_{i}"]["whole_case_recovery"] for i in range(8)])
                for r in clean_recs
            ])),
            "net_blank_cell_gain": float(np.mean([
                np.mean([r["causal_steering"]["same_board_corrupt_repair_by_depth"][d_k][f"signed_jac_sham_{i}"]["net_blank_cell_gain"] for i in range(8)])
                for r in clean_recs
            ])),
        }
        steering_table[d_k] = row_d

    # Transport geometry & BPD sparse factor summary
    geometry_summary = {
        "mean_one_step_subspace_closure_defect": float(np.mean(_series(lambda r: r["causal_steering"]["depth_transport_geometry"]["one_step_subspace_closure_defect"]))),
        "mean_one_step_protected_kernel_escape": float(np.mean(_series(lambda r: r["causal_steering"]["depth_transport_geometry"]["one_step_protected_kernel_escape"]))),
        "overlap_d6": float(np.mean(_series(lambda r: r["causal_steering"]["depth_transport_geometry"]["principal_angle_drift_by_depth"]["6"]["subspace_overlap_fraction"]))),
        "overlap_d8": float(np.mean(_series(lambda r: r["causal_steering"]["depth_transport_geometry"]["principal_angle_drift_by_depth"]["8"]["subspace_overlap_fraction"]))),
        "overlap_d12": float(np.mean(_series(lambda r: r["causal_steering"]["depth_transport_geometry"]["principal_angle_drift_by_depth"]["12"]["subspace_overlap_fraction"]))),
        "mean_angle_deg_d6": float(np.mean(_series(lambda r: r["causal_steering"]["depth_transport_geometry"]["principal_angle_drift_by_depth"]["6"]["mean_principal_angle_deg"]))),
        "mean_angle_deg_d8": float(np.mean(_series(lambda r: r["causal_steering"]["depth_transport_geometry"]["principal_angle_drift_by_depth"]["8"]["mean_principal_angle_deg"]))),
        "mean_angle_deg_d12": float(np.mean(_series(lambda r: r["causal_steering"]["depth_transport_geometry"]["principal_angle_drift_by_depth"]["12"]["mean_principal_angle_deg"]))),
        "bpd_sparse_top12_exact_d6": float(np.mean(_series(lambda r: r["causal_steering"]["bpd_physical_factor_audit"]["by_depth"]["6"]["sparse_topk_exact_board"]))),
        "bpd_sparse_top12_exact_d8": float(np.mean(_series(lambda r: r["causal_steering"]["bpd_physical_factor_audit"]["by_depth"]["8"]["sparse_topk_exact_board"]))),
        "bpd_sparse_top12_exact_d12": float(np.mean(_series(lambda r: r["causal_steering"]["bpd_physical_factor_audit"]["by_depth"]["12"]["sparse_topk_exact_board"]))),
        "bpd_sparse_top12_kl_d6": float(np.mean(_series(lambda r: r["causal_steering"]["bpd_physical_factor_audit"]["by_depth"]["6"]["sparse_topk_kl_nats"]))),
        "bpd_sparse_top12_kl_d8": float(np.mean(_series(lambda r: r["causal_steering"]["bpd_physical_factor_audit"]["by_depth"]["8"]["sparse_topk_kl_nats"]))),
        "bpd_sparse_top12_kl_d12": float(np.mean(_series(lambda r: r["causal_steering"]["bpd_physical_factor_audit"]["by_depth"]["12"]["sparse_topk_kl_nats"]))),
    }

    # Registered 6-endpoint family across all admitted roots
    endpoint_defs = [
        ("E1_gramian_vs_no_edit_d8_exact", lambda r: r["causal_steering"]["root_contrasts_by_depth"]["8"]["gramian_minus_no_edit_exact"], 0.10),
        ("E2_gramian_vs_signed_sham_d8_exact", lambda r: r["causal_steering"]["root_contrasts_by_depth"]["8"]["gramian_minus_mean_sham_exact"], 0.10),
        ("E3_gramian_vs_signed_sham_d12_exact", lambda r: r["causal_steering"]["root_contrasts_by_depth"]["12"]["gramian_minus_mean_sham_exact"], 0.10),
        ("E4_gramian_vs_signed_sham_d8_blank_gain", lambda r: r["causal_steering"]["root_contrasts_by_depth"]["8"]["gramian_minus_mean_sham_blank_gain"], 0.05),
        ("E5_bpd_vu_vs_signed_sham_d8_exact", lambda r: r["causal_steering"]["root_contrasts_by_depth"]["8"]["bpd_minus_mean_sham_exact"], 0.10),
        ("E6_gramian_vs_signed_sham_d6_fit_exact", lambda r: r["causal_steering"]["root_contrasts_by_depth"]["6"]["gramian_minus_mean_sham_exact"], 0.10),
    ]
    endpoints_report = {}
    for idx, (ep_name, fn, min_floor) in enumerate(endpoint_defs):
        vals = _series(fn)
        stats = _whole_root_bootstrap(vals, idx) if len(vals) >= 2 else None
        passed = bool(
            cohort_eligible
            and stats is not None
            and stats["ci_familywise_bonferroni"][0] > min_floor
        )
        endpoints_report[ep_name] = {
            "admitted_roots": len(admitted_recs),
            "required_roots": len(expected_roots),
            "eligible": cohort_eligible,
            "min_lower_bound_required": min_floor,
            "bootstrap": stats if cohort_eligible else None,
            "descriptive_mean": float(np.mean(vals)) if vals else None,
            "decision": "PASS" if passed else ("FAIL" if cohort_eligible else "INELIGIBLE"),
        }

    # Pinned EvalMath post-development preflight on primary depth-8 Gramian vs signed-Jacobian sham contrast
    pilot_rows = []
    for r in clean_recs:
        c8 = r["causal_steering"]["root_contrasts_by_depth"]["8"]
        pilot_rows.append({
            "unit_id": f"root_{r['root']}",
            "paired_effect": _fmt_frac_str(c8["gramian_minus_mean_sham_exact"]),
            "sham_effects": [_fmt_frac_str(v) for v in c8["sham_centered_exact_list"]],
        })

    evalmath_spec = {
        "schema_version": 1,
        "experiment_id": "BPD-DT04-depth8-gramian-vs-signed-sham",
        "independent_unit": "independently trained PureSudokuTRM model root",
        "alpha": "0.05",
        "scientific_min_effect": "0.10",
        "target_effect": "0.15",
        "measurement_resolution": str(Fraction(1, 16)),
        "sham_quantile": "0.95",
        "sham_confidence": "0.95",
        "planned_comparisons": FAMILY_SIZE,
        "confirm_units_planned": 128,
        "pilot_from_development_only": True,
        "heldout_sealed": True,
        "units_independent": True,
        "pilot": pilot_rows,
    }
    evalmath_out = preflight(evalmath_spec) if pilot_rows else None

    overall_verdict = (
        "DEVELOPMENT_MECHANISM_GATE_PASSED_PENDING_HUMAN_RELEASE"
        if (
            cohort_eligible
            and endpoints_report["E1_gramian_vs_no_edit_d8_exact"]["decision"] == "PASS"
            and endpoints_report["E2_gramian_vs_signed_sham_d8_exact"]["decision"] == "PASS"
            and endpoints_report["E3_gramian_vs_signed_sham_d12_exact"]["decision"] == "PASS"
            and evalmath_out is not None
            and evalmath_out["decision"] == "READY_FOR_CONFIRMATION"
        )
        else "COMPETENCE_ADMITTED_STEERING_PILOT_ONLY_OR_NULL"
        if cohort_eligible
        else "INCOMPLETE_OR_INELIGIBLE"
    )

    # Write per-root CSV
    csv_path = out_dir / "PER_ROOT_20261119.csv"
    with csv_path.open("w", newline="\n", encoding="utf-8") as cf:
        fieldnames = [
            "root", "admitted", "sudoku_d6_exact", "sudoku_d8_exact", "sudoku_d12_exact",
            "arc_legacy_id_d8", "arc_legacy_ho_d8", "arc_lut_id_d8", "arc_lut_ho_cell_d8", "arc_lut_ho_exact_d8",
            "story_d8_success", "story_d12_success",
            "no_edit_d8_exact", "full_dark_d8_exact", "gramian_d8_exact", "bpd_vu_d8_exact", "sham_mean_d8_exact",
            "gram_minus_sham_d6", "gram_minus_sham_d8", "gram_minus_sham_d12",
            "angle_deg_d8", "angle_deg_d12", "closure_defect", "bpd_gauge_err",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in clean_recs:
            s_ev = r["families"]["sudoku4"]["eval_by_depth"]
            a_leg = r["families"]["arc_style"]["arc_analogue_strict_2rule"]["eval"]
            a_lut = r["families"]["arc_style"]["arc_lut_192_to_64_induction"]["eval"]
            st_ev = r["families"]["storyworld"]["eval_by_depth"]
            cs = r["causal_steering"]
            d8_arms = cs["same_board_corrupt_repair_by_depth"]["8"]
            rc = cs["root_contrasts_by_depth"]
            geom = cs["depth_transport_geometry"]
            sham_m8 = sum(d8_arms[f"signed_jac_sham_{i}"]["gold_whole_board_exact"] for i in range(8)) / 8.0
            writer.writerow({
                "root": r["root"],
                "admitted": r["admission"]["all_gates_admitted"],
                "sudoku_d6_exact": f"{s_ev['6']['whole_board_exact']:.6f}",
                "sudoku_d8_exact": f"{s_ev['8']['whole_board_exact']:.6f}",
                "sudoku_d12_exact": f"{s_ev['12']['whole_board_exact']:.6f}",
                "arc_legacy_id_d8": f"{a_leg['in_dist']['8']['exact_grid_accuracy']:.6f}",
                "arc_legacy_ho_d8": f"{a_leg['heldout']['8']['exact_grid_accuracy']:.6f}",
                "arc_lut_id_d8": f"{a_lut['in_dist']['8']['exact_grid_accuracy']:.6f}",
                "arc_lut_ho_cell_d8": f"{a_lut['heldout']['8']['cell_accuracy']:.6f}",
                "arc_lut_ho_exact_d8": f"{a_lut['heldout']['8']['exact_grid_accuracy']:.6f}",
                "story_d8_success": f"{st_ev['8']['host_ending_success_rate']:.6f}",
                "story_d12_success": f"{st_ev['12']['host_ending_success_rate']:.6f}",
                "no_edit_d8_exact": f"{d8_arms['no_edit']['gold_whole_board_exact']:.6f}",
                "full_dark_d8_exact": f"{d8_arms['full_dark_donor']['gold_whole_board_exact']:.6f}",
                "gramian_d8_exact": f"{d8_arms['task_gramian_r8']['gold_whole_board_exact']:.6f}",
                "bpd_vu_d8_exact": f"{d8_arms['bpd_vu_r8_dose']['gold_whole_board_exact']:.6f}",
                "sham_mean_d8_exact": f"{sham_m8:.6f}",
                "gram_minus_sham_d6": f"{rc['6']['gramian_minus_mean_sham_exact']:+.6f}",
                "gram_minus_sham_d8": f"{rc['8']['gramian_minus_mean_sham_exact']:+.6f}",
                "gram_minus_sham_d12": f"{rc['12']['gramian_minus_mean_sham_exact']:+.6f}",
                "angle_deg_d8": f"{geom['principal_angle_drift_by_depth']['8']['mean_principal_angle_deg']:.4f}",
                "angle_deg_d12": f"{geom['principal_angle_drift_by_depth']['12']['mean_principal_angle_deg']:.4f}",
                "closure_defect": f"{geom['one_step_subspace_closure_defect']:.6f}",
                "bpd_gauge_err": f"{cs['bpd_physical_factor_audit']['gauge_invariance']['max_rollout_logit_abs_error']:.3e}",
            })

    audit_payload = {
        "schema": "BPD-DT04/independent-audit/v1",
        "study": STUDY_ID,
        "source_verification": src_ver,
        "expected_roots": len(expected_roots),
        "loaded_roots": len(loaded),
        "missing_roots": missing_roots,
        "all_receipts_passed_integrity": all_audit_passed,
        "admitted_roots_count": len(admitted_recs),
        "total_verified_sham_certificates": total_sham_certs,
        "per_root_audit": audit_by_root,
    }
    (out_dir / "DT04_ORIGINAL_AUDIT.json").write_bytes(
        (json.dumps(audit_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )

    if evalmath_out is not None:
        (out_dir / "POST_DEVELOPMENT_PREFLIGHT.json").write_bytes(
            (json.dumps({"input_spec": evalmath_spec, "preflight": evalmath_out}, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )

    summary = {
        "schema": "BPD-DT04/original-summary/v1",
        "study": STUDY_ID,
        "verdict": overall_verdict,
        "source_verification": src_ver,
        "roots_expected": len(expected_roots),
        "roots_completed": len(clean_recs),
        "roots_admitted": len(admitted_recs),
        "competence_and_depth_transport": competence_summary,
        "steering_arms_by_depth": steering_table,
        "geometry_and_bpd_factors": geometry_summary,
        "registered_endpoints": endpoints_report,
        "evalmath_post_development": evalmath_out,
        "per_root_csv_sha256": sha256_file(csv_path),
        "confirmation_cohort_status": "SEALED_NOT_GENERATED_OR_RUN",
        "arg03_n_status": "SEALED_UNTOUCHED",
    }
    summary_path = out_dir / "DT04_ORIGINAL_SUMMARY.json"
    summary_path.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    # Build immutable artifact manifest covering all generated files
    members = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "ORIGINAL_ARTIFACT_MANIFEST.json":
            members.append({
                "path": str(p.relative_to(out_dir)).replace("\\", "/"),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    manifest = {
        "schema": "BPD-DT04/artifact-manifest/v1",
        "study": STUDY_ID,
        "member_count": len(members),
        "total_bytes": sum(m["bytes"] for m in members),
        "summary_sha256": sha256_file(summary_path),
        "members": members,
    }
    (out_dir / "ORIGINAL_ARTIFACT_MANIFEST.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    summary = aggregate_and_audit(args.out_dir, allow_partial_dev=args.allow_partial)
    print(
        json.dumps({
            "verdict": summary["verdict"],
            "roots_completed": summary["roots_completed"],
            "roots_admitted": summary["roots_admitted"],
            "evalmath_decision": summary["evalmath_post_development"]["decision"]
            if summary["evalmath_post_development"]
            else None,
        }, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
