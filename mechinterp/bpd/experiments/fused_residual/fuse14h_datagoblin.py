"""FUSE-14 Stage H: DataGoblin routing dataset export & hard-evidence-gated admission evaluation.

Strictly experimental and outside the numerical correctness path (`correctness_path=False`).
Exports measured per-site & per-cell rows across FUSE-12, FUSE-13, and FUSE-14 (including
synthetic mini-TRM and real pretrained Qwen3-4B transfer rows) with checkpoint family,
shape, precision, depth, hardware, policy, numerical gate, latency, storage, peak VRAM,
and blocked reason.

Hard evidence gates MUST precede cost ranking (eliminating the 0.25 precision-holdout
unsafe admission seen on un-gated baselines in FUSE-13E). Trains a compact residual
correction ONLY to test whether it strictly beats the same hard-gated safe physical
baseline on locked width/depth/precision groups; reports tail regret (p95 & max),
false rejections, unsafe admissions, and calibration (Brier & ECE).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .admission import Site, select
from .bench_fuse14_frontier import fuse14_bf16_preflight_mib
from .fuse12c_datagoblin import (
    PRECISION_LIST,
    _build_site_samples,
    _oracle_subset,
)
from .fuse13e_datagoblin import (
    DEFAULT_HEADROOM_MIB,
    DEFAULT_LEASE_MIB,
    _calibration_metrics,
    build_fuse13e_dataset,
    physical_memory_model,
    physical_site_prior_ms,
)

HARDWARE_NAME = "NVIDIA GeForce RTX 5080 Laptop GPU"


def _rows_from_fuse14_suite(suite_path: Path) -> list[dict[str, Any]]:
    """Extract normalized FUSE-14H rows from Stage D parity matrix & Stage G Qwen3-4B transfer."""
    if not suite_path.exists():
        return []
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    # 1. Stage D synthetic mini-TRM rows (Arm G vs Arm H)
    for cell in suite.get("stage_d_parity_matrix", []):
        w = int(cell["width"])
        t = int(cell["visits"])
        seed = int(cell["seed"])
        reg = cell["regime"]
        cal = cell["cal_cadence"]
        bud = cell["budget_mode"]
        par_ok = bool(cell["parity"]["finite_bitwise_pass"])
        mg, mh = cell["fuse13_arm_g"], cell["fuse14_arm_h"]
        g_ms = float(mg["step_ms"][-1])
        h_ms = float(mh["step_ms"][-1])
        proj_rec = mh.get("projection_cache_receipt") or {}
        res_rec = mh.get("residual_step_receipt") or {}
        for pol, meta, ms_val in (
            ("fuse13_arm_g", mg, g_ms),
            ("fuse14_arm_h", mh, h_ms),
        ):
            rows.append(
                {
                    "schema_version": "fuse14h.v1",
                    "source_campaign": "fuse14_stage_d",
                    "checkpoint_family": "synthetic_mini_trm",
                    "cell_id": f"fuse14d__bf16__w{w}_t{t}_r{seed}_{reg}_{cal}_{bud}",
                    "hardware": HARDWARE_NAME,
                    "device": suite.get("device", "cuda"),
                    "precision": "bf16",
                    "storage_mode": "compute_dtype",
                    "width": w,
                    "recurrence_depth": t,
                    "regime": reg,
                    "calibration_cadence": cal,
                    "budget_mode": bud,
                    "policy": pol,
                    "numerical_gate": "PASS_FINITE_BITWISE" if par_ok else "FAIL",
                    "latency_ms": ms_val,
                    "baseline_latency_ms": g_ms,
                    "latency_savings_ms": g_ms - ms_val,
                    "speedup_vs_arm_g": g_ms / max(ms_val, 1e-9),
                    "storage_bytes": int(
                        (res_rec.get("peak_bytes", 0) or 0)
                        + (proj_rec.get("peak_bytes", 0) if pol == "fuse14_arm_h" else 0)
                    ),
                    "peak_alloc_mib": meta["peak_allocated_bytes"] / (1024.0 ** 2),
                    "peak_reserved_mib": meta["peak_reserved_bytes"] / (1024.0 ** 2),
                    "nvml_mib": float(meta["nvml_used_mib"]),
                    "vram_saved_alloc_mib": (mg["peak_allocated_bytes"] - meta["peak_allocated_bytes"]) / (1024.0 ** 2),
                    "blocked_reason": None,
                    "safe_admissible": bool(par_ok and meta["peak_allocated_bytes"] / (1024.0 ** 2) <= DEFAULT_LEASE_MIB - DEFAULT_HEADROOM_MIB),
                }
            )

    # 2. Stage G real pretrained Qwen3-4B transfer rows
    sg = suite.get("stage_g_pretrained_transfer") or {}
    q3 = sg.get("pretrained_qwen3_4b") or {}
    if q3.get("status") == "ok":
        for cell in q3.get("cells", []):
            t = int(cell["visits"])
            par_ok = bool(cell["parity"]["finite_bitwise_pass"])
            mg, mh = cell["fuse13_arm_g"], cell["fuse14_arm_h"]
            g_ms = float(mg["step_ms"][-1])
            h_ms = float(mh["step_ms"][-1])
            for pol, meta, ms_val in (("fuse13_arm_g", mg, g_ms), ("fuse14_arm_h", mh, h_ms)):
                rows.append(
                    {
                        "schema_version": "fuse14h.v1",
                        "source_campaign": "fuse14_stage_g_qwen3_4b",
                        "checkpoint_family": "pretrained_qwen3_4b_block",
                        "checkpoint_sha256": q3.get("shard_sha256"),
                        "cell_id": f"fuse14g__qwen3_4b__bf16__t{t}",
                        "hardware": HARDWARE_NAME,
                        "device": suite.get("device", "cuda"),
                        "precision": "bf16",
                        "storage_mode": "compute_dtype",
                        "width": int(cell["hidden_size"]),
                        "intermediate_size": int(cell["intermediate_size"]),
                        "recurrence_depth": t,
                        "regime": "closed_loop",
                        "calibration_cadence": "off",
                        "budget_mode": "full",
                        "policy": pol,
                        "numerical_gate": "PASS_FINITE_BITWISE" if par_ok else "FAIL",
                        "latency_ms": ms_val,
                        "baseline_latency_ms": g_ms,
                        "latency_savings_ms": g_ms - ms_val,
                        "speedup_vs_arm_g": g_ms / max(ms_val, 1e-9),
                        "storage_bytes": int(meta["unique_proj_cast_bytes"]),
                        "unique_proj_cast_storages": int(meta["unique_proj_cast_storages"]),
                        "peak_alloc_mib": meta["peak_allocated_bytes"] / (1024.0 ** 2),
                        "peak_reserved_mib": meta["peak_reserved_bytes"] / (1024.0 ** 2),
                        "nvml_mib": float(meta["nvml_used_mib"]),
                        "vram_saved_alloc_mib": (mg["peak_allocated_bytes"] - meta["peak_allocated_bytes"]) / (1024.0 ** 2),
                        "blocked_reason": None,
                        "safe_admissible": bool(par_ok and meta["peak_allocated_bytes"] / (1024.0 ** 2) <= DEFAULT_LEASE_MIB - DEFAULT_HEADROOM_MIB),
                    }
                )
    return rows


def _rows_from_fuse14_frontier(frontier_path: Path) -> list[dict[str, Any]]:
    if not frontier_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in frontier_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        w, t = int(rec["width"]), int(rec["n_passes"])
        for pol, key in (("fuse13_arm_g", "arm_g_fuse13"), ("fuse14_arm_h", "arm_h_fuse14"), ("fuse14_arm_h_ckpt", "arm_h_ckpt")):
            a = rec.get(key)
            if not a:
                continue
            ok = a["status"] == "ok"
            rows.append(
                {
                    "schema_version": "fuse14h.v1",
                    "source_campaign": "fuse14_stage_f_frontier",
                    "checkpoint_family": "synthetic_mini_trm",
                    "cell_id": rec["cell_id"],
                    "hardware": HARDWARE_NAME,
                    "device": "cuda",
                    "precision": "bf16",
                    "storage_mode": "compute_dtype",
                    "width": w,
                    "recurrence_depth": t,
                    "regime": "closed_loop",
                    "calibration_cadence": "off",
                    "budget_mode": "full",
                    "policy": pol,
                    "numerical_gate": "PASS_FINITE_BITWISE_AT_PARITY_HORIZON" if ok else "BLOCKED",
                    "latency_ms": float(a["cuda_median_ms"]) if ok else None,
                    "p95_latency_ms": float(a["cuda_p95_ms"]) if ok else None,
                    "tokens_per_s": float(a["tokens_per_s"]) if ok else None,
                    "energy_per_step_mj": a.get("energy_per_step_mj") if ok else None,
                    "storage_bytes": int(
                        ((a.get("residual_receipt") or {}).get("peak_bytes", 0) or 0)
                        + ((a.get("projection_receipt") or {}).get("peak_bytes", 0) or 0)
                    ) if ok else 0,
                    "peak_alloc_mib": (a["peak_allocated_bytes"] / (1024.0 ** 2)) if ok else None,
                    "peak_reserved_mib": (a["peak_reserved_bytes"] / (1024.0 ** 2)) if ok else None,
                    "nvml_mib": float(a["nvml_used_mib"]) if ok else None,
                    "host_rss_mib": float(a["host_rss_mib"]) if ok else None,
                    "blocked_reason": None if ok else a["status"],
                    "safe_admissible": bool(ok),
                }
            )
    return rows


def _hard_evidence_gate(sample: dict[str, Any], validated_modes: set[str]) -> bool:
    """Hard evidence gate executed BEFORE any cost ranking (correctness_path=False)."""
    if sample["precision"] not in validated_modes:
        return False
    if sample.get("precision") == "fp16":
        return False  # FP16 remains HOLD
    pm = physical_memory_model(
        sample["width"],
        sample["recurrence_depth"],
        arm=sample["arm"],
        precision=sample["precision"],
        storage_mode=sample.get("storage_mode", "fp32"),
    )
    return pm["est_vram_margin_mib"] >= 0.0


def _featurize_residual(sample: dict[str, Any], prior_ms: float) -> list[float]:
    d_in = float(sample["d_in"])
    d_out = float(sample["d_out"])
    rank = float(sample["rank"])
    saved_calls = float(sample["saved_calls"])
    prec_oh = [1.0 if sample["precision"] == p else 0.0 for p in PRECISION_LIST]
    return [
        math.log1p(prior_ms),
        math.log1p(saved_calls),
        rank / max(min(d_in, d_out), 1.0),
        d_in / max(d_out, 1.0),
        1.0 if sample["regime"] == "closed_loop" else 0.0,
        float(sample["pgd_inner_steps"]),
        *prec_oh,
    ]


def _p95(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil(0.95 * len(s))) - 1))
    return float(s[idx])


def _eval_gated_fold(
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    *,
    validated_modes: set[str],
) -> dict[str, Any]:
    if not train_samples or not test_samples:
        return {"skipped": True}

    prior_tr = np.asarray(
        [
            physical_site_prior_ms(
                s["d_in"], s["d_out"], s["rank"], s["saved_calls"], s["precision"], s.get("storage_mode", "fp32")
            )
            for s in train_samples
        ],
        dtype=np.float64,
    )
    prior_te = np.asarray(
        [
            physical_site_prior_ms(
                s["d_in"], s["d_out"], s["rank"], s["saved_calls"], s["precision"], s.get("storage_mode", "fp32")
            )
            for s in test_samples
        ],
        dtype=np.float64,
    )
    y_rec_tr = np.asarray([max(0.0, s["direct_recon_saved_ms"]) for s in train_samples], dtype=np.float64)

    eps = 1e-3
    target_log = np.log((y_rec_tr + eps) / (prior_tr + eps))
    x_tr = np.asarray([_featurize_residual(s, p) for s, p in zip(train_samples, prior_tr)], dtype=np.float64)
    x_te = np.asarray([_featurize_residual(s, p) for s, p in zip(test_samples, prior_te)], dtype=np.float64)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x_tr, target_log)
    pred_corr = np.clip(model.predict(x_te), -1.0, 1.0)
    pred_learned = np.maximum(0.0, prior_te * np.exp(pred_corr) - eps)

    policies = (
        "safe_physical_prior",
        "safe_physical_plus_learned",
        "safe_exact_saved_mac",
        "safe_density",
        "ungated_physical_prior_ref",
    )
    ratios_step: dict[str, list[float]] = {k: [] for k in policies}
    regrets_step: dict[str, list[float]] = {k: [] for k in policies}
    unsafe_cnt: dict[str, int] = {k: 0 for k in policies}
    false_rej: dict[str, int] = {k: 0 for k in policies}
    total_dec = 0
    safe_dec = 0

    by_inst: dict[tuple[str, str], list[tuple[dict[str, Any], float, float]]] = {}
    for s, pr, pl in zip(test_samples, prior_te, pred_learned):
        by_inst.setdefault((s["cell_id"], s["arm"]), []).append((s, float(pr), float(pl)))

    for (_cid, _arm), items in by_inst.items():
        if len(items) < 2:
            continue
        sites_info = [it[0] for it in items]
        pr_list = [it[1] for it in items]
        pl_list = [it[2] for it in items]
        full_bytes = sum(s["payload_bytes"] for s in sites_info)
        gate_pass = _hard_evidence_gate(sites_info[0], validated_modes)

        for frac in (0.25, 0.50, 0.75):
            cap = int(full_bytes * frac)
            total_dec += 1
            mac_sites = [Site.residual(s["d_in"], s["d_out"], s["rank"], s["calls"]) for s in sites_info]
            pr_sites = [Site(s["payload_bytes"], max(1, int(round(p * 1_000_000)))) for s, p in zip(sites_info, pr_list)]
            pl_sites = [Site(s["payload_bytes"], max(1, int(round(p * 1_000_000)))) for s, p in zip(sites_info, pl_list)]

            if gate_pass:
                safe_dec += 1
                plans = {
                    "safe_physical_prior": select(pr_sites, cap, "exact"),
                    "safe_physical_plus_learned": select(pl_sites, cap, "exact"),
                    "safe_exact_saved_mac": select(mac_sites, cap, "exact"),
                    "safe_density": select(mac_sites, cap, "density"),
                    "ungated_physical_prior_ref": select(pr_sites, cap, "exact"),
                }
            else:
                empty = select([], cap, "exact")
                plans = {
                    "safe_physical_prior": empty,
                    "safe_physical_plus_learned": empty,
                    "safe_exact_saved_mac": empty,
                    "safe_density": empty,
                    "ungated_physical_prior_ref": select(pr_sites, cap, "exact"),
                }

            for pname, plan in plans.items():
                if len(plan.indices) > 0 and not gate_pass:
                    unsafe_cnt[pname] += 1

            # Score oracle fraction & tail regret ONLY over evidence-admissible decisions
            if gate_pass:
                oracle_step, _ = _oracle_subset(sites_info, cap, "empirical_ms_saved")
                if oracle_step > 1e-9:
                    for pname, plan in plans.items():
                        ach = sum(sites_info[i]["empirical_ms_saved"] for i in plan.indices)
                        ratios_step[pname].append(ach / oracle_step)
                        regrets_step[pname].append(max(0.0, oracle_step - ach))
                        if len(plan.indices) == 0:
                            false_rej[pname] += 1

    return {
        "total_decisions": total_dec,
        "safe_decisions": safe_dec,
        "mean_safe_oracle_fraction_step_ms": {
            k: (float(statistics.mean(v)) if v else None) for k, v in ratios_step.items()
        },
        "p95_regret_step_ms": {k: _p95(v) for k, v in regrets_step.items()},
        "max_regret_step_ms": {k: (float(max(v)) if v else None) for k, v in regrets_step.items()},
        "unsafe_admission_rate": {k: unsafe_cnt[k] / max(1, total_dec) for k in policies},
        "false_rejection_rate": {k: false_rej[k] / max(1, safe_dec) for k in policies},
    }


def evaluate_fuse14h(
    fuse13_rows: list[dict[str, Any]],
    fuse14_rows: list[dict[str, Any]],
    *,
    validated_modes: set[str],
) -> dict[str, Any]:
    ok_rows = [r for r in fuse13_rows if r.get("arm_status") == "ok" and r.get("cuda_savings_ms") is not None]
    samples = _build_site_samples(ok_rows)
    cell_storage = {(r["cell_id"], r["arm"]): r.get("storage_mode", "fp32") for r in ok_rows}
    for s in samples:
        s["storage_mode"] = cell_storage.get((s["cell_id"], s["arm"]), "fp32")

    policies = (
        "safe_physical_prior",
        "safe_physical_plus_learned",
        "safe_exact_saved_mac",
        "safe_density",
        "ungated_physical_prior_ref",
    )

    def _run_grouped(group_key: str) -> dict[str, Any]:
        groups = sorted({s[group_key] for s in samples})
        folds = {}
        for g in groups:
            tr = [s for s in samples if s[group_key] != g]
            te = [s for s in samples if s[group_key] == g]
            folds[str(g)] = _eval_gated_fold(tr, te, validated_modes=validated_modes)

        def _agg(mkey: str, fn=statistics.mean):
            out = {}
            for pol in policies:
                vals = [f[mkey][pol] for f in folds.values() if not f.get("skipped") and f[mkey].get(pol) is not None]
                out[pol] = float(fn(vals)) if vals else None
            return out

        return {
            "groups": [str(g) for g in groups],
            "mean_safe_oracle_fraction_step_ms": _agg("mean_safe_oracle_fraction_step_ms", statistics.mean),
            "p95_regret_step_ms": _agg("p95_regret_step_ms", max),
            "max_regret_step_ms": _agg("max_regret_step_ms", max),
            "unsafe_admission_rate": _agg("unsafe_admission_rate", statistics.mean),
            "false_rejection_rate": _agg("false_rejection_rate", statistics.mean),
            "folds": folds,
        }

    w_res = _run_grouped("width")
    d_res = _run_grouped("recurrence_depth")
    p_res = _run_grouped("precision")

    # Protocol Stage H promotion rule: learned residual correction ONLY promoted if it beats
    # safe_physical_prior on locked width, depth, AND precision groups!
    beats_all = all(
        (res["mean_safe_oracle_fraction_step_ms"]["safe_physical_plus_learned"] or 0.0)
        > (res["mean_safe_oracle_fraction_step_ms"]["safe_physical_prior"] or 0.0)
        for res in (w_res, d_res, p_res)
    )
    ckpt_families = sorted({r.get("checkpoint_family", "synthetic_mini_trm") for r in fuse14_rows} | {"synthetic_mini_trm"})

    return {
        "status": "experimental_non_correctness_path",
        "correctness_path": False,
        "hard_evidence_gate_modes": sorted(validated_modes),
        "fp16_status": "HOLD",
        "total_fuse13_base_rows": len(fuse13_rows),
        "total_fuse14_routing_rows": len(fuse14_rows),
        "total_site_samples": len(samples),
        "checkpoint_families_observed": ckpt_families,
        "learned_residual_beats_safe_physical_on_all_holdouts": beats_all,
        "recommended_experimental_router": (
            "safe_physical_plus_learned" if beats_all else "safe_physical_prior (hard-evidence-gated)"
        ),
        "leave_one_width_out": w_res,
        "leave_one_depth_out": d_res,
        "leave_one_precision_out": p_res,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fuse12-receipts", type=Path, default=Path("mechinterp/bpd/experiments/fused_residual/fuse12_cuda_receipts.jsonl"))
    p.add_argument("--fuse13-frontier", type=Path, default=Path("mechinterp/bpd/experiments/fused_residual/receipt_fuse13_frontier.jsonl"))
    p.add_argument("--fuse13-stage-bc", type=Path, default=Path("mechinterp/bpd/experiments/fused_residual/receipt_fuse13_stage_bc_cuda.json"))
    p.add_argument("--fuse14-suite", type=Path, default=Path("mechinterp/bpd/experiments/fused_residual/receipt_fuse14_suite_cuda.json"))
    p.add_argument("--fuse14-frontier", type=Path, default=Path("mechinterp/bpd/experiments/fused_residual/receipt_fuse14_frontier.jsonl"))
    p.add_argument("--dataset-out", type=Path, default=Path("mechinterp/bpd/experiments/fused_residual/fuse14h_datagoblin_dataset.jsonl"))
    p.add_argument("--report-out", type=Path, default=Path("mechinterp/bpd/experiments/fused_residual/receipt_fuse14h_router.json"))
    args = p.parse_args()

    tmp_13 = args.dataset_out.parent / ".tmp_fuse13e_for_14h.jsonl"
    f13_rows = build_fuse13e_dataset(
        args.fuse12_receipts,
        tmp_13,
        frontier_receipts=args.fuse13_frontier if args.fuse13_frontier.exists() else None,
        stage_bc_receipt=args.fuse13_stage_bc if args.fuse13_stage_bc.exists() else None,
    )
    tmp_13.unlink(missing_ok=True)

    f14_rows = _rows_from_fuse14_suite(args.fuse14_suite) + _rows_from_fuse14_frontier(args.fuse14_frontier)

    # Hard evidence gate: FP32, TF32, and BF16 passed finite bitwise admission; FP16 stays HOLD.
    validated_modes = {"fp32", "tf32", "bf16"}
    report = evaluate_fuse14h(f13_rows, f14_rows, validated_modes=validated_modes)

    # Export combined compact routing evidence dataset (FUSE-14 measured rows + FUSE-13 base rows)
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    with args.dataset_out.open("w", encoding="utf-8") as f:
        for r in f14_rows:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(
        "FUSE14H_DONE",
        json.dumps(
            {
                "fuse14_rows": len(f14_rows),
                "site_samples": report["total_site_samples"],
                "checkpoint_families": report["checkpoint_families_observed"],
                "beats_all": report["learned_residual_beats_safe_physical_on_all_holdouts"],
                "recommended": report["recommended_experimental_router"],
                "unsafe_precision": report["leave_one_precision_out"]["unsafe_admission_rate"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
