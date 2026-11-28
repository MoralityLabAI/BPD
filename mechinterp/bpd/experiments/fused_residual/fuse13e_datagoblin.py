"""FUSE-13 Stage E: DataGoblin physical cost model + compact learned admission correction.

Strictly experimental and outside the numerical correctness path.
Extends the FUSE-12C curriculum dataset with physical VRAM / live-residual / precision-cast /
lease-headroom accounting, fits a regularized log-residual correction on top of an explicit
physical cost prior, enforces safety abstention on predicted VRAM spillover or unvalidated
configurations, and compares against exact_saved_mac, density, first_fit, and FUSE-12C log-Ridge
across grouped width, depth, precision/device, and hardware/checkpoint holdouts.
"""
from __future__ import annotations

import argparse
import itertools
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
from .bench_fuse13_frontier import preflight_estimate_mib
from .fuse12c_datagoblin import (
    PRECISION_LIST,
    _build_site_samples,
    _featurize as _fuse12c_featurize,
    _oracle_subset,
    export_datagoblin_dataset as export_fuse12c_dataset,
)

# RTX 5080 Laptop GPU (80 W WDDM lease) physical reference constants for transparent prior
DEFAULT_LEASE_MIB = 12288.0
DEFAULT_HEADROOM_MIB = 2048.0
SAFE_BUDGET_MIB = DEFAULT_LEASE_MIB - DEFAULT_HEADROOM_MIB  # 10,240 MiB

# Effective sustained throughput priors (GFLOP/s and GB/s) per precision mode
PHYS_GFLOPS = {"fp32": 14_000.0, "tf32": 32_000.0, "bf16": 45_000.0, "fp16": 45_000.0}
PHYS_MEM_GBS = 420.0
PHYS_LAUNCH_MS = 0.012


def physical_site_prior_ms(
    d_in: int,
    d_out: int,
    rank: int,
    saved_calls: int,
    precision: str,
    storage_mode: str = "fp32",
) -> float:
    """Analytical physical prior for per-site GPU time saved (ms) without fitted parameters."""
    if saved_calls <= 0:
        return 0.0
    elem_bytes = 2 if (storage_mode == "compute_dtype" and precision in ("bf16", "fp16")) else 4
    gflops = PHYS_GFLOPS.get(precision, PHYS_GFLOPS["fp32"])
    # 1. Compute savings: saved GEMMs (V @ U: 2 * d_in * rank * d_out FLOPs per call)
    mac_flops = 2.0 * float(d_in) * float(rank) * float(d_out)
    compute_ms = (mac_flops / (gflops * 1e6)) * saved_calls
    # 2. Memory traffic savings: writing/reading materialized [d_out, d_in] residual + W_target subtraction
    res_bytes = float(d_in) * float(d_out) * 4.0
    mem_ms = ((2.5 * res_bytes) / (PHYS_MEM_GBS * 1e6)) * saved_calls
    # 3. Precision conversion savings when compute_dtype avoids repeated FP32->low casts in F.linear
    cast_ms = (
        ((float(d_in) * float(d_out) * (4.0 + elem_bytes)) / (PHYS_MEM_GBS * 1e6)) * saved_calls
        if (storage_mode == "compute_dtype" and precision in ("bf16", "fp16"))
        else 0.0
    )
    # 4. Kernel launch overhead (2 kernels per saved reconstruction: matmul + sub)
    launch_ms = PHYS_LAUNCH_MS * saved_calls
    return max(1e-6, max(compute_ms, mem_ms) + cast_ms + launch_ms)


def physical_memory_model(
    width: int,
    recurrence_depth: int,
    *,
    arm: str,
    precision: str,
    storage_mode: str,
    lease_mib: float = DEFAULT_LEASE_MIB,
    headroom_mib: float = DEFAULT_HEADROOM_MIB,
) -> dict[str, float]:
    """Physical peak VRAM & spill-risk model for an arm under the active lease."""
    # Map fp32 storage_mode under bf16/fp16 to native 4-byte storage for live residual estimate
    eff_mode = precision if storage_mode == "compute_dtype" else "fp32"
    est_native = preflight_estimate_mib(width, recurrence_depth, arm="A", mode=precision)
    est_cand = preflight_estimate_mib(width, recurrence_depth, arm=("A" if arm == "A" else "G"), mode=eff_mode)
    safe_budget = max(512.0, lease_mib - headroom_mib)
    margin_mib = safe_budget - est_cand["total_est_mib"]
    # Smooth logistic spill-risk score centered at 0 MiB margin (width 512 MiB)
    spill_risk = 1.0 / (1.0 + math.exp(max(-30.0, min(30.0, margin_mib / 512.0))))
    return {
        "est_live_residuals_native_mib": est_native["live_residuals_mib"],
        "est_live_residuals_cached_mib": est_cand["live_residuals_mib"],
        "est_activation_opt_overhead_mib": round(
            est_cand["base_weights_opt_mib"] + est_cand["activations_mib"], 2
        ),
        "est_total_vram_mib": est_cand["total_est_mib"],
        "safe_lease_budget_mib": safe_budget,
        "est_vram_margin_mib": round(margin_mib, 2),
        "est_spill_risk": round(spill_risk, 6),
    }


def _rows_from_fuse13_frontier(frontier_path: Path) -> list[dict[str, Any]]:
    """Extract normalized FUSE-13E dataset rows from Stage D frontier receipts."""
    if not frontier_path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in frontier_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind") != "frontier_scaling":
            continue
        mode = rec["mode"]
        width = int(rec["width"])
        depth = int(rec["n_passes"])
        cid = rec["cell_id"]
        arm_a = rec.get("arm_a", {})
        arm_g = rec.get("arm_g", {})
        elem_bytes = 2 if mode in ("bf16", "fp16") else 4
        raw_inter = int(round(4 * width * 2.0 / 3.0))
        inter = ((raw_inter + 255) // 256) * 256 if width >= 256 else width * 3
        ranks = [max(2, width // 32), max(4, width // 16), max(6, 3 * width // 32), max(8, width // 8)]
        site_dims = {
            "layers.0.o_proj": {"d_in": width, "d_out": width, "rank": ranks[0], "payload_bytes": width * width * elem_bytes},
            "layers.0.down_proj": {"d_in": inter, "d_out": width, "rank": ranks[1], "payload_bytes": inter * width * elem_bytes},
            "layers.1.o_proj": {"d_in": width, "d_out": width, "rank": ranks[2], "payload_bytes": width * width * elem_bytes},
            "layers.1.down_proj": {"d_in": inter, "d_out": width, "rank": ranks[3], "payload_bytes": inter * width * elem_bytes},
        }
        full_payload = sum(v["payload_bytes"] for v in site_dims.values())
        ref_cuda = float(arm_a["cuda_median_ms"]) if arm_a.get("status") == "ok" else None
        ref_wall = float(arm_a["wall_median_ms"]) if arm_a.get("status") == "ok" else None
        ref_alloc = int(arm_a["peak_allocated_bytes"]) if arm_a.get("status") == "ok" else None
        ref_energy = arm_a.get("energy_per_step_mj") if arm_a.get("status") == "ok" else None

        for arm_id, a in (("A", arm_a), ("G", arm_g)):
            status = a.get("status", "unknown")
            storage_mode = "compute_dtype" if (arm_id == "G" and mode in ("bf16", "fp16")) else "fp32"
            phys = physical_memory_model(width, depth, arm=arm_id, precision=mode, storage_mode=storage_mode)
            is_ok = status == "ok"
            cand_cuda = float(a["cuda_median_ms"]) if is_ok else None
            cand_wall = float(a["wall_median_ms"]) if is_ok else None
            cand_alloc = int(a["peak_allocated_bytes"]) if is_ok else None
            cand_energy = a.get("energy_per_step_mj") if is_ok else None
            speedup = (ref_cuda / cand_cuda) if (ref_cuda and cand_cuda) else None
            out.append(
                {
                    "schema_version": "fuse13e.v1",
                    "source_campaign": "fuse13_frontier",
                    "dataset_family": "synthetic_mini_trm",
                    "cell_id": cid,
                    "tier": "frontier",
                    "hardware": "NVIDIA GeForce RTX 5080 Laptop GPU",
                    "device": "cuda",
                    "precision": mode,
                    "storage_mode": storage_mode,
                    "width": width,
                    "recurrence_depth": depth,
                    "regime": "closed_loop",
                    "pgd_inner_steps": 2,
                    "calibration_cadence": "off",
                    "budget_mode": "full",
                    "cache_budget_bytes": full_payload,
                    "full_payload_bytes": full_payload,
                    "n_sites": len(site_dims),
                    "site_dims": site_dims,
                    "site_recon_ms": {},
                    "component_ranks": ranks,
                    "arm": arm_id,
                    "arm_label": "native" if arm_id == "A" else "fuse09_10_11_compute_dtype",
                    "arm_status": status,
                    "admitted_sites": list(site_dims.keys()) if arm_id == "G" else [],
                    "admitted_payload_bytes": int(a.get("payload_peak_bytes", 0)) if is_ok else 0,
                    "reconstruction_calls_native": 0,
                    "reconstruction_calls_cached": 0,
                    "reconstruction_savings": 0,
                    "planned_saved_macs": sum(
                        (depth * 4 - 1) * sd["d_in"] * sd["rank"] * sd["d_out"] for sd in site_dims.values()
                    )
                    if arm_id == "G"
                    else 0,
                    "baseline_wall_ms": ref_wall,
                    "measured_wall_ms": cand_wall,
                    "wall_savings_ms": (ref_wall - cand_wall) if (ref_wall is not None and cand_wall is not None) else None,
                    "baseline_cuda_ms": ref_cuda,
                    "measured_cuda_ms": cand_cuda,
                    "cuda_savings_ms": (ref_cuda - cand_cuda) if (ref_cuda is not None and cand_cuda is not None) else None,
                    "speedup_cuda": speedup,
                    "baseline_peak_allocated_bytes": ref_alloc,
                    "peak_allocated_bytes": cand_alloc,
                    "vram_saved_bytes": (ref_alloc - cand_alloc) if (ref_alloc is not None and cand_alloc is not None) else None,
                    "peak_reserved_bytes": int(a.get("peak_reserved_bytes", 0)) if is_ok else None,
                    "nvml_used_mib": float(a.get("nvml_used_mib", 0.0)) if is_ok else None,
                    "host_rss_mib": float(a.get("host_rss_mib", 0.0)) if is_ok else None,
                    "mean_power_w": a.get("mean_power_w") if is_ok else None,
                    "energy_per_step_mj": cand_energy,
                    "energy_savings_mj": (
                        float(ref_energy - cand_energy)
                        if (ref_energy is not None and cand_energy is not None)
                        else None
                    ),
                    **phys,
                    "safe_admissible": bool(is_ok and phys["est_vram_margin_mib"] >= 0.0),
                }
            )
    return out


def build_fuse13e_dataset(
    fuse12_receipts: Path,
    dataset_out: Path,
    *,
    frontier_receipts: Path | None = None,
    stage_bc_receipt: Path | None = None,
) -> list[dict[str, Any]]:
    """Build extended FUSE-13E DataGoblin dataset combining FUSE-12 + FUSE-13 receipts."""
    tmp_12 = dataset_out.parent / ".tmp_fuse12c_rows.jsonl"
    base_rows = export_fuse12c_dataset(fuse12_receipts, tmp_12)
    try:
        tmp_12.unlink(missing_ok=True)
    except OSError:
        pass

    # Determine which precision modes passed Stage C finite bitwise admission (if receipt available)
    validated_modes = {"fp32", "tf32"}
    if stage_bc_receipt and stage_bc_receipt.exists():
        bc = json.loads(stage_bc_receipt.read_text(encoding="utf-8"))
        for m in ("bf16", "fp16"):
            reg = bc.get("stage_c_stress_regression", {}).get(m, {})
            if reg.get("admission_pass"):
                validated_modes.add(m)

    extended: list[dict[str, Any]] = []
    for r in base_rows:
        width = int(r["width"])
        depth = int(r["recurrence_depth"])
        prec = str(r["precision"])
        arm = str(r["arm"])
        storage_mode = "fp32"  # FUSE-12 stored fp32 payloads
        phys = physical_memory_model(width, depth, arm=arm, precision=prec, storage_mode=storage_mode)
        peak_mib = float(r.get("peak_allocated_bytes", 0)) / (1024.0 ** 2)
        safe_vram = (peak_mib <= phys["safe_lease_budget_mib"]) and (phys["est_vram_margin_mib"] >= 0.0)
        row = {
            **r,
            "schema_version": "fuse13e.v1",
            "source_campaign": "fuse12",
            "dataset_family": "synthetic_mini_trm",
            "storage_mode": storage_mode,
            "arm_status": "ok",
            **phys,
            "safe_admissible": bool(safe_vram and prec in validated_modes),
        }
        extended.append(row)

    if frontier_receipts is not None and frontier_receipts.exists():
        extended.extend(_rows_from_fuse13_frontier(frontier_receipts))

    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    with dataset_out.open("w", encoding="utf-8") as f:
        for row in extended:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    return extended


def _featurize_correction(sample: dict[str, Any], prior_ms: float) -> list[float]:
    """Scale-invariant compact features for the learned log-ratio correction to the physical prior."""
    d_in = float(sample["d_in"])
    d_out = float(sample["d_out"])
    rank = float(sample["rank"])
    saved_calls = float(sample["saved_calls"])
    prec_oh = [1.0 if sample["precision"] == p else 0.0 for p in PRECISION_LIST]
    is_compute_dt = 1.0 if sample.get("storage_mode") == "compute_dtype" else 0.0
    return [
        math.log1p(prior_ms),
        math.log1p(saved_calls),
        rank / max(min(d_in, d_out), 1.0),
        d_in / max(d_out, 1.0),
        1.0 if sample["regime"] == "closed_loop" else 0.0,
        float(sample["pgd_inner_steps"]),
        is_compute_dt,
        *prec_oh,
    ]


def _calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 5) -> dict[str, float]:
    """Compute Brier score and Expected Calibration Error (ECE)."""
    y = y_true.astype(np.float64)
    p = np.clip(probs.astype(np.float64), 0.0, 1.0)
    brier = float(np.mean((p - y) ** 2)) if len(y) else 0.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    if n > 0:
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
            cnt = int(np.sum(mask))
            if cnt > 0:
                ece += (cnt / n) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return {"brier_score": round(brier, 6), "ece": round(ece, 6)}


def _eval_fold(
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    *,
    validated_modes: set[str],
) -> dict[str, Any]:
    if not train_samples or not test_samples:
        return {"skipped": True}

    # Physical priors (zero training required)
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

    # FUSE-12C baseline: log-Ridge predicting direct_recon_saved_ms from scratch
    x12_tr = np.asarray([_fuse12c_featurize(s) for s in train_samples], dtype=np.float64)
    x12_te = np.asarray([_fuse12c_featurize(s) for s in test_samples], dtype=np.float64)
    y_rec_tr = np.asarray([max(0.0, s["direct_recon_saved_ms"]) for s in train_samples], dtype=np.float64)
    y_rec_te = np.asarray([max(0.0, s["direct_recon_saved_ms"]) for s in test_samples], dtype=np.float64)

    m12 = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    m12.fit(x12_tr, np.log1p(y_rec_tr))
    pred_12c = np.expm1(np.clip(m12.predict(x12_te), 0.0, 20.0))

    # FUSE-13E: compact learned log-ratio correction to the physical prior
    eps = 1e-3
    target_log_corr = np.log((y_rec_tr + eps) / (prior_tr + eps))
    xc_tr = np.asarray([_featurize_correction(s, p) for s, p in zip(train_samples, prior_tr)], dtype=np.float64)
    xc_te = np.asarray([_featurize_correction(s, p) for s, p in zip(test_samples, prior_te)], dtype=np.float64)
    mc = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    mc.fit(xc_tr, target_log_corr)
    pred_log_corr = np.clip(mc.predict(xc_te), -1.0, 1.0)
    pred_phys_learned = np.maximum(0.0, prior_te * np.exp(pred_log_corr) - eps)

    policies = (
        "physical_plus_learned_gated",
        "physical_plus_learned",
        "physical_prior",
        "fuse12c_log_ridge",
        "exact_saved_mac",
        "density",
        "first_fit",
    )
    ratios_step: dict[str, list[float]] = {k: [] for k in policies}
    ratios_recon: dict[str, list[float]] = {k: [] for k in policies}
    regrets_step: dict[str, list[float]] = {k: [] for k in policies}
    regrets_recon: dict[str, list[float]] = {k: [] for k in policies}
    unsafe_counts: dict[str, int] = {k: 0 for k in policies}
    total_decisions = 0
    abstentions = 0

    by_inst: dict[tuple[str, str], list[tuple[dict[str, Any], float, float, float]]] = {}
    for s, p_pl, p_pr, p_12 in zip(test_samples, pred_phys_learned, prior_te, pred_12c):
        by_inst.setdefault((s["cell_id"], s["arm"]), []).append((s, float(p_pl), float(p_pr), float(p_12)))

    for (_cid, _arm), items in by_inst.items():
        if len(items) < 2:
            continue
        sites_info = [it[0] for it in items]
        p_pl_list = [it[1] for it in items]
        p_pr_list = [it[2] for it in items]
        p_12_list = [it[3] for it in items]
        full_bytes = sum(s["payload_bytes"] for s in sites_info)
        sample0 = sites_info[0]
        phys_mem = physical_memory_model(
            sample0["width"],
            sample0["recurrence_depth"],
            arm=sample0["arm"],
            precision=sample0["precision"],
            storage_mode=sample0.get("storage_mode", "fp32"),
        )
        is_safe_config = (phys_mem["est_vram_margin_mib"] >= 0.0) and (sample0["precision"] in validated_modes)

        for frac in (0.25, 0.50, 0.75):
            cap = int(full_bytes * frac)
            total_decisions += 1
            mac_sites = [Site.residual(s["d_in"], s["d_out"], s["rank"], s["calls"]) for s in sites_info]
            pl_sites = [Site(s["payload_bytes"], max(1, int(round(p * 1_000_000)))) for s, p in zip(sites_info, p_pl_list)]
            pr_sites = [Site(s["payload_bytes"], max(1, int(round(p * 1_000_000)))) for s, p in zip(sites_info, p_pr_list)]
            l12_sites = [Site(s["payload_bytes"], max(1, int(round(p * 1_000_000)))) for s, p in zip(sites_info, p_12_list)]

            pl_raw_plan = select(pl_sites, cap, "exact")
            if is_safe_config:
                pl_gated_plan = pl_raw_plan
            else:
                abstentions += 1
                pl_gated_plan = select([], cap, "exact")

            plans = {
                "physical_plus_learned_gated": pl_gated_plan,
                "physical_plus_learned": pl_raw_plan,
                "physical_prior": select(pr_sites, cap, "exact"),
                "fuse12c_log_ridge": select(l12_sites, cap, "exact"),
                "exact_saved_mac": select(mac_sites, cap, "exact"),
                "density": select(mac_sites, cap, "density"),
                "first_fit": select(mac_sites, cap, "first"),
            }
            for pname, plan in plans.items():
                if len(plan.indices) > 0 and not is_safe_config:
                    unsafe_counts[pname] += 1

            oracle_step, _ = _oracle_subset(sites_info, cap, "empirical_ms_saved")
            if oracle_step > 1e-9:
                for pname, plan in plans.items():
                    ach = sum(sites_info[i]["empirical_ms_saved"] for i in plan.indices)
                    ratios_step[pname].append(ach / oracle_step)
                    regrets_step[pname].append(max(0.0, oracle_step - ach))

            oracle_rec, _ = _oracle_subset(sites_info, cap, "direct_recon_saved_ms")
            if oracle_rec > 1e-9:
                for pname, plan in plans.items():
                    ach = sum(sites_info[i]["direct_recon_saved_ms"] for i in plan.indices)
                    ratios_recon[pname].append(ach / oracle_rec)
                    regrets_recon[pname].append(max(0.0, oracle_rec - ach))

    return {
        "n_train_sites": len(train_samples),
        "n_test_sites": len(test_samples),
        "prior_recon_mae_ms": float(np.mean(np.abs(y_rec_te - prior_te))),
        "physical_plus_learned_recon_mae_ms": float(np.mean(np.abs(y_rec_te - pred_phys_learned))),
        "fuse12c_log_ridge_recon_mae_ms": float(np.mean(np.abs(y_rec_te - pred_12c))),
        "budget_decisions": total_decisions,
        "abstention_rate": abstentions / max(1, total_decisions),
        "mean_oracle_fraction_step_ms": {
            k: (float(statistics.mean(v)) if v else None) for k, v in ratios_step.items()
        },
        "mean_oracle_fraction_recon_ms": {
            k: (float(statistics.mean(v)) if v else None) for k, v in ratios_recon.items()
        },
        "worst_case_regret_step_ms": {
            k: (float(max(v)) if v else None) for k, v in regrets_step.items()
        },
        "worst_case_regret_recon_ms": {
            k: (float(max(v)) if v else None) for k, v in regrets_recon.items()
        },
        "unsafe_admission_rate": {
            k: (unsafe_counts[k] / max(1, total_decisions)) for k in policies
        },
    }


def _eval_safe_admission_classifier(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    """Calibrated worth-caching + safety-abstention classifier grouped by `group_key`."""
    cand_rows = [
        r for r in rows
        if r.get("arm") != "A"
        and r.get("arm_status") == "ok"
        and r.get("speedup_cuda") is not None
        and r.get("admitted_payload_bytes", 0) > 0
    ]
    groups = sorted({str(r[group_key]) for r in cand_rows})
    if len(groups) < 2:
        return {"skipped": True, "reason": f"only {len(groups)} group(s) for {group_key}", "groups": groups}

    def _x(r: dict[str, Any]) -> list[float]:
        return [
            math.log1p(r["width"]),
            math.log1p(r["recurrence_depth"]),
            float(r["pgd_inner_steps"]),
            math.log1p(r["admitted_payload_bytes"]),
            math.log1p(max(0, r["planned_saved_macs"])),
            float(r["est_vram_margin_mib"]) / 1024.0,
            float(r["est_spill_risk"]),
            1.0 if r["regime"] == "closed_loop" else 0.0,
            1.0 if r["arm"] in ("F", "G") else 0.0,
            1.0 if r["calibration_cadence"] == "1" else 0.0,
            1.0 if r.get("storage_mode") == "compute_dtype" else 0.0,
            *[1.0 if r["precision"] == p else 0.0 for p in PRECISION_LIST],
        ]

    def _y(r: dict[str, Any]) -> int:
        vram_saved_mib = (r.get("vram_saved_bytes") or 0) / (1024.0 ** 2)
        beneficial = (float(r["speedup_cuda"]) >= 1.03) or (vram_saved_mib >= 256.0)
        return int(beneficial and bool(r.get("safe_admissible", True)))

    all_y, all_p, all_pred, all_safe = [], [], [], []
    folds = {}
    for g in groups:
        tr = [r for r in cand_rows if str(r[group_key]) != g]
        te = [r for r in cand_rows if str(r[group_key]) == g]
        if not tr or not te:
            continue
        y_tr = np.asarray([_y(r) for r in tr], dtype=np.int64)
        y_te = np.asarray([_y(r) for r in te], dtype=np.int64)
        safe_te = np.asarray([bool(r.get("safe_admissible", True)) for r in te], dtype=bool)
        if len(set(y_tr.tolist())) < 2:
            probs = np.full(len(y_te), float(y_tr[0]), dtype=np.float64)
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
            clf.fit(np.asarray([_x(r) for r in tr], dtype=np.float64), y_tr)
            probs = clf.predict_proba(np.asarray([_x(r) for r in te], dtype=np.float64))[:, 1]
        # Hard physical safety gate forces probability 0 when predicted VRAM spill or unvalidated
        gated_probs = np.where(safe_te, probs, 0.0)
        preds = (gated_probs >= 0.5).astype(np.int64)
        pos_mask = y_te == 1
        fnr = float(np.mean(preds[pos_mask] == 0)) if np.any(pos_mask) else 0.0
        unsafe = float(np.mean((preds == 1) & (~safe_te)))
        cal = _calibration_metrics(y_te, gated_probs)
        folds[g] = {
            "n_test": len(te),
            "accuracy": float(np.mean(preds == y_te)),
            "false_rejection_rate": fnr,
            "unsafe_admission_rate": unsafe,
            **cal,
        }
        all_y.extend(y_te.tolist())
        all_p.extend(gated_probs.tolist())
        all_pred.extend(preds.tolist())
        all_safe.extend(safe_te.tolist())

    y_arr = np.asarray(all_y, dtype=np.int64)
    p_arr = np.asarray(all_p, dtype=np.float64)
    pred_arr = np.asarray(all_pred, dtype=np.int64)
    safe_arr = np.asarray(all_safe, dtype=bool)
    pos_mask = y_arr == 1
    return {
        "criterion": "(speedup_cuda >= 1.03 OR vram_saved_mib >= 256) AND safe_admissible",
        "n_eval": int(len(y_arr)),
        "accuracy": float(np.mean(pred_arr == y_arr)) if len(y_arr) else None,
        "false_rejection_rate": float(np.mean(pred_arr[pos_mask] == 0)) if np.any(pos_mask) else 0.0,
        "unsafe_admission_rate": float(np.mean((pred_arr == 1) & (~safe_arr))) if len(y_arr) else 0.0,
        **_calibration_metrics(y_arr, p_arr),
        "folds": folds,
    }


def evaluate_fuse13e(rows: list[dict[str, Any]], *, validated_modes: set[str]) -> dict[str, Any]:
    """Run all Stage E grouped holdouts: width, depth, precision/device, and hardware/checkpoint."""
    ok_rows = [r for r in rows if r.get("arm_status") == "ok" and r.get("cuda_savings_ms") is not None]
    samples = _build_site_samples(ok_rows)
    # Attach storage_mode to each site sample
    cell_storage = {(r["cell_id"], r["arm"]): r.get("storage_mode", "fp32") for r in ok_rows}
    for s in samples:
        s["storage_mode"] = cell_storage.get((s["cell_id"], s["arm"]), "fp32")

    policies = (
        "physical_plus_learned_gated",
        "physical_plus_learned",
        "physical_prior",
        "fuse12c_log_ridge",
        "exact_saved_mac",
        "density",
        "first_fit",
    )

    def _run_grouped(group_key: str) -> dict[str, Any]:
        groups = sorted({s[group_key] for s in samples})
        if len(groups) < 2:
            return {
                "skipped": True,
                "reason": f"single observed {group_key} family: {groups}",
                "groups": groups,
            }
        folds = {}
        for g in groups:
            tr = [s for s in samples if s[group_key] != g]
            te = [s for s in samples if s[group_key] == g]
            folds[str(g)] = _eval_fold(tr, te, validated_modes=validated_modes)

        def _agg(metric_key: str, mode: str = "mean") -> dict[str, Any]:
            out = {}
            for pol in policies:
                vals = [
                    f[metric_key][pol]
                    for f in folds.values()
                    if not f.get("skipped") and f[metric_key].get(pol) is not None
                ]
                if not vals:
                    out[pol] = None
                elif mode == "max":
                    out[pol] = float(max(vals))
                else:
                    out[pol] = float(statistics.mean(vals))
            return out

        return {
            "groups": [str(g) for g in groups],
            "aggregate_oracle_fraction_step_ms": _agg("mean_oracle_fraction_step_ms", "mean"),
            "aggregate_oracle_fraction_recon_ms": _agg("mean_oracle_fraction_recon_ms", "mean"),
            "worst_case_regret_step_ms": _agg("worst_case_regret_step_ms", "max"),
            "worst_case_regret_recon_ms": _agg("worst_case_regret_recon_ms", "max"),
            "mean_unsafe_admission_rate": _agg("unsafe_admission_rate", "mean"),
            "folds": folds,
        }

    hw_families = sorted({r.get("hardware", "unknown") for r in rows})
    ds_families = sorted({r.get("dataset_family", "synthetic_mini_trm") for r in rows})

    return {
        "status": "experimental_non_correctness_path",
        "correctness_path": False,
        "validated_modes_for_safety_gate": sorted(validated_modes),
        "total_dataset_rows": len(rows),
        "total_site_samples": len(samples),
        "hardware_families": hw_families,
        "dataset_families": ds_families,
        "checkpoint_holdout_status": (
            "available"
            if len(ds_families) >= 2
            else "blocked_single_family_synthetic_only"
        ),
        "hardware_holdout_status": (
            "available"
            if len(hw_families) >= 2
            else "blocked_single_gpu_rtx5080"
        ),
        "leave_one_width_out": _run_grouped("width"),
        "leave_one_depth_out": _run_grouped("recurrence_depth"),
        "leave_one_precision_out": _run_grouped("precision"),
        "safe_admission_classifier": {
            "leave_one_width_out": _eval_safe_admission_classifier(rows, "width"),
            "leave_one_depth_out": _eval_safe_admission_classifier(rows, "recurrence_depth"),
            "leave_one_precision_out": _eval_safe_admission_classifier(rows, "precision"),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fuse12-receipts",
        type=Path,
        default=Path("mechinterp/bpd/experiments/fused_residual/fuse12_cuda_receipts.jsonl"),
    )
    p.add_argument("--frontier-receipts", type=Path, default=None)
    p.add_argument("--stage-bc-receipt", type=Path, default=None)
    p.add_argument(
        "--dataset-out",
        type=Path,
        default=Path("mechinterp/bpd/experiments/fused_residual/fuse13e_datagoblin_dataset.jsonl"),
    )
    p.add_argument(
        "--report-out",
        type=Path,
        default=Path("mechinterp/bpd/experiments/fused_residual/receipt_fuse13e_predictor.json"),
    )
    args = p.parse_args()

    validated_modes = {"fp32", "tf32"}
    if args.stage_bc_receipt and args.stage_bc_receipt.exists():
        bc = json.loads(args.stage_bc_receipt.read_text(encoding="utf-8"))
        for m in ("bf16", "fp16"):
            if bc.get("stage_c_stress_regression", {}).get(m, {}).get("admission_pass"):
                validated_modes.add(m)

    rows = build_fuse13e_dataset(
        args.fuse12_receipts,
        args.dataset_out,
        frontier_receipts=args.frontier_receipts,
        stage_bc_receipt=args.stage_bc_receipt,
    )
    report = evaluate_fuse13e(rows, validated_modes=validated_modes)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        "FUSE13E_DONE",
        json.dumps(
            {
                "dataset_rows": len(rows),
                "site_samples": report["total_site_samples"],
                "width_oracle_step": report["leave_one_width_out"].get("aggregate_oracle_fraction_step_ms"),
                "depth_oracle_step": report["leave_one_depth_out"].get("aggregate_oracle_fraction_step_ms"),
                "precision_oracle_step": report["leave_one_precision_out"].get("aggregate_oracle_fraction_step_ms"),
                "unsafe_width": report["leave_one_width_out"].get("mean_unsafe_admission_rate"),
                "clf_width_ece": report["safe_admission_classifier"]["leave_one_width_out"].get("ece"),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
