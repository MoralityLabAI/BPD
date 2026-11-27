"""FUSE-12C: DataGoblin curriculum dataset exporter & experimental admission-policy predictor.

Strictly experimental and outside the numerical correctness path.
Exports a compact machine-readable JSONL dataset of measured residual-caching
tradeoffs and evaluates a small regularized predictor against the exact saved-MAC
planner, first-fit, and density-based selection under identical byte budgets
with strict leave-one-width-family-out and leave-one-recurrence-depth-out holdouts.
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .admission import Site, select

PRECISION_LIST = ("fp32", "tf32", "bf16", "fp16")


def export_datagoblin_dataset(receipts_path: Path, dataset_path: Path) -> list[dict[str, Any]]:
    """Convert raw FUSE-12 JSONL benchmark receipts into compact DataGoblin curriculum rows."""
    rows: list[dict[str, Any]] = []
    if not receipts_path.exists():
        raise FileNotFoundError(f"Receipts file not found: {receipts_path}")

    for line in receipts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") != "ok":
            continue
        arms = rec.get("arms", {})
        if "A" not in arms:
            continue
        ref = arms["A"]
        site_dims = rec.get("site_dims", {})
        ranks = [int(site_dims[k]["rank"]) for k in sorted(site_dims)]
        hw = rec.get("nvml_pre", {}).get("name") or rec.get("device", "unknown")

        for arm_id, a in arms.items():
            recons = a.get("reconstructions", {})
            ref_cuda = float(ref["cuda_median_ms"])
            cand_cuda = float(a["cuda_median_ms"])
            ref_wall = float(ref["wall_median_ms"])
            cand_wall = float(a["wall_median_ms"])
            ref_energy = ref.get("energy_per_step_mj")
            cand_energy = a.get("energy_per_step_mj")
            energy_savings = (
                float(ref_energy - cand_energy)
                if (ref_energy is not None and cand_energy is not None)
                else None
            )
            row = {
                "schema_version": "fuse12c.v1",
                "source_dataset_kind": rec.get("source_dataset_kind", "synthetic_native_trainer"),
                "cell_id": rec["cell_id"],
                "tier": rec["tier"],
                "hardware": hw,
                "device": rec["device"],
                "precision": rec["precision_mode"],
                "width": int(rec["width"]),
                "recurrence_depth": int(rec["n_passes"]),
                "regime": rec["regime"],
                "pgd_inner_steps": int(rec["pgd_inner"]),
                "calibration_cadence": rec["cal_cadence"],
                "budget_mode": rec["budget_mode"],
                "cache_budget_bytes": int(rec["cap_bytes"]),
                "full_payload_bytes": int(rec["full_payload_bytes"]),
                "n_sites": len(site_dims),
                "site_dims": site_dims,
                "site_recon_ms": rec.get("site_recon_ms", {}),
                "component_ranks": ranks,
                "arm": arm_id,
                "arm_label": a["label"],
                "admitted_sites": a.get("admitted_sites", []),
                "admitted_payload_bytes": int(a.get("payload_peak_bytes", 0)),
                "reconstruction_calls_native": int(recons.get("native_total_reconstructions", 0)),
                "reconstruction_calls_cached": int(recons.get("cached_total_reconstructions", 0)),
                "reconstruction_savings": int(recons.get("total_saved_reconstructions", 0)),
                "planned_saved_macs": int(a.get("planned_saved_macs", 0)),
                "baseline_wall_ms": ref_wall,
                "measured_wall_ms": cand_wall,
                "wall_savings_ms": ref_wall - cand_wall,
                "baseline_cuda_ms": ref_cuda,
                "measured_cuda_ms": cand_cuda,
                "cuda_savings_ms": ref_cuda - cand_cuda,
                "speedup_cuda": float(a.get("speedup_cuda", 1.0)),
                "speedup_per_mib": float(a.get("speedup_per_mib", 0.0)),
                "ms_saved_per_mib": float(a.get("ms_saved_per_mib", 0.0)),
                "peak_allocated_bytes": int(a.get("peak_allocated_bytes", 0)),
                "peak_reserved_bytes": int(a.get("peak_reserved_bytes", 0)),
                "nvml_used_mib": float(a.get("nvml_used_mib", 0.0)),
                "host_rss_mib": float(a.get("host_rss_mib", 0.0)),
                "planner_overhead_ms": float(a.get("phases_ms", {}).get("planner_overhead_ms", 0.0)),
                "mean_power_w": a.get("mean_power_w"),
                "energy_per_step_mj": cand_energy,
                "energy_savings_mj": energy_savings,
            }
            rows.append(row)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    return rows


def _site_calls_for_arm(row: dict[str, Any]) -> int:
    arm = row["arm"]
    t = int(row["recurrence_depth"])
    eff_t = t if row["regime"] == "closed_loop" else max(1, t // 2)
    k = int(row["pgd_inner_steps"]) if row["regime"] == "closed_loop" else 0
    if arm == "B":
        return eff_t * (1 if row["regime"] == "closed_loop" else 2)
    if arm == "C":
        return eff_t * ((1 + k) if row["regime"] == "closed_loop" else 2)
    if arm == "D":
        return t
    if arm == "E":
        return eff_t * ((1 + k) if row["regime"] == "closed_loop" else 2) + eff_t
    if arm == "F":
        return eff_t * ((2 + k) if row["regime"] == "closed_loop" else 2)
    if arm == "G":
        return eff_t * ((2 + k) if row["regime"] == "closed_loop" else 2) + eff_t
    return 0


def _build_site_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per-site empirical GPU savings records from measured full/binding benchmark rows."""
    samples: list[dict[str, Any]] = []
    for r in rows:
        if r["arm"] == "A":
            continue
        admitted = r.get("admitted_sites", [])
        if not admitted:
            continue
        site_dims = r.get("site_dims", {})
        site_recon_ms = r.get("site_recon_ms", {})
        calls = _site_calls_for_arm(r)
        if calls <= 1:
            continue
        saved_calls = calls - 1
        # Empirical site reconstruction weight inside this cell
        raw_site_ms = {
            s: float(site_recon_ms.get(s, 0.0)) * saved_calls for s in admitted
        }
        total_raw = sum(raw_site_ms.values())
        measured_total = float(r["cuda_savings_ms"])
        # Reconcile empirical site contributions to match exact measured step savings
        for s in admitted:
            sd = site_dims[s]
            d_in, d_out, rank, payload = (
                int(sd["d_in"]),
                int(sd["d_out"]),
                int(sd["rank"]),
                int(sd["payload_bytes"]),
            )
            share = (raw_site_ms[s] / total_raw) if total_raw > 1e-12 else (1.0 / len(admitted))
            emp_ms = measured_total * share
            saved_macs = saved_calls * d_in * rank * d_out
            samples.append(
                {
                    "cell_id": r["cell_id"],
                    "arm": r["arm"],
                    "site": s,
                    "width": int(r["width"]),
                    "recurrence_depth": int(r["recurrence_depth"]),
                    "regime": r["regime"],
                    "pgd_inner_steps": int(r["pgd_inner_steps"]),
                    "precision": r["precision"],
                    "d_in": d_in,
                    "d_out": d_out,
                    "rank": rank,
                    "payload_bytes": payload,
                    "calls": calls,
                    "saved_calls": saved_calls,
                    "saved_macs": saved_macs,
                    "empirical_ms_saved": emp_ms,
                }
            )
    return samples


def _featurize(sample: dict[str, Any]) -> list[float]:
    d_in = float(sample["d_in"])
    d_out = float(sample["d_out"])
    rank = float(sample["rank"])
    saved_calls = float(sample["saved_calls"])
    payload = float(sample["payload_bytes"])
    saved_macs = float(sample["saved_macs"])
    prec_oh = [1.0 if sample["precision"] == p else 0.0 for p in PRECISION_LIST]
    return [
        math.log1p(d_in),
        math.log1p(d_out),
        math.log1p(rank),
        math.log1p(saved_calls),
        math.log1p(payload),
        math.log1p(saved_macs),
        d_in / max(d_out, 1.0),
        rank / max(min(d_in, d_out), 1.0),
        1.0 if sample["regime"] == "closed_loop" else 0.0,
        float(sample["pgd_inner_steps"]),
        *prec_oh,
    ]


def _oracle_subset_ms(sites_info: list[dict[str, Any]], cap: int) -> tuple[float, tuple[int, ...]]:
    best_val = -1e18
    best_idx: tuple[int, ...] = ()
    n = len(sites_info)
    for r in range(n + 1):
        for comb in itertools.combinations(range(n), r):
            sz = sum(sites_info[i]["payload_bytes"] for i in comb)
            if sz <= cap:
                val = sum(sites_info[i]["empirical_ms_saved"] for i in comb)
                if val > best_val:
                    best_val = val
                    best_idx = comb
    return max(0.0, best_val), best_idx


def _eval_policies_on_group(
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    if not train_samples or not test_samples:
        return {"skipped": True}

    x_train = np.asarray([_featurize(s) for s in train_samples], dtype=np.float64)
    y_train = np.asarray([s["empirical_ms_saved"] for s in train_samples], dtype=np.float64)
    x_test = np.asarray([_featurize(s) for s in test_samples], dtype=np.float64)
    y_test = np.asarray([s["empirical_ms_saved"] for s in test_samples], dtype=np.float64)

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae = float(np.mean(np.abs(y_test - y_pred)))
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # Group test samples by (cell_id, arm) where all sites are present to evaluate budget selection
    by_instance: dict[tuple[str, str], list[tuple[dict[str, Any], float]]] = {}
    for s, pred in zip(test_samples, y_pred):
        by_instance.setdefault((s["cell_id"], s["arm"]), []).append((s, float(pred)))

    policy_ratios: dict[str, list[float]] = {
        "learned_predictor": [],
        "exact_saved_mac": [],
        "first_fit": [],
        "density": [],
    }
    policy_ms_per_mib: dict[str, list[float]] = {k: [] for k in policy_ratios}
    instances_evaluated = 0

    for (_cid, _arm), items in by_instance.items():
        if len(items) < 2:
            continue
        sites_info = [it[0] for it in items]
        preds = [it[1] for it in items]
        full_bytes = sum(s["payload_bytes"] for s in sites_info)
        # Evaluate across binding budget fractions (25%, 50%, 75%)
        for frac in (0.25, 0.50, 0.75):
            cap = int(full_bytes * frac)
            oracle_ms, _ = _oracle_subset_ms(sites_info, cap)
            if oracle_ms <= 1e-9:
                continue
            instances_evaluated += 1

            mac_sites = [
                Site.residual(s["d_in"], s["d_out"], s["rank"], s["calls"]) for s in sites_info
            ]
            # Learned predictor uses predicted GPU ms saved scaled to non-negative integers for exact knapsack
            pred_sites = [
                Site(s["payload_bytes"], max(0, int(round(max(0.0, p) * 1_000_000))))
                for s, p in zip(sites_info, preds)
            ]

            plans = {
                "learned_predictor": select(pred_sites, cap, "exact"),
                "exact_saved_mac": select(mac_sites, cap, "exact"),
                "first_fit": select(mac_sites, cap, "first"),
                "density": select(mac_sites, cap, "density"),
            }
            for pname, plan in plans.items():
                achieved_ms = sum(sites_info[i]["empirical_ms_saved"] for i in plan.indices)
                policy_ratios[pname].append(achieved_ms / oracle_ms)
                used_mib = plan.size / (1024.0 * 1024.0)
                if used_mib > 0:
                    policy_ms_per_mib[pname].append(achieved_ms / used_mib)

    return {
        "n_train_sites": len(train_samples),
        "n_test_sites": len(test_samples),
        "predictor_mae_ms": mae,
        "predictor_r2": r2,
        "budget_instances_evaluated": instances_evaluated,
        "mean_oracle_fraction": {
            k: (float(statistics.mean(v)) if v else None) for k, v in policy_ratios.items()
        },
        "mean_ms_saved_per_mib": {
            k: (float(statistics.mean(v)) if v else None) for k, v in policy_ms_per_mib.items()
        },
    }


def evaluate_admission_predictor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run leave-one-width-family-out and leave-one-recurrence-depth-out evaluations."""
    samples = _build_site_samples(rows)
    widths = sorted({s["width"] for s in samples})
    depths = sorted({s["recurrence_depth"] for s in samples})

    width_folds = {}
    for w in widths:
        train = [s for s in samples if s["width"] != w]
        test = [s for s in samples if s["width"] == w]
        width_folds[str(w)] = _eval_policies_on_group(train, test)

    depth_folds = {}
    for d in depths:
        train = [s for s in samples if s["recurrence_depth"] != d]
        test = [s for s in samples if s["recurrence_depth"] == d]
        depth_folds[str(d)] = _eval_policies_on_group(train, test)

    def _agg(folds: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for pol in ("learned_predictor", "exact_saved_mac", "first_fit", "density"):
            vals = [
                f["mean_oracle_fraction"][pol]
                for f in folds.values()
                if not f.get("skipped") and f["mean_oracle_fraction"].get(pol) is not None
            ]
            out[pol] = float(statistics.mean(vals)) if vals else None
        return out

    return {
        "status": "experimental_non_correctness_path",
        "total_dataset_rows": len(rows),
        "total_site_samples": len(samples),
        "widths_observed": widths,
        "depths_observed": depths,
        "leave_one_width_out": {
            "aggregate_oracle_fraction": _agg(width_folds),
            "folds": width_folds,
        },
        "leave_one_depth_out": {
            "aggregate_oracle_fraction": _agg(depth_folds),
            "folds": depth_folds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True, help="Input JSONL benchmark receipts")
    parser.add_argument("--dataset-out", type=Path, required=True, help="Output DataGoblin JSONL dataset")
    parser.add_argument("--report-out", type=Path, required=True, help="Output FUSE-12C predictor report JSON")
    args = parser.parse_args()

    rows = export_datagoblin_dataset(args.receipts, args.dataset_out)
    report = evaluate_admission_predictor(rows)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        "FUSE12C_DONE",
        json.dumps(
            {
                "dataset_rows": len(rows),
                "site_samples": report["total_site_samples"],
                "width_holdout": report["leave_one_width_out"]["aggregate_oracle_fraction"],
                "depth_holdout": report["leave_one_depth_out"]["aggregate_oracle_fraction"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
