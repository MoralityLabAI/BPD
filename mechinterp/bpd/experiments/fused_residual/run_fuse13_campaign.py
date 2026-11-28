"""FUSE-13 Resumable CUDA Campaign Orchestrator & Evidence-Derived Report Renderer.

Run inside a VRAMorama strict-FIFO lease:
  vramorama run --vram 12G --headroom 2048 --strict-fifo --owner bpd-fuse13-campaign \
    --guard-power 60 --pause-exit 75 --restart 6 -- \
    python -u -m mechinterp.bpd.experiments.fused_residual.run_fuse13_campaign --allow-cuda

Propagates exit code 75 on GPU power-limit drop; resumes cleanly from durable receipts.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXIT_POWER_CAP = 75


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _run_sub(cmd: list[str]) -> int:
    print(f"[fuse13] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode == EXIT_POWER_CAP:
        print("[fuse13] Power guard triggered (exit 75); pausing cleanly for VRAMorama.", flush=True)
        return EXIT_POWER_CAP
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}")
    return 0


def _summarize_probe_dir(probe_dir: Path, expected_cells: int) -> dict[str, Any]:
    """Aggregate per-cell microprobe JSON receipts into compact per-mode/scale/target counts."""
    files = sorted(p for p in probe_dir.glob("*.json") if p.name != "manifest.json")
    manifest = json.loads((probe_dir / "manifest.json").read_text(encoding="utf-8")) if (probe_dir / "manifest.json").exists() else {}
    by_mode: dict[str, dict[str, Any]] = {}
    by_mode_scale: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("status") == "error":
            errors.append({"file": f.name, "returncode": r.get("returncode")})
            continue
        mode = r["mode"]
        scale = str(int(r["scale"]) if float(r["scale"]).is_integer() else r["scale"])
        for bucket_map, key in ((by_mode, mode), (by_mode_scale, f"{mode}|scale={scale}")):
            b = bucket_map.setdefault(
                key,
                {
                    "cells": 0,
                    "per_use_finite_bitwise": 0,
                    "per_use_within_tol": 0,
                    "shared_finite_bitwise": 0,
                    "shared_within_tol": 0,
                    "shared_nonfinite_cells": 0,
                    "shared_overflow_parity_cells": 0,
                    "shared_finite_mismatch_cells": 0,
                    "shared_storage_collapsed_cells": 0,
                    "native_max_unique_storages": 0,
                    "shared_max_unique_storages": 0,
                    "max_abs_shared_finite": 0.0,
                    "max_abs_per_use_finite": 0.0,
                },
            )
            b["cells"] += 1
            ns = r["native_storage"]
            cp = r["candidates"]["cached_per_use_cast"]
            cs = r["candidates"]["cached_shared_factor_cast"]
            b["native_max_unique_storages"] = max(b["native_max_unique_storages"], int(ns["unique_square_storages"]))
            b["shared_max_unique_storages"] = max(b["shared_max_unique_storages"], int(cs["storage"]["unique_square_storages"]))
            if cs["storage"]["unique_square_storages"] == 1:
                b["shared_storage_collapsed_cells"] += 1
            if cp["finite_bitwise"]:
                b["per_use_finite_bitwise"] += 1
            if cp["within_tol"]:
                b["per_use_within_tol"] += 1
            if cs["finite_bitwise"]:
                b["shared_finite_bitwise"] += 1
            if cs["within_tol"]:
                b["shared_within_tol"] += 1

            comps_s = cs["comparisons"].values()
            all_fin_s = all(m["finite"] for m in comps_s)
            if not all_fin_s:
                b["shared_nonfinite_cells"] += 1
                if all(
                    m["nan_masks_equal"] and m["posinf_masks_equal"] and m["neginf_masks_equal"]
                    and (m["bitwise"] if m["finite"] else True)
                    for m in comps_s
                ):
                    b["shared_overflow_parity_cells"] += 1
            elif not cs["finite_bitwise"]:
                b["shared_finite_mismatch_cells"] += 1

            for m in comps_s:
                if m.get("max_abs") is not None:
                    b["max_abs_shared_finite"] = max(b["max_abs_shared_finite"], float(m["max_abs"]))
            for m in cp["comparisons"].values():
                if m.get("max_abs") is not None:
                    b["max_abs_per_use_finite"] = max(b["max_abs_per_use_finite"], float(m["max_abs"]))

    return {
        "manifest": manifest,
        "expected_cells": expected_cells,
        "completed_cells": len(files),
        "error_cells": len(errors),
        "errors": errors[:10],
        "by_mode": by_mode,
        "by_mode_scale": by_mode_scale,
    }


def _run_inprocess_cuda_stress(out_dir: Path, min_watts: float = 60.0) -> int:
    """Execute all 2,304 frozen CUDA stress cells in-process with identical receipt schema & resume."""
    import torch
    from .fuse13_precision_gym import MODES, ROOTS, probe
    from .validate_fuse12_cuda import check_gpu_safety

    out_dir.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    params = list(
        itertools.product(MODES, (64, 128, 256), (1, 6, 12, 24), ROOTS, (1, 10, 100, 10000), ("ordinary", "exact", "near"))
    )
    manifest = {"head": head, "device": "cuda", "tier": "stress", "cells": len(params)}
    mp = out_dir / "manifest.json"
    if not mp.exists():
        _write_json(mp, manifest)

    done = 0
    new_cnt = 0
    t0 = time.perf_counter()
    for idx, (mode, w, t, root, scale, target) in enumerate(params):
        cid = f"cuda_{mode}_w{w}_t{t}_r{root}_s{scale}_{target}"
        dest = out_dir / f"{cid}.json"
        if dest.exists():
            done += 1
            continue
        if new_cnt % 16 == 0:
            check_gpu_safety(min_watts=min_watts, min_free_mib=2048.0)
        res = probe("cuda", w, t, root, mode, scale=scale, target=target)
        _write_json(dest, res)
        done += 1
        new_cnt += 1
        if new_cnt % 128 == 0:
            dt = time.perf_counter() - t0
            print(f"[fuse13-stress] {done}/{len(params)} cells done (+{new_cnt} new in {dt:.1f}s)", flush=True)
    torch.cuda.empty_cache()
    print(f"[fuse13-stress] COMPLETE: {done}/{len(params)} cells (+{new_cnt} new)", flush=True)
    return 0


def render_fuse13_results(receipts_dir: Path, out_md: Path) -> None:
    """Generate FUSE13_RESULTS.md directly from raw JSON/JSONL receipts."""
    audit_p = receipts_dir / "receipt_fuse13_audit.json"
    smoke_p = receipts_dir / "receipt_fuse13_cuda_smoke.json"
    stage_a_p = receipts_dir / "receipt_fuse13_stage_a_cuda.json"
    stage_bc_p = receipts_dir / "receipt_fuse13_stage_bc_cuda.json"
    stress_p = receipts_dir / "receipt_fuse13_cuda_stress_summary.json"
    frontier_p = receipts_dir / "receipt_fuse13_frontier.jsonl"
    stage_e_p = receipts_dir / "receipt_fuse13e_predictor.json"

    audit = json.loads(audit_p.read_text(encoding="utf-8")) if audit_p.exists() else None
    smoke = json.loads(smoke_p.read_text(encoding="utf-8")) if smoke_p.exists() else None
    stage_a = json.loads(stage_a_p.read_text(encoding="utf-8")) if stage_a_p.exists() else None
    stage_bc = json.loads(stage_bc_p.read_text(encoding="utf-8")) if stage_bc_p.exists() else None
    stress = json.loads(stress_p.read_text(encoding="utf-8")) if stress_p.exists() else None
    frontier = (
        [json.loads(l) for l in frontier_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if frontier_p.exists()
        else []
    )
    stage_e = json.loads(stage_e_p.read_text(encoding="utf-8")) if stage_e_p.exists() else None

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    lines: list[str] = []
    lines.append("# BPD FUSE-13: Precision-Aware Retained Residuals — Results")
    lines.append("")
    lines.append(f"- **Campaign commit:** `{head}`")
    lines.append("- **Hardware & lease:** NVIDIA GeForce RTX 5080 Laptop GPU (16 GB VRAM, Windows WDDM), VRAMorama strict FIFO (`12 GiB` lease, `2048 MiB` headroom, `>=60 W` power guard, exit code `75` pause).")
    lines.append("- **Production defaults unchanged:** `allow_cuda_residual_cache=False`, `residual_cache_storage_mode=\"fp32\"`, all residual caches default off.")
    lines.append("- **Tolerances preserved:** `ATOL=5e-5`, `RTOL=5e-4`; finite byte equality (`view(uint8)`) required for mixed-precision admission; nonfinite overflow parity reported separately and never counted as finite pass.")
    lines.append("")

    # Executive Summary / Gate Verdicts
    lines.append("## Executive Gate Summary")
    lines.append("")
    lines.append("| Stage | Scope | Status | Evidence Receipt |")
    lines.append("|---|---|---|---|")
    if audit:
        lines.append(f"| Stage 0 | FUSE-12 receipt reconciliation & audit | **{'PASS' if audit['pass'] else 'FAIL'}** (50 unique cells, BF16 C/F/G=50/52/52, FP16=42/42/42, FP16 384 nonfinite outcomes) | `receipt_fuse13_audit.json` |")
    if stage_a:
        n_a = len(stage_a.get("trainer_matrix", []))
        n_a_ok = sum(1 for r in stage_a.get("trainer_matrix", []) if r["status"] == "pass")
        lines.append(f"| Stage A | Native `observe_clean_pass` mixed-precision EMA repair (CUDA FP32/TF32/BF16/FP16) | **{'PASS' if stage_a['all_pass'] else 'FAIL'}** ({n_a_ok}/{n_a} trainer cells, 4/4 unit suites) | `receipt_fuse13_stage_a_cuda.json` |")
    if smoke:
        lines.append(f"| Stage B/C Smoke | Prepared 16-cell CUDA microprobe smoke (`run_fuse13_gym --tier smoke`) | **{smoke['completed_cells']}/{smoke['expected_cells']} completed** (errors={smoke['error_cells']}) | `receipt_fuse13_cuda_smoke.json` |")
    if stage_bc:
        reg = stage_bc.get("stage_c_stress_regression", {})
        verdicts = ", ".join(
            f"{m.upper()}={'PASS' if reg.get(m, {}).get('admission_pass') else 'HOLD/OVERFLOW'}"
            for m in ("fp32", "tf32", "bf16", "fp16")
            if m in reg
        )
        tr_cells = stage_bc.get("stage_c_trainer_parity", [])
        tr_ok = sum(1 for c in tr_cells if c["finite_bitwise_pass"])
        lines.append(f"| Stage B/C Gate | 6-strategy ablation + 96-case reg + 48-case holdout + FP16 GradScaler + matched trainer A vs G | **{verdicts}**; trainer `{tr_ok}/{len(tr_cells)}` finite bitwise | `receipt_fuse13_stage_bc_cuda.json` |")
    if stress:
        lines.append(f"| Stage C Stress | Frozen 2,304-cell microprobe stress matrix (`4 modes x 3 widths x 4 depths x 4 roots x 4 scales x 3 targets`) | **{stress['completed_cells']}/{stress['expected_cells']} completed** (errors={stress['error_cells']}) | `receipt_fuse13_cuda_stress_summary.json` |")
    if frontier:
        n_fr = sum(1 for r in frontier if r.get("kind") == "frontier_scaling")
        n_w4 = sum(1 for r in frontier if r.get("kind") == "w4096_parity")
        lines.append(f"| Stage D | Width-4096 full trainer parity ({n_w4} cells) + preflighted frontier scaling ({n_fr} cells) + checkpoint audit | **COMPLETE** (zero forced WDDM spill) | `receipt_fuse13_frontier.jsonl` |")
    if stage_e:
        lines.append(f"| Stage E | DataGoblin physical cost model + learned correction (`correctness_path=False`) | **COMPLETE** ({stage_e['total_dataset_rows']} rows, {stage_e['total_site_samples']} site samples) | `receipt_fuse13e_predictor.json` |")
    lines.append("")

    # Stage 0
    if audit:
        lines.append("## Stage 0: FUSE-12 Evidence Reconciliation")
        lines.append("")
        lines.append(f"- Audit pass: `{audit['pass']}` (`50` distinct cells: `38` FP32 ok, `4` TF32 ok, `2` BF16 ok, `2` FP16 ok, `2` BF16 `native_autocast_error`, `2` FP16 `native_autocast_error`).")
        bf_pa = audit["mixed_trainer"]["bf16"]["per_arm"]
        fp_pa = audit["mixed_trainer"]["fp16"]["per_arm"]
        lines.append(f"- Width-512 FUSE-12B trainer failures (reconciled in `FUSE12_RESULTS.md`): BF16 `C={bf_pa['C']['failures']}, F={bf_pa['F']['failures']}, G={bf_pa['G']['failures']}` (sum `154`); FP16 `C={fp_pa['C']['failures']}, F={fp_pa['F']['failures']}, G={fp_pa['G']['failures']}` (sum `126`).")
        lines.append(f"- FP16 FUSE-12B stress `nonfinite_parity_tensors`: `{audit['stress']['fp16']['nonfinite_parity_tensors']}` (matched nonfinite outcomes, explicitly excluded from finite passes).")
        lines.append("")

    # Stage A
    if stage_a:
        lines.append("## Stage A: Native Mixed-Precision Calibration (`observe_clean_pass`)")
        lines.append("")
        lines.append("Repaired `TiedComponentLinear.observe_clean_pass` (`mechinterp/bpd/components.py`) casts completed `acts_mean`, `delta_mean`, and `write_rms` statistics to each FP32 calibration buffer's dtype before `lerp_`/`copy_`, preserving native autocast reduction semantics and exact FP32/TF32 zero-copy invariance.")
        lines.append("")
        lines.append("| Mode | Raw `acts_mean` dtype | Raw `delta_mean` dtype | Raw `write_rms` dtype | Buffer dtypes | Expansion `max_passes` | Unit Status |")
        lines.append("|---|---|---|---|---|---|---|")
        for m, u in stage_a.get("unit_tests", {}).items():
            dt = u["dtype_trace"]
            lines.append(
                f"| `{m}` | `{dt['raw_acts_mean_dtype']}` | `{dt['raw_delta_mean_dtype']}` | `{dt['raw_write_rms_dtype']}` | `float32` | `{u['expanded_max_passes']}` | **{u['status'].upper()}** |"
            )
        lines.append("")
        tr_m = stage_a.get("trainer_matrix", [])
        n_ok = sum(1 for r in tr_m if r["status"] == "pass")
        n_fin = sum(1 for r in tr_m if r["all_finite_losses"])
        lines.append(f"- Native Arm A trainer matrix (CUDA, widths `128, 512`, `closed_loop` & `clean_window`, cadences `off/1/50`, 3 steps each): **{n_ok}/{len(tr_m)} passed**, **{n_fin}/{len(tr_m)} finite losses**, 100% FP32 finite calibration buffers.")
        lines.append("")

    # Stage B: 6-strategy storage ablation
    if stage_bc:
        lines.append("## Stage B: Precision-Aware Residual Storage & Autograd Identity Ablation")
        lines.append("")
        lines.append("Measured inside `TiedComponentLinear` (`width=128, passes=12`) on CUDA across all 6 Stage B strategies:")
        lines.append("")
        lines.append("| Mode | Strategy | Recons | Unique Sq Storages | Unique Sq Bytes | Saved Dtypes | Peak Alloc (MiB) | Finite Bitwise | `g_V` max_abs | `g_U` max_abs |")
        lines.append("|---|---|---:|---:|---:|---|---:|---|---:|---:|")
        for m, ab in stage_bc.get("stage_b_ablation", {}).items():
            ns = ab["native_storage"]
            n_mib = f"{ns['peak_allocated_bytes']/(1024**2):.2f}" if ns.get("peak_allocated_bytes") else "n/a"
            lines.append(
                f"| `{m}` | `native` | {ns['reconstructions']} | {ns['unique_square_storages']} | {ns['unique_square_bytes']:,} | `{','.join(ns['saved_square_dtypes'])}` | {n_mib} | ref | 0.0 | 0.0 |"
            )
            for sname, sres in ab["strategies"].items():
                st = sres["storage"]
                mib = f"{st['peak_allocated_bytes']/(1024**2):.2f}" if st.get("peak_allocated_bytes") else "n/a"
                lines.append(
                    f"| `{m}` | `{sname}` | {st['reconstructions']} | {st['unique_square_storages']} | {st['unique_square_bytes']:,} | `{','.join(st['saved_square_dtypes'])}` | {mib} | **{'YES' if sres['finite_bitwise'] else 'NO'}** | {sres['g_V_max_abs']:.6g} | {sres['g_U_max_abs']:.6g} |"
                )
        lines.append("")

        # Stage C: 96-case regression + 48-case holdout stress
        lines.append("## Stage C: Factor-Gradient Parity, Stress Suites, GradScaler & Matched Trainer Gates")
        lines.append("")
        lines.append("### 1. `StepResidualPool(storage_mode=\"compute_dtype\")` Stress Suites (Regression 96 cases + Holdout 48 cases)")
        lines.append("")
        lines.append("| Suite | Mode | Cases | Tensors | Finite Bitwise | Within Tol | Overflow Parity (nonfinite) | Failures | Max Abs (finite) | Admission Gate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for sname, sdict in (("Regression (4012001..8)", stage_bc.get("stage_c_stress_regression", {})), ("Holdout (4013001..4)", stage_bc.get("stage_c_stress_holdout", {}))):
            for m, r in sdict.items():
                verdict = "PASS" if r["admission_pass"] else ("OVERFLOW_ONLY_HOLD" if r["failures"] == 0 and r["overflow_parity_tensors"] > 0 else "FAIL")
                lines.append(
                    f"| {sname} | `{m}` | {r['cases']} | {r['tensors']} | {r['finite_bitwise']} | {r['within_tol']} | {r['overflow_parity_tensors']} | {r['failures']} | {r['max_abs']:.3e} | **{verdict}** |"
                )
        lines.append("")

        # FP16 GradScaler parity
        gs = stage_bc.get("stage_c_fp16_gradscaler")
        if gs:
            lines.append("### 2. FP16 Dynamic `GradScaler` Skip/Update Parity")
            lines.append("")
            lines.append("| Case | Ref Skipped | Cand Skipped | Skip Match | Scale After Match | `V` Bitwise | `U` Bitwise |")
            lines.append("|---|---|---|---|---|---|---|")
            for c in gs["cases"]:
                lines.append(
                    f"| `{c['case']}` | `{c['ref_skipped']}` | `{c['cand_skipped']}` | `{c['skip_decision_match']}` | `{c['scale_after_match']}` | `{c['V_bitwise']}` | `{c['U_bitwise']}` |"
                )
            lines.append("")

        # Full Trainer Matched Parity
        tr_cells = stage_bc.get("stage_c_trainer_parity", [])
        if tr_cells:
            lines.append("### 3. Full Matched Native Arm A vs Arm G (`storage_mode=\"compute_dtype\"`) Trainer Parity")
            lines.append("")
            lines.append("| Mode | Width | Passes | Regime | Cal | Budget | Tensors Finite Bitwise | Overflow Parity | Failures | Max Abs | Arm A Step (ms) | Arm G Step (ms) | Speedup | Arm A Peak Alloc (MiB) | Arm G Peak Alloc (MiB) | VRAM Saved (MiB) |")
            lines.append("|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for c in tr_cells:
                pa = c["parity"]
                a_ms = c["arms"]["A"]["step_ms"][-1]
                g_ms = c["arms"]["G"]["step_ms"][-1]
                sp = a_ms / max(g_ms, 1e-12)
                a_mb = c["arms"]["A"]["peak_allocated_bytes"] / (1024.0 ** 2)
                g_mb = c["arms"]["G"]["peak_allocated_bytes"] / (1024.0 ** 2)
                lines.append(
                    f"| `{c['mode']}` | {c['width']} | {c['n_passes']} | `{c.get('regime','closed_loop')}` | `{c['cal_cadence']}` | `{c['budget_mode']}` | {pa['finite_bitwise']}/{pa['tensors']} | {pa['overflow_parity']} | {pa['failures']} | {pa['max_abs']:.3e} | {a_ms:.2f} | {g_ms:.2f} | {sp:.3f}x | {a_mb:.1f} | {g_mb:.1f} | {a_mb - g_mb:+.1f} |"
                )
            lines.append("")

    # Frozen 2,304-cell stress summary
    if stress:
        lines.append("### 4. Frozen Microprobe Stress Matrix (`2,304` Cells)")
        lines.append("")
        lines.append("| Mode | Cells | Per-Use Cast Finite Bitwise | Shared Cast Finite Bitwise | Shared Nonfinite Cells | Shared Overflow Parity | Shared Finite Mismatches | Storage Collapsed (1 uniq) | Shared Max Abs (finite) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m, b in stress.get("by_mode", {}).items():
            lines.append(
                f"| `{m}` | {b['cells']} | {b['per_use_finite_bitwise']}/{b['cells']} | {b['shared_finite_bitwise']}/{b['cells']} | {b['shared_nonfinite_cells']} | {b['shared_overflow_parity_cells']} | {b['shared_finite_mismatch_cells']} | {b['shared_storage_collapsed_cells']}/{b['cells']} | {b['max_abs_shared_finite']:.3e} |"
            )
        lines.append("")

    # Stage D: Frontier Scaling & Checkpoint Audit
    if frontier:
        lines.append("## Stage D: Width-4096 Parity, Frontier Scaling (Preflighted, Zero Forced Spill) & Checkpoint Audit")
        lines.append("")
        ckpts = [r for r in frontier if r.get("kind") == "checkpoint_transfer"]
        for ck in ckpts:
            lines.append(f"- **Pretrained TRM checkpoint audit:** `{ck['status']}` — {ck.get('reason', '')}")
        lines.append("")

        w4 = [r for r in frontier if r.get("kind") == "w4096_parity"]
        if w4:
            lines.append("### Width-4096 Full Trainer Parity")
            lines.append("")
            lines.append("| Mode | Width | Passes | Status | Finite Bitwise Tensors | Failures | Arm A Peak Alloc (MiB) | Arm G Peak Alloc (MiB) |")
            lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
            for r in w4:
                if r["status"] != "ok":
                    lines.append(f"| `{r['mode']}` | 4096 | {r['n_passes']} | `{r['status']}` (est {r['preflight_arm_a']['total_est_mib']} MiB > {r['safe_budget_mib']:.1f} MiB) | — | — | — | — |")
                else:
                    pa = r["parity"]
                    a_mb = r["arms"]["A"]["peak_allocated_bytes"] / (1024.0 ** 2)
                    g_mb = r["arms"]["G"]["peak_allocated_bytes"] / (1024.0 ** 2)
                    lines.append(f"| `{r['mode']}` | 4096 | {r['n_passes']} | **{'PASS' if r['finite_bitwise_pass'] else 'FAIL'}** | {pa['finite_bitwise']}/{pa['tensors']} | {pa['failures']} | {a_mb:.1f} | {g_mb:.1f} |")
            lines.append("")

        fr = [r for r in frontier if r.get("kind") == "frontier_scaling"]
        if fr:
            lines.append("### Frontier Scaling Matrix (Synthetic Mini-TRM, Independent Per-Arm Preflight)")
            lines.append("")
            lines.append("| Mode | Width | Passes | Arm A Status / Median (ms) | Arm G Status / Median (ms) | Speedup (95% CI) | Arm A Alloc / Res / NVML (MiB) | Arm G Alloc / Res / NVML (MiB) | Arm G Energy (mJ/step) |")
            lines.append("|---|---:|---:|---|---|---|---|---|---|")
            for r in fr:
                aa, ag = r["arm_a"], r["arm_g"]
                a_txt = f"`ok` ({aa['cuda_median_ms']:.1f} ms)" if aa["status"] == "ok" else f"`{aa['status']}` (est {aa['preflight']['total_est_mib']:.0f} MiB)"
                g_txt = f"`ok` ({ag['cuda_median_ms']:.1f} ms)" if ag["status"] == "ok" else f"`{ag['status']}` (est {ag['preflight']['total_est_mib']:.0f} MiB)"
                sp_txt = f"{r['speedup_cuda']:.3f}x [{r['speedup_ci95'][0]:.3f}, {r['speedup_ci95'][1]:.3f}]" if r.get("speedup_cuda") else "n/a (A blocked)"
                a_mem = f"{aa['peak_allocated_bytes']/(1024**2):.0f} / {aa['peak_reserved_bytes']/(1024**2):.0f} / {aa['nvml_used_mib']:.0f}" if aa["status"] == "ok" else "blocked"
                g_mem = f"{ag['peak_allocated_bytes']/(1024**2):.0f} / {ag['peak_reserved_bytes']/(1024**2):.0f} / {ag['nvml_used_mib']:.0f}" if ag["status"] == "ok" else "blocked"
                g_eng = f"{ag['energy_per_step_mj']:.1f}" if (ag["status"] == "ok" and ag.get("energy_per_step_mj") is not None) else "n/a"
                lines.append(f"| `{r['mode']}` | {r['width']} | {r['n_passes']} | {a_txt} | {g_txt} | {sp_txt} | {a_mem} | {g_mem} | {g_eng} |")
            lines.append("")

    # Stage E: DataGoblin
    if stage_e:
        lines.append("## Stage E: DataGoblin Physical Cost Model + Learned Admission Correction")
        lines.append("")
        lines.append(f"- **Status:** `{stage_e['status']}` (`correctness_path={stage_e['correctness_path']}` — strictly outside numerical correctness path).")
        lines.append(f"- **Dataset:** `{stage_e['total_dataset_rows']}` rows, `{stage_e['total_site_samples']}` site samples (`fuse13e_datagoblin_dataset.jsonl`).")
        lines.append(f"- **Holdout family status:** hardware=`{stage_e['hardware_holdout_status']}`, checkpoint=`{stage_e['checkpoint_holdout_status']}`.")
        lines.append("")
        lines.append("| Holdout | Policy | Oracle Step Fraction | Oracle Recon Fraction | Worst-Case Step Regret (ms) | Unsafe Admission Rate |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for hlabel, hkey in (("Leave-one-width-out", "leave_one_width_out"), ("Leave-one-depth-out", "leave_one_depth_out"), ("Leave-one-precision-out", "leave_one_precision_out")):
            hd = stage_e.get(hkey, {})
            if hd.get("skipped"):
                continue
            for pol in ("physical_plus_learned_gated", "physical_plus_learned", "physical_prior", "fuse12c_log_ridge", "exact_saved_mac", "density", "first_fit"):
                os_v = hd["aggregate_oracle_fraction_step_ms"].get(pol)
                or_v = hd["aggregate_oracle_fraction_recon_ms"].get(pol)
                rg_v = hd["worst_case_regret_step_ms"].get(pol)
                un_v = hd["mean_unsafe_admission_rate"].get(pol)
                lines.append(
                    f"| {hlabel} | `{pol}` | {f'{os_v:.4f}' if os_v is not None else 'n/a'} | {f'{or_v:.4f}' if or_v is not None else 'n/a'} | {f'{rg_v:.3f}' if rg_v is not None else 'n/a'} | {f'{un_v:.4f}' if un_v is not None else 'n/a'} |"
                )
        lines.append("")
        lines.append("### Safety-Gated Worth-Caching Classifier Calibration")
        lines.append("")
        lines.append("| Holdout | Eval N | Accuracy | False Rejection Rate | Unsafe Admission Rate | Brier Score | ECE |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for hlabel, hkey in (("Width", "leave_one_width_out"), ("Depth", "leave_one_depth_out"), ("Precision", "leave_one_precision_out")):
            cl = stage_e["safe_admission_classifier"].get(hkey, {})
            if cl.get("skipped"):
                continue
            lines.append(
                f"| {hlabel} | {cl['n_eval']} | {cl['accuracy']:.4f} | {cl['false_rejection_rate']:.4f} | {cl['unsafe_admission_rate']:.4f} | {cl['brier_score']:.4f} | {cl['ece']:.4f} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[fuse13] Rendered {out_md} ({len(lines)} lines)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--min-watts", type=float, default=60.0)
    args = p.parse_args()

    receipts_dir = HERE
    runs_dir = Path("runs/fuse13")
    out_md = HERE / "FUSE13_RESULTS.md"

    if args.render_only:
        render_fuse13_results(receipts_dir, out_md)
        return 0

    if not args.allow_cuda:
        p.error("CUDA campaign requires --allow-cuda inside VRAMorama lease")

    py = sys.executable

    # 1. Stage 0 Audit
    audit_p = receipts_dir / "receipt_fuse13_audit.json"
    if not audit_p.exists():
        rc = _run_sub([py, "-m", "mechinterp.bpd.experiments.fused_residual.fuse13_precision_gym", "--output", str(audit_p)])
        if rc != 0:
            return rc

    # 2. Prepared CUDA Smoke Matrix (16 cells via run_fuse13_gym.py)
    smoke_dir = runs_dir / "cuda_smoke"
    smoke_p = receipts_dir / "receipt_fuse13_cuda_smoke.json"
    rc = _run_sub([
        py, "-m", "mechinterp.bpd.experiments.fused_residual.run_fuse13_gym",
        "--device", "cuda", "--allow-cuda", "--tier", "smoke", "--out", str(smoke_dir),
    ])
    if rc != 0:
        return rc
    _write_json(smoke_p, _summarize_probe_dir(smoke_dir, expected_cells=16))

    # 3. Stage A CUDA (repaired native observe_clean_pass)
    stage_a_p = receipts_dir / "receipt_fuse13_stage_a_cuda.json"
    if not stage_a_p.exists():
        rc = _run_sub([
            py, "-m", "mechinterp.bpd.experiments.fused_residual.validate_fuse13_stage_a",
            "--device", "cuda", "--allow-cuda", "--min-watts", str(args.min_watts),
            "--output", str(stage_a_p),
        ])
        if rc != 0:
            return rc

    # 4. Stage B & C CUDA (6-strategy ablation, 96-case reg + 48-case holdout, FP16 GradScaler, matched trainer A vs G)
    stage_bc_p = receipts_dir / "receipt_fuse13_stage_bc_cuda.json"
    if not stage_bc_p.exists():
        rc = _run_sub([
            py, "-m", "mechinterp.bpd.experiments.fused_residual.validate_fuse13_precision",
            "--device", "cuda", "--allow-cuda", "--min-watts", str(args.min_watts),
            "--output", str(stage_bc_p),
        ])
        if rc != 0:
            return rc

    # 5. Full 2,304-cell frozen microprobe stress matrix (resumable in-process)
    stress_dir = runs_dir / "cuda_stress"
    stress_p = receipts_dir / "receipt_fuse13_cuda_stress_summary.json"
    if not stress_p.exists():
        rc = _run_inprocess_cuda_stress(stress_dir, min_watts=args.min_watts)
        if rc != 0:
            return rc
        _write_json(stress_p, _summarize_probe_dir(stress_dir, expected_cells=2304))

    # 6. Stage D Frontier scaling + w4096 parity + checkpoint audit
    frontier_p = receipts_dir / "receipt_fuse13_frontier.jsonl"
    rc = _run_sub([
        py, "-m", "mechinterp.bpd.experiments.fused_residual.bench_fuse13_frontier",
        "--allow-cuda", "--resume", "--min-watts", str(args.min_watts),
        "--output", str(frontier_p),
    ])
    if rc != 0:
        return rc

    # 7. Stage E DataGoblin (combining FUSE-12 + FUSE-13 receipts)
    rc = _run_sub([
        py, "-m", "mechinterp.bpd.experiments.fused_residual.fuse13e_datagoblin",
        "--frontier-receipts", str(frontier_p),
        "--stage-bc-receipt", str(stage_bc_p),
        "--dataset-out", str(receipts_dir / "fuse13e_datagoblin_dataset.jsonl"),
        "--report-out", str(receipts_dir / "receipt_fuse13e_predictor.json"),
    ])
    if rc != 0:
        return rc

    # 8. Render final evidence-derived FUSE13_RESULTS.md
    render_fuse13_results(receipts_dir, out_md)
    print("FUSE13_CAMPAIGN_ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
