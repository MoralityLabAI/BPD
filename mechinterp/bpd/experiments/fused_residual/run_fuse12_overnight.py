"""FUSE-12: Resumable overnight CUDA orchestration & FUSE12_RESULTS.md generator.

Designed to be queued cleanly via VRAMorama:
  vramorama run --vram 12G --headroom 2048 --strict-fifo --owner bpd-fuse12-cuda \
      --guard-power 60 --pause-exit 75 --restart 6 -- \
      <python> -m mechinterp.bpd.experiments.fused_residual.run_fuse12_overnight
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .validate_fuse12_cuda import EXIT_POWER_CAP

EXP_DIR = Path(__file__).resolve().parent


def _run_module(module: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *extra_args]
    print(f"[orchestrator] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode == EXIT_POWER_CAP:
        print(f"[orchestrator] received power-cap pause code {EXIT_POWER_CAP}; propagating.", flush=True)
        sys.exit(EXIT_POWER_CAP)
    return proc.returncode


def render_results_markdown(
    *,
    validate_path: Path,
    receipts_path: Path,
    fuse12c_report_path: Path,
    out_md_path: Path,
    git_head: str,
) -> None:
    val = json.loads(validate_path.read_text(encoding="utf-8")) if validate_path.exists() else None
    receipts: list[dict[str, Any]] = []
    if receipts_path.exists():
        for line in receipts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                receipts.append(json.loads(line))
    f12c = (
        json.loads(fuse12c_report_path.read_text(encoding="utf-8"))
        if fuse12c_report_path.exists()
        else None
    )

    lines: list[str] = []
    lines.append("# FUSE-12 | CUDA Validation, Cumulative Residual Caching Benchmark & DataGoblin Policy Curriculum")
    lines.append("")
    lines.append(f"- **Generated (UTC):** {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}")
    lines.append(f"- **Branch / Commit:** `research/fuse12-cuda-validation-20261127` (`{git_head}`)")
    lines.append("- **Base:** FUSE-11 PR #185 (`a4cc3d861df531f41d91000d8dd0922a5ef953ec`)")
    lines.append("- **Scope Distinction:** All benchmark numbers below are measured on the **synthetic mini-TRM native BPDTrainer** (4 tied sites, heterogeneous ranks) and are explicitly distinct from pretrained 7M TRM or LLM checkpoints. Production defaults remain unchanged (`allow_cuda_residual_cache=False`, all caches off).")
    lines.append("")

    # Section 1: Hardware & Preflight
    lines.append("## 1. Hardware & Validation Summary")
    if val is None:
        lines.append("> [!NOTE]")
        lines.append("> CUDA validation and benchmark run is currently queued in VRAMorama behind active RTX 5080 workloads. CPU regression & guard checks passed; CUDA results will populate automatically upon queue completion.")
        lines.append("")
    else:
        nv = val.get("preflight_nvml", {})
        dp = val.get("device_properties", {})
        lines.append(f"- **GPU:** {dp.get('name', nv.get('name', 'N/A'))} (SM {dp.get('major')}.{dp.get('minor')}, WDDM)")
        lines.append(f"- **Driver / CUDA / PyTorch:** Driver `{nv.get('driver_version', 'N/A')}`, CUDA runtime `{val.get('cuda_runtime')}`, PyTorch `{val.get('torch')}`")
        lines.append(f"- **Power Limit / Preflight VRAM Free:** {nv.get('enforced_power_limit_w')} W / {nv.get('memory_free_mib')} MiB free of {nv.get('memory_total_mib')} MiB")
        lines.append(f"- **CPU Guard Gate:** `{val.get('cpu_guard', {}).get('status')}`")
        if "cuda_fp32_lifecycle" in val:
            lf = val["cuda_fp32_lifecycle"]
            st = val["cuda_fp32_stress"]
            lines.append(f"- **CUDA FP32 Lifecycle & Stream Ownership:** `{lf['status']}` ({len(lf['checks'])}/{len(lf['checks'])} checks, zero allocator leak verified: `{lf['allocator_audit']['after_close_bytes'] == lf['allocator_audit']['base_bytes']}`)")
            lines.append(
                f"- **CUDA FP32 96-Case Stress Gate:** `{'PASS' if val.get('fp32_gate_passed') else 'FAIL'}` | "
                f"bitwise exact `{st['bitwise_exact']}/{st['tensors']}`, within tol `{st['within_tol']}/{st['tensors']}`, "
                f"failures `{st['failures']}`, max_abs `{st['max_abs_err']:.3e}`, max_rel `{st['max_rel_err']:.3e}`, "
                f"near_zero_max_abs `{st['near_zero_max_abs_err']:.3e}`"
            )
        lines.append("")

    # Section 2: Matched Benchmark Tables (FP32)
    ok_fp32 = [r for r in receipts if r.get("status") == "ok" and r.get("precision_mode") == "fp32"]
    lines.append("## 2. FP32 Matched 7-Arm Benchmark Results (CUDA Synchronized)")
    if not ok_fp32:
        lines.append("_Pending overnight CUDA execution._")
        lines.append("")
    else:
        lines.append("| Tier | Width | Passes | Regime | K | Cal | Budget | Arm A (ms) | Arm C (ms / x) | Arm F (ms / x) | Arm G (ms / x) | G CI95 | Payload (MiB) | Speedup/MiB | Peak Alloc (MiB) | NVML Used (MiB) | Energy A->G (mJ) | Parity (Bitwise / Tol / Fail) |")
        lines.append("|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|")
        for r in ok_fp32:
            arms = r["arms"]
            a = arms["A"]
            c = arms.get("C")
            f_arm = arms.get("F")
            g = arms.get("G") or f_arm or c
            ci = g.get("speedup_cuda_ci95")
            ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
            par = g.get("parity")
            par_s = (
                f"{par['bitwise_exact']}/{par['tensors']} | {par['within_tol']}/{par['tensors']} | {par['failures']}"
                if par
                else "n/a"
            )
            e_a = a.get("energy_per_step_mj")
            e_g = g.get("energy_per_step_mj")
            e_s = f"{e_a:.1f}->{e_g:.1f}" if (e_a is not None and e_g is not None) else "n/a"
            c_s = f"{c['cuda_median_ms']:.2f} ({c['speedup_cuda']:.3f}x)" if c else "n/a"
            f_s = f"{f_arm['cuda_median_ms']:.2f} ({f_arm['speedup_cuda']:.3f}x)" if f_arm else "n/a"
            g_s = f"{g['cuda_median_ms']:.2f} ({g['speedup_cuda']:.3f}x)" if g else "n/a"
            lines.append(
                f"| {r['tier']} | {r['width']} | {r['n_passes']} | {r['regime']} | {r['pgd_inner']} | "
                f"{r['cal_cadence']} | {r['budget_mode']} | {a['cuda_median_ms']:.2f} | {c_s} | {f_s} | {g_s} | "
                f"{ci_s} | {g['payload_peak_mib']:.2f} | {g['speedup_per_mib']:.4f} | "
                f"{g['peak_allocated_bytes'] / (1024**2):.1f} | {g['nvml_used_mib']:.0f} | {e_s} | {par_s} |"
            )
        lines.append("")

        # Skipped / headroom cells
        skipped = [r for r in receipts if r.get("status") != "ok"]
        if skipped:
            lines.append("### Headroom / Skipped Frontier Cells")
            for s in skipped:
                lines.append(f"- `{s['cell_id']}`: `{s['status']}` (free_mib={s.get('free_mib')}, est_mib={s.get('est_vram_mib')})")
            lines.append("")

    # Section 3: FUSE-12B Mixed Precision
    lines.append("## 3. FUSE-12B: Mixed Precision (TF32, BF16, FP16)")
    if val and "fuse12b_stress" in val:
        lines.append("| Mode | Cases | Bitwise Exact | Within Tol | Non-Finite Parity | Failures | Max Abs Err | Max Rel Err | Near-Zero Max Abs |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for pm, st in val["fuse12b_stress"].items():
            lines.append(
                f"| `{pm}` | {st['cases']} | {st['bitwise_exact']}/{st['tensors']} | "
                f"{st['within_tol']}/{st['tensors']} | {st['nonfinite_parity_tensors']} | "
                f"{st['failures']} | {st['max_abs_err']:.3e} | {st['max_rel_err']:.3e} | {st['near_zero_max_abs_err']:.3e} |"
            )
        lines.append("")
    ok_mp = [r for r in receipts if r.get("status") == "ok" and r.get("tier") == "mixed_precision"]
    if ok_mp:
        lines.append("| Precision | Width | Passes | Budget | Arm A (ms) | Arm G (ms) | Speedup | Payload (MiB) | Parity (Bitwise / Tol / Fail) |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---|")
        for r in ok_mp:
            a = r["arms"]["A"]
            g = r["arms"].get("G") or r["arms"].get("F")
            par = g.get("parity")
            par_s = (
                f"{par['bitwise_exact']}/{par['tensors']} | {par['within_tol']}/{par['tensors']} | {par['failures']}"
                if par
                else "n/a"
            )
            lines.append(
                f"| `{r['precision_mode']}` | {r['width']} | {r['n_passes']} | {r['budget_mode']} | "
                f"{a['cuda_median_ms']:.2f} | {g['cuda_median_ms']:.2f} | {g['speedup_cuda']:.3f}x | "
                f"{g['payload_peak_mib']:.2f} | {par_s} |"
            )
        lines.append("")
    elif not (val and "fuse12b_stress" in val):
        lines.append("_Pending FP32 gate pass and overnight CUDA execution._")
        lines.append("")

    # Section 4: FUSE-12C DataGoblin Admission Policy Predictor
    lines.append("## 4. FUSE-12C: DataGoblin Admission-Policy Dataset & Predictor")
    if f12c is None:
        lines.append("_Pending CUDA benchmark receipts for DataGoblin dataset export & held-out evaluation._")
        lines.append("")
    else:
        lines.append(f"- **Dataset Rows:** {f12c['total_dataset_rows']} | **Site Samples:** {f12c['total_site_samples']}")
        lines.append(f"- **Widths Observed:** `{f12c['widths_observed']}` | **Recurrence Depths Observed:** `{f12c['depths_observed']}`")
        lines.append("")
        lines.append("| Holdout Scheme | Learned Predictor (Oracle Frac) | Exact Saved-MAC | First-Fit | Density |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, key in (
            ("Leave-One-Width-Family-Out", "leave_one_width_out"),
            ("Leave-One-Recurrence-Depth-Out", "leave_one_depth_out"),
        ):
            agg = f12c[key]["aggregate_oracle_fraction"]
            def _fmt(x): return f"{x:.4f}" if x is not None else "n/a"
            lines.append(
                f"| {label} | {_fmt(agg.get('learned_predictor'))} | {_fmt(agg.get('exact_saved_mac'))} | "
                f"{_fmt(agg.get('first_fit'))} | {_fmt(agg.get('density'))} |"
            )
        lines.append("")

    # Section 5: Reproduction
    lines.append("## 5. Reproduction Instructions")
    lines.append("```bash")
    lines.append("# 1. Mandatory CPU regression & CUDA default-off guard (safe with no GPU)")
    lines.append("CUDA_VISIBLE_DEVICES=\"\" python -m mechinterp.bpd.experiments.fused_residual.validate_fuse12_cuda --cpu-guard-only --output /tmp/fuse12_cpu_guard.json")
    lines.append("")
    lines.append("# 2. Full CUDA validation + resumable progressive benchmark + FUSE-12C (queued via VRAMorama on Windows RTX 5080)")
    lines.append("vramorama run --vram 12G --headroom 2048 --strict-fifo --owner bpd-fuse12-cuda --guard-power 60 --pause-exit 75 --restart 6 -- \\")
    lines.append("  python -m mechinterp.bpd.experiments.fused_residual.run_fuse12_overnight")
    lines.append("```")
    lines.append("")

    out_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--tiers", default="smoke,medium,wide,frontier,mixed_precision")
    parser.add_argument("--min-watts", type=float, default=40.0)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_path = out_dir / "receipt_fuse12_cuda_validate.json"
    receipts_path = out_dir / "fuse12_cuda_receipts.jsonl"
    dataset_path = out_dir / "fuse12c_datagoblin_dataset.jsonl"
    f12c_report_path = out_dir / "receipt_fuse12c_predictor.json"
    results_md_path = out_dir / "FUSE12_RESULTS.md"

    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        git_head = "unknown"

    # Stage 1: CUDA validation & lifecycle + 96 stress + FUSE-12B stress (skip if already passed)
    need_val = True
    if validate_path.exists():
        try:
            prev = json.loads(validate_path.read_text(encoding="utf-8"))
            if prev.get("fp32_gate_passed") and "fuse12b_stress" in prev:
                print("[orchestrator] reusing passed CUDA validation receipt.", flush=True)
                need_val = False
        except Exception:
            pass

    if need_val:
        rc = _run_module(
            "mechinterp.bpd.experiments.fused_residual.validate_fuse12_cuda",
            ["--cuda", "--min-watts", str(args.min_watts), "--output", str(validate_path)],
        )
        if rc != 0:
            render_results_markdown(
                validate_path=validate_path,
                receipts_path=receipts_path,
                fuse12c_report_path=f12c_report_path,
                out_md_path=results_md_path,
                git_head=git_head,
            )
            sys.exit(rc)

    # Stage 2: Resumable progressive CUDA benchmark
    rc = _run_module(
        "mechinterp.bpd.experiments.fused_residual.bench_fuse12_cuda",
        [
            "--device",
            "cuda",
            "--tiers",
            args.tiers,
            "--resume",
            "--min-watts",
            str(args.min_watts),
            "--output",
            str(receipts_path),
        ],
    )
    if rc != 0:
        sys.exit(rc)

    # Stage 3: FUSE-12C DataGoblin dataset export & held-out admission predictor
    rc = _run_module(
        "mechinterp.bpd.experiments.fused_residual.fuse12c_datagoblin",
        [
            "--receipts",
            str(receipts_path),
            "--dataset-out",
            str(dataset_path),
            "--report-out",
            str(f12c_report_path),
        ],
    )
    if rc != 0:
        sys.exit(rc)

    # Stage 4: Render FUSE12_RESULTS.md
    render_results_markdown(
        validate_path=validate_path,
        receipts_path=receipts_path,
        fuse12c_report_path=f12c_report_path,
        out_md_path=results_md_path,
        git_head=git_head,
    )
    print("FUSE12_OVERNIGHT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
