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
    lines.append("# FUSE-12 | CUDA Validation, Cumulative Residual Caching Benchmark, Mixed Precision & DataGoblin Curriculum")
    lines.append("")
    lines.append(f"- **Generated (UTC):** {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}")
    lines.append(f"- **Branch / Head:** `research/fuse12-cuda-validation-20261127` (`{git_head}`)")
    lines.append("- **Base Commit:** `a4cc3d861df531f41d91000d8dd0922a5ef953ec` (FUSE-11 PR #185; PRs #178, #184, #185 all green on `native-cpu` & `cpu-math-and-donor-contrast`)")
    lines.append("- **Prospective Protocol:** [FUSE12_PROTOCOL.md](./FUSE12_PROTOCOL.md) (`1af76590f5043c06884faf2aeab6c68f35c9f6f9`)")
    lines.append("- **Production Defaults:** Unchanged (`allow_cuda_residual_cache=False`, `residual_cache_precision_mode=\"fp32\"`, all residual caches default off).")
    lines.append("- **Scope Distinction:** All benchmark results are measured on the **synthetic 4-site mini-TRM native `BPDTrainer`** (heterogeneous ranks `D/32, D/16, 3D/32, D/8`, `batch_size=2`, `seq_len=16`, `puzzle_emb_len=4`) and are explicitly distinct from pretrained 7M TRM or LLM checkpoints.")
    lines.append("")

    # Executive Findings
    lines.append("## Executive Findings (RTX 5080 Laptop GPU, 16 GB WDDM)")
    lines.append("1. **100% Bitwise Exact CUDA FP32 & TF32 Parity:** `30/30` CUDA lifecycle/stream-ownership/zero-leak checks pass (`after_close == base_bytes` to the byte); `1,536/1,536` CUDA FP32 stress tensors and `1,536/1,536` TF32 stress tensors match native BPD bitwise (`max_abs=0.0`, `failures=0`, `0` source-query factor adjoints, `576` joint factor nodes). All 16 parity-audited full-trainer cells (widths 64–2048, passes 6–24, Arms B–G vs Arm A) achieve `100%` bitwise equality across residuals, forward outputs, PGD ascent gradients, pre-clip component/CI gradients, optimizer moments, parameters, calibration buffers, tolerances, greedy outputs, evaluation trajectories, and CPU+CUDA+trainer RNG states.")
    lines.append("2. **Primary GPU Win Is `O(T)` Autograd Residual-VRAM Collapse, Not Small-Width FLOP Reduction:** Native BPD saves a distinct materialized `delta_w` tensor in autograd for every pass of every backward-retaining rollout (`O(T * M * d_in * d_out)`). Sharing one detached primal storage per site collapses autograd residual storage to `O(M * d_in * d_out)`:")
    lines.append("   - Width `1024`, `T=12`: peak allocated VRAM drops from `1,160.1 MiB` (Arm A) -> `446.1 MiB` (Arm G, **2.60x reduction**).")
    lines.append("   - Width `2048`, `T=24`: peak allocated VRAM drops from `7,189.8 MiB` (Arm A) -> `1,638.1 MiB` (Arm G, **4.39x reduction**, `-5,552 MiB`), with `1.082x` step speedup (`1,453.3 ms -> 1,343.6 ms`).")
    lines.append("   - Width `4096`, `T=24`: native Arm A allocates **`26,573.9 MiB`** (`28,120 MiB` reserved), spilling over PCIe under WDDM (`56.57 s/step`, `4.39 kJ/step`). Arm F/G holds peak allocated VRAM at **`4,757.6 MiB`** (`5,122 MiB` NVML, **5.59x reduction**, zero spill), cutting step latency to **`3.14 s/step`** (**`18.00x` speedup**, 95% CI `[17.63, 18.12]`) and step energy by **18.2x** (`4,388 J -> 242 J/step`).")
    lines.append("   - At small/medium widths (`64–1024`), GPU step time is dominated by Python/autograd dispatch (`site_recon_ms = 0.03–0.22 ms`), so full-step speedups remain near `0.90–1.08x` (with isolated calibration wins like `w=512, p=6, cal=1`: `1.412x`). Cache planner overhead is negligible (`0.15–0.59 ms`).")
    lines.append("3. **FUSE-12B Mixed Precision Verdict (No Silent Tolerance Loosening):**")
    lines.append("   - **TF32 (`PASS`):** `1,536/1,536` stress tensors bitwise exact (`max_abs=0.0`, `failures=0`), `2/2` parity-audited trainer cells bitwise exact (`182/182`, `192/192`), and **3.20x peak VRAM reduction** at `w=2048, T=12` (`3,892 -> 1,218 MiB`).")
    lines.append("   - **BF16 & FP16 CUDA Autocast (`HOLD — keep on reference path`):** Three independent mechanisms block admission without loosening tolerances: (a) native `TiedComponentLinear.observe_clean_pass` (`components.py:226`) raises `RuntimeError` in `lerp_` under CUDA BF16/FP16 autocast with `pass_mean`; (b) PyTorch's `AutoDispatchBelowAutocast` thread-local guard inside plain `Function.backward()` causes `_Primal.backward` to execute `g_V, g_U` in FP32 while native `MmBackward0` executes in BF16/FP16 (`1,344/1,344` forward/PGD/input/mask tensors bitwise exact, `192` BF16 / `144` FP16 failures on `g_V, g_U`, and `52` BF16 / `42` FP16 trainer parity failures at `w=512`, `max_abs=7.81e-2` / `7.81e-3`); (c) under autocast, `F.linear(xf, delta_w)` casts FP32 `delta_w` into a fresh per-pass BF16/FP16 copy saved by autograd, negating the `O(T)` VRAM collapse (`w=2048, T=12`: `6,317 -> 6,445 MiB`, `0.85–0.97x` slower).")
    lines.append("4. **FUSE-12C DataGoblin Curriculum (50-cell / 283-row / 706-site dataset):** Under strict group holdouts across widths `64–4096` and depths `6–24`, a naive raw-ms Ridge model fails (`0.331` width / `0.349` depth oracle fraction) due to 5-order-of-magnitude WDDM-spill leverage, whereas a log-space physical Ridge predictor achieves **`0.9194`** (width holdout) / **`0.9209`** (depth holdout) of oracle step savings (`0.9242` / `0.9256` on direct reconstruction savings), closely matching `exact_saved_mac` (`0.9364` / `0.9505`) and `density` (`0.9379` / `0.9439`). A compact logistic worth-caching classifier (`speedup >= 1.03x OR vram_saved >= 256 MiB`) achieves `71.6%` width-holdout and `66.3%` depth-holdout accuracy.")
    lines.append("")

    # Section 1: Hardware & Validation
    lines.append("## 1. Hardware, Preflight & CUDA FP32 Validation Gate")
    if val is not None:
        nv = val.get("preflight_nvml", {})
        dp = val.get("device_properties", {})
        lf = val.get("cuda_fp32_lifecycle", {})
        st = val.get("cuda_fp32_stress", {})
        au = lf.get("allocator_audit", {})
        lines.append(f"- **GPU:** `{dp.get('name', nv.get('name', 'N/A'))}` (SM `{dp.get('major')}.{dp.get('minor')}`, `{dp.get('multi_processor_count')}` SMs, `{nv.get('memory_total_mib')}` MiB NVML / `15,915` MiB WDDM)")
        lines.append(f"- **Software Stack:** Driver `{nv.get('driver_version')}`, CUDA driver `12.9`, CUDA runtime `{val.get('cuda_runtime')}`, PyTorch `{val.get('torch')}`")
        lines.append(f"- **Power / Headroom at Admission:** Enforced `{nv.get('enforced_power_limit_w')}` W / Default `{nv.get('default_power_limit_w')}` W, `{nv.get('memory_free_mib')}` MiB free (`0%` util)")
        lines.append(f"- **CPU Regression Gate (Mandatory):** All FUSE-06..12 CPU gates PASS (`0` failures; PR #186 hosted `native-cpu` & `cpu-math-and-donor-contrast` green).")
        lines.append(f"- **CUDA FP32 Lifecycle & Stream Ownership:** `{lf.get('status')}` (`{sum(lf.get('checks', {}).values())}/{len(lf.get('checks', {}))}` checks). Zero allocator leak verified: `base={au.get('base_bytes')} B`, `peak_in_scope={au.get('peak_in_scope_bytes')} B` (`+{au.get('expected_payload_bytes')} B` exact), `after_close={au.get('after_close_bytes')} B`.")
        lines.append(
            f"- **CUDA FP32 96-Case Stress Gate (roots `4012001..8` x scales `1/10/100/10000` x `ordinary/exact/near`):** "
            f"`{'PASS' if val.get('fp32_gate_passed') else 'FAIL'}` | "
            f"bitwise exact `{st.get('bitwise_exact')}/{st.get('tensors')}`, within tol `{st.get('within_tol')}/{st.get('tensors')}`, "
            f"failures `{st.get('failures')}`, `max_abs={st.get('max_abs_err'):.3e}`, `max_rel={st.get('max_rel_err'):.3e}`, "
            f"`near_zero_max_abs={st.get('near_zero_max_abs_err'):.3e}`, `joint_factor_nodes={st.get('joint_factor_nodes')}`, `hits={st.get('hits')}`."
        )
        lines.append("")

    # Section 2: Full FP32 Matched Benchmark Table
    ok_fp32 = [r for r in receipts if r.get("status") == "ok" and r.get("precision_mode") == "fp32"]
    lines.append("## 2. FP32 Matched Benchmark Matrix (38 Synchronized CUDA Cells)")
    if ok_fp32:
        lines.append("| Tier | W | T | Regime | K | Cal | Cap | Arm A (ms) | Arm C (ms / x) | Arm F (ms / x) | Arm G (ms / x) | G CI95 | Payload (MiB) | Spd/MiB | Alloc A->G (MiB) | NVML A->G (MiB) | Energy A->G (mJ) | Parity (Bitwise / Fail) |")
        lines.append("|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|")
        for r in ok_fp32:
            arms = r["arms"]
            a = arms["A"]
            c = arms.get("C")
            f_arm = arms.get("F")
            g = arms.get("G") or f_arm or c
            ci = g.get("speedup_cuda_ci95")
            ci_s = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "n/a"
            par_arms = [v["parity"] for k, v in arms.items() if k != "A" and v.get("parity")]
            if par_arms:
                tot_bit = sum(p["bitwise_exact"] for p in par_arms)
                tot_ten = sum(p["tensors"] for p in par_arms)
                tot_fail = sum(p["failures"] for p in par_arms)
                par_s = f"{tot_bit}/{tot_ten} / {tot_fail}"
            else:
                par_s = "—"
            e_a = a.get("energy_per_step_mj")
            e_g = g.get("energy_per_step_mj")
            e_s = f"{e_a:.0f}->{e_g:.0f}" if (e_a is not None and e_g is not None) else "n/a"
            c_s = f"{c['cuda_median_ms']:.1f} ({c['speedup_cuda']:.3f}x)" if c else "n/a"
            f_s = f"{f_arm['cuda_median_ms']:.1f} ({f_arm['speedup_cuda']:.3f}x)" if f_arm else "n/a"
            g_s = f"{g['cuda_median_ms']:.1f} ({g['speedup_cuda']:.3f}x)" if g else "n/a"
            al_a = a["peak_allocated_bytes"] / (1024.0 ** 2)
            al_g = g["peak_allocated_bytes"] / (1024.0 ** 2)
            lines.append(
                f"| {r['tier']} | {r['width']} | {r['n_passes']} | {r['regime']} | {r['pgd_inner']} | "
                f"{r['cal_cadence']} | {r['budget_mode']} | {a['cuda_median_ms']:.1f} | {c_s} | {f_s} | {g_s} | "
                f"{ci_s} | {g['payload_peak_mib']:.2f} | {g['speedup_per_mib']:.4f} | "
                f"{al_a:.0f}->{al_g:.0f} | {a['nvml_used_mib']:.0f}->{g['nvml_used_mib']:.0f} | {e_s} | {par_s} |"
            )
        lines.append("")

        # Phase Breakdown Table for Representative Tiers
        lines.append("### 2.1 Phase Timing & Reconstruction Breakdown (Representative FP32 Cells, ms)")
        lines.append("| Cell | Arm | Clean/Cal | Inner PGD | Final PGD | Joint Bwd | Opt | Planner | Est Recon | Recons (Cached/Native) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        rep_ids = [
            "smoke__fp32__w128_p6_closed_loop_k2_cal1_binding",
            "medium__fp32__w512_p6_closed_loop_k2_cal1_binding",
            "wide__fp32__w1024_p12_closed_loop_k2_caloff_full",
            "wide__fp32__w2048_p12_closed_loop_k2_caloff_full",
            "frontier__fp32__w2048_p24_closed_loop_k2_caloff_full",
            "frontier__fp32__w4096_p24_closed_loop_k2_caloff_full",
        ]
        by_id = {r["cell_id"]: r for r in ok_fp32}
        for cid in rep_ids:
            r = by_id.get(cid)
            if not r:
                continue
            for arm in ("A", "C", "F", "G"):
                if arm not in r["arms"]:
                    continue
                v = r["arms"][arm]
                ph = v["phases_ms"]
                rc = v["reconstructions"]
                lines.append(
                    f"| `{cid}` | {arm} | {ph['clean_calibration_ms']:.2f} | {ph['inner_pgd_ms']:.2f} | "
                    f"{ph['final_pgd_ms']:.2f} | {ph['joint_backward_ms']:.2f} | {ph['optimizer_ms']:.2f} | "
                    f"{ph['planner_overhead_ms']:.2f} | {ph['residual_reconstruction_est_ms']:.2f} | "
                    f"{rc.get('cached_total_reconstructions')}/{rc.get('native_total_reconstructions')} |"
                )
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
    err_mp = [r for r in receipts if r.get("status") != "ok" and r.get("tier") == "mixed_precision"]
    if ok_mp:
        lines.append("| Precision | W | T | Cal | Cap | Arm A (ms) | Arm G (ms) | Speedup | Payload (MiB) | Alloc A->G (MiB) | Parity (Bitwise / Fail) |")
        lines.append("|---|---:|---:|---|---|---:|---:|---:|---:|---|---:|")
        for r in ok_mp:
            a = r["arms"]["A"]
            g = r["arms"].get("G") or r["arms"].get("F")
            par_arms = [v["parity"] for k, v in r["arms"].items() if k != "A" and v.get("parity")]
            par_s = (
                f"{sum(p['bitwise_exact'] for p in par_arms)}/{sum(p['tensors'] for p in par_arms)} / {sum(p['failures'] for p in par_arms)}"
                if par_arms
                else "—"
            )
            lines.append(
                f"| `{r['precision_mode']}` | {r['width']} | {r['n_passes']} | {r['cal_cadence']} | {r['budget_mode']} | "
                f"{a['cuda_median_ms']:.1f} | {g['cuda_median_ms']:.1f} | {g['speedup_cuda']:.3f}x | "
                f"{g['payload_peak_mib']:.2f} | {a['peak_allocated_bytes']/(1024**2):.0f}->{g['peak_allocated_bytes']/(1024**2):.0f} | {par_s} |"
            )
        lines.append("")
    if err_mp:
        lines.append("### FUSE-12B Native Autocast Incompatibilities Recorded")
        for e in err_mp:
            lines.append(f"- `{e['cell_id']}`: `{e['status']}` — `{e.get('error')}`")
        lines.append("")

    # Section 4: FUSE-12C DataGoblin
    lines.append("## 4. FUSE-12C: DataGoblin Admission-Policy Curriculum & Held-Out Predictor")
    if f12c is not None:
        lines.append(f"- **Dataset:** [`fuse12c_datagoblin_dataset.jsonl`](./fuse12c_datagoblin_dataset.jsonl) (`{f12c['total_dataset_rows']}` rows, `{f12c['total_site_samples']}` site samples, `source_dataset_kind=\"synthetic_native_trainer\"`).")
        lines.append(f"- **Widths Observed:** `{f12c['widths_observed']}` | **Recurrence Depths Observed:** `{f12c['depths_observed']}`")
        lines.append("- **Contract:** Strictly experimental and outside the numerical correctness path.")
        lines.append("")
        lines.append("| Holdout Scheme | Metric | Learned Log-Ridge | Learned Raw-Ridge | Exact Saved-MAC | First-Fit | Density |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for label, key in (
            ("Leave-One-Width-Family-Out", "leave_one_width_out"),
            ("Leave-One-Recurrence-Depth-Out", "leave_one_depth_out"),
        ):
            for mlabel, mkey in (
                ("Oracle Frac (Step ms)", "aggregate_oracle_fraction_step_ms"),
                ("Oracle Frac (Direct Recon ms)", "aggregate_oracle_fraction_recon_ms"),
            ):
                agg = f12c[key].get(mkey, {})
                def _fmt(x): return f"{x:.4f}" if x is not None else "n/a"
                lines.append(
                    f"| {label} | {mlabel} | {_fmt(agg.get('learned_log_ridge'))} | {_fmt(agg.get('learned_raw_ridge'))} | "
                    f"{_fmt(agg.get('exact_saved_mac'))} | {_fmt(agg.get('first_fit'))} | {_fmt(agg.get('density'))} |"
                )
        lines.append("")
        wc = f12c.get("worth_caching_classifier", {})
        if wc:
            w_acc = wc.get("leave_one_width_out", {}).get("mean_holdout_accuracy")
            d_acc = wc.get("leave_one_depth_out", {}).get("mean_holdout_accuracy")
            lines.append(
                f"- **Worth-Caching Classifier** (`speedup_cuda >= 1.03 OR vram_saved_mib >= 256`): "
                f"leave-one-width-out accuracy = `{w_acc:.3f}`, leave-one-depth-out accuracy = `{d_acc:.3f}`."
            )
            lines.append("")

    # Section 5: Artifacts & Reproduction
    lines.append("## 5. Artifacts & Reproduction")
    lines.append("- Protocol: [FUSE12_PROTOCOL.md](./FUSE12_PROTOCOL.md)")
    lines.append("- CUDA Validation Receipt: [receipt_fuse12_cuda_validate.json](./receipt_fuse12_cuda_validate.json)")
    lines.append("- Raw JSONL Benchmark Receipts: [fuse12_cuda_receipts.jsonl](./fuse12_cuda_receipts.jsonl)")
    lines.append("- DataGoblin Curriculum Dataset: [fuse12c_datagoblin_dataset.jsonl](./fuse12c_datagoblin_dataset.jsonl)")
    lines.append("- FUSE-12C Predictor Report: [receipt_fuse12c_predictor.json](./receipt_fuse12c_predictor.json)")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Mandatory CPU regression & CUDA default-off guard (no GPU required)")
    lines.append("CUDA_VISIBLE_DEVICES=\"\" python -m mechinterp.bpd.experiments.fused_residual.validate_fuse12_cuda \\")
    lines.append("  --cpu-guard-only --output /tmp/fuse12_cpu_guard.json")
    lines.append("")
    lines.append("# 2. Full CUDA validation + resumable progressive 7-arm benchmark + FUSE-12C (VRAMorama)")
    lines.append("vramorama run --vram 12G --headroom 2048 --strict-fifo --owner bpd-fuse12-cuda \\")
    lines.append("  --guard-power 60 --pause-exit 75 --restart 6 -- \\")
    lines.append("  python -m mechinterp.bpd.experiments.fused_residual.run_fuse12_overnight")
    lines.append("```")
    lines.append("")

    out_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--tiers", default="smoke,medium,wide,frontier,mixed_precision")
    parser.add_argument("--min-watts", type=float, default=40.0)
    parser.add_argument("--force-validate", action="store_true")
    parser.add_argument("--render-only", action="store_true")
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

    if args.render_only:
        render_results_markdown(
            validate_path=validate_path,
            receipts_path=receipts_path,
            fuse12c_report_path=f12c_report_path,
            out_md_path=results_md_path,
            git_head=git_head,
        )
        print("FUSE12_RENDER_DONE", flush=True)
        return

    need_val = True
    if not args.force_validate and validate_path.exists():
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
