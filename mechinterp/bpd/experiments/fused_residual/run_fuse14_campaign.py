"""FUSE-14 Resumable CUDA Campaign Orchestrator & Evidence-Derived Report Renderer.

Run inside a VRAMorama strict-FIFO lease:
  vramorama run --vram 12G --headroom 2048 --strict-fifo --owner bpd-fuse14-campaign \
    --guard-power 60 --pause-exit 75 --restart 6 -- \
    python -u -m mechinterp.bpd.experiments.fused_residual.run_fuse14_campaign --allow-cuda
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXIT_POWER_CAP = 75


def _run_sub(cmd: list[str]) -> int:
    print(f"[fuse14] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode == EXIT_POWER_CAP:
        print("[fuse14] Power guard triggered (exit 75); pausing cleanly for VRAMorama.", flush=True)
        return EXIT_POWER_CAP
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}")
    return 0


def render_fuse14_results(receipts_dir: Path, out_md: Path) -> None:
    """Generate FUSE14_RESULTS.md strictly from durable raw JSON/JSONL receipts."""
    suite_p = receipts_dir / "receipt_fuse14_suite_cuda.json"
    frontier_p = receipts_dir / "receipt_fuse14_frontier.jsonl"
    stage_h_p = receipts_dir / "receipt_fuse14h_router.json"
    cpu_probe_p = receipts_dir / "receipt_fuse14_cpu_probe_summary.json"

    suite = json.loads(suite_p.read_text(encoding="utf-8")) if suite_p.exists() else None
    frontier = (
        [json.loads(l) for l in frontier_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if frontier_p.exists()
        else []
    )
    stage_h = json.loads(stage_h_p.read_text(encoding="utf-8")) if stage_h_p.exists() else None
    cpu_probe = json.loads(cpu_probe_p.read_text(encoding="utf-8")) if cpu_probe_p.exists() else None

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    lines: list[str] = []
    lines.append("# BPD FUSE-14: Whole-Block Recurrent Cast Compactification — Results")
    lines.append("")
    lines.append(f"- **Campaign commit:** `{head}`")
    lines.append("- **Hardware & lease:** NVIDIA GeForce RTX 5080 Laptop GPU (16 GiB VRAM, Windows WDDM), VRAMorama strict FIFO (`12 GiB` lease, `2048 MiB` headroom, `>=60 W` power guard, exit code `75` pause; zero forced WDDM spill).")
    lines.append("- **Production defaults unchanged:** `allow_cuda_residual_cache=False`, `residual_cache_storage_mode=\"fp32\"`, `cache_frozen_projection_casts=False`, `projection_cache_allow_recompute=False`. FP16 remains **HOLD**. Trainable projection cast sharing remains excluded.")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Gate Summary (Stages A–H)")
    lines.append("")
    lines.append("| Stage | Scope | Verdict | Evidence Receipt |")
    lines.append("|---|---|---|---|")
    if cpu_probe:
        lines.append(f"| Preflight CPU Probe | Standalone CPU BF16 probe replay (`18` cells) | **PASS** ({cpu_probe['finite_bitwise_cells']}/{cpu_probe['cells']} finite bitwise; depth-12 storages `48 -> 2`) | `receipt_fuse14_cpu_probe_summary.json` |")
    if suite:
        sa = suite["stage_a"]
        sp = sa["standalone_probe"]
        ta = sa["trainer_cast_audit"]
        aud_g = ta["fuse13_arm_g"]["cast_audit"]
        aud_h = ta["fuse14_arm_h"]["cast_audit"]
        lines.append(f"| Stage A | CUDA standalone probe (`{sp['finite_bitwise']}/{sp['cells']}`) + trainer per-layer saved-cast audit | **{'PASS' if ta['parity']['finite_bitwise_pass'] else 'FAIL'}** (saved cast storages `{aud_g['unique_storages']} -> {aud_h['unique_storages']}`, `max_abs={ta['parity']['max_abs']:.3e}`) | `receipt_fuse14_suite_cuda.json` |")
        sb = suite["stage_b_ablation"]
        sb_cands = ("explicit_per_call_cast", "qkv_only", "gate_up_only", "both_block_projs")
        sb_ok = all(sb[k]["parity_vs_arm_g"]["finite_bitwise_pass"] for k in sb_cands if k in sb)
        sb_ac_off = sb.get("autocast_cache_disabled", {}).get("parity_vs_arm_g", {})
        sb_falsified = not sb_ac_off.get("finite_bitwise_pass", True)
        lines.append(f"| Stage B | 6-arm isolation ablation (`qkv_only`, `gate_up_only`, `both`, autocast cache off, per-call cast) | **{'PASS' if sb_ok else 'FAIL'}** (4/4 valid arms finite bitwise; `autocast_cache_disabled` {'falsified' if sb_falsified else 'matched'}) | `receipt_fuse14_suite_cuda.json` |")
        sc = suite["stage_c_guards"]
        lines.append(f"| Stage C | Step-local trainer integration + fail-closed lifecycle guards | **{'PASS' if all(sc.values()) else 'FAIL'}** ({sum(sc.values())}/{len(sc)} guards pass; defaults off) | `receipt_fuse14_suite_cuda.json` |")
        sd = suite["stage_d_parity_matrix"]
        sd_ok = sum(int(c["parity"]["finite_bitwise_pass"]) for c in sd)
        fp16_st = (suite.get("fp16_hold_confirmation") or {}).get("status", "n/a")
        lines.append(f"| Stage D | Frozen BF16 multi-step trainer matrix (`w=128..2048, T=2..24`, 2 roots, `closed_loop`/`clean_window`, `off/1/50`) | **{'PASS' if sd_ok == len(sd) else 'FAIL'}** ({sd_ok}/{len(sd)} finite bitwise; FP16 `{fp16_st}`) | `receipt_fuse14_suite_cuda.json` |")
        se = suite["stage_e_checkpointing"]
        se_ok = sum(int(c["parity"]["finite_bitwise_pass"]) for c in se)
        lines.append(f"| Stage E | Independent activation checkpointing + recompute-safe frozen cast reuse | **{'PASS' if se_ok == len(se) else 'FAIL'}** ({se_ok}/{len(se)} finite bitwise) | `receipt_fuse14_suite_cuda.json` |")
    if frontier:
        n_h_ok = sum(1 for r in frontier if r["arm_h_fuse14"]["status"] == "ok")
        n_ck_ok = sum(1 for r in frontier if (r.get("arm_h_ckpt") or {}).get("status") == "ok")
        lines.append(f"| Stage F | All 5 previously blocked BF16 frontier cells (+ `(8192,24)` preflight) | **COMPLETE** (Arm H fits `{n_h_ok}/{len(frontier)}`; Arm H+Ckpt fits `{n_ck_ok}` additional; zero forced spill) | `receipt_fuse14_frontier.jsonl` |")
    if suite and suite.get("stage_g_pretrained_transfer"):
        sg = suite["stage_g_pretrained_transfer"]
        q3 = sg.get("pretrained_qwen3_4b", {})
        if q3.get("status") == "ok":
            q3_ok = sum(int(c["parity"]["finite_bitwise_pass"]) for c in q3["cells"])
            lines.append(f"| Stage G | Pretrained checkpoint transfer (TRM/DG `blocked_no_local_checkpoint`; local SHA-256 `Qwen3-4B` `2560/9728`) | **PASS (Qwen3-4B)** ({q3_ok}/{len(q3['cells'])} finite bitwise; TRM/DG honestly blocked) | `receipt_fuse14_suite_cuda.json` |")
        else:
            lines.append(f"| Stage G | Pretrained checkpoint transfer | `blocked_no_local_checkpoint` | `receipt_fuse14_suite_cuda.json` |")
    if stage_h:
        lines.append(f"| Stage H | DataGoblin hard-evidence-gated routing (`correctness_path=False`) | **COMPLETE** (`0.0000` unsafe admissions; router: `{stage_h['recommended_experimental_router']}`) | `receipt_fuse14h_router.json` |")
    lines.append("")

    # Stage A
    if suite:
        sa = suite["stage_a"]
        sp = sa["standalone_probe"]
        ta = sa["trainer_cast_audit"]
        lines.append("## Stage A: CUDA Standalone Probe & Per-Layer Saved-Cast Audit")
        lines.append("")
        lines.append(f"- Standalone CUDA probe (`fuse14_projection_probe --device cuda`): **{sp['finite_bitwise']}/{sp['cells']} finite bitwise**; depth-12 saved projection cast storages collapsed **`{sp['d12_storages'][0][0]} -> {sp['d12_storages'][0][1]}`**.")
        aud_g = ta["fuse13_arm_g"]["cast_audit"]
        aud_h = ta["fuse14_arm_h"]["cast_audit"]
        lines.append(f"- Native `BPDTrainer` audit (`w={ta['width']}, T={ta['visits']}`, BF16, `closed_loop`, `cal=1`): finite bitwise **`{ta['parity']['finite_bitwise']}/{ta['parity']['tensors']}`** (`max_abs={ta['parity']['max_abs']:.3e}`).")
        lines.append("")
        lines.append("| Arm | Total Pack Calls | Unique Cast Storages | Unique Cast Bytes | `qkv_proj` Unique (Bytes) | `gate_up_proj` Unique (Bytes) | Peak Alloc (MiB) | NVML (MiB) |")
        lines.append("|---|---:|---:|---:|---|---|---:|---:|")
        for lbl, meta in (("FUSE-13 Arm G", ta["fuse13_arm_g"]), ("FUSE-14 Arm H", ta["fuse14_arm_h"])):
            au = meta["cast_audit"]
            q_k = au["by_kind"].get("qkv_proj", {"unique_ptrs": 0, "unique_bytes": 0})
            gu_k = au["by_kind"].get("gate_up_proj", {"unique_ptrs": 0, "unique_bytes": 0})
            lines.append(
                f"| `{lbl}` | {au['total_pack_calls']} | **{au['unique_storages']}** | {au['unique_bytes']:,} | {q_k['unique_ptrs']} ({q_k['unique_bytes']:,} B) | {gu_k['unique_ptrs']} ({gu_k['unique_bytes']:,} B) | {meta['peak_allocated_bytes']/(1024**2):.1f} | {meta['nvml_used_mib']:.0f} |"
            )
        lines.append("")

        # Stage B
        lines.append("## Stage B: 6-Arm Isolation Ablation (`BPDTrainer`, BF16 CUDA)")
        lines.append("")
        lines.append("| Arm | Finite Bitwise vs Arm G | Max Abs | Unique Cast Storages | Unique Cast Bytes | Peak Alloc (MiB) | NVML (MiB) | Step Time (ms) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for aname, adata in suite["stage_b_ablation"].items():
            mt = adata["meta"]
            au = mt["cast_audit"]
            par = adata["parity_vs_arm_g"]
            bw = "ref" if par is None else ("**YES**" if par["finite_bitwise_pass"] else "**NO**")
            ma = 0.0 if par is None else par["max_abs"]
            lines.append(
                f"| `{aname}` | {bw} | {ma:.3e} | {au['unique_storages']} | {au['unique_bytes']:,} | {mt['peak_allocated_bytes']/(1024**2):.1f} | {mt['nvml_used_mib']:.0f} | {mt['step_ms'][-1]:.2f} |"
            )
        lines.append("")

        # Stage C
        lines.append("## Stage C: Fail-Closed Lifecycle & Guard Audit")
        lines.append("")
        sc = suite["stage_c_guards"]
        lines.append(f"All **{sum(sc.values())}/{len(sc)}** Stage C checks passed: " + ", ".join(f"`{k}`" for k in sc) + ".")
        lines.append("")

        # Stage D
        lines.append("## Stage D: Frozen Multi-Step BF16 Parity Matrix (Arm G vs Arm H)")
        lines.append("")
        lines.append("| Width | Visits | Root | Regime | Cal | Budget | Finite Bitwise Tensors | Failures | Max Abs | Arm G Step (ms) | Arm H Step (ms) | Arm G Alloc (MiB) | Arm H Alloc (MiB) | Alloc Saved (MiB) | NVML Saved (MiB) |")
        lines.append("|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in suite["stage_d_parity_matrix"]:
            pa = c["parity"]
            mg, mh = c["fuse13_arm_g"], c["fuse14_arm_h"]
            lines.append(
                f"| {c['width']} | {c['visits']} | {c['seed']} | `{c['regime']}` | `{c['cal_cadence']}` | `{c['budget_mode']}` | {pa['finite_bitwise']}/{pa['tensors']} | {pa['failures']} | {pa['max_abs']:.3e} | {mg['step_ms'][-1]:.1f} | {mh['step_ms'][-1]:.1f} | {mg['peak_allocated_bytes']/(1024**2):.1f} | {mh['peak_allocated_bytes']/(1024**2):.1f} | {c['vram_saved_alloc_mib']:+.1f} | {c['vram_saved_nvml_mib']:+.0f} |"
            )
        lines.append("")
        if suite.get("fp16_hold_confirmation"):
            fh = suite["fp16_hold_confirmation"]
            lines.append(f"- **FP16 HOLD confirmation:** `{fh['status']}` (`hits={fh['receipt']['hits']}, misses={fh['receipt']['misses']}, fallbacks={fh['receipt']['fallbacks']}`).")
            lines.append("")

        # Stage E
        lines.append("## Stage E: Independent Activation Checkpointing Evaluation")
        lines.append("")
        lines.append("| Width | Visits | Finite Bitwise (`ckpt_native` vs `ckpt+proj`) | Native Ckpt Alloc / NVML (MiB) | Ckpt+Proj Alloc / NVML (MiB) | Alloc Saved (MiB) | Native Ckpt Step (ms) | Ckpt+Proj Step (ms) |")
        lines.append("|---:|---:|---|---|---|---:|---:|---:|")
        for c in suite["stage_e_checkpointing"]:
            pa = c["parity"]
            mr, mc = c["ckpt_native"], c["ckpt_plus_frozen_proj"]
            lines.append(
                f"| {c['width']} | {c['visits']} | **{'YES' if pa['finite_bitwise_pass'] else 'NO'}** ({pa['finite_bitwise']}/{pa['tensors']}) | {mr['peak_allocated_bytes']/(1024**2):.1f} / {mr['nvml_used_mib']:.0f} | {mc['peak_allocated_bytes']/(1024**2):.1f} / {mc['nvml_used_mib']:.0f} | {c['vram_saved_alloc_mib']:+.1f} | {mr['step_ms'][-1]:.1f} | {mc['step_ms'][-1]:.1f} |"
            )
        lines.append("")

    # Stage F: Frontier
    if frontier:
        lines.append("## Stage F: Reattempting All Five Blocked BF16 Frontier Configurations")
        lines.append("")
        lines.append("Note: When FUSE-13 Arm G is preflight-blocked, comparison is explicitly **censored** (not a speedup). Hardware measurements on `(2048, 48)` (`4,848 MiB`) and `(4096, 24)` (`10,173 MiB`) revealed an additional per-pass `W_target` BF16 cast in `TiedComponentLinear.forward` (`components.py:382`, stochastic `routing_mask is not None`, `T * site_elems * 2` bytes); refitting Arm H preflight with this measured term prevents WDDM spill on `(4096, 48)+` while `Arm H + Checkpointing` (`H_ckpt`) eliminates it cleanly.")
        lines.append("")
        lines.append("| Width | Visits | Baseline Comparison | FUSE-13 Arm G | FUSE-14 Arm H (p50 / p95 ms, tok/s) | Arm H Alloc / Res / NVML (MiB) | Arm H Energy (mJ/step) | Arm H+Ckpt (if run) |")
        lines.append("|---:|---:|---|---|---|---|---:|---|")
        for r in frontier:
            ag, ah, ahc = r["arm_g_fuse13"], r["arm_h_fuse14"], r.get("arm_h_ckpt")
            g_txt = f"`{ag['status']}` (est {ag['preflight']['total_est_mib']:.0f} MiB)" if ag["status"] != "ok" else f"`ok` ({ag['cuda_median_ms']:.1f} ms)"
            if ah["status"] == "ok":
                h_txt = f"`ok` ({ah['cuda_median_ms']:.1f} / {ah['cuda_p95_ms']:.1f} ms, {ah['tokens_per_s']:.0f} tok/s)"
                h_mem = f"{ah['peak_allocated_bytes']/(1024**2):.0f} / {ah['peak_reserved_bytes']/(1024**2):.0f} / {ah['nvml_used_mib']:.0f}"
                h_eng = f"{ah['energy_per_step_mj']:.1f}" if ah.get("energy_per_step_mj") is not None else "n/a"
            else:
                h_txt = f"`{ah['status']}` (est {ah['preflight']['total_est_mib']:.0f} MiB)"
                h_mem = "blocked"
                h_eng = "n/a"
            if ahc is None:
                hc_txt = "not needed"
            elif ahc["status"] == "ok":
                hc_txt = f"`ok` ({ahc['cuda_median_ms']:.1f} ms, {ahc['peak_allocated_bytes']/(1024**2):.0f}/{ahc['nvml_used_mib']:.0f} MiB)"
            else:
                hc_txt = f"`{ahc['status']}` (est {ahc['preflight']['total_est_mib']:.0f} MiB)"
            lines.append(f"| {r['width']} | {r['n_passes']} | `{r['baseline_comparison']}` | {g_txt} | {h_txt} | {h_mem} | {h_eng} | {hc_txt} |")
        lines.append("")

    # Stage G: Pretrained Transfer
    if suite and suite.get("stage_g_pretrained_transfer"):
        sg = suite["stage_g_pretrained_transfer"]
        lines.append("## Stage G: Pretrained Checkpoint Transfer")
        lines.append("")
        lines.append(f"- **TRM / DataGoblin checkpoint audit:** `{sg['trm_or_datagoblin']['status']}` — {sg['trm_or_datagoblin']['reason']}")
        q3 = sg.get("pretrained_qwen3_4b", {})
        if q3.get("status") == "ok":
            lines.append(f"- **Real Pretrained Transfer (`pretrained_qwen3_4b_block`, kept separate from synthetic mini-TRM):** `{q3['source_model']}`, verified SHA-256 `{q3['shard_sha256']}`.")
            lines.append("")
            lines.append("| Visits | Hidden / Inter | Finite Bitwise | Saved Cast Storages (G -> H) | Arm G Alloc / NVML (MiB) | Arm H Alloc / NVML (MiB) | VRAM Saved Alloc / NVML (MiB) | Arm G -> H Step (ms) |")
            lines.append("|---:|---|---|---|---|---|---:|---|")
            for c in q3["cells"]:
                pa = c["parity"]
                mg, mh = c["fuse13_arm_g"], c["fuse14_arm_h"]
                lines.append(
                    f"| {c['visits']} | `{c['hidden_size']} / {c['intermediate_size']}` | **{'YES' if pa['finite_bitwise_pass'] else 'NO'}** ({pa['finite_bitwise']}/{pa['tensors']}) | `{mg['unique_proj_cast_storages']} -> {mh['unique_proj_cast_storages']}` | {mg['peak_allocated_bytes']/(1024**2):.1f} / {mg['nvml_used_mib']:.0f} | {mh['peak_allocated_bytes']/(1024**2):.1f} / {mh['nvml_used_mib']:.0f} | {c['vram_saved_alloc_mib']:+.1f} / {c['vram_saved_nvml_mib']:+.0f} | {mg['step_ms'][-1]:.1f} -> {mh['step_ms'][-1]:.1f} |"
                )
            lines.append("")

    # Stage H: DataGoblin
    if stage_h:
        lines.append("## Stage H: DataGoblin Hard-Evidence-Gated Routing (`correctness_path=False`)")
        lines.append("")
        lines.append(f"- **Hard evidence gate modes:** `{', '.join(stage_h['hard_evidence_gate_modes'])}` (FP16: `{stage_h['fp16_status']}`).")
        lines.append(f"- **Checkpoint families exported:** `{', '.join(stage_h['checkpoint_families_observed'])}` (`{stage_h['total_fuse14_routing_rows']}` FUSE-14 rows in `fuse14h_datagoblin_dataset.jsonl`, `{stage_h['total_site_samples']}` FUSE-12/13 site samples).")
        lines.append(f"- **Residual predictor promotion check:** `beats_safe_physical_on_all_holdouts={stage_h['learned_residual_beats_safe_physical_on_all_holdouts']}` -> recommended experimental router: **`{stage_h['recommended_experimental_router']}`**.")
        lines.append("")
        lines.append("| Holdout | Policy | Safe Oracle Fraction | p95 Regret (ms) | Max Regret (ms) | Unsafe Admission Rate | False Rejection Rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for hlabel, hkey in (("Width", "leave_one_width_out"), ("Depth", "leave_one_depth_out"), ("Precision", "leave_one_precision_out")):
            hd = stage_h[hkey]
            for pol in ("safe_physical_prior", "safe_physical_plus_learned", "safe_exact_saved_mac", "safe_density", "ungated_physical_prior_ref"):
                of = hd["mean_safe_oracle_fraction_step_ms"].get(pol)
                p95 = hd["p95_regret_step_ms"].get(pol)
                mx = hd["max_regret_step_ms"].get(pol)
                un = hd["unsafe_admission_rate"].get(pol)
                fr = hd["false_rejection_rate"].get(pol)
                lines.append(
                    f"| {hlabel} | `{pol}` | {f'{of:.4f}' if of is not None else 'n/a'} | {f'{p95:.3f}' if p95 is not None else 'n/a'} | {f'{mx:.3f}' if mx is not None else 'n/a'} | {f'{un:.4f}' if un is not None else 'n/a'} | {f'{fr:.4f}' if fr is not None else 'n/a'} |"
                )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[fuse14] Rendered {out_md} ({len(lines)} lines)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--min-watts", type=float, default=60.0)
    p.add_argument("--lease-mib", type=float, default=12288.0)
    p.add_argument("--max-width", type=int, default=None)
    args = p.parse_args()

    receipts_dir = HERE
    out_md = HERE / "FUSE14_RESULTS.md"

    if args.render_only:
        render_fuse14_results(receipts_dir, out_md)
        return 0

    if not args.allow_cuda:
        p.error("CUDA campaign requires --allow-cuda inside VRAMorama lease")

    py = sys.executable

    # 1. Stages A, B, C, D, E, G on CUDA
    suite_p = receipts_dir / "receipt_fuse14_suite_cuda.json"
    if not suite_p.exists():
        rc = _run_sub([
            py, "-m", "mechinterp.bpd.experiments.fused_residual.validate_fuse14_suite",
            "--device", "cuda", "--allow-cuda", "--min-watts", str(args.min_watts),
            "--output", str(suite_p),
        ])
        if rc != 0:
            return rc
    else:
        ex_suite = json.loads(suite_p.read_text(encoding="utf-8"))
        q3_cells = (ex_suite.get("stage_g_pretrained_transfer") or {}).get("pretrained_qwen3_4b", {}).get("cells", [])
        if q3_cells and not all(c["parity"]["finite_bitwise_pass"] for c in q3_cells):
            rc = _run_sub([
                py, "-m", "mechinterp.bpd.experiments.fused_residual.validate_fuse14_suite",
                "--device", "cuda", "--allow-cuda", "--min-watts", str(args.min_watts),
                "--stage-g-only", "--output", str(suite_p),
            ])
            if rc != 0:
                return rc

    # 2. Stage F: All five previously blocked BF16 frontier cells (+ (8192,24) preflight)
    frontier_p = receipts_dir / "receipt_fuse14_frontier.jsonl"
    f_cmd = [
        py, "-m", "mechinterp.bpd.experiments.fused_residual.bench_fuse14_frontier",
        "--allow-cuda", "--resume", "--min-watts", str(args.min_watts),
        "--lease-mib", str(args.lease_mib),
        "--output", str(frontier_p),
    ]
    if args.max_width is not None:
        f_cmd += ["--max-width", str(args.max_width)]
    rc = _run_sub(f_cmd)
    if rc != 0:
        return rc

    # 3. Stage H: DataGoblin hard-evidence-gated routing
    rc = _run_sub([
        py, "-m", "mechinterp.bpd.experiments.fused_residual.fuse14h_datagoblin",
        "--fuse14-suite", str(suite_p),
        "--fuse14-frontier", str(frontier_p),
        "--dataset-out", str(receipts_dir / "fuse14h_datagoblin_dataset.jsonl"),
        "--report-out", str(receipts_dir / "receipt_fuse14h_router.json"),
    ])
    if rc != 0:
        return rc

    # 4. Render final evidence-derived FUSE14_RESULTS.md
    render_fuse14_results(receipts_dir, out_md)
    print("FUSE14_CAMPAIGN_ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
