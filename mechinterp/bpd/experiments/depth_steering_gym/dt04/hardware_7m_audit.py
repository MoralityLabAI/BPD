"""BPD-DT04 empirical hardware envelope & offline SHA-verified 7M TRM bridge audit.

Directly measures the actual RTX 5080 VRAM and host DRAM footprint behind the
`ui/index.html` Compute Planning arithmetic lower bounds, and exercises
`training.NativeSudokuTRMBridge` on locally cached SHA-verified `yagizdevre`
and `gaoxin492` 7M checkpoints using fresh non-overlapping 9x9 Sudoku boards.

Safety invariants:
- Strictly offline (`HF_HUB_OFFLINE=1`, `local_files_only=True`); never downloads weights.
- Verifies exact checkpoint SHA-256 before loading.
- Never opens or references the sealed ARG-03-N 168-case dataset.
- Enforces the host RSS (`<= 12 GiB`) and CUDA (`<= 12.5 GiB`) ceilings throughout.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch

from mechinterp.bpd.adapters.trm import encode_sudoku_pairs
from mechinterp.dark_fiber_gyms import sudoku_roster as SR
from mechinterp.loopscope02.scripts import sudoku_prop as SP
from mechinterp.loopscope02.scripts.trm_sudoku import TRM_CHECKPOINTS, PureSudokuTRM

from ..dt03_causal import output_weight_basis
from ..training import NativeSudokuTRMBridge
from .memory_guard import enforce_memory_ceiling, memory_snapshot

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
LOOP02_BUNDLE = REPO / "mechinterp" / "loopscope02" / "results" / "benchmark_bundle.json"
LOOP04_REP = REPO / "supplements" / "loopscope04_arg03_receipts" / "loopscope04" / "rep_eval_strata.json"
DT04_9X9_SEED = 2604700


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_offline_7m_checkpoint(tag: str) -> tuple[Path, str]:
    """Locate local HF-cached 7M TRM checkpoint with `local_files_only=True` and check SHA-256."""
    if tag not in TRM_CHECKPOINTS:
        raise ValueError(f"Unknown 7M TRM checkpoint tag: {tag}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    meta = TRM_CHECKPOINTS[tag]
    from huggingface_hub import hf_hub_download

    local_path = Path(
        hf_hub_download(repo_id=meta["repo_id"], filename=meta["filename"], local_files_only=True)
    )
    digest = _sha256_file(local_path)
    if digest != meta["sha256"]:
        raise RuntimeError(f"7M checkpoint SHA-256 mismatch for {tag}: {digest} != {meta['sha256']}")
    return local_path, digest


def generate_screened_9x9_boards(
    *, n_stalled: int = 8, n_prop: int = 8, seed: int = DT04_9X9_SEED
) -> tuple[list[dict], dict]:
    """Generate fresh solver-unique 9x9 boards screened against public LOOPSCOPE02/04 sets."""
    if not LOOP02_BUNDLE.is_file() or not LOOP04_REP.is_file():
        raise FileNotFoundError("Public LOOPSCOPE02/04 exclusion files missing")
    rep_ids = {r["case_id"] for r in json.loads(LOOP04_REP.read_text(encoding="utf-8"))["rows"]}
    banned, banned_wl, excluded_sha = SR._build_exclusion_index([LOOP02_BUNDLE])
    rng = random.Random(int(seed))
    buckets: dict[str, list[dict]] = {"S4_stalled": [], "WB-Prop": []}
    attempts = 0
    while (len(buckets["S4_stalled"]) < n_stalled or len(buckets["WB-Prop"]) < n_prop) and attempts < 50_000:
        attempts += 1
        target = "S4_stalled" if len(buckets["S4_stalled"]) < n_stalled else "WB-Prop"
        sol = SR.generate_diverse_solution(rng)
        if target == "WB-Prop":
            clues = rng.randint(34, 40)
            puz = SR.delete_clues(sol, clues, rng)
            if puz is None:
                continue
        else:
            # Preferentially remove clues until naked/hidden-single propagation stalls (`not prop.solved`)
            # while maintaining exact uniqueness (`count_solutions == 1`).
            grid = list(sol)
            idxs = list(range(81))
            rng.shuffle(idxs)
            cur_clues = 81
            for cell_i in idxs:
                if cur_clues <= 25:
                    break
                old_ch = grid[cell_i]
                grid[cell_i] = "."
                if SR.count_solutions("".join(grid)) != 1:
                    grid[cell_i] = old_ch
                else:
                    cur_clues -= 1
            puz = "".join(grid)
        fp = hashlib.sha256(puz.encode("ascii")).hexdigest()
        if fp in rep_ids:
            continue
        if SR.count_solutions(puz) != 1:
            raise AssertionError("Non-unique 9x9 puzzle generated")
        prop = SP.trace_propagation(puz)
        stratum = "WB-Prop" if prop.solved else "S4_stalled"
        if stratum != target:
            continue
        if not SR._is_novel_and_register(puz, banned, banned_wl):
            continue
        buckets[stratum].append({
            "id": fp,
            "question": puz,
            "answer": sol,
            "stratum": stratum,
            "clues": sum(ch != "." for ch in puz),
        })
    if len(buckets["S4_stalled"]) != n_stalled or len(buckets["WB-Prop"]) != n_prop:
        raise RuntimeError(f"Could not complete 9x9 roster after {attempts} attempts")
    rows = buckets["S4_stalled"] + buckets["WB-Prop"]
    return rows, {
        "seed": seed,
        "attempts": attempts,
        "counts": {k: len(v) for k, v in buckets.items()},
        "loopscope02_exclusion_sha256": excluded_sha,
        "loopscope04_rep_ids_checked": len(rep_ids),
        "arg03_n_accessed": False,
    }


def benchmark_compute_envelope(device: torch.device | str) -> dict:
    """Measure actual VRAM & RSS vs `ui/index.html` arithmetic lower bounds."""
    dev = torch.device(device)
    configs = [
        ("mini_14k", dict(hidden_size=32, num_heads=4, l_layers=1, h_cycles=1, l_cycles=1, puzzle_emb_len=1, seq_len=16), 64),
        ("dt04_242k", dict(hidden_size=96, num_heads=4, l_layers=2, h_cycles=1, l_cycles=2, puzzle_emb_len=2, seq_len=16), 64),
        ("full_7m", dict(hidden_size=512, num_heads=8, l_layers=2, h_cycles=3, l_cycles=6, puzzle_emb_len=16, seq_len=81), 729),
    ]
    rows = []
    for name, kw, out_coords in configs:
        seq_total = kw["puzzle_emb_len"] + kw["seq_len"]
        d = kw["hidden_size"]
        for dtype_name, dt, bpp in (("bf16", torch.bfloat16, 2), ("fp32", torch.float32, 4)):
            for batch in (1, 4, 16):
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(dev)
                m = PureSudokuTRM(**kw).to(device=dev, dtype=dt).eval()
                m.cos = m.cos.float()
                m.sin = m.sin.float()
                params = sum(p.numel() for p in m.parameters())
                x = torch.ones((batch, kw["seq_len"]), dtype=torch.long, device=dev)
                t0 = time.perf_counter()
                with torch.no_grad():
                    emb = m.encode_inputs(x)
                    zh, zl = m.init_states(batch)
                    for _ in range(8):
                        zh, zl = m.act_step(zh, zl, emb)
                    _ = m.digit_logits(zh)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                snap = memory_snapshot(dev)
                # UI arithmetic lower bounds (`ui/index.html` lines 208-210)
                ui_state_bytes = batch * 2 * seq_total * d * bpp
                ui_dense_jac_fp32_bytes = out_coords * 2 * seq_total * d * 4
                ui_param_bf16_bytes = params * 2
                rows.append({
                    "config": name,
                    "parameters": params,
                    "dtype": dtype_name,
                    "batch": batch,
                    "depth8_infer_ms": round(elapsed_ms, 2),
                    "ui_lower_bound_state_mib": round(ui_state_bytes / (1024.0 ** 2), 3),
                    "ui_lower_bound_dense_jac_b1_mib": round(ui_dense_jac_fp32_bytes / (1024.0 ** 2), 3),
                    "ui_lower_bound_param_bf16_mib": round(ui_param_bf16_bytes / (1024.0 ** 2), 3),
                    "measured_cuda_max_allocated_mib": snap.get("cuda_max_allocated_mib"),
                    "measured_cuda_max_reserved_mib": snap.get("cuda_max_reserved_mib"),
                    "measured_host_rss_mib": snap.get("host_rss_mib"),
                })
                del m, x, emb, zh, zl
                enforce_memory_ceiling(dev)
    return {"device": str(dev), "measurements": rows}


def run_hardware_and_7m_bridge_audit(device: torch.device | str = "cuda:0") -> dict:
    """Execute full hardware benchmark + SHA-verified 7M Sudoku depth-transport audit."""
    dev = torch.device(device)
    envelope = benchmark_compute_envelope(dev)
    boards, roster_meta = generate_screened_9x9_boards(n_stalled=8, n_prop=8)
    questions = [b["question"] for b in boards]
    answers = [b["answer"] for b in boards]
    inputs, gold = encode_sudoku_pairs(questions, answers, device=dev)

    checkpoints_out = {}
    for tag in ("yagizdevre", "gaoxin492"):
        ckpt_path, sha = resolve_offline_7m_checkpoint(tag)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = PureSudokuTRM.from_state_dict(sd, dtype=torch.float32).to(dev).eval()
        del sd
        bridge = NativeSudokuTRMBridge(model)
        # Overall + per-stratum horizon audit across steps (4, 8, 12, 16, 24, 32)
        steps = (4, 8, 12, 16, 24, 32)
        all_h = bridge.audit(inputs, gold, steps=steps)
        s4_h = bridge.audit(inputs[:8], gold[:8], steps=steps)
        wb_h = bridge.audit(inputs[8:], gold[8:], steps=steps)

        # Protected step-4 dark-state perturbation diagnostic on S4_stalled (8 boards):
        # compare clean vs left-SVD8 dark kick vs random head-null kick at matched L2 dose
        with torch.no_grad():
            s4_in = inputs[:8]
            s4_gold = gold[:8]
            emb = model.encode_inputs(s4_in)
            zh, zl = model.init_states(8)
            for _ in range(4):
                zh, zl = model.act_step(zh, zl, emb)
            # Full 11-vocab head nullspace on 512D hidden_size
            w_h = model.lm_head.weight.detach().double().cpu()
            _u, s_v, vh = torch.linalg.svd(w_h, full_matrices=True)
            r_h = int((s_v > 1e-8 * float(s_v.max().item())).sum().item())
            p_null = (vh[r_h:].T @ vh[r_h:]).float().to(dev)  # [512, 512]
            left_o = output_weight_basis(model.layers[-1].o_proj.weight.detach(), 8).to(dev)  # [512, 8]

            g7 = torch.Generator(device="cpu").manual_seed(DT04_9X9_SEED + (1 if tag == "yagizdevre" else 2))
            raw_noise = torch.randn(tuple(zh[:, 16:, :].shape), generator=g7, dtype=torch.float32).to(dev)
            svd_dir = ((raw_noise @ left_o) @ left_o.T) @ p_null
            rand_dir = raw_noise @ p_null
            target_dose = 0.25 * (zh[:, 16:, :] @ p_null).norm(dim=-1, keepdim=True).clamp_min(1e-6)
            svd_kick = svd_dir * (target_dose / svd_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            rand_kick = rand_dir * (target_dose / rand_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8))

            zh_svd = zh.clone(); zh_svd[:, 16:, :] += svd_kick
            zh_rnd = zh.clone(); zh_rnd[:, 16:, :] += rand_kick
            p_base = torch.softmax(model.lm_head(zh)[:, 16:, :], dim=-1)
            prot_err = max(
                float((torch.softmax(model.lm_head(zh_svd)[:, 16:, :], dim=-1) - p_base).abs().max().item()),
                float((torch.softmax(model.lm_head(zh_rnd)[:, 16:, :], dim=-1) - p_base).abs().max().item()),
            )
            if prot_err > 1e-5:
                raise AssertionError(f"7M head protection violated: {prot_err}")

            probe_horizons = {}
            for arm_tag, h0 in (("clean", zh), ("left_o_svd8_dark_kick", zh_svd), ("matched_random_dark_kick", zh_rnd)):
                hc, lc = h0.clone(), zl.clone()
                arm_steps = {}
                for s_step in range(5, 33):
                    hc, lc = model.act_step(hc, lc, emb)
                    if s_step in (8, 12, 16, 32):
                        pred = model.digit_logits(hc).argmax(-1)
                        arm_steps[str(s_step)] = {
                            "cell_acc": float((pred == s4_gold).float().mean().item()),
                            "board_exact": float((pred == s4_gold).all(-1).float().mean().item()),
                        }
                probe_horizons[arm_tag] = arm_steps

        snap = enforce_memory_ceiling(dev)
        checkpoints_out[tag] = {
            "sha256": sha,
            "horizon_all_16": all_h,
            "horizon_s4_stalled_8": s4_h,
            "horizon_wb_prop_8": wb_h,
            "s4_step4_protected_perturbation": {
                "protected_all_vocab_max_error": prot_err,
                "arms_by_outer_step": probe_horizons,
            },
            "post_run_memory": snap,
        }
        del model, bridge
        enforce_memory_ceiling(dev)

    return {
        "schema": "BPD-DT04/hardware-and-7m-bridge-audit/v1",
        "evidence_class": "empirical hardware benchmark and descriptive 16-board 7M depth-transport audit",
        "arg03_n_sealed": True,
        "offline_only": True,
        "compute_envelope": envelope,
        "roster_9x9_meta": roster_meta,
        "checkpoints_7m": checkpoints_out,
    }
