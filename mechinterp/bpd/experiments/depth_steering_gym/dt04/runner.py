"""BPD-DT04 prospective execution runner.

Executes only:
- Excluded engineering smoke root `2603999`, or
- The 59 registered development + sham-calibration roots `2604000..2604058`, or
- The offline hardware envelope & 7M bridge audit.

Preserves every root attempt, reserves `started_<root>.json` prior to model
initialization, saves trained Sudoku4 weights + SHA-256 digests, and enforces
host DRAM (`<= 12 GiB`) and CUDA VRAM (`<= 12.5 GiB`) ceilings between roots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch

from . import DEV_ROOTS, FIT_DEPTHS, HELDOUT_DEPTHS, SMOKE_ROOT, STUDY_ID
from .causal import evaluate_root_causal_steering
from .memory_guard import configure_device_budget, enforce_memory_ceiling
from .tasks import (
    build_root_sudoku_splits,
    encode_sudoku_cases,
    train_and_eval_arc_family,
    train_and_eval_storyworld,
    train_and_eval_sudoku4,
)
from .verify_source import verify_historical_locks, verify_source


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, val in sorted(model.state_dict().items()):
        arr = np.ascontiguousarray(val.detach().cpu().numpy())
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(json.dumps(list(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    data = text.encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()


def guard_root(root: int) -> None:
    if type(root) is not int or (root != SMOKE_ROOT and root not in DEV_ROOTS):
        raise ValueError(
            f"BPD-DT04 permits only excluded smoke {SMOKE_ROOT} and dev roots 2604000..2604058; got {root!r}"
        )


def execute_single_root(
    root: int,
    out_dir: Path,
    *,
    device: str,
    source_verification: dict,
) -> dict:
    guard_root(root)
    roots_dir = out_dir / "roots"
    weights_dir = out_dir / "weights"
    roots_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    receipt_path = roots_dir / f"root_{root}.json"
    started_path = roots_dir / f"started_{root}.json"
    if receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))

    pre_mem = enforce_memory_ceiling(device)
    _atomic_write_json(
        started_path,
        {
            "study": STUDY_ID,
            "root": root,
            "started_at_utc": _utc_now(),
            "source_lock_sha256": source_verification.get("execute_lock_sha256"),
            "pre_memory": pre_mem,
        },
    )

    t0 = time.perf_counter()
    try:
        # 1. Canonical-disjoint Sudoku4 competence + depth transport
        splits = build_root_sudoku_splits(root)
        sud_model, sud_report = train_and_eval_sudoku4(root, splits, device)
        state_hash = _model_state_hash(sud_model)

        ckpt_file = weights_dir / f"sudoku4_root_{root}.pt"
        buf = BytesIO()
        torch.save(sud_model.cpu().state_dict(), buf)
        ckpt_bytes = buf.getvalue()
        ckpt_sha = hashlib.sha256(ckpt_bytes).hexdigest()
        ckpt_file.write_bytes(ckpt_bytes)
        sud_model.to(device)
        sud_report["model_state_hash"] = state_hash
        sud_report["checkpoint_file"] = str(ckpt_file.relative_to(out_dir)).replace("\\", "/")
        sud_report["checkpoint_sha256"] = ckpt_sha

        # Competence gate check at supervised boundary depth 6 (>= 0.85 whole-board exact)
        sud_d6_exact = float(sud_report["eval_by_depth"]["6"]["whole_board_exact"])
        sud_competent = bool(sud_d6_exact >= 0.85)

        # 2. Complete-state causal steering + BPD (V,U) + 8 signed-Jacobian shams on Sudoku4
        tr_tok, _ = encode_sudoku_cases(splits["train_cases"][:512], device)
        cal_tok, _ = encode_sudoku_cases(splits["calib_cases"], device)
        ev_tok, ev_gold = encode_sudoku_cases(splits["eval_cases"], device)
        dn_tok, _ = encode_sudoku_cases(splits["donor_cases"], device)
        causal_report = evaluate_root_causal_steering(
            sud_model,
            tr_tok,
            cal_tok,
            ev_tok,
            ev_gold,
            dn_tok,
            seed=root,
        )
        # Verify frozen target model weights remained completely unchanged after causal/BPD probes
        if _model_state_hash(sud_model) != state_hash:
            raise AssertionError("Frozen Sudoku4 model weights mutated during causal evaluation")
        del sud_model, tr_tok, cal_tok, ev_tok, ev_gold, dn_tok
        enforce_memory_ceiling(device)

        # 3. Demo-conditioned Synthetic ARC (both strict 2-rule + 192->64 LUT induction)
        arc_report = train_and_eval_arc_family(root, device)
        enforce_memory_ceiling(device)

        # 4. Closed-loop horizon-12 Storyworld terminal ending evaluation
        story_report = train_and_eval_storyworld(root, device)
        post_mem = enforce_memory_ceiling(device)

        elapsed = round(time.perf_counter() - t0, 3)
        arc_id_d6 = float(arc_report["arc_lut_192_to_64_induction"]["eval"]["in_dist"]["6"]["exact_grid_accuracy"])
        story_d12 = float(story_report["eval_by_depth"]["12"]["host_ending_success_rate"])
        gauge_ok = bool(causal_report["bpd_physical_factor_audit"]["gauge_invariance"]["passed"])
        shams_ok = len(causal_report["sham_validation_ledger"]) == 8 and all(
            bool(s["distinct_contrast_available"]) for s in causal_report["sham_validation_ledger"]
        )

        record = {
            "schema": "BPD-DT04/root-receipt/v1",
            "study": STUDY_ID,
            "root": root,
            "is_excluded_smoke": root == SMOKE_ROOT,
            "status": "COMPLETED",
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "supervised_depths": list(FIT_DEPTHS),
            "heldout_depths": list(HELDOUT_DEPTHS),
            "source_verification": source_verification,
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": device,
            },
            "admission": {
                "sudoku4_d6_competent": sud_competent,
                "sudoku4_split_admissible": bool(sud_report["split_audit"]["admissible"]),
                "arc_indist_d6_competent": bool(arc_id_d6 >= 0.85),
                "storyworld_d12_competent": bool(story_d12 >= 0.85),
                "bpd_gauge_passed": gauge_ok,
                "eight_signed_shams_admitted": shams_ok,
                "protected_readout_passed": float(causal_report["protected_all_vocab_max_error"]) <= 1e-6,
                "all_gates_admitted": bool(
                    sud_competent
                    and sud_report["split_audit"]["admissible"]
                    and arc_id_d6 >= 0.85
                    and story_d12 >= 0.85
                    and gauge_ok
                    and shams_ok
                ),
            },
            "families": {
                "sudoku4": sud_report,
                "arc_style": arc_report,
                "storyworld": story_report,
            },
            "causal_steering": causal_report,
            "memory_telemetry": {
                "pre": pre_mem,
                "post": post_mem,
            },
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 3)
        record = {
            "schema": "BPD-DT04/root-receipt/v1",
            "study": STUDY_ID,
            "root": root,
            "is_excluded_smoke": root == SMOKE_ROOT,
            "status": "FAILED",
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "source_verification": source_verification,
        }

    _atomic_write_json(receipt_path, record)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "dev", "hardware-7m"), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--start-index", type=int, default=0, help="0-based start index within DEV_ROOTS")
    ap.add_argument("--count", type=int, default=len(DEV_ROOTS), help="Max number of DEV_ROOTS to run")
    ap.add_argument("--allow-prelock-smoke", action="store_true")
    args = ap.parse_args()

    dev_info = configure_device_budget(args.device, cuda_fraction=0.75, cpu_threads=4)
    if args.mode == "smoke" and args.allow_prelock_smoke:
        src_ver = {
            "source_check": "PRELOCK_SMOKE_ONLY",
            "execute_lock_sha256": None,
            **verify_historical_locks(),
        }
    else:
        src_ver = verify_source(require_runtime=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "hardware-7m":
        from .hardware_7m_audit import run_hardware_and_7m_bridge_audit

        report = run_hardware_and_7m_bridge_audit(args.device)
        report["source_verification"] = src_ver
        report["device_budget"] = dev_info
        out_path = args.out_dir / "DT04_HARDWARE_AND_7M_AUDIT.json"
        sha = _atomic_write_json(out_path, report)
        print(json.dumps({"mode": "hardware-7m", "out": str(out_path), "sha256": sha}, sort_keys=True), flush=True)
        return 0

    roster = (SMOKE_ROOT,) if args.mode == "smoke" else DEV_ROOTS[args.start_index : args.start_index + args.count]
    total = len(roster)
    print(
        json.dumps({
            "event": "DT04_RUN_START",
            "mode": args.mode,
            "roots_planned": total,
            "first_root": roster[0] if roster else None,
            "last_root": roster[-1] if roster else None,
            "device_budget": dev_info,
            "source_lock_sha256": src_ver.get("execute_lock_sha256"),
        }, sort_keys=True),
        flush=True,
    )

    failures = 0
    for idx, root in enumerate(roster, start=1):
        rec = execute_single_root(root, args.out_dir, device=args.device, source_verification=src_ver)
        if rec["status"] != "COMPLETED":
            failures += 1
            print(
                f"[{idx}/{total}] root={root} FAILED ({rec.get('error_type')}: {rec.get('error_message')})",
                flush=True,
            )
            continue
        sud_d8 = rec["families"]["sudoku4"]["eval_by_depth"]["8"]["whole_board_exact"]
        sud_d12 = rec["families"]["sudoku4"]["eval_by_depth"]["12"]["whole_board_exact"]
        arc_ho_d8 = rec["families"]["arc_style"]["arc_lut_192_to_64_induction"]["eval"]["heldout"]["8"]["cell_accuracy"]
        st_d12 = rec["families"]["storyworld"]["eval_by_depth"]["12"]["host_ending_success_rate"]
        g_vs_s_d8 = rec["causal_steering"]["root_contrasts_by_depth"]["8"]["gramian_minus_mean_sham_exact"]
        g_vs_s_d12 = rec["causal_steering"]["root_contrasts_by_depth"]["12"]["gramian_minus_mean_sham_exact"]
        rss = rec["memory_telemetry"]["post"]["host_rss_mib"]
        vram = rec["memory_telemetry"]["post"].get("cuda_max_reserved_mib", 0.0)
        print(
            f"[{idx:02d}/{total:02d}] root={root} ok ({rec['elapsed_seconds']:.1f}s) | "
            f"sud_d8={sud_d8:.3f} sud_d12={sud_d12:.3f} arc_ho_cell_d8={arc_ho_d8:.3f} "
            f"story_d12={st_d12:.3f} | gram-sham_d8={g_vs_s_d8:+.3f} d12={g_vs_s_d12:+.3f} | "
            f"rss={rss:.0f}MiB max_vram={vram:.0f}MiB",
            flush=True,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
