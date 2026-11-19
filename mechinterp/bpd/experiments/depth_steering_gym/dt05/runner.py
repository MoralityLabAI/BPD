"""DT05 resource calibration, excluded smoke, and prospective development runs.

Examples and the development evidence boundary are in README.md. This entry
point has no confirmation mode and never downloads pretrained weights.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from mechinterp.bpd.adapters.trm import build_mini_trm
from mechinterp.loopscope02.scripts.trm_sudoku import PureSudokuTRM, TRM_CHECKPOINTS
from ..dt04.tasks import build_root_sudoku_splits, encode_sudoku_cases
from ..dt04.verify_source import verify_source
from ..worlds import SudokuCase, count_solutions, sudoku4
from . import DEV_ROOTS, EVAL_DEPTHS, FIT_DEPTHS, SMOKE_ROOT, STUDY_ID
from .resources import BoundedTensorCache, MemoryBudget, assert_budget, memory_snapshot
from .training import (MODEL_CONFIGS, TrainConfig, fit_online_factors, model_digest,
                       routing_control, train_sudoku_model)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PARENT_COMMIT = "1082496445cb99e85cf346597aa87d4cb28f7059"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def runtime() -> dict:
    return {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cuda_available": torch.cuda.is_available()}


def source_snapshot() -> dict:
    historical = verify_source(require_runtime=False)
    files = (list(HERE.glob("*.py")) + list((HERE / "evidence").glob("*.json"))
             + [REPO / "AGENTS.md", HERE / "README.md", HERE / "PROTOCOL_DT05.md", HERE / "REVIEW_DT04.md"])
    lock = json.loads((HERE.parent / "dt04/EXECUTE_LOCK.json").read_text())
    files += [REPO / rel for rel in lock["files_sha256"]]
    digests = {str(p.relative_to(REPO)).replace("\\", "/"): sha256_file(p) for p in sorted(set(files))}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=REPO, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return {"historical": historical, "files_sha256": digests,
            "digest": hashlib.sha256(canonical_bytes(digests)).hexdigest(),
            "git_head": head, "git_dirty": dirty, "dt04_parent_commit": PARENT_COMMIT}


def guard_development_root(root: int) -> None:
    if type(root) is not int or root not in DEV_ROOTS:
        raise ValueError("DT05 development accepts only registered roots 2605000..2605058")


def configure_runtime(device: str, threads: int, budget: MemoryBudget) -> None:
    if type(threads) is not int or threads < 1:
        raise ValueError("CPU threads must be positive")
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError("Only tested CPU and CUDA execution paths are enabled")
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, budget.max_cuda_reserved_gib * 1024**3 / total), dev)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert_budget(memory_snapshot(device), budget)


def make_plan(args) -> dict:
    config = TrainConfig(steps=args.steps, logical_batch=args.logical_batch, microbatch=args.microbatch,
                         learning_rate=args.learning_rate, ema_decay=args.ema_decay, precision=args.precision)
    if args.profile not in ("mini", "dt04"):
        raise ValueError("Development trains Sudoku4; full_7m is available for calibration and local replay")
    if not 1 <= args.top_k <= args.components <= MODEL_CONFIGS[args.profile]["hidden_size"]:
        raise ValueError("Require 1 <= top_k <= components <= hidden_size")
    if type(args.factor_steps) is not int or args.factor_steps < 0:
        raise ValueError("factor steps must be nonnegative")
    if not np.isfinite(args.cache_gib) or args.cache_gib < 0:
        raise ValueError("cache GiB must be finite and nonnegative")
    return {
        "schema": "BPD-DT05/development-plan/v1", "study": STUDY_ID,
        "scope": "development_exposed_catalog", "confirmation_permitted": False,
        "development_roots": list(DEV_ROOTS), "excluded_smoke_root": SMOKE_ROOT,
        "source": source_snapshot(), "profile": args.profile, "train": asdict(config),
        "factor_steps": args.factor_steps, "components": args.components, "top_k": args.top_k,
        "cache_gib": args.cache_gib, "memory_budget": asdict(budget_from_args(args)),
        "fit_depths": list(FIT_DEPTHS), "eval_depths": list(EVAL_DEPTHS),
        "arms": ["raw", "ema", "online_factors", "fixed_factor_control", "random_factor_control"],
        "primary_development_contrast": "online_factors minus fixed_factor_control exact board at depth 8",
        "independent_unit": "trained model root; boards and depths are repeated measurements",
        "selection_exposure": "existing Sudoku4 generator/catalog and DT04 report informed this redesign",
        "new_causal_benefit": "unmeasured; matched dose/sham calibration still required",
        "training_selection": "fixed final update; raw and EMA both scored; EMA teacher fixed before runs",
    }


def read_plan(path: Path) -> dict:
    if path.stat().st_size > 2_000_000:
        raise ValueError("plan exceeds size limit")
    plan = json.loads(path.read_text())
    if (plan.get("schema") != "BPD-DT05/development-plan/v1"
            or plan.get("scope") != "development_exposed_catalog"
            or plan.get("confirmation_permitted") is not False
            or plan.get("development_roots") != list(DEV_ROOTS)):
        raise ValueError("Invalid DT05 development plan or cohort")
    if plan.get("source", {}).get("digest") != source_snapshot()["digest"]:
        raise ValueError("source changed after the plan was frozen; create a new development plan")
    TrainConfig(**plan["train"])
    MemoryBudget(**plan["memory_budget"])
    if plan.get("profile") not in ("mini", "dt04"):
        raise ValueError("invalid development model profile")
    if plan.get("fit_depths") != list(FIT_DEPTHS) or plan.get("eval_depths") != list(EVAL_DEPTHS):
        raise ValueError("fit/evaluation depths differ from the implemented plan")
    if any(type(plan.get(k)) is not int for k in ("factor_steps", "components", "top_k")):
        raise ValueError("factor settings must be integers")
    if plan["factor_steps"] < 0 or not 1 <= plan["top_k"] <= plan["components"] <= MODEL_CONFIGS[plan["profile"]]["hidden_size"]:
        raise ValueError("invalid factor settings")
    if not isinstance(plan.get("cache_gib"), (int, float)) or not np.isfinite(plan["cache_gib"]) or plan["cache_gib"] < 0:
        raise ValueError("invalid cache budget")
    return plan


def save_checkpoint(out_dir: Path, name: str, model, metadata: dict) -> dict:
    path = out_dir / f"{name}.pt"
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)
    receipt = {"file": path.name, "sha256": sha256_file(path), "state_sha256": model_digest(model), **metadata}
    write_json(out_dir / f"{name}.json", receipt)
    return receipt


def run_development(plan: dict, root: int, out_dir: Path, device: str, threads: int) -> dict:
    from .evaluation import evaluate_policy, transport_diagnostic

    guard_development_root(root)
    budget = MemoryBudget(**plan["memory_budget"])
    configure_runtime(device, threads, budget)
    # Global attempt lease is relative to this frozen plan's campaign directory.
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    plan_sha = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    write_json(out_dir / "STARTED.json", {"root": root, "plan_sha256": plan_sha,
               "started_utc": datetime.now(timezone.utc).isoformat(), "status": "STARTED"})
    write_json(out_dir / "PLAN.json", plan)
    record = {"schema": "BPD-DT05/development-root/v1", "root": root,
              "plan_sha256": plan_sha, "scope": "development_exposed_catalog", "runtime": runtime(),
              "source": source_snapshot(), "device": device, "confirmation_run": False}
    try:
        splits = build_root_sudoku_splits(root)
        split_manifest = {key: [asdict(case) for case in splits[key]] for key in
                          ("train_cases", "calib_cases", "eval_cases", "donor_cases")}
        write_json(out_dir / "SPLITS.json", split_manifest)
        record["splits"] = {"sha256": sha256_file(out_dir / "SPLITS.json"),
                            "counts": splits["canonical_class_counts"],
                            "audit": splits["audit_train_vs_eval"]}
        config = TrainConfig(**plan["train"])
        raw, ema, training = train_sudoku_model(root, splits["train_cases"], config=config,
                                               device=device, budget=budget, profile=plan["profile"])
        record["training"] = training
        record["checkpoints"] = {
            "raw": save_checkpoint(out_dir, "raw", raw, {"profile": plan["profile"], "format": "pure"}),
            "ema": save_checkpoint(out_dir, "ema", ema, {"profile": plan["profile"], "format": "pure"}),
        }
        tx, _ = encode_sudoku_cases(splits["train_cases"][:512], "cpu")
        ex, ey = encode_sudoku_cases(splits["eval_cases"], "cpu")
        cache = BoundedTensorCache(int(plan["cache_gib"] * 1024**3))
        student, fit = fit_online_factors(ema, tx, steps=plan["factor_steps"], components=plan["components"],
                  top_k=plan["top_k"], logical_batch=config.logical_batch, microbatch=config.microbatch,
                  seed=root, budget=budget, cache=cache)
        record["factor_fit"] = fit
        record["checkpoints"]["online_factors"] = save_checkpoint(out_dir, "online_factors", student,
                   {"profile": plan["profile"], "format": "online_factors", "components": plan["components"],
                    "top_k": plan["top_k"], "routing": "online"})
        record["arms"] = {}
        for name, model in (("raw", raw), ("ema", ema), ("online_factors", student)):
            record["arms"][name] = evaluate_policy(model, ex, ey, microbatch=config.microbatch, budget=budget)
        for name, mode in (("fixed_factor_control", "fixed"), ("random_factor_control", "random")):
            control = routing_control(student, mode, root)
            record["arms"][name] = evaluate_policy(control, ex, ey, microbatch=config.microbatch, budget=budget)
            del control
        record["transport_instrument"] = transport_diagnostic(ema, tx[:1], rank=min(8, plan["components"]))
        record["memory_final"] = memory_snapshot(device)
        assert_budget(record["memory_final"], budget)
        record["status"] = "COMPLETED_DEVELOPMENT_ONLY"
    except Exception as exc:
        record.update(status="FAILED", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    record["elapsed_seconds"] = time.perf_counter() - started
    write_json(out_dir / "RECEIPT.json", record)
    return record


def smoke(out_dir: Path, device: str, budget: MemoryBudget, threads: int) -> dict:
    from .evaluation import evaluate_policy, transport_diagnostic

    configure_runtime(device, threads, budget)
    source = source_snapshot()
    out_dir.mkdir(parents=True, exist_ok=False)
    cases = [sudoku4(860_000 + i, "engineering", blanks=4 + i % 3) for i in range(8)]
    config = TrainConfig(steps=2, logical_batch=8, microbatch=4, ema_decay=0.9)
    raw, ema, train = train_sudoku_model(SMOKE_ROOT, cases, config=config, device=device, budget=budget, profile="mini")
    x, y = encode_sudoku_cases(cases, "cpu")
    cache = BoundedTensorCache(8 * 1024**2)
    student, fit = fit_online_factors(ema, x, steps=2, components=8, top_k=4, logical_batch=8,
                     microbatch=4, seed=SMOKE_ROOT, budget=budget, cache=cache)
    result = {"schema": "BPD-DT05/engineering-smoke/v1", "scope": "engineering_only",
              "root": SMOKE_ROOT, "scientific_root_count": 0, "status": "PASS", "source": source,
              "runtime": runtime(), "training": train, "factor_fit": fit, "arms": {}}
    for name, model in (("raw", raw), ("ema", ema), ("online", student),
                        ("fixed", routing_control(student, "fixed", SMOKE_ROOT)),
                        ("random", routing_control(student, "random", SMOKE_ROOT))):
        result["arms"][name] = evaluate_policy(model, x, y, microbatch=4, budget=budget)
    result["transport_instrument"] = transport_diagnostic(ema, x[:1], rank=4)
    result["checkpoint"] = save_checkpoint(out_dir, "engineering_ema", ema, {"profile": "mini", "format": "pure"})
    write_json(out_dir / "CASES.json", {"schema": "BPD-DT05/sudoku-cases/v1", "scope": "engineering",
                "cases": [asdict(case) for case in cases]})
    result["memory_final"] = memory_snapshot(device)
    write_json(out_dir / "SMOKE.json", result)
    return result


def load_replay_cases(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("replay cases exceed 8 MiB limit")
    data = json.loads(path.read_text())
    if data.get("schema") != "BPD-DT05/sudoku-cases/v1" or data.get("scope") not in ("development", "engineering"):
        raise ValueError("replay needs an explicit development/engineering case manifest")
    rows = data.get("cases")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
        raise ValueError("replay requires 1..4096 cases")
    tokens, targets, ids = [], [], []
    size = None
    for row in rows:
        givens, solution = row["givens"], row["solution"]
        if not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("case IDs must be unique strings")
        if size is None:
            size = len(givens)
        if len(givens) != size or size not in (4, 9):
            raise ValueError("all cases must share a supported Sudoku size")
        if any(type(v) is not int for grid in (givens, solution) for r in grid for v in r):
            raise ValueError("Sudoku digits must be integers")
        count, unique = count_solutions(tuple(tuple(r) for r in givens), limit=2)
        if count != 1 or unique != tuple(tuple(r) for r in solution):
            raise ValueError("each replay puzzle needs a unique, solver-verified solution")
        ids.append(row["id"])
        tokens.append([v + 1 for r in givens for v in r])
        targets.append([v - 1 for r in solution for v in r])
    return torch.tensor(tokens), torch.tensor(targets), {
        "sha256": sha256_file(path), "scope": data["scope"], "case_ids": ids, "unique_solutions_verified": True}


def replay(args) -> dict:
    from .evaluation import evaluate_policy
    from .techniques import replace_online_factors

    if args.out.exists():
        raise FileExistsError("replay output already exists")
    budget = budget_from_args(args)
    configure_runtime(args.device, args.cpu_threads, budget)
    source = source_snapshot()
    if not args.checkpoint.is_file() or sha256_file(args.checkpoint) != args.sha256.lower():
        raise ValueError("local checkpoint SHA-256 does not match")
    x, y, cases = load_replay_cases(args.cases)
    if args.checkpoint_format == "official":
        if args.sha256.lower() not in {c["sha256"] for c in TRM_CHECKPOINTS.values()}:
            raise ValueError("official checkpoint must match a published native bridge SHA lock")
        if args.profile != "full_7m":
            raise ValueError("official checkpoints require full_7m profile")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.checkpoint_format == "official":
        model = PureSudokuTRM.from_state_dict(state, dtype=torch.float32)
    else:
        model = build_mini_trm(**MODEL_CONFIGS[args.profile], seed=0)
        if args.checkpoint_format == "online_factors":
            names = [f"layers.{i}.{s}" for i in range(len(model.layers)) for s in ("o_proj", "down_proj")]
            replace_online_factors(model, names, args.components, args.top_k)
        model.load_state_dict(state, strict=True)
    if x.shape[1] != model.seq_len:
        raise ValueError("case shape differs from checkpoint profile")
    model.to(args.device).eval()
    out = {"schema": "BPD-DT05/local-replay/v1", "scope": cases["scope"], "source": source,
           "checkpoint_sha256": args.sha256.lower(), "checkpoint_format": args.checkpoint_format,
           "runtime": runtime(), "cases": cases,
           "evaluation": evaluate_policy(model, x, y, depths=tuple(args.depths), microbatch=args.microbatch, budget=budget)}
    write_json(args.out, out)
    return out


def budget_from_args(args) -> MemoryBudget:
    return MemoryBudget(args.max_host_rss_gib, args.min_host_available_gib,
                        args.max_cuda_reserved_gib, args.min_cuda_free_gib)


def add_budget(parser):
    parser.add_argument("--max-host-rss-gib", type=float, default=12.0)
    parser.add_argument("--min-host-available-gib", type=float, default=4.0)
    parser.add_argument("--max-cuda-reserved-gib", type=float, default=12.5)
    parser.add_argument("--min-cuda-free-gib", type=float, default=1.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="freeze a development recipe and exact source hashes")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--profile", choices=("mini", "dt04"), default="dt04")
    p.add_argument("--steps", type=int, default=480)
    p.add_argument("--factor-steps", type=int, default=80)
    p.add_argument("--logical-batch", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--components", type=int, default=24)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--cache-gib", type=float, default=2.0)
    add_budget(p)
    p = sub.add_parser("run", help="attempt one registered development root from a frozen plan")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--root", type=int, required=True)
    p.add_argument("--out", type=Path, required=True, help="campaign directory; each root gets a unique attempt")
    for command in ("smoke", "calibrate", "replay"):
        q = sub.add_parser(command)
        q.add_argument("--out", type=Path, required=True)
        add_budget(q)
    for name in ("run", "smoke", "calibrate", "replay"):
        q = sub.choices[name]
        q.add_argument("--device", default="cpu")
        q.add_argument("--cpu-threads", type=int, default=4)
    q = sub.choices["calibrate"]
    q.add_argument("--profile", choices=tuple(MODEL_CONFIGS), default="dt04")
    q.add_argument("--stage", choices=("inference", "training"), default="training")
    q.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    q.add_argument("--logical-batch", type=int, default=128)
    q.add_argument("--candidates", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    q.add_argument("--warmups", type=int, default=1)
    q.add_argument("--repeats", type=int, default=3)
    q = sub.choices["replay"]
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--sha256", required=True)
    q.add_argument("--checkpoint-format", choices=("pure", "official", "online_factors"), default="pure")
    q.add_argument("--profile", choices=tuple(MODEL_CONFIGS), default="dt04")
    q.add_argument("--components", type=int, default=24)
    q.add_argument("--top-k", type=int, default=12)
    q.add_argument("--cases", type=Path, required=True)
    q.add_argument("--microbatch", type=int, default=16)
    q.add_argument("--depths", type=int, nargs="+", default=list(EVAL_DEPTHS))
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.out.exists():
            raise FileExistsError("plan exists; use a new path for another frozen recipe")
        result = make_plan(args)
        write_json(args.out, result)
    elif args.command == "run":
        guard_development_root(args.root)
        plan = read_plan(args.plan)
        # Validate device and available memory before reserving a model attempt.
        configure_runtime(args.device, args.cpu_threads, MemoryBudget(**plan["memory_budget"]))
        # One marker per root per campaign, including failures and interrupted runs.
        args.out.mkdir(parents=True, exist_ok=True)
        lease = args.out / f"attempted_{args.root}.json"
        with lease.open("x") as stream:
            json.dump({"root": args.root, "plan_sha256": sha256_file(args.plan),
                       "status": "ATTEMPT_RESERVED", "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
        result = run_development(plan, args.root, args.out / f"root_{args.root}", args.device, args.cpu_threads)
    elif args.command == "smoke":
        result = smoke(args.out, args.device, budget_from_args(args), args.cpu_threads)
    elif args.command == "calibrate":
        from .calibration import run_calibration
        if args.out.exists():
            raise FileExistsError("calibration output already exists")
        budget = budget_from_args(args)
        configure_runtime(args.device, args.cpu_threads, budget)
        result = run_calibration(profile=args.profile, stage=args.stage, logical_batch=args.logical_batch,
                    candidates=args.candidates, device=args.device, budget=budget, precision=args.precision,
                    warmups=args.warmups, repeats=args.repeats)
        result.update(source=source_snapshot(), runtime=runtime())
        write_json(args.out, result)
    else:
        result = replay(args)
    print(json.dumps({"command": args.command, "output": str(args.out), "status": result.get("status", "WRITTEN")}, sort_keys=True))
    return 1 if result.get("status") in ("FAILED", "NO_ADMISSIBLE_MICROBATCH") else 0


if __name__ == "__main__":
    raise SystemExit(main())
