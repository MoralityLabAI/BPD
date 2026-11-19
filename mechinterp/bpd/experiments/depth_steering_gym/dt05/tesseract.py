"""Provisional Tesseract Rust desktop runner manifest and cancellation adapter (§6).

Because the upstream Tesseract Rust desktop runner repository is not present in
this checkout or public GitHub index, this module defines an explicitly versioned
provisional manifest contract (``Tesseract/gym-manifest-provisional/v1``), strict
validator, and graceful cancellation context manager documented in
``TESSERACT_INTEGRATION.md``.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import signal
from typing import Any

from . import DEV_ROOTS, EVAL_DEPTHS, FIT_DEPTHS, SMOKE_ROOT, STUDY_ID
from .resources import MemoryBudget
from .training import MODEL_CONFIGS

TESSERACT_SCHEMA = "Tesseract/gym-manifest-provisional/v1"
VALID_SCOPES = ("engineering", "development", "protected")


class RunCancelled(RuntimeError):
    """Raised when the desktop runner delivers SIGINT/SIGTERM cancellation."""


@contextmanager
def graceful_cancellation_guard(on_cancel=None):
    """Translate SIGINT/SIGTERM into ``RunCancelled`` so receipts flush cleanly."""
    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)

    def _handler(signum, _frame):
        sig_name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)
        if on_cancel is not None:
            on_cancel(sig_name)
        raise RunCancelled(f"Cancelled by signal {sig_name}")

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        yield
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


def build_tesseract_manifest(
    *,
    git_revision: str,
    source_digest: str,
    repository_url: str = "https://github.com/MoralityLabAI/BPD",
    branch: str = "gym/TRM-Sodoku",
    device: str = "cuda:0",
    cpu_threads: int = 8,
    budget: MemoryBudget = MemoryBudget(),
    dt04_train_microbatch: int = 128,
    full_7m_infer_microbatch: int = 64,
    full_7m_train_microbatch: int = 32,
) -> dict[str, Any]:
    """Construct a complete, machine-validated Tesseract JSON gym manifest."""
    if not isinstance(git_revision, str) or len(git_revision) != 40:
        raise ValueError("git_revision must be a 40-hex-char commit SHA")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise ValueError("source_digest must be a 64-hex-char SHA-256")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if 128 % dt04_train_microbatch or 256 % full_7m_infer_microbatch or 32 % full_7m_train_microbatch:
        raise ValueError("calibrated microbatches must divide their workload logical batches")

    module = "mechinterp.bpd.experiments.depth_steering_gym.dt05.runner"
    common_budget_args = [
        "--max-host-rss-gib", str(budget.max_host_rss_gib),
        "--min-host-available-gib", str(budget.min_host_available_gib),
        "--max-cuda-reserved-gib", str(budget.max_cuda_reserved_gib),
        "--min-cuda-free-gib", str(budget.min_cuda_free_gib),
    ]

    manifest: dict[str, Any] = {
        "schema": TESSERACT_SCHEMA,
        "schema_status": "provisional_adapter",
        "schema_mapping_doc": "mechinterp/bpd/experiments/depth_steering_gym/dt05/TESSERACT_INTEGRATION.md",
        "gym_id": "bpd-dt05-trm-sudoku",
        "study_id": STUDY_ID,
        "repository": {
            "url": repository_url,
            "branch": branch,
            "git_revision": git_revision,
            "source_lock_sha256": source_digest,
            "working_directory": ".",
            "gym_directory": "mechinterp/bpd/experiments/depth_steering_gym/dt05",
            "line_endings": "lf",
        },
        "environment": {
            "python_min_version": "3.11",
            "required_env": {
                "HF_HUB_OFFLINE": "1",
                "PYTHONUNBUFFERED": "1",
            },
            "network_egress_permitted": False,
            "weight_autodownload_permitted": False,
        },
        "target_hardware_envelope": {
            "reference_gpu": "NVIDIA GeForce RTX 5080 (16 GB VRAM)",
            "device": device,
            "cpu_threads_request": cpu_threads,
            "system_ram_request_gib": 8.0,
            "system_ram_ceiling_gib": budget.max_host_rss_gib,
            "system_ram_min_available_reserve_gib": budget.min_host_available_gib,
            "vram_request_gib": 10.0,
            "vram_ceiling_gib": budget.max_cuda_reserved_gib,
            "vram_min_free_reserve_gib": budget.min_cuda_free_gib,
        },
        "cohorts": {
            "excluded_engineering_smoke_root": SMOKE_ROOT,
            "reserved_development_roots": {
                "start": DEV_ROOTS[0],
                "end": DEV_ROOTS[-1],
                "count": len(DEV_ROOTS),
                "auto_launch_all_prohibited": True,
            },
            "sealed_protected_cohorts": [
                "BPD-DT04 confirmation roots 2604100..2604355",
                "ARG-03-N 168-case test.jsonl",
                "NC09 / AMM-03..06 confirmation populations",
            ],
        },
        "workloads": [
            {
                "id": "engineering_smoke_cuda",
                "evidence_scope": "engineering",
                "permitted": True,
                "description": "Excluded root 2605999 end-to-end mini-TRM smoke on GPU.",
                "profile": "mini",
                "model_config": MODEL_CONFIGS["mini"],
                "device": device,
                "precision": "fp32",
                "logical_batch": 8,
                "microbatch": 4,
                "accumulation_steps": 2,
                "command": ["python", "-m", module, "smoke"],
                "args": [
                    "--device", device,
                    "--cpu-threads", str(cpu_threads),
                    "--execution-mode", "optimized",
                    "--out", "runs/dt05/tesseract/smoke-${RUN_ID}",
                    *common_budget_args,
                ],
                "resources": {
                    "cpu_threads": cpu_threads,
                    "system_ram_gib": 4.0,
                    "vram_gib": 2.0,
                },
                "timeout": {
                    "wall_seconds": 300,
                    "graceful_cancel_signal": "SIGINT",
                    "kill_escalation_seconds": 15,
                },
                "artifacts": {
                    "output_dir": "runs/dt05/tesseract/smoke-${RUN_ID}",
                    "primary_receipt": "runs/dt05/tesseract/smoke-${RUN_ID}/SMOKE.json",
                    "stdout_protocol": "jsonl_events_and_final_summary",
                },
                "retry_policy": {
                    "auto_retry_permitted": True,
                    "max_attempts": 2,
                    "requires_fresh_out_path": True,
                    "resume_in_place_permitted": False,
                },
            },
            {
                "id": "calibrate_dt04_training_bf16",
                "evidence_scope": "engineering",
                "permitted": True,
                "description": "Isolated microbatch forward/backward calibration for dt04 profile.",
                "profile": "dt04",
                "model_config": MODEL_CONFIGS["dt04"],
                "device": device,
                "precision": "bf16",
                "logical_batch": 128,
                "microbatch": dt04_train_microbatch,
                "accumulation_steps": 128 // dt04_train_microbatch,
                "command": ["python", "-m", module, "calibrate"],
                "args": [
                    "--profile", "dt04",
                    "--stage", "training",
                    "--device", device,
                    "--precision", "bf16",
                    "--cuda-graph",
                    "--logical-batch", "128",
                    "--candidates", "8", "16", "32", "64", "128",
                    "--cpu-threads", str(cpu_threads),
                    "--out", "runs/dt05/tesseract/cal-dt04-train-${RUN_ID}.json",
                    *common_budget_args,
                ],
                "resources": {
                    "cpu_threads": cpu_threads,
                    "system_ram_gib": 4.0,
                    "vram_gib": 2.5,
                },
                "timeout": {
                    "wall_seconds": 600,
                    "graceful_cancel_signal": "SIGINT",
                    "kill_escalation_seconds": 15,
                },
                "artifacts": {
                    "output_dir": "runs/dt05/tesseract",
                    "primary_receipt": "runs/dt05/tesseract/cal-dt04-train-${RUN_ID}.json",
                    "stdout_protocol": "json_summary",
                },
                "retry_policy": {
                    "auto_retry_permitted": True,
                    "max_attempts": 2,
                    "requires_fresh_out_path": True,
                    "resume_in_place_permitted": False,
                },
            },
            {
                "id": "benchmark_full_step_7m_checkpointed",
                "evidence_scope": "engineering",
                "permitted": True,
                "description": "Full optimizer-step + CPU prep + pinned H2D + EMA benchmark for full_7m.",
                "profile": "full_7m",
                "model_config": MODEL_CONFIGS["full_7m"],
                "device": device,
                "precision": "bf16",
                "logical_batch": 32,
                "microbatch": full_7m_train_microbatch,
                "accumulation_steps": 32 // full_7m_train_microbatch,
                "command": ["python", "-m", module, "calibrate-step"],
                "args": [
                    "--profile", "full_7m",
                    "--device", device,
                    "--precision", "bf16",
                    "--execution-mode", "optimized",
                    "--checkpoint-act",
                    "--logical-batch", "32",
                    "--candidates", "1", "2", "4", "8", "16", "32",
                    "--cpu-threads", str(cpu_threads),
                    "--out", "runs/dt05/tesseract/step-7m-train-${RUN_ID}.json",
                    *common_budget_args,
                ],
                "resources": {
                    "cpu_threads": cpu_threads,
                    "system_ram_gib": 6.0,
                    "vram_gib": 10.0,
                },
                "timeout": {
                    "wall_seconds": 1200,
                    "graceful_cancel_signal": "SIGINT",
                    "kill_escalation_seconds": 20,
                },
                "artifacts": {
                    "output_dir": "runs/dt05/tesseract",
                    "primary_receipt": "runs/dt05/tesseract/step-7m-train-${RUN_ID}.json",
                    "stdout_protocol": "json_summary",
                },
                "retry_policy": {
                    "auto_retry_permitted": True,
                    "max_attempts": 2,
                    "requires_fresh_out_path": True,
                    "resume_in_place_permitted": False,
                },
            },
            {
                "id": "calibrate_7m_inference_bf16",
                "evidence_scope": "engineering",
                "permitted": True,
                "description": "Isolated 8-ACT-step inference microbatch calibration for full_7m.",
                "profile": "full_7m",
                "model_config": MODEL_CONFIGS["full_7m"],
                "device": device,
                "precision": "bf16",
                "logical_batch": 256,
                "microbatch": full_7m_infer_microbatch,
                "accumulation_steps": 256 // full_7m_infer_microbatch,
                "command": ["python", "-m", module, "calibrate"],
                "args": [
                    "--profile", "full_7m",
                    "--stage", "inference",
                    "--device", device,
                    "--precision", "bf16",
                    "--cuda-graph",
                    "--logical-batch", "256",
                    "--candidates", "16", "32", "64", "128", "256",
                    "--cpu-threads", str(cpu_threads),
                    "--out", "runs/dt05/tesseract/cal-7m-infer-${RUN_ID}.json",
                    *common_budget_args,
                ],
                "resources": {
                    "cpu_threads": cpu_threads,
                    "system_ram_gib": 4.0,
                    "vram_gib": 6.0,
                },
                "timeout": {
                    "wall_seconds": 600,
                    "graceful_cancel_signal": "SIGINT",
                    "kill_escalation_seconds": 15,
                },
                "artifacts": {
                    "output_dir": "runs/dt05/tesseract",
                    "primary_receipt": "runs/dt05/tesseract/cal-7m-infer-${RUN_ID}.json",
                    "stdout_protocol": "json_summary",
                },
                "retry_policy": {
                    "auto_retry_permitted": True,
                    "max_attempts": 2,
                    "requires_fresh_out_path": True,
                    "resume_in_place_permitted": False,
                },
            },
            {
                "id": "development_single_root_dt04",
                "evidence_scope": "development",
                "permitted": True,
                "requires_explicit_user_root_selection": True,
                "requires_frozen_plan_and_evalmath_preflight": True,
                "description": "Execute ONE registered DT05 development root (2605000..2605058) from a frozen plan.",
                "profile": "dt04",
                "model_config": MODEL_CONFIGS["dt04"],
                "device": device,
                "precision": "bf16",
                "fit_depths": list(FIT_DEPTHS),
                "eval_depths": list(EVAL_DEPTHS),
                "logical_batch": 128,
                "microbatch": dt04_train_microbatch,
                "accumulation_steps": 128 // dt04_train_microbatch,
                "command": ["python", "-m", module, "run"],
                "args": [
                    "--plan", "${FROZEN_PLAN_JSON}",
                    "--root", "${DEV_ROOT}",
                    "--device", device,
                    "--cpu-threads", str(cpu_threads),
                    "--execution-mode", "optimized",
                    "--out", "${CAMPAIGN_DIR}",
                ],
                "resources": {
                    "cpu_threads": cpu_threads,
                    "system_ram_gib": 8.0,
                    "vram_gib": 4.0,
                },
                "timeout": {
                    "wall_seconds": 1800,
                    "graceful_cancel_signal": "SIGINT",
                    "kill_escalation_seconds": 30,
                    "on_timeout_or_cancel": "writes RECEIPT.json with status CANCELLED or FAILED; root lease remains consumed",
                },
                "artifacts": {
                    "output_dir": "${CAMPAIGN_DIR}/root_${DEV_ROOT}",
                    "attempt_lease": "${CAMPAIGN_DIR}/attempted_${DEV_ROOT}.json",
                    "primary_receipt": "${CAMPAIGN_DIR}/root_${DEV_ROOT}/RECEIPT.json",
                    "checkpoints": [
                        "${CAMPAIGN_DIR}/root_${DEV_ROOT}/raw.pt",
                        "${CAMPAIGN_DIR}/root_${DEV_ROOT}/ema.pt",
                        "${CAMPAIGN_DIR}/root_${DEV_ROOT}/online_factors.pt",
                    ],
                    "stdout_protocol": "jsonl_events_and_final_summary",
                },
                "retry_policy": {
                    "auto_retry_permitted": False,
                    "max_attempts": 1,
                    "requires_fresh_out_path": False,
                    "resume_in_place_permitted": False,
                    "restriction_reason": "Each development root 2605000..2605058 reserves an immutable attempt lease; failed/cancelled roots must remain in the campaign ledger.",
                },
            },
            {
                "id": "protected_confirmation_seal",
                "evidence_scope": "protected",
                "permitted": False,
                "description": "Placeholder entry documenting sealed confirmation cohorts; runner must refuse launch.",
                "profile": "dt04",
                "device": device,
                "precision": "bf16",
                "logical_batch": 128,
                "microbatch": dt04_train_microbatch,
                "accumulation_steps": 128 // dt04_train_microbatch,
                "command": [],
                "args": [],
                "resources": {"cpu_threads": 0, "system_ram_gib": 0.0, "vram_gib": 0.0},
                "timeout": {"wall_seconds": 0, "graceful_cancel_signal": "SIGINT", "kill_escalation_seconds": 0},
                "artifacts": {"output_dir": "", "primary_receipt": "", "stdout_protocol": "none"},
                "retry_policy": {
                    "auto_retry_permitted": False,
                    "max_attempts": 0,
                    "requires_fresh_out_path": True,
                    "resume_in_place_permitted": False,
                    "restriction_reason": "EvalMath verdict is REDESIGN_OR_PILOT_ONLY; confirmation cohorts remain strictly sealed.",
                },
            },
        ],
    }
    validate_tesseract_manifest(manifest)
    return manifest


def validate_tesseract_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed on malformed or scientifically unsafe Tesseract gym manifests."""
    if manifest.get("schema") != TESSERACT_SCHEMA:
        raise ValueError(f"Unsupported Tesseract manifest schema: {manifest.get('schema')}")
    repo = manifest.get("repository", {})
    if not isinstance(repo.get("git_revision"), str) or len(repo["git_revision"]) != 40:
        raise ValueError("Manifest requires exact 40-char git_revision")
    if not isinstance(repo.get("source_lock_sha256"), str) or len(repo["source_lock_sha256"]) != 64:
        raise ValueError("Manifest requires 64-char source_lock_sha256")
    env = manifest.get("environment", {})
    if env.get("network_egress_permitted") is not False or env.get("weight_autodownload_permitted") is not False:
        raise ValueError("DT05 manifest must forbid network egress and weight autodownload")
    if env.get("required_env", {}).get("HF_HUB_OFFLINE") != "1":
        raise ValueError("DT05 manifest must enforce HF_HUB_OFFLINE=1")
    cohorts = manifest.get("cohorts", {}).get("reserved_development_roots", {})
    if cohorts.get("auto_launch_all_prohibited") is not True:
        raise ValueError("DT05 manifest must prohibit automatic full-cohort launch")

    workloads = manifest.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("Manifest requires nonempty workloads list")
    seen_ids = set()
    for wl in workloads:
        wid = wl.get("id")
        if not isinstance(wid, str) or not wid or wid in seen_ids:
            raise ValueError(f"Duplicate or invalid workload id: {wid}")
        seen_ids.add(wid)
        scope = wl.get("evidence_scope")
        if scope not in VALID_SCOPES:
            raise ValueError(f"Invalid evidence_scope on {wid}: {scope}")
        if scope == "protected":
            if wl.get("permitted") is not False or wl.get("command"):
                raise ValueError("Protected confirmation workloads must remain blocked with empty command")
            continue
        if wl.get("profile") not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model profile on {wid}")
        lb = wl.get("logical_batch")
        mb = wl.get("microbatch")
        if type(lb) is not int or type(mb) is not int or lb < 1 or mb < 1 or lb % mb != 0:
            raise ValueError(f"Invalid logical/microbatch on {wid}")
        if wl.get("accumulation_steps") != lb // mb:
            raise ValueError(f"Mismatched accumulation_steps on {wid}")
        retry = wl.get("retry_policy", {})
        if scope == "development":
            if retry.get("auto_retry_permitted") is not False or retry.get("max_attempts") != 1:
                raise ValueError("Development workloads must enforce max_attempts=1 and auto_retry_permitted=False")
            if retry.get("resume_in_place_permitted") is not False:
                raise ValueError("Development workloads must forbid in-place resume")


def stamp_cancellation_receipt(root: int | None, reason: str) -> dict[str, Any]:
    return {
        "status": "CANCELLED",
        "root": root,
        "reason": reason,
        "cancelled_utc": datetime.now(timezone.utc).isoformat(),
    }
