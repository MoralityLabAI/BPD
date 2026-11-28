"""Prepare, inspect and run the BPD-COL-02 Nanbeige gym on a 4-GiB RTX 3050.

The default run is an exposed development pilot. It executes the checkpoint's
native two-loop forward with streamed weights, one prompt at a time. No model
training, external model API, or existing sealed evaluation is involved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cases
import interventions
import scoring
import streaming

SYSTEM_PROMPT = "Reply with one letter."
ANSWER_SUFFIXES = {"A": " A", "B": " B"}
RUN_SCHEMA = "BPD-COL-02/native-streamed-run/v1"
RUNTIME_PACKAGES = ("torch", "transformers", "safetensors", "accelerate", "huggingface_hub",
                    "tokenizers", "sentencepiece", "protobuf", "psutil", "numpy")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    return {path.name: streaming.sha256_file(path) for path in sorted(ROOT.glob("*.py"))}


def find_snapshot(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(streaming.MODEL_ID, revision=streaming.MODEL_REVISION,
                                      local_files_only=True))
    except Exception as error:
        raise RuntimeError("Pinned Nanbeige snapshot is absent. Supply --model-dir or run the download command.") from error


def tokenize_case(tokenizer, case: dict, max_tokens: int = 128) -> dict:
    """Check real continuation boundaries and donor token alignment."""
    variants = {}
    shared_ids = None
    for role in cases.ROLES:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case[role]["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False) + "Answer:"
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not 2 <= len(ids) <= max_tokens:
            raise ValueError(f"{role} has {len(ids)} tokens; cap is {max_tokens}")
        answer_ids = {}
        for label, suffix in ANSWER_SUFFIXES.items():
            extended = tokenizer.encode(text + suffix, add_special_tokens=False)
            if len(extended) != len(ids) + 1 or extended[:-1] != ids:
                raise ValueError(f"Answer {label} is not a single continuation token at the rendered boundary")
            answer_ids[label] = extended[-1]
        if len(set(answer_ids.values())) != 2:
            raise ValueError("Answer tokens must differ")
        if shared_ids is None:
            shared_ids = answer_ids
        elif shared_ids != answer_ids:
            raise ValueError("Answer token IDs changed across matched prompts")
        variants[role] = {"text": text, "input_ids": ids, "answer_token_ids": answer_ids}
    lengths = {len(variant["input_ids"]) for variant in variants.values()}
    if len(lengths) != 1:
        raise ValueError("Donor prompts have different token counts; no padding or silent alignment is permitted")
    if len({tuple(variant["input_ids"]) for variant in variants.values()}) != 3:
        raise ValueError("Distinct oracle roles collapsed to identical model prompts")
    return variants


def score_vector(backend, hidden, answer_ids: dict[str, int]) -> dict:
    import torch
    logits = backend.score_hidden(hidden, [answer_ids["A"], answer_ids["B"]])[0].float()
    if logits.shape != (2,) or not torch.isfinite(logits).all():
        raise ValueError("Invalid answer logits")
    probs = torch.softmax(logits.double(), dim=0).tolist()
    return {"probabilities": dict(zip(("A", "B"), probs)),
            "answer_logits": dict(zip(("A", "B"), logits.tolist())),
            "score_semantics": "conditional next-token probability over the two declared answers"}


class ForwardBudget:
    def __init__(self, maximum: int, seconds: float):
        if type(maximum) is not int or maximum < 1 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive forward and wall-time budgets are required")
        self.maximum, self.seconds = maximum, seconds
        self.used = 0
        self.started = time.monotonic()

    def admit_case(self) -> bool:
        return self.used + 8 <= self.maximum and time.monotonic() - self.started < self.seconds

    def run(self, backend, *args, **kwargs):
        if self.used >= self.maximum or time.monotonic() - self.started >= self.seconds:
            raise RuntimeError("Native forward/time budget exhausted")
        self.used += 1
        return backend.run(*args, **kwargs)


def execute_case(backend, tokenized: dict, case: dict, budget: ForwardBudget, *,
                 dose_fraction: float = 0.1, progress: dict | None = None) -> tuple[dict, dict]:
    import torch
    clean_results, tensors = {}, {}
    progress = {} if progress is None else progress
    clean_scores, arms = progress.setdefault("clean", {}), progress.setdefault("arms", {})
    answer_ids = tokenized["recipient"]["answer_token_ids"]
    for role in cases.ROLES:
        progress["active_stage"] = "clean." + role
        result = budget.run(backend, tokenized[role]["input_ids"], capture_prefix=True)
        clean_results[role] = result
        clean_scores[role] = {
            **score_vector(backend, result.final_hidden_cpu, answer_ids),
            "first_pass": score_vector(backend, result.boundary_last_cpu[0], answer_ids),
            "runtime": result.receipt,
        }
        tensors[role + ".pass1_boundary"] = result.boundary_states_cpu[0].contiguous()
        tensors[role + ".final_query"] = result.final_hidden_cpu.contiguous()
    progress["active_stage"] = "prepare_interventions"
    patches = interventions.prepare_arms(
        *(clean_results[role].boundary_states_cpu[0] for role in cases.ROLES),
        case_id=case["case_id"], dose_fraction=dose_fraction)
    for name in interventions.ARM_NAMES:
        progress["active_stage"] = "arm." + name
        patch = patches[name]
        result = budget.run(backend, tokenized["recipient"]["input_ids"],
                            intervention=patch, capture_prefix=False)
        if patch.calls != 1 or patch.measured_dose is None:
            raise RuntimeError("Intervention did not execute exactly once at the declared loop boundary")
        if not torch.equal(result.boundary_last_cpu[0], clean_results["recipient"].boundary_last_cpu[0]):
            raise RuntimeError("The protected first-pass query differs from the clean recipient")
        arms[name] = {**score_vector(backend, result.final_hidden_cpu, answer_ids),
                      **patch.metadata(), "runtime": result.receipt}
        tensors["arm." + name + ".final_query"] = result.final_hidden_cpu.contiguous()
    progress["active_stage"] = "completed"
    record = {
        "case_id": case["case_id"], "root_id": case["root_id"], "status": "completed",
        "clean": clean_scores, "arms": arms, "answer_token_ids": answer_ids,
        "input_tokens": len(tokenized["recipient"]["input_ids"]),
        "protected_observer": "complete native vocabulary head at final prompt position after pass one",
        "patched_site": "model.norm output, first native loop, all prefix positions except final prompt position",
        "selection_exposure": "public development; fixed dose and random seed before output scoring",
    }
    return record, tensors


def _append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Incomplete record at line {number}; preserve the file before repairing it") from error
    return records


def command_prepare(args) -> None:
    manifest = cases.generate_manifest(args.roots, tuple(args.horizons), args.tier,
                                       cases_per_root=args.cases_per_root)
    if args.output.exists() and json.loads(args.output.read_text()) != manifest:
        raise ValueError("An existing manifest has a different contract; choose a new output path")
    write_json(args.output, manifest)
    print(json.dumps({"manifest": str(args.output), "roots": manifest["n_roots"],
                      "cases": manifest["n_cases"], "sha256": manifest["manifest_sha256"]}))


def command_download(args) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    target = Path(args.model_dir).expanduser() if args.model_dir else Path(HF_HUB_CACHE)
    ancestor = target
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 10 * streaming.GIB:
        raise RuntimeError("Allow at least 10 GiB free disk for the pinned checkpoint download")
    patterns = list(streaming.SOURCE_SHA256) + list(streaming.WEIGHT_FILES) + [
        "tokenizer.json", "tokenizer.model", "special_tokens_map.json", "added_tokens.json"]
    snapshot = snapshot_download(streaming.MODEL_ID, revision=streaming.MODEL_REVISION,
                                 allow_patterns=sorted(set(patterns)),
                                 local_dir=args.model_dir, max_workers=1)
    result = streaming.inspect_snapshot(snapshot, hash_weights=True)
    print(json.dumps({"snapshot": snapshot, "verified": result}, indent=2))


def command_fetch_source(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in streaming.SOURCE_SHA256.items():
        if name == "model.safetensors.index.json":
            continue
        destination = args.output / name
        if destination.exists() and streaming.sha256_file(destination) == expected:
            continue
        url = f"https://huggingface.co/{streaming.MODEL_ID}/resolve/{streaming.MODEL_REVISION}/{name}"
        temporary = destination.with_name(destination.name + ".partial")
        digest, total = hashlib.sha256(), 0
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 64 * 1024 * 1024:
                        raise ValueError("A test-source asset exceeds its bounded download size")
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Official source hash mismatch for {name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"source": str(args.output), "revision": streaming.MODEL_REVISION,
                      "pretrained_weights_downloaded": False}))


def command_doctor(args) -> None:
    report = {"model_id": streaming.MODEL_ID, "revision": streaming.MODEL_REVISION,
              "python": sys.version, "platform": platform.platform(), "cuda_budget_gib": 1.75,
              "added_rss_budget_gib": 2.75, "runtime_ready": False, "errors": []}
    for name in ("torch", "transformers", "safetensors", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "psutil"):
        try:
            report.setdefault("packages", {})[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"Missing package: {name}")
    try:
        import psutil
        report["available_system_ram_bytes"] = psutil.virtual_memory().available
        if report["available_system_ram_bytes"] < 4 * streaming.GIB:
            report["errors"].append("Keep at least 4 GiB of system RAM available before this profile starts")
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            report["errors"].append("CUDA is unavailable; native pretrained run needs the RTX 3050")
        else:
            device = torch.device(args.device)
            props = torch.cuda.get_device_properties(device)
            free, total = torch.cuda.mem_get_info(device)
            report["gpu"] = {"name": props.name, "free_bytes": free, "total_bytes": total,
                             "bf16_supported": torch.cuda.is_bf16_supported()}
            if not torch.cuda.is_bf16_supported():
                report["errors"].append("The registered stream uses native BF16")
            if free < 2 * streaming.GIB:
                report["errors"].append("Keep at least 2 GiB of dedicated GPU memory free")
    except (ImportError, ValueError, RuntimeError, AssertionError) as error:
        report["errors"].append(str(error))
    try:
        snapshot = find_snapshot(args.model_dir)
        report["snapshot"] = str(snapshot)
        report["checkpoint"] = streaming.inspect_snapshot(snapshot, hash_weights=args.verify_weights)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        report["errors"].append(str(error))
    report["runtime_ready"] = not report["errors"]
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["runtime_ready"]:
        raise SystemExit(2)


def command_run(args) -> None:
    if not math.isfinite(args.dose_fraction) or not 0 < args.dose_fraction <= 1:
        raise ValueError("dose-fraction must be finite and in (0,1]")
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 16:
        raise ValueError("cpu-threads must be an integer from 1 through 16")
    if type(args.forward_budget) is not int or args.forward_budget < 8:
        raise ValueError("forward-budget must allow at least one complete eight-forward case")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("max-seconds must be finite and positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases.validate_manifest(manifest)
    import psutil
    baseline_rss = psutil.Process().memory_info().rss
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(20261125)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.device(args.device).type != "cuda":
        raise ValueError("The pretrained profile requires a CUDA device; CPU fixtures have a separate test entry point")
    snapshot = find_snapshot(args.model_dir)
    # Check local executable model/tokenizer source before loading either.
    streaming.inspect_snapshot(snapshot, hash_weights=False)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    tokenized, token_errors = {}, {}
    for case in manifest["cases"]:
        try:
            tokenized[case["case_id"]] = tokenize_case(tokenizer, case)
        except ValueError as error:
            token_errors[case["case_id"]] = str(error)
    contract = {
        "schema": RUN_SCHEMA, "cases_manifest_sha256": manifest["manifest_sha256"],
        "model_id": streaming.MODEL_ID, "revision": streaming.MODEL_REVISION,
        "source_sha256": source_hashes(), "loop_count": 2,
        "dtype": "bfloat16", "max_tokens": 128, "batch_size": 1, "use_cache": False,
        "cuda_budget_gib": 1.75, "added_rss_budget_gib": 2.75,
        "runtime_versions": {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES},
        "python_version": platform.python_version(), "device": args.device,
        "cpu_threads": args.cpu_threads, "random_seed": 20261125,
        "deterministic_algorithms": True, "allow_tf32": False,
        "dose_fraction_of_minimum_donor_distance": args.dose_fraction,
        "delivered_dose_relative_tolerance": 0.02,
        "delivered_dose_absolute_tolerance": 1e-6,
        "scientific_floor_answer_probability": 0.05,
        "system_prompt": SYSTEM_PROMPT, "assistant_prefill": "Answer:",
        "answer_suffixes": ANSWER_SUFFIXES, "arms": list(interventions.ARM_NAMES),
        "score_scope": "conditional two-answer next-token scores; source-parameter-clustered public development",
        "current_observer": "complete vocabulary head at final prompt position after native pass one",
        "native_model_status": "execution requested; success determined by recorded runtime results",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    contract_path, records_path = args.output / "run_contract.json", args.output / "records.jsonl"
    if contract_path.exists():
        if not args.resume:
            raise ValueError("Run directory exists; use --resume for its unchanged contract or choose a fresh output")
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Resume would change the frozen run contract")
        if json.loads((args.output / "cases_manifest.json").read_text()) != manifest:
            raise ValueError("Saved cases differ from the frozen input manifest")
        if json.loads((args.output / "tokenized_prompts.json").read_text()) != tokenized:
            raise ValueError("Tokenized prompts changed since this run was frozen")
    elif args.resume:
        raise ValueError("No frozen run exists to resume")
    else:
        if records_path.exists():
            raise ValueError("Records exist without their frozen contract")
        write_json(contract_path, contract)
        write_json(args.output / "cases_manifest.json", manifest)
        write_json(args.output / "tokenized_prompts.json", tokenized)
    records = _load_records(records_path)
    scoring.summarize(records, manifest)  # Reject duplicates/unknown cases before any GPU work.
    completed = {record["case_id"] for record in records}
    budget = ForwardBudget(args.forward_budget, args.max_seconds)
    exit_code = 0
    session = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "forward_budget": args.forward_budget, "wall_time_budget_seconds": args.max_seconds,
               "baseline_process_rss_bytes": baseline_rss, "resume": args.resume,
               "runtime_versions": contract["runtime_versions"]}
    backend = None
    try:
        for case in manifest["cases"]:
            ident = case["case_id"]
            if ident in completed:
                continue
            if ident in token_errors:
                record = {"case_id": ident, "root_id": case["root_id"], "status": "skipped",
                          "reason": "tokenization: " + token_errors[ident]}
            elif not budget.admit_case():
                session["stop_reason"] = "case-level forward/time budget"
                break
            else:
                progress = {}
                try:
                    if backend is None:
                        backend = streaming.Backend(snapshot, device=args.device, baseline_rss_bytes=baseline_rss)
                        props = torch.cuda.get_device_properties(torch.device(args.device))
                        session["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                                          "compute_capability": [props.major, props.minor]}
                    record, tensors = execute_case(backend, tokenized[ident], case, budget,
                                                   dose_fraction=args.dose_fraction, progress=progress)
                    activation = args.output / "activations" / (ident + ".safetensors")
                    activation.parent.mkdir(parents=True, exist_ok=True)
                    save_file(tensors, str(activation))
                    record["activations"] = {"path": str(activation.relative_to(args.output)),
                                             "sha256": streaming.sha256_file(activation)}
                    del tensors
                except (Exception, KeyboardInterrupt) as error:
                    record = {"case_id": ident, "root_id": case["root_id"], "status": "error",
                              "reason": f"{type(error).__name__}: {error}",
                              "partial_progress": progress,
                              "partial_progress_scope": "diagnostic evidence only; excluded from completed-case estimates"}
                    exit_code = 130 if isinstance(error, KeyboardInterrupt) else 2
            _append_record(records_path, record)
            records.append(record)
            summary = scoring.summarize(records, manifest)
            write_json(args.output / "summary.json", summary)
            print(json.dumps({"case_id": ident, "status": record["status"],
                              "native_forwards_this_session": budget.used,
                              "summary": str(args.output / "summary.json")}), flush=True)
            if exit_code:
                session["stop_reason"] = record["reason"]
                break
    except (Exception, KeyboardInterrupt) as error:
        session["stop_reason"] = f"{type(error).__name__}: {error}"
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 2
    finally:
        session["native_forwards_attempted"] = budget.used
        session["elapsed_seconds"] = time.monotonic() - budget.started
        if backend is not None:
            try:
                session["memory"] = backend.memory_receipt()
            except Exception as error:
                session["memory_receipt_error"] = f"{type(error).__name__}: {error}"
                exit_code = exit_code or 2
            try:
                backend.close()
            except Exception as error:
                session["cleanup_error"] = f"{type(error).__name__}: {error}"
                exit_code = exit_code or 2
        session["runtime_exit_code"] = exit_code
        session_path = args.output / "sessions" / (str(time.time_ns()) + ".json")
        write_json(session_path, session)
        write_json(args.output / "summary.json", scoring.summarize(records, manifest))
    print(json.dumps({"verdict": "PILOT_ONLY", "summary": str(args.output / "summary.json"),
                      "session": str(session_path), "runtime_exit_code": exit_code}, indent=2))
    if exit_code:
        raise SystemExit(exit_code)


def command_score(args) -> None:
    contract = json.loads((args.run / "run_contract.json").read_text())
    if contract.get("schema") != RUN_SCHEMA:
        raise ValueError("Unknown frozen run contract")
    for name in ("cases.py", "scoring.py"):
        if streaming.sha256_file(ROOT / name) != contract.get("source_sha256", {}).get(name):
            raise ValueError(f"Frozen scorer source changed: {name}; preserve this run and use its original code")
    manifest = json.loads((args.run / "cases_manifest.json").read_text())
    if manifest.get("manifest_sha256") != contract["cases_manifest_sha256"]:
        raise ValueError("Saved manifest differs from the frozen run contract")
    summary = scoring.summarize(_load_records(args.run / "records.jsonl"), manifest)
    write_json(args.run / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Freeze public arithmetic/trace cases without a model")
    prepare.add_argument("--roots", type=int, default=4)
    prepare.add_argument("--horizons", type=int, nargs="+", default=[1])
    prepare.add_argument("--cases-per-root", type=int, default=1)
    prepare.add_argument("--tier", choices=cases.TIERS, default="trace")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(function=command_prepare)
    download = sub.add_parser("download", help="Download only the immutable model snapshot")
    download.add_argument("--model-dir")
    download.set_defaults(function=command_download)
    source = sub.add_parser("fetch-test-source", help="Download pinned architecture/tokenizer assets without checkpoint weights")
    source.add_argument("--output", type=Path, required=True)
    source.set_defaults(function=command_fetch_source)
    doctor = sub.add_parser("doctor", help="Inspect dependencies, hardware and checkpoint inventory")
    doctor.add_argument("--model-dir")
    doctor.add_argument("--device", default="cuda:0")
    doctor.add_argument("--verify-weights", action="store_true")
    doctor.add_argument("--output", type=Path)
    doctor.set_defaults(function=command_doctor)
    run = sub.add_parser("run", help="Run the native streamed, inference-only development experiment")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model-dir")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--dose-fraction", type=float, default=0.1)
    run.add_argument("--forward-budget", type=int, default=32)
    run.add_argument("--max-seconds", type=float, default=7200)
    run.add_argument("--cpu-threads", type=int, default=4)
    run.add_argument("--resume", action="store_true")
    run.set_defaults(function=command_run)
    score = sub.add_parser("score", help="Recompute the parameter-clustered summary from saved records")
    score.add_argument("--run", type=Path, required=True)
    score.set_defaults(function=command_score)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
