"""INK-BPD-05 optional *local* pretrained Inkling 2 Mini development runner.

Requires an already available full HF checkpoint, explicit frozen task JSONL,
and the exact source implementation. This script never calls network download,
runs no training, and refuses the word 'confirmation' in the task phase.

The lightweight CI tests exercise its instrument on synthetic torch; no
pretrained model is loaded by CI. Use the protocol before spending GPU time.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch

from pretrained_message_hook import (
    InklingMessageHook, PatchSpec, UPSTREAM_MODELING_BLOB,
    UPSTREAM_TRANSFORMERS_COMMIT,
)

ARMS = ["zero", "candidate"] + [f"{group}_{n}" for group in ("coord", "haar") for n in range(4)]
EXPECTED_TEXT_CONFIG = {
    "hidden_size": 2304,
    "num_hidden_layers": 24,
    "loop_layer_index_start": 6,
    "loop_layer_index_end": 18,
    "loop_num_passes": 3,
    "n_layers_per_attn_res_block": 6,
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(model_dir: Path) -> dict:
    """Hash every local model shard, configuration and tokenizer metadata."""
    root = model_dir.expanduser().resolve(strict=True)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError("model-dir must contain a local config.json")
    selected = sorted(
        p for p in root.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(root).parts
        and p.suffix in {".safetensors", ".json", ".model", ".tiktoken"}
    )
    shards = [p for p in selected if p.suffix == ".safetensors"]
    if not shards:
        raise ValueError("no local safetensors shard found; external download disabled")
    files = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
              "sha256": sha256_path(p)} for p in selected]
    manifest = {"files": files}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "file_count": len(files), "shard_count": len(shards),
            "weight_bytes": sum(p.stat().st_size for p in shards), "files": files}


def load_development_tasks(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    if not 0 < len(raw) <= 1024 * 1024:
        raise ValueError("bounded JSONL task file required")
    tasks: list[dict] = []
    seen = set()
    for line_num, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid task JSON at line {line_num}") from exc
        if not isinstance(item, dict) or item.get("phase") != "development":
            raise ValueError("INK-BPD-05 only accepts explicit development tasks")
        for field in ("root_id", "family_id", "prompt", "correct", "distractor"):
            if not isinstance(item.get(field), str) or not item[field]:
                raise ValueError(f"missing {field} on line {line_num}")
        if item["root_id"] in seen or item["correct"] == item["distractor"]:
            raise ValueError("duplicate root_id or identical token candidates")
        seen.add(item["root_id"])
        tasks.append({k: item[k] for k in
                      ("root_id", "family_id", "prompt", "correct", "distractor")})
    if not tasks or len(tasks) > 256:
        raise ValueError("task count must be between 1 and 256")
    return tasks, hashlib.sha256(raw).hexdigest()


def source_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()


def verify_transformers_source() -> str:
    """Fail closed on upstream model implementation drift."""
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopTextModel
    source = inspect.getsourcefile(Inkling2LoopTextModel)
    if source is None:
        raise RuntimeError("unable to inspect Inkling2LoopTextModel source")
    blob = source_blob_sha(Path(source))
    if blob != UPSTREAM_MODELING_BLOB:
        raise RuntimeError(
            "HF Inkling2Loop model code differs from the registered implementation: "
            f"got Git blob {blob}, expected {UPSTREAM_MODELING_BLOB}"
        )
    return blob


def validate_loaded_model(model):
    if getattr(model.config, "model_type", None) != "inkling2_loop":
        raise ValueError("require official inkling2_loop model type")
    core = model.model.language_model
    for key, expected in EXPECTED_TEXT_CONFIG.items():
        observed = getattr(core.config, key, None)
        if observed != expected:
            raise RuntimeError(f"model geometry {key}={observed} vs {expected}")
    if core.message_input.__class__.__name__ != "Inkling2LoopMessageAttentionResidual":
        raise RuntimeError("message input is not the expected pretrained module")
    if tuple(core.message_input.query.shape) != (12, 4, 2304):
        raise RuntimeError("cross-loop reader query tensor changed")
    return core


def candidate_token_id(tokenizer, value: str) -> int:
    ids = tokenizer.encode(value, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"answer must be exactly one checkpoint token: {value!r}")
    return int(ids[0])


def evaluate_logits(logits: torch.Tensor, correct_id: int, distractor_id: int) -> dict:
    """Full-head top-1 and a separately labeled restricted two-token contrast."""
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("requires one final-token logit vector per scored root")
    x = logits[0].float()
    if max(correct_id, distractor_id) >= x.numel():
        raise ValueError("candidate token ID out of LM head range")
    margin = (x[correct_id] - x[distractor_id]).item()
    return {
        "fullhead_correct": int(x.argmax().item() == correct_id),
        "restricted_pair_correct": int(margin > 0),
        "correct_vs_distractor_margin": margin,
        "next_token_top1_id": x.argmax().item(),
        "correct_logit": x[correct_id].item(),
        "distractor_logit": x[distractor_id].item(),
    }


def execute_model(model, inputs: dict, core, spec: PatchSpec | None):
    with torch.inference_mode():
        if spec is None:
            out = model(**inputs, use_cache=False, logits_to_keep=1)
            return out.logits[:, -1].detach().float().cpu(), None
        with InklingMessageHook(core.message_input, spec) as hooked:
            out = model(**inputs, use_cache=False, logits_to_keep=1)
        return out.logits[:, -1].detach().float().cpu(), hooked.finish()


def evaluate_root(model, tokenizer, core, task: dict, *,
                  seed: int, dose_fraction: float, coordinate: int) -> dict:
    correct = candidate_token_id(tokenizer, task["correct"])
    distractor = candidate_token_id(tokenizer, task["distractor"])
    encoded = tokenizer(task["prompt"], add_special_tokens=False, return_tensors="pt")
    if encoded.input_ids.shape[-1] < 2 or encoded.input_ids.shape[-1] > 192:
        raise ValueError("input token length outside registered [2,192] range")
    device = model.get_input_embeddings().weight.device
    if device.type == "meta":
        raise RuntimeError("unmaterialized embedding device, cannot run hook")
    inputs = {k: v.to(device) for k, v in encoded.items()}
    clean_logits, _ = execute_model(model, inputs, core, None)
    args = dict(coordinate=coordinate, dose_fraction=dose_fraction, seed=seed)
    zero_logits, zero_receipt = execute_model(
        model, inputs, core, PatchSpec(arm="zero", **args)
    )
    clean_parity_max = (clean_logits - zero_logits).abs().max().item()
    if clean_parity_max > 0.003:
        raise RuntimeError(f"full-head identity replay failed: {clean_parity_max}")
    outcomes = {}
    for arm in ARMS:
        if arm == "zero":
            scored, audit = zero_logits, zero_receipt
        else:
            scored, audit = execute_model(
                model, inputs, core, PatchSpec(arm=arm, **args)
            )
        outcomes[arm] = {"score": evaluate_logits(scored, correct, distractor),
                         "hook": audit}
    return {
        "root_id": task["root_id"],
        "family_id": task["family_id"],
        "n_input_tokens": int(encoded.input_ids.shape[-1]),
        "correct_token_id": correct,
        "distractor_token_id": distractor,
        "clean": evaluate_logits(clean_logits, correct, distractor),
        "clean_zero_fullhead_max_abs": clean_parity_max,
        "arms": outcomes,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-roots", type=int, default=4)
    p.add_argument("--dose-fraction", type=float, default=0.08)
    p.add_argument("--coordinate", type=int, default=2027)
    p.add_argument("--seed", type=int, default=504271)
    args = p.parse_args()
    if not 1 <= args.max_roots <= 16 or args.coordinate != 2027 or abs(args.dose_fraction - .08) > 1e-12:
        p.error("fixed INK-BPD-05 engineering settings: <=16 roots, coord 2027, dose_fraction 0.08")
    if args.out.resolve() in (args.tasks.resolve(), args.model_dir.resolve()):
        p.error("output cannot replace task file or checkpoint directory")
    # Fail-closed weight loading: all files must be present already.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tasks, tasks_sha = load_development_tasks(args.tasks)
    tasks = tasks[:args.max_roots]
    source = verify_transformers_source()
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "inkling2_loop":
        p.error("local checkpoint must identify inkling2_loop")
    manifest = checkpoint_manifest(args.model_dir)
    from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False
    )
    model = Inkling2LoopForConditionalGeneration.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False,
        device_map="auto", dtype="auto", attn_implementation="eager",
    ).eval()
    core = validate_loaded_model(model)
    observed = []
    for i, task in enumerate(tasks):
        row = evaluate_root(
            model, tokenizer, core, task, seed=args.seed + 1009 * i,
            dose_fraction=args.dose_fraction, coordinate=args.coordinate,
        )
        observed.append(row)
    valid_clean = sum(row["clean"]["fullhead_correct"] for row in observed)
    report = {
        "schema": "bpd/ink-bpd05-pretrained-development/v1",
        "evidence_class": "source-locked_native_pretrained_development_only",
        "model": {"transformers_source_blob": source,
                  "transformers_source_commit": UPSTREAM_TRANSFORMERS_COMMIT,
                  "checkpoint": manifest},
        "tasks": {"sha256": tasks_sha, "assigned": len(observed),
                  "independent_weight_checkpoints": 1,
                  "family_ids": sorted(set(x["family_id"] for x in observed))},
        "protocol": {"arm_names": ARMS, "candidate_coord": args.coordinate,
                     "dose_fraction": args.dose_fraction, "seed": args.seed,
                     "passes": [1], "probe_layer": 11,
                     "sham_family": "4 coordinate + 4 seeded Haar",
                     "label": "restricted next-token choice plus fullhead exact token"},
        "n_fullhead_clean_correct": valid_clean,
        "fullhead_clean_competence_fraction": valid_clean / len(observed),
        "instrument_gate": (
            "DEVELOPMENT_ENGINEERING_ONLY"
            if valid_clean / len(observed) >= .75 else
            "FAIL_CLEAN_COMPETENCE"
        ),
        "per_root": observed,
        "limitations": [
            "Goodfire's coordinate 2027 is externally selection-exposed.",
            "Next-token scoring is a restricted behavioral instrument, not verifier-backed serial reasoning.",
            "One checkpoint, at most 16 development roots, and eight correlated sham directions per root.",
            "No 59-independent-root sham tail calibration and no sealed confirmation release.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"class": report["evidence_class"], "roots": len(observed),
                      "fullhead_clean_correct": valid_clean,
                      "instrument_gate": report["instrument_gate"],
                      "out": str(args.out)}))


if __name__ == "__main__":
    main()
