"""Independent checkpoint and receipt replay for BPD-RP-02.

This auditor implements the GRU recurrence directly from saved tensors. It
imports neither the model nor the experiment implementation. The deterministic
synthetic input generator is shared; semantic labels, tuple partitioning and
input encoding are checked independently after generation. Training gradients
are checked through their receipts rather than by rerunning optimization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from .data import make_pairs
from .stats import summarize


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ARMS = ("A_task", "B_causal", "C_rotated")
TOLERANCE = 1e-5


class AuditFailure(ValueError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0
        self.max_scalar_error = 0.0

    def require(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AuditFailure(message)

    def close(self, actual, expected, message: str, tolerance: float = TOLERANCE) -> None:
        actual, expected = float(actual), float(expected)
        error = abs(actual - expected)
        self.require(math.isfinite(actual) and math.isfinite(expected) and error <= tolerance,
                     f"{message}: actual={actual}, expected={expected}, error={error}")
        self.max_scalar_error = max(self.max_scalar_error, error)


def _read(path: Path):
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(root: int, domain: str, index: int = 0) -> int:
    digest = hashlib.sha256(f"BPD-RP-02:{root}:{domain}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _state_hash(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch_hash(batch) -> str:
    digest = hashlib.sha256()
    for name in ("receiver", "donor", "hybrid", "labels_r", "labels_d", "labels_h",
                 "edit_slot", "query"):
        digest.update(name.encode())
        digest.update(getattr(batch, name).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch(root: int, recipe: dict, domain: str, index: int, n: int,
           delay: int, split: str, checks: Checks):
    batch = make_pairs(_seed(root, domain, index), n, root_seed=root, delay=delay,
                       split=split, slots=recipe["slots"], values=recipe["values"],
                       noise=recipe["noise"])
    _check_semantics(batch, root, recipe, split, delay, checks)
    return batch


def _check_semantics(batch, root: int, recipe: dict, split: str,
                     delay: int, checks: Checks) -> None:
    n, slots, values = len(batch.query), recipe["slots"], recipe["values"]
    checks.require(batch.prefix_steps == slots, "Prefix length disagrees with the number of writes")
    checks.require(int((batch.edit_slot == batch.query).sum()) == (n + 1) // 2,
                   "Edited-slot query balance differs")
    for slot in range(slots):
        checks.require(abs(int((batch.edit_slot == slot).sum()) - n / slots) <= 1,
                       "Edit slots are imbalanced")
    generator = torch.Generator().manual_seed((root * 104729 + 9029) % (2**63 - 1))
    address_permutation = torch.randperm(slots, generator=generator)
    value_permutation = torch.randperm(values, generator=generator)
    address_inverse, value_inverse = address_permutation.argsort(), value_permutation.argsort()
    for feature_name, memory_name, label_name in (
            ("receiver", "values_r", "labels_r"),
            ("donor", "values_d", "labels_d"),
            ("hybrid", "values_h", "labels_h")):
        x = getattr(batch, feature_name)
        memory, labels = getattr(batch, memory_name), getattr(batch, label_name)
        checks.require(tuple(x.shape) == (n, slots + delay + 1, 3 + slots + values),
                       "Feature shape differs from the recipe")
        checks.require(bool(torch.isfinite(x).all()), "Nonfinite model input")
        expected_types = torch.zeros_like(x[:, :, :3])
        expected_types[:, :slots, 0] = 1
        expected_types[:, slots:-1, 1] = 1
        expected_types[:, -1, 2] = 1
        checks.require(torch.equal(x[:, :, :3], expected_types), "Token type encoding differs")
        address = address_inverse[x[:, :slots, 3:3+slots].argmax(-1)]
        written = value_inverse[x[:, :slots, 3+slots:].argmax(-1)]
        checks.require(torch.equal(address.sort(dim=1).values,
                                   torch.arange(slots).expand(n, -1)),
                       "Every semantic slot must be written exactly once")
        reconstructed = torch.empty_like(memory).scatter(1, address, written)
        checks.require(torch.equal(reconstructed, memory), "Raw write inputs disagree with semantic memory")
        decoded_query = address_inverse[x[:, -1, 3:3+slots].argmax(-1)]
        checks.require(torch.equal(decoded_query, batch.query), "Raw query input disagrees with query metadata")
        expected_labels = reconstructed[torch.arange(n), decoded_query]
        checks.require(torch.equal(labels, expected_labels), "Host label disagrees with decoded model inputs")
        code = sum(memory[:, slot] * values ** slot for slot in range(slots))
        is_evaluation = code.remainder(5) <= 1
        checks.require(bool((is_evaluation if split == "eval" else ~is_evaluation).all()),
                       "Semantic tuple crosses the declared train/evaluation partition")
    expected_hybrid = batch.values_r.clone()
    expected_hybrid[torch.arange(n), batch.edit_slot] = batch.values_d[torch.arange(n), batch.edit_slot]
    checks.require(torch.equal(expected_hybrid, batch.values_h), "Hybrid tuple changes the wrong memory")
    checks.require(bool((batch.values_r != batch.values_h).sum(1).eq(1).all()),
                   "A hybrid must change exactly one slot")
    checks.require(torch.equal(batch.receiver[:, slots:], batch.donor[:, slots:]) and
                   torch.equal(batch.receiver[:, slots:], batch.hybrid[:, slots:]),
                   "Paired suffixes differ")


@torch.no_grad()
def manual_rollout(state: dict[str, Tensor], x: Tensor, patch: Tensor | None = None,
                   patch_step: int = 4) -> dict[str, Tensor]:
    """Reconstruct the trained recurrence using only input/output weights and V/U."""
    width = state["recurrent.V"].shape[0]
    h = torch.zeros((len(x), width), dtype=x.dtype, device=x.device)
    encoded = F.linear(x, state["input.weight"], state["input.bias"])
    states, writes = [], []
    for step in range(x.shape[1]):
        write = (h @ state["recurrent.V"]) @ state["recurrent.U"]
        if patch is not None and step == patch_step:
            write = write + patch
        ir, iz, inn = encoded[:, step].chunk(3, dim=-1)
        hr, hz, hn = write.chunk(3, dim=-1)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        h = (1 - update) * candidate + update * h
        states.append(h)
        writes.append(write)
    return {"logits": F.linear(h, state["output.weight"], state["output.bias"]),
            "states": torch.stack(states, dim=1),
            "recurrent_writes": torch.stack(writes, dim=1)}


@torch.no_grad()
def manual_patch(state: dict[str, Tensor], recipient: Tensor, donor: Tensor,
                 groups: Tensor, edit_slot: Tensor, dose: float) -> tuple[Tensor, Tensor]:
    difference = donor @ state["recurrent.V"] - recipient @ state["recurrent.V"]
    selected = groups[edit_slot]
    mask = torch.zeros_like(difference).scatter(1, selected, 1.0)
    raw = (difference * mask) @ state["recurrent.U"]
    norm = torch.linalg.vector_norm(raw, dim=-1)
    valid = norm > 1e-9
    scale = torch.where(valid, dose / norm.clamp_min(1e-9), torch.zeros_like(norm))
    return raw * scale[:, None], valid


def _load_checkpoint(path: Path, recipe: dict, checks: Checks) -> dict[str, Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    checks.require(isinstance(state, dict) and all(isinstance(v, Tensor) for v in state.values()),
                   f"Unexpected checkpoint payload: {path.name}")
    width, components, values = recipe["width"], recipe["components"], recipe["values"]
    shapes = {"input.weight": (3*width, 3+recipe["slots"]+values),
              "input.bias": (3*width,), "output.weight": (values, width),
              "output.bias": (values,), "recurrent.V": (width, components),
              "recurrent.U": (components, 3*width)}
    for name, shape in shapes.items():
        checks.require(name in state and tuple(state[name].shape) == shape,
                       f"Unexpected checkpoint tensor {name}")
        checks.require(bool(torch.isfinite(state[name]).all()), f"Nonfinite checkpoint tensor {name}")
    return state


def _score(logits: Tensor, labels: Tensor, record: dict, checks: Checks, context: str) -> dict:
    predictions = logits.argmax(-1)
    correct = predictions.eq(labels).int()
    checks.require(record["labels"] == labels.tolist(), f"{context}: labels disagree")
    checks.require(record["predictions"] == predictions.tolist(), f"{context}: predictions disagree")
    checks.require(record["correct"] == correct.tolist(), f"{context}: correctness bits disagree")
    checks.close(record["accuracy"], float(correct.float().mean()), f"{context}: accuracy")
    checks.close(record["cross_entropy"], float(F.cross_entropy(logits, labels)), f"{context}: CE")
    return {**record, "labels": labels.tolist(), "predictions": predictions.tolist(),
            "correct": correct.tolist()}


def _verify_attribution(result: dict, warm: dict, recipe: dict, checks: Checks) -> Tensor:
    root = result["root"]
    batch = _batch(root, recipe, "attribution_train", 0, recipe["calibration_n"],
                   recipe["iid_delay"], "train", checks)
    record = result["attribution"]
    checks.require(record["batch_sha256"] == _batch_hash(batch), "Attribution input hash differs")
    candidates = torch.arange(recipe["components"]).reshape(-1, recipe["group_size"])
    checks.require(record["candidate_groups"] == candidates.tolist(), "Attribution candidates differ")
    full = manual_rollout(warm, torch.cat((batch.receiver, batch.donor)))
    n = len(batch.labels_r)
    rs, ds = full["states"][:n, batch.prefix_steps-1], full["states"][n:, batch.prefix_steps-1]
    base = F.cross_entropy(full["logits"][:n], batch.labels_h, reduction="none")
    checks.close(record["clean_training_calibration_accuracy"],
                 float(full["logits"][:n].argmax(-1).eq(batch.labels_r).float().mean()),
                 "Attribution clean accuracy")
    for slot in range(recipe["slots"]):
        selected = batch.edit_slot == slot
        checks.require(record["cases_by_slot"][slot] == int(selected.sum()), "Attribution slot count differs")
        checks.close(record["base_counterfactual_ce_by_slot"][slot], float(base[selected].mean()),
                     "Attribution baseline CE")
    maximum_dose_error = 0.0
    for candidate_id, group in enumerate(candidates):
        delta, valid = manual_patch(warm, rs, ds, group.expand(recipe["slots"], -1),
                                     batch.edit_slot, recipe["intervention_l2"])
        checks.require(bool(valid.any()), "Attribution candidate has no nonzero intervention")
        maximum_dose_error = max(maximum_dose_error,
                                float((delta.norm(dim=-1)[valid]-recipe["intervention_l2"]).abs().max()))
        logits = manual_rollout(warm, batch.receiver, delta, batch.prefix_steps)["logits"]
        losses = F.cross_entropy(logits, batch.labels_h, reduction="none")
        for slot in range(recipe["slots"]):
            checks.close(record["candidate_ce_by_slot"][candidate_id][slot],
                         float(losses[batch.edit_slot == slot].mean()), "Attribution candidate CE")
    chosen = torch.tensor(record["candidate_ce_by_slot"]).argmin(dim=0)
    checks.require(record["selected_candidate"] == chosen.tolist(), "Attribution retention differs")
    groups = candidates[chosen]
    checks.require(record["groups"] == groups.tolist(), "Frozen component groups differ")
    checks.close(record["maximum_dose_error"], maximum_dose_error, "Attribution dose error")
    checks.require(maximum_dose_error <= TOLERANCE, "Attribution physical dose exceeds tolerance")
    return groups


def _verify_training(result: dict, recipe: dict, checks: Checks) -> int:
    root, steps, n = result["root"], recipe["training_steps"], recipe["batch_size"]
    for arm in ARMS:
        checks.require(len(result["arms"][arm]["training"]) == steps, "Training budget differs")
    stream = hashlib.sha256()
    for step in range(steps):
        delay = recipe["train_delays"][step % len(recipe["train_delays"])]
        batch = _batch(root, recipe, "paired_train", step, n, delay, "train", checks)
        digest = _batch_hash(batch)
        stream.update(digest.encode())
        for arm in ARMS:
            row = result["arms"][arm]["training"][step]
            checks.require(row["step"] == step and row["batch_sha256"] == digest,
                           "An arm received a different training batch or step order")
            for field, expected in (("trajectories", 5*n), ("gradient_evaluations", 3),
                                    ("ordinary_examples", 3*n), ("auxiliary_examples", 2*n)):
                checks.require(row[field] == expected, f"Training compute accounting differs: {field}")
            task, auxiliary = row["task_gradient_l2"], row["raw_auxiliary_gradient_l2"]
            scale, applied = row["auxiliary_scale"], row["applied_auxiliary_gradient_l2"]
            checks.require(all(math.isfinite(v) and v >= 0 for v in (task, auxiliary, scale, applied)),
                           "Invalid gradient magnitude")
            degenerate = auxiliary <= 1e-12
            checks.require(row["auxiliary_degenerate"] == degenerate, "Degenerate-gradient flag differs")
            expected_scale = (recipe["auxiliary_gradient_ratio"] * task / auxiliary
                              if arm != "A_task" and not degenerate else 0.0)
            checks.close(scale, expected_scale, "Auxiliary scale", tolerance=max(TOLERANCE, abs(expected_scale)*1e-7))
            checks.close(applied, scale*auxiliary, "Applied auxiliary gradient norm")
            expected_ratio = (recipe["auxiliary_gradient_ratio"]
                              if arm != "A_task" and not degenerate and task > 0 else 0.0)
            checks.close(row["realized_auxiliary_ratio"], expected_ratio, "Auxiliary/task gradient ratio")
            cosine = row["task_auxiliary_cosine"]
            checks.require(math.isfinite(cosine) and abs(cosine) <= 1+TOLERANCE,
                           "Invalid gradient cosine")
            expected_combined = math.sqrt(max(0.0, task*task + applied*applied + 2*task*applied*cosine))
            checks.close(row["preclip_combined_gradient_l2"], expected_combined,
                         "Combined pre-clip gradient norm", tolerance=max(TOLERANCE, expected_combined*1e-6))
            for field in ("task_ce", "causal_ce", "rotated_ce", "parameter_update_l2"):
                checks.require(math.isfinite(row[field]) and row[field] >= 0,
                               f"Invalid training scalar {field}")
            invalid = row["invalid_interventions"]
            checks.require(type(invalid) is int and 0 <= invalid <= n, "Invalid intervention count")
            checks.close(row["physical_l2_min"], 0.0 if invalid else recipe["intervention_l2"],
                         "Training minimum physical dose")
            checks.close(row["physical_l2_max"], 0.0 if invalid == n else recipe["intervention_l2"],
                         "Training maximum physical dose")
            checks.require(0 <= row["physical_sham_l2_max_error"] <= TOLERANCE,
                           "Training sham changes the physical dose")
    checks.require(stream.hexdigest() == result["training_stream_sha256"], "Training stream hash differs")
    return steps * len(ARMS)


def _evaluate_arm(state: dict, record: dict, batches: dict, groups: Tensor,
                  permutation: Tensor, signs: Tensor, recipe: dict,
                  checks: Checks, context: str) -> dict:
    rebuilt = copy.deepcopy(record)
    for name in ("iid", "long"):
        batch = batches[name]
        endpoint = record["eval"][name]
        checks.require(endpoint["batch_sha256"] == _batch_hash(batch), "Ordinary input hash differs")
        checks.require(endpoint["steering_enabled"] is False, "Ordinary evaluation enabled steering")
        logits = manual_rollout(state, batch.receiver)["logits"]
        rebuilt["eval"][name] = _score(logits, batch.labels_r, endpoint, checks, context+":"+name)
    batch = batches["causal"]
    n, prefix = len(batch.labels_r), batch.prefix_steps
    clean = manual_rollout(state, torch.cat((batch.receiver, batch.donor)))
    rs, ds = clean["states"][:n, prefix-1], clean["states"][n:, prefix-1]
    delta, valid = manual_patch(state, rs, ds, groups, batch.edit_slot, recipe["intervention_l2"])
    sham = delta[:, permutation] * signs
    patched = manual_rollout(state, batch.receiver, delta, prefix)
    rotated = manual_rollout(state, batch.receiver, sham, prefix)
    for name, rollout in (("causal", patched), ("causal_sham", rotated)):
        endpoint = record[name]
        checks.require(endpoint["batch_sha256"] == _batch_hash(batch), "Causal input hash differs")
        checks.require(endpoint["targeted"] == batch.query.eq(batch.edit_slot).tolist(),
                       "Causal targeted subgroup differs")
        rebuilt[name] = _score(rollout["logits"], batch.labels_h, endpoint, checks, context+":"+name)
    clean_correct = clean["logits"][:n].argmax(-1).eq(batch.labels_r)
    original_correct = patched["logits"].argmax(-1).eq(batch.labels_r)
    untargeted = batch.query != batch.edit_slot
    write_delta = patched["recurrent_writes"][:, prefix] - clean["recurrent_writes"][:n, prefix]
    metrics = {
        "invalid_interventions": int((~valid).sum()),
        "delivered_l2_min": float(delta.norm(dim=-1).min()),
        "delivered_l2_max": float(delta.norm(dim=-1).max()),
        "sham_norm_max_error": float((delta.norm(dim=-1)-sham.norm(dim=-1)).abs().max()),
        "native_write_max_error": float((write_delta-delta).abs().max()),
        "prefix_state_max_error": float((patched["states"][:, :prefix]-clean["states"][:n, :prefix]).abs().max()),
        "recipient_state_rms": float(rs.square().mean().sqrt()),
        "patched_state_rms": float(patched["states"].square().mean().sqrt()),
        "all_head_max_change": float((F.linear(patched["states"], state["output.weight"], state["output.bias"])
                                      - F.linear(clean["states"][:n], state["output.weight"], state["output.bias"])).abs().max()),
        "terminal_state_distance_mean": float((patched["states"][:, -1]-clean["states"][:n, -1]).norm(dim=-1).mean()),
        "untargeted_original_answer_corruptions": int((untargeted & clean_correct & ~original_correct).sum()),
        "untargeted_original_answer_recoveries": int((untargeted & ~clean_correct & original_correct).sum()),
    }
    checks.require(set(record["physical_audit"]) == set(metrics), "Physical audit metric schema differs")
    for key, value in metrics.items():
        if isinstance(value, int):
            checks.require(record["physical_audit"][key] == value, f"Physical count differs: {key}")
        else:
            checks.close(record["physical_audit"][key], value, f"Physical trace metric: {key}")
    for key in ("sham_norm_max_error", "native_write_max_error", "prefix_state_max_error"):
        checks.require(metrics[key] <= TOLERANCE, f"Physical identity failed: {key}")
    if bool(valid.any()):
        checks.require(float((delta.norm(dim=-1)[valid]-recipe["intervention_l2"]).abs().max()) <= TOLERANCE,
                       "Evaluated physical dose differs")
    rebuilt["physical_audit"] = metrics
    return rebuilt


def audit_root(directory: Path, checks: Checks | None = None) -> tuple[dict, dict]:
    checks = checks or Checks()
    hashes = _read(directory / "artifact_hashes.json")
    expected_files = {"result.json", "shared_warmup.pt", *(arm+".pt" for arm in ARMS)}
    checks.require(set(hashes) == expected_files, "Root artifact manifest is incomplete or unexpected")
    for filename, digest in hashes.items():
        checks.require(_sha(directory / filename) == digest, f"Artifact hash differs: {filename}")
    result = _read(directory / "result.json")
    root, recipe = result["root"], result["recipe"]
    checks.require(directory.name == f"root_{root}", "Root directory identity differs")
    checks.require(result["evidence_class"] == "DEVELOPMENT_ONLY", "Evidence scope changed")
    checks.require(set(result["arms"]) == set(ARMS), "Training arms differ")
    checks.close(recipe["intervention_l2"], .75, "Registered physical dose", tolerance=0)
    checks.close(recipe["auxiliary_gradient_ratio"], .25, "Registered auxiliary ratio", tolerance=0)
    warm = _load_checkpoint(directory / "shared_warmup.pt", recipe, checks)
    shared_hash = _state_hash(warm)
    checks.require(shared_hash == result["shared_initial_state_sha256"], "Shared warmup hash differs")
    checks.require(result["arm_initial_state_sha256"] == {arm: shared_hash for arm in ARMS},
                   "Arms started from different checkpoints")
    parameter_count = sum(warm[name].numel() for name in (
        "input.weight", "input.bias", "output.weight", "output.bias", "recurrent.V", "recurrent.U"))
    checks.require(parameter_count == result["parameter_count"], "Trainable parameter count differs")
    groups = _verify_attribution(result, warm, recipe, checks)
    training_rows = _verify_training(result, recipe, checks)
    generator = torch.Generator().manual_seed(_seed(root, "physical_sham"))
    permutation = torch.randperm(3*recipe["width"], generator=generator)
    signs = 2*torch.randint(0, 2, (3*recipe["width"],), generator=generator).float()-1
    checks.require(result["sham_permutation"] == permutation.tolist() and result["sham_signs"] == signs.tolist(),
                   "Frozen physical sham differs from its assigned seed")
    batches = {name: _batch(root, recipe, "evaluation_"+name, 0, n, delay, "eval", checks)
               for name, n, delay in (("iid", recipe["evaluation_n"], recipe["iid_delay"]),
                                      ("long", recipe["evaluation_n"], recipe["long_delay"]),
                                      ("causal", recipe["causal_n"], recipe["long_delay"]))}
    rebuilt = copy.deepcopy(result)
    for arm in ARMS:
        state = _load_checkpoint(directory / (arm+".pt"), recipe, checks)
        checks.require(_state_hash(state) == result["arms"][arm]["final_state_sha256"],
                       f"Final checkpoint hash differs: {arm}")
        rebuilt["arms"][arm] = _evaluate_arm(state, result["arms"][arm], batches, groups,
                                             permutation, signs, recipe, checks, f"{root}:{arm}")
    report = {"root": root, "verdict": "PASS", "artifact_hashes_verified": len(hashes),
              "checkpoint_hashes_verified": 4, "training_rows_checked": training_rows,
              "attribution_candidates_replayed": recipe["components"] // recipe["group_size"],
              "ordinary_case_conditions_replayed": len(ARMS)*2*recipe["evaluation_n"],
              "causal_case_conditions_replayed": len(ARMS)*2*recipe["causal_n"]}
    return report, rebuilt


def audit_run(directory: Path, *, check_current_source: bool = True) -> dict:
    torch.set_num_threads(1)
    checks = Checks()
    manifest = _read(directory / "run_manifest.json")
    roots = manifest["roots"]
    checks.require(bool(roots) and len(roots) == len(set(roots)), "Root roster is empty or duplicated")
    discovered = sorted(p.name for p in directory.glob("root_*") if p.is_dir())
    checks.require(discovered == sorted(f"root_{root}" for root in roots), "Root roster is incomplete or extended")
    source_hashes = manifest["source_hashes"]
    if manifest.get("source_lock") is not None:
        lock = manifest["source_lock"]
        checks.require(lock["source_hashes"] == source_hashes, "Manifest/source-freeze hashes differ")
        checks.require(lock["recipe"] == manifest["recipe"], "Manifest/source-freeze recipe differs")
    if check_current_source:
        for relative, digest in source_hashes.items():
            path = (REPO / relative).resolve()
            checks.require(path.is_relative_to(REPO), "Frozen source path escapes the repository")
            checks.require(path.is_file() and _sha(path) == digest, f"Current frozen source differs: {relative}")
    reports, rebuilt = [], []
    for root in roots:
        report, result = audit_root(directory / f"root_{root}", checks)
        checks.require(result["recipe"] == manifest["recipe"], "Root recipe differs from run manifest")
        reports.append(report)
        rebuilt.append(result)
    summary = summarize(rebuilt)
    checks.require(summary == _read(directory / "summary.json"), "Independent aggregate statistics differ")
    completion = _read(directory / "completion.json")
    checks.require(completion["completed"] is True and completion["roots"] == roots,
                   "Completion roster differs")
    expected_result_hashes = {str(root): _sha(directory / f"root_{root}" / "result.json") for root in roots}
    checks.require(completion["result_sha256"] == expected_result_hashes, "Completion hashes differ")
    return {"schema_version": 1, "experiment": "BPD-RP-02", "verdict": "PASS",
            "scope": "independent saved-checkpoint recurrence replay and training-receipt audit; optimizer trajectories are not replayed",
            "generator_scope": "shared deterministic input generator with independent raw-input label and semantic-partition checks",
            "source_hashes_checked": len(source_hashes) if check_current_source else 0,
            "current_source_check_skipped": not check_current_source,
            "n_roots": len(roots), "checks": checks.count,
            "scalar_tolerance": TOLERANCE, "maximum_scalar_comparison_error": checks.max_scalar_error,
            "prediction_bits_match": True, "aggregate_statistics_match": True,
            "ordinary_case_conditions_replayed": sum(r["ordinary_case_conditions_replayed"] for r in reports),
            "causal_case_conditions_replayed": sum(r["causal_case_conditions_replayed"] for r in reports),
            "roots": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-current-source", action="store_true",
                        help="Audit preserved historical receipts after a disclosed source amendment")
    args = parser.parse_args()
    try:
        result = audit_run(args.results_dir, check_current_source=not args.skip_current_source)
    except (AuditFailure, KeyError, OSError, ValueError, RuntimeError) as error:
        result = {"verdict": "FAIL", "error": str(error)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(result, sort_keys=True))
    if result["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
