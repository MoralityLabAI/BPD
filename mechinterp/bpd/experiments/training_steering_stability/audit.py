"""Independent checkpoint, geometry and receipt replay for BPD-RP-03.

Inference is reconstructed directly from saved tensors. This module imports
neither RP03 model nor experiment code. Synthetic inputs use the shared data
generator, followed by independent decoding of semantic labels and tuple folds.
Warmup and paired-training input streams are regenerated, while recorded
gradient accounting is checked arithmetically. Optimizer updates are not
replayed. Anchors at the saved warmup and final checkpoints are replayed;
intermediate anchors have receipt checks without saved-checkpoint replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor
import torch.nn.functional as F

from ..training_steering.audit import (
    AuditFailure, Checks, _batch_hash, _check_semantics, _load_checkpoint,
    _score, _state_hash,
)
from .data import binding_batch, make_pairs
from .stats import summarize


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ARMS = ("A_task", "B_causal_fixed", "C_rotated_fixed", "D_causal_live")
CAUSAL_NAMES = ("causal", "causal_sham", "causal_live")
SMOKE_ROOTS = (27010, 27011)
DEVELOPMENT_ROOTS = tuple(range(27101, 27109))
TOLERANCE = 1e-5


def _read(path: Path):
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(root: int, domain: str, index: int = 0) -> int:
    digest = hashlib.sha256(f"BPD-RP-03:{root}:{domain}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _batch(root: int, recipe: dict, domain: str, index: int, n: int,
           delay: int, split: str, checks: Checks):
    batch = make_pairs(_seed(root, domain, index), n, root_seed=root, delay=delay,
                       split=split, slots=recipe["slots"], values=recipe["values"],
                       noise=recipe["noise"])
    _check_semantics(batch, root, recipe, split, delay, checks)
    return batch


def _tensor_close(actual: Tensor, expected: Tensor, checks: Checks,
                  context: str, tolerance: float = TOLERANCE) -> None:
    checks.require(actual.shape == expected.shape, context + ": shape differs")
    checks.require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
                   context + ": nonfinite tensor")
    error = float((actual - expected).detach().abs().max())
    checks.close(error, 0., context, tolerance)


@torch.no_grad()
def manual_rollout(state: dict[str, Tensor], x: Tensor, patch: Tensor | None = None,
                   patch_step: int = 4,
                   state_replacement: Tensor | None = None) -> dict[str, Tensor]:
    """Direct native recurrence; W_target and all calibration buffers are unused."""
    width = state["recurrent.V"].shape[0]
    h = x.new_zeros((len(x), width))
    encoded = F.linear(x, state["input.weight"], state["input.bias"])
    states, incoming, writes = [], [], []
    for step in range(x.shape[1]):
        if step == patch_step:
            if state_replacement is not None:
                h = state_replacement
            elif patch is not None:
                h = h + patch
        incoming.append(h)
        write = (h @ state["recurrent.V"]) @ state["recurrent.U"]
        ir, iz, inn = encoded[:, step].chunk(3, dim=-1)
        hr, hz, hn = write.chunk(3, dim=-1)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        h = (1 - update) * candidate + update * h
        states.append(h)
        writes.append(write)
    return {"logits": F.linear(h, state["output.weight"], state["output.bias"]),
            "states": torch.stack(states, 1), "incoming_states": torch.stack(incoming, 1),
            "recurrent_writes": torch.stack(writes, 1)}


@torch.no_grad()
def manual_projectors(state: dict[str, Tensor], groups: Tensor) -> Tensor:
    factor = state["recurrent.V"]
    if groups.ndim != 2 or groups.dtype != torch.long or not groups.numel():
        raise AuditFailure("Invalid projector group indices")
    if bool((groups < 0).any()) or bool((groups >= factor.shape[1]).any()):
        raise AuditFailure("Projector group index exceeds factor dimensions")
    output = []
    for group in groups:
        if len(torch.unique(group)) != len(group) or len(group) > factor.shape[0]:
            raise AuditFailure("Projector group has repeated or excessive columns")
        selected = factor[:, group]
        singular = torch.linalg.svdvals(selected)
        threshold = max(selected.shape) * torch.finfo(selected.dtype).eps * singular.max()
        if bool(singular.min() <= threshold):
            raise AuditFailure("Projector group is rank deficient")
        frame = torch.linalg.qr(selected, mode="reduced").Q
        output.append(frame @ frame.T)
    return torch.stack(output)


@torch.no_grad()
def manual_patch(recipient: Tensor, donor: Tensor, projections: Tensor,
                 edit_slot: Tensor, dose: float) -> tuple[Tensor, Tensor]:
    selected = projections[edit_slot]
    difference = donor - recipient
    raw = torch.bmm(difference[:, None, :], selected).squeeze(1)
    length = torch.linalg.vector_norm(raw, dim=-1)
    valid = length > 1e-9
    scale = torch.where(valid, dose / length.clamp_min(1e-9), torch.zeros_like(length))
    return raw * scale[:, None], valid


def _verify_attribution(result: dict, warm: dict, recipe: dict,
                        checks: Checks) -> tuple[Tensor, Tensor]:
    batch = _batch(result["root"], recipe, "attribution_train", 0,
                   recipe["calibration_n"], recipe["iid_delay"], "train", checks)
    record = result["attribution"]
    checks.require(record["batch_sha256"] == _batch_hash(batch), "Attribution input hash differs")
    candidates = torch.arange(recipe["components"]).reshape(-1, recipe["group_size"])
    checks.require(record["candidate_groups"] == candidates.tolist(), "Attribution candidates differ")
    projections = manual_projectors(warm, candidates)
    full = manual_rollout(warm, torch.cat((batch.receiver, batch.donor)))
    n, prefix = len(batch.query), batch.prefix_steps
    receiver, donor = full["states"][:n, prefix-1], full["states"][n:, prefix-1]
    checks.close(record["clean_calibration_accuracy"],
                 float(full["logits"][:n].argmax(-1).eq(batch.labels_r).float().mean()),
                 "Attribution clean accuracy")
    candidate_losses = []
    for index, projection in enumerate(projections):
        delta, valid = manual_patch(receiver, donor, projection.expand(recipe["slots"], -1, -1),
                                     batch.edit_slot, recipe["intervention_l2"])
        logits = manual_rollout(warm, batch.receiver, delta, prefix)["logits"]
        losses = F.cross_entropy(logits, batch.labels_h, reduction="none")
        correct = logits.argmax(-1).eq(batch.labels_h)
        per_slot = []
        for slot in range(recipe["slots"]):
            selected = batch.edit_slot == slot
            targeted = selected & (batch.query == slot)
            checks.require(bool(targeted.any()), "Attribution slot lacks targeted examples")
            loss = float(losses[selected].mean())
            per_slot.append(loss)
            checks.close(record["candidate_ce_by_slot"][index][slot], loss,
                         "Attribution candidate counterfactual CE")
            checks.close(record["candidate_targeted_accuracy_by_slot"][index][slot],
                         float(correct[targeted].float().mean()), "Attribution targeted accuracy")
        candidate_losses.append(per_slot)
        checks.require(record["invalid_by_candidate"][index] == int((~valid).sum()),
                       "Attribution invalid-direction count differs")
        if bool(valid.any()):
            checks.close(float((delta.norm(dim=-1)[valid]-recipe["intervention_l2"]).abs().max()),
                         0., "Attribution physical dose")
    selected = torch.tensor(candidate_losses).argmin(0)
    groups = candidates[selected]
    checks.require(record["selected_candidate"] == selected.tolist(), "Attribution selected candidate differs")
    checks.require(record["groups"] == groups.tolist(), "Attribution selected groups differ")
    return groups, projections[selected]


def _verify_geometry(directory: Path, result: dict, warm: dict,
                     groups: Tensor, fixed: Tensor, recipe: dict,
                     checks: Checks) -> tuple[Tensor, Tensor]:
    saved = torch.load(directory / "geometry.pt", weights_only=True, map_location="cpu")
    checks.require(set(saved) == {"groups", "fixed_projectors", "permutation", "signs"},
                   "Saved geometry schema differs")
    checks.require(saved["groups"].dtype == torch.long and torch.equal(saved["groups"], groups),
                   "Saved geometry groups differ")
    _tensor_close(saved["fixed_projectors"], fixed, checks, "Saved fixed projectors")
    _tensor_close(fixed, manual_projectors(warm, groups), checks, "Fixed geometry from common warmup")
    _tensor_close(fixed, fixed.transpose(1, 2), checks, "Projector symmetry")
    _tensor_close(fixed @ fixed, fixed, checks, "Projector idempotence")
    checks.close(float((fixed.diagonal(dim1=1, dim2=2).sum(-1)-recipe["group_size"]).abs().max()),
                 0., "Projector rank through trace")
    generator = torch.Generator().manual_seed(_seed(result["root"], "physical_sham"))
    permutation = torch.randperm(recipe["width"], generator=generator)
    signs = 2*torch.randint(0, 2, (recipe["width"],), generator=generator).float()-1
    checks.require(torch.equal(saved["permutation"], permutation) and torch.equal(saved["signs"], signs),
                   "Saved physical rotation differs from its seed")
    checks.require(result["sham_permutation"] == permutation.tolist()
                   and result["sham_signs"] == signs.tolist(), "Rotation receipt differs")
    return permutation, signs


def _verify_training(result: dict, recipe: dict, checks: Checks) -> tuple[int, int]:
    root, steps, n = result["root"], recipe["training_steps"], recipe["batch_size"]
    warm_stream = hashlib.sha256()
    for step in range(recipe["warmup_steps"]):
        delay = recipe["train_delays"][step % len(recipe["train_delays"])]
        batch = _batch(root, recipe, "warmup", step, n, delay, "train", checks)
        warm_stream.update(_batch_hash(batch).encode())
    checks.require(warm_stream.hexdigest() == result["warmup_stream_sha256"], "Warmup input stream hash differs")
    expected_trace_steps = list(range(100, recipe["warmup_steps"]+1, 100))
    checks.require([row["step"] for row in result["warmup_trace"]] == expected_trace_steps,
                   "Warmup logging budget differs")
    for row in result["warmup_trace"]:
        checks.require(math.isfinite(row["task_ce"]) and row["task_ce"] >= 0, "Invalid warmup loss receipt")
    for arm in ARMS:
        checks.require(len(result["arms"][arm]["training"]) == steps, "Paired training budget differs")
    stream = hashlib.sha256()
    for step in range(steps):
        delay = recipe["train_delays"][step % len(recipe["train_delays"])]
        batch = _batch(root, recipe, "paired_train", step, n, delay, "train", checks)
        digest = _batch_hash(batch)
        stream.update(digest.encode())
        for arm in ARMS:
            row = result["arms"][arm]["training"][step]
            checks.require(row["step"] == step and row["batch_sha256"] == digest,
                           "Arm input batch or training step differs")
            for field, expected in (("trajectories", 6*n), ("gradient_evaluations", 4),
                                    ("ordinary_examples", 3*n), ("auxiliary_examples", 3*n)):
                checks.require(row[field] == expected, "Compute accounting differs: " + field)
            task, auxiliary = row["task_gradient_l2"], row["raw_auxiliary_gradient_l2"]
            scale, applied = row["auxiliary_scale"], row["applied_auxiliary_gradient_l2"]
            checks.require(all(math.isfinite(value) and value >= 0
                               for value in (task, auxiliary, scale, applied)), "Invalid gradient magnitude")
            degenerate = auxiliary <= 1e-12
            checks.require(row["auxiliary_degenerate"] == degenerate, "Degenerate-gradient flag differs")
            expected_scale = (recipe["auxiliary_gradient_ratio"] * task / auxiliary
                              if arm != "A_task" and not degenerate else 0.)
            checks.close(scale, expected_scale, "Auxiliary scale",
                         tolerance=max(TOLERANCE, abs(expected_scale)*1e-7))
            checks.close(applied, scale*auxiliary, "Applied auxiliary norm")
            expected_ratio = (recipe["auxiliary_gradient_ratio"]
                              if arm != "A_task" and not degenerate and task > 0 else 0.)
            checks.close(row["realized_auxiliary_ratio"], expected_ratio, "Auxiliary/task gradient ratio")
            cosine = row["task_auxiliary_cosine"]
            checks.require(math.isfinite(cosine) and abs(cosine) <= 1+TOLERANCE, "Invalid gradient cosine")
            combined = math.sqrt(max(0., task*task+applied*applied+2*task*applied*cosine))
            checks.close(row["preclip_combined_gradient_l2"], combined, "Combined pre-clip norm",
                         tolerance=max(TOLERANCE, combined*1e-6))
            for field in ("task_ce", "fixed_ce", "rotated_ce", "live_ce", "parameter_update_l2"):
                checks.require(math.isfinite(row[field]) and row[field] >= 0, "Invalid training scalar: " + field)
            for name in ("fixed", "live"):
                invalid = row[name+"_invalid"]
                checks.require(type(invalid) is int and 0 <= invalid <= n, "Invalid direction count")
                checks.close(row[name+"_physical_l2_min"], 0. if invalid else recipe["intervention_l2"],
                             name + " training minimum dose")
                checks.close(row[name+"_physical_l2_max"], 0. if invalid == n else recipe["intervention_l2"],
                             name + " training maximum dose")
            checks.require(row["projectors_detached"] is True, "Projector detach receipt differs")
            checks.require(0 <= row["sham_norm_max_error"] <= TOLERANCE, "Training rotation changes dose")
    checks.require(stream.hexdigest() == result["training_stream_sha256"], "Paired training stream hash differs")
    return steps * len(ARMS), recipe["warmup_steps"]


def _binding(root: int, recipe: dict, checks: Checks):
    x, labels, metadata = binding_batch(root, "evaluation_binding", recipe["binding_n"],
                                        recipe["long_delay"], "eval", SimpleNamespace(**recipe))
    n, slots, values = len(x), recipe["slots"], recipe["values"]
    checks.require(n == recipe["binding_n"] and n % slots == 0 and slots == values,
                   "Binding cohort dimensions differ")
    checks.require(tuple(x.shape) == (n, slots+recipe["long_delay"]+1, 3+slots+values),
                   "Binding feature dimensions differ")
    checks.require(bool(torch.isfinite(x).all()), "Binding inputs are nonfinite")
    expected_types = torch.zeros_like(x[:, :, :3])
    expected_types[:, :slots, 0] = 1
    expected_types[:, slots:-1, 1] = 1
    expected_types[:, -1, 2] = 1
    checks.require(torch.equal(x[:, :, :3], expected_types), "Binding token types differ")
    generator = torch.Generator().manual_seed((root*104729+9029) % (2**63-1))
    address = torch.randperm(slots, generator=generator)
    value = torch.randperm(values, generator=generator)
    decoded_address = address.argsort()[x[:, :slots, 3:3+slots].argmax(-1)]
    decoded_value = value.argsort()[x[:, :slots, 3+slots:].argmax(-1)]
    checks.require(torch.equal(decoded_address.sort(1).values, torch.arange(slots).expand(n, -1)),
                   "Binding memory does not write every address once")
    memory = torch.empty((n, slots), dtype=torch.long).scatter(1, decoded_address, decoded_value)
    query = address.argsort()[x[:, -1, 3:3+slots].argmax(-1)]
    checks.require(torch.equal(memory.sort(1).values, torch.arange(values).expand(n, -1)),
                   "Binding memory values are not all distinct")
    code = sum(memory[:, slot] * values**slot for slot in range(slots))
    checks.require(bool(code.remainder(5).le(1).all()), "Binding tuple crosses its evaluation partition")
    checks.require(torch.equal(query, torch.arange(slots).repeat(n//slots)), "Binding query groups differ")
    decoded_labels = memory[torch.arange(n), query]
    checks.require(torch.equal(labels, decoded_labels), "Binding labels disagree with raw encoded inputs")
    grouped = x.reshape(n//slots, slots, x.shape[1], x.shape[2])
    checks.require(torch.equal(grouped[:, :, :-1], grouped[:, :1, :-1].expand(-1, slots, -1, -1)),
                   "Binding query variants change their memory prefix")
    query_residual = x[:, -1].clone()
    query_residual[torch.arange(n), 3+address[query]] -= 1
    grouped_residual = query_residual.reshape(n//slots, slots, -1)
    _tensor_close(grouped_residual, grouped_residual[:, :1].expand_as(grouped_residual), checks,
                  "Binding query variants preserve distractor channels and noise")
    checks.require(metadata["memories"] == memory.tolist() and metadata["queries"] == query.tolist(),
                   "Binding semantic metadata disagrees with raw inputs")
    digest = hashlib.sha256(x.contiguous().numpy().tobytes()+labels.numpy().tobytes()).hexdigest()
    checks.require(metadata["batch_sha256"] == digest, "Binding input hash differs")
    checks.require(metadata["split"] == "eval" and metadata["n_base_sequences"] == n//slots,
                   "Binding metadata scope differs")
    checks.close(metadata["modal_baseline_accuracy"], 1/values, "Binding modal baseline")
    return x, labels, metadata


def _compare_metrics(record: dict, metrics: dict, checks: Checks, context: str) -> None:
    checks.require(set(record) == set(metrics), context + ": metric schema differs")
    for name, value in metrics.items():
        if isinstance(value, int):
            checks.require(record[name] == value, context + ": count differs: " + name)
        elif isinstance(value, list):
            checks.require(len(record[name]) == len(value), context + ": list length differs")
            for actual, expected in zip(record[name], value, strict=True):
                checks.close(actual, expected, context + ": " + name)
        else:
            checks.close(record[name], value, context + ": " + name)


def _causal_replay(state: dict, batch, groups: Tensor, fixed: Tensor,
                   permutation: Tensor, signs: Tensor, recipe: dict, record: dict,
                   checks: Checks, context: str) -> dict:
    rebuilt = copy.deepcopy(record)
    n, step = len(batch.query), batch.prefix_steps
    ordinary = manual_rollout(state, torch.cat((batch.receiver, batch.donor, batch.hybrid)))
    receiver, donor, hybrid = (ordinary["states"][i*n:(i+1)*n, step-1] for i in range(3))
    live = manual_projectors(state, groups)
    fixed_delta, fixed_valid = manual_patch(receiver, donor, fixed, batch.edit_slot, recipe["intervention_l2"])
    live_delta, live_valid = manual_patch(receiver, donor, live, batch.edit_slot, recipe["intervention_l2"])
    deltas = {"causal": fixed_delta, "causal_sham": fixed_delta[:, permutation]*signs,
              "causal_live": live_delta}
    target = batch.query == batch.edit_slot
    clean_correct = ordinary["logits"][:n].argmax(-1).eq(batch.labels_r)
    for name, delta in deltas.items():
        run = manual_rollout(state, batch.receiver, delta, step)
        endpoint = record[name]
        checks.require(endpoint["batch_sha256"] == _batch_hash(batch), context + ": causal input hash differs")
        checks.require(endpoint["targeted"] == target.tolist(), context + ": targeted mask differs")
        rebuilt[name] = _score(run["logits"], batch.labels_h, endpoint, checks, context+":"+name)
        predicted_original = run["logits"].argmax(-1).eq(batch.labels_r)
        metrics = {
            "delivered_l2_min": float(delta.norm(dim=-1).min()),
            "delivered_l2_max": float(delta.norm(dim=-1).max()),
            "incoming_state_delta_max_error": float((run["incoming_states"][:, step]-receiver-delta).abs().max()),
            "prefix_state_max_error": float((run["states"][:, :step]-ordinary["states"][:n, :step]).abs().max()),
            "recipient_state_rms": float(receiver.square().mean().sqrt()),
            "edited_incoming_state_rms": float(run["incoming_states"][:, step].square().mean().sqrt()),
            "terminal_state_distance_mean": float((run["states"][:, -1]-ordinary["states"][:n, -1]).norm(dim=-1).mean()),
            "all_head_max_change": float((F.linear(run["states"], state["output.weight"], state["output.bias"])
                - F.linear(ordinary["states"][:n], state["output.weight"], state["output.bias"])).abs().max()),
            "untargeted_original_recoveries": int((~target & ~clean_correct & predicted_original).sum()),
            "untargeted_original_corruptions": int((~target & clean_correct & ~predicted_original).sum()),
        }
        _compare_metrics(record["physical_audit"][name], metrics, checks, context+":"+name+":physical")
        for field in ("incoming_state_delta_max_error", "prefix_state_max_error"):
            checks.require(metrics[field] <= TOLERANCE, context + ": physical identity failed: " + field)
        valid = live_valid if name == "causal_live" else fixed_valid
        if bool(valid.any()):
            checks.close(float((delta.norm(dim=-1)[valid]-recipe["intervention_l2"]).abs().max()),
                         0., context + ": delivered hidden dose")
        checks.require(bool(delta[~valid].eq(0).all()), context + ": invalid directions are nonzero")
        rebuilt["physical_audit"][name] = metrics
    oracle = manual_rollout(state, batch.receiver, patch_step=step, state_replacement=hybrid)
    oracle_record = record["hybrid_state_oracle"]
    checks.require(oracle_record["targeted"] == target.tolist(), context + ": oracle targeted mask differs")
    rebuilt["hybrid_state_oracle"] = _score(oracle["logits"], batch.labels_h, oracle_record,
                                            checks, context+":hybrid_state_oracle")
    error = float((oracle["logits"]-ordinary["logits"][2*n:]).abs().max())
    checks.close(oracle_record["native_hybrid_logits_max_error"], error, context + ": oracle logit parity")
    checks.require(error <= TOLERANCE, context + ": full-state oracle fails native hybrid parity")
    checks.close(oracle_record["physical_l2_mean"], float((hybrid-receiver).norm(dim=-1).mean()),
                 context + ": full-state oracle dose")
    geometry = {
        "projector_chordal_by_slot": ((live-fixed).double().square().sum((-1, -2))/(2*recipe["group_size"])).sqrt().tolist(),
        "fixed_live_delta_cosine_mean": float(F.cosine_similarity(fixed_delta, live_delta, dim=-1).mean()),
        "fixed_live_delta_distance_mean": float((fixed_delta-live_delta).norm(dim=-1).mean()),
        "fixed_invalid": int((~fixed_valid).sum()), "live_invalid": int((~live_valid).sum()),
        "fixed_sham_norm_max_error": float((fixed_delta.norm(dim=-1)-deltas["causal_sham"].norm(dim=-1)).abs().max()),
    }
    _compare_metrics(record["geometry"], geometry, checks, context+":geometry")
    checks.require(geometry["fixed_sham_norm_max_error"] <= TOLERANCE, context + ": physical rotation changes dose")
    rebuilt["geometry"] = geometry
    rebuilt["ordinary_on_causal"] = _score(ordinary["logits"][:n], batch.labels_r,
                                           record["ordinary_on_causal"], checks, context+":ordinary_on_causal")
    return rebuilt


def _record_arithmetic(record: dict, labels: Tensor, checks: Checks, context: str) -> None:
    checks.require(record["labels"] == labels.tolist(), context + ": labels differ")
    predictions = record["predictions"]
    checks.require(len(predictions) == len(labels) and all(type(v) is int and 0 <= v < 4 for v in predictions),
                   context + ": prediction labels invalid")
    correct = [int(a == b) for a, b in zip(predictions, labels.tolist(), strict=True)]
    checks.require(record["correct"] == correct, context + ": correctness bits differ from labels/predictions")
    checks.close(record["accuracy"], float(torch.tensor(correct).float().mean()), context + ": accuracy arithmetic")
    checks.require(math.isfinite(record["cross_entropy"]) and record["cross_entropy"] >= 0,
                   context + ": invalid CE receipt")


def _anchor_receipt(record: dict, batch, recipe: dict, checks: Checks, context: str) -> None:
    """Check intermediate receipts without claiming to recover unsaved models."""
    target = (batch.query == batch.edit_slot).tolist()
    for name in (*CAUSAL_NAMES, "hybrid_state_oracle"):
        _record_arithmetic(record[name], batch.labels_h, checks, context+":"+name)
        checks.require(record[name]["targeted"] == target, context + ": targeted anchor mask differs")
        if name != "hybrid_state_oracle":
            checks.require(record[name]["batch_sha256"] == _batch_hash(batch), context + ": anchor batch hash differs")
    _record_arithmetic(record["ordinary_on_causal"], batch.labels_r, checks, context+":ordinary")
    geometry = record["geometry"]
    checks.require(len(geometry["projector_chordal_by_slot"]) == recipe["slots"], context + ": slot count differs")
    checks.require(all(math.isfinite(value) and 0 <= value <= 1+TOLERANCE
                       for value in geometry["projector_chordal_by_slot"]), context + ": invalid chordal distance")
    checks.require(0 <= geometry["fixed_sham_norm_max_error"] <= TOLERANCE, context + ": rotation dose receipt fails")
    checks.require(abs(geometry["fixed_live_delta_cosine_mean"]) <= 1+TOLERANCE,
                   context + ": invalid direction cosine")
    for name in CAUSAL_NAMES:
        physical = record["physical_audit"][name]
        invalid = geometry["live_invalid" if name == "causal_live" else "fixed_invalid"]
        checks.require(type(invalid) is int and 0 <= invalid <= len(batch.query), context + ": invalid direction count")
        checks.close(physical["delivered_l2_min"], 0. if invalid else recipe["intervention_l2"],
                     context + ": minimum dose receipt")
        checks.close(physical["delivered_l2_max"], 0. if invalid == len(batch.query) else recipe["intervention_l2"],
                     context + ": maximum dose receipt")
        for field in ("incoming_state_delta_max_error", "prefix_state_max_error"):
            checks.require(0 <= physical[field] <= TOLERANCE, context + ": physical identity receipt fails")


def audit_root(directory: Path, checks: Checks | None = None) -> tuple[dict, dict]:
    checks = checks or Checks()
    hashes = _read(directory / "artifact_hashes.json")
    expected = {"result.json", "shared_warmup.pt", "geometry.pt", *(arm+".pt" for arm in ARMS)}
    checks.require(set(hashes) == expected, "Root artifact inventory differs")
    for filename, digest in hashes.items():
        checks.require(_sha(directory / filename) == digest, "Artifact hash differs: " + filename)
    result = _read(directory / "result.json")
    root, recipe = result["root"], result["recipe"]
    checks.require(directory.name == f"root_{root}", "Root directory identity differs")
    checks.require(result["evidence_class"] == "DEVELOPMENT_ONLY", "Root evidence scope differs")
    checks.require(set(result["arms"]) == set(ARMS), "Arm roster differs")
    checks.close(recipe["intervention_l2"], .75, "Declared hidden-state dose", tolerance=0)
    checks.close(recipe["auxiliary_gradient_ratio"], .25, "Declared auxiliary ratio", tolerance=0)
    warm = _load_checkpoint(directory / "shared_warmup.pt", recipe, checks)
    warm_hash = _state_hash(warm)
    checks.require(warm_hash == result["shared_initial_state_sha256"], "Common warmup checkpoint hash differs")
    checks.require(result["arm_initial_state_sha256"] == {arm: warm_hash for arm in ARMS},
                   "Arm initialization receipts differ")
    parameter_count = sum(warm[name].numel() for name in (
        "input.weight", "input.bias", "output.weight", "output.bias", "recurrent.V", "recurrent.U"))
    checks.require(parameter_count == result["parameter_count"], "Trainable parameter count differs")
    groups, fixed = _verify_attribution(result, warm, recipe, checks)
    permutation, signs = _verify_geometry(directory, result, warm, groups, fixed, recipe, checks)
    training_rows, warmup_batches = _verify_training(result, recipe, checks)
    batches = {name: _batch(root, recipe, "evaluation_"+name, 0, n, delay, "eval", checks)
               for name, n, delay in (("iid", recipe["evaluation_n"], recipe["iid_delay"]),
                                      ("long", recipe["evaluation_n"], recipe["long_delay"]),
                                      ("causal", recipe["causal_n"], recipe["long_delay"]))}
    binding_x, binding_labels, binding_metadata = _binding(root, recipe, checks)
    anchor = _batch(root, recipe, "anchor_train", 0, recipe["anchor_n"], recipe["long_delay"], "train", checks)
    expected_steps = [0] + [step for step in range(1, recipe["training_steps"]+1)
                            if step in recipe["anchor_steps"]]
    checks.require(recipe["training_steps"] in expected_steps, "The final anchor was not captured")
    rebuilt = copy.deepcopy(result)
    replayed_anchors = 0
    receipt_only_anchors = 0
    for arm in ARMS:
        record = result["arms"][arm]
        state = _load_checkpoint(directory / (arm+".pt"), recipe, checks)
        checks.require(_state_hash(state) == record["final_state_sha256"], "Final checkpoint hash differs: " + arm)
        rebuilt_arm = copy.deepcopy(record)
        for name in ("iid", "long"):
            batch, endpoint = batches[name], record["eval"][name]
            checks.require(endpoint["batch_sha256"] == _batch_hash(batch), "Ordinary input hash differs")
            checks.require(endpoint["steering_enabled"] is False, "Ordinary evaluation enabled steering")
            rebuilt_arm["eval"][name] = _score(manual_rollout(state, batch.receiver)["logits"], batch.labels_r,
                                                endpoint, checks, f"{root}:{arm}:{name}")
        checks.require(record["binding"]["steering_enabled"] is False, "Binding evaluation enabled steering")
        for key, value in binding_metadata.items():
            checks.require(record["binding"][key] == value, "Binding metadata differs: " + key)
        rebuilt_arm["binding"] = _score(manual_rollout(state, binding_x)["logits"], binding_labels,
                                         record["binding"], checks, f"{root}:{arm}:binding")
        rebuilt_arm = _causal_replay(state, batches["causal"], groups, fixed, permutation, signs,
                                     recipe, rebuilt_arm, checks, f"{root}:{arm}:evaluation")
        anchors = record["anchors"]
        checks.require([row["step"] for row in anchors] == expected_steps, "Anchor step roster differs")
        checks.require(anchors[0] == result["arms"]["A_task"]["anchors"][0], "Common warmup anchors differ across arms")
        new_anchors = []
        for row in anchors:
            step = row["step"]
            context = f"{root}:{arm}:anchor:{step}"
            checks.require(row["scope"] == "fixed training-partition anchor; never used for updates or selection",
                           context + ": scope differs")
            if step in (0, recipe["training_steps"]):
                anchor_state = warm if step == 0 else state
                new_anchors.append(_causal_replay(anchor_state, anchor, groups, fixed, permutation,
                                                  signs, recipe, row, checks, context))
                replayed_anchors += 1
            else:
                _anchor_receipt(row, anchor, recipe, checks, context)
                new_anchors.append(copy.deepcopy(row))
                receipt_only_anchors += 1
        rebuilt_arm["anchors"] = new_anchors
        rebuilt["arms"][arm] = rebuilt_arm
    report = {
        "root": root, "verdict": "PASS", "artifact_hashes_verified": len(hashes),
        "saved_model_checkpoints_verified": 1+len(ARMS), "geometry_checkpoint_verified": True,
        "arm_initialization_receipts_checked": len(ARMS), "training_rows_checked": training_rows,
        "warmup_input_batches_reconstructed": warmup_batches,
        "paired_training_input_batches_reconstructed": recipe["training_steps"],
        "attribution_candidates_replayed": recipe["components"]//recipe["group_size"],
        "ordinary_case_conditions_replayed": len(ARMS)*2*recipe["evaluation_n"],
        "binding_case_conditions_replayed": len(ARMS)*recipe["binding_n"],
        "causal_case_conditions_replayed": len(ARMS)*3*recipe["causal_n"],
        "oracle_case_conditions_replayed": len(ARMS)*recipe["causal_n"],
        "ordinary_on_causal_case_conditions_replayed": len(ARMS)*recipe["causal_n"],
        "endpoint_anchor_records_replayed": replayed_anchors,
        "anchor_causal_case_conditions_replayed": replayed_anchors*3*recipe["anchor_n"],
        "anchor_oracle_case_conditions_replayed": replayed_anchors*recipe["anchor_n"],
        "anchor_ordinary_case_conditions_replayed": replayed_anchors*recipe["anchor_n"],
        "intermediate_anchor_records_receipt_checked_only": receipt_only_anchors,
    }
    return report, rebuilt


def _verify_preflight(directory: Path, results: list[dict], manifest: dict,
                       checks: Checks) -> dict:
    """Rebuild planning inputs from replayed bits, then invoke the pinned planner."""
    rows = []
    for result in results:
        ordinary = {arm: result["arms"][arm]["eval"]["long"]["correct"] for arm in ARMS}
        checks.require(all(len(bits) == 512 for bits in ordinary.values()),
                       "Preflight denominator differs from the registered 512 cases per root")
        counts = {arm: sum(bits) for arm, bits in ordinary.items()}
        rows.append({"unit_id": str(result["root"]),
                     "paired_effect": f"{counts['B_causal_fixed']-counts['A_task']}/512",
                     "sham_effects": [f"{counts['C_rotated_fixed']-counts['A_task']}/512"]})
    unopened = manifest["phase"] == "smoke"
    expected = {
        "schema_version": 1, "experiment_id": "BPD-RP-03-hidden-state-steering-development",
        "independent_unit": "independently trained model root with four paired descendants",
        "alpha": ".05", "scientific_min_effect": ".05", "target_effect": ".10",
        "measurement_resolution": "1/512", "sham_quantile": ".95", "sham_confidence": ".95",
        "planned_comparisons": 3, "confirm_units_planned": len(DEVELOPMENT_ROOTS), "pilot": rows,
        "pilot_from_development_only": True, "heldout_sealed": unopened, "units_independent": True,
        "planning_scope": "Eight bounded development roots only; confirmation outside this protocol",
        "heldout_scope": ("Eight assigned development roots remain unopened" if unopened
                          else "Declared development readout has been consumed; no confirmation cohort allocated"),
        "sham_scope": "same-site physical norm and auxiliary gradient ratio matched; small-n population sham tail uncalibrated",
    }
    checks.require(_read(directory / "preflight_input.json") == expected,
                   "Preflight input effects, family, floor or cohort scope differs")
    path = HERE.parent / "repair_portfolio" / "evidence" / "preflight.py"
    relative = str(path.relative_to(REPO))
    checks.require(manifest["source_hashes"].get(relative) == _sha(path),
                   "Preflight implementation differs from its frozen hash")
    loader = importlib.util.spec_from_file_location("rp03_independent_pinned_preflight", path)
    checks.require(loader is not None and loader.loader is not None, "Pinned preflight cannot be loaded")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    output = module.preflight(expected)
    checks.require(output == _read(directory / "preflight_result.json"), "Recomputed pinned preflight result differs")
    return output


def audit_run(directory: Path, *, check_current_source: bool = True) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checks = Checks()
    manifest = _read(directory / "run_manifest.json")
    roots = manifest["roots"]
    checks.require(manifest["phase"] in ("smoke", "development"), "Unsupported run phase")
    allowed = SMOKE_ROOTS if manifest["phase"] == "smoke" else DEVELOPMENT_ROOTS
    checks.require(bool(roots) and len(roots) == len(set(roots)) and all(root in allowed for root in roots),
                   "Root roster is empty, duplicated or outside the phase cohort")
    discovered = sorted(path.name for path in directory.glob("root_*") if path.is_dir())
    checks.require(discovered == sorted(f"root_{root}" for root in roots), "Root roster is incomplete or extended")
    checks.require(manifest["evidence_class"] == "DEVELOPMENT_ONLY", "Run evidence scope differs")
    source_hashes = manifest["source_hashes"]
    checks.require(str(Path(__file__).relative_to(REPO)) in source_hashes, "Auditor was absent from the source inventory")
    lock = manifest.get("source_lock")
    if manifest["phase"] == "development":
        checks.require(isinstance(lock, dict), "Development source freeze is missing")
        checks.require(check_current_source, "Current-source skipping is limited to preserved smoke receipts")
    if lock is not None:
        checks.require(lock["source_hashes"] == source_hashes, "Manifest/source-freeze hashes differ")
        checks.require(lock["recipe"] == manifest["recipe"], "Manifest/source-freeze recipe differs")
        checks.require(lock["development_roots"] == list(DEVELOPMENT_ROOTS), "Frozen development roster differs")
        commit = lock.get("remote_commit", "")
        checks.require(len(commit) == 40 and all(char in "0123456789abcdef" for char in commit),
                       "Published source freeze commit is malformed")
    if check_current_source:
        for relative, digest in source_hashes.items():
            path = (REPO / relative).resolve()
            checks.require(path.is_relative_to(REPO), "Frozen source path escapes the repository")
            checks.require(path.is_file() and _sha(path) == digest, "Current frozen source differs: " + relative)
    reports, rebuilt = [], []
    for root in roots:
        report, result = audit_root(directory / f"root_{root}", checks)
        checks.require(result["recipe"] == manifest["recipe"], "Root recipe differs from run manifest")
        reports.append(report)
        rebuilt.append(result)
        print(json.dumps({"audit_root": root, "verdict": "PASS", "checks_so_far": checks.count}), flush=True)
    checks.require(summarize(rebuilt) == _read(directory / "summary.json"), "Independent aggregate summary differs")
    preflight = _verify_preflight(directory, rebuilt, manifest, checks)
    completion = _read(directory / "completion.json")
    checks.require(completion["completed"] is True and completion["roots"] == roots, "Completion roster differs")
    expected_hashes = {str(root): _sha(directory / f"root_{root}" / "result.json") for root in roots}
    checks.require(completion["result_sha256"] == expected_hashes, "Completion result hashes differ")
    counters = {key: sum(report[key] for report in reports) for key in reports[0]
                if key.endswith("_replayed") or key.endswith("_reconstructed")
                or key in ("training_rows_checked", "intermediate_anchor_records_receipt_checked_only")}
    return {
        "schema_version": 1, "experiment": "BPD-RP-03", "verdict": "PASS",
        "scope": "independent saved-checkpoint hidden-state recurrence, geometry and endpoint-anchor replay; input-stream and training-receipt audit",
        "scope_limits": [
            "Optimizer updates and the full optimization trajectory are not replayed.",
            "Gradient and compute budgets are checked through arithmetic receipts and input hashes.",
            "Intermediate anchors have no saved models; their labels, prediction arithmetic and physical receipts are checked without inference replay.",
            "Shared deterministic data generation is followed by independent raw-input label, query and semantic-partition checks.",
            "The aggregate statistics module is independent of model/training code and is reused for receipt recomputation.",
        ],
        "source_hashes_checked": len(source_hashes) if check_current_source else 0,
        "current_source_check_skipped": not check_current_source,
        "n_roots": len(roots), "checks": checks.count, "scalar_tolerance": TOLERANCE,
        "maximum_scalar_comparison_error": checks.max_scalar_error,
        "prediction_bits_match": True, "aggregate_statistics_match": True,
        "preflight_input_verified": True, "preflight_output_matches": True,
        "preflight_decision": preflight["decision"],
        **counters, "roots": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-current-source", action="store_true",
                        help="Audit a preserved smoke run after a disclosed source amendment")
    args = parser.parse_args()
    try:
        result = audit_run(args.results_dir, check_current_source=not args.skip_current_source)
    except (AuditFailure, KeyError, OSError, ValueError, RuntimeError, TypeError) as error:
        result = {"experiment": "BPD-RP-03", "verdict": "FAIL", "error": str(error)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(result, sort_keys=True))
    if result["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
