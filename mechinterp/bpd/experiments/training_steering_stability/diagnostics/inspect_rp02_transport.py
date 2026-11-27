"""Post-hoc transport diagnostics on exposed RP02 B checkpoints.

This script performs inference on the original scored development pairs. It
does not train, select new groups, test a hypothesis, or provide confirmation
evidence. Full-hybrid interventions use the known counterfactual input and
are privileged structural controls. Natural doses differ across conditions.
The numerical dose 0.75 in gate space and hidden space has different physical
meaning; only each hidden-space direction and its rotated control share a
physical intervention norm and site.

Run from the repository root:
python -m mechinterp.bpd.experiments.training_steering_stability.diagnostics.inspect_rp02_transport
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from mechinterp.bpd.experiments.training_steering.data import make_pairs


HERE = Path(__file__).resolve().parent
RP02 = HERE.parents[1] / "training_steering"
REPO = HERE.parents[4]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(root: int, domain: str, index: int = 0) -> int:
    digest = hashlib.sha256(f"BPD-RP-02:{root}:{domain}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def batch_hash(batch) -> str:
    digest = hashlib.sha256()
    for name in ("receiver", "donor", "hybrid", "labels_r", "labels_d",
                 "labels_h", "edit_slot", "query"):
        digest.update(name.encode())
        digest.update(getattr(batch, name).contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def rollout(state: dict[str, Tensor], x: Tensor, *, boundary_step: int = 4,
            state_delta: Tensor | None = None,
            state_replacement: Tensor | None = None,
            write_delta: Tensor | None = None) -> dict[str, Tensor]:
    """Direct GRU evaluation with an explicit pre-token hidden intervention."""
    width = state["recurrent.V"].shape[0]
    h = x.new_zeros((len(x), width))
    encoded = F.linear(x, state["input.weight"], state["input.bias"])
    states, writes, updates, resets, before, carries = [], [], [], [], [], []
    for step in range(x.shape[1]):
        if step == boundary_step and state_replacement is not None:
            h = state_replacement
        if step == boundary_step and state_delta is not None:
            h = h + state_delta
        before.append(h)
        write = (h @ state["recurrent.V"]) @ state["recurrent.U"]
        if step == boundary_step and write_delta is not None:
            write = write + write_delta
        ir, iz, inn = encoded[:, step].chunk(3, dim=-1)
        hr, hz, hn = write.chunk(3, dim=-1)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        carry = update * h
        h = (1 - update) * candidate + carry
        states.append(h)
        writes.append(write)
        updates.append(update)
        resets.append(reset)
        carries.append(carry)
    return {"logits": F.linear(h, state["output.weight"], state["output.bias"]),
            **{name: torch.stack(value, 1) for name, value in (
                ("states", states), ("writes", writes), ("updates", updates),
                ("resets", resets), ("before", before), ("carries", carries))}}


def fixed_dose(delta: Tensor, dose: float) -> Tensor:
    length = delta.norm(dim=-1)
    return delta * torch.where(length > 1e-9, dose / length.clamp_min(1e-9),
                               torch.zeros_like(length))[:, None]


def group_projection(delta: Tensor, factor: Tensor, groups: Tensor,
                     edit_slot: Tensor) -> Tensor:
    """Project a state difference onto the selected V-column span."""
    result = torch.zeros_like(delta)
    for slot, group in enumerate(groups):
        selected = edit_slot == slot
        columns = factor[:, group]
        singular = torch.linalg.svdvals(columns)
        if float(singular.min()) <= 1e-7:
            raise ValueError("A selected V group is rank deficient")
        frame = torch.linalg.qr(columns, mode="reduced").Q
        result[selected] = (delta[selected] @ frame) @ frame.T
    return result


def norms(value: Tensor) -> dict[str, float]:
    length = value.norm(dim=-1)
    return {"mean": float(length.mean()), "min": float(length.min()),
            "max": float(length.max())}


def condition_record(out: dict[str, Tensor], clean: dict[str, Tensor],
                     hybrid: dict[str, Tensor], labels: Tensor,
                     targeted: Tensor, step: int) -> dict:
    prediction = out["logits"].argmax(-1)
    base_prediction = clean["logits"].argmax(-1)
    correct = prediction.eq(labels)
    base_correct = base_prediction.eq(labels)
    actual = out["states"][:, step] - clean["states"][:, step]
    desired = hybrid["states"][:, step] - clean["states"][:, step]
    result = {
        "predictions": prediction.tolist(),
        "cross_entropy_toward_hybrid": float(F.cross_entropy(out["logits"], labels)),
        "boundary_hidden_delta_l2": norms(out["before"][:, step] - clean["before"][:, step]),
        "first_recurrent_write_delta_l2": norms(out["writes"][:, step] - clean["writes"][:, step]),
        "first_post_token_hidden_delta_l2": norms(actual),
        "terminal_hidden_delta_l2": norms(out["states"][:, -1] - clean["states"][:, -1]),
        "first_carry_delta_l2": norms(out["carries"][:, step] - clean["carries"][:, step]),
        "post_token_delta_cosine_to_hybrid_mean": float(F.cosine_similarity(actual, desired).mean()),
    }
    for name, mask in (("targeted", targeted), ("untargeted", ~targeted)):
        result[name] = {
            "n": int(mask.sum()), "correct": int(correct[mask].sum()),
            "prediction_changes": int((prediction[mask] != base_prediction[mask]).sum()),
            "recovery_toward_hybrid": int((correct & ~base_correct & mask).sum()),
            "corruption_toward_hybrid": int((~correct & base_correct & mask).sum()),
        }
    return result


def factor_drift(warm: dict[str, Tensor], final: dict[str, Tensor], groups: Tensor) -> dict:
    old_v, new_v = warm["recurrent.V"], final["recurrent.V"]
    old_u, new_u = warm["recurrent.U"], final["recurrent.U"]
    v_cos = F.cosine_similarity(old_v, new_v, dim=0)
    u_cos = F.cosine_similarity(old_u, new_u, dim=1)
    subspace = []
    for slot, group in enumerate(groups):
        old_q = torch.linalg.qr(old_v[:, group], mode="reduced").Q
        new_q = torch.linalg.qr(new_v[:, group], mode="reduced").Q
        principal = torch.linalg.svdvals(old_q.T @ new_q)
        subspace.append({"slot": slot, "principal_cosines": principal.tolist(),
                         "mean_squared_overlap": float(principal.square().mean())})
    return {
        "V_condition_number_warmup": float(torch.linalg.cond(old_v)),
        "V_condition_number_final": float(torch.linalg.cond(new_v)),
        "V_column_cosine_mean": float(v_cos.mean()),
        "V_column_cosine_min": float(v_cos.min()),
        "U_row_cosine_mean": float(u_cos.mean()),
        "U_row_cosine_min": float(u_cos.min()),
        "rank_one_component_cosine_mean": float((v_cos * u_cos).mean()),
        "rank_one_component_cosine_min": float((v_cos * u_cos).min()),
        "selected_group_subspaces": subspace,
        "interpretation_limit": "Geometry of corresponding indices does not measure preserved slot semantics.",
    }


@torch.no_grad()
def inspect_root(directory: Path, root: int, recipe: dict) -> dict:
    root_dir = directory / f"root_{root}"
    manifest = json.loads((root_dir / "artifact_hashes.json").read_text())
    used = ("B_causal.pt", "shared_warmup.pt", "result.json")
    for name in used:
        if sha(root_dir / name) != manifest[name]:
            raise ValueError(f"Archived artifact hash mismatch: root {root}, {name}")
    record = json.loads((root_dir / "result.json").read_text())
    if record["recipe"] != recipe:
        raise ValueError("Root recipe differs from its run manifest")
    state = torch.load(root_dir / "B_causal.pt", weights_only=True, map_location="cpu")
    warm = torch.load(root_dir / "shared_warmup.pt", weights_only=True, map_location="cpu")
    batch = make_pairs(seed(root, "evaluation_causal"), recipe["causal_n"],
                       root_seed=root, delay=recipe["long_delay"], split="eval",
                       slots=recipe["slots"], values=recipe["values"], noise=recipe["noise"])
    batch_sha = batch_hash(batch)
    if batch_sha != record["arms"]["B_causal"]["causal"]["batch_sha256"]:
        raise ValueError("Diagnostic examples differ from exposed RP02 causal examples")
    groups = torch.tensor(record["attribution"]["groups"], dtype=torch.long)
    step = batch.prefix_steps
    clean = rollout(state, batch.receiver)
    donor = rollout(state, batch.donor)
    hybrid = rollout(state, batch.hybrid)
    recipient_h = clean["states"][:, step - 1]
    donor_h = donor["states"][:, step - 1]
    hybrid_h = hybrid["states"][:, step - 1]
    difference = donor_h - recipient_h
    v, u = state["recurrent.V"], state["recurrent.U"]
    activation_difference = donor_h @ v - recipient_h @ v
    mask = torch.zeros_like(activation_difference).scatter(1, groups[batch.edit_slot], 1.)
    dose = recipe["intervention_l2"]
    gate_patch = fixed_dose((activation_difference * mask) @ u, dose)
    hidden_natural = group_projection(difference, v, groups, batch.edit_slot)
    hidden_patch = fixed_dose(hidden_natural, dose)
    frozen_hidden = fixed_dose(group_projection(difference, warm["recurrent.V"],
                                                groups, batch.edit_slot), dose)
    generator = torch.Generator().manual_seed(seed(root, "posthoc_hidden_rotation"))
    permutation = torch.randperm(v.shape[0], generator=generator)
    signs = torch.randint(0, 2, (v.shape[0],), generator=generator).float() * 2 - 1
    rotated_hidden = hidden_patch[:, permutation] * signs
    full_gate = (hybrid_h @ v) @ u - (recipient_h @ v) @ u
    outputs = {
        "unchanged": clean,
        "old_group_gate_l2_075": rollout(state, batch.receiver, write_delta=gate_patch),
        "current_Vgroup_hidden_l2_075": rollout(state, batch.receiver, state_delta=hidden_patch),
        "current_Vgroup_hidden_l2_075_rotated": rollout(state, batch.receiver, state_delta=rotated_hidden),
        "warmup_Vgroup_hidden_l2_075": rollout(state, batch.receiver, state_delta=frozen_hidden),
        "current_Vgroup_hidden_natural": rollout(state, batch.receiver, state_delta=hidden_natural),
        "privileged_hybrid_full_state": rollout(state, batch.receiver, state_replacement=hybrid_h),
        "full_donor_state": rollout(state, batch.receiver, state_replacement=donor_h),
        "privileged_hybrid_full_gate": rollout(state, batch.receiver, write_delta=full_gate),
        "hybrid_native": hybrid,
    }
    old_predictions = outputs["old_group_gate_l2_075"]["logits"].argmax(-1).tolist()
    if old_predictions != record["arms"]["B_causal"]["causal"]["predictions"]:
        raise ValueError("Original RP02 causal predictions fail replay")
    hybrid_error = float((outputs["privileged_hybrid_full_state"]["logits"] - hybrid["logits"]).abs().max())
    if hybrid_error > 1e-5:
        raise ValueError("Full hybrid state does not reproduce the native hybrid continuation")
    # With hybrid gates and recipient carry, h_after differs from the native
    # hybrid state by exactly z_hybrid * (h_recipient - h_hybrid), up to rounding.
    gate_only_state = outputs["privileged_hybrid_full_gate"]["states"][:, step]
    predicted_carry_gap = hybrid["updates"][:, step] * (recipient_h - hybrid_h)
    carry_identity_error = float((gate_only_state - hybrid["states"][:, step]
                                  - predicted_carry_gap).abs().max())
    if carry_identity_error > 1e-5:
        raise ValueError("The gate-only versus full-state carry identity fails")
    targeted = batch.query == batch.edit_slot
    return {
        "root": root, "arm": "B_causal", "batch_sha256": batch_sha,
        "source_artifacts_sha256": {name: manifest[name] for name in used},
        "groups": groups.tolist(), "unique_selected_groups": len({tuple(g) for g in groups.tolist()}),
        "labels_hybrid": batch.labels_h.tolist(), "targeted": targeted.tolist(),
        "hidden_rotation": {"permutation": permutation.tolist(), "signs": signs.tolist()},
        "checks": {
            "original_causal_predictions_exact": True,
            "hybrid_state_vs_native_logit_max_error": hybrid_error,
            "gate_only_carry_identity_max_error": carry_identity_error,
            "hidden_rotation_norm_max_error": float((hidden_patch.norm(dim=-1) - rotated_hidden.norm(dim=-1)).abs().max()),
            "hidden_dose_max_error": float((hidden_patch.norm(dim=-1) - dose).abs().max()),
            "gate_dose_max_error": float((gate_patch.norm(dim=-1) - dose).abs().max()),
        },
        "clean_first_NOOP_update_mean": float(clean["updates"][:, step].mean()),
        "clean_first_NOOP_update_above_095_fraction": float((clean["updates"][:, step] > .95).float().mean()),
        "clean_first_NOOP_reset_mean": float(clean["resets"][:, step].mean()),
        "gate_only_missing_carry_l2": norms(predicted_carry_gap),
        "factor_drift_after_warmup": factor_drift(warm, state, groups),
        "conditions": {name: condition_record(out, clean, hybrid, batch.labels_h, targeted, step)
                       for name, out in outputs.items()},
    }


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for condition in rows[0]["conditions"]:
        records = [row["conditions"][condition] for row in rows]
        result[condition] = {
            name: {field: sum(record[name][field] for record in records)
                   for field in records[0][name]}
            for name in ("targeted", "untargeted")
        }
        for name in ("boundary_hidden_delta_l2", "first_recurrent_write_delta_l2",
                     "first_post_token_hidden_delta_l2", "terminal_hidden_delta_l2",
                     "first_carry_delta_l2"):
            result[condition][name + "_mean"] = sum(record[name]["mean"] for record in records) / len(rows)
        result[condition]["post_token_delta_cosine_to_hybrid_mean"] = sum(
            record["post_token_delta_cosine_to_hybrid_mean"] for record in records) / len(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RP02 / "results" / "development")
    parser.add_argument("--out", type=Path, default=HERE / "rp02_transport_receipt.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest_path = args.results / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["phase"] != "development":
        raise ValueError("Only already exposed RP02 development receipts are supported")
    data_path = RP02 / "data.py"
    data_relative = str(data_path.relative_to(REPO))
    if sha(data_path) != manifest["source_hashes"][data_relative]:
        raise ValueError("The deterministic input generator differs from the RP02 source freeze")
    rows = [inspect_root(args.results, root, manifest["recipe"]) for root in manifest["roots"]]
    receipt = {
        "evidence_class": "POSTHOC_EXPOSED_CHECKPOINT_DIAGNOSTIC_ONLY",
        "training_models_created": 0,
        "hypothesis_tests": [],
        "scope": "Original scored RP02 causal examples, B_causal checkpoints, fixed original groups; every diagnostic condition is retained.",
        "limitations": [
            "This reuse can guide RP03 engineering and cannot estimate confirmation performance.",
            "Hybrid controls use privileged counterfactual prefixes; natural full-state and gate doses are unmatched.",
            "A numerical dose of 0.75 in hidden space differs physically from 0.75 in gate space.",
            "The hidden rotation supplies a negative control at the same intervention site and dose.",
            "The old gate-selected groups were reused without selecting new hidden-space groups.",
            "Geometric factor stability does not establish stable semantic slot meaning.",
        ],
        "source": {"script_sha256": sha(Path(__file__)), "data_sha256": sha(data_path),
                   "run_manifest_sha256": sha(manifest_path),
                   "RP02_source_freeze": manifest["source_lock"]["remote_commit"],
                   "torch": torch.__version__},
        "recipe": manifest["recipe"], "roots": manifest["roots"],
        "aggregate": aggregate(rows), "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "roots": len(rows),
                      "aggregate": receipt["aggregate"],
                      "maximum_hybrid_logit_error": max(row["checks"]["hybrid_state_vs_native_logit_max_error"] for row in rows),
                      "maximum_carry_identity_error": max(row["checks"]["gate_only_carry_identity_max_error"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
