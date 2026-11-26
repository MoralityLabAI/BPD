"""Independent recurrence and factor-level replay for RP04.

The audit never imports the diagnostic runner or intervention implementation.
Inputs may be regenerated with the shared deterministic sampler; their raw
encodings, labels and query expansion are then decoded independently. Native
BPD inference is reconstructed directly from the saved model tensors.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import time

import torch
from torch import Tensor
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DIRECTIONS = ("donor_span", "hybrid_span", "hybrid_live_span", "donor_dual", "hybrid_dual", "hybrid_full")
ROUTES = ("full", "carry_only", "all_write_only", "selected_write_only", "outside_write_only", "carry_selected", "carry_outside")
CHECKPOINTS = ("shared_warmup", "A_task", "B_causal_fixed", "C_rotated_fixed", "D_causal_live")
TOLERANCE = 2e-5


class AuditFailure(ValueError):
    pass


@dataclass
class Checks:
    count: int = 0
    maximum_scalar_error: float = 0.0
    maximum_tensor_error: float = 0.0

    def require(self, condition, context: str):
        self.count += 1
        if not bool(condition):
            raise AuditFailure(context)

    def close(self, actual, expected, context: str, tolerance: float = TOLERANCE):
        actual, expected = float(actual), float(expected)
        error = abs(actual - expected)
        self.require(math.isfinite(actual) and math.isfinite(expected) and error <= tolerance,
                     f"{context}: actual={actual}, expected={expected}, error={error}")
        self.maximum_scalar_error = max(self.maximum_scalar_error, error)

    def tensor(self, actual: Tensor, expected: Tensor, context: str, tolerance: float = TOLERANCE):
        self.require(actual.shape == expected.shape, context + ": shape")
        self.require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), context + ": finite")
        error = float((actual.double() - expected.double()).abs().max()) if actual.numel() else 0.0
        self.close(error, 0., context, tolerance)
        self.maximum_tensor_error = max(self.maximum_tensor_error, error)


def read(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_for(groups: Tensor, edit_slot: Tensor, components: int, dtype, device):
    selected = groups.to(device)[edit_slot.to(device)]
    return torch.zeros((len(edit_slot), components), dtype=dtype, device=device).scatter_(1, selected, 1.)


@torch.no_grad()
def manual_rollout(state: dict[str, Tensor], x: Tensor, delta: Tensor | None = None,
                   groups: Tensor | None = None, edit_slot: Tensor | None = None,
                   route: str = "full", patch_step: int = 4,
                   state_replacement: Tensor | None = None) -> dict[str, Tensor]:
    """Execute the GRU from tensor equations with explicit carry/write paths."""
    if route not in ROUTES:
        raise AuditFailure("Unknown route")
    if delta is not None and state_replacement is not None:
        raise AuditFailure("Replacement and additive delta are exclusive")
    V, U = state["recurrent.V"], state["recurrent.U"]
    width = V.shape[0]
    if delta is not None and delta.shape != (len(x), width):
        raise AuditFailure("Delta shape differs")
    if delta is not None and route in ("selected_write_only", "outside_write_only", "carry_selected", "carry_outside"):
        if groups is None or edit_slot is None:
            raise AuditFailure("Grouped route needs component groups")
        mask = mask_for(groups, edit_slot, V.shape[1], V.dtype, V.device)
    else:
        mask = None
    encoded = F.linear(x, state["input.weight"], state["input.bias"])
    h = x.new_zeros((len(x), width))
    states, incoming, writes = [], [], []
    for step in range(x.shape[1]):
        baseline = h
        if step == patch_step and state_replacement is not None:
            h = state_replacement
            baseline = h
        carry = h
        if step == patch_step and delta is not None:
            carry_changed = route in ("full", "carry_only", "carry_selected", "carry_outside")
            carry = h + delta if carry_changed else h
            if route == "full":
                write = (carry @ V) @ U
            else:
                # Reproduce the native output-hook arithmetic in float32:
                # the recurrent write is evaluated at the actual carry, then
                # a route-specific correction is applied with the same order.
                write = (carry @ V) @ U
                dreads = delta @ V
                total = dreads @ U
                if route == "carry_only":
                    write = write - total
                elif route == "all_write_only":
                    write = write + total
                elif route == "selected_write_only":
                    write = write + (dreads * mask) @ U
                elif route == "outside_write_only":
                    write = write + (dreads * (1 - mask)) @ U
                elif route == "carry_selected":
                    write = write - (dreads * (1 - mask)) @ U
                elif route == "carry_outside":
                    write = write - (dreads * mask) @ U
        else:
            write = (h @ V) @ U
        incoming.append(carry)
        ir, iz, inn = encoded[:, step].chunk(3, dim=-1)
        hr, hz, hn = write.chunk(3, dim=-1)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        h = (1 - update) * candidate + update * carry
        states.append(h)
        writes.append(write)
    return {"logits": F.linear(h, state["output.weight"], state["output.bias"]),
            "states": torch.stack(states, 1), "incoming_states": torch.stack(incoming, 1),
            "recurrent_writes": torch.stack(writes, 1)}


@torch.no_grad()
def manual_projectors(V: Tensor, groups: Tensor) -> Tensor:
    out = []
    for group in groups:
        columns = V[:, group]
        frame = torch.linalg.qr(columns, mode="reduced").Q
        out.append(frame @ frame.T)
    return torch.stack(out)


def normalize(raw: Tensor, dose: float):
    length = torch.linalg.vector_norm(raw, dim=-1)
    valid = length > 1e-9
    return raw * torch.where(valid, dose / length.clamp_min(1e-9), torch.zeros_like(length))[:, None], valid


@torch.no_grad()
def manual_directions(receiver: Tensor, donor: Tensor, hybrid: Tensor,
                      V: Tensor, groups: Tensor, fixed: Tensor,
                      edit_slot: Tensor, dose: float = .75):
    """Construct all declared edits from independently written linear algebra."""
    d_donor, d_hybrid = donor.double() - receiver.double(), hybrid.double() - receiver.double()
    warm = fixed.double()[edit_slot]
    current = manual_projectors(V.double(), groups)[edit_slot]
    project = lambda x, p: torch.einsum("bi,bij->bj", x, p)
    mask = mask_for(groups, edit_slot, V.shape[1], torch.float64, V.device)
    raw = {
        "donor_span": project(d_donor, warm),
        "hybrid_span": project(d_hybrid, warm),
        "hybrid_live_span": project(d_hybrid, current),
        "hybrid_full": d_hybrid,
    }
    V64 = V.double()
    # Solve the transposed system. GELSD computes the Moore-Penrose solution
    # with the registered relative singular-value cutoff.
    for name, difference in (("donor_dual", d_donor), ("hybrid_dual", d_hybrid)):
        desired = (difference.double() @ V64) * mask
        solution = torch.linalg.lstsq(V64.T, desired.T, rcond=1e-10, driver="gelsd").solution.T
        raw[name] = solution
    result, valid = {}, {}
    for name in DIRECTIONS:
        value, valid[name] = normalize(raw[name], dose)
        result[name] = value.to(receiver.dtype)
    return result, raw, valid


@torch.no_grad()
def manual_physical_metrics(delta: Tensor, V: Tensor, U: Tensor, groups: Tensor, edit_slot: Tensor):
    d = delta.double()
    V64, U64 = V.double(), U.double()
    reads = d @ V64
    mask = mask_for(groups, edit_slot, V.shape[1], torch.float64, V.device)
    group = (reads * mask) @ U64
    outside = (reads * (1 - mask)) @ U64
    atoms = reads[:, :, None] * U64[None]
    energies = atoms.square().sum(-1)
    group2, outside2 = group.square().sum(-1), outside.square().sum(-1)
    gnorm, onorm = group2.sqrt(), outside2.sqrt()
    eps = 1e-30
    return {
        "delivered_norm": d.norm(dim=-1),
        "selected_write_norm": gnorm,
        "outside_write_norm": onorm,
        "grouped_write_leakage": outside2 / (group2 + outside2).clamp_min(eps),
        "grouped_write_cosine": (group * outside).sum(-1) / (gnorm * onorm).clamp_min(eps),
        "cancellation": 1 - (group + outside).norm(dim=-1) / (gnorm + onorm).clamp_min(eps),
        "atom_energy_leakage": (energies * (1 - mask)).sum(-1) / energies.sum(-1).clamp_min(eps),
        "atom_energy_by_component": energies,
        "atom_sum_error": (atoms.sum(1) - d @ V64 @ U64).abs().max(),
        "group_sum_error": (group + outside - d @ V64 @ U64).abs().max(),
    }


@torch.no_grad()
def manual_gate_identity(state: dict[str, Tensor], x: Tensor, delta: Tensor, patch_step: int = 4):
    """Verify the exact finite difference across carry and gate-response paths."""
    clean = manual_rollout(state, x)
    h = clean["incoming_states"][:, patch_step]
    V, U = state["recurrent.V"], state["recurrent.U"]
    w, wp = (h @ V) @ U, ((h + delta) @ V) @ U
    encoded = F.linear(x[:, patch_step], state["input.weight"], state["input.bias"])
    ir, iz, inn = encoded.chunk(3, -1)
    def gates(write):
        wr, wz, wn = write.chunk(3, -1)
        reset = torch.sigmoid(ir + wr)
        update = torch.sigmoid(iz + wz)
        candidate = torch.tanh(inn + reset * wn)
        return update, candidate
    z, n = gates(w)
    zp, np = gates(wp)
    f00 = (1-z) * n + z * h
    f10 = (1-z) * n + z * (h+delta)
    f01 = (1-zp) * np + zp * h
    f11 = (1-zp) * np + zp * (h+delta)
    carry = zp * delta
    update = (zp-z) * (h-n)
    candidate = (1-zp) * (np-n)
    return {
        "finite_decomposition_error": float(((f11-f00) - carry-update-candidate).abs().max()),
        "carry_only_identity_error": float(((f10-f00) - z*delta).abs().max()),
        "interaction_identity_error": float(((f11-f10-f01+f00) - (zp-z)*delta).abs().max()),
        "carry_term_norm": carry.norm(dim=-1),
        "update_term_norm": update.norm(dim=-1),
        "candidate_term_norm": candidate.norm(dim=-1),
        "interaction_norm": ((zp-z)*delta).norm(dim=-1),
    }


def semantic_checks(base, expanded, root: int, checks: Checks):
    """Decode raw features without relying on the sampler's semantic helpers."""
    slots = base.values_r.shape[1]
    values = base.receiver.shape[-1] - 3 - slots
    n = len(base.query)
    checks.require(len(expanded.query) == slots*n, "Every base pair must have every query")
    checks.require(torch.equal(expanded.query, torch.arange(slots).repeat(n)), "Query expansion order differs")
    checks.require(torch.equal(expanded.edit_slot, base.edit_slot.repeat_interleave(slots)), "Edited slot expansion differs")
    for slot in range(slots):
        checks.require(int((base.edit_slot == slot).sum()) == n // slots, "Base edit slots are unbalanced")
    generator = torch.Generator().manual_seed((root * 104729 + 9029) % (2**63 - 1))
    address_encoding = torch.randperm(slots, generator=generator)
    value_encoding = torch.randperm(values, generator=generator)
    inverse_address, inverse_value = address_encoding.argsort(), value_encoding.argsort()
    for batch, label in ((base, "base"), (expanded, "expanded")):
        rows = len(batch.query)
        checks.require(batch.prefix_steps == slots, label + ": prefix length")
        for feature, memory_name, label_name in (("receiver", "values_r", "labels_r"),
                ("donor", "values_d", "labels_d"), ("hybrid", "values_h", "labels_h")):
            x, memory, truth = (getattr(batch, name) for name in (feature, memory_name, label_name))
            checks.require(bool(torch.isfinite(x).all()), label + ": finite raw input")
            types = torch.zeros_like(x[:, :, :3])
            types[:, :slots, 0] = 1
            types[:, slots:-1, 1] = 1
            types[:, -1, 2] = 1
            checks.require(torch.equal(types, x[:, :, :3]), label + ": operation types")
            address = inverse_address[x[:, :slots, 3:3+slots].argmax(-1)]
            written = inverse_value[x[:, :slots, 3+slots:].argmax(-1)]
            checks.require(torch.equal(address.sort(1).values, torch.arange(slots).expand(rows, -1)), label + ": each address written once")
            reconstructed = torch.empty_like(memory).scatter_(1, address, written)
            checks.require(torch.equal(reconstructed, memory), label + ": memory decoded from features")
            query = inverse_address[x[:, -1, 3:3+slots].argmax(-1)]
            checks.require(torch.equal(query, batch.query), label + ": query decoded from features")
            checks.require(torch.equal(truth, reconstructed[torch.arange(rows), query]), label + ": label decoded from features")
            code = sum(reconstructed[:, slot] * values**slot for slot in range(slots))
            checks.require(bool((code.remainder(5) <= 1).all()), label + ": evaluation semantic partition")
        expected_hybrid = batch.values_r.clone()
        expected_hybrid[torch.arange(rows), batch.edit_slot] = batch.values_d[torch.arange(rows), batch.edit_slot]
        checks.require(torch.equal(expected_hybrid, batch.values_h), label + ": one-slot hybrid value")
        checks.require(bool((batch.values_r != batch.values_h).sum(1).eq(1).all()), label + ": exactly one changed slot")
        checks.require(torch.equal(batch.receiver[:, slots:], batch.donor[:, slots:]) and
                       torch.equal(batch.receiver[:, slots:], batch.hybrid[:, slots:]), label + ": shared pair suffix")
    for feature in ("receiver", "donor", "hybrid"):
        original, all_queries = getattr(base, feature), getattr(expanded, feature)
        checks.require(torch.equal(original[:, :-1].repeat_interleave(slots, 0), all_queries[:, :-1]), "Expanded prefix/distractors changed")
        checks.require(torch.equal(original[:, -1, 3+slots:].repeat_interleave(slots, 0), all_queries[:, -1, 3+slots:]), "Expanded query value noise changed")
        old_noise = original[:, -1, 3:3+slots].clone()
        old_noise[torch.arange(n), address_encoding[base.query]] -= 1
        new_noise = all_queries[:, -1, 3:3+slots].clone()
        new_noise[torch.arange(n*slots), address_encoding[expanded.query]] -= 1
        checks.tensor(new_noise, old_noise.repeat_interleave(slots, 0), "Expanded query address noise", 2e-7)
    for name in ("values_r", "values_d", "values_h"):
        checks.require(torch.equal(getattr(expanded, name), getattr(base, name).repeat_interleave(slots, 0)), "Expanded memory metadata changed")


def nested_close(actual, expected, checks: Checks, path: str):
    """Compare serialized arithmetic receipts recursively with exact structure."""
    if isinstance(expected, dict):
        checks.require(isinstance(actual, dict) and set(actual) == set(expected), path + ": keys")
        for key in expected:
            nested_close(actual[key], expected[key], checks, path + "/" + key)
    elif isinstance(expected, list):
        checks.require(isinstance(actual, list) and len(actual) == len(expected), path + ": list length")
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            nested_close(a, b, checks, path + f"/{i}")
    elif isinstance(expected, float):
        checks.close(actual, expected, path)
    else:
        checks.require(type(actual) is type(expected) and actual == expected, path + ": exact value")


def nullable_ratio(a: Tensor, b: Tensor, threshold: float = 1e-9):
    return [float(x / y) if float(y) > threshold else None for x, y in zip(a, b, strict=True)]


def cosine_list(a: Tensor, b: Tensor):
    return nullable_ratio((a*b).sum(-1), a.norm(dim=-1)*b.norm(dim=-1), 1e-18)


@torch.no_grad()
def manual_direction_record(delta, V, U, groups, edit_slot):
    d, v, u = delta.double(), V.double(), U.double()
    mask = mask_for(groups, edit_slot, V.shape[1], torch.float64, V.device)
    reads = d@v
    atoms = reads[:, :, None]*u[None]
    energy = atoms.square().sum(-1)
    g, o = (reads*mask)@u, (reads*(1-mask))@u
    total = reads@u
    gn, on = g.norm(dim=-1), o.norm(dim=-1)
    return {
        "delivered_norm": d.norm(dim=-1).tolist(),
        "selected_write_norm": gn.tolist(), "outside_write_norm": on.tolist(),
        "total_write_norm": total.norm(dim=-1).tolist(),
        "grouped_write_leakage": nullable_ratio(on.square(), gn.square()+on.square(), 1e-18),
        "grouped_write_cosine": cosine_list(g, o),
        "cancellation": [None if value is None else 1-value for value in
                         nullable_ratio(total.norm(dim=-1), gn+on)],
        "atom_energy_leakage": nullable_ratio((energy*(1-mask)).sum(-1), energy.sum(-1), 1e-18),
        "atom_energy_by_component_mean": energy.mean(0).tolist(),
        "atom_energy_by_slot_and_component_mean": [energy[edit_slot==slot].mean(0).tolist()
            if bool((edit_slot==slot).any()) else None for slot in range(len(groups))],
        "selected_read_norm_gauge_dependent": (reads*mask).norm(dim=-1).tolist(),
        "outside_read_norm_gauge_dependent": (reads*(1-mask)).norm(dim=-1).tolist(),
        "dense_write_max_abs_error": float((d@(v@u)-total).abs().max()),
        "group_write_sum_max_abs_error": float((g+o-total).abs().max()),
    }


@torch.no_grad()
def manual_diagnostic_record(receiver, donor, hybrid, V, U, groups, fixed, edit_slot, dose=.75):
    deltas, raw, valid = manual_directions(receiver, donor, hybrid, V, groups, fixed, edit_slot, dose)
    v, u = V.double(), U.double()
    differences = {"donor": donor.double()-receiver.double(), "hybrid": hybrid.double()-receiver.double()}
    live_all = manual_projectors(v, groups)
    live = live_all[edit_slot]
    projected = {key: torch.einsum("bi,bij->bj", value, live) for key, value in differences.items()}
    mask = mask_for(groups, edit_slot, V.shape[1], torch.float64, V.device)
    desired = {key: (value@v)*mask for key, value in differences.items()}
    records = {}
    for name, delta in deltas.items():
        source = "donor" if name.startswith("donor") else "hybrid"
        raw_norm = raw[name].norm(dim=-1)
        scale = torch.where(valid[name], dose/raw_norm.clamp_min(1e-9), torch.zeros_like(raw_norm))
        d = delta.double()
        record = manual_direction_record(delta, V, U, groups, edit_slot)
        record.update({
            "raw_norm": raw_norm.tolist(), "valid": valid[name].tolist(),
            "normalization_scale": scale.tolist(),
            "raw_amplification": nullable_ratio(raw_norm, differences[source].norm(dim=-1)),
        })
        if name.endswith("dual"):
            target = desired[source]
            residual = (raw[name]@v-target).norm(dim=-1)
            scaled_residual = (d@v-target*scale[:, None]).norm(dim=-1)
            span_norm = projected[source].norm(dim=-1)
            perpendicular_norm = (raw[name]-projected[source]).norm(dim=-1)
            pyth = raw_norm.square()-span_norm.square()-perpendicular_norm.square()
            record.update({
                "desired_read_norm": target.norm(dim=-1).tolist(),
                "raw_read_residual_relative": nullable_ratio(residual, target.norm(dim=-1)),
                "raw_read_residual_absolute": residual.tolist(),
                "normalized_read_residual_relative": nullable_ratio(scaled_residual, (target*scale[:,None]).norm(dim=-1)),
                "normalized_read_residual_absolute": scaled_residual.tolist(),
                "normalized_read_scale": scale.tolist(),
                "raw_live_span_norm": span_norm.tolist(),
                "dual_to_live_span_norm_ratio": nullable_ratio(raw_norm, span_norm),
                "dual_minus_live_span_norm": perpendicular_norm.tolist(),
                "pythagorean_absolute_residual": pyth.abs().tolist(),
                "pythagorean_relative_residual": nullable_ratio(pyth.abs(), raw_norm.square(), 1e-18),
                "selected_read_amplitude_vs_live_span": nullable_ratio(span_norm, raw_norm),
            })
        records[name] = record
    singular = torch.linalg.svdvals(v)
    rank = int((singular>singular.max()*1e-10).sum())
    group_ranks = []
    write_ranks = []
    write_projectors = []
    for group in groups:
        s = torch.linalg.svdvals(v[:,group])
        group_ranks.append(int((s>s.max()*1e-10).sum()))
        _, write_s, vh = torch.linalg.svd(u[group], full_matrices=False)
        write_rank = int((write_s > write_s.max()*1e-10).sum())
        write_ranks.append(write_rank)
        write_frame = vh[:write_rank].T
        write_projectors.append(write_frame@write_frame.T)
    write_projectors = torch.stack(write_projectors)[edit_slot]
    hybrid_full_write = (differences["hybrid"]@v)@u
    selected_write_projection = torch.einsum("bi,bij->bj", hybrid_full_write, write_projectors)
    return deltas, {
        "factor": {"shape": list(v.shape), "rank": rank, "pinv_rtol": 1e-10,
                   "singular_values": singular.tolist(),
                   "condition_number": float(singular.max()/singular.min()) if rank==min(v.shape) else None,
                   "selected_group_ranks": group_ranks,
                   "selected_U_group_ranks": write_ranks},
        "geometry": {
            "hybrid_difference_norm": differences["hybrid"].norm(dim=-1).tolist(),
            "donor_difference_norm": differences["donor"].norm(dim=-1).tolist(),
            "hybrid_fixed_projection_residual": nullable_ratio((differences["hybrid"]-raw["hybrid_span"]).norm(dim=-1), differences["hybrid"].norm(dim=-1)),
            "hybrid_live_projection_residual": nullable_ratio((differences["hybrid"]-raw["hybrid_live_span"]).norm(dim=-1), differences["hybrid"].norm(dim=-1)),
            "donor_hybrid_projected_cosine": cosine_list(raw["donor_span"], raw["hybrid_span"]),
            "projector_chordal_by_slot": ((live_all-fixed.double()).square().sum((-1,-2))/(2*groups.shape[1])).sqrt().tolist(),
            "hybrid_full_write_norm": hybrid_full_write.norm(dim=-1).tolist(),
            "hybrid_selected_U_projection_residual": nullable_ratio((hybrid_full_write-selected_write_projection).norm(dim=-1), hybrid_full_write.norm(dim=-1)),
        },
        "directions": records,
    }


@torch.no_grad()
def manual_pathway_record(state, x, delta, patch_step=4):
    clean = manual_rollout(state, x)
    full = manual_rollout(state, x, delta, patch_step=patch_step)
    h = clean["incoming_states"][:, patch_step]
    w, wp = clean["recurrent_writes"][:, patch_step], full["recurrent_writes"][:, patch_step]
    ir, iz, inn = F.linear(x[:,patch_step], state["input.weight"], state["input.bias"]).chunk(3,-1)
    def gates(write):
        wr,wz,wn=write.chunk(3,-1)
        r=torch.sigmoid(ir+wr)
        z=torch.sigmoid(iz+wz)
        n=torch.tanh(inn+r*wn)
        return r,z,n
    r,z,n=gates(w); rp,zp,np=gates(wp)
    f00=(1-z)*n+z*h
    f10=(1-z)*n+z*(h+delta)
    f01=(1-zp)*np+zp*h
    f11=(1-zp)*np+zp*(h+delta)
    carry=zp*delta; update=(zp-z)*(h-n); candidate=(1-zp)*(np-n)
    gate=update+candidate; interaction=(zp-z)*delta
    actual=full["states"][:,patch_step]-clean["states"][:,patch_step]
    write_prediction=(delta@state["recurrent.V"])@state["recurrent.U"]
    norm=lambda t:t.double().norm(dim=-1).tolist()
    return {
        "state_delta_norm":norm(actual),"carry_term_norm":norm(carry),
        "update_gate_term_norm":norm(update),"candidate_term_norm":norm(candidate),
        "gate_term_norm":norm(gate),"carry_gate_cosine":cosine_list(carry.double(),gate.double()),
        "carry_only_state_delta_norm":norm(f10-f00),"write_only_state_delta_norm":norm(f01-f00),
        "interaction_norm":norm(interaction),"reset_gate_delta_norm":norm(rp-r),
        "update_gate_delta_norm":norm(zp-z),"candidate_delta_norm":norm(np-n),
        "write_delta_norm":norm(wp-w),"final_all_head_delta_norm":norm(full["logits"]-clean["logits"]),
        "decomposition_max_abs_error":float((actual-carry-update-candidate).abs().max()),
        "interaction_max_abs_error":float((f11-f10-f01+f00-interaction).abs().max()),
        "carry_only_identity_max_abs_error":float((f10-f00-z*delta).abs().max()),
        "native_clean_state_max_abs_error":float((f00-clean["states"][:,patch_step]).abs().max()),
        "native_full_state_max_abs_error":float((f11-full["states"][:,patch_step]).abs().max()),
        "linear_write_max_abs_error":float((wp-w-write_prediction).abs().max()),
    }


def batch_fingerprint(batch):
    digest = hashlib.sha256()
    for field in fields(batch):
        value = getattr(batch, field.name)
        digest.update(field.name.encode("ascii") + b"\0")
        if isinstance(value, Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        else:
            digest.update(str(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def manual_condition_record(run, ordinary, batch, step):
    logits = run["logits"]
    predictions = logits.argmax(-1)
    log_probabilities = logits.log_softmax(-1)
    rows = torch.arange(len(logits))
    return {
        "predictions": predictions.tolist(),
        "correct": (predictions == batch.labels_h).int().tolist(),
        "cross_entropy_hybrid_mean": float(-log_probabilities[rows, batch.labels_h].mean()),
        "cross_entropy_receiver_mean": float(-log_probabilities[rows, batch.labels_r].mean()),
        "logits_abs_max": float(logits.abs().max()),
        "physical": {
            "incoming_state_delta_l2_mean": float((run["incoming_states"][:,step]-ordinary["states"][:,step-1]).norm(dim=-1).mean()),
            "recurrent_write_delta_l2_mean": float((run["recurrent_writes"][:,step]-ordinary["recurrent_writes"][:,step]).norm(dim=-1).mean()),
            "next_state_delta_l2_mean": float((run["states"][:,step]-ordinary["states"][:,step]).norm(dim=-1).mean()),
            "terminal_state_delta_l2_mean": float((run["states"][:,-1]-ordinary["states"][:,-1]).norm(dim=-1).mean()),
            "all_head_change_max": float((logits-ordinary["logits"]).abs().max()),
            "prefix_state_max_error": float((run["states"][:,:step]-ordinary["states"][:,:step]).abs().max()),
            "incoming_state_rms": float(run["incoming_states"][:,step].square().mean().sqrt()),
        },
    }


@torch.no_grad()
def algebra_checks(V, U, delta, groups, edit_slot, checks):
    """Check saved-factor identities under deterministic changes of basis."""
    checks.require(all(len(group.unique()) == len(group) for group in groups),
                   "Repeated component index within a frozen group")
    v, u, d = V.double(), U.double(), delta.double()
    mask = mask_for(groups, edit_slot, V.shape[1], torch.float64, V.device)
    reads = d@v
    atoms = reads[:,:,None]*u[None]
    total = reads@u
    selected, outside = (reads*mask)@u, (reads*(1-mask))@u
    checks.tensor(atoms.sum(1), total, "Atomic writes reconstruct recurrent write", 1e-10)
    checks.tensor(selected+outside, total, "Selected and outside writes sum", 1e-10)
    checks.tensor(d@(v@u), total, "Dense and factorized writes agree", 1e-10)
    scale = torch.linspace(.5, 1.5, v.shape[1], dtype=torch.float64)
    scaled_atoms = (d@(v*scale))[:,:,None]*(u/scale[:,None])[None]
    checks.tensor(scaled_atoms, atoms, "Inverse factor scaling preserves physical atoms", 1e-10)
    transform = torch.eye(v.shape[1], dtype=torch.float64)
    # Slots may share some or all selected components. Mix components that
    # have identical membership across slots, so this change of basis
    # commutes with every selected/outside mask even when groups overlap.
    memberships = torch.zeros((v.shape[1], len(groups)), dtype=torch.bool)
    for slot, group in enumerate(groups):
        memberships[group, slot] = True
    classes = {}
    for component, membership in enumerate(memberships.tolist()):
        classes.setdefault(tuple(membership), []).append(component)
    for members in classes.values():
        indices = torch.tensor(members, dtype=torch.long)
        transform[indices[:-1], indices[1:]] = .2
    vm = v@transform
    um = torch.linalg.solve(transform, u)
    checks.tensor((d@vm*mask)@um, selected, "Selected block mixing gauge", 1e-10)
    checks.tensor((d@vm*(1-mask))@um, outside, "Outside block mixing gauge", 1e-10)
    checks.tensor(manual_projectors(vm, groups), manual_projectors(v, groups), "Selected span basis invariance", 1e-10)


def check_hash_inventory(base, inventory, checks, context):
    checks.require(isinstance(inventory, dict) and bool(inventory), context+": empty inventory")
    for relative, expected in inventory.items():
        path = (base/relative).resolve()
        checks.require(path.is_relative_to(base.resolve()), context+": path escapes its root")
        checks.require(isinstance(expected, str) and len(expected)==64, context+": malformed hash")
        checks.require(path.is_file() and sha256(path)==expected, context+": hash differs for "+relative)


@torch.no_grad()
def audit_snapshot(path, root, checkpoint, phase, recipe, base, batch, metadata, geometry, model_path, checks):
    from .common import CONDITIONS, PATH_DIRECTIONS
    saved = read(path)
    for key, value in {"schema_version":1,"root":root,"phase":phase,"checkpoint":checkpoint,
                       "checkpoint_sha256":sha256(model_path),"batch_sha256":metadata["batch_sha256"]}.items():
        checks.require(saved[key]==value, f"{checkpoint}: {key} differs")
    state = torch.load(model_path, weights_only=True, map_location="cpu")
    V, U = state["recurrent.V"], state["recurrent.U"]
    checks.require(V.shape==(32,32) and U.shape==(32,96), "Frozen factor shape differs")
    checks.require(all(bool(torch.isfinite(x).all()) for x in state.values()), "Checkpoint has nonfinite tensor")
    parameter_names = ("input.weight","input.bias","output.weight","output.bias","recurrent.V","recurrent.U")
    checks.require(saved["parameter_count"]==sum(state[k].numel() for k in parameter_names), "Parameter count differs")
    groups, fixed = geometry["groups"], geometry["fixed_projectors"]
    checks.require(saved["groups"]==groups.tolist(), "Snapshot component assignment differs")
    checks.require(set(saved["conditions"])==set(CONDITIONS), "Intervention inventory differs")
    n, step = len(batch.query), batch.prefix_steps
    all_clean = manual_rollout(state, torch.cat((batch.receiver,batch.donor,batch.hybrid)))
    clean = {k:v[:n] for k,v in all_clean.items()}
    states = [all_clean["states"][i*n:(i+1)*n,step-1][::recipe.slots] for i in range(3)]
    for i in range(3):
        checks.tensor(states[i].repeat_interleave(recipe.slots,0), all_clean["states"][i*n:(i+1)*n,step-1], "All-query prefix states agree", 1e-7)
    directions, diagnostics = manual_diagnostic_record(*states,V,U,groups,fixed,base.edit_slot,recipe.dose)
    nested_close(saved["direction_diagnostics"], diagnostics, checks, checkpoint+"/direction_diagnostics")
    checks.require(set(saved["rotated_diagnostics"])=={x+"_rotated" for x in DIRECTIONS}, "Rotated diagnostic inventory differs")
    checks.require(set(saved["pathway_diagnostics"])==set(PATH_DIRECTIONS), "Pathway diagnostic inventory differs")
    expected_conditions = {}
    for condition in CONDITIONS:
        if condition=="clean":
            run = clean
        elif condition=="hybrid_state_oracle":
            run = manual_rollout(state,batch.receiver,state_replacement=states[2].repeat_interleave(recipe.slots,0),patch_step=step)
            hybrid_logits=all_clean["logits"][2*n:]
            checks.tensor(run["logits"],hybrid_logits,"Full hybrid continuation")
            checks.require(torch.equal(run["logits"].argmax(-1),hybrid_logits.argmax(-1)), "Full hybrid prediction parity")
            checks.close(saved["checks"]["oracle_logits_max_error"],float((run["logits"]-hybrid_logits).abs().max()), "Saved hybrid continuation error")
            checks.require(saved["checks"]["oracle_prediction_bits_match"] is True, "Native hybrid parity receipt failed")
        else:
            name, route = condition.split("__") if "__" in condition else (condition,"full")
            if name.endswith("_rotated"):
                delta=directions[name.removesuffix("_rotated")][:,geometry["permutation"]]*geometry["signs"]
                record=manual_direction_record(delta,V,U,groups,base.edit_slot)
                nested_close(saved["rotated_diagnostics"][name],record,checks,checkpoint+"/rotated/"+name)
                original=directions[name.removesuffix("_rotated")]
                checks.tensor(delta.double().norm(dim=-1),original.double().norm(dim=-1),"Signed rotation preserves dose",1e-7)
            else:
                delta=directions[name]
            run=manual_rollout(state,batch.receiver,delta.repeat_interleave(recipe.slots,0),groups,batch.edit_slot,route,patch_step=step)
        expected = manual_condition_record(run,clean,batch,step)
        nested_close(saved["conditions"][condition],expected,checks,checkpoint+"/"+condition)
        expected_conditions[condition]=expected
    for name in DIRECTIONS:
        expected_norm=torch.tensor(diagnostics["directions"][name]["valid"],dtype=torch.float64)*recipe.dose
        checks.tensor(directions[name].double().norm(dim=-1),expected_norm,"Declared equal physical state dose",1e-7)
        record=diagnostics["directions"][name]
        checks.close(record["dense_write_max_abs_error"],0.,"Dense-factor BPD identity",1e-10)
        checks.close(record["group_write_sum_max_abs_error"],0.,"Grouped BPD identity",1e-10)
    for name in PATH_DIRECTIONS:
        record=manual_pathway_record(state,base.receiver,directions[name],step)
        nested_close(saved["pathway_diagnostics"][name],record,checks,checkpoint+"/pathway/"+name)
        for key,value in record.items():
            if key.endswith("max_abs_error"):
                checks.close(value,0.,"Gate/carry identity: "+key)
        algebra_checks(V,U,directions[name],groups,base.edit_slot,checks)
    checks.require(saved["checks"]["checkpoint_unchanged"] is True,"Native mutation receipt failed")
    checks.require(saved["checks"]["n_prediction_cases"]==n*len(CONDITIONS),"Prediction case count differs")
    checks.require(sha256(model_path)==saved["checkpoint_sha256"],"Checkpoint changed during audit")
    return expected_conditions


def independent_counts(predictions, labels_h, labels_r, targeted, ordinary, indices):
    changed=[i for i in indices if targeted[i]]
    unchanged=[i for i in indices if not targeted[i]]
    initially_right=[i for i in unchanged if ordinary[i]==labels_r[i]]
    initially_wrong=[i for i in unchanged if ordinary[i]!=labels_r[i]]
    return {"n_cases":len(indices),"correct_hybrid":sum(predictions[i]==labels_h[i] for i in indices),
        "targeted_n":len(changed),"targeted_correct":sum(predictions[i]==labels_h[i] for i in changed),
        "untargeted_n":len(unchanged),"untargeted_correct":sum(predictions[i]==labels_r[i] for i in unchanged),
        "untargeted_original_correct":len(initially_right),"untargeted_original_wrong":len(initially_wrong),
        "untargeted_corrupted":sum(predictions[i]!=labels_r[i] for i in initially_right),
        "untargeted_recovered":sum(predictions[i]==labels_r[i] for i in initially_wrong)}


def audit_summary(summary, rebuilt, checks):
    from .common import CONTRASTS, CONDITIONS
    checks.require(summary["n_units"]==len(rebuilt) and summary["root_roster_complete"] is True,"Statistical root unit differs")
    checks.require(summary["planned_comparisons"]==summary["actual_inferential_comparisons"]==4,"Comparison family differs")
    for row, expected in zip(summary["per_root"],rebuilt,strict=True):
        root, metadata, snapshots=expected
        checks.require(row["root"]==root,"Summary root order differs")
        for checkpoint in CHECKPOINTS:
            ordinary=snapshots[checkpoint]["clean"]["predictions"]
            for condition in CONDITIONS:
                prediction=snapshots[checkpoint][condition]["predictions"]
                counts=independent_counts(prediction,metadata["labels_h"],metadata["labels_r"],metadata["targeted"],ordinary,list(range(len(prediction))))
                nested_close(row["checkpoints"][checkpoint][condition]["counts"],counts,checks,"Summary integer counts")
                for slot in range(4):
                    indices=[i for i,s in enumerate(metadata["edited_slot"]) if s==slot]
                    slot_counts=independent_counts(prediction,metadata["labels_h"],metadata["labels_r"],metadata["targeted"],ordinary,indices)
                    nested_close(row["checkpoints"][checkpoint][condition]["by_edited_slot"][str(slot)],slot_counts,checks,"Summary per-slot counts")
    probabilities=[]
    for contrast,(candidate,baseline,role) in zip(summary["contrasts"],CONTRASTS,strict=True):
        checks.require(contrast["candidate"]==candidate and contrast["baseline"]==baseline and contrast["role"]==role,"Contrast identity differs")
        checks.require(contrast["checkpoint"]=="B_causal_fixed" and contrast["confirmation_claim"] is False,"Contrast scope differs")
        effects=[]
        for root,metadata,snapshots in rebuilt:
            target=metadata["targeted"]; labels=metadata["labels_h"]
            pred_c=snapshots["B_causal_fixed"][candidate]["predictions"]
            pred_b=snapshots["B_causal_fixed"][baseline]["predictions"]
            numerator=sum(int(a==truth)-int(b==truth) for a,b,truth,t in zip(pred_c,pred_b,labels,target,strict=True) if t)
            effects.append(Fraction(numerator,sum(target)))
        observed=abs(sum(effects))
        extreme=sum(abs(sum(sign*effect for sign,effect in zip(signs,effects,strict=True)))>=observed
                    for signs in itertools.product((-1,1),repeat=len(effects)))
        p=Fraction(extreme,2**len(effects)); probabilities.append(p)
        checks.require(Fraction(contrast["sign_flip"]["p_fraction"])==p,"Exact root sign-flip differs")
        checks.close(contrast["mean_effect"],float(sum(effects)/len(effects)),"Mean root effect")
    adjusted=[None]*len(probabilities); running=Fraction(0)
    for position,index in enumerate(sorted(range(len(probabilities)),key=probabilities.__getitem__)):
        running=max(running,min(Fraction(1),probabilities[index]*(len(probabilities)-position)))
        adjusted[index]=running
    for contrast,p in zip(summary["contrasts"],adjusted,strict=True):
        checks.require(Fraction(contrast["holm_adjusted_p_fraction"])==p,"Holm family adjustment differs")


def audit_preflight(directory, rebuilt, manifest, checks):
    rows=[]
    for root,metadata,snapshots in rebuilt:
        target=metadata["targeted"]; truth=metadata["labels_h"]; n=sum(target)
        records=snapshots["B_causal_fixed"]
        def success(name):
            return sum(p==y for p,y,t in zip(records[name]["predictions"],truth,target,strict=True) if t)
        effect=success("hybrid_span")-success("donor_span")
        sham=success("hybrid_span_rotated")-success("donor_span_rotated")
        rows.append({"unit_id":str(root),"paired_effect":f"{effect}/{n}","sham_effects":[f"{sham}/{n}"]})
    expected={"schema_version":1,"experiment_id":"BPD-RP-04-component-selectivity",
        "independent_unit":"exposed independently trained RP03 model root",
        "alpha":".05","scientific_min_effect":".05","target_effect":".10",
        "measurement_resolution":f"1/{n}","sham_quantile":".95","sham_confidence":".95",
        "planned_comparisons":4,"confirm_units_planned":8,"pilot":rows,
        "pilot_from_development_only":True,"units_independent":True,"heldout_sealed":False,
        "heldout_scope":"fresh stochastic input draws on already exposed model roots",
        "planning_scope":"bounded diagnostic; no confirmation cohort allocated",
        "sham_scope":"paired rotated hybrid-span minus rotated donor-span; uncalibrated population tail"}
    nested_close(read(directory/"preflight_input.json"),expected,checks,"Preflight input")
    path=HERE.parent/"repair_portfolio/evidence/preflight.py"
    checks.require(manifest["source_hashes"][str(path.relative_to(REPO))]==sha256(path),"Pinned planner source differs")
    loader=importlib.util.spec_from_file_location("rp04_independent_pinned_preflight",path)
    checks.require(loader is not None and loader.loader is not None,"Cannot load pinned planner")
    module=importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    result=module.preflight(expected)
    nested_close(read(directory/"preflight_result.json"),result,checks,"Preflight result")
    return result


@torch.no_grad()
def audit_run(directory: Path):
    from .common import (Recipe, ROOTS, SMOKE_ROOTS, PARENT_COMMIT, CONDITIONS,
                         checkpoint_paths, source_hashes, input_hashes)
    from .data import make_batch
    from .diagnostics import summarize_diagnostics
    from .stats import summarize
    started=time.time(); directory=Path(directory)
    checks=Checks(); manifest=read(directory/"run_manifest.json")
    phase=manifest["phase"]
    checks.require(phase in ("smoke","development"),"Unsupported run phase")
    roots=SMOKE_ROOTS if phase=="smoke" else ROOTS
    checks.require(manifest["roots"]==list(roots),"Incomplete or altered assigned root roster")
    checks.require(sorted(p.name for p in directory.glob("root_*") if p.is_dir())==sorted(f"root_{r}" for r in roots),"Extra or missing root folder")
    checks.require(manifest["experiment"]=="BPD-RP-04" and manifest["parent_commit"]==PARENT_COMMIT,"Experiment lineage differs")
    checks.require(manifest["evidence_class"]=="DEVELOPMENT_ONLY" and manifest["new_training_updates"]==0,"Diagnostic scope differs")
    recipe=Recipe(**manifest["recipe"])
    checks.require(manifest["recipe"]==asdict(Recipe()),"Frozen recipe differs")
    torch.set_num_threads(recipe.threads); torch.use_deterministic_algorithms(True)
    checks.require(manifest["source_hashes"]==source_hashes(),"Current source inventory differs")
    checks.require(manifest["input_hashes"]==input_hashes(phase),"Input custody inventory differs")
    check_hash_inventory(REPO,manifest["source_hashes"],checks,"Frozen source")
    check_hash_inventory(REPO,manifest["input_hashes"],checks,"Parent checkpoint")
    lock=manifest.get("source_lock")
    if phase=="development":
        checks.require(isinstance(lock,dict),"Published source lock is required")
        for key in ("source_hashes","input_hashes","recipe","roots"):
            checks.require(lock[key]==manifest[key],"Source lock differs: "+key)
        commit=lock.get("remote_commit","")
        checks.require(len(commit)==40 and all(c in "0123456789abcdef" for c in commit),"Published freeze commit is malformed")
    completion=read(directory/"completion.json")
    checks.require(completion["completed"] is True and completion["roots"]==list(roots),"Completion roster differs")
    artifact_names={str(p.relative_to(directory)) for p in directory.rglob("*.json")
                    if p.name not in ("completion.json","independent_audit.json")}
    checks.require(set(completion["artifact_sha256"])==artifact_names,"Completion artifact inventory differs")
    check_hash_inventory(directory,completion["artifact_sha256"],checks,"Run artifact")
    rebuilt=[]; reports=[]
    for root in roots:
        before=checks.count; folder=directory/f"root_{root}"
        base,batch,metadata=make_batch(root,phase,recipe)
        semantic_checks(base,batch,root,checks)
        checks.require(batch_fingerprint(base)==metadata["base_batch_sha256"],"Base input fingerprint differs")
        checks.require(batch_fingerprint(batch)==metadata["batch_sha256"],"Expanded input fingerprint differs")
        nested_close(read(folder/"cases.json"),metadata,checks,"Case metadata")
        index=read(folder/"result.json")
        checks.require(index["root"]==root and index["phase"]==phase and index["checkpoint_roster"]==list(CHECKPOINTS),"Root checkpoint roster differs")
        checks.require(set(index["artifact_sha256"])=={"cases.json"}|{x+".json" for x in CHECKPOINTS},"Root artifact inventory differs")
        check_hash_inventory(folder,index["artifact_sha256"],checks,"Root artifact")
        paths=checkpoint_paths(root,phase)
        geometry=torch.load(paths["geometry"],weights_only=True,map_location="cpu")
        groups=geometry["groups"]; fixed=geometry["fixed_projectors"]
        checks.require(groups.shape==(4,4) and groups.dtype==torch.long,"Frozen group shape differs")
        checks.require(bool((groups>=0).all()) and bool((groups<32).all()),"Frozen component index outside range")
        checks.require(sorted(geometry["permutation"].tolist())==list(range(32)),"Rotation is not a permutation")
        checks.require(geometry["signs"].shape==(32,) and bool(geometry["signs"].abs().eq(1).all()),"Rotation signs differ")
        warm=torch.load(paths["shared_warmup"],weights_only=True,map_location="cpu")
        checks.tensor(fixed,manual_projectors(warm["recurrent.V"],groups),"Fixed projectors match shared warmup",1e-6)
        snapshots={checkpoint:audit_snapshot(folder/(checkpoint+".json"),root,checkpoint,phase,recipe,
                   base,batch,metadata,geometry,paths[checkpoint],checks) for checkpoint in CHECKPOINTS}
        rebuilt.append((root,metadata,snapshots))
        report={"root":root,"checkpoints_replayed":len(CHECKPOINTS),"conditions_per_checkpoint":len(CONDITIONS),
                "base_pairs":len(base.query),"query_cases":len(batch.query),
                "prediction_bits_replayed":len(batch.query)*len(CONDITIONS)*len(CHECKPOINTS),"checks":checks.count-before}
        reports.append(report)
        print(json.dumps({"audit_root":root,"verdict":"PASS","checks_so_far":checks.count}),flush=True)
    summary=read(directory/"summary.json")
    nested_close(summary,summarize(directory),checks,"Aggregate statistical receipt")
    audit_summary(summary,rebuilt,checks)
    nested_close(read(directory/"diagnostic_summary.json"),summarize_diagnostics(directory),
                 checks,"Aggregate diagnostic receipt")
    preflight=audit_preflight(directory,rebuilt,manifest,checks)
    return {"schema_version":1,"experiment":"BPD-RP-04","phase":phase,"verdict":"PASS",
        "scope":"Independent saved-tensor recurrence, all 26 intervention conditions, factor geometry, gate/carry identities, labels and root-level count/sign-flip replay",
        "scope_limits":[
            "RP03 optimizer trajectories are inherited inputs and are outside this inference-only replay.",
            "The deterministic input sampler is shared; raw encodings, semantic labels, partition membership, query expansion and fingerprints are checked independently.",
            "The statistics-only module is reused for bootstrap receipts; integer endpoints, exact root sign flips and Holm correction are also computed independently.",
            "The diagnostic aggregation helper is reused after every underlying numeric array is independently replayed.",
            "Source custody is verified through local file hashes and the recorded published commit; this audit does not contact the remote repository.",
            "Runtime and peak-memory fields are hash-bound receipts; their original measurements are not independently replayed.",
            "Preflight effects and scope are reconstructed independently; its pinned planning algorithm is reused."],
        "metric_fields_replayed":["condition predictions/correct bits/cross-entropies/logit maxima/all physical scalars",
            "all direction and rotated diagnostics","all pathway diagnostics","all root and per-slot counts",
            "diagnostic aggregate via shared helper after independent raw replay",
            "exact root sign flips and Holm adjustment","preflight effects, family and scope; pinned planner output"],
        "n_roots":len(roots),"n_checkpoints":len(roots)*len(CHECKPOINTS),"n_conditions":len(CONDITIONS),
        "source_hashes_checked":len(manifest["source_hashes"]),"input_hashes_checked":len(manifest["input_hashes"]),
        "artifact_hashes_checked":len(completion["artifact_sha256"]),"prediction_bits_match":True,
        "prediction_bits_replayed":sum(r["prediction_bits_replayed"] for r in reports),"aggregate_statistics_match":True,
        "aggregate_diagnostics_match":True,
        "preflight_input_verified":True,"preflight_output_matches":True,"preflight_decision":preflight["decision"],
        "checks":checks.count,"scalar_tolerance":TOLERANCE,"maximum_scalar_comparison_error":checks.maximum_scalar_error,
        "maximum_tensor_comparison_error":checks.maximum_tensor_error,"elapsed_seconds":time.time()-started,"roots":reports}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    try:
        result=audit_run(args.results_dir)
    except (AuditFailure,KeyError,OSError,ValueError,RuntimeError,TypeError) as error:
        result={"experiment":"BPD-RP-04","verdict":"FAIL","error":str(error)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result,sort_keys=True))
    if result["verdict"]!="PASS":
        raise SystemExit(1)


if __name__=="__main__":
    main()
