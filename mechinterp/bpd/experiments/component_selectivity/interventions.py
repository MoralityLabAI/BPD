"""Native BPD state edits, component-write accounting, and pathway controls.

Direction construction uses float64 and a fixed pseudoinverse cutoff. Every
realized intervention is cast to the model state's dtype before execution.
The native recurrence retains all component gates and zero residual weight.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


DIRECTIONS = (
    "donor_span", "hybrid_span", "hybrid_live_span",
    "donor_dual", "hybrid_dual", "hybrid_full",
)
ROUTES = (
    "full", "carry_only", "all_write_only", "selected_write_only",
    "outside_write_only", "carry_selected", "carry_outside",
)
PINV_RTOL = 1e-10
ZERO_NORM = 1e-9


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(*values: Tensor) -> None:
    _require(all(bool(torch.isfinite(x).all()) for x in values), "Nonfinite intervention input")


def _mask(groups: Tensor, edit_slot: Tensor, batch: int, components: int,
          device: torch.device, dtype: torch.dtype) -> Tensor:
    _require(groups.ndim == 2 and groups.shape[0] > 0 and groups.shape[1] > 0,
             "groups must have shape [slots, nonempty group]")
    _require(groups.dtype == torch.long and edit_slot.dtype == torch.long,
             "Group and slot indices must use torch.long")
    _require(edit_slot.shape == (batch,), "edit_slot must contain one index per case")
    _require(bool((groups >= 0).all()) and bool((groups < components).all()),
             "Component index is out of range")
    _require(bool((edit_slot >= 0).all()) and bool((edit_slot < len(groups)).all()),
             "Slot index is out of range")
    _require(all(len(torch.unique(g)) == len(g) for g in groups),
             "Repeated index inside a component group")
    selected = groups.to(device)[edit_slot.to(device)]
    return torch.zeros((batch, components), dtype=dtype, device=device).scatter_(1, selected, 1.)


def _ratio(numerator: Tensor, denominator: Tensor, epsilon: float = ZERO_NORM) -> list:
    """Undefined scalar ratios remain explicit nulls in JSON receipts."""
    numerator, denominator = numerator.detach().cpu(), denominator.detach().cpu()
    return [float(a / b) if float(b) > epsilon else None
            for a, b in zip(numerator, denominator, strict=True)]


def _cosine(a: Tensor, b: Tensor) -> list:
    return _ratio((a * b).sum(-1), a.norm(dim=-1) * b.norm(dim=-1), 1e-18)


def _normalize(raw: Tensor, like: Tensor, dose: float):
    norm = raw.norm(dim=-1)
    valid = norm > ZERO_NORM
    scale = torch.where(valid, dose / norm.clamp_min(ZERO_NORM), torch.zeros_like(norm))
    delta = (raw * scale[:, None]).to(like.dtype)
    return delta, norm, valid, scale


def _project(difference: Tensor, projection: Tensor) -> Tensor:
    return torch.bmm(difference[:, None, :], projection).squeeze(1)


@torch.no_grad()
def direction_diagnostics(delta: Tensor, V: Tensor, U: Tensor,
                          groups: Tensor, edit_slot: Tensor) -> dict:
    """Describe current-factor reads and physical component writes.

    Group-summed write leakage is invariant under invertible basis changes
    within selected and unselected blocks. Atom-energy leakage is invariant
    under inverse column/row scaling, but can change under within-block mixing.
    It is a fraction of summed atom energies, not of net write energy.
    """
    _require(delta.ndim == 2 and delta.shape[0] > 0, "delta must be nonempty [batch,width]")
    _require(V.ndim == 2 and U.ndim == 2 and V.shape[0] == delta.shape[1]
             and V.shape[1] == U.shape[0], "Incompatible BPD factor shapes")
    _finite(delta, V, U)
    device = delta.device
    _require(V.device == U.device == device, "Factors and delta must share a device")
    d, v, u = delta.double(), V.detach().double(), U.detach().double()
    mask = _mask(groups, edit_slot, len(delta), v.shape[1], device, torch.float64)
    reads = d @ v
    selected = (reads * mask) @ u
    outside = (reads * (1 - mask)) @ u
    total = reads @ u
    dense = d @ (v @ u)
    selected_norm, outside_norm = selected.norm(dim=-1), outside.norm(dim=-1)
    denom = selected_norm.square() + outside_norm.square()
    atom_energy = reads.square() * u.square().sum(dim=-1)[None, :]
    atom_total = atom_energy.sum(-1)
    atom_outside = (atom_energy * (1 - mask)).sum(-1)
    cancellation_denominator = selected_norm + outside_norm
    cancellation = [1. - value if value is not None else None
                    for value in _ratio(total.norm(dim=-1), cancellation_denominator)]
    per_slot = []
    for slot in range(len(groups)):
        chosen = edit_slot.to(delta.device) == slot
        per_slot.append(atom_energy[chosen].mean(0).tolist() if bool(chosen.any()) else None)
    return {
        "delivered_norm": d.norm(dim=-1).tolist(),
        "selected_write_norm": selected_norm.tolist(),
        "outside_write_norm": outside_norm.tolist(),
        "total_write_norm": total.norm(dim=-1).tolist(),
        "grouped_write_leakage": _ratio(outside_norm.square(), denom, 1e-18),
        "grouped_write_cosine": _cosine(selected, outside),
        "cancellation": cancellation,
        "atom_energy_leakage": _ratio(atom_outside, atom_total, 1e-18),
        "atom_energy_by_component_mean": atom_energy.mean(0).tolist(),
        "atom_energy_by_slot_and_component_mean": per_slot,
        "selected_read_norm_gauge_dependent": (reads * mask).norm(dim=-1).tolist(),
        "outside_read_norm_gauge_dependent": (reads * (1 - mask)).norm(dim=-1).tolist(),
        "dense_write_max_abs_error": float((dense - total).abs().max()),
        "group_write_sum_max_abs_error": float((selected + outside - total).abs().max()),
    }


@torch.no_grad()
def build_directions(state_r: Tensor, state_d: Tensor, state_h: Tensor,
                     V: Tensor, U: Tensor, groups: Tensor,
                     fixed_projectors: Tensor, edit_slot: Tensor,
                     dose: float = .75) -> tuple[dict[str, Tensor], dict]:
    """Construct the six fixed-recipe directions and their physical receipts."""
    _require(math.isfinite(dose) and dose > 0, "dose must be finite and positive")
    _require(state_r.ndim == 2 and state_r.shape[0] > 0 and state_r.shape[1] > 0,
             "States must have nonempty shape [batch,width]")
    _require(state_d.shape == state_h.shape == state_r.shape, "State shapes differ")
    _require(state_r.is_floating_point() and state_d.dtype == state_h.dtype == state_r.dtype,
             "States must share a floating-point dtype")
    _require(state_r.device == state_d.device == state_h.device == V.device == U.device
             == fixed_projectors.device, "Intervention inputs must share a device")
    _require(V.ndim == 2 and V.shape[0] == state_r.shape[1] and U.ndim == 2
             and U.shape[0] == V.shape[1], "Incompatible BPD factors")
    width, components = V.shape
    _require(groups.ndim == 2 and groups.shape[0] > 0 and groups.shape[1] > 0,
             "groups must have shape [slots, nonempty group]")
    _require(fixed_projectors.shape == (len(groups), width, width),
             "Fixed projector shape differs from groups")
    _require(groups.shape[1] <= width, "Group exceeds state width")
    _finite(state_r, state_d, state_h, V, U, fixed_projectors)
    v, u = V.detach().double(), U.detach().double()
    mask = _mask(groups, edit_slot, len(state_r), components, state_r.device, torch.float64)
    differences = {"donor": state_d.double() - state_r.double(),
                   "hybrid": state_h.double() - state_r.double()}
    fixed = fixed_projectors.detach().double()[edit_slot.to(fixed_projectors.device)]
    live_all, group_ranks = [], []
    for group in groups.to(v.device):
        columns = v[:, group]
        singular = torch.linalg.svdvals(columns)
        rank = int((singular > singular.max() * PINV_RTOL).sum())
        _require(rank == len(group), "Selected live component span is rank deficient")
        frame = torch.linalg.qr(columns, mode="reduced").Q
        live_all.append(frame @ frame.T)
        group_ranks.append(rank)
    live_all = torch.stack(live_all)
    live = live_all[edit_slot.to(v.device)]
    write_projectors, write_group_ranks = [], []
    for group in groups.to(u.device):
        _, write_singular, vh = torch.linalg.svd(u[group], full_matrices=False)
        write_rank = int((write_singular > write_singular.max() * PINV_RTOL).sum())
        write_frame = vh[:write_rank].T
        write_projectors.append(write_frame @ write_frame.T)
        write_group_ranks.append(write_rank)
    write_projectors = torch.stack(write_projectors)[edit_slot.to(u.device)]
    hybrid_full_write = (differences["hybrid"] @ v) @ u
    hybrid_selected_U_projection = _project(hybrid_full_write, write_projectors)
    singular = torch.linalg.svdvals(v)
    rank = int((singular > singular.max() * PINV_RTOL).sum())
    inverse = torch.linalg.pinv(v, rtol=PINV_RTOL, atol=0.)
    desired = {name: (d @ v) * mask for name, d in differences.items()}
    live_span = {name: _project(d, live) for name, d in differences.items()}
    raw = {
        "donor_span": _project(differences["donor"], fixed),
        "hybrid_span": _project(differences["hybrid"], fixed),
        "hybrid_live_span": live_span["hybrid"],
        "donor_dual": desired["donor"] @ inverse,
        "hybrid_dual": desired["hybrid"] @ inverse,
        "hybrid_full": differences["hybrid"],
    }
    directions, records = {}, {}
    for name in DIRECTIONS:
        delta, raw_norm, valid, scale = _normalize(raw[name], state_r, dose)
        directions[name] = delta
        record = direction_diagnostics(delta, V, U, groups, edit_slot)
        source = "donor" if name.startswith("donor") else "hybrid"
        record.update({
            "raw_norm": raw_norm.tolist(), "valid": valid.tolist(),
            "normalization_scale": scale.tolist(),
            "raw_amplification": _ratio(raw_norm, differences[source].norm(dim=-1)),
        })
        if name.endswith("dual"):
            target = desired[source]
            target_norm = target.norm(dim=-1)
            residual = (raw[name] @ v - target).norm(dim=-1)
            normalized_residual = (delta.double() @ v - target * scale[:, None]).norm(dim=-1)
            span_norm = live_span[source].norm(dim=-1)
            perpendicular_norm = (raw[name] - live_span[source]).norm(dim=-1)
            pythagorean = raw_norm.square() - span_norm.square() - perpendicular_norm.square()
            record.update({
                "desired_read_norm": target_norm.tolist(),
                "raw_read_residual_relative": _ratio(residual, target_norm),
                "raw_read_residual_absolute": residual.tolist(),
                "normalized_read_residual_relative": _ratio(
                    normalized_residual, (target * scale[:, None]).norm(dim=-1)),
                "normalized_read_residual_absolute": normalized_residual.tolist(),
                "normalized_read_scale": scale.tolist(),
                "raw_live_span_norm": span_norm.tolist(),
                "dual_to_live_span_norm_ratio": _ratio(raw_norm, span_norm),
                "dual_minus_live_span_norm": perpendicular_norm.tolist(),
                "pythagorean_absolute_residual": pythagorean.abs().tolist(),
                "pythagorean_relative_residual": _ratio(pythagorean.abs(), raw_norm.square(), 1e-18),
                "selected_read_amplitude_vs_live_span": _ratio(span_norm, raw_norm),
            })
        records[name] = record
    drifts = ((live_all - fixed_projectors.double()).square().sum((-1, -2))
              / (2 * groups.shape[1])).sqrt()
    return directions, {
        "factor": {
            "shape": list(v.shape), "rank": rank, "pinv_rtol": PINV_RTOL,
            "singular_values": singular.tolist(),
            "condition_number": (float(singular.max() / singular.min())
                                 if rank == min(v.shape) else None),
            "selected_group_ranks": group_ranks,
            "selected_U_group_ranks": write_group_ranks,
        },
        "geometry": {
            "hybrid_difference_norm": differences["hybrid"].norm(dim=-1).tolist(),
            "hybrid_full_write_norm": hybrid_full_write.norm(dim=-1).tolist(),
            "hybrid_selected_U_projection_residual": _ratio(
                (hybrid_full_write - hybrid_selected_U_projection).norm(dim=-1),
                hybrid_full_write.norm(dim=-1)),
            "donor_difference_norm": differences["donor"].norm(dim=-1).tolist(),
            "hybrid_fixed_projection_residual": _ratio(
                (differences["hybrid"] - raw["hybrid_span"]).norm(dim=-1),
                differences["hybrid"].norm(dim=-1)),
            "hybrid_live_projection_residual": _ratio(
                (differences["hybrid"] - raw["hybrid_live_span"]).norm(dim=-1),
                differences["hybrid"].norm(dim=-1)),
            "donor_hybrid_projected_cosine": _cosine(raw["donor_span"], raw["hybrid_span"]),
            "projector_chordal_by_slot": drifts.tolist(),
        },
        "directions": records,
    }


@torch.no_grad()
def native_rollout(model, x: Tensor, delta: Tensor | None = None,
                   groups: Tensor | None = None, edit_slot: Tensor | None = None,
                   route: str = "full", patch_step: int = 4,
                   state_replacement: Tensor | None = None) -> dict[str, Tensor]:
    """Execute a state edit or a one-token carry/component-write pathway control.

    Full edits use RP03's native state patch. Other routes add or remove the
    corresponding physical BPD writes through a temporary forward-output hook.
    Their write-site doses are inherited from the parent state edit.
    """
    _require(route in ROUTES, "Unknown pathway route")
    _require(delta is None or state_replacement is None,
             "State replacement and additive delta are exclusive")
    if state_replacement is not None:
        _require(route == "full", "State replacement requires the full route")
        return model.rollout(x, state_replacement=state_replacement, patch_step=patch_step)
    if delta is None:
        _require(route == "full", "A pathway control requires a delta")
        return model.rollout(x)
    _require(x.ndim == 3 and delta.shape == (len(x), model.width), "Delta shape differs")
    _require(delta.dtype == x.dtype and delta.device == x.device,
             "Delta and features must share dtype and device")
    _require(type(patch_step) is int and 0 <= patch_step < x.shape[1], "Invalid patch step")
    _finite(delta)
    if route == "full":
        return model.rollout(x, patch=delta, patch_step=patch_step)
    v, u = model.recurrent.V.detach(), model.recurrent.U.detach()
    reads = delta @ v
    total = reads @ u
    selected = outside = None
    if route in ("selected_write_only", "outside_write_only", "carry_selected", "carry_outside"):
        _require(groups is not None and edit_slot is not None, "Selected-write routes require groups and slots")
        mask = _mask(groups, edit_slot, len(delta), v.shape[1], delta.device, delta.dtype)
        selected = (reads * mask) @ u
        outside = (reads * (1 - mask)) @ u
    correction = {
        "carry_only": -total,
        "all_write_only": total,
        "selected_write_only": selected,
        "outside_write_only": outside,
        "carry_selected": -outside if outside is not None else None,
        "carry_outside": -selected if selected is not None else None,
    }[route]
    carry = route in ("carry_only", "carry_selected", "carry_outside")
    call = 0
    interventions = 0

    def edit_output(module, inputs, output):
        nonlocal call, interventions
        current = call
        call += 1
        if current != patch_step:
            return output
        if (module.mode != "component" or module.schedule is None
                or not bool((module.schedule.delta_mask == 0).all())
                or not bool((module.schedule.comp_mask == 1).all())):
            raise RuntimeError("Pathway hook requires native all-open component execution with zero residual")
        interventions += 1
        return output + correction

    handle = model.recurrent.register_forward_hook(edit_output)
    try:
        result = model.rollout(x, patch=delta if carry else None, patch_step=patch_step)
        if interventions != 1 or call != x.shape[1]:
            raise RuntimeError("Native pathway hook missed its declared recurrent pass")
        return result
    finally:
        handle.remove()


@torch.no_grad()
def pathway_diagnostics(model, x: Tensor, delta: Tensor, patch_step: int = 4) -> dict:
    """Check exact finite carry/gate identities at the intervention token.

    The corner construction uses the observed native clean and edited writes.
    Floating-point associativity can make the separately routed write-only
    rollout differ slightly from that algebraic corner.
    """
    clean = native_rollout(model, x)
    edited = native_rollout(model, x, delta=delta, patch_step=patch_step)
    h = clean["incoming_states"][:, patch_step]
    w = clean["recurrent_writes"][:, patch_step]
    wp = edited["recurrent_writes"][:, patch_step]
    ir, iz, inn = model.input(x[:, patch_step]).chunk(3, dim=-1)

    def gate_values(write):
        wr, wz, wn = write.chunk(3, dim=-1)
        r = torch.sigmoid(ir + wr)
        z = torch.sigmoid(iz + wz)
        n = torch.tanh(inn + r * wn)
        return r, z, n

    r, z, n = gate_values(w)
    rp, zp, np = gate_values(wp)
    f00 = (1 - z) * n + z * h
    f10 = (1 - z) * n + z * (h + delta)
    f01 = (1 - zp) * np + zp * h
    f11 = (1 - zp) * np + zp * (h + delta)
    carry = zp * delta
    update = (zp - z) * (h - n)
    candidate = (1 - zp) * (np - n)
    gate = update + candidate
    interaction = (zp - z) * delta
    actual = edited["states"][:, patch_step] - clean["states"][:, patch_step]
    reads = delta @ model.recurrent.V
    predicted_write = reads @ model.recurrent.U
    return {
        "state_delta_norm": actual.double().norm(dim=-1).tolist(),
        "carry_term_norm": carry.double().norm(dim=-1).tolist(),
        "update_gate_term_norm": update.double().norm(dim=-1).tolist(),
        "candidate_term_norm": candidate.double().norm(dim=-1).tolist(),
        "gate_term_norm": gate.double().norm(dim=-1).tolist(),
        "carry_gate_cosine": _cosine(carry.double(), gate.double()),
        "carry_only_state_delta_norm": (f10 - f00).double().norm(dim=-1).tolist(),
        "write_only_state_delta_norm": (f01 - f00).double().norm(dim=-1).tolist(),
        "interaction_norm": interaction.double().norm(dim=-1).tolist(),
        "reset_gate_delta_norm": (rp - r).double().norm(dim=-1).tolist(),
        "update_gate_delta_norm": (zp - z).double().norm(dim=-1).tolist(),
        "candidate_delta_norm": (np - n).double().norm(dim=-1).tolist(),
        "write_delta_norm": (wp - w).double().norm(dim=-1).tolist(),
        "final_all_head_delta_norm": (edited["logits"] - clean["logits"]).double().norm(dim=-1).tolist(),
        "decomposition_max_abs_error": float((actual - carry - update - candidate).abs().max()),
        "interaction_max_abs_error": float((f11 - f10 - f01 + f00 - interaction).abs().max()),
        "carry_only_identity_max_abs_error": float((f10 - f00 - z * delta).abs().max()),
        "native_clean_state_max_abs_error": float((f00 - clean["states"][:, patch_step]).abs().max()),
        "native_full_state_max_abs_error": float((f11 - edited["states"][:, patch_step]).abs().max()),
        "linear_write_max_abs_error": float((wp - w - predicted_write).abs().max()),
    }
