"""BPD-PD-01: source-locked phase-disjoint dictionary-learning control.

This intentionally uses the real BPDModel/TiedComponentLinear wrapper. It is a
differentiable sparse-execution surrogate, not a claim of exact expected-L0
optimization or recovery of a neural network mechanism.

See PHASE_DISJOINT_PROTOCOL.md for source and evaluation constraints.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig


class PhaseTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.core.weight.copy_(torch.eye(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


class SharedPhaseGate(nn.Module):
    """Shared CI surrogate sees absolute input coordinates; no phase label."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x.abs()))


def draw_batch(gen: torch.Generator, batch_size: int, *, mixed: bool) -> torch.Tensor:
    """Fresh independently signed nonzero amplitudes, [B,T=4,P=1,D=2]."""
    magnitude = .75 + .5 * torch.rand(batch_size, 4, 1, 2, generator=gen)
    signs = torch.where(
        torch.randint(0, 2, (batch_size, 4, 1, 2), generator=gen).bool(),
        1.0, -1.0,
    )
    x = magnitude * signs
    if not mixed:
        x[:, :2, :, 1] = 0.0
        x[:, 2:, :, 0] = 0.0
    return x


def masked_forward(
    bpd: BPDModel, gate: SharedPhaseGate, x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masks = gate(x)
    schedule = PassMaskSchedule(
        comp_mask=masks,
        delta_mask=torch.zeros_like(masks[..., 0]),
    )
    with bpd.masked_components({"core": schedule}):
        out = torch.stack(
            [bpd.target_model(x[:, t]) for t in range(x.shape[1])], dim=1,
        )
    return out, masks


def loss_parts(bpd: BPDModel, gate: SharedPhaseGate, x: torch.Tensor) -> dict[str, torch.Tensor]:
    y, g = masked_forward(bpd, gate, x)
    recon = (y - x).square().mean()
    sparse = g.mean()
    faith = bpd.wrappers["core"].weight_delta().square().mean()
    switch = (g[:, 1:] - g[:, :-1]).abs().mean()
    return {
        "reconstruction": recon,
        "sparsity": sparse,
        "faithfulness": faith,
        "switch": switch,
        "stage1": recon + .30 * faith + .05 * sparse,
    }


def read_axis_match(bpd: BPDModel) -> float:
    v = F.normalize(bpd.wrappers["core"].V.detach(), dim=0)
    return max(
        float((v[0, 0].abs() + v[1, 1].abs()) / 2),
        float((v[0, 1].abs() + v[1, 0].abs()) / 2),
    )


@torch.no_grad()
def evaluate(bpd: BPDModel, gate: SharedPhaseGate, x: torch.Tensor) -> dict:
    d = loss_parts(bpd, gate, x)
    g = gate(x)
    mean_phase = g.mean(dim=(0, 2))  # [4,C]
    return {
        "reconstruction_mse": float(d["reconstruction"]),
        "sparsity": float(d["sparsity"]),
        "faithfulness_mse": float(d["faithfulness"]),
        "switch": float(d["switch"]),
        "read_axis_match": read_axis_match(bpd),
        "gate_mean_by_pass": mean_phase.tolist(),
        "gate_active_over_half_by_pass": (mean_phase > .5).tolist(),
    }


def run_arm(
    seed: int, *, mixed: bool, freeze_dictionary: bool,
    steps: int = 100, stage2_steps: int = 25,
    batch_size: int = 16, calibration_size: int = 128, heldout_size: int = 128,
) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    bpd = BPDModel(
        PhaseTeacher(), {"core": 2}, max_passes=4,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=freeze_dictionary,
    )
    gate = SharedPhaseGate()
    factor_params = bpd.component_parameters()
    originals = [p.detach().clone() for p in factor_params]
    for p in factor_params:
        p.requires_grad_(not freeze_dictionary)

    train_gen = torch.Generator().manual_seed(100000 + seed)
    calibration = draw_batch(
        torch.Generator().manual_seed(300000 + seed), calibration_size, mixed=mixed,
    )
    heldout = draw_batch(
        torch.Generator().manual_seed(200000 + seed), heldout_size, mixed=mixed,
    )
    optimizer = torch.optim.Adam(
        list(gate.parameters()) + ([] if freeze_dictionary else factor_params),
        lr=.04,
    )
    first_grad_norm = 0.0
    for step in range(steps):
        x = draw_batch(train_gen, batch_size, mixed=mixed)
        optimizer.zero_grad(set_to_none=True)
        d = loss_parts(bpd, gate, x)
        d["stage1"].backward()
        if step == 0 and not freeze_dictionary:
            first_grad_norm = math.sqrt(
                sum(float(p.grad.square().sum()) for p in factor_params if p.grad is not None),
            )
        optimizer.step()

    factor_update_norm = math.sqrt(
        sum(float((p.detach() - p0).square().sum())
            for p, p0 in zip(factor_params, originals)),
    )
    calibration_before = evaluate(bpd, gate, calibration)
    factor_snapshot = [p.detach().clone() for p in factor_params]

    # The tolerance-constrained selection is on calibration only.
    # Stage 2 can optimize GATES. It cannot modify the dictionary.
    for p in factor_params:
        p.requires_grad_(False)
    gate_stage1 = copy.deepcopy(gate.state_dict())
    best_switch = calibration_before["switch"]
    best_coeff = 0.0
    best_gate_state = copy.deepcopy(gate_stage1)
    candidates = []
    for coeff in (.01, .05, .20):
        gate.load_state_dict(gate_stage1)
        optimizer2 = torch.optim.Adam(gate.parameters(), lr=.01)
        for _ in range(stage2_steps):
            x = draw_batch(train_gen, batch_size, mixed=mixed)
            optimizer2.zero_grad(set_to_none=True)
            parts = loss_parts(bpd, gate, x)
            (parts["stage1"] + coeff * parts["switch"]).backward()
            optimizer2.step()
        candidate = evaluate(bpd, gate, calibration)
        feasible = (
            candidate["reconstruction_mse"] <= calibration_before["reconstruction_mse"] + .005
            and candidate["sparsity"] <= calibration_before["sparsity"] + .005
        )
        candidates.append({
            "switch_coeff": coeff, "calibration_feasible": bool(feasible),
            "calibration_reconstruction_mse": candidate["reconstruction_mse"],
            "calibration_sparsity": candidate["sparsity"],
            "calibration_switch": candidate["switch"],
        })
        if feasible and candidate["switch"] < best_switch:
            best_switch = candidate["switch"]
            best_coeff = coeff
            best_gate_state = copy.deepcopy(gate.state_dict())

    gate.load_state_dict(best_gate_state)
    factors_unchanged = all(
        torch.equal(p.detach(), before) for p, before in zip(factor_params, factor_snapshot),
    )
    if not factors_unchanged:
        raise AssertionError("Stage 2 modified a frozen dictionary")
    calibration_after = evaluate(bpd, gate, calibration)
    heldout_after = evaluate(bpd, gate, heldout)

    phase_gates = heldout_after["gate_active_over_half_by_pass"]
    phase_0 = phase_gates[0] == phase_gates[1]
    phase_1 = phase_gates[2] == phase_gates[3]
    opposite = phase_gates[0] != phase_gates[2]
    one_each = sum(phase_gates[0]) == 1 and sum(phase_gates[2]) == 1
    gate_routing = bool(phase_0 and phase_1 and opposite and one_each)
    directional_pass = (
        not mixed and not freeze_dictionary
        and heldout_after["read_axis_match"] >= .90
        and heldout_after["reconstruction_mse"] <= .02
        and factor_update_norm > 0.0
        and gate_routing
    )
    return {
        "study": "BPD-PD-01",
        "seed": seed, "mixed": mixed,
        "dictionary_trainable_in_stage1": not freeze_dictionary,
        "training_updates": steps, "stage2_updates_per_coeff": stage2_steps,
        "batch_size": batch_size,
        "factor_first_step_grad_norm": first_grad_norm,
        "factor_stage1_update_norm": factor_update_norm,
        "stage2_dictionary_bitwise_frozen": factors_unchanged,
        "calibration_stage1": calibration_before,
        "calibration_selected": calibration_after,
        "stage2_candidates": candidates,
        "selected_switch_coeff": best_coeff,
        "heldout_selected": heldout_after,
        "phase_routing_meets_binary_pattern": gate_routing,
        "directional_gate_passed": directional_pass,
    }


def run_registered(steps: int = 100, stage2_steps: int = 25) -> dict:
    arms = []
    for seed in (71, 89):
        for mixed, frozen in ((False, False), (True, False), (False, True)):
            arms.append(run_arm(
                seed, mixed=mixed, freeze_dictionary=frozen,
                steps=steps, stage2_steps=stage2_steps,
            ))
    trained_disjoint = [x for x in arms if not x["mixed"] and x["dictionary_trainable_in_stage1"]]
    return {
        "study": "BPD-PD-01",
        "scope": "2D shared BPD factor training control; prospective seed pair; not a neural mechanism result",
        "torch_version": torch.__version__,
        "steps": steps,
        "stage2_steps_per_candidate": stage2_steps,
        "registered_primary_criterion_passed": bool(
            steps == 100 and stage2_steps == 25
            and all(x["directional_gate_passed"] for x in trained_disjoint)
        ),
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--stage2-steps", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    receipt = run_registered(args.steps, args.stage2_steps)
    data = json.dumps(receipt, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data + "\n", encoding="utf-8")
    else:
        print(data)


if __name__ == "__main__":
    main()
