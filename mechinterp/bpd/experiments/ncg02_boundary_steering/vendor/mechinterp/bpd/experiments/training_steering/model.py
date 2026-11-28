"""A trainable recurrent model whose shared operator uses native BPD factors.

Every token executes ``TiedComponentLinear`` in component mode with all
component gates open and residual delta zero. ``W_target`` is an initialization
buffer; it contributes zero to the executed recurrent write. Input/output
weights and the physical V/U factor pair remain trainable.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear
from mechinterp.bpd.config import AblationConfig


class ComponentMemory(nn.Module):
    def __init__(self, width: int = 32, input_dim: int = 11,
                 values: int = 4, components: int = 32) -> None:
        super().__init__()
        if min(width, input_dim, values, components) < 1:
            raise ValueError("Model dimensions must be positive")
        self.width = width
        self.input_dim = input_dim
        self.values = values
        self.components = components
        self.input = nn.Linear(input_dim, 3 * width)
        self.output = nn.Linear(width, values)
        self.recurrent = TiedComponentLinear(
            nn.Linear(width, 3 * width, bias=False), components,
            max_passes=64,
            ablation=AblationConfig(mode="zero", sphere_rescale=False),
            init_svd=True,
        )
        self.recurrent.mode = "component"

    def rollout(self, x: Tensor, patch: Tensor | None = None,
                patch_step: int = 4, return_states: bool = False) -> dict[str, Tensor]:
        """Execute a complete sequence with an optional single write edit.

        ``states`` and ``recurrent_writes`` are always returned so audit code
        receives a stable schema. ``return_states`` is accepted as an explicit
        caller hint. The patch acts on the recurrent preactivation before its
        reset/update/candidate gates are evaluated.
        """
        if x.ndim != 3 or x.shape[2] != self.input_dim or x.shape[1] < 1:
            raise ValueError("x must have shape [batch, time >= 1, input_dim]")
        if not x.is_floating_point() or x.shape[0] < 1:
            raise ValueError("A nonempty floating-point input batch is required")
        batch, steps, _ = x.shape
        if patch is not None:
            if patch.shape != (batch, 3 * self.width):
                raise ValueError("patch must have shape [batch, 3 * width]")
            if type(patch_step) is not int or not 0 <= patch_step < steps:
                raise ValueError("patch_step must identify a token in the sequence")
            if patch.device != x.device:
                raise ValueError("patch and features must use the same device")
        state = x.new_zeros((batch, self.width))
        self.recurrent.reset_pass_state()
        self.recurrent.mode = "component"
        self.recurrent.schedule = PassMaskSchedule(
            comp_mask=x.new_ones((batch, steps, self.components)),
            delta_mask=x.new_zeros((batch, steps)),
        )
        states, writes = [], []
        encoded = self.input(x)
        try:
            for step in range(steps):
                write = self.recurrent(state)
                if patch is not None and step == patch_step:
                    write = write + patch
                input_reset, input_update, input_candidate = encoded[:, step].chunk(3, dim=-1)
                recurrent_reset, recurrent_update, recurrent_candidate = write.chunk(3, dim=-1)
                reset = torch.sigmoid(input_reset + recurrent_reset)
                update = torch.sigmoid(input_update + recurrent_update)
                candidate = torch.tanh(input_candidate + reset * recurrent_candidate)
                state = (1.0 - update) * candidate + update * state
                states.append(state)
                writes.append(write)
        finally:
            self.recurrent.schedule = None
        return {"logits": self.output(state), "states": torch.stack(states, dim=1),
                "recurrent_writes": torch.stack(writes, dim=1)}

    def forward(self, x: Tensor) -> Tensor:
        return self.rollout(x)["logits"]

    def prefix_state(self, x: Tensor, prefix_steps: int = 4) -> Tensor:
        if type(prefix_steps) is not int or not 1 <= prefix_steps <= x.shape[1]:
            raise ValueError("prefix_steps must identify a nonempty input prefix")
        return self.rollout(x[:, :prefix_steps])["states"][:, -1]


def component_patch(model: ComponentMemory, recipient_state: Tensor,
                    donor_state: Tensor, groups: Tensor, edit_slot: Tensor,
                    dose: float = .75) -> tuple[Tensor, dict[str, Tensor]]:
    """Replace selected donor reads and fix the delivered write L2 dose.

    The donor state is detached. Gradients remain available to recipient states
    and both learned factors. A raw norm at or below 1e-9 yields an explicit
    zero edit and ``valid=False``. No unrelated direction fills that case.
    """
    if not math.isfinite(dose) or dose <= 0:
        raise ValueError("dose must be positive and finite")
    if (recipient_state.ndim != 2 or donor_state.shape != recipient_state.shape
            or recipient_state.shape[1] != model.width):
        raise ValueError("Recipient and donor states must share shape [batch, width]")
    if groups.ndim != 2 or groups.shape[0] < 1 or groups.shape[1] < 1:
        raise ValueError("groups must have shape [slots, nonempty component group]")
    if groups.dtype != torch.long or edit_slot.dtype != torch.long:
        raise ValueError("groups and edit_slot must use torch.long indices")
    if edit_slot.shape != (len(recipient_state),):
        raise ValueError("edit_slot must contain one slot index per case")
    if (bool((groups < 0).any()) or bool((groups >= model.components).any())
            or bool((edit_slot < 0).any()) or bool((edit_slot >= len(groups)).any())):
        raise ValueError("Component or slot index is out of range")
    if any(len(torch.unique(group)) != len(group) for group in groups):
        raise ValueError("A component group must contain distinct indices")
    groups = groups.to(device=recipient_state.device)
    edit_slot = edit_slot.to(device=recipient_state.device)
    factor = model.recurrent
    difference = donor_state.detach().float() @ factor.V - recipient_state.float() @ factor.V
    selected = groups[edit_slot]
    mask = torch.zeros_like(difference).scatter(1, selected, 1.0)
    raw = (difference * mask) @ factor.U
    raw_norm = torch.linalg.vector_norm(raw, dim=-1)
    valid = raw_norm > 1e-9
    scale = torch.where(valid, dose / raw_norm.clamp_min(1e-9), torch.zeros_like(raw_norm))
    delta = raw * scale[:, None]
    return delta, {
        "raw_norm": raw_norm,
        "delivered_norm": torch.linalg.vector_norm(delta, dim=-1),
        "valid": valid,
        "recipient_state_norm": torch.linalg.vector_norm(recipient_state, dim=-1),
        "donor_state_norm": torch.linalg.vector_norm(donor_state.detach(), dim=-1),
    }

