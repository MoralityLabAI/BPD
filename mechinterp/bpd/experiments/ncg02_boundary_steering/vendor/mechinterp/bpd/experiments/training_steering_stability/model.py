"""RP03 hidden-state interventions on the native RP02 BPD recurrence.

The input/output layers, GRU equations and shared V/U operator match RP02.
A state intervention occurs before one token and enters both the recurrent
operator and the GRU carry. Projectors have an explicit stop-gradient boundary.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from mechinterp.bpd.components import PassMaskSchedule
from mechinterp.bpd.experiments.training_steering.model import ComponentMemory


class ComponentStateMemory(ComponentMemory):
    """The RP02 model with a hidden-state intervention before a chosen token."""

    def rollout(self, x: Tensor, patch: Tensor | None = None,
                patch_step: int = 4, return_states: bool = False,
                state_replacement: Tensor | None = None) -> dict[str, Tensor]:
        """Return logits and traces, including the state entering each token.

        ``patch`` adds to hidden state before the recurrent write and carry are
        evaluated. ``state_replacement`` supplies a full hidden state for a
        privileged structural control; the two interventions are exclusive.
        The clean sequence reads features alone and executes native BPD V/U
        with every component open and residual delta zero at every token.
        """
        if x.ndim != 3 or x.shape[2] != self.input_dim or x.shape[1] < 1:
            raise ValueError("x must have shape [batch, time >= 1, input_dim]")
        if not x.is_floating_point() or x.shape[0] < 1:
            raise ValueError("A nonempty floating-point input batch is required")
        if patch is not None and state_replacement is not None:
            raise ValueError("An additive patch and a full state replacement are exclusive")
        batch, steps, _ = x.shape
        intervention = patch if patch is not None else state_replacement
        if intervention is not None:
            if intervention.shape != (batch, self.width):
                raise ValueError("Hidden intervention must have shape [batch, width]")
            if type(patch_step) is not int or not 0 <= patch_step < steps:
                raise ValueError("patch_step must identify a token in the sequence")
            if intervention.device != x.device or intervention.dtype != x.dtype:
                raise ValueError("Hidden intervention and features must share dtype and device")
            if not bool(torch.isfinite(intervention).all()):
                raise ValueError("Hidden intervention must be finite")
        state = x.new_zeros((batch, self.width))
        self.recurrent.reset_pass_state()
        self.recurrent.mode = "component"
        self.recurrent.schedule = PassMaskSchedule(
            comp_mask=x.new_ones((batch, steps, self.components)),
            delta_mask=x.new_zeros((batch, steps)),
        )
        states, incoming, writes = [], [], []
        encoded = self.input(x)
        try:
            for step in range(steps):
                if step == patch_step:
                    if state_replacement is not None:
                        state = state_replacement
                    elif patch is not None:
                        state = state + patch
                incoming.append(state)
                write = self.recurrent(state)
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
                "incoming_states": torch.stack(incoming, dim=1),
                "recurrent_writes": torch.stack(writes, dim=1)}


@torch.no_grad()
def projectors(model: ComponentStateMemory, groups: Tensor) -> Tensor:
    """Return detached orthogonal projectors onto the selected V-column spans.

    Each group must have full column rank. QR sign changes and invertible
    within-group basis changes preserve the resulting projector. A caller can
    freeze this tensor after common warmup or recompute it at a later step;
    both choices have the same stop-gradient rule.
    """
    if groups.ndim != 2 or groups.shape[0] < 1 or groups.shape[1] < 1:
        raise ValueError("groups must have shape [slots, nonempty group]")
    if groups.dtype != torch.long:
        raise ValueError("groups must use torch.long indices")
    if groups.shape[1] > model.width:
        raise ValueError("A group cannot have more independent columns than model width")
    if bool((groups < 0).any()) or bool((groups >= model.components).any()):
        raise ValueError("Component index is out of range")
    if any(len(torch.unique(group)) != len(group) for group in groups):
        raise ValueError("Each component group must contain distinct indices")
    factor = model.recurrent.V.detach()
    groups = groups.to(factor.device)
    matrices = []
    for group in groups:
        columns = factor[:, group]
        if not bool(torch.isfinite(columns).all()):
            raise ValueError("Selected V columns must be finite")
        singular = torch.linalg.svdvals(columns)
        tolerance = max(columns.shape) * torch.finfo(columns.dtype).eps * singular.max()
        if bool(singular.min() <= tolerance):
            raise ValueError("Selected V group is rank deficient")
        frame = torch.linalg.qr(columns, mode="reduced").Q
        matrices.append(frame @ frame.T)
    return torch.stack(matrices).detach()


def state_patch(recipient_state: Tensor, donor_state: Tensor,
                frozen_projectors: Tensor, edit_slot: Tensor,
                dose: float = .75) -> tuple[Tensor, dict[str, Tensor]]:
    """Project donor-minus-recipient state and normalize its physical L2 dose.

    Donor state and projector values are detached. The recipient state remains
    differentiable. Raw norm <= 1e-9 produces a zero delta with ``valid=False``.
    The projector supplies direction; dose normalization rescales that direction.
    """
    if not math.isfinite(dose) or dose <= 0:
        raise ValueError("dose must be positive and finite")
    if (recipient_state.ndim != 2 or recipient_state.shape[0] < 1
            or recipient_state.shape[1] < 1 or donor_state.shape != recipient_state.shape
            or not recipient_state.is_floating_point()):
        raise ValueError("Recipient and donor states must share nonempty shape [batch, width]")
    batch, width = recipient_state.shape
    if (frozen_projectors.ndim != 3 or frozen_projectors.shape[0] < 1
            or frozen_projectors.shape[1:] != (width, width)):
        raise ValueError("Projectors must have shape [slots, width, width]")
    for value in (donor_state, frozen_projectors):
        if value.device != recipient_state.device or value.dtype != recipient_state.dtype:
            raise ValueError("States and projectors must share dtype and device")
    if edit_slot.shape != (batch,) or edit_slot.dtype != torch.long:
        raise ValueError("edit_slot must use one torch.long slot index per case")
    if bool((edit_slot < 0).any()) or bool((edit_slot >= len(frozen_projectors)).any()):
        raise ValueError("Slot index is out of range")
    if not all(bool(torch.isfinite(value).all())
               for value in (recipient_state, donor_state, frozen_projectors)):
        raise ValueError("States and projectors must be finite")
    selected = frozen_projectors.detach()[edit_slot.to(recipient_state.device)]
    difference = donor_state.detach() - recipient_state
    raw = torch.bmm(difference[:, None, :], selected).squeeze(1)
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

