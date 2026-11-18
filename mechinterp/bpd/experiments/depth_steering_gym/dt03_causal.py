"""DT03 source-only causal repairs for the archived DT01/DT02 depth gym.

This module leaves both first-run source locks and protected evaluation roots
untouched. Every causal edit is measured in the full physical recurrent state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch


def flat_observer_jacobian(
    observable: Callable[[torch.Tensor], torch.Tensor],
    states: torch.Tensor,
) -> torch.Tensor:
    """Return [batch, every_output_coordinate, entire_state_coordinate].

    Cross-example derivatives are rejected; the entire native task tensor is
    differentiated, rather than summing normalized softmax classes to zero.
    """
    if states.ndim < 2 or not torch.isfinite(states).all():
        raise ValueError("finite batch of full physical states required")
    B = states.shape[0]
    flat_input_size = states[0].numel()
    x = states.detach().clone().requires_grad_(True)
    y = observable(x)
    if y.ndim < 2 or y.shape[0] != B or not torch.isfinite(y).all():
        raise ValueError("observer must return finite [batch, ...] outputs")
    target = y.reshape(B, -1)
    rows = []
    for b in range(B):
        batch_rows = []
        for j in range(target.shape[1]):
            (grad,) = torch.autograd.grad(
                target[b, j], x,
                retain_graph=True, allow_unused=True,
            )
            if grad is None:
                raise ValueError("observer detached from the physical state")
            others = grad.detach().clone()
            others[b] = 0
            if torch.max(torch.abs(others)) > 1e-7:
                raise ValueError("batch examples coupled inside causal observer")
            batch_rows.append(grad[b].reshape(-1))
        rows.append(torch.stack(batch_rows))
    J = torch.stack(rows)
    if J.shape != (B, target.shape[1], flat_input_size):
        raise AssertionError("full output/state Jacobian shape mismatch")
    if not torch.isfinite(J).all():
        raise ValueError("nonfinite task Jacobian")
    return J.detach()


def physical_task_gramian(jacobian: torch.Tensor) -> torch.Tensor:
    if jacobian.ndim != 3 or not torch.isfinite(jacobian).all():
        raise ValueError("finite [root, output, state] Jacobian required")
    return torch.einsum("bmd,bme->de", jacobian.double(), jacobian.double()) / len(jacobian)


def select_full_coordinate_block(
    continuation: Callable[[torch.Tensor, int], torch.Tensor],
    task: Callable[[torch.Tensor], torch.Tensor],
    fit_states: torch.Tensor,
    protected_basis: torch.Tensor,
    fit_depths: tuple[int, ...],
    rank: int,
) -> tuple[torch.Tensor, dict]:
    """Train-only physical task Gramian using every native output coordinate."""
    if not fit_depths or len(set(fit_depths)) != len(fit_depths):
        raise ValueError("invalid fit-depth roster")
    D = fit_states[0].numel()
    if protected_basis.ndim != 2 or protected_basis.shape[0] != D:
        raise ValueError("protected basis belongs to flattened full state")
    if rank < 1 or rank > protected_basis.shape[1]:
        raise ValueError("invalid physical block rank")
    Q = protected_basis.double()
    G = torch.zeros(Q.shape[1], Q.shape[1], dtype=torch.float64)
    for depth in fit_depths:
        J = flat_observer_jacobian(
            lambda h: task(continuation(h, depth)), fit_states
        ).double()
        reduced = J @ Q
        G += physical_task_gramian(reduced)
    G /= len(fit_depths)
    eigvals, eigvecs = torch.linalg.eigh(G)
    block = (Q @ eigvecs[:, -rank:]).float()
    if float((block.T @ block - torch.eye(rank)).abs().max()) > 1e-4:
        raise AssertionError("physical block lost orthogonality")
    return block, {"fit_depths": list(fit_depths), "rank": rank,
                   "eigenvalues": eigvals.flip(0).tolist(),
                   "full_output_coordinates": int(J.shape[1]),
                   "full_state_coordinates": D}


def whole_case_recovery(candidate: torch.Tensor, donor: torch.Tensor,
                        threshold: float) -> dict:
    """Every coordinate of a case must recover to pass that case."""
    if candidate.shape != donor.shape or candidate.ndim < 2:
        raise ValueError("paired case/output tensors required")
    if threshold < 0 or not torch.isfinite(candidate).all() or not torch.isfinite(donor).all():
        raise ValueError("invalid recovery tolerance or nonfinite response")
    diff = (candidate - donor).abs().reshape(candidate.shape[0], -1)
    recovered = diff.amax(-1) <= threshold
    return {
        "independent_case_count": len(diff),
        "whole_case_recovery": float(recovered.float().mean()),
        "coordinate_recovery": float((diff <= threshold).float().mean()),
        "per_case_recovered": recovered.tolist(),
        "mean_squared_response_error": float(diff.square().mean()),
    }


def validate_delivered_sham(edit: torch.Tensor, control: torch.Tensor,
                            physical_basis: torch.Tensor, jac: torch.Tensor,
                            tolerance: float = 1e-5) -> dict:
    """Verify full-state physical dose, protected span and signed response.

    edit/control: [batch, full_state_dim], jac: [batch, output, full_state_dim].
    The full task Jacobian must match, not a scalar surrogate.
    """
    if edit.shape != control.shape or edit.ndim != 2:
        raise ValueError("invalid full physical edit/control shape")
    B, D = edit.shape
    if physical_basis.ndim != 2 or physical_basis.shape[0] != D:
        raise ValueError("wrong protected physical basis")
    if jac.ndim != 3 or jac.shape[0] != B or jac.shape[2] != D:
        raise ValueError("wrong full signed task Jacobian shape")
    if not all(torch.isfinite(z).all() for z in (edit,control,physical_basis,jac)):
        raise ValueError("nonfinite sham or physical observer")
    Q = physical_basis.double()
    P = Q @ Q.T
    e, c = edit.double(), control.double()
    span_error = float(torch.maximum(
        (e - e@P).abs().max(), (c-c@P).abs().max()
    ))
    dose_error = float((torch.linalg.vector_norm(e,dim=-1) -
                        torch.linalg.vector_norm(c,dim=-1)).abs().max())
    signed_error = float((torch.einsum("bmd,bd->bm", jac.double(), e-c)).abs().max())
    if span_error > tolerance or dose_error > tolerance or signed_error > tolerance:
        raise AssertionError("sham changes physical span, dose or signed future-task effect")
    separations = torch.linalg.vector_norm(e-c,dim=-1)
    return {"span_error":span_error, "delivered_l2_error":dose_error,
            "full_signed_jacobian_error":signed_error,
            "max_candidate_sham_separation":float(separations.max()),
            "distinct_contrast_available":bool((separations > tolerance).any())}


def output_weight_basis(weight: torch.Tensor, rank: int) -> torch.Tensor:
    """Physical output-side singular directions of y=W x."""
    if weight.ndim != 2 or rank < 1 or rank > min(weight.shape):
        raise ValueError("invalid output-weight SVD request")
    left, _, _ = torch.linalg.svd(weight.double(), full_matrices=False)
    return left[:, :rank].float()


@dataclass(frozen=True)
class DualState:
    z_h: torch.Tensor
    z_l: torch.Tensor

    def flatten(self) -> torch.Tensor:
        if self.z_h.shape != self.z_l.shape or self.z_h.ndim < 2:
            raise ValueError("H/L state shape mismatch")
        return torch.cat((self.z_h.flatten(1), self.z_l.flatten(1)), dim=-1)

    @classmethod
    def unflatten(cls, packed: torch.Tensor, shape: tuple[int, ...]) -> "DualState":
        per = 1
        for dim in shape:
            per *= dim
        if packed.ndim != 2 or packed.shape[1] != 2*per:
            raise ValueError("incorrect full H/L packing")
        return cls(packed[:, :per].reshape(-1,*shape),
                   packed[:, per:].reshape(-1,*shape))


def native_resume_recipient(
    model, recipient_tokens: torch.Tensor,
    receiver_state: DualState, depth: int,
) -> torch.Tensor:
    """Continue *receiver context* with explicitly supplied complete H/L state."""
    if depth < 0:
        raise ValueError("negative future depth")
    emb = model.encode_inputs(recipient_tokens)
    h, l = receiver_state.z_h, receiver_state.z_l
    for _ in range(depth):
        h, l = model.act_step(h, l, emb)
    return model.digit_logits(h)
