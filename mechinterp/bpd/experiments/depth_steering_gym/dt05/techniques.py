"""Gold-free stopping, autonomous sparse factors, and bounded transport instruments.

These are development instruments. Factor routing uses the current compressed
activation; it never asks a dense teacher for masks at inference. Geometry uses
thin matrices and Jacobian-vector products, with explicit rank-loss receipts.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn


def _integer_matrix(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, cells]")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"{name} must contain integer digits")
    if value.shape[0] < 1:
        raise ValueError(f"{name} batch must be nonempty")


def _sudoku_groups(flat: Tensor, size: int) -> tuple[Tensor, Tensor, Tensor]:
    box = math.isqrt(size)
    grid = flat.reshape(-1, size, size)
    boxes = grid.reshape(-1, box, box, box, box).permute(0, 1, 3, 2, 4)
    return grid, grid.transpose(1, 2), boxes.reshape(-1, size, size)


def _validate_givens(givens: Tensor) -> int:
    _integer_matrix(givens, "givens")
    size = math.isqrt(givens.shape[1])
    if size not in (4, 9) or size * size != givens.shape[1]:
        raise ValueError("Only flat 4x4 and 9x9 Sudoku boards are supported")
    if bool(((givens < 0) | (givens > size)).any()):
        raise ValueError(f"Givens must be blank 0 or digits 1..{size}")
    for group in _sudoku_groups(givens, size):
        ordered = group.sort(dim=-1).values
        duplicate = (ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] != 0)
        if bool(duplicate.any()):
            raise ValueError("Givens contain conflicting row, column, or box clues")
    return size


def _valid_mask(pred_digits: Tensor, givens: Tensor, size: int) -> Tensor:
    _integer_matrix(pred_digits, "pred_digits")
    if pred_digits.shape != givens.shape or pred_digits.device != givens.device:
        raise ValueError("Predictions and givens must share shape and device")
    expected = torch.arange(1, size + 1, device=pred_digits.device)
    valid = ((givens == 0) | (pred_digits == givens)).all(dim=-1)
    for group in _sudoku_groups(pred_digits, size):
        valid = valid & (group.sort(dim=-1).values == expected).all(dim=-1).all(dim=-1)
    return valid


def sudoku_valid_mask(pred_digits: Tensor, givens_digits: Tensor) -> Tensor:
    """Check complete solutions against public clues, rows, columns and boxes.

    Inputs are flat ``[B,16]`` or ``[B,81]``, using real digits ``1..N`` and
    public blank ``0``. There is no gold-solution argument. Invalid prediction
    digits produce False; malformed or contradictory public clues raise.
    """
    return _valid_mask(pred_digits, givens_digits, _validate_givens(givens_digits))


class FirstValidStopper:
    """Retain each board's earliest valid candidate using only public clues.

    The caller owns recurrent execution and can compact active cases using
    ``valid_found``. Calling this on a complete trajectory preserves a raw-final
    diagnostic, but does not itself reduce model compute. Unsolved cases use
    the final candidate and final depth. Unique-puzzle admission belongs to the
    dataset loader; validity alone does not certify agreement with a gold label
    for an ambiguous puzzle.
    """

    def __init__(self, givens_digits: Tensor):
        self.size = _validate_givens(givens_digits)
        self.givens = givens_digits.detach().clone()
        self.valid_found = torch.zeros(self.givens.shape[0], dtype=torch.bool, device=self.givens.device)
        self.selected_depths = torch.zeros(self.givens.shape[0], dtype=torch.long, device=self.givens.device)
        self.predictions = torch.zeros_like(self.givens)
        self._last_depth = 0
        self._last_prediction: Tensor | None = None

    def update(self, pred_digits: Tensor, depth: int) -> Tensor:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= self._last_depth:
            raise ValueError("Observed depths must be positive and strictly increasing")
        valid = _valid_mask(pred_digits, self.givens, self.size)
        newly_valid = valid & ~self.valid_found
        self.predictions[newly_valid] = pred_digits.detach()[newly_valid]
        self.selected_depths[newly_valid] = depth
        self.valid_found |= newly_valid
        self._last_depth = depth
        self._last_prediction = pred_digits.detach().clone()
        return newly_valid

    def finish(self, final_pred: Tensor, final_depth: int) -> dict[str, Tensor]:
        if isinstance(final_depth, bool) or not isinstance(final_depth, int) or final_depth < self._last_depth:
            raise ValueError("Final depth cannot precede the latest observation")
        if final_depth > self._last_depth:
            self.update(final_pred, final_depth)
        elif self._last_prediction is None or not torch.equal(final_pred, self._last_prediction):
            raise ValueError("The final candidate must match the last observed candidate")
        predictions = torch.where(self.valid_found[:, None], self.predictions, final_pred)
        depths = torch.where(self.valid_found, self.selected_depths, final_depth)
        return {
            "predictions": predictions.detach().clone(),
            "selected_depths": depths.detach().clone(),
            "valid_found": self.valid_found.detach().clone(),
        }


class OnlineTopKLinear(nn.Module):
    """Factor-only ``(x V) U`` with live, fixed, or seeded-random component use.

    ``online`` scores the current input with ``abs(x V) * norm(U)`` each call.
    ``fixed`` uses the largest ``norm(V_column) * norm(U_row)`` factors; after
    SVD initialization these are the leading singular components. ``random``
    uses a seeded fixed subset. Every mode retains the same factors for fair
    comparisons. No dense target weight or residual is stored or evaluated.

    The online path still performs dense small-C matrix multiplication after
    masking. Active-component sparsity is not a promise of wall-clock speedup.
    """

    def __init__(self, V: Tensor, U: Tensor, top_k: int, *, bias: Tensor | None = None,
                 mode: str = "online", seed: int = 0):
        super().__init__()
        if V.ndim != 2 or U.ndim != 2 or V.shape[1] != U.shape[0] or min(*V.shape, *U.shape) < 1:
            raise ValueError("Factors must have shapes [input, C] and [C, output]")
        if not V.is_floating_point() or not U.is_floating_point() or V.dtype != U.dtype or V.device != U.device:
            raise ValueError("Factors must share floating dtype and device")
        if not bool(torch.isfinite(V).all() & torch.isfinite(U).all()):
            raise ValueError("Factors must be finite")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= V.shape[1]:
            raise ValueError("top_k must lie in 1..C")
        if mode not in {"online", "fixed", "random"}:
            raise ValueError("mode must be online, fixed, or random")
        if bias is not None and (bias.shape != (U.shape[1],) or not bool(torch.isfinite(bias).all())):
            raise ValueError("Bias must be a finite output-width vector")
        self.V = nn.Parameter(V.detach().clone())
        self.U = nn.Parameter(U.detach().clone())
        self.bias = nn.Parameter(bias.detach().to(device=V.device, dtype=V.dtype).clone()) if bias is not None else None
        self.in_features, self.out_features = V.shape[0], U.shape[1]
        self.C, self.top_k, self.mode = V.shape[1], top_k, mode
        strength = V.detach().float().norm(dim=0) * U.detach().float().norm(dim=1)
        if mode == "random":
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            indices = torch.randperm(self.C, generator=generator)[:top_k].to(V.device)
        else:
            indices = strength.argsort(descending=True, stable=True)[:top_k]
        self.register_buffer("control_indices", indices)

    @classmethod
    def from_factors(cls, V: Tensor, U: Tensor, top_k: int, *, bias: Tensor | None = None,
                     mode: str = "online", seed: int = 0) -> "OnlineTopKLinear":
        return cls(V, U, top_k, bias=bias, mode=mode, seed=seed)

    @classmethod
    def from_linear(cls, linear: nn.Linear, components: int, top_k: int, *,
                    mode: str = "online", seed: int = 0) -> "OnlineTopKLinear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("from_linear expects nn.Linear")
        if isinstance(components, bool) or not isinstance(components, int) or not 1 <= components <= min(linear.weight.shape):
            raise ValueError("components must lie in 1..min(input, output)")
        left, singular, right = torch.linalg.svd(linear.weight.detach().double().cpu(), full_matrices=False)
        scale = singular[:components].sqrt()
        V = (right[:components].T * scale).to(linear.weight)
        U = (scale[:, None] * left[:, :components].T).to(linear.weight)
        return cls(V, U, top_k, bias=linear.bias, mode=mode, seed=seed)

    @classmethod
    def from_tied(cls, wrapper: nn.Module, top_k: int, *, mode: str = "online", seed: int = 0) -> "OnlineTopKLinear":
        if not isinstance(getattr(wrapper, "V", None), Tensor) or not isinstance(getattr(wrapper, "U", None), Tensor):
            raise TypeError("Expected a factor wrapper with V and U tensors")
        return cls(wrapper.V, wrapper.U, top_k, bias=getattr(wrapper, "bias", None), mode=mode, seed=seed)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError("Input must have the factor input width and floating dtype")
        xf = x.to(dtype=self.V.dtype)
        if self.top_k == self.C:
            output = (xf @ self.V) @ self.U
        elif self.mode == "online":
            acts = xf @ self.V
            scores = acts.abs() * self.U.norm(dim=-1)
            selected = scores.topk(self.top_k, dim=-1).indices
            mask = torch.zeros_like(acts).scatter_(-1, selected, 1)
            output = (acts * mask) @ self.U
        else:
            indices = self.control_indices
            output = (xf @ self.V[:, indices]) @ self.U[indices]
        if self.bias is not None:
            output = output + self.bias
        return output.to(dtype=x.dtype)


def replace_online_factors(model: nn.Module, module_names: Sequence[str], components: int,
                           top_k: int, *, mode: str = "online", seed: int = 0) -> dict[str, OnlineTopKLinear]:
    """Replace named linears after validating every replacement; return wrappers."""
    if isinstance(module_names, str) or not module_names or len(set(module_names)) != len(module_names):
        raise ValueError("Provide a nonempty sequence of unique module names")
    replacements = {}
    for index, name in enumerate(module_names):
        if not name or name.startswith(".") or name.endswith("."):
            raise ValueError("Module names must be valid nonempty child paths")
        original = model.get_submodule(name)
        if isinstance(original, nn.Linear):
            wrapper = OnlineTopKLinear.from_linear(original, components, top_k, mode=mode, seed=seed + index)
        else:
            if getattr(original, "C", None) != components:
                raise ValueError("Existing factor width must match components")
            wrapper = OnlineTopKLinear.from_tied(original, top_k, mode=mode, seed=seed + index)
        replacements[name] = wrapper
    for name, wrapper in replacements.items():
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child_name, wrapper)
    return replacements


class HeadNullProjector:
    """Project against a thin head row-space basis, storing at most vocab x width.

    The exact head kernel is stronger than preserving softmax up to a common
    logit shift. Floating arithmetic still requires measuring delivered error.
    """

    def __init__(self, head_weight: Tensor, *, rtol: float = 1e-7, atol: float = 1e-10):
        _check_tolerances(rtol, atol)
        if head_weight.ndim != 2 or not head_weight.is_floating_point() or min(head_weight.shape) < 1:
            raise ValueError("Head must be a nonempty floating matrix")
        if not bool(torch.isfinite(head_weight).all()):
            raise ValueError("Head must be finite")
        _, singular, right = torch.linalg.svd(head_weight.detach().double().cpu(), full_matrices=False)
        self.threshold = atol + rtol * float(singular.max())
        self.rank = int((singular > self.threshold).sum())
        work_dtype = torch.float64 if head_weight.dtype == torch.float64 else torch.float32
        self.row_basis = right[:self.rank].T.to(device=head_weight.device, dtype=work_dtype).contiguous()
        self.width = head_weight.shape[1]

    def __call__(self, vectors: Tensor) -> Tensor:
        if vectors.shape[-1] != self.width or not vectors.is_floating_point():
            raise ValueError("Projection vectors must have the floating head width")
        basis = self.row_basis.to(device=vectors.device)
        work = vectors.to(dtype=basis.dtype)
        return (work - (work @ basis) @ basis.T).to(dtype=vectors.dtype)

    def project_dual(self, packed: Tensor, token_count: int, prefix_len: int) -> Tensor:
        """Project packed H/L edits, locking both prefixes and protecting H readout."""
        if not isinstance(token_count, int) or not isinstance(prefix_len, int) or not 0 <= prefix_len <= token_count:
            raise ValueError("Require 0 <= prefix_len <= token_count")
        if token_count < 1 or packed.shape[-1] != 2 * token_count * self.width:
            raise ValueError("Packed dual state has the wrong width")
        dual = packed.reshape(*packed.shape[:-1], 2, token_count, self.width)
        high = self(dual[..., 0, :, :])
        low = dual[..., 1, :, :].clone()
        high[..., :prefix_len, :] = 0
        low[..., :prefix_len, :] = 0
        return torch.stack((high, low), dim=-3).reshape_as(packed)


def _check_tolerances(rtol: float, atol: float) -> None:
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("Tolerances must be finite and nonnegative")


def transport_basis_jvp(step: Callable[[Tensor], Tensor], state: Tensor, basis: Tensor, *,
                        project: Callable[[Tensor], Tensor] | None = None,
                        rtol: float = 1e-6, atol: float = 1e-9) -> tuple[Tensor, dict]:
    """Transport one case's orthonormal basis through one local recurrent step.

    ``basis`` is ``[state.numel(), rank]``; optional ``project`` maps rows of
    vectors back to the permitted edit space. Thin SVD reveals collapse instead
    of filling lost directions with arbitrary QR columns. A returned rank loss
    requires a matched-rank control or a failed arm in subsequent studies.
    This instrument establishes local derivatives, not causal task improvement.
    """
    _check_tolerances(rtol, atol)
    if not state.is_floating_point() or basis.ndim != 2 or not basis.is_floating_point():
        raise ValueError("State and rank-2 basis must be floating tensors")
    if basis.shape[0] != state.numel() or not 1 <= basis.shape[1] <= state.numel():
        raise ValueError("Basis rows must match state.numel() with nonempty rank")
    if state.device != basis.device or state.dtype != basis.dtype:
        raise ValueError("State and basis must share device and dtype")
    if not bool(torch.isfinite(state).all() & torch.isfinite(basis).all()):
        raise ValueError("State and basis must be finite")
    work_dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
    columns = basis.to(dtype=work_dtype)
    identity = torch.eye(basis.shape[1], device=state.device, dtype=work_dtype)
    orth_error = float(((columns.T @ columns) - identity).abs().max())
    if orth_error > 1e-4:
        raise ValueError("Input transport basis must be orthonormal")
    derivatives = []
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for index in range(basis.shape[1]):
            value, derivative = torch.func.jvp(step, (state,), (basis[:, index].reshape_as(state),))
            if not isinstance(value, Tensor) or value.shape != state.shape or derivative.shape != state.shape:
                raise ValueError("Recurrent step must preserve state shape")
            if not bool(torch.isfinite(value).all() & torch.isfinite(derivative).all()):
                raise ValueError("Transport produced a nonfinite state or derivative")
            derivatives.append(derivative.detach().reshape(-1).to(dtype=work_dtype))
    transported = torch.stack(derivatives, dim=1)
    denom = transported.norm().clamp_min(torch.finfo(work_dtype).tiny)
    closure = float((transported - columns @ (columns.T @ transported)).norm() / denom)
    projected = transported if project is None else project(transported.T).T
    if projected.shape != transported.shape or not bool(torch.isfinite(projected).all()):
        raise ValueError("Projection must preserve finite transported-vector shape")
    escape = float((transported - projected).norm() / denom)
    left, singular, _ = torch.linalg.svd(projected, full_matrices=False)
    threshold = atol + rtol * float(singular.max())
    rank = int((singular > threshold).sum())
    result = left[:, :rank].to(dtype=basis.dtype)
    return result, {
        "input_rank": int(basis.shape[1]),
        "output_rank": rank,
        "rank_lost": int(basis.shape[1]) - rank,
        "rank_threshold": threshold,
        "singular_values": singular.detach().cpu().tolist(),
        "input_orthogonality_max_error": orth_error,
        "one_step_closure_defect": closure,
        "protected_projection_escape": escape,
        "evidence_class": "local derivative instrument; causal benefit unmeasured",
        "dense_state_jacobian_allocated": False,
    }
