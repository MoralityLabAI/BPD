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


def _sudoku_groups_stacked(flat: Tensor, size: int) -> Tensor:
    """Return stacked ``[batch, 3, size, size]`` rows, columns, and boxes."""
    box = math.isqrt(size)
    grid = flat.reshape(-1, size, size)
    boxes = grid.reshape(-1, box, box, box, box).permute(0, 1, 3, 2, 4).reshape(-1, size, size)
    return torch.stack((grid, grid.transpose(1, 2), boxes), dim=1)


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
    ordered = _sudoku_groups_stacked(givens, size).sort(dim=-1).values
    duplicate = (ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] != 0)
    if bool(duplicate.any()):
        raise ValueError("Givens contain conflicting row, column, or box clues")
    return size


def _valid_mask(pred_digits: Tensor, givens: Tensor, size: int) -> Tensor:
    _integer_matrix(pred_digits, "pred_digits")
    if pred_digits.shape != givens.shape or pred_digits.device != givens.device:
        raise ValueError("Predictions and givens must share shape and device")
    expected = torch.arange(1, size + 1, device=pred_digits.device, dtype=pred_digits.dtype)
    clue_ok = ((givens == 0) | (pred_digits == givens)).all(dim=-1)
    ordered = _sudoku_groups_stacked(pred_digits, size).sort(dim=-1).values
    groups_ok = (ordered == expected).all(dim=-1).all(dim=-1).all(dim=-1)
    return clue_ok & groups_ok


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
        # Non-persistent runtime caches (never alter state_dict keys).
        self._cache_sig: tuple | None = None
        self._cached_u_norm: Tensor | None = None
        self._cached_v_ctrl: Tensor | None = None
        self._cached_u_ctrl: Tensor | None = None
        self._external_mask_queue: list[Tensor] | None = None
        self._telemetry_enabled: bool = False
        self._telemetry_counts: Tensor | None = None
        self._telemetry_token_calls: int = 0
        self._telemetry_active_sum: int = 0

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

    def _ensure_eval_cache(self) -> tuple[Tensor, Tensor, Tensor]:
        sig = (
            self.V.data_ptr(), self.V._version,
            self.U.data_ptr(), self.U._version,
            self.control_indices.data_ptr(), self.control_indices._version,
            self.V.device, self.V.dtype,
        )
        if self._cache_sig != sig:
            with torch.no_grad():
                self._cached_u_norm = self.U.detach().norm(dim=-1)
                idx = self.control_indices
                self._cached_v_ctrl = self.V.detach().index_select(1, idx).contiguous()
                self._cached_u_ctrl = self.U.detach().index_select(0, idx).contiguous()
                self._cache_sig = sig
        return self._cached_u_norm, self._cached_v_ctrl, self._cached_u_ctrl  # type: ignore[return-value]

    def reset_routing_telemetry(self, enabled: bool = True) -> None:
        self._telemetry_enabled = bool(enabled)
        self._telemetry_counts = torch.zeros(self.C, dtype=torch.int64, device=self.V.device) if enabled else None
        self._telemetry_token_calls = 0
        self._telemetry_active_sum = 0

    def routing_telemetry(self) -> dict:
        if self._telemetry_counts is None:
            return {"enabled": False}
        counts = self._telemetry_counts.detach().cpu()
        total_tok = max(1, self._telemetry_token_calls)
        probs = counts.double() / max(1.0, float(counts.sum().item()))
        nz = probs[probs > 0]
        entropy = float(-(nz * nz.log()).sum().item()) if nz.numel() else 0.0
        return {
            "enabled": True,
            "components": self.C,
            "declared_top_k": self.top_k,
            "mode": self.mode,
            "token_positions_routed": self._telemetry_token_calls,
            "mean_active_components_per_token": float(self._telemetry_active_sum / total_tok),
            "unique_active_components": int((counts > 0).sum().item()),
            "active_component_fraction": float((counts > 0).float().mean().item()),
            "effective_component_entropy_rank": float(math.exp(entropy)) if nz.numel() else 0.0,
            "component_activation_counts": counts.tolist(),
        }

    def _record_selected(self, selected: Tensor, n_tokens: int) -> None:
        if not self._telemetry_enabled or self._telemetry_counts is None:
            return
        with torch.no_grad():
            flat = selected.reshape(-1)
            self._telemetry_counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int64))
            self._telemetry_token_calls += int(n_tokens)
            self._telemetry_active_sum += int(flat.numel())

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError("Input must have the factor input width and floating dtype")
        xf = x.to(dtype=self.V.dtype)
        need_grad = torch.is_grad_enabled() and (self.V.requires_grad or self.U.requires_grad or xf.requires_grad)
        n_tokens = int(xf.numel() // self.in_features)
        if self._external_mask_queue is not None:
            if not self._external_mask_queue:
                raise RuntimeError("External routing mask queue exhausted during forward")
            ext_selected = self._external_mask_queue.pop(0)
            acts = xf @ self.V
            mask = torch.zeros_like(acts).scatter_(-1, ext_selected, 1)
            output = (acts * mask) @ self.U
            self._record_selected(ext_selected, n_tokens)
        elif self.top_k == self.C:
            output = (xf @ self.V) @ self.U
            if self._telemetry_enabled and self._telemetry_counts is not None:
                with torch.no_grad():
                    self._telemetry_counts.add_(n_tokens)
                    self._telemetry_token_calls += n_tokens
                    self._telemetry_active_sum += n_tokens * self.C
        elif self.mode == "online":
            acts = xf @ self.V
            u_norm = self.U.norm(dim=-1) if need_grad else self._ensure_eval_cache()[0]
            scores = acts.abs() * u_norm
            selected = scores.topk(self.top_k, dim=-1).indices
            mask = torch.zeros_like(acts).scatter_(-1, selected, 1)
            output = (acts * mask) @ self.U
            self._record_selected(selected, n_tokens)
        else:
            if need_grad:
                indices = self.control_indices
                output = (xf @ self.V[:, indices]) @ self.U[indices]
            else:
                _, v_ctrl, u_ctrl = self._ensure_eval_cache()
                output = (xf @ v_ctrl) @ u_ctrl
            if self._telemetry_enabled:
                expanded = self.control_indices.expand(n_tokens, self.top_k)
                self._record_selected(expanded, n_tokens)
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


def principal_angles_deg(basis_a: Tensor, basis_b: Tensor) -> dict:
    """Compute canonical principal angles (degrees) and Frobenius subspace overlap."""
    if basis_a.ndim != 2 or basis_b.ndim != 2 or basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError("Bases must be 2D matrices sharing ambient state dimension")
    if basis_a.shape[1] == 0 or basis_b.shape[1] == 0:
        return {
            "rank_a": int(basis_a.shape[1]),
            "rank_b": int(basis_b.shape[1]),
            "cosines": [],
            "angles_deg": [],
            "min_angle_deg": 90.0,
            "mean_angle_deg": 90.0,
            "max_angle_deg": 90.0,
            "subspace_overlap": 0.0,
        }
    a64 = basis_a.detach().double()
    b64 = basis_b.detach().double()
    for tag, mat in (("basis_a", a64), ("basis_b", b64)):
        eye = torch.eye(mat.shape[1], device=mat.device, dtype=torch.float64)
        if float((mat.T @ mat - eye).abs().max()) > 1e-4:
            raise ValueError(f"{tag} must have orthonormal columns")
    svals = torch.linalg.svdvals(a64.T @ b64).clamp(0.0, 1.0)
    angles = [float(math.degrees(math.acos(float(sv.item())))) for sv in svals]
    overlap = float(svals.pow(2).mean().item())
    return {
        "rank_a": int(a64.shape[1]),
        "rank_b": int(b64.shape[1]),
        "cosines": svals.detach().cpu().tolist(),
        "angles_deg": angles,
        "min_angle_deg": min(angles),
        "mean_angle_deg": sum(angles) / len(angles),
        "max_angle_deg": max(angles),
        "subspace_overlap": overlap,
    }


def _jvp_columns_loop(step: Callable[[Tensor], Tensor], state: Tensor, basis: Tensor, work_dtype: torch.dtype) -> Tensor:
    derivatives = []
    for index in range(basis.shape[1]):
        value, derivative = torch.func.jvp(step, (state,), (basis[:, index].reshape_as(state),))
        if not isinstance(value, Tensor) or value.shape != state.shape or derivative.shape != state.shape:
            raise ValueError("Recurrent step must preserve state shape")
        if not bool(torch.isfinite(value).all() & torch.isfinite(derivative).all()):
            raise ValueError("Transport produced a nonfinite state or derivative")
        derivatives.append(derivative.detach().reshape(-1).to(dtype=work_dtype))
    return torch.stack(derivatives, dim=1)


def _jvp_columns_chunked_vmap(
    step: Callable[[Tensor], Tensor],
    state: Tensor,
    basis: Tensor,
    work_dtype: torch.dtype,
    chunk_size: int,
) -> Tensor:
    primal = step(state)
    if not isinstance(primal, Tensor) or primal.shape != state.shape:
        raise ValueError("Recurrent step must preserve state shape")
    if not bool(torch.isfinite(primal).all()):
        raise ValueError("Transport produced a nonfinite state or derivative")

    def single_jvp(col: Tensor) -> Tensor:
        val, deriv = torch.func.jvp(step, (state,), (col.reshape_as(state),))
        if not isinstance(val, Tensor) or val.shape != state.shape or deriv.shape != state.shape:
            raise ValueError("Recurrent step must preserve state shape")
        return deriv.reshape(-1)

    batched = torch.func.vmap(single_jvp, in_dims=1, out_dims=1)
    chunks = []
    for offset in range(0, basis.shape[1], chunk_size):
        sub = basis[:, offset:offset + chunk_size]
        derivs = batched(sub)
        if not bool(torch.isfinite(derivs).all()):
            raise ValueError("Transport produced a nonfinite state or derivative")
        chunks.append(derivs.detach().to(dtype=work_dtype))
    return torch.cat(chunks, dim=1)


def transport_basis_jvp(
    step: Callable[[Tensor], Tensor],
    state: Tensor,
    basis: Tensor,
    *,
    project: Callable[[Tensor], Tensor] | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    chunk_size: int = 8,
    mode: str = "auto",
) -> tuple[Tensor, dict]:
    """Transport one case's orthonormal basis through one local recurrent step.

    ``basis`` is ``[state.numel(), rank]``; optional ``project`` maps rows of
    vectors back to the permitted edit space. Thin SVD reveals collapse instead
    of filling lost directions with arbitrary QR columns. A returned rank loss
    requires a matched-rank control or a failed arm in subsequent studies.
    This instrument establishes local derivatives, not causal task improvement.
    """
    _check_tolerances(rtol, atol)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if mode not in {"auto", "vmap", "loop"}:
        raise ValueError("mode must be auto, vmap, or loop")
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
    executed_mode = mode
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        if mode == "loop" or basis.shape[1] == 1:
            transported = _jvp_columns_loop(step, state, basis, work_dtype)
            executed_mode = "loop"
        elif mode == "vmap":
            transported = _jvp_columns_chunked_vmap(step, state, basis, work_dtype, chunk_size)
            executed_mode = "vmap"
        else:
            try:
                transported = _jvp_columns_chunked_vmap(step, state, basis, work_dtype, chunk_size)
                executed_mode = "vmap"
            except RuntimeError:
                transported = _jvp_columns_loop(step, state, basis, work_dtype)
                executed_mode = "loop"
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
    drift = principal_angles_deg(columns, result.to(dtype=work_dtype)) if rank > 0 else principal_angles_deg(columns, result.to(dtype=work_dtype))
    return result, {
        "input_rank": int(basis.shape[1]),
        "output_rank": rank,
        "rank_lost": int(basis.shape[1]) - rank,
        "rank_threshold": threshold,
        "singular_values": singular.detach().cpu().tolist(),
        "input_orthogonality_max_error": orth_error,
        "one_step_closure_defect": closure,
        "protected_projection_escape": escape,
        "principal_angle_drift_from_input": drift,
        "jvp_execution_mode": executed_mode,
        "jvp_chunk_size": chunk_size,
        "evidence_class": "local derivative instrument; causal benefit unmeasured",
        "dense_state_jacobian_allocated": False,
    }
