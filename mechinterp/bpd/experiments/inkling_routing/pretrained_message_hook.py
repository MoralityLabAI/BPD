"""INK-BPD-05: hook the *real* Inkling2LoopMessageAttentionResidual input.

Source-pinned to Hugging Face transformers 046bdb8 (Nov 21, 2026).
The model's message_input receives previous-pass block-2 state and the final
block-3 state. Alter the latter at its reader boundary, for all 12 readers
of a chosen pass. Each reader then performs its native RMSNorm, attention,
message-gate integration, and full subsequent transformer continuation.

A repeated per-layer read of a single edited block state is one physical
intervention per task root, not twelve independent intervention units.
This module makes no claims about pretrained weights until a separate
source-locked development run actually executes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

UPSTREAM_TRANSFORMERS_COMMIT = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
UPSTREAM_MODELING_BLOB = "78498834e2806e83a1bb96eea5cf5447d9b1531f"


@dataclass(frozen=True)
class PatchSpec:
    arm: str = "candidate"
    passes: tuple[int, ...] = (1,)
    coordinate: int = 2027
    dose_fraction: float = 0.08
    seed: int = 504271
    num_loop_layers: int = 12
    total_passes: int = 3
    probe_layer: int = 11
    token_index: int = -1

    def __post_init__(self) -> None:
        accepted = {"zero", "candidate"} | {
            f"{family}_{i}" for family in ("coord", "haar") for i in range(4)
        }
        if self.arm not in accepted:
            raise ValueError(f"unsupported physical arm: {self.arm}")
        if not self.passes or len(set(self.passes)) != len(self.passes):
            raise ValueError("pass selection must be nonempty and unique")
        if any(not (1 <= t < self.total_passes) for t in self.passes):
            raise ValueError("cross-loop message exists only after first pass")
        if not math.isfinite(self.dose_fraction) or not (0 < self.dose_fraction < 0.6):
            raise ValueError("physical dose_fraction must be in (0, 0.6)")
        if not 0 <= self.probe_layer < self.num_loop_layers:
            raise ValueError("invalid probe layer")
        if self.token_index != -1:
            raise ValueError("INK-BPD-05 is registered for last-token intervention only")
        if not (0 <= self.coordinate) or self.seed < 0:
            raise ValueError("invalid candidate coordinate or sham seed")


def direction_for_arm(x: Tensor, spec: PatchSpec) -> Tensor:
    """Frozen physical directions; source-selected candidate versus 8 shams."""
    width = x.shape[-1]
    if spec.coordinate >= width:
        raise ValueError("candidate coordinate exceeds checkpoint hidden width")
    if spec.arm in ("zero", "candidate"):
        v = torch.zeros(width, device=x.device, dtype=torch.float32)
        v[spec.coordinate] = 1
        return v
    family, index_s = spec.arm.rsplit("_", 1)
    i = int(index_s)
    if family == "coord":
        other = (spec.coordinate + 97 * (i + 1)) % width
        if other == spec.coordinate:
            raise ValueError("sham coordinate collision")
        v = torch.zeros(width, device=x.device, dtype=torch.float32)
        v[other] = 1
        return v
    g = torch.Generator(device="cpu").manual_seed(spec.seed + 4099 * i)
    return torch.randn(width, generator=g, dtype=torch.float32).to(x.device)


def fixed_chord_rotation(x: Tensor, direction: Tensor, fraction: float) -> tuple[Tensor, dict]:
    """Apply a tangent rotation at exact intended L2 = fraction * ||x||.

    The target and each sham see the same unpatched source at the first
    edited pass. Delivered physical metrics are recomputed *after* casting
    back to the checkpoint's dtype. A degenerate radial candidate is
    ineligible rather than silently substituted.
    """
    if x.ndim != 1 or direction.shape != x.shape:
        raise ValueError("expected single raw-state [hidden] vector and equal direction")
    if not math.isfinite(fraction) or not 0 < fraction < 0.6:
        raise ValueError("invalid fractional L2 dose")
    raw = x.float()
    radius = torch.linalg.vector_norm(raw)
    if not bool(torch.isfinite(radius).item()) or radius.item() <= 1e-10:
        raise ValueError("zero or nonfinite raw block state")
    v = direction.to(device=x.device, dtype=torch.float32)
    if not bool(torch.isfinite(v).all().item()):
        raise ValueError("nonfinite patch direction")
    tangent = v - (v * raw).sum() / radius.square() * raw
    tangent_norm = torch.linalg.vector_norm(tangent)
    if tangent_norm.item() < 1e-8:
        raise ValueError("candidate/sham direction degenerate after tangent projection")
    cosine = 1.0 - fraction * fraction / 2.0
    sine = fraction * math.sqrt(1.0 - fraction * fraction / 4.0)
    y = (cosine * raw + sine * radius * tangent / tangent_norm).to(dtype=x.dtype)
    actual = torch.linalg.vector_norm(y.float() - raw)
    ratio = actual / radius
    norm_ratio = torch.linalg.vector_norm(y.float()) / radius
    if abs(ratio.item() - fraction) > max(0.02 * fraction, 0.002):
        raise RuntimeError("cast-delivered L2 outside registered tolerance")
    if abs(norm_ratio.item() - 1) > 0.005:
        raise RuntimeError("cast-delivered RMS outside registered tolerance")
    return y, {
        "source_l2": radius.item(),
        "intended_l2": (radius * fraction).item(),
        "delivered_l2": actual.item(),
        "delivered_l2_fraction": ratio.item(),
        "delivered_norm_ratio": norm_ratio.item(),
        "tangent_source_cosine": ((tangent * raw).sum() / (radius * tangent_norm)).item(),
    }


def recompute_message(module, block2: Tensor, block3: Tensor, loop_layer_idx: int) -> tuple[Tensor, Tensor]:
    """Native 4-reader, 2-slot message from the inspected HF equations.

    Values have shape [4] beta3 and [4, hidden] message, both float32.
    The module's own RMSNorm is used so its epsilon matches the live model.
    """
    if block2.shape != block3.shape or block2.ndim != 1:
        raise ValueError("two raw block outputs must share [hidden] shape")
    if not hasattr(module, "query") or not hasattr(module, "norm"):
        raise ValueError("model reader missing query or norm")
    states = torch.stack([block2.float(), block3.float()], dim=0)
    normalized = module.norm(states).float()
    queries = module.query[loop_layer_idx].float()
    if tuple(queries.shape) != (4, states.shape[-1]):
        raise ValueError("not a 4-reader Inkling message module")
    scores = normalized @ queries.T / math.sqrt(states.shape[-1])
    weights = torch.softmax(scores, dim=0)
    return weights[1], weights.T @ normalized


def probe_coordinate_attribution(
    module, block2: Tensor, block3: Tensor, patched3: Tensor,
    loop_layer_idx: int, coordinate: int,
) -> dict:
    """Exact score-gap coordinate term and epsilon-aware local gate Jacobian.

    This is a local *reader* instrument. The full model continuation must
    independently establish task effects. The unweighted RMSNorm epsilon
    makes a radial derivative small but nonzero in principle; report it.
    """
    if not 0 <= coordinate < block3.numel():
        raise ValueError("outlier coordinate exceeds reader dimension")
    u = module.query[loop_layer_idx].float()  # [4, d]
    a, b, c = block2.float(), block3.float(), patched3.float()
    if a.ndim != 1 or a.shape != b.shape or b.shape != c.shape:
        raise ValueError("invalid cross-loop block-state shape")
    d = b.numel()
    eps = float(module.norm.eps)
    r3 = (b.square().sum() + d * eps).sqrt()
    an, bn = module.norm(a).float(), module.norm(b).float()
    gaps = u @ (bn - an) / math.sqrt(d)
    coordinate_gaps = u[:, coordinate] * (bn[coordinate] - an[coordinate]) / math.sqrt(d)
    beta_before, _ = recompute_message(module, a, b, loop_layer_idx)
    beta_after, _ = recompute_message(module, a, c, loop_layer_idx)
    dv = c - b
    scores3 = (u @ b) / r3
    grad = u / r3 - scores3[:, None] * b[None, :] / r3.square()
    predicted_score_shift = grad @ dv
    predicted_gate_shift = beta_before * (1 - beta_before) * predicted_score_shift
    actual_gate_shift = beta_after - beta_before
    return {
        "coordinate": coordinate,
        "block3_coordinate_raw": b[coordinate].item(),
        "block3_coordinate_rms_normalized": bn[coordinate].item(),
        "block3_coordinate_energy_share": (b[coordinate].square() / b.square().sum()).item(),
        "score_gap": gaps.detach().cpu().tolist(),
        "coordinate_score_gap_contribution": coordinate_gaps.detach().cpu().tolist(),
        "coordinate_gap_abs_share": [
            (coordinate_gaps[i].abs() / gaps[i].abs()).item()
            if gaps[i].abs().item() > 1e-8 else None for i in range(4)
        ],
        "tangent_jacobian_predicted_gate_shift": predicted_gate_shift.detach().cpu().tolist(),
        "physical_recomputed_gate_shift": actual_gate_shift.detach().cpu().tolist(),
        "reader_nonlinearity_residual": (
            actual_gate_shift - predicted_gate_shift
        ).detach().cpu().tolist(),
        "epsilon_aware_radial_score_jacobian": (grad @ b).detach().cpu().tolist(),
        "rmsnorm_eps": eps,
    }


class InklingMessageHook:
    """Native input intervention + independent module-output parity audit."""

    def __init__(self, module, spec: PatchSpec):
        self.module = module
        self.spec = spec
        self.calls = 0
        self.delivered: list[dict] = []
        self.readers: list[dict] = []
        self.parity_errors: list[float] = []
        self._pending: tuple[int, Tensor] | None = None
        self._handles = []

    def __enter__(self):
        if self._handles:
            raise RuntimeError("hook already active")
        self._handles = [
            self.module.register_forward_pre_hook(self._pre_hook),
            self.module.register_forward_hook(self._post_hook),
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        return False

    def _pre_hook(self, module, inputs):
        if len(inputs) != 3:
            raise RuntimeError("Inkling message_input signature drift")
        blocks, partial, loop_layer_idx = inputs
        if not isinstance(loop_layer_idx, int):
            raise RuntimeError("unexpected loop reader index type")
        if blocks.ndim != 4 or partial.ndim != 3:
            raise RuntimeError("unexpected previous block layout")
        if tuple(blocks.shape[:2]) != tuple(partial.shape[:2]) or blocks.shape[-2] != 1:
            raise RuntimeError("expected one previous complete block plus block-3 partial")
        if blocks.shape[-1] != partial.shape[-1] or partial.shape[0] != 1:
            raise RuntimeError("width mismatch or unsupported multi-example batch")
        layer = self.calls % self.spec.num_loop_layers
        pass_idx = self.calls // self.spec.num_loop_layers + 1
        if loop_layer_idx != layer or pass_idx >= self.spec.total_passes:
            raise RuntimeError("Inkling cross-loop call sequence changed")
        self.calls += 1
        old = partial[0, self.spec.token_index]
        changed = partial
        active = pass_idx in self.spec.passes and self.spec.arm != "zero"
        if active:
            direction = direction_for_arm(old, self.spec)
            edited, receipt = fixed_chord_rotation(old, direction, self.spec.dose_fraction)
            changed = partial.clone()
            changed[0, self.spec.token_index] = edited
            self.delivered.append({
                "pass": pass_idx, "reader_layer": layer, **receipt,
            })
        if layer == self.spec.probe_layer:
            b2 = blocks[0, self.spec.token_index, 0]
            b3 = changed[0, self.spec.token_index]
            beta_clean, _ = recompute_message(module, b2, old, layer)
            beta_after, expected = recompute_message(module, b2, b3, layer)
            diag = probe_coordinate_attribution(
                module, b2, old, b3, layer, self.spec.coordinate
            )
            self.readers.append({
                "pass": pass_idx,
                "beta3_clean": beta_clean.detach().cpu().tolist(),
                "beta3_patched": beta_after.detach().cpu().tolist(),
                "beta3_change": (beta_after - beta_clean).detach().cpu().tolist(),
                "outlier_attribution": diag,
            })
            self._pending = (self.calls, expected.detach())
        return (blocks, changed, loop_layer_idx)

    def _post_hook(self, module, inputs, output):
        if self._pending is not None and self._pending[0] == self.calls:
            _, expected = self._pending
            if output.ndim != 4 or output.shape[0] != 1 or output.shape[-2:] != expected.shape:
                raise RuntimeError("native message output layout drift")
            observed = output[0, self.spec.token_index].float()
            difference = (observed - expected).abs().max().item()
            self.parity_errors.append(difference)
            if difference > 0.003:
                raise RuntimeError(f"native four-reader message parity failed ({difference})")
            self._pending = None

    def finish(self) -> dict:
        expected = self.spec.num_loop_layers * (self.spec.total_passes - 1)
        if self.calls != expected or self._pending is not None:
            raise RuntimeError(f"cross-loop hook coverage {self.calls}/{expected}")
        expected_doses = len(self.spec.passes) * self.spec.num_loop_layers
        if self.spec.arm == "zero":
            expected_doses = 0
        if len(self.delivered) != expected_doses:
            raise RuntimeError("incomplete physical dose ledger")
        if len(self.readers) != self.spec.total_passes - 1 or len(self.parity_errors) != len(self.readers):
            raise RuntimeError("incomplete four-reader parity ledger")
        return {
            "schema": "bpd/ink-bpd05-native-reader-hook/v1",
            "arm": self.spec.arm,
            "pass_indices": list(self.spec.passes),
            "coordinate": self.spec.coordinate,
            "dose_fraction": self.spec.dose_fraction,
            "call_count": self.calls,
            "physical_read_count": len(self.delivered),
            "delivered": self.delivered,
            "readers": self.readers,
            "max_message_parity_abs": max(self.parity_errors),
        }
