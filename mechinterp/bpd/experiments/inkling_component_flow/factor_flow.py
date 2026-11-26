"""Read-only BPD factor accounting at a two-block RMS attention reader.

The clean wrapper output is decomposed at one observed execution. Finite
edits operate on the previous loop's completed block-3 boundary while holding
block 2 fixed. Declared sites must establish the adjacent-loop relationship.
An upstream writer intervention and subsequent model execution require a
separate native hook. All returned tensors are detached CPU float64 values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import torch
from torch import Tensor

from mechinterp.bpd.components import TiedComponentLinear


@dataclass(frozen=True)
class ExecutionSite:
    """Physical layers are one-based; token and loop indices are zero-based.

    ``depth`` repeats computation for a token. ``sequence_time`` denotes
    recurrent processing across input steps. The caller supplies identity;
    a wrapper invocation counter alone cannot establish either interpretation.
    """

    root_id: str
    checkpoint_id: str
    example_id: str
    token_index: int
    loop_index: int
    physical_layer: int
    module_name: str
    step_kind: Literal["depth", "sequence_time"]

    def __post_init__(self) -> None:
        for name in ("root_id", "checkpoint_id", "example_id", "module_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("token_index", "loop_index", "physical_layer"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "physical_layer" else 0):
                raise ValueError(f"invalid {name}")
        if self.step_kind not in ("depth", "sequence_time"):
            raise ValueError("step_kind must identify depth or sequence_time recurrence")


def _real(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {name}")
    return value


def _tensor(value: Tensor, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty rank-{ndim} tensor")
    if not value.is_floating_point() or value.is_complex():
        raise ValueError(f"{name} must be a real floating tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _finite(value: Any, name: str) -> Any:
    """Fail explicitly when finite inputs overflow a derived calculation."""
    if isinstance(value, Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite derived {name}")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite derived {name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite(item, f"{name}[{index}]")
    return value


def _closure(actual: Tensor, expected: Tensor, atol: float, rtol: float, label: str) -> float:
    if actual.shape != expected.shape:
        raise ValueError(f"{label} shape mismatch")
    _finite(actual, label)
    _finite(expected, label)
    error = float(_finite((actual - expected).abs().max(), label))
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f"{label} failed: max absolute error {error:.9g}")
    return error


@dataclass(frozen=True)
class WriteLedger:
    site: ExecutionSite
    component_writes: Tensor
    residual_write: Tensor
    bias_write: Tensor
    observed_output: Tensor
    component_activations: Tensor
    native_output_max_abs_error: float
    closure_atol: float
    closure_rtol: float
    input_dtype: str
    observed_dtype: str


def clean_target_writes(
    module: TiedComponentLinear,
    x: Tensor,
    observed_output: Tensor,
    site: ExecutionSite,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> WriteLedger:
    """Account for a separately observed clean target-mode native output.

    The residual is recomputed from the stored target weight and factors in
    float64. This retains the exact parameter identity and reports its error
    against the supplied native output. Active masks, mean replacement and
    sphere rescaling are outside this clean target-mode interface.
    """
    if not isinstance(module, TiedComponentLinear):
        raise TypeError("module must be a native TiedComponentLinear")
    if not isinstance(site, ExecutionSite):
        raise TypeError("site must be an ExecutionSite")
    if module.mode != "target" or module.schedule is not None or module.intervention is not None:
        raise ValueError("clean target mode with no schedule or intervention is required")
    atol, rtol = _real(atol, "atol"), _real(rtol, "rtol")
    xx = _tensor(x, "x", 1)
    observed = _tensor(observed_output, "observed_output", 1)
    v, u = _tensor(module.V, "V", 2), _tensor(module.U, "U", 2)
    weight = _tensor(module.W_target, "W_target", 2)
    if v.shape[0] != xx.numel() or v.shape[1] != u.shape[0]:
        raise ValueError("factor/input dimensions disagree")
    if tuple(weight.shape) != (u.shape[1], xx.numel()) or observed.numel() != u.shape[1]:
        raise ValueError("target/output dimensions disagree")
    activations = _finite(xx @ v, "component activations")
    writes = _finite(activations[:, None] * u, "component writes")
    # Apply the residual to this input directly; avoid a second dense weight
    # matrix or any [tokens, components, width] diagnostic allocation.
    target_write = _finite(xx @ weight.T, "target write")
    factor_write = _finite(activations @ u, "summed factor write")
    residual = _finite(target_write - factor_write, "weight residual write")
    bias = torch.zeros_like(observed) if module.bias is None else _tensor(module.bias, "bias", 1)
    if bias.shape != observed.shape:
        raise ValueError("bias/output dimensions disagree")
    error = _closure(writes.sum(0) + residual + bias, observed, atol, rtol, "native output closure")
    return WriteLedger(site, writes, residual, bias, observed, activations, error,
                       atol, rtol, str(x.dtype), str(observed_output.dtype))


@dataclass(frozen=True)
class ReaderSpec:
    """Queries at one execution site; IDs preserve Q/K/V/MoE identity."""

    site: ExecutionSite
    reader_ids: tuple[str, ...]
    queries: Tensor  # [reader, width]
    eps: float
    rms_scale: Tensor | None = None  # optional learned elementwise RMS scale

    def __post_init__(self) -> None:
        if not isinstance(self.site, ExecutionSite):
            raise TypeError("reader site must be an ExecutionSite")
        queries = _tensor(self.queries, "queries", 2)
        if (len(self.reader_ids) != queries.shape[0]
                or len(set(self.reader_ids)) != len(self.reader_ids)
                or any(not isinstance(s, str) or not s.strip() for s in self.reader_ids)):
            raise ValueError("one distinct nonempty reader ID is required per query")
        eps = _real(self.eps, "eps")
        scale = torch.ones(queries.shape[1], dtype=torch.float64)
        if self.rms_scale is not None:
            scale = _tensor(self.rms_scale, "rms_scale", 1)
            if scale.numel() != queries.shape[1]:
                raise ValueError("RMS scale width mismatch")
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "rms_scale", scale)


def reader_state(reader: ReaderSpec, block2: Tensor, block3: Tensor) -> dict[str, Tensor]:
    """Recompute normalized values, softmax and the raw block-3 Jacobian."""
    a, b = _tensor(block2, "block2", 1), _tensor(block3, "block3", 1)
    d = reader.queries.shape[1]
    if a.numel() != d or b.numel() != d:
        raise ValueError("reader/block widths disagree")
    rho2 = _finite((a.square().sum() + d * reader.eps).sqrt(), "block2 RMS denominator")
    rho3 = _finite((b.square().sum() + d * reader.eps).sqrt(), "block3 RMS denominator")
    if float(rho2) == 0 or float(rho3) == 0:
        raise ValueError("zero block at epsilon zero has undefined normalization")
    effective = _finite(reader.queries * reader.rms_scale, "scaled queries")
    score2, score3 = effective @ a / rho2, effective @ b / rho3
    gap = score3 - score2
    beta3 = torch.softmax(torch.stack((score2, score3), dim=-1), dim=-1)[:, 1]
    normalized2 = a * reader.rms_scale * math.sqrt(d) / rho2
    normalized3 = b * reader.rms_scale * math.sqrt(d) / rho3
    gradient = effective / rho3 - score3[:, None] * b[None, :] / rho3.square()
    beta_gradient = beta3[:, None] * (1 - beta3[:, None]) * gradient
    message = ((1 - beta3[:, None]) * normalized2
               + beta3[:, None] * normalized3)
    return _finite({"score2": score2, "score3": score3, "gap": gap, "beta3": beta3,
            "normalized2": normalized2, "normalized3": normalized3, "message": message,
            "gap_gradient_block3": gradient, "beta3_gradient_block3": beta_gradient,
            "radial_gap_derivative": gradient @ b,
            "analytic_radial_gap_derivative": score3 * (d * reader.eps) / rho3.square(),
            "rho3": rho3}, "reader state")


def _same_context(writer: ExecutionSite, reader: ExecutionSite) -> None:
    """Validate the declared preceding-loop edge and its token ordering.

    This validates metadata supplied by the caller. Observed native execution
    identity still requires a separately audited trace or hook.
    """
    if not isinstance(writer, ExecutionSite) or not isinstance(reader, ExecutionSite):
        raise TypeError("source and reader sites must be ExecutionSite values")
    for name in ("root_id", "checkpoint_id", "example_id", "step_kind"):
        if getattr(writer, name) != getattr(reader, name):
            raise ValueError(f"writer and reader disagree on {name}")
    if reader.loop_index != writer.loop_index + 1:
        raise ValueError("reader must consume the immediately previous loop")
    if writer.step_kind == "depth" and reader.token_index != writer.token_index:
        raise ValueError("depth recurrence must preserve token_index")
    if writer.step_kind == "sequence_time" and reader.token_index <= writer.token_index:
        raise ValueError("sequence_time reader token_index must strictly increase")


def _reader_snapshot(reader: ReaderSpec) -> dict[str, Any]:
    return {"site": asdict(reader.site), "reader_ids": reader.reader_ids,
            "queries": reader.queries.clone(), "eps": reader.eps,
            "rms_scale": reader.rms_scale.clone()}


def component_reader_flow(
    ledger: WriteLedger,
    reader: ReaderSpec,
    block2: Tensor,
    block3: Tensor,
    other_block_writes: Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    share_norm_floor: float = 1e-12,
) -> dict[str, Any]:
    """Signed observed accounting and local sensitivity of each physical write.

    ``other_block_writes`` must come from an independently assembled remainder.
    This API verifies closure but cannot attest how the caller obtained it.
    Declared sites must describe an adjacent-loop edge, with matching depth
    tokens or increasing sequence-time tokens. Component entries depend on
    the factor gauge. Inverse diagonal rescaling
    preserves each write; arbitrary component mixing can alter the entries.
    """
    _same_context(ledger.site, reader.site)
    atol, rtol = _real(atol, "atol"), _real(rtol, "rtol")
    floor = _real(share_norm_floor, "share_norm_floor", positive=True)
    a, b = _tensor(block2, "block2", 1), _tensor(block3, "block3", 1)
    remainder = _tensor(other_block_writes, "other_block_writes", 1)
    if remainder.shape != ledger.observed_output.shape:
        raise ValueError("remainder/output dimensions disagree")
    output_closure = _closure(ledger.observed_output + remainder, b, atol, rtol,
                              "observed block closure")
    terms = torch.cat((ledger.component_writes, ledger.residual_write[None],
                       ledger.bias_write[None], remainder[None]), dim=0)
    reconstruction_error = _closure(terms.sum(0), b, atol, rtol, "factor block closure")
    labels = tuple(f"component:{c}" for c in range(ledger.component_writes.shape[0]))
    labels += ("weight_residual", "bias", "other_block_writes")
    state = reader_state(reader, a, b)
    score3_terms = (reader.queries * reader.rms_scale) @ terms.T / state["rho3"]
    gap_terms = torch.cat((score3_terms, -state["score2"][:, None]), dim=1)
    norm2 = _finite(b.square().sum(), "projection denominator")
    shares_valid = float(norm2.sqrt()) > floor
    shares = (terms @ b / norm2) if shares_valid else None
    return _finite({
        "schema": "bpd/ink-bpd06-component-reader-flow/v1",
        "evidence_class": "observed_write_accounting_and_local_reader_geometry",
        "intervention_site": "previous_loop_completed_block3_boundary",
        "writer_site": asdict(ledger.site), "reader_site": asdict(reader.site),
        "reader_ids": reader.reader_ids, "term_labels": labels,
        "reader_spec": _reader_snapshot(reader), "block2": a, "block3": b,
        "score_gap_term_labels": labels + ("negative_block2_score",),
        "component_activations": ledger.component_activations,
        "physical_writes": terms,
        "native_output_max_abs_error": ledger.native_output_max_abs_error,
        "native_closure_atol": ledger.closure_atol,
        "native_closure_rtol": ledger.closure_rtol,
        "observed_block_max_abs_error": output_closure,
        "factor_block_max_abs_error": reconstruction_error,
        "block_closure_atol": atol, "block_closure_rtol": rtol,
        "score3_terms": score3_terms, "score_gap_terms": gap_terms,
        "score_gap_closure_error": (gap_terms.sum(-1) - state["gap"]).abs(),
        "signed_projection_shares": shares, "projection_shares_valid": shares_valid,
        "share_norm_floor": floor,
        "projection_share_sum": float(shares.sum()) if shares_valid else None,
        "gap_derivative_per_write": state["gap_gradient_block3"] @ terms.T,
        "beta3_derivative_per_write": state["beta3_gradient_block3"] @ terms.T,
        "rmsnorm_eps": reader.eps, "reader": state,
    }, "component reader flow")


def patch_reader_audit(
    reader: ReaderSpec,
    block2: Tensor,
    block3: Tensor,
    delta: Tensor,
    *,
    source_site: ExecutionSite,
    max_dose: float,
) -> dict[str, Any]:
    """Audit a previous-loop block-3 edit with block 2 and queries fixed.

    The message decomposition uses clean normalized content with changed
    routing, clean routing with changed content, and their interaction. Its
    terms sum to the recomputed message change. Model output/task behavior is
    outside this reader-local calculation.
    """
    _same_context(source_site, reader.site)
    budget = _real(max_dose, "max_dose")
    a = _tensor(block2, "block2", 1)
    b, displacement = _tensor(block3, "block3", 1), _tensor(delta, "delta", 1)
    if b.shape != displacement.shape:
        raise ValueError("patch width mismatch")
    dose = float(_finite(torch.linalg.vector_norm(displacement), "patch dose"))
    if (budget == 0 and dose != 0) or dose > budget + 1e-12 * max(1.0, budget):
        raise ValueError("delivered patch exceeds max_dose")
    patched = _finite(b + displacement, "patched block3")
    radius_before = float(_finite(torch.linalg.vector_norm(b), "source radius"))
    radius_after = float(_finite(torch.linalg.vector_norm(patched), "patched radius"))
    before = reader_state(reader, a, b)
    after = reader_state(reader, a, patched)
    gate_change = after["beta3"] - before["beta3"]
    content_change = after["normalized3"] - before["normalized3"]
    route = gate_change[:, None] * (before["normalized3"] - before["normalized2"])
    content = before["beta3"][:, None] * content_change
    interaction = gate_change[:, None] * content_change
    message_change = after["message"] - before["message"]
    predicted_gap = before["gap_gradient_block3"] @ displacement
    predicted_gate = before["beta3_gradient_block3"] @ displacement
    actual_gap = after["gap"] - before["gap"]
    return _finite({
        "schema": "bpd/ink-bpd06-boundary-patch/v1",
        "evidence_class": "finite_recomputed_reader_boundary_edit",
        "intervention_site": "previous_loop_completed_block3_boundary",
        "source_site": asdict(source_site),
        "reader_site": asdict(reader.site), "reader_ids": reader.reader_ids,
        "reader_spec": _reader_snapshot(reader), "block2": a,
        "block3_before": b, "applied_displacement": displacement,
        "dose_l2": dose, "max_dose": budget,
        "radius_before": radius_before, "radius_after": radius_after,
        "relative_dose": dose / radius_before if radius_before > 0 else None,
        "rms_before": radius_before / math.sqrt(b.numel()),
        "rms_after": radius_after / math.sqrt(b.numel()),
        "predicted_gap_change": predicted_gap, "actual_gap_change": actual_gap,
        "gap_linearization_residual": actual_gap - predicted_gap,
        "predicted_beta3_change": predicted_gate, "actual_beta3_change": gate_change,
        "beta3_linearization_residual": gate_change - predicted_gate,
        "route_delta": route, "content_delta": content,
        "interaction_delta": interaction, "message_delta": message_change,
        "decomposition_max_abs_error": float(
            (route + content + interaction - message_change).abs().max()),
    }, "reader patch audit")


def component_direction_patch(
    ledger: WriteLedger,
    component: int,
    dose: float,
    *,
    orientation: int = -1,
) -> Tensor:
    """An absolute-L2 boundary displacement along one physical factor write.

    Negative orientation points toward removal. The actual coefficient
    removal dose is the write's norm. The caller chooses and records the dose.
    """
    dose = _real(dose, "dose")
    if type(component) is not int or not 0 <= component < ledger.component_writes.shape[0]:
        raise ValueError("invalid component index")
    if type(orientation) is not int or orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or +1")
    direction = ledger.component_writes[component]
    norm = float(_finite(torch.linalg.vector_norm(direction), "component direction norm"))
    if dose == 0:
        return torch.zeros_like(direction)
    if norm == 0:
        raise ValueError("zero component write cannot supply a nonzero dose")
    displacement = _finite(orientation * (direction / norm) * dose, "component patch")
    delivered = float(_finite(torch.linalg.vector_norm(displacement), "component patch dose"))
    if not math.isclose(delivered, dose, rel_tol=1e-10, abs_tol=0.):
        raise ValueError("component patch failed to deliver requested dose")
    return displacement


def to_jsonable(value: Any) -> Any:
    """Convert a diagnostic result to JSON values, preserving undefined shares."""
    _finite(value, "JSON diagnostic")
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value
