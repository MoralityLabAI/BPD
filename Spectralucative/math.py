"""Finite spectral diagnostics for BPD, using chronological column-state maps.

The analytic identities concern real matrices. NumPy evaluates them in float64;
these routines provide numerical checks and analytic truncation formulas, not
outward-rounded interval certificates. ``OntologicalProofs`` PR231 supplies the
stationary Lean results. The nonautonomous identity below is derived explicitly
from its one-step recurrence and is independently checked in this study.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


def _real_array(value, name: str, *, ndim: int | None = None) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real numeric array") from error
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _natural(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def matrix_inf_norm(matrix) -> float:
    """Induced infinity norm, including rectangular and empty matrices."""
    matrix = _real_array(matrix, "matrix", ndim=2)
    if not matrix.shape[0] or not matrix.shape[1]:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.max(np.sum(np.abs(matrix), axis=1)))
    if not math.isfinite(result):
        raise ValueError("matrix norm exceeds the float64 range")
    return result


def _float_slack(*scales: float, operations: int = 1) -> float:
    return float(128 * np.finfo(np.float64).eps * max(1, operations) *
                 max(1.0, *(abs(float(scale)) for scale in scales)))


def trace_logdet_diagnostic(matrix, *, z: float, terms: int = 12) -> dict:
    """Compare Σ(k=1..K) tr((zA)^k)/k with −log det(I−zA).

    For q=|z|‖A‖∞<1, the exact-real truncation error is bounded by
    d*q**(K+1)/((K+1)*(1-q)). This follows from |tr(M^k)|≤d‖M‖∞**k.
    Outside this sufficient norm domain, no series or tail certificate is
    returned, even when the spectral radius happens to be below one.

    The determinant is evaluated through ``slogdet``. Its sign must be positive
    inside the norm domain. The numerical slack is a comparison tolerance;
    floating-point roundoff has no certified bound here.
    """
    matrix = _real_array(matrix, "matrix", ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("trace/logdet requires a square matrix")
    terms = _natural(terms, "terms")
    z = float(z)
    if not math.isfinite(z):
        raise ValueError("z must be finite")
    dimension = matrix.shape[0]
    norm = matrix_inf_norm(matrix)
    q = abs(z) * norm
    if not math.isfinite(q):
        raise ValueError("scaled matrix norm exceeds the float64 range")
    scaled = z * matrix
    sign, logabsdet = np.linalg.slogdet(np.eye(dimension) - scaled)
    negative_logdet = -float(logabsdet) if sign > 0 and math.isfinite(logabsdet) else None
    result = {
        "dimension": dimension, "z": z, "terms": terms,
        "matrix_inf_norm": norm, "q": q,
        "convergence_condition": q < 1.0,
        "status": "within_norm_domain" if q < 1.0 else "outside_norm_domain",
        "logdet_sign": float(sign), "negative_logdet": negative_logdet,
        "trace_log_partial": None, "analytic_tail_bound": None,
        "absolute_log_error": None, "floating_point_slack": None,
        "floating_point_check_passed": None,
        "roundoff_certified": False,
        "evidence_class": "floating_point_diagnostic_with_analytic_truncation_bound",
    }
    if q >= 1.0:
        return result
    if negative_logdet is None:
        raise FloatingPointError("slogdet failed inside the strict norm domain")
    power = np.eye(dimension)
    summands = []
    for k in range(1, terms + 1):
        power = scaled @ power
        summands.append(float(np.trace(power)) / k)
    partial = math.fsum(summands)
    tail = dimension * q ** (terms + 1) / ((terms + 1) * (1 - q))
    error = abs(partial - negative_logdet)
    slack = _float_slack(partial, negative_logdet, tail,
                         operations=max(1, dimension, terms))
    result.update(trace_log_partial=partial, analytic_tail_bound=tail,
                  absolute_log_error=error, floating_point_slack=slack,
                  floating_point_check_passed=error <= tail + slack)
    return result


def ordered_product(operators: Sequence, *, dimension: int | None = None) -> np.ndarray:
    """Return A[T−1]…A[0]; rectangular successive maps are supported.

    An empty sequence needs ``dimension`` to specify the identity matrix.
    """
    operators = [_real_array(a, f"operators[{i}]", ndim=2) for i, a in enumerate(operators)]
    if not operators:
        if dimension is None:
            raise ValueError("an empty product requires its identity dimension")
        return np.eye(_natural(dimension, "dimension"))
    initial = operators[0].shape[1]
    if dimension is not None and _natural(dimension, "dimension") != initial:
        raise ValueError("dimension disagrees with the first operator")
    product = np.eye(initial)
    for operator in operators:
        if operator.shape[1] != product.shape[0]:
            raise ValueError("successive operator dimensions do not compose")
        product = operator @ product
        if not np.all(np.isfinite(product)):
            raise FloatingPointError("ordered product exceeds the float64 range")
    return product


def transport_diagnostic(A_steps: Sequence, B_steps: Sequence, L_steps: Sequence) -> dict:
    """Check transport through time-varying real maps, including rectangular L.

    At time t, A_t maps n_t→n_(t+1), B_t maps m_t→m_(t+1), and
    L_t maps n_t→m_t. Set E_t=B_t L_t−L_(t+1) A_t and
    D_t=Φ_B(t,0)L_0−L_t Φ_A(t,0). Then

        D_(t+1)=B_t D_t+E_t Φ_A(t,0),   D_0=0.

    Consequently e_(t+1)=‖B_t‖∞ e_t+‖E_t‖∞‖Φ_A(t,0)‖∞ is an
    exact-real norm envelope. The finite telescoping sum is reconstructed
    separately. No stationarity, contraction, or invertibility is assumed.
    """
    A = [_real_array(a, f"A_steps[{i}]", ndim=2) for i, a in enumerate(A_steps)]
    B = [_real_array(b, f"B_steps[{i}]", ndim=2) for i, b in enumerate(B_steps)]
    L = [_real_array(l, f"L_steps[{i}]", ndim=2) for i, l in enumerate(L_steps)]
    horizon = len(A)
    if len(B) != horizon or len(L) != horizon + 1:
        raise ValueError("T A maps and T B maps require exactly T+1 transport maps")
    for t, (a, b) in enumerate(zip(A, B)):
        expected_a = (L[t + 1].shape[1], L[t].shape[1])
        expected_b = (L[t + 1].shape[0], L[t].shape[0])
        if a.shape != expected_a or b.shape != expected_b:
            raise ValueError(f"operator dimensions disagree with transport maps at step {t}")
    source_prefix = [np.eye(L[0].shape[1])]
    target_prefix = [np.eye(L[0].shape[0])]
    defects = []
    differences = [np.zeros_like(L[0])]
    envelopes = [0.0]
    defect_norms = []
    target_norms = []
    prefix_norms = [matrix_inf_norm(source_prefix[0])]
    for t, (a, b) in enumerate(zip(A, B)):
        defect = b @ L[t] - L[t + 1] @ a
        defects.append(defect)
        defect_norms.append(matrix_inf_norm(defect))
        target_norms.append(matrix_inf_norm(b))
        envelopes.append(target_norms[-1] * envelopes[-1] + defect_norms[-1] * prefix_norms[-1])
        source_prefix.append(a @ source_prefix[-1])
        target_prefix.append(b @ target_prefix[-1])
        prefix_norms.append(matrix_inf_norm(source_prefix[-1]))
        difference = target_prefix[-1] @ L[0] - L[t + 1] @ source_prefix[-1]
        differences.append(difference)
        if not math.isfinite(envelopes[-1]) or not np.all(np.isfinite(difference)):
            raise FloatingPointError("transport calculation exceeds the float64 range")
    telescoping = np.zeros((L[-1].shape[0], L[0].shape[1]))
    suffix = np.eye(L[-1].shape[0])
    for t in reversed(range(horizon)):
        telescoping += suffix @ defects[t] @ source_prefix[t]
        suffix = suffix @ B[t]
    error_norms = [matrix_inf_norm(difference) for difference in differences]
    reconstruction_error = matrix_inf_norm(telescoping - differences[-1])
    slack = _float_slack(*error_norms, *envelopes,
                         operations=max(1, horizon, *L[0].shape))
    return {
        "n_steps": horizon,
        "source_dimensions": [l.shape[1] for l in L],
        "target_dimensions": [l.shape[0] for l in L],
        "local_defect_norms": defect_norms,
        "source_prefix_norms": prefix_norms,
        "target_operator_norms": target_norms,
        "product_error_norms": error_norms,
        "norm_envelope": envelopes,
        "final_error_norm": error_norms[-1],
        "final_bound": envelopes[-1],
        "final_difference": differences[-1].tolist(),
        "telescoping_reconstruction": telescoping.tolist(),
        "reconstruction_residual_norm": reconstruction_error,
        "floating_point_slack": slack,
        "floating_point_check_passed": bool(
            reconstruction_error <= slack and
            all(error <= bound + slack for error, bound in zip(error_norms, envelopes))
        ),
        "roundoff_certified": False,
    }


def walsh_transform(values) -> np.ndarray:
    """Normalized Walsh coefficients along axis 0, with integer/LSB mask order.

    For N=2^k, coefficient s is N⁻¹ Σ_x values[x] (−1)^popcount(s&x).
    Values may be scalar or have one output-vector axis.
    """
    values = _real_array(values, "values")
    if values.ndim not in (1, 2) or values.shape[0] < 1:
        raise ValueError("values must be a nonempty mask vector or mask-by-output matrix")
    count = values.shape[0]
    if count & (count - 1):
        raise ValueError("the complete cube must contain a power of two values")
    if values.ndim == 2 and values.shape[1] < 1:
        raise ValueError("at least one output coordinate is required")
    output = values.copy()
    width = 1
    while width < count:
        for start in range(0, count, 2 * width):
            left = output[start:start + width].copy()
            right = output[start + width:start + 2 * width].copy()
            output[start:start + width] = left + right
            output[start + width:start + 2 * width] = left - right
        width *= 2
    output /= count
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("Walsh transform exceeds the float64 range")
    return output


def walsh_inverse(coefficients) -> np.ndarray:
    """Inverse of the normalized transform: H times the coefficient vector."""
    coefficients = _real_array(coefficients, "coefficients")
    return walsh_transform(coefficients) * coefficients.shape[0]


def walsh_diagnostic(mask_ids: Sequence[int], values, *, max_degree: int = 1) -> dict:
    """Parseval and low-degree residual energy on a complete uniform mask cube.

    IDs must contain each integer 0..2^k−1 exactly once. Input order may vary;
    coefficients and reconstructed values are returned in sorted mask-ID order.
    For vector responses, energy sums squared output coordinates and averages
    only over masks. A higher Walsh degree describes interaction among the
    chosen intervention bits, without asserting a causal mechanism rank.
    """
    values = _real_array(values, "values")
    if values.ndim not in (1, 2) or not len(values):
        raise ValueError("values must be a nonempty mask vector or mask-by-output matrix")
    count = len(values)
    if count & (count - 1):
        raise ValueError("an incomplete mask cube cannot receive a Parseval diagnostic")
    if len(mask_ids) != count:
        raise ValueError("mask_ids and values must have equal length")
    ids = [_natural(mask_id, "mask ID") for mask_id in mask_ids]
    if sorted(ids) != list(range(count)):
        raise ValueError("mask IDs must enumerate the complete cube exactly once")
    degree = _natural(max_degree, "max_degree")
    bits = count.bit_length() - 1
    degree = min(degree, bits)
    ordered = values[np.argsort(ids)]
    coefficients = walsh_transform(ordered)
    coefficients_2d = coefficients[:, None] if coefficients.ndim == 1 else coefficients
    ordered_2d = ordered[:, None] if ordered.ndim == 1 else ordered
    coefficient_energy = np.sum(coefficients_2d ** 2, axis=1)
    degrees = np.array([mask.bit_count() for mask in range(count)])
    energy_by_degree = [float(np.sum(coefficient_energy[degrees == d])) for d in range(bits + 1)]
    truncated = coefficients.copy()
    truncated[degrees > degree] = 0.0
    reconstructed = walsh_inverse(truncated)
    residual = ordered - reconstructed
    residual_2d = residual[:, None] if residual.ndim == 1 else residual
    mean_squared = float(np.mean(np.sum(ordered_2d ** 2, axis=1)))
    spectral_energy = float(np.sum(coefficient_energy))
    residual_energy = float(np.mean(np.sum(residual_2d ** 2, axis=1)))
    discarded_energy = float(np.sum(coefficient_energy[degrees > degree]))
    variance = float(np.mean(np.sum((ordered_2d - ordered_2d.mean(axis=0)) ** 2, axis=1)))
    high_order = float(np.sum(coefficient_energy[degrees >= 2]))
    if not all(math.isfinite(number) for number in (
        *energy_by_degree, mean_squared, spectral_energy, residual_energy,
        discarded_energy, variance, high_order,
    )):
        raise FloatingPointError("Walsh energy exceeds the float64 range")
    slack = _float_slack(mean_squared, spectral_energy, residual_energy,
                         operations=max(count, ordered_2d.shape[1]))
    return {
        "n_bits": bits, "n_masks": count, "output_dimension": ordered_2d.shape[1],
        "measure": "uniform_complete_cube", "retained_degree": degree,
        "mask_ids": list(range(count)), "coefficients": coefficients.tolist(),
        "energy_by_degree": energy_by_degree,
        "mean_squared_l2_response": mean_squared,
        "spectral_energy": spectral_energy,
        "parseval_absolute_error": abs(mean_squared - spectral_energy),
        "variance": variance, "high_degree_energy": high_order,
        "high_degree_fraction_of_variance": high_order / variance if variance > 0 else 0.0,
        "truncated_response": reconstructed.tolist(),
        "residual_mean_squared_l2_response": residual_energy,
        "discarded_coefficient_energy": discarded_energy,
        "residual_identity_absolute_error": abs(residual_energy - discarded_energy),
        "floating_point_slack": slack,
        "floating_point_check_passed": bool(
            abs(mean_squared - spectral_energy) <= slack and
            abs(residual_energy - discarded_energy) <= slack
        ),
        "roundoff_certified": False,
    }


def factor_closure_witness(V, U, target_weight, *, mask=None) -> dict:
    """Exact-real read-space closure witness for a square BPD linear recurrence.

    BPD stores V[d,C], U[C,d], W_hat=UᵀVᵀ and Δ=W_target−W_hat.
    With retained residual and component mask M, W_mask=Uᵀ M Vᵀ+Δ.
    The read map L=Vᵀ and reduced map K=VᵀUᵀM satisfy
    K L−L W_mask=−LΔ. Nonlinear activations and sphere rescaling require
    the Jacobian of their full update; this linear witness does not include them.
    """
    V = _real_array(V, "V", ndim=2)
    U = _real_array(U, "U", ndim=2)
    target = _real_array(target_weight, "target_weight", ndim=2)
    dimension, components = V.shape
    if U.shape != (components, dimension) or target.shape != (dimension, dimension):
        raise ValueError("factor closure requires V[d,C], U[C,d], and square W[d,d]")
    if mask is None:
        mask = np.ones(components)
    mask = _real_array(mask, "mask", ndim=1)
    if mask.shape != (components,) or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("mask must have C entries in [0,1]")
    reconstruction = U.T @ V.T
    residual = target - reconstruction
    full = (U.T * mask) @ V.T + residual
    reduced = (V.T @ U.T) * mask
    read_map = V.T
    defect = reduced @ read_map - read_map @ full
    predicted_defect = -read_map @ residual
    error = matrix_inf_norm(defect - predicted_defect)
    slack = _float_slack(matrix_inf_norm(defect), matrix_inf_norm(predicted_defect),
                         operations=max(1, dimension, components))
    return {
        "state_dimension": dimension, "component_count": components,
        "full_masked_operator": full.tolist(), "reduced_operator": reduced.tolist(),
        "read_map": read_map.tolist(), "weight_residual": residual.tolist(),
        "closure_defect": defect.tolist(), "predicted_closure_defect": predicted_defect.tolist(),
        "closure_defect_norm": matrix_inf_norm(defect),
        "identity_residual_norm": error, "floating_point_slack": slack,
        "floating_point_check_passed": error <= slack, "roundoff_certified": False,
    }
