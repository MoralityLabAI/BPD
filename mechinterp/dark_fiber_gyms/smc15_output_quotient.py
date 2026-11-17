"""BPD-OUT01: finite-horizon two-atom observable quotient certificate.

The input columns are distinct physical directions. The observed matrix L
may map them to nearly parallel native response directions. We certify a
physical unit vector with bounded normalized output. This is a local linear
algebra theorem, not a claim about a trained recurrent model's global fiber.

Requires numpy. No model weights, heldout puzzles or source artifact needed.
"""
from __future__ import annotations

import numpy as np


def two_atom_observable_null_certificate(U: np.ndarray, L: np.ndarray) -> dict:
    """Finite-dimensional singular-vector construction.

    U is d x 2, L is m x d. Assume rank(U)=2, Lu and Lv are both nonzero.
    Put nu = max(||Lu||, ||Lv||), gamma = sigma_min(U), and epsilon =
    sigma_min([Lu/||Lu||, Lv/||Lv||]). For the normalized right singular
    vector c of the output frame, x = U diag(1/||Lu||,1/||Lv||) c.

    ||Lx|| = epsilon and ||x|| >= gamma/nu, hence
    ||L(x/||x||)|| <= epsilon*nu/gamma.

    The bounds concern a linear map L. For nonlinear G, L=DG(h) gives a
    local tangent statement until a remainder bound has been supplied.
    """
    U = np.asarray(U, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    if U.ndim != 2 or U.shape[1] != 2 or L.ndim != 2 or L.shape[1] != U.shape[0]:
        raise ValueError("expected U d×2 and L m×d")
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(L)):
        raise ValueError("nonfinite physical/observable map")
    singular_in = np.linalg.svd(U, full_matrices=False, compute_uv=False)
    gamma = float(singular_in[-1])
    if gamma <= 1e-10:
        raise ValueError("input physical pair rank deficient")
    observed = L @ U
    norms = np.linalg.norm(observed, axis=0)
    if float(norms.min()) <= 1e-10:
        raise ValueError("a feature already has zero downstream output")
    normalised = observed / norms
    _, sig, v_t = np.linalg.svd(normalised, full_matrices=False)
    epsilon = float(sig[-1])
    nu = float(norms.max())
    c = v_t[-1, :]
    x = U @ (c / norms)
    physnorm = float(np.linalg.norm(x))
    if physnorm <= 1e-10:
        raise ValueError("singular combination has zero physical norm")
    w = x / physnorm
    predicted = epsilon * nu / gamma
    actual = float(np.linalg.norm(L @ w))
    if actual > predicted + 2e-10:
        raise ArithmeticError("linear output quotient bound failed")
    if physnorm + 2e-10 < gamma / nu:
        raise ArithmeticError("lower physical norm bound failed")
    return {
        "epsilon_normalised_output_sigma_min": epsilon,
        "gamma_input_sigma_min": gamma,
        "nu_max_raw_column_response": nu,
        "physical_combination_norm": physnorm,
        "physical_unit_nearnull": w.tolist(),
        "actual_output_norm": actual,
        "certified_output_upper_bound": predicted,
        "input_physical_condition_number": float(singular_in[0] / gamma),
        "certificate_is_local_linear": True,
    }
