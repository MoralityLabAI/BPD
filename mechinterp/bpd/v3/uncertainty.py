"""Independent-board bootstrap for a fixed dither experiment.

Resampling each cell or dither as an independent unit would understate uncertainty.
"""
from __future__ import annotations

import numpy as np
from mechinterp.bpd.v3.floor_and_basis import kl, softmax


def board_js(target_logits: np.ndarray) -> np.ndarray:
    """Return finite-K plug-in J for every independent board and readout [B,R]."""
    logits = np.asarray(target_logits, dtype=np.float64)
    if logits.ndim != 5 or logits.shape[1] < 2:
        raise ValueError("expected logits [board,dither,readout,cell,class] with K >= 2")
    p = softmax(logits)
    mean_p = p.mean(axis=1)
    return kl(p, mean_p[:, None]).mean(axis=(1, 3))


def bootstrap_board_means(per_board_readout: np.ndarray, seed: int = 20261117,
                          repetitions: int = 5000) -> dict:
    """Board-cluster percentile interval. K finite-size bias remains uncorrected."""
    x = np.asarray(per_board_readout, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("expected finite [board,readout] with at least two boards")
    if repetitions < 100:
        raise ValueError("bootstrap repetitions must be >= 100")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, x.shape[0], size=(repetitions, x.shape[0]))
    by_readout = np.mean(x[indexes], axis=1)
    overall = np.mean(by_readout, axis=1)
    return {
        "point": float(x.mean()),
        "percentile95": np.percentile(overall, [2.5, 97.5]).tolist(),
        "readout_point": x.mean(axis=0).tolist(),
        "readout_percentile95": np.percentile(by_readout, [2.5, 97.5], axis=0).T.tolist(),
        "independent_boards": int(x.shape[0]),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "finite_K_bias_note": "J plug-in is downward biased; bootstrap resamples boards, not noise realizations",
    }
