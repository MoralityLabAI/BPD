"""SAU-TRACE02: retained latent information versus native recurrent access.

Independent, source-inspired synthetic linear recurrence with 50 latent items.
No language model checkpoints, hidden test data, or upstream source copied.
Run "python -m mechinterp.dark_fiber_gyms.sau_trace02" for 8 independent source
seeds; "python -m pytest mechinterp/dark_fiber_gyms/tests/test_sau_trace02.py".
"""
from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

K = 50
VISIBLE_REPLY = "I understand."
ETA = 0.8
RHO = 0.5
DEPTH = 4
GAIN = 7.0
GATE_OPEN = 0.6
NOISE = 0.30
SEEDS = (103, 131, 167, 211, 307, 401, 503, 601)


def propagation_coefficient(depth: int = DEPTH, gate: float = GATE_OPEN) -> float:
    """C F^k P_memory scalar for [[eta I,0],[gate I,rho I]]."""
    return gate * sum((RHO ** (depth - 1 - j)) * (ETA ** j) for j in range(depth))


def native_logits(memory: np.ndarray, gate: float, depth: int = DEPTH,
                  *, nuisance: np.ndarray | None = None) -> np.ndarray:
    """Physical state contains memory, nuisance and action variables.

    Nuisance is carried as inert state. Changing it at matched edit norm is
    a zero-effect control; changing memory matters iff the route is open.
    """
    memory = np.asarray(memory, dtype=np.float64).copy()
    if nuisance is not None and nuisance.shape != memory.shape:
        raise ValueError("nuisance must match memory shape")
    decision = np.zeros_like(memory)
    for _ in range(depth):
        decision = RHO * decision + gate * memory
        memory *= ETA
    return GAIN * decision


def true_probability(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(centered)
    p = weights / weights.sum(axis=1, keepdims=True)
    return p[np.arange(len(labels)), labels]


def fit_prototypes(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.stack([X[labels == k].mean(axis=0) for k in range(K)])


def classify(X: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    # Nearest Euclidean prototype is a linear scoring rule in X with biases.
    return np.argmax(X @ prototypes.T - 0.5 * (prototypes * prototypes).sum(axis=1), axis=1)


def run_seed(seed: int, *, n_test: int = 1000, per_class_cal: int = 32) -> dict:
    if n_test % K or per_class_cal < 2:
        raise ValueError("balanced test and >=2 calibrations per class required")
    cal = np.random.default_rng(seed * 13 + 1)
    test = np.random.default_rng(seed * 13 + 2)
    null_rng = np.random.default_rng(seed * 13 + 3)
    identities = np.eye(K)
    y_train = np.repeat(np.arange(K), per_class_cal)
    cal.shuffle(y_train)
    x_train = identities[y_train] + cal.normal(0, NOISE, (len(y_train), K))
    train_features_hash = hashlib.sha256(x_train.tobytes()).hexdigest()
    labels_hash = hashlib.sha256(y_train.tobytes()).hexdigest()
    fitted = fit_prototypes(x_train, y_train)
    shuffled_labels = null_rng.permutation(y_train)
    null = fit_prototypes(x_train, shuffled_labels)

    y_test = np.repeat(np.arange(K), n_test // K)
    test.shuffle(y_test)
    x_test = identities[y_test] + test.normal(0, NOISE, (n_test, K))
    test_features_hash = hashlib.sha256(x_test.tobytes()).hexdigest()
    hit = float(np.mean(classify(x_test, fitted) == y_test))
    null_hit = float(np.mean(classify(x_test, null) == y_test))

    z = np.zeros_like(x_test)
    closed = native_logits(x_test, gate=0.0, nuisance=z)
    stripped = native_logits(z, gate=0.0, nuisance=z)
    open_logits = native_logits(x_test, gate=GATE_OPEN, nuisance=z)
    partly_open = native_logits(x_test, gate=0.10, nuisance=z)
    donor = np.roll(x_test, shift=17, axis=1)
    open_donor = native_logits(donor, gate=GATE_OPEN, nuisance=z)
    closed_donor = native_logits(donor, gate=0.0, nuisance=z)
    matched_nuisance = donor - x_test
    sham = native_logits(x_test, gate=GATE_OPEN, nuisance=matched_nuisance)
    visible = GAIN * identities[y_test]

    assert np.array_equal(closed, stripped)
    assert np.array_equal(closed_donor, closed)
    assert np.array_equal(open_logits, sham)
    assert np.max(np.abs(open_logits - GAIN * propagation_coefficient() * x_test)) < 1e-12

    return {
        "seed": seed, "n_cal": len(y_train), "n_test": n_test,
        "calibration_sha256": train_features_hash,
        "calibration_labels_sha256": labels_hash,
        "test_sha256": test_features_hash,
        "external_probe_top1": round(hit, 6),
        "permuted_calibration_top1": round(null_hit, 6),
        "native_retained_true_p_gate_closed": round(float(true_probability(closed, y_test).mean()), 6),
        "native_stripped_true_p_gate_closed": round(float(true_probability(stripped, y_test).mean()), 6),
        "native_true_p_gate_010": round(float(true_probability(partly_open, y_test).mean()), 6),
        "native_true_p_gate_060": round(float(true_probability(open_logits, y_test).mean()), 6),
        "native_true_p_donor_gate_060": round(float(true_probability(open_donor, y_test).mean()), 6),
        "native_true_p_nuisance_sham_gate_060": round(float(true_probability(sham, y_test).mean()), 6),
        "native_true_p_visible_control": round(float(true_probability(visible, y_test).mean()), 6),
        "physical_donor_edit_mean_l2": round(float(np.linalg.norm(donor - x_test, axis=1).mean()), 6),
        "physical_nuisance_sham_edit_mean_l2": round(float(np.linalg.norm(matched_nuisance, axis=1).mean()), 6),
        "operator_coeff_depth4": round(propagation_coefficient(), 8),
    }


def run(n_test: int = 1000, per_class_cal: int = 32, seeds=SEEDS) -> dict:
    runs = [run_seed(s, n_test=n_test, per_class_cal=per_class_cal) for s in seeds]
    keys = ["external_probe_top1", "permuted_calibration_top1",
            "native_retained_true_p_gate_closed", "native_true_p_gate_010",
            "native_true_p_gate_060", "native_true_p_donor_gate_060",
            "native_true_p_nuisance_sham_gate_060", "native_true_p_visible_control"]
    summary = {k: round(float(np.mean([r[k] for r in runs])), 6) for k in keys}
    ranges = {k: [min(r[k] for r in runs), max(r[k] for r in runs)] for k in keys}
    return {"protocol": "SAU-TRACE02", "type": "synthetic_linear_recurrence",
            "visible_reply": VISIBLE_REPLY, "candidate_items": K,
            "test_episodes": len(runs) * n_test,
            "calibration_episodes": len(runs) * K * per_class_cal,
            "independent_rng_seeds": list(seeds), "depth": DEPTH,
            "gate_open": GATE_OPEN, "noise_sigma": NOISE,
            "native_readout": "softmax fixed 50-way task logits",
            "external_readout": "calibration-fitted nearest-template linear probe",
            "mean": summary, "seedwise_range": ranges, "seeds": runs}


class TestCausalAccess(unittest.TestCase):
    def test_closed_access_and_native_operator(self):
        x = np.eye(K)[:4]
        self.assertTrue(np.array_equal(native_logits(x, 0), np.zeros_like(x)))
        self.assertAlmostEqual(propagation_coefficient(1), GATE_OPEN)
        self.assertAlmostEqual(propagation_coefficient(DEPTH), 0.6942, places=10)
        self.assertTrue(np.allclose(native_logits(x, GATE_OPEN),
                                    GAIN * propagation_coefficient() * x, atol=1e-12))

    def test_readout_access_dissociation(self):
        result = run_seed(103, n_test=250, per_class_cal=16)
        self.assertGreater(result["external_probe_top1"], 0.7)
        self.assertLess(result["permuted_calibration_top1"], 0.1)
        self.assertEqual(result["native_retained_true_p_gate_closed"], 0.02)
        self.assertEqual(result["native_stripped_true_p_gate_closed"], 0.02)
        self.assertGreater(result["native_true_p_gate_060"], result["native_true_p_gate_010"])
        self.assertGreater(result["native_true_p_gate_010"], 0.02)

    def test_causal_donor_and_matched_nuisance(self):
        result = run_seed(211, n_test=250, per_class_cal=16)
        self.assertGreater(result["native_true_p_gate_060"],
                           result["native_true_p_donor_gate_060"] + 0.2)
        self.assertEqual(result["native_true_p_nuisance_sham_gate_060"],
                         result["native_true_p_gate_060"])
        self.assertEqual(result["physical_donor_edit_mean_l2"],
                         result["physical_nuisance_sham_edit_mean_l2"])
        self.assertGreater(result["native_true_p_visible_control"], 0.9)

    def test_seed_aggregate(self):
        result = run(n_test=250, per_class_cal=16, seeds=(103, 131, 167))
        self.assertEqual(result["test_episodes"], 750)
        self.assertEqual(result["calibration_episodes"], 2400)
        for row in result["seeds"]:
            self.assertGreater(row["external_probe_top1"], 0.7)
            self.assertGreater(row["native_true_p_gate_060"],
                               row["native_true_p_donor_gate_060"])


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
