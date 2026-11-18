"""Exact four-class six-token oracle and frozen SUB-08 data partition.

This module uses only NumPy and the Python standard library.  Labels and
restricted-feature ceilings are finite-domain arithmetic; no model is loaded.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
from math import fsum, log2
import hashlib
import json

import numpy as np

VOCAB = 4
SEQ_LEN = 6
TOKEN_NAMES = ("X", "S", "C", "R")
SPLIT_SEED = 246810
PARTITION_SLICES = {
    "train": (0, 2048),
    "validation": (2048, 2560),
    "reserved_dose": (2560, 3072),
    "heldout": (3072, 4096),
}
FEATURE_NAMES = tuple(f"prefix_{t}" for t in range(SEQ_LEN + 1)) + (
    "last_token", "token_counts"
)


def all_sequences() -> np.ndarray:
    """Return the full lexicographic universe as an (4096, 6) int64 array."""
    return np.asarray(list(product(range(VOCAB), repeat=SEQ_LEN)), dtype=np.int64)


def oracle_labels(seq: np.ndarray) -> np.ndarray:
    """Evaluate six ordered tokens, returning final-state class ``2*a+b``."""
    seq = np.asarray(seq)
    if seq.ndim != 2 or seq.shape[1] != SEQ_LEN:
        raise ValueError(f"Expected shape (N, {SEQ_LEN}); got {seq.shape}")
    if not np.issubdtype(seq.dtype, np.integer):
        raise TypeError("Token IDs must be integers")
    if np.any(seq < 0) or np.any(seq >= VOCAB):
        raise ValueError("Token IDs must be in {0,1,2,3}")
    a = np.zeros(len(seq), dtype=np.int64)
    b = np.ones(len(seq), dtype=np.int64)
    for position in range(SEQ_LEN):
        token = seq[:, position]
        # Both next-state coordinates use the original coordinates at this step.
        next_a = np.where(token == 0, a ^ 1,
                          np.where(token == 1, b,
                                   np.where(token == 3, a ^ b, a)))
        next_b = np.where(token == 1, a, np.where(token == 2, a ^ b, b))
        a, b = next_a, next_b
    return 2 * a + b


def partition(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sequences, labels, and original lexicographic IDs for one split."""
    if name not in PARTITION_SLICES:
        raise ValueError(f"Unknown partition {name!r}; choose {tuple(PARTITION_SLICES)}")
    start, stop = PARTITION_SLICES[name]
    ids = np.random.default_rng(SPLIT_SEED).permutation(VOCAB ** SEQ_LEN)[start:stop]
    ids = np.asarray(ids, dtype=np.int64)
    seq = all_sequences()[ids]
    return seq, oracle_labels(seq), ids


def _sha_i64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _entropy(counts) -> float:
    values = [int(x) for x in counts]
    total = sum(values)
    if total == 0:
        return 0.0
    return -fsum((count / total) * log2(count / total) for count in values if count)


def _feature(seq: tuple[int, ...], name: str):
    if name.startswith("prefix_"):
        return seq[:int(name.rsplit("_", 1)[1])]
    if name == "last_token":
        return (seq[-1],)
    if name == "token_counts":
        return tuple(seq.count(token) for token in range(VOCAB))
    raise ValueError(name)


def _feature_certificate(seq: np.ndarray, labels: np.ndarray, name: str) -> dict:
    groups: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    for tokens, label in zip(seq, labels, strict=True):
        key = _feature(tuple(int(token) for token in tokens), name)
        groups[key][int(label)] += 1
    n = len(labels)
    correct = sum(max(counts.values()) for counts in groups.values())
    conditional_entropy = fsum(
        sum(groups[key].values()) / n
        * _entropy([groups[key][label] for label in range(VOCAB)])
        for key in sorted(groups)
    )
    return {
        "feature": name,
        "n_cases": n,
        "n_feature_values": len(groups),
        "bayes_correct_numerator": correct,
        "bayes_denominator": n,
        "bayes_accuracy": correct / n,
        "conditional_entropy_bits": conditional_entropy,
    }


def exact_certificate() -> dict:
    """Return a JSON-native finite task certificate, including split ceilings.

    Counts are exact. Entropies numerically evaluate logarithms of exact
    frequencies. Every split's ceiling refers to its own empirical distribution.
    """
    seq = all_sequences()
    labels = oracle_labels(seq)
    class_counts = np.bincount(labels, minlength=VOCAB).tolist()
    permutation = np.random.default_rng(SPLIT_SEED).permutation(len(seq))
    partitions = {}
    for name, (start, stop) in PARTITION_SLICES.items():
        ids = permutation[start:stop]
        part_seq, part_labels = seq[ids], labels[ids]
        part_counts = np.bincount(part_labels, minlength=VOCAB).tolist()
        partitions[name] = {
            "n_cases": len(ids),
            "class_counts": part_counts,
            "entropy_bits": _entropy(part_counts),
            "ids_sha256": _sha_i64(ids),
            "features": {
                feature: _feature_certificate(part_seq, part_labels, feature)
                for feature in FEATURE_NAMES
            },
        }
    witness = np.asarray(((0, 0, 1, 1, 3, 2), (0, 0, 1, 1, 2, 3)), dtype=np.int64)
    witness_labels = oracle_labels(witness).tolist()
    return {
        "schema": "bpd-sub08-exact-oracle-v1",
        "token_names": list(TOKEN_NAMES),
        "class_encoding": "2*a+b",
        "initial_state": [0, 1],
        "sequence_length": SEQ_LEN,
        "n_cases": len(seq),
        "class_counts": class_counts,
        "entropy_bits": _entropy(class_counts),
        "sequences_le_i64_sha256": _sha_i64(seq),
        "labels_le_i64_sha256": _sha_i64(labels),
        "population_features": {
            feature: _feature_certificate(seq, labels, feature)
            for feature in FEATURE_NAMES
        },
        "partition_seed": SPLIT_SEED,
        "partition_rng": "numpy.random.default_rng/PCG64",
        "permutation_ids_sha256": _sha_i64(permutation),
        "partitions": partitions,
        "order_witness": {
            "tokens_a": witness[0].tolist(),
            "class_a": witness_labels[0],
            "tokens_b": witness[1].tolist(),
            "class_b": witness_labels[1],
            "common_counts": [2, 2, 1, 1],
        },
        "scope": "Exact finite-domain information certificate; no architecture-depth theorem.",
    }


if __name__ == "__main__":
    print(json.dumps(exact_certificate(), indent=2, sort_keys=True, allow_nan=False))
