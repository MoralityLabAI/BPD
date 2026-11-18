"""Independent arithmetic tests for SUB-08. No model package is imported."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from math import log2
from pathlib import Path
import unittest

import numpy as np

_SPEC = importlib.util.spec_from_file_location("sub08_oracle_under_test", Path(__file__).with_name("oracle.py"))
assert _SPEC is not None and _SPEC.loader is not None
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)

# Literal state-transition table, independently written for classes 2*a+b.
# Rows name the token X,S,C,R; columns name the starting class 0,1,2,3.
TRANSITIONS = ((2, 3, 0, 1), (0, 2, 1, 3), (0, 1, 3, 2), (0, 3, 2, 1))


def literal_labels(sequences):
    result = []
    for tokens in sequences:
        state = 1
        for token in tokens:
            state = TRANSITIONS[int(token)][state]
        result.append(state)
    return np.asarray(result, dtype=np.int64)


def independent_feature_counts(sequences, labels, feature):
    groups = defaultdict(Counter)
    for tokens, label in zip(sequences, labels, strict=True):
        if isinstance(feature, int):
            key = tuple(tokens[:feature])
        elif feature == "last":
            key = int(tokens[-1])
        else:
            key = tuple(sum(int(x) == token for x in tokens) for token in range(4))
        groups[key][int(label)] += 1
    correct = sum(max(counts.values()) for counts in groups.values())
    entropy = 0.0
    for counts in groups.values():
        n = sum(counts.values())
        entropy -= sum(count / len(labels) * log2(count / n) for count in counts.values())
    return correct, entropy


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seq = oracle.all_sequences()
        cls.labels = literal_labels(cls.seq)
        cls.cert = oracle.exact_certificate()

    def test_exhaustive_literal_transition_match(self):
        self.assertEqual(self.seq.shape, (4096, 6))
        self.assertEqual(self.seq.dtype, np.dtype("int64"))
        encoded = self.seq @ (4 ** np.arange(5, -1, -1))
        np.testing.assert_array_equal(encoded, np.arange(4096))
        np.testing.assert_array_equal(oracle.oracle_labels(self.seq), self.labels)
        np.testing.assert_array_equal(np.bincount(self.labels, minlength=4), [781, 1146, 1024, 1145])

    def test_all_population_feature_certificates(self):
        expected = {0: 1146, 1: 1147, 2: 1232, 3: 1355, 4: 1556, 5: 1967, 6: 4096,
                    "last": 1188, "counts": 1603}
        for feature, numerator in expected.items():
            name = f"prefix_{feature}" if isinstance(feature, int) else {"last": "last_token", "counts": "token_counts"}[feature]
            correct, entropy = independent_feature_counts(self.seq, self.labels, feature)
            self.assertEqual(correct, numerator)
            observed = self.cert["population_features"][name]
            self.assertEqual(observed["bayes_correct_numerator"], numerator)
            self.assertEqual(observed["bayes_denominator"], 4096)
            self.assertAlmostEqual(observed["conditional_entropy_bits"], entropy, places=12)

    def test_same_counts_order_witness(self):
        a = [0, 0, 1, 1, 3, 2]
        b = [0, 0, 1, 1, 2, 3]
        self.assertEqual(Counter(a), Counter(b))
        np.testing.assert_array_equal(literal_labels([a, b]), [2, 3])
        self.assertEqual(self.cert["order_witness"]["class_a"], 2)
        self.assertEqual(self.cert["order_witness"]["class_b"], 3)

    def test_exact_partition_and_hashes(self):
        permutation = np.random.Generator(np.random.PCG64(246810)).permutation(4096)
        sizes = {"train": 2048, "validation": 512, "reserved_dose": 512, "heldout": 1024}
        counts = {"train": [384, 548, 525, 591], "validation": [106, 158, 123, 125],
                  "reserved_dose": [102, 136, 122, 152], "heldout": [189, 304, 254, 277]}
        all_ids = []
        offset = 0
        for name, size in sizes.items():
            seq, labels, ids = oracle.partition(name)
            np.testing.assert_array_equal(ids, permutation[offset:offset + size])
            np.testing.assert_array_equal(seq, self.seq[ids])
            np.testing.assert_array_equal(labels, literal_labels(seq))
            self.assertEqual(ids.dtype, np.dtype("int64"))
            self.assertEqual(len(set(ids)), size)
            cert = self.cert["partitions"][name]
            self.assertEqual(cert["class_counts"], counts[name])
            self.assertEqual(cert["ids_sha256"], hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest())
            all_ids.extend(int(i) for i in ids)
            offset += size
        self.assertEqual(len(all_ids), 4096)
        self.assertEqual(set(all_ids), set(range(4096)))

    def test_frozen_heldout_feature_ceilings(self):
        seq, labels, _ = oracle.partition("heldout")
        expected = {0: 304, 1: 310, 2: 347, 3: 408, 4: 531, 5: 756, 6: 1024,
                    "last": 304, "counts": 449}
        for feature, numerator in expected.items():
            name = f"prefix_{feature}" if isinstance(feature, int) else {"last": "last_token", "counts": "token_counts"}[feature]
            correct, entropy = independent_feature_counts(seq, labels, feature)
            self.assertEqual(correct, numerator)
            observed = self.cert["partitions"]["heldout"]["features"][name]
            self.assertEqual(observed["bayes_correct_numerator"], numerator)
            self.assertEqual(observed["bayes_denominator"], 1024)
            self.assertAlmostEqual(observed["conditional_entropy_bits"], entropy, places=12)

    def test_json_native_and_archived_certificate(self):
        self.assertEqual(json.loads(json.dumps(self.cert, allow_nan=False)), self.cert)
        archived = Path(__file__).with_name("EXACT_ORACLE.json")
        if archived.exists():
            self.assertEqual(json.loads(archived.read_text()), self.cert)

    def test_bad_inputs_rejected(self):
        for seq in (np.zeros((6,), dtype=np.int64), np.zeros((2, 5), dtype=np.int64),
                    np.full((1, 6), 4, dtype=np.int64), np.full((1, 6), -1, dtype=np.int64)):
            with self.assertRaises(ValueError):
                oracle.oracle_labels(seq)
        with self.assertRaises(TypeError):
            oracle.oracle_labels(np.zeros((1, 6), dtype=np.float64))
        with self.assertRaises(ValueError):
            oracle.partition("discovery")


if __name__ == "__main__":
    unittest.main()
