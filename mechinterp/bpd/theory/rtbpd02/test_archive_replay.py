"""Public microfixtures for the private archive replay contract."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from archive_analysis import CompactWrite, outside_support, split_root, train_candidates, transfer
from archive_checkpoint import Shard, double_rms_jvp, round_bf16
from archive_inputs import Bank, Occurrence, select_cal, state_arrays, verify_case


class ArchiveReplayTests(unittest.TestCase):
    def test_whole_root_selection_and_token_custody_fail_closed(self):
        pairs = [
            {"id": f"root-{w}", "root": "root", "world": w, "split": "calibration", "arm": "focused"}
            for w in ("A", "B")
        ]
        rows = [*pairs, dict(pairs[0], split="development", id="development-A")]
        spec = {"expected_roots": 1, "fit_split": "calibration", "arm": "focused"}
        payload = "\n".join(map(json.dumps, rows)).encode()
        selected, roots = select_cal(payload, spec)
        self.assertEqual(roots, ("root",))
        self.assertEqual(selected, list(enumerate(pairs)))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_cal(payload + b"\n" + json.dumps(pairs[0]).encode(), spec)

        ids = [1, 72, 13]
        digest = sha256(json.dumps(ids).encode()).hexdigest()
        case = dict(pairs[0], status="completed", initial_prompt_ids=ids, initial_prompt_ids_sha256=digest,
                    state_file="case_0000_states.npz", state_file_sha256="state", initial_phases=[{"loop": 1}, {"loop": 2}],
                    steps=[{"step": 0, "history_tokens": 3}])
        audit = dict(pairs[0], state_sha256="state")
        token = {"tokens": 3, "ids_sha256": digest}
        verify_case(case, audit, pairs[0], token, "case_0000", [1, 2])
        with self.assertRaisesRegex(ValueError, "token"):
            verify_case(case, audit, pairs[0], dict(token, ids_sha256="alien"), "case_0000", [1, 2])

    def test_vector_index_joins_physical_site_and_phase(self):
        indices, arrays = [], {}
        for p in (1, 2):
            for site, width in (("down_proj_output", 2), ("residual_after_layer", 2), ("down_proj_input", 3)):
                key = f"v{len(indices)}"
                indices.append(dict(loop=p, site=site, layer_zero_based=21, array_key=key))
                arrays[key] = np.full(width, len(indices), dtype=np.float32)
        buf = BytesIO()
        np.savez(buf, **arrays)
        values = state_arrays({"state_index": indices}, buf.getvalue(), 21, [1, 2], 3, 2)
        np.testing.assert_array_equal(values[1][0], [3., 3., 3.])
        np.testing.assert_array_equal(values[2][1], [4., 4.])
        broken_index = [*indices, dict(indices[-1], array_key="duplicate")]
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            state_arrays({"state_index": broken_index}, buf.getvalue(), 21, [1, 2], 3, 2)

    def test_safetensors_row_stream_and_round_to_nearest_even(self):
        header = json.dumps({"weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
        data = np.array([0x3F80, 0x4000, 0x4040, 0xC080], dtype="<u2").tobytes()
        content = len(header).to_bytes(8, "little") + header + data
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.safetensors"
            path.write_bytes(content)
            with path.open("rb") as stream:
                checked = Shard(stream, sha256(content).hexdigest(), lambda: None)
                np.testing.assert_array_equal(checked.tensor("weight", (2, 2), (1, 2)), [[3., -4.]])
                with self.assertRaisesRegex(ValueError, "shape"):
                    checked.tensor("weight", (1, 4))
            with path.open("rb") as stream, self.assertRaisesRegex(ValueError, "hash"):
                Shard(stream, "bad", lambda: None)
        midpoint = np.array([0x3F808000, 0x3F818000], dtype=np.uint32).view(np.float32)
        np.testing.assert_array_equal(round_bf16(midpoint).view(np.uint32), [0x3F800000, 0x3F820000])

    def test_fixed_double_rms_tangent_via_finite_difference(self):
        x, g, h = np.array([1., 4., 2.]), np.array([.75, .25, 2.]), np.array([2., -1., 1.])
        basis = np.array([[1., .25], [1., 2.], [-1., 2.]])
        eps, step = .13, 1e-5

        def nonlinear(y):
            first = y * g / np.sqrt(np.mean(y ** 2) + eps)
            return first * h / np.sqrt(np.mean(first ** 2) + eps)

        difference = np.stack([(nonlinear(x + step * v) - nonlinear(x - step * v)) / (2 * step) for v in basis.T], axis=1)
        np.testing.assert_allclose(double_rms_jvp(x, basis, g, h, eps), difference, atol=1e-9)

    def test_heldout_root_does_not_enter_reach_or_carrier_fit(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(5, 2, 4))
        carrier = rng.normal(size=(5, 2, 3))
        original = Occurrence(x, np.zeros_like(carrier), carrier)
        modified = Occurrence(x.copy(), np.zeros_like(carrier), carrier.copy())
        modified.x[2] += 1000
        modified.carrier[2] += 1000
        a, r = split_root(original, 2)
        b, s = split_root(modified, 2)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(r, s)
        bank = Bank(np.eye(4)[:, :3], np.diag([4., 3., 1.]))
        target = CompactWrite(bank)
        first, _ = train_candidates(target, {"t2": bank}, a, 1, np.eye(3), 1e-12, np.eye(3))
        second, _ = train_candidates(target, {"t2": bank}, b, 1, np.eye(3), 1e-12, np.eye(3))
        for name in first:
            np.testing.assert_array_equal(first[name].apply(x[2]), second[name].apply(x[2]))

    def test_transport_reports_off_support_and_interference(self):
        a, b = Bank(np.array([[1.]]), np.array([[1.]])), Bank(np.array([[2.]]), np.array([[1.]]))
        report = transfer(a, b, np.array([[2.], [4.]]), np.array([[1.], [2.]]), 1e-12)
        self.assertEqual(report["total_below_sum_due_to_cancellation"], 2)
        self.assertEqual(report["mean_root_energy"]["total"], 0.)
        self.assertEqual(outside_support(np.array([[1., 0.]]), np.array([[0., 1.]]), 1e-12), 1.)


if __name__ == "__main__":
    unittest.main()
