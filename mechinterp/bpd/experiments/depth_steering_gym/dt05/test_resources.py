"""DT05 resource tests use toy buffers and injected clocks; no study roots."""
from __future__ import annotations

import itertools
import math
import unittest
from unittest.mock import patch

import torch

from .resources import (
    GIB, BoundedTensorCache, BudgetExceeded, MemoryBudget,
    ReusableTransferBuffer, assert_budget, calibrate_microbatch, memory_snapshot,
)


def snapshot(device="cpu", **extra):
    return {
        "device": device, "host_rss_bytes": GIB,
        "host_peak_rss_bytes": 100 * GIB,
        "host_available_bytes": 20 * GIB,
        "cuda_reserved_bytes": GIB, "cuda_free_bytes": 12 * GIB,
        "cuda_peak_reserved_bytes": GIB, **extra,
    }


class ResourceTests(unittest.TestCase):
    def test_current_rss_separate_from_lifetime_peak(self):
        assert_budget(snapshot(), MemoryBudget())
        with self.assertRaises(BudgetExceeded):
            assert_budget(snapshot(host_rss_bytes=13 * GIB), MemoryBudget())

    def test_missing_required_telemetry_fails_closed(self):
        for key in ("host_rss_bytes", "host_available_bytes"):
            with self.subTest(key=key), self.assertRaises(BudgetExceeded):
                assert_budget(snapshot(**{key: None}), MemoryBudget())
        for key in ("cuda_reserved_bytes", "cuda_free_bytes"):
            with self.subTest(key=key), self.assertRaises(BudgetExceeded):
                assert_budget(snapshot("cuda", **{key: None}), MemoryBudget())
        with self.assertRaises(BudgetExceeded):
            assert_budget(snapshot(host_rss_bytes=math.nan), MemoryBudget())

    def test_budget_limits_validate_and_keep_reserves(self):
        for kwargs in ({"max_host_rss_gib": 0}, {"min_cuda_free_gib": -1}, {"min_host_available_gib": math.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MemoryBudget(**kwargs)
        with self.assertRaises(BudgetExceeded):
            assert_budget(snapshot(host_available_bytes=3 * GIB), MemoryBudget())
        with self.assertRaises(BudgetExceeded):
            assert_budget(snapshot("cuda", cuda_free_bytes=GIB // 2), MemoryBudget())

    def test_actual_snapshot_has_current_host_memory(self):
        measured = memory_snapshot()
        self.assertGreater(measured["host_rss_bytes"], 0)
        self.assertGreater(measured["host_available_bytes"], 0)
        self.assertIsNone(measured["cuda_free_bytes"])
        self.assertIsInstance(measured["errors"], list)

    def test_unavailable_cuda_is_explicit(self):
        with patch("torch.cuda.is_available", return_value=False):
            measured = memory_snapshot("cuda:0")
        self.assertIsNone(measured["cuda_free_bytes"])
        self.assertTrue(any("CUDA unavailable" in err for err in measured["errors"]))
        with self.assertRaises(BudgetExceeded):
            assert_budget(measured, MemoryBudget())

    def test_calibration_keeps_logical_batch_discards_warmups(self):
        # batch 2: warmup100s, measured1s/1s/1s; batch4:2s/2s/2s, tie goes smaller.
        durations = iter([100, 1, 1, 1, 100, 2, 2, 2])
        elapsed = [0.0]
        ticks = []
        for duration in durations:
            ticks.extend((elapsed[0], elapsed[0] + duration))
            elapsed[0] += duration
        clock = iter(ticks)
        calls = []
        report = calibrate_microbatch([4, 2], 16, calls.append, snapshot_fn=snapshot, clock=lambda: next(clock))
        self.assertEqual(report["selected_microbatch"], 2)
        self.assertEqual(report["accumulation_steps"], 8)
        self.assertEqual(report["trials"][0]["seconds"], [1, 1, 1])
        self.assertEqual(calls, [2] * 4 + [4] * 4)

    def test_inadmissible_or_failed_larger_trial_keeps_receipt(self):
        ticks = itertools.count(1)
        def work(batch):
            if batch == 4:
                raise MemoryError("toy exhausted")
        report = calibrate_microbatch([2, 4, 8], 16, work, warmups=0, repeats=1,
                                     snapshot_fn=snapshot, clock=lambda: next(ticks))
        self.assertEqual([r["status"] for r in report["trials"]], ["OK", "OOM", "SKIPPED_AFTER_LIMIT"])
        self.assertEqual(report["selected_microbatch"], 2)
        called = []
        report = calibrate_microbatch([2, 4], 16, called.append,
                                     snapshot_fn=lambda d: snapshot(d, host_available_bytes=0))
        self.assertEqual(called, [])
        self.assertIsNone(report["selected_microbatch"])
        self.assertEqual(report["trials"][0]["status"], "BUDGET_REJECTED")

    def test_calibration_rejects_changed_logical_batch_and_propagates_bugs(self):
        for candidates in ([], [3], [True], [0], [2.0]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                calibrate_microbatch(candidates, 16, lambda b: None)
        def bad_work(_):
            raise RuntimeError("shape bug")
        with self.assertRaisesRegex(RuntimeError, "shape bug"):
            calibrate_microbatch([2], 16, bad_work, snapshot_fn=snapshot)

    def test_growth_predictor_holds_host_reserve(self):
        snapshots = iter([
            snapshot(), snapshot(host_rss_bytes=5 * GIB),
            snapshot(host_rss_bytes=5 * GIB, host_available_bytes=5 * GIB),
        ])
        ticks = itertools.count(1)
        calls = []
        result = calibrate_microbatch([2, 4], 16, calls.append, warmups=0, repeats=1,
                                     snapshot_fn=lambda d: next(snapshots), clock=lambda: next(ticks))
        self.assertEqual(calls, [2])
        self.assertEqual(result["trials"][1]["status"], "BUDGET_REJECTED")

    def test_cache_owns_independent_storage_and_counts_lru(self):
        cache = BoundedTensorCache(32)
        source = torch.arange(4, dtype=torch.float32)
        self.assertTrue(cache.put("split-a", source))
        source.fill_(99)
        hit = cache.get("split-a")
        torch.testing.assert_close(hit, torch.arange(4, dtype=torch.float32))
        hit.zero_()
        torch.testing.assert_close(cache.get("split-a"), torch.arange(4, dtype=torch.float32))
        cache.put("split-b", torch.ones(4))
        cache.get("split-a")
        cache.put("split-c", torch.zeros(4))
        self.assertIsNone(cache.get("split-b"))
        self.assertEqual(cache.stats()["bytes_used"], 32)
        self.assertEqual(cache.stats()["evictions"], 1)

    def test_oversize_or_empty_cache_entry_preserves_previous(self):
        cache = BoundedTensorCache(16)
        cache.put("key", torch.ones(4))
        self.assertFalse(cache.put("key", torch.zeros(5)))
        self.assertFalse(cache.put("empty", torch.zeros(0)))
        torch.testing.assert_close(cache.get("key"), torch.ones(4))
        with self.assertRaises(ValueError):
            cache.put("bad", [1, 2, 3])
        with self.assertRaises(ValueError):
            cache.put("", torch.ones(4))

    def test_cache_rows_copy_only_requested_rows_and_validate_indices(self):
        cache = BoundedTensorCache(1024)
        source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        cache.put("train-only", source)
        indices = torch.tensor([4, 1, 4], dtype=torch.int64)
        selected = cache.get_rows("train-only", indices)
        torch.testing.assert_close(selected, source[indices])
        selected.zero_()
        torch.testing.assert_close(cache.get_rows("train-only", indices), source[indices])
        self.assertEqual(cache.get_rows("train-only", torch.tensor([], dtype=torch.int64)).shape, (0, 4))
        self.assertIsNone(cache.get_rows("missing", indices))
        for bad in (torch.tensor([-1]), torch.tensor([6]), torch.tensor([1.0]), torch.tensor([[1]]), [1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cache.get_rows("train-only", bad)
        cache.put("scalar", torch.tensor(1))
        with self.assertRaises(ValueError):
            cache.get_rows("scalar", torch.tensor([0]))

    def test_cache_entry_cap_and_replacement_accounting(self):
        cache = BoundedTensorCache(1024, max_entries=2)
        cache.put("a", torch.ones(4))
        cache.put("b", torch.ones(4))
        cache.put("a", torch.ones(8))
        self.assertEqual(cache.stats()["bytes_used"], 48)
        cache.put("c", torch.ones(4))
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.stats()["entries"], 2)
        cache.clear()
        self.assertEqual(cache.stats()["bytes_used"], 0)
        self.assertEqual(cache.stats()["entries"], 0)

    def test_live_ram_admission_get_rows_into_and_transfer_buffer(self):
        cache = BoundedTensorCache(1024)
        budget = MemoryBudget(max_host_rss_gib=12.0, min_host_available_gib=4.0)
        self.assertTrue(cache.put_admitted("ok", torch.arange(12, dtype=torch.float32).reshape(3, 4),
                                           budget, snapshot=snapshot()))
        self.assertFalse(cache.put_admitted("deny", torch.ones(4),
                                            budget, snapshot=snapshot(host_available_bytes=3 * GIB)))
        self.assertGreaterEqual(cache.stats()["rejected_admissions"], 1)
        out = torch.empty((2, 4), dtype=torch.float32)
        idx = torch.tensor([2, 0], dtype=torch.int64)
        res = cache.get_rows_into("ok", idx, out)
        self.assertIs(res, out)
        torch.testing.assert_close(out[0], torch.tensor([8., 9., 10., 11.]))
        self.assertEqual(cache.stats()["recomputations_avoided"], 2)
        self.assertGreater(cache.stats()["hit_rate"], 0.0)

        buf = ReusableTransferBuffer((4, 3), torch.long, "cpu", pin_memory=False)
        src = torch.arange(6, dtype=torch.long).reshape(2, 3)
        sent = buf.stage_and_send(src, count=2)
        self.assertTrue(torch.equal(sent, src))
        self.assertEqual(buf.transfers, 1)


if __name__ == "__main__":
    unittest.main()
