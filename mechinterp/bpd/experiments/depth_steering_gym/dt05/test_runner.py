"""Admission and artifact regressions, using synthetic files and mocked launches."""
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import DEV_ROOTS
from .resources import MemoryBudget
from . import runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_cases(self, **overrides):
        solved = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]
        givens = [row[:] for row in solved]
        givens[0][1] = 0
        data = {"schema": "BPD-DT05/sudoku-cases/v1", "scope": "engineering",
                "cases": [{"id": "fixture", "givens": givens, "solution": solved}]}
        data.update(overrides)
        path = self.root / "cases.json"
        path.write_text(json.dumps(data))
        return path

    def test_replay_admits_unique_cases_with_explicit_native_encoding(self):
        path = self.write_cases()
        x, y, receipt = runner.load_replay_cases(path)
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(x[0, 1].item(), 1)
        self.assertEqual(y[0, 1].item(), 1)
        self.assertTrue(receipt["unique_solutions_verified"])
        self.assertEqual(receipt["sha256"], runner.sha256_file(path))

    def test_replay_rejects_confirmation_ambiguous_and_wrong_labels(self):
        with self.assertRaisesRegex(ValueError, "development/engineering"):
            runner.load_replay_cases(self.write_cases(scope="confirmation"))
        path = self.write_cases()
        data = json.loads(path.read_text())
        data["cases"][0]["givens"] = [[0] * 4 for _ in range(4)]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.load_replay_cases(path)
        path = self.write_cases()
        data = json.loads(path.read_text())
        data["cases"][0]["solution"][0][1] = 4
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.load_replay_cases(path)

    def test_root_guard_runs_before_any_plan_read(self):
        for root in (2603999, 2604000, 2604100, 2604355, 2605999, -1):
            with self.subTest(root=root), patch.object(runner, "read_plan") as read:
                with self.assertRaises(ValueError):
                    runner.main(["run", "--root", str(root), "--plan", "missing.json", "--out", str(self.root / "campaign")])
                read.assert_not_called()
        runner.guard_development_root(DEV_ROOTS[0])
        runner.guard_development_root(DEV_ROOTS[-1])
        for invalid in (True, "2605000", 2605000.0):
            with self.assertRaises(ValueError):
                runner.guard_development_root(invalid)

    def test_device_failure_happens_before_attempt_reservation(self):
        out = self.root / "campaign"
        plan = {"memory_budget": MemoryBudget().__dict__}
        with patch.object(runner, "read_plan", return_value=plan), patch.object(runner, "configure_runtime", side_effect=RuntimeError("missing CUDA")):
            with self.assertRaisesRegex(RuntimeError, "missing CUDA"):
                runner.main(["run", "--root", str(DEV_ROOTS[0]), "--plan", "missing.json", "--out", str(out)])
        self.assertFalse(out.exists())

    def test_duplicate_attempt_does_not_launch_again(self):
        plan_file = self.root / "plan.json"
        plan_file.write_text("{}")
        out = self.root / "campaign"
        plan = {"memory_budget": MemoryBudget().__dict__}
        args = ["run", "--root", str(DEV_ROOTS[0]), "--plan", str(plan_file), "--out", str(out)]
        with patch.object(runner, "read_plan", return_value=plan), patch.object(runner, "configure_runtime"), patch.object(runner, "run_development", return_value={"status": "FAILED"}) as launch, redirect_stdout(StringIO()):
            self.assertEqual(runner.main(args), 1)
            with self.assertRaises(FileExistsError):
                runner.main(args)
        self.assertEqual(launch.call_count, 1)
        self.assertTrue((out / f"attempted_{DEV_ROOTS[0]}.json").is_file())

    def test_replay_existing_output_precedes_model_access(self):
        out = self.root / "existing.json"
        out.write_text("keep me")
        with patch.object(runner.torch, "load") as load, patch.object(runner, "configure_runtime") as configure:
            with self.assertRaises(FileExistsError):
                runner.replay(Namespace(out=out))
        load.assert_not_called()
        configure.assert_not_called()
        self.assertEqual(out.read_text(), "keep me")

    def test_plan_source_and_cohort_are_bound(self):
        path = self.root / "plan.json"
        with patch.object(runner, "source_snapshot", return_value={"digest": "locked"}), redirect_stdout(StringIO()):
            runner.main(["plan", "--out", str(path)])
            self.assertEqual(runner.read_plan(path)["development_roots"], list(DEV_ROOTS))
            with self.assertRaises(FileExistsError):
                runner.main(["plan", "--out", str(path)])
        with patch.object(runner, "source_snapshot", return_value={"digest": "changed"}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                runner.read_plan(path)
        data = json.loads(path.read_text())
        data["development_roots"].append(2604100)
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "cohort"):
            runner.read_plan(path)


if __name__ == "__main__":
    unittest.main()
