"""Deterministic regressions for Tesseract gym manifest and CLI subcommands."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
from io import StringIO
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from . import runner
from .tesseract import (
    RunCancelled,
    TESSERACT_SCHEMA,
    build_tesseract_manifest,
    graceful_cancellation_guard,
    validate_tesseract_manifest,
)


class TesseractManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def sample_manifest(self):
        return build_tesseract_manifest(
            git_revision="a" * 40,
            source_digest="b" * 64,
            device="cuda:0",
            cpu_threads=8,
        )

    def test_manifest_builds_and_validates_all_scopes(self):
        manifest = self.sample_manifest()
        self.assertEqual(manifest["schema"], TESSERACT_SCHEMA)
        scopes = {w["id"]: w["evidence_scope"] for w in manifest["workloads"]}
        self.assertEqual(scopes["engineering_smoke_cuda"], "engineering")
        self.assertEqual(scopes["development_single_root_dt04"], "development")
        self.assertEqual(scopes["protected_confirmation_seal"], "protected")

    def test_validator_fails_closed_on_unsafe_mutations(self):
        base = self.sample_manifest()

        bad_net = copy.deepcopy(base)
        bad_net["environment"]["network_egress_permitted"] = True
        with self.assertRaisesRegex(ValueError, "network egress"):
            validate_tesseract_manifest(bad_net)

        bad_prot = copy.deepcopy(base)
        for w in bad_prot["workloads"]:
            if w["evidence_scope"] == "protected":
                w["permitted"] = True
        with self.assertRaisesRegex(ValueError, "Protected"):
            validate_tesseract_manifest(bad_prot)

        bad_retry = copy.deepcopy(base)
        for w in bad_retry["workloads"]:
            if w["evidence_scope"] == "development":
                w["retry_policy"]["auto_retry_permitted"] = True
        with self.assertRaisesRegex(ValueError, "max_attempts=1"):
            validate_tesseract_manifest(bad_retry)

    def test_graceful_cancellation_guard_raises_and_restores_handlers(self):
        orig_int = signal.getsignal(signal.SIGINT)
        seen = []
        with self.assertRaises(RunCancelled):
            with graceful_cancellation_guard(on_cancel=seen.append):
                handler = signal.getsignal(signal.SIGINT)
                handler(signal.SIGINT, None)
        self.assertEqual(seen, ["SIGINT"])
        self.assertEqual(signal.getsignal(signal.SIGINT), orig_int)

    def test_runner_cli_tesseract_manifest_and_calibrate_step(self):
        manifest_out = self.root / "tesseract.json"
        with patch.object(runner, "source_snapshot", return_value={"digest": "c" * 64, "git_head": "d" * 40}), \
                redirect_stdout(StringIO()):
            rc = runner.main(["tesseract-manifest", "--out", str(manifest_out), "--device", "cpu"])
        self.assertEqual(rc, 0)
        loaded = json.loads(manifest_out.read_text())
        validate_tesseract_manifest(loaded)

        step_out = self.root / "step_bench.json"
        with patch.object(runner, "source_snapshot", return_value={"digest": "c" * 64, "git_head": "d" * 40}), \
                redirect_stdout(StringIO()):
            rc2 = runner.main([
                "calibrate-step", "--profile", "mini", "--device", "cpu",
                "--logical-batch", "4", "--candidates", "2", "4",
                "--warmups", "0", "--repeats", "1",
                "--min-host-available-gib", "0.1",
                "--out", str(step_out),
            ])
        self.assertEqual(rc2, 0)
        step_data = json.loads(step_out.read_text())
        self.assertEqual(step_data["schema"], "BPD-DT05/full-step-benchmark/v1")
        self.assertEqual(step_data["status"], "BENCHMARKED")


if __name__ == "__main__":
    unittest.main()
