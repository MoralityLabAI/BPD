"""Read-only local UI regression checks on archived engineering evidence.

Never open ARG-03-N, start training, or generate an experimental root cohort.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from .dt03_tasks import rollout_reachable
from .ui_server import (
    ARCHIVE, INDEX, EXPECTED_ARCHIVE_SHA256, ThreadingHTTPServer,
    api_response, fixture_seed, handler_for, load_archive, overview,
    task_fixture,
)


class UIDataContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive=load_archive()

    def test_archive_is_immutable_original_source(self):
        result=self.archive
        self.assertEqual(result["experiment"],"BPD-DT02-A")
        self.assertEqual(result["supervised_depths"],[2,4,6])
        self.assertEqual(result["heldout_depths"],[8,12])
        self.assertFalse(result["official_arc1_evaluated"])
        self.assertEqual([v["seed"] for v in result["roots"]],[901,902])
        self.assertEqual(len(EXPECTED_ARCHIVE_SHA256),64)
        self.assertTrue(INDEX.is_file())

    def test_unverified_alternative_archive_rejected(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            p=Path(d)/"fake.json"
            p.write_text('{"experiment":"BPD-DT02-A"}',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                load_archive(p)

    def test_overview_preserves_task_failure_and_steering(self):
        report=overview(self.archive)
        self.assertEqual(report["schema"],"bpd-dt03-ui-overview-v1")
        self.assertEqual(len(report["roots"]),2)
        self.assertEqual(report["evidence"],"SOURCE_LOCKED_ENGINEERING_ONLY")
        for root in report["roots"]:
            self.assertEqual(set(root["families"]),{"sudoku4","arc_style","storyworld"})
            self.assertEqual(root["families"]["sudoku4"]["depths"]["8"]["exact_task_accuracy"],0)
            self.assertEqual(root["families"]["arc_style"]["depths"]["12"]["exact_task_accuracy"],0)
            self.assertEqual(root["families"]["storyworld"]["depths"]["12"]["host_ending_success_rate"],0)
            self.assertIn("protected_all_vocab_max_error",root["families"]["sudoku4"]["steering"])

    def test_sudoku_ui_fixture_is_unique_and_source_only(self):
        r=task_fixture("sudoku",42)
        self.assertEqual(r["task"],"sudoku")
        self.assertFalse(r["registered_evaluation_root"])
        self.assertEqual(r["metadata"]["blanks"],6)
        self.assertTrue(r["metadata"]["unique_solution"])
        self.assertEqual(len(r["public"]["givens"]),4)
        self.assertEqual(len(r["host_gold"]["solution"]),4)
        self.assertEqual(r,task_fixture("sudoku",42))
        self.assertNotEqual(fixture_seed(42,"sudoku"),fixture_seed(42,"arc"))

    def test_arc_public_demos_and_query_exclude_host_gold(self):
        for split in ("train","heldout"):
            r=task_fixture("arc",13,split=split)
            self.assertEqual(len(r["public"]["train"]),3)
            self.assertEqual(len(r["public"]["test"]),1)
            self.assertNotIn("output",r["public"]["test"][0])
            self.assertNotIn("host_gold",r["public"])
            self.assertEqual(r["metadata"]["input_token_length"],127)
            self.assertEqual(r["metadata"]["query_position_count"],16)
            self.assertEqual(len(r["host_gold"]["answer"]),4)

    def test_story_observation_hides_ending_until_terminal(self):
        r=task_fixture("story",22,horizon=12)
        self.assertNotIn("host_gold",r)
        self.assertNotIn("secret_cause",r["public"])
        self.assertNotIn("ending",r["public"])
        derived=fixture_seed(22,"story")
        actions=rollout_reachable(derived,12)["actions"]
        done=task_fixture("story",22,horizon=12,actions=tuple(actions))
        self.assertEqual(done["public"]["day"],12)
        self.assertTrue(done["public"]["terminal"])
        self.assertTrue(done["host_gold"]["success"])
        self.assertNotEqual(done["host_gold"]["ending"],"unfinished")

    def test_invalid_and_reused_action_requests_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"root|seed"):
            task_fixture("sudoku",-2)
        with self.assertRaisesRegex(ValueError,"split"):
            task_fixture("arc",3,split="private")
        with self.assertRaisesRegex(ValueError,"horizon"):
            task_fixture("story",3,horizon=3)
        with self.assertRaisesRegex(ValueError,"action"):
            task_fixture("story",3,actions=("invent_action",))
        with self.assertRaisesRegex(ValueError,"after terminal"):
            task_fixture("story",3,actions=("listen",)*13)
        with self.assertRaises(KeyError):
            api_response("/api/secret",{},self.archive)

    def test_html_no_remote_script_or_scientific_action(self):
        markup=INDEX.read_text(encoding="utf-8")
        self.assertIn("TRM Depth Lab",markup)
        self.assertIn("Task explorer",markup)
        self.assertIn("Causal diagnostics",markup)
        self.assertNotIn("src=\"https://",markup)
        self.assertNotIn("window.localStorage",markup)
        self.assertNotIn("training.run",markup)


class UIRouteContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report=load_archive()
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),
                                       handler_for(report,INDEX.read_bytes()))
        cls.thread=Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.base=f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_serves_browser_and_valid_json(self):
        with urlopen(self.base+"/",timeout=8) as response:
            self.assertEqual(response.status,200)
            self.assertIn(b"TRM Depth Lab",response.read())
            self.assertEqual(response.headers["X-Frame-Options"],"DENY")
        with urlopen(self.base+"/api/overview",timeout=8) as response:
            body=json.load(response)
            self.assertEqual(body["roots"][0]["seed"],901)
        with urlopen(self.base+"/api/fixture?task=arc&seed=14",timeout=8) as response:
            body=json.load(response)
            self.assertNotIn("output",body["public"]["test"][0])

    def test_rejects_unknown_endpoints_and_invalid_input(self):
        for url,status in (("/etc/passwd",404),("/api/fixture?task=story&horizon=1",400),
                           ("/api/fixture?task=sudoku&seed=-1",400)):
            with self.assertRaises(HTTPError) as ctx:
                urlopen(self.base+url,timeout=8)
            self.assertEqual(ctx.exception.code,status)


if __name__=="__main__":
    unittest.main()
