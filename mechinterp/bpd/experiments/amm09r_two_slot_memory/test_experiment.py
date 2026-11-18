"""Count-accounting and access-boundary fixtures; no model fitting or root sampling."""
from __future__ import annotations

import copy
from contextlib import ExitStack, contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
PACKAGE="_amm09r1_count_fixtures"
package=types.ModuleType(PACKAGE)
package.__path__=[str(ROOT)]
sys.modules[PACKAGE]=package


def load(name):
    spec=importlib.util.spec_from_file_location(PACKAGE+"."+name,ROOT/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


m=load("models")
case=load("case")
e=load("experiment")


@contextmanager
def accepted_runtime_fixture(*,wrong_binary=False,wrong_source=False):
    import numpy
    manifest=json.loads((ROOT/"RUNTIME.json").read_text())
    binaries=dict(manifest["torch_binary_sha256"])
    if wrong_binary:
        binaries["_C.so"]="0"*64
    with ExitStack() as stack:
        stack.enter_context(patch.dict(m.os.environ,manifest["environment"]))
        stack.enter_context(patch.object(m.platform,"python_version",return_value=manifest["versions"]["python"]))
        stack.enter_context(patch.object(m.torch,"__version__",manifest["versions"]["torch"]))
        stack.enter_context(patch.object(numpy,"__version__",manifest["versions"]["numpy"]))
        stack.enter_context(patch.object(m.torch,"get_num_threads",return_value=1))
        stack.enter_context(patch.object(m.torch,"get_num_interop_threads",return_value=1))
        stack.enter_context(patch.object(m.torch,"are_deterministic_algorithms_enabled",return_value=True))
        stack.enter_context(patch.object(m.torch,"is_deterministic_algorithms_warn_only_enabled",return_value=False))
        stack.enter_context(patch.object(m.torch,"get_default_dtype",return_value=m.torch.float32))
        stack.enter_context(patch.object(m.torch.backends.cpu,"get_cpu_capability",return_value="DEFAULT"))
        stack.enter_context(patch.object(m.torch.version,"git_version","0"*40 if wrong_source else manifest["torch_git"]))
        stack.enter_context(patch.object(m,"_binary_hashes",return_value=binaries))
        yield


def assay_fixture(root,split,n,offset=0):
    counts=(20,120+offset,4,50,0,1)
    bits={f"mask_{action}":"1"*min(n,count)+"0"*(n-min(n,count))
          for action,count in enumerate(counts)}
    bits.update({arm:"0"*n for arm in e.CONTROL_ARMS})
    return {"root":root,"split":split,"n":n,"candidate_bits":bits,
            "candidate_hits":{key:value.count("1") for key,value in bits.items()},
            "diagnostics":{"casewise_oracle_hits_audit_only":min(n,max(counts)),
                           "full_donor_hits_audit_only":0,
                           "max_equal_dose_error":0.,
                           "max_immediate_readout_displacement":0.}}


def root_fixture(index):
    root=f"development/{index:04d}"
    held=assay_fixture(root,"heldout",512,index%2)
    repeat=assay_fixture(root,"repeatability",512,index%2)
    selected={"memory_only":1,"memory_plus_local":3,"local_ridge":2,
              "pooled":0,"local_count":0,"random_mask":5}
    actions={arm:[action]*512 for arm,action in selected.items()}
    bits={arm:held["candidate_bits"][f"mask_{action}"] for arm,action in selected.items()}
    bits.update({arm:held["candidate_bits"][arm] for arm in e.CONTROL_ARMS})
    reports={f"mask_{action}":"0"*16 for action in range(6)}
    return {"root":root,"family":m.family_for(index),
            "fit":{"teacher_task_mse":.00001,"factor_task_mse":.001,
                   "weight_reconstruction_relative":.05,"min_rank3_gap":.05},
            "fidelity_audit_only":True,
            "calibration_clean_audit_only":assay_fixture(root,"calibration",16),
            "agent_noisy_feedback":{"reported_bits":reports,
                "reported_hits":{key:0 for key in reports},"noisy_labels":96,
                "n_local_cases":16,"known_judge_error":.20},
            "local_policy":{"pooled_action":0,"local_count_action":0,"prior_weights":[0.]*18},
            "heldout":held,"policy_actions":actions,"policy_bits":bits,
            "policy_hits":{key:value.count("1") for key,value in bits.items()},
            "repeatability":{"assay":repeat,"local_ridge_actions":[2]*512,
                "local_ridge_bits":repeat["candidate_bits"]["mask_2"],"local_ridge_hits":4,
                "pooled_bits":repeat["candidate_bits"]["mask_0"],"pooled_hits":20}}


def cohort_fixture():
    rows=[root_fixture(index) for index in range(64)]
    return {"phase":"development","n_independent_roots":64,
            "n_test_pairs_per_root":512,"total_test_cases":64*512,
            "pooled_action":0,"global_coefficients":[0.]*18,
            "rows":rows,"totals":{arm:sum(row["policy_hits"][arm] for row in rows) for arm in m.ARMS}}


class AMM09R1AccountingTests(unittest.TestCase):
    def test_same_version_with_wrong_torch_binary_is_rejected(self):
        with accepted_runtime_fixture(wrong_binary=True):
            with self.assertRaisesRegex(ValueError,"torch binary bytes"):
                m.require_runtime()

    def test_same_version_with_wrong_torch_source_is_rejected(self):
        with accepted_runtime_fixture(wrong_source=True):
            with self.assertRaisesRegex(ValueError,"torch source"):
                m.require_runtime()

    def test_accepted_runtime_signature_keeps_expected_binary_identity(self):
        with accepted_runtime_fixture():
            signature=m.runtime_signature()
        manifest=json.loads((ROOT/"RUNTIME.json").read_text())
        self.assertEqual(signature["torch_git"],manifest["torch_git"])
        self.assertEqual(signature["torch_binary_sha256"],manifest["torch_binary_sha256"])

    def test_engineering_report_hash_is_checked_before_parsing_its_verdict(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            fixture=Path(directory)
            (fixture/"results").mkdir()
            (fixture/"RUNTIME.json").write_bytes((ROOT/"RUNTIME.json").read_bytes())
            (fixture/"results"/"ENGINEERING_ACCEPTANCE.json").write_text("{invalid JSON")
            with patch.object(m,"ROOT",fixture):
                with self.assertRaisesRegex(ValueError,"acceptance byte digest"):
                    m.require_phase_admission("training")

    def test_primary_uses_memory_only_and_all_comparisons_name_their_arms(self):
        result=e.analysis(cohort_fixture())
        primary=result["comparisons"]["memory_only_vs_pooled"]
        secondary=result["comparisons"]["memory_plus_local_vs_pooled"]
        self.assertEqual(primary["mean_root_effect"],100.5/512)
        self.assertEqual(secondary["mean_root_effect"],30/512)
        self.assertEqual(primary["positive_roots"],64)
        self.assertEqual(result["comparison_arms"]["memory_only_vs_pooled"],
                         {"treatment":"memory_only","control":"pooled"})
        self.assertFalse(result["confirmation_authorized"])

    def test_swapped_root_counts_cannot_hide_behind_unchanged_aggregate(self):
        result=cohort_fixture()
        before=copy.deepcopy(result["totals"])
        first,second=result["rows"][:2]
        first["policy_hits"]["memory_only"],second["policy_hits"]["memory_only"]=(
            second["policy_hits"]["memory_only"],first["policy_hits"]["memory_only"])
        self.assertEqual(before,result["totals"])
        self.assertEqual(sum(row["policy_hits"]["memory_only"] for row in result["rows"]),
                         before["memory_only"])
        with self.assertRaisesRegex(ValueError,"per-root policy count mismatch"):
            e.analysis(result)
        with self.assertRaisesRegex(ValueError,"per-root policy count mismatch"):
            e.preflight_input(result)

    def test_admitted_action_cannot_select_a_seventh_mask(self):
        row=root_fixture(0)
        row["policy_actions"]["memory_only"][0]=6
        with self.assertRaisesRegex(ValueError,"inadmissible"):
            e.validated_root_counts(row)

    def test_boolean_action_is_not_an_integer_mask_index(self):
        row=root_fixture(0)
        row["policy_actions"]["memory_only"][0]=True
        with self.assertRaisesRegex(ValueError,"inadmissible"):
            e.validated_root_counts(row)

    def test_seventh_candidate_cannot_enter_an_assay(self):
        row=root_fixture(0)
        row["heldout"]["candidate_bits"]["mask_6"]="0"*512
        row["heldout"]["candidate_hits"]["mask_6"]=0
        with self.assertRaisesRegex(ValueError,"candidate or arm set"):
            e.validated_root_counts(row)

    def test_selected_bits_are_rebuilt_from_candidate_and_action(self):
        row=root_fixture(0)
        row["policy_bits"]["memory_only"]="0"*512
        row["policy_hits"]["memory_only"]=0
        with self.assertRaisesRegex(ValueError,"selected outcome bits"):
            e.validated_root_counts(row)

    def test_calibration_alphabet_and_counts_are_checked(self):
        row=root_fixture(0)
        row["agent_noisy_feedback"]["reported_bits"]["mask_0"]="x"+"0"*15
        with self.assertRaisesRegex(ValueError,"malformed binary"):
            e.validated_root_counts(row)
        row=root_fixture(0)
        row["agent_noisy_feedback"]["reported_hits"]["mask_0"]=1
        with self.assertRaisesRegex(ValueError,"noisy calibration count"):
            e.validated_root_counts(row)

    def test_fidelity_and_repeatability_cannot_be_forged(self):
        row=root_fixture(0)
        row["fidelity_audit_only"]=False
        with self.assertRaisesRegex(ValueError,"fidelity flag"):
            e.validated_root_counts(row)
        row=root_fixture(0)
        row["repeatability"]["pooled_hits"]+=1
        with self.assertRaisesRegex(ValueError,"repeatability count"):
            e.validated_root_counts(row)

    def test_root_membership_and_aggregate_counts_are_checked(self):
        result=cohort_fixture()
        result["rows"][1]=copy.deepcopy(result["rows"][0])
        with self.assertRaisesRegex(ValueError,"missing, duplicated, or reordered"):
            e.analysis(result)
        result=cohort_fixture()
        result["totals"]["memory_only"]+=1
        with self.assertRaisesRegex(ValueError,"cohort aggregate"):
            e.analysis(result)

    def test_preflight_uses_reconstructed_primary_and_same_policy_sham(self):
        result=e.preflight_input(cohort_fixture())
        self.assertEqual(result["pilot"][0]["paired_effect"],"100/512")
        self.assertEqual(result["pilot"][1]["paired_effect"],"101/512")
        self.assertEqual(result["pilot"][0]["sham_effects"],["0/512"])
        self.assertEqual(result["planned_comparisons"],4)
        self.assertEqual(result["scientific_min_effect"],"0.08")

    def test_root_pooled_policy_must_match_frozen_global_memory(self):
        result=cohort_fixture()
        result["pooled_action"]=1
        with self.assertRaisesRegex(ValueError,"frozen cohort memory"):
            e.analysis(result)

    def test_confirmation_rejected_before_runtime_or_rng_access(self):
        with (patch.object(m,"require_runtime",side_effect=AssertionError("runtime touched")) as runtime,
              patch.object(m.torch,"Generator",side_effect=AssertionError("RNG touched")) as generator):
            calls=(lambda:m.root_key("confirmation",0),
                   lambda:m.seed("confirmation/0000","teacher_train"),
                   lambda:m.gen("confirmation/0000","teacher_train"),
                   lambda:m.data("confirmation",0,16,"fixture"),
                   lambda:m.donor_pairs("confirmation",0,128,"training_pairs"),
                   lambda:m.Physical("confirmation",0),
                   lambda:m.Factors("confirmation",0),
                   lambda:m.fit("confirmation",0))
            for call in calls:
                with self.assertRaisesRegex(ValueError,"sealed"):
                    call()
            runtime.assert_not_called()
            generator.assert_not_called()

    def test_confirmation_evaluation_ignores_fabricated_release_and_preflight(self):
        with (patch.object(e,"validate_memory",side_effect=AssertionError("memory touched")) as memory,
              patch.object(e,"score_root",side_effect=AssertionError("model touched")) as score):
            with self.assertRaisesRegex(ValueError,"sealed"):
                e.evaluate("confirmation",{}, {"decision":"READY_FOR_CONFIRMATION"},{"human_approved":True})
            memory.assert_not_called()
            score.assert_not_called()

    def test_cli_has_no_confirmation_phase(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"should-not-exist.json"
            result=subprocess.run([sys.executable,str(ROOT/"run.py"),"--phase","confirmation","--out",str(path)],
                                  capture_output=True,text=True,check=False)
            self.assertEqual(result.returncode,2)
            self.assertIn("invalid choice",result.stderr)
            self.assertFalse(path.exists())


class SingleDevelopmentExecutionTests(unittest.TestCase):
    def test_only_the_registered_first_attempt_can_generate_development(self):
        allowed={"workflow":"BPD-AMM-09R1 prospective development", "event":"pull_request",
                 "run_number":2,"run_attempt":1}
        expected={"GITHUB_WORKFLOW":allowed["workflow"],"GITHUB_EVENT_NAME":"pull_request",
                  "GITHUB_RUN_NUMBER":"2","GITHUB_RUN_ATTEMPT":"1"}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"results").mkdir()
            (root/"results"/"TRAINING_RECEIPTS.json").write_text(json.dumps({"development_execution":allowed}))
            with patch.object(m,"ROOT",root), patch.dict(m.os.environ,expected):
                m.require_development_execution()
                for key,value in (("GITHUB_RUN_ATTEMPT","2"),("GITHUB_RUN_NUMBER","3"),
                                  ("GITHUB_EVENT_NAME","workflow_dispatch")):
                    with self.subTest(key=key), patch.dict(m.os.environ,{key:value}):
                        with self.assertRaisesRegex(ValueError,"sealed"):
                            m.require_development_execution()

    def test_missing_execution_record_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m,"ROOT",Path(directory)):
            with self.assertRaisesRegex(ValueError,"single development execution"):
                m.require_development_execution()


if __name__=="__main__":
    unittest.main()
