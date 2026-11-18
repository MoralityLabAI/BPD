"""NC09-Fixed12 independent original sixteen-root publication/ZIP evidence audit.

This file was added after the first registered positive engineering development
run. It parses preserved original CSV and protected source hashes, and
independently recomputes every root bootstrap. It never trains a model or
samples from any unexposed calibration/confirmation population.

To verify the raw first GitHub Actions artifact additionally:
  python -m mechinterp.bpd.experiments.nc09_fixed12.test_published_results
    --archive /path/to/bpd-nc09-fixed12-first16-44268539237.zip
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
ORIGINAL_RUN_ID=44268539237
ORIGINAL_ARTIFACT_ID=14410589675
ORIGINAL_ZIP_SHA="585bec0fea7fd43fa21c2e7d3e32a0794ef760692a644ec52c8a6faf9e10fc89"
ORIGINAL_SUMMARY_SHA="4ffe6d47dc85200f283ab1b982da3c205cc3f0194fb400da3a4aaa2b7a812027"
PROTOCOL_SHA="dc9423f1fd9e308221c41ed2af83e7398a59dd0eb0beefa8394091e373156e02"
EXPERIMENT_SHA="8d1b88eea8518a83669f03f9126136ca3d56ac5bac5fe041e2e03fb12922d380"
ROOT_HASHES={
    900:"b8d6353cd9befd7875b0f9d0252392859a8399ae6c5c1714fb6d24e33b1232af",
    901:"a05dd6f2accd750770949191987e7e2fdc3bc9280b272523fb26410e7f114266",
    902:"4350b721f793acccc14452356d169624501ebc8b2e0042df969726df5b2f94a9",
    903:"df98f97c5a3b32340a01abcd5f32cc9421b6c60b98f2cac727d912568fb44801",
    904:"418380896c15cfd6a49407e91b6ea5cedc166ead8374a25424dfb7927851a7af",
    905:"bf5dfd46dce96e8e87c80d357739bd4e01416fe121daac46c4bfd9b94b9de7a6",
    906:"1c5cf0ce95dc4e2e9b3bca34a4c06d95803682d147c6238d29f93484ad59a7a7",
    907:"6e363dcf2a8cf5b8655d7f71033c0f0c880794b7e59f0611ea74a3f77582b6a1",
    908:"0f777b81eb474076fcef7c9bb7b45c207e65d44f9aa70b07710985eea17b4388",
    909:"e38525376cd5bdda3ebb02bd5c745fb7d18483d2398976292946664a4161ca09",
    910:"b9d8c03d368c776f59cc35351f41b4fd63df1fe5111da602c373e1e45d8d0633",
    911:"a23f07b181af72da1c9ed56daef40d0e0ba0da85e61e2b5c5fc2991980c0f551",
    912:"2bc1f29c14e1ce8b686c15d43e4b8fb6c1e4a5617bf14ec86c213b2527d23368",
    913:"f903eae491dcb77898459a4a1ddaa8794416718d457e2ae9df43dcf2bab99dbe",
    914:"f4a79976c0e6384b177dc5e9c3852902e8d7c1afe1cea1362ee5c48790a52649",
    915:"4746f9e70001988ea444ece4d09db8ff3d773c8fb9db722c8d05b05f4e26092d",
}
FIELDS=(
    "root","r0","r1","r2","r3","eligible","shams",
    "qualified_sham_proposals","dose_gain","clean_nll","dp12_nll",
    "flat3_nll","flat3_dose_nll","flat4_nll","reverse_nll",
    "flat3_minus_dp12","doseflat3_minus_dp12","dp12_minus_clean",
    "dp12_minus_flat4","matched_sham_minus_dp12","reverse_minus_dp12",
)
METRICS=(
    "flat3_minus_dp12","doseflat3_minus_dp12","dp12_minus_clean",
    "dp12_minus_flat4","sham_minus_dp12",
)
ORIGINAL_STATS={
    "flat3_minus_dp12":(.027179472148418427,.011696710134856403,.04645781440194696),
    "doseflat3_minus_dp12":(.028017306234687567,.011852192459627986,.048272684682160616),
    "dp12_minus_clean":(.014932033605873585,.007738164742477238,.023801426927093416),
    "dp12_minus_flat4":(.007012439426034689,.0026691347011364996,.01242969580926001),
    "matched_sham_minus_dp12":(.05340614961460233,.038984497950877994,.0676910299807787),
}


def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def original_rows() -> list[dict]:
    with (ROOT/"PER_ROOT_20261118.csv").open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        assert tuple(r.fieldnames)==FIELDS
        rows=list(r)
    assert len(rows)==16 and {int(x["root"]) for x in rows}==set(ROOT_HASHES)
    ranks=Counter(tuple(int(x[f"r{i}"]) for i in range(4)) for x in rows)
    assert ranks=={(4,3,3,2):9,(4,4,2,2):4,(5,3,2,2):3}
    for x in rows:
        assert int(x["eligible"])==1 and int(x["shams"])==8
        assert int(x["qualified_sham_proposals"])>=8
        assert sum(int(x[f"r{i}"]) for i in range(4))==12
        assert 0.5<=float(x["dose_gain"])<=1.5
        assert float(x["flat3_minus_dp12"])>0
        assert float(x["doseflat3_minus_dp12"])>0
        for value,k,a,b in (
            ("flat3_minus_dp12",None,"flat3_nll","dp12_nll"),
            ("doseflat3_minus_dp12",None,"flat3_dose_nll","dp12_nll"),
            ("dp12_minus_clean",None,"dp12_nll","clean_nll"),
            ("dp12_minus_flat4",None,"dp12_nll","flat4_nll"),
            ("reverse_minus_dp12",None,"reverse_nll","dp12_nll"),
        ):
            assert abs(float(x[value])-(float(x[a])-float(x[b])))<1e-10
    assert sum(float(x["reverse_minus_dp12"])>0 for x in rows)==16
    assert min(int(x["qualified_sham_proposals"]) for x in rows)==128
    return rows


def independent_bootstrap(rows:list[dict]):
    arr=np.asarray([[float(row[k if k!="matched_sham_minus_dp12" else "sham_minus_dp12"])
                     for k in METRICS] for row in rows],dtype=float)
    assert arr.shape==(16,5)
    rng=np.random.default_rng(909009)
    sampled=rng.integers(0,16,size=(4096,16))
    means=arr[sampled].mean(axis=1)
    results={}
    for j,k in enumerate(METRICS):
        observed=(float(arr[:,j].mean()),
                  float(np.percentile(means[:,j],2.5)),
                  float(np.percentile(means[:,j],97.5)))
        assert max(abs(a-b) for a,b in zip(observed,ORIGINAL_STATS[k]))<1e-12,(k,observed)
        results[k]=observed
    assert results["flat3_minus_dp12"][0]>=.008 and results["flat3_minus_dp12"][1]>0
    assert results["doseflat3_minus_dp12"][0]>=.006 and results["doseflat3_minus_dp12"][1]>0
    assert results["dp12_minus_clean"][0]<=.05 and results["dp12_minus_clean"][2]<=.05
    assert results["dp12_minus_flat4"][0]<=.02 and results["dp12_minus_flat4"][2]<=.02
    assert results["matched_sham_minus_dp12"][0]>=.02 and results["matched_sham_minus_dp12"][1]>0
    return results


def test_immutable_first_run_source_and_paired_root_publication():
    assert digest((ROOT/"PROTOCOL.md").read_bytes())==PROTOCOL_SHA
    assert digest((ROOT/"experiment.py").read_bytes())==EXPERIMENT_SHA
    independent_bootstrap(original_rows())


def audit_full_raw_first_run(path:Path):
    original=path.read_bytes()
    assert len(original)==60611 and digest(original)==ORIGINAL_ZIP_SHA
    with zipfile.ZipFile(path) as z:
        expected={"NC09_FIXED12_ORIGINAL_SUMMARY.json"} | {
            f"nc09_fixed12_root_{root}.json" for root in ROOT_HASHES
        }
        assert set(z.namelist())==expected
        summary_data=z.read("NC09_FIXED12_ORIGINAL_SUMMARY.json")
        assert len(summary_data)==16775 and digest(summary_data)==ORIGINAL_SUMMARY_SHA
        summary=json.loads(summary_data)
        assert summary["study"]=="BPD-NC-09-fixed12"
        assert summary["phase"]=="DEVELOPMENT"
        assert summary["evidence_class"]=="ENGINEERING_ONLY"
        assert summary["registered_verdict"]=="DEVELOPMENT_POSITIVE"
        assert summary["independent_model_roots"]==summary["admitted_roots"]==16
        assert summary["model_budget"]==12 and summary["physical_reference_budget"]==16
        assert summary["nonflat_allocations"]==16
        assert summary["gates"]=={
            "all_sixteen_native_roots_and_eight_shams":True,
            "fixed12_beats_equal12_flat3":True,
            "fixed12_beats_dose_matched_flat3":True,
            "absolute_task_fidelity":True,
            "near_rank4_task_fidelity":True,
            "source_dose_matched_sham_specificity":True,
        }
        assert summary["bootstrap"]["draws"]==4096
        assert summary["bootstrap"]["seed"]==909009
        assert summary["confirmed_scientific_result"] is False
        assert summary["calibration_roots_evaluated"]==summary["confirmation_roots_evaluated"]==0
        independently=independent_bootstrap(original_rows())
        for key,stats in independently.items():
            actual=summary["bootstrap"]["metrics"][key]
            assert max(abs(actual[field]-number)
                       for field,number in zip(("mean","lower95","upper95"),stats))<1e-12
        by={x["root"]:x for x in summary["per_root"]}
        for csvrow in original_rows():
            root=int(csvrow["root"])
            data=z.read(f"nc09_fixed12_root_{root}.json")
            assert digest(data)==ROOT_HASHES[root]
            rec=json.loads(data)
            assert rec["root"]==root and rec["study"]=="BPD-NC-09-fixed12"
            assert rec["phase"]=="DEVELOPMENT" and rec["eligible"]
            assert rec["training"]["steps"]==384
            assert rec["splits"]=={
                "train":2048,"discovery":1024,"selector_diagnostics":512,"heldout":1024,
            }
            assert rec["protocol_sha256"]==PROTOCOL_SHA
            assert rec["source_sha256"]==EXPERIMENT_SHA
            assert rec["clean_closure"]["passed"]
            assert rec["sham_admission"]["selected"]==8
            assert rec["sham_admission"]["proposals"]==4096
            assert rec["sham_admission"]["qualified"]==int(csvrow["qualified_sham_proposals"])
            assert abs(rec["dose_matched_flat3_gain"]-
                       float(csvrow["dose_gain"]))<1e-12
            assert abs(rec["dose_matched_flat3_gain"]*
                       rec["flat3_source_write_rms"]-
                       rec["dp12_source_write_rms"])<1e-8
            assert rec["fixed_budget12_allocation"]["ranks"]==[
                int(csvrow[f"r{i}"]) for i in range(4)
            ]
            # Independent exhaustive finite knapsack replay from the *original*
            # logged source-only per-pass proxy, not the experiment optimizer.
            candidates={int(t):{int(r):p for r,p in rows.items()}
                        for t,rows in rec["passwise_source_candidates"].items()}
            best=None
            for ranktuple in itertools.product((2,3,4,5,6),repeat=4):
                if sum(ranktuple)!=12 or not all(
                    candidates[t][ranktuple[t]]["eligible"] for t in range(4)
                ):
                    continue
                value=sum(candidates[t][ranktuple[t]]["squared_local_omitted_model_score"]
                          for t in range(4))
                if best is None or value<best[0]-1e-12 or (
                    abs(value-best[0])<=1e-12 and ranktuple<best[1]):
                    best=(value,ranktuple)
            assert best is not None
            assert list(best[1])==rec["fixed_budget12_allocation"]["ranks"]
            for arm in ("dp12","flat3","flat3_dose","flat4"):
                assert rec["heldout"][arm] is not None
                assert rec["gauge"][arm]["physical_map_relative_error"]<=1e-4
                assert rec["gauge"][arm]["max_complete_rollout_logit_error"]<=5e-5
            assert len(rec["sham_task_nlls"])==8
            assert abs(rec["heldout"]["dp12"]["nll_nats"]-
                       float(csvrow["dp12_nll"]))<1e-10
            assert abs(by[root]["flat3_minus_dp12"]-
                       float(csvrow["flat3_minus_dp12"]))<1e-10
            assert abs(by[root]["doseflat3_minus_dp12"]-
                       float(csvrow["doseflat3_minus_dp12"]))<1e-10
        print("PASS: immutable original 16-model NC09 source, independent DP and all six developmental gates")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path)
    args=p.parse_args()
    test_immutable_first_run_source_and_paired_root_publication()
    if args.archive is not None:
        audit_full_raw_first_run(args.archive)
    print("PASS: original registered NC09 sixteen-root source and developmental PASS")


if __name__=="__main__":
    main()
