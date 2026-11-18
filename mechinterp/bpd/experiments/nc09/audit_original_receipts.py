"""Independently replay published NC-09 engineering statistics from original CI ZIP.

This script is a *post-run audit*. It never trains models, edits original
receipts, inspects confirmation data or performs rank selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

EXPECTED_ZIP_SHA256="13795a201c54a8cd251685bc35a2b4908d8e08985ce5d33a4dccb71bc3be2a03"
EXPECTED_SUMMARY_SHA256="cd4381f621e114ebfb1d504e19aec547a616006beeb7467f9be4e0bc7d7d360e"
EXPECTED_SOURCE_SHA256="05b57528a412a039d79c42f1728283226bedc62e14c21a93007ec4334fb03108"
EXPECTED_CORE_SHA256="2e43ae9965d168050843a7dc43b88c12a40359be93aeb8040681e2c23923d644"
EXPECTED_PROTOCOL_SHA256="f642abb3f65b361364096170a258d13093a8b4a14f3656e44d117c8f2737125f"


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def audit(zip_path: Path) -> dict:
    assert sha(zip_path.read_bytes())==EXPECTED_ZIP_SHA256,"original ZIP byte hash mismatch"
    with zipfile.ZipFile(zip_path) as archive:
        summary_bytes=archive.read("NC09_ENGINEERING_SUMMARY.json")
        assert sha(summary_bytes)==EXPECTED_SUMMARY_SHA256
        summary=json.loads(summary_bytes)
        assert summary["development_resolution"]=="ENGINEERING_NULL_OR_INELIGIBLE"
        assert summary["eligible_seeds"]==list(range(900,908))
        assert summary["source_sha256"]==EXPECTED_SOURCE_SHA256
        assert summary["core_sha256"]==EXPECTED_CORE_SHA256
        assert summary["protocol_sha256"]==EXPECTED_PROTOCOL_SHA256
        independently_recomputed={key:[] for key in (
            "shared_transfer_excess",
            "specificity_private_minus_shared",
            "shared_matched_sham_advantage",
        )}
        for seed in range(900,908):
            original=json.loads(archive.read(f"nc09_root_{seed}.json"))
            assert original["eligible"] and original["seed"]==seed
            for key,expected in (("source_sha256",EXPECTED_SOURCE_SHA256),
                                 ("core_sha256",EXPECTED_CORE_SHA256),
                                 ("protocol_sha256",EXPECTED_PROTOCOL_SHA256)):
                assert original[key]==expected
            # Recover original context assignments with the source-registered
            # public generator; do not use the experiment's scorer or summary.
            generator=torch.Generator().manual_seed(seed*100_003 + 3*1_009)
            base=torch.randn(1024,5,generator=generator)
            mask_shared=(base[:,4]>0).numpy()
            mask_private=~mask_shared
            def score(label: str, context: str) -> float:
                rec=original["heldout"][label]
                losses=np.array(rec["per_case_nll"],dtype=np.float64)
                assert len(losses)==1024
                mask=mask_shared if context=="shared" else mask_private
                value=float(losses[mask].mean())
                # Original case-aggregates used float32 PyTorch reduction.
                assert abs(value-rec["by_context"][context]["nll_nats"])<2e-6, (seed,label,context)
                return value
            for arm in ("clean_B","B_to_B","A_to_B"):
                for context in ("shared","private"):
                    score(arm,context)
            a_plus,b_plus=score("A_to_B","shared"),score("B_to_B","shared")
            a_minus,b_minus=score("A_to_B","private"),score("B_to_B","private")
            gshared=a_plus-b_plus
            specificity=(a_minus-b_minus)-gshared
            shams=original["heldout"]["A_shams_to_B"]
            assert len(shams)==8
            null_scores=[]
            for sham in shams:
                arr=np.array(sham["per_case_nll"],dtype=np.float64)
                null_scores.append(float(arr[mask_shared].mean()))
            advantage=float(np.median(null_scores))-a_plus
            independently_recomputed["shared_transfer_excess"].append(gshared)
            independently_recomputed["specificity_private_minus_shared"].append(specificity)
            independently_recomputed["shared_matched_sham_advantage"].append(advantage)
        for key,seed in (
            ("shared_transfer_excess",909009),
            ("specificity_private_minus_shared",909010),
            ("shared_matched_sham_advantage",909011)
        ):
            data=np.asarray(independently_recomputed[key],dtype=np.float64)
            orig=summary["causal_endpoints"][key]
            assert abs(data.mean()-orig["mean"])<2e-6
            resamples=data[np.random.default_rng(seed).integers(0,8,size=(4096,8))].mean(axis=1)
            interval=np.quantile(resamples,[0.05/3/2,1-0.05/3/2])
            assert np.allclose(interval,orig["familywise_interval"],atol=2e-6)
        return {
            "audit":"PASS",
            "independent_roots":8,
            "registered_development_verdict":summary["development_resolution"],
            "per_root_external_case_nll":"recomputed from all 8 original raw files",
            "familywise_intervals":"independently recomputed and matched",
            "original_zip_sha256":EXPECTED_ZIP_SHA256,
            "confirmation":"SEALED",
        }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-zip",required=True,type=Path)
    p.add_argument("--out",type=Path)
    opts=p.parse_args()
    result=audit(opts.original_zip)
    report=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if opts.out:
        with opts.out.open("x",encoding="utf-8") as f:
            f.write(report)
    print(report)


if __name__=="__main__":
    main()
