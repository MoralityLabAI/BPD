"""Independent, model-free NC-11 first-run artifact audit.

Usage:
  python -m mechinterp.bpd.experiments.nc11.audit_original_receipts \
    /path/to/nc11-original-eight-pair-receipts.zip

This verification was written AFTER the first hosted development run. It
recomputes the frozen originally registered bootstrap from immutable raw
root JSONs and cannot retune or re-evaluate a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOTS=tuple(range(1100,1108))
EXPECTED_ZIP_SHA256="3c2ae80c545d06bcae4946e599ebe98c163dacd245fc7a8839a5c4f17fdcba35"
EXPECTED_SOURCE_SHA256="685405114b49f3fb615fc7b5d31b5f0839c0245687ef4d1445ac66c2c102177a"
EXPECTED_PROTOCOL_SHA256="733d347a516416a2a687c33469236351ae230958157261cdcc88620795d8f541"
PROTOCOL_COMMIT="60b65642c03cb3064912aa3929582479aa21dd20"
COMPARISONS=(
    "jacobian_vs_activation_shared",
    "jacobian_vs_permuted_shared",
    "jacobian_minus_target_cal_shared",
    "matched_sham_advantage_shared",
)
BOOTS=4096
FAMILY_ALPHA=.05/4
EXPECTED_MEANS=(
    .20236086007207632,
    -.03495992440730333,
    .12918895203620195,
    .04818188399076462,
)


def audit(original_zip:Path)->dict:
    data=original_zip.read_bytes()
    sha=hashlib.sha256(data).hexdigest()
    if sha!=EXPECTED_ZIP_SHA256:
        raise ValueError(f"Original artifact SHA mismatch: {sha}")
    with zipfile.ZipFile(original_zip) as z:
        expected={f"nc11_root_{n}.json" for n in ROOTS}
        expected.add("NC11_ORIGINAL_SUMMARY.json")
        if set(z.namelist())!=expected or len(z.namelist())!=len(expected):
            raise ValueError("Original complete nine-file roster mismatch")
        summary=json.loads(z.read("NC11_ORIGINAL_SUMMARY.json"))
        roots=[json.loads(z.read(f"nc11_root_{root}.json")) for root in ROOTS]
    assert summary["development_verdict"]=="ENGINEERING_NULL_OR_INELIGIBLE"
    assert summary["scientific_status"]=="NOT_CONFIRMATORY"
    assert summary["roots"]==list(ROOTS)
    assert summary["technical_admission_roots"]==list(ROOTS)
    assert summary["protocol_commit"]==PROTOCOL_COMMIT
    assert summary["experiment_source_sha256"]==EXPECTED_SOURCE_SHA256
    assert summary["protocol_sha256"]==EXPECTED_PROTOCOL_SHA256
    for root,rec in zip(ROOTS,roots):
        assert rec["root"]==root and rec["phase"]=="ENGINEERING"
        assert rec["study"]=="BPD-NC-11" and rec["eligible"] is True
        assert rec["excluded_reasons"]==[]
        assert rec["source_hash_sha256"]==EXPECTED_SOURCE_SHA256
        assert rec["protocol_sha256"]==EXPECTED_PROTOCOL_SHA256
        assert rec["protocol_commit"]==PROTOCOL_COMMIT
        assert rec["budgets"]=={
            "steps":384,"train_A":2048,"train_B":2048,
            "discovery":1024,"calibration_pool":512,
            "target_gradient_calibration":128,
            "dose":1024,"evaluation":1024,"sham_proposals":2048,
        }
        assert rec["calibration"]["calibration_used"]==128
        assert rec["calibration"]["positive_context_examples_in_pool"]>=128
        assert rec["closure"]["A"]["passed"] and rec["closure"]["B"]["passed"]
        assert rec["negative_control"]["zero_head_subspace_rejected"]
        assert rec["negative_control"]["zero_head_jacobian_max_abs"]==0
        assert rec["shams"]["admitted"] and rec["shams"]["selected"]==8
        assert rec["shams"]["qualified"]>=8
        assert rec["gauge"]["full_weight_parity"]["passed"]
        assert rec["gauge"]["physical_operator_error_relative"]<1e-4
        assert rec["gauge"]["heldout_free_source_kept_max_logit_error"]<5e-5
        for method in ("activation","permuted_jacobian","target_calibration"):
            assert rec["calibration"]["dose_matched_gains"][method]["eligible"]
    recomputed={}
    for index,name in enumerate(COMPARISONS):
        root_effects=np.asarray([r["comparisons"][name] for r in roots],dtype=float)
        rng=np.random.default_rng(1_110_011+101*index)
        sample_idx=rng.integers(0,len(ROOTS),size=(BOOTS,len(ROOTS)))
        means=root_effects[sample_idx].mean(axis=1)
        ci=np.quantile(means,[FAMILY_ALPHA/2,1-FAMILY_ALPHA/2])
        original=summary["contrasts"][name]
        if not np.isclose(root_effects.mean(),EXPECTED_MEANS[index],atol=1e-12):
            raise ValueError(f"{name}: original mean mismatch")
        if not np.isclose(root_effects.mean(),original["mean_nats"],atol=1e-12):
            raise ValueError(f"{name}: original aggregate mean mismatch")
        if not np.allclose(ci,original["familywise_9875pct_ci"],atol=1e-12):
            raise ValueError(f"{name}: original familywise interval mismatch")
        recomputed[name]={"mean_nats":float(root_effects.mean()),
                          "familywise_9875pct_ci":[float(x) for x in ci]}
    return {
        "verdict":"INDEPENDENT_REPLAY_PASS",
        "scientific_status":"ORIGINAL_ENGINEERING_NULL_OR_INELIGIBLE",
        "independent_root_pairs":len(ROOTS),
        "source_sha256":EXPECTED_SOURCE_SHA256,
        "original_zip_sha256":EXPECTED_ZIP_SHA256,
        "recomputed":recomputed,
        "confirmation":"SEALED"
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_zip",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.original_zip),indent=2,sort_keys=True,
                     allow_nan=False))


if __name__=="__main__":
    main()
