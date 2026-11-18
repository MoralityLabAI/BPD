"""NC-12 independent original-result audit, created after first run.

Reads the immutable original GitHub Actions ZIP. Recalculates every frozen
16-family bootstrap without importing the training/experiment package.

Usage:
  python -m mechinterp.bpd.experiments.nc12.audit_original_receipts /path/to/archive.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOTS=tuple(range(1200,1208))
QUERIES=(4,8,16,32)
CONTRASTS=(
    "target_only_minus_transfer",
    "activation_minus_transfer",
    "median_sham_minus_transfer",
    "transfer_minus_clean_B",
)
ORIGINAL_ZIP_SHA256="3b97a25641caa553b12b67da992bc72e2a7c4c82ecc8712fd79ef498a76eabe5"
ORIGINAL_SOURCE_SHA256="30aa76ef6b2150d96a37fa696d25d714ed11bfcb0359d6e6205fd97842d2f1b5"
FROZEN_PROTOCOL_SHA256="2b0bcb1e7f99536c7c1d5717c8802800671a6ea79d42c7efcd0ab854de62a152"
PROTOCOL_FIRST_COMMIT="6a4b5ebc16fae730f17b0e5dac5564eaf419c071"
BOOTS=4096
FAMILY=16
ALPHA=.05/FAMILY

EXPECTED_BONLY_MINUS_TRANSFER=(
    -.0620215330272913,
    -.07272148318588734,
    -.0794768501073122,
    -.09501091297715902,
)


def audit(path:Path)->dict:
    if not path.is_file():
        raise ValueError("missing original results archive")
    raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=ORIGINAL_ZIP_SHA256:
        raise ValueError(f"original artifact SHA-256 mismatch: {digest}")
    with zipfile.ZipFile(path) as z:
        required={"NC12_ORIGINAL_SUMMARY.json"}
        required.update(f"nc12_root_{r}.json" for r in ROOTS)
        if set(z.namelist())!=required:
            raise ValueError("original archived file roster not exactly nine")
        summary=json.loads(z.read("NC12_ORIGINAL_SUMMARY.json"))
        roots=[json.loads(z.read(f"nc12_root_{n}.json")) for n in ROOTS]
    if (summary["source_sha256"]!=ORIGINAL_SOURCE_SHA256
        or summary["protocol_sha256"]!=FROZEN_PROTOCOL_SHA256
        or summary["protocol_commit"]!=PROTOCOL_FIRST_COMMIT
        or summary["original_development_verdict"]!="ENGINEERING_NULL_OR_INELIGIBLE"
        or summary["lowest_qualifying_low_budget"] is not None):
        raise ValueError("frozen original study metadata differs")
    if summary["independent_training_pairs"]!=8 or summary["family_directional_comparisons"]!=16:
        raise ValueError("incorrect independent units or multiple-testing family")
    for r,receipt in zip(ROOTS,roots):
        if (receipt["root"]!=r or receipt["study"]!="BPD-NC-12"
            or receipt["phase"]!="ENGINEERING"
            or receipt["protocol_commit"]!=PROTOCOL_FIRST_COMMIT
            or receipt["source_sha256"]!=ORIGINAL_SOURCE_SHA256
            or receipt["protocol_sha256"]!=FROZEN_PROTOCOL_SHA256
            or receipt["excluded_global"]):
            raise ValueError(f"root {r} metadata or source admission mismatch")
        if receipt["budgets"]!={
            "steps":384,"train_A":2048,"train_B":2048,"discovery":1024,
            "calibration_pool":512,"activation_calibration":128,
            "target_gradient_union":32,"dose":1024,"evaluation":1024,
            "sham_proposals":2048,
        }:
            raise ValueError("unexpected training, source or gradient budget")
        if receipt["calibration"]["nested_gradient_union"]!=32:
            raise ValueError("target-gradient query union differs from registration")
        if not (receipt["closure"]["A"]["passed"] and receipt["closure"]["B"]["passed"]
                and receipt["negative_control"]["zero_head_rejected"]
                and receipt["negative_control"]["gradient_max_abs"]==0
                and receipt["gauge"]["rotated_fullmask"]["passed"]
                and receipt["gauge"]["max_physical_operator_relative_error"]<=1e-4
                and receipt["gauge"]["max_closed_loop_logit_delta"]<=5e-5):
            raise ValueError(f"root {r} technical control failed")
        for q in QUERIES:
            v=receipt["q_arms"][str(q)]
            if (v["excluded_reasons"] or not v["technically_admitted"]
                or v["target_gradient_queries"]!=q
                or v["activation_rows_used"]!=128):
                raise ValueError(f"root {r}, query {q}: original admission changed")
            if not (v["sham_matching"]["admitted"]
                and v["sham_matching"]["selected"]==8
                and v["sham_matching"]["qualified"]>=8):
                raise ValueError(f"root {r}, query {q}: local physical shams failed")
            if not all(v["dose_gains"][name]["eligible"]
                       for name in ("transfer","target_only","activation","permuted")):
                raise ValueError(f"root {r}, query {q}: dose bounds changed")
    replay={}
    for q_ix,q in enumerate(QUERIES):
        orig=summary["arms"][str(q)]
        if orig["admitted_roots"]!=list(ROOTS) or orig["eligible_for_scientific_development_claim"]:
            raise ValueError(f"query {q}: original joint gate changed")
        replay[str(q)]={}
        for i,name in enumerate(CONTRASTS):
            effects=np.asarray([r["q_arms"][str(q)]["comparisons"][name]
                                for r in roots],dtype=float)
            rng=np.random.default_rng(1_212_012+q*31+i*101)
            idx=rng.integers(0,8,size=(BOOTS,8))
            boots=effects[idx].mean(1)
            interval=np.quantile(boots,[ALPHA/2,1.-ALPHA/2])
            observed=orig["contrasts"][name]
            if (not np.isclose(effects.mean(),observed["mean_nats"],atol=1e-12)
                or not np.allclose(interval,observed["familywise_ci"],atol=1e-12)):
                raise ValueError(f"q{q} {name} failed independent bootstrap replay")
            replay[str(q)][name]={
                "root_mean":float(effects.mean()),
                "familywise_996875pct_ci":[float(x) for x in interval],
            }
        if not np.isclose(
            replay[str(q)]["target_only_minus_transfer"]["root_mean"],
            EXPECTED_BONLY_MINUS_TRANSFER[q_ix],atol=1e-12
        ):
            raise ValueError(f"q{q}: equal-target-information effect changed")
    return {
        "audit":"INDEPENDENT_SOURCE_LOCKED_REPLAY_PASS",
        "original_scientific_verdict":"ENGINEERING_NULL_OR_INELIGIBLE",
        "model_pairs":8,"comparisons_replayed":16,
        "original_zip_sha256":digest,
        "experiment_source_sha256":ORIGINAL_SOURCE_SHA256,
        "protocol_sha256":FROZEN_PROTOCOL_SHA256,
        "recomputed":replay,
        "confirmation":"SEALED",
    }


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_zip",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.original_zip),sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
