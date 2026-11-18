"""PD17 engineering-only source and exact-math tests on excluded seed 4001.

None of the 32 registered roots may be sampled, optimized or scored here.
"""
from __future__ import annotations

import hashlib

import torch

from mechinterp.bpd.theory.pd12.experiment import basis_alignment
from mechinterp.bpd.theory.pd17.angular import angular_basis
from mechinterp.bpd.theory.pd17.exact_source_math import (
    SINGLETON, exact_hyperplane_zero_mass,
    original_norm_tilt_profile, pair_dominance_certificate,
    source_support_distribution
)
from mechinterp.bpd.theory.pd17.source import (
    CONFIG, FAMILIES, training_source, heldout_source
)

SMOKE=CONFIG["excluded_engineering_seed"]


def _corr(x,y):
    x=x.double().reshape(-1)
    y=y.double().reshape(-1)
    x=x-x.mean()
    y=y-y.mean()
    return float((x@y)/(x.norm()*y.norm()))


def test_frozen_roster_and_source_controls():
    assert CONFIG["study_id"]=="BPD-PD-17"
    seeds=CONFIG["seeds"]
    assert len(seeds)==len(set(seeds))==32
    assert SMOKE not in seeds and max(seeds)<SMOKE and min(seeds)>3539
    assert set(CONFIG["source_laws"])=={"original","radial_camouflage"}
    assert [a["key"] for a in CONFIG["arms"]]==[
        "uniform_givens","norm_givens","angular_atoms","planted_Q_oracle"
    ]
    assert len(FAMILIES)==8
    assert CONFIG["givens"]["candidate_angle_scores"]==39960
    assert CONFIG["scoring"]["bootstrap_repeats"]==10000


def test_exact_support_mass_and_strict_pair_gap():
    p=source_support_distribution()
    assert sum(p.values())==1
    assert SINGLETON.numerator==1 and SINGLETON.denominator==64
    assert len(p)>100
    z0=exact_hyperplane_zero_mass(0)
    assert z0==1
    audit=pair_dominance_certificate()
    assert audit["pairs"]==120
    assert audit["minimum_normalized_basis_mixed_column_gap"]=="1/1024"
    prof=original_norm_tilt_profile(1.)
    assert abs(prof["phi"]-0.37530734)<1e-7
    assert abs(prof["effective_cardinality_probabilities"][0]-0.6373)<2e-4
    assert abs(prof["expected_ESS_fraction"]-0.48430)<2e-4


def test_paired_train_support_masks_and_radial_independence_excluded():
    orig=training_source(SMOKE,"original")
    cam=training_source(SMOKE,"radial_camouflage")
    assert orig["x"].shape==cam["x"].shape==(2048,4,1,16)
    assert torch.equal(orig["q"],cam["q"])
    assert torch.equal(orig["mask"],cam["mask"])
    assert orig["support_sha256"]==cam["support_sha256"]
    assert orig["pool_sha256"]!=cam["pool_sha256"]
    assert torch.allclose(cam["x"].squeeze(2)@cam["q"],
                          cam["latent"].squeeze(2),atol=3e-6,rtol=0)
    radial=cam["latent"].square().sum(-1).sqrt()
    assert float(radial.min())>=.99999 and float(radial.max())<=1.10001
    assert float(cam["latent"][cam["mask"].bool()].abs().min()) > .375
    k=cam["mask"].reshape(-1,16).sum(-1)
    orig_r2=orig["x"].reshape(-1,16).double().square().sum(-1)
    cam_r2=cam["x"].reshape(-1,16).double().square().sum(-1)
    assert _corr(orig_r2,k)>.70
    assert abs(_corr(cam_r2,k))<.05
    assert hashlib.sha256(orig["q"].numpy().tobytes()).hexdigest()==orig["q_sha256"]
    assert cam["pool_sha256"]==training_source(SMOKE,"radial_camouflage")["pool_sha256"]


def test_paired_heldout_support_and_oracle_lower_active_bound():
    for fam in FAMILIES:
        orig=heldout_source(SMOKE,"original",fam,boards=16)
        cam=heldout_source(SMOKE,"radial_camouflage",fam,boards=16)
        assert torch.equal(orig["mask"],cam["mask"])
        assert orig["support_sha256"]==cam["support_sha256"]
        assert cam["x"].shape==(16,4,1,16)
        assert float(cam["latent"][cam["mask"].bool()].abs().min())>.375
        assert torch.allclose(cam["x"].squeeze(2)@cam["q"],
                              cam["latent"].squeeze(2),atol=3e-6,rtol=0)


def test_observed_only_angular_atoms_recover_camouflaged_axes():
    cam=training_source(SMOKE,"radial_camouflage",boards=512)
    v,t=angular_basis(cam["x"])
    assert v is not None and not t["abstained"]
    assert t["candidate_count"]==1024 and t["observed_vector_count"]==2048
    assert t["angular_scalar_multiplications"]==1024*2048*16
    assert t["nonzero_moves"]==16
    assert len(set(t["selected_training_indices"]))==16
    assert all(n>=12 for n in t["selected_neighbor_counts"])
    assert t["orthogonality_max_error"]<1e-8
    align=basis_alignment(v,cam["q"])
    assert align["mean_cosine"]>.999999
