"""BPD-PD-13 engineering tests using excluded model seed 4001 only.

These tests do not fit or score any of the 32 registered source rotations.
"""
from __future__ import annotations

import inspect
import math
import torch

from mechinterp.bpd.theory.pd12.experiment import support_banks
from mechinterp.bpd.theory.pd13.experiment import (
    ARMS, CELLS, CONFIG, DIM, FAMILIES, angle_grid,
    pool_for_seed, priority_order, initial_random_basis,
    search_basis, run_arm, summarize,
)

SMOKE=CONFIG["smoke_seed"]


def patterns(mask):
    return [tuple(torch.nonzero(mask[b,t,0]).flatten().tolist())
            for b in range(mask.shape[0]) for t in range(4)]


def test_registered_seed_roster_and_frozen_scientific_gates():
    assert CONFIG["study_id"]=="BPD-PD-13"
    assert len(CELLS)==2
    assert {c["margin"] for c in CELLS}=={"strong","weak"}
    assert all(c["dimension"]==16 and c["support_distribution"]=="clustered" for c in CELLS)
    seeds=[s for cell in CELLS for s in cell["seeds"]]
    assert len(seeds)==len(set(seeds))==32
    assert SMOKE not in seeds and min(seeds)>2357
    assert ARMS==("priority2048","lex2048","priority256","lex256","planted_Q_oracle")
    assert CONFIG["scoring"]["primary_arm"]=="priority2048"
    assert CONFIG["scoring"]["primary_control"]=="lex2048"
    assert CONFIG["objective"]["sweeps"]==3
    assert CONFIG["objective"]["candidate_grid"]==37
    assert CONFIG["objective"]["pair_evaluations_per_arm"]==13320
    assert CONFIG["scoring"]["primary_min_successes"]==18


def test_unchanged_pd12_correlated_banks_and_novel_supports():
    b=support_banks(DIM)
    for k in (1,2,3,4):
        assert set(b["train"][k]).isdisjoint(b["heldout"][k])
        assert set(b["cluster"][k]).issubset(b["train"][k])
    for cell in CELLS:
        q,x,sha,mask,latent=pool_for_seed(SMOKE,cell,boards=128)
        assert x.shape==mask.shape==latent.shape==(128,4,1,DIM)
        assert float((x.abs()>1e-6).float().mean())>.995
        assert torch.allclose(x.squeeze(2)@q,latent.squeeze(2),atol=3e-6,rtol=0)
        assert all(p in b["train"][len(p)] for p in patterns(mask))
        lo,hi=(.75,1.25) if cell["margin"]=="strong" else (.25,1.25)
        assert bool((latent[mask.bool()].abs()>=lo).all())
        assert bool((latent[mask.bool()].abs()<=hi).all())
        assert bool((latent[~mask.bool()]==0).all())


def test_large_training_pool_contains_nested_small_volume_prefix():
    cell=CELLS[0]
    q,x,digest,masks,latent=pool_for_seed(SMOKE,cell)
    assert x.shape==(2048,4,1,16)
    assert x[:256].shape==(256,4,1,16)
    assert digest==pool_for_seed(SMOKE,cell)[2]
    assert torch.equal(x,pool_for_seed(SMOKE,cell)[1])
    assert torch.equal(masks,pool_for_seed(SMOKE,cell)[3])
    assert all(p in support_banks(DIM)["train"][len(p)] for p in patterns(masks[:256]))


def test_pair_priority_uses_only_observed_read_mass_and_changes_order():
    z=torch.zeros(128,16,dtype=torch.float64)
    z[:,10]=2.0
    z[:,11]=2.0
    ranking,masses=priority_order(z,.15)
    assert ranking[0]==(10,11)
    assert ranking[-1]!=(10,11)
    assert len(ranking)==len(set(ranking))==120
    assert len(masses)==16
    assert all(0<=m<=1 for m in masses)
    assert set(inspect.signature(search_basis).parameters)=={
        "observed","initial","margin","arm","sweeps"
    }
    assert "support" not in inspect.signature(search_basis).parameters
    assert "q" not in inspect.signature(search_basis).parameters


def test_equal_angle_cost_and_monotone_source_only_search():
    cell=CELLS[0]
    q,x,_,_,_=pool_for_seed(SMOKE,cell,boards=32)
    initial=initial_random_basis(SMOKE,16)
    for arm in ("priority2048","lex2048","priority256","lex256"):
        read,diag=search_basis(x,initial,cell["margin"],arm,sweeps=2)
        assert diag["candidate_angle_scores"]==2*120*37
        assert diag["candidate_scalar_projection_evaluations"]==2*120*37*128
        assert diag["training_observed_vectors"]==128
        assert diag["orthogonality_max_error"]<2e-6
        assert torch.allclose(read.T@read,torch.eye(16,dtype=torch.float64),atol=2e-6)
        assert len(diag["pair_orders_by_sweep"])==2
        assert len(diag["pair_orders_by_sweep"][0])==120
        for a,b in zip(diag["mass_by_sweep"],diag["mass_by_sweep"][1:]):
            assert b>=a-1e-9,(arm,a,b)
        if arm.startswith("priority"):
            assert diag["priority_proxy_marginal_scalar_evaluations"]==2*128*16
        else:
            assert diag["priority_proxy_marginal_scalar_evaluations"]==0
        replay,_=search_basis(x.clone(),initial.clone(),cell["margin"],arm,sweeps=2)
        assert torch.equal(read,replay)


def test_heldout_oracle_on_both_amplitude_regimes():
    for cell in CELLS:
        q,x,digest,mask,lat=pool_for_seed(SMOKE,cell,boards=24)
        r=run_arm(SMOKE,cell,"planted_Q_oracle",pool=x,q=q,pool_hash=digest,
                  heldout_boards=12)
        assert r["uses_true_Q_for_optimizer"]
        assert r["success"]
        assert r["heldout"]["alignment"]["mean_cosine"]>.999999
        assert all(v["binary_mask"] and v["support_fraction"]==1
                   and v["false_off"]==0 and v["false_on"]==0
                   for v in r["heldout"]["per_family"].values())


def test_five_arms_per_margin_with_common_pool_and_exact_bpd_masks():
    rows=[]
    for cell in CELLS:
        q,pool,pool_hash,_,_=pool_for_seed(SMOKE,cell)
        for arm in ARMS:
            row=run_arm(SMOKE,cell,arm,pool=pool,q=q,pool_hash=pool_hash,
                        heldout_boards=12,sweeps=1)
            rows.append(row)
            assert row["training_full_pool_sha256"]==pool_hash
            assert row["heldout"]["alignment"]["mean_cosine"]>=0
            assert len(row["heldout"]["per_family"])==8
            assert all(v["binary_mask"] for v in row["heldout"]["per_family"].values())
            if arm=="planted_Q_oracle":
                assert row["success"]
                assert row["training"]["candidate_angle_scores"]==0
            else:
                assert not row["uses_true_Q_for_optimizer"]
                assert row["training"]["candidate_angle_scores"]==120*37
                n=2048 if arm.endswith("2048") else 256
                assert row["training"]["training_board_count"]==n
                assert row["training"]["candidate_scalar_projection_evaluations"]==120*37*4*n
        s=[r for r in rows if r["cell"]==cell["id"]]
        assert len({r["initial_random_basis_sha256"] for r in s})==1
        assert len({r["hidden_rotation_sha256"] for r in s})==1
    # Excluded smoke uses the same seed under two *different* margin cells;
    # the registered cohort uses 32 distinct seeds once each.
    assert len(rows)==10
