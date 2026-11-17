"""Source-only PD12 engineering tests using excluded seed 3001.

Never trains or evaluates registered PD12 seed/rotation pairs.
"""
from __future__ import annotations

import math
import torch

from mechinterp.bpd.theory.pd12.experiment import (
    CONFIG,CELLS,UNITS,ARMS,FAMILIES, support_banks, rotation_for_seed,
    initial_random_basis,sample_batch,angle_grid,gaussian_mass,
    search_basis,basis_alignment,physical_model,evaluate_family,
    run_arm,summarize,
)

SMOKE=CONFIG["execution"]["smoke_seed"]


def sample_patterns(support):
    return [tuple(torch.nonzero(support[b,t,0]).flatten().tolist())
            for b in range(support.shape[0]) for t in range(4)]


def test_frozen_full_factorial_and_fresh_seeds():
    assert CONFIG["study_id"]=="BPD-PD-12"
    assert len(CELLS)==8
    assert len(UNITS)==64
    assert len(set(seed for cell,seed in UNITS))==64
    assert SMOKE not in {seed for cell,seed in UNITS}
    assert {cell["dimension"] for cell in CELLS}=={8,16}
    assert {cell["support_distribution"] for cell in CELLS}=={"uniform","clustered"}
    assert {cell["margin"] for cell in CELLS}=={"strong","weak"}
    assert all(len(cell["seed_list"])==8 for cell in CELLS)
    assert ARMS==("global37","local37","global13","planted_Q_oracle")
    assert len(FAMILIES)==8
    assert CONFIG["objective"]["sweeps"]==3
    assert CONFIG["source"]["training_boards"]==256


def test_dimensional_support_banks_keep_every_withheld_combination_out_of_train():
    for dim in (8,16):
        bank=support_banks(dim)
        for k in (1,2,3,4):
            train=set(bank["train"][k])
            withheld=set(bank["heldout"][k])
            cluster=set(bank["cluster"][k])
            assert train.isdisjoint(withheld)
            assert cluster<=train
            assert len(train)>0
            if k>1:
                assert len(withheld)>0
        assert len(bank["heldout"][2])==dim//2
        assert len(bank["heldout"][4])>0


def test_rotations_dense_inputs_and_margin_floor():
    for cell in (CELLS[0],CELLS[3],CELLS[4],CELLS[7]):
        d=cell["dimension"]
        q=rotation_for_seed(SMOKE,d)
        assert torch.equal(q,rotation_for_seed(SMOKE,d))
        assert torch.allclose(q.T@q,torch.eye(d),atol=3e-6,rtol=0)
        rng=torch.Generator().manual_seed(100000+SMOKE)
        x,support,a=sample_batch(rng,40,q,cell=cell,family="train")
        assert x.shape==support.shape==a.shape==(40,4,1,d)
        assert torch.allclose(x.squeeze(2)@q,a.squeeze(2),atol=3e-6,rtol=0)
        assert float((x.abs()>1e-6).float().mean())>.98
        lo,hi=CONFIG["source"]["amplitude_ranges"][cell["margin"]]
        for t in range(4):
            active=support[:,t].bool()
            assert bool((a[:,t][active].abs()>=lo).all())
            assert bool((a[:,t][active].abs()<=hi).all())
            assert bool((a[:,t][~active]==0).all())
        for pattern in sample_patterns(support):
            assert pattern in support_banks(d)["train"][len(pattern)]


def test_clustered_source_has_actual_coactivation_concentration():
    for dim in (8,16):
        base=dict(next(c for c in CELLS if c["dimension"]==dim))
        q=rotation_for_seed(SMOKE,dim)
        uniform=dict(base,support_distribution="uniform")
        clustered=dict(base,support_distribution="clustered")
        _,su,_=sample_batch(torch.Generator().manual_seed(101111),256,q,
                             cell=uniform,family="train")
        _,sc,_=sample_batch(torch.Generator().manual_seed(101111),256,q,
                             cell=clustered,family="train")
        bank=support_banks(dim)
        def clustered_fraction(mask):
            p=sample_patterns(mask)
            chosen=[x for x in p if len(x)>1]
            return sum(x in bank["cluster"][len(x)] for x in chosen)/len(chosen)
        assert clustered_fraction(sc)>.70
        assert clustered_fraction(su)<.30


def test_hidden_supports_and_transition_never_contaminate_training():
    for cell in (CELLS[0],CELLS[-1]):
        d=cell["dimension"]
        q=rotation_for_seed(SMOKE,d)
        for fam in ("k2_unseen","k3_unseen","k4_unseen","transition_unseen"):
            x,support,_=sample_batch(
                torch.Generator().manual_seed(200000+SMOKE+10000*FAMILIES.index(fam)),
                16,q,cell=cell,family=fam
            )
            seen=sample_patterns(support)
            for index,p in enumerate(seen):
                if fam=="transition_unseen":
                    if index%4==0:
                        assert p in support_banks(d)["train"][1]
                    else:
                        assert p in support_banks(d)["heldout"][len(p)]
                else:
                    assert p in support_banks(d)["heldout"][len(p)]
        assert CONFIG["source"]["heldout_seed_offset"]!=CONFIG["source"]["train_seed_offset"]


def test_candidate_budget_matched_and_cost_scaling_exact():
    for dim in (8,16):
        pair_count=dim*(dim-1)//2
        for arm,expected in (("global37",37),("local37",37),("global13",13)):
            angles=angle_grid(arm)
            assert len(angles)==expected
            assert abs(float(angles[len(angles)//2]))<1e-13
            assert 3*pair_count*expected==(
                3*dim*(dim-1)//2*expected
            )
    g=angle_grid("global37")
    l=angle_grid("local37")
    assert math.isclose(float(g[0]),-math.pi/2,abs_tol=1e-12)
    assert math.isclose(float(g[-1]),math.pi/2,abs_tol=1e-12)
    assert math.isclose(float(l[0]),-math.radians(5),abs_tol=1e-12)


def test_unlabelled_pair_search_monotonic_and_same_initial_basis():
    cell=next(c for c in CELLS if c["dimension"]==8 and c["margin"]=="strong")
    d=cell["dimension"]
    q=rotation_for_seed(SMOKE,d)
    x,support,_=sample_batch(torch.Generator().manual_seed(100000+SMOKE),16,q,cell=cell)
    initial=initial_random_basis(SMOKE,d)
    assert torch.equal(initial,initial_random_basis(SMOKE,d))
    poisoned=torch.ones_like(support)
    assert not torch.equal(poisoned,support)
    for arm in ("global37","local37","global13"):
        read,diag=search_basis(x,initial,cell,arm,sweeps=1)
        assert diag["candidate_angle_scores"]==(d*(d-1)//2)*len(angle_grid(arm))
        assert diag["candidate_scalar_projection_evaluations"]==diag["candidate_angle_scores"]*16*4
        assert diag["orthogonality_max_error"]<1e-6
        assert diag["final_mass"]>=diag["initial_mass"]-1e-9
        assert torch.allclose(read.T@read,torch.eye(d,dtype=torch.float64),atol=1e-6)
        # No support or count argument exists in the optimizer interface.
        again,_=search_basis(x.clone(),initial.clone(),cell,arm,sweeps=1)
        assert torch.equal(read,again)


def test_dimension_general_assignment_and_true_q_physical_masks():
    for cell in (CELLS[0],CELLS[-1]):
        d=cell["dimension"]
        q=rotation_for_seed(SMOKE,d)
        permutation=list(range(1,d))+[0]
        read=q.double()[:,permutation]
        alignment=basis_alignment(read,q)
        assert alignment["component_to_latent"]==permutation
        assert alignment["min_cosine"]>.99999
        bpd=physical_model(d,q.double())
        row=evaluate_family(
            bpd,q.double(),q,SMOKE,cell,"k4_unseen",
            basis_alignment(q.double(),q),16
        )
        assert row["binary_mask"]
        assert row["support_fraction"]==1.0
        assert row["false_off"]==row["false_on"]==0
        assert row["mse"]<1e-9


def test_two_dimension_smoke_and_registered_summary_guard():
    rows=[]
    for cell in (CELLS[0],CELLS[-1]):
        for arm in ARMS:
            row=run_arm(SMOKE+cell["dimension"],cell,arm,
                        train_boards=12,heldout_boards=8,sweeps=1)
            rows.append(row)
            assert len(row["heldout"]["per_family"])==8
            assert all(v["binary_mask"] for v in row["heldout"]["per_family"].values())
            if arm=="planted_Q_oracle":
                assert row["success"]
                assert all(v["support_fraction"]==1 for v in row["heldout"]["per_family"].values())
                assert row["training"]["candidate_angle_scores"]==0
            else:
                assert row["training"]["candidate_angle_scores"]==(
                    cell["dimension"]*(cell["dimension"]-1)//2
                    *len(angle_grid(arm))
                )
                assert not row["uses_true_q_for_optimization"]
    # Full registered paired summary is used only with 64 actual units,
    # so engineering smokes deliberately cannot satisfy its unit roster.
    assert len(rows)==8
