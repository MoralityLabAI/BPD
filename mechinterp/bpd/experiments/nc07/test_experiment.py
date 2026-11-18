"""NC07 excluded-seed and math/unit controls. No development root is trained here."""
from __future__ import annotations

import torch

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import PASSES,WIDTH,sample_x,parity
from mechinterp.bpd.experiments.nc04.experiment import factor_basis
from mechinterp.bpd.experiments.nc07.experiment import (
    SMOKE,DEVELOPMENT,RANKS,SELECTION,
    train_model,rank_projectors,clean_model_kl_selection,
    local_physical_shams,physical_gauge_audit,evaluate_root
)


def test_frozen_seed_roster_and_rank_selector():
    assert SMOKE==699
    assert DEVELOPMENT==tuple(range(700,708))
    assert SMOKE not in DEVELOPMENT
    assert RANKS==(2,3,4,6)
    assert SELECTION["mean_final_kl_max"]==.027
    assert SELECTION["p95_final_kl_max"]==.25
    assert SELECTION["min_discovery_retained_write_fraction"]==.350


def test_eigenspace_subspace_invariants_and_nestedness():
    gen=torch.Generator().manual_seed(171)
    g=torch.randn((512,PASSES,WIDTH),generator=gen)
    # Separate passwise-coordinate scales to guarantee stable eigengaps.
    scales=torch.linspace(3,.1,WIDTH)
    g=g*scales
    out=rank_projectors(g)
    assert set(out)==set(RANKS)
    for rank,c in out.items():
        p=c["p"]
        assert c["eigengap_eligible"]
        assert p is not None and p.shape==(PASSES,WIDTH,WIDTH)
        assert c["idempotence_max"]<1e-10
        assert c["symmetry_max"]<1e-10
        assert all(row["eigengap_ratio"]>=1.001 for row in c["spectrum"])
        assert all(abs(float(torch.trace(p[t]))-rank)<1e-8
                   for t in range(PASSES))
    assert (out[2]["p"]@out[4]["p"]-out[2]["p"]).abs().max()<1e-8


def test_predata_rank_criterion_has_no_true_label_input():
    model,_=train_model(SMOKE,steps=8)
    xd=sample_x(SMOKE,2,64)
    xs=sample_x(SMOKE,4,64)
    assert not torch.equal(xd,xs)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
        ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd,"native",SMOKE)
    assert parity(bpd,model,xs[:16])["passed"]
    # Use source-only pseudo gradients to exercise the rank-selector path.
    gen=torch.Generator().manual_seed(171)
    grad=torch.randn((64,PASSES,WIDTH),generator=gen)*torch.linspace(3,.1,WIDTH)
    cand=rank_projectors(grad)
    with bpd.target_capture():
        model(xd,capture=True)
        activ=torch.stack(model.acts,1).detach()
    chosen,cards=clean_model_kl_selection(bpd,model,xs,activ,cand)
    assert set(cards)==set(RANKS)
    assert chosen is None or chosen in RANKS
    assert chosen==next((r for r in RANKS if cards[r]["eligible"]),None)
    assert all("spectrum" in x and "eigengap_eligible" in x for x in cards.values())


def test_physical_shams_are_rank_and_discovery_dose_matched():
    gen=torch.Generator().manual_seed(119)
    x=torch.randn((256,PASSES,WIDTH),generator=gen)
    g=torch.randn((256,PASSES,WIDTH),generator=gen)*torch.linspace(3,.1,WIDTH)
    candidate=rank_projectors(g)[3]
    assert candidate["p"] is not None
    u=torch.eye(WIDTH,dtype=torch.float64)
    shams,diag=local_physical_shams(x,u,candidate["eigbasis"],SMOKE,
                                   proposals=128,want=2)
    assert diag["proposal_budget"]==128
    assert diag["selected"]<=2
    if len(shams)==2:
        assert all(v["relative_discovery_dose_error"]<=.15 for v in shams)
        assert all(v["p"].shape==(PASSES,WIDTH,WIDTH) for v in shams)


def test_excluded_699_native_end_to_end_smoke_keeps_holdout_separate():
    rec=evaluate_root(SMOKE,steps=8,derivative_n=64,selection_n=64,
        heldout_n=64,sham_proposals=64,shams=2)
    assert rec["root"]==SMOKE and rec["phase"]=="DEVELOPMENT"
    assert rec["training"]["steps"]==8
    assert rec["split_sizes"]=={
        "train":2048,"derivative_discovery":64,
        "rank_selection":64,"final_heldout":64}
    assert rec["full_mask_closure"]["passed"]
    assert set(rec["selector_by_rank"])=={"2","3","4","6"}
    assert rec["selected_rank"] is None or rec["selected_rank"] in RANKS
    assert rec["protocol_sha256"] and rec["source_sha256"]
    # The source-only selector has already fixed its rank before final labels.
    chosen=rec["selected_rank"]
    assert chosen is None or rec["selector_by_rank"][str(chosen)]["eligible"]
