"""Synthetic eight-model-root outcome tests; never evaluate new development sources."""
from __future__ import annotations

import pytest

from mechinterp.bpd.experiments.nc07.aggregate import aggregate,paired_root_bootstrap
from mechinterp.bpd.experiments.nc07.experiment import DEVELOPMENT,PROTO_HASH,RANKS


def synthetic_roots(*, chosen_ranks=None, keep_cost=.020, sham_adv=.050,
                    extra_rank4_cost=.008, fail_root=None):
    ranks=chosen_ranks or [3,3,3,3,4,4,4,4]
    assert len(ranks)==8
    rows=[]
    for i,seed in enumerate(DEVELOPMENT):
        rank=ranks[i]
        model_clean=.10
        selected_nll=model_clean+keep_cost
        cards={str(r):{"rank":r,"eligible":bool(r==rank)} for r in RANKS}
        cards[str(rank)]["eligible"]=True
        # The earliest passing rank must equal the registered selector result.
        if rank==4:
            cards["2"]["eligible"]=False
            cards["3"]["eligible"]=False
        elif rank==3:
            cards["2"]["eligible"]=False
        if rank==6:
            cards["2"]["eligible"]=cards["3"]["eligible"]=cards["4"]["eligible"]=False
        out={
            "study":"BPD-NC-07","phase":"DEVELOPMENT","root":seed,
            "protocol_sha256":PROTO_HASH,
            "eligible":i!=fail_root,
            "selected_rank":rank,
            "selector_by_rank":cards,
            "clean_final_nll_nats":model_clean,
            "heldout":{
                "selected":{"nll_nats":selected_nll},
                "fixed_rank3":{"nll_nats":selected_nll+.010},
                "fixed_rank4":{"nll_nats":selected_nll-extra_rank4_cost},
            },
            "median_sham_minus_selected_nll_nats":sham_adv,
        }
        rows.append(out)
    return rows


def test_root_bootstrap_preserves_paired_metric_and_is_deterministic():
    values=[(.02,.04,.008)]*8
    x=paired_root_bootstrap(values)
    assert x["repeats"]==2000 and x["seed"]==707007
    assert abs(x["selected_cost"]["upper95"]-.02)<1e-12
    assert x==paired_root_bootstrap(values)


def test_positive_fixed_source_only_adaptive_rank_gates():
    out=aggregate(synthetic_roots())
    assert out["registered_verdict"]=="ENGINEERING_POSITIVE"
    assert out["eligible_roots"]==8
    assert out["selected_rank_le3_roots"]==4
    assert out["mean_selected_rank"]==3.5
    assert all(out["gates"].values())
    assert not out["confirmed_scientific_result"]
    assert out["calibration_roots_evaluated"]==0
    assert out["confirmation_roots_evaluated"]==0


def test_negative_compression_task_fidelity_shams_and_root_integrity():
    assert not aggregate(synthetic_roots(chosen_ranks=[4]*8))["gates"]["compress_dimension_vs_rank4"]
    assert not aggregate(synthetic_roots(keep_cost=.07))["gates"]["absolute_true_label_fidelity"]
    assert not aggregate(synthetic_roots(sham_adv=0.))["gates"]["physical_dose_sham_specificity"]
    assert not aggregate(synthetic_roots(extra_rank4_cost=.04))["gates"]["bounded_task_cost_vs_rank4"]
    assert not aggregate(synthetic_roots(fail_root=0))["gates"]["eight_full_model_roots_and_shams"]
    with pytest.raises(ValueError):
        aggregate(synthetic_roots()[:-1])
    rows=synthetic_roots()
    rows[1]["root"]=700
    with pytest.raises(ValueError):
        aggregate(rows)
