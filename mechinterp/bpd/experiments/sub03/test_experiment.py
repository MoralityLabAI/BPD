"""SUB-03: source-only tests and excluded synthetic fixtures."""
import math
import pytest
import torch

from mechinterp.bpd.experiments.sub01.experiment import (
    WIDTH, Looper, sample_x, pca_projectors,
)
from mechinterp.bpd.experiments.sub03.experiment import (
    SMOKE, CALIBRATION_ROOTS, ANGLE_GRID, RANK, SHAMS,
    orthogonal, local_gramian_rotation, observable_gramian,
    spectrum_preserving_shams, geometry_admitted, bounded_signed,
    inverse_positive, upper_null_quantile, preflight_spec,
    train_source, aggregate,
)


def test_spectrum_and_causal_rank_exact_for_all_eight_shams():
    torch.manual_seed(19)
    m=Looper().eval()
    x=sample_x(SMOKE,2,96)
    cov, vals, vecs=observable_gramian(m,x)
    p=(vecs[:,-RANK:]@vecs[:,-RANK:].T).float()
    arms=spectrum_preserving_shams(SMOKE,cov,vals,vecs,p)
    assert len(arms)==SHAMS
    assert [v["theta_radians"] for v in arms[:4]]==list(ANGLE_GRID)
    assert all(a["theta_radians"] is None for a in arms[4:])
    assert geometry_admitted(arms)
    for a in arms:
        P=a["projector"]
        assert float((P@P-P).abs().max()) < 1e-5
        assert float(torch.trace(P))==pytest.approx(8.,abs=1e-4)
        assert a["geometry"]["eigenvalue_law_max_error"] < 1e-9


def test_local_rotation_has_exact_angle_and_keeps_top_four():
    torch.manual_seed(20)
    m=Looper().eval()
    cov, vals,vec=observable_gramian(m,sample_x(SMOKE,2,64))
    for theta in ANGLE_GRID:
        q=local_gramian_rotation(vec,theta,770131)
        assert torch.allclose(q.T@q,torch.eye(WIDTH,dtype=torch.float64),atol=1e-9)
        assert torch.allclose(q@vec[:,-4:],vec[:,-4:],atol=1e-9)
        weak=vec[:,4:8]
        assert torch.allclose(weak.T@q@weak,
                              math.cos(theta)*torch.eye(4,dtype=torch.float64),
                              atol=1e-9)


def test_root_level_tolerance_requires_59_independent_worlds():
    assert upper_null_quantile([.2]*58)==(None,None)
    assert upper_null_quantile([.2]*59)[1]==59
    assert upper_null_quantile([.1]*30+[.4]*34)[0]==pytest.approx(.4)
    assert 0 < bounded_signed(.4335) < 1
    assert -1 < bounded_signed(-.4335) < 0
    assert inverse_positive(bounded_signed(.4335))==pytest.approx(.4335)
    assert inverse_positive(0.)==0


def test_preflight_units_are_full_roots_and_null_family_is_complete():
    fake={
      "root":1400,
      "pilot":{"bounded_effect": bounded_signed(.4)},
      "calibration":{"shams":[{"bounded_sham_effect":.1}]*8},
    }
    p=preflight_spec([fake])
    assert p["planned_comparisons"]==1
    assert p["confirm_units_planned"]==64
    assert p["pilot"][0]["unit_id"]=="root-1400"
    assert len(p["pilot"][0]["sham_effects"])==8
    assert p["target_effect"]=="0.30"
    assert p["heldout_sealed"] is True


def test_guardrails_keep_independent_root_roster_sealed():
    assert CALIBRATION_ROOTS==tuple(range(1400,1464))
    assert SMOKE not in CALIBRATION_ROOTS
    with pytest.raises(ValueError,match="Outside registered"):
        train_source(1400-2,steps=1)
    with pytest.raises(ValueError,match="64"):
        aggregate([])
