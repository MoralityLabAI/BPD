"""Stage 1 physical native BPDModel/TiedComponentLinear full-mask parity.

Only excluded engineering seed 901 is used. This checks the real repository
wrapper, not an independent reimplementation of its masking semantics.
"""
from __future__ import annotations

import json
import torch
from torch import nn

from mechinterp.bpd.components import (
    BPDModel, PassMaskSchedule, ComponentIntervention,
)
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.theory.sf05b_source_families.source_families import (
    MAX_WRITES, make_batch, change_last_source,
)
from mechinterp.bpd.theory.sf05b_stage1.teacher import (
    HIDDEN, INNER, COMPONENTS, Teacher, Factors, train_teacher, train_factors,
)


class Core(nn.Module):
    def __init__(self,teacher:Teacher):
        super().__init__()
        self.core=nn.Linear(INNER,HIDDEN,bias=False)
        with torch.no_grad():
            self.core.weight.copy_(teacher.C.T)

    def forward(self,x):
        return self.core(x)


def verify_native(seed:int=901,teacher_updates:int=2600,
                  factor_updates:int=450,n:int=64):
    if seed!=901:
        raise ValueError("stage 1 native engineering seed pinned to 901")
    torch.set_num_threads(1)
    teacher=train_teacher(seed,teacher_updates)
    factors=train_factors(seed,teacher,factor_updates)
    wrapper_model=BPDModel(
        Core(teacher),{"core":COMPONENTS},max_passes=MAX_WRITES+1,
        ablation=AblationConfig(mode="zero",sphere_rescale=False),
        init_svd=False)
    wrapper=wrapper_model.wrappers["core"]
    batch=make_batch(seed*430000+981100,n,"train")
    with torch.no_grad():
        wrapper.V.copy_(factors.V)
        wrapper.U.copy_(factors.U)
        mask=PassMaskSchedule(torch.ones(n,MAX_WRITES+1,1,COMPONENTS),
                              torch.zeros(n,MAX_WRITES+1,1))
        h=torch.zeros(n,HIDDEN)
        f=factors.weight()
        independent=torch.zeros(n,HIDDEN)
        with wrapper_model.masked_components({"core":mask}):
            for j in range(MAX_WRITES):
                fv=teacher.features(h,batch.events[:,j])
                wn=wrapper_model.target_model(fv.unsqueeze(1))[:,0]
                h=torch.where(batch.event_mask[:,j,None],h+wn,h)
                ff=teacher.features(independent,batch.events[:,j])
                independent=torch.where(batch.event_mask[:,j,None],
                                        independent+ff@f,independent)
            native=h+wrapper_model.target_model(
                teacher.features(h,batch.query).unsqueeze(1))[:,0]
        reference=independent+teacher.features(
            independent,batch.query)@f
        full_mask_max=float((native-reference).abs().max())
        if full_mask_max>1e-5:
            raise AssertionError(f"native fullmask parity {full_mask_max}")

        donor_events,_,_=change_last_source(batch,seed+730100)
        r=teacher.rollout(batch)
        row=torch.arange(n)
        last=batch.lengths-1
        last_rec=batch.events[row,last]
        last_donor=donor_events[row,last]
        feat_rec=teacher.features(r.before_last,last_rec)
        feat_don=teacher.features(r.before_last,last_donor)
        donor_acts=(feat_don@factors.V).unsqueeze(1)
        wrapper.mode="target"
        wrapper.reset_pass_state(pass_offset=0)
        wrapper.intervention=ComponentIntervention(
            components=tuple(range(COMPONENTS)),passes=frozenset({0}),
            mode="swap_acts",donor_acts=donor_acts)
        try:
            native_swapped=wrapper_model.target_model(
                feat_rec.unsqueeze(1))[:,0]
        finally:
            wrapper.intervention=None
        expected=feat_rec@teacher.C+(feat_don-feat_rec)@f
        swap_max=float((native_swapped-expected).abs().max())
        if swap_max>1e-5:
            raise AssertionError(f"native donor physical swap {swap_max}")
    return {
        "seed":seed,"n":n,
        "native_fullmask_seven_calls_max_error":full_mask_max,
        "native_source_label_swap_max_error":swap_max,
        "factor_weight_mse":float((factors.weight()-teacher.C).square().mean()),
        "passed":True,
        "evidence":"excluded-root physical wrapper numerical parity only",
    }


if __name__=="__main__":
    print(json.dumps(verify_native(),indent=2))
