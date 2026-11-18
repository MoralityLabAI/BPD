"""FC01: exactly 59 fresh trained-model independent physical sham calibration roots.

The prospective calibration PROTOCOL was committed before this code or
training any 12000..12058 root. All NC09 original source/estimator functions
are imported unchanged, and their source SHA256 is checked at runtime.

No scientific-confirmation root may be passed to this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import (
    PASSES, WIDTH, sample_x, external_labels, parity,
)
from mechinterp.bpd.experiments.nc04.experiment import (
    DenseLooper, factor_basis, self_score_discovery, context_accuracies,
)
from mechinterp.bpd.experiments.nc05.experiment import clean_dose_rms
from mechinterp.bpd.experiments.nc08.experiment import passwise_source_candidates
from mechinterp.bpd.experiments.nc09_fixed12 import experiment as nc09

ROOT=Path(__file__).resolve().parent
PROTO_SHA=hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest()
PARENT_PROTO_SHA="dc9423f1fd9e308221c41ed2af83e7398a59dd0eb0beefa8394091e373156e02"
PARENT_SOURCE_SHA="8d1b88eea8518a83669f03f9126136ca3d56ac5bac5fe041e2e03fb12922d380"
ROOTS=tuple(range(12000,12059))
SMOKE=11999
ARMS=("dp12","flat3","flat3_dose","flat4","reverse")


def protect_original_nc09_source():
    parent=Path(nc09.__file__).resolve()
    actual=hashlib.sha256(parent.read_bytes()).hexdigest()
    proto=hashlib.sha256(parent.with_name("PROTOCOL.md").read_bytes()).hexdigest()
    if actual!=PARENT_SOURCE_SHA or proto!=PARENT_PROTO_SHA:
        raise AssertionError("First registered NC09 source changed: calibration blocked")
    return {"nc09_original_experiment_sha256":actual,"nc09_original_protocol_sha256":proto}


def train_model(seed:int,*,steps:int=384):
    if seed not in ROOTS and seed!=SMOKE:
        raise ValueError("Only excluded 11999 and original 12000..12058 calibration")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=DenseLooper()
    source=sample_x(seed,1,2048)
    signs=torch.where(external_labels(source),1.,-1.)
    gen=torch.Generator().manual_seed(seed+1_737_701)
    optimizer=torch.optim.Adam(model.parameters(),lr=.005)
    for _ in range(steps):
        ix=torch.randint(len(source),(128,),generator=gen)
        logit=model(source[ix])[0][:,-1]
        loss=F.softplus(-2*signs[ix]*logit).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out=model(source)[0][:,-1]
        acc=float(((out>=0)==external_labels(source)).float().mean())
        density=float((model.mix.weight.detach().abs()>=1e-4).float().mean())
    return model,{
        "root":seed,"steps":steps,"train_size":len(source),
        "train_accuracy":acc,"mix_density":density,
        "last_batch_loss":float(loss.detach()),
    }


def evaluate_calibration_root(seed:int,*,steps:int=384,discovery_n:int=1024,
                              selector_n:int=512,heldout_n:int=1024,
                              sham_proposals:int=4096,shams:int=8):
    torch.set_num_threads(1)
    source_integrity=protect_original_nc09_source()
    model,training=train_model(seed,steps=steps)
    xd=sample_x(seed,2,discovery_n)
    xs=sample_x(seed,4,selector_n)
    bpd=BPDModel(model,{"mix":WIDTH},max_passes=PASSES,
                 ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(bpd,"native",seed)
    closure=parity(bpd,model,xs[:min(64,len(xs))])
    z,g,hits,modelscore= self_score_discovery(bpd,model,xd)
    with bpd.target_capture():
        model(xd,capture=True)
        acts=torch.stack(model.acts,1).detach()
    candidate=passwise_source_candidates(z,g)
    optimum=nc09.optimal_fixed12(candidate)
    arms=nc09.fixed_arm_projectors(candidate,optimum)
    U=bpd.wrappers["mix"].U.detach()
    dpdose=(clean_dose_rms(acts,U,arms["dp12"]["projector"])
            if arms["dp12"]["eligible"] else None)
    flatdose=(clean_dose_rms(acts,U,arms["flat3"]["projector"])
              if arms["flat3"]["eligible"] else None)
    gain=dpdose/flatdose if (dpdose is not None and
                            flatdose is not None and flatdose>1e-12) else None
    source_gain_ok=bool(gain is not None and .5<=gain<=1.5)
    for name,arm in arms.items():
        arm["gain"]=float(gain) if name=="flat3_dose" and gain is not None else 1.
        arm["source_write_rms"]=(clean_dose_rms(acts,U,arm["projector"])*arm["gain"]
                                 if arm["eligible"] else None)
        arm["model_only_validation"]=(nc09.model_only_diagnostic(
            bpd,model,xs,arm) if arm["eligible"] else None)
    if not source_gain_ok:
        arms["flat3_dose"]["eligible"]=False
    sham_pool=[]
    sham_admission=None
    if optimum is not None:
        sham_pool,sham_admission=nc09.source_matched_local_shams(
            acts,U,nc09.tuple_basis(candidate,optimum["ranks"]),seed,
            proposals=sham_proposals,want=shams
        )
    gauges={}
    for name in ("dp12","flat3","flat3_dose","flat4"):
        if arms[name]["eligible"]:
            gauges[name]=nc09.gauge_audit(
                bpd,model,xs,seed,arms[name]["projector"],arms[name]["gain"]
            )
    factor_basis(bpd,"native",seed)
    # All discovery, dose, shams, model-only diagnostics and gauge selections
    # are frozen BEFORE independent labeled task calibration samples are read.
    x_eval=sample_x(seed,3,heldout_n)
    labels=external_labels(x_eval)
    context=context_accuracies(model,x_eval,labels)
    with bpd.target_capture():
        clean,_=model(x_eval,capture=True)
        clean_writes=torch.stack(model.writes,1)
    true_sign=torch.where(labels,1.,-1.)
    clean_nll=float(F.softplus(-2*true_sign*clean[:,-1]).mean())
    result={
        "study":"BPD-NC09-FC01","phase":"INDEPENDENT_SHAM_CALIBRATION",
        "root":seed,"protocol_sha256":PROTO_SHA,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parent_integrity":source_integrity,
        "training":training,
        "splits":{"train":2048,"derivative_discovery":len(xd),
                  "model_validation":len(xs),"labeled_calibration":len(x_eval)},
        "full_mask_closure":closure,
        "modelscore_nonzero_count":modelscore["model_score_nonzero_count"],
        "source_rank12_optimum":optimum,
        "physical_source_gain":gain,
        "dp12_source_write_rms":dpdose,
        "flat3_source_write_rms":flatdose,
        "source_gain_eligible":source_gain_ok,
        "model_only_diagnostics":{k:v["model_only_validation"] for k,v in arms.items()},
        "source_arms":{k:{field:value for field,value in v.items() if field!="projector"}
                       for k,v in arms.items()},
        "sham_admission":sham_admission,
        "native_rotated_gauge":gauges,
        "clean_nll":clean_nll,
        "clean_task_context":context,
        "task_nll_by_arm":{},
        "sham_nlls":[],
    }
    for name,arm in arms.items():
        score=nc09.evaluate_task(bpd,model,x_eval,labels,clean,clean_writes,arm) if arm["eligible"] else None
        result["task_nll_by_arm"][name]=score
    if result["task_nll_by_arm"]["dp12"] is not None:
        for sham in sham_pool:
            scored=nc09.evaluate_task(
                bpd,model,x_eval,labels,clean,clean_writes,
                {"projector":sham["projector"],"gain":1.}
            )
            result["sham_nlls"].append(float(scored["nll_nats"]))
    mandatory=("dp12","flat3","flat3_dose","flat4")
    eligible=bool(
        min(context.values())>=.85 and training["mix_density"]>=.9
        and closure["passed"] and optimum is not None
        and source_gain_ok and
        all(arms[k]["eligible"] for k in mandatory)
        and sham_admission is not None and sham_admission["selected"]==shams
        and len(result["sham_nlls"])==shams
        and all(result["task_nll_by_arm"][k] is not None for k in mandatory)
        and all(gauges[k]["physical_map_relative_error"]<=1e-4 and
                gauges[k]["max_complete_rollout_logit_error"]<=5e-5
                for k in mandatory)
    )
    result["eligible"]=eligible
    if len(result["sham_nlls"])==shams and all(result["task_nll_by_arm"][k] is not None for k in mandatory):
        nll={k:result["task_nll_by_arm"][k]["nll_nats"] for k in mandatory}
        median=float(np.median(result["sham_nlls"]))
        null=[float(v-median) for v in result["sham_nlls"]]
        result["effects"]={
            "A_flat3_minus_dp12":float(nll["flat3"]-nll["dp12"]),
            "D_dose_flat3_minus_dp12":float(nll["flat3_dose"]-nll["dp12"]),
            "S_median_sham_minus_dp12":float(median-nll["dp12"]),
            "C_dp12_minus_clean":float(nll["dp12"]-clean_nll),
            "F_dp12_minus_flat4":float(nll["dp12"]-nll["flat4"]),
            "null_sham_signed_effects":null,
            "T_null_worst_abs":float(max(abs(x) for x in null)),
        }
    else:
        result["effects"]=None
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=int,default=SMOKE)
    p.add_argument("--steps",type=int,default=16)
    p.add_argument("--discovery",type=int,default=128)
    p.add_argument("--selector",type=int,default=128)
    p.add_argument("--heldout",type=int,default=128)
    p.add_argument("--sham-proposals",type=int,default=128)
    p.add_argument("--shams",type=int,default=2)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=evaluate_calibration_root(
        args.root,steps=args.steps,discovery_n=args.discovery,
        selector_n=args.selector,heldout_n=args.heldout,
        sham_proposals=args.sham_proposals,shams=args.shams)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"root":result["root"],"eligible":result["eligible"],
                      "qualified_shams":(result["sham_admission"] or {}).get("qualified"),
                      "null_worst":(result["effects"] or {}).get("T_null_worst_abs")},sort_keys=True))


if __name__=="__main__":
    main()
