"""BPD FSI-04 native pilot: donor-causal correction with a dose-matched sham.

This is a development-only model runner. It requires an existing pinned 7M
TRM checkpoint and a fresh externally audited Sudoku CSV. It NEVER downloads
weights or reads ARG-03's 168-case sealed test set.

Usage:
 python -m mechinterp.dark_fiber_gyms.fsi04.native_pilot \
    --repo-root . --checkpoint-file /local/step_21700 --tag gaoxin492 \
    --eval-csv runs/fsi04/development.csv \
    --exclude-csv mechinterp/loopscope02/results/benchmark_bundle.json \
    --out runs/fsi04/gaoxin_development.json --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from mechinterp.dark_fiber_gyms.core import sha256_file, trm_manifold_swap
from mechinterp.dark_fiber_gyms.loopscope04_trm import (
    audit_boards, classify_sudoku_board, import_sudoku_prop, import_trm,
    load_model, read_boards,
)
from .geometry import matched_random_joint

REG=Path(__file__).with_name("REGISTRATION.json")
ARMS=("clean","donor_joint","random_joint_dose_matched","self_control")


def _bad_path(p: Path) -> bool:
    s=p.as_posix().lower()
    return p.suffix.lower()==".jsonl" or "arg03" in s or "test.jsonl" in s


@torch.inference_mode()
def native_case(TS,model,row,*,patch_step=4,horizon=32,sigma=.35,seed=2026111744):
    if row.get("stratum")!="S4_stalled" and row.get("stratum")!="WB-Prop":
        raise ValueError("pilot admits only S4_stalled or WB-Prop negative-control roots")
    device=model.H_init.device
    x=model.encode_inputs(TS.encode_questions([row["question"]],device))
    zha,zla=model.init_states(1);zhb,zlb=model.init_states(1)
    g=torch.Generator(device="cpu").manual_seed(seed)
    def kick(t):
        noise=torch.randn(tuple(t.shape),generator=g,dtype=torch.float32).to(t.device)
        return t+sigma*t.float().square().mean().sqrt()*noise
    zhb,zlb=kick(zhb),kick(zlb)
    for _ in range(patch_step):
        zha,zla=model.act_step(zha,zla,x)
        zhb,zlb=model.act_step(zhb,zlb,x)

    board_start=model.puzzle_emb_len
    h_swap,donor_cert=trm_manifold_swap(zha,zhb,model.lm_head.weight,board_start=board_start)
    l_swap=zla.clone();l_swap[:,board_start:,:]=zlb[:,board_start:,:]
    h_random,l_random,dose_cert=matched_random_joint(
        zha,h_swap,zla,l_swap,model.lm_head.weight,board_start,seed+9137)
    states={"clean":(zha,zla),"donor_joint":(h_swap,l_swap),
            "random_joint_dose_matched":(h_random,l_random),
            "self_control":(zha.clone(),zla.clone()),"donor_reference":(zhb,zlb)}
    h_all=torch.cat([states[k][0] for k in (*ARMS,"donor_reference")])
    l_all=torch.cat([states[k][1] for k in (*ARMS,"donor_reference")])
    x_all=x.expand(len(ARMS)+1,-1,-1)
    pre=model.lm_head(h_all)[:,board_start:,:].float()
    preerr=float((pre[:len(ARMS)]-pre[0:1]).abs().max().item())
    if preerr>1e-3:raise AssertionError(f"readout parity failed at t={patch_step}: {preerr}")
    if not torch.equal(h_all[0],h_all[ARMS.index("self_control")]):
        raise AssertionError("self control is not exactly clean")

    first_cos={}
    final_pred=None
    for t in range(patch_step+1,horizon+1):
        h_all,l_all=model.act_step(h_all,l_all,x_all)
        if t==patch_step+1:
            logits=model.digit_logits(h_all).float()
            ref=logits[0]
            donor=(logits[-1]-ref).reshape(-1)
            donorm=float(donor.norm().item())
            for i,k in enumerate(ARMS):
                shift=(logits[i]-ref).reshape(-1)
                n=float(shift.norm().item())
                first_cos[k]=float(torch.dot(shift,donor).item()/(n*donorm)) if n>1e-7 and donorm>1e-7 else 0.0
        if t==horizon:
            final_pred=model.digit_logits(h_all).argmax(dim=-1).add(1).cpu().tolist()
    if final_pred is None:raise AssertionError("no continuation")
    answer=[int(c) for c in row["answer"]]
    clean=final_pred[0]
    outcomes={}
    for i,k in enumerate(ARMS):
        pred=final_pred[i]
        gained=sum(z!=a and v==a for z,v,a in zip(clean,pred,answer))
        lost=sum(z==a and v!=a for z,v,a in zip(clean,pred,answer))
        correct=sum(v==a for v,a in zip(pred,answer))
        assert correct-sum(z==a for z,a in zip(clean,answer))==gained-lost
        outcomes[k]={"correct_cells":int(correct),"exact":bool(pred==answer),
                     "gained_cells_vs_clean":int(gained),"lost_cells_vs_clean":int(lost),
                     "net_cells_vs_clean":int(gained-lost),
                     "first_successor_cos_to_donor":first_cos[k]}
    if outcomes["self_control"]!=dict(outcomes["clean"],first_successor_cos_to_donor=first_cos["self_control"]):
        # The first-successor cosine is zero for both clean and self.
        if (outcomes["self_control"]["correct_cells"]!=outcomes["clean"]["correct_cells"] or
            outcomes["self_control"]["exact"]!=outcomes["clean"]["exact"]):
            raise AssertionError("self-control trajectory diverged from clean")
    if dose_cert["zH_dose_mismatch_max_abs"]>2e-3 or dose_cert["zL_dose_mismatch_max_abs"]>2e-3:
        raise AssertionError("sham dose mismatch")
    return {"case_id":hashlib.sha256(row["question"].encode()).hexdigest(),
            "stratum":row["stratum"],"patch_step":patch_step,"horizon":horizon,
            "sigma":sigma,"seed":seed,"arms":outcomes,
            "certificate":{"donor_sphere":donor_cert,"dose_matched_sham":dose_cert,
                           "pre_patch_full_head_max_abs":preerr}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--checkpoint-file",type=Path,required=True)
    p.add_argument("--tag",choices=("yagizdevre","gaoxin492"),required=True)
    p.add_argument("--eval-csv",type=Path,required=True)
    p.add_argument("--exclude-csv",type=Path,action="append",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--limit",type=int,default=0)
    a=p.parse_args(argv)
    reg=json.loads(REG.read_text())
    if reg["schema"]!="BPD-FSI04/native-causal-dose-matched-development-pilot/v1":
        p.error("registration changed")
    if _bad_path(a.eval_csv) or any(_bad_path(z) for z in a.exclude_csv):
        p.error("ARG03/test.jsonl or other non-development path forbidden")
    if not a.checkpoint_file.is_file():p.error("local checkpoint missing; no model download")
    if a.limit<0:p.error("limit must be nonnegative")
    rows=read_boards(a.eval_csv,a.limit)
    if len({r["id"] for r in rows})!=len(rows):raise ValueError("duplicate root")
    audit=audit_boards([],rows,a.exclude_csv,check_full_automorphisms=True)
    TS=import_trm(a.repo_root)
    SP=import_sudoku_prop(a.repo_root)
    for row in rows:
        actual=classify_sudoku_board(SP,row["question"])
        required="WB-Prop" if actual["wb_group"]=="WB-Prop" else actual["fine_stratum"]
        if row.get("stratum") not in ("S4_stalled","WB-Prop") or row["stratum"]!=required:
            raise ValueError(f"roster stratum mismatch: {row.get('stratum')} vs {required}")
    torch.set_grad_enabled(False)
    model,sha=load_model(TS,a.checkpoint_file,a.tag,a.device)
    t0=time.perf_counter();results=[]
    for j,row in enumerate(rows):
        case_seed=reg["sampling"]["seed"]^int(row["id"][:8],16)
        results.append(native_case(TS,model,row,
          patch_step=reg["intervention"]["patch_step"],
          horizon=reg["intervention"]["horizon"],
          sigma=reg["intervention"]["sigma"],seed=case_seed))
        print(f"{a.tag}: {j+1}/{len(rows)} native development roots",flush=True)
    output={"schema":"BPD-FSI04/native-development-causal-v1",
        "status":"exploratory-development-only",
        "checkpoint_tag":a.tag,"checkpoint_sha256":sha,
        "registration_sha256":sha256_file(REG),
        "eval_csv_sha256":sha256_file(a.eval_csv),
        "audit":audit,"rows":results,
        "duration_s":time.perf_counter()-t0,
        "heldout_test":"ARG-03-N protected 168-case test untouched"}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(f"{a.tag}: wrote {len(results)} development roots to {a.out}",flush=True)


if __name__=="__main__":
    main()
