"""DT02-A: bounded CI CPU training of actual PureSudokuTRM physical recurrence.

Engineering roots 901/902 only. Task targets originate in independent exact
4x4 Sudoku, ARC-style (NOT official ARC-1), and host-observed storyworlds.
Depths 2/4/6 are supervised; depth 8/12 is held out. A separate physical
SVD steering diagnostic measures exact current full-logit conservation.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from mechinterp.loopscope02.scripts.trm_sudoku import PureSudokuTRM
from mechinterp.bpd.experiments.depth_steering_gym.worlds import (
    sudoku4, sudoku_legal, arc_analogue, HiddenEndingWorld,
)

FIT=(2,4,6)
TEST=(8,12)
FAMILIES=("sudoku4", "arc_style", "storyworld")
ACTIONS=HiddenEndingWorld.ACTIONS


def expert_action(obs):
    if obs["trust"] < 2: return "listen"
    if obs["evidence"] < 2: return "research"
    if not obs["token"]: return "share"
    if obs["stability"] < 2: return "stabilize"
    return "advance"


def story_encoding(obs):
    arr=[min(obs["day"],9),max(0, min(9,obs["trust"]+2)),
         min(9,obs["evidence"]),min(9,obs["stability"]),
         int(obs["token"]),len(obs["actions"]),int(obs["terminal"])]
    return [int(x)+1 for x in (arr+[0]*16)[:16]]


def cases(family,seed,n,split):
    g=random.Random(seed*7+29)
    xs=[];ys=[]
    for i in range(n):
        source=seed*100000+i*97
        if family=="sudoku4":
            c=sudoku4(source,split,blanks=6)
            xs.append([v+1 for row in c.givens for v in row]);ys.append([v-1 for row in c.solution for v in row])
        elif family=="arc_style":
            c=arc_analogue(source,"train" if split=="train" else "test")
            xs.append([v+1 for row in c.test_input for v in row]);ys.append([v for row in c.answer for v in row])
        elif family=="storyworld":
            world=HiddenEndingWorld(source,split,horizon=8)
            # A small supervised public-state curriculum. Hidden gate never enters the encoder.
            for _ in range(g.randrange(8)):
                if world.state.closed: break
                world.step(g.choice(ACTIONS))
            obs=world.observe()
            xs.append(story_encoding(obs));ys.append([ACTIONS.index(expert_action(obs))]+[0]*15)
        else: raise ValueError(family)
    x=torch.tensor(xs,dtype=torch.long)
    gold=torch.tensor(ys,dtype=torch.long)
    return x,gold


def init_model(seed):
    torch.manual_seed(seed)
    model=PureSudokuTRM(hidden_size=32,num_heads=4,l_layers=1,
                         h_cycles=1,l_cycles=1,puzzle_emb_len=1,seq_len=16)
    return model


def forward_depths(model, inputs, depths):
    emb=model.encode_inputs(inputs)
    zh,zl=model.init_states(len(inputs))
    result={}
    for t in range(1,max(depths)+1):
        zh,zl=model.act_step(zh,zl,emb)
        if t in depths: result[t]=model.digit_logits(zh)
    return result


def train(seed,family,updates=64,batch=16):
    model=init_model(seed+{ "sudoku4":0,"arc_style":10000,"storyworld":20000}[family])
    opt=torch.optim.AdamW(model.parameters(),lr=.004)
    for i in range(updates):
        x,gold=cases(family,seed*20000+113+i,batch,"train")
        logits=forward_depths(model,x,FIT)
        loss=0.
        for d in FIT:
            output=logits[d][:,:,:5 if family=="storyworld" else 4]
            if family=="storyworld":
                loss=loss+F.cross_entropy(output[:,0,:],gold[:,0])
            else:
                loss=loss+F.cross_entropy(output.flatten(0,1),gold.flatten())
        loss/=len(FIT)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
        if not torch.isfinite(loss):raise ValueError("nonfinite recurrent training loss")
    return model.eval(),float(loss.detach())


@torch.no_grad()
def evaluate(model,seed,family,n=64):
    x,gold=cases(family,seed*70000+8001,n,"eval")
    outputs=forward_depths(model,x,TEST)
    report={}
    for depth,logit in outputs.items():
        pred=logit[:,:,:5 if family=="storyworld" else 4].argmax(-1)
        if family=="storyworld":
            accuracy=float((pred[:,0]==gold[:,0]).float().mean());exact=accuracy
            board_legality=None
        else:
            accuracy=float((pred==gold).float().mean())
            exact=float((pred==gold).all(1).float().mean())
            board_legality=(sum(sudoku_legal(tuple(tuple(int(c+1) for c in row)
                             for row in pred[i].reshape(4,4)),require_complete=True)
                             for i in range(n))/n if family=="sudoku4" else None)
        report[str(depth)]={"token_accuracy":accuracy,"exact_task_accuracy":exact,
                            "board_legality":board_legality,"n":n,
                            "task_label_provenance":"independent exact host oracle",
                            "official_arc1":False}
    if family=="storyworld":
        endings=[]
        for i in range(min(n,16)):
            world=HiddenEndingWorld(seed*50000+9001+i,"eval",horizon=12)
            for turn in range(12):
                xx=torch.tensor([story_encoding(world.observe())],dtype=torch.long)
                action=ACTIONS[forward_depths(model,xx,(12,))[12][:,:,:5].argmax(-1)[0,0].item()]
                world.step(action)
            endings.append(world.ending())
        report["12"]["terminal_episodes"]=len(endings)
        report["12"]["host_ending_success_rate"]=sum(e!="unfinished" for e in endings)/len(endings)
        report["12"]["ending_histogram"]={v:endings.count(v) for v in sorted(set(endings))}
    return report


def resume_after(model, x, zh, zl, start, end):
    emb=model.encode_inputs(x)
    for _ in range(start,end): zh,zl=model.act_step(zh,zl,emb)
    return model.digit_logits(zh)


@torch.no_grad()
def steering(model,seed,n=8):
    recipient,_=cases("sudoku4",seed*4000+101,n,"eval")
    donor,_=cases("sudoku4",seed*4000+393,n,"eval")
    rx=model.encode_inputs(recipient);dx=model.encode_inputs(donor)
    hr,lr=model.init_states(n);hd,ld=model.init_states(n)
    for _ in range(2):
        hr,lr=model.act_step(hr,lr,rx)
        hd,ld=model.act_step(hd,ld,dx)
    # Protect all 11 physical logits, rather than only four visible task digits.
    W=model.lm_head.weight.detach().double()
    _,s,vh=torch.linalg.svd(W,full_matrices=True)
    rank=int((s>1e-8*s[0]).sum())
    ker=vh[rank:].T.float()
    if ker.numel()==0:raise RuntimeError("no protected head kernel")
    P=ker@ker.T
    delta=(hd-hr)@P
    delta[:,0,:]=0.0
    weight=model.layers[0].o_proj.weight.detach().double()
    _,_,basis_v=torch.linalg.svd(weight,full_matrices=False)
    B=basis_v[:min(8,len(basis_v))].T.float()
    partial=((hd-hr)@B)@B.T@P
    partial[:,0,:]=0
    probs=lambda h:torch.softmax(model.lm_head(h)[:,1:,:].float(),dim=-1)
    error=float((probs(hr+delta)-probs(hr)).abs().max())
    part_error=float((probs(hr+partial)-probs(hr)).abs().max())
    if max(error,part_error)>1e-6:raise AssertionError("head protected readout changed")
    variants={"zero":torch.zeros_like(delta),"full_dark":delta,"physical_svd8_dark":partial}
    result={"protected_all_vocab_max_error":max(error,part_error),"rank_head":rank,"n":n,
            "physical_svd_basis_rank":B.shape[1],"depths":{}}
    for horizon in (4,8,12):
        donor_future=resume_after(model,donor,hd,ld,2,horizon)[:,:,:4]
        pairs={}
        for name,edit in variants.items():
            future=resume_after(model,recipient,hr+edit,lr,2,horizon)[:,:,:4]
            p=future.argmax(-1);y=donor_future.argmax(-1)
            pairs[name]={"donor_token_agreement":float((p==y).float().mean()),
                         "donor_board_agreement":float((p==y).all(-1).float().mean()),
                         "logit_mse":float((future-donor_future).square().mean()),
                         "edit_norm":float(edit.norm(dim=-1).mean())}
        result["depths"][str(horizon)]=pairs
    result["claim_boundary"]="state-level physical SVD steering engineering control; no fitted BPD sparse components or full signed-Jacobian null"
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",nargs="+",type=int,default=[901,902])
    ap.add_argument("--updates",type=int,default=64)
    ap.add_argument("--samples",type=int,default=64)
    ap.add_argument("--out",type=Path,required=True)
    x=ap.parse_args()
    if x.seeds!=[901,902]:raise ValueError("development seeds 901 and 902 only")
    torch.set_num_threads(2)
    result={"experiment":"BPD-DT02-A","evidence_class":"trained engineering on excluded seeds",
            "supervised_depths":list(FIT),"heldout_depths":list(TEST),
            "synthetic_arc_only":True,"official_arc1_evaluated":False,"full_task_jacobian_null_evaluated":False,
            "torch":torch.__version__,"updates":x.updates,"roots":[]}
    for seed in x.seeds:
        row={"seed":seed,"family":{}}
        for family in FAMILIES:
            model,loss=train(seed,family,updates=x.updates)
            metrics=evaluate(model,seed,family,n=x.samples)
            buff=BytesIO();torch.save(model.state_dict(),buff)
            entry={"final_loss":loss,"checkpoint_sha256":sha256(buff.getvalue()).hexdigest(),
                   "parameter_count":sum(p.numel() for p in model.parameters()),
                   "eval":metrics}
            if family=="sudoku4":entry["protected_steering"]=steering(model,seed)
            row["family"][family]=entry
        result["roots"].append(row)
    x.out.parent.mkdir(exist_ok=True,parents=True)
    x.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"complete","seed_roots":x.seeds,"updates":x.updates,
                      "samples":x.samples,"out":str(x.out)},sort_keys=True))

if __name__=="__main__":main()
