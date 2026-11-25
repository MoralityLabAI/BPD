"""BPD-NBR-03: finite nonlinear mask-risk distillation; development only."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from mechinterp.bpd.experiments.nbr02_cq06 import run as R

E=R.E
ROOTS=tuple(range(202611250301,202611250309))
EXCLUDED=202611250300
PAT=R.PATTERNS
TOL=1e-5
FIRST=("physical","local_joint","path_joint","finite_probe","distilled","random")
SECOND=("physical","local_joint","path_joint","finite_oracle","distilled","learned","random")

def hash_tensors(*xs):
    h=hashlib.sha256()
    for x in xs:
        v=x.detach().cpu().contiguous()
        h.update(str(v.dtype).encode()+str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()

def mask_index(m):
    eq=(m[:,None,:]==PAT[None,:,:]).all(-1)
    assert eq.sum(1).eq(1).all()
    return eq.int().argmax(1)

def secant(b,p):
    diff=p-b
    # stable derivative at zero; avoid division by zero in either branch
    safe=torch.where(diff.abs()>1e-6,diff,torch.ones_like(diff))
    return torch.where(diff.abs()>1e-6,(p.tanh()-b.tanh())/safe,1-b.tanh().square())

def path_second(t,f,s,cue):
    w,p,_,_=R.local(t,f,s,cue)
    g=t.head.weight[0]*secant(t.bias,p)
    return torch.einsum("nkd,nd->nk",w,g)[:,:,None]

def path_first(t,f,u0,probes):
    w,p,_,_=R.local(t,f,torch.zeros(len(u0),E.D),u0)
    W=f.weight()[:E.D,:]
    avg=None
    for frac,weight in ((0.,1/6),(.5,4/6),(1.,1/6)):
        s=torch.tanh(t.bias+frac*(p-t.bias))
        deriv=1-s.square()
        current=torch.einsum("nkd,nd->nk",w,t.head.weight[0]*deriv)[:,:,None]
        future=[]
        for q in probes:
            s2=torch.tanh(torch.cat((s,q.expand(len(s),-1)),1)@f.V@f.U+t.bias)
            g2=t.head.weight[0]*(1-s2.square())
            future.append(torch.einsum("nkd,nd->nk",w,(g2@W.T)*deriv))
        sig=torch.cat((current,torch.stack(future,-1)/(len(probes)**.5)),dim=-1)
        avg=weight*sig if avg is None else avg+weight*sig
    return avg

def first_risks(native,u0,probes):
    z=torch.zeros(len(u0),E.D)
    s=native.step(z,u0,tau=0)
    def signature(s):
        return torch.stack([native.logit(s)]+[
            native.logit(native.step(s,q.expand(len(s),-1),tau=1))
            for q in probes],1)
    reference=signature(s)
    risks=[]
    for m in PAT:
        cand=signature(native.step(z,u0,m.expand(len(u0),-1),0))
        err=(cand-reference).square()
        risks.append(err[:,0]+err[:,1:].mean(1))
    return torch.stack(risks,1),s

def second_risks(native,s,cue):
    full=native.logit(native.step(s,cue,tau=1))
    risks=torch.stack([(native.logit(native.step(s,cue,m.expand(len(s),-1),1))-full).square()
                       for m in PAT],1)
    return risks,full

def policy(seed,inputs,risks):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model=nn.Sequential(nn.Linear(inputs.shape[1],48),nn.Tanh(),
                            nn.Linear(48,48),nn.Tanh(),nn.Linear(48,len(PAT)))
    gen=torch.Generator().manual_seed(seed+1)
    lo=risks.min(1,keepdim=True).values
    norm=(risks-lo)/(risks.max(1,keepdim=True).values-lo).clamp_min(1e-8)
    gold=risks.argmin(1)
    opt=torch.optim.Adam(model.parameters(),lr=.009)
    for _ in range(600):
        ix=torch.randint(len(inputs),(256,),generator=gen)
        logits=model(inputs[ix])
        loss=(logits.softmax(1)*norm[ix]).sum(1).mean()+.2*F.cross_entropy(logits,gold[ix])
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        pred=model(inputs).argmax(1)
        reg=risks.gather(1,pred[:,None])[:,0]-lo[:,0]
    diag={"calibration_sha256":hash_tensors(inputs,risks),
          "parameter_sha256":R.digest(model),"parameters":sum(p.numel() for p in model.parameters()),
          "calibration_choice_agreement":float((pred==gold).float().mean()),
          "calibration_mean_regret":float(reg.mean())}
    return model.eval(),diag

def first_masks(t,f,u0,probes,model,native,seed):
    w,_,_,_,v=R.first_signatures(t,f,u0,probes)
    p=path_first(t,f,u0,probes)
    risks,_=first_risks(native,u0,probes)
    return {"physical":R.diagonal(w),"local_joint":R.joint(v),"path_joint":R.joint(p),
            "finite_probe":PAT[risks.argmin(1)],"distilled":PAT[model(u0).argmax(1)],
            "random":R.random_mask(seed,len(u0),True)},risks

def second_masks(t,f,s,cue,model,gate,native,rand):
    w,_,_,v=R.local(t,f,s,cue)
    p=path_second(t,f,s,cue)
    risks,full=second_risks(native,s,cue)
    return {"physical":R.diagonal(w),"local_joint":R.joint(v[:,:,None]),"path_joint":R.joint(p),
            "finite_oracle":PAT[risks.argmin(1)],
            "distilled":PAT[model(torch.cat((s,cue),1)).argmax(1)],
            "learned":R.topk(gate.scores(s,cue)),"random":rand},risks,full

def record(m,y):
    R.check_mask(m,len(y));R.finite(y)
    return {"mask":m.int().tolist(),"logits":y.tolist()}

@torch.no_grad()
def task(seed,t,f,gate,p1,p2,native,probes):
    u0,u1,y=E.source(seed,256,810000,paired_z=True)
    full,s=native.pair(u0,u1)
    teacher=t(u0,u1)[0]
    masks,risks=first_masks(t,f,u0,probes,p1,native,seed+933000)
    first={}
    for name,m in masks.items():
        if not torch.equal(m[::2],m[1::2]):raise ArithmeticError("first cue leakage")
        logits,_=native.pair(u0,u1,m1=m)
        first[name]=record(m,logits)
    masks,r2,_=second_masks(t,f,s,u1,p2,gate,native,R.random_mask(seed+933001,len(y)))
    second={}
    for name,m in masks.items():
        out=native.logit(native.step(s,u1,m,1))
        second[name]=record(m,torch.stack((full[:,0],out),1))
    return {"targets":y.tolist(),"teacher":teacher.tolist(),"full":full.tolist(),
            "first_risks":risks.tolist(),"second_risks":r2.tolist(),"first":first,"second":second}

@torch.no_grad()
def donor(seed,t,f,gate,p2,native):
    ur,ud,u1,yr,yd=E.donor_inputs(seed,128)
    z=torch.zeros(len(ur),E.D)
    sr=native.step(z,ur,tau=0);sd=native.step(z,ud,tau=0)
    rand=R.random_mask(seed+933002,len(ur))
    frozen,_,clean=second_masks(t,f,sr,u1,p2,gate,native,rand)
    masks,risks,full=second_masks(t,f,sd,u1,p2,gate,native,rand)
    out={}
    for name in SECOND:
        out[name]={mode:record(m,native.logit(native.step(sd,u1,m,1))[:,None])
                   for mode,m in (("feedback",masks[name]),("frozen",frozen[name]))}
    return {"recipient_targets":yr.tolist(),"donor_targets":yd.tolist(),
            "clean_full":clean.tolist(),"donor_full":full.tolist(),
            "outside":(sd.abs()>1.00001).any(1).tolist(),"risks":risks.tolist(),"arms":out}

def self_test():
    R.check_source();E.configure();torch.manual_seed(EXCLUDED)
    # Saturation: local derivative dramatically underpredicts a large finite deletion.
    p=torch.tensor([8.]);b=torch.tensor([0.])
    linear=(1-p.tanh().square())*p
    finite=p.tanh()-b.tanh()
    assert linear.abs().item()<.001 and finite.item()>.99
    torch.testing.assert_close(secant(b,p)*p,finite)
    t=E.Teacher(EXCLUDED);f=E.Factors(EXCLUDED)
    with torch.no_grad():
        n=R.Native(t,f);u0,u1,_=E.source(EXCLUDED,8,932000,paired_z=True)
        s1=n.step(torch.zeros(len(u0),E.D),u0,tau=0)
        w,p,s,v=R.local(t,f,s1,u1)
        observed=E.logits(t,s)-E.logits(t,t.bias.tanh().expand_as(s))
        torch.testing.assert_close(path_second(t,f,s1,u1).sum(1)[:,0],observed,atol=2e-6,rtol=2e-5)
        rs,_=first_risks(n,u0,u1);rt,_=second_risks(n,s1,u1)
        assert rs.shape==rt.shape==(8,6)
        assert torch.equal(rs[::2],rs[1::2])
        R.check_mask(R.joint(path_first(t,f,u0,u1)),len(u0))
    print("BPD-NBR-03 saturation and native parity tests PASS")

def run(out):
    R.check_source();E.configure();out.mkdir(parents=True,exist_ok=False)
    for seed in ROOTS:
        torch.manual_seed(seed)
        t=E.train_teacher(seed);f=E.train_factors(seed,t)
        geom=E.canonicalize(f);gate,planner,_=E.train_policies(seed,t,f)
        native=R.Native(t,f)
        for m in (t,f,gate):m.eval().requires_grad_(False)
        with torch.no_grad():
            u0,u1,_=E.source(seed,2048,931000,paired_z=True)
            _,probes,_=E.source(seed,8,932000,paired_z=True)
            r1,s=first_risks(native,u0,probes);r2,_=second_risks(native,s,u1)
            x2=torch.cat((s,u1),1)
        p1,d1=policy(seed+934000,u0,r1)
        p2,d2=policy(seed+935000,x2,r2)
        a=task(seed,t,f,gate,p1,p2,native,probes)
        b=donor(seed,t,f,gate,p2,native)
        r={"schema":"BPD/NBR03/root-v1","development_only":True,"seed":seed,
           "source_blob":R.SOURCE_SHA,"model_sha256":R.digest(t,f,gate),
           "torch":torch.__version__,"python":platform.python_version(),"geometry":geom,
           "native_max_error":native.max_error,"probes":probes.tolist(),
           "distill":{"first":d1,"second":d2},"task":a,"donor":b,
           "access":{"first_probe_oracle_current_forwards":7,"first_probe_oracle_delayed_forwards":56,
                     "first_path_gradient_points":3,"first_path_future_probes":24,
                     "second_finite_oracle_forwards":7,
                     "distilled_selection_model_forwards":0,"distilled_selection_policy_forwards":1,
                     "scoring_reference_forwards_excluded_from_policy_cost":True}}
        p=out/f"{seed}.json"
        p.write_text(json.dumps(r,allow_nan=False,separators=(",",":"))+"\n")
        print("BPD-NBR-03 root",seed,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true");p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.self_test:self_test()
    elif a.out:run(a.out)
    else:p.error("supply --self-test or --out")
if __name__=="__main__":main()
