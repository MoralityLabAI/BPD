"""BPD-NBR-02: nonlinear native gate development pilot, fixed k=2.

All scores use current reached state and cue (first visit uses only
independent calibration future-cue probes). Source truth is scoring only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import torch
from mechinterp.bpd.theory.cq06_nonlinear_feedback import experiment as E

ROOTS=tuple(range(202611250201,202611250209))
EXCLUDED=202611250200
K=2
TOL=1e-5
PATTERNS=E.PATTERNS[E.PATTERNS.sum(1)==K]
SOURCE_SHA="6f9e03268ea5c3d8a706ee547b8e76da296e86e7"
FIRST_ARMS=("physical","current_diagonal","delayed_diagonal","delayed_greedy","delayed_joint","random")
SECOND_ARMS=("physical","observable_diagonal","observable_greedy","observable_joint","learned","random","finite_oracle")

def git_blob(p):
    b=Path(p).read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()

def check_source():
    if git_blob(E.__file__)!=SOURCE_SHA:
        raise RuntimeError("CQ06 source drift")

def finite(*xs):
    if not all(torch.isfinite(x).all() for x in xs):
        raise ArithmeticError("nonfinite tensor")

def digest(*modules):
    h=hashlib.sha256()
    for m in modules:
        for k,v in sorted(m.state_dict().items()):
            h.update(k.encode()+b"\0")
            h.update(str(v.dtype).encode()+str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def topk(score):
    ix=torch.argsort(score,dim=1,descending=True,stable=True)[:,:K]
    return torch.zeros_like(score).scatter(1,ix,1.)

def random_mask(seed,n,paired=False):
    assert not paired or n%2==0
    g=torch.Generator().manual_seed(seed)
    m=topk(torch.rand(n//2 if paired else n,E.COMPONENTS,generator=g))
    return m.repeat_interleave(2,0) if paired else m

def risk(v,m):
    return ((v*(1.-m)[:,:,None]).sum(1)).square().sum(1)

def diagonal(v):
    return topk(v.square().sum(-1))

def greedy(v):
    m=torch.zeros(v.shape[:2],dtype=v.dtype)
    for _ in range(K):
        residual=(v*(1.-m)[:,:,None]).sum(1)
        costs=(residual[:,None,:]-v).square().sum(-1)
        ix=costs.masked_fill(m.bool(),float("inf")).argmin(1,keepdim=True)
        m.scatter_(1,ix,1.)
    return m

def joint(v):
    costs=torch.einsum("mk,nkh->nmh",1.-PATTERNS.to(v.dtype),v).square().sum(-1)
    return PATTERNS[costs.argmin(1)].to(v.dtype)

def local(t,f,state,cue):
    a=torch.cat((state,cue),1)@f.V
    w=a[:,:,None]*f.U[None,:,:]
    pre=a@f.U+t.bias
    s=torch.tanh(pre)
    g=t.head.weight[0]*(1-s.square())
    v=torch.einsum("nkd,nd->nk",w,g)
    finite(w,s,v)
    return w,pre,s,v

def first_signatures(t,f,u0,probes):
    w,p,s,v=local(t,f,torch.zeros(len(u0),E.D),u0)
    current=v[:,:,None]
    W=f.weight()[:E.D,:]
    future=[]
    for q in probes:
        s2=torch.tanh(torch.cat((s,q.expand(len(s),-1)),1)@f.V@f.U+t.bias)
        g2=t.head.weight[0]*(1-s2.square())
        g1=(g2@W.T)*(1-s.square())
        future.append(torch.einsum("nkd,nd->nk",w,g1))
    delayed=torch.cat((current,torch.stack(future,-1)/(len(probes)**.5)),dim=-1)
    return w,p,s,current,delayed

def check_mask(m,n):
    if m.shape!=(n,E.COMPONENTS) or not ((m==0)|(m==1)).all() or not (m.sum(1)==K).all():
        raise ArithmeticError("mask cardinality")

class Native:
    """Every scored arm executes real BPD; independent tensor formula checks parity."""
    def __init__(self,t,f):
        self.t=t;self.f=f;self.model=E.NativeModel(t,f);self.max_error=0.
    def step(self,s,u,m=None,tau=0):
        a=E.step(self.t,self.f,s,u,m,backend="native",native=self.model,tau=tau)
        b=E.independent_step(self.t,self.f,s,u,m)
        finite(a,b)
        err=max(float((a-b).abs().max()),float((E.logits(self.t,a)-E.logits(self.t,b)).abs().max()))
        self.max_error=max(err,self.max_error)
        if err>TOL:
            raise ArithmeticError(f"native parity {err}")
        return a
    def logit(self,s):
        return E.logits(self.t,s)
    def pair(self,u0,u1,m1=None,m2=None):
        s1=self.step(torch.zeros(len(u0),E.D),u0,m1,0)
        s2=self.step(s1,u1,m2,1)
        return torch.stack((self.logit(s1),self.logit(s2)),1),s1

def second_masks(t,f,gate,s,cue,rand,native,finite_oracle=True):
    w,pre,full,v=local(t,f,s,cue)
    v=v[:,:,None]
    masks={"physical":diagonal(w),"observable_diagonal":diagonal(v),
           "observable_greedy":greedy(v),"observable_joint":joint(v),
           "learned":topk(gate.scores(s,cue)),"random":rand}
    if finite_oracle:
        # Exhaustive finite-output oracle pays six actual native masked
        # continuations and a full reference; no label or future truth.
        ls=torch.stack([native.logit(native.step(s,cue,m.expand(len(s),-1),1))
                        for m in PATTERNS],1)
        full_l=native.logit(native.step(s,cue,tau=1))
        masks["finite_oracle"]=PATTERNS[(ls-full_l[:,None]).square().argmin(1)]
    return masks,v,w

def packed(mask,logits,proxy,physical):
    return {"mask":mask.int().tolist(),"logits":logits.tolist(),
            "proxy_risk":proxy.tolist(),"physical_risk":physical.tolist()}

def task(seed,t,f,gate,native):
    u0,u1,y=E.source(seed,256,810000,paired_z=True)
    _,probes,_=E.source(seed,8,922000,paired_z=True)
    full,s= native.pair(u0,u1)
    teacher=t(u0,u1)[0]
    w,p,s_local,current,v=first_signatures(t,f,u0,probes)
    if (s-s_local).abs().max()>TOL: raise ArithmeticError("full first parity")
    masks={"physical":diagonal(w),"current_diagonal":diagonal(current),
           "delayed_diagonal":diagonal(v),"delayed_greedy":greedy(v),
           "delayed_joint":joint(v),"random":random_mask(seed+923000,len(y),True)}
    first={}
    for name,m in masks.items():
        check_mask(m,len(y))
        if not torch.equal(m[::2],m[1::2]): raise ArithmeticError("future cue leakage")
        logits,_=native.pair(u0,u1,m1=m)
        first[name]=packed(m,logits,risk(v,m),risk(w,m))
    first_signature={"physical":w.tolist(),"current":current.tolist(),"delayed":v.tolist()}
    masks,v,w=second_masks(t,f,gate,s,u1,random_mask(seed+923001,len(y)),native)
    second={}
    for name,m in masks.items():
        check_mask(m,len(y))
        s2=native.step(s,u1,m,1)
        logits=torch.stack((full[:,0],native.logit(s2)),1)
        second[name]=packed(m,logits,risk(v,m),risk(w,m))
    check_oracle(second,full[:,1])
    return {"targets":y.tolist(),"full":full.tolist(),"teacher":teacher.tolist(),
            "first":first,"second":second,"probes":probes.tolist(),
            "first_signature":first_signature,"second_signature":{"physical":w.tolist(),"observable":v.tolist()}}

def check_oracle(arms,full):
    oracle=(torch.tensor(arms["finite_oracle"]["logits"])[:,-1]-full).square()
    for name,r in arms.items():
        error=(torch.tensor(r["logits"])[:,-1]-full).square()
        if ((oracle-error)>TOL).any():
            raise ArithmeticError(f"finite oracle dominated by {name}")

def donor(seed,t,f,gate,native):
    ur,ud,u1,yr,yd=E.donor_inputs(seed,128)
    sr=native.step(torch.zeros(len(ur),E.D),ur,tau=0)
    sd=native.step(torch.zeros(len(ud),E.D),ud,tau=0)
    ar=torch.cat((torch.zeros(len(ur),E.D),ur),1)@f.V
    ad=torch.cat((torch.zeros(len(ud),E.D),ud),1)@f.V
    partial=((ad-ar)*torch.tensor([1.,1.,0.,0.]))@f.U
    raw=torch.tanh(ar@f.U+t.bias+partial)-sr
    dose=raw.norm(dim=1)
    axis=t.head.weight[0]/t.head.weight[0].norm()
    def dark(q): return q-(q@axis)[:,None]*axis
    def match(q):
        q=dark(q)
        norm=q.norm(dim=1)
        valid=(norm>1e-8)&(dose>1e-8)
        out=torch.where(valid[:,None],q/norm.clamp_min(1e-8)[:,None]*dose[:,None],torch.zeros_like(q))
        return out,valid
    p,valid=match(raw)
    random=torch.randn(sr.shape,generator=torch.Generator().manual_seed(seed+923002))
    q,qvalid=match(random)
    rand=random_mask(seed+923003,len(ur))
    frozen,_,_=second_masks(t,f,gate,sr,u1,rand,native)
    results={}
    for name,reached,ok in (("partial_dark",sr+p,valid),("random_dark",sr+q,qvalid),
                            ("sham",sr,torch.ones_like(valid)),("full_donor",sd,torch.ones_like(valid))):
        delta=reached-sr
        full=native.logit(native.step(reached,u1,tau=1))
        current=native.logit(reached)
        dyn,v,w=second_masks(t,f,gate,reached,u1,rand,native)
        out={}
        for policy in SECOND_ARMS:
            entry={}
            for mode,m in (("feedback",dyn[policy]),("frozen",frozen[policy])):
                check_mask(m,len(ur))
                pred=native.logit(native.step(reached,u1,m,1))
                entry[mode]=packed(m,pred[:,None],risk(v,m),risk(w,m))
            entry["switches"]=int((dyn[policy]!=frozen[policy]).any(1).sum())
            if name=="sham":
                if not torch.equal(dyn[policy],frozen[policy]) or entry["feedback"]["logits"]!=entry["frozen"]["logits"]:
                    raise ArithmeticError("sham mismatch")
            out[policy]=entry
        check_oracle({a:r["feedback"] for a,r in out.items()},full)
        norm=delta.norm(dim=1);darkerr=(delta@axis).abs()
        if name in ("partial_dark","random_dark") and ok.any():
            if (norm[ok]-dose[ok]).abs().max()>TOL or darkerr[ok].max()>TOL:
                raise ArithmeticError("matched dark patch parity")
        results[name]={"full":full.tolist(),"current":current.tolist(),
                       "dose":norm.tolist(),"dark_error":darkerr.tolist(),
                       "valid":ok.tolist(),"outside":(reached.abs()>1.00001).any(1).tolist(),
                       "signature":{"physical":w.tolist(),"observable":v.tolist()},"arms":out}
    return {"recipient_targets":yr.tolist(),"donor_targets":yd.tolist(),
            "clean_current":native.logit(sr).tolist(),"raw_dose":dose.tolist(),
            "patches":results}

def self_test():
    check_source();E.configure();torch.manual_seed(EXCLUDED)
    t=E.Teacher(EXCLUDED);f=E.Factors(EXCLUDED)
    u0,u1,_=E.source(EXCLUDED,8,922000,paired_z=True)
    with torch.no_grad():
        native=Native(t,f)
        for m in E.PATTERNS:
            native.pair(u0,u1,m.expand(len(u0),-1),m.expand(len(u0),-1))
        w,p,s,c,v=first_signatures(t,f,u0,u1)
        for selector in (diagonal,greedy,joint):check_mask(selector(v),len(u0))
        assert (risk(v,joint(v))<=risk(v,greedy(v))+1e-7).all()
        assert (risk(v,joint(v))<=risk(v,diagonal(v))+1e-7).all()
    # Independent autograd gradients of first preactivation and delayed readout.
    p=p.detach().requires_grad_()
    s=torch.tanh(p)
    first=E.logits(t,s)
    g=torch.autograd.grad(first.sum(),p,retain_graph=True)[0]
    torch.testing.assert_close(torch.einsum("nkd,nd->nk",w,g),c[:,:,0],rtol=1e-5,atol=1e-6)
    for j,q in enumerate(u1):
        future=E.logits(t,torch.tanh(torch.cat((s,q.expand(len(s),-1)),1)@f.V@f.U+t.bias))
        g=torch.autograd.grad(future.sum(),p,retain_graph=True)[0]
        torch.testing.assert_close(torch.einsum("nkd,nd->nk",w,g),v[:,:,j+1]*(len(u1)**.5),rtol=1e-5,atol=1e-6)
    # Terminal derivative: compare every component signature to autograd.
    z=torch.zeros(len(u0),E.D)
    with torch.no_grad():w,p,_,v=local(t,f,z,u0)
    p=p.detach().requires_grad_()
    g=torch.autograd.grad(E.logits(t,torch.tanh(p)).sum(),p)[0]
    torch.testing.assert_close(torch.einsum("nkd,nd->nk",w,g),v,rtol=1e-5,atol=1e-6)
    # Cancellation witness: magnitude selects coordinate 0, joint selects 1
    # when k=1; for k=2 compare exhaustive costs on four coordinates.
    vec=torch.tensor([[[6.],[5.],[-6.],[1.]]])
    assert float(risk(vec,joint(vec)))==0.
    assert float(risk(vec,diagonal(vec)))>0.
    print("BPD-NBR-02 derivative/native-mask engineering tests PASS")

def run(out):
    check_source();E.configure()
    out.mkdir(parents=True,exist_ok=False)
    for seed in ROOTS:
        torch.manual_seed(seed)
        t=E.train_teacher(seed);f=E.train_factors(seed,t)
        fit=float((f.weight().detach()-t.weight().detach()).abs().max())
        geo=E.canonicalize(f);gate,planner,_=E.train_policies(seed,t,f)
        t.eval();f.eval();gate.eval()
        with torch.no_grad():
            native=Native(t,f)
            tasks=task(seed,t,f,gate,native)
            donors=donor(seed,t,f,gate,native)
        rec={"schema":"BPD/NBR02/root-v1","seed":seed,"engineering_only":True,
             "source_blob":SOURCE_SHA,"torch":torch.__version__,"python":platform.python_version(),
             "platform":platform.platform(),"threads":torch.get_num_threads(),
             "fit_max_weight_error":fit,"geometry":geo,"parameter_sha256":digest(t,f,gate),
             "native_max_abs_error":native.max_error,"task":tasks,"donor":donors,
             "access":{"first":1,"future_probe_full_forwards":8,
                       "second":1,"finite_oracle_candidate_forwards":6,
                       "cardinality":2,"compute_savings_claimed":False}}
        p=out/f"{seed}.json"
        p.write_text(json.dumps(rec,allow_nan=False,separators=(",",":"))+"\n")
        print("BPD-NBR-02 root",seed,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true");p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.self_test:self_test()
    elif a.out:run(a.out)
    else:p.error("supply --self-test or --out")

if __name__=="__main__":main()
