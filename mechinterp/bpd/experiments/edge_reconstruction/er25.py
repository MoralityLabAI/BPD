"""ER-25: survivor-conditioned exact refinement kernel selection."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from er23 import Bank, cost, EPS, decision

WIDTHS=(32,64,128)
CAL=(42,43,44)
TEST=(45,46,47)
CHOICES=('sort16','sort64','sort256','filter64')
NOISE=(.1,1.,3.)


def train(d,root):
    import torch
    from torch import nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rec=nn.Linear(d,d,bias=False)
            self.out=nn.Linear(d,d,bias=False)
        def forward(self,x):
            h=x
            for _ in range(4):
                h=torch.tanh(self.rec(h)+.25*x)
            return self.out(h)
    torch.set_num_threads(1)
    torch.manual_seed(261140+root+1000*d)
    teacher,model=Net(),Net()
    opt=torch.optim.Adam(model.parameters(),lr=.015)
    for _ in range(350):
        x=torch.randn(128,d)
        with torch.no_grad():
            y=teacher(x)
        loss=(model(x)-y).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w=model.rec.weight.detach().numpy().astype(np.float64)
    o=model.out.weight.detach().numpy().astype(np.float64)
    u,s,v=np.linalg.svd(w,full_matrices=False)
    return w,o,(u*s).T.copy(),v


def world(d,root):
    w,o,u,v=train(d,root)
    rng=np.random.default_rng(np.random.SeedSequence([261141,d,root]))
    x,c=rng.standard_normal((2,24,d))
    cal=Bank(w,o,u,v,x,c)
    h=x
    for _ in range(4):
        h=np.tanh(h@w.T+.25*x)
    clean=((h@o.T)*c).sum(-1)
    keys=rng.choice(d*d,min(d*d,256),replace=False)
    scale=max(1e-5,float(np.sqrt(np.mean((cal.exact(keys)-clean)**2))))
    for mult in NOISE:
        x,c=rng.standard_normal((2,12,d))
        k=int(rng.integers(d*d))
        i,j=divmod(k,d)
        h=x
        for t in range(4):
            a=w if t not in (1,2) else w-np.outer(u[i if t==1 else j],v[i if t==1 else j])
            h=np.tanh(h@a.T+.25*x)
        truth=((h@o.T)*c).sum(-1)
        assert np.allclose(truth,Bank(w,o,u,v,x,c).exact([k])[0],atol=1e-10,rtol=0)
        yield (w,o,u,v,x,c),truth+rng.normal(0,mult*scale,12),k,mult


def screen(bank,y,block=8):
    bank.setup()
    n=bank.d**2
    lo=np.empty(n)
    best_upper=np.inf
    incumbent=-1
    for i in range(0,bank.d,block):
        end=min(i+block,bank.d)
        pred,err=bank.bounds(i,end)
        delta=abs(pred-y)
        a,b=i*bank.d,end*bank.d
        lo[a:b]=(np.maximum(0,delta-err)**2).sum(-1)
        hi=((delta+err)**2).sum(-1)
        j=int(hi.argmin())
        if hi[j]<best_upper:
            best_upper,incumbent=hi[j],a+j
    score=float(cost(bank.exact([incumbent]),y)[0])
    assert lo[incumbent]-EPS<=score<=best_upper+EPS
    eligible=np.flatnonzero(lo<=score+EPS)
    return lo,eligible[eligible!=incumbent],incumbent,score


def finish(bank,y,lo,eligible,best,score,mode):
    if mode.startswith('sort'):
        batch=int(mode[4:])
        order=eligible[np.argsort(lo[eligible],kind='stable')]
    elif mode=='filter64':
        batch=64
        order=eligible
    else:
        raise ValueError(mode)
    calls=1
    for pos in range(0,len(order),batch):
        ids=order[pos:pos+batch]
        if mode.startswith('sort') and lo[ids[0]]>score+EPS:
            break
        ids=ids[lo[ids]<=score+EPS]
        if not len(ids):
            continue
        v=cost(bank.exact(ids),y)
        assert np.all(lo[ids]<=v+EPS)
        m=v.min()
        k=int(ids[v==m].min())
        if m<score or (m==score and k<best):
            best,score=k,float(m)
        calls+=len(ids)
    return best,score,calls


def choose(cal,d,fraction=None):
    """Nearest three calibration episodes by log survivor count; median finish time."""
    rows=[r for r in cal if r['width']==d]
    if fraction is not None:
        rows=sorted(rows,key=lambda r:abs(np.log1p(d*d*r['fraction'])-np.log1p(d*d*fraction)))[:3]
    return min(CHOICES,key=lambda m:float(np.median([r['arms'][m]['finish_ms'] for r in rows])))


def run_one(args,y,mode,cal=None):
    t=time.perf_counter()
    bank=Bank(*args)
    if mode=='eager':
        best,score,calls,_=decision(bank,y,'eager',batch=64)
        return dict(best=best,score=score,calls=calls,ms=(time.perf_counter()-t)*1000)
    lo,eligible,best,score=screen(bank,y)
    fraction=len(eligible)/bank.d**2
    s=time.perf_counter()
    selected=choose(cal,bank.d,fraction) if mode=='adaptive' else (choose(cal,bank.d) if mode=='static' else mode)
    best,score,calls=finish(bank,y,lo,eligible,best,score,selected)
    end=time.perf_counter()
    return dict(best=best,score=score,calls=calls,fraction=fraction,selected=selected,screen_ms=(s-t)*1000,finish_ms=(end-s)*1000,ms=(end-t)*1000)


def run(split,cal,width):
    for root in CAL if split=='cal' else TEST:
        for ep,(args,y,hidden,noise) in enumerate(world(width,root)):
            modes=list(CHOICES)+['eager']+(['adaptive','static'] if split=='test' else [])
            out={m:[] for m in modes}
            for rep in range(2):
                shift=(root+ep+rep)%len(modes)
                for m in modes[shift:]+modes[:shift]:
                    out[m].append(run_one(args,y,m,cal))
            ref=out['eager'][0]
            arms={}
            for m,rr in out.items():
                assert all(r['best']==ref['best'] and abs(r['score']-ref['score'])<1e-10 for r in rr)
                arms[m]=rr[0].copy()
                for k in ('ms','screen_ms','finish_ms'):
                    if k in rr[0]:
                        arms[m][k]=float(np.median([r[k] for r in rr]))
            print(json.dumps(dict(width=width,root=root,ep=ep,noise_multiplier=noise,hidden=hidden,hit=ref['best']==hidden,fraction=arms['sort64']['fraction'],arms=arms)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('split',choices=['cal','test'])
    p.add_argument('width',type=int,choices=WIDTHS)
    p.add_argument('--cal',type=Path)
    a=p.parse_args()
    cal=[json.loads(l) for l in a.cal.read_text().splitlines()] if a.cal else None
    if a.split=='test':
        assert cal is not None and {r['root'] for r in cal if r['width']==a.width}==set(CAL)
    run(a.split,cal,a.width)
