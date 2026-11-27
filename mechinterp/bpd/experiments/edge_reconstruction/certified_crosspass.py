"""ER-18: analytical tanh interaction envelopes and interval MAP refinement.

Bounds hold in real arithmetic. Float64 checks use a small numerical slack;
this is not an outward-rounded interval implementation.
"""
import json
import numpy as np
import torch
from neural_bridge import train, D, T

ROOTS = range(24, 32)
CAL, P, EP, B = 24, 48, 48, 12
BETA = 4/(3*np.sqrt(3))
EPS = 1e-10


def slopes(lo, hi):
    """Maximum |tanh'| on a coordinate interval."""
    nearest = np.maximum(np.maximum(lo, -hi), 0)
    return 1-np.tanh(nearest)**2


def curvature(lo, hi):
    """Maximum |tanh''| on a coordinate interval."""
    l = np.maximum(np.maximum(lo, -hi), 0)
    u = np.maximum(abs(lo), abs(hi))
    t = np.tanh(np.clip(np.arctanh(1/np.sqrt(3)), l, u))
    return 2*t*(1-t*t)


def mixed_bound(z, a, b, c_abs, local):
    """Bound |tanh(z+a+b+c)-tanh(z+a)-tanh(z+b)+tanh(z)|."""
    if not local:
        return c_abs + BETA*abs(a*b)
    lo = z + np.minimum(a,0) + np.minimum(b,0)
    hi = z + np.maximum(a,0) + np.maximum(b,0)
    center = z+a+b
    return (slopes(center-c_abs,center+c_abs)*c_abs
            + curvature(lo,hi)*abs(a*b))


class Model:
    def __init__(self, model):
        self.W = model.rec.weight.detach().numpy().astype(np.float64)
        self.O = model.out.weight.detach().numpy().astype(np.float64)
        u,s,vh = np.linalg.svd(self.W,full_matrices=False)
        self.M = np.einsum('ik,k,kj->kij',u,s,vh)

    def rollout(self,x,c,faults=()):
        h=x; states=[h]
        for t in range(T):
            w=self.W
            for tau,i in faults:
                if t==tau: w=w-self.M[i]
            h=np.tanh(h@w.T+.25*x); states.append(h)
        return (h@self.O.T*c).sum(axis=-1),states

    def pair(self,x,c,k):
        return self.rollout(x,c,((1,k//D),(2,k%D)))[0]

    def envelopes(self,x,c):
        clean,s=self.rollout(x,c)
        first=[self.rollout(x,c,((1,i),)) for i in range(D)]
        second=[self.rollout(x,c,((2,j),)) for j in range(D)]
        add=np.stack([a[0]+b[0]-clean for a in first for b in second])
        global_b=[]; local_b=[]; tangent=[]
        for i,(_,si) in enumerate(first):
            d=si[2]-s[2]
            a=d@self.W.T
            for j,(_,sj) in enumerate(second):
                b=-s[2]@self.M[j].T; cross=-d@self.M[j].T
                z=s[2]@self.W.T+.25*x
                z4=s[3]@self.W.T+.25*x
                a4=(si[3]-s[3])@self.W.T
                b4=(sj[3]-s[3])@self.W.T
                for local,store in ((False,global_b),(True,local_b)):
                    r3=mixed_bound(z,a,b,abs(cross),local)
                    r4=mixed_bound(z4,a4,b4,r3@abs(self.W).T,local)
                    store.append((abs(c@self.O)*r4).sum(axis=-1)+EPS)
                g3=((c@self.O)*(1-s[4]**2))@self.W
                g2=g3*(1-s[3]**2)
                tangent.append(np.sqrt(np.mean(np.sum(cross*g2,axis=-1)**2)))
        return add,np.array(global_b),np.array(local_b),np.array(tangent),clean


def refine(pred,err,y,exact_query):
    """Return exact MAP via lower bounds and a lazily evaluated incumbent.

    Exact-query function is called only after selecting a hypothesis using
    observable intervals. Uniform prior, common Gaussian sigma, so squared
    residual ranks MAP. Strict separation is required for a skipped candidate;
    ties among evaluated candidates follow np.argmin.
    """
    d=abs(pred-y)
    lower=(np.maximum(0,d-err)**2).sum(axis=1)
    upper=((d+err)**2).sum(axis=1)
    evaluated=np.zeros(len(pred),bool)
    values=np.full(len(pred),np.inf)
    calls=[]
    # An initial approximate incumbent is chosen by minimum upper energy.
    k=int(np.argmin(upper))
    while True:
        val=exact_query(k)
        energy=float(np.sum((val-y)**2))
        assert lower[k]-EPS <= energy <= upper[k]+EPS
        values[k]=energy; evaluated[k]=True; calls.append(k)
        best=int(np.argmin(values))
        unresolved=np.where(evaluated,np.inf,lower)
        if evaluated.all() or values[best]+EPS < unresolved.min():
            return best,calls
        k=int(np.argmin(unresolved))


def run():
    torch.set_num_threads(1); rows=[]; checked=0
    for root in ROOTS:
        trained,mse=train(root); model=Model(trained)
        torch.manual_seed(261134+root)
        x,c=[v.numpy().astype(np.float64) for v in (torch.randn(CAL+P,D),torch.randn(CAL+P,D))]
        _,_,_,rank,clean=model.envelopes(x[:CAL],c[:CAL])
        cal_exact=np.stack([model.pair(x[:CAL],c[:CAL],k) for k in range(D*D)])
        sigma=max(1e-5,.1*float(np.sqrt(np.mean((cal_exact-clean)**2))))
        x,c=x[CAL:],c[CAL:]
        add,glob,loc,_,_=model.envelopes(x,c)
        exact=np.stack([model.pair(x,c,k) for k in range(D*D)])
        err=abs(exact-add)
        assert np.all(err<=glob) and np.all(err<=loc)
        assert np.all(loc<=glob+EPS)
        checked+=2*err.size
        rng=np.random.default_rng(np.random.SeedSequence([261135,root]))
        random_order=rng.permutation(D*D)
        episodes=[]
        for episode in range(EP):
            h=int(rng.integers(D*D)); probes=rng.permutation(P)[:B]
            y=exact[h,probes]+rng.normal(0,sigma,B)
            truth=int(np.argmin(np.sum((exact[:,probes]-y)**2,axis=1)))
            r=dict(hidden=h,exact=truth,approx=int(np.argmin(np.sum((add[:,probes]-y)**2,axis=1))))
            for name,bound in [('global',glob),('local',loc)]:
                query=lambda k:model.pair(x[probes],c[probes],k)
                best,calls=refine(add[:,probes],bound[:,probes],y,query)
                assert best==truth
                r[name]=dict(calls=len(calls),pairs=calls,best=best)
            for name,order in [('random',random_order),('tangent',np.argsort(-rank,kind='stable'))]:
                pred=add[:,probes].copy(); ids=order[:16]; pred[ids]=exact[ids][:,probes]
                r[name]=int(np.argmin(np.sum((pred-y)**2,axis=1)))
            episodes.append(r)
        # Tolerance-based global cache selection from observable bounds (no joint labels).
        # If a row is omitted its error is at most tau*sigma on every available probe.
        tol={}
        for tau in (.5,1,2,4,8):
            ids=np.flatnonzero(loc.max(axis=1)>tau*sigma)
            omitted=np.setdiff1d(np.arange(D*D),ids)
            assert np.all(err[omitted]<=tau*sigma+EPS)
            tol[str(tau)]=dict(cached=len(ids),max_omitted_error_sigma=float(err[omitted].max()/sigma) if len(omitted) else 0.)
        rows.append(dict(root=root,sigma=sigma,teacher_mse=mse,mean_abs_error=float(err.mean()),mean_global_bound=float(glob.mean()),mean_local_bound=float(loc.mean()),max_error_over_local=float(np.max(err/loc)),tol=tol,episodes=episodes))
    es=[e for r in rows for e in r['episodes']]
    stats={}
    for name in ('global','local'):
        ns=np.array([e[name]['calls'] for e in es]); stats[name]=dict(mean=float(ns.mean()),median=float(np.median(ns)),p90=float(np.quantile(ns,.9)),max=int(ns.max()),at_most16=int((ns<=16).sum()),full=int((ns==D*D).sum()),matches=sum(e[name]['best']==e['exact'] for e in es))
    return dict(roots=rows,bound_checks=checked,episodes=len(es),refinement=stats,correct={name:sum(e[name]==e['hidden'] for e in es) for name in ('exact','approx','random','tangent')},oracle_agreement={name:sum(e[name]==e['exact'] for e in es) for name in ('approx','random','tangent')},tol_summary={str(tau):[r['tol'][str(tau)]['cached'] for r in rows] for tau in (.5,1,2,4,8)})

if __name__=='__main__': print(json.dumps(run(),indent=2))
