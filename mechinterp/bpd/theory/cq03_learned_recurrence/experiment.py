"""BPD-CQ-03: trained physical recurrence and counterfactual hidden-variable transfer.

A standalone backend makes the original protocol reproducible on CPU without
importing BPD. The optional native backend uses the repository's actual
BPDModel / TiedComponentLinear masked execution and checks numerical parity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ENGINEERING_SEED=997
SEEDS=(607,613,617,619,631,641,643,647)
ROOT=Path(__file__).resolve().parent


def draw(seed: int, n: int):
    gen=torch.Generator().manual_seed(seed)
    ab=(torch.rand((n,2),generator=gen)*1.8-.9)
    h=torch.zeros(n,4)
    h[:,2:]=ab
    return h,ab


class Teacher(nn.Module):
    """Rank-two shared transition, trained from the two visible behavior targets."""
    def __init__(self,seed):
        super().__init__()
        g=torch.Generator().manual_seed(seed+902000)
        self.in_factor=nn.Parameter(torch.randn(4,2,generator=g)*.45)
        self.out_factor=nn.Parameter(torch.randn(2,4,generator=g)*.45)
    def weight(self):return (self.in_factor@self.out_factor).T
    def forward(self,h):return (h@self.in_factor)@self.out_factor


def train_teacher(seed,steps=520):
    teacher=Teacher(seed);opt=torch.optim.Adam(teacher.parameters(),lr=.035)
    g=torch.Generator().manual_seed(seed+390000)
    for j in range(steps):
        ab=torch.rand((96,2),generator=g)*1.8-.9
        h=torch.zeros(96,4);h[:,2:]=ab
        one=teacher(h);two=teacher(one)
        loss=(one[:,0]-ab[:,0]).square().mean()+(two[:,0]-ab[:,1]).square().mean()
        loss=loss+.0002*teacher.weight().square().mean()
        if not torch.isfinite(loss):raise ValueError('teacher loss nonfinite')
        opt.zero_grad();loss.backward();opt.step()
    return teacher


class Factors(nn.Module):
    """One shared rank-two V,U decomposition of the learned physical T."""
    def __init__(self, seed, dim=4):
        super().__init__();g=torch.Generator().manual_seed(seed+981000)
        self.V=nn.Parameter(torch.randn(dim,2,generator=g)*.35)
        self.U=nn.Parameter(torch.randn(2,dim,generator=g)*.35)
    def forward(self,h):return (h@self.V)@self.U
    def weight(self):return (self.V@self.U).T


class NativeFactors(Factors):
    """Exact BPD wrapper backend, parameters copied from standalone factors."""
    def __init__(self,seed,weight):
        nn.Module.__init__(self)
        from mechinterp.bpd.components import BPDModel
        from mechinterp.bpd.config import AblationConfig
        class Base(nn.Module):
            def __init__(self,w):
                super().__init__();self.core=nn.Linear(4,4,bias=False)
                with torch.no_grad():self.core.weight.copy_(w)
            def forward(self,h):return self.core(h)
        self.bpd=BPDModel(Base(weight.detach()),{'core':2},max_passes=2,
          ablation=AblationConfig(mode='zero',sphere_rescale=False),init_svd=False)
        init=Factors(seed)
        with torch.no_grad():
            self.bpd.wrappers['core'].V.copy_(init.V)
            self.bpd.wrappers['core'].U.copy_(init.U)
    @property
    def V(self):return self.bpd.wrappers['core'].V
    @property
    def U(self):return self.bpd.wrappers['core'].U
    def forward(self,h):
        # One invocation, only within a two-visit schedule context.
        return self.bpd.target_model(h.unsqueeze(1))[:,0]
    def rollout(self,h):
        from mechinterp.bpd.components import PassMaskSchedule
        b=len(h); mask=torch.ones((b,2,1,2));delta=torch.zeros((b,2,1))
        with self.bpd.masked_components({'core':PassMaskSchedule(mask,delta)}):
            one=self.forward(h);two=self.forward(one)
        return one,two
    def weight(self):return (self.V@self.U).T


def rollout(factors,h):
    if isinstance(factors,NativeFactors):return factors.rollout(h)
    one=factors(h);two=factors(one);return one,two


def train_factors(seed,teacher,backend='independent',steps=360):
    f=(NativeFactors(seed,teacher.weight()) if backend=='native' else Factors(seed))
    opt=torch.optim.Adam([f.V,f.U],lr=.035)
    initial=[f.V.detach().clone(),f.U.detach().clone()]
    target=teacher.weight().detach()
    g=torch.Generator().manual_seed(seed+680000)
    for step in range(steps):
        ab=torch.rand((96,2),generator=g)*1.8-.9
        h=torch.zeros(96,4);h[:,2:]=ab
        # Independent factor training receives only a learned weight source,
        # trajectories, and teacher states. It never sees heldout donor pairs.
        with torch.no_grad():ref1=teacher(h);ref2=teacher(ref1)
        one,two=rollout(f,h)
        loss=((one-ref1).square().mean()+(two-ref2).square().mean()
              +0.5*(f.weight()-target).square().mean())
        if not torch.isfinite(loss):raise ValueError('factor loss nonfinite')
        opt.zero_grad();loss.backward();opt.step()
    f.training_factor_update_norms=[float((p-old).norm()) for p,old in zip((f.V,f.U),initial)]
    return f


def calibration(seed,f):
    # Calibration chooses one component whose first-pass write carries the
    # largest b-derivative into the SECOND visible readout. No donor/test gold.
    h,ab=draw(seed+510000,128)
    w=f.weight().detach()
    scores=[]
    with torch.no_grad():
        for c in range(2):
            read=f.V[3,c];write=f.U[c]
            score=float((write@w.T[:,0]).abs()*read.abs())
            scores.append(score)
    return int(max(range(2),key=lambda c:(scores[c],-c))),scores


def assay(seed,teacher,f,pairs=128,cutoff=.12):
    # Input roles share a and vary b. Behavioral target is donor b, known
    # from task generation before any teacher or factor was fitted.
    gr=torch.Generator().manual_seed(seed+710000)
    a=torch.rand(pairs,generator=gr)*1.6-.8
    br=torch.rand(pairs,generator=gr)*.8-.4
    sign=torch.where(torch.rand(pairs,generator=gr)<.5,-1.,1.)
    bd=br+sign*(.30+torch.rand(pairs,generator=gr)*.15)
    hr=torch.zeros(pairs,4);hd=hr.clone();hr[:,2]=a;hd[:,2]=a;hr[:,3]=br;hd[:,3]=bd
    chosen,scores=calibration(seed,f)
    w=f.weight().detach()
    with torch.no_grad():
        x1,x2=rollout(f,hr);d1,d2=rollout(f,hd)
        teacher_one,teacher_two=teacher(hr),teacher(teacher(hr))
        clean=(x1[:,0]-a).abs()<=cutoff
        clean &= (d1[:,0]-a).abs()<=cutoff
        clean &= (x2[:,0]-br).abs()<=cutoff
        clean &= (d2[:,0]-bd).abs()<=cutoff
        def component(c):
            return ((hd-hr)@f.V[:,c:c+1])@f.U[c:c+1]
        delta=component(chosen)
        norms=delta.norm(dim=-1,keepdim=True)
        elig=clean & (norms.squeeze(-1)>1e-7)
        # Native write difference has an exact matched norm in every control.
        native=delta.clone()
        dark=native.clone();dark[:,0]=0.
        other=component(1-chosen)
        def match(x):
            norm=x.norm(dim=-1,keepdim=True)
            stable=torch.zeros_like(x);stable[:,1]=1.
            return torch.where(norm>1e-8,x/norm.clamp_min(1e-8),stable)*norms
        dark=match(dark)
        other=match(other)
        random=torch.randn((pairs,4),generator=gr);random[:,0]=0.;random=match(random)
        visible=torch.zeros_like(native);visible[:,0]=norms.squeeze(-1)
        zero=torch.zeros_like(native)
        # Exact same patch coordinates for all arms, applied after first pass.
        # Subsequent transition is the real LEARNED recurrent operator W_hat.
        def measure(v):
            state=x1+v
            return (state@w.T)[:,0]
        target=bd
        variants={'native':native,'dark_native':dark,'other_component':other,
                  'random_dark':random,'visible':visible,'sham':zero,
                  'full_state':d1-x1}
        successes={k:int((((measure(v)-target).abs()<=cutoff)&elig).sum()) for k,v in variants.items()}
        immediate_pass={k:int(((v[:,0].abs()<=.05)&elig).sum()) for k,v in variants.items()}
        clean_n=int(elig.sum())
        # Reparameterize factor coordinates by invertible shear R. Physical
        # unmasked W is unchanged, but singleton component patches may change.
        R=torch.tensor([[1.,1.],[0.,1.]])
        V2=f.V.detach()@R;U2=torch.linalg.inv(R)@f.U.detach()
        assert torch.allclose((V2@U2),(f.V@f.U),atol=2e-6)
        relative=((hd-hr)@V2[:,chosen:chosen+1])@U2[chosen:chosen+1]
        gauge_drift=float((relative-native).norm(dim=-1)[elig].mean()) if clean_n else 0.
        full_invariant=float((((hd-hr)@V2)@U2 - ((hd-hr)@f.V)@f.U).abs().max())
        all_writes_delta=((hd-hr)@f.V)@f.U
        error_norm=max(float((variants[k].norm(dim=-1)-norms.squeeze(-1)).abs()[elig].max()) if clean_n else 0. for k in ['native','dark_native','other_component','random_dark','visible'])
        parity=None
        if isinstance(f,NativeFactors):
            test_h,_=draw(seed+520000,32)
            n1,n2=rollout(f,test_h);factors=Factors(seed)
            with torch.no_grad():factors.V.copy_(f.V);factors.U.copy_(f.U)
            q1,q2=rollout(factors,test_h)
            parity=max(float((n1-q1).abs().max()),float((n2-q2).abs().max()))
        return {'seed':seed,'teacher_mse':float(((teacher_one[:,0]-a).square().mean()+(teacher_two[:,0]-br).square().mean())/2),
                'student_mse':float(((x1[:,0]-a).square().mean()+(x2[:,0]-br).square().mean())/2),
                'clean_n':clean_n,'pairs':pairs,'selected_component':chosen,'calibration_scores':scores,
                'native_parity_max':parity,'factor_update_norms':f.training_factor_update_norms,'transfer':successes,
                'current_visible_preserved':immediate_pass,
                'max_norm_error':error_norm,'gauge_singleton_drift':gauge_drift,
                'gauge_full_write_error':full_invariant,
                'delayed_observable_rank':int(torch.linalg.matrix_rank(torch.stack((f.U.detach()[:,0],f.U.detach()@w.T[:,0])),tol=1e-4)),
                'right_after_first_current_effect_mean':float(native[:,0].abs()[elig].mean()) if clean_n else None,
                'future_native_effect_mean':float((native@w.T[:,0]).abs()[elig].mean()) if clean_n else None}


def run_seed(seed,backend='independent',steps_teacher=520,steps_factors=360,pairs=128):
    torch.set_num_threads(1);torch.manual_seed(seed)
    teacher=train_teacher(seed,steps_teacher)
    factors=train_factors(seed,teacher,backend,steps_factors)
    return assay(seed,teacher,factors,pairs)


def cohort(backend='independent',seeds=SEEDS,pairs=128):
    start=time.perf_counter()
    rows=[run_seed(seed,backend,pairs=pairs) for seed in seeds]
    total=sum(r['clean_n'] for r in rows)
    return {'schema':'bpd.cq03.learned_recurrence.v1','backend':backend,'seeds':list(seeds),
            'teacher_steps':520,'factor_steps':360,'pairs_per_seed':pairs,'rows':rows,
            'eligible_total':total,'success_counts':{k:sum(r['transfer'][k] for r in rows) for k in rows[0]['transfer']},
            'runtime_seconds':time.perf_counter()-start,'torch':torch.__version__,'python':platform.python_version(),
            'scope':'synthetic rank-two trained linear recurrence; independent copy targets, not pretrained RMP or ARG03'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=['independent','native'],default='independent');p.add_argument('--engineering',action='store_true')
    args=p.parse_args();result=cohort(args.backend,(ENGINEERING_SEED,) if args.engineering else SEEDS,32 if args.engineering else 128)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for r in result['rows']:print('SEED',r['seed'],r['clean_n'],r['transfer'],'teacher=',r['teacher_mse'],'student=',r['student_mse'])

if __name__=='__main__':main()
