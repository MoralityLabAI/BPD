"""AMM-07: fresh independently trained recurrent models for selective trust.

See PROTOCOL.md. Never use a development or confirmation root in source tests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

NAME = 'BPD-AMM-07'
SCHEMA = 'bpd.amm07.replicated_feedback.v1'
FAMILIES = ('wren', 'basalt')
PHASE_N = {'engineering': 8, 'training': 64, 'development': 64, 'confirmation': 128}
PAIRS = 1024
CAL_PAIRS = 16
TRAIN_PAIRS = 128
STEPS_TEACHER = 200
STEPS_FACTORS = 140
LR = 0.035
TOL = 0.18
EPS = 1e-9
ARMS = ('quality_aware', 'quality_blind', 'blind_all', 'memory_only', 'local_ridge', 'pooled', 'random_component', 'random_dark', 'random_dark_b', 'sham', 'full_donor')
COMPARISONS = {'aware_vs_blind': 'quality_blind', 'aware_vs_blind_all': 'blind_all', 'aware_vs_memory_only': 'memory_only'}


def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(x: Any) -> str:
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def seed(*parts: object) -> int:
    return int(hashlib.sha256(('AMM07/'+'/'.join(map(str,parts))).encode()).hexdigest()[:15],16)


def gen(*parts: object) -> torch.Generator:
    return torch.Generator(device='cpu').manual_seed(seed(*parts))


def family_for(index: int) -> str:
    return FAMILIES[index % 2]


def root_key(phase: str, index: int) -> str:
    if phase not in PHASE_N or not 0 <= index < PHASE_N[phase]:
        raise ValueError('unknown or out of range root')
    return f'{phase}/{index:04d}'


def data(phase: str, index: int, batch: int, mode: str):
    root=root_key(phase,index)
    g=gen(root,mode)
    ab=torch.rand((batch,2), generator=g)*1.6-.8
    h=torch.zeros((batch,4),dtype=torch.float32)
    h[:,2:4]=ab
    return h,ab


def labels(ab: torch.Tensor, family: str):
    if family=='wren':return ab[:,0],ab[:,1]
    if family=='basalt':return ab[:,1],ab[:,0]
    raise ValueError('bad family')


class Physical(nn.Module):
    def __init__(self,phase: str,index:int):
        super().__init__()
        g=gen(root_key(phase,index),'teacher_init')
        self.i=nn.Parameter(torch.randn(4,2,generator=g)*0.45)
        self.o=nn.Parameter(torch.randn(2,4,generator=g)*0.45)
    def forward(self,h):return (h@self.i)@self.o
    def weight(self):return (self.i@self.o).T


class Factors(nn.Module):
    def __init__(self,phase: str,index:int):
        super().__init__()
        g=gen(root_key(phase,index),'factor_init')
        self.V=nn.Parameter(torch.randn(4,2,generator=g)*.35)
        self.U=nn.Parameter(torch.randn(2,4,generator=g)*.35)
    def forward(self,h):return (h@self.V)@self.U
    def weight(self):return (self.V@self.U).T


def fit(phase: str,index: int):
    """Per-root trained teacher and independently fit BPD factors."""
    # Seed-independent PyTorch execution; every training stream keyed by root.
    torch.set_num_threads(1)
    root=root_key(phase,index)
    fam=family_for(index)
    m=Physical(phase,index)
    initial_m=(m.i.detach().clone(),m.o.detach().clone())
    opt=torch.optim.Adam(m.parameters(),lr=LR)
    g=gen(root,'teacher_train')
    for _ in range(STEPS_TEACHER):
        ab=torch.rand((96,2),generator=g)*1.6-.8
        h=torch.zeros((96,4));h[:,2:]=ab
        a,b=labels(ab,fam)
        p1=m(h);p2=m(p1)
        loss=(p1[:,0]-a).square().mean()+(p2[:,0]-b).square().mean()+.0002*m.weight().square().mean()
        opt.zero_grad();loss.backward();opt.step()
    teacher_updates=[float((p.detach()-q).norm()) for p,q in zip((m.i,m.o),initial_m)]
    f=Factors(phase,index)
    initial_f=(f.V.detach().clone(),f.U.detach().clone())
    opt2=torch.optim.Adam(f.parameters(),lr=LR)
    g2=gen(root,'factor_train')
    w=m.weight().detach()
    for _ in range(STEPS_FACTORS):
        ab=torch.rand((96,2),generator=g2)*1.6-.8
        h=torch.zeros((96,4));h[:,2:]=ab
        with torch.no_grad(): ref1=m(h);ref2=m(ref1)
        x1=f(h);x2=f(x1)
        loss=(x1-ref1).square().mean()+(x2-ref2).square().mean()+.5*(f.weight()-w).square().mean()
        opt2.zero_grad();loss.backward();opt2.step()
    factor_updates=[float((p.detach()-q).norm()) for p,q in zip((f.V,f.U),initial_f)]
    with torch.no_grad():
        product=(f.V.double()@f.U.double())
        q,s,vt=torch.linalg.svd(product,full_matrices=False)
        scaled=s[:2].sqrt()
        vc=q[:,:2]*scaled
        uc=scaled[:,None]*vt[:2]
        f.V.copy_(vc.float());f.U.copy_(uc.float())
        h,ab=data(phase,index,256,'independent_validation')
        y1,y2=labels(ab,fam)
        t1=m(h);t2=m(t1)
        x1=f(h);x2=f(x1)
        metrics={
            'teacher_update_norms': teacher_updates,
            'factor_update_norms': factor_updates,
            'teacher_behavioral_mse':float(((t1[:,0]-y1).square().mean()+(t2[:,0]-y2).square().mean())/2),
            'factor_behavioral_mse':float(((x1[:,0]-y1).square().mean()+(x2[:,0]-y2).square().mean())/2),
            'weight_reconstruction_relative':float((f.weight()-w).norm()/w.norm().clamp_min(1e-8)),
            'singular_values': [float(s[0]),float(s[1])],
            'singular_gap':float(s[0]-s[1]),
            'canonical_reconstruction_max_error':float((product-vc@uc).abs().max()),
            'teacher_weight_sha256':hashlib.sha256(w.numpy().tobytes()).hexdigest(),
            'factor_full_weight_sha256':hashlib.sha256(f.weight().detach().numpy().tobytes()).hexdigest(),
        }
    return m,f,metrics


def donors(phase: str, index: int, n: int, split: str):
    """Mixed relevant + nuisance source edits; labels predetermined from donors."""
    if n not in (CAL_PAIRS, TRAIN_PAIRS, PAIRS):
        raise ValueError('unregistered donor count')
    if split not in ('training_pairs','local_calibration','heldout_test',
                     'repeatability','engineering_test'):
        raise ValueError('unregistered donor split')
    g=gen(root_key(phase,index),split)
    other=torch.rand(n,generator=g)*1.0-.5
    v=torch.rand(n,generator=g)*.4-.2
    sign=torch.where(torch.rand(n,generator=g)<.5,-1.,1.)
    dv=v+sign*(.30+torch.rand(n,generator=g)*.15)
    nuisance_sign=torch.where(torch.rand(n,generator=g)<.5,-1.,1.)
    nuisance_shift=nuisance_sign*(.15+torch.rand(n,generator=g)*.65)
    recipient=torch.zeros((n,4));donor=recipient.clone()
    if family_for(index)=='wren':
        recipient[:,2]=other
        donor[:,2]=other+nuisance_shift
        recipient[:,3]=v
        donor[:,3]=dv
    else:
        recipient[:,3]=other
        donor[:,3]=other+nuisance_shift
        recipient[:,2]=v
        donor[:,2]=dv
    return recipient,donor,dv

def normalize_dark(raw: torch.Tensor, dose: torch.Tensor):
    z=raw.clone();z[:,0]=0.
    norm=z.norm(dim=1,keepdim=True)
    fallback=torch.zeros_like(z);fallback[:,1]=1.
    out=torch.where(norm>1e-8,z/norm.clamp_min(1e-8),fallback)*dose
    return out,int(((norm[:,0]<=1e-8)&(dose[:,0]>1e-8)).sum())


def geometry(f: Factors, family: str):
    """Gauge-stable candidate read/write features, no donor test labels."""
    with torch.no_grad():
        j=3 if family=='wren' else 2
        v=f.V.detach().double();u=f.U.detach().double()
        full=v@u
        s=torch.linalg.svdvals(full)
        gap=float((s[0]-s[1])/(s[0]+s[1]+1e-8))
        ref=(v[j,:]@u).norm().clamp_min(1e-8)
        x=[]; gains=[]
        for c in (0,1):
            dark=u[c].clone(); dark[0]=0
            darknorm=dark.norm()
            sign=1. if float(v[j,c])>=0 else -1.
            direction=sign*dark/darknorm.clamp_min(1e-8)
            gain=float(((direction@full)[0])) if float(darknorm)>1e-8 else 0.
            fraction=float(abs(v[j,c])*darknorm/ref)
            gains.append(gain)
            x.append([1.,gain,abs(gain),gain*gain,fraction,gap,float(family=='basalt')])
        return {'features':x,'signed_gains':gains,'singular_gap':float(s[0]-s[1]),
                'canonical_singular_values':[float(s[0]),float(s[1])]}


def assay(phase: str,index: int,f: Factors,split: str,n: int):
    """One matched physical dose per pair for every active action."""
    rec,don,target=donors(phase,index,n,split)
    with torch.no_grad():
        r1=f(rec);d1=f(don)
        full_delta=d1-r1
        dose=full_delta.norm(dim=1,keepdim=True)
        patches={}; degeneracy={}
        for c in (0,1):
            raw=((don-rec)@f.V[:,c:c+1])@f.U[c:c+1]
            patches[f'component_{c}'],degeneracy[f'component_{c}']=normalize_dark(raw,dose)
        for label in ('random_dark','random_dark_b'):
            noise=torch.randn((n,4),generator=gen(root_key(phase,index),split,label))
            patches[label],_=normalize_dark(noise,dose)
        patches['sham']=torch.zeros_like(full_delta)
        patches['full_donor']=full_delta
        preds={k:f(r1+p)[:,0] for k,p in patches.items()}
        bits={k:''.join(str(int(x)) for x in ((y-target).abs()<=TOL)) for k,y in preds.items()}
        active=('component_0','component_1','random_dark','random_dark_b')
        max_dose_error=max(float((patches[k].norm(dim=1)-dose[:,0]).abs().max()) for k in active)
        max_readout_error=max(float(patches[k][:,0].abs().max()) for k in active)
        return {'split':split,'n':n,'candidate_bits':bits,
                'candidate_hits':{k:bits[k].count('1') for k in bits},
                'dose_max_abs_error':max_dose_error,'current_readout_max_abs_error':max_readout_error,
                'zero_dose_pairs':int((dose[:,0]<=1e-8).sum()),
                'degenerate_projection_pairs':degeneracy,
                'mean_dose':float(dose.mean()),
                'full_donor_visible_change_mean_abs':float(full_delta[:,0].abs().mean())}


def gauge_check(f: Factors, family: str) -> dict:
    """Independent GL2 shear and canonical rank-one invariance on engineered roots."""
    with torch.no_grad():
        v=f.V.detach().double();u=f.U.detach().double();full=v@u
        R=torch.tensor([[1.,.6],[0.,1.]],dtype=torch.float64)
        sheared=(v@R)@(torch.linalg.inv(R)@u)
        max_full=float((full-sheared).abs().max())
        def components(z):
            left,s,right=torch.linalg.svd(z,full_matrices=False)
            return [left[:,:2][:,c:c+1]@right[c:c+1,:]*s[c] for c in (0,1)]
        a,b=components(full),components(sheared)
        rankone=max(float((x-y).abs().max()) for x,y in zip(a,b))
        return {'full_gl2_max_abs_error':max_full,
                'recanonicalized_rank1_max_abs_error':rankone,
                'gap':float(torch.linalg.svdvals(full)[0]-torch.linalg.svdvals(full)[1])}
