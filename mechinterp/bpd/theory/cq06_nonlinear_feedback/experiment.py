"""CQ-06: nonlinear delayed-state BPD with prospective sham/confirmation partition.

All behavioral targets arise from the generator, independently of BPD masks.
Root = newly rotated cue world, learned nonlinear recurrent physical model,
separately trained BPD rank-four factorization, and trained second-pass gate.
The first pass is deliberately full; the second is masked from reached state.
"""
from __future__ import annotations
import argparse
import functools
import itertools
import json
import math
import platform
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

D = 12
INPUT = 4
COMPONENTS = 4
DEVELOPMENT = (997,1009,1013,1021,1033,1039,1049,1051)
SHAM = tuple(4001+2*i for i in range(59))
CONFIRMATION = tuple(6001+2*i for i in range(26))
PATTERNS = torch.tensor(list(itertools.product((0.,1.),repeat=COMPONENTS)),dtype=torch.float32)
ROOT = Path(__file__).resolve().parent


def configure():
    torch.set_num_threads(1)


@functools.lru_cache(maxsize=512)
def rotation(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed+3300)
    a=torch.randn(INPUT,INPUT,generator=gen)
    q,r=torch.linalg.qr(a)
    return q*torch.sign(torch.diag(r)).unsqueeze(0)


def source(seed:int,n:int,offset:int,paired_z:bool=False):
    assert n>0 and (n%2==0 if paired_z else True)
    gen=torch.Generator().manual_seed(seed+offset)
    count=n//2 if paired_z else n
    signs=torch.where(torch.randint(0,2,(count,3),generator=gen)==0,-1.,1.)
    values=signs*(.65+.55*torch.rand(count,3,generator=gen))
    first=torch.stack((values[:,0],values[:,1],torch.zeros(count),-torch.ones(count)),1)
    second=torch.stack((torch.zeros(count),torch.zeros(count),values[:,2],torch.ones(count)),1)
    labels=torch.stack((signs[:,0]*signs[:,1],signs[:,0]*signs[:,2]),1)
    if paired_z:
        first=first.repeat_interleave(2,dim=0)
        zflip=second.clone();zflip[:,2]*=-1
        second=torch.stack((second,zflip),1).reshape(n,INPUT)
        labels=torch.stack((labels,labels*torch.tensor([1.,-1.])),1).reshape(n,2)
    q=rotation(seed)
    return first@q,second@q,labels


def donor_inputs(seed,n=128):
    ur0,u1,targets=source(seed,n,811000)
    rot=rotation(seed)
    unmixed=ur0@rot.T
    flip=torch.tensor((-1.,-1.,1.,1.))
    ud0=(unmixed*flip)@rot
    return ur0,ud0,u1,targets,torch.stack((targets[:,0],-targets[:,1]),1)


class Teacher(nn.Module):
    def __init__(self,seed:int):
        super().__init__()
        gen=torch.Generator().manual_seed(seed+1000)
        self.A=nn.Parameter(torch.randn(D+INPUT,COMPONENTS,generator=gen)*.21)
        self.B=nn.Parameter(torch.randn(COMPONENTS,D,generator=gen)*.21)
        self.bias=nn.Parameter(torch.zeros(D))
        self.head=nn.Linear(D,1,bias=False)
        with torch.no_grad():self.head.weight.copy_(torch.randn(1,D,generator=gen)*.2)
    def weight(self):return self.A@self.B
    def step(self,state,cue):return torch.tanh(torch.cat((state,cue),1)@self.A@self.B+self.bias)
    def forward(self,u0,u1):
        first=self.step(torch.zeros(len(u0),D),u0)
        second=self.step(first,u1)
        return torch.cat((self.head(first),self.head(second)),1),first,second


def train_teacher(seed:int,steps:int=850):
    t=Teacher(seed);opt=torch.optim.Adam(t.parameters(),lr=.018)
    for i in range(steps):
        u0,u1,y=source(seed,128,20000+i)
        logits,_,_=t(u0,u1)
        loss=F.softplus(-logits*y).mean()+.0004*(t.A.square().mean()+t.B.square().mean())
        opt.zero_grad();loss.backward();opt.step()
    return t


class Factors(nn.Module):
    def __init__(self,seed):
        super().__init__()
        g=torch.Generator().manual_seed(seed+87000)
        self.V=nn.Parameter(torch.randn(D+INPUT,COMPONENTS,generator=g)*.28)
        self.U=nn.Parameter(torch.randn(COMPONENTS,D,generator=g)*.28)
    def weight(self):return self.V@self.U


def train_factors(seed:int,t:Teacher,steps:int=600):
    f=Factors(seed);opt=torch.optim.Adam([f.V,f.U],lr=.05)
    w=t.weight().detach()
    for i in range(steps):
        u0,u1,_=source(seed,96,240000+i)
        with torch.no_grad():
            ref1=t.step(torch.zeros(len(u0),D),u0)
            ref2=t.step(ref1,u1)
        first=independent_step(t,f,torch.zeros(len(u0),D),u0)
        second=independent_step(t,f,first,u1)
        loss=(first-ref1).square().mean()+(second-ref2).square().mean()+.5*(f.weight()-w).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    return f


def canonicalize(f:Factors):
    with torch.no_grad():
        w=f.weight().detach().double()
        q,s,vh=torch.linalg.svd(w,full_matrices=False)
        root=s[:COMPONENTS].clamp_min(0.).sqrt()
        v=q[:,:COMPONENTS]*root[None,:]
        u=root[:,None]*vh[:COMPONENTS,:]
        before=float((w-v@u).abs().max())
        f.V.copy_(v.float());f.U.copy_(u.float())
    gaps=[float(s[i]-s[i+1]) for i in range(COMPONENTS-1)]
    return {'initial_full_matrix_reconstruction_error':before,
            'min_nonzero_singular_gap':min(gaps),
            'singular_values':[float(z) for z in s[:COMPONENTS]]}


class NativeModel(nn.Module):
    """Actual repository BPDModel/TiedComponentLinear shared physical core."""
    def __init__(self,t:Teacher,f:Factors):
        super().__init__()
        from mechinterp.bpd.components import BPDModel
        from mechinterp.bpd.config import AblationConfig
        class Base(nn.Module):
            def __init__(self,weight,bias):
                super().__init__()
                self.core=nn.Linear(D+INPUT,D,bias=False)
                with torch.no_grad():self.core.weight.copy_(weight.T)
                self.register_buffer('fixed_bias',bias.clone())
            def forward(self,cue_and_state):return torch.tanh(self.core(cue_and_state)+self.fixed_bias)
        self.bpd=BPDModel(Base(t.weight().detach(),t.bias.detach()),{'core':COMPONENTS},
                      max_passes=2,ablation=AblationConfig(mode='zero',sphere_rescale=False),init_svd=False)
        with torch.no_grad():
            wrap=self.bpd.wrappers['core']
            wrap.V.copy_(f.V);wrap.U.copy_(f.U)
    def step(self,state,cue,mask,tau):
        from mechinterp.bpd.components import PassMaskSchedule
        n=len(state)
        schedule=PassMaskSchedule(mask.reshape(n,1,1,COMPONENTS),torch.zeros(n,1,1))
        with self.bpd.masked_components({'core':schedule},pass_offset=tau):
            return self.bpd.target_model(torch.cat((state,cue),1).unsqueeze(1))[:,0]


def independent_step(t,f,state,cue,mask=None):
    joined=torch.cat((state,cue),1)
    acts=joined@f.V
    if mask is not None:acts=acts*mask
    return torch.tanh(acts@f.U+t.bias)


def step(t,f,state,cue,mask=None,backend='independent',native=None,tau=0):
    if mask is None:mask=torch.ones(len(state),COMPONENTS)
    if backend=='native':return native.step(state,cue,mask,tau)
    return independent_step(t,f,state,cue,mask)


def logits(t,state):return t.head(state).squeeze(-1)


def both(t,f,u0,u1,mask2=None,backend='independent',native=None):
    s1=step(t,f,torch.zeros(len(u0),D),u0,backend=backend,native=native,tau=0)
    s2=step(t,f,s1,u1,mask2,backend,native,tau=1)
    return torch.stack((logits(t,s1),logits(t,s2)),1),s1,s2


def oracle_masks(t,f,u0,u1):
    """Mask labels distill full model; independent behavioral labels are unused."""
    with torch.no_grad():
        full,s1,s2=both(t,f,u0,u1)
        x=torch.cat((s1,u1),1)@f.V
        candidate=torch.tanh((x[:,None,:]*PATTERNS[None,:,:])@f.U+t.bias)
        cand_logits=(candidate@t.head.weight.T).squeeze(-1)
        difference=(cand_logits-full[:,None,1]).abs()
        state_rms=(candidate-s2[:,None,:]).square().mean(-1).sqrt()
        acceptable=(difference<=.7)&(state_rms<=1.15)
        scores=torch.where(acceptable,PATTERNS.sum(1)[None,:]+.005*difference,
                           100+difference+state_rms)
        chosen=scores.argmin(1)
        return PATTERNS[chosen],{'all_have_feasible_masks':bool(acceptable.any(1).all()),
              'feasible_count':int(acceptable.any(1).sum()),
              'chosen_pattern_histogram':torch.bincount(chosen,minlength=16).tolist()}


class Policy(nn.Module):
    def __init__(self,seed,kind='feedback'):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed+(50200 if kind=='feedback' else 50300))
            n_in=D+INPUT if kind=='feedback' else INPUT
            self.net=nn.Sequential(nn.Linear(n_in,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,COMPONENTS))
    def scores(self,first,u0_or_u1):return self.net(torch.cat((first,u0_or_u1),1)) if first is not None else self.net(u0_or_u1)
    def mask(self,first,u0_or_u1,threshold=0.):return (self.scores(first,u0_or_u1)>=threshold).float()


def train_policies(seed,t,f,n=4096,updates=500):
    u0,u1,_=source(seed,n,610000)
    gold,diag=oracle_masks(t,f,u0,u1)
    with torch.no_grad():_,s1,_=both(t,f,u0,u1)
    gate=Policy(seed,'feedback');planner=Policy(seed,'planner')
    features=torch.cat((s1,u1),1)
    for model,inputs,offset in ((gate,features,610002),(planner,u0,610003)):
        gen=torch.Generator().manual_seed(seed+offset)
        optim=torch.optim.Adam(model.parameters(),lr=.009)
        for i in range(updates):
            idx=torch.randint(0,n,(256,),generator=gen)
            loss=F.binary_cross_entropy_with_logits(model.net(inputs[idx]),gold[idx])
            optim.zero_grad();loss.backward();optim.step()
    return gate,planner,diag


def pass2_masks(t,f,u0,u1,gate,planner=None,calibration=None,backend='independent',native=None):
    with torch.no_grad():
        full,s1,_=both(t,f,u0,u1,backend=backend,native=native)
        g=gate.mask(s1,u1)
        return g,s1


def calibrate(seed,t,f,gate,planner,n=1024):
    u0,u1,y=source(seed,n,620000)
    with torch.no_grad():
        _,s1,_=both(t,f,u0,u1)
        mask=gate.mask(s1,u1)
        budget=float(mask.sum(1).mean())
        a=[]
        for pattern in PATTERNS:
            prediction,_,_=both(t,f,u0,u1,pattern.expand(n,-1))
            good=(prediction.sign()==y).all(1)
            a.append(float(good.float().mean()))
        costs=[int(row.sum()) for row in PATTERNS]
        best=(-1.,0,0,1.)
        for i in range(16):
            if costs[i]<=budget and a[i]>best[0]:best=(a[i],i,i,1.)
            for j in range(i+1,16):
                if costs[i]==costs[j]:continue
                p=(budget-costs[j])/(costs[i]-costs[j])
                if 0<=p<=1:
                    val=p*a[i]+(1-p)*a[j]
                    if val>best[0]+1e-12:best=(val,i,j,float(p))
        score=planner.scores(None,u0).flatten()
        keep=min(max(budget/COMPONENTS,0.),1.)
        threshold=float(torch.quantile(score,(1.-keep)))
        planner_cost=float((score>=threshold).float().view(n,COMPONENTS).sum(1).mean())
    return {'gate_calibration_budget_second':budget,'static_patterns':[best[1],best[2]],
            'static_p_first':best[3],'static_cal_accuracy':best[0],
            'planner_threshold':threshold,'planner_calibration_budget_second':planner_cost}


def constant_mix(seed,n,policy):
    gen=torch.Generator().manual_seed(seed+630000)
    choose=torch.rand(n,generator=gen)<policy['static_p_first']
    return torch.where(choose[:,None],PATTERNS[policy['static_patterns'][0]],PATTERNS[policy['static_patterns'][1]])


def random_slots(seed,m):
    gen=torch.Generator().manual_seed(seed+631000)
    r=torch.rand(len(m),COMPONENTS,generator=gen).argsort(1).argsort(1)
    return (r<m.sum(1,keepdim=True)).float()


def scores_on_task(seed,t,f,gate,planner,policy,backend='independent',native=None,n=256):
    u0,u1,y=source(seed,n,810000,paired_z=True)
    with torch.no_grad():
        _,s1,_=both(t,f,u0,u1,backend=backend,native=native)
        adaptive=gate.mask(s1,u1)
        masks={'feedback':adaptive,'input_planner':planner.mask(None,u0,policy['planner_threshold']),
               'static_mix':constant_mix(seed,n,policy),'random_slots':random_slots(seed,adaptive),
               'full':PATTERNS[-1].expand(n,-1)}
        result={}
        for k,mask in masks.items():
            pred,_,_=both(t,f,u0,u1,mask,backend,native)
            first=(pred[:,0]*y[:,0]>0)
            second=(pred[:,1]*y[:,1]>0)
            result[k]={'both_correct':int((first&second).sum()),'second_correct':int(second.sum()),
                       'mean_active_slots':4+float(mask.sum(1).mean()),
                       'first_correct':int(first.sum())}
        exact_teacher,_,_=t(u0,u1)
        result['teacher']={'both_correct':int(((exact_teacher*y)>0).all(1).sum()),
                           'second_correct':int((exact_teacher[:,1]*y[:,1]>0).sum()),
                           'first_correct':int((exact_teacher[:,0]*y[:,0]>0).sum()),
                           'mean_active_slots':8.}
        result['algebraic_oracle']={'both_correct':n,'second_correct':n,'first_correct':n,'mean_active_slots':None}
        myopic=t.head(s1).squeeze(-1)
        result['myopic_same_first_cue_pairs']={'n':n,'correct':int((myopic*y[:,1]>0).sum()),
                         'pairs_exactly_opposite_second_label':bool((y[::2,1]==-y[1::2,1]).all()),
                         'pairs_identical_first_cue':bool(torch.equal(u0[::2],u0[1::2]))}
        return result


def donor_assay(seed,t,f,gate,backend='independent',native=None,n=128,include_pairs=True):
    ur0,ud0,u1,yr,yd=donor_inputs(seed,n)
    with torch.no_grad():
        _,sr,_=both(t,f,ur0,u1,backend=backend,native=native)
        _,sd,_=both(t,f,ud0,u1,backend=backend,native=native)
        r0=torch.cat((torch.zeros(n,D),ur0),1)
        d0=torch.cat((torch.zeros(n,D),ud0),1)
        chosen=torch.tensor((1.,1.,0.,0.))
        pre_r=(r0@f.V)@f.U+t.bias
        pre_delta=(((d0-r0)@f.V)*chosen)@f.U
        raw=torch.tanh(pre_r+pre_delta)-sr
        dose=raw.norm(dim=1,keepdim=True)
        axis=t.head.weight.detach()[0]
        axis=axis/axis.norm()
        def dark(x):return x-(x@axis)[:,None]*axis
        def match(x):
            norm=x.norm(dim=1,keepdim=True)
            fallback=torch.zeros_like(x);fallback[:,0]=1.
            fallback=dark(fallback)
            fallback=fallback/fallback.norm(dim=1,keepdim=True).clamp_min(1e-12)
            return torch.where(norm>1e-8,x/norm.clamp_min(1e-8),fallback)*dose
        p=match(dark(raw))
        rng1=torch.Generator().manual_seed(seed+812000)
        rng2=torch.Generator().manual_seed(seed+813000)
        r1=match(dark(torch.randn(n,D,generator=rng1)))
        r2=match(dark(torch.randn(n,D,generator=rng2)))
        # Stronger on-manifold null: an unrelated clean donor state with the
        # same first-step parity label, projected to the current readout kernel.
        # The world/generator changes per independent trained model root.
        natural=[]
        for index,other_offset in enumerate((814000,815000)):
            other_u0,_,other_labels=source(seed,n,other_offset)
            signs=torch.where(other_labels[:,0]==yr[:,0],1.,-1.)
            local=other_u0@rotation(seed).T
            local[:,1]*=signs
            other_u0=local@rotation(seed)
            _,other_s1,_=both(t,f,other_u0,u1,backend=backend,native=native)
            natural.append(match(dark(other_s1-sr)))
        visible=dose*axis[None,:]
        arms={'dark_bpd':p,'random_dark_1':r1,'random_dark_2':r2,
              'manifold_dark_1':natural[0],'manifold_dark_2':natural[1], 'raw_bpd':raw,
              'current_visible':visible,'sham':torch.zeros_like(raw),'full_donor':sd-sr}
        frozen=gate.mask(sr,u1)
        reports={'feedback':{},'frozen':{}}
        switches={}
        out_of_tanh_range={}
        pairs={} if include_pairs else None
        for name,delta in arms.items():
            reached=sr+delta
            out_of_tanh_range[name]=int((reached.abs()>1.00001).any(1).sum())
            changed=gate.mask(reached,u1)
            switches[name]=int((changed!=frozen).any(1).sum())
            entry={}
            for policy_name,mask in [('feedback',changed),('frozen',frozen)]:
                out=step(t,f,reached,u1,mask,backend,native,tau=1)
                ok=(logits(t,out)*yd[:,1]>0)
                reports[policy_name][name]=int(ok.sum())
                entry[policy_name]=ok.int().tolist() if include_pairs else None
            if include_pairs:pairs[name]=entry
        eq_dose_error=max(float((x.norm(dim=1)-dose[:,0]).abs().max()) for x in (p,r1,r2,natural[0],natural[1],visible))
        immediate_dark=float((p@axis).abs().max())
        full_first=logits(t,sr)*yr[:,0]>0
        donor_first=logits(t,sd)*yd[:,0]>0
        return {'n_pairs':n,'all_pairs_scored':True,'per_root_target_flipped':bool((yr[:,1]==-yd[:,1]).all()),
                'first_target_retained':bool((yr[:,0]==yd[:,0]).all()),
                'clean_first_recipient':int(full_first.sum()),'clean_first_donor':int(donor_first.sum()),
                'zero_dose':int((dose[:,0]<=1e-8).sum()),
                'mean_native_partial_state_dose':float(dose.mean()),
                'mean_full_donor_state_dose':float((sd-sr).norm(dim=1).mean()),
                'max_equal_dose_error':eq_dose_error,'max_immediate_readout_dark_effect':immediate_dark,
                'switches':switches,'out_of_tanh_range_by_arm':out_of_tanh_range,
                'counts':reports,'pair_outcomes':pairs}


def gauge_assay(seed,t,f,gate,n=64):
    u0,u1,_=source(seed,n,820000)
    with torch.no_grad():
        _,s1,_=both(t,f,u0,u1)
        m=gate.mask(s1,u1)
        normal,_,_=both(t,f,u0,u1,m)
        v=f.V.detach().double();u=f.U.detach().double();w=v@u
        R1=torch.eye(COMPONENTS,dtype=torch.float64);R1[0,1]=.6
        R2=torch.eye(COMPONENTS,dtype=torch.float64);R2[0,2]=.55;R2[2,0]=-.35
        errors=[];raw=[];changes=[];full=[];singular_gaps=[]
        for R in (R1,R2):
            vs=v@R;us=torch.linalg.inv(R)@u
            full.append(float((vs@us-w).abs().max()))
            f1=torch.tanh(((torch.cat((torch.zeros(n,D,dtype=torch.float64),u0.double()),1)@vs))@us+t.bias.double())
            f2=torch.tanh(((torch.cat((f1,u1.double()),1)@vs)*m.double())@us+t.bias.double())
            raw.append(float((logits(t,f2.float())-normal[:,1]).abs().max()))
            q,s,vh=torch.linalg.svd(vs@us,full_matrices=False)
            sq=s[:COMPONENTS].sqrt()
            vc=q[:,:COMPONENTS]*sq[None,:];uc=sq[:,None]*vh[:COMPONENTS,:]
            c1=torch.tanh(((torch.cat((torch.zeros(n,D,dtype=torch.float64),u0.double()),1)@vc))@uc+t.bias.double())
            new_mask=gate.mask(c1.float(),u1)
            c2=torch.tanh(((torch.cat((c1,u1.double()),1)@vc)*new_mask.double())@uc+t.bias.double())
            errors.append(float((logits(t,c2.float())-normal[:,1]).abs().max()))
            changes.append(int((new_mask!=m).any(1).sum()))
            singular_gaps.append(min(float(s[i]-s[i+1]) for i in range(COMPONENTS-1)))
        return {'max_full_operator_shear_error':max(full),
                'max_raw_shear_masked_logit_drift':max(raw),
                'max_recanonicalized_logit_drift':max(errors),
                'recanonicalized_mask_switches':sum(changes),
                'min_singular_gap_after_shear':min(singular_gaps)}


def native_parity(seed,t,f,native):
    u0,u1,_=source(seed,64,830000)
    gen=torch.Generator().manual_seed(seed+830001)
    m=(torch.rand(64,COMPONENTS,generator=gen)>.5).float()
    with torch.no_grad():
        a,a1,a2=both(t,f,u0,u1,m,'independent')
        b,b1,b2=both(t,f,u0,u1,m,'native',native)
        return max(float((a-b).abs().max()),float((a1-b1).abs().max()),float((a2-b2).abs().max()))


def run_root(seed,phase='development',backend='independent',include_pairs=True):
    configure()
    torch.manual_seed(seed)
    t=train_teacher(seed)
    f=train_factors(seed,t)
    fit_error=float((f.weight().detach()-t.weight().detach()).abs().max())
    geometry=canonicalize(f)
    gate,planner,oracle=train_policies(seed,t,f)
    native=NativeModel(t,f) if backend=='native' else None
    parity=native_parity(seed,t,f,native) if native is not None else None
    with torch.no_grad():
        donor=donor_assay(seed,t,f,gate,backend,native,include_pairs=include_pairs)
        outcome={'seed':seed,'phase':phase,'fit_max_weight_error':fit_error,'geometry':geometry,
                 'oracle':oracle,'native_max_parity_error':parity,'donor':donor}
        if phase!='sham':
            calibration=calibrate(seed,t,f,gate,planner)
            task=scores_on_task(seed,t,f,gate,planner,calibration,backend,native)
            gauge=gauge_assay(seed,t,f,gate)
            outcome.update({'calibration':calibration,'task':task,'gauge':gauge})
        return outcome


def cohort(phase='development',backend='independent',seeds=None,include_pairs=True):
    if seeds is None:seeds={'development':DEVELOPMENT,'sham':SHAM,'confirmation':CONFIRMATION}[phase]
    seeds=tuple(seeds)
    rows=[run_root(s,phase,backend,include_pairs=include_pairs) for s in seeds]
    return {'schema':'bpd.cq06.nonlinear.delayed-state.v1','phase':phase,'backend':backend,'seeds':list(seeds),
            'teacher_steps':850,'factor_steps':600,'gate_steps':500,'n_donor_pairs_per_root':128,
            'n_task_cases_per_root':256 if phase!='sham' else 0,
            'rows':rows,'torch':torch.__version__,'python':platform.python_version(),
            'evidence_class':'synthetic nonlinear recurrent family, not natural-language pretrained checkpoints'}


def stats(run):
    ds=[(r['donor']['counts']['feedback']['dark_bpd']-r['donor']['counts']['feedback']['manifold_dark_1'])/128 for r in run['rows']]
    second=[(r['donor']['counts']['feedback']['dark_bpd']-r['donor']['counts']['frozen']['dark_bpd'])/128 for r in run['rows']]
    shams=[abs(r['donor']['counts']['feedback']['manifold_dark_1']-r['donor']['counts']['feedback']['manifold_dark_2'])/128 for r in run['rows']]
    counts={policy:sum(r['donor']['counts']['feedback'][policy] for r in run['rows']) for policy in run['rows'][0]['donor']['counts']['feedback']}
    task=None
    if run['phase']!='sham':
        task={a:{'correct':sum(r['task'][a]['both_correct'] for r in run['rows'])}
           for a in ('feedback','input_planner','static_mix','random_slots','full','teacher','algebraic_oracle')}
        for a in task:task[a]['mean_slots']=None if a=='algebraic_oracle' else sum(r['task'][a]['mean_active_slots'] for r in run['rows'])/len(run['rows'])
    return {'roots':len(run['rows']),'donor_feedback_counts':counts,'primary_effects_by_root':ds,
            'feedback_vs_frozen_by_root':second,'random_vs_random_abs_by_root':shams,
            'max_random_vs_random_abs':max(shams),'task':task,
            'max_patch_norm_error':max(r['donor']['max_equal_dose_error'] for r in run['rows'])}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=('development','sham','confirmation'),default='development')
    ap.add_argument('--backend',choices=('independent','native'),default='independent')
    ap.add_argument('--seeds',nargs='*',type=int)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--no-raw-pairs',action='store_true')
    args=ap.parse_args()
    run=cohort(args.phase,args.backend,args.seeds if args.seeds else None,include_pairs=not args.no_raw_pairs)
    report=stats(run)
    if args.out:
        if args.out.exists():raise FileExistsError(f'Write-once result file already exists: {args.out}')
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps({'run':run,'stats':report},sort_keys=True,indent=2)+'\n')
    print(json.dumps({'phase':args.phase,'backend':args.backend,'seeds':run['seeds'],'stats':report},indent=2))

if __name__=='__main__':main()
