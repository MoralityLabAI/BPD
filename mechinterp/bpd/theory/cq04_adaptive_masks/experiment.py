"""BPD-CQ-04: input-adaptive two-pass execution masks over learned CQ-03 recurrence.

Task labels (a,b) precede the BPD decomposition. The controller learns from
independent development examples only. Native parity uses PassMaskSchedule.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import platform
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from mechinterp.bpd.theory.cq03_learned_recurrence.experiment import train_teacher,train_factors

ROOT=Path(__file__).resolve().parent
SEEDS=(701,709,719,727,733,739,743,751)
ENGINEERING_SEEDS=(997,1009,1013,1021)
N_TRAIN=4096
N_CAL=2048
N_TEST=256
CUTOFF=.12
PATTERNS=torch.tensor(list(itertools.product((0.,1.),repeat=4)),dtype=torch.float32)


def source(seed,n,offset):
    """Independent source streams. 40% dual-source, 30% A, 30% B."""
    rng=torch.Generator().manual_seed(seed+offset)
    ab=torch.rand((n,2),generator=rng)*1.6-.8
    mode=torch.randint(0,10,(n,),generator=rng)
    ab[(mode>=4)&(mode<7),1]=0.
    ab[mode>=7,0]=0.
    h=torch.zeros(n,4);h[:,2:]=ab
    return h,ab,torch.where(mode<4,0,torch.where(mode<7,1,2))


def eval_mask(f,h,mask,backend='independent'):
    """Mask shape [batch,4], two component slots at each of two passes."""
    assert mask.shape==(len(h),4)
    if backend=='native':
        from mechinterp.bpd.components import PassMaskSchedule
        schedule=PassMaskSchedule(mask.reshape(len(h),2,1,2),torch.zeros(len(h),2,1))
        with f.bpd.masked_components({'core':schedule}):
            first=f.forward(h);second=f.forward(first)
    else:
        first=(h@f.V*mask[:,:2])@f.U
        second=(first@f.V*mask[:,2:])@f.U
    return torch.stack((first[:,0],second[:,0]),1),first,second


def oracle_labels(f,h,target):
    """16 masks, calibration-only target-guided imitation; never used in test routing."""
    scores=[]
    with torch.no_grad():
        for pattern in PATTERNS:
            pred,_,_=eval_mask(f,h,pattern.expand(len(h),-1))
            err=(pred-target).abs().amax(1)
            card=int(pattern.sum())
            scores.append(torch.where(err<=CUTOFF,card+err*.01,100+err))
    return PATTERNS[torch.stack(scores).argmin(0)]


class Controller(nn.Module):
    """Four hard pass-component bits, conditioned on observed initial hidden state."""
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed+18300)
            self.net=nn.Sequential(nn.Linear(4,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,4))
    def forward(self,h):
        a,b=h[:,2],h[:,3]
        return self.net(torch.stack((a,b,a.abs(),b.abs()),1))
    def masks(self,h):
        return (self(h)>=0).float()


def train_controller(seed,f):
    h,t,_=source(seed,N_TRAIN,432000)
    gold=oracle_labels(f,h,t)
    controller=Controller(seed)
    opt=torch.optim.Adam(controller.parameters(),lr=.006)
    rng=torch.Generator().manual_seed(seed+432001)
    for _ in range(800):
        idx=torch.randint(0,len(h),(256,),generator=rng)
        loss=F.binary_cross_entropy_with_logits(controller(h[idx]),gold[idx])
        opt.zero_grad();loss.backward();opt.step()
    return controller


def correct(pred,target):
    return (pred-target).abs().amax(1)<=CUTOFF


def static_budget_mix(seed,f,controller):
    """Calibration-only optimal input-blind mixture of two constant patterns."""
    h,t,_=source(seed,N_CAL,433000)
    with torch.no_grad():
        cap=float(controller.masks(h).sum(1).mean())
        acc=[float(correct(eval_mask(f,h,m.expand(len(h),-1))[0],t).float().mean()) for m in PATTERNS]
    costs=[float(m.sum()) for m in PATTERNS]
    # The exact two-support LP extreme points suffice for one linear budget.
    best=(-1.,0,0,1.)
    for i in range(16):
        if costs[i]<=cap and (acc[i],-i)>(best[0],-best[1]):
            best=(acc[i],i,i,1.)
        for j in range(i+1,16):
            if costs[i]==costs[j]:continue
            p=(cap-costs[j])/(costs[i]-costs[j])
            if not 0<=p<=1:continue
            a=p*acc[i]+(1-p)*acc[j]
            if a>best[0]+1e-10:best=(a,i,j,p)
    assert best[0]>=0
    return {'patterns':[best[1],best[2]],'p_first':best[3],
            'calibration_expected_accuracy':best[0], 'calibration_gate_cost':cap,
            'calibration_static_expected_cost':best[3]*costs[best[1]]+(1-best[3])*costs[best[2]]}


def static_masks(seed,n,policy):
    g=torch.Generator().manual_seed(seed+445000)
    choose=torch.rand(n,generator=g)<policy['p_first']
    return torch.where(choose[:,None],PATTERNS[policy['patterns'][0]],PATTERNS[policy['patterns'][1]])


def permutation_masks(seed,mask):
    g=torch.Generator().manual_seed(seed+446000)
    n=len(mask);r=torch.rand((n,4),generator=g).argsort(1).argsort(1)
    return (r<mask.sum(1,keepdim=True)).float()


def rate(row):return int(row.sum())


def donor_assay(seed,f,controller,backend='independent',n=128):
    """Source-heldout donor-b target, masked recipient with frozen gate plan."""
    g=torch.Generator().manual_seed(seed+447000)
    a=torch.rand(n,generator=g)*1.6-.8
    br=torch.rand(n,generator=g)*.4-.2
    sign=torch.where(torch.rand(n,generator=g)<.5,-1.,1.)
    bd=br+sign*(.30+.15*torch.rand(n,generator=g))
    hr=torch.zeros(n,4);hd=hr.clone()
    hr[:,2]=a;hd[:,2]=a;hr[:,3]=br;hd[:,3]=bd
    with torch.no_grad():
        mr=controller.masks(hr);md=controller.masks(hd)
        xr,x1,_=eval_mask(f,hr,mr,backend)
        xd,dx1,_=eval_mask(f,hd,md,backend)
        good=correct(xr,torch.stack((a,br),1))&correct(xd,torch.stack((a,bd),1))
        same=(mr==md).all(1)
        delta=((hd-hr)@f.V*mr[:,:2])@f.U
        norms=delta.norm(dim=1,keepdim=True)
        eligible=good&same&(norms[:,0]>1e-7)
        dark=delta.clone();dark[:,0]=0
        random=torch.randn((n,4),generator=g);random[:,0]=0
        def match(z):
            d=z.norm(dim=1,keepdim=True)
            fallback=torch.zeros_like(z);fallback[:,1]=1
            return torch.where(d>1e-8,z/d.clamp_min(1e-8),fallback)*norms
        dark=match(dark);random=match(random)
        arms={'native':delta,'dark_native':dark,'random_dark':random,
              'sham':torch.zeros_like(delta),'full_state':dx1-x1}
        scores={};norm_errors={}
        for name,patch in arms.items():
            if backend=='native':
                from mechinterp.bpd.components import PassMaskSchedule
                mask=mr[:,2:].reshape(n,1,1,2)
                schedule=PassMaskSchedule(mask,torch.zeros(n,1,1))
                with f.bpd.masked_components({'core':schedule},pass_offset=1):
                    y=f.forward(x1+patch)[:,0]
            else:
                y=(((x1+patch)@f.V)*mr[:,2:])@f.U
                y=y[:,0]
            scores[name]=int(((y-bd).abs().le(CUTOFF)&eligible).sum())
            if name in ('native','dark_native','random_dark'):
                norm_errors[name]=float((patch.norm(dim=1)-norms[:,0]).abs().max())
        return {'pairs':n,'eligible':int(eligible.sum()),'clean':int(good.sum()),
                'same_mask':int(same.sum()),'success':scores,
                'max_equal_dose_error':max(norm_errors.values()),
                'mean_native_immediate_visibility':float(delta[:,0].abs().mean())}


def assay(seed,backend='independent'):
    torch.set_num_threads(1)
    teacher=train_teacher(seed)
    f=train_factors(seed,teacher,backend)
    controller=train_controller(seed,f)
    policy=static_budget_mix(seed,f,controller)
    h,t,mode=source(seed,N_TEST,444000)
    with torch.no_grad():
        masks={'adaptive':controller.masks(h), 'static_mix':static_masks(seed,len(h),policy),
               'full':PATTERNS[-1].expand(len(h),-1)}
        masks['random_slots']=permutation_masks(seed,masks['adaptive'])
        pred={name:eval_mask(f,h,mask,backend)[0] for name,mask in masks.items()}
        results={name:{'correct':rate(correct(out,t)),'mean_components':float(masks[name].sum(1).mean()),
                       'by_source':[{'n':int((mode==k).sum()),'correct':rate(correct(out,t)&(mode==k))} for k in range(3)]}
                 for name,out in pred.items()}
        parity=None
        if backend=='native':
            x1=(h@f.V*masks['adaptive'][:,:2])@f.U
            x2=(x1@f.V*masks['adaptive'][:,2:])@f.U
            parity=float((pred['adaptive']-torch.stack((x1[:,0],x2[:,0]),1)).abs().max())
        donor=donor_assay(seed,f,controller,backend)
        R=torch.tensor([[1.,1.],[0.,1.]])
        full_err=float(((f.V@R)@(torch.linalg.inv(R)@f.U)-f.V@f.U).abs().max())
        # Fixed component execution masks in the original learned gauge.
        V2=f.V@R;U2=torch.linalg.inv(R)@f.U
        x1=(h@V2*masks['adaptive'][:,:2])@U2
        x2=(x1@V2*masks['adaptive'][:,2:])@U2
        gauge_correct=rate(correct(torch.stack((x1[:,0],x2[:,0]),1),t))
        return {'seed':seed,'backend':backend,'metrics':results,'static_calibration':policy,
                'donor':donor,'native_parity_max':parity,'gauge_full_error':full_err,
                'gauge_masked_accuracy_after_shear':gauge_correct}


def cohort(backend='independent',seeds=SEEDS):
    rows=[assay(s,backend) for s in seeds]
    counts={a:sum(r['metrics'][a]['correct'] for r in rows) for a in rows[0]['metrics']}
    gate_cost=sum(r['metrics']['adaptive']['mean_components'] for r in rows)/len(rows)
    diffs=[(r['metrics']['adaptive']['correct']-r['metrics']['static_mix']['correct'])/N_TEST for r in rows]
    return {'schema':'bpd.cq04.adaptive_masks.pilot.v1','backend':backend,'seeds':list(seeds),
            'cases_per_seed':N_TEST,'rows':rows,'success_counts':counts,'mean_adaptive_cost':gate_cost,
            'per_root_primary_diffs':diffs,'positive_root_count':sum(x>0 for x in diffs),
            'evidence_class':'prospective synthetic pilot, root-limited; EvalMath preflight confirmation unresolved',
            'torch':torch.__version__,'python':platform.python_version()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['independent','native'],default='independent')
    ap.add_argument('--engineering',action='store_true');ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    res=cohort(args.backend,ENGINEERING_SEEDS if args.engineering else SEEDS)
    print(json.dumps({k:v for k,v in res.items() if k!='rows'},indent=2))
    for r in res['rows']:
        print(r['seed'],{k:v['correct'] for k,v in r['metrics'].items()},r['metrics']['adaptive']['mean_components'],r['donor']['eligible'],r['donor']['success'])
    if args.out:
        if args.out.exists():raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
