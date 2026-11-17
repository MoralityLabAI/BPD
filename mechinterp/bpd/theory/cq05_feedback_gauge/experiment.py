"""CQ-05 input-conditioned first mask and state-feedback second mask.

Independent labels, canonicalized shared rank-two BPD operator, ITT donors.
"""
from __future__ import annotations
import argparse, itertools, json, math, platform
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

PILOT_SEEDS=(1201,1213,1217,1223,1229,1231,1237,1249)
DEV_SEEDS=(997,1009,1013,1021)
PATTERNS=torch.tensor(list(itertools.product((0.,1.),repeat=4)),dtype=torch.float32)
TOL=.12

class Physical(nn.Module):
    def __init__(self,seed):
        super().__init__();g=torch.Generator().manual_seed(seed+902000)
        self.i=nn.Parameter(torch.randn(4,2,generator=g)*.45)
        self.o=nn.Parameter(torch.randn(2,4,generator=g)*.45)
    def weight(self):return (self.i@self.o).T
    def forward(self,h):return (h@self.i)@self.o

class Factors(nn.Module):
    def __init__(self,seed):
        super().__init__();g=torch.Generator().manual_seed(seed+981000)
        self.V=nn.Parameter(torch.randn(4,2,generator=g)*.35)
        self.U=nn.Parameter(torch.randn(2,4,generator=g)*.35)
    def weight(self):return (self.V@self.U).T
    def forward(self,h):return (h@self.V)@self.U

class NativeFactors(nn.Module):
    def __init__(self,seed,weight):
        super().__init__()
        from mechinterp.bpd.components import BPDModel
        from mechinterp.bpd.config import AblationConfig
        class Base(nn.Module):
            def __init__(self,w):
                super().__init__();self.core=nn.Linear(4,4,bias=False)
                with torch.no_grad():self.core.weight.copy_(w)
            def forward(self,x):return self.core(x)
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
    def weight(self):return (self.V@self.U).T
    def forward(self,h):return self.bpd.target_model(h.unsqueeze(1))[:,0]
    def rollout_full(self,h):
        from mechinterp.bpd.components import PassMaskSchedule
        n=len(h)
        s=PassMaskSchedule(torch.ones(n,2,1,2),torch.zeros(n,2,1))
        with self.bpd.masked_components({'core':s}):
            a=self.forward(h);b=self.forward(a)
        return a,b


def physical_fit(seed):
    m=Physical(seed);opt=torch.optim.Adam(m.parameters(),lr=.035)
    g=torch.Generator().manual_seed(seed+390000)
    for _ in range(520):
        ab=torch.rand((96,2),generator=g)*1.8-.9
        h=torch.zeros(96,4);h[:,2:]=ab
        one=m(h);two=m(one)
        loss=(one[:,0]-ab[:,0]).square().mean()+(two[:,0]-ab[:,1]).square().mean()+.0002*m.weight().square().mean()
        opt.zero_grad();loss.backward();opt.step()
    return m


def factor_fit(seed,teacher,backend='independent'):
    f=NativeFactors(seed,teacher.weight()) if backend=='native' else Factors(seed)
    opt=torch.optim.Adam([f.V,f.U],lr=.035)
    g=torch.Generator().manual_seed(seed+680000)
    w=teacher.weight().detach()
    for _ in range(360):
        ab=torch.rand((96,2),generator=g)*1.8-.9
        h=torch.zeros(96,4);h[:,2:]=ab
        with torch.no_grad():ref1=teacher(h);ref2=teacher(ref1)
        if backend=='native':one,two=f.rollout_full(h)
        else:one=f(h);two=f(one)
        loss=(one-ref1).square().mean()+(two-ref2).square().mean()+.5*(f.weight()-w).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    return f


def canonicalize(f):
    """SVD fixes GL(2) gauge off the full reconstructed physical product."""
    with torch.no_grad():
        product=f.V.double()@f.U.double()
        uu,ss,vh=torch.linalg.svd(product,full_matrices=False)
        root=ss[:2].clamp_min(0).sqrt()
        v=uu[:,:2]*root[None,:]
        u=root[:,None]*vh[:2]
        error=float((product-v@u).abs().max())
        f.V.copy_(v.float());f.U.copy_(u.float())
        return {'max_reconstruction_error':error,'s1':float(ss[0]),'s2':float(ss[1]),
                'singular_gap':float(ss[0]-ss[1])}


def make(seed,n,offset):
    g=torch.Generator().manual_seed(seed+offset)
    ab=torch.rand(n,2,generator=g)*1.6-.8
    mode=torch.randint(0,10,(n,),generator=g)
    ab[(mode>=4)&(mode<7),1]=0
    ab[mode>=7,0]=0
    h=torch.zeros(n,4);h[:,2:]=ab
    return h,ab,torch.where(mode<4,0,torch.where(mode<7,1,2))


def step(f,h,m,tau=0,backend='independent'):
    assert m.shape==(len(h),2)
    if backend=='native':
        from mechinterp.bpd.components import PassMaskSchedule
        s=PassMaskSchedule(m[:,None,None,:],torch.zeros(len(h),1,1))
        with f.bpd.masked_components({'core':s},pass_offset=tau):return f.forward(h)
    return (h@f.V*m)@f.U


def two_step(f,h,bits,backend='independent'):
    a=step(f,h,bits[:,:2],0,backend)
    b=step(f,a,bits[:,2:],1,backend)
    return torch.stack((a[:,0],b[:,0]),1),a,b


def success(pred,target):return (pred-target).abs().amax(1)<=TOL


def first_oracle(f,h,targets):
    with torch.no_grad():
        vals=[]
        for p in PATTERNS:
            pred,_,_=two_step(f,h,p.expand(len(h),-1))
            err=(pred-targets).abs().amax(1)
            vals.append(torch.where(err<=TOL,p.sum()+err*.01,100+err))
        return PATTERNS[torch.stack(vals).argmin(0),:2]


def second_oracle(f,state,targets):
    with torch.no_grad():
        vals=[]
        for p in PATTERNS[:4,2:].unique(dim=0):
            pred=step(f,state,p.expand(len(state),-1),1)[:,0]
            err=(pred-targets).abs()
            vals.append(torch.where(err<=TOL,p.sum()+err*.01,100+err))
        patterns=PATTERNS[:4,2:].unique(dim=0)
        return patterns[torch.stack(vals).argmin(0)]


class Feedback(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed+28000)
            self.first_net=nn.Sequential(nn.Linear(4,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,2))
            self.second_net=nn.Sequential(nn.Linear(8,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,2))
    def first(self,h):
        x=h[:,2:]
        return (self.first_net(torch.cat((x,x.abs()),1))>=0).float()
    def second(self,one):
        return (self.second_net(torch.cat((one,one.abs()),1))>=0).float()


def train_feedback(seed,f,n=4096,updates=800):
    h,targets,_=make(seed,n,532000)
    labels1=first_oracle(f,h,targets)
    g=Feedback(seed)
    rng=torch.Generator().manual_seed(seed+532001)
    opt=torch.optim.Adam(g.first_net.parameters(),lr=.006)
    features=torch.cat((h[:,2:],h[:,2:].abs()),1)
    for _ in range(updates):
        idx=torch.randint(0,n,(256,),generator=rng)
        loss=F.binary_cross_entropy_with_logits(g.first_net(features[idx]),labels1[idx])
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        first=step(f,h,g.first(h))
        labels2=second_oracle(f,first,targets[:,1]);features2=torch.cat((first,first.abs()),1)
    opt=torch.optim.Adam(g.second_net.parameters(),lr=.006)
    for _ in range(updates):
        idx=torch.randint(0,n,(256,),generator=rng)
        loss=F.binary_cross_entropy_with_logits(g.second_net(features2[idx]),labels2[idx])
        opt.zero_grad();loss.backward();opt.step()
    return g


def feedback_rollout(f,h,g,backend='independent'):
    m1=g.first(h);one=step(f,h,m1,0,backend)
    m2=g.second(one);two=step(f,one,m2,1,backend)
    return torch.stack((one[:,0],two[:,0]),1),torch.cat((m1,m2),1),one


def calibrate_rules(seed,f,g,n=1536):
    """Best source-support rule under feedback's calibration mean slot budget."""
    h,t,mode=make(seed,n,533000)
    with torch.no_grad():
        _,fbits,_=feedback_rollout(f,h,g)
        budget=float(fbits.sum(1).mean())
        all_preds=[success(two_step(f,h,p.expand(n,-1))[0],t) for p in PATTERNS]
        successes=[[int((all_preds[k]&(mode==j)).sum()) for k in range(16)] for j in range(3)]
        nn=[int((mode==j).sum()) for j in range(3)]
        costs=[int(x.sum()) for x in PATTERNS]
        best=(-1,-1,None)
        for i,j,k in itertools.product(range(16),repeat=3):
            cost=(nn[0]*costs[i]+nn[1]*costs[j]+nn[2]*costs[k])/n
            if cost>budget+1e-8:continue
            hit=successes[0][i]+successes[1][j]+successes[2][k]
            candidate=(hit,-cost,(i,j,k))
            if candidate>best:best=candidate
        assert best[2] is not None
        s=[int(p.sum()) for p in PATTERNS]
        a=[float(x.float().mean()) for x in all_preds]
        opt=(-1.,0,0,1.)
        for i in range(16):
            if s[i]<=budget and a[i]>opt[0]:opt=(a[i],i,i,1.)
            for j in range(i+1,16):
                if s[i]==s[j]:continue
                w=(budget-s[j])/(s[i]-s[j])
                if 0<=w<=1:
                    val=w*a[i]+(1-w)*a[j]
                    if val>opt[0]+1e-9:opt=(val,i,j,float(w))
        return {'budget':budget,'support_patterns':best[2],
                'support_cal_accuracy':best[0]/n,'support_cal_cost':-best[1],
                'static_patterns':[opt[1],opt[2]],'static_p':opt[3],
                'static_cal_accuracy':opt[0]}


def fixed_masks(seed,n,policy):
    g=torch.Generator().manual_seed(seed+535000)
    pick=torch.rand(n,generator=g)<policy['static_p']
    return torch.where(pick[:,None],PATTERNS[policy['static_patterns'][0]],PATTERNS[policy['static_patterns'][1]])


def permute_masks(seed,bits):
    g=torch.Generator().manual_seed(seed+536000)
    draws=torch.rand(len(bits),4,generator=g)
    rank=draws.argsort(1).argsort(1)
    return (rank<bits.sum(1,keepdim=True)).float()


def donor_assay(seed,f,g,backend='independent',n=128):
    rng=torch.Generator().manual_seed(seed+537000)
    a=torch.rand(n,generator=rng)*1.6-.8
    br=torch.rand(n,generator=rng)*.4-.2
    sign=torch.where(torch.rand(n,generator=rng)<.5,-1.,1.)
    bd=br+sign*(.30+torch.rand(n,generator=rng)*.15)
    hr=torch.zeros(n,4);hd=hr.clone()
    hr[:,2]=a;hd[:,2]=a;hr[:,3]=br;hd[:,3]=bd
    with torch.no_grad():
        yr,mr,x1=feedback_rollout(f,hr,g,backend)
        yd,md,d1=feedback_rollout(f,hd,g,backend)
        delta=((hd-hr)@f.V*mr[:,:2])@f.U
        dose=delta.norm(dim=1,keepdim=True)
        dark=delta.clone();dark[:,0]=0
        random=torch.randn(n,4,generator=rng);random[:,0]=0
        def matched(x):
            ln=x.norm(dim=1,keepdim=True)
            fallback=torch.zeros_like(x);fallback[:,1]=1
            return torch.where(ln>1e-8,x/ln.clamp_min(1e-8),fallback)*dose
        dark=matched(dark);random=matched(random)
        arms={'native':delta,'dark_native':dark,'random_dark':random,
              'sham':torch.zeros_like(delta),'full_state':d1-x1}
        report={'feedback':{},'frozen':{}}
        switch={};sham_mask=mr[:,2:]
        for key,add in arms.items():
            state=x1+add
            m2=g.second(state)
            switch[key]=int((m2!=sham_mask).any(1).sum())
            for mode,mask in [('feedback',m2),('frozen',sham_mask)]:
                pred=step(f,state,mask,1,backend)[:,0]
                report[mode][key]=int(((pred-bd).abs()<=TOL).sum())
        per_pair_err=(dark.norm(dim=1)-dose[:,0]).abs()
        per_pair_err=torch.maximum(per_pair_err,(random.norm(dim=1)-dose[:,0]).abs())
        clean=success(yr,torch.stack((a,br),1))&success(yd,torch.stack((a,bd),1))
        stable=(mr==md).all(1)
        return {'n':n,'clean_pairs':int(clean.sum()),'same_plan':int(stable.sum()),
                'clean_same_plan':int((clean&stable).sum()),'zero_dose':int((dose[:,0]<=1e-7).sum()),
                'mean_native_dose':float(dose.mean()),'max_equal_dose_error':float(per_pair_err.max()),
                'mean_native_current_readout_abs':float(delta[:,0].abs().mean()),
                'pass_two_switches':switch,'success_all_pairs':report}


def gauge_assay(seed,f,g,h,baseline_bits):
    """Fixed-plan canonical mask execution remains invariant after GL(2) shear."""
    with torch.no_grad():
        v=f.V.detach().clone().double();u=f.U.detach().clone().double();product=v@u
        mats=(torch.tensor([[1.,.6],[0.,1.]],dtype=torch.float64),
              torch.tensor([[1.,0.],[-.7,1.]],dtype=torch.float64))
        original_pred=two_step(f,h,baseline_bits)[0]
        v_raw=[];v_can=[];max_full=0.;gaps=[];mask_changes=[]
        for R in mats:
            vt=v@R;ut=torch.linalg.inv(R)@u
            max_full=max(max_full,float((product-vt@ut).abs().max()))
            x1=(h.double()@vt*baseline_bits[:,:2])@ut
            x2=(x1@vt*baseline_bits[:,2:])@ut
            v_raw.append(float((torch.stack((x1[:,0],x2[:,0]),1)-original_pred).abs().max()))
            uu,s,vh=torch.linalg.svd(vt@ut,full_matrices=False)
            root=s[:2].sqrt();vc=uu[:,:2]*root;uc=root[:,None]*vh[:2]
            cc1=(h.double()@vc*baseline_bits[:,:2])@uc
            recomputed=g.second(cc1.float())
            cc2=(cc1@vc*recomputed)@uc
            mask_changes.append(int((recomputed!=baseline_bits[:,2:]).any(1).sum()))
            v_can.append(float((torch.stack((cc1[:,0],cc2[:,0]),1)-original_pred).abs().max()))
            gaps.append(float(s[0]-s[1]))
        return {'max_full_operator_delta':max_full,'max_raw_masked_readout_delta':max(v_raw),
                'max_canonical_masked_readout_delta':max(v_can),
                'min_singular_gap':min(gaps),
                'max_feedback_mask_switches_after_canonicalization':max(mask_changes)}


def assay(seed,backend='independent',small=False):
    torch.set_num_threads(1);torch.manual_seed(seed)
    t=physical_fit(seed);f=factor_fit(seed,t,backend)
    geom=canonicalize(f)
    gate=train_feedback(seed,f,n=1536 if small else 4096,updates=240 if small else 800)
    pol=calibrate_rules(seed,f,gate,n=512 if small else 1536)
    h,target,mode=make(seed,128 if small else 256,534000)
    with torch.no_grad():
        fbpred,fbmask,_=feedback_rollout(f,h,gate,backend)
        support_mask=torch.stack([PATTERNS[pol['support_patterns'][int(m)]] for m in mode])
        masks={'feedback':fbmask,'support':support_mask,
               'static_mix':fixed_masks(seed,len(h),pol),'full':PATTERNS[-1].expand(len(h),-1)}
        masks['random_slots']=permute_masks(seed,fbmask)
        preds={'feedback':fbpred}
        for name in masks:
            if name!='feedback':preds[name]=two_step(f,h,masks[name],backend)[0]
        scores={name:{'correct':int(success(preds[name],target).sum()),
                      'mean_slots':float(masks[name].sum(1).mean()),
                      'by_source':[{'n':int((mode==k).sum()),'correct':int((success(preds[name],target)&(mode==k)).sum())} for k in range(3)]}
                for name in masks}
        gauge=gauge_assay(seed,f,gate,h,fbmask)
        donor=donor_assay(seed,f,gate,backend,n=32 if small else 128)
        return {'seed':seed,'backend':backend,'geometry':geom,'calibration':pol,
                'scores':scores,'gauge':gauge,'donor':donor}


def cohort(backend='independent',seeds=PILOT_SEEDS,small=False):
    rows=[assay(s,backend,small) for s in seeds]
    return {'schema':'bpd.cq05.feedback.gauge.pilot.v1','backend':backend,
            'seeds':list(seeds),'cases_per_seed':128 if small else 256,
            'rows':rows,'totals':{n:sum(row['scores'][n]['correct'] for row in rows) for n in rows[0]['scores']},
            'donor_totals':{mode:{name:sum(row['donor']['success_all_pairs'][mode][name] for row in rows)
                     for name in rows[0]['donor']['success_all_pairs'][mode]} for mode in ('feedback','frozen')},
            'pilot_only':True,'torch':torch.__version__,'python':platform.python_version()}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--backend',choices=('independent','native'),default='independent')
    a.add_argument('--engineering',action='store_true');a.add_argument('--out',type=Path)
    opt=a.parse_args();r=cohort(opt.backend,DEV_SEEDS[:2] if opt.engineering else PILOT_SEEDS,small=opt.engineering)
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
    for row in r['rows']:
        print(row['seed'],{k:v['correct'] for k,v in row['scores'].items()},row['donor']['success_all_pairs'])
    if opt.out:
        if opt.out.exists():raise FileExistsError(opt.out)
        opt.out.parent.mkdir(parents=True,exist_ok=True)
        opt.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
