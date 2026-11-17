"""AMM-02: cross-model strategy memory for trained BPD tied recurrences.

Pre-data protocol committed at ce390c8; independent external labels,
all-world ITT donor assay, phase-separated roots. Torch CPU only.
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

NAME = 'BPD-AMM-02'
SCHEMA = 'bpd.amm02.cross_model_memory.v1'
FAMILIES = ('wren', 'basalt')
PHASE_N = {'engineering': 8, 'training': 64, 'development': 64, 'confirmation': 128}
PAIRS = 48
STEPS_TEACHER = 200
STEPS_FACTORS = 140
LR = 0.035
TOL = 0.18
EPS = 1e-9
ARMS = ('memory', 'pooled', 'swapped', 'random', 'random_dark', 'random_dark_b', 'sham', 'full_donor')
COMPARISONS = {'memory_vs_pooled': 'pooled', 'memory_vs_swapped': 'swapped', 'memory_vs_random_dark': 'random_dark'}


def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(x: Any) -> str:
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def seed(*parts: object) -> int:
    return int(hashlib.sha256(('AMM02/'+'/'.join(map(str,parts))).encode()).hexdigest()[:15],16)


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


def donors(phase:str,index:int):
    fam=family_for(index)
    root=root_key(phase,index)
    g=gen(root,'donor_pairs')
    other=torch.rand(PAIRS,generator=g)*1.6-.8
    v=torch.rand(PAIRS,generator=g)*.4-.2
    direction=torch.where(torch.rand(PAIRS,generator=g)<.5,-1.,1.)
    dv=v+direction*(.30+torch.rand(PAIRS,generator=g)*.15)
    r=torch.zeros((PAIRS,4));d=r.clone()
    if fam=='wren':
        r[:,2],d[:,2]=other,other
        r[:,3],d[:,3]=v,dv
    else:
        r[:,2],d[:,2]=v,dv
        r[:,3],d[:,3]=other,other
    external_target=dv
    return r,d,external_target


def normalize_dark(x:torch.Tensor,dose:torch.Tensor):
    raw=x.clone()
    raw[:,0]=0
    n=raw.norm(dim=1,keepdim=True)
    fallback=torch.zeros_like(raw);fallback[:,1]=1.
    result=torch.where(n>1e-8,raw/n.clamp_min(1e-8),fallback)*dose
    return result, int(((n[:,0]<=1e-8)&(dose[:,0]>1e-8)).sum())


def candidate_assay(phase:str,index:int,f:Factors,m:Physical,metrics:dict):
    """Independent behavioral label; same one-shot dose for both BPD actions."""
    fam=family_for(index)
    hr,hd,target=donors(phase,index)
    with torch.no_grad():
        h1r=f(hr);h1d=f(hd)
        delta=h1d-h1r
        D=delta.norm(dim=1,keepdim=True)
        components=[];zeros=[]
        for c in (0,1):
            raw=((hd-hr)@f.V[:,c:c+1])@f.U[c:c+1]
            x,n_zero=normalize_dark(raw,D)
            components.append(x);zeros.append(n_zero)
        rn=torch.randn((PAIRS,4),generator=gen(root_key(phase,index),'random_dark_a'))
        random_dark,_=normalize_dark(rn,D)
        rn2=torch.randn((PAIRS,4),generator=gen(root_key(phase,index),'random_dark_b'))
        random_dark_b,_=normalize_dark(rn2,D)
        zero=torch.zeros_like(delta)
        arms={
            'component_0':components[0],
            'component_1':components[1],
            'random_dark':random_dark,
            'random_dark_b':random_dark_b,
            'sham':zero,
            'full_donor':delta,
        }
        # All active arms use current-readout dark vectors and same physical dose.
        err=max(float((patch.norm(dim=1)-D[:,0]).abs().max()) for key,patch in arms.items() if key in ('component_0','component_1','random_dark','random_dark_b'))
        dark_error=max(float(patch[:,0].abs().max()) for key,patch in arms.items() if key in ('component_0','component_1','random_dark','random_dark_b'))
        pred={key:f(h1r+patch)[:,0] for key,patch in arms.items()}
        bits={key:((y-target).abs()<=TOL).int().tolist() for key,y in pred.items()}
        test0,test1=labels(hr[:,2:],fam)
        testd0,testd1=labels(hd[:,2:],fam)
        clean=(f(hr)[:,0]-test0).abs()<=TOL
        clean&=(f(hd)[:,0]-testd0).abs()<=TOL
        clean&=(f(f(hr))[:,0]-test1).abs()<=TOL
        clean&=(f(f(hd))[:,0]-testd1).abs()<=TOL
        native_delta=float(D.mean())
        return {'root':root_key(phase,index),'family':fam,'n':PAIRS,
                'candidate_bits':{key:''.join(map(str,b)) for key,b in bits.items()},
                'candidate_hits':{key:sum(b) for key,b in bits.items()},
                'teacher_factor_diagnostics':metrics,
                'pair_diagnostics':{'clean_both':int(clean.sum()),
                                    'max_active_dose_error':err,'max_active_current_readout_error':dark_error,
                                    'zero_dose':int((D[:,0]<=1e-8).sum()),
                                    'degenerate_component_projected':zeros,
                                    'mean_dose':native_delta,
                                    'full_donor_current_visible_mean_abs':float(delta[:,0].abs().mean())}}


def fit_one_root(phase:str,index:int):
    m,f,metrics=fit(phase,index)
    return candidate_assay(phase,index,f,m,metrics)


def training_memory():
    rows=[fit_one_root('training',i) for i in range(PHASE_N['training'])]
    hits={fam:{str(i):sum(r['candidate_hits'][f'component_{i}'] for r in rows if r['family']==fam) for i in (0,1)} for fam in FAMILIES}
    assignment={fam: (0 if hits[fam]['0']>=hits[fam]['1'] else 1) for fam in FAMILIES}
    p0=sum(hits[fam]['0'] for fam in FAMILIES);p1=sum(hits[fam]['1'] for fam in FAMILIES)
    record={'phase':'training','schema':SCHEMA+'/training-memory','root_count':PHASE_N['training'],
            'pairs_per_root':PAIRS,'family_counts':{fam:sum(r['family']==fam for r in rows) for fam in FAMILIES},
            'action_hits':hits,'family_action':assignment,'pooled_action':0 if p0>=p1 else 1,
            'rows':rows,'experiment_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    record['sha256']=digest(record)
    return record


def check_memory(memory:dict):
    if memory.get('schema')!=SCHEMA+'/training-memory' or memory.get('root_count')!=PHASE_N['training']:
        raise ValueError('incompatible training memory')
    old=memory.copy();h=old.pop('sha256',None)
    if h!=digest(old) or len(memory.get('rows',[]))!=PHASE_N['training']:
        raise ValueError('training memory integrity failure')
    if memory.get('experiment_source_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('training memory source revision mismatch')
    if {r.get('root') for r in memory['rows']}!={root_key('training',i) for i in range(PHASE_N['training'])}:
        raise ValueError('training root provenance mismatch')


def choose(arm:str,memory:dict,family:str,root:str,case:int):
    if arm=='memory':return memory['family_action'][family]
    if arm=='pooled':return memory['pooled_action']
    if arm=='swapped':return memory['family_action'][FAMILIES[1-FAMILIES.index(family)]]
    if arm=='random':return seed(root,'random_component',case)%2
    raise ValueError('unsupported policy')


def score_row(raw:dict,memory:dict):
    x=raw['candidate_bits']; fam=raw['family']; root=raw['root']
    out={};bits={}
    for arm in ('memory','pooled','swapped','random'):
        z=''.join(x[f'component_{choose(arm,memory,fam,root,c)}'][c] for c in range(PAIRS))
        bits[arm]=z
    for arm in ('random_dark','random_dark_b','sham','full_donor'):
        bits[arm]=x[arm]
    for arm in ARMS:
        out[arm]=bits[arm].count('1')
    return {**raw,'policy_hits':out,'policy_bits':bits,
            'policy_action':{arm:(choose(arm,memory,fam,root,0) if arm!='random' else None) for arm in ('memory','pooled','swapped','random')}}


def admission(phase:str,memory:dict|None,preflight:dict|None):
    """Must run BEFORE any root or train function for sealed confirmation."""
    if phase not in ('development','confirmation'):
        raise ValueError('unknown evaluation phase')
    if memory is None:
        raise ValueError('frozen training memory required')
    check_memory(memory)
    if phase=='confirmation':
        if not isinstance(preflight,dict) or any((
            preflight.get('decision')!='READY_FOR_CONFIRMATION',
            preflight.get('experiment_id')!=NAME,
            preflight.get('planned_confirmation_worlds')!=PHASE_N['confirmation'],
            preflight.get('planned_comparisons')!=len(COMPARISONS),
            preflight.get('pilot_worlds')!=PHASE_N['development'],
            preflight.get('upper_sham_quantile') is None,
            bool(preflight.get('reasons')),
        )):
            raise ValueError('strict EvalMath independent-development preflight required before any confirmation read')


def evaluate(phase:str,memory:dict|None=None,preflight:dict|None=None):
    admission(phase,memory,preflight)
    assert memory is not None
    rows=[score_row(fit_one_root(phase,i),memory) for i in range(PHASE_N[phase])]
    result={'schema':SCHEMA+'/cohort','phase':phase,'n_independent_roots':PHASE_N[phase],
            'n_pairs_per_root':PAIRS,'total_cases':PAIRS*PHASE_N[phase],
            'training_memory_sha256':memory['sha256'],
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'totals':{a:sum(r['policy_hits'][a] for r in rows) for a in ARMS},
            'rows':rows}
    if phase=='development':result['preflight_input']=preflight_input(result)
    else:
        result['preflight_sha256']=digest(preflight)
        result['analysis']=analysis(result,preflight)
    return result


def preflight_input(run:dict):
    if run['phase']!='development':raise ValueError('preflight needs development roots')
    rows=[]
    for r in run['rows']:
        p=r['policy_hits']
        rows.append({'unit_id':r['root'],
                     'paired_effect':f"{p['memory']-p['pooled']}/{PAIRS}",
                     'sham_effects':[f"{p['random_dark']-p['random_dark_b']}/{PAIRS}"]})
    return {'schema_version':1,'experiment_id':NAME,
            'independent_unit':'one separately trained physical recurrent model and BPD factor root',
            'alpha':'0.05','scientific_min_effect':'0.12','target_effect':'0.20',
            'measurement_resolution':f'1/{PAIRS}',
            'sham_quantile':'0.95','sham_confidence':'0.95',
            'planned_comparisons':len(COMPARISONS),'confirm_units_planned':PHASE_N['confirmation'],
            'pilot_from_development_only':True,'heldout_sealed':True,'units_independent':True,'pilot':rows}


def root_sign_p(xs:list[int]):
    k=sum(x>0 for x in xs);n=sum(x!=0 for x in xs)
    if n==0:return 1.0
    # SF exact upper tail, stable for n<=128.
    return sum(math.comb(n,j) for j in range(k,n+1))/2**n


def holm(ps:dict[str,float]):
    out={};floor=0.
    for i,key in enumerate(sorted(ps,key=lambda k:(ps[k],k))):
        floor=max(floor,min(1.,ps[key]*(len(ps)-i)))
        out[key]=floor
    return out


def bootstrap(xs:list[int],key:str):
    r=random.Random(seed('bootstrap',key));n=len(xs)
    values=sorted(sum(xs[r.randrange(n)] for _ in range(n))/(PAIRS*n) for _ in range(2000))
    return [values[50],values[1949]]


def analysis(cohort:dict,pf:dict):
    if cohort['phase']!='confirmation':raise ValueError('confirmation only')
    diffs={key:[r['policy_hits']['memory']-r['policy_hits'][control] for r in cohort['rows']] for key,control in COMPARISONS.items()}
    p={key:root_sign_p(xs) for key,xs in diffs.items()}; adjusted=holm(p)
    contrasts={key:{'root_mean_effect':sum(xs)/(len(xs)*PAIRS),
                    'root_bootstrap_95_ci':bootstrap(xs,key),
                    'positive_roots':sum(x>0 for x in xs),
                    'negative_roots':sum(x<0 for x in xs),
                    'tied_roots':sum(x==0 for x in xs),
                    'exact_one_sided_root_sign_p':p[key],
                    'holm_p':adjusted[key]} for key,xs in diffs.items()}
    floor=max(.12,pf['mechanistic_variation_floor'])
    primary=contrasts['memory_vs_pooled']
    teacher_all=[r['teacher_factor_diagnostics']['teacher_behavioral_mse'] for r in cohort['rows']]
    factor_all=[r['teacher_factor_diagnostics']['factor_behavioral_mse'] for r in cohort['rows']]
    maxdose=max(r['pair_diagnostics']['max_active_dose_error'] for r in cohort['rows'])
    maxdark=max(r['pair_diagnostics']['max_active_current_readout_error'] for r in cohort['rows'])
    return {'schema':SCHEMA+'/analysis','evidence':'trained synthetic linear BPD models; no real LLM agents',
            'comparisons':contrasts,'mechanistic_variation_floor':floor,
            'max_equal_dose_error':maxdose,'max_current_readout_displacement':maxdark,
            'teacher_mse_range':[min(teacher_all),max(teacher_all)],
            'factor_mse_range':[min(factor_all),max(factor_all)],
            'primary_gate':bool(primary['root_bootstrap_95_ci'][0]>floor and primary['holm_p']<.05 and maxdose<=1e-5 and maxdark<=1e-5)}


def small_summary(o:dict):
    if o.get('phase')=='training':return {'kind':'training','family_action':o['family_action'],'pooled_action':o['pooled_action']}
    return {'phase':o['phase'],'roots':o['n_independent_roots'],'cases':o['total_cases'],'totals':o['totals'],
            'preflight_needed':o.get('phase')=='development',
            'primary_gate':o.get('analysis',{}).get('primary_gate')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('engineering','training','development','confirmation'),required=True)
    p.add_argument('--memory',type=Path)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.phase=='training':o=training_memory()
    elif a.phase=='engineering':
        o={'phase':'engineering','rows':[fit_one_root('engineering',i) for i in range(2)]}
    else:
        if not a.memory:raise ValueError('--memory required')
        m=json.loads(a.memory.read_text())
        pf=json.loads(a.preflight.read_text()) if a.preflight else None
        o=evaluate(a.phase,m,pf)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open('x') as f:
            json.dump(o,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(small_summary(o),sort_keys=True))


if __name__=='__main__':main()
