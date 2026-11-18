"""AMM-06: budgeted per-case causal-probe features and independent outcome audit.

Three unlabeled continuation readouts per case are charged equally: one
unpatched reference and two component-patched continuations. Full donor
continuation and the independently defined donor goal are audit-only.
"""
from __future__ import annotations
import torch
from . import models as m

FEATURE_COUNT = 11


def candidate_features(f:m.Factors, rec:torch.Tensor, donor:torch.Tensor, family:str):
    """Two candidate vectors built without the external donor target."""
    with torch.no_grad():
        n=len(rec)
        r1=f(rec);d1=f(donor)
        full=d1-r1
        dose=full.norm(dim=1,keepdim=True)
        reference=f(r1)[:,0]
        geometry=m.geometry(f,family)
        values=[];probes=[];patches=[];degenerate=[]
        for c in (0,1):
            raw=((donor-rec)@f.V[:,c:c+1])@f.U[c:c+1]
            patch,deg=m.normalize_dark(raw,dose)
            probe=f(r1+patch)[:,0]
            response=probe-reference
            alignment=(patch*full).sum(dim=1)/(dose[:,0].square()+1e-8)
            normalized=response/(dose[:,0]+1e-8)
            features=torch.stack((
                torch.ones(n),r1[:,0],d1[:,0]-r1[:,0],dose[:,0],
                normalized,normalized.abs(),response,alignment,
                torch.full((n,),geometry['signed_gains'][c]),
                torch.full((n,),geometry['singular_gap']),
                torch.full((n,),float(family=='basalt'))),dim=1)
            values.append(features);probes.append(probe)
            patches.append(patch);degenerate.append(deg)
        return {'features':torch.stack(values,dim=1),
                'probes':torch.stack(probes,dim=1),
                'patches':patches,'geometry':geometry,
                'recipient_first':r1,'donor_first':d1,'full':full,
                'dose':dose,'reference':reference,'degenerate':degenerate}


def assay(phase:str,index:int,f:m.Factors,split:str,n:int):
    """The policy receives features only; labels and donor continuation are audit-only."""
    rec,donor,target=m.donors(phase,index,n,split)
    candidate=candidate_features(f,rec,donor,m.family_for(index))
    with torch.no_grad():
        p=candidate['probes'];dose=candidate['dose']
        bits={f'component_{c}':''.join(str(int(v)) for v in ((p[:,c]-target).abs()<=m.TOL))
              for c in (0,1)}
        active=[]
        for name in ('random_dark','random_dark_b'):
            vec=torch.randn((n,4),generator=m.gen(m.root_key(phase,index),split,name))
            patch,_=m.normalize_dark(vec,dose)
            bits[name]=''.join(str(int(v)) for v in
                ((f(candidate['recipient_first']+patch)[:,0]-target).abs()<=m.TOL))
            active.append(patch)
        bits['sham']=''.join(str(int(v)) for v in ((candidate['reference']-target).abs()<=m.TOL))
        donor_future=f(candidate['donor_first'])[:,0]  # audit-only oracle proxy
        bits['full_donor']=''.join(str(int(v)) for v in ((donor_future-target).abs()<=m.TOL))
        proxy_pick=(p-donor_future[:,None]).abs().argmin(dim=1)
        proxy_hits=sum(int(bits[f'component_{int(c)}'][i]=='1') for i,c in enumerate(proxy_pick))
        oracle_hits=sum((a=='1' or b=='1') for a,b in
                        zip(bits['component_0'],bits['component_1']))
        max_dose=max(float((v.norm(dim=1)-dose[:,0]).abs().max())
                     for v in candidate['patches']+active)
        max_visible=max(float(v[:,0].abs().max()) for v in candidate['patches']+active)
        return {'root':m.root_key(phase,index),'split':split,'n':n,
                'features':candidate['features'].tolist(),
                'candidate_bits':bits,
                'candidate_hits':{name:v.count('1') for name,v in bits.items()},
                'diagnostics':{'max_equal_dose_error':max_dose,
                               'max_immediate_readout_displacement':max_visible,
                               'zero_dose_pairs':int((dose[:,0]<=1e-8).sum()),
                               'degenerate_projection_pairs':candidate['degenerate'],
                               'mean_dose':float(dose.mean()),
                               'casewise_oracle_hits_audit_only':oracle_hits,
                               'model_proxy_hits_audit_only':proxy_hits,
                               'singular_gap':candidate['geometry']['singular_gap']}}
