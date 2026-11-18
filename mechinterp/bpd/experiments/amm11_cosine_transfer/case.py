"""AMM-11: six bounded two-slot canonical interventions and two delayed readouts.

The policy sees features constructed from the current physical states and
unpatched/candidate continuations. The donor target and full donor future are
held solely by scoring/audit code, never passed to any policy function.
"""
from __future__ import annotations
import torch
from . import models as m

def candidate_features(f:m.Factors,recipient:torch.Tensor,donor:torch.Tensor,family:str):
    """Six admissible rank<=2 component masks with same physical dose."""
    with torch.no_grad():
        n=len(recipient)
        r1=f(recipient);d1=f(donor)
        full=d1-r1
        dose=full.norm(dim=1,keepdim=True)
        ref2=f(r1);ref3=f(ref2)
        spectrum=torch.linalg.svdvals(f.V.double()@f.U.double())[:m.R]
        smallest_gap=float(min(spectrum[0]-spectrum[1],spectrum[1]-spectrum[2]))
        relative_gap=smallest_gap/(float(spectrum[0])+1e-8)
        kind=float(family=="basalt")
        cardinal=[];masks=[];probes=[];degenerate=[]
        for subset in m.MASKS:
            raw=((donor-recipient)@f.V[:,list(subset)])@f.U[list(subset),:]
            patch,deg=m.normalize_dark(raw,dose)
            out2=f(r1+patch)
            out3=f(out2)
            shift2=out2[:,0]-ref2[:,0]
            shift3=out3[:,0]-ref3[:,0]
            cos=(patch*full).sum(dim=1)/(dose[:,0].square()+1e-8)
            features=torch.stack((
                torch.ones(n),torch.full((n,),len(subset)/3.),
                r1[:,0],d1[:,0]-r1[:,0],dose[:,0],
                shift2,shift3,shift2.abs(),shift3.abs(),
                shift2.square(),shift3.square(),shift2*shift3,
                out2[:,0],out3[:,0],cos,
                torch.full((n,),relative_gap),
                (d1[:,0]-r1[:,0]).abs()/(dose[:,0]+1e-8),
                torch.full((n,),kind)),dim=1)
            cardinal.append(features)
            masks.append(patch)
            probes.append(torch.stack((out2[:,0],out3[:,0]),dim=1))
            degenerate.append(deg)
        return {"features":torch.stack(cardinal,dim=1),
                "probes":torch.stack(probes,dim=1),
                "patches":masks,"degenerate_masks":degenerate,
                "recipient_first":r1,"donor_first":d1,
                "full_first_delta":full,"dose":dose,
                "reference_two":torch.stack((ref2[:,0],ref3[:,0]),dim=1),
                "min_singular_gap":smallest_gap}

def score(phase:str,index:int,f:m.Factors,split:str,n:int):
    rec,don,target=m.donor_pairs(phase,index,n,split)
    features=candidate_features(f,rec,don,m.family_for(index))
    if features["features"].shape!=(n,len(m.MASKS),m.FEATURES):
        raise ValueError("fixed 18-feature/6-action two-slot protocol broken")
    with torch.no_grad():
        yy=features["probes"]
        success=((yy-target[:,None,:]).abs()<=m.TOL).all(dim=2)
        bits={f"mask_{i}":"".join(str(int(b)) for b in success[:,i])
              for i in range(len(m.MASKS))}
        random_patches=[]
        for name in ("random_dark","random_dark_b"):
            z=torch.randn((n,m.D),generator=m.gen(m.root_key(phase,index),split,name))
            patch,_=m.normalize_dark(z,features["dose"])
            random_patches.append(patch)
            p2=f(features["recipient_first"]+patch);p3=f(p2)
            result=((torch.stack((p2[:,0],p3[:,0]),dim=1)-target).abs()<=m.TOL).all(1)
            bits[name]="".join(str(int(b)) for b in result)
        sham=((features["reference_two"]-target).abs()<=m.TOL).all(1)
        bits["sham"]="".join(str(int(b)) for b in sham)
        # Full donor continuation is an AUDIT/CEILING label and is NEVER an agent input.
        full2=f(features["donor_first"]);full3=f(full2)
        full=((torch.stack((full2[:,0],full3[:,0]),dim=1)-target).abs()<=m.TOL).all(1)
        bits["full_donor"]="".join(str(int(b)) for b in full)
        active=features["patches"]+random_patches
        maxnorm=max(float((p.norm(dim=1)-features["dose"][:,0]).abs().max())
                    for p in active)
        maxvisible=max(float(p[:,0].abs().max()) for p in active)
        return {"root":m.root_key(phase,index),"split":split,"n":n,
                "features":features["features"].tolist(),
                "candidate_bits":bits,
                "candidate_hits":{name:value.count("1") for name,value in bits.items()},
                "diagnostics":{
                    "casewise_oracle_hits_audit_only":int(success.any(dim=1).sum()),
                    "full_donor_hits_audit_only":int(full.sum()),
                    "max_equal_dose_error":maxnorm,
                    "max_immediate_readout_displacement":maxvisible,
                    "zero_dose_pairs":int((features["dose"][:,0]<=1e-8).sum()),
                    "degenerate_mask_projection_pairs":features["degenerate_masks"],
                    "min_singular_gap":features["min_singular_gap"],
                    "mean_dose":float(features["dose"].mean())}}
