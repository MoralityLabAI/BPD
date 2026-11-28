"""AMM-11: fresh rank-three recurrent worlds for causal cosine competition.

Protocol registered before source in PROTOCOL.md. Engineering roots only
for unit tests and architecture checks. Source targets never enter policy.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
from pathlib import Path
from functools import lru_cache
from itertools import combinations
from typing import Any
import torch
from torch import nn

NAME="BPD-AMM-11"
SCHEMA="bpd.amm11.cosine_transfer.v1"
D=6
R=3
PHASE_N={"engineering":8,"training":64,"development":64,"confirmation":128}
FAMILIES=("wren","basalt")
ORDER={"wren":(0,1,2),"basalt":(2,0,1)}
MASKS=tuple(sub for k in (1,2) for sub in combinations(range(R),k))
TRAIN_PAIRS=128
CAL_PAIRS=16
PAIRS=512
TEACHER_STEPS=400
FACTOR_STEPS=240
TOL=.18
LR=.035
FEATURES=18
ARMS=("full_memory","cosine","compact_memory","pooled",
      "local_ridge","memory_plus_local","max_cardinality","random_mask",
      "random_dark","random_dark_b","sham","full_donor")
COMPARISONS=(("full_vs_cosine","full_memory","cosine"),
             ("compact_vs_cosine","compact_memory","cosine"),
             ("full_vs_compact","full_memory","compact_memory"),
             ("full_vs_pooled","full_memory","pooled"),
             ("local_update_vs_full","memory_plus_local","full_memory"))

ROOT=Path(__file__).resolve().parent
PINNED_ENV={
 "ATEN_CPU_CAPABILITY":"default","MKL_CBWR":"COMPATIBLE",
 "MKL_DYNAMIC":"FALSE","MKL_NUM_THREADS":"1","OMP_DYNAMIC":"FALSE",
 "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PYTHONHASHSEED":"0"}

@lru_cache(maxsize=1)
def _binary_hashes():
 def h(p):
  sha=hashlib.sha256()
  with Path(p).open("rb") as f:
   for chunk in iter(lambda:f.read(1024*1024),b""):sha.update(chunk)
  return sha.hexdigest()
 return {"_C.so":h(torch._C.__file__),
         "libtorch_cpu.so":h(Path(torch.__file__).parent/"lib"/"libtorch_cpu.so")}

def runtime_signature():
 expected=json.loads((ROOT/"RUNTIME.json").read_text())
 import numpy
 versions={"python":platform.python_version(),"torch":torch.__version__,
           "numpy":numpy.__version__}
 if versions!=expected["versions"] or torch.version.git_version!=expected["torch_git"]:
  raise ValueError("AMM11 frozen CPU package versions/source required")
 if expected["environment"]!=PINNED_ENV or any(os.environ.get(k)!=v for k,v in PINNED_ENV.items()):
  raise ValueError("AMM11 use source-locked deterministic CPU environment")
 if _binary_hashes()!=expected["torch_binary_sha256"]:
  raise ValueError("AMM11 torch CPU binary checksum mismatch")
 if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
  raise ValueError("AMM11 one CPU worker thread and one interop thread required")
 if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
  raise ValueError("AMM11 deterministic operations must fail closed")
 if torch.get_default_dtype()!=torch.float32 or torch.backends.cpu.get_cpu_capability() not in ("NO AVX","DEFAULT"):
  raise ValueError("AMM11 float32 generic CPU dispatch required")
 return {"versions":versions,"environment":dict(PINNED_ENV),
         "torch_git":torch.version.git_version,
         "torch_binary_sha256":_binary_hashes(),
         "cpu_capability":torch.backends.cpu.get_cpu_capability(),
         "deterministic":True}

def canonical(value:Any)->str:
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)

def digest(value:Any)->str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def seed(*parts:object)->int:
    return int(hashlib.sha256(("AMM11/"+"/".join(map(str,parts))).encode()).hexdigest()[:15],16)

def gen(*parts:object)->torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed(*parts))

def root_key(phase:str,index:int)->str:
    if phase=="confirmation":
        raise ValueError("AMM11 confirmation namespace sealed")
    if phase not in PHASE_N or not 0<=index<PHASE_N[phase]:
        raise ValueError("unregistered model root")
    return f"{phase}/{index:04d}"

def family_for(index:int)->str:
    return FAMILIES[index%2]

def labels(src:torch.Tensor,family:str)->tuple[torch.Tensor,...]:
    return tuple(src[:,i] for i in ORDER[family])

def data(phase:str,index:int,n:int,stream:str):
    g=gen(root_key(phase,index),stream)
    src=torch.rand((n,R),generator=g)*1.6-.8
    h=torch.zeros(n,D)
    h[:,3:]=src
    return h,src

class Physical(nn.Module):
    def __init__(self,phase:str,index:int):
        super().__init__()
        g=gen(root_key(phase,index),"teacher_initialization")
        self.I=nn.Parameter(torch.randn(D,R,generator=g)*.45)
        self.O=nn.Parameter(torch.randn(R,D,generator=g)*.45)
    def forward(self,h):return (h@self.I)@self.O
    def weight(self):return (self.I@self.O).T

class Factors(nn.Module):
    def __init__(self,phase:str,index:int):
        super().__init__()
        g=gen(root_key(phase,index),"bpd_initialization")
        self.V=nn.Parameter(torch.randn(D,R,generator=g)*.35)
        self.U=nn.Parameter(torch.randn(R,D,generator=g)*.35)
    def forward(self,h):return (h@self.V)@self.U
    def weight(self):return (self.V@self.U).T

def rollout(f,h):
    a=f(h);b=f(a);c=f(b)
    return (a,b,c)

def fit(phase:str,index:int):
    """Independently optimize physical teacher, then independent BPD factors."""
    torch.set_num_threads(1)
    root=root_key(phase,index)
    runtime_signature()
    family=family_for(index)
    teacher=Physical(phase,index)
    initial=(teacher.I.detach().clone(),teacher.O.detach().clone())
    optimizer=torch.optim.Adam(teacher.parameters(),lr=LR)
    g=gen(root,"teacher_train")
    for _ in range(TEACHER_STEPS):
        src=torch.rand((96,R),generator=g)*1.6-.8
        h=torch.zeros((96,D));h[:,3:]=src
        target=labels(src,family)
        preds=rollout(teacher,h)
        loss=sum((preds[k][:,0]-target[k]).square().mean() for k in range(3))
        loss=loss+.0002*teacher.weight().square().mean()
        if not bool(torch.isfinite(loss)):raise ArithmeticError("teacher training nonfinite")
        optimizer.zero_grad();loss.backward();optimizer.step()
    teacher_update=[float((param.detach()-before).norm()) for param,before
                    in zip((teacher.I,teacher.O),initial)]
    factor=Factors(phase,index)
    factor_initial=(factor.V.detach().clone(),factor.U.detach().clone())
    optimizer2=torch.optim.Adam(factor.parameters(),lr=LR)
    g2=gen(root,"factor_train")
    frozen_weight=teacher.weight().detach()
    for _ in range(FACTOR_STEPS):
        src=torch.rand((96,R),generator=g2)*1.6-.8
        h=torch.zeros((96,D));h[:,3:]=src
        with torch.no_grad(): references=rollout(teacher,h)
        outputs=rollout(factor,h)
        loss=sum((outputs[k]-references[k]).square().mean() for k in range(3))
        loss=loss+.5*(factor.weight()-frozen_weight).square().mean()
        if not bool(torch.isfinite(loss)):raise ArithmeticError("factor training nonfinite")
        optimizer2.zero_grad();loss.backward();optimizer2.step()
    factor_update=[float((param.detach()-before).norm()) for param,before
                   in zip((factor.V,factor.U),factor_initial)]
    with torch.no_grad():
        full=factor.V.double()@factor.U.double()
        left,s,right=torch.linalg.svd(full,full_matrices=False)
        root_s=s[:R].clamp_min(0).sqrt()
        vc=left[:,:R]*root_s[None,:]
        uc=root_s[:,None]*right[:R,:]
        reconstruction_error=float((full-vc@uc).abs().max())
        factor.V.copy_(vc.float());factor.U.copy_(uc.float())
        h,src=data(phase,index,256,"source_validation")
        targets=labels(src,family)
        teacher_pred=rollout(teacher,h)
        factor_pred=rollout(factor,h)
        def mse(outputs):
            return float(sum((outputs[i][:,0]-targets[i]).square().mean() for i in range(3))/3)
        gaps=(float(s[0]-s[1]),float(s[1]-s[2]))
        metrics={
            "teacher_update_norms":teacher_update,
            "factor_update_norms":factor_update,
            "teacher_task_mse":mse(teacher_pred),
            "factor_task_mse":mse(factor_pred),
            "weight_reconstruction_relative":float((factor.weight()-frozen_weight).norm()/
                frozen_weight.norm().clamp_min(1e-8)),
            "rank3_singular_values":[float(x) for x in s[:R]],
            "min_rank3_gap":min(gaps),
            "canonical_reconstruction_max_abs_error":reconstruction_error,
            "teacher_weight_sha256":hashlib.sha256(frozen_weight.cpu().numpy().tobytes()).hexdigest(),
            "factor_weight_sha256":hashlib.sha256(factor.weight().cpu().numpy().tobytes()).hexdigest()}
    return teacher,factor,metrics

def donor_pairs(phase:str,index:int,n:int,split:str):
    if n not in (TRAIN_PAIRS,CAL_PAIRS,PAIRS):
        raise ValueError("unregistered pair count")
    if split not in ("training_pairs","calibration","heldout","repeatability","engineering"):
        raise ValueError("unregistered pair split")
    family=family_for(index)
    g=gen(root_key(phase,index),split)
    src=torch.rand((n,R),generator=g)*.6-.3
    magnitude=torch.rand((n,R),generator=g)
    magnitude[:,0]=.15+.55*magnitude[:,0]
    magnitude[:,1:]=.25+.20*magnitude[:,1:]
    sign=torch.where(torch.rand((n,R),generator=g)<.5,-1.,1.)
    donor=src.clone()
    for k,column in enumerate(ORDER[family]):
        donor[:,column]+=sign[:,k]*magnitude[:,k]
    recipient_h=torch.zeros((n,D));recipient_h[:,3:]=src
    donor_h=torch.zeros((n,D));donor_h[:,3:]=donor
    target=torch.stack((donor[:,ORDER[family][1]],donor[:,ORDER[family][2]]),dim=1)
    return recipient_h,donor_h,target

def normalize_dark(raw:torch.Tensor,dose:torch.Tensor):
    dark=raw.clone();dark[:,0]=0
    length=dark.norm(dim=1,keepdim=True)
    fallback=torch.zeros_like(dark);fallback[:,1]=1
    unit=torch.where(length>1e-8,dark/length.clamp_min(1e-8),fallback)
    patch=unit*dose
    degenerate=int(((length[:,0]<=1e-8)&(dose[:,0]>1e-8)).sum())
    return patch,degenerate

def gauge_check(f:Factors):
    """GL3 shear keeps physical full W and recanonicalized rank-one writes."""
    with torch.no_grad():
        V=f.V.detach().double();U=f.U.detach().double();W=V@U
        shear=torch.tensor([[1.,.6,0.],[0.,1.,.3],[.2,0.,1.]],dtype=torch.float64)
        transformed=(V@shear)@(torch.linalg.inv(shear)@U)
        def components(W):
            left,s,right=torch.linalg.svd(W,full_matrices=False)
            return [left[:,i:i+1]@right[i:i+1]*s[i] for i in range(R)]
        before=components(W);after=components(transformed)
        return {"max_full_gl3_error":float((W-transformed).abs().max()),
                "max_canonical_rankone_error":max(float((a-b).abs().max())
                         for a,b in zip(before,after)),
                "min_singular_gap":min(float(torch.linalg.svdvals(W)[j]-
                                             torch.linalg.svdvals(W)[j+1]) for j in range(2))}
