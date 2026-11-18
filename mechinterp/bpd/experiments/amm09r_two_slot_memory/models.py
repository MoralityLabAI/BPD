"""AMM-09R1: six admissible two-slot masks in trained rank-three recurrence.

Governed by PROTOCOL.md, with source locked before training. Engineering roots
are excluded from inference. Source targets never enter policy.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
from functools import lru_cache
from pathlib import Path
from itertools import combinations
from typing import Any
import torch
from torch import nn

NAME="BPD-AMM-09R1"
SCHEMA="bpd.amm09r1.two_slot_memory.v1"
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
ARMS=("memory_plus_local","local_ridge","memory_only","local_count",
      "pooled","random_mask","random_dark","random_dark_b","sham","full_donor")
COMPARISONS=(
    ("memory_only_vs_pooled", "memory_only", "pooled"),
    ("memory_plus_local_vs_pooled", "memory_plus_local", "pooled"),
    ("memory_plus_local_vs_local_ridge", "memory_plus_local", "local_ridge"),
    ("memory_plus_local_vs_memory_only", "memory_plus_local", "memory_only"),
)
ROOT=Path(__file__).resolve().parent
PINNED_ENV={
    "ATEN_CPU_CAPABILITY":"default", "MKL_CBWR":"COMPATIBLE",
    "MKL_NUM_THREADS":"1", "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1",
    "MKL_DYNAMIC":"FALSE", "OMP_DYNAMIC":"FALSE", "PYTHONHASHSEED":"0",
}


def canonical(value:Any)->str:
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)

def digest(value:Any)->str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_sha256()->dict[str,str]:
    names=("__init__.py","models.py","case.py","experiment.py","run.py","audit.py",
           "test_experiment.py","RUNTIME.json","PROTOCOL.md",
           "../../../../.github/workflows/bpd-amm09r1-fresh-development.yml")
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def require_runtime()->None:
    import numpy
    expected=json.loads((ROOT/"RUNTIME.json").read_text())
    versions={"python":platform.python_version(), "torch":torch.__version__, "numpy":numpy.__version__}
    if versions!=expected["versions"] or expected["environment"]!=PINNED_ENV:
        raise ValueError("registered AMM09R1 runtime versions required")
    if any(os.environ.get(key)!=value for key,value in PINNED_ENV.items()):
        raise ValueError("launch run.py in a fresh process with the registered CPU environment")
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
        raise ValueError("AMM09R1 requires one intra-op and one inter-op thread")
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("AMM09R1 requires deterministic algorithms")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError("AMM09R1 nondeterministic operations must fail")
    if torch.get_default_dtype()!=torch.float32 or torch.backends.cpu.get_cpu_capability() not in ("NO AVX","DEFAULT"):
        raise ValueError("registered float32/default CPU dispatch required")
    if torch.version.git_version!=expected["torch_git"]:
        raise ValueError("torch source differs from accepted engineering runtime")
    if _binary_hashes()!=expected["torch_binary_sha256"]:
        raise ValueError("torch binary bytes differ from accepted engineering runtime")


@lru_cache(maxsize=1)
def _binary_hashes()->dict[str,str]:
    def file_hash(path):
        digest=hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""):
                digest.update(chunk)
        return digest.hexdigest()
    return {"libtorch_cpu.so":file_hash(Path(torch.__file__).parent/"lib"/"libtorch_cpu.so"),
            "_C.so":file_hash(torch._C.__file__)}


def runtime_signature()->dict:
    require_runtime()
    import numpy
    return {"python":platform.python_version(), "torch":torch.__version__,
            "numpy":numpy.__version__, "torch_git":torch.version.git_version,
            "torch_binary_sha256":dict(_binary_hashes()), "environment":dict(PINNED_ENV),
            "num_threads":torch.get_num_threads(),
            "num_interop_threads":torch.get_num_interop_threads(),
            "default_dtype":str(torch.get_default_dtype()),
            "cpu_capability":torch.backends.cpu.get_cpu_capability(),
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled()}


def require_development_execution()->None:
    """Admit the one prospectively recorded first Actions attempt only."""
    try:
        manifest=json.loads((ROOT/"results"/"TRAINING_RECEIPTS.json").read_text())
        allowed=manifest["development_execution"]
    except (OSError,ValueError,KeyError) as error:
        raise ValueError("registered single development execution required") from error
    if (not isinstance(allowed,dict)
            or allowed.get("workflow")!="BPD-AMM-09R1 prospective development"
            or allowed.get("event")!="pull_request"
            or type(allowed.get("run_number")) is not int or allowed["run_number"]<1
            or allowed.get("run_attempt")!=1):
        raise ValueError("invalid registered development execution")
    expected={"GITHUB_WORKFLOW":allowed["workflow"],"GITHUB_EVENT_NAME":allowed["event"],
              "GITHUB_RUN_NUMBER":str(allowed["run_number"]),"GITHUB_RUN_ATTEMPT":"1"}
    if any(os.environ.get(key)!=value for key,value in expected.items()):
        raise ValueError("development execution is sealed outside its registered first attempt")


def require_phase_admission(phase:str,*,generate:bool=True)->str|None:
    if phase=="confirmation":
        raise ValueError("AMM09R1 confirmation generation is sealed")
    if phase=="engineering":
        return None
    if phase not in ("training","development"):
        raise ValueError("unregistered phase")
    if generate and (ROOT/"results"/"DEVELOPMENT_LOCK.json").exists():
        raise ValueError("development is already locked; use archived receipt audit")
    path=ROOT/"results"/"ENGINEERING_ACCEPTANCE.json"
    try:
        raw=path.read_bytes()
        expected=json.loads((ROOT/"RUNTIME.json").read_text())
    except (OSError,ValueError) as error:
        raise ValueError("committed engineering acceptance required before scientific roots") from error
    acceptance_sha=hashlib.sha256(raw).hexdigest()
    if acceptance_sha!=expected["engineering_acceptance_sha256"]:
        raise ValueError("engineering acceptance byte digest differs from the frozen report")
    try:
        gate=json.loads(raw)
    except ValueError as error:
        raise ValueError("invalid frozen engineering acceptance JSON") from error
    expected_gates={"all_registered_repeats_present", "all_receipts_valid",
                    "pinned_payloads_bit_exact", "pinned_sources_equal",
                    "all_registered_sources_equal", "registered_software",
                    "pinned_runtime_settings", "torch_binaries_equal", "two_distinct_hosts"}
    if (gate.get("schema")!="bpd.amm09r.comparison.v1"
        or gate.get("registration_commit")!="ffc7143e3b597d840c5000a94a45b45d964931cf"
        or gate.get("registered_gate_passed") is not True
        or gate.get("descriptive_only") is True
        or gate.get("confirmation_authorized") is not False
        or gate.get("scientific_evidence") is not False
        or not isinstance(gate.get("gates"),dict)
        or not expected_gates.issubset(gate["gates"])
        or any(value is not True for value in gate["gates"].values())):
        raise ValueError("registered pinned32 engineering acceptance failed")
    if phase=="development":
        if generate:
            require_development_execution()
        lock_path=ROOT/"results"/"TRAINING_LOCK.json"
        try:
            lock=json.loads(lock_path.read_text())
        except (OSError,ValueError) as error:
            raise ValueError("committed training lock required before development") from error
        if (lock.get("schema")!=SCHEMA+"/training_lock"
                or lock.get("source_sha256")!=source_sha256()
                or lock.get("independent_roots")!=64):
            raise ValueError("current source-locked training memory required")
    return acceptance_sha

def seed(*parts:object)->int:
    joined="/".join(map(str,parts))
    if "confirmation" in joined.split("/"):
        raise ValueError("AMM09R1 confirmation namespace is sealed")
    return int(hashlib.sha256(("AMM09R1/"+joined).encode()).hexdigest()[:15],16)

def gen(*parts:object)->torch.Generator:
    value=seed(*parts)  # Confirmation rejection precedes runtime checks or RNG access.
    require_runtime()
    for part in parts:
        phase=str(part).split("/",1)[0]
        if phase in ("training","development"):
            require_phase_admission(phase)
    return torch.Generator(device="cpu").manual_seed(value)

def root_key(phase:str,index:int)->str:
    if phase=="confirmation":
        raise ValueError("AMM09R1 confirmation generation is sealed")
    if phase not in PHASE_N or type(index) is not int or not 0<=index<PHASE_N[phase]:
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
    root=root_key(phase,index)
    require_runtime()
    require_phase_admission(phase)
    torch.set_num_threads(1)
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
