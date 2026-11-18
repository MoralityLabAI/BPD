"""AMM-07: independent duplicated noisy judge reports and Bayesian reliability.

Only the 16x2x2 noisy calibration bits and candidate features reach policies.
The clean label, hidden judge regime, donor continuation and target are audit only.
"""
from __future__ import annotations
import math
import torch
from . import models as m

FEATURES=11
CAL=16
QUALITY_CAL=8
ETA_LOW=.10
ETA_HIGH=.45
LAM_RESIDUAL=20.
LAM_LOCAL=2.
EPS=1e-6

def hidden_quality(index:int)->str:
    return "low_noise" if (index//2)%2==0 else "high_noise"

def true_flip_probability(index:int)->float:
    return ETA_LOW if hidden_quality(index)=="low_noise" else ETA_HIGH

def reported_calibration(phase:str,index:int,cal:dict)->dict:
    """Two independent reports for each candidate on each calibration pair."""
    if cal.get("split")!="local_calibration" or cal.get("n")!=CAL:
        raise ValueError("requires 16 disjoint local calibration cases")
    true_q=true_flip_probability(index)  # generator only, omitted from agent output
    results={}
    for c in (0,1):
        key=f"component_{c}"
        actual=cal["candidate_bits"][key]
        reports=[]
        for rep in (0,1):
            g=m.gen(m.root_key(phase,index),"judge_duplicate",c,rep)
            flips=(torch.rand(CAL,generator=g)<true_q).tolist()
            reports.append("".join(str(int((bit=="1")!=flip)) for bit,flip in zip(actual,flips)))
        results[key]=reports
    return {"schema":"BPD-AMM-07/agent-visible-replicates/v1",
            "reported_replicates":results,"n_calibration_pairs":CAL,
            "candidate_actions":2,"judge_repeats":2,
            "n_noisy_reports":4*CAL,
            "public_noise_population":[ETA_LOW,ETA_HIGH]}

def observations(feedback:dict)->torch.Tensor:
    if not isinstance(feedback,dict) or feedback.get("n_calibration_pairs")!=CAL or feedback.get("n_noisy_reports")!=4*CAL:
        raise ValueError("malformed calibration count")
    reps=feedback.get("reported_replicates")
    if not isinstance(reps,dict) or set(reps)!={"component_0","component_1"}:
        raise ValueError("calibration needs two candidate response lists")
    if any(not isinstance(reps[f"component_{c}"],list) or len(reps[f"component_{c}"])!=2 or
           any(not isinstance(v,str) or len(v)!=CAL or set(v)-{"0","1"} for v in reps[f"component_{c}"])
           for c in (0,1)):
        raise ValueError("malformed duplicated noisy labels")
    return torch.tensor([[[int(reps[f"component_{c}"][r][i]) for r in (0,1)]
                           for c in (0,1)] for i in range(CAL)],dtype=torch.float64)

def quality_posterior(reports:torch.Tensor)->dict:
    """Classify hidden shared q using ONLY first eight calibration cases."""
    if reports.shape!=(CAL,2,2):raise ValueError("wrong duplicate shape")
    d=int((reports[:QUALITY_CAL,: ,0]!=reports[:QUALITY_CAL,: ,1]).sum().item())
    n=QUALITY_CAL*2
    def log_binomial_kernel(q:float)->float:
        mismatch=2*q*(1-q)
        return d*math.log(mismatch)+(n-d)*math.log1p(-mismatch)
    logodds=log_binomial_kernel(ETA_LOW)-log_binomial_kernel(ETA_HIGH)
    p_low=1/(1+math.exp(-logodds))
    return {"posterior_low":p_low,"posterior_high":1-p_low,
            "log_posterior_odds_low":logodds,
            "disagreements_first_half":d,"n_independent_duplicate_pairs":n,
            "classify_low":bool(p_low>=.5)}

def posterior_clean_truth(prior:torch.Tensor,reports:torch.Tensor,w_low:float)->torch.Tensor:
    """Bayesian E[clean y|both reports], integrating the root-q mixture."""
    if reports.shape!=prior.shape+(2,):raise ValueError("invalid posterior shapes")
    if not 0.<=w_low<=1.:raise ValueError("quality posterior outside probability range")
    p=prior.double().clamp(EPS,1.-EPS)
    count=reports.double().sum(dim=-1)
    num=torch.zeros_like(p);den_other=torch.zeros_like(p)
    for q,w in ((ETA_LOW,w_low),(ETA_HIGH,1-w_low)):
        l1=(1-q)**count * q**(2-count)
        l0=q**count*(1-q)**(2-count)
        num+=w*p*l1
        den_other+=w*(1-p)*l0
    return (num/(num+den_other)).clamp(EPS,1.-EPS)

def ridge(X:torch.Tensor,y:torch.Tensor,lam:float,penalize_intercept:bool)->torch.Tensor:
    if X.ndim!=2 or X.shape[1]!=FEATURES or y.shape!=(len(X),):
        raise ValueError("invalid local ridge inputs")
    X=X.double();y=y.double()
    reg=torch.eye(FEATURES,dtype=torch.float64)*lam
    if not penalize_intercept:reg[0,0]=0.
    return torch.linalg.solve(X.T@X+reg,X.T@y)

def local_decision(features:list,feedback:dict,memory_coefficients:list[float])->dict:
    """Quality-aware and blind fits share the same 64 noisy reports.

    Both registered 8-pair policies fit only calibration cases 8..15.
    Blind-all is a stronger control using all 16 calibration cases.
    """
    if len(memory_coefficients)!=FEATURES:raise ValueError("memory coefficient shape")
    X=torch.tensor(features,dtype=torch.float64)
    if X.shape!=(CAL,2,FEATURES):raise ValueError("wrong calibration feature matrix")
    reports=observations(feedback)
    quality=quality_posterior(reports)
    weights=torch.tensor(memory_coefficients,dtype=torch.float64)
    prior=(X@weights).clamp(EPS,1.-EPS)
    def residual(w_low:float,indices:slice)->list[float]:
        soft=posterior_clean_truth(prior[indices],reports[indices],w_low)
        return ridge(X[indices].reshape(-1,FEATURES),
                     (soft-prior[indices]).reshape(-1),
                     LAM_RESIDUAL,True).tolist()
    aware=residual(quality["posterior_low"],slice(QUALITY_CAL,None))
    blind=residual(.5,slice(QUALITY_CAL,None))
    blind_all=residual(.5,slice(None))
    local_soft=posterior_clean_truth(torch.full_like(prior,.5),reports,.5)
    local=ridge(X.reshape(-1,FEATURES),local_soft.reshape(-1),
                LAM_LOCAL,False).tolist()
    return {"quality":quality,"memory_coefficients":memory_coefficients,
            "aware_residual":aware,"blind_residual":blind,
            "blind_all_residual":blind_all,"local_ridge":local,
            "n_noisy_reports":4*CAL,
            "n_quality_cases":QUALITY_CAL,
            "n_shared_fit_cases":CAL-QUALITY_CAL}

def choose_actions(features:list,decision:dict,root:str,pooled:int)->dict[str,list[int]]:
    X=torch.tensor(features,dtype=torch.float64)
    if X.ndim!=3 or X.shape[1:]!=(2,FEATURES):
        raise ValueError("candidate features shape")
    memory=torch.tensor(decision["memory_coefficients"],dtype=torch.float64)
    prior=(X@memory).clamp(0,1)
    def decide(choices:torch.Tensor)->list[int]:
        return torch.argmax(choices,dim=1).tolist()
    def updated(residual_key:str)->list[int]:
        w=torch.tensor(decision[residual_key],dtype=torch.float64)
        return decide((prior+X@w).clamp(0,1))
    n=len(X)
    return {"quality_aware":updated("aware_residual"),
            "quality_blind":updated("blind_residual"),
            "blind_all":updated("blind_all_residual"),
            "memory_only":decide(prior),
            "local_ridge":decide(X@torch.tensor(decision["local_ridge"],dtype=torch.float64)),
            "pooled":[pooled]*n,
            "random_component":[m.seed(root,"random_component",i)%2 for i in range(n)]}
