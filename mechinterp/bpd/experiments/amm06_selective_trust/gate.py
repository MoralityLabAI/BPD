"""AMM-06 fixed noisy-label holdout likelihood gate.

The policy never receives clean labels, a judge-mode tag or a donor-future
continuation. Its only local supervised inputs are the 16 noisy bits.
"""
from __future__ import annotations
import math
import torch
from . import models as m

F=11
CAL=8
HALF=4
ETA_LOW=.10
ETA_HIGH=.45
ETA_MARGINAL=(ETA_LOW+ETA_HIGH)/2.
THRESHOLD_LOG_BF=math.log(2.)
LAMBDA_RESIDUAL=20.
LAMBDA_LOCAL=2.


def hidden_quality(index:int)->str:
    """Audit only, balanced independently of family in each phase."""
    return "low_noise" if (index//2)%2==0 else "high_noise"


def true_flip_probability(index:int)->float:
    return ETA_LOW if hidden_quality(index)=="low_noise" else ETA_HIGH


def reported_calibration(phase:str,index:int,cal:dict)->dict:
    """Produce one deterministic noisy answer for each component/pair."""
    if cal.get("split")!="local_calibration" or cal.get("n")!=CAL:
        raise ValueError("calibration cohort must be eight disjoint pairs")
    p=true_flip_probability(index)
    bits={}
    for c in (0,1):
        key=f"component_{c}"
        clean=cal["candidate_bits"][key]
        gen=m.gen(m.root_key(phase,index),"judge_noise",c)
        flips=(torch.rand(CAL,generator=gen)<p).tolist()
        bits[key]="".join(str(int((b=="1")!=flip)) for b,flip in zip(clean,flips))
    return {
        "schema":"BPD-AMM-06/agent-visible-calibration/v1",
        "reported_bits":bits,
        "reported_hits":{k:v.count("1") for k,v in bits.items()},
        "n_pairs":CAL,"n_feedback_labels":2*CAL,
        "known_noise_population":[ETA_LOW,ETA_HIGH]
    }


def observed_labels(feedback:dict)->torch.Tensor:
    if feedback.get("n_pairs")!=CAL or feedback.get("n_feedback_labels")!=2*CAL:
        raise ValueError("eight calibration pairs and 16 reported bits required")
    bits=feedback["reported_bits"]
    if set(bits)!={"component_0","component_1"} or any(
        len(bits[k])!=CAL or set(bits[k])-{"0","1"} for k in bits
    ):
        raise ValueError("malformed agent-visible calibration")
    return torch.tensor([[int(bits[f"component_{c}"][j]) for c in (0,1)]
                         for j in range(CAL)],dtype=torch.float64)


def fit_ridge(X:torch.Tensor,target:torch.Tensor,lam:float,free_intercept:bool)->torch.Tensor:
    X=X.double(); target=target.double()
    if X.ndim!=2 or X.shape[1]!=F or target.shape!=(len(X),):
        raise ValueError("invalid ridge rows")
    penalty=torch.eye(F,dtype=torch.float64)*lam
    if free_intercept: penalty[0,0]=0
    result=torch.linalg.solve(X.T@X+penalty,X.T@target)
    if not bool(torch.isfinite(result).all()):raise ValueError("invalid ridge solution")
    return result


def log_likelihood(pred:torch.Tensor,observed:torch.Tensor,q:float)->torch.Tensor:
    """Likelihood of noisy binary reports conditional on one shared root q."""
    probs=(q+(1-2*q)*pred.clamp(0,1)).clamp(1e-9,1.-1e-9)
    return (observed*probs.log()+(1-observed)*torch.log1p(-probs)).sum()


def log_mixture_likelihood(pred:torch.Tensor,observed:torch.Tensor)->float:
    a=log_likelihood(pred,observed,ETA_LOW)
    b=log_likelihood(pred,observed,ETA_HIGH)
    return float(torch.logaddexp(a,b)-math.log(2.))


def local_decision(features:list, feedback:dict, memory_coefficients:list[float])->dict:
    """Freeze gate before heldout inference, with fit and validation split."""
    X=torch.tensor(features,dtype=torch.float64)
    if X.shape!=(CAL,2,F): raise ValueError("invalid calibration feature shape")
    if len(memory_coefficients)!=F: raise ValueError("invalid prior dimension")
    y=observed_labels(feedback)
    weights=torch.tensor(memory_coefficients,dtype=torch.float64)
    prior=(X@weights).clamp(0,1)
    corrected=(y-ETA_MARGINAL)/(1.-2.*ETA_MARGINAL)
    Xtrain=X[:HALF].reshape(-1,F)
    delta_train=(corrected[:HALF]-prior[:HALF]).reshape(-1)
    proposed=fit_ridge(Xtrain,delta_train,LAMBDA_RESIDUAL,free_intercept=False)
    Xval=X[HALF:]
    p0=prior[HALF:].reshape(-1)
    p1=(prior[HALF:]+Xval@proposed).clamp(0,1).reshape(-1)
    labels=y[HALF:].reshape(-1)
    log_bf=log_mixture_likelihood(p1,labels)-log_mixture_likelihood(p0,labels)
    accept=bool(log_bf>THRESHOLD_LOG_BF)
    Xall=X.reshape(-1,F)
    residual=fit_ridge(Xall,(corrected-prior).reshape(-1),LAMBDA_RESIDUAL,
                       free_intercept=False)
    local=fit_ridge(Xall,corrected.reshape(-1),LAMBDA_LOCAL,free_intercept=True)
    return {"log_bayes_factor":log_bf,"accepted":accept,
            "threshold":THRESHOLD_LOG_BF,
            "prior_weights":weights.tolist(),
            "full_residual_weights":residual.tolist(),
            "local_ridge_weights":local.tolist(),
            "fit_calibration_pairs":HALF,
            "validation_calibration_pairs":CAL-HALF,
            "n_noisy_labels":2*CAL}


def choose_actions(features:list,decision:dict,root:str,pooled_action:int)->dict[str,list[int]]:
    """Per-case choices using model-derived features and the frozen gate only."""
    x=torch.tensor(features,dtype=torch.float64)
    if x.ndim!=3 or x.shape[1:]!=(2,F): raise ValueError("invalid heldout features")
    coeff=torch.tensor(decision["prior_weights"],dtype=torch.float64)
    residual=torch.tensor(decision["full_residual_weights"],dtype=torch.float64)
    local=torch.tensor(decision["local_ridge_weights"],dtype=torch.float64)
    prior=(x@coeff).clamp(0,1)
    updated=prior+x@residual
    local_scores=x@local
    # Shared deterministic tie-breaking selects component zero.
    memory=torch.argmax(prior,dim=1).tolist()
    always=torch.argmax(updated,dim=1).tolist()
    local_choices=torch.argmax(local_scores,dim=1).tolist()
    n=len(x)
    return {
        "selective_trust": always if decision["accepted"] else memory,
        "always_update":always,
        "memory_only":memory,
        "local_ridge":local_choices,
        "pooled":[pooled_action]*n,
        "random_component":[m.seed(root,"random_component",i)%2 for i in range(n)],
    }
