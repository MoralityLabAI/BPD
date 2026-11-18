"""AMM-09R1: source-locked bounded two-slot causal strategy-memory experiment.

Governed by PROTOCOL.md, with source locked before training. Physical model
roots and donor cases have separate phase streams. Confirmation is unavailable.
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
from . import models as m, case

ROOT=Path(__file__).resolve().parent
NAME=m.NAME
SCHEMA=m.SCHEMA
N_ACTIONS=len(m.MASKS)
N_FEATURES=m.FEATURES
GLOBAL_RIDGE=10.
LOCAL_RIDGE=2.
RESIDUAL_RIDGE=20.
JUDGE_ERROR=.20
SCIENCE_FLOOR=.08
BOOTSTRAP_REPS=2000
COMPARISONS=m.COMPARISONS
ARMS=m.ARMS
LAST_PARTIAL={}

def source_sha256()->dict[str,str]:
    return m.source_sha256()

def ridge(X:torch.Tensor,y:torch.Tensor,lam:float,penalize_intercept:bool)->list[float]:
    X=X.double();y=y.double()
    if X.ndim!=2 or X.shape[1]!=N_FEATURES or y.shape!=(len(X),):
        raise ValueError("fixed 18-dimensional ridge features")
    reg=torch.eye(N_FEATURES,dtype=torch.float64)*lam
    if not penalize_intercept:reg[0,0]=0
    weights=torch.linalg.solve(X.T@X+reg,X.T@y)
    if not bool(torch.isfinite(weights).all()):raise ArithmeticError("nonfinite regression")
    return [float(x) for x in weights]

def regression_prediction(features:list,weights:list[float])->torch.Tensor:
    x=torch.tensor(features,dtype=torch.float64)
    if x.ndim!=3 or x.shape[1:]!=(N_ACTIONS,N_FEATURES):
        raise ValueError("wrong six-action candidate features")
    return x@torch.tensor(weights,dtype=torch.float64)

def train():
    global LAST_PARTIAL
    engineering_sha=m.require_phase_admission("training")
    runtime=m.runtime_signature()
    sources=source_sha256()  # Required executable files must exist before any root access.
    rows=[];xs=[];ys=[]
    LAST_PARTIAL={"phase":"training","stage":"root_fitting","rows":rows}
    total=[0]*N_ACTIONS
    for i in range(m.PHASE_N["training"]):
        LAST_PARTIAL["current_root"]=m.root_key("training",i)
        _,f,diagnostics=m.fit("training",i)
        observed=case.score("training",i,f,"training_pairs",m.TRAIN_PAIRS)
        outcomes=torch.tensor([[int(observed["candidate_bits"][f"mask_{j}"][k])
                          for j in range(N_ACTIONS)]
                          for k in range(m.TRAIN_PAIRS)],dtype=torch.float64)
        features=torch.tensor(observed["features"],dtype=torch.float64)
        xs.append(features.reshape(-1,N_FEATURES))
        ys.append(outcomes.reshape(-1))
        for a in range(N_ACTIONS):
            total[a]+=observed["candidate_hits"][f"mask_{a}"]
        rows.append({"root":m.root_key("training",i),
                     "family":m.family_for(i),"fit":diagnostics,"assay":observed})
    LAST_PARTIAL["stage"]="global_policy_fitting"
    raw=ridge(torch.cat(xs),torch.cat(ys),GLOBAL_RIDGE,penalize_intercept=False)
    # Registered portable CPU memory, exact 4-place policy quantization.
    weights=[round(float(x),4) for x in raw]
    training_bits=m.digest([{"root":r["root"],
        "candidate_bits":r["assay"]["candidate_bits"]} for r in rows])
    pool=min((a for a in range(N_ACTIONS)),
             key=lambda a:(-total[a],a))
    out={"schema":SCHEMA+"/training_memory","phase":"training","experiment":NAME,
         "source_sha256":sources,"n_independent_training_roots":64,
         "training_cases_per_root":m.TRAIN_PAIRS,
         "candidate_hits":{f"mask_{a}":total[a] for a in range(N_ACTIONS)},
         "pooled_action":pool,"global_coefficients":weights,
         "ridge_lambda":GLOBAL_RIDGE,"train_bits_sha256":training_bits,
         "training_rows_sha256":m.digest(rows), "runtime_signature":runtime,
         "engineering_acceptance_sha256":engineering_sha}
    out["sha256"]=m.digest(out)
    out["rows"]=rows
    return out

CANDIDATES=tuple(f"mask_{a}" for a in range(6))
CONTROL_ARMS=("random_dark","random_dark_b","sham","full_donor")
CHOICE_ARMS=("memory_plus_local","local_ridge","memory_only","local_count","pooled","random_mask")


def bit_counts(bits:dict,n:int,keys:tuple[str,...])->dict[str,int]:
    if not isinstance(bits,dict) or set(bits)!=set(keys):
        raise ValueError("candidate or arm set differs from registration")
    if any(not isinstance(value,str) or len(value)!=n or set(value)-{"0","1"}
           for value in bits.values()):
        raise ValueError("malformed binary outcome string")
    return {key:bits[key].count("1") for key in keys}


def equal_counts(recorded:dict,expected:dict,context:str)->None:
    if (not isinstance(recorded,dict) or set(recorded)!=set(expected)
            or any(type(recorded[key]) is not int or recorded[key]!=count
                   for key,count in expected.items())):
        raise ValueError(context+" count mismatch")


def validate_assay(observed:dict,root:str,split:str,n:int)->dict[str,int]:
    if (not isinstance(observed,dict) or observed.get("root")!=root
            or observed.get("split")!=split or observed.get("n")!=n):
        raise ValueError("assay membership/split/case count mismatch")
    counts=bit_counts(observed.get("candidate_bits"),n,CANDIDATES+CONTROL_ARMS)
    equal_counts(observed.get("candidate_hits"),counts,"candidate")
    bits=observed["candidate_bits"]
    oracle=sum(any(bits[key][i]=="1" for key in CANDIDATES) for i in range(n))
    diagnostics=observed.get("diagnostics",{})
    if (diagnostics.get("casewise_oracle_hits_audit_only")!=oracle
            or diagnostics.get("full_donor_hits_audit_only")!=counts["full_donor"]):
        raise ValueError("audit-only outcome accounting mismatch")
    if "features" in observed and m.digest(observed["features"])!=observed.get("features_sha256"):
        raise ValueError("feature receipt digest mismatch")
    return counts


def validate_memory(memory:dict):
    if not isinstance(memory,dict) or memory.get("schema")!=SCHEMA+"/training_memory":
        raise ValueError("AMM09R1 frozen training policy required")
    if memory.get("source_sha256")!=source_sha256():
        raise ValueError("AMM09R1 source freeze changed")
    if memory.get("phase")!="training" or memory.get("experiment")!=NAME:
        raise ValueError("wrong training experiment/phase")
    rows=memory.get("rows",[])
    if (memory.get("n_independent_training_roots")!=64 or len(rows)!=64
            or [r.get("root") for r in rows]!=[m.root_key("training",i) for i in range(64)]):
        raise ValueError("incorrect/duplicated/out-of-order training roots")
    if memory.get("training_cases_per_root")!=m.TRAIN_PAIRS or memory.get("ridge_lambda")!=GLOBAL_RIDGE:
        raise ValueError("training budget or ridge penalty changed")
    if memory.get("engineering_acceptance_sha256")!=m.require_phase_admission("training",generate=False):
        raise ValueError("engineering acceptance receipt changed")
    declared=json.loads((ROOT/"RUNTIME.json").read_text())
    runtime=memory.get("runtime_signature",{})
    if (any(runtime.get(key)!=value for key,value in declared["versions"].items())
            or runtime.get("environment")!=declared["environment"]
            or runtime.get("torch_git")!=declared["torch_git"]
            or runtime.get("torch_binary_sha256")!=declared["torch_binary_sha256"]
            or runtime.get("num_threads")!=1 or runtime.get("num_interop_threads")!=1
            or runtime.get("deterministic_algorithms") is not True
            or runtime.get("default_dtype")!="torch.float32"
            or runtime.get("cpu_capability") not in ("NO AVX","DEFAULT")):
        raise ValueError("unregistered stored training runtime")
    total={key:0 for key in CANDIDATES}
    xs=[];ys=[]
    for index,row in enumerate(rows):
        if row.get("family")!=m.family_for(index):
            raise ValueError("training family mismatch")
        observed=row["assay"]
        counts=validate_assay(observed,row["root"],"training_pairs",m.TRAIN_PAIRS)
        for key in CANDIDATES:total[key]+=counts[key]
        X=torch.tensor(observed.get("features"),dtype=torch.float64)
        if X.shape!=(m.TRAIN_PAIRS,N_ACTIONS,N_FEATURES) or not bool(torch.isfinite(X).all()):
            raise ValueError("full finite training feature receipt required")
        y=torch.tensor([[int(observed["candidate_bits"][key][i]) for key in CANDIDATES]
                        for i in range(m.TRAIN_PAIRS)],dtype=torch.float64)
        xs.append(X.reshape(-1,N_FEATURES));ys.append(y.reshape(-1))
    equal_counts(memory.get("candidate_hits"),total,"training aggregate")
    pooled=min(range(N_ACTIONS),key=lambda a:(-total[f"mask_{a}"],a))
    if type(memory.get("pooled_action")) is not int or memory["pooled_action"]!=pooled:
        raise ValueError("pooled action is not the training-selected permitted mask")
    bit_rows=[{"root":r["root"],"candidate_bits":r["assay"]["candidate_bits"]} for r in rows]
    if m.digest(bit_rows)!=memory.get("train_bits_sha256") or m.digest(rows)!=memory.get("training_rows_sha256"):
        raise ValueError("training candidate/full-row digest mismatch")
    weights=memory.get("global_coefficients",[])
    if len(weights)!=N_FEATURES or any(type(x) not in (int,float) or not math.isfinite(x) or round(x,4)!=x for x in weights):
        raise ValueError("invalid/unquantized memory weights")
    refit=[round(float(x),4) for x in ridge(torch.cat(xs),torch.cat(ys),GLOBAL_RIDGE,False)]
    if refit!=weights:
        raise ValueError("retained training features do not reproduce frozen coefficients")
    content={key:value for key,value in memory.items() if key not in ("sha256","rows")}
    if m.digest(content)!=memory.get("sha256"):
        raise ValueError("training policy digest mismatch")
    return True


def selected_bits(candidates:dict,actions:list[int],n:int)->str:
    if (not isinstance(actions,list) or len(actions)!=n
            or any(type(action) is not int or not 0<=action<N_ACTIONS for action in actions)):
        raise ValueError("invalid or inadmissible policy action")
    return "".join(candidates[f"mask_{action}"][i] for i,action in enumerate(actions))


def validated_root_counts(row:dict)->tuple[dict[str,int],int]:
    root=row.get("root")
    if not isinstance(root,str) or not root.startswith("development/"):
        raise ValueError("development root receipt required")
    validate_assay(row.get("heldout"),root,"heldout",m.PAIRS)
    candidates=row["heldout"]["candidate_bits"]
    actions=row.get("policy_actions",{})
    if not isinstance(actions,dict) or set(actions)!=set(CHOICE_ARMS):
        raise ValueError("policy action set differs from registration")
    reconstructed={arm:selected_bits(candidates,actions[arm],m.PAIRS) for arm in CHOICE_ARMS}
    reconstructed.update({arm:candidates[arm] for arm in CONTROL_ARMS})
    bit_counts(row.get("policy_bits"),m.PAIRS,ARMS)
    if reconstructed!=row["policy_bits"]:
        raise ValueError("selected outcome bits differ from admitted action/candidate bits")
    counts={arm:bits.count("1") for arm,bits in reconstructed.items()}
    equal_counts(row.get("policy_hits"),counts,"per-root policy")
    policy=row.get("local_policy",{})
    pooled=policy.get("pooled_action")
    if type(pooled) is not int or not 0<=pooled<N_ACTIONS or actions["pooled"]!=[pooled]*m.PAIRS:
        raise ValueError("pooled policy action changed within root")
    local_count=policy.get("local_count_action")
    if type(local_count) is not int or not 0<=local_count<N_ACTIONS or actions["local_count"]!=[local_count]*m.PAIRS:
        raise ValueError("local-count policy action changed within root")
    calibration=row.get("calibration_clean_audit_only")
    validate_assay(calibration,root,"calibration",m.CAL_PAIRS)
    feedback=row.get("agent_noisy_feedback",{})
    reports=bit_counts(feedback.get("reported_bits"),m.CAL_PAIRS,CANDIDATES)
    equal_counts(feedback.get("reported_hits"),reports,"noisy calibration")
    if (feedback.get("noisy_labels")!=m.CAL_PAIRS*N_ACTIONS
            or feedback.get("n_local_cases")!=m.CAL_PAIRS
            or feedback.get("known_judge_error")!=JUDGE_ERROR):
        raise ValueError("noisy calibration budget/known error changed")
    expected_local=min(range(N_ACTIONS),key=lambda a:(-reports[f"mask_{a}"],a))
    if local_count!=expected_local:
        raise ValueError("local-count policy disagrees with noisy calibration")
    repeat=row.get("repeatability",{})
    validate_assay(repeat.get("assay"),root,"repeatability",m.PAIRS)
    repeat_candidates=repeat["assay"]["candidate_bits"]
    pooled_bits=selected_bits(repeat_candidates,[pooled]*m.PAIRS,m.PAIRS)
    local_bits=selected_bits(repeat_candidates,repeat.get("local_ridge_actions"),m.PAIRS)
    if repeat.get("pooled_bits")!=pooled_bits or repeat.get("local_ridge_bits")!=local_bits:
        raise ValueError("repeatability selected outcome mismatch")
    if (type(repeat.get("pooled_hits")) is not int or repeat["pooled_hits"]!=pooled_bits.count("1")
            or type(repeat.get("local_ridge_hits")) is not int or repeat["local_ridge_hits"]!=local_bits.count("1")):
        raise ValueError("repeatability count mismatch")
    if type(row.get("fidelity_audit_only")) is not bool or row["fidelity_audit_only"]!=model_fidelity(row["fit"]):
        raise ValueError("fidelity flag disagrees with fitted diagnostics")
    return counts,pooled_bits.count("1")


def validated_development_counts(result:dict)->list[tuple[dict[str,int],int]]:
    if (result.get("phase")!="development" or result.get("n_independent_roots")!=64
            or result.get("n_test_pairs_per_root")!=m.PAIRS
            or result.get("total_test_cases")!=64*m.PAIRS):
        raise ValueError("registered 64-root development cohort required")
    rows=result.get("rows",[])
    if [row.get("root") for row in rows]!=[m.root_key("development",i) for i in range(64)]:
        raise ValueError("development roots missing, duplicated, or reordered")
    for index,row in enumerate(rows):
        if row.get("family")!=m.family_for(index):
            raise ValueError("development family mismatch")
        if row.get("local_policy",{}).get("pooled_action")!=result.get("pooled_action"):
            raise ValueError("root pooled action differs from frozen cohort memory")
        if row.get("local_policy",{}).get("prior_weights")!=result.get("global_coefficients"):
            raise ValueError("root global prior differs from frozen cohort memory")
    if type(result.get("pooled_action")) is not int or not 0<=result["pooled_action"]<N_ACTIONS:
        raise ValueError("invalid frozen cohort pooled action")
    weights=result.get("global_coefficients",[])
    if len(weights)!=N_FEATURES or any(type(value) not in (int,float) or not math.isfinite(value)
                                       or round(value,4)!=value for value in weights):
        raise ValueError("invalid frozen cohort prior coefficients")
    derived=[validated_root_counts(row) for row in rows]
    totals={arm:sum(counts[arm] for counts,_ in derived) for arm in ARMS}
    equal_counts(result.get("totals"),totals,"cohort aggregate")
    return derived

def corrupted_feedback(phase:str,index:int,calibration:dict)->dict:
    if calibration["split"]!="calibration" or calibration["n"]!=m.CAL_PAIRS:
        raise ValueError("registered local calibration sample required")
    bits={}
    for action in range(N_ACTIONS):
        key=f"mask_{action}"
        g=m.gen(m.root_key(phase,index),"judge_noise",action)
        flips=(torch.rand(m.CAL_PAIRS,generator=g)<JUDGE_ERROR).tolist()
        bits[key]="".join(str(int((b=="1")!=flip))
                          for b,flip in zip(calibration["candidate_bits"][key],flips))
    return {"schema":SCHEMA+"/agent_noisy_feedback",
            "reported_bits":bits,
            "reported_hits":{k:v.count("1") for k,v in bits.items()},
            "n_local_cases":m.CAL_PAIRS,"noisy_labels":m.CAL_PAIRS*N_ACTIONS,
            "known_judge_error":JUDGE_ERROR}

def policy_fit(calibration_features:list,feedback:dict,memory:dict):
    if feedback.get("noisy_labels")!=m.CAL_PAIRS*N_ACTIONS:
        raise ValueError("must use exact 96-label local budget")
    bits=feedback["reported_bits"]
    if set(bits)!={f"mask_{j}" for j in range(N_ACTIONS)} or any(
        len(v)!=m.CAL_PAIRS or set(v)-{"0","1"} for v in bits.values()
    ):
        raise ValueError("invalid judge feedback bitstrings")
    equal_counts(feedback.get("reported_hits"),{key:value.count("1") for key,value in bits.items()},"noisy calibration")
    y=torch.tensor([[int(bits[f"mask_{j}"][i]) for j in range(N_ACTIONS)]
                    for i in range(m.CAL_PAIRS)],dtype=torch.float64)
    corrected=(y-JUDGE_ERROR)/(1.-2.*JUDGE_ERROR)
    X=torch.tensor(calibration_features,dtype=torch.float64)
    if X.shape!=(m.CAL_PAIRS,N_ACTIONS,N_FEATURES):
        raise ValueError("source-shared six-action calibration geometry required")
    prior=(X@torch.tensor(memory["global_coefficients"],dtype=torch.float64)).clamp(0,1)
    residual=ridge(X.reshape(-1,N_FEATURES),(corrected-prior).reshape(-1),
                   RESIDUAL_RIDGE,penalize_intercept=True)
    local=ridge(X.reshape(-1,N_FEATURES),corrected.reshape(-1),
                LOCAL_RIDGE,penalize_intercept=False)
    most_frequent=min(range(N_ACTIONS),key=lambda a:(-feedback["reported_hits"][f"mask_{a}"],a))
    return {"prior_weights":memory["global_coefficients"],
            "residual_weights":residual,"local_weights":local,
            "local_count_action":most_frequent,
            "pooled_action":memory["pooled_action"],
            "noisy_label_count":m.CAL_PAIRS*N_ACTIONS}

def choose(features:list,policy:dict,root:str)->dict[str,list[int]]:
    x=torch.tensor(features,dtype=torch.float64)
    if x.ndim!=3 or x.shape[1:]!=(N_ACTIONS,N_FEATURES):
        raise ValueError("invalid source-independent test geometry")
    prior=(x@torch.tensor(policy["prior_weights"],dtype=torch.float64)).clamp(0,1)
    residual=x@torch.tensor(policy["residual_weights"],dtype=torch.float64)
    local=x@torch.tensor(policy["local_weights"],dtype=torch.float64)
    n=len(x)
    argmax=lambda t:torch.argmax(t,dim=1).tolist()
    return {"memory_plus_local":argmax((prior+residual).clamp(0,1)),
            "local_ridge":argmax(local),
            "memory_only":argmax(prior),
            "local_count":[policy["local_count_action"]]*n,
            "pooled":[policy["pooled_action"]]*n,
            "random_mask":[m.seed(root,"random_mask",i)%N_ACTIONS for i in range(n)]}

def score_root(phase:str,index:int,memory:dict)->dict:
    """No test success bit enters any policy fit or action selection."""
    _,f,fit=m.fit(phase,index)
    root=m.root_key(phase,index)
    calibration=case.score(phase,index,f,"calibration",m.CAL_PAIRS)
    agent_feedback=corrupted_feedback(phase,index,calibration)
    trained_policy=policy_fit(calibration.pop("features"),agent_feedback,memory)
    held=case.score(phase,index,f,"heldout",m.PAIRS)
    policy_choices=choose(held.pop("features"),trained_policy,root)
    bits={name:"".join(held["candidate_bits"][f"mask_{a}"][i] for i,a in enumerate(actions))
          for name,actions in policy_choices.items()}
    for arm in ("random_dark","random_dark_b","sham","full_donor"):
        bits[arm]=held["candidate_bits"][arm]
    repeat=case.score(phase,index,f,"repeatability",m.PAIRS)
    repeat_choices=choose(repeat.pop("features"),trained_policy,root)
    pooled_repeat_bits="".join(repeat["candidate_bits"][f"mask_{a}"][i]
                              for i,a in enumerate(repeat_choices["pooled"]))
    local_repeat_bits="".join(repeat["candidate_bits"][f"mask_{a}"][i]
                              for i,a in enumerate(repeat_choices["local_ridge"]))
    return {"root":root,"family":m.family_for(index),"fit":fit,
            "fidelity_audit_only":model_fidelity(fit),
            "calibration_clean_audit_only":calibration,
            "agent_noisy_feedback":agent_feedback,
            "local_policy":trained_policy,
            "heldout":held,"policy_actions":policy_choices,
            "policy_bits":bits,"policy_hits":{k:v.count("1") for k,v in bits.items()},
            "repeatability":{"assay":repeat,
                             "local_ridge_actions":repeat_choices["local_ridge"],
                             "local_ridge_bits":local_repeat_bits,
                             "local_ridge_hits":local_repeat_bits.count("1"),
                             "pooled_bits":pooled_repeat_bits,
                             "pooled_hits":pooled_repeat_bits.count("1")}}

def model_fidelity(d:dict)->bool:
    return bool(d["teacher_task_mse"]<=1e-4 and
                d["factor_task_mse"]<=.02 and
                d["weight_reconstruction_relative"]<=.10 and
                d["min_rank3_gap"]>.01)

def admit(phase:str,memory:dict|None,preflight:dict|None=None,release:dict|None=None):
    """Reject confirmation before memory validation or any model access."""
    if phase=="confirmation":
        raise ValueError("AMM09R1 confirmation remains sealed; no release interface is implemented")
    if phase!="development":
        raise ValueError("unknown evaluation phase")
    m.require_phase_admission("development")
    m.require_runtime()
    validate_memory(memory)
    if memory["runtime_signature"]!=m.runtime_signature():
        raise ValueError("training runtime/binaries differ from development runtime")
    lock=json.loads((ROOT/"results"/"TRAINING_LOCK.json").read_text())
    if compact(memory)!=lock:
        raise ValueError("supplied full training memory differs from committed training lock")

def preflight_input(result:dict)->dict:
    derived=validated_development_counts(result)
    shams=[]
    for row,(counts,repeat_pooled) in zip(result["rows"],derived):
        shams.append({"unit_id":row["root"],
                      "paired_effect":f"{counts['memory_only']-counts['pooled']}/{m.PAIRS}",
                      "sham_effects":[f"{counts['pooled']-repeat_pooled}/{m.PAIRS}"]})
    return {"schema_version":1,"experiment_id":NAME,
            "independent_unit":"one 6D rank-three three-pass trained physical and BPD model root",
            "alpha":"0.05","scientific_min_effect":"0.08","target_effect":"0.16",
            "measurement_resolution":f"1/{m.PAIRS}",
            "sham_quantile":"0.95","sham_confidence":"0.95",
            "planned_comparisons":len(COMPARISONS),"confirm_units_planned":128,
            "pilot_from_development_only":True,"heldout_sealed":True,
            "units_independent":True,"pilot":shams}

def exact_root_sign_p(xs:list[int])->float:
    n=sum(x!=0 for x in xs)
    if n==0:return 1.
    k=sum(x>0 for x in xs)
    return sum(math.comb(n,j) for j in range(k,n+1))/(2**n)

def holm(ps:dict[str,float])->dict[str,float]:
    out={};floor=0.
    for i,key in enumerate(sorted(ps,key=lambda k:(ps[k],k))):
        floor=max(floor,min(1.,(len(ps)-i)*ps[key]))
        out[key]=floor
    return out

def cluster_bootstrap(xs:list[int],key:str)->list[float]:
    rand=random.Random(m.seed("bootstrap",key))
    n=len(xs)
    draws=sorted(sum(xs[rand.randrange(n)] for _ in range(n))/(n*m.PAIRS)
                 for _ in range(BOOTSTRAP_REPS))
    return [draws[50],draws[1949]]

def analysis(result:dict,preflight:dict|None=None)->dict:
    derived=validated_development_counts(result)
    comparisons={name:[counts[treatment]-counts[control] for counts,_ in derived]
                 for name,treatment,control in COMPARISONS}
    pvals={key:exact_root_sign_p(values) for key,values in comparisons.items()}
    adjusted=holm(pvals)
    metrics={key:{"mean_root_effect":sum(values)/(len(values)*m.PAIRS),
                  "root_bootstrap_95_ci":cluster_bootstrap(values,key),
                  "positive_roots":sum(x>0 for x in values),
                  "negative_roots":sum(x<0 for x in values),
                  "tied_roots":sum(x==0 for x in values),
                  "exact_one_sided_root_sign_p":pvals[key],
                  "holm_adjusted_p":adjusted[key]} for key,values in comparisons.items()}
    out={"schema":SCHEMA+"/analysis", "status":"development_only", "comparisons":metrics,
         "confirmation_authorized":False,
         "comparison_arms":{name:{"treatment":treatment,"control":control}
                            for name,treatment,control in COMPARISONS}}
    if preflight is not None:
        if (preflight.get("experiment_id")!=NAME or preflight.get("pilot_worlds")!=64
                or preflight.get("planned_comparisons")!=4):
            raise ValueError("preflight does not match the registered independent-root study")
        floor=max(SCIENCE_FLOOR,preflight["mechanistic_variation_floor"])
        primary=metrics["memory_only_vs_pooled"]
        maxerr=max(row["heldout"]["diagnostics"]["max_equal_dose_error"] for row in result["rows"])
        maxvisible=max(row["heldout"]["diagnostics"]["max_immediate_readout_displacement"] for row in result["rows"])
        high_fidelity=sum(row["fidelity_audit_only"] for row in result["rows"])
        headroom=sum(row["heldout"]["diagnostics"]["casewise_oracle_hits_audit_only"]-counts["pooled"]
                     for row,(counts,_) in zip(result["rows"],derived))/(64*m.PAIRS)
        gates={"primary_lower_bound_above_floor":primary["root_bootstrap_95_ci"][0]>floor,
               "primary_holm_significant":primary["holm_adjusted_p"]<.05,
               "source_fidelity":high_fidelity>=56,
               "allowed_oracle_headroom":headroom>=.08,
               "equal_dose":maxerr<=1e-5,"current_readout_unchanged":maxvisible==0,
               "evalmath_ready":preflight.get("decision")=="READY_FOR_CONFIRMATION"}
        out.update({"mechanistic_variation_floor":floor,
                    "max_dose_error":maxerr,"max_immediate_drift":maxvisible,
                    "high_fidelity_roots":high_fidelity,"allowed_oracle_headroom_rate":headroom,
                    "development_gates":gates,"registered_primary_gate":all(gates.values())})
    return out

def evaluate(phase:str,memory:dict|None=None,preflight:dict|None=None,release:dict|None=None)->dict:
    global LAST_PARTIAL
    admit(phase,memory,preflight,release)
    assert memory is not None
    rows=[]
    LAST_PARTIAL={"phase":phase,"stage":"root_scoring","rows":rows}
    for index in range(m.PHASE_N[phase]):
        LAST_PARTIAL["current_root"]=m.root_key(phase,index)
        rows.append(score_root(phase,index,memory))
    LAST_PARTIAL["stage"]="cohort_accounting"
    out={"schema":SCHEMA+"/cohort","phase":phase,"experiment":NAME,
         "n_independent_roots":len(rows),"training_policy_sha256":memory["sha256"],
         "source_sha256":source_sha256(),"runtime_signature":m.runtime_signature(),
         "engineering_acceptance_sha256":memory["engineering_acceptance_sha256"],
         "pooled_action":memory["pooled_action"],
         "global_coefficients":memory["global_coefficients"],
         "n_calibration_pairs_per_root":m.CAL_PAIRS,
         "n_test_pairs_per_root":m.PAIRS,
         "n_repeatability_pairs_per_root":m.PAIRS,
         "total_test_cases":m.PAIRS*len(rows),
         "totals":{arm:sum(r["policy_hits"][arm] for r in rows) for arm in ARMS},
         "rows":rows}
    if phase=="development":
        out["preflight_input"]=preflight_input(out)
        out["analysis"]=analysis(out)
        out["high_fidelity_roots"]=sum(r["fidelity_audit_only"] for r in rows)
        out["oracle_headroom_hits_audit_only"]=sum(
            r["heldout"]["diagnostics"]["casewise_oracle_hits_audit_only"]
            -r["policy_hits"]["pooled"] for r in rows)
        out["oracle_headroom_rate_audit_only"]=out["oracle_headroom_hits_audit_only"]/out["total_test_cases"]
        out["max_equal_dose_error"]=max(r["heldout"]["diagnostics"]["max_equal_dose_error"]
                                       for r in rows)
        out["max_immediate_readout_displacement"]=max(
            r["heldout"]["diagnostics"]["max_immediate_readout_displacement"] for r in rows)
    return out

def compact(run:dict)->dict:
    """Stable semantic replay; raw machine-specific floats archived separately."""
    if run.get("phase")=="training":
        return {"schema":SCHEMA+"/training_lock",
                "source_sha256":run["source_sha256"],
                "training_policy_sha256":run["sha256"],
                "independent_roots":64,
                "pooled_action":run["pooled_action"],
                "train_candidate_hits":run["candidate_hits"],
                "training_case_bits_sha256":run["train_bits_sha256"],
                "training_rows_sha256":run["training_rows_sha256"],
                "runtime_signature":run["runtime_signature"],
                "engineering_acceptance_sha256":run["engineering_acceptance_sha256"],
                "global_coefficients":run["global_coefficients"]}
    validated_development_counts(run)
    out={"schema":SCHEMA+"/cohort_lock","phase":run["phase"],
         "source_sha256":run["source_sha256"],
         "runtime_signature":run["runtime_signature"],
         "engineering_acceptance_sha256":run["engineering_acceptance_sha256"],
         "training_policy_sha256":run["training_policy_sha256"],
         "pooled_action":run["pooled_action"],
         "global_coefficients":run["global_coefficients"],
         "independent_roots":run["n_independent_roots"],
         "total_test_cases":run["total_test_cases"],
         "totals":run["totals"],
         "all_candidate_action_case_bits_sha256":m.digest([
            {"root":r["root"],"clean_calibration":r["calibration_clean_audit_only"]["candidate_bits"],
             "noisy_calibration":r["agent_noisy_feedback"]["reported_bits"],
             "test_candidates":r["heldout"]["candidate_bits"],
             "selected_actions":r["policy_actions"],"selected_outcomes":r["policy_bits"],
             "repeat_local_outcomes":r["repeatability"]["local_ridge_bits"],
             "repeat_pooled_outcomes":r["repeatability"]["pooled_bits"],
             "fidelity_flag":r["fidelity_audit_only"]}
            for r in run["rows"]])}
    if run["phase"]=="development":
        out.update({"high_fidelity_roots":run["high_fidelity_roots"],
                    "oracle_headroom_rate_audit_only":run["oracle_headroom_rate_audit_only"],
                    "max_equal_dose_error":run["max_equal_dose_error"],
                    "max_immediate_readout_displacement":run["max_immediate_readout_displacement"],
                    "preflight_input_sha256":m.digest(run["preflight_input"]),
                    "analysis":run["analysis"]})
    return out

if __name__=="__main__":
    raise SystemExit("Invoke run.py directly so the CPU environment is fixed before importing torch")
