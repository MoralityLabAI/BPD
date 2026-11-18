"""AMM-08: source-locked seven-action strategy-memory experiment.

Registered in PROTOCOL.md at 88855522 before implementation. Physical model
roots and donor cases are deterministic, source-independent across phases.
The confirmation branch is fail-closed before ANY model fitting.
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

def source_sha256()->dict[str,str]:
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("models.py","case.py","experiment.py")}

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
        raise ValueError("wrong seven-action candidate features")
    return x@torch.tensor(weights,dtype=torch.float64)

def train():
    rows=[];xs=[];ys=[]
    total=[0]*N_ACTIONS
    for i in range(m.PHASE_N["training"]):
        _,f,diagnostics=m.fit("training",i)
        observed=case.score("training",i,f,"training_pairs",m.TRAIN_PAIRS)
        outcomes=torch.tensor([[int(observed["candidate_bits"][f"mask_{j}"][k])
                          for j in range(N_ACTIONS)]
                          for k in range(m.TRAIN_PAIRS)],dtype=torch.float64)
        features=torch.tensor(observed.pop("features"),dtype=torch.float64)
        xs.append(features.reshape(-1,N_FEATURES))
        ys.append(outcomes.reshape(-1))
        for a in range(N_ACTIONS):
            total[a]+=observed["candidate_hits"][f"mask_{a}"]
        rows.append({"root":m.root_key("training",i),
                     "family":m.family_for(i),"fit":diagnostics,"assay":observed})
    raw=ridge(torch.cat(xs),torch.cat(ys),GLOBAL_RIDGE,penalize_intercept=False)
    # Registered portable CPU memory, exact 4-place policy quantization.
    weights=[round(float(x),4) for x in raw]
    training_bits=m.digest([{"root":r["root"],
        "candidate_bits":r["assay"]["candidate_bits"]} for r in rows])
    pool=min((a for a in range(N_ACTIONS)),
             key=lambda a:(-total[a],a))
    out={"schema":SCHEMA+"/training_memory","phase":"training","experiment":NAME,
         "source_sha256":source_sha256(),"n_independent_training_roots":64,
         "training_cases_per_root":m.TRAIN_PAIRS,
         "candidate_hits":{f"mask_{a}":total[a] for a in range(N_ACTIONS)},
         "pooled_action":pool,"global_coefficients":weights,
         "ridge_lambda":GLOBAL_RIDGE,"train_bits_sha256":training_bits}
    out["sha256"]=m.digest(out)
    out["rows"]=rows
    return out

def validate_memory(memory:dict):
    if not isinstance(memory,dict) or memory.get("schema")!=SCHEMA+"/training_memory":
        raise ValueError("AMM08 frozen training policy required")
    if memory.get("source_sha256")!=source_sha256():
        raise ValueError("AMM08 source freeze changed")
    if memory.get("n_independent_training_roots")!=64 or len(memory.get("rows",[]))!=64:
        raise ValueError("training world count changed")
    if {r.get("root") for r in memory["rows"]}!={m.root_key("training",i) for i in range(64)}:
        raise ValueError("incorrect/duplicated training worlds")
    rows=[{"root":r["root"],"candidate_bits":r["assay"]["candidate_bits"]}
          for r in memory["rows"]]
    if m.digest(rows)!=memory.get("train_bits_sha256"):
        raise ValueError("training candidate-case hash changed")
    if len(memory.get("global_coefficients",[]))!=N_FEATURES or any(
        not math.isfinite(x) or round(x,4)!=x for x in memory["global_coefficients"]
    ):
        raise ValueError("invalid/unquantized memory weights")
    if memory.get("pooled_action") not in range(N_ACTIONS):
        raise ValueError("pooled action invalid")
    content={key:value for key,value in memory.items() if key not in ("sha256","rows")}
    if m.digest(content)!=memory.get("sha256"):
        raise ValueError("training policy digest mismatch")
    return True

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
        raise ValueError("must use exact 112-label local budget")
    bits=feedback["reported_bits"]
    if set(bits)!={f"mask_{j}" for j in range(N_ACTIONS)} or any(
        len(v)!=m.CAL_PAIRS or set(v)-{"0","1"} for v in bits.values()
    ):
        raise ValueError("invalid judge feedback bitstrings")
    y=torch.tensor([[int(bits[f"mask_{j}"][i]) for j in range(N_ACTIONS)]
                    for i in range(m.CAL_PAIRS)],dtype=torch.float64)
    corrected=(y-JUDGE_ERROR)/(1.-2.*JUDGE_ERROR)
    X=torch.tensor(calibration_features,dtype=torch.float64)
    if X.shape!=(m.CAL_PAIRS,N_ACTIONS,N_FEATURES):
        raise ValueError("source-shared seven-action calibration geometry required")
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
                             "local_ridge_hits":local_repeat_bits.count("1")}}

def model_fidelity(d:dict)->bool:
    return bool(d["teacher_task_mse"]<=1e-4 and
                d["factor_task_mse"]<=.02 and
                d["weight_reconstruction_relative"]<=.10 and
                d["min_rank3_gap"]>.01)

def admit(phase:str,memory:dict|None,preflight:dict|None=None,release:dict|None=None):
    """Execute before any model generation or donor sampling."""
    if phase not in ("development","confirmation"):
        raise ValueError("unknown evaluation phase")
    validate_memory(memory)
    if phase=="confirmation":
        pf=preflight
        if not isinstance(pf,dict) or any((
            pf.get("schema")!="EvalMath/mechinterp-power-preflight/v1",
            pf.get("decision")!="READY_FOR_CONFIRMATION",
            pf.get("experiment_id")!=NAME,
            pf.get("planned_confirmation_worlds")!=128,
            pf.get("planned_comparisons")!=3,
            pf.get("pilot_worlds")!=64,
            pf.get("mean_pilot_effect",0)<=pf.get("mechanistic_variation_floor",1),
            bool(pf.get("reasons")),
        )):raise ValueError("corrected independent-root EvalMath preflight REQUIRED")
        if not isinstance(release,dict) or any((
            release.get("human_approved") is not True,
            release.get("approval_commit") in (None,""),
            release.get("experiment")!=NAME,
            release.get("source_sha256")!=source_sha256(),
            release.get("memory_sha256")!=memory["sha256"],
            release.get("preflight_sha256")!=m.digest(pf),
            release.get("confirmation_population")!="AMM08/confirmation/0000..0127",
            release.get("n_confirmation_roots")!=128,
            release.get("development_high_fidelity_roots",0)<56,
            release.get("development_oracle_headroom_rate",0)<.08,
            release.get("max_equal_dose_error",1)>1e-5,
            release.get("max_immediate_readout_displacement",1)>0,
        )):raise ValueError("explicit source-locked human confirmation authorization REQUIRED")

def preflight_input(result:dict)->dict:
    if result.get("phase")!="development" or result.get("n_independent_roots")!=64:
        raise ValueError("64 independently trained development roots required")
    shams=[]
    for r in result["rows"]:
        p=r["policy_hits"]
        shams.append({"unit_id":r["root"],
                      "paired_effect":f"{p['memory_plus_local']-p['local_ridge']}/{m.PAIRS}",
                      "sham_effects":[f"{p['local_ridge']-r['repeatability']['local_ridge_hits']}/{m.PAIRS}"]})
    return {"schema_version":1,"experiment_id":NAME,
            "independent_unit":"one 6D rank-three three-pass trained physical and BPD model root",
            "alpha":"0.05","scientific_min_effect":"0.08","target_effect":"0.18",
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
    if result.get("phase") not in ("development","confirmation"):
        raise ValueError("root analysis requires measured cohort")
    comparisons={key:[r["policy_hits"]["memory_plus_local"]-r["policy_hits"][arm]
                      for r in result["rows"]] for key,arm in COMPARISONS.items()}
    pvals={k:exact_root_sign_p(v) for k,v in comparisons.items()}
    adjusted=holm(pvals)
    metrics={k:{"mean_root_effect":sum(v)/(len(v)*m.PAIRS),
                "root_bootstrap_95_ci":cluster_bootstrap(v,k),
                "positive_roots":sum(x>0 for x in v),
                "negative_roots":sum(x<0 for x in v),
                "tied_roots":sum(x==0 for x in v),
                "exact_one_sided_root_sign_p":pvals[k],
                "holm_adjusted_p":adjusted[k]} for k,v in comparisons.items()}
    out={"schema":SCHEMA+"/analysis","status":"development_only" if result["phase"]=="development" else "confirmation",
         "comparisons":metrics}
    if preflight is not None:
        floor=max(SCIENCE_FLOOR,preflight["mechanistic_variation_floor"])
        primary=metrics["memory_vs_local_ridge"]
        maxerr=max(r["heldout"]["diagnostics"]["max_equal_dose_error"]
                   for r in result["rows"])
        maxvisible=max(r["heldout"]["diagnostics"]["max_immediate_readout_displacement"]
                       for r in result["rows"])
        out.update({"mechanistic_variation_floor":floor,
            "max_dose_error":maxerr,"max_immediate_drift":maxvisible,
            "registered_primary_gate":bool(primary["root_bootstrap_95_ci"][0]>floor
                and primary["holm_adjusted_p"]<.05
                and metrics["memory_vs_memory_only"]["mean_root_effect"]>=0
                and maxerr<=1e-5 and maxvisible==0)})
    return out

def evaluate(phase:str,memory:dict|None=None,preflight:dict|None=None,release:dict|None=None)->dict:
    admit(phase,memory,preflight,release)
    assert memory is not None
    rows=[score_root(phase,i,memory) for i in range(m.PHASE_N[phase])]
    out={"schema":SCHEMA+"/cohort","phase":phase,"experiment":NAME,
         "n_independent_roots":len(rows),"training_policy_sha256":memory["sha256"],
         "source_sha256":source_sha256(),"n_calibration_pairs_per_root":m.CAL_PAIRS,
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
            -r["policy_hits"]["local_ridge"] for r in rows)
        out["oracle_headroom_rate_audit_only"]=out["oracle_headroom_hits_audit_only"]/out["total_test_cases"]
        out["max_equal_dose_error"]=max(r["heldout"]["diagnostics"]["max_equal_dose_error"]
                                       for r in rows)
        out["max_immediate_readout_displacement"]=max(
            r["heldout"]["diagnostics"]["max_immediate_readout_displacement"] for r in rows)
    else:
        out["preflight_sha256"]=m.digest(preflight)
        out["release_sha256"]=m.digest(release)
        out["analysis"]=analysis(out,preflight)
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
                "global_coefficients":run["global_coefficients"]}
    out={"schema":SCHEMA+"/cohort_lock","phase":run["phase"],
         "source_sha256":run["source_sha256"],
         "training_policy_sha256":run["training_policy_sha256"],
         "independent_roots":run["n_independent_roots"],
         "total_test_cases":run["total_test_cases"],
         "totals":run["totals"],
         "all_candidate_action_case_bits_sha256":m.digest([
            {"root":r["root"],"clean_calibration":r["calibration_clean_audit_only"]["candidate_bits"],
             "noisy_calibration":r["agent_noisy_feedback"]["reported_bits"],
             "test_candidates":r["heldout"]["candidate_bits"],
             "selected_actions":r["policy_actions"],"selected_outcomes":r["policy_bits"],
             "repeat_local_outcomes":r["repeatability"]["local_ridge_bits"],
             "fidelity_flag":r["fidelity_audit_only"]}
            for r in run["rows"]])}
    if run["phase"]=="development":
        out.update({"high_fidelity_roots":run["high_fidelity_roots"],
                    "oracle_headroom_rate_audit_only":run["oracle_headroom_rate_audit_only"],
                    "max_equal_dose_error":run["max_equal_dose_error"],
                    "max_immediate_readout_displacement":run["max_immediate_readout_displacement"],
                    "preflight_input_sha256":m.digest(run["preflight_input"]),
                    "analysis":run["analysis"]})
    else:out["analysis"]=run["analysis"]
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("engineering","training","development","confirmation"),required=True)
    parser.add_argument("--memory",type=Path)
    parser.add_argument("--preflight",type=Path)
    parser.add_argument("--release",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--check-lock",type=Path)
    args=parser.parse_args()
    if args.phase=="training":result=train()
    elif args.phase=="engineering":
        rows=[]
        for i in range(2):
            teacher,f,metrics=m.fit("engineering",i)
            rows.append({"root":m.root_key("engineering",i),"fit":metrics,
                         "gauge":m.gauge_check(f),
                         "case":case.score("engineering",i,f,"engineering",m.TRAIN_PAIRS)})
        result={"phase":"engineering","rows":rows}
    else:
        if args.memory is None:raise ValueError("requires source-hashed --memory")
        memory=json.loads(args.memory.read_text())
        preflight=json.loads(args.preflight.read_text()) if args.preflight else None
        release=json.loads(args.release.read_text()) if args.release else None
        result=evaluate(args.phase,memory,preflight,release)
    if args.check_lock:
        if args.phase=="engineering":raise ValueError("engineering cannot be locked as scientific result")
        locked=json.loads(args.check_lock.read_text())
        if compact(result)!=locked:raise AssertionError("AMM08 source/outcome replay mismatch")
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write("\n")
    print(json.dumps({"experiment":NAME,"phase":args.phase,
                      "n_roots":result.get("n_independent_roots"),
                      "totals":result.get("totals"),
                      "training_policy_sha256":result.get("sha256")}))

if __name__=="__main__":main()
