"""AMM11: preregistered source-locked memory versus cosine causal routing.

Every scientific root is freshly trained under the pinned CPU runtime.
The experiment has no confirmation execution path. Donor targets, future
donor states and heldout candidate outcome bits never enter policy functions.
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
from . import models as m,case

ROOT=Path(__file__).resolve().parent
NAME=m.NAME
SCHEMA=m.SCHEMA
N_ACTIONS=6
N_FEATURES=18
COMPACT=(0,1,5,6,7,8,9,10,11,14)
ARMS=m.ARMS
COMPARISONS=m.COMPARISONS
GLOBAL_RIDGE=10.
LOCAL_RIDGE=2.
RESIDUAL_RIDGE=20.
JUDGE_ERROR=.20
SCIENCE_MINIMUM=.08
REPS=2000
MAIN_KEYS=tuple(f"mask_{i}" for i in range(N_ACTIONS))
KNOWN_CONTROLS=("random_dark","random_dark_b","sham","full_donor")
POLICY_ARMS=("full_memory","cosine","compact_memory","pooled","local_ridge",
             "memory_plus_local","max_cardinality","random_mask")
assert set(ARMS)==set(POLICY_ARMS+KNOWN_CONTROLS)
assert m.MASKS==((0,),(1,),(2,),(0,1),(0,2),(1,2))

def source_sha256()->dict[str,str]:
    files=("__init__.py","models.py","case.py","experiment.py",
           "test_experiment.py","RUNTIME.json","PROTOCOL.md")
    return {file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in files}

def ridge(X:torch.Tensor,target:torch.Tensor,columns:tuple[int,...],
          lam:float,penalize_intercept:bool=False)->list[float]:
    if X.ndim!=2 or X.shape[1]!=N_FEATURES or target.shape!=(len(X),):
        raise ValueError("fixed 18-dimensional causal-response regression required")
    x=X[:,list(columns)].double()
    y=target.double()
    penalty=torch.eye(len(columns),dtype=torch.float64)*lam
    if not penalize_intercept:penalty[0,0]=0.
    weights=torch.linalg.solve(x.T@x+penalty,x.T@y)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("invalid frozen ridge solution")
    return [float(v) for v in weights]

def training_policy(rows:list,all_features:list[torch.Tensor],all_outcomes:list[torch.Tensor])->dict:
    full=ridge(torch.cat(all_features),torch.cat(all_outcomes),tuple(range(N_FEATURES)),GLOBAL_RIDGE)
    compact=ridge(torch.cat(all_features),torch.cat(all_outcomes),COMPACT,GLOBAL_RIDGE)
    full=[round(x,4) for x in full]
    compact=[round(x,4) for x in compact]
    hits={key:sum(r["assay"]["candidate_hits"][key] for r in rows) for key in MAIN_KEYS}
    pool=min(range(N_ACTIONS),key=lambda i:(-hits[f"mask_{i}"],i))
    bits=m.digest([{"root":r["root"],"candidate_bits":r["assay"]["candidate_bits"]} for r in rows])
    fields={"schema":SCHEMA+"/training_memory","experiment":NAME,"phase":"training",
            "source_sha256":source_sha256(),"runtime":m.runtime_signature(),
            "independent_training_roots":64,"cases_per_root":m.TRAIN_PAIRS,
            "full_coefficients":full,"compact_coefficients":compact,
            "compact_columns":list(COMPACT),"ridge_penalty":GLOBAL_RIDGE,
            "pooled_action":pool,"candidate_hits":hits,"training_candidate_bits_sha256":bits,
            "training_rows_sha256":m.digest(rows)}
    fields["sha256"]=m.digest(fields)
    fields["rows"]=rows
    return fields

def training():
    m.runtime_signature()
    rows=[];features=[];outcomes=[]
    for i in range(64):
        _,f,fit=m.fit("training",i)
        assay=case.score("training",i,f,"training_pairs",m.TRAIN_PAIRS)
        X=torch.tensor(assay["features"],dtype=torch.float64)
        y=torch.tensor([[int(assay["candidate_bits"][f"mask_{a}"][j])
                         for a in range(N_ACTIONS)] for j in range(m.TRAIN_PAIRS)],
                        dtype=torch.float64)
        features.append(X.reshape(-1,N_FEATURES))
        outcomes.append(y.reshape(-1))
        rows.append({"root":m.root_key("training",i),"family":m.family_for(i),
                     "fitting":fit,"assay":assay})
    return training_policy(rows,features,outcomes)

def validate_training(memory:dict,check_live:bool=True)->None:
    if not isinstance(memory,dict) or memory.get("schema")!=SCHEMA+"/training_memory":
        raise ValueError("source-locked AMM11 training memory required")
    if memory.get("source_sha256")!=source_sha256() or memory.get("experiment")!=NAME:
        raise ValueError("experimental source-lock changed")
    if memory.get("independent_training_roots")!=64 or memory.get("cases_per_root")!=128:
        raise ValueError("wrong independent root/case count")
    rows=memory.get("rows",[])
    if len(rows)!=64 or [r.get("root") for r in rows]!=[m.root_key("training",i) for i in range(64)]:
        raise ValueError("training membership or root order drift")
    for i,row in enumerate(rows):
        if row.get("family")!=m.family_for(i) or row["assay"].get("root")!=row["root"]:
            raise ValueError("training family/assay mismatch")
        a=row["assay"]
        if a.get("n")!=128 or a.get("split")!="training_pairs":
            raise ValueError("training donor scope changed")
        if set(a.get("candidate_bits",{}))!=set(MAIN_KEYS+KNOWN_CONTROLS):
            raise ValueError("training physical action space changed")
        if any(len(bits)!=128 or set(bits)-{"0","1"} for bits in a["candidate_bits"].values()):
            raise ValueError("corrupt training candidate outcome bits")
        if m.digest(a.get("features"))!=a.get("features_sha256"):
            raise ValueError("training feature receipt changed")
    if m.digest(rows)!=memory.get("training_rows_sha256"):
        raise ValueError("source-locked full training root receipt changed")
    bit_hash=m.digest([{"root":r["root"],"candidate_bits":r["assay"]["candidate_bits"]} for r in rows])
    if bit_hash!=memory.get("training_candidate_bits_sha256"):
        raise ValueError("training binary case trace mismatch")
    hit_counts={key:sum(r["assay"]["candidate_bits"][key].count("1") for r in rows) for key in MAIN_KEYS}
    if hit_counts!=memory.get("candidate_hits"):
        raise ValueError("training candidate counts mismatch")
    pooled=min(range(6),key=lambda i:(-hit_counts[f"mask_{i}"],i))
    if memory.get("pooled_action")!=pooled or memory.get("compact_columns")!=list(COMPACT):
        raise ValueError("training pooled action/compact feature family changed")
    for key,n in (("full_coefficients",18),("compact_coefficients",10)):
        w=memory.get(key,[])
        if len(w)!=n or any(type(x) not in (int,float) or not math.isfinite(x) or round(x,4)!=x for x in w):
            raise ValueError("training policy quantization invalid")
    fields={k:v for k,v in memory.items() if k not in ("sha256","rows")}
    if m.digest(fields)!=memory.get("sha256"):
        raise ValueError("training policy checksum mismatch")
    if check_live and memory.get("runtime")!=m.runtime_signature():
        raise ValueError("AMM11 pinned CPU runtime changed")

def check_training_lock(memory:dict)->None:
    lock=ROOT/"results/TRAINING_LOCK.json"
    if not lock.exists():raise ValueError("TRAINING_LOCK must be committed before development root access")
    if compact(memory)!=json.loads(lock.read_text()):
        raise ValueError("new model training policy does not match committed lock")

def calibration_feedback(phase:str,index:int,cal:dict)->dict:
    if cal.get("split")!="calibration" or cal.get("n")!=m.CAL_PAIRS:
        raise ValueError("disjoint calibration cohort required")
    root=m.root_key(phase,index)
    bits={}
    for a in range(N_ACTIONS):
        key=f"mask_{a}"
        flips=(torch.rand(m.CAL_PAIRS,generator=m.gen(root,"judge_noise",a))<JUDGE_ERROR).tolist()
        bits[key]="".join(str(int((b=="1")!=flip))
                           for b,flip in zip(cal["candidate_bits"][key],flips))
    return {"schema":SCHEMA+"/reported_feedback","reported_bits":bits,
            "reported_hits":{k:v.count("1") for k,v in bits.items()},
            "noisy_labels":m.CAL_PAIRS*N_ACTIONS,"n_cases":m.CAL_PAIRS,
            "known_error":JUDGE_ERROR}

def fit_local(features:list,feedback:dict,memory:dict)->dict:
    if feedback.get("n_cases")!=m.CAL_PAIRS or feedback.get("noisy_labels")!=96:
        raise ValueError("96-label fixed calibration budget required")
    bits=feedback["reported_bits"]
    if set(bits)!=set(MAIN_KEYS) or any(len(v)!=16 or set(v)-{"0","1"} for v in bits.values()):
        raise ValueError("invalid binary calibration judge outcome")
    X=torch.tensor(features,dtype=torch.float64)
    if X.shape!=(m.CAL_PAIRS,6,18):raise ValueError("invalid calibration causal features")
    y=torch.tensor([[int(bits[f"mask_{a}"][i]) for a in range(6)]
                    for i in range(m.CAL_PAIRS)],dtype=torch.float64)
    corrected=(y-JUDGE_ERROR)/(1-2*JUDGE_ERROR)
    prior=(X@torch.tensor(memory["full_coefficients"],dtype=torch.float64)).clamp(0,1)
    residual=ridge(X.reshape(-1,18),(corrected-prior).reshape(-1),
                   tuple(range(18)),RESIDUAL_RIDGE,penalize_intercept=True)
    local=ridge(X.reshape(-1,18),corrected.reshape(-1),tuple(range(18)),LOCAL_RIDGE)
    return {"full_coefficients":memory["full_coefficients"],
            "compact_coefficients":memory["compact_coefficients"],
            "residual_coefficients":residual,"local_coefficients":local,
            "pooled_action":memory["pooled_action"],
            "feedback_bits":96}

def choose(features:list,policy:dict,root:str)->dict[str,list[int]]:
    X=torch.tensor(features,dtype=torch.float64)
    if X.ndim!=3 or X.shape[1:]!=(6,18):
        raise ValueError("six permitted mask signatures required")
    full=(X@torch.tensor(policy["full_coefficients"],dtype=torch.float64)).clamp(0,1)
    compact=(X[:,:,list(COMPACT)]@torch.tensor(policy["compact_coefficients"],dtype=torch.float64)).clamp(0,1)
    residual=X@torch.tensor(policy["residual_coefficients"],dtype=torch.float64)
    local=X@torch.tensor(policy["local_coefficients"],dtype=torch.float64)
    return {"full_memory":torch.argmax(full,dim=1).tolist(),
            "cosine":torch.argmax(X[:,:,14],dim=1).tolist(),
            "compact_memory":torch.argmax(compact,dim=1).tolist(),
            "pooled":[policy["pooled_action"]]*len(X),
            "local_ridge":torch.argmax(local,dim=1).tolist(),
            "memory_plus_local":torch.argmax((full+residual).clamp(0,1),dim=1).tolist(),
            "max_cardinality":torch.argmax(X[:,:,1],dim=1).tolist(),
            "random_mask":[m.seed(root,"random_mask",i)%6 for i in range(len(X))]}

def fidelity(metrics:dict)->bool:
    return bool(metrics["teacher_task_mse"]<=1e-4
      and metrics["factor_task_mse"]<=.02
      and metrics["weight_reconstruction_relative"]<=.10
      and metrics["min_rank3_gap"]>.01)

def score_root(phase:str,index:int,memory:dict)->dict:
    if phase!="development":raise ValueError("AMM11 confirmation model root sealed")
    _,f,fit=m.fit(phase,index)
    root=m.root_key(phase,index)
    cal=case.score(phase,index,f,"calibration",m.CAL_PAIRS)
    feedback=calibration_feedback(phase,index,cal)
    policy=fit_local(cal["features"],feedback,memory)
    held=case.score(phase,index,f,"heldout",m.PAIRS)
    choices=choose(held["features"],policy,root)
    bits={arm:"".join(held["candidate_bits"][f"mask_{action}"][j]
                      for j,action in enumerate(actions)) for arm,actions in choices.items()}
    for key in KNOWN_CONTROLS:bits[key]=held["candidate_bits"][key]
    repeat=case.score(phase,index,f,"repeatability",m.PAIRS)
    repeat_choice=choose(repeat["features"],policy,root)["cosine"]
    repeat_bits="".join(repeat["candidate_bits"][f"mask_{a}"][j]
                        for j,a in enumerate(repeat_choice))
    # Candidate response geometry was used only for policy actions and then
    # removed from archived development: no future posthoc policy rescoring.
    cal.pop("features")
    held.pop("features")
    repeat.pop("features")
    return {"root":root,"family":m.family_for(index),
            "fitting":fit,"fidelity_flag_audit_only":fidelity(fit),
            "calibration_clean_audit_only":cal,"noisy_feedback":feedback,
            "policy":policy,"heldout":held,
            "policy_actions":choices,"policy_bits":bits,
            "policy_hits":{k:v.count("1") for k,v in bits.items()},
            "repeatability":{"assay":repeat,"cosine_actions":repeat_choice,
                             "cosine_bits":repeat_bits,"cosine_hits":repeat_bits.count("1")}}

def exact_sign(xs:list[int])->float:
    n=sum(x!=0 for x in xs);k=sum(x>0 for x in xs)
    return (sum(math.comb(n,j) for j in range(k,n+1))/(2**n)) if n else 1.

def holm(p:dict[str,float])->dict[str,float]:
    adj={};floor=0.
    for i,name in enumerate(sorted(p,key=lambda k:(p[k],k))):
        floor=max(floor,min(1.,(len(p)-i)*p[name]))
        adj[name]=floor
    return adj

def bootstrap(xs:list[int],name:str)->list[float]:
    rng=random.Random(m.seed("bootstrap",name))
    size=len(xs)
    samples=sorted(sum(xs[rng.randrange(size)] for _ in range(size))/(size*m.PAIRS)
                   for _ in range(REPS))
    return [samples[50],samples[1949]]

def analysis(result:dict)->dict:
    rows=result["rows"]
    effects={name:[r["policy_hits"][left]-r["policy_hits"][right] for r in rows]
             for name,left,right in COMPARISONS}
    ps={k:exact_sign(v) for k,v in effects.items()}
    adjusted=holm(ps)
    return {"schema":SCHEMA+"/independent_root_development","comparisons":{
        name:{"treatment":left,"control":right,
              "difference_hits":sum(effects[name]),
              "mean_root_effect":sum(effects[name])/(64*m.PAIRS),
              "positive_roots":sum(z>0 for z in effects[name]),
              "negative_roots":sum(z<0 for z in effects[name]),
              "tied_roots":sum(z==0 for z in effects[name]),
              "exact_sign_p":ps[name],"holm_adjusted_p":adjusted[name],
              "root_bootstrap_95":bootstrap(effects[name],name)}
        for name,left,right in COMPARISONS}}

def preflight_input(result:dict)->dict:
    if result["phase"]!="development" or len(result["rows"])!=64:
        raise ValueError("independent 64-root development required")
    pilot=[{"unit_id":r["root"],
            "paired_effect":f"{r['policy_hits']['full_memory']-r['policy_hits']['cosine']}/{m.PAIRS}",
            "sham_effects":[f"{r['policy_hits']['cosine']-r['repeatability']['cosine_hits']}/{m.PAIRS}"]}
           for r in result["rows"]]
    return {"schema_version":1,"experiment_id":NAME,
            "independent_unit":"one independently trained 6D rank3 recurrent teacher plus separately trained BPD VU root",
            "alpha":"0.05","scientific_min_effect":"0.08","target_effect":"0.12",
            "measurement_resolution":f"1/{m.PAIRS}",
            "sham_quantile":"0.95","sham_confidence":"0.95",
            "planned_comparisons":len(COMPARISONS),"confirm_units_planned":128,
            "pilot_from_development_only":True,"heldout_sealed":True,
            "units_independent":True,"pilot":pilot}

def development(memory:dict)->dict:
    if (ROOT/"results/DEVELOPMENT_LOCK.json").exists():
        raise ValueError("Development already scored: only immutable receipt auditing allowed")
    validate_training(memory)
    check_training_lock(memory)
    rows=[score_root("development",i,memory) for i in range(64)]
    result={"schema":SCHEMA+"/cohort","experiment":NAME,"phase":"development",
            "source_sha256":source_sha256(),"training_policy_sha256":memory["sha256"],
            "independent_roots":64,"n_cases_per_root":m.PAIRS,"total_test_cases":64*m.PAIRS,
            "totals":{arm:sum(r["policy_hits"][arm] for r in rows) for arm in ARMS},
            "n_calibration_pairs_per_root":m.CAL_PAIRS,
            "noise_labels_per_root":96,
            "high_fidelity_roots":sum(r["fidelity_flag_audit_only"] for r in rows),
            "casewise_oracle_headroom_rate_audit_only":sum(
                r["heldout"]["diagnostics"]["casewise_oracle_hits_audit_only"]
                    -r["policy_hits"]["cosine"] for r in rows)/(64*m.PAIRS),
            "max_equal_dose_error":max(r["heldout"]["diagnostics"]["max_equal_dose_error"] for r in rows),
            "max_immediate_readout_displacement":max(r["heldout"]["diagnostics"]["max_immediate_readout_displacement"] for r in rows),
            "rows":rows}
    result["preflight_input"]=preflight_input(result)
    result["analysis"]=analysis(result)
    return result

def compact(result:dict)->dict:
    if result["phase"]=="training":
        return {"schema":SCHEMA+"/training_lock",
                "experiment":NAME,"source_sha256":result["source_sha256"],
                "training_policy_sha256":result["sha256"],
                "independent_roots":64,
                "train_mask_hits":result["candidate_hits"],
                "pooled_action":result["pooled_action"],
                "full_coefficients":result["full_coefficients"],
                "compact_coefficients":result["compact_coefficients"],
                "train_case_bits_sha256":result["training_candidate_bits_sha256"],
                "training_rows_sha256":result["training_rows_sha256"]}
    if result["phase"]!="development":raise ValueError("only scientific training/development may lock")
    return {"schema":SCHEMA+"/development_lock",
            "source_sha256":result["source_sha256"],
            "training_policy_sha256":result["training_policy_sha256"],
            "independent_roots":result["independent_roots"],
            "total_test_cases":result["total_test_cases"],
            "totals":result["totals"],
            "case_outcome_action_sha256":m.digest([{
                "root":r["root"],"calibration":r["calibration_clean_audit_only"]["candidate_bits"],
                "noisy_reports":r["noisy_feedback"]["reported_bits"],
                "candidates":r["heldout"]["candidate_bits"],
                "selected_actions":r["policy_actions"],
                "selected_outcomes":r["policy_bits"],
                "repeat_cosine":r["repeatability"]["cosine_bits"],
                "fidelity":r["fidelity_flag_audit_only"]} for r in result["rows"]]),
            "preflight_input_sha256":m.digest(result["preflight_input"]),
            "fidelity_roots":result["high_fidelity_roots"],
            "headroom_rate":result["casewise_oracle_headroom_rate_audit_only"],
            "max_dose_error":result["max_equal_dose_error"],
            "max_readout_drift":result["max_immediate_readout_displacement"],
            "analysis":result["analysis"]}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("engineering","training","development"),required=True)
    parser.add_argument("--memory",type=Path)
    parser.add_argument("--check-lock",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.phase=="engineering":
        _,f,fit=m.fit("engineering",0)
        result={"phase":"engineering","fit":fit,"gauge":m.gauge_check(f),
                "sample":case.score("engineering",0,f,"engineering",m.TRAIN_PAIRS)}
    elif args.phase=="training":
        if (ROOT/"results/TRAINING_LOCK.json").exists() and not args.check_lock:
            raise ValueError("training is locked: supply --check-lock to replay")
        result=training()
    else:
        if args.memory is None:raise ValueError("source-locked --memory required")
        result=development(json.loads(args.memory.read_text()))
    if args.check_lock:
        if compact(result)!=json.loads(args.check_lock.read_text()):
            raise AssertionError("source-locked numerical/case replay mismatch")
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x") as file:
            json.dump(result,file,indent=2,sort_keys=True,allow_nan=False)
            file.write("\n")
    print(f"AMM11_PHASE={args.phase}")
    if args.phase=="training":print("AMM11_TRAINING_LOCK="+json.dumps(compact(result),sort_keys=True))
    if args.phase=="development":print("AMM11_DEVELOPMENT_LOCK="+json.dumps(compact(result),sort_keys=True))

if __name__=="__main__":main()
