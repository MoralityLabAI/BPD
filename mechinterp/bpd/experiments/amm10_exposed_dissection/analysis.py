"""AMM10: replay exposed AMM09R1 roots; train-CV ablations, development diagnostics.

No model generation, no opening confirmation, no development feature reconstruction.
The published 64+64 roots are explicitly selection exposed. All results exploratory.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random
from pathlib import Path
from collections import Counter
import numpy as np

TRAIN_SHA="015f8c78181514a0702e56ad707444699e48b377d9bf29da8ec60a42a0d48d30"
DEV_SHA="682473974fa15db6005dab588d66d73c2a58bba1d5c8f1f4a91a93899ea88e5a"
POLICY_SHA="b34b9a322ff5aeaa517eb5edd4bf0aced9ccf70f0bf97383d8e6cca79de2f0e9"
NAME="BPD-AMM-09R1"
CANDS=tuple(f"mask_{i}" for i in range(6))
CONTROL=("random_dark","random_dark_b","sham","full_donor")
SOURCE_ARMS=("memory_only","pooled","memory_plus_local","local_ridge","local_count","random_mask")
VARIANTS={
 "full18":tuple(range(18)),
 "no_family":tuple(i for i in range(18) if i!=17),
 "no_alignment":tuple(i for i in range(18) if i!=14),
 "no_delayed_probes":(0,1,2,3,4,14,15,16,17),
 "no_geometry":(0,1,2,3,5,6,7,8,9,10,11,12,13,17),
 "linear_probe":(0,1,5,6,12,13,14),
 "compact_causal":(0,1,5,6,7,8,9,10,11,14),
 "cardinality_only":(0,1),
 "alignment_only":(0,14),
}
HEURISTICS=("max_cosine","max_cardinality","max_balanced_shift",
 "max_total_shift","max_signed_shift_product","max_shift_norm")
CV_BASELINES=("pooled_fold","family_pooled_fold","root_fixed_hindsight","casewise_oracle")
FAMILIES=("wren","basalt")

def canonical(o):return json.dumps(o,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(o):return hashlib.sha256(canonical(o).encode()).hexdigest()
def raw_json(p,sha):
 b=Path(p).read_bytes()
 if hashlib.sha256(b).hexdigest()!=sha:raise ValueError("original source-of-record raw hash mismatch: "+str(p))
 return json.loads(b)

def members(rows,phase,n):
 if not isinstance(rows,list) or len(rows)!=64:raise ValueError("64 exposed model roots required")
 for i,r in enumerate(rows):
  if r.get("root")!=f"{phase}/{i:04d}" or "confirmation" in r["root"] or r.get("family")!=FAMILIES[i%2]:
   raise ValueError("phase/family/root-order mismatch: "+str(i))
 return rows

def read(training,development):
 tr=raw_json(training,TRAIN_SHA);dev=raw_json(development,DEV_SHA)
 if tr.get("experiment")!=NAME or tr.get("phase")!="training" or tr.get("sha256")!=POLICY_SHA:
  raise ValueError("different historical training policy")
 if dev.get("experiment")!=NAME or dev.get("phase")!="development" or dev.get("training_policy_sha256")!=POLICY_SHA:
  raise ValueError("different historical development policy")
 if tr.get("source_sha256")!=dev.get("source_sha256") or dev.get("total_test_cases")!=32768 or dev.get("pooled_action")!=4:
  raise ValueError("source/case/pooled policy mismatch")
 members(tr["rows"],"training",64);members(dev["rows"],"development",64)
 return tr,dev

def training_arrays(tr):
 xx=[];yy=[]
 for r in tr["rows"]:
  a=r["assay"]
  if a.get("root")!=r["root"] or a.get("split")!="training_pairs" or a.get("n")!=128 or a.get("features_sha256")!=digest(a.get("features")):
   raise ValueError("training feature/assay commitment failed")
  X=np.asarray(a["features"],dtype=np.float64)
  bits=a["candidate_bits"]
  if X.shape!=(128,6,18) or not np.isfinite(X).all() or set(bits)!=set(CANDS+CONTROL):
   raise ValueError("invalid candidate feature/bit shape")
  if any(len(z)!=128 or set(z)-{"0","1"} for z in bits.values()):raise ValueError("invalid original training candidate bits")
  y=np.asarray([[int(bits[f"mask_{j}"][i]) for j in range(6)] for i in range(128)],dtype=np.int8)
  xx.append(X);yy.append(y)
 X=np.stack(xx);Y=np.stack(yy)
 if [int(Y[:,:,i].sum()) for i in range(6)]!=[tr["candidate_hits"][key] for key in CANDS]:
  raise ValueError("original 64-root training hits changed")
 return X,Y

def score_identity(X,coeff):
 """Prove/check preclip full18 = common + nine candidate-varying terms."""
 b=np.asarray(coeff,dtype=np.float64);x=X.reshape(-1,6,18)
 common=(x[:,0,0]*b[0]+x[:,0,2]*b[2]+x[:,0,3]*b[3]+x[:,0,4]*b[4]
  +b[12]*(x[:,0,12]-x[:,0,5])+b[13]*(x[:,0,13]-x[:,0,6])
  +x[:,0,15]*b[15]+x[:,0,16]*b[16]+x[:,0,17]*b[17])
 variable=(b[1]*x[:,:,1]+(b[5]+b[12])*x[:,:,5]+(b[6]+b[13])*x[:,:,6]
  +b[7]*x[:,:,7]+b[8]*x[:,:,8]+b[9]*x[:,:,9]+b[10]*x[:,:,10]
  +b[11]*x[:,:,11]+b[14]*x[:,:,14])
 full=x@b;reduced=common[:,None]+variable
 keep=(0,2,3,4,15,16,17)
 spread=float(np.max(abs(x[:,:,keep]-x[:,0:1,keep])))
 return {"n_training_cases":int(len(x)),"max_candidate_invariant_spread":spread,
  "max_preclip_score_error":float(np.max(abs(full-reduced))),
  "preclip_action_mismatch_count":int(np.count_nonzero(np.argmax(full,axis=1)!=np.argmax(reduced,axis=1))),
  "postclip_action_mismatch_count":int(np.count_nonzero(np.argmax(np.clip(full,0,1),axis=1)!=np.argmax(np.clip(reduced,0,1),axis=1))),
  "effective_coefficients":{
   "cardinality":float(b[1]),"delta2":float(b[5]+b[12]),"delta3":float(b[6]+b[13]),
   "abs_delta2":float(b[7]),"abs_delta3":float(b[8]),"delta2_sq":float(b[9]),
   "delta3_sq":float(b[10]),"delta2_delta3":float(b[11]),"cosine":float(b[14])}}

def ridge(x,y,cols):
 cols=list(cols);X=x[:,cols]
 reg=np.eye(len(cols),dtype=np.float64)*10.;reg[0,0]=0
 coef=np.linalg.solve(X.T@X+reg,X.T@y)
 if not np.isfinite(coef).all():raise ValueError("unstable root-held-out ridge")
 return coef

def heuristic(x,name):
 if name=="max_cosine":sc=x[:,:,14]
 elif name=="max_cardinality":sc=x[:,:,1]
 elif name=="max_balanced_shift":sc=np.minimum(x[:,:,7],x[:,:,8])
 elif name=="max_total_shift":sc=x[:,:,7]+x[:,:,8]
 elif name=="max_signed_shift_product":sc=x[:,:,11]
 elif name=="max_shift_norm":sc=x[:,:,9]+x[:,:,10]
 else:raise ValueError("heuristic arm not in frozen plan")
 return np.argmax(sc,axis=1)

def effect(a,b,root_size,tag):
 delta=[int(x-y) for x,y in zip(a,b)]
 rng=random.Random(int(hashlib.sha256(("AMM10/exposed/bootstrap/"+tag).encode()).hexdigest()[:15],16))
 draws=sorted(sum(delta[rng.randrange(64)] for _ in range(64))/(64*root_size) for _ in range(2000))
 pos=sum(d>0 for d in delta);negative=sum(d<0 for d in delta);n=pos+negative
 p=sum(math.comb(n,k) for k in range(pos,n+1))/(2**n) if n else 1.
 return {"delta_hits":sum(delta),"delta_rate":sum(delta)/(64*root_size),
         "positive_roots":pos,"negative_roots":negative,"tied_roots":64-n,
         "descriptive_sign_p":p,"descriptive_cluster_bootstrap_95":[draws[50],draws[1949]]}

def crossfit(X,Y):
 arms=tuple(VARIANTS)+HEURISTICS+CV_BASELINES
 hits={key:[None]*64 for key in arms}
 decisions={key:[None]*64 for key in tuple(VARIANTS)+HEURISTICS+CV_BASELINES[:2]}
 folds=[]
 for fold in range(8):
  fit=[i for i in range(64) if i%8!=fold]
  hold=[i for i in range(64) if i%8==fold]
  xfit=X[fit].reshape(-1,18);yfit=Y[fit].reshape(-1)
  models={k:ridge(xfit,yfit,v) for k,v in VARIANTS.items()}
  pool=int(np.sum(Y[fit],axis=(0,1)).argmax())
  per_family={j:int(np.sum(Y[[i for i in fit if i%2==j]],axis=(0,1)).argmax()) for j in (0,1)}
  folds.append({"fold":fold,"heldout_training_roots":hold,"pooled_mask":pool,
   "family_pooled_masks":{FAMILIES[j]:per_family[j] for j in (0,1)}})
  for r in hold:
   x=X[r];y=Y[r];ix=np.arange(128)
   selected={k:np.argmax(np.clip(x[:,:,list(VARIANTS[k])]@w,0,1),axis=1) for k,w in models.items()}
   selected.update({k:heuristic(x,k) for k in HEURISTICS})
   selected["pooled_fold"]=np.full(128,pool,dtype=np.int32)
   selected["family_pooled_fold"]=np.full(128,per_family[r%2],dtype=np.int32)
   for k,actions in selected.items():
    decisions[k][r]=actions.tolist()
    hits[k][r]=int(y[ix,actions].sum())
   hits["root_fixed_hindsight"][r]=int(y.sum(axis=0).max())
   hits["casewise_oracle"][r]=int(y.any(axis=1).sum())
 if any(any(v is None for v in a) for a in hits.values()):raise ValueError("missing root-held-out fold")
 output={k:{"hits":sum(v),"rate":sum(v)/8192,"per_root_hits":v,
  "vs_pooled_fold":effect(v,hits["pooled_fold"],128,k)} for k,v in hits.items()}
 pairs=(("full18","compact_causal"),("full18","max_cosine"),("full18","no_alignment"),
        ("full18","no_delayed_probes"),("alignment_only","max_cosine"))
 agreement={}
 for a,b in pairs:
  different=sum(sum(x!=y for x,y in zip(decisions[a][i],decisions[b][i])) for i in range(64))
  agreement[a+"__vs__"+b]={"different_actions":different,"same_action_fraction":(8192-different)/8192,
   "behavioral_delta":effect(hits[a],hits[b],128,a+"__vs__"+b)}
 return {"schema":"bpd.amm10.exploratory.crossfit.v1","root_count":64,"folds":folds,
  "fold_assignment":"root_index%8","root_holdout_budget":"8 heldout roots and 56 training roots per fold",
  "case_count":8192,"ridge_lambda":10,"ridge_unpenalized_intercept":True,
  "variant_feature_indices":{k:list(v) for k,v in VARIANTS.items()},
  "analytic_rules":list(HEURISTICS),"arms":output,"action_agreement":agreement}

def development(tr,dev):
 family_train=np.zeros((2,6),dtype=np.int32)
 for i,r in enumerate(tr["rows"]):
  for a in range(6):family_train[i%2,a]+=r["assay"]["candidate_bits"][f"mask_{a}"].count("1")
 family_best=[int(np.argmax(family_train[j])) for j in (0,1)]
 per_root={k:[] for k in SOURCE_ARMS+("family_pooled","root_fixed_hindsight","casewise_oracle")}
 hist={k:np.zeros((2,6),dtype=np.int32) for k in SOURCE_ARMS}
 fixed=np.zeros(6,dtype=np.int32);root_best=[];multi=0;switches=0;beats_hindsight=0
 for i,r in enumerate(dev["rows"]):
  c=r["heldout"]["candidate_bits"]
  if set(c)!=set(CANDS+CONTROL) or any(len(v)!=512 or set(v)-{"0","1"} for v in c.values()):
   raise ValueError("corrupt exposed development action space")
  if "features" in r["heldout"] or not isinstance(r["heldout"].get("features_sha256"),str):
   raise ValueError("development features are deliberately absent: no ablated rescoring")
  outcomes=np.asarray([[int(c[f"mask_{a}"][j]) for a in range(6)] for j in range(512)],dtype=np.int8)
  bymask=outcomes.sum(axis=0);fixed+=bymask
  best=int(np.argmax(bymask));root_best.append(best)
  per_root["root_fixed_hindsight"].append(int(bymask[best]))
  per_root["casewise_oracle"].append(int(outcomes.any(axis=1).sum()))
  per_root["family_pooled"].append(int(bymask[family_best[i%2]]))
  if per_root["casewise_oracle"][-1]!=r["heldout"]["diagnostics"]["casewise_oracle_hits_audit_only"]:
   raise ValueError("audit-only oracle count changed")
  for k in SOURCE_ARMS:
   chosen=r["policy_actions"][k]
   if len(chosen)!=512 or any(type(a) is not int or a not in range(6) for a in chosen):
    raise ValueError("invalid exposed development chosen mask")
   expected="".join(c[f"mask_{a}"][j] for j,a in enumerate(chosen))
   if expected!=r["policy_bits"][k] or expected.count("1")!=r["policy_hits"][k]:
    raise ValueError("selected case bits disagree with original data")
   per_root[k].append(expected.count("1"));hist[k][i%2]+=np.bincount(chosen,minlength=6)
  if any(a!=4 for a in r["policy_actions"]["pooled"]):raise ValueError("original pooled mask changed")
  chosen_memory=r["policy_actions"]["memory_only"]
  switches+=sum(a!=4 for a in chosen_memory)
  multi+=len(set(chosen_memory))>1
  beats_hindsight+=int(per_root["memory_only"][-1]>int(bymask.max()))
 if (sum(per_root["memory_only"]),sum(per_root["pooled"]),sum(per_root["memory_plus_local"]))!=(12502,8038,11340):
  raise ValueError("original published development totals changed")
 arms={k:{"hits":sum(v),"rate":sum(v)/32768,"per_root_hits":v,
  "vs_pooled":effect(v,per_root["pooled"],512,k)} for k,v in per_root.items()}
 return {"schema":"bpd.amm10.exploratory.development.v1","exposed_development_roots":64,
  "n_cases":32768,"train_selected_family_pooled_masks":{FAMILIES[j]:family_best[j] for j in (0,1)},
  "training_family_mask_hits":{FAMILIES[j]:family_train[j].tolist() for j in (0,1)},
  "hindsight_fixed_global_mask":int(np.argmax(fixed)),
  "hindsight_fixed_global_hits":int(fixed.max()),"fixed_mask_development_hits":fixed.tolist(),
  "hindsight_root_best_fixed_masks":root_best,"memory_only_multi_mask_roots":multi,
  "memory_only_cases_different_from_pooled":switches,
  "memory_only_roots_exceeding_their_hindsight_best_fixed":beats_hindsight,
  "mask_histograms_by_family":{k:{FAMILIES[j]:hist[k][j].tolist() for j in (0,1)} for k in SOURCE_ARMS},
  "arms":arms,"full_development_features_used":False,
  "note":"all development rows exposed, hindsight rules have access to development labels; no new prospective policies scored"}

def analyze(train_file,dev_file):
 tr,dev=read(train_file,dev_file)
 X,Y=training_arrays(tr)
 return {"schema":"bpd.amm10.exposed_dissection.v1",
  "evidence":"exposed retrospective diagnostics and training root-held-out exploratory refits",
  "raw_training_sha256":TRAIN_SHA,"raw_development_sha256":DEV_SHA,
  "source_experiment":NAME,"frozen_training_policy_sha256":POLICY_SHA,
  "exact_rank_identity":score_identity(X,tr["global_coefficients"]),
  "training_cv":crossfit(X,Y),"development":development(tr,dev),
  "confirmation_roots_opened":0,"verdict":"EXPLORATORY_ONLY_NO_CONFIRMATION"}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--training",type=Path,required=True)
 parser.add_argument("--development",type=Path,required=True)
 parser.add_argument("--out",type=Path,required=True)
 parser.add_argument("--check-lock",type=Path)
 args=parser.parse_args()
 result=analyze(args.training,args.development)
 if args.check_lock:
  if result!=json.loads(args.check_lock.read_text()):raise AssertionError("AMM10 exact original-receipt replay drift")
 args.out.parent.mkdir(parents=True,exist_ok=True)
 args.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
 print("AMM10_IDENTITY="+json.dumps(result["exact_rank_identity"],sort_keys=True))
 print("AMM10_CV_RATES="+json.dumps({k:round(v["rate"],8) for k,v in result["training_cv"]["arms"].items()},sort_keys=True))
 print("AMM10_DEV_RATES="+json.dumps({k:round(v["rate"],8) for k,v in result["development"]["arms"].items()},sort_keys=True))
 print("AMM10_CV_ACTION_AGREEMENT="+json.dumps(result["training_cv"]["action_agreement"],sort_keys=True))
 print("AMM10_RESULT_SHA256="+hashlib.sha256(args.out.read_bytes()).hexdigest())

if __name__=="__main__":main()
