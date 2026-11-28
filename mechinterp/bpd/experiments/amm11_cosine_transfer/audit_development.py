"""Independent, source-only AMM11 development receipt audit.

Read only the once-scored original GitHub Actions JSON archives and committed
locks. No import of torch, no model fitting, no access to confirmation roots.
Registered comparison family and bootstrap seed stream come from PROTOCOL.md.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random
from collections import Counter
from pathlib import Path

TRAIN_SHA="88f4ff475f301f77f21839d25ac4fa42930a7c3ed723a87428fcc43a479ca2f6"
DEV_SHA="074d6d1ca8c3f3dc9f4847b437a7fa028393b7291a9004d905d088c8efe1a0cd"
PF_SHA="ab347a148fee80342e8b580bff4ef323972cbf3556756e2ee681f53983f03844"
TRAIN_POLICY_SHA="eef8c22c96737ee027297b3fbffe58d11df3d13b31a7b9d55a5a0ae98840ab08"
NAME="BPD-AMM-11"
SCHEMA="bpd.amm11.cosine_transfer.v1"
MASKS=tuple(f"mask_{i}" for i in range(6))
CONTROLS=("random_dark","random_dark_b","sham","full_donor")
POLICIES=("full_memory","cosine","compact_memory","pooled","local_ridge",
          "memory_plus_local","max_cardinality","random_mask")
ARMS=POLICIES+CONTROLS
COMPARISONS=(
 ("full_vs_cosine","full_memory","cosine"),
 ("compact_vs_cosine","compact_memory","cosine"),
 ("full_vs_compact","full_memory","compact_memory"),
 ("full_vs_pooled","full_memory","pooled"),
 ("local_update_vs_full","memory_plus_local","full_memory"),
)

def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def original(path,expected):
 if sha(path)!=expected:raise AssertionError("original source-of-record SHA256 mismatch "+str(path))
 return json.loads(Path(path).read_text())

def root_rows(rows,phase):
 if not isinstance(rows,list) or len(rows)!=64:raise AssertionError("64 independent model roots required")
 for i,row in enumerate(rows):
  if row.get("root")!=f"{phase}/{i:04d}":raise AssertionError("root namespace/order or uniqueness changed")
  if row.get("family")!=("wren" if i%2==0 else "basalt"):raise AssertionError("root task family changed")
  if "confirmation" in row["root"]:raise AssertionError("protected root access")

def bitstrings(x,n,keys):
 if not isinstance(x,dict) or set(x)!=set(keys):raise AssertionError("candidate/arm set differs from registration")
 if any(not isinstance(v,str) or len(v)!=n or set(v)-{"0","1"} for v in x.values()):
  raise AssertionError("malformed binary outcome string")
 return {key:x[key].count("1") for key in keys}

def validate_assay(assay,root,split,n):
 if assay.get("root")!=root or assay.get("split")!=split or assay.get("n")!=n:
  raise AssertionError("wrong assay source/split")
 counts=bitstrings(assay.get("candidate_bits"),n,MASKS+CONTROLS)
 if assay.get("candidate_hits")!=counts:raise AssertionError("candidate label counts changed")
 case_oracle=sum(any(assay["candidate_bits"][key][i]=="1" for key in MASKS) for i in range(n))
 d=assay["diagnostics"]
 if d["casewise_oracle_hits_audit_only"]!=case_oracle or d["full_donor_hits_audit_only"]!=counts["full_donor"]:
  raise AssertionError("audit-only oracle/ceiling accounting drift")
 if len(assay.get("features_sha256",""))!=64:raise AssertionError("retained feature SHA missing")
 if "features" in assay and digest(assay["features"])!=assay["features_sha256"]:
  raise AssertionError("original full feature tensor SHA mismatch")
 return counts

def one_sided_sign(xs):
 nz=[v for v in xs if v!=0]
 n=len(nz);k=sum(v>0 for v in nz)
 return sum(math.comb(n,j) for j in range(k,n+1))/(2**n) if n else 1.

def holm(ps):
 output={};floor=0.
 for i,key in enumerate(sorted(ps,key=lambda k:(ps[k],k))):
  floor=max(floor,min(1.,(len(ps)-i)*ps[key]))
  output[key]=floor
 return output

def bootstrap(xs,name):
 rng=random.Random(int(hashlib.sha256(("AMM11/bootstrap/"+name).encode()).hexdigest()[:15],16))
 n=len(xs)
 samples=sorted(sum(xs[rng.randrange(n)] for _ in range(n))/(n*512)
                for _ in range(2000))
 return [samples[50],samples[1949]]

def report(training,dev,planner,training_lock,dev_lock):
 if training.get("experiment")!=NAME or training.get("phase")!="training":raise AssertionError("wrong original training")
 if dev.get("experiment")!=NAME or dev.get("phase")!="development":raise AssertionError("wrong original development")
 if training["sha256"]!=dev["training_policy_sha256"] or training["sha256"]!=TRAIN_POLICY_SHA:
  raise AssertionError("training policy/independent development binding changed")
 if training["source_sha256"]!=dev["source_sha256"] or training_lock["source_sha256"]!=dev_lock["source_sha256"]:
  raise AssertionError("source/frozen hash mismatch")
 if dev["source_sha256"]!=dev_lock["source_sha256"] or dev_lock["training_policy_sha256"]!=TRAIN_POLICY_SHA:
  raise AssertionError("invalid development source lock")
 if training_lock["training_policy_sha256"]!=TRAIN_POLICY_SHA or training_lock["independent_roots"]!=64:
  raise AssertionError("training lock incorrect")
 if dev.get("independent_roots")!=64 or dev.get("total_test_cases")!=32768 or dev.get("noise_labels_per_root")!=96:
  raise AssertionError("development unit/case budget changed")
 root_rows(training["rows"],"training");root_rows(dev["rows"],"development")
 trbits=[]
 for r in training["rows"]:
  a=r["assay"];validate_assay(a,r["root"],"training_pairs",128)
  trbits.append({"root":r["root"],"candidate_bits":a["candidate_bits"]})
 if digest(trbits)!=training_lock["train_case_bits_sha256"]:
  raise AssertionError("training source outcome bit commitment mismatch")
 if training_lock["pooled_action"]!=training["pooled_action"] or training["pooled_action"] not in range(6):
  raise AssertionError("pooled fixed action changed")
 if training_lock["train_mask_hits"]!={k:sum(r["assay"]["candidate_hits"][k] for r in training["rows"]) for k in MASKS}:
  raise AssertionError("training mask counts changed")
 if training_lock["full_coefficients"]!=training["full_coefficients"] or training_lock["compact_coefficients"]!=training["compact_coefficients"]:
  raise AssertionError("training coefficients changed")
 root_totals=Counter();case_digest=[];shams=[]
 hist={arm:Counter() for arm in POLICIES}
 by_family={family:Counter() for family in ("wren","basalt")}
 flips=0;fidelity_count=0;physical_dose_error=0.;readout_error=0.
 for i,r in enumerate(dev["rows"]):
  root=r["root"];cal=r["calibration_clean_audit_only"];held=r["heldout"];repeat=r["repeatability"]
  validate_assay(cal,root,"calibration",16)
  validate_assay(held,root,"heldout",512)
  validate_assay(repeat["assay"],root,"repeatability",512)
  if "features" in held or "features" in repeat["assay"]:
   raise AssertionError("posthoc development policy features must remain excluded")
  noisy=r["noisy_feedback"];reports=noisy["reported_bits"]
  bitstrings(reports,16,MASKS)
  if noisy.get("noisy_labels")!=96 or noisy.get("known_error")!=.2:
   raise AssertionError("judge calibration budget or noise changed")
  flips+=sum(a!=b for key in MASKS for a,b in zip(cal["candidate_bits"][key],reports[key]))
  choices=r["policy_actions"];bits=r["policy_bits"]
  if set(choices)!=set(POLICIES):raise AssertionError("policy family changed")
  bitstrings(bits,512,ARMS)
  reconstructed={}
  for arm in POLICIES:
   vals=choices[arm]
   if (not isinstance(vals,list) or len(vals)!=512 or
       any(type(a) is not int or a not in range(6) for a in vals)):
    raise AssertionError("inadmissible choice")
   reconstructed[arm]="".join(held["candidate_bits"][f"mask_{a}"][j] for j,a in enumerate(vals))
   hist[arm].update(vals)
  for arm in CONTROLS:reconstructed[arm]=held["candidate_bits"][arm]
  if reconstructed!=bits or {key:bits[key].count("1") for key in ARMS}!=r["policy_hits"]:
   raise AssertionError("selected outcomes do not match registered intervention masks")
  if choices["pooled"]!=[training["pooled_action"]]*512:
   raise AssertionError("frozen pooled action drift")
  rb="".join(repeat["assay"]["candidate_bits"][f"mask_{a}"][j]
             for j,a in enumerate(repeat["cosine_actions"]))
  if rb!=repeat["cosine_bits"] or repeat["cosine_hits"]!=rb.count("1"):
   raise AssertionError("frozen cosine repeatability sham mismatch")
  root_totals.update(r["policy_hits"])
  by_family[r["family"]].update(r["policy_hits"])
  fid=(r["fitting"]["teacher_task_mse"]<=1e-4 and
       r["fitting"]["factor_task_mse"]<=.02 and
       r["fitting"]["weight_reconstruction_relative"]<=.10 and
       r["fitting"]["min_rank3_gap"]>.01)
  if bool(r["fidelity_flag_audit_only"])!=fid:raise AssertionError("source fit flag mismatch")
  fidelity_count+=int(fid)
  physical_dose_error=max(physical_dose_error,held["diagnostics"]["max_equal_dose_error"])
  readout_error=max(readout_error,held["diagnostics"]["max_immediate_readout_displacement"])
  pilot=dev["preflight_input"]["pilot"][i]
  e=f"{r['policy_hits']['full_memory']-r['policy_hits']['cosine']}/512"
  sham=f"{r['policy_hits']['cosine']-repeat['cosine_hits']}/512"
  if pilot!={"unit_id":root,"paired_effect":e,"sham_effects":[sham]}:
   raise AssertionError("registered primary/sham row misidentified")
  shams.append(abs(r["policy_hits"]["cosine"]-repeat["cosine_hits"])/512)
  case_digest.append({"root":root,"calibration":cal["candidate_bits"],"noisy_reports":reports,
            "candidates":held["candidate_bits"],"selected_actions":choices,
            "selected_outcomes":bits,"repeat_cosine":repeat["cosine_bits"],
            "fidelity":r["fidelity_flag_audit_only"]})
 if dict(root_totals)!=dev["totals"] or dict(root_totals)!=dev_lock["totals"]:
  raise AssertionError("development all-arm totals mismatch")
 if digest(case_digest)!=dev_lock["case_outcome_action_sha256"]:
  raise AssertionError("original casewise semantic SHA mismatch")
 if digest(dev["preflight_input"])!=dev_lock["preflight_input_sha256"]:
  raise AssertionError("original pilot source SHA mismatch")
 if dev_lock["fidelity_roots"]!=fidelity_count:
  raise AssertionError("fidelity root audit mismatch")
 if max(shams)!=planner["upper_sham_quantile"] or planner["observed_worst_sham"]!=max(shams):
  raise AssertionError("95/95 independent-root sham tolerance mismatch")
 if planner.get("planned_comparisons")!=len(COMPARISONS) or planner.get("pilot_worlds")!=64:
  raise AssertionError("preflight comparison family or model count changed")
 if planner["experiment_id"]!=NAME or planner["mean_pilot_effect"]!=(root_totals["full_memory"]-root_totals["cosine"])/32768:
  raise AssertionError("preflight primary effect mismatch")
 comparisons={};ps={}
 for name,left,right in COMPARISONS:
  differences=[r["policy_hits"][left]-r["policy_hits"][right] for r in dev["rows"]]
  p=one_sided_sign(differences);ps[name]=p
  comparisons[name]={"treatment":left,"control":right,
    "difference_hits":sum(differences),"mean_root_effect":sum(differences)/32768,
    "positive_roots":sum(x>0 for x in differences),
    "negative_roots":sum(x<0 for x in differences),
    "tied_roots":sum(x==0 for x in differences),
    "exact_sign_p":p,"root_bootstrap_95":bootstrap(differences,name)}
 adjusted=holm(ps)
 for name in comparisons:
  comparisons[name]["holm_adjusted_p"]=adjusted[name]
  original=dev["analysis"]["comparisons"][name]
  if comparisons[name]!=original:
   raise AssertionError("published root stats/bootstraps mismatch "+name)
 if comparisons!=dev_lock["analysis"]["comparisons"]:
  raise AssertionError("committed independent-root inference changed")
 if fidelity_count!=dev["high_fidelity_roots"] or abs(physical_dose_error-dev["max_equal_dose_error"])>1e-12:
  raise AssertionError("original physical integrity data mismatch")
 if readout_error!=dev["max_immediate_readout_displacement"]:
  raise AssertionError("current-readout parity changed")
 if planner["decision"]!="REDESIGN_OR_PILOT_ONLY" or planner["mean_pilot_effect"]>=planner["mechanistic_variation_floor"]:
  raise AssertionError("registered null decision changed")
 return {"schema":SCHEMA+"/independent_raw_receipt_audit",
         "evidence":"development-only synthetic trained linear BPD, original archives",
         "n_independent_model_roots":64,"n_test_cases":32768,
         "raw_training_sha256":TRAIN_SHA,"raw_development_sha256":DEV_SHA,
         "raw_preflight_sha256":PF_SHA,
         "training_policy_sha256":TRAIN_POLICY_SHA,
         "source_sha256":dev["source_sha256"],
         "casewise_semantic_sha256":dev_lock["case_outcome_action_sha256"],
         "comparisons":comparisons,"totals":dict(root_totals),
         "mask_histograms":{a:{f"mask_{i}":hist[a][i] for i in range(6)} for a in POLICIES},
         "source_families":{k:dict(v) for k,v in by_family.items()},
         "actual_noisy_feedback_flips":flips,"n_feedback_bits":64*96,
         "fidelity_roots":fidelity_count,
         "max_physical_dose_error":physical_dose_error,
         "max_immediate_readout_drift":readout_error,
         "independent_cosine_sham_95_95_max":max(shams),
         "mean_primary_effect":planner["mean_pilot_effect"],
         "original_evalmath_verdict":planner["decision"],
         "registered_scientific_minimum":.08,
         "confirmation_roots_scored":0,
         "scientific_verdict":"REDESIGN_OR_PILOT_ONLY"}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ("training","development","preflight","training-lock","development-lock"):
  p.add_argument("--"+x,required=True,type=Path)
 p.add_argument("--out",type=Path)
 p.add_argument("--check-lock",type=Path)
 a=p.parse_args()
 data=report(original(a.training,TRAIN_SHA),original(a.development,DEV_SHA),
             original(a.preflight,PF_SHA),
             json.loads(a.training_lock.read_text()),json.loads(a.development_lock.read_text()))
 if a.check_lock and data!=json.loads(a.check_lock.read_text()):
  raise AssertionError("independent receipt audit source lock mismatch")
 if a.out:
  a.out.parent.mkdir(parents=True,exist_ok=True)
  a.out.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print("AMM11_AUDIT_PRIMARY="+json.dumps(data["comparisons"]["full_vs_cosine"],sort_keys=True))
 print("AMM11_AUDIT_VERDICT="+data["original_evalmath_verdict"])
 print("AMM11_AUDIT_RECEIPT="+json.dumps(data,sort_keys=True))

if __name__=="__main__":main()
